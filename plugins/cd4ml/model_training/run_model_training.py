from model_training import load_train_data, calculate_class_weights, train_final_model
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, classification_report
import yaml
import subprocess                    

# ---------- Dagshub ENV ----------
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME")

# ---------- Pfade ----------
INPUT_DIR  = os.getenv("DATA_PROCESSED_DIR")
OUTPUT_DIR = os.getenv("MODEL_DIR")

X_TRAIN_TFIDF_PATH    = os.path.join(INPUT_DIR,  os.getenv("X_TRAIN_TFIDF"))
Y_TRAIN_PATH          = os.path.join(INPUT_DIR,  os.getenv("Y_TRAIN"))
X_VALIDATE_TFIDF_PATH = os.path.join(INPUT_DIR,  os.getenv("X_VALIDATE_TFIDF"))
Y_VALIDATE_PATH       = os.path.join(INPUT_DIR,  os.getenv("Y_VALIDATE"))
MODEL_PATH            = os.path.join(OUTPUT_DIR, os.getenv("MODEL"))
class_report_path     = os.path.join(OUTPUT_DIR, "training_class_report.txt")

def load_config(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, filename), "r") as fh:
        return yaml.safe_load(fh)

#### max ####
def dvc_track_and_commit_shared_model():
    """
    Track shared_volume/models dir and stage its .dvc file.
    """
    # we want 'shared_volume/models' → shared_volume/models.dvc
    subprocess.run(
        ["dvc", "add", "--force", "models"],  # /app/models
        check=True, text=True
    )
    subprocess.run(
        ["git", "add", "models.dvc"],         # root-level models.dvc
        check=True, text=True
    )
    # move it into shared_volume
    subprocess.run(
        ["mv", "models.dvc", "shared_volume/models.dvc"],
        check=True, text=True
    )
    subprocess.run(
        ["git", "add", "shared_volume/models.dvc"],
        check=True, text=True
    )
#### max ####

def main():
    print(f"Loading X_train_tfidf from {X_TRAIN_TFIDF_PATH}")
    print(f"Loading y_train        from {Y_TRAIN_PATH}")
    print(f"Loading X_validate_tfidf from {X_VALIDATE_TFIDF_PATH}")
    print(f"Loading y_validate       from {Y_VALIDATE_PATH}")

    # 1. Load data
    X_train_tfidf, y_train = load_train_data(X_TRAIN_TFIDF_PATH, Y_TRAIN_PATH)
    X_validate_tfidf, y_validate = load_train_data(X_VALIDATE_TFIDF_PATH, Y_VALIDATE_PATH)

    # 2. Class weights
    custom_class_weights = calculate_class_weights(y_train)

    # 3. MLflow Tracking
    tracking_uri = (
        f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}"
        f"@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten_final_model")

    # 4. Train model
    config_param = load_config("param_config.yml")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model = train_final_model(custom_class_weights, X_train_tfidf, y_train)

        y_pred = model.predict(X_validate_tfidf)
        f1_weighted = f1_score(y_validate, y_pred, average="weighted")

        report = classification_report(y_validate, y_pred)
        with open(class_report_path, "w") as fh:
            fh.write(report)

        mlflow.log_param("loss",      config_param["model"]["params"]["loss"])
        mlflow.log_param("alpha",     config_param["model"]["params"]["alpha"])
        mlflow.log_param("max_iter",  config_param["model"]["params"]["max_iter"])
        mlflow.log_metric("f1_weighted", f1_weighted)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SGDClassifier_Model"
        )
        mlflow.log_artifact(class_report_path)

        with open(f"{OUTPUT_DIR}/current_run_id.txt", "w") as fh:
            fh.write(run_id)

    # 5. Persist model locally
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()
    #max# produce shared_volume/models.dvc
    dvc_track_and_commit_shared_model()