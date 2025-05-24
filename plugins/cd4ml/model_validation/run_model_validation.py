import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess          

from model_validation import load_model, prediction_and_metrics, save_txt_file
from model_training import load_train_data

# ---------- Pfade ----------
INPUT_DIR  = os.getenv("DATA_PROCESSED_DIR")
OUTPUT_DIR = os.getenv("MODEL_DIR")
X_TEST_TFIDF_PATH = os.path.join(INPUT_DIR, os.getenv("X_TEST_TFIDF"))
Y_TEST_PATH       = os.path.join(INPUT_DIR, os.getenv("Y_TEST"))
MODEL_FILE        = os.getenv("MODEL")

# ---------- Dagshub ----------
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME")

import os
import subprocess

def track_and_push(paths, description: str):
    """
    Für jeden Pfad in `paths` (absolute Container-Pfad, z.B. "/app/data/processed" oder "/app/models"):
      1. Erzeuge den zu trackenden Pfad unter shared_volume
      2. 'dvc add --force shared_volume/<relpath>'
      3. git add shared_volume/<relpath>.dvc
    Anschließend git commit & git push.
    """
    cwd = os.getcwd()

    dvc_files = []
    for p in paths:
        # z.B. p="/app/data/processed"  → rel="data/processed"
        rel = os.path.relpath(p, cwd)
        # das ist der bereits per Compose gemountete Host-Ordner:
        shared_rel = os.path.join("shared_volume", rel)

        # 1) Tracken (Meta-Datei landet unter shared_volume/...)
        subprocess.run(
            ["dvc", "add", "--force", shared_rel],
            check=True, text=True
        )

        # 2) Git-Stage der automatisch erzeugten .dvc-Datei
        dvc_file = f"{shared_rel}.dvc"
        subprocess.run(
            ["git", "add", dvc_file],
            check=True, text=True
        )
        dvc_files.append(dvc_file)

    # 3) Commit & Push
    subprocess.run(
        ["git", "commit", "-m", f"dvc: {description}"],
        check=True, text=True
    )
    subprocess.run(
        ["git", "push"],
        check=True, text=True
    )

    print(f"Tracked & committed: {', '.join(dvc_files)}")


def main():
    # 1. MLflow connection
    tracking_uri = (
        f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}"
        f"@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 2. Compare with current production model
    latest_prod = client.get_latest_versions("SGDClassifier_Model", stages=["Production"])
    prod_f1 = None
    if latest_prod:
        prod_run_id = latest_prod[0].run_id
        prod_f1 = client.get_run(prod_run_id).data.metrics.get("f1_weighted")

    with open(f"{OUTPUT_DIR}/current_run_id.txt", "r") as fh:
        new_run_id = fh.read().strip()

    model = load_model(OUTPUT_DIR, MODEL_FILE)
    X_test_tfidf, y_test = load_train_data(X_TEST_TFIDF_PATH, Y_TEST_PATH)

    val_acc, val_f1, classification_repo = prediction_and_metrics(X_test_tfidf, y_test, model)

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation F1:       {val_f1:.4f}")

    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{MODEL_FILE}.txt")
    save_txt_file(OUTPUT_DIR, f"classification_report_{MODEL_FILE}", classification_repo)

    # 3. Promote model if better
    model_versions = client.search_model_versions("name='SGDClassifier_Model'")
    new_version = next((mv for mv in model_versions if mv.run_id == new_run_id), None)

    if prod_f1 is None or val_f1 > prod_f1:
        if new_version is None:
            raise ValueError(f"No model version found for run_id {new_run_id}")
        client.set_registered_model_alias(
            name="SGDClassifier_Model",
            alias="production",
            version=new_version.version
        )
        print(f"Model version {new_version.version} is now 'production'.")
    else:
        print("Existing production model performs better.")


if __name__ == "__main__":
    main()
    track_and_push([OUTPUT_DIR], "track validated model")