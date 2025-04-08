from model_training import load_train_data, calculate_class_weights, train_final_model
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, classification_report

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

def main():
        
    # Set paths for X and y data and output model folder

    input_dir = os.getenv("DATA_INPUT_DIR", "app/data/processed")
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "app/models")
    X_train_tfidf_path = os.path.join(input_dir, "X_train_tfidf.pkl")
    y_train_path = os.path.join(input_dir, "y_train.pkl")
    X_validate_tfidf_path = os.path.join(input_dir, "X_validate_tfidf.pkl")
    y_validate_path = os.path.join(input_dir, "y_validate.pkl")
    class_report_path = os.path.join(output_dir, "training_class_report.txt")

    print(f"Loading X_train_tfidf from {X_train_tfidf_path}")
    print(f"Loading y_train from {y_train_path}")
    print(f"Loading X_validate_tfidf from {X_validate_tfidf_path}")
    print(f"Loading y_validate from {y_validate_path}")

    # 1. Load preprocessed data

    X_train_tfidf, y_train = load_train_data(X_train_tfidf_path, y_train_path)
    X_validate_tfidf, y_validate = load_train_data(X_validate_tfidf_path, y_validate_path)

    print("Data loading successful")

    # 2. Calculate adapted weights

    custom_class_weights = calculate_class_weights(y_train)

    print("Start model training")

    # 3. Train selected model
    # 3.1 Set MLFlow

    tracking_uri = f"https://{DAGSHUB_REPO_OWNER}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten_final_model")
    
    # 3.2 training

    with mlflow.start_run():
        model = train_final_model(custom_class_weights, X_train_tfidf, y_train)

        y_pred = model.predict(X_validate_tfidf)
        f1_weighted = f1_score(y_validate, y_pred, average="weighted")

        report = classification_report(y_validate, y_pred)
        with open(class_report_path, "w") as f:
            f.write(report)
        
        mlflow.log_param("loss", "log_loss")
        mlflow.log_param("alpha", 1.1616550847757421e-06)
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("f1_weighted", f1_weighted)

        mlflow.sklearn.log_model(model, artifact_path="modell", registered_model_name="SGDClassifier_Model")
        mlflow.log_artifact(class_report_path)

    
    print("Training finished")  



    # 4. Save trained model
    
    model_path = f"{output_dir}/sgd_text_model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    main()