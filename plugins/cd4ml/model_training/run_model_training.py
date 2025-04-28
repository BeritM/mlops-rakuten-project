from model_training import load_train_data, calculate_class_weights, train_final_model
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, classification_report

#import dagshub env variables

DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

# Set paths for X and y data and output model folder

input_dir = os.getenv("DATA_INPUT_DIR", "app/data/processed")
output_dir = os.getenv("MODEL_OUTPUT_DIR", "app/models")
X_train_tfidf_path = os.path.join(input_dir, "X_train_tfidf.pkl")
y_train_path = os.path.join(input_dir, "y_train.pkl")
X_validate_tfidf_path = os.path.join(input_dir, "X_validate_tfidf.pkl")
y_validate_path = os.path.join(input_dir, "y_validate.pkl")
class_report_path = os.path.join(output_dir, "training_class_report.txt")
model_path = f"{output_dir}/sgd_text_model.pkl"



def main():
        
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
    # 3.1 Set MLFlow, including input_example and signature

    tracking_uri = f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten_final_model")

    #input_example = X_train_tfidf.iloc[:1]
    #signature = infer_signature(X_train_tfidf, model.predict(X_train_tfidf))

    #input_example = X_train_tfidf[:1].toarray()
    #prediction_example = model.predict(input_example)
    #signature = infer_signature(input_example, prediction_example)
    
    # 3.2 training

    with mlflow.start_run() as run:
        # Get run ID for later model comparison
        run_id = run.info.run_id
        
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

        mlflow.sklearn.log_model(model, 
                                 artifact_path="model", 
                                 registered_model_name="SGDClassifier_Model"
                                 #signature=signature,
                                 #input_example=input_example
                                 )
        mlflow.log_artifact(class_report_path)

        with open(f"{output_dir}/current_run_id.txt", "w") as f:
            f.write(run_id)
        
    
    print("Training finished")  



    # 4. Save trained model
    
    joblib.dump(model, model_path)

    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    main()