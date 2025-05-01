import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

from model_validation import load_model, prediction_and_metrics, save_txt_file
from model_training import load_train_data

# Load path environment variables

INPUT_DIR = os.getenv("DATA_PROCESSED_DIR")
OUTPUT_DIR = os.getenv("MODEL_DIR")

X_TEST_TFIDF_PATH = os.path.join(INPUT_DIR, os.getenv("X_TEST_TFIDF"))
Y_TEST_PATH = os.path.join(INPUT_DIR, os.getenv("Y_TEST"))
MODEL_FILE = os.getenv("MODEL")

#import dagshub env variables
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")


def main():

    # Load MLFlow model

    tracking_uri = f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    # Get latest model in production
    
    client = MlflowClient(tracking_uri=tracking_uri)

    latest_prod = client.get_latest_versions("SGDClassifier_Model", stages=["Production"])

    if latest_prod:
        prod_run_id = latest_prod[0].run_id
        prod_metrics = client.get_run(prod_run_id).data.metrics
        prod_f1 = prod_metrics["f1_weighted"]
    else:
        prod_f1 = None

    with open(f"{OUTPUT_DIR}/current_run_id.txt", "r") as f:
        new_run_id = f.read()

    # Neccessary if validation f1 scoring is tracked in MLFlow:
    
    #new_metrics = client.get_run(new_run_id).data.metrics
    #new_f1 = new_metrics["f1_weighted"]

    # Loading new model and calculate f1 on test set
    
    model = load_model(OUTPUT_DIR, MODEL_FILE)

    X_validate_tfidf, y_validate = load_train_data(X_TEST_TFIDF_PATH, Y_TEST_PATH)

    val_acc, val_f1, classification_repo = prediction_and_metrics(X_validate_tfidf, y_validate, model)

    print("Prediction finished. Results:")
    print(f"Validation accuracy: {val_acc}")
    print(f"Validation F1: {val_f1}")
    print(f"classification report: \n {classification_repo}")

    save_txt_file(OUTPUT_DIR, f"classification_report_{MODEL_FILE}", classification_repo)

    print("Classification report saved successfully")

    
    # Get model versions registered under 'SGDClassifier_Model'
    model_versions = client.search_model_versions(f"name='SGDClassifier_Model'")

    # Find latest production model via alias
    prod_model = None
    for mv in model_versions:
        if "production" in (mv.aliases or []):
            prod_model = mv
            break

    prod_f1 = None
    if prod_model:
        prod_run = client.get_run(prod_model.run_id)
        prod_f1 = prod_run.data.metrics.get("f1_weighted")

    # Compares old model with latest model
    if prod_f1 is None or val_f1 > prod_f1:
        # Search for new model by run_id
        new_version = None
        for mv in model_versions:
            if mv.run_id == new_run_id:
                new_version = mv
                break

        if new_version is None:
            raise ValueError(f"No model version found for run_id {new_run_id}.")

        # Put alias 'production' on new model
        client.set_registered_model_alias(
            name="SGDClassifier_Model",
            alias="production",
            version=new_version.version
        )

        print(f"Model version {new_version.version} is now assigned to alias 'production'.")

    else:
        print("Last production model performs better.")


if __name__ == "__main__":
    main()
