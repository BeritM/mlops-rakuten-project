import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

from model_validation import load_model, prediction_and_metrics, save_txt_file
from model_training import load_train_data

# Set paths for X and y data and output model folder

input_dir = os.getenv("DATA_INPUT_DIR", "./data/processed")
output_dir = os.getenv("MODEL_OUTPUT_DIR", "./models")

# Change to test after test run!!!
X_test_tfidf_path = os.path.join(input_dir, "X_test_tfidf.pkl")
y_test_path = os.path.join(input_dir, "y_test.pkl")
model_file = "sgd_text_model.pkl"

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

    with open(f"{output_dir}/current_run_id.txt", "r") as f:
        new_run_id = f.read()

    # Neccessary if validation f1 scoring is tracked in MLFlow:
    
    #new_metrics = client.get_run(new_run_id).data.metrics
    #new_f1 = new_metrics["f1_weighted"]

    # Loading new model and calculate f1 on test set
    
    model = load_model(output_dir, model_file)

    X_validate_tfidf, y_validate = load_train_data(X_test_tfidf_path, y_test_path)

    val_acc, val_f1, classification_repo = prediction_and_metrics(X_validate_tfidf, y_validate, model)

    print("Prediction finished. Results:")
    print(f"Validation accuracy: {val_acc}")
    print(f"Validation F1: {val_f1}")
    print(f"classification report: \n {classification_repo}")

    save_txt_file(output_dir, f"classification_report_{model_file}", classification_repo)

    print("Classification report saved successfully")

    # Compare latest model in production with new model

    #if prod_f1 is None or val_f1 > prod_f1:
    #    # Get new model version
    #    model_versions = client.search_model_versions("name='SGDClassifier_Model'")
    #    new_version = None
    #    for mv in model_versions:
    #        if mv.run_id == new_run_id:
    #            new_version = mv.version
    #            break
        
    #    if new_version is None:
    #        raise ValueError(f"No model version found for run_id {new_run_id}.")

    #    # Put new model in production stage
    #    client.transition_model_version_stage(
    #        name="SGDClassifier_Model",
    #        version=new_version,
    #        stage="Production"
    #    )
    #    print(f"Model version {new_version} in production.")
    #else:
    #    print("Last production model performs better.")


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

    # Vergleich der neuen mit der aktuellen Production-Version
    if prod_f1 is None or val_f1 > prod_f1:
        # Suche neue Version passend zur run_id
        new_version = None
        for mv in model_versions:
            if mv.run_id == new_run_id:
                new_version = mv
                break

        if new_version is None:
            raise ValueError(f"No model version found for run_id {new_run_id}.")

        # Setze Alias 'production' auf die neue Version
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
