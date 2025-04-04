import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../model_training")))

from model_validation import load_model, prediction_and_metrics, save_txt_file
from model_training import load_train_data


def main():

    # Set paths for X and y data and output model folder

    input_dir = os.getenv("DATA_INPUT_DIR", "./data/processed")
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "./models")

    X_validate_tfidf_path = os.path.join(input_dir, "X_validate_tfidf.pkl")
    y_validate_path = os.path.join(input_dir, "y_validate.pkl")
    model_file = "sgd_text_model.pkl"

    model = load_model(output_dir, model_file)

    X_validate_tfidf, y_validate = load_train_data(X_validate_tfidf_path, y_validate_path)

    val_acc, val_f1, classification_repo = prediction_and_metrics(X_validate_tfidf, y_validate, model)

    print("Prediction finished. Results:")
    print(f"Validation accuracy: {val_acc}")
    print(f"Validation F1: {val_f1}")
    print(f"classification report: \n {classification_repo}")

    save_txt_file(output_dir, f"classification_report_{model_file}", classification_repo)

    print("Classification report saved successfully")


if __name__ == "__main__":
    main()
