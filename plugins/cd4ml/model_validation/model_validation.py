import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import joblib
from sklearn.base import BaseEstimator

def load_model(output_dir: str, model_file: str) -> BaseEstimator:
    model_path = f"{output_dir}/{model_file}"
    model = joblib.load(model_path)

    return model

def prediction_and_metrics(X_validate_tfidf: pd.DataFrame, y_validate: pd.DataFrame, model: BaseEstimator):
    y_pred_validate = model.predict(X_validate_tfidf)
    val_accuracy = accuracy_score(y_validate, y_pred_validate)
    val_f1 = f1_score(y_validate, y_pred_validate, average='weighted')
    class_report = classification_report(y_validate, y_pred_validate)

    return val_accuracy, val_f1, class_report, y_pred_validate

def save_txt_file(output_dir: str, file_name: str, content: str):
    report_path = f"{output_dir}/{file_name}.txt"
    with open(report_path, "w") as f:
        f.write(content)