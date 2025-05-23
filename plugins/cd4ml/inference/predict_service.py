### --- predict_api.py ---
import os
import joblib
import re
import mlflow
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status, Header, Body
import csv
import pathlib
from datetime import datetime
from pydantic import BaseModel
from jose import jwt, JWTError
from mlflow.tracking import MlflowClient
from plugins.cd4ml.data_processing.preprocessing_core import ProductTypePredictorMLflow

# ----- Environment Variables ---
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
FEEDBACK_DIR = os.getenv("DATA_FEEDBACK_DIR")
FEEDBACK_FILENAME = os.getenv("FEEDBACK_CSV_PATH")

# --- Feedback path setup from env ---
FEEDBACK_CSV_PATH = os.path.join(FEEDBACK_DIR, FEEDBACK_FILENAME)
pathlib.Path(FEEDBACK_DIR).mkdir(parents=True, exist_ok=True)

# --- FastAPI Setup ---
predict_app = FastAPI()

# --- JWT Config ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"

# --- Auth Helper ---
def verify_token(token: str = Header(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Models ---
class PredictionRequest(BaseModel):
    designation: str
    description: str

class PredictionResponse(BaseModel):
    predicted_class: str

class FeedbackEntry(BaseModel):
    designation: str
    description: str
    predicted_label: str
    correct_label: str


# --- Load MLflow Model ---
tracking_uri = f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("rakuten_final_model")

model_name = "SGDClassifier_Model"
model_uri = f"models:/{model_name}@production"
production_model = mlflow.pyfunc.load_model(model_uri=model_uri)
client = MlflowClient()
prod_model_version = client.get_model_version_by_alias(model_name, "production")
run_id = prod_model_version.run_id
run_info = client.get_run(run_id) # type: ignore
model_params = run_info.data.params
model_metrics = run_info.data.metrics

# --- Load TFIDF Vectorizer and Product Dictionary ---
vectorizer_path_dir = client.download_artifacts(run_id=run_id, path="vectorizer") # type: ignore
vectorizer_path = os.path.join(vectorizer_path_dir, "tfidf_vectorizer.pkl")
product_dict_dir = client.download_artifacts(run_id=run_id, path="product_dictionary") # type: ignore
product_dictionary_path = os.path.join(product_dict_dir, "product_dictionary.pkl")

predictor: Optional[ProductTypePredictorMLflow] = None

# product_dict: {code → label}
with open(product_dictionary_path, "rb") as f:
    product_dict = joblib.load(f)
label_to_code = {v: k for k, v in product_dict.items()}  # reverse dict
valid_labels = set(label_to_code.keys())

# --- Load predictor at app startup ---
@predict_app.on_event("startup")
def load_predictor():
    global predictor
    predictor = ProductTypePredictorMLflow(
        model=production_model,
        vectorizer_path=vectorizer_path,
        product_dictionary_path=product_dictionary_path
    )

# --- Health Check Endpoint ---
@predict_app.get("/model-info")
def get_model_info(user=Depends(verify_token)):
    info = {
        "model_version": prod_model_version.version,
        "registered_at": datetime.fromtimestamp(prod_model_version.creation_timestamp / 1000).isoformat(),
        "parameters": {
            "alpha": model_params.get("alpha"),
            "loss": model_params.get("loss"),
            "max_iter": model_params.get("max_iter"),
        },
        "metrics": {
            "f1_weighted": model_metrics.get("f1_weighted")
        }
    }
    return info


# --- Prediction Endpoint ---
@predict_app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest, user=Depends(verify_token)):
    prediction = predictor.predict(request.designation, request.description)
    return {"predicted_class": prediction}


@predict_app.post("/feedback")
def submit_feedback(entry: FeedbackEntry, user=Depends(verify_token)):
    # Validate corrected label
    if entry.correct_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"'{entry.correct_label}' is not a known product category. Valid categories are: {sorted(valid_labels)}"
        )

    if entry.predicted_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"'{entry.predicted_label}' is not a known product category. Valid categories are: {sorted(valid_labels)}"
        )

    is_correct = entry.predicted_label == entry.correct_label

    # Map to integer codes
    predicted_code = label_to_code[entry.predicted_label]
    correct_code = label_to_code[entry.correct_label]

    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "model_version": prod_model_version.version,
        "designation": entry.designation,
        "description": entry.description,
        "predicted_code": predicted_code,
        "predicted_label": entry.predicted_label,
        "correct_code": correct_code,
        "correct_label": entry.correct_label,
        "is_correct": is_correct
    }

    file_exists = pathlib.Path(FEEDBACK_CSV_PATH).is_file()
    with open(FEEDBACK_CSV_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=feedback_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(feedback_data)

    return {"status": "success", "message": "Feedback recorded."}
