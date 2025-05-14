### --- predict_api.py ---
import os
import joblib
import re
import mlflow
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel
from jose import jwt, JWTError
from mlflow.tracking import MlflowClient
from plugins.cd4ml.data_processing.preprocessing_core import ProductTypePredictorMLflow

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

#def verify_token(authorization: str = Header(...)):
#    try:
#        scheme, token = authorization.split()
#        if scheme.lower() != "bearer":
#            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth scheme")
#        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#        username = payload.get("sub")
#        if username is None:
#            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
#        return payload
#    except (JWTError, ValueError):
#        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

# --- Models ---
class PredictionRequest(BaseModel):
    designation: str
    description: str

class PredictionResponse(BaseModel):
    predicted_class: str

# --- Path Variables ---
#MODEL_DIR = os.getenv("MODEL_DIR")
#TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, os.getenv("TFIDF_VECTORIZER"))
#PRODUCT_DICTIONARY_PATH = os.path.join(MODEL_DIR, os.getenv("PRODUCT_DICTIONARY"))

DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

tracking_uri = f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("rakuten_final_model")

model_name = "SGDClassifier_Model"
model_uri = f"models:/{model_name}@production"
production_model = mlflow.pyfunc.load_model(model_uri=model_uri)
client = MlflowClient()
prod_model_version = client.get_model_version_by_alias(model_name, "production")
run_id = prod_model_version.run_id
run_info = client.get_run(run_id)
model_params = run_info.data.params
model_metrics = run_info.data.metrics

# --- Load TFIDF Vectorizer and Product Dictionary ---
vectorizer_path_dir = client.download_artifacts(run_id=run_id, path="vectorizer")
vectorizer_path = os.path.join(vectorizer_path_dir, "tfidf_vectorizer.pkl")
product_dict_dir = client.download_artifacts(run_id=run_id, path="product_dictionary")
product_dictionary_path = os.path.join(product_dict_dir, "product_dictionary.pkl")

predictor: Optional[ProductTypePredictorMLflow] = None

@predict_app.on_event("startup")
def load_predictor():
    global predictor
    predictor = ProductTypePredictorMLflow(
        model=production_model,
        vectorizer_path=vectorizer_path,
        product_dictionary_path=product_dictionary_path
    )

@predict_app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest, user=Depends(verify_token)):
    prediction = predictor.predict(request.designation, request.description)
    return {"predicted_class": prediction}

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
