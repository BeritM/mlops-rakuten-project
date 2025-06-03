import os
import csv
import pathlib
import joblib
import mlflow
from datetime import datetime
from typing import Optional
from dvc_push_manager import track_and_push_with_retry
import threading

from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel
from jose import jwt, JWTError
from mlflow.tracking import MlflowClient
from plugins.cd4ml.data_processing.preprocessing_core import ProductTypePredictorMLflow

# --- FastAPI App ---
predict_app = FastAPI()

# --- Auth Config ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"

# --- Globals ---
FEEDBACK_CSV_PATH = None
predictor: Optional[ProductTypePredictorMLflow] = None
prod_model_version = None
model_params = {}
model_metrics = {}
label_to_code = {}
valid_labels = set()

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

# --- DVC Push Helper ---
def _async_track_and_push(description: str) -> None:
    def _worker():
        try:
            ok = track_and_push_with_retry(description=description, max_retries=3)
            if ok:
                print("[INFO] DVC/Git push succeeded.")
            else:
                print("[WARNING] DVC/Git push completed with warnings.")
        except Exception as e:
            print(f"[ERROR] DVC/Git push failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()

# --- Pydantic Models ---
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

# --- Startup Hook ---
@predict_app.on_event("startup")
def startup():
    global FEEDBACK_CSV_PATH, predictor, prod_model_version, model_params, model_metrics, label_to_code, valid_labels

    # Load env variables
    feedback_dir = os.getenv("DATA_FEEDBACK_DIR")
    feedback_filename = os.getenv("FEEDBACK_CSV_PATH")
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")

    # Setup feedback path
    if feedback_dir and feedback_filename:
        pathlib.Path(feedback_dir).mkdir(parents=True, exist_ok=True)
        FEEDBACK_CSV_PATH = os.path.join(feedback_dir, feedback_filename)
        print(f"[INFO] Feedback CSV path: {FEEDBACK_CSV_PATH}")
    else:
        print("[WARNING] Feedback vars not set. Feedback functionality disabled.")
        FEEDBACK_CSV_PATH = None

    # Setup MLflow
    model_name = "SGDClassifier_Model"
    try:
        tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("rakuten_final_model")
        print(f"[INFO] MLflow tracking URI set")

        client = MlflowClient()
        print(f"[DEBUG] Fetching model alias 'production'...")
        prod_model_version = client.get_model_version_by_alias(model_name, "production")
        run_id = prod_model_version.run_id
        print(f"[INFO] Got run ID: {run_id}")

        print(f"[DEBUG] Loading production model...")
        production_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")

        run_info = client.get_run(run_id)
        model_params = run_info.data.params
        model_metrics = run_info.data.metrics
        print(f"[INFO] Model loaded successfully.")

        print(f"[DEBUG] Downloading vectorizer...")
        vectorizer_path_dir = client.download_artifacts(run_id=run_id, path="vectorizer")
        vectorizer_path = os.path.join(vectorizer_path_dir, "tfidf_vectorizer.pkl")
        print(f"[INFO] Vectorizer path: {vectorizer_path}")
        # vectorizer_path = "/app/models/tfidf_vectorizer.pkl"
        print(f"[INFO] Using remote vectorizer: {vectorizer_path}")

        print(f"[DEBUG] Downloading product dictionary...")
        product_dict_dir = client.download_artifacts(run_id=run_id, path="product_dictionary")
        product_dictionary_path = os.path.join(product_dict_dir, "product_dictionary.pkl")
        print(f"[INFO] Product dictionary path: {product_dictionary_path}")
        # product_dictionary_path = "/app/models/product_dictionary.pkl"
        print(f"[INFO] Using remote product dictionary: {product_dictionary_path}")


        with open(product_dictionary_path, "rb") as f:
            product_dict = joblib.load(f)
        label_to_code = {v: k for k, v in product_dict.items()}
        valid_labels = set(label_to_code.keys())

        predictor = ProductTypePredictorMLflow(
            model=production_model,
            vectorizer_path=vectorizer_path,
            product_dictionary_path=product_dictionary_path
        )

        print("[INFO] Predict service startup complete.")

    except Exception as e:
        print(f"[ERROR] Failed during startup: {e}")
        raise RuntimeError("Predict service startup failed. See logs for details.")

# --- Endpoints ---

@predict_app.get("/model-info")
def get_model_info(user=Depends(verify_token)):
    return {
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

@predict_app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest, user=Depends(verify_token)):
    prediction = predictor.predict(request.designation, request.description)
    return {"predicted_class": prediction}

@predict_app.get("/health")
def health_check():
    return {"status": "healthy", "service": "predict_service"}

@predict_app.post("/feedback")
def submit_feedback(entry: FeedbackEntry, user=Depends(verify_token)):
    if FEEDBACK_CSV_PATH is None:
        raise HTTPException(status_code=503, detail="Feedback is not enabled.")

    if entry.correct_label not in valid_labels or entry.predicted_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail="Invalid product label. Check valid categories."
        )

    predicted_code = label_to_code[entry.predicted_label]
    correct_code = label_to_code[entry.correct_label]
    is_correct = predicted_code == correct_code

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
        
    _async_track_and_push(description="append feedback entry")  

    return {"status": "success", "message": "Feedback recorded."}
