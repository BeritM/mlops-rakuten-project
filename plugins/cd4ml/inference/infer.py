import re
import joblib
import mlflow
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from nltk.corpus import stopwords
from jose import JWTError, jwt
from typing import Dict
from hashlib import sha256
from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient



# --- JWT Configuration ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- FastAPI Setup ---
app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Simple In-Memory User Database ---
users_db: Dict[str, Dict[str, str]] = {
    "admin": {"username": "admin", "password": sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "user": {"username": "user", "password": sha256("user123".encode()).hexdigest(), "role": "user"},
}

# --- JWT Token Utility ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Models ---
class PredictionRequest(BaseModel):
    designation: str
    description: str

class PredictionResponse(BaseModel):
    predicted_class: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

# --- Import dagshub ---
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

# --- Load Model from MLflow/DagsHub ---
tracking_uri = f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("rakuten_final_model")

# Load the latest model version
model_name = "SGDClassifier_Model"
model_uri = f"models:/{model_name}@production"
production_model = mlflow.pyfunc.load_model(model_uri=model_uri)

client = MlflowClient()
prod_model_version = client.get_model_version_by_alias(model_name, "production")

# Extract parameters and metrics
run_info = client.get_run(prod_model_version.run_id)
model_params = run_info.data.params
model_metrics = run_info.data.metrics

# --- Predictor Wrapper using MLflow ---
class ProductTypePredictorMLflow:
    def __init__(self, model, vectorizer_path, product_dictionary_path):
        self.model = model
        self.stop_words = self._load_stopwords()
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = joblib.load(f)
        with open(product_dictionary_path, "rb") as f:
            self.product_dictionary = joblib.load(f)

    def _load_stopwords(self):
        stop_words_eng = set(stopwords.words("english"))
        stop_words_fr = set(stopwords.words("french"))
        custom_stopwords = set(["chez", "der", "plu", "haut", "peut", "non", "100", "produit",
                                "lot", "tout", "cet", "cest", "sou", "san"])
        return stop_words_eng.union(stop_words_fr).union(custom_stopwords)

    def preprocess(self, text):
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        cleaned = " ".join(word for word in cleaned.split() if word not in self.stop_words)
        return self.vectorizer.transform([cleaned])

    def predict(self, designation, description=""):
        # input validation - only strings allowed
        if not isinstance(designation, str):
            raise ValueError("designation has to be a string.")
        if not isinstance(description, str):
            raise ValueError("description has to be a string.")
        
        combined_text = f"{designation} {description}"
        vectorized = self.preprocess(combined_text)
        prediction = self.model.predict(vectorized)[0]
        print(prediction, type(prediction))
        prediction = self.product_dictionary[int(prediction)]
        return prediction


predictor = ProductTypePredictorMLflow(
    #model=latest_model,
    model=production_model,
    vectorizer_path="data/processed/tfidf_vectorizer.pkl",
    product_dictionary_path = "models/product_dictionary.pkl"
)


# --- Authentication Endpoints ---
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    hashed_pw = sha256(form_data.password.encode()).hexdigest()
    if not user or user["password"] != hashed_pw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]},
                                       expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest, user=Depends(verify_token)):
    prediction = predictor.predict(request.designation, request.description)
    return {"predicted_class": prediction}

# --- Model Information Endpoint ---
@app.get("/model-info")
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

# --- Admin-only Endpoints ---
def admin_required(user=Depends(verify_token)):
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return user

@app.post("/users")
def create_user(user_data: UserCreate, user=Depends(admin_required)):
    if user_data.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    users_db[user_data.username] = {
        "username": user_data.username,
        "password": sha256(user_data.password.encode()).hexdigest(),
        "role": user_data.role
    }
    return {"msg": "User created successfully."}

@app.delete("/users/{username}")
def delete_user(username: str, user=Depends(admin_required)):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[username]
    return {"msg": "User deleted successfully."}
