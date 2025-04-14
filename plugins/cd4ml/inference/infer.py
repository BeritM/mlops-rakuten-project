import re
import joblib
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from nltk.corpus import stopwords
from jose import JWTError, jwt
from typing import Dict
from hashlib import sha256
from datetime import datetime, timedelta

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

# --- Product Type Predictor Class (unchanged) ---
class ProductTypePredictor:
    def __init__(self, vectorizer_path, model_path, product_dictionary_path):
        self.stop_words = self._load_stopwords()
        self.vectorizer = self._load_pickle(vectorizer_path)
        self.model = self._load_pickle(model_path)
        self.product_dictionary = self._load_pickle(product_dictionary_path)

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            return joblib.load(f)

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
        combined_text = f"{designation} {description}"
        vectorized = self.preprocess(combined_text)
        prediction = self.model.predict(vectorized)[0]
        return self.product_dictionary[int(prediction)]

predictor = ProductTypePredictor(
    vectorizer_path="data/processed/tfidf_vectorizer.pkl",
    model_path="models/sgd_text_model.pkl",
    product_dictionary_path="models/product_dictionary.pkl"
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
