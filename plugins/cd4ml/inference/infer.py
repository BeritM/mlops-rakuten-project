import re
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords

# Define file paths

# --- Class Definition ---
class ProductTypePredictor:
    def __init__(self, vectorizer_path, model_path, product_dictionary_path):
        self.stop_words = self._load_stopwords()
        self.vectorizer = self._load_pickle(vectorizer_path)
        self.model = self._load_pickle(model_path)
        self.product_dictionary = self._load_pickle(product_dictionary_path)

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_stopwords(self):
        stop_words_eng = set(stopwords.words("english"))
        stop_words_fr = set(stopwords.words("french"))
        custom_stopwords = set([
            "chez", "der", "plu", "haut", "peut", "non", "100", "produit",
            "lot", "tout", "cet", "cest", "sou", "san"
        ])
        return stop_words_eng.union(stop_words_fr).union(custom_stopwords)

    def preprocess(self, text):
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        cleaned = " ".join(word for word in cleaned.split() if word not in self.stop_words)
        return self.vectorizer.transform([cleaned])

    def predict(self, designation, description=""):
        # input validation - only strings allowed
        if not isinstance(designation, str):
            raise ValueError("designation muss ein String sein.")
        if not isinstance(description, str):
            raise ValueError("description muss ein String sein.")
        
        combined_text = f"{designation} {description}"
        vectorized = self.preprocess(combined_text)
        prediction = self.model.predict(vectorized)[0]
        print(prediction, type(prediction))
        prediction = self.product_dictionary[prediction]
        return prediction


# --- FastAPI Setup ---
app = FastAPI()

# Load model and vectorizer
predictor = ProductTypePredictor(
    vectorizer_path="models/tfidf_vectorizer.pkl",
    model_path="models/sgd_text_model.pkl",
    product_dictionary_path = "models/product_dictionary.pkl"
)

# --- Request Body Schema ---
class PredictionRequest(BaseModel):
    designation: str
    description: str = ""

# --- Response Schema (optional) ---
class PredictionResponse(BaseModel):
    predicted_class: str

# --- POST Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest):
    prediction = predictor.predict(request.designation, request.description)
    return {"predicted_class": str(prediction)}
