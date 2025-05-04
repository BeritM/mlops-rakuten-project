import re
import joblib
from nltk.corpus import stopwords

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
        if not isinstance(designation, str):
            raise ValueError("designation has to be a string.")
        if not isinstance(description, str):
            raise ValueError("description has to be a string.")
        
        combined_text = f"{designation} {description}"
        vectorized = self.preprocess(combined_text)
        prediction = self.model.predict(vectorized)[0]
        return self.product_dictionary[int(prediction)]