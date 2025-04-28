import os
from dotenv import load_dotenv

# Projekt‑Root ermitteln
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
# .env laden
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Danach erst das MLflow‑Mocking usw.
import sys
import types
import pytest
import numpy as np
import joblib

# ---------------------------------------------------
# Wechsel ins shared_volume als CWD
# ---------------------------------------------------
SHARED_DIR = os.path.join(PROJECT_ROOT, "shared_volume")
os.chdir(SHARED_DIR)

# ---------------------------------------------------
# Sys.path anpassen
# ---------------------------------------------------
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------
# Echte Importe (jetzt mit geladenen ENV‑Variablen)
# ---------------------------------------------------
from fastapi.testclient import TestClient
import plugins.cd4ml.inference.infer as infer_mod
from plugins.cd4ml.inference.infer import ProductTypePredictorMLflow as ProductTypePredictor, app

# ---------------------------------------------------
# Pfade zu den realen Artefakten
# ---------------------------------------------------
VECTOR_PATH       = os.path.join("data", "processed", "tfidf_vectorizer.pkl")
MODEL_PATH        = os.path.join("models", "sgd_text_model.pkl")
PRODUCT_DICT_PATH = os.path.join("models", "product_dictionary.pkl")

# ---------------------------------------------------
# Fixtures
# ---------------------------------------------------
@pytest.fixture(scope="module")
def predictor():
    for p in (VECTOR_PATH, MODEL_PATH, PRODUCT_DICT_PATH):
        if not os.path.exists(p):
            pytest.skip(f"Required file not found: {p}")
    model = joblib.load(MODEL_PATH)
    return ProductTypePredictor(
        model=model,
        vectorizer_path=VECTOR_PATH,
        product_dictionary_path=PRODUCT_DICT_PATH
    )

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# ---------------------------------------------------
# 7) Unit-Tests für den Predictor
# ---------------------------------------------------
def test_predictor_real_data(predictor):
    designation = "Test product"
    description = "This is a description"
    result = predictor.predict(designation, description)
    assert result is not None
    assert isinstance(result, str)

def test_predictor_empty_inputs(predictor):
    assert predictor.predict("Test product", "") is not None
    assert predictor.predict("", "Description") is not None

def test_predictor_invalid_input(predictor):
    with pytest.raises(Exception):
        predictor.predict(None, "Description")
    with pytest.raises(Exception):
        predictor.predict("Test product", None)
    with pytest.raises(Exception):
        predictor.predict(123, 456)

def test_predictor_special_characters(predictor):
    assert predictor.predict("Test!@#", "Desc???") is not None


# ---------------------------------------------------
# 7) Integrationstest für /predict
# ---------------------------------------------------
def test_fastapi_endpoint(client, predictor):
    # Ersetze globalen Predictor, damit infer.py nicht seinen eigenen lädt
    infer_mod.predictor = predictor

    # Login
    login_resp = client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert login_resp.status_code == 200

    token = login_resp.json()["access_token"]
    # Predict‑Call
    resp = client.post(
        "/predict",
        json={"designation": "Test", "description": "Desc"},
        params={"token": token},
        headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in resp.json()
    print("Raw /predict response JSON:", data)