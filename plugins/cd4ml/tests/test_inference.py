import os
import sys
import pytest
import joblib
from fastapi.testclient import TestClient

# Make sure the project root (/app) is on the import path
sys.path.insert(0, os.getcwd())

# Import the FastAPI app and predictor class
from plugins.cd4ml.inference.infer import ProductTypePredictorMLflow as ProductTypePredictor, app
import plugins.cd4ml.inference.infer as infer_mod

# ---------------------------------------------------
# # Determine data/model directories from environment
# ---------------------------------------------------

VECTOR_PATH = os.path.join(os.getenv("MODEL_DIR"), os.getenv("TFIDF_VECTORIZER"))
MODEL_PATH = os.path.join(os.getenv("MODEL_DIR"), os.getenv("MODEL"))
PRODUCT_DICT_PATH = os.path.join(os.getenv("MODEL_DIR"), os.getenv("PRODUCT_DICTIONARY"))

@pytest.fixture(scope="module")
def predictor():
    # Skip tests if artifacts are missing
    for path in (VECTOR_PATH, MODEL_PATH, PRODUCT_DICT_PATH):
        if not os.path.exists(path):
            pytest.skip(f"Required artifact not found: {path}")
    model = joblib.load(MODEL_PATH)
    return ProductTypePredictor(
        model=model,
        vectorizer_path=VECTOR_PATH,
        product_dictionary_path=PRODUCT_DICT_PATH,
    )

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_predictor_real_data(predictor):
    result = predictor.predict("Test product", "This is a description")
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

def test_fastapi_endpoint(client, predictor):
    # Replace the global predictor in the inference module
    infer_mod.predictor = predictor

    # Perform login
    login_resp = client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    # Call the /predict endpoint
    resp = client.post(
        "/predict",
        json={"designation": "Test", "description": "Desc"},
        params={"token": token},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    json_data = resp.json()
    assert "predicted_class" in json_data
    print("Raw /predict response JSON:", json_data)