import os
import sys
import pytest
from fastapi.testclient import TestClient
from jose import jwt
from datetime import datetime, timedelta

# Sicherstellen, dass das Projekt-Root im Pfad ist
sys.path.insert(0, os.getcwd())

# Importiere die App
from plugins.cd4ml.inference.predict_service import predict_app, SECRET_KEY, ALGORITHM, predictor

client = TestClient(predict_app)

# JWT Token generieren
def create_test_token(username="admin"):
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture(scope="module")
def token():
    return create_test_token()

@pytest.mark.skipif(predictor is None, reason="Predictor was not loaded correctly")
def test_predict_valid(token):
    response = client.post(
        "/predict",
        json={"designation": "Test", "description": "This is a description."},
        headers={"Authorization": token}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_class" in json_data
    assert isinstance(json_data["predicted_class"], str)

@pytest.mark.skipif(predictor is None, reason="Predictor was not loaded correctly")
def test_predict_empty_fields(token):
    response = client.post(
        "/predict",
        json={"designation": "", "description": ""},
        headers={"Authorization": token}
    )
    assert response.status_code == 200
    assert "predicted_class" in response.json()

@pytest.mark.skipif(predictor is None, reason="Predictor was not loaded correctly")
def test_predict_invalid_fields(token):
    response = client.post(
        "/predict",
        json={"designation": None, "description": 123},
        headers={"Authorization": token}
    )
    assert response.status_code == 422  # Validation error

def test_predict_missing_token():
    response = client.post(
        "/predict",
        json={"designation": "Test", "description": "Desc"}
    )
    assert response.status_code == 422 or response.status_code == 401

@pytest.mark.skipif(predictor is None, reason="Predictor was not loaded correctly")
def test_model_info(token):
    response = client.get("/model-info", headers={"Authorization": token})
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "parameters" in data
    assert "metrics" in data
    assert "f1_weighted" in data["metrics"]
