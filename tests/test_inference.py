import sys
import os
import pytest
import numpy as np

# Add the project directory to sys.path so that the "plugins" folder is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from plugins.cd4ml.inference.infer import ProductTypePredictor, app

VECTOR_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "sgd_text_model.pkl")

# Fixture: Initializes the predictor or skips the test if files are missing.
@pytest.fixture(scope="module")
def predictor():
    if not os.path.exists(VECTOR_PATH) or not os.path.exists(MODEL_PATH):
        pytest.skip("Model or vectorizer file not found.")
    return ProductTypePredictor(VECTOR_PATH, MODEL_PATH)

# Fixture: FastAPI TestClient for integration tests.
@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# ---------------------------
# Unit Tests for the Predictor
# ---------------------------
def test_predictor_real_data(predictor):
    """
    Test the predict method of the ProductTypePredictor.
    Checks:
      - Input (designation and description) are strings.
      - A non-None result is returned.
      - Output is a numeric product code (Python int or numpy integer).
      - Fails if the output format is not as expected.
    """
    designation = "Test product"
    description = "This is a description"
    assert isinstance(designation, str), "Designation must be a string."
    assert isinstance(description, str), "Description must be a string."
    
    result = predictor.predict(designation, description)
    assert result is not None, "Prediction should not be None."
    
    if not (isinstance(result, (int, str)) or np.issubdtype(type(result), np.integer)):
        pytest.fail(f"Result should be a numeric value (int or np.integer) or string, but got {type(result)}.")
    
    if isinstance(result, str):
        pytest.fail("Result is a string; expected a numeric product code.")
    
    print(f"Input: designation: {designation} (Type: {type(designation).__name__}), "
          f"description: {description} (Type: {type(description).__name__})")
    print(f"Output: result: {result} (Type: {type(result).__name__})")

def test_predictor_empty_inputs(predictor):
    """
    Test how the predictor handles empty inputs.
    Checks that a result is returned even if one of the inputs is empty.
    """
    # Empty description
    designation = "Test product"
    description = ""
    result = predictor.predict(designation, description)
    assert result is not None, "Result should not be None when description is empty."
    
    # Empty designation
    designation = ""
    description = "Description"
    result = predictor.predict(designation, description)
    assert result is not None, "Result should not be None when designation is empty."

def test_predictor_invalid_input(predictor):
    """
    Test that the predictor raises an exception for invalid inputs,
    e.g., None or numeric values instead of strings.
    """
    with pytest.raises(Exception):
        predictor.predict(None, "Description")
    
    with pytest.raises(Exception):
        predictor.predict("Test product", None)
    
    with pytest.raises(Exception):
        predictor.predict(123, 456)

def test_predictor_special_characters(predictor):
    """
    Test that special characters in the input are handled correctly.
    At least, a result should be returned.
    """
    designation = "Test!@# product$$%"
    description = "This--is a description???"
    result = predictor.predict(designation, description)
    assert result is not None, "Result should not be None when special characters are present."

# ---------------------------
# Integration Test for FastAPI Endpoint
# ---------------------------
def test_fastapi_endpoint(client):
    """
    Test the /predict endpoint of the FastAPI app.
    Checks:
      - HTTP status code is 200.
      - JSON response contains 'predicted_class'.
      - 'predicted_class' is not None.
    """
    payload = {
        "designation": "Test product",
        "description": "This is a description"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, "HTTP status code should be 200."
    
    data = response.json()
    assert "predicted_class" in data, "Response must contain 'predicted_class'."
    assert data["predicted_class"] is not None, "'predicted_class' should not be None."
    print(f"FastAPI Response: {data}")
