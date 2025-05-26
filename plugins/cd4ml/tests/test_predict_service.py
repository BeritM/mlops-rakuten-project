import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import jwt
from datetime import datetime, timedelta

# Importiere die FastAPI App aus deinem Skript
from predict_api import predict_app, SECRET_KEY, ALGORITHM

# Erstelle einen TestClient für deine FastAPI App
client = TestClient(predict_app)

# --- Fixtures for Mocking ---

@pytest.fixture(scope="module", autouse=True)
def mock_env_vars():
    """Mock DagsHub environment variables."""
    with patch.dict(os.environ, {
        "DAGSHUB_USER_NAME": "test_user",
        "DAGSHUB_USER_TOKEN": "test_token",
        "DAGSHUB_REPO_OWNER": "test_owner",
        "DAGSHUB_REPO_NAME": "test_repo"
    }):
        yield

@pytest.fixture
def mock_mlflow_client():
    """Mocks mlflow.tracking.MlflowClient and its methods."""
    with patch("predict_api.MlflowClient") as MockClient:
        mock_client_instance = MockClient.return_value

        # Mock get_model_version_by_alias
        mock_client_instance.get_model_version_by_alias.return_value = MagicMock(
            version="1",
            run_id="test_run_id",
            creation_timestamp=datetime.now().timestamp() * 1000
        )

        # Mock get_run
        mock_client_instance.get_run.return_value = MagicMock(
            data=MagicMock(
                params={"alpha": "0.0001", "loss": "log_loss", "max_iter": "1000"},
                metrics={"f1_weighted": 0.85}
            )
        )

        # Mock download_artifacts
        # This will simulate the creation of temporary directories and files
        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("os.makedirs", return_value=None):
                with patch("builtins.open", MagicMock()): # Mock file opening
                    mock_client_instance.download_artifacts.side_effect = lambda run_id, path: f"/tmp/{path}"
                    yield mock_client_instance

@pytest.fixture
def mock_mlflow_pyfunc_load_model():
    """Mocks mlflow.pyfunc.load_model."""
    with patch("predict_api.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        yield mock_model

@pytest.fixture
def mock_product_type_predictor():
    """Mocks ProductTypePredictorMLflow."""
    with patch("predict_api.ProductTypePredictorMLflow") as MockPredictor:
        mock_predictor_instance = MockPredictor.return_value
        mock_predictor_instance.predict.return_value = "Electronics"  # Default prediction
        yield mock_predictor_instance

@pytest.fixture
def valid_token():
    """Generiert einen gültigen JWT Token."""
    payload = {"sub": "testuser", "exp": datetime.utcnow() + timedelta(minutes=30)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture
def invalid_token():
    """Generiert einen ungültigen JWT Token (falsche Signatur)."""
    return "invalid.token.signature"

@pytest.fixture
def expired_token():
    """Generiert einen abgelaufenen JWT Token."""
    payload = {"sub": "testuser", "exp": datetime.utcnow() - timedelta(minutes=5)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# --- Tests for /predict endpoint ---

def test_predict_success(mock_mlflow_client, mock_mlflow_pyfunc_load_model, mock_product_type_predictor, valid_token):
    """Test successful prediction with valid token."""
    response = client.post(
        "/predict",
        headers={"token": valid_token},
        json={"designation": "Smartphone", "description": "Latest model, great camera"}
    )
    assert response.status_code == 200
    assert response.json() == {"predicted_class": "Electronics"}
    mock_product_type_predictor.predict.assert_called_once_with("Smartphone", "Latest model, great camera")

def test_predict_missing_designation(valid_token):
    """Test prediction with missing designation."""
    response = client.post(
        "/predict",
        headers={"token": valid_token},
        json={"description": "Latest model, great camera"}
    )
    assert response.status_code == 422  # Unprocessable Entity for Pydantic validation errors

def test_predict_missing_description(valid_token):
    """Test prediction with missing description."""
    response = client.post(
        "/predict",
        headers={"token": valid_token},
        json={"designation": "Smartphone"}
    )
    assert response.status_code == 422

def test_predict_no_token():
    """Test prediction without any token."""
    response = client.post(
        "/predict",
        json={"designation": "Smartphone", "description": "Latest model, great camera"}
    )
    assert response.status_code == 403  # Forbidden (FastAPI's default for missing header)

def test_predict_invalid_token(invalid_token):
    """Test prediction with an invalid token."""
    response = client.post(
        "/predict",
        headers={"token": invalid_token},
        json={"designation": "Smartphone", "description": "Latest model, great camera"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

def test_predict_expired_token(expired_token):
    """Test prediction with an expired token."""
    response = client.post(
        "/predict",
        headers={"token": expired_token},
        json={"designation": "Smartphone", "description": "Latest model, great camera"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

# --- Tests for /model-info endpoint ---

def test_model_info_success(mock_mlflow_client, mock_mlflow_pyfunc_load_model, valid_token):
    """Test successful retrieval of model info with valid token."""
    response = client.get(
        "/model-info",
        headers={"token": valid_token}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_version"] == "1"
    assert "registered_at" in json_response
    assert json_response["parameters"] == {
        "alpha": "0.0001",
        "loss": "log_loss",
        "max_iter": "1000"
    }
    assert json_response["metrics"] == {
        "f1_weighted": 0.85
    }

def test_model_info_no_token():
    """Test model info without any token."""
    response = client.get("/model-info")
    assert response.status_code == 403

def test_model_info_invalid_token(invalid_token):
    """Test model info with an invalid token."""
    response = client.get(
        "/model-info",
        headers={"token": invalid_token}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

def test_model_info_expired_token(expired_token):
    """Test model info with an expired token."""
    response = client.get(
        "/model-info",
        headers={"token": expired_token}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

# --- Tests for startup/initialization (challenging to test directly in FastAPI context) ---
# These are more integration tests or rely on mocking at a higher level.
# For example, if MLflow setup fails, the app might not even start.
# We'll focus on testing the successful startup by ensuring mocks are called.

def test_startup_loads_predictor(mock_mlflow_client, mock_mlflow_pyfunc_load_model, mock_product_type_predictor):
    """
    Verifies that the on_event("startup") function correctly initializes
    the predictor by checking if its constructor was called with the
    expected arguments from MLflow.
    """
    # The fixtures already ensure that the mocks are in place before the test runs.
    # The TestClient automatically calls the startup events.
    # We just need to assert that the predictor was initialized.

    mock_mlflow_pyfunc_load_model.assert_called_once_with(model_uri="models:/SGDClassifier_Model@production")
    mock_mlflow_client.get_model_version_by_alias.assert_called_once_with("SGDClassifier_Model", "production")
    mock_mlflow_client.get_run.assert_called_once_with("test_run_id")
    mock_mlflow_client.download_artifacts.assert_any_call(run_id="test_run_id", path="vectorizer")
    mock_mlflow_client.download_artifacts.assert_any_call(run_id="test_run_id", path="product_dictionary")

    # Since os.path.join is mocked, the arguments for ProductTypePredictorMLflow
    # will reflect the mocked paths.
    mock_product_type_predictor.assert_called_once_with(
        model=mock_mlflow_pyfunc_load_model.return_value,
        vectorizer_path="/tmp/vectorizer/tfidf_vectorizer.pkl",
        product_dictionary_path="/tmp/product_dictionary/product_dictionary.pkl"
    )

# --- Edge Case: Predictor not loaded (e.g., if startup failed) ---
# This simulates a scenario where predictor might be None due to an error during startup.
# While the current setup makes this hard to *directly* test by making startup fail,
# we can manually set predictor to None for a specific test.

@patch('predict_api.predictor', new=None)
def test_predict_when_predictor_not_loaded(valid_token):
    """
    Test prediction endpoint when predictor is not initialized (e.g., startup failed).
    This specifically tests the scenario where 'predictor' is None, leading to an AttributeError
    when trying to call 'predict'. FastAPI should return a 500 Internal Server Error.
    """
    response = client.post(
        "/predict",
        headers={"token": valid_token},
        json={"designation": "Smartphone", "description": "Latest model, great camera"}
    )
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Internal Server Error" in response.json()["detail"] # Or a more specific error message if you add one

# --- Test `verify_token` directly (more granular testing) ---

def test_verify_token_valid(valid_token):
    """Test verify_token with a valid token."""
    from predict_api import verify_token
    payload = verify_token(valid_token)
    assert payload["sub"] == "testuser"

def test_verify_token_invalid_signature():
    """Test verify_token with an invalid signature."""
    from predict_api import verify_token
    token = jwt.encode({"sub": "testuser"}, "wrong_secret", algorithm=ALGORITHM) # Wrong secret
    with pytest.raises(HTTPException) as exc_info:
        verify_token(token)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid token"

def test_verify_token_missing_sub():
    """Test verify_token with a token missing 'sub'."""
    from predict_api import verify_token
    payload = {"exp": datetime.utcnow() + timedelta(minutes=30)} # No 'sub'
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        verify_token(token)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid token"

def test_verify_token_expired(expired_token):
    """Test verify_token with an expired token."""
    from predict_api import verify_token
    with pytest.raises(HTTPException) as exc_info:
        verify_token(expired_token)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid token"