import pytest
import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH

@pytest.fixture
def model():
    """Load trained model fixture."""
    return joblib.load(MODEL_OUTPUT_PATH)

VALID_INPUT = np.array([[50, 1, 100.0, 6, 15, 1, 3, 1, 1, 1]])

# ── Model loading ──
def test_model_loads_successfully():
    """Model file should load without errors."""
    model = joblib.load(MODEL_OUTPUT_PATH)
    assert model is not None

def test_model_has_predict_method(model):
    """Model should have predict method."""
    assert hasattr(model, "predict")

def test_model_has_predict_proba_method(model):
    """Model should have predict_proba method."""
    assert hasattr(model, "predict_proba")

# ── Predictions ──
def test_model_predict_returns_array(model):
    """Model predict should return a numpy array."""
    result = model.predict(VALID_INPUT)
    assert isinstance(result, np.ndarray)

def test_model_predict_valid_label(model):
    """Model prediction should be 0 or 1."""
    result = model.predict(VALID_INPUT)
    assert result[0] in [0, 1]

def test_model_predict_proba_shape(model):
    """Predict proba should return 2 classes."""
    result = model.predict_proba(VALID_INPUT)
    assert result.shape == (1, 2)

def test_model_predict_proba_sum(model):
    """Probabilities should sum to 1."""
    result = model.predict_proba(VALID_INPUT)
    assert abs(result[0].sum() - 1.0) < 1e-6

def test_model_features_count(model):
    """Model should expect 10 features."""
    assert model.n_features_ == 10

def test_model_batch_prediction(model):
    """Model should handle batch inputs."""
    batch = np.repeat(VALID_INPUT, 5, axis=0)
    result = model.predict(batch)
    assert len(result) == 5