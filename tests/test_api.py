import pytest
import json
from api.app import create_app

@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

VALID_PAYLOAD = {
    "lead_time": 50,
    "no_of_special_requests": 1,
    "avg_price_per_room": 100.0,
    "arrival_month": 6,
    "arrival_date": 15,
    "market_segment_type": 1,
    "no_of_week_nights": 3,
    "no_of_weekend_nights": 1,
    "type_of_meal_plan": 1,
    "room_type_reserved": 1
}

HEADERS = {"X-API-Key": "hotel-dev-key-2024"}

# ── Health check ──
def test_health_returns_200(client):
    """Health endpoint should return 200 when model exists."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200

def test_health_response_structure(client):
    """Health response should contain required fields."""
    r = client.get("/api/v1/health")
    data = json.loads(r.data)
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data

# ── Authentication ──
def test_predict_without_api_key_returns_401(client):
    """Prediction without API key should return 401."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD
    )
    assert r.status_code == 401

def test_predict_with_invalid_api_key_returns_401(client):
    """Prediction with wrong API key should return 401."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers={"X-API-Key": "wrong-key"}
    )
    assert r.status_code == 401

# ── Prediction ──
def test_predict_valid_input_returns_200(client):
    """Valid prediction request should return 200."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers=HEADERS
    )
    assert r.status_code == 200

def test_predict_response_structure(client):
    """Prediction response should contain required fields."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers=HEADERS
    )
    data = json.loads(r.data)
    assert "prediction" in data
    assert "probability" in data
    assert "latency_ms" in data

def test_predict_valid_label(client):
    """Prediction result should be a valid label."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers=HEADERS
    )
    data = json.loads(r.data)
    assert data["prediction"] in ["Cancelled", "Not Cancelled"]

def test_predict_probability_range(client):
    """Probability should be between 0 and 1."""
    r = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
        headers=HEADERS
    )
    data = json.loads(r.data)
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_missing_features_returns_400(client):
    """Missing features should return 400."""
    r = client.post(
        "/api/v1/predict",
        json={"lead_time": 50},
        headers=HEADERS
    )
    assert r.status_code == 400

def test_predict_empty_body_returns_400(client):
    """Empty body should return 400."""
    r = client.post(
        "/api/v1/predict",
        json={},
        headers=HEADERS
    )
    assert r.status_code == 400

# ── Batch prediction ──
def test_batch_predict_returns_200(client):
    """Batch prediction should return 200."""
    r = client.post(
        "/api/v1/predict/batch",
        json={"reservations": [VALID_PAYLOAD, VALID_PAYLOAD]},
        headers=HEADERS
    )
    assert r.status_code == 200

def test_batch_predict_correct_count(client):
    """Batch prediction should return correct number of results."""
    r = client.post(
        "/api/v1/predict/batch",
        json={"reservations": [VALID_PAYLOAD, VALID_PAYLOAD]},
        headers=HEADERS
    )
    data = json.loads(r.data)
    assert data["total"] == 2
    assert len(data["results"]) == 2

def test_batch_predict_empty_returns_400(client):
    """Empty batch should return 400."""
    r = client.post(
        "/api/v1/predict/batch",
        json={"reservations": []},
        headers=HEADERS
    )
    assert r.status_code == 400