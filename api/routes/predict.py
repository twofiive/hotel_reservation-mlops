from flask import Blueprint, jsonify, request
from api.auth import require_api_key
from prometheus_client import Counter, Histogram
import joblib
import numpy as np
import time
from src.logger import get_logger
from config.paths_config import MODEL_OUTPUT_PATH

logger = get_logger(__name__)
predict_bp = Blueprint("predict", __name__)

# Load model once at startup
model = joblib.load(MODEL_OUTPUT_PATH)

# ── Prometheus metrics (C11) ──
PREDICTION_COUNTER = Counter(
    "hotel_predictions_total",
    "Total number of predictions made",
    ["result"]
)
PREDICTION_LATENCY = Histogram(
    "hotel_prediction_latency_seconds",
    "Prediction latency in seconds"
)
REQUEST_COUNTER = Counter(
    "hotel_api_requests_total",
    "Total API requests",
    ["endpoint", "status"]
)

FEATURES = [
    "lead_time", "no_of_special_requests", "avg_price_per_room",
    "arrival_month", "arrival_date", "market_segment_type",
    "no_of_week_nights", "no_of_weekend_nights",
    "type_of_meal_plan", "room_type_reserved"
]

LABELS = {0: "Not Cancelled", 1: "Cancelled"}

@predict_bp.route("/predict", methods=["POST"])
@require_api_key
def predict():
    """
    Predict hotel reservation cancellation.
    
    Body (JSON):
        lead_time, no_of_special_requests, avg_price_per_room,
        arrival_month, arrival_date, market_segment_type,
        no_of_week_nights, no_of_weekend_nights,
        type_of_meal_plan, room_type_reserved
    
    Returns:
        prediction: "Cancelled" or "Not Cancelled"
        probability: float
        latency_ms: float
    """
    start_time = time.time()

    try:
        data = request.get_json()
        if not data:
            REQUEST_COUNTER.labels(endpoint="/predict", status="400").inc()
            return jsonify({"error": "No JSON body provided"}), 400

        # Validate required features
        missing = [f for f in FEATURES if f not in data]
        if missing:
            REQUEST_COUNTER.labels(endpoint="/predict", status="400").inc()
            return jsonify({
                "error": "Missing features",
                "missing": missing
            }), 400

        # Build input array
        input_array = np.array([[data[f] for f in FEATURES]])

        # Predict
        with PREDICTION_LATENCY.time():
            pred = model.predict(input_array)[0]
            proba = model.predict_proba(input_array)[0].max()

        result = LABELS.get(int(pred), "Unknown")
        latency_ms = (time.time() - start_time) * 1000

        # Update Prometheus counters
        PREDICTION_COUNTER.labels(result=result).inc()
        REQUEST_COUNTER.labels(endpoint="/predict", status="200").inc()

        logger.info(f"Prediction: {result} | Proba: {proba:.4f} | Latency: {latency_ms:.2f}ms")

        return jsonify({
            "prediction": result,
            "probability": round(float(proba), 4),
            "latency_ms": round(latency_ms, 2),
            "features_used": FEATURES
        }), 200

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@predict_bp.route("/predict/batch", methods=["POST"])
@require_api_key
def predict_batch():
    """
    Batch prediction for multiple reservations.
    
    Body (JSON):
        {"reservations": [{"lead_time": ..., ...}, ...]}
    
    Returns:
        List of predictions with probabilities
    """
    start_time = time.time()

    try:
        data = request.get_json()
        reservations = data.get("reservations", [])

        if not reservations:
            return jsonify({"error": "No reservations provided"}), 400

        results = []
        for i, res in enumerate(reservations):
            missing = [f for f in FEATURES if f not in res]
            if missing:
                results.append({
                    "index": i,
                    "error": f"Missing features: {missing}"
                })
                continue

            input_array = np.array([[res[f] for f in FEATURES]])
            pred = model.predict(input_array)[0]
            proba = model.predict_proba(input_array)[0].max()
            result = LABELS.get(int(pred), "Unknown")

            PREDICTION_COUNTER.labels(result=result).inc()
            results.append({
                "index": i,
                "prediction": result,
                "probability": round(float(proba), 4)
            })

        latency_ms = (time.time() - start_time) * 1000
        REQUEST_COUNTER.labels(endpoint="/predict/batch", status="200").inc()

        return jsonify({
            "total": len(results),
            "latency_ms": round(latency_ms, 2),
            "results": results
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500