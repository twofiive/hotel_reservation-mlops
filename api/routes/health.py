from flask import Blueprint, jsonify
import joblib
import os
from config.paths_config import MODEL_OUTPUT_PATH

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint — no auth required."""
    model_exists = os.path.exists(MODEL_OUTPUT_PATH)
    return jsonify({
        "status": "ok" if model_exists else "degraded",
        "model_loaded": model_exists,
        "service": "Hotel Reservation Prediction API",
        "version": "1.0.0"
    }), 200 if model_exists else 503