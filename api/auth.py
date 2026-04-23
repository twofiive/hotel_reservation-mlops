from functools import wraps
from flask import request, jsonify
import os

# API key loaded from environment variable
API_KEY = os.getenv("API_KEY", "hotel-dev-key-2024")

def require_api_key(f):
    """
    Decorator to protect endpoints with an API key.
    Key must be passed in header: X-API-Key: <key>
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Missing or invalid API key"
            }), 401
        return f(*args, **kwargs)
    return decorated