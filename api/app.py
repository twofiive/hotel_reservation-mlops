from flask import Flask
from api.routes.predict import predict_bp
from api.routes.health import health_bp
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

def create_app():
    """Flask application factory."""
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp, url_prefix="/api/v1")

    # Expose Prometheus metrics at /metrics
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        "/metrics": make_wsgi_app()
    })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)