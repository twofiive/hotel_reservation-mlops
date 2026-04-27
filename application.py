import joblib
import numpy as np
from datetime import datetime, date
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from config.paths_config import MODEL_OUTPUT_PATH
from src.logger import get_logger
from src.db_queries import (
    get_total_predictions, get_cancelled_count, get_not_cancelled_count,
    get_avg_confidence, get_avg_latency, get_today_count,
    get_confidence_distribution, get_timeline_data,
    get_market_segment_distribution, get_avg_price_by_result
)


logger = get_logger(__name__)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

LABELS = {0: "Non annulé", 1: "Annulé"}
MONTHS = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}
MARKET_SEGMENTS = {
    0: "Aviation", 1: "Complémentaire", 2: "Corporate",
    3: "Hors ligne", 4: "En ligne"
}
MEAL_PLANS = {
    0: "Formule 1", 1: "Formule 2",
    2: "Formule 3", 3: "Non sélectionné"
}
ROOM_TYPES = {
    0: "Chambre Type 1", 1: "Chambre Type 2",
    2: "Chambre Type 3", 3: "Chambre Type 4",
    4: "Chambre Type 5", 5: "Chambre Type 6",
    6: "Chambre Type 7"
}


class Prediction(db.Model):
    """Stocke l'historique des prédictions."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    lead_time = db.Column(db.Integer)
    no_of_special_requests = db.Column(db.Integer)
    avg_price_per_room = db.Column(db.Float)
    arrival_month = db.Column(db.Integer)
    arrival_date = db.Column(db.Integer)
    market_segment_type = db.Column(db.Integer)
    no_of_week_nights = db.Column(db.Integer)
    no_of_weekend_nights = db.Column(db.Integer)
    type_of_meal_plan = db.Column(db.Integer)
    room_type_reserved = db.Column(db.Integer)
    prediction = db.Column(db.String(20))
    probability = db.Column(db.Float)
    latency_ms = db.Column(db.Float)


with app.app_context():
    db.create_all()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    probability = None
    error = None
    latency_ms = None

    if request.method == "POST":
        try:
            start = datetime.utcnow()

            lead_time = int(request.form["lead_time"])
            no_of_special_requests = int(request.form["no_of_special_requests"])
            avg_price_per_room = float(request.form["avg_price_per_room"])
            arrival_month = int(request.form["arrival_month"])
            arrival_date = int(request.form["arrival_date"])
            market_segment_type = int(request.form["market_segment_type"])
            no_of_week_nights = int(request.form["no_of_week_nights"])
            no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
            type_of_meal_plan = int(request.form["type_of_meal_plan"])
            room_type_reserved = int(request.form["room_type_reserved"])

            features = np.array([[
                lead_time, no_of_special_requests, avg_price_per_room,
                arrival_month, arrival_date, market_segment_type,
                no_of_week_nights, no_of_weekend_nights,
                type_of_meal_plan, room_type_reserved
            ]])

            pred = loaded_model.predict(features)[0]
            proba = loaded_model.predict_proba(features)[0].max()
            prediction_result = LABELS.get(int(pred), "Inconnu")
            probability = round(float(proba) * 100, 1)
            latency_ms = round(
                (datetime.utcnow() - start).total_seconds() * 1000, 2
            )

            record = Prediction(
                lead_time=lead_time,
                no_of_special_requests=no_of_special_requests,
                avg_price_per_room=avg_price_per_room,
                arrival_month=arrival_month,
                arrival_date=arrival_date,
                market_segment_type=market_segment_type,
                no_of_week_nights=no_of_week_nights,
                no_of_weekend_nights=no_of_weekend_nights,
                type_of_meal_plan=type_of_meal_plan,
                room_type_reserved=room_type_reserved,
                prediction=prediction_result,
                probability=round(float(proba), 4),
                latency_ms=latency_ms
            )
            db.session.add(record)
            db.session.commit()
            logger.info(
                f"Prédiction: {prediction_result} "
                f"({probability}%) en {latency_ms}ms"
            )

        except Exception as e:
            error = str(e)
            logger.error(f"Erreur prédiction: {e}")

    history = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).limit(5).all()

    return render_template(
        "index.html",
        prediction=prediction_result,
        probability=probability,
        latency_ms=latency_ms,
        error=error,
        history=history,
        months=MONTHS,
        market_segments=MARKET_SEGMENTS,
        meal_plans=MEAL_PLANS,
        room_types=ROOM_TYPES
    )


@app.route("/history")
def history():
    predictions = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).limit(50).all()
    return render_template(
        "history.html",
        predictions=predictions,
        months=MONTHS
    )


@app.route("/monitoring")
def monitoring():
    """Page de monitoring — métriques calculées via requêtes SQL explicites."""

    stats = {
        "total": get_total_predictions(db),
        "cancelled": get_cancelled_count(db),
        "not_cancelled": get_not_cancelled_count(db),
        "avg_confidence": get_avg_confidence(db),
        "avg_latency": get_avg_latency(db),
        "today": get_today_count(db)
    }

    timeline_labels, timeline_cancelled, timeline_not_cancelled = get_timeline_data(db)
    market_labels, market_counts = get_market_segment_distribution(db)

    chart_data = {
        "confidence_dist": get_confidence_distribution(db),
        "timeline_labels": timeline_labels,
        "timeline_cancelled": timeline_cancelled,
        "timeline_not_cancelled": timeline_not_cancelled,
        "market_labels": market_labels,
        "market_counts": market_counts,
        "avg_price_by_result": get_avg_price_by_result(db)
    }

    return render_template(
        "monitoring.html",
        stats=stats,
        chart_data=chart_data
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "loaded",
        "version": "1.0.0"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)