from sqlalchemy import text
from src.logger import get_logger

logger = get_logger(__name__)

"""
Database query module — SQL queries for prediction history analytics.

Database : SQLite (development) — sqlite:///predictions.db
           PostgreSQL (production) — configurable via SQLALCHEMY_DATABASE_URI
ORM      : Flask-SQLAlchemy
Table    : prediction (see Prediction model in application.py)

Usage:
    from src.db_queries import get_total_predictions
    total = get_total_predictions(db)  # db = SQLAlchemy instance from application.py
"""

def get_total_predictions(db):
    """Total number of predictions stored in the database."""
    result = db.session.execute(
        text("SELECT COUNT(*) FROM prediction")
    ).scalar()
    return result or 0


def get_cancelled_count(db):
    """Number of cancelled predictions."""
    result = db.session.execute(
        text("SELECT COUNT(*) FROM prediction WHERE prediction = 'Annulé'")
    ).scalar()
    return result or 0


def get_not_cancelled_count(db):
    """Number of not cancelled predictions."""
    result = db.session.execute(
        text("SELECT COUNT(*) FROM prediction WHERE prediction = 'Non annulé'")
    ).scalar()
    return result or 0


def get_avg_confidence(db):
    """Average prediction confidence across all predictions."""
    result = db.session.execute(
        text("SELECT ROUND(AVG(probability) * 100, 1) FROM prediction")
    ).scalar()
    return float(result) if result else 0.0


def get_avg_latency(db):
    """Average model latency in milliseconds."""
    result = db.session.execute(
        text("""
            SELECT ROUND(AVG(latency_ms), 2)
            FROM prediction
            WHERE latency_ms IS NOT NULL
        """)
    ).scalar()
    return float(result) if result else 0.0


def get_today_count(db):
    """Number of predictions made today."""
    result = db.session.execute(
        text("""
            SELECT COUNT(*)
            FROM prediction
            WHERE DATE(timestamp) = DATE('now')
        """)
    ).scalar()
    return result or 0


def get_confidence_distribution(db):
    """
    Distribution of prediction confidence in 5 buckets.
    Returns list of counts: [50-60%, 60-70%, 70-80%, 80-90%, 90-100%]
    """
    result = db.session.execute(
        text("""
            SELECT
                SUM(CASE WHEN probability*100 < 60 THEN 1 ELSE 0 END),
                SUM(CASE WHEN probability*100 >= 60 AND probability*100 < 70 THEN 1 ELSE 0 END),
                SUM(CASE WHEN probability*100 >= 70 AND probability*100 < 80 THEN 1 ELSE 0 END),
                SUM(CASE WHEN probability*100 >= 80 AND probability*100 < 90 THEN 1 ELSE 0 END),
                SUM(CASE WHEN probability*100 >= 90 THEN 1 ELSE 0 END)
            FROM prediction
        """)
    ).fetchone()
    return [int(v or 0) for v in result]


def get_timeline_data(db):
    """
    Predictions count per day for last 7 days, split by result.
    Returns dict with labels, cancelled counts, not_cancelled counts.
    """
    result = db.session.execute(
        text("""
            SELECT
                DATE(timestamp) as day,
                SUM(CASE WHEN prediction = 'Annulé' THEN 1 ELSE 0 END) as cancelled,
                SUM(CASE WHEN prediction = 'Non annulé' THEN 1 ELSE 0 END) as not_cancelled
            FROM prediction
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY day ASC
        """)
    ).fetchall()

    labels = [str(r[0]) for r in result]
    cancelled = [int(r[1]) for r in result]
    not_cancelled = [int(r[2]) for r in result]
    return labels, cancelled, not_cancelled


def get_market_segment_distribution(db):
    """
    Count of predictions per market segment type.
    Returns dict with segment labels and counts.
    """
    result = db.session.execute(
        text("""
            SELECT market_segment_type, COUNT(*) as total
            FROM prediction
            GROUP BY market_segment_type
            ORDER BY market_segment_type ASC
        """)
    ).fetchall()

    market_names = {
        0: "Aviation", 1: "Complémentaire", 2: "Corporate",
        3: "Hors ligne", 4: "En ligne"
    }
    labels = [market_names.get(int(r[0]), str(r[0])) for r in result]
    counts = [int(r[1]) for r in result]
    return labels, counts


def get_avg_price_by_result(db):
    """
    Average room price grouped by prediction result.
    Returns [avg_price_cancelled, avg_price_not_cancelled].
    """
    result = db.session.execute(
        text("""
            SELECT
                ROUND(AVG(CASE WHEN prediction = 'Annulé'
                    THEN avg_price_per_room END), 2),
                ROUND(AVG(CASE WHEN prediction = 'Non annulé'
                    THEN avg_price_per_room END), 2)
            FROM prediction
        """)
    ).fetchone()
    return [float(result[0] or 0), float(result[1] or 0)]


def get_recent_predictions(db, limit=5):
    """
    Fetch the most recent predictions ordered by timestamp.
    """
    result = db.session.execute(
        text("""
            SELECT id, timestamp, lead_time, avg_price_per_room,
                   prediction, probability
            FROM prediction
            ORDER BY timestamp DESC
            LIMIT :limit
        """),
        {"limit": limit}
    ).fetchall()
    return result


def get_full_history(db, limit=50):
    """
    Fetch full prediction history with all relevant columns.
    """
    result = db.session.execute(
        text("""
            SELECT id, timestamp, lead_time, avg_price_per_room,
                   arrival_month, market_segment_type,
                   prediction, probability
            FROM prediction
            ORDER BY timestamp DESC
            LIMIT :limit
        """),
        {"limit": limit}
    ).fetchall()
    return result