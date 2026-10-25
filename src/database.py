"""
Database models and storage utilities for AI Image Detector.
Handles logging of user image submissions, audit metadata, and rate limiting.
"""

from datetime import datetime, timedelta
import os
import shutil
from typing import Dict, Any, Tuple, Optional
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

db = SQLAlchemy()

UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
THUMBNAILS_DIR = os.path.join(UPLOADS_DIR, "thumbnails")


class PredictionLog(db.Model):
    """Stores all prediction requests and uploaded image metadata."""
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    uuid_filename = db.Column(db.String(64), unique=True, nullable=False, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, default=0)  # in bytes
    mime_type = db.Column(db.String(64), default="image/jpeg")
    prediction = db.Column(db.String(32), nullable=False, index=True)  # 'Real' or 'AI-generated'
    confidence = db.Column(db.Float, nullable=False)  # e.g. 98.5
    prob_real = db.Column(db.Float, default=0.0)
    prob_fake = db.Column(db.Float, default=0.0)
    ip_address = db.Column(db.String(64), default="127.0.0.1", index=True)
    session_id = db.Column(db.String(64), default="", index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Dataset ingestion tracking
    training_status = db.Column(db.String(32), default="none")  # 'none', 'train_real', 'train_fake'
    training_moved_at = db.Column(db.DateTime, nullable=True)

    # User Feedback on Prediction
    user_feedback = db.Column(db.String(32), default="none")  # 'none', 'agree', 'disagree'
    user_actual_label = db.Column(db.String(32), default="")  # 'Real' or 'AI-generated' if reported by user
    user_comment = db.Column(db.String(500), default="")      # Optional feedback remarks
    feedback_timestamp = db.Column(db.DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "uuid_filename": self.uuid_filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_size_formatted": self.formatted_size(),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "prob_real": self.prob_real,
            "prob_fake": self.prob_fake,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if self.timestamp else "",
            "training_status": self.training_status,
            "user_feedback": self.user_feedback,
            "user_actual_label": self.user_actual_label,
            "user_comment": self.user_comment,
            "feedback_timestamp": self.feedback_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if self.feedback_timestamp else ""
        }

    def formatted_size(self) -> str:
        """Return human-readable file size."""
        size = self.file_size or 0
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"


class LoginAttempt(db.Model):
    """Tracks failed login attempts for brute-force rate limiting and lockout."""
    __tablename__ = "login_attempts"

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(64), index=True, nullable=False)
    username = db.Column(db.String(64), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    success = db.Column(db.Boolean, default=False)


class GeneralFeedback(db.Model):
    """Stores user thoughts, bug reports, ratings, and feature suggestions."""
    __tablename__ = "general_feedback"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), default="Anonymous")
    email = db.Column(db.String(120), default="")
    category = db.Column(db.String(64), default="General")  # 'General', 'Accuracy', 'Feature Request', 'Bug Report'
    rating = db.Column(db.Integer, default=5)  # 1 to 5 stars
    message = db.Column(db.Text, nullable=False)
    ip_address = db.Column(db.String(64), default="127.0.0.1")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "category": self.category,
            "rating": self.rating,
            "message": self.message,
            "ip_address": self.ip_address,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if self.timestamp else ""
        }


def init_db(app):
    """Initialize database and ensure directories exist."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        # Enable WAL mode on SQLite for high-concurrency read/write
        engine = db.engine
        if "sqlite" in engine.url.drivername:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        db.create_all()


def record_prediction(
    uuid_filename: str,
    original_filename: str,
    file_size: int,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    ip_address: str,
    session_id: str = "",
    mime_type: str = "image/jpeg"
) -> PredictionLog:
    """Save an inference result and upload metadata to SQLite."""
    prob_real = float(probabilities.get("Real", 0.0))
    prob_fake = float(probabilities.get("AI-generated", 0.0))

    log_entry = PredictionLog(
        uuid_filename=uuid_filename,
        original_filename=original_filename,
        file_size=file_size,
        mime_type=mime_type,
        prediction=prediction,
        confidence=confidence,
        prob_real=prob_real,
        prob_fake=prob_fake,
        ip_address=ip_address,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        training_status="none"
    )
    db.session.add(log_entry)
    db.session.commit()
    return log_entry


def is_ip_locked_out(ip_address: str, max_attempts: int = 5, lockout_minutes: int = 15) -> Tuple[bool, int]:
    """
    Check if an IP address is currently locked out due to consecutive failed attempts.
    Returns: (is_locked, remaining_seconds_locked)
    """
    cutoff = datetime.utcnow() - timedelta(minutes=lockout_minutes)
    
    # Check attempts since cutoff
    attempts = LoginAttempt.query.filter(
        LoginAttempt.ip_address == ip_address,
        LoginAttempt.timestamp >= cutoff
    ).order_by(LoginAttempt.timestamp.desc()).all()

    consecutive_fails = 0
    latest_fail_time = None
    for att in attempts:
        if att.success:
            break
        consecutive_fails += 1
        if latest_fail_time is None:
            latest_fail_time = att.timestamp

    if consecutive_fails >= max_attempts and latest_fail_time is not None:
        unlock_time = latest_fail_time + timedelta(minutes=lockout_minutes)
        remaining = int((unlock_time - datetime.utcnow()).total_seconds())
        if remaining > 0:
            return True, remaining

    return False, 0


def get_failed_attempts_count(ip_address: str, lockout_minutes: int = 15) -> int:
    """Return consecutive failed login attempts for this IP in the current window."""
    cutoff = datetime.utcnow() - timedelta(minutes=lockout_minutes)
    attempts = LoginAttempt.query.filter(
        LoginAttempt.ip_address == ip_address,
        LoginAttempt.timestamp >= cutoff
    ).order_by(LoginAttempt.timestamp.desc()).all()

    count = 0
    for att in attempts:
        if att.success:
            break
        count += 1
    return count


def record_login_attempt(ip_address: str, username: str, success: bool):
    """Log an admin login attempt for security auditing and lockout tracking."""
    attempt = LoginAttempt(
        ip_address=ip_address,
        username=username,
        timestamp=datetime.utcnow(),
        success=success
    )
    db.session.add(attempt)
    db.session.commit()

    if success:
        # Clear previous failed attempts to reset the counter on successful login
        LoginAttempt.query.filter_by(ip_address=ip_address, success=False).delete()
        db.session.commit()


def get_summary_statistics() -> Dict[str, Any]:
    """Compute aggregate counts, feedback metrics, and percentage breakdown for admin dashboard."""
    total = PredictionLog.query.count()
    real_count = PredictionLog.query.filter_by(prediction="Real").count()
    fake_count = PredictionLog.query.filter_by(prediction="AI-generated").count()
    
    avg_confidence = db.session.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
    
    real_pct = round((real_count / total * 100.0), 1) if total > 0 else 0.0
    fake_pct = round((fake_count / total * 100.0), 1) if total > 0 else 0.0

    # User feedback metrics
    feedback_agree = PredictionLog.query.filter_by(user_feedback="agree").count()
    feedback_disagree = PredictionLog.query.filter_by(user_feedback="disagree").count()
    feedback_total = feedback_agree + feedback_disagree
    agreement_rate = round((feedback_agree / feedback_total * 100.0), 1) if feedback_total > 0 else 0.0
    general_feedback_count = GeneralFeedback.query.count()

    # Submissions grouped by date for chart (last 14 days)
    daily_query = db.session.query(
        func.date(PredictionLog.timestamp).label("date"),
        func.count(PredictionLog.id).label("count")
    ).group_by(func.date(PredictionLog.timestamp))\
     .order_by(func.date(PredictionLog.timestamp).asc())\
     .limit(14).all()

    chart_dates = [row.date for row in daily_query] if daily_query else []
    chart_counts = [row.count for row in daily_query] if daily_query else []

    # If empty, provide today as 0
    if not chart_dates:
        chart_dates = [datetime.utcnow().strftime("%Y-%m-%d")]
        chart_counts = [0]

    return {
        "total_submissions": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_percentage": real_pct,
        "fake_percentage": fake_pct,
        "average_confidence": round(float(avg_confidence), 1),
        "feedback_total": feedback_total,
        "feedback_agree": feedback_agree,
        "feedback_disagree": feedback_disagree,
        "agreement_rate": agreement_rate,
        "general_feedback_count": general_feedback_count,
        "chart_dates": chart_dates,
        "chart_counts": chart_counts
    }


def record_prediction_feedback(log_id: int, feedback: str, actual_label: str = "", comment: str = "") -> Tuple[bool, str]:
    """Store user feedback for a specific prediction result."""
    log = db.session.get(PredictionLog, log_id)
    if not log:
        return False, "Prediction record not found."
    
    clean_fb = (feedback or "").strip().lower()
    if clean_fb not in ("agree", "disagree"):
        return False, "Feedback must be 'agree' or 'disagree'."

    log.user_feedback = clean_fb
    log.user_actual_label = (actual_label or "").strip()[:32]
    log.user_comment = (comment or "").strip()[:500]
    log.feedback_timestamp = datetime.utcnow()
    
    try:
        db.session.commit()
        return True, "Feedback recorded successfully."
    except Exception as e:
        db.session.rollback()
        return False, f"Database error: {str(e)}"


def record_general_feedback(name: str, email: str, category: str, rating: int, message: str, ip_address: str) -> Tuple[bool, str]:
    """Save user suggestion, review, or bug report."""
    msg = (message or "").strip()
    if not msg:
        return False, "Feedback message cannot be empty."

    rating_val = max(1, min(5, int(rating or 5)))
    clean_category = (category or "General").strip()[:64]
    clean_name = (name or "Anonymous").strip()[:100] or "Anonymous"
    clean_email = (email or "").strip()[:120]

    entry = GeneralFeedback(
        name=clean_name,
        email=clean_email,
        category=clean_category,
        rating=rating_val,
        message=msg[:2000],
        ip_address=ip_address,
        timestamp=datetime.utcnow()
    )

    try:
        db.session.add(entry)
        db.session.commit()
        return True, "Feedback received. Thank you for helping us improve!"
    except Exception as e:
        db.session.rollback()
        return False, f"Failed to save feedback: {str(e)}"


def move_image_to_dataset(log_id: int, target_class: str, database_dir: str = "database") -> Tuple[bool, str]:
    """
    Copy a submitted image from uploads/ to database/train/REAL or database/train/FAKE.
    Updates the log entry to track that it was added to training data.
    """
    log = PredictionLog.query.get(log_id)
    if not log:
        return False, "Submission record not found."

    src_path = os.path.join(UPLOADS_DIR, log.uuid_filename)
    if not os.path.exists(src_path):
        return False, f"Image file '{log.uuid_filename}' not found in uploads."

    class_clean = target_class.lower()
    if class_clean not in ("real", "fake"):
        return False, f"Invalid target class: {target_class}. Must be 'real' or 'fake'."

    # Find or create target directory
    train_dir = os.path.join(database_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    dest_folder = None
    for entry in os.listdir(train_dir):
        if entry.lower() == class_clean and os.path.isdir(os.path.join(train_dir, entry)):
            dest_folder = os.path.join(train_dir, entry)
            break

    if not dest_folder:
        dest_folder = os.path.join(train_dir, class_clean.upper())
        os.makedirs(dest_folder, exist_ok=True)

    # Name the file uniquely in database
    _, ext = os.path.splitext(log.original_filename or log.uuid_filename)
    if not ext:
        ext = ".jpg"
    dest_filename = f"user_sub_{log.id}_{log.uuid_filename[:12]}{ext}"
    dest_path = os.path.join(dest_folder, dest_filename)

    try:
        shutil.copy2(src_path, dest_path)
        log.training_status = f"train_{class_clean}"
        log.training_moved_at = datetime.utcnow()
        db.session.commit()
        return True, f"Successfully added to {os.path.relpath(dest_folder)} as '{dest_filename}'."
    except Exception as e:
        return False, f"Failed to copy file: {str(e)}"
