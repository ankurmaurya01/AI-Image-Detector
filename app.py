"""
AI-Generated Image Detector - Web Application & Secure Admin Console
Flask web application featuring:
1. Public Real vs. AI-Generated image detection with high-concurrency serialized inference.
2. Full logging of submissions (randomized UUID storage, SQLite audit log, client metadata).
3. Secure, single-operator admin dashboard (/admin) with filtering, statistics, CSV export,
   thumbnail viewer, training dataset ingestion, CSRF protection, and brute-force lockout.
"""

from datetime import datetime, timedelta
import csv
import gc
import io
import json
import os
import sys
import tempfile
import threading
import uuid
import webbrowser

from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    flash, send_from_directory, Response, abort
)
from flask_login import login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from PIL import Image

# Restrict multithreading to stay under 512MB RAM on Render / shared hosts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import load_config, get_device
from src.predict import DetectorPredictor
from src.data_loader import safe_image_loader, get_transforms
from src.model import build_model
from src.database import (
    db, init_db, PredictionLog, UPLOADS_DIR, THUMBNAILS_DIR,
    record_prediction, is_ip_locked_out, record_login_attempt,
    get_summary_statistics, move_image_to_dataset, get_failed_attempts_count
)
from src.auth import init_auth, AdminUser, verify_admin_credentials
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

# Application Configuration
app = Flask(__name__)
application = app  # WSGI compatibility alias

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-insecure-secret-key-change-in-env")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.abspath("admin_data.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))  # 10 MB limit

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Initialize CSRF Protection and Database/Auth extensions
csrf = CSRFProtect(app)
init_db(app)
init_auth(app)

PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

config = load_config("config.yaml")
weights_path = "models/inference_model.pth" if os.path.exists("models/inference_model.pth") else config["paths"].get("best_model_path", "models/best_model.pth")

# Global predictor reference & lock for handling concurrent users without RAM multiplication
_predictor = None
_predict_lock = threading.Lock()


def get_model():
    """Lazy-load the model to enable instant server port binding on cloud hosts."""
    global _predictor
    if _predictor is not None:
        return _predictor

    torch.set_num_threads(1)

    try:
        if os.path.exists(weights_path):
            _predictor = DetectorPredictor(config=config, weights_path=weights_path)
            print(f"[INFO] Loaded trained model from {weights_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{weights_path}'")
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint ({e}). Initializing fallback mode...")

        class FallbackPredictor:
            def __init__(self, cfg):
                self.config = cfg
                self.device = get_device("auto")
                cfg_copy = dict(cfg)
                if "model" in cfg_copy:
                    cfg_copy["model"] = dict(cfg_copy["model"])
                    cfg_copy["model"]["pretrained"] = False
                self.model = build_model(cfg_copy).to(self.device)
                self.model.eval()
                self.transform = get_transforms(cfg, split="test")

            def predict(self, image_path):
                pil_img = safe_image_loader(image_path)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs.float(), dim=1).squeeze(0).cpu().numpy()
                real_prob = float(probs[0]) * 100
                fake_prob = float(probs[1]) * 100
                label = "AI-generated" if fake_prob >= 50.0 else "Real"
                confidence = fake_prob if label == "AI-generated" else real_prob
                return {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "probabilities": {
                        "Real": round(real_prob, 2),
                        "AI-generated": round(fake_prob, 2),
                    }
                }

        _predictor = FallbackPredictor(config)
        print("[INFO] Fallback model initialized successfully.")

    gc.collect()
    return _predictor


def get_client_ip() -> str:
    """Extract real client IP behind reverse proxies (Render, Cloudflare, etc.)."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


# ==============================================================================
# PUBLIC ROUTES
# ==============================================================================

@app.route("/", methods=["GET"])
def home():
    """Public Image Detector interface (contains no link to /admin)."""
    return render_template("index.html")


@app.route("/index.html", methods=["GET"])
def index_html():
    return redirect(url_for("home"))


@app.route("/predict", methods=["POST"])
@csrf.exempt  # Exempt public image submission API from CSRF
def predict():
    """
    Public Prediction Endpoint:
    1. Validates file format and image integrity.
    2. Runs neural network inference with concurrency lock.
    3. Securely logs the image and metadata (UUID filename) into SQLite database.
    4. Returns prediction verdict and confidence percentages.
    """
    pred = get_model()
    if pred is None:
        return jsonify({"error": "Model initialization failed."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided in upload request."}), 400

    file = request.files["image"]
    original_filename = file.filename or "upload.jpg"
    if original_filename.strip() == "":
        return jsonify({"error": "Empty filename."}), 400

    # Extension check
    _, ext = os.path.splitext(original_filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Generate random unique filename for storage (never use original filename)
    uuid_filename = f"{uuid.uuid4().hex}{ext}"
    permanent_upload_path = os.path.join(UPLOADS_DIR, uuid_filename)

    # Temporary working file for PIL validation and prediction
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_path = tmp.name
    tmp.close()

    try:
        # Save user stream to disk
        file.save(tmp_path)
        file_size = os.path.getsize(tmp_path)

        # Image content integrity check using PIL
        try:
            with Image.open(tmp_path) as img:
                img.verify()
        except Exception:
            return jsonify({"error": "Uploaded file is corrupted or not a valid image."}), 400

        # Copy verified image to uploads/ with randomized UUID
        try:
            with open(tmp_path, "rb") as src_f, open(permanent_upload_path, "wb") as dst_f:
                dst_f.write(src_f.read())
        except Exception as e:
            print(f"[WARNING] Could not save upload copy: {e}")

        # Serialized inference: multiple users queue smoothly without spiking RAM beyond 512MB
        with _predict_lock:
            result = pred.predict(tmp_path)

        # Log metadata in SQLite database
        client_ip = get_client_ip()
        session_id = request.cookies.get("session_id") or uuid.uuid4().hex[:16]

        try:
            record_prediction(
                uuid_filename=uuid_filename,
                original_filename=secure_filename(original_filename),
                file_size=file_size,
                prediction=result.get("label", "Unknown"),
                confidence=result.get("confidence", 0.0),
                probabilities=result.get("probabilities", {}),
                ip_address=client_ip,
                session_id=session_id,
                mime_type=file.mimetype or "image/jpeg"
            )
        except Exception as db_err:
            print(f"[ERROR] Failed to log prediction to database: {db_err}")

        gc.collect()

        response = jsonify(result)
        # Ensure session ID cookie is set on client for audit correlation
        if not request.cookies.get("session_id"):
            response.set_cookie("session_id", session_id, max_age=30*24*3600, httponly=True, samesite="Lax")
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e) or repr(e)}), 400
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ==============================================================================
# SECURE ADMIN AUTHENTICATION ROUTES
# ==============================================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """
    Administrator Sign-In:
    - Session-based authentication (single owner account from .env).
    - Rate-limiting: locks out after 5 consecutive failed attempts for 15 minutes.
    - CSRF protection enabled.
    """
    if current_user.is_authenticated:
        return redirect(url_for("admin_dashboard"))

    client_ip = get_client_ip()
    is_locked, remaining_secs = is_ip_locked_out(client_ip, max_attempts=5, lockout_minutes=15)
    remaining_minutes = int(remaining_secs / 60) + 1 if is_locked else 0

    if request.method == "POST":
        if is_locked:
            flash(f"Access blocked due to excessive failed attempts. Please wait {remaining_minutes} minute(s).", "danger")
            return render_template("admin/login.html", is_locked=True, remaining_minutes=remaining_minutes)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if verify_admin_credentials(username, password):
            record_login_attempt(client_ip, username, success=True)
            login_user(AdminUser(username), remember=False)
            flash("Signed in successfully to the Admin Console.", "success")
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/admin"):
                return redirect(next_page)
            return redirect(url_for("admin_dashboard"))
        else:
            record_login_attempt(client_ip, username, success=False)
            failed_count = get_failed_attempts_count(client_ip, lockout_minutes=15)
            remaining_attempts = max(0, 5 - failed_count)

            if remaining_attempts == 0:
                flash("Maximum failed attempts reached. Your IP has been temporarily locked out for 15 minutes.", "danger")
                return render_template("admin/login.html", is_locked=True, remaining_minutes=15)
            else:
                flash(f"Invalid administrator credentials. {remaining_attempts} attempt(s) remaining before lockout.", "danger")
                return render_template("admin/login.html", is_locked=False, remaining_minutes=0)

    return render_template("admin/login.html", is_locked=is_locked, remaining_minutes=remaining_minutes)


@app.route("/admin/logout", methods=["POST"])
@login_required
def admin_logout():
    """Signs out the administrator and terminates the session."""
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("admin_login"))


# ==============================================================================
# PROTECTED ADMIN DASHBOARD & MANAGEMENT ROUTES
# ==============================================================================

@app.route("/admin", methods=["GET"])
@login_required
def admin_dashboard():
    """
    Main Administrator Dashboard:
    - Summary analytics (total count, real %, AI %, average confidence, trend chart).
    - Submissions table with search, filter (real/fake, date range, training status), sorting, and pagination.
    """
    page = request.args.get("page", 1, type=int)
    per_page = 15

    search_query = request.args.get("search", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()
    training_filter = request.args.get("training_status", "").strip()
    sort_by = request.args.get("sort_by", "newest").strip()
    start_date_str = request.args.get("start_date", "").strip()
    end_date_str = request.args.get("end_date", "").strip()

    # Base query
    query = PredictionLog.query

    # Search filter
    if search_query:
        search_pattern = f"%{search_query}%"
        query = query.filter(
            db.or_(
                PredictionLog.original_filename.ilike(search_pattern),
                PredictionLog.ip_address.ilike(search_pattern),
                PredictionLog.uuid_filename.ilike(search_pattern)
            )
        )

    # Class filter
    if prediction_filter in ("Real", "AI-generated"):
        query = query.filter(PredictionLog.prediction == prediction_filter)

    # Training ingestion filter
    if training_filter:
        query = query.filter(PredictionLog.training_status == training_filter)

    # Date range filters
    if start_date_str:
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            query = query.filter(PredictionLog.timestamp >= start_dt)
        except ValueError:
            pass

    if end_date_str:
        try:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(PredictionLog.timestamp < end_dt)
        except ValueError:
            pass

    # Sorting
    if sort_by == "oldest":
        query = query.order_by(PredictionLog.timestamp.asc())
    elif sort_by == "conf_high":
        query = query.order_by(PredictionLog.confidence.desc())
    elif sort_by == "conf_low":
        query = query.order_by(PredictionLog.confidence.asc())
    elif sort_by == "size_desc":
        query = query.order_by(PredictionLog.file_size.desc())
    else:  # newest first
        query = query.order_by(PredictionLog.timestamp.desc())

    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    logs = pagination.items

    stats = get_summary_statistics()

    return render_template(
        "admin/dashboard.html",
        logs=logs,
        pagination=pagination,
        stats=stats,
        search_query=search_query,
        prediction_filter=prediction_filter,
        training_filter=training_filter,
        sort_by=sort_by,
        start_date=start_date_str,
        end_date=end_date_str
    )


@app.route("/admin/image/<filename>", methods=["GET"])
@login_required
def admin_image(filename):
    """Serve full-resolution stored image from protected uploads/ directory."""
    # Prevent directory traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        abort(400)
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/admin/thumbnail/<filename>", methods=["GET"])
@login_required
def admin_thumbnail(filename):
    """
    Serve or generate on-the-fly small cached thumbnail (100x100).
    Keeps dashboard memory usage minimal even with hundreds of user uploads.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        abort(400)

    thumb_path = os.path.join(THUMBNAILS_DIR, filename)
    src_path = os.path.join(UPLOADS_DIR, filename)

    if not os.path.exists(thumb_path) and os.path.exists(src_path):
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                img.save(thumb_path, format="JPEG", quality=85)
        except Exception as e:
            print(f"[WARNING] Could not create thumbnail for {filename}: {e}")
            return send_from_directory(UPLOADS_DIR, filename)

    if os.path.exists(thumb_path):
        return send_from_directory(THUMBNAILS_DIR, filename)
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/admin/move/<int:log_id>/<string:target_class>", methods=["POST"])
@login_required
def admin_move_to_dataset(log_id, target_class):
    """
    Dataset Ingestion Action:
    Copies an uploaded image into database/train/REAL or database/train/FAKE
    so the owner can retrain or fine-tune models on new user inputs.
    """
    success, message = move_image_to_dataset(log_id, target_class, database_dir="database")
    if success:
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(request.referrer or url_for("admin_dashboard"))


@app.route("/admin/delete/<int:log_id>", methods=["POST"])
@login_required
def admin_delete_log(log_id):
    """Permanently delete a submission log record and its stored upload file."""
    log = PredictionLog.query.get_or_404(log_id)
    uuid_filename = log.uuid_filename

    # Delete physical file from uploads/
    img_path = os.path.join(UPLOADS_DIR, uuid_filename)
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
        except OSError as e:
            gc.collect()
            try:
                os.remove(img_path)
            except OSError as e2:
                print(f"[WARNING] Could not delete image file: {e2}")

    # Delete thumbnail
    thumb_path = os.path.join(THUMBNAILS_DIR, uuid_filename)
    if os.path.exists(thumb_path):
        try:
            os.remove(thumb_path)
        except OSError:
            pass

    # Delete database record
    db.session.delete(log)
    db.session.commit()

    flash(f"Record #{log_id} and its associated file were permanently removed.", "success")
    return redirect(request.referrer or url_for("admin_dashboard"))


@app.route("/admin/export-csv", methods=["GET"])
@login_required
def admin_export_csv():
    """Export complete submissions audit log as a downloadable CSV file."""
    logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Timestamp (UTC)",
        "Original Filename",
        "Stored UUID Filename",
        "File Size (Bytes)",
        "Verdict",
        "Confidence (%)",
        "Prob Real (%)",
        "Prob AI-Generated (%)",
        "Client IP",
        "Session ID",
        "Training Dataset Status"
    ])

    for log in logs:
        writer.writerow([
            log.id,
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S") if log.timestamp else "",
            log.original_filename,
            log.uuid_filename,
            log.file_size,
            log.prediction,
            log.confidence,
            log.prob_real,
            log.prob_fake,
            log.ip_address,
            log.session_id,
            log.training_status
        ])

    csv_data = output.getvalue()
    filename = f"detector_submissions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print("\n" + "=" * 65)
    print(" 🚀 AI-Generated Image Detector & Admin Console is LIVE!")
    print(f" 👉 Public Web Interface : {url}")
    print(f" 👉 Admin Portal URL      : {url}/admin")
    print(" Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65 + "\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=False)
