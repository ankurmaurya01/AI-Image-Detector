"""
Authentication and session management module for AI Image Detector.
Provides single-admin user session model, password verification, and security helpers.
"""

from datetime import timedelta
import os
from flask_login import LoginManager, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

login_manager = LoginManager()


class AdminUser(UserMixin):
    """Single-owner admin user model for Flask-Login."""
    def __init__(self, username: str):
        self.id = username
        self.username = username

    def get_id(self):
        return self.id


@login_manager.user_loader
def load_user(user_id):
    """Load the single admin user if ID matches configured admin username."""
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    if user_id == admin_username:
        return AdminUser(username=admin_username)
    return None


def init_auth(app):
    """Configure Flask-Login and secure cookie/session settings on the Flask app."""
    # Session cookie security settings
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
    app.config.setdefault("PERMANENT_SESSION_LIFETIME", timedelta(minutes=60))

    login_manager.init_app(app)
    login_manager.login_view = "admin_login"
    login_manager.login_message = "Please sign in to access the administrator dashboard."
    login_manager.login_message_category = "warning"


def verify_admin_credentials(username: str, password: str) -> bool:
    """Verify input credentials against environment variables."""
    expected_username = os.environ.get("ADMIN_USERNAME", "admin").strip()
    password_hash = os.environ.get("ADMIN_PASSWORD_HASH", "").strip()

    if not expected_username or not password_hash:
        return False

    if username.strip() != expected_username:
        return False

    return check_password_hash(password_hash, password)


def create_password_hash(password: str) -> str:
    """Generate a secure Werkzeug password hash."""
    return generate_password_hash(password, method="scrypt")
