import os

# Bind to Render's dynamically assigned $PORT or fallback to 5000
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# 1 worker with 4 threads ensures concurrent user handling within Render's 512MB RAM limit
workers = 1
threads = 4
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
