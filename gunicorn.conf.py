# Linux/macOS: gunicorn -c gunicorn.conf.py wsgi:app
bind = "0.0.0.0:8000"
workers = 1        # IMPORTANTE: estado de sesión en memoria → un único worker
threads = 4
timeout = 120
accesslog = "-"