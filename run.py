"""Servidor de DESARROLLO. Para producción use run_production.py o gunicorn."""
import config
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=config.APP_HOST, port=config.APP_PORT, debug=False, threaded=True)