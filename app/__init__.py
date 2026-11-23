from flask import Flask

from src.inference import DrowsinessPredictor
from .sessions import SessionManager


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024   # rechaza payloads > 3 MB

    predictor, error = None, None
    try:
        predictor = DrowsinessPredictor()
    except Exception as exc:   # la app arranca igual y muestra el error en la UI
        error = str(exc)
        app.logger.error("No se pudo cargar el modelo: %s", exc)

    app.extensions["predictor"] = predictor
    app.extensions["model_error"] = error
    app.extensions["sessions"] = SessionManager(predictor)

    from .routes import bp
    app.register_blueprint(bp)
    return app