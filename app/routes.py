import base64
import binascii
import time

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("main", __name__)


def _decode_image(data_url):
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("Campo 'image' ausente.")
    if "," in data_url:                      # quita "data:image/jpeg;base64,"
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Imagen base64 inválida.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return img


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/health")
def health():
    pred = current_app.extensions["predictor"]
    return jsonify(
        status="ok" if pred else "model_missing",
        model_loaded=pred is not None,
        error=current_app.extensions["model_error"],
        class_names=pred.class_names if pred else [],
        seq_len=pred.seq_len if pred else None,
        sample_fps=pred.sample_fps if pred else None,
    )


@bp.post("/api/predict")
def predict():
    if current_app.extensions["predictor"] is None:
        return jsonify(error=current_app.extensions["model_error"] or "Modelo no cargado."), 503
    payload = request.get_json(silent=True) or {}
    sid = str(payload.get("session_id", ""))[:64]
    if not sid:
        return jsonify(error="Falta session_id."), 400
    try:
        frame = _decode_image(payload.get("image"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    analyzer = current_app.extensions["sessions"].get(sid)
    t0 = time.perf_counter()
    result = analyzer.process(frame)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["frame_size"] = [frame.shape[1], frame.shape[0]]   # para escalar la caja en el cliente
    return jsonify(result)


@bp.post("/api/reset")
def reset():
    payload = request.get_json(silent=True) or {}
    current_app.extensions["sessions"].reset(str(payload.get("session_id", ""))[:64])
    return jsonify(ok=True)