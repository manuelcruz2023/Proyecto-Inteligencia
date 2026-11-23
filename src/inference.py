"""Inferencia: predictor (modelo) y analizador de flujo (estado por cámara/sesión)."""
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.face import FaceCropper
from src.model import split_encoder_and_head


class DrowsinessPredictor:
    """Carga el modelo una vez y expone encode() (CNN por frame) y classify() (LSTM)."""

    def __init__(self, model_path=config.MODEL_PATH, config_path=config.MODEL_CONFIG_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. Ejecute scripts/03_train.py primero.")
        cfg = {}
        if Path(config_path).exists():
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.class_names = cfg.get("class_names", config.CLASS_NAMES)
        self.img_size = cfg.get("img_size", config.IMG_SIZE)
        self.seq_len = cfg.get("seq_len", config.SEQ_LEN)
        self.sample_fps = cfg.get("sample_fps", config.SAMPLE_FPS)

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.encoder, self.head = split_encoder_and_head(self.model)
        self.feature_dim = self.encoder.output_shape[-1]
        self._lock = threading.Lock()   # TF no es thread-safe con tf.function compartidas

        S, T, F = self.img_size, self.seq_len, self.feature_dim
        self._encode = tf.function(
            lambda x: self.encoder(x, training=False),
            input_signature=[tf.TensorSpec([1, S, S, 3], tf.float32)])
        self._classify = tf.function(
            lambda f: self.head(f, training=False),
            input_signature=[tf.TensorSpec([1, T, F], tf.float32)])
        # Calentamiento: evita latencia alta en la primera petición
        self.classify(np.stack([self.encode(np.zeros((S, S, 3), np.uint8)) for _ in range(T)]))

    def encode(self, crop_rgb):
        with self._lock:
            x = crop_rgb[None].astype(np.float32)
            return self._encode(x)[0].numpy()

    def classify(self, features):
        with self._lock:
            return self._classify(features[None].astype(np.float32))[0].numpy()

    def predict_full(self, sequence):
        """Modelo completo (T,H,W,3). Se usa solo para validar equivalencia en pruebas."""
        return self.model(sequence[None].astype(np.float32), training=False)[0].numpy()


class StreamAnalyzer:
    """Estado por cámara: recorte facial, buffer de features y suavizado temporal."""

    def __init__(self, predictor, alpha=0.6):
        self.p = predictor
        self.cropper = FaceCropper(img_size=predictor.img_size)
        self.features = deque(maxlen=predictor.seq_len)
        self.alpha = alpha            # peso de la predicción nueva en la media móvil exponencial
        self.ema = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.cropper.reset()
            self.features.clear()
            self.ema = None

    def process(self, frame_bgr):
        with self._lock:
            crop, bbox = self.cropper.crop(frame_bgr)
            if crop is None:   # se pierde el rostro → se rompe la continuidad temporal
                self.features.clear()
                self.ema = None
                return {"face_detected": False, "ready": False, "progress": 0.0, "bbox": None}

            self.features.append(self.p.encode(crop))
            n = len(self.features)
            out = {"face_detected": True, "bbox": bbox, "progress": n / self.p.seq_len}
            if n < self.p.seq_len:
                out["ready"] = False
                return out

            probs = self.p.classify(np.stack(self.features))
            self.ema = probs if self.ema is None else self.alpha * probs + (1 - self.alpha) * self.ema
            k = int(np.argmax(self.ema))
            out.update({
                "ready": True,
                "label": self.p.class_names[k],
                "label_index": k,
                "confidence": float(self.ema[k]),
                "probs": {c: float(v) for c, v in zip(self.p.class_names, self.ema)},
            })
            return out