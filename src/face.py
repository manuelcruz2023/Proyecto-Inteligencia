"""Detección y recorte facial. Se usa TAL CUAL en extracción de frames e inferencia."""
import cv2
import numpy as np


class FaceCropper:
    """Detecta el rostro más grande (Haar cascade) y devuelve un recorte cuadrado RGB.

    Es con estado: suaviza la caja entre frames y tolera pérdidas breves de detección.
    Use una instancia por video / por sesión de cámara.
    """

    def __init__(self, img_size=96, margin=0.2, detect_width=480,
                 max_missed=5, smoothing=0.5):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"No se pudo cargar el clasificador Haar: {path}")
        self.img_size = img_size
        self.margin = margin
        self.detect_width = detect_width
        self.max_missed = max_missed
        self.smoothing = smoothing  # peso de la caja anterior (0 = sin suavizado)
        self.reset()

    def reset(self):
        self._last = None
        self._missed = 0

    def _detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.detect_width / w if w > self.detect_width else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return np.array([x, y, fw, fh], dtype=np.float32) / scale

    def crop(self, frame_bgr):
        """Devuelve (recorte_RGB_uint8[img,img,3], bbox(x,y,w,h)) o (None, None)."""
        det = self._detect(frame_bgr)
        if det is not None:
            self._missed = 0
            if self._last is not None:
                det = self.smoothing * self._last + (1.0 - self.smoothing) * det
            self._last = det
        elif self._last is not None and self._missed < self.max_missed:
            self._missed += 1          # reutiliza la última caja por unos frames
        else:
            self._last = None
            return None, None

        x, y, w, h = self._last
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * self.margin)
        H, W = frame_bgr.shape[:2]
        x0, y0 = int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2)))
        x1, y1 = int(min(W, round(cx + side / 2))), int(min(H, round(cy + side / 2)))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None, None
        face = cv2.resize(frame_bgr[y0:y1, x0:x1], (self.img_size, self.img_size),
                          interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face, (int(x), int(y), int(w), int(h))