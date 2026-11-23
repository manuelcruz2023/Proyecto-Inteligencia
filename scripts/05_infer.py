"""Inferencia local con OpenCV: webcam (--source 0) o archivo de video (--source ruta.mp4)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.inference import DrowsinessPredictor, StreamAnalyzer

COLORS = {"Alert": (80, 200, 80), "Low Vigilance": (0, 190, 255), "Drowsy": (60, 60, 240)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--drowsy-seconds", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    predictor = DrowsinessPredictor()
    analyzer = StreamAnalyzer(predictor)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la fuente de video.")

    interval = 1.0 / predictor.sample_fps    # mismo ritmo temporal que en el entrenamiento
    last_t, result, drowsy_since = 0.0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last_t >= interval:
            last_t = now
            result = analyzer.process(frame)
            is_drowsy = bool(result.get("ready") and result["label"] == "Drowsy"
                             and result["probs"]["Drowsy"] >= args.threshold)
            drowsy_since = (drowsy_since or now) if is_drowsy else None

        view = frame.copy()
        if result and result.get("bbox"):
            x, y, w, h = result["bbox"]
            color = COLORS.get(result.get("label"), (200, 200, 200))
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            if result.get("ready"):
                cv2.putText(view, f"{result['label']} {result['confidence']:.0%}", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                for i, (n, p) in enumerate(result["probs"].items()):
                    cv2.putText(view, f"{n}: {p:.2f}", (10, 25 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[n], 2)
            else:
                cv2.putText(view, f"Recopilando {result['progress']:.0%}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif result:
            cv2.putText(view, "Sin rostro", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if drowsy_since and time.time() - drowsy_since >= args.drowsy_seconds:
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 12)
            cv2.putText(view, "ALERTA: SOMNOLENCIA", (20, view.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print("\a", end="", flush=True)   # campana del terminal

        cv2.imshow("Deteccion de somnolencia (q para salir)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()