"""Paso 4: evaluación en test (o val): accuracy, precision, recall, F1, matriz de confusión."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config
from src.data_generator import SequenceGenerator


def plot_confusion(cm, names, path):
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, title, fmt in [(axes[0], cm, "Matriz de confusión (conteo)", "{:d}"),
                                 (axes[1], cmn, "Matriz normalizada por fila (recall)", "{:.2f}")]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(title)
        thr = data.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                v = data[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center",
                        va="center", color="white" if v > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    gen = SequenceGenerator(config.PROCESSED_DIR / f"sequences_{args.split}.csv", config.FACES_DIR,
                            batch_size=args.batch_size, seq_len=config.SEQ_LEN,
                            augment=False, shuffle=False)
    probs = model.predict(gen, verbose=1)
    y_true, y_pred = gen.labels, np.argmax(probs, axis=1)   # sin shuffle → mismo orden
    names = config.CLASS_NAMES
    labels = list(range(len(names)))

    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "split": args.split, "n_sequences": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": dict(zip(["precision", "recall", "f1"], map(float, precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0)[:3]))),
        "weighted": dict(zip(["precision", "recall", "f1"], map(float, precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0)[:3]))),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, n in enumerate(names)},
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   digits=4, zero_division=0)
    print("\n" + report)
    print("Matriz de confusión:\n", cm)
    print(f"\nAccuracy: {metrics['accuracy']:.4f} | F1 macro: {metrics['macro']['f1']:.4f}")

    (config.REPORTS_DIR / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    (config.REPORTS_DIR / f"classification_report_{args.split}.txt").write_text(report)
    plot_confusion(cm, names, config.REPORTS_DIR / f"confusion_matrix_{args.split}.png")
    print(f"Resultados en {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()