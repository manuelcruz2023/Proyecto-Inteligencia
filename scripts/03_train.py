"""Paso 3: entrenamiento CNN + LSTM en dos fases (cabeza → fine-tuning)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from src.data_generator import SequenceGenerator
from src.model import build_model, set_backbone_trainable


def make_gen(split, batch_size, train):
    return SequenceGenerator(config.PROCESSED_DIR / f"sequences_{split}.csv", config.FACES_DIR,
                             batch_size=batch_size, seq_len=config.SEQ_LEN,
                             augment=train, shuffle=train, seed=config.SEED)


def plot_history(hist: pd.DataFrame, path, phase_boundary):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (m, t) in zip(ax, [("loss", "Pérdida"), ("accuracy", "Exactitud")]):
        a.plot(hist[m], label="train")
        a.plot(hist[f"val_{m}"], label="val")
        if phase_boundary:
            a.axvline(phase_boundary - 0.5, color="gray", ls="--", label="inicio fine-tuning")
        a.set_title(t); a.set_xlabel("época"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-frozen", type=int, default=config.EPOCHS_FROZEN)
    ap.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--no-pretrained", action="store_true", help="Sin pesos ImageNet (offline)")
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    train_gen = make_gen("train", args.batch_size, True)
    val_gen = make_gen("val", args.batch_size, False)

    # Desbalance de clases → pesos inversamente proporcionales a la frecuencia
    cw = compute_class_weight("balanced", classes=np.arange(len(config.CLASS_NAMES)),
                              y=train_gen.labels)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Pesos de clase:", class_weight)

    model = build_model(pretrained=not args.no_pretrained)
    model.summary(line_length=110)

    def compile_model(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def callbacks(threshold=None):
        kw = {"initial_value_threshold": threshold} if threshold is not None else {}
        return [
            tf.keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss",
                                               save_best_only=True, verbose=1, **kw),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                                 min_lr=1e-7, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir=str(config.REPORTS_DIR / "tensorboard")),
        ]

    # ---------------- Fase 1: backbone congelado ----------------
    set_backbone_trainable(model, False)
    compile_model(config.LR_FROZEN)
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_frozen,
                   class_weight=class_weight, callbacks=callbacks(), workers=2,
                   max_queue_size=10)
    hist = pd.DataFrame(h1.history)
    boundary = len(hist)

    # ---------------- Fase 2: fine-tuning ----------------
    if not args.skip_finetune and args.epochs_finetune > 0:
        set_backbone_trainable(model, True)
        compile_model(config.LR_FINETUNE)     # recompilar tras cambiar trainable
        best1 = float(min(h1.history["val_loss"]))
        h2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_finetune,
                       class_weight=class_weight, callbacks=callbacks(best1), workers=2,
                       max_queue_size=10)
        hist = pd.concat([hist, pd.DataFrame(h2.history)], ignore_index=True)

    hist.to_csv(config.REPORTS_DIR / "training_history.csv", index=False)
    plot_history(hist, config.REPORTS_DIR / "training_curves.png", boundary)

    cfg = {"img_size": config.IMG_SIZE, "seq_len": config.SEQ_LEN,
           "sample_fps": config.SAMPLE_FPS, "class_names": config.CLASS_NAMES,
           "backbone": "MobileNetV2", "temporal": "LSTM(128)"}
    config.MODEL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nMejor modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    main()