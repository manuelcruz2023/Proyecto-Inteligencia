"""Prueba rápida SIN dataset ni internet: valida modelo, pérdida, split encoder/head y FaceCropper.
Ejecutar: python scripts/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

import config
from src.face import FaceCropper
from src.model import build_model, set_backbone_trainable, split_encoder_and_head

model = build_model(pretrained=False)
X = np.random.randint(0, 255, (4, config.SEQ_LEN, config.IMG_SIZE, config.IMG_SIZE, 3)).astype("float32")
y = np.array([0, 1, 2, 1])
p = model(X, training=False).numpy()
assert p.shape == (4, 3) and np.allclose(p.sum(1), 1, atol=1e-4)

model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
print("1 paso de entrenamiento:", model.train_on_batch(X, y))
set_backbone_trainable(model, True)
model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
print("1 paso de fine-tuning :", model.train_on_batch(X, y))

# El modelo dividido (encoder + head) debe dar la MISMA salida que el modelo completo
enc, head = split_encoder_and_head(model)
feats = np.stack([enc(X[0, t][None], training=False).numpy()[0] for t in range(config.SEQ_LEN)])
p_split = head(feats[None], training=False).numpy()[0]
p_full = model(X[:1], training=False).numpy()[0]
assert np.allclose(p_split, p_full, atol=1e-4), (p_split, p_full)

print("FaceCropper OK:", FaceCropper().crop(np.zeros((480, 640, 3), np.uint8)))
print("SMOKE TEST OK")