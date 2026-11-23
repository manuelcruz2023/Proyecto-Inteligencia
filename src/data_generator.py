"""Generador de secuencias desde arrays .npy (memory-mapped) con data augmentation."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, faces_dir, batch_size=16, seq_len=16,
                 augment=False, shuffle=False, seed=42):
        super().__init__()
        df = pd.read_csv(csv_path, dtype={"video_id": str})
        self.video_ids = df["video_id"].to_numpy()
        self.starts = df["start"].to_numpy(dtype=np.int64)
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.faces_dir = Path(faces_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self.indices = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def _video(self, vid):
        if vid not in self._cache:   # mmap: no carga todo el video en RAM
            self._cache[vid] = np.load(self.faces_dir / f"{vid}.npy", mmap_mode="r")
        return self._cache[vid]

    def _augment(self, seq):
        """Misma transformación para TODA la secuencia (preserva coherencia temporal)."""
        if self.rng.random() < 0.5:
            seq = seq[:, :, ::-1, :]                              # espejo horizontal
        seq = seq * self.rng.uniform(0.7, 1.3)                    # brillo
        mean = seq.mean()
        seq = (seq - mean) * self.rng.uniform(0.8, 1.2) + mean    # contraste
        seq = seq + self.rng.normal(0, 3.0, size=seq.shape)       # ruido gaussiano
        return np.clip(seq, 0, 255)

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        size = self._video(self.video_ids[idx[0]]).shape[1]
        X = np.empty((len(idx), self.seq_len, size, size, 3), dtype=np.float32)
        y = np.empty(len(idx), dtype=np.int64)
        for j, k in enumerate(idx):
            arr = self._video(self.video_ids[k])
            s = self.starts[k]
            seq = np.asarray(arr[s:s + self.seq_len], dtype=np.float32)
            X[j] = self._augment(seq) if self.augment else seq
            y[j] = self.labels[k]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)