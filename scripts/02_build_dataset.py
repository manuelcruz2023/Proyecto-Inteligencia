"""Paso 2: divide por SUJETO (train/val/test) y genera el índice de secuencias deslizantes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    meta = pd.read_csv(config.METADATA_CSV, dtype={"subject": str, "video_id": str})
    subjects = np.array(sorted(meta["subject"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, round(n * config.SPLIT_RATIOS["test"]))
    n_val = max(1, round(n * config.SPLIT_RATIOS["val"]))
    splits = {"test": subjects[:n_test].tolist(),
              "val": subjects[n_test:n_test + n_val].tolist(),
              "train": subjects[n_test + n_val:].tolist()}

    # Garantía anti-fuga: ningún sujeto en dos particiones
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    config.SPLITS_JSON.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    T = config.SEQ_LEN
    for name, subs in splits.items():
        stride = config.SEQ_STRIDE_TRAIN if name == "train" else config.SEQ_STRIDE_EVAL
        rows = []
        for _, v in meta[meta["subject"].isin(subs)].iterrows():
            for s in range(0, int(v["n_frames"]) - T + 1, stride):
                rows.append((v["video_id"], v["subject"], int(v["label"]), s))
        df = pd.DataFrame(rows, columns=["video_id", "subject", "label", "start"])
        out = config.PROCESSED_DIR / f"sequences_{name}.csv"
        df.to_csv(out, index=False)
        dist = df["label"].value_counts().sort_index().to_dict()
        print(f"[{name:5s}] sujetos={len(subs):2d} secuencias={len(df):6d} "
              f"distribución={{{', '.join(f'{config.CLASS_NAMES[k]}: {v}' for k, v in dist.items())}}}")


if __name__ == "__main__":
    main()