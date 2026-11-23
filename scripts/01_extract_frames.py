"""Paso 1: extrae frames de los videos de UTA-RLDD, recorta el rostro y guarda un .npy por video.

Estructura esperada (la original del dataset):
    data/raw/UTA-RLDD/Fold1_part1/01/0.mov   (0 = alert, 5 = low vigilance, 10 = drowsy)
El script busca recursivamente cualquier video cuyo nombre sea 0, 5 o 10.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.face import FaceCropper

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
_cropper = None   # una instancia por proceso worker


def list_videos(raw_dir: Path):
    jobs = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            raw_label = int(p.stem)
        except ValueError:
            print(f"[AVISO] Se ignora (nombre no numérico): {p}")
            continue
        if raw_label in config.LABEL_MAP:
            jobs.append((str(p), p.parent.name, raw_label))
    return jobs


def process_video(job):
    global _cropper
    path, subject, raw_label, out_dir, sample_fps, max_seconds, overwrite = job
    video_id = f"{subject}_{raw_label}"
    out_path = Path(out_dir) / f"{video_id}.npy"
    meta = {"video_id": video_id, "subject": subject, "raw_label": raw_label,
            "label": config.LABEL_MAP[raw_label], "source": path}

    if out_path.exists() and not overwrite:
        meta.update(n_frames=int(np.load(out_path, mmap_mode="r").shape[0]), fps=np.nan,
                    face_ratio=np.nan)
        return meta

    if _cropper is None:
        _cropper = FaceCropper(img_size=config.IMG_SIZE)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1:
        fps = 30.0
    step = max(1, int(round(fps / sample_fps)))
    max_frames = int(max_seconds * fps) if max_seconds > 0 else float("inf")

    _cropper.reset()
    faces, sampled, idx = [], 0, 0
    while idx < max_frames and cap.grab():          # grab() es barato: no decodifica a imagen
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                sampled += 1
                crop, _ = _cropper.crop(frame)
                if crop is not None:
                    faces.append(crop)
        idx += 1
    cap.release()

    if len(faces) < config.SEQ_LEN:
        return None
    np.save(out_path, np.stack(faces).astype(np.uint8))
    meta.update(n_frames=len(faces), fps=float(fps), face_ratio=len(faces) / max(sampled, 1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--sample-fps", type=float, default=config.SAMPLE_FPS)
    ap.add_argument("--max-seconds", type=int, default=config.MAX_SECONDS_PER_VIDEO,
                    help="Segundos por video a procesar (0 = completo)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = list_videos(args.raw_dir)
    if not jobs:
        sys.exit(f"No se encontraron videos en {args.raw_dir}")
    print(f"{len(jobs)} videos encontrados.")
    jobs = [(p, s, l, str(config.FACES_DIR), args.sample_fps, args.max_seconds, args.overwrite)
            for p, s, l in jobs]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_video, jobs), total=len(jobs), desc="Extrayendo"):
            if res is not None:
                rows.append(res)
    df = pd.DataFrame(rows).sort_values("video_id")
    df.to_csv(config.METADATA_CSV, index=False)
    print(f"\nVideos válidos: {len(df)}/{len(jobs)}  → {config.METADATA_CSV}")
    print(df.groupby("label")["n_frames"].agg(["count", "sum"]).rename(
        index=dict(enumerate(config.CLASS_NAMES))))


if __name__ == "__main__":
    main()