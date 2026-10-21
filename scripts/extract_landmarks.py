"""Extrae landmarks de MediaPipe Holistic de cada video S2 y los guarda como .npy (T, 98).

    python scripts/extract_landmarks.py            # todos los videos de la campaña S2 no excluidos
    python scripts/extract_landmarks.py --limit 20 # prueba rápida
    python scripts/extract_landmarks.py --workers 4  # procesos en paralelo

Requiere mediapipe==0.10.21 (API clásica `mp.solutions.holistic`) y los videos de
`python scripts/download_data.py`. Escribe data/processed/landmarks/<video_id>.npy (ignorado por Git)
y data/inventory/landmarks_audit.csv (versionable): resolución, fps, cuadros, tasa de detección y
tiempo de extracción por video. Es idempotente: omite los videos ya procesados.

Cada video se procesa con una instancia nueva de Holistic. En modo video, MediaPipe sigue a la
persona de un cuadro al siguiente; reutilizar la instancia haría que los primeros cuadros de un
video dependieran del video anterior (posiblemente de otro señante) y que el resultado cambiara
según el orden de procesamiento.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.features.landmarks import D, frame_vector  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "landmarks"


def to_array(lms):
    return None if lms is None else np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)


def extract(path: Path, holistic) -> tuple[np.ndarray, int, int, float]:
    """Devuelve (secuencia (T, 98), ancho, alto, fps). Las coordenadas se pasan a píxeles antes de normalizar."""
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append(frame_vector(to_array(res.pose_landmarks), to_array(res.left_hand_landmarks),
                                   to_array(res.right_hand_landmarks), width, height))
    cap.release()
    seq = np.stack(frames) if frames else np.empty((0, D), dtype=np.float32)
    return seq, width, height, fps


def process_one(job: tuple[str, str, str]) -> dict:
    """Procesa un video (o relee su .npy si ya existe) y devuelve su fila de auditoría."""
    import mediapipe as mp

    video_id, signer_key, src = job
    dst = OUT_DIR / f"{video_id}.npy"
    seconds = np.nan
    if not dst.exists():
        t0 = time.perf_counter()
        # instancia nueva por video: sin estado de seguimiento heredado del video anterior
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
            seq, w, h, fps = extract(Path(src), holistic)
        seconds = time.perf_counter() - t0
        np.save(dst, seq)
    else:  # el .npy ya existe: solo se relee el tamaño del video para la auditoría
        seq = np.load(dst)
        cap = cv2.VideoCapture(src)
        w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
    n = len(seq)
    return {
        "video_id": video_id, "signer_key": signer_key, "width": w, "height": h, "fps": fps,
        "n_frames": n, "extract_seconds": seconds,
        "pose_rate": float(np.mean(~np.isnan(seq[:, 0]))) if n else 0.0,
        "left_rate": float(np.mean(~np.isnan(seq[:, 14]))) if n else 0.0,
        "right_rate": float(np.mean(~np.isnan(seq[:, 56]))) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="procesos en paralelo (la extracción es CPU-bound; medir antes de subirlo)")
    args = ap.parse_args()

    inv = pd.read_csv(ROOT / "data/inventory/lsc54_v4_inventory.csv")
    inv = inv[(~inv.excluded) & (inv.session == "S2")]
    if args.limit:
        inv = inv.head(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(r.video_id, r.signer_key,
             str(ROOT / "data/raw/videos" / r.session / f"{r.signer_folder:02d}" / r.label / f"{r.rep}.avi"))
            for r in inv.itertuples()]

    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(tqdm(pool.map(process_one, jobs, chunksize=4), total=len(jobs)))
    else:
        rows = [process_one(j) for j in tqdm(jobs)]
    wall = time.perf_counter() - t0

    audit = pd.DataFrame(rows)
    audit.to_csv(ROOT / "data/inventory/landmarks_audit.csv", index=False)
    print(f"[ok] {len(audit)} videos en {wall:.0f} s con {args.workers} proceso(s); "
          f"{audit.n_frames.sum()} cuadros; {audit.n_frames.sum() / wall:.1f} cuadros/s")


if __name__ == "__main__":
    main()
