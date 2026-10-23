"""Representación de landmarks por cuadro y estadísticas por video (baseline clásico).

Por cuadro se conservan (x, y) de:
  * 7 puntos de pose de tren superior (nariz, hombros, codos, muñecas)  -> 14
  * 21 puntos de mano izquierda y 21 de mano derecha                    -> 84
  Total: 98 valores por cuadro (D = 98).

Normalización por cuadro (invariante a posición y distancia a la cámara):
  1) MediaPipe entrega x/ancho e y/alto; se convierte a píxeles (x*ancho, y*alto) para que dos
     resoluciones distintas (640x360 y 640x480) no deformen la seña de forma distinta;
  2) origen = punto medio de los hombros; escala = distancia entre hombros (en píxeles).
Si no hay pose en un cuadro, todo el cuadro queda en NaN. Si una mano no se
detecta, sus 42 valores quedan en NaN (la ausencia se resume como tasa de detección).
"""
from __future__ import annotations

import warnings

import numpy as np

POSE_IDX = (0, 11, 12, 13, 14, 15, 16)  # nariz, hombros, codos, muñecas (MediaPipe Pose)
N_POSE, N_HAND = len(POSE_IDX), 21
D = 2 * (N_POSE + 2 * N_HAND)  # 98
L_SHOULDER, R_SHOULDER = 1, 2  # posiciones dentro de POSE_IDX


def frame_vector(pose: np.ndarray | None, left: np.ndarray | None, right: np.ndarray | None,
                 width: int = 1, height: int = 1) -> np.ndarray:
    """pose (33,2+), left/right (21,2+) en coordenadas normalizadas de MediaPipe -> vector (D,).

    width/height: tamaño del video en píxeles. Con el valor por defecto (1, 1) no se corrige
    la relación de aspecto; en el pipeline real siempre deben pasarse.
    """
    out = np.full(D, np.nan, dtype=np.float32)
    if pose is None:
        return out
    px = np.array([width, height], dtype=np.float32)
    p = np.asarray(pose, dtype=np.float32)[list(POSE_IDX), :2] * px
    origin = (p[L_SHOULDER] + p[R_SHOULDER]) / 2
    scale = float(np.linalg.norm(p[L_SHOULDER] - p[R_SHOULDER]))
    if scale < 1e-6:
        return out
    parts = [(p - origin) / scale]
    for hand in (left, right):
        if hand is None:
            parts.append(np.full((N_HAND, 2), np.nan, dtype=np.float32))
        else:
            parts.append((np.asarray(hand, dtype=np.float32)[:, :2] * px - origin) / scale)
    return np.concatenate(parts).reshape(-1)


def _hand_slices() -> tuple[slice, slice]:
    start = 2 * N_POSE
    return slice(start, start + 2 * N_HAND), slice(start + 2 * N_HAND, start + 4 * N_HAND)


def drop_implausible_frames(seq: np.ndarray, max_abs: float) -> np.ndarray:
    """Marca como ausentes (NaN) los cuadros con alguna coordenada a más de `max_abs` anchos de hombro.

    Ocurre cuando MediaPipe estima los hombros casi superpuestos: la escala tiende a cero y las
    coordenadas normalizadas se disparan. Con videos de 640 px y hombros de ~140 px, ningún punto
    real queda a más de ~5 anchos de hombro del centro. El filtro es fijo (no aprende de los datos).
    """
    seq = np.array(seq, dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        bad = np.nanmax(np.abs(seq), axis=1, initial=0.0) > max_abs
    seq[bad] = np.nan
    return seq


def video_features(seq: np.ndarray, max_abs: float | None = None) -> np.ndarray:
    """Secuencia (T, D) con NaN -> vector fijo de 5*D + 3 = 493 valores.

    media, desviación, mínimo, máximo y velocidad media absoluta de cada coordenada,
    más tasa de detección de cada mano y número de cuadros. Si `max_abs` se define,
    antes se descartan los cuadros físicamente implausibles (ver drop_implausible_frames).
    """
    seq = np.asarray(seq, dtype=np.float64)
    if max_abs is not None:
        seq = drop_implausible_frames(seq, max_abs)
    T = len(seq)
    lh, rh = _hand_slices()
    det_l = float(np.mean(~np.isnan(seq[:, lh.start]))) if T else 0.0
    det_r = float(np.mean(~np.isnan(seq[:, rh.start]))) if T else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # columnas totalmente NaN
        mean = np.nanmean(seq, axis=0)
        std = np.nanstd(seq, axis=0)
        mn = np.nanmin(seq, axis=0)
        mx = np.nanmax(seq, axis=0)
        vel = np.nanmean(np.abs(np.diff(seq, axis=0)), axis=0) if T > 1 else np.full(D, np.nan)
    return np.concatenate([mean, std, mn, mx, vel, [det_l, det_r, float(T)]])


N_FEATURES = 5 * D + 3
