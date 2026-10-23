"""Representación secuencial para el modelo recurrente (Entrega 2).

Parte de la misma secuencia (T, 98) que usa la línea base (ver landmarks.py: coordenadas en
píxeles, origen en el centro de los hombros, escala = ancho de hombros, NaN si no se detecta)
y la lleva a un tensor de tamaño fijo (n_frames, 98 + 3):

  1) filtro de plausibilidad: cuadros con alguna coordenada a más de `max_abs` anchos de hombro
     se marcan como ausentes (mismo criterio que la variante `logreg_plausible` de la Entrega 1);
  2) relleno de vacíos cortos: cada bloque (pose, mano izq., mano der.) se interpola linealmente
     en el tiempo si el vacío dura como máximo `max_gap` cuadros y está acotado por cuadros
     detectados a ambos lados (no se extrapola al inicio ni al final);
  3) remuestreo a `n_frames` cuadros equiespaciados (vecino más cercano: no inventa posiciones
     entre cuadros y conserva la máscara coherente con el valor);
  4) máscara: 3 canales binarios (pose, mano izq., mano der. presentes) y los NaN restantes
     pasan a 0. Así la red distingue "mano en el origen" de "mano no detectada".

Todo es determinista salvo `augment`, que solo se aplica sobre entrenamiento.
"""
from __future__ import annotations

import numpy as np

from .landmarks import D, N_HAND, N_POSE

# bloques de columnas (inicio, fin) dentro del vector de 98: pose, mano izquierda, mano derecha
BLOCKS = ((0, 2 * N_POSE), (2 * N_POSE, 2 * N_POSE + 2 * N_HAND), (2 * N_POSE + 2 * N_HAND, D))
N_MASK = len(BLOCKS)
D_IN = D + N_MASK  # 101


def drop_implausible(seq: np.ndarray, max_abs: float | None) -> np.ndarray:
    seq = np.array(seq, dtype=np.float32, copy=True)
    if max_abs is None or len(seq) == 0:
        return seq
    with np.errstate(invalid="ignore"):
        bad = np.nanmax(np.where(np.isnan(seq), 0, np.abs(seq)), axis=1) > max_abs
    seq[bad] = np.nan
    return seq


def fill_short_gaps(seq: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola linealmente, por bloque, los vacíos internos de hasta `max_gap` cuadros."""
    seq = np.array(seq, dtype=np.float32, copy=True)
    T = len(seq)
    if max_gap <= 0 or T < 3:
        return seq
    for a, b in BLOCKS:
        present = ~np.isnan(seq[:, a])
        idx = np.flatnonzero(present)
        if len(idx) < 2:
            continue
        for left, right in zip(idx[:-1], idx[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                w = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
                seq[left + 1:right, a:b] = (1 - w) * seq[left, a:b] + w * seq[right, a:b]
    return seq


def resample_indices(T: int, n_frames: int, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Índices de `n_frames` cuadros equiespaciados entre `start` y `end` (vecino más cercano)."""
    end = (T - 1) if end is None else end
    return np.clip(np.rint(np.linspace(start, end, n_frames)), 0, T - 1).astype(int)


def to_model_input(seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Secuencia (T, 98) ya filtrada y rellenada -> (len(idx), 101) con máscara y sin NaN."""
    x = seq[idx]
    mask = np.stack([~np.isnan(x[:, a]) for a, _ in BLOCKS], axis=1).astype(np.float32)
    return np.concatenate([np.nan_to_num(x, nan=0.0), mask], axis=1).astype(np.float32)


def prepare(seq: np.ndarray, n_frames: int = 32, max_gap: int = 5, max_abs: float | None = 6.0) -> np.ndarray:
    """Pipeline determinista completo (validación y prueba)."""
    s = fill_short_gaps(drop_implausible(seq, max_abs), max_gap)
    return to_model_input(s, resample_indices(len(s), n_frames))


def augment(seq: np.ndarray, rng: np.random.Generator, n_frames: int = 32, max_gap: int = 5,
            max_abs: float | None = 6.0, crop: float = 0.1, scale: float = 0.1, rot_deg: float = 10.0,
            noise: float = 0.01) -> np.ndarray:
    """Versión aleatoria de `prepare` para entrenamiento.

    - desplazamiento temporal: recorta hasta `crop` de la duración por cada extremo antes de remuestrear;
    - escala isotrópica en [1-scale, 1+scale] y rotación en ±rot_deg alrededor del centro de hombros
      (el origen de la representación), comunes a todo el video;
    - ruido gaussiano de desviación `noise` (en anchos de hombro) sobre los puntos detectados.
    Sin reflejo horizontal: cambiaría la lateralidad de la seña.
    """
    s = fill_short_gaps(drop_implausible(seq, max_abs), max_gap)
    T = len(s)
    c = crop * (T - 1)
    start, end = rng.uniform(0, c), (T - 1) - rng.uniform(0, c)
    x = to_model_input(s, resample_indices(T, n_frames, start, end))
    xy = x[:, :D].reshape(n_frames, -1, 2)
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    k = rng.uniform(1 - scale, 1 + scale)
    R = k * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    xy = xy @ R.T + rng.normal(0, noise, xy.shape).astype(np.float32)
    present = np.repeat(x[:, D:], [N_POSE, N_HAND, N_HAND], axis=1)[..., None]  # (n, 49, 1)
    x[:, :D] = (xy * present).reshape(n_frames, D)  # lo ausente sigue en 0
    return x
