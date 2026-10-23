"""Métricas comunes a la línea base y al modelo secuencial (mismas definiciones que train_baseline.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred, proba, classes) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))}
    if proba is not None:
        top3 = np.array(classes)[np.argsort(proba, axis=1)[:, -3:]]
        out["top3_accuracy"] = float(np.mean([t in row for t, row in zip(y_true, top3)]))
    return out


def per_signer_macro_f1(oof: pd.DataFrame) -> dict:
    """macro-F1 de cada señante sobre las clases que grabó (oof con columnas y, p, signer)."""
    return {s: float(f1_score(g.y, g.p, labels=sorted(g.y.unique()), average="macro", zero_division=0))
            for s, g in oof.groupby("signer")}


def per_class_f1(oof: pd.DataFrame, classes) -> dict:
    return dict(zip(classes, map(float, f1_score(oof.y, oof.p, labels=classes, average=None, zero_division=0))))


def signer_bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, classes, n_boot: int = 2000, seed: int = 2026) -> dict:
    """IC por remuestreo de señantes de la diferencia de macro-F1 agrupado (b - a).

    a y b son predicciones fuera de muestra de dos modelos sobre los MISMOS videos (columnas
    video_id, y, p, signer). En cada réplica se sortean señantes con reemplazo y se recalcula el
    macro-F1 agrupado de ambos modelos sobre los videos de esos señantes: la unidad de
    remuestreo es la persona, igual que la unidad de partición.
    """
    a = a.set_index("video_id").sort_index()
    b = b.set_index("video_id").loc[a.index]
    signers = np.array(sorted(a.signer.unique()))
    by_signer = {s: np.flatnonzero(a.signer.to_numpy() == s) for s in signers}
    rng = np.random.default_rng(seed)
    ya, pa, pb = a.y.to_numpy(), a.p.to_numpy(), b.p.to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by_signer[s] for s in rng.choice(signers, len(signers), replace=True)])
        fa = f1_score(ya[idx], pa[idx], labels=classes, average="macro", zero_division=0)
        fb = f1_score(ya[idx], pb[idx], labels=classes, average="macro", zero_division=0)
        diffs.append(fb - fa)
    diffs = np.array(diffs)
    point = (f1_score(ya, pb, labels=classes, average="macro", zero_division=0)
             - f1_score(ya, pa, labels=classes, average="macro", zero_division=0))
    return {"diff": float(point), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_boot_leq0": float(np.mean(diffs <= 0)), "n_boot": n_boot}
