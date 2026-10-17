"""Partición independiente del señante, estratificada por cobertura de vocabulario.

Diseño (configs/split.yaml):
  1. Se clasifica cada señante en 'full' (graba todas las clases) o 'partial'.
  2. Test: se sortean k señantes de cada grupo con una semilla fija. Quedan
     bloqueados y no se usan para entrenar, ajustar ni elegir modelos.
  3. Dev: los señantes restantes se emparejan (1 full + 1 partial) y cada pareja
     es un fold de validación cruzada agrupada.
Así ningún señante aparece en dos particiones y todas las clases están
presentes en test y en cada fold de validación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_groups(videos: pd.DataFrame) -> pd.Series:
    """'full' si el señante tiene videos de todas las clases; si no, 'partial'."""
    n_classes = videos["label"].nunique()
    per_signer = videos.groupby("signer_key")["label"].nunique()
    return per_signer.map(lambda n: "full" if n == n_classes else "partial").rename("group")


def make_signer_split(videos: pd.DataFrame, seed: int, test_per_group: int, n_folds: int) -> pd.DataFrame:
    """Devuelve una fila por señante: signer_key, group, partition ('test'|'dev'), fold."""
    groups = coverage_groups(videos)
    rng = np.random.default_rng(seed)

    test, dev_by_group = [], {}
    for g in sorted(groups.unique()):
        members = sorted(groups[groups == g].index)
        shuffled = list(rng.permutation(members))
        test += shuffled[:test_per_group]
        dev_by_group[g] = shuffled[test_per_group:]

    sizes = {len(v) for v in dev_by_group.values()}
    if sizes != {n_folds}:
        raise ValueError(f"Cada grupo debe aportar {n_folds} señantes a dev; hay {sizes}")

    rows = [{"signer_key": s, "group": groups[s], "partition": "test", "fold": -1} for s in test]
    for g, members in dev_by_group.items():
        rows += [{"signer_key": s, "group": g, "partition": "dev", "fold": i + 1} for i, s in enumerate(members)]
    return pd.DataFrame(rows).sort_values(["partition", "fold", "group"]).reset_index(drop=True)


def check_split(split: pd.DataFrame, videos: pd.DataFrame) -> dict:
    """Controles anti-fuga. Lanza AssertionError si alguno falla."""
    assert split["signer_key"].is_unique, "un señante aparece más de una vez"
    assert set(split["signer_key"]) == set(videos["signer_key"]), "hay señantes sin asignar o sobrantes"

    labeled = videos.merge(split, on="signer_key")
    all_labels = set(videos["label"])
    test_labels = set(labeled.loc[labeled.partition == "test", "label"])
    assert test_labels == all_labels, f"faltan clases en test: {all_labels - test_labels}"

    dev = labeled[labeled.partition == "dev"]
    for fold, part in dev.groupby("fold"):
        assert set(part["label"]) == all_labels, f"faltan clases en el fold {fold}"
        train_signers = set(dev.loc[dev.fold != fold, "signer_key"])
        assert train_signers.isdisjoint(part["signer_key"]), f"fuga de señante en el fold {fold}"

    weight = "n_videos" if "n_videos" in labeled else None
    count = (lambda d: int(d[weight].sum())) if weight else (lambda d: int(len(d)))
    return {
        "signers_test": sorted(split.loc[split.partition == "test", "signer_key"]),
        "videos_test": count(labeled[labeled.partition == "test"]),
        "videos_dev": count(dev),
        "videos_per_fold": {int(f): count(p) for f, p in dev.groupby("fold")},
    }
