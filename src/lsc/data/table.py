"""Tabla de videos del alcance (S2, sin excluidos) unida a la partición versionada."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_split_config(root: Path) -> dict:
    return yaml.safe_load((root / "configs" / "split.yaml").read_text(encoding="utf-8"))


def load_table(root: Path, cfg: dict | None = None) -> pd.DataFrame:
    """Mismo filtrado que scripts/train_baseline.py: los modelos se comparan sobre los mismos videos."""
    cfg = cfg or load_split_config(root)
    inv = pd.read_csv(root / "data/inventory/lsc54_v4_inventory.csv")
    split = pd.read_csv(root / "splits" / f"split_{cfg['version']}.csv")
    inv = inv[(~inv.excluded) & inv.session.isin(cfg["sessions"])]
    return (inv.merge(split, on="signer_key", how="inner")[["video_id", "signer_key", "label", "partition", "fold"]]
            .reset_index(drop=True))
