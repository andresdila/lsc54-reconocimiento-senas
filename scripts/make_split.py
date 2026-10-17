"""Genera splits/split_<version>.csv a partir del inventario y configs/split.yaml.

    python scripts/make_split.py                     # usa el inventario real (tras download_data.py)
    python scripts/make_split.py --check-only        # verifica que el split versionado se reproduce

Si el split ya existe, el script NO lo sobrescribe: lo regenera y comprueba que
sea idéntico. Cambiar la partición exige una nueva versión (v2), de forma explícita.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.data.splits import check_split, make_signer_split  # noqa: E402


def load_videos(inventory: Path, coverage: Path, sessions: list[str]) -> pd.DataFrame:
    if inventory.exists():
        inv = pd.read_csv(inventory)
        return inv[(~inv["excluded"]) & inv["session"].isin(sessions)][["signer_key", "label", "video_id"]]
    print(f"[aviso] no existe {inventory.name}; se usa la cobertura leída del índice de los zips")
    cov = pd.read_csv(coverage)
    return cov[cov["signer_key"].str.split("-").str[0].isin(sessions)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "split.yaml")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    videos = load_videos(
        ROOT / "data/inventory/lsc54_v4_inventory.csv",
        ROOT / "data/inventory/s2_coverage_zip_index.csv",
        cfg["sessions"],
    )
    split = make_signer_split(videos, cfg["seed"], cfg["test_signers_per_group"], cfg["n_folds"])
    report = check_split(split, videos)

    out = ROOT / "splits" / f"split_{cfg['version']}.csv"
    if out.exists():
        committed = pd.read_csv(out)
        same = committed[split.columns].reset_index(drop=True).equals(split)
        if not same:
            sys.exit(f"[error] el split regenerado difiere de {out.name}: no se sobrescribe")
        print(f"[ok] {out.name} se reproduce exactamente")
    elif args.check_only:
        sys.exit(f"[error] no existe {out.name}")
    else:
        out.parent.mkdir(exist_ok=True)
        split.to_csv(out, index=False)
        print(f"[ok] escrito {out.name}")

    summary = {"config": cfg, **report}
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
