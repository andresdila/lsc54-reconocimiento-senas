"""Descarga, verifica y audita los videos crudos de LSC-54.

Uso (desde la raíz del repo):
    python scripts/download_data.py             # descarga + verifica + extrae + inventario
    python scripts/download_data.py --no-extract  # solo inventario (no escribe videos)

Es idempotente: si un zip ya existe y su MD5 coincide, no se vuelve a bajar;
si la descarga se corta, se reanuda desde donde quedó.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.data.inventory import audit, build_inventory  # noqa: E402

CHUNK = 1 << 20  # 1 MiB


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_bytes: int) -> None:
    """Descarga con reanudación (HTTP Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if done and r.status_code != 206:  # el servidor ignoró el Range
            done = 0
        mode = "ab" if done else "wb"
        with part.open(mode) as f, tqdm(
            total=expected_bytes, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(CHUNK):
                f.write(block)
                bar.update(len(block))
    part.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "data.yaml", type=Path)
    parser.add_argument("--no-extract", action="store_true", help="no escribe los videos, solo el inventario")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    zips_dir = ROOT / cfg["paths"]["raw_zips"]

    for spec in cfg["files"]:
        dest = zips_dir / spec["name"]
        if dest.exists() and md5sum(dest) == spec["md5"]:
            print(f"[ok] {spec['name']} ya descargado y verificado")
            continue
        url = cfg["dataset"]["download_url"].format(file_id=spec["file_id"])
        download(url, dest, spec["bytes"])
        got = md5sum(dest)
        if got != spec["md5"]:
            dest.unlink()
            sys.exit(f"[error] MD5 no coincide para {spec['name']}: {got} != {spec['md5']}")
        print(f"[ok] {spec['name']} verificado (MD5)")

    extract_to = None if args.no_extract else ROOT / cfg["paths"]["raw_videos"]
    inv = build_inventory(
        zip_specs=cfg["files"],
        zips_dir=zips_dir,
        exclude_rules=cfg.get("exclude", []),
        label_fixes=cfg.get("label_fixes", {}),
        extract_to=extract_to,
    )
    inv_path = ROOT / cfg["paths"]["inventory"]
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(inv_path, index=False)

    summary = audit(inv)
    (inv_path.with_suffix(".audit.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["duplicates_remaining"] or not summary["video_ids_unique"]:
        sys.exit("[error] quedan duplicados o ids repetidos: revisar inventario antes de continuar")


if __name__ == "__main__":
    main()
