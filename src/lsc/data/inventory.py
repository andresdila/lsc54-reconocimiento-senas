"""Inventario y auditoría de los videos crudos de LSC-54.

Cada video del dataset queda descrito por una fila con su origen, su
identidad (sesión + carpeta de señante), su etiqueta y su huella digital
(sha256). Este inventario es la base para:
  * detectar duplicados exactos (misma huella en dos rutas),
  * construir el split por señante sin fuga de identidad,
  * dejar trazabilidad de qué datos entraron al experimento.
"""
from __future__ import annotations

import fnmatch
import hashlib
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class VideoRecord:
    video_id: str          # id estable: <sesión>-<señante>-<etiqueta>-<repetición>
    source_zip: str
    path_in_zip: str
    session: str           # campaña de grabación (S1 colores, S2 cortesía+números)
    signer_folder: int     # número de carpeta tal como viene en el zip
    signer_key: str        # identidad conservadora = sesión + carpeta
    category: str
    label: str
    rep: int
    bytes: int
    sha256: str
    excluded: bool
    exclusion_reason: str


def slugify(text: str) -> str:
    """'Buenos Días' -> 'buenos_dias' (sin tildes, minúsculas, '_')."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "_".join(text.lower().split())


def parse_member(path: str) -> dict | None:
    """Interpreta una ruta del zip: '<raíz>/<señante>/<categoría>/<seña>/<rep>.avi'.

    Devuelve None si la ruta no es un video con esa estructura.
    """
    path = unicodedata.normalize("NFC", path)
    parts = path.split("/")
    if len(parts) != 5 or not parts[-1].lower().endswith(".avi"):
        return None
    _, signer, category, sign, filename = parts
    return {
        "signer_folder": int(signer),
        "category": slugify(category),
        "sign_raw": sign,
        "rep": int(Path(filename).stem),
    }


def is_excluded(path: str, rules: Iterable[dict]) -> tuple[bool, str]:
    path = unicodedata.normalize("NFC", path)
    for rule in rules:
        if fnmatch.fnmatch(path, unicodedata.normalize("NFC", rule["pattern"])):
            return True, rule["reason"]
    return False, ""


def build_inventory(
    zip_specs: list[dict],
    zips_dir: Path,
    exclude_rules: list[dict],
    label_fixes: dict[str, str],
    extract_to: Path | None = None,
) -> pd.DataFrame:
    """Recorre los zips, calcula sha256 de cada video y (opcional) los extrae.

    Los videos excluidos se registran igual en el inventario (excluded=True)
    para que la decisión quede auditada, pero no se extraen.
    """
    records: list[VideoRecord] = []
    for spec in zip_specs:
        zpath = zips_dir / spec["name"]
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                meta = parse_member(info.filename)
                if meta is None:
                    continue
                label = slugify(meta["sign_raw"])
                label = label_fixes.get(label, label)
                excluded, reason = is_excluded(info.filename, exclude_rules)

                digest = hashlib.sha256()
                with zf.open(info) as src:
                    data = src.read()
                digest.update(data)

                session = spec["session"]
                signer_key = f"{session}-{meta['signer_folder']:02d}"
                video_id = f"{signer_key}-{label}-{meta['rep']}"

                if extract_to is not None and not excluded:
                    out = extract_to / session / f"{meta['signer_folder']:02d}" / label / f"{meta['rep']}.avi"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)

                records.append(VideoRecord(
                    video_id=video_id,
                    source_zip=spec["name"],
                    path_in_zip=unicodedata.normalize("NFC", info.filename),
                    session=session,
                    signer_folder=meta["signer_folder"],
                    signer_key=signer_key,
                    category=meta["category"],
                    label=label,
                    rep=meta["rep"],
                    bytes=info.file_size,
                    sha256=digest.hexdigest(),
                    excluded=excluded,
                    exclusion_reason=reason,
                ))

    df = pd.DataFrame([asdict(r) for r in records])
    return df.sort_values(["session", "signer_folder", "label", "rep"]).reset_index(drop=True)


def audit(df: pd.DataFrame) -> dict:
    """Resumen de calidad que se reporta en el plan experimental."""
    kept = df[~df["excluded"]]
    dup_mask = kept.duplicated("sha256", keep=False)
    per_class = kept.groupby("label").size()
    signers_per_class = kept.groupby("label")["signer_key"].nunique()
    return {
        "videos_total": int(len(df)),
        "videos_excluded": int(df["excluded"].sum()),
        "videos_kept": int(len(kept)),
        "classes": int(kept["label"].nunique()),
        "signer_keys": int(kept["signer_key"].nunique()),
        "duplicates_remaining": int(dup_mask.sum()),
        "videos_per_class_min": int(per_class.min()),
        "videos_per_class_max": int(per_class.max()),
        "signers_per_class_min": int(signers_per_class.min()),
        "signers_per_class_max": int(signers_per_class.max()),
        "video_ids_unique": bool(kept["video_id"].is_unique),
    }
