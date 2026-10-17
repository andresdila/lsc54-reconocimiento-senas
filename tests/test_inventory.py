"""Pruebas del inventario con zips sintéticos que imitan la estructura real de LSC-54."""
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.data.inventory import audit, build_inventory, parse_member, slugify  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def _make_zips(tmp: Path, extra_duplicate: bool = False) -> None:
    """Crea los 3 zips con la misma convención de rutas que el dataset original."""
    with zipfile.ZipFile(tmp / "Videos Colors.zip", "w") as z:
        for signer in (1, 2):
            for sign in ("amarillo", "cafe"):
                for rep in (1, 2):
                    z.writestr(f"Videos Colors/{signer}/Colores/{sign}/{rep}.avi", f"C{signer}{sign}{rep}")
    with zipfile.ZipFile(tmp / "Videos Numbers.zip", "w") as z:
        for signer in (1, 34):
            for sign in ("dies", "uno"):
                for rep in (0, 1):
                    z.writestr(f"Videos Numbers/{signer}/Numeros/{sign}/{rep}.avi", f"N{signer}{sign}{rep}")
    with zipfile.ZipFile(tmp / "Videos Courtesy.zip", "w") as z:
        for signer in (1, 34):
            for sign in ("Buenos Dias", "denada"):
                for rep in (0, 1):
                    z.writestr(f"Videos Courtesy/{signer}/Cordialidad/{sign}/{rep}.avi", f"K{signer}{sign}{rep}")
        # copia de los números del señante 34 (como en el zip real)
        for sign in ("dies", "uno"):
            for rep in (0, 1):
                z.writestr(f"Videos Courtesy/34/Números/{sign}/{rep}.avi", f"N34{sign}{rep}")
        if extra_duplicate:  # un video repetido con otra etiqueta: debe detectarse
            z.writestr("Videos Courtesy/1/Cordialidad/hola/0.avi", "C1amarillo1")


def _inventory(tmp: Path, extract: bool = False):
    return build_inventory(CFG["files"], tmp, CFG["exclude"], CFG["label_fixes"],
                           extract_to=(tmp / "videos") if extract else None)


def test_slugify_and_parse():
    assert slugify("Buenos Días") == "buenos_dias"
    meta = parse_member("Videos Numbers/34/Numeros/uno/2.avi")
    assert meta == {"signer_folder": 34, "category": "numeros", "sign_raw": "uno", "rep": 2}
    assert parse_member("Videos Numbers/34/") is None


def test_excludes_courtesy_numbers_copy_and_fixes_labels(tmp_path):
    _make_zips(tmp_path)
    inv = _inventory(tmp_path)
    excluded = inv[inv.excluded]
    assert len(excluded) == 4 and excluded.path_in_zip.str.contains("Números").all()
    kept = inv[~inv.excluded]
    assert "diez" in set(kept.label) and "dies" not in set(kept.label)
    assert "de_nada" in set(kept.label)
    summary = audit(inv)
    assert summary["duplicates_remaining"] == 0
    assert summary["video_ids_unique"]


def test_signer_identity_is_session_scoped(tmp_path):
    """La carpeta 1 de colores (S1) y la carpeta 1 de cortesía (S2) NO se asumen la misma persona."""
    _make_zips(tmp_path)
    kept = _inventory(tmp_path).query("~excluded")
    assert {"S1-01", "S2-01"} <= set(kept.signer_key)


def test_detects_cross_label_duplicate(tmp_path):
    _make_zips(tmp_path, extra_duplicate=True)
    assert audit(_inventory(tmp_path))["duplicates_remaining"] == 2


def test_extraction_skips_excluded(tmp_path):
    _make_zips(tmp_path)
    _inventory(tmp_path, extract=True)
    files = list((tmp_path / "videos").rglob("*.avi"))
    assert len(files) == 8 + 8 + 8  # colores + números + cortesía, sin la copia
