"""Entrena la BiGRU base con los 14 señantes de desarrollo y guarda sus pesos, sin tocar el test.

    python scripts/export_model.py

Usa la misma configuración (configs/model.yaml), la misma semilla y el mismo procedimiento
que el entrenamiento final de scripts/train_sequence.py --final-test, pero no evalúa el test
bloqueado: sirve para regenerar los pesos del modelo final en cualquier máquina.
Salida: outputs/bigru_base_final.pt (pesos, estandarizador, clases, configuración y versiones)
y outputs/bigru_base_final.sha256.

Para cargar el modelo:
    ckpt = torch.load("outputs/bigru_base_final.pt", weights_only=False)
    model = build_model(ckpt["config"], len(ckpt["classes"])); model.load_state_dict(ckpt["state_dict"])
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_sequence import LANDMARKS, git_info, load_config, train_fold  # noqa: E402
from lsc.data.table import load_split_config, load_table  # noqa: E402


def main() -> None:
    split_cfg = load_split_config(ROOT)
    seed = split_cfg["seed"]
    cfg = load_config([])
    df = load_table(ROOT, split_cfg)
    classes = sorted(df.label.unique())
    y_all = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy().astype(np.int64)
    dev = np.flatnonzero((df.partition == "dev").to_numpy())
    seqs = [np.load(LANDMARKS / f"{v}.npy") for v in df.video_id.iloc[dev]]
    y_dev = y_all[dev]
    # la función de entrenamiento exige un conjunto de validación; se usa el propio desarrollo
    # (solo para las curvas de registro). El test bloqueado no se carga en ningún momento.
    model, st, curves, _, cost = train_fold(cfg, seqs, y_dev, seqs, y_dev, classes, seed)
    commit, dirty = git_info()
    out = ROOT / "outputs" / "bigru_base_final.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "standardizer": st.state(), "classes": classes,
                "config": cfg, "seed": seed, "split_version": split_cfg["version"], "n_train": int(len(dev)),
                "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
                "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
                "fit_seconds": cost["fit_seconds"]}, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB) sha256={digest} ajuste={cost['fit_seconds']:.0f}s")


if __name__ == "__main__":
    main()
