"""Corre la batería de experimentos de la Entrega 2 (ver reports/e2_avance/EXPERIMENTOS.md).

    python scripts/run_experiments.py                 # todos, uno tras otro
    python scripts/run_experiments.py base lr3e-3     # solo los nombrados
    python scripts/run_experiments.py --list

Funciona igual en Windows, macOS y Linux. Omite los experimentos que ya tienen summary.json y,
si uno se interrumpe, al relanzarlo retoma desde el último fold terminado.
Al final ejecuta scripts/compare_models.py para regenerar la tabla y las figuras.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# nombre -> cambios sobre configs/model.yaml (un factor por experimento)
EXPERIMENTS: dict[str, list[str]] = {
    "base": [],
    "sin_aumentos": ["augment.enabled=false"],
    "lr3e-3": ["train.lr=0.003"],
    "lr3e-4": ["train.lr=0.0003"],
    "wd0.1": ["train.weight_decay=0.1"],
    "dropout0.5": ["model.dropout=0.5"],
    "opt_adam": ["train.optimizer=adam"],
    "opt_sgd": ["train.optimizer=sgd", "train.lr=0.03"],
    "lstm": ["model.encoder=lstm"],
    "conv1d": ["model.encoder=conv"],
    "sin_ncuadros": ["data.use_n_frames=false"],
    "pesos_clase": ["train.class_weights=true"],
    "sin_mascara": ["data.use_mask=false"],
    "sin_interpolacion": ["data.max_gap=0"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="experimentos a correr (por defecto, todos)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="hilos de PyTorch por experimento")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, s in EXPERIMENTS.items():
            print(f"{n:20s} {' '.join(s) or '(configuración base)'}")
        return
    names = args.names or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"[error] experimentos desconocidos: {unknown}. Ver --list")

    for n in names:
        if (ROOT / "reports" / "e2_avance" / "runs" / n / "summary.json").exists():
            print(f"[omitido] {n}: ya tiene resultados")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "train_sequence.py"), "--name", n]
        if EXPERIMENTS[n]:
            cmd += ["--set", *EXPERIMENTS[n]]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        print(f"\n=== {n} ===  {' '.join(EXPERIMENTS[n]) or '(base)'}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    if not args.no_compare:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_models.py"), "--candidate", "base"],
                       check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
