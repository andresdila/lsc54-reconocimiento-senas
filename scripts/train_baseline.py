"""Baseline clásico: estadísticas de landmarks por video + regresión logística.

    python scripts/train_baseline.py --model dummy_majority   # piso de referencia (no necesita landmarks)
    python scripts/train_baseline.py --model dummy_prior
    python scripts/train_baseline.py --model logreg           # requiere data/processed/landmarks/*.npy
    python scripts/train_baseline.py --model logreg --final-test   # UNA sola vez, al congelar el baseline

Evaluación en dev: validación cruzada agrupada por señante usando los folds de splits/split_v1.csv
(el modelo se reajusta en cada fold con los señantes restantes de dev). El test bloqueado solo
se toca con --final-test y el script se niega a repetirlo.
Salida: reports/e1_plan/baseline_<model>_cv.json (o ..._test.json).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.features.landmarks import N_FEATURES, video_features  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text(encoding="utf-8"))
SEED = CFG["seed"]


def make_model(name: str):
    if name == "dummy_majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_prior":
        return DummyClassifier(strategy="stratified", random_state=SEED)
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    raise ValueError(name)


def load_table() -> pd.DataFrame:
    inv = pd.read_csv(ROOT / "data/inventory/lsc54_v4_inventory.csv")
    split = pd.read_csv(ROOT / "splits" / f"split_{CFG['version']}.csv")
    inv = inv[(~inv.excluded) & inv.session.isin(CFG["sessions"])]
    return inv.merge(split, on="signer_key", how="inner")[["video_id", "signer_key", "label", "partition", "fold"]]


def load_features(df: pd.DataFrame, needs_landmarks: bool, max_abs: float | None) -> np.ndarray:
    if not needs_landmarks:
        return np.zeros((len(df), 1))
    d = ROOT / "data" / "processed" / "landmarks"
    return np.stack([video_features(np.load(d / f"{v}.npy"), max_abs) for v in df.video_id])


def metrics(y_true, y_pred, proba, classes) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))}
    if proba is not None:
        top3 = np.array(classes)[np.argsort(proba, axis=1)[:, -3:]]
        out["top3_accuracy"] = float(np.mean([t in row for t, row in zip(y_true, top3)]))
    return out


def fit_predict(model_name, X_tr, y_tr, X_te, classes):
    """Ajusta y predice; devuelve también el costo (tiempo de ajuste y latencia por video)."""
    t0 = time.perf_counter()
    m = make_model(model_name).fit(X_tr, y_tr)
    t1 = time.perf_counter()
    pred = m.predict(X_te)
    t2 = time.perf_counter()
    proba = None
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X_te)
        proba = np.zeros((len(X_te), len(classes)))
        proba[:, [classes.index(c) for c in m.classes_]] = p
    cost = {"fit_seconds": t1 - t0, "predict_ms_per_video": 1000 * (t2 - t1) / max(len(X_te), 1)}
    return pred, proba, cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["dummy_majority", "dummy_prior", "logreg"], required=True)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--max-abs", type=float, default=None,
                    help="descarta cuadros con coordenadas a más de N anchos de hombro (filtro de plausibilidad)")
    ap.add_argument("--tag", default="", help="sufijo del archivo de resultados para distinguir variantes")
    args = ap.parse_args()

    df = load_table()
    classes = sorted(df.label.unique())
    X = load_features(df, needs_landmarks=args.model == "logreg", max_abs=args.max_abs)
    name = args.model + (f"_{args.tag}" if args.tag else "")
    y = df.label.to_numpy()
    out_dir = ROOT / "reports" / "e1_plan"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts", "configs", "splits"],
                                             text=True).strip())
    except Exception:
        commit, dirty = None, None
    meta = {"model": args.model, "variant": args.tag or None, "max_abs": args.max_abs, "seed": SEED, "split_version": CFG["version"], "n_features": int(X.shape[1]),
            "expected_features": N_FEATURES if args.model == "logreg" else None, "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}

    dev = (df.partition == "dev").to_numpy()
    if args.final_test:
        dst = out_dir / f"baseline_{name}_test.json"
        if dst.exists():
            sys.exit(f"[error] {dst.name} ya existe: el test bloqueado se evalúa una sola vez")
        te = ~dev
        pred, proba, cost = fit_predict(args.model, X[dev], y[dev], X[te], classes)
        res = {**meta, "n_train": int(dev.sum()), "n_test": int(te.sum()), **metrics(y[te], pred, proba, classes), **cost}
        dst.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    folds, oof_true, oof_pred, oof_signer = [], [], [], []
    for f in sorted(df.loc[dev, "fold"].unique()):
        tr = dev & (df.fold != f).to_numpy()
        va = dev & (df.fold == f).to_numpy()
        pred, proba, cost = fit_predict(args.model, X[tr], y[tr], X[va], classes)
        folds.append({"fold": int(f), "n_train": int(tr.sum()), "n_val": int(va.sum()),
                      **metrics(y[va], pred, proba, classes), **cost})
        oof_true += list(y[va]); oof_pred += list(pred); oof_signer += list(df.signer_key[va])
    keys = [k for k in folds[0] if k not in ("fold", "n_train", "n_val")]
    summary = {k: {"mean": float(np.mean([r[k] for r in folds])), "std": float(np.std([r[k] for r in folds], ddof=1))}
               for k in keys}
    # F1 sobre las predicciones fuera de muestra de los 7 folds juntos: menos ruidoso por clase
    pooled = {"macro_f1": float(f1_score(oof_true, oof_pred, labels=classes, average="macro", zero_division=0)),
              "accuracy": float(accuracy_score(oof_true, oof_pred))}
    # macro-F1 por señante (cada señante sobre las clases que grabó): la experiencia de una persona nueva
    oof = pd.DataFrame({"y": oof_true, "p": oof_pred, "signer": oof_signer})
    per_signer = {s: float(f1_score(g.y, g.p, labels=sorted(g.y.unique()), average="macro", zero_division=0))
                  for s, g in oof.groupby("signer")}
    per_class = dict(zip(classes, map(float, f1_score(oof_true, oof_pred, labels=classes, average=None,
                                                         zero_division=0))))
    res = {**meta, "folds": folds, "summary": summary, "pooled_oof": pooled,
           "per_signer_macro_f1": per_signer, "per_class_f1_pooled": per_class}
    (out_dir / f"baseline_{name}_cv.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "pooled_oof": pooled, "folds": [(r["fold"], round(r["macro_f1"], 4)) for r in folds]}, indent=2))


if __name__ == "__main__":
    main()
