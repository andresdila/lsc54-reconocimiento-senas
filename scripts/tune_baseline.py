"""Sensibilidad de la línea base: ¿la brecha con la BiGRU se debe a que la línea base no se afinó?

    python scripts/tune_baseline.py

Evalúa, con los mismos 7 folds por señante de dev y las mismas 493 variables por video,
la regresión logística con varios valores de C y dos clasificadores clásicos adicionales
(bosque aleatorio y SVM con kernel RBF). Elegir el mejor de esta lista con los mismos folds
favorece a la línea base, de modo que la brecha resultante es una cota conservadora.
El test bloqueado no se toca.
Salida: reports/e3_final/baseline_sensibilidad_cv.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_baseline import SEED, load_features, load_table  # noqa: E402


def candidates() -> dict:
    out = {}
    for c in (0.01, 0.1, 1.0, 10.0):
        out[f"logreg_C{c:g}"] = lambda c=c: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=c, max_iter=3000, random_state=SEED))
    out["random_forest"] = lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))
    for c in (1.0, 10.0):
        out[f"svm_rbf_C{c:g}"] = lambda c=c: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), SVC(C=c, gamma="scale", random_state=SEED))
    return out


def main() -> None:
    df = load_table()
    classes = sorted(df.label.unique())
    X = load_features(df, needs_landmarks=True, max_abs=None)
    y = df.label.to_numpy()
    dev = (df.partition == "dev").to_numpy()
    fold_ids = sorted(df.loc[dev, "fold"].unique())
    results = {}
    for name, make in candidates().items():
        scores, oof_t, oof_p, t0 = [], [], [], time.perf_counter()
        for f in fold_ids:
            tr = dev & (df.fold != f).to_numpy()
            va = dev & (df.fold == f).to_numpy()
            pred = make().fit(X[tr], y[tr]).predict(X[va])
            scores.append(float(f1_score(y[va], pred, labels=classes, average="macro", zero_division=0)))
            oof_t += list(y[va]); oof_p += list(pred)
        results[name] = {"folds_macro_f1": scores, "mean": float(np.mean(scores)),
                         "std": float(np.std(scores, ddof=1)),
                         "pooled_macro_f1": float(f1_score(oof_t, oof_p, labels=classes, average="macro", zero_division=0)),
                         "seconds": time.perf_counter() - t0}
        print(f"{name:16s} {results[name]['mean']:.3f} ± {results[name]['std']:.3f}")
    best = max(results, key=lambda k: results[k]["mean"])
    out = {"seed": SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
           "best_by_mean": best, "results": results}
    dst = ROOT / "reports" / "e3_final" / "baseline_sensibilidad_cv.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("mejor:", best)


if __name__ == "__main__":
    main()
