"""Cifras de diagnóstico del informe final que no salen directamente de compare_models.py.

    python scripts/report_numbers.py

Lee los resultados ya versionados (no entrena ni toca el test de nuevo) y calcula:
  * calibración de la BiGRU base en validación cruzada (ECE con 15 intervalos, confianza media y exactitud);
  * diagnóstico de las curvas de aprendizaje (época del mínimo de la pérdida de validación por fold,
    pérdida final y época en que el macro-F1 de entrenamiento alcanza 0,99);
  * errores en el test bloqueado (por señante y posición de la seña correcta en el top-3).
Salida: reports/e3_final/cifras_informe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports" / "e2_avance" / "runs" / "base"


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main() -> None:
    oof = pd.read_csv(RUN / "oof.csv")
    correct = (oof.y == oof.p).to_numpy().astype(float)
    conf = oof.p_proba.to_numpy()
    wrong = oof[oof.y != oof.p]
    calib = {"ece_15_bins": ece(conf, correct), "mean_confidence": float(conf.mean()),
             "accuracy": float(correct.mean()),
             "mean_confidence_when_wrong": float(wrong.p_proba.mean()),
             "true_in_top3_when_wrong": float(np.mean([t in s.split("|") for t, s in zip(wrong.y, wrong.top3)]))}

    cur = pd.read_csv(RUN / "curves.csv")
    mean = cur.groupby("epoch")[["val_loss", "train_macro_f1", "val_macro_f1"]].mean()
    per_fold_min = cur.loc[cur.groupby("fold").val_loss.idxmin(), ["fold", "epoch"]]
    curves = {"mean_val_loss_min_epoch": int(mean.val_loss.idxmin()),
              "mean_val_loss_min": float(mean.val_loss.min()),
              "mean_val_loss_final": float(mean.val_loss.iloc[-1]),
              "per_fold_val_loss_min_epoch": {int(f): int(e) for f, e in per_fold_min.values},
              "mean_train_f1_reaches_0.99_epoch": int(mean.index[mean.train_macro_f1 >= 0.99][0]),
              "mean_val_f1_final": float(mean.val_macro_f1.iloc[-1])}

    te = pd.read_csv(ROOT / "reports" / "e3_final" / "sequence_base_test.preds.csv")
    errs = te[te.y != te.p]
    test = {"n_test": int(len(te)), "n_errors": int(len(errs)),
            "errors_by_signer": {s: [int((g.y != g.p).sum()), int(len(g))] for s, g in te.groupby("signer")},
            "top_confusions": errs.groupby(["y", "p"]).size().sort_values(ascending=False).head(10)
                                  .reset_index().values.tolist()}

    out = {"calibration_dev_oof": calib, "learning_curves_base": curves, "test_errors": test}
    dst = ROOT / "reports" / "e3_final" / "cifras_informe.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
