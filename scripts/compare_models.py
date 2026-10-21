"""Comparación cuantitativa: línea base vs. modelos secuenciales, sobre los mismos 7 folds de dev.

    python scripts/compare_models.py                       # todos los runs de reports/e2_avance/runs
    python scripts/compare_models.py --candidate base      # run que se contrasta con la línea base

1) Reconstruye las predicciones fuera de muestra (OOF) de la línea base de la Entrega 1 con su propio
   código (scripts/train_baseline.py, sin cambios) y las guarda en reports/e2_avance/baseline/.
2) Tabla de experimentos: macro-F1 por fold (media ± d.e.), agrupado, exactitud, top-3, parámetros, costo.
3) Regla de decisión fijada en el plan (sección 8) para el candidato frente a la línea base:
   Δ macro-F1 ≥ 0,05 y mejora en ≥ 5 de 7 folds, con IC 95 % por remuestreo de señantes.
4) Figuras: curvas de aprendizaje, macro-F1 por fold, por señante y por clase, matrices de confusión.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.eval.metrics import (classification_metrics, per_class_f1,  # noqa: E402
                              per_signer_macro_f1, signer_bootstrap_diff)

RUNS = ROOT / "reports" / "e2_avance" / "runs"
OUT = ROOT / "reports" / "e2_avance"
FIG = OUT / "figures"
NUMBERS = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"]


def baseline_oof(variant: str, max_abs: float | None) -> pd.DataFrame:
    """OOF de la regresión logística de la Entrega 1 (mismo código, mismos folds)."""
    dst = OUT / "baseline" / f"{variant}_oof.csv"
    if dst.exists():
        return pd.read_csv(dst)
    spec = importlib.util.spec_from_file_location("train_baseline", ROOT / "scripts" / "train_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    df = tb.load_table()
    classes = sorted(df.label.unique())
    X = tb.load_features(df, needs_landmarks=True, max_abs=max_abs)
    y = df.label.to_numpy()
    dev = (df.partition == "dev").to_numpy()
    parts, folds = [], []
    for f in sorted(df.loc[dev, "fold"].unique()):
        tr = dev & (df.fold != f).to_numpy()
        va = dev & (df.fold == f).to_numpy()
        pred, proba, cost = tb.fit_predict("logreg", X[tr], y[tr], X[va], classes)
        top3 = np.array(classes)[np.argsort(proba, 1)[:, -3:][:, ::-1]]
        parts.append(pd.DataFrame({"video_id": df.video_id[va].to_numpy(), "y": y[va], "p": pred,
                                   "signer": df.signer_key[va].to_numpy(), "fold": int(f), "p_proba": proba.max(1),
                                   "top3": ["|".join(r) for r in top3]}))
        folds.append({"fold": int(f), **classification_metrics(y[va], pred, proba, classes), **cost})
    oof = pd.concat(parts, ignore_index=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(dst, index=False)
    (dst.parent / f"{variant}_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    return oof


def fold_metrics(oof: pd.DataFrame, classes) -> pd.DataFrame:
    rows = []
    for f, g in oof.groupby("fold"):
        top3 = np.mean([t in s.split("|") for t, s in zip(g.y, g.top3)])
        rows.append({"fold": f, "macro_f1": f1_score(g.y, g.p, labels=classes, average="macro", zero_division=0),
                     "accuracy": float(np.mean(g.y == g.p)), "top3_accuracy": top3})
    return pd.DataFrame(rows).set_index("fold")


def load_runs() -> dict[str, dict]:
    runs = {}
    for d in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(d.read_text(encoding="utf-8"))
        if "folds" in s and len(s["folds"]) == 7:
            s["oof"] = pd.read_csv(d.parent / "oof.csv")
            s["curves"] = pd.read_csv(d.parent / "curves.csv")
            runs[s["name"]] = s
    return runs


def plot_curves(run: dict, name: str) -> Path:
    c = run["curves"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for f, g in c.groupby("fold"):
        ax[0].plot(g.epoch, g.train_loss, color="#2a6fdb", alpha=0.25, lw=0.8)
        ax[0].plot(g.epoch, g.val_loss, color="#d9480f", alpha=0.25, lw=0.8)
        ax[1].plot(g.epoch, g.train_macro_f1, color="#2a6fdb", alpha=0.25, lw=0.8)
        ax[1].plot(g.epoch, g.val_macro_f1, color="#d9480f", alpha=0.25, lw=0.8)
    m = c.groupby("epoch").mean(numeric_only=True)
    ax[0].plot(m.index, m.train_loss, color="#2a6fdb", lw=2, label="entrenamiento (sin aumentos)")
    ax[0].plot(m.index, m.train_aug_loss, color="#2a6fdb", lw=1.2, ls="--", label="entrenamiento (con aumentos)")
    ax[0].plot(m.index, m.val_loss, color="#d9480f", lw=2, label="validación (señantes nuevos)")
    ax[1].plot(m.index, m.train_macro_f1, color="#2a6fdb", lw=2, label="entrenamiento")
    ax[1].plot(m.index, m.val_macro_f1, color="#d9480f", lw=2, label="validación")
    ax[0].set(xlabel="época", ylabel="entropía cruzada", title=f"Pérdida — {name}")
    ax[1].set(xlabel="época", ylabel="macro-F1", title=f"Macro-F1 — {name}", ylim=(0, 1.02))
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / f"curvas_{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_lr_curves(runs: dict, names: list[str], fname: str, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for n in names:
        if n not in runs:
            continue
        m = runs[n]["curves"].groupby("epoch").mean(numeric_only=True)
        line = ax[0].plot(m.index, m.val_loss, lw=1.6, label=n)[0]
        ax[0].plot(m.index, m.train_loss, lw=1, ls=":", color=line.get_color())
        ax[1].plot(m.index, m.val_macro_f1, lw=1.6, label=n, color=line.get_color())
    ax[0].set(xlabel="época", ylabel="pérdida (— val · ··· train)", title=title)
    ax[1].set(xlabel="época", ylabel="macro-F1 de validación (media de 7 folds)", ylim=(0, 1))
    ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="base")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    models = {"logreg": baseline_oof("logreg", None), "logreg_plausible": baseline_oof("logreg_plausible", 6.0)}
    classes = sorted(models["logreg"].y.unique())
    runs = load_runs()
    for n, r in runs.items():
        models[n] = r["oof"]

    # --- tabla de experimentos
    rows = []
    base_folds = fold_metrics(models["logreg"], classes)
    for n, oof in models.items():
        fm = fold_metrics(oof, classes)
        r = runs.get(n)
        rows.append({"modelo": n, "cambio": " ".join(r["overrides"]) if r else ("línea base E1" if n == "logreg" else "línea base E1, filtro max_abs=6"),
                     "macro_f1_media": fm.macro_f1.mean(), "macro_f1_de": fm.macro_f1.std(ddof=1),
                     "macro_f1_agrupado": f1_score(oof.y, oof.p, labels=classes, average="macro", zero_division=0),
                     "exactitud": fm.accuracy.mean(), "top3": fm.top3_accuracy.mean(),
                     "folds_mejor_que_logreg": int((fm.macro_f1 > base_folds.macro_f1).sum()) if n != "logreg" else None,
                     "peor_señante": min(per_signer_macro_f1(oof).values()),
                     "parametros": r["n_params"] if r else 43 * 494,
                     "ajuste_s_por_fold": r["summary"]["fit_seconds"]["mean"] if r else None,
                     "ms_por_video_1x1": r["summary"]["predict_ms_single_video"]["mean"] if r else None})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tabla_experimentos.csv", index=False)

    # --- decisión del candidato frente a la línea base (regla fijada en el plan)
    decision = {}
    if args.candidate in models:
        cand = models[args.candidate]
        for ref in ("logreg", "logreg_plausible"):
            fb, fc = fold_metrics(models[ref], classes), fold_metrics(cand, classes)
            boot = signer_bootstrap_diff(models[ref], cand, classes)
            delta = float(fc.macro_f1.mean() - fb.macro_f1.mean())
            wins = int((fc.macro_f1 > fb.macro_f1).sum())
            decision[ref] = {"delta_macro_f1_media_folds": delta, "folds_mejorados": wins,
                             "delta_por_fold": (fc.macro_f1 - fb.macro_f1).round(4).to_dict(),
                             "bootstrap_señantes": boot,
                             "cumple_regla": bool(delta >= 0.05 and wins >= 5)}
        (OUT / f"decision_{args.candidate}.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")

        # por fold
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        x = np.arange(1, 8)
        for k, (n, col) in enumerate([("logreg", "#868e96"), (args.candidate, "#d9480f")]):
            ax.bar(x + (k - 0.5) * 0.38, fold_metrics(models[n], classes).macro_f1, 0.38, label=n, color=col)
        ax.set(xlabel="fold de validación (2 señantes nuevos)", ylabel="macro-F1", xticks=x, ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "macro_f1_por_fold.png", dpi=150)
        plt.close(fig)

        # por señante
        sb, sc = per_signer_macro_f1(models["logreg"]), per_signer_macro_f1(cand)
        sig = sorted(sb, key=lambda s: sb[s])
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for s in sig:
            ax.plot([s, s], [sb[s], sc[s]], color="#adb5bd", lw=1)
        ax.scatter(sig, [sb[s] for s in sig], color="#868e96", label="logreg", zorder=3)
        ax.scatter(sig, [sc[s] for s in sig], color="#d9480f", label=args.candidate, zorder=3)
        ax.set(ylabel="macro-F1 del señante", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "macro_f1_por_senante.png", dpi=150)
        plt.close(fig)

        # por clase
        cb, cc = per_class_f1(models["logreg"], classes), per_class_f1(cand, classes)
        order = sorted(classes, key=lambda c: cc[c] - cb[c])
        fig, ax = plt.subplots(figsize=(10, 3.4))
        colors = ["#1c7ed6" if c in NUMBERS else "#495057" for c in order]
        ax.bar(order, [cc[c] - cb[c] for c in order], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(ylabel=f"F1 {args.candidate} − F1 logreg", title="Cambio de F1 por clase (azul = números)")
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "delta_f1_por_clase.png", dpi=150)
        plt.close(fig)

        # matrices de confusión de los números (donde la línea base más falla)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
        for a, n in zip(ax, ("logreg", args.candidate)):
            o = models[n][models[n].y.isin(NUMBERS)]
            cm = confusion_matrix(o.y, o.p.where(o.p.isin(NUMBERS), "otra"), labels=NUMBERS + ["otra"])[:10]
            cm = cm / cm.sum(1, keepdims=True)
            a.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
            a.set(xticks=range(11), yticks=range(10), title=f"Números — {n}", xlabel="predicción", ylabel="real")
            a.set_xticklabels(NUMBERS + ["otra"], rotation=60, fontsize=7)
            a.set_yticklabels(NUMBERS, fontsize=7)
            for i in range(10):
                for j in range(11):
                    if cm[i, j] >= 0.1:
                        a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "confusion_numeros.png", dpi=150)
        plt.close(fig)

        per = pd.DataFrame({"logreg": cb, args.candidate: cc})
        per["delta"] = per[args.candidate] - per["logreg"]
        per["tipo"] = ["número" if c in NUMBERS else "cortesía" for c in per.index]
        per.sort_values("delta").to_csv(OUT / f"f1_por_clase_{args.candidate}.csv")
        pd.DataFrame({"logreg": sb, args.candidate: sc}).to_csv(OUT / f"f1_por_senante_{args.candidate}.csv")

    for n, r in runs.items():
        plot_curves(r, n)
    plot_lr_curves(runs, ["lr3e-4", "base", "lr3e-3"], "barrido_lr.png", "Tasa de aprendizaje (3e-4, 1e-3, 3e-3)")
    plot_lr_curves(runs, ["base", "wd0.1", "dropout0.5", "sin_aumentos"], "barrido_regularizacion.png", "Regularización")
    plot_lr_curves(runs, ["base", "opt_adam", "opt_sgd"], "barrido_optimizador.png", "Optimizador")
    plot_lr_curves(runs, ["base", "lstm", "conv1d"], "barrido_encoder.png", "Codificador temporal")

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
