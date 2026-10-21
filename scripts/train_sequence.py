"""Modelo secuencial (BiGRU / BiLSTM / Conv1D) sobre landmarks, con validación cruzada por señante.

    python scripts/train_sequence.py --name base                       # configs/model.yaml tal cual
    python scripts/train_sequence.py --name lr3e-3 --set train.lr=0.003  # un factor distinto
    python scripts/train_sequence.py --name lstm --set model.encoder=lstm
    python scripts/train_sequence.py --name base --folds 1 2             # prueba rápida
    python scripts/train_sequence.py --name final --final-test           # UNA sola vez, al congelar el modelo

Protocolo (idéntico al de la línea base):
  * mismos videos (src/lsc/data/table.py) y mismos 7 folds de splits/split_v1.csv;
  * en cada fold el modelo se entrena desde cero con los otros 6 folds de dev;
  * estandarización y log(n_frames) se ajustan solo con el entrenamiento del fold;
  * número FIJO de épocas: la métrica que se reporta es la de la última época. No se elige la
    época con el fold de validación (eso inflaría la estimación). La mejor época se guarda solo
    como diagnóstico de las curvas;
  * el test bloqueado solo se toca con --final-test y el script se niega a repetirlo.

Salidas:
  reports/e2_avance/runs/<name>/summary.json  configuración, git, métricas por fold, resumen, costo
  reports/e2_avance/runs/<name>/curves.csv    pérdida/exactitud/macro-F1 por época y fold
  reports/e2_avance/runs/<name>/oof.csv       predicciones fuera de muestra (para comparar y analizar errores)
  mlruns/                                     el mismo registro en MLflow (ignorado por Git)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.data.table import load_split_config, load_table  # noqa: E402
from lsc.eval.metrics import (classification_metrics, per_class_f1,  # noqa: E402
                              per_signer_macro_f1)
from lsc.features.landmarks import D  # noqa: E402
from lsc.features.sequence import D_IN, augment, prepare  # noqa: E402
from lsc.models.sequence import SeqClassifier, n_params  # noqa: E402

LANDMARKS = ROOT / "data" / "processed" / "landmarks"


# ---------------------------------------------------------------- configuración
def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(overrides: list[str]) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for p in path:
            node = node[p]
        if last not in node:
            raise KeyError(f"clave desconocida: {key}")
        node[last] = parse_value(value)
    return cfg


def git_info() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts",
                                              "configs", "splits"], text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


# ---------------------------------------------------------------- datos
class Standardizer:
    """Media y desviación de cada coordenada sobre los cuadros DETECTADOS del entrenamiento del fold."""

    def fit(self, X: np.ndarray, log_t: np.ndarray) -> "Standardizer":
        coords = X[..., :D].reshape(-1, D)
        present = np.repeat(X[..., D:], [14, 42, 42], axis=-1).reshape(-1, D) > 0
        cnt = present.sum(0).clip(min=1)
        self.mean = (coords * present).sum(0) / cnt
        self.std = np.sqrt((((coords - self.mean) * present) ** 2).sum(0) / cnt).clip(min=1e-3)
        self.t_mean, self.t_std = float(log_t.mean()), float(log_t.std() + 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        present = np.repeat(X[..., D:], [14, 42, 42], axis=-1) > 0
        X[..., :D] = np.where(present, (X[..., :D] - self.mean) / self.std, 0.0)
        return X.astype(np.float32)

    def transform_t(self, log_t: np.ndarray) -> np.ndarray:
        return ((log_t - self.t_mean) / self.t_std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "t_mean": self.t_mean, "t_std": self.t_std}


def make_batches(n: int, batch_size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


# ---------------------------------------------------------------- entrenamiento
def build_model(cfg: dict, n_classes: int) -> SeqClassifier:
    m = cfg["model"]
    return SeqClassifier(D_IN, n_classes, encoder=m["encoder"], d_model=m["d_model"], hidden=m["hidden"],
                         layers=m["layers"], dropout=m["dropout"], rnn_dropout=m["rnn_dropout"],
                         use_n_frames=cfg["data"]["use_n_frames"])


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    t = cfg["train"]
    # sin decaimiento de pesos en sesgos ni en LayerNorm/BatchNorm (práctica estándar con AdamW)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=t["lr"])
    if t["optimizer"] == "adam":  # Adam con L2 acoplado (para contrastar con AdamW)
        return torch.optim.Adam(groups, lr=t["lr"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=t["lr"], momentum=0.9, nesterov=True)
    raise ValueError(t["optimizer"])


def lr_at(epoch: int, cfg: dict) -> float:
    """Calentamiento lineal y luego coseno hasta 0, fijado al número total de épocas."""
    t = cfg["train"]
    if epoch < t["warmup_epochs"]:
        return t["lr"] * (epoch + 1) / t["warmup_epochs"]
    prog = (epoch - t["warmup_epochs"]) / max(1, t["epochs"] - t["warmup_epochs"])
    return t["lr"] * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def predict(model, X, T, batch_size=256):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]), torch.from_numpy(T[i:i + batch_size]))
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def evaluate(model, X, T, y, classes, loss_fn) -> dict:
    logits = predict(model, X, T)
    loss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(y)).item())
    proba = torch.softmax(torch.from_numpy(logits), 1).numpy()
    m = classification_metrics(np.array(classes)[y], np.array(classes)[proba.argmax(1)], proba, classes)
    return {"loss": loss, **m, "proba": proba}


def train_fold(cfg, seqs_tr, y_tr, seqs_va, y_va, classes, seed, log_epoch=None):
    """Entrena desde cero; devuelve modelo final, curvas, probabilidades de validación y costo."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, a, t = cfg["data"], cfg["augment"], cfg["train"]
    kw = dict(n_frames=d["n_frames"], max_gap=d["max_gap"], max_abs=d["max_abs"])
    Xtr_clean = np.stack([prepare(s, **kw) for s in seqs_tr])
    Xva = np.stack([prepare(s, **kw) for s in seqs_va])
    ltr = np.log(np.array([len(s) for s in seqs_tr], dtype=np.float32))
    lva = np.log(np.array([len(s) for s in seqs_va], dtype=np.float32))
    st = Standardizer().fit(Xtr_clean, ltr)
    Xtr_c, Xva_s = st.transform(Xtr_clean), st.transform(Xva)
    if not d.get("use_mask", True):  # ablación: la red no ve qué está detectado (lo ausente sigue en 0)
        Xtr_c[..., D:] = 0.0
        Xva_s[..., D:] = 0.0
    Ttr, Tva = st.transform_t(ltr), st.transform_t(lva)

    model = build_model(cfg, len(classes))
    opt = build_optimizer(cfg, model)
    weight = None
    if t["class_weights"]:
        cnt = np.bincount(y_tr, minlength=len(classes)).astype(np.float32)
        weight = torch.from_numpy(cnt.sum() / (len(classes) * cnt.clip(min=1)))
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=t["label_smoothing"])
    eval_loss = nn.CrossEntropyLoss()  # las curvas se comparan con la misma pérdida sin ponderar ni suavizar
    ytr_t = torch.from_numpy(y_tr)

    curves, t0 = [], time.perf_counter()
    for epoch in range(t["epochs"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, cfg)
        model.train()
        if a["enabled"]:
            Xep = st.transform(np.stack([augment(s, rng, **kw, crop=a["crop"], scale=a["scale"],
                                                 rot_deg=a["rot_deg"], noise=a["noise"]) for s in seqs_tr]))
            if not d.get("use_mask", True):
                Xep[..., D:] = 0.0
        else:
            Xep = Xtr_c
        run_loss, run_correct = 0.0, 0
        for idx in make_batches(len(Xep), t["batch_size"], rng):
            xb, tb, yb = torch.from_numpy(Xep[idx]), torch.from_numpy(Ttr[idx]), ytr_t[idx]
            logits = model(xb, tb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            if t["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            run_loss += float(loss.item()) * len(idx)
            run_correct += int((logits.argmax(1) == yb).sum())
        tr = evaluate(model, Xtr_c, Ttr, y_tr, classes, eval_loss)
        va = evaluate(model, Xva_s, Tva, y_va, classes, eval_loss)
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"],
               "train_aug_loss": run_loss / len(Xep), "train_aug_acc": run_correct / len(Xep),
               "train_loss": tr["loss"], "train_acc": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
               "val_loss": va["loss"], "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
               "val_top3": va["top3_accuracy"]}
        curves.append(row)
        if log_epoch:
            log_epoch(row)
    fit_seconds = time.perf_counter() - t0

    # costo de inferencia en CPU: lote completo y un video a la vez (escenario de la herramienta)
    t1 = time.perf_counter()
    predict(model, Xva_s, Tva)
    batch_ms = 1000 * (time.perf_counter() - t1) / len(Xva_s)
    model.eval()
    with torch.no_grad():
        t2 = time.perf_counter()
        for i in range(min(50, len(Xva_s))):
            model(torch.from_numpy(Xva_s[i:i + 1]), torch.from_numpy(Tva[i:i + 1]))
        single_ms = 1000 * (time.perf_counter() - t2) / min(50, len(Xva_s))
    cost = {"fit_seconds": fit_seconds, "predict_ms_per_video": batch_ms, "predict_ms_single_video": single_ms}
    return model, st, curves, va["proba"], cost


# ---------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="nombre del experimento (carpeta de resultados y run de MLflow)")
    ap.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR", help="cambios sobre configs/model.yaml")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="subconjunto de folds (solo pruebas)")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    split_cfg = load_split_config(ROOT)
    seed = split_cfg["seed"]
    cfg = load_config(args.set)
    df = load_table(ROOT, split_cfg)
    classes = sorted(df.label.unique())
    y_all = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy().astype(np.int64)
    seqs = [np.load(LANDMARKS / f"{v}.npy") for v in df.video_id]
    commit, dirty = git_info()
    meta = {"name": args.name, "overrides": args.set, "config": cfg, "seed": seed, "split_version": split_cfg["version"],
            "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
            "n_params": n_params(build_model(cfg, len(classes)))}

    mlf = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
            mlflow.set_experiment("lsc54-e2-secuencial")
            mlf = mlflow
        except ImportError:
            print("[aviso] MLflow no está instalado; solo se escriben los JSON/CSV")

    dev = (df.partition == "dev").to_numpy()
    if args.final_test:
        out = ROOT / "reports" / "e3_final" / f"sequence_{args.name}_test.json"
        if out.exists():
            sys.exit(f"[error] {out.name} ya existe: el test bloqueado se evalúa una sola vez")
        out.parent.mkdir(parents=True, exist_ok=True)
        te = ~dev
        model, st, curves, proba, cost = train_fold(cfg, [seqs[i] for i in np.flatnonzero(dev)], y_all[dev],
                                                    [seqs[i] for i in np.flatnonzero(te)], y_all[te], classes, seed)
        yt, yp = np.array(classes)[y_all[te]], np.array(classes)[proba.argmax(1)]
        oof = pd.DataFrame({"video_id": df.video_id[te].to_numpy(), "y": yt, "p": yp, "signer": df.signer_key[te].to_numpy()})
        res = {**meta, "n_train": int(dev.sum()), "n_test": int(te.sum()), **classification_metrics(yt, yp, proba, classes),
               **cost, "per_signer_macro_f1": per_signer_macro_f1(oof)}
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        oof.to_csv(out.with_suffix(".preds.csv"), index=False)
        (ROOT / "outputs").mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "standardizer": st.state(), "classes": classes, "config": cfg},
                   ROOT / "outputs" / f"sequence_{args.name}_final.pt")
        print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2, ensure_ascii=False))
        return

    out_dir = ROOT / "reports" / "e2_avance" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = sorted(df.loc[dev, "fold"].unique()) if args.folds is None else args.folds
    folds, all_curves, oof_parts = [], [], []
    run_ctx = mlf.start_run(run_name=args.name) if mlf else None
    try:
        if mlf:
            mlf.log_params({"overrides": " ".join(args.set) or "(base)", "git_commit": commit, "n_params": meta["n_params"],
                            **{f"{s}.{k}": v for s, sec in cfg.items() for k, v in sec.items()}})
        for f in fold_ids:
            tr = dev & (df.fold != f).to_numpy()
            va = dev & (df.fold == f).to_numpy()
            t_idx, v_idx = np.flatnonzero(tr), np.flatnonzero(va)

            def log_epoch(row, f=f):
                if mlf:
                    mlf.log_metrics({f"f{f}_{k}": v for k, v in row.items() if k != "epoch"}, step=row["epoch"])

            model, st, curves, proba, cost = train_fold(cfg, [seqs[i] for i in t_idx], y_all[tr],
                                                        [seqs[i] for i in v_idx], y_all[va], classes, seed + int(f),
                                                        log_epoch)
            yt, yp = np.array(classes)[y_all[va]], np.array(classes)[proba.argmax(1)]
            best = max(curves, key=lambda r: r["val_macro_f1"])
            folds.append({"fold": int(f), "n_train": int(tr.sum()), "n_val": int(va.sum()),
                          **classification_metrics(yt, yp, proba, classes), **cost,
                          "best_epoch_diagnostic": best["epoch"], "best_val_macro_f1_diagnostic": best["val_macro_f1"]})
            all_curves += [{"fold": int(f), **r} for r in curves]
            part = pd.DataFrame({"video_id": df.video_id[va].to_numpy(), "y": yt, "p": yp,
                                 "signer": df.signer_key[va].to_numpy(), "fold": int(f),
                                 "p_proba": proba.max(1)})
            top3 = np.array(classes)[np.argsort(proba, 1)[:, -3:][:, ::-1]]
            part["top3"] = ["|".join(r) for r in top3]
            oof_parts.append(part)
            print(f"[fold {f}] macro-F1 {folds[-1]['macro_f1']:.4f} · acc {folds[-1]['accuracy']:.4f} · "
                  f"{cost['fit_seconds']:.0f} s · mejor época (diagnóstico) {best['epoch']}", flush=True)

        keys = [k for k in folds[0] if k not in ("fold", "n_train", "n_val")]
        summary = {k: {"mean": float(np.mean([r[k] for r in folds])),
                       "std": float(np.std([r[k] for r in folds], ddof=1)) if len(folds) > 1 else 0.0} for k in keys}
        oof = pd.concat(oof_parts, ignore_index=True)
        pooled = classification_metrics(oof.y, oof.p, None, classes)
        res = {**meta, "folds": folds, "summary": summary, "pooled_oof": pooled,
               "per_signer_macro_f1": per_signer_macro_f1(oof), "per_class_f1_pooled": per_class_f1(oof, classes)}
        (out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(all_curves).to_csv(out_dir / "curves.csv", index=False)
        oof.to_csv(out_dir / "oof.csv", index=False)
        if mlf:
            mlf.log_metrics({"cv_macro_f1_mean": summary["macro_f1"]["mean"], "cv_macro_f1_std": summary["macro_f1"]["std"],
                             "cv_accuracy_mean": summary["accuracy"]["mean"], "cv_top3_mean": summary["top3_accuracy"]["mean"],
                             "pooled_macro_f1": pooled["macro_f1"], "fit_seconds_mean": summary["fit_seconds"]["mean"]})
            mlf.log_artifacts(str(out_dir))
        print(json.dumps({"name": args.name, "macro_f1": summary["macro_f1"], "pooled_oof": pooled,
                          "folds": [(r["fold"], round(r["macro_f1"], 4)) for r in folds]}, indent=2))
    finally:
        if run_ctx:
            mlf.end_run()


if __name__ == "__main__":
    main()
