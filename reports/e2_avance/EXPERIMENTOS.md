# Entrega 2 — protocolo de experimentos

Todos los experimentos usan la partición v1 (mismos 7 folds de dev que la línea base), semilla 2026
(+ número de fold en cada fold) y la configuración base de `configs/model.yaml`. Cada experimento
cambia **un solo factor** respecto a la base (guía de Google, *A scientific approach to improving
model performance*), de modo que la diferencia se puede atribuir a ese factor.

La métrica que se reporta es la de la **última época** (número fijo de épocas). La mejor época por
fold se guarda solo como diagnóstico de las curvas: elegirla con el fold de validación inflaría la
estimación. El test bloqueado no se usa en ningún experimento de esta entrega.

| Experimento | Factor que cambia | Pregunta |
|---|---|---|
| `base` | — | Configuración candidata del plan (sección 9) |
| `lr3e-4`, `lr3e-3` | `train.lr` | ¿La tasa de aprendizaje inicial (1e-3) está en la zona estable? |
| `wd0.1` | `train.weight_decay` 0,01 → 0,1 | ¿Más regularización L2 desacoplada reduce la brecha train/val? |
| `dropout0.5` | `model.dropout` 0,3 → 0,5 | Ídem, en la cabeza |
| `sin_aumentos` | `augment.enabled=false` | ¿Cuánto aportan los aumentos a la generalización a señantes nuevos? |
| `opt_adam` | AdamW → Adam (L2 acoplado) | ¿Importa desacoplar el decaimiento de pesos? |
| `opt_sgd` | AdamW → SGD Nesterov (lr 0,03) | ¿Un optimizador sin adaptación converge igual en 80 épocas? |
| `lstm` | GRU → LSTM | Variante del plan |
| `conv1d` | GRU → convolución temporal 1D | ¿La recurrencia es lo que aporta? |
| `sin_ncuadros` | `data.use_n_frames=false` | ¿El modelo usa la duración como atajo? |
| `pesos_clase` | `train.class_weights=true` | ¿Compensar el desbalance mejora el macro-F1? |
| `sin_mascara` | `data.use_mask=false` | ¿Sirve decirle a la red qué no se detectó? |
| `sin_interpolacion` | `data.max_gap=0` | ¿Sirve rellenar vacíos cortos de detección? |

Reproducir un experimento:

```bash
python scripts/train_sequence.py --name lr3e-3 --set train.lr=0.003
python scripts/compare_models.py --candidate base
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional: explorar los runs
```

Resultados por experimento en `runs/<nombre>/` (`summary.json`, `curves.csv`, `oof.csv`);
tabla consolidada en `tabla_experimentos.csv`; figuras en `figures/`.
