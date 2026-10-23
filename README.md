# Reconocimiento de señas aisladas de Lengua de Señas Colombiana (LSC-54)

Proyecto integrador — Técnicas Avanzadas de Modelado, Maestría en Analítica Aplicada e IA, Universidad de La Sabana (2026-II).

**Equipo:** Andrés Díaz · Lorena Valero · Jorge Anaya · Sabrina Miranda

**Pregunta experimental:** bajo una partición independiente del señante, ¿un modelo secuencial (LSTM/GRU) sobre landmarks de MediaPipe mejora el macro-F1 frente a un baseline clásico sobre estadísticas de los landmarks, y a qué costo computacional?

> Estado: Entrega 1 (plan experimental y línea base) en `reports/e1_plan/`.

## Reproducir

```bash
# 1. Entorno (Python 3.11; MediaPipe 0.10.21 no funciona con 3.13)
conda create -n lsc54 python=3.11 -y      # Windows: usar "Anaconda Prompt"
conda activate lsc54
pip install -r requirements.txt

# 2. Datos: descarga (~0,97 GB), verifica MD5, deduplica, extrae y genera el inventario
python scripts/download_data.py
python scripts/make_split.py --check-only   # confirma que la partición v1 se reproduce

# 3. Landmarks de MediaPipe (1.740 videos; ~45 min con 2 procesos)
python scripts/extract_landmarks.py --workers 2

# 4. Línea base en validación cruzada por señante (7 folds de dev)
python scripts/train_baseline.py --model dummy_majority
python scripts/train_baseline.py --model dummy_prior
python scripts/train_baseline.py --model logreg

# 5. Modelo secuencial (Entrega 2): BiGRU y experimentos de un factor a la vez
python scripts/train_sequence.py --name base             # configs/model.yaml (~35 min en CPU, 7 folds)
python scripts/run_experiments.py --list                 # batería completa (reports/e2_avance/EXPERIMENTOS.md)
python scripts/run_experiments.py                        # corre todos y al final la comparación
python scripts/compare_models.py --candidate base        # tabla, regla de decisión, IC y figuras

# 6. Pruebas
pytest -q
```

El test bloqueado (4 señantes) **no** se usa en estas ejecuciones; solo se evalúa una vez al final con `--final-test`.

Los datos **no** se versionan. `scripts/download_data.py` los obtiene de la fuente oficial y deja:
- `data/raw/videos/<sesión>/<señante>/<seña>/<rep>.avi` (ignorado por Git)
- `data/inventory/lsc54_v4_inventory.csv` + `.audit.json` (versionados: evidencia de auditoría)
- `data/processed/landmarks/<video_id>.npy` (ignorado por Git) y `data/inventory/landmarks_audit.csv` (versionado)

## Partición (v1, fijada antes de modelar)

- **Alcance:** campaña S2 de LSC-54 (cortesía + números): 43 clases, 18 señantes, 1.740 videos únicos. Se excluye la campaña de colores (S1) porque su numeración de señantes es independiente de S2 y no permite garantizar que una persona no quede a ambos lados del split.
- **Unidad de generalización:** el señante. Ningún señante aparece en dos particiones.
- **Test bloqueado:** 4 señantes (2 que graban las 43 clases + 2 que graban 21), sorteados con semilla 2026 → 381 videos.
- **Dev:** 14 señantes en validación cruzada agrupada de 7 folds (cada fold valida con 1+1 señantes; todas las clases presentes).
- `python scripts/make_split.py --check-only` verifica que la partición versionada se reproduce; `tests/test_split.py` incluye los controles anti-fuga.

## Estructura

```
configs/      decisiones experimentales (YAML)
scripts/      puntos de entrada (descarga, extracción, entrenamiento)
src/lsc/      lógica reutilizable
tests/        pruebas automáticas (incluye controles anti-fuga)
splits/       asignación señante → partición (versionada)
reports/      informes LaTeX por entrega (e1_plan, e2_avance, e3_final)
notebooks/    exploración y comunicación
```

## Forma de trabajo

Nadie sube cambios directamente a `main`: cada integrante trabaja en su rama (`nombre/tarea`), abre un Pull Request y otra persona del equipo lo revisa antes de unirlo.

## Datos y licencia

LSC-54 (Mora-Zarate, Garzón-Castro & Castellanos-Rivillas, 2025, *Data in Brief* 63:112145), Science Data Bank, DOI [10.57760/sciencedb.25639](https://doi.org/10.57760/sciencedb.25639), licencia **CC BY-NC 4.0**. Uso académico, no comercial; los videos no se redistribuyen en este repositorio.
