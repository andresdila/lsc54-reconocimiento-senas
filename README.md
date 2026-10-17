# Reconocimiento de señas aisladas de Lengua de Señas Colombiana (LSC-54)

Proyecto integrador — Técnicas Avanzadas de Modelado, Maestría en Analítica Aplicada e IA, Universidad de La Sabana (2026-II).

**Equipo:** Andrés Díaz · Lorena Valero · Jorge Anaya · Sabrina Miranda

**Pregunta experimental (borrador):** bajo una partición independiente del señante, ¿un modelo secuencial (LSTM/GRU) sobre landmarks de MediaPipe mejora el macro-F1 frente a un baseline clásico sobre estadísticas de los landmarks, y a qué costo computacional?

> Estado: Entrega 1 (plan experimental) en construcción.

## Reproducir

```bash
# 1. Entorno (Python 3.11 o 3.12)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Datos: descarga (~0,97 GB), verifica MD5, deduplica, extrae y genera el inventario
python scripts/download_data.py

# 3. Pruebas
pytest -q
```

Los datos **no** se versionan. `scripts/download_data.py` los obtiene de la fuente oficial y deja:
- `data/raw/videos/<sesión>/<señante>/<seña>/<rep>.avi` (ignorado por Git)
- `data/inventory/lsc54_v4_inventory.csv` + `.audit.json` (versionados: evidencia de auditoría)

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

## Datos y licencia

LSC-54 (Mora-Zarate, Garzón-Castro & Castellanos-Rivillas, 2025, *Data in Brief* 63:112145), Science Data Bank, DOI [10.57760/sciencedb.25639](https://doi.org/10.57760/sciencedb.25639), licencia **CC BY-NC 4.0**. Uso académico, no comercial; los videos no se redistribuyen en este repositorio.
