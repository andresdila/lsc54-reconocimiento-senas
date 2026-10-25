# Ficha del modelo y de los datos: BiGRU sobre landmarks para LSC-54

Proyecto integrador de *Técnicas Avanzadas de Modelado*, Maestría en Analítica Aplicada e IA, Universidad de La Sabana (2026-II).
Equipo: Andrés Díaz, Lorena Valero, Jorge Anaya y Sabrina Miranda.
Formato basado en *Model Cards for Model Reporting* (Mitchell et al., 2019) y *Datasheets for Datasets* (Gebru et al., 2021).

> **Resumen.** Clasificador de 43 señas aisladas de la Lengua de Señas Colombiana (LSC) a partir de landmarks de MediaPipe, evaluado siempre sobre señantes que no participaron en el entrenamiento. Macro-F1 de **0,811 ± 0,079** en validación cruzada por señante y **0,881** en un test bloqueado de 4 señantes (cifra optimista, ver limitaciones). Es una herramienta de apoyo a la práctica de vocabulario, no un traductor.

---

## 1. Detalles del modelo

| Campo | Contenido |
|---|---|
| Nombre | BiGRU base (`configs/model.yaml`, sin cambios) |
| Tipo | Red recurrente bidireccional con atención temporal para clasificación multiclase de secuencias |
| Entrada | Secuencia de 32 cuadros: 98 coordenadas (x, y) de MediaPipe Holistic (7 puntos de pose del tren superior + 21 por mano), normalizadas por hombros, más 3 canales de máscara de presencia y log(n_frames) |
| Arquitectura | Proyección 101 → 128 con LayerNorm, GRU bidireccional de 2 capas × 128 unidades, atención temporal, dropout 0,3, capa densa de 43 clases |
| Parámetros | 552.023 |
| Entrenamiento | Entropía cruzada; AdamW (lr 1e-3, weight decay 0,01); lote 32; 80 épocas fijas; calentamiento de 5 épocas y descenso coseno; recorte de gradiente 1,0; semilla 2026 |
| Aumentos (solo en entrenamiento) | Recorte temporal 0,10; escala 0,10; rotación ±10°; ruido gaussiano 0,01; sin reflejo horizontal (alteraría la lateralidad) |
| Marco | PyTorch 2.14.1 (CPU), scikit-learn 1.9.1, NumPy 1.26.4, Python 3.11.17 |
| Línea base de referencia | Regresión logística sobre 493 estadísticas por video (21.242 parámetros) |
| Evaluación final | Commit `9431af2`, árbol de trabajo limpio; configuración congelada en el commit `dbbe611` |
| Licencia del código | Por definir por el equipo (el repositorio aún no incluye un archivo `LICENSE`) |

## 2. Uso previsto

- **Uso previsto:** retroalimentación de práctica de vocabulario básico de LSC (cortesía y números), con una persona frente a una cámara. Se recomienda mostrar las **tres señas más probables** (top-3 = 0,955 en test) y no una única respuesta.
- **Usuarios previstos:** estudiantes de LSC y docentes, como apoyo y nunca como autoridad.
- **Usos fuera de alcance:** traducción o interpretación, sustitución de intérpretes, evaluación o calificación de personas, reconocimiento de lengua de señas continua, uso comercial (la licencia de los datos lo impide), vigilancia o identificación de personas.

## 3. Factores relevantes

El desempeño depende de factores que el modelo no controla:

- **La persona que firma.** El macro-F1 por señante en test va de 0,69 a 0,92.
- **La calidad de detección de MediaPipe**, que varía entre personas (detección de manos de 0,46 a 0,97) y con la iluminación, la ropa y las oclusiones.
- **La cámara.** Los videos tienen dos resoluciones (640×360 y 640×480); las coordenadas se convierten a píxeles antes de normalizar para que el modelo no aprenda la cámara en lugar de la seña.
- **El tipo de seña.** Los números se confunden más que las expresiones de cortesía.

No se dispone de metadatos demográficos de los señantes (se desconoce si son sordos, oyentes o intérpretes, su edad o su lateralidad).

## 4. Métricas

Métrica principal: **macro-F1** (promedio del F1 de las 43 clases, con igual peso). Secundarias: exactitud, exactitud top-3, F1 por clase, F1 por señante y costo (parámetros, tiempo de ajuste, latencia).

Regla de decisión fijada antes de ver resultados: el modelo candidato es mejor si su macro-F1 en validación cruzada supera al de la línea base en al menos 0,05 y mejora en al menos 5 de los 7 folds, con intervalo de confianza por remuestreo de señantes.

## 5. Datos de evaluación

Partición v1 **por señante**, fijada antes de modelar (semilla 2026). Ningún señante aparece en dos particiones.

| Partición | Señantes | Videos | Uso |
|---|---|---|---|
| Desarrollo | 14 | 1.359 | Validación cruzada agrupada de 7 folds (192 a 206 videos por fold) y entrenamiento final |
| Test bloqueado | 4 (S2-01, S2-03, S2-29, S2-31) | 381 | Evaluado **una sola vez** al final; el script se niega a repetirlo |

## 6. Datos de entrenamiento y procedencia

Ver la ficha de datos (sección 9). El modelo final se entrenó con los 14 señantes de desarrollo (1.359 videos).

## 7. Análisis cuantitativo

**Validación cruzada por señante (7 folds, media ± desviación estándar):**

| Modelo | Macro-F1 | Exactitud | Top-3 | Peor señante |
|---|---|---|---|---|
| Clase mayoritaria | 0,001 | 0,032 | 0,079 | - |
| Regresión logística | 0,548 ± 0,166 | 0,569 | 0,782 | 0,108 |
| **BiGRU base** | **0,811 ± 0,079** | **0,811** | **0,923** | **0,527** |

La BiGRU mejora en los 7 folds, con diferencia media de +0,263 (IC 95 % por remuestreo de señantes: [+0,16; +0,37]).

**Test bloqueado (4 señantes, 381 videos, evaluación única):**

| Métrica | Regresión logística | BiGRU base |
|---|---|---|
| Macro-F1 | 0,685 | **0,881** |
| Exactitud | 0,669 | **0,869** |
| Top-3 | 0,829 | **0,955** |
| Tiempo de ajuste (dev completo, CPU) | 0,9 s | 227 s |
| Latencia por video (lote / individual) | 0,01 ms | 0,39 ms / 6,8 ms |

**Macro-F1 por señante en test:** S2-01 0,90 · S2-03 0,92 · S2-29 0,82 · S2-31 0,69.

**Errores en test:** 50 de 381 videos (13 %). Diez de las 43 clases tienen F1 = 1,0. La clase más difícil es *seis* (F1 = 0,375). Los números son el 31 % de los videos de test pero el 54 % de los errores (tasa de error de 22,5 % frente a 8,8 % en cortesía), y 20 de esos 27 errores son confusiones entre números. Las confusiones más frecuentes son *seis → uno* (6 videos), *bien → por favor* (3), *cuatro → tres* (3), *nueve → ocho* (2) y *ocho → siete* (2). En validación cruzada el F1 medio de los números es 0,709 frente a 0,849 de las expresiones de cortesía.

**Ablaciones (un factor a la vez, macro-F1 en CV):** con la tasa y el weight decay de la base sin reajustar, Adam con L2 acoplado (0,599) y SGD (0,635) subajustan (macro-F1 de entrenamiento de 0,79 a 0,80 en la época 80), lo que describe su comportamiento con esa configuración y no una superioridad general de AdamW; sin aumentos 0,771; LSTM 0,754; sin máscara 0,794; sin log(n_frames) 0,791. Las variantes que superan a la base (Conv1D 0,828; weight decay 0,1 y sin interpolación 0,819) lo hacen por menos de 0,02 con intervalos que incluyen el cero, por lo que **no se seleccionaron** (evitar sesgo de selección).

## 8. Consideraciones éticas, riesgos y limitaciones

| Riesgo | Estado |
|---|---|
| Representatividad: 18 señantes sin metadatos, 4 en test | Se reporta el desempeño por señante; las conclusiones se limitan a este grupo |
| Fuga de identidad entre particiones | Controlada (S1 excluido, partición por señante, pruebas automáticas con fuga plantada) |
| Detección desigual de MediaPipe | Medida y reportada; se recomienda avisar al usuario cuando sea baja |
| Atajos no lingüísticos (duración, resolución, detección) | Ablaciones sin n_frames (−0,020) y sin máscara (−0,016), diferencias sobre predicciones agrupadas: no son atajos dominantes |
| Desbalance y cobertura (22 clases con solo 9 señantes) | Macro-F1 como métrica; pesos de clase sin efecto (−0,002) |
| Sobreconfianza (ECE 0,114; confianza media 0,925 frente a exactitud 0,812 en CV) | Pendiente: recalibrar con escalado de temperatura antes de definir un umbral de abstención |
| Privacidad | Rostros pixelados por los autores; no se redistribuyen videos; sin landmarks faciales |
| Uso indebido o sobreafirmación | No presentar como traductor; validar con la comunidad sorda y docentes de LSC |

**Limitaciones principales**

1. **Pocas personas.** Con 18 señantes y 4 en test no hay evidencia de generalización a otras poblaciones, cámaras o condiciones de iluminación.
2. **El test es optimista.** Los señantes de test tienen mejor detección de manos que los de desarrollo (0,73 frente a 0,57). La cifra de validación cruzada (0,811 ± 0,079) es la mejor referencia de lo que cabe esperar.
3. **Alcance léxico.** 43 señas aisladas de una sola campaña; reconocer vocabulario no equivale a comprender la lengua (gramática, expresión facial, uso del espacio).
4. **Comparación acotada.** La línea base se fijó con C = 1; en un análisis de sensibilidad con los mismos folds (`scripts/tune_baseline.py`), la mejor opción entre regresión logística con C de 0,01 a 10, bosque aleatorio y SVM RBF alcanza 0,554 ± 0,163, de modo que la brecha con la BiGRU se mantiene en 0,26. No se probó gradient boosting.
5. **Incertidumbre.** Algunos folds validan clases con un único señante; los F1 por clase en test (6 a 13 videos) son indicativos y no concluyentes.

**Recomendaciones:** usar top-3 y, tras recalibrar las probabilidades, un umbral de confianza para abstenerse; medir la detección por usuario; validar con la comunidad sorda y docentes de LSC antes de cualquier uso real; reentrenar con más señantes y condiciones.

---

## 9. Ficha de datos (datasheet resumida)

| Aspecto | Descripción |
|---|---|
| **Conjunto** | LSC-54 (Mora-Zarate, Garzón-Castro y Castellanos-Rivillas, 2025, *Data in Brief* 63:112145). DOI: 10.57760/sciencedb.25639 (Science Data Bank) |
| **Motivación** | Respaldar una herramienta de enseñanza de vocabulario básico de LSC |
| **Contenido original** | 54 señas (11 colores, 33 de cortesía, 10 números), 22 participantes, entre 3 y 5 repeticiones por seña |
| **Subconjunto de trabajo** | Campaña S2 (cortesía y números): **1.740 videos únicos, 43 clases, 18 señantes**. El inventario completo tiene 2.850 videos sin duplicados. Se excluye S1 (colores) porque su numeración de señantes es independiente y no permite garantizar la partición por persona |
| **No se usa** | La versión aumentada ×45 (51,4 GB): multiplicar los videos antes de particionar es una fuente de fuga |
| **Licencia** | **CC BY-NC 4.0**: uso académico y no comercial, con atribución. Los videos **no** se redistribuyen en el repositorio |
| **Privacidad** | Rostros pixelados por los autores; el proyecto no usa landmarks faciales y versiona solo inventario, métricas y particiones |
| **Auditoría de calidad** | 113.658 cuadros procesados; pose detectada en el 99,5 % de los cuadros; mano más activa en el 60 %; el 32 % de los videos queda bajo el 50 % de detección de manos (se conservan y se usa la tasa como eje de análisis). Detección de manos por señante de 0,46 a 0,97 |
| **Preprocesamiento** | MediaPipe Holistic 0.10.21; conversión a píxeles; centrado en el punto medio de los hombros y escala por su distancia; remuestreo a 32 cuadros; interpolación de vacíos de hasta 5 cuadros; máscara de presencia |
| **Sesgos conocidos** | Sin metadatos demográficos; detección desigual entre personas; dos resoluciones de video |
| **Mantenimiento** | El inventario y la partición están versionados (`data/inventory/`, `splits/`); el script `download_data.py` obtiene los videos de la fuente oficial y verifica su MD5 |

---

## 10. Reproducibilidad

Requiere Python 3.11 o 3.12 (MediaPipe 0.10.21 no funciona con 3.13 o superior).

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/make_split.py --check-only            # confirma que la partición v1 se reproduce
python scripts/extract_landmarks.py --workers 2      # ≈45 min
python scripts/train_baseline.py --model logreg      # línea base en validación cruzada
python scripts/train_sequence.py --name base         # BiGRU, ≈35 min en CPU, 7 folds
python scripts/run_experiments.py                    # los 14 experimentos de un factor
python scripts/compare_models.py --candidate base    # tabla, regla de decisión, IC y figuras
pytest -q                                            # 22 pruebas, incluye controles anti-fuga
```

El test bloqueado **no** forma parte de esta secuencia: `--final-test` se evaluó una sola vez y los scripts se niegan a repetirlo. Los resultados quedaron versionados en `reports/e1_plan/baseline_logreg_test.json` y `reports/e3_final/sequence_base_test.json`; la decisión previa está en `reports/e3_final/DECISION.md`.

Verificación de reproducibilidad: el mismo experimento base reentrenado en otra máquina (Linux, Python 3.13) dio macro-F1 0,808 ± 0,079 frente a 0,811 ± 0,079 en la máquina original, con diferencias por fold de a lo sumo 0,03.

**Pendiente:** fijar versiones exactas en un `requirements.lock` y agregar un archivo `LICENSE` al repositorio.

## 11. Referencias

- Mora-Zarate, J. E., Garzón-Castro, C. L. y Castellanos-Rivillas, J. A. (2025). LSC-54: A landmark-based dataset for Colombian Sign Language. *Data in Brief, 63*, 112145. https://doi.org/10.1016/j.dib.2025.112145
- Mitchell, M., et al. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.
- Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92. https://doi.org/10.1145/3458723
