# Entrega 3 — decisión previa a la evaluación del test bloqueado

Registrada el 22 de octubre de 2026, **antes** de evaluar el test bloqueado (S2-01, S2-03, S2-29, S2-31).

## Modelo que se evalúa
**BiGRU base** (`configs/model.yaml` sin cambios, commit `dbbe611`): proyección 101→128 con LayerNorm,
BiGRU de 2 capas y 128 unidades, atención temporal, dropout 0,3; AdamW (lr 1e-3, weight decay 0,01),
80 épocas fijas con calentamiento y coseno; aumentos solo en entrenamiento.
Se entrena una vez con los 14 señantes de dev y se evalúa una sola vez sobre los 4 de test.

## Justificación
1. Es la arquitectura candidata registrada en el plan (Entrega 1, sección 9) antes de ver resultados.
2. Cumple la regla de decisión del plan en dev: macro-F1 0,811 ± 0,079 frente a 0,548 ± 0,166 de la
   regresión logística; mejora en 7/7 folds; IC 95 % por remuestreo de señantes [+0,16; +0,37].
3. Ninguna variante de la ronda 1 lo supera de forma distinguible del ruido: `conv1d` +0,019
   [−0,011; +0,044], `wd0.1` +0,006 [−0,005; +0,020], `sin_interpolacion` +0,005 [−0,008; +0,019].
   Elegir la de mayor macro-F1 entre 14 variantes evaluadas en los mismos folds inflaría la
   estimación (sesgo de selección). Esas variantes se reportan como oportunidades de mejora.

## Comparación en el test
Contra la línea base de la Entrega 1 tal como fue especificada (regresión logística, `train_baseline.py
--model logreg --final-test`). Ambos modelos se entrenan con dev completo y se evalúan una única vez.
Con 4 señantes, el resultado del test se reporta con la salvedad de su amplia incertidumbre y de que
esos señantes tienen mejor detección de manos que dev (0,73 frente a 0,57; plan, sección 3).
