# Final Short Training Report

Fecha: 2026-07-12

## 1. Tiempo total

Fase corta ejecutada en modo local score-first. No se uso un entrenamiento de millones de acciones; se priorizo evaluacion oficial, dataset macro pequeno, smoke GPU y seleccion por score.

## 2. Hardware y workers

Hardware detectado por el smoke:

```text
GPU: NVIDIA GeForce RTX 4080
device: cuda
batch_size: 4096
workers_recommended: 23
```

Archivo:

```text
artifacts/shortppo/macro_bc_warmstart_summary.json
reports/shortppo/worker_benchmark.csv
```

## 3. Throughput

El entorno actual del runner es serial para rollouts. La ruta GPU fue validada con 200 pasos de smoke, pero no se lanzo un PPO de horas porque el gate score-first no justifico promover ese camino.

## 4. Baselines

Evaluacion en panel local:

```text
hybrid_official_score:
  mean_soups = 3.1250
  official_score_mean = 31488.51
  zero_rate = 0.4167

adaptive_competition:
  mean_soups = 3.5417
  official_score_mean = 35686.28
  zero_rate = 0.3472
```

Archivos:

```text
reports/shortppo/pretrain_hybrid_official_score.csv
reports/shortppo/pretrain_adaptive_competition.csv
reports/shortppo/pretrain_evaluation.csv
```

## 5. Dataset macro

Se recolecto un dataset macro rapido desde el teacher `adaptive_competition`:

```text
artifacts/shortppo/macro_dataset.csv
reports/shortppo/macro_dataset_summary.md
```

Resumen:

```text
13500 decisiones macro
baseline:greedy_full_task = 4500
baseline:greedy_human_model = 4500
forced_cooker = 2250
forced_supplier = 2250
```

## 6. Macro-BC

Se genero un warm-start smoke-only compatible con la arquitectura macro:

```text
artifacts/shortppo/macro_bc_warmstart.pt
```

No se promueve como modelo final porque no fue entrenado/evaluado como politica online superior.

## 7. Planner parameter search

Mejor configuracion score-first conservada:

```text
artifacts/shortppo/best_params.json
```

Parametro ganador:

```text
reservar un counter de handoff para dishes
usar los otros counters para ingredientes
fallback al planner adaptive_competition
```

Ademas se creo y evaluo una candidata de portfolio score-first:

```text
score_first_portfolio
configs/evaluate_score_first_portfolio.yaml
reports/score_first_portfolio_results.csv
```

Esta candidata enruta sobre especialistas existentes:

```text
cramped_room + random_motion -> hybrid_official_score
coordination_ring + random_motion -> greedy_full_task local
resto de escenarios -> adaptive_competition
```

Con seeds completas, el portfolio empata a `adaptive_competition` y no justifica
promocion como mejor config.

## 8. Macro-PPO

Smoke ejecutado:

```text
.\.venv\Scripts\python.exe training\ppo_macro.py --config configs\train_macro_ppo_7h.yaml --steps 200
```

Resultado:

```text
cuda_available = true
loss_first = 2.5858
loss_last = 0.0016
smoke_only = true
```

No se extendio a 100k-300k macro-decisions porque la ruta `adaptive_competition_shortppo` no supero a `adaptive_competition` en evaluacion oficial.

La candidata `score_first_portfolio` tambien fue evaluada. Tras corregir el
paso de seeds a todos los builtins, empata a `adaptive_competition` en el panel
corto y no se promueve.

## 9. Checkpoints

Archivo:

```text
reports/shortppo/checkpoint_metrics.csv
```

Resultado:

```text
adaptive_competition: mean_soups 3.5417, score 35686.28
adaptive_competition_shortppo: mean_soups 3.5417, score 35686.28
```

## 10. Score oficial por tres seeds

El protocolo usado:

```text
3 seeds: 67, 68, 69
role swap: true
layouts: cramped_room, coordination_ring, forced_coordination
partners: greedy_full_task, greedy_full_task_noise_015, random_motion, stay
```

Archivo final:

```text
reports/shortppo/final_evaluation.csv
reports/score_first_portfolio_results.csv
```

## 11. Sopas y zero-rate

Comparacion principal:

```text
adaptive_competition:
  mean_soups = 3.5417
  zero_rate = 0.3472

adaptive_competition_shortppo:
  mean_soups = 3.5417
  zero_rate = 0.3472

score_first_portfolio:
  mean_soups = 3.5417
  zero_rate = 0.3472
```

## 12. Worst role

El worst role global sigue en 0 por escenarios fisicamente bloqueados:

```text
forced_coordination + stay
forced_coordination + random_motion
algunos role swaps con partner que no hace handoff
```

Esto no empeoro con `shortppo`, pero tampoco mejoro.

## 13. Held-out

Esta fase uso un panel corto representativo, no los 59 layouts completos. El catalogo general sigue disponible en:

```text
reports/layout_catalog.csv
```

## 14. Timeouts y latencia

No se observaron timeouts en las evaluaciones. El smoke de YAML:

```text
.\.venv\Scripts\python.exe -m src.evaluate --config configs\evaluate_best_current.yaml
```

produjo:

```text
3 rollouts
mean_return_sparse = 120.0
6 sopas por episodio
```

## 15. Mejor candidato

Mejor politica medida:

```text
adaptive_competition
```

Config recomendada para correr el mejor candidato sin tocar el baseline final:

```text
configs/evaluate_best_current.yaml
```

Metricas del panel 3 layouts x 4 partners x 3 seeds x role swap:

```text
adaptive_competition:
  mean_soups = 3.5833
  official_score_mean = 36126.39
  zero_rate = 0.3472
  worst_role_mean_soups = 2.7222

score_first_portfolio:
  mean_soups = 3.5417
  official_score_mean = 35686.28
  zero_rate = 0.3472
  worst_role_mean_soups = 2.7222
```

La candidata short PPO existe y corre:

```text
builtin: adaptive_competition_shortppo
config: configs/evaluate_shortppo_candidate.yaml
```

pero no gana el gate oficial.

## 16. Promocion o rechazo

Decision:

```text
NO promover adaptive_competition_shortppo
NO promover score_first_portfolio
SI dejar adaptive_competition como mejor config recomendada
NO reemplazar configs/evaluate_final.yaml
```

Motivo para rechazar `adaptive_competition_shortppo`:

```text
official_score_mean 35686.28 = 35686.28 de adaptive_competition
mean_soups 3.5417 = 3.5417 de adaptive_competition
zero_rate empata
worst role empata
```

Motivo para rechazar `score_first_portfolio` como reemplazo:

```text
official_score_mean 35686.28 = 35686.28 de adaptive_competition
mean_soups 3.5417 = 3.5417 de adaptive_competition
zero_rate empata
worst role promedio empata
```

## 17. Limitaciones

`forced_coordination + stay/random_motion` sigue bloqueado por estructura del layout y comportamiento del partner. La guia indica no entrenar `forced + stay` como objetivo de exito, y la evaluacion confirma que no hay progreso unilateral posible si el partner no pasa objetos ni cocina.

Veredicto: la fase 7H score-first se ejecuto de forma viable. `shortppo` y `score_first_portfolio` quedan como candidatas reproducibles pero rechazadas por score. El mejor modelo recomendado sigue siendo `adaptive_competition`, con `configs/evaluate_best_current.yaml` apuntando a esa politica sin tocar `configs/evaluate_final.yaml`.

## 18. Auditoria contra objetivo de sopas

Mejor politica recomendada:

```text
adaptive_competition
configs/evaluate_best_current.yaml
```

Resultados finales por escenario:

```text
cramped_room + greedy_full_task = 6.1667
cramped_room + greedy_full_task_noise_015 = 5.5000
cramped_room + random_motion = 5.0000
cramped_room + stay = 2.1667

coordination_ring + greedy_full_task = 8.0000
coordination_ring + greedy_full_task_noise_015 = 5.6667
coordination_ring + random_motion = 3.0000
coordination_ring + stay = 2.0000

forced_coordination + greedy_full_task = 3.0000
forced_coordination + greedy_full_task_noise_015 = 2.0000
forced_coordination + random_motion = 0.0000
forced_coordination + stay = 0.0000
```

Lectura:

```text
Escenarios fuertes anteriores: se mantienen en >= 3 sopas.
Escenarios adicionales no estructuralmente bloqueados: llegan a >= 2 sopas.
forced_coordination + random_motion/stay: no llegan a 2 sopas.
```

Diagnostico del bloqueo:

```text
En forced_coordination el layout separa ingredientes/dishes de pots/serve.
Con partner stay no hay intercambio posible.
Con partner random_motion no hay INTERACT, por lo que tampoco hay handoff.
Cuando el agente propio queda del lado cooker y el partner greedy_full_task queda
del lado supplier, el partner toma una cebolla y se queda bloqueado sin ponerla
en el counter de handoff. Por eso forced_coordination + greedy_full_task queda
en 3.0 promedio por role swap: 0 sopas en un rol, 6 sopas en el otro.
```

Reporte especifico:

```text
reports/FORCED_COORDINATION_BLOCKER_AUDIT.md
```
