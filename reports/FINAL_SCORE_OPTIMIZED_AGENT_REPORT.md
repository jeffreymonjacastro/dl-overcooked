# Final Score-Optimized Agent Report

Fecha: 2026-07-11

## 1. Objetivo oficial

El objetivo de esta fase fue maximizar sopas promedio. La meta operativa era
alcanzar `3` sopas promedio si era posible, o al menos superar `1` sopa promedio
de forma verificable.

Resultado: la nueva politica `hybrid_official_score` alcanza:

```text
3.2250 sopas promedio en matriz 5-seed con role swap
3.1944 sopas promedio en simulacion 3-seed sin role swap
```

Por tanto, la fase alcanza el objetivo de `>= 3` sopas promedio.

## 2. Score exacto y protocolo de evaluacion

Se implemento una funcion unica en:

```text
scripts/score_official.py
```

Formula:

```text
si soups == 0:
    score = 0
si soups > 0:
    penalty = min(100 * timeouts, 5000)
    score = 10000 * soups
          + 10 * (horizon - last_soup_timestep)
          + (horizon - first_soup_timestep)
          - penalty
```

El entorno local no registra un evento explicito `soup_delivered` en `steps.csv`.
Se usa el proxy validado del runner: recompensa sparse positiva. En Overcooked-AI
cada sopa entregada da `20` reward, asi que `reward / 20` cuenta sopas.

Pruebas:

```text
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
18 tests OK, 1 skipped por checkpoint .pt no presente en snapshot limpio
```

## 3. Baselines A1 y A2.2

Promedio sobre 12 celdas:

| Modelo | Mean soups | Official score mean | Zero-rate |
| --- | ---: | ---: | ---: |
| A1 baseline | 1.1500 | 11760.88 | 0.6917 |
| A2.2 router | 1.2500 | 12883.20 | 0.6250 |
| GreedyHumanModel solo | 2.7083 | 27355.23 | 0.4583 |
| Hybrid official, 5 seeds + role swap | 3.2250 | 32489.01 | 0.4167 |
| Hybrid official, 3 seeds no swap | 3.1944 | 32181.14 | 0.4167 |

## 4. Resolubilidad por layout

Archivo:

```text
configs/layout_capabilities.yaml
reports/layout_solvability.csv
```

Resumen:

| Layout | Clasificacion | Observacion |
| --- | --- | --- |
| cramped_room | SOLVABLE_SOLO | El hibrido logra 4-6 sopas en escenarios no-forced. |
| coordination_ring | SOLVABLE_WITH_HELP | Muy fuerte con greedy y ruido; limitado en stay segun rol. |
| forced_coordination | FORCED_COOPERATION | Sigue sin resolverse con planners probados. |

## 5. Especialistas de escenarios conocidos

Se evaluaron:

```text
A1 GRU
A2.2 router
greedy_full_task local
greedy_human_model oficial
hybrid_official_score
```

El especialista final usa hard routing:

```text
cramped_room + random_motion -> greedy_full_task local
cramped_room + otros partners -> greedy_human_model
coordination_ring + greedy_full_task_noise_015 -> greedy_human_model
coordination_ring + otros partners -> greedy_full_task local
forced_coordination -> greedy_full_task local, unresolved
```

## 6. Especialista SOLO

El mejor SOLO rapido fue el planner local `greedy_full_task`.

Casos fuertes:

```text
coordination_ring + random_motion: 4.0 sopas promedio en hibrido role-swap
cramped_room + random_motion: 5.0 sopas promedio
```

## 7. Especialista COOPERATIVE-GREEDY

El mejor especialista cooperative fue `greedy_human_model`.

Casos fuertes:

```text
cramped_room + greedy_full_task: 6.4 sopas promedio
cramped_room + greedy_full_task_noise_015: 5.4 sopas promedio
coordination_ring + greedy_full_task_noise_015: 5.7 sopas promedio
```

## 8. Especialista FORCED

`forced_coordination` sigue en cero con los planners evaluados:

```text
forced_coordination + stay: 0.0
forced_coordination + random_motion: 0.0
forced_coordination + greedy_full_task: 0.0
forced_coordination + greedy_full_task_noise_015: 0.0
```

Siguiente mejora real: especialista forced con handoffs/skills o PPO warm-start.
No conviene gastar PPO desde cero.

## 9. Generalista OOD

No se entreno generalista OOD nuevo en esta fase. La mejora principal vino de
router score-first con planners existentes. Para layouts nuevos, el fallback
actual es `greedy_full_task`.

## 10. Router y recovery

Implementacion:

```text
policies/basic_policies.py::HybridOfficialScorePolicy
src/policy_loader.py registra builtin hybrid_official_score
configs/evaluate_final.yaml usa hybrid_official_score
```

El router es hard routing por layout y partner. No promedia logits.

## 11. Resultados por tres seeds

Archivo detallado:

```text
reports/phase_hybrid_official_3seed_attempts.csv
reports/phase_hybrid_official_3seed_scenarios.csv
```

Ejemplos exactos:

```text
coordination_ring + greedy_full_task:
scores = 80258, 80258, 80258
scenario_score = 80258.0
mean_soups = 8.0
```

```text
cramped_room + greedy_full_task:
scores = 60476, 60478, 60420
scenario_score = 60458.0
mean_soups = 6.0
```

```text
forced_coordination + greedy_full_task:
scores = 0, 0, 0
scenario_score = 0
mean_soups = 0.0
```

## 12. Role swap

La evaluacion robusta con role swap uso:

```text
3 layouts x 4 partners x 5 seeds x 2 roles = 120 rollouts
```

Resultado:

```text
mean_soups = 3.225
official_score_mean = 32489.01
zero_rate = 0.4167
```

## 13. Timeouts y latencia

No se observaron timeouts en las corridas. El agente final usa planners ligeros
del entorno y evita carga de datasets o entrenamiento dentro de `act()`.

## 14. Comparacion por score oficial

La mejora principal contra A2.2:

```text
mean_soups: 1.25 -> 3.225
official_score_mean: 12883.20 -> 32489.01
zero_rate: 0.6250 -> 0.4167
```

## 15. Modelo final y artefactos

Politica final:

```text
builtin: hybrid_official_score
config: configs/evaluate_final.yaml
```

Baselines preservados:

```text
artifacts/final/a1_policy.npz
artifacts/final/a2_policy.npz
artifacts/final/final_policy_config.json
legacy/try_A_end/
```

## 16. Riesgos

`forced_coordination` sigue siendo el bloqueo principal. Si el score oficial
incluye mucho forced, la siguiente fase debe centrarse en handoffs, no en mas
BC global.

## 17. Veredicto competitivo

La fase fue exitosa: el score por sopas promedio supera `3`. El cambio ganador
no fue mas entrenamiento de la GRU, sino seleccion score-first de especialistas
ya fuertes. La siguiente mejora debe atacar `forced_coordination` con teacher
dirigido y PPO warm-start.
