# Final Incremental Improvement Report

Fecha: 2026-07-13

## 1. Baseline protegido

Baseline congelado:

```text
score_first_portfolio
```

Candidato incremental:

```text
score_first_portfolio_v2
```

La V2 no reemplaza globalmente al baseline. Solo agrega rutas especialistas
medidas para layouts donde el baseline estaba bajo.

## 2. Score oficial y protocolo

La seleccion se hizo con score oficial:

```text
10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- min(100 * timeouts, 5000)
```

Para rutas nuevas se uso:

```text
partner = greedy_full_task
seeds = 67..76
horizon = 250
role_swap = false
```

## 3. Triage de layouts bajos

Se genero:

```text
reports/final_pass/low_layout_triage.csv
reports/final_pass/low_layout_triage.md
```

El triage separa layouts por familias:

```text
recipe/order-aware
handoff estructural
bottleneck/deadlock
custom geometry
planner pathological / invalid
```

## 4. Recipe-aware family

No se promovio una ruta recipe-aware global. Las pruebas mostraron que algunos
layouts tomate siguen fallando cuando el partner `greedy_full_task` contamina la
olla con onion. Cambiar globalmente a recipe-aware podria danar rutas que ya
funcionan.

Decision:

```text
mantener recipe_aware_greedy protegido para counter_circuit
no promoverlo globalmente
```

## 5. Handoff family

No se promovio una nueva ruta de handoff en esta pasada. Los layouts como
`forced_coordination`, `forced_coordination_tomato`, `pipeline` y
`soup_coordination` siguen requiriendo una politica de handoff simetrica mas
profunda.

Decision:

```text
mantener adaptive_competition como fallback actual
no tocar la ruta protegida de escenarios revelados
```

## 6. Bottleneck recovery

No se implemento todavia una politica nueva de deadlock/yield. Los layouts
`bottleneck`, `cramped_corridor`, `small_corridor`, `m_shaped_s` y similares
siguen como candidatos para una futura `BottleneckRecoveryPolicy`.

## 7. Planner acotado

No se reemplazo el planner. Los casos `corridor`, `you_shall_not_pass` y
`multiplayer_schelling` deben tratarse como planner pathological por ahora:

```text
corridor: timeout
you_shall_not_pass: timeout
multiplayer_schelling: MemoryError por matriz enorme
```

## 8. Busqueda de parametros

No se hizo PPO ni busqueda larga. Se hizo una busqueda corta de especialistas
existentes por layout, priorizando score oficial y sopas.

Especialista promovido:

```text
GreedyHumanModel
```

Solo en rutas donde supero claramente al baseline.

## 9. Entrenamiento opcional

No se entreno en esta pasada. La mejora actual viene de routing score-first,
no de checkpoints nuevos.

## 10. Comparacion por layout/familia

Resultado de rutas nuevas:

| Layout | Baseline sopas | V2 sopas | Delta sopas | Baseline score | V2 score | Promovido |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bottleneck` | 0.0000 | 4.0000 | +4.0000 | 0.00 | 40347.00 | true |
| `centre_objects` | 1.0000 | 5.0000 | +4.0000 | 12343.00 | 50505.00 | true |
| `chavez_room` | 0.0000 | 8.0000 | +8.0000 | 0.00 | 80733.00 | true |
| `cramped_room_o_3orders` | 0.0000 | 5.0000 | +5.0000 | 0.00 | 50450.00 | true |
| `cramped_room_tomato` | 0.0000 | 8.0000 | +8.0000 | 0.00 | 80569.00 | true |
| `jamcy_room` | 0.0000 | 11.2000 | +11.2000 | 0.00 | 113088.00 | true |
| `large_room` | 0.0000 | 5.1000 | +5.1000 | 0.00 | 51423.60 | true |
| `mdp_test` | 0.0000 | 4.8000 | +4.8000 | 0.00 | 49888.80 | true |
| `m_room` | 0.0000 | 10.0000 | +10.0000 | 0.00 | 100205.60 | true |
| `scenario1_s` | 0.0000 | 4.0000 | +4.0000 | 0.00 | 40347.00 | true |
| `scenario2_s` | 1.0000 | 5.6000 | +4.6000 | 12332.00 | 56372.30 | true |
| `schelling` | 0.0000 | 7.7000 | +7.7000 | 0.00 | 77313.10 | true |
| `schelling_s` | 0.0000 | 9.1000 | +9.1000 | 0.00 | 91325.50 | true |
| `soup_coordination` | 0.0000 | 14.0000 | +14.0000 | 0.00 | 140336.00 | true |
| `unident` | 1.0000 | 5.9000 | +4.9000 | 12200.00 | 59281.70 | true |

Archivos:

```text
reports/final_pass/portfolio_v2_results.csv
reports/final_pass/portfolio_v2_summary.md
```

## 11. Escenarios revelados

La regresion protegida de V2 paso:

| Escenario | Layout | Sopas promedio | Score promedio | Grupos aprobados |
| ---: | --- | ---: | ---: | ---: |
| 1 | `asymmetric_advantages` | 5.9333 | 59754.27 | 10/10 |
| 2 | `coordination_ring` | 4.9000 | 49736.90 | 10/10 |
| 3 | `counter_circuit` | 5.3667 | 54149.23 | 10/10 |

Archivo:

```text
reports/final_pass/protected_score_first_portfolio_v2.md
```

## 12. Held-out

No se completo una evaluacion held-out amplia. Esta pasada promovio solo rutas
exactas medidas, no familias generales.

## 13. Worst role

No se ejecuto role swap para las rutas promovidas en esta pasada. Antes de una
promocion final de entrega general, debe agregarse role swap en el panel V2.

## 14. Timeouts y latencia

No hubo timeouts en:

```text
rutas promovidas V2
escenarios revelados V2
```

Los timeouts conocidos siguen aislados:

```text
corridor
you_shall_not_pass
```

## 15. Rutas promovidas

Se creo:

```text
configs/validated_routes.yaml
```

Rutas promovidas:

```text
bottleneck + greedy_full_task -> BottleneckKickstartPolicy
centre_objects + greedy_full_task -> GreedyHumanModel
chavez_room + greedy_full_task -> GreedyHumanModel
cramped_room_o_3orders + greedy_full_task -> GreedyHumanModel
cramped_room_tomato + greedy_full_task -> GreedyHumanModel
jamcy_room + greedy_full_task -> GreedyHumanModel
large_room + greedy_full_task -> GreedyHumanModel
mdp_test + greedy_full_task -> GreedyHumanModel
m_room + greedy_full_task -> GreedyHumanModel
scenario1_s + greedy_full_task -> BottleneckKickstartPolicy
scenario2_s + greedy_full_task -> GreedyHumanModel
schelling + greedy_full_task -> GreedyHumanModel
schelling_s + greedy_full_task -> GreedyHumanModel
soup_coordination + greedy_full_task -> RecipeAwareGreedyPolicy
unident + greedy_full_task -> GreedyHumanModel
```

## 16. Rutas rechazadas

No se promovieron:

```text
simple_tomato
bonus_order_test
forced_coordination_tomato
forced_coordination
cramped_corridor
long_cook_time
m_shaped_s
pipeline
scenario4
small_corridor
diagonal_run
```

Motivo principal:

```text
no superaron el baseline con especialistas existentes o requieren handoff/deadlock logic nueva
```

## 17. Config final

Config candidata:

```text
configs/evaluate_score_first_portfolio_v2.yaml
```

Politica candidata:

```text
score_first_portfolio_v2
```

El baseline anterior queda disponible:

```text
score_first_portfolio
configs/evaluate_best_current.yaml
```

## 18. Veredicto competitivo

La V2 es una mejora incremental segura para los layouts promovidos y mantiene
intactos los tres escenarios revelados. En el sweep completo paso de:

```text
baseline: 18/44 layouts evaluados >= 3 sopas
V2:       33/44 layouts evaluados >= 3 sopas
```

Tambien redujo los layouts evaluados con cero sopas:

```text
baseline: 23 layouts en 0 sopas
V2:       11 layouts en 0 sopas
```

Aun no completa el objetivo total de subir todos los layouts a 3 sopas; los
restantes requieren handoff, deadlock recovery, receta/partner control o planner
acotado.

Veredicto:

```text
promover V2 como candidato incremental
mantener score_first_portfolio como baseline comparativo
continuar siguiente fase en handoff/bottleneck/planner
```

## Verificacion ejecutada

```text
python -m py_compile policies/score_first_portfolio_v2.py src/policy_loader.py scripts/classify_low_layouts.py scripts/run_final_regression_suite.py scripts/evaluate_final_portfolio_v2.py scripts/evaluate_revealed_scenarios.py scripts/evaluate_competition_protocol.py
python scripts/classify_low_layouts.py
python scripts/evaluate_final_portfolio_v2.py --seeds 67-76
python scripts/run_final_regression_suite.py --policy score_first_portfolio_v2 --seeds 67-96 --group-size 3
python scripts/evaluate_layout_sweep.py --policy score_first_portfolio_v2 --seeds 67-76 --timeout-seconds 90
python scripts/classify_low_layouts.py --sweep reports/layout_sweep_score_first_portfolio_v2.csv --omitted reports/layout_sweep_score_first_portfolio_v2_omitted.csv --output reports/final_pass/low_layout_triage_v2.csv --report reports/final_pass/low_layout_triage_v2.md
python -m unittest discover -s tests -v
```
