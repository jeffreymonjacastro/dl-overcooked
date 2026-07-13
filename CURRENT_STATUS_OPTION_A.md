# Current Status Option A

Fecha: 2026-07-11

Estado actual: fase hibrida score-first completada.

## Resultado

El agente activo es:

```text
hybrid_official_score
```

Config principal:

```text
configs/evaluate_final.yaml
```

Metricas clave:

```text
Matriz 5 seeds + role swap: 3.2250 sopas promedio
Simulacion 3 seeds no swap: 3.1944 sopas promedio
A2.2 baseline: 1.2500 sopas promedio
```

## Archivos importantes

```text
reports/FINAL_SCORE_OPTIMIZED_AGENT_REPORT.md
reports/phase_hybrid_official_matrix.csv
reports/phase_hybrid_official_3seed_no_swap.csv
reports/phase_hybrid_official_3seed_attempts.csv
reports/phase_hybrid_official_3seed_scenarios.csv
scripts/score_official.py
configs/layout_capabilities.yaml
```

## Riesgo principal

`forced_coordination` sigue sin resolverse. La siguiente fase debe enfocarse en
handoffs/skills y PPO warm-start especifico para forced.

---

## Actualizacion 2026-07-12: fase generalizacion macro

Se implemento la candidata:

```text
adaptive_competition
```

Archivos principales:

```text
planning/layout_graph.py
planning/macro_actions.py
planning/macro_executor.py
planning/task_state.py
models/partner_tracker.py
models/macro_actor_critic.py
policies/adaptive_competition_policy.py
scripts/catalog_layouts.py
scripts/evaluate_competition_protocol.py
training/ppo_macro.py
```

Resultados con protocolo 3 seeds + role swap tras reservar un handoff para platos:

```text
hybrid_official_score mean_cell_soups = 3.1944
adaptive_competition mean_cell_soups = 3.5833
hybrid_official_score mean_zero_rate = 0.4167
adaptive_competition mean_zero_rate = 0.3472
forced_coordination + greedy_full_task = 3.0 sopas promedio
forced_coordination + greedy_full_task_noise_015 = 2.0 sopas promedio
```

No se reemplazo `configs/evaluate_final.yaml` porque `adaptive_competition` aun no
cumple los gates de peor rol ni puede resolver forced contra partners sin
interacciones utiles (`random_motion`/`stay`).

Reporte:

```text
reports/FINAL_COMPETITION_GENERALIZATION_REPORT.md
```

---

## Actualizacion 2026-07-12: fase short PPO score-first

Se siguio `GUIA_ENTRENAMIENTO_VIABLE_7H_SCORE_FIRST.md` sin reemplazar
`configs/evaluate_final.yaml`.

Nueva candidata creada:

```text
adaptive_competition_shortppo
configs/evaluate_shortppo_candidate.yaml
artifacts/shortppo/
reports/shortppo/
```

Resultado score-first:

```text
adaptive_competition mean_soups = 3.5417
adaptive_competition official_score_mean = 35686.28

adaptive_competition_shortppo mean_soups = 3.5417
adaptive_competition_shortppo official_score_mean = 35686.28
```

Decision:

```text
No promover shortppo porque empata pero no mejora.
Mejor politica actual: adaptive_competition.
Config recomendada: configs/evaluate_best_current.yaml
```

Reporte:

```text
reports/FINAL_SHORT_TRAINING_REPORT.md
```

---

## Actualizacion 2026-07-12: portfolio score-first y seeds completas

Se corrigio `scripts/evaluate_competition_protocol.py` para pasar contexto
`layout_name` y `partner_name` a `score_first_portfolio`.

Candidata evaluada:

```text
score_first_portfolio
configs/evaluate_score_first_portfolio.yaml
reports/score_first_portfolio_results.csv
```

Resultado contra `adaptive_competition` en el mismo panel 3 layouts x 4 partners
x 3 seeds x role swap:

```text
adaptive_competition:
  mean_soups = 3.5417
  official_score_mean = 35686.28
  zero_rate = 0.3472
  worst_role_mean_soups = 2.7222

score_first_portfolio:
  mean_soups = 3.5417
  official_score_mean = 35686.28
  zero_rate = 0.3472
  worst_role_mean_soups = 2.7222
```

Decision:

```text
Usar configs/evaluate_best_current.yaml para probar adaptive_competition.
No promover score_first_portfolio porque empata pero no mejora.
No reemplazar configs/evaluate_final.yaml.
forced_coordination + stay/random_motion sigue sin solucion unilateral.
```

---

## Actualizacion 2026-07-12: escenarios revelados

Se evaluaron los escenarios revelados sin entrenamiento:

```text
policy = score_first_portfolio
seeds = 67..96
group_size = 3
role_swap = false
sticky_action_prob = 0.10
random_action_prob = 0.15 cuando aplica
```

Se agrego soporte de evaluacion para:

```text
sticky_action_prob
recipe_aware_greedy
```

`score_first_portfolio` ahora usa:

```text
asymmetric_advantages + greedy_full_task -> GreedyHumanModel
counter_circuit + greedy_full_task -> recipe_aware_greedy
otros casos -> rutas previas
```

Resultado revelado:

```text
Escenario 1 asymmetric_advantages + greedy_full_task:
  mean_soups = 5.9333
  official_score_mean = 59754.27
  groups_passed = 10/10

Escenario 2 coordination_ring + greedy_full_task sticky:
  mean_soups = 4.9000
  official_score_mean = 49736.90
  groups_passed = 10/10

Escenario 3 counter_circuit + greedy_full_task sticky+random:
  mean_soups = 5.3667
  official_score_mean = 54149.23
  groups_passed = 10/10
```

Decision:

```text
Mejor config actual para escenarios revelados: configs/evaluate_best_current.yaml
Agente recomendado: score_first_portfolio
Reporte: reports/REVEALED_SCENARIOS_EVAL_REPORT.md
```
