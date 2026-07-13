# Guía de pasada final — Mejora incremental, score-first y sin regresiones

## 0. Instrucción principal para Codex

Trabaja únicamente dentro de:

```text
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
```

Esta fase se ejecuta **solo en local**.

Lee primero:

```text
COMPETITION_ADAPTATION_CHANGES_REPORT.md
COMPETITION_SCENARIOS_SUCCESS_REPORT.md
LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO.md
PROJECT_GENERAL_STATUS_REPORT.md
CURRENT_STATUS_OPTION_A.md
reports/REVEALED_SCENARIOS_EVAL_REPORT.md
```

La política base que no debe romperse es:

```text
score_first_portfolio
```

Config recomendada actual:

```text
configs/evaluate_best_current.yaml
```

Regla absoluta:

```text
No reemplazar una ruta que ya funciona por una política generalista nueva.
Toda mejora debe entrar como una ruta adicional, especialista o fallback.
```

---

# 1. Objetivo de esta pasada final

No se busca rehacer el agente completo.

Se busca:

```text
1. conservar el rendimiento de los tres escenarios ya aprobados;
2. mejorar layouts que hoy tienen 0–1 sopa;
3. aumentar el score oficial promedio;
4. reducir zero-rate y dependencia del rol;
5. agregar generalización por familias de problema;
6. promover solo mejoras demostradas.
```

Resultados protegidos:

```text
Escenario 1 — asymmetric_advantages:
  5.9333 sopas
  59754.27 score
  10/10 grupos aprobados

Escenario 2 — coordination_ring + sticky:
  4.9000 sopas
  49736.90 score
  10/10 grupos aprobados

Escenario 3 — counter_circuit + sticky + random:
  5.3667 sopas
  54149.23 score
  10/10 grupos aprobados
```

Estos resultados son gates de regresión.

---

# 2. Métrica obligatoria de selección

Toda comparación debe usar el score oficial:

```text
Score =
10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- min(100 * timeouts, 5000)
```

Si no hay sopa:

```text
Score = 0
```

Para cada escenario:

```text
scenario_score =
mean(score_seed_1, score_seed_2, score_seed_3)
```

Orden de prioridad:

```text
1. mayor score oficial por grupos de 3 seeds;
2. mayor número de sopas;
3. menor zero-rate;
4. mayor percentil 10;
5. mejor peor rol;
6. earlier last soup;
7. earlier first soup;
8. cero timeouts.
```

No seleccionar por:

```text
validation loss
training reward
accuracy
último checkpoint
promedio global sin desglose
```

---

# 3. Diagnóstico actual

El portfolio actual ya resuelve bien:

```text
asymmetric_advantages
coordination_ring
counter_circuit
cramped_room
five_by_five
scenario2
scenario3
simple_o
centre_pots
```

Los layouts bajos se agrupan en cinco familias.

## 3.1 Handoff o cooperación estricta

Ejemplos:

```text
forced_coordination
forced_coordination_tomato
soup_coordination
pipeline
schelling
small_corridor
```

Problema probable:

```text
handoff no generalizado;
counter equivocado;
plato no transferido;
dependencia fuerte del rol;
espera o bloqueo mutuo.
```

## 3.2 Receta tomate o multi-receta

Ejemplos:

```text
simple_tomato
cramped_room_tomato
cramped_room_o_3orders
bonus_order_test
forced_coordination_tomato
```

Problema probable:

```text
rutas onion-only;
orden real ignorado;
ingrediente equivocado en olla;
recipe-aware aplicado solo a counter_circuit.
```

## 3.3 Bottlenecks, corredores y bloqueo

Ejemplos:

```text
bottleneck
cramped_corridor
small_corridor
m_shaped_s
scenario1_s
scenario4
```

Problema probable:

```text
ambos agentes persiguen el mismo objetivo;
no existe yield;
planner no replantea;
un agente ocupa el único paso.
```

## 3.4 Layouts custom o geometría no cubierta

Ejemplos:

```text
chavez_room
jamcy_room
m_room
diagonal_run
large_room
```

Problema probable:

```text
el router no reconoce la familia;
targets incorrectos;
distancias o estaciones no resueltas;
fallback demasiado específico.
```

## 3.5 Planner patológico o layout inválido

Ejemplos:

```text
corridor
you_shall_not_pass
multiplayer_schelling
maze_kitchen
tutorial_1
cramped_room_single
```

Problema:

```text
timeout;
matriz de planificación gigante;
MemoryError;
configuración incompatible;
suposiciones de dos agentes.
```

Estos casos deben aislarse. No deben bloquear el resto de la mejora.

---

# 4. Estrategia final: portfolio jerárquico seguro

La política nueva debe extender el portfolio actual:

```text
ScoreFirstPortfolioV2
```

Orden de selección:

```text
1. rutas exactas protegidas de escenarios revelados;
2. rutas exactas ya demostradas en layouts fuertes;
3. especialista por familia topológica/receta;
4. adaptive_competition;
5. fallback seguro.
```

Nunca invertir este orden.

Pseudológica:

```python
if is_revealed_scenario_route(layout, partner, noise):
    return protected_policy

if has_validated_layout_route(layout, partner):
    return validated_layout_policy

family = classify_layout_and_recipe(state, mdp)

if family == "FORCED_HANDOFF":
    return handoff_family_policy

if family == "TOMATO_OR_MIXED":
    return recipe_aware_family_policy

if family == "BOTTLENECK":
    return bottleneck_recovery_policy

if family == "SOLO_CAPABLE" and partner_is_unhelpful:
    return solo_topology_policy

return adaptive_competition
```

---

# 5. Fase 1 — Congelar baseline y crear regresión automática

Crear:

```text
configs/evaluate_score_first_portfolio_v2.yaml
scripts/run_final_regression_suite.py
reports/final_pass/
```

El script debe ejecutar primero:

```text
python scripts/evaluate_revealed_scenarios.py \
  --policy score_first_portfolio \
  --seeds 67-96 \
  --group-size 3
```

Guardar resultados base en:

```text
reports/final_pass/protected_baseline.csv
reports/final_pass/protected_baseline.md
```

Gate obligatorio:

```text
Escenario 1: 10/10 grupos aprobados
Escenario 2: 10/10 grupos aprobados
Escenario 3: 10/10 grupos aprobados
```

Además:

```text
no perder más de 0.10 sopas promedio por escenario;
no perder score oficial promedio;
no crear timeouts.
```

Si una mejora falla este gate, se rechaza o se limita a una ruta más específica.

---

# 6. Fase 2 — Triage de layouts bajos

Crear:

```text
scripts/classify_low_layouts.py
reports/final_pass/low_layout_triage.csv
reports/final_pass/low_layout_triage.md
```

Clasificar cada layout bajo como:

```text
A. competition_like_and_solvable
B. slow_but_working
C. structurally_forced
D. planner_pathological
E. invalid_or_test_only
```

Prioridad:

```text
1. A
2. B
3. C
4. D
5. E
```

No gastar tiempo principal en layouts de prueba inválidos o imposibles.

Panel prioritario sugerido:

```text
simple_tomato
cramped_room_tomato
forced_coordination
forced_coordination_tomato
soup_coordination
bottleneck
cramped_corridor
small_corridor
large_room
centre_objects
scenario2_s
unident
```

---

# 7. Fase 3 — Recipe-aware general sin tocar rutas onion exitosas

Crear:

```text
policies/recipe_aware_family_policy.py
tests/test_recipe_aware_family_policy.py
```

Debe leer del MDP:

```text
receta activa;
ingredientes requeridos;
órdenes pendientes;
pot state;
ingrediente sostenido;
```

Reglas:

```text
no colocar ingrediente que no pertenece a la receta;
priorizar receta que puede completarse antes;
si pot tiene ingrediente parcial, completar esa misma receta;
recoger dish cuando una soup estará lista;
entregar orden válida antes de iniciar una nueva receta.
```

Integración:

```text
solo activar para layouts detectados como tomato/mixed/order-aware;
no reemplazar asymmetric_advantages, coordination_ring o rutas onion validadas.
```

Evaluar mínimo:

```text
simple_tomato
cramped_room_tomato
asymmetric_advantages_tomato
counter_circuit
counter_circuit_o_1order
forced_coordination_tomato
```

Gate:

```text
subir de 0 a >=1 sopa en al menos dos layouts actualmente fallidos;
no bajar counter_circuit;
no bajar asymmetric_advantages.
```

---

# 8. Fase 4 — HandoffFamilyPolicy con simetría de rol

Crear o mejorar:

```text
policies/handoff_family_policy.py
planning/handoff_protocol.py
tests/test_handoff_role_symmetry.py
```

Debe detectar por topología:

```text
componentes separadas;
recursos disponibles por lado;
pot/serving por lado;
counters accesibles desde ambas componentes;
número de counters de handoff.
```

Roles:

```text
SUPPLIER
COOKER
DISH_SUPPLIER
RECOVERY
```

Protocolo:

```text
reservar un counter para ingredientes;
reservar un counter para dish cuando sea posible;
no llenar todos los counters con onions;
retirar inmediatamente objetos recibidos;
pasar dish cuando pot esté próximo o ready;
recalcular el counter si el partner bloquea;
hacer probe de cooperación;
```

Si el partner no coopera:

```text
en layout solo-capable -> cambiar a SOLO;
en layout forced -> dejar recursos útiles y evitar loops;
no esperar indefinidamente.
```

Evaluar con ambos roles:

```text
forced_coordination
forced_coordination_tomato
soup_coordination
pipeline
small_corridor
```

Partners:

```text
greedy_full_task
greedy_full_task_noise_015
random_motion
```

Gate:

```text
forced + greedy:
  ambos roles > 0 sopas;
  promedio >= resultado actual;

forced + noisy:
  zero-rate menor;

no degradar forced role que ya funciona.
```

---

# 9. Fase 5 — BottleneckRecoveryPolicy

Crear:

```text
policies/bottleneck_recovery_policy.py
planning/deadlock_detector.py
tests/test_bottleneck_recovery.py
```

Detectar:

```text
posición repetida;
acciones alternantes;
sin progreso de receta;
ambos agentes intentando cruzar;
partner bloqueando tile crítico;
```

Acciones de recuperación:

```text
YIELD
BACK_OFF
WAIT_AT_SAFE_TILE
REPLAN_TO_ALTERNATE_STATION
TAKE_COMPLEMENTARY_TASK
```

Reglas:

```text
un agente cede según agent_index y costo de ruta;
no cambiar de objetivo cada timestep;
mantener hysteresis de 10–20 pasos;
cancelar macro si no progresa;
```

Evaluar:

```text
bottleneck
cramped_corridor
small_corridor
m_shaped_s
scenario1_s
```

Gate:

```text
romper al menos dos zero-rates;
no degradar coordination_ring;
no aumentar latencia de forma significativa.
```

---

# 10. Fase 6 — Planner acotado para mapas grandes

Crear:

```text
planning/bounded_motion_planner.py
tests/test_bounded_planner_limits.py
```

Requisitos:

```text
no construir matrices NxN gigantes;
búsqueda lazy;
BFS/A* por consulta;
cache limitado;
LRU eviction;
límite de nodos;
timeout interno corto;
fallback a movimiento local.
```

Ante fallo:

```text
devolver acción segura;
registrar planner_fallback;
no producir MemoryError;
no exceder el límite de acción.
```

Evaluar:

```text
corridor
you_shall_not_pass
multiplayer_schelling
large_room
```

Objetivo principal:

```text
eliminar timeout y MemoryError;
```

El rendimiento de sopas es secundario en esta fase.

---

# 11. Fase 7 — Búsqueda corta de parámetros antes de entrenar

Crear o usar:

```text
scripts/search_policy_parameters.py
```

Optimizar por familia:

```text
handoff wait timeout;
dish handoff timing;
counter reservation;
partner passive threshold;
yield duration;
macro cancellation threshold;
replanning interval;
```

Usar:

```text
random search o CEM;
20–50 configuraciones por familia;
2–3 seeds para búsqueda;
10 seeds para confirmación.
```

Objetivo:

```text
mejorar score sin entrenamiento neuronal largo.
```

Conservar la mejor configuración por familia.

---

# 12. Fase 8 — Entrenamiento opcional y dirigido

PPO no es obligatorio.

Solo usar Macro-PPO si:

```text
las reglas ya consiguen sopas;
existen macro-demostraciones exitosas;
la familia sigue bajo el objetivo;
```

No entrenar una policy global nueva.

Entrenar un especialista por familia:

```text
macro_ppo_handoff
macro_ppo_bottleneck
macro_ppo_recipe
```

Warm-start:

```text
planner/reglas -> macro dataset -> Macro-BC -> short Macro-PPO
```

Presupuesto:

```text
50k–150k macro-decisions por especialista;
early stopping;
máximo 2–3 horas total de PPO.
```

Selección:

```text
score oficial held-out de la familia;
soups;
zero-rate;
worst role.
```

Si PPO no supera las reglas, descartar el checkpoint.

---

# 13. Fase 9 — Evaluación score-first por portfolio

Crear:

```text
scripts/evaluate_final_portfolio_v2.py
reports/final_pass/portfolio_v2_results.csv
reports/final_pass/portfolio_v2_summary.md
```

Comparar:

```text
score_first_portfolio
adaptive_competition
score_first_portfolio_v2
cada especialista individual
```

Evaluación A — protegida:

```text
escenarios 1–3 exactos;
seeds 67..96;
grupos de 3;
ruido exacto;
```

Evaluación B — layouts bajos:

```text
10 seeds;
role swap;
partner greedy;
partner noisy;
partner random cuando corresponda;
```

Evaluación C — held-out:

```text
layouts de cada familia no usados para ajustar parámetros;
3 seeds;
role swap;
```

Calcular:

```text
attempt_score;
scenario_score;
mean soups;
zero-rate;
percentil 10;
worst role;
first soup;
last soup;
timeouts;
latency.
```

---

# 14. Promoción por ruta, no promoción global

La versión final puede usar una tabla de rutas validadas:

```text
validated_routes.yaml
```

Cada entrada:

```yaml
layout_or_family:
  partner_condition:
  policy:
  score_mean:
  soups_mean:
  zero_rate:
  worst_role:
  validation_date:
```

Regla:

```text
si la ruta nueva supera baseline en esa celda:
    activar ruta nueva
si no:
    mantener ruta anterior
```

No promover una política completa por mejorar solo el promedio global.

---

# 15. Gates finales

## 15.1 Escenarios revelados

Obligatorio:

```text
Escenario 1: 10/10 grupos aprobados
Escenario 2: 10/10 grupos aprobados
Escenario 3: 10/10 grupos aprobados
```

Y:

```text
sin caída material de soups;
sin caída de score;
cero timeouts.
```

## 15.2 Layouts bajos

Objetivo:

```text
convertir al menos 5 layouts de 0 sopa a >=1 sopa;
subir al menos 3 layouts a >=2 sopas;
reducir zero-rate global del sweep;
```

## 15.3 Generalización

Obligatorio:

```text
mejora en layouts held-out de la misma familia;
no depender de nombres o coordenadas fijas salvo rutas reveladas;
```

## 15.4 Rendimiento técnico

```text
p95 de decisión <100 ms;
sin MemoryError;
sin timeout de planner;
```

---

# 16. Orden exacto de trabajo

```text
1. Congelar score_first_portfolio.
2. Crear suite de regresión de escenarios 1–3.
3. Clasificar layouts bajos por familia.
4. Implementar RecipeAwareFamilyPolicy.
5. Implementar HandoffFamilyPolicy simétrica.
6. Implementar BottleneckRecoveryPolicy.
7. Implementar planner acotado.
8. Buscar parámetros por familia.
9. Evaluar rutas nuevas.
10. Entrenar Macro-PPO solo donde reglas + BC todavía queden cortas.
11. Construir validated_routes.yaml.
12. Ejecutar regresión final completa.
13. Promover únicamente rutas ganadoras.
```

---

# 17. Archivos finales requeridos

```text
policies/score_first_portfolio_v2.py
policies/recipe_aware_family_policy.py
policies/handoff_family_policy.py
policies/bottleneck_recovery_policy.py
planning/handoff_protocol.py
planning/deadlock_detector.py
planning/bounded_motion_planner.py
configs/validated_routes.yaml
configs/evaluate_score_first_portfolio_v2.yaml
scripts/run_final_regression_suite.py
scripts/classify_low_layouts.py
scripts/evaluate_final_portfolio_v2.py
reports/final_pass/
reports/FINAL_INCREMENTAL_IMPROVEMENT_REPORT.md
```

---

# 18. Reporte final obligatorio

Crear:

```text
reports/FINAL_INCREMENTAL_IMPROVEMENT_REPORT.md
```

Debe incluir:

```markdown
# Final Incremental Improvement Report

## 1. Baseline protegido
## 2. Score oficial y protocolo
## 3. Triage de layouts bajos
## 4. Recipe-aware family
## 5. Handoff family
## 6. Bottleneck recovery
## 7. Planner acotado
## 8. Búsqueda de parámetros
## 9. Entrenamiento opcional
## 10. Comparación por layout/familia
## 11. Escenarios revelados
## 12. Held-out
## 13. Worst role
## 14. Timeouts y latencia
## 15. Rutas promovidas
## 16. Rutas rechazadas
## 17. Config final
## 18. Veredicto competitivo
```

Actualizar también:

```text
CURRENT_STATUS_OPTION_A.md
```

---

# 19. Prompt corto para Codex

```text
Trabaja únicamente dentro de:
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked

Sigue:
GUIA_PASADA_FINAL_MEJORA_INCREMENTAL_SCORE_FIRST.md

No reemplaces score_first_portfolio con una política generalista.

Primero congela y protege los tres escenarios revelados mediante una suite de
regresión. Después clasifica los layouts bajos por familia y agrega especialistas
incrementales para receta tomate/multi-receta, handoff/coordinación, bottlenecks
y planners grandes.

Todas las decisiones de promoción deben usar la fórmula oficial:

10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- min(100 * timeouts, 5000)

Evalúa por grupos de tres seeds, role swap, score, soups, zero-rate, percentil
10, worst role, first/last soup, timeouts y latencia.

Usa búsqueda de parámetros antes de PPO. Si una familia todavía queda baja,
entrena solo un Macro-PPO especialista con warm-start y 50k–150k macro-decisions.

Promueve rutas individualmente mediante validated_routes.yaml. Si una ruta nueva
no supera al baseline en su layout/familia, conserva la ruta anterior.

Al finalizar crea:
reports/FINAL_INCREMENTAL_IMPROVEMENT_REPORT.md

y actualiza:
CURRENT_STATUS_OPTION_A.md
```
