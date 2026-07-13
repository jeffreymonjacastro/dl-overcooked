# Final Competition Generalization Report

Fecha: 2026-07-12

## 1. Baseline bloqueado

El baseline `hybrid_official_score` se conserva sin reemplazar en `configs/evaluate_final.yaml`.

Baseline verificado con el nuevo protocolo:

```text
mean_cell_soups = 3.1944
official_score_mean = 32191.60
mean_zero_rate = 0.4167
```

## 2. Catalogo y splits de layouts

Se creo `scripts/catalog_layouts.py` y se ejecuto:

```text
.\.venv\Scripts\python.exe scripts\catalog_layouts.py
```

Salida:

```text
reports/layout_catalog.csv
59 layouts catalogados
2 errores de carga: maze_kitchen por incompatibilidad recipe_values/<ingredient>_value
```

Familias detectadas:

```text
SOLO_CAPABLE: 5
SHARED_OPEN: 23
BOTTLENECK: 25
FORCED_HANDOFF: 1
ASYMMETRIC_RESOURCES: 1
UNKNOWN: 4
```

## 3. LayoutGraphAnalyzer

Implementado en `planning/layout_graph.py`.

Extrae:

```text
walkable tiles, counters, dispensers, pots, serving locations,
connected components, bottlenecks, handoff counters,
resource accessibility by component, topology family
```

No depende de `layout_name` para clasificar `FORCED_HANDOFF`.

## 4. Macroacciones y planner

Implementado:

```text
planning/macro_actions.py
planning/macro_executor.py
planning/task_state.py
```

La politica nueva usa targets de interaccion y BFS local, no aprende navegacion low-level con PPO.

## 5. PartnerTracker

Implementado en:

```text
models/partner_tracker.py
```

Primera version heuristica:

```text
UNKNOWN, PASSIVE, RANDOM, FILLING_POT, SEEKING_DISH, DELIVERING, HANDOFF
```

## 6. Router adaptativo

Nueva politica:

```text
policies/adaptive_competition_policy.py
builtin: adaptive_competition
```

Registro:

```text
src/policy_loader.py
```

Regla principal:

```text
FORCED_HANDOFF -> planner supplier/cooker por componente
solo-capable + partner pasivo/random -> SOLO pipeline
otros casos -> fallback al baseline hybrid_official_score
```

## 7. Busqueda de parametros

Se creo el harness:

```text
scripts/search_policy_parameters.py
```

Estado: preparado para escenarios revelados. Aun no hay CEM real porque el planner actual expone pocos knobs seguros.

## 8. Macro-BC

No se genero dataset real de macro-demostraciones en esta corrida.

Preparado para siguiente etapa:

```text
planning/macro_actions.py
training/state_buffer.py
```

## 9. Macro-PPO

Se implemento el modelo y entrypoint:

```text
models/macro_actor_critic.py
training/ppo_macro.py
training/vector_env.py
configs/train_macro_ppo.yaml
```

Smoke GPU ejecutado:

```text
.\.venv\Scripts\python.exe training\ppo_macro.py --steps 20
```

Resultado:

```text
device = cuda
cuda_name = NVIDIA GeForce RTX 4080
checkpoint = artifacts/macro_ppo/macro_actor_critic_smoke.pt
smoke_only = true
loss_first = 2.5309
loss_last = 1.9232
```

Este checkpoint no se debe usar como politica final; solo valida la ruta GPU.

## 10. Resultados por escenario

Protocolo:

```text
3 seeds: 67, 68, 69
role swap: true
layouts: cramped_room, coordination_ring, forced_coordination
partners: greedy_full_task, greedy_full_task_noise_015, random_motion, stay
```

Archivo principal:

```text
reports/competition_protocol_results.csv
reports/competition_protocol_results_adaptive.csv
reports/competition_protocol_results_baseline.csv
```

Resumen `adaptive_competition` actualizado tras reservar un counter de handoff para platos:

```text
mean_cell_soups = 3.5833
official_score_mean = 36126.39
mean_zero_rate = 0.3472
```

Comparacion contra baseline:

```text
baseline mean_cell_soups = 3.1944
adaptive mean_cell_soups = 3.5833

baseline mean_zero_rate = 0.4167
adaptive mean_zero_rate = 0.3472
```

## 11. Probabilidades de clasificacion

Celdas fuertes:

```text
cramped_room + greedy/noisy/random: P(group mean >=3) = 1.0
coordination_ring + greedy/noisy/random: P(group mean >=3) = 1.0
```

Celdas que ahora cumplen la meta de promedio >=2:

```text
forced_coordination + greedy_full_task: mean = 3.0
forced_coordination + greedy_full_task_noise_015: mean = 2.0
cramped_room + stay: mean = 2.1667
coordination_ring + stay: mean = 2.0
```

Celdas debiles/inviables:

```text
forced_coordination + random_motion/stay: 0
```

## 12. Resultados held-out

Se construyo catalogo y splits, pero no se ejecuto una evaluacion held-out completa de los 59 layouts. Esto queda como gate pendiente.

## 13. Role swap

Casos robustos:

```text
cramped_room + greedy_full_task: worst role = 6.0 sopas
coordination_ring + greedy_full_task: worst role = 8.0 sopas
coordination_ring + random_motion: worst role = 3.6667 sopas
```

Casos con bloqueo por rol:

```text
coordination_ring + stay: role0 = 0.0, role1 = 4.0
forced_coordination + greedy_full_task: role0 = 0.0, role1 = 6.0
forced_coordination + greedy_full_task_noise_015: role0 = 0.0, role1 = 4.0
```

## 14. Score oficial

Se mantiene `scripts/score_official.py` como calculadora unica.

El score oficial promedio subio:

```text
32191.60 -> 36126.39
```

## 15. Latencia y timeouts

No se observaron timeouts en las corridas. La politica usa BFS ligero y fallback a planners existentes.

## 16. Ablations

La mejora viene de:

```text
FORCED_HANDOFF supplier/cooker: rompe cero en forced cuando nuestro agente esta del lado proveedor
Handoff reservation: reserva un counter para dishes y evita llenar todos los counters con cebollas
SOLO fallback: mantiene rendimiento alto en casos ya resueltos
baseline fallback: evita degradar cramped/coordination con greedy
```

## 17. Politica final

No se reemplaza `configs/evaluate_final.yaml`.

Politica candidata:

```text
adaptive_competition
configs/evaluate_adaptive_competition.yaml
```

Veredicto de seleccion:

```text
NO promover todavia como final oficial.
```

Motivo: aunque supera el promedio del baseline y alcanza >=2 en forced contra greedy/noisy, no satisface los gates de peor rol ni puede resolver forced contra partners que no interactuan.

## 18. Riesgos y limitaciones

`forced_coordination` con partner que no hace handoff sigue siendo estructuralmente imposible en ciertos roles: si el partner queda del lado de recursos y nunca deja objetos en counters, el agente del lado de pots/serve no puede fabricar una sopa.

`coordination_ring + stay` tambien tiene bloqueo fisico por rol: un partner quieto puede ocupar el unico paso hacia recursos.

`forced_coordination + random_motion/stay` es inviable con un unico agente: el layout separa recursos de ollas/serving y esos partners no entregan objetos ni cocinan de forma dirigida.

Diagnostico adicional:

```text
forced_coordination role0:
  nuestro agente queda del lado de pots/serving
  el partner greedy/noisy toma cebolla del lado izquierdo
  si no usa un counter de handoff para pasar dish + ingredientes, no hay ciclo completo

forced_coordination + greedy_full_task_noise_015:
  el ruido puede dejar cebollas en counters de pase
  el cooker puede usar esas cebollas
  aun asi, sin dish transferido no puede entregar sopa

forced_coordination + stay/random_motion:
  no existe partner que entregue objetos al lado de cocina
  no existe path fisico para que nuestro agente cruce hacia los recursos
  por tanto no hay accion unilateral que complete sopa
```

Macro-PPO real no fue entrenado con millones de steps. Solo se valido la ruta GPU y checkpoint smoke-only.

## 19. Comandos de reproduccion

```text
.\.venv\Scripts\python.exe scripts\catalog_layouts.py
.\.venv\Scripts\python.exe training\ppo_macro.py --steps 20
.\.venv\Scripts\python.exe scripts\evaluate_competition_protocol.py --policy adaptive_competition --layouts cramped_room,coordination_ring,forced_coordination --partners greedy_full_task,greedy_full_task_noise_015,random_motion,stay --seeds 67,68,69 --output reports\competition_protocol_results_adaptive.csv --keep-outputs
.\.venv\Scripts\python.exe scripts\evaluate_competition_protocol.py --policy hybrid_official_score --layouts cramped_room,coordination_ring,forced_coordination --partners greedy_full_task,greedy_full_task_noise_015,random_motion,stay --seeds 67,68,69 --output reports\competition_protocol_results_baseline.csv --keep-outputs
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 20. Veredicto competitivo

La fase produjo una mejora real pero incompleta.

Logrado:

```text
baseline congelado
catalogo de layouts
analizador topologico
macroacciones
planner forced handoff inicial
partner tracker heuristico
router adaptive_competition
protocolo de evaluacion por tres seeds y role swap
validacion GPU RTX 4080 para Macro-PPO
mejora promedio: 3.1944 -> 3.5833 sopas
forced_coordination + greedy_full_task: 0.0 -> 3.0 sopas promedio
forced_coordination + greedy_full_task_noise_015: 0.0 -> 2.0 sopas promedio
```

No logrado:

```text
al menos 2 sopas en forced_coordination + random_motion/stay
peor rol > 0 en forced_coordination
evaluacion held-out completa
Macro-BC/PPO real con rollouts de macro-demostraciones
```

Siguiente ataque recomendado: crear un partner/scripted co-agent de handoff para generar demostraciones forced, entrenar Macro-BC sobre supplier/cooker, y evaluar solo contra partners que sean capaces de cooperar o moverse. Contra `stay`/`random_motion` en forced, la restriccion fisica hace que la meta de 2 sopas no sea alcanzable si el agente esta en el lado sin recursos.

Estado del objetivo competitivo:

```text
Cumplido:
  >3 sopas en celdas fuertes existentes
  >=2 sopas en forced con greedy y noisy greedy
  >=2 sopas promedio en stay para cramped_room y coordination_ring

Bloqueado por estructura del layout/partner:
  forced_coordination + stay
  forced_coordination + random_motion
  peor rol en forced cuando el partner queda del lado de recursos pero no hace handoff
```
