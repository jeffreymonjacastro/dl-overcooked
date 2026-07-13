# Guía viable de 6–7 horas — Entrenamiento local Score-First

## 0. Instrucción para Codex

Trabaja únicamente dentro de:

```text
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
```

Esta fase se ejecuta **solo en la máquina local**. No preparar notebooks ni versiones para Kaggle.

Conserva sin modificar:

```text
hybrid_official_score
adaptive_competition
configs/evaluate_final.yaml
```

La nueva candidata debe guardarse aparte:

```text
adaptive_competition_shortppo
configs/evaluate_shortppo_candidate.yaml
artifacts/shortppo/
reports/shortppo/
```

No reemplazar el agente final hasta superar los gates de esta guía.

---

# 1. Decisión principal

No se necesitan millones de acciones para esta fase.

La política ya dispone de:

```text
analizador topológico
macroacciones
planner
partner tracker
adaptive_competition
```

Por tanto, el entrenamiento no empieza desde cero. El objetivo es **ajustar la selección de macroacciones** sobre una política ya competente.

Presupuesto máximo:

```text
6–7 horas de reloj
```

Objetivo inicial de entrenamiento:

```text
100 000–300 000 macro-decisions
```

No fijar 2M de antemano. El número final depende del throughput medido.

Detener el entrenamiento cuando:

```text
se agote el presupuesto;
held-out deje de mejorar;
el score oficial empeore;
la política colapse a una macro;
```

---

# 2. Objetivo oficial obligatorio

Para cada intento:

```text
Score =
10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- min(100 * timeouts, 5000)
```

Si no se entrega ninguna sopa:

```text
Score = 0
```

Para cada escenario:

```text
scenario_score =
mean(attempt_score_seed_1,
     attempt_score_seed_2,
     attempt_score_seed_3)
```

## 2.1 Prioridad de selección

La métrica principal debe ser:

```text
mean scenario_score sobre grupos de tres seeds
```

La fórmula ya prioriza fuertemente el número de sopas.

Además, usar como restricciones de seguridad:

```text
mean soups
zero-soup rate
10th percentile scenario score
worst role
timeouts
```

Orden práctico:

```text
1. superar el score oficial del baseline;
2. aumentar mean soups;
3. reducir zero-rate;
4. mejorar percentil 10 y peor rol;
5. optimizar first/last soup;
6. cero timeouts.
```

No seleccionar por:

```text
PPO loss
train reward
validation loss
último checkpoint
```

---

# 3. Metas de esta corrida

Baseline actual aproximado:

```text
adaptive_competition:
  mean soups = 3.5833
  official score mean = 36126.39
  zero-rate = 0.3472
```

Metas deseables de la fase corta:

```text
mean soups >= 3.8
official score mean >= 38000
zero-rate <= 0.30
forced + cooperative partner:
  worst role > 0
```

Meta ambiciosa:

```text
mean soups >= 4.0
official score mean >= 40000
```

No afirmar éxito si la mejora aparece solo en training layouts.

---

# 4. Plan total de 6–7 horas

## Bloque A — Benchmark y sanity checks

Duración máxima:

```text
30 minutos
```

Tareas:

```text
1. ejecutar tests;
2. verificar score_official.py;
3. probar 8, 16 y 32 workers;
4. medir macro-decisions/s;
5. escoger el throughput más alto estable;
6. ejecutar un smoke de 2 000–5 000 macro-decisions.
```

Abortar si:

```text
hay NaN/Inf;
el checkpoint no reanuda;
el score se calcula incorrectamente;
el action mask falla;
```

---

## Bloque B — Evaluación corta antes de entrenar

Duración máxima:

```text
45 minutos
```

No evaluar los 59 layouts profundamente.

Seleccionar un panel de 12–16 layouts representativos:

```text
3 SOLO_CAPABLE
3 SHARED_OPEN
3 BOTTLENECK
2–3 FORCED/ASYMMETRIC
2 adversariales o difíciles
```

Partners:

```text
greedy_full_task
greedy_full_task_noise_015
random_motion
```

Protocolo:

```text
3 seeds
role swap
```

Comparar:

```text
hybrid_official_score
adaptive_competition
```

Guardar:

```text
reports/shortppo/pretrain_evaluation.csv
reports/shortppo/pretrain_summary.md
```

---

## Bloque C — Dataset macro rápido

Duración máxima:

```text
45–60 minutos
```

Recolectar demostraciones únicamente de políticas ya disponibles:

```text
adaptive_competition
GreedyHumanModel
macro planner
scripted handoff si ya existe
```

No construir una infraestructura nueva compleja durante esta corrida.

Priorizar:

```text
entregas exitosas
handoffs exitosos
recuperaciones
forced supplier/cooker
acciones previas a una sopa
```

Objetivo mínimo:

```text
10 000–30 000 decisiones macro válidas
```

Balancear:

```text
SOLO/SHARED
BOTTLENECK
FORCED/ASYMMETRIC
```

---

## Bloque D — Macro-BC corto

Duración máxima:

```text
30–45 minutos
```

Entrenar:

```text
10–20 épocas
early stopping
```

Evaluar online cada pocas épocas.

Checkpoint de warm-start:

```text
artifacts/shortppo/macro_bc_warmstart.pt
```

Gate:

```text
la policy debe producir macros válidas;
no debe colapsar;
debe mantener al menos 90 % del score de adaptive_competition;
```

Si Macro-BC es claramente peor, iniciar PPO desde los pesos compatibles de la policy/planner actual o cancelar PPO y dedicar el tiempo a búsqueda de parámetros.

---

## Bloque E — Búsqueda rápida de parámetros del planner

Duración máxima:

```text
30–45 minutos
```

Ejecutar antes o en paralelo con el entrenamiento corto.

Buscar:

```text
timeout de espera por partner
umbral de partner pasivo
prioridad dish/onion
momento de pasar dish
counter reservado
duración mínima de modo
umbral de recovery
```

Usar:

```text
random search o CEM pequeño
20–50 configuraciones
```

Evaluar cada configuración con pocas seeds y conservar las 3 mejores.

Esta fase puede producir una mejora más rápida que PPO.

---

## Bloque F — Macro-PPO corto y dirigido

Duración objetivo:

```text
2.5–3.5 horas
```

Presupuesto:

```text
100k–300k macro-decisions
```

No entrenar low-level actions.

Inicialización:

```text
Macro-BC warm-start
o mejor checkpoint compatible disponible
```

Curriculum compacto:

```text
40 % greedy / noisy greedy
25 % random_motion en layouts solo-capable
20 % bottlenecks
15 % forced handoff con partner cooperativo
```

No entrenar `forced + stay` como objetivo de éxito.

Guardar checkpoints:

```text
25k
50k
100k
150k
200k
250k
300k
```

Evaluar cada:

```text
25k–50k macro-decisions
```

---

## Bloque G — Evaluación final

Duración máxima:

```text
60–75 minutos
```

Comparar:

```text
hybrid_official_score
adaptive_competition
mejor planner parametrizado
Macro-BC
mejor Macro-PPO
```

Usar:

```text
panel held-out
3 seeds
role swap
partners oficiales disponibles
```

Calcular:

```text
attempt_score por seed
scenario_score por grupo de tres seeds
mean soups
zero-rate
percentil 10
worst role
timeouts
latency
```

---

# 5. PPO viable

## 5.1 Acción

El actor selecciona macroacciones:

```text
GET_INGREDIENT
PLACE_IN_POT
GET_DISH
PASS_INGREDIENT
PASS_DISH
RECEIVE_OBJECT
PICK_SOUP
DELIVER
YIELD
UNBLOCK
```

El planner convierte cada macro en acciones primitivas.

## 5.2 Semi-MDP

Para una macro de duración `k`:

```text
R_macro = sum(gamma^i * reward[t+i])
bootstrap = gamma^k * V(next_state)
```

No usar descuento de un solo paso para todas las macros.

## 5.3 Reward de entrenamiento

La selección final siempre usa el score oficial exacto.

Para estabilidad, usar reward escalada:

```text
+100 por sopa
+shaping pequeño por progreso real
```

El shaping total no debe equivaler a una sopa.

Permitido:

```text
ingrediente correcto en pot
dish útil
soup recogida
handoff correcto
desbloqueo real
```

No permitido:

```text
recompensar caminar
recompensar interact sin efecto
shaping que domine entregas
```

---

# 6. Evaluación durante training

Cada evaluación debe producir:

```text
mean official scenario score
mean soups
zero-rate
10th percentile score
worst role
first soup
last soup
timeouts
```

Guardar:

```text
reports/shortppo/checkpoint_metrics.csv
```

La selección del checkpoint debe usar:

```text
1. mayor official scenario score;
2. mayor mean soups;
3. menor zero-rate;
4. mejor percentil 10;
5. mejor worst role.
```

---

# 7. Early stopping obligatorio

Detener antes de agotar las 7 horas si durante tres evaluaciones consecutivas:

```text
official score no mejora;
held-out soups no mejora;
zero-rate aumenta;
worst role empeora;
la policy colapsa a una sola macro;
```

Conservar siempre el mejor checkpoint anterior.

---

# 8. Qué hacer si PPO no mejora

No extender automáticamente.

Usar el mejor resultado entre:

```text
adaptive_competition
planner parametrizado por CEM
Macro-BC
Macro-PPO
```

Una búsqueda de parámetros puede ser el ganador aunque PPO no funcione.

El objetivo es producir más sopas y más score, no demostrar que PPO funciona.

---

# 9. Criterios para promover el candidato

Promover solo si cumple en held-out:

```text
official score > adaptive_competition
mean soups > 3.5833
zero-rate no empeora
worst role no empeora
cero timeouts
```

Preferencia:

```text
mean soups >= 3.8
official score >= 38000
```

No reemplazar `configs/evaluate_final.yaml` automáticamente.

Crear primero:

```text
configs/evaluate_shortppo_candidate.yaml
```

---

# 10. Archivos que deben quedar

```text
configs/train_macro_ppo_7h.yaml
configs/evaluate_shortppo_candidate.yaml
scripts/run_shortppo_7h.ps1
scripts/evaluate_shortppo_candidate.py
artifacts/shortppo/
reports/shortppo/pretrain_evaluation.csv
reports/shortppo/checkpoint_metrics.csv
reports/shortppo/final_evaluation.csv
reports/FINAL_SHORT_TRAINING_REPORT.md
```

---

# 11. Reporte final

Crear:

```text
reports/FINAL_SHORT_TRAINING_REPORT.md
```

Debe incluir:

```markdown
# Final Short Training Report

## 1. Tiempo total
## 2. Hardware y workers
## 3. Throughput
## 4. Baselines
## 5. Dataset macro
## 6. Macro-BC
## 7. Planner parameter search
## 8. Macro-PPO
## 9. Checkpoints
## 10. Score oficial por tres seeds
## 11. Sopas y zero-rate
## 12. Worst role
## 13. Held-out
## 14. Timeouts y latencia
## 15. Mejor candidato
## 16. Promoción o rechazo
## 17. Limitaciones
```

Actualizar también:

```text
CURRENT_STATUS_OPTION_A.md
```

---

# 12. Prompt corto para Codex

```text
Trabaja únicamente dentro de:
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked

Sigue:
GUIA_ENTRENAMIENTO_VIABLE_7H_SCORE_FIRST.md

Presupuesto máximo total: 6–7 horas.

No ejecutes millones de acciones por defecto. Usa 100k–300k macro-decisions,
warm-start, evaluación periódica y early stopping.

La métrica principal es el score oficial:

10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- min(100 * timeouts, 5000).

Calcula el score por intento y el promedio exacto de tres seeds por escenario.

Antes de PPO:
1. benchmark de workers;
2. evaluación held-out corta;
3. dataset macro rápido;
4. Macro-BC;
5. búsqueda rápida de parámetros del planner.

Después ejecuta Macro-PPO dirigido durante 2.5–3.5 horas como máximo.

Evalúa checkpoints cada 25k–50k macro-decisions y detén si el score oficial,
held-out soups o peor rol dejan de mejorar.

No reemplaces configs/evaluate_final.yaml salvo que el candidato supere
adaptive_competition en score oficial, mean soups, zero-rate y worst role.

Crea:
reports/FINAL_SHORT_TRAINING_REPORT.md
```
