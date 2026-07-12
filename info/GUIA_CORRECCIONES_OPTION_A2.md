# Guía de correcciones y siguiente iteración — Overcooked Option A2

## 0. Instrucción principal para Codex

Trabaja **únicamente** dentro de:

```text
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
```

Tienes permisos de lectura, escritura, ejecución, modificación y eliminación **solo dentro de esa carpeta**. No accedas, modifiques ni inspecciones archivos fuera de esa ruta.

Antes de cambiar código:

1. Lee completamente:
   - `CURRENT_STATUS_OPTION_A.md`
   - `LINEAMIENTO_PROYECTO_OVERCooked.md`
   - `DATA_STRUCTURE.md`
   - cualquier reporte previo en `reports/`
2. Inspecciona la implementación real de:
   - `training/train_option_a_gru_bc.py`
   - `training/datasets.py`
   - `models/option_a_gru_policy.py`
   - `policies/template.py`
   - scripts de exportación y evaluación
3. No asumas que el diagnóstico previo es correcto: **verifica cada hipótesis ejecutando pruebas**.
4. No empieces otro entrenamiento largo hasta completar la Fase 1 de integridad.

El objetivo es corregir Option A, producir una versión **A2 medible y más robusta**, decidir con evidencia si Option B merece continuar, y volver a generar el reporte final de estado con resultados reales.

---

# 1. Restricciones del proyecto

## 1.1 Partes que deben permanecer estables

No modificar salvo que exista un bug confirmado y documentado:

```text
src/
data/
policies/basic_policies.py
policies/human_keyboard_policy.py
configs/evaluate.yaml
```

La carpeta `data/` es de solo lectura: no borrar, renombrar ni sobrescribir datasets originales.

## 1.2 Partes permitidas para cambios

```text
training/
models/
scripts/
artifacts/
reports/
configs/ nuevos
policies/template.py
tests/
```

Si se detecta un bug que obligue a tocar `src/`, detenerse primero, documentar la evidencia y evitar el cambio mientras exista una solución externa.

## 1.3 Preservar el baseline actual

Antes de modificar pesos o reportes:

```text
artifacts/option_a_baseline_before_a2/
reports/baseline_before_a2/
```

Copiar allí:

- checkpoint PyTorch actual;
- exportación `.npz`;
- configuración del modelo;
- evaluación actual;
- reporte `CURRENT_STATUS_OPTION_A.md`.

No sobrescribir el baseline sin respaldo.

---

# 2. Objetivo de la corrección

El checkpoint actual demuestra que el pipeline funciona, pero presenta comportamiento frágil:

- funciona muy bien en algunas combinaciones;
- obtiene cero sopas en muchas otras;
- el criterio de selección se basó principalmente en `validation loss`;
- no está demostrada la paridad exacta entre PyTorch y NumPy;
- deben revisarse `previous_action`, el hidden state, los cortes de secuencia y el role swap;
- los datos humanos y teacher mezclan políticas y calidades incompatibles;
- los teacher rollouts son pocos y algunos no son exitosos.

La nueva iteración debe seguir este orden:

```text
integridad técnica
    ↓
diagnóstico de comportamiento
    ↓
corrección del dataset
    ↓
A2.1 GRU corregida
    ↓
A2.2 especialistas + router, solo si A2.1 no basta
    ↓
PPO reducido Option B, solo si supera gates
    ↓
selección final por score oficial
```

No resolver esto únicamente aumentando épocas, `hidden_size` o `seq_len`.

---

# 3. Fase 1 — Pruebas de integridad obligatorias

Crear:

```text
tests/
  test_gru_export_parity.py
  test_previous_action_alignment.py
  test_sequence_state_handling.py
  test_agent_reset.py
  test_action_mapping.py
  test_reward_and_soup_metrics.py

scripts/
  diagnose_option_a.py

reports/diagnostics/
```

Ejecutar todas las pruebas y guardar resultados en:

```text
reports/diagnostics/integrity_report.md
reports/diagnostics/parity_metrics.json
```

No continuar con entrenamiento si alguna prueba crítica falla.

## 3.1 Paridad PyTorch vs `.npz`

Comparar:

```text
artifacts/option_a/best_checkpoint.pt
artifacts/option_a/final_policy.npz
```

Usar las mismas secuencias de observaciones, índices de agente, acciones previas y estados iniciales.

Comparar en cada timestep:

- logits;
- hidden state;
- acción argmax;
- normalización de entrada.

Criterios:

```text
max_abs_error_logits <= 1e-5
max_abs_error_hidden <= 1e-5
action_match_rate = 100 %
```

La prueba debe cubrir:

- secuencia corta;
- episodio completo de 250 pasos;
- `agent_index = 0`;
- `agent_index = 1`;
- varios resets consecutivos.

Revisar especialmente la implementación de GRU:

- orden de compuertas de PyTorch;
- transposición de matrices;
- tratamiento separado de `bias_ih` y `bias_hh`;
- ecuación exacta del estado candidato;
- ecuación final de actualización de `h_t`.

Si no hay paridad, corregir la inferencia NumPy antes de volver a evaluar.

## 3.2 Alineación de `previous_action`

Verificar que el entrenamiento use:

```text
input[t].previous_action = action[t - 1]
```

y nunca `action[t]`.

Para `t = 0`, utilizar un token BOS o vector cero claramente documentado.

En inferencia:

```text
1. construir input con la acción anterior;
2. ejecutar forward;
3. seleccionar action_t;
4. almacenar action_t como previous_action del siguiente paso.
```

Crear un test con una secuencia artificial conocida que detecte desplazamientos de una posición.

## 3.3 Manejo de secuencias y hidden state

Determinar exactamente cómo se usan `seq_len` y las ventanas.

No aceptar este comportamiento:

```text
cada ventana intermedia comienza con hidden = 0
y start_flag = 1
```

si la ventana no representa el inicio real del episodio.

Implementar una de estas opciones correctamente:

### Opción recomendada: episodios completos

Entrenar secuencias de hasta 250 pasos con masking y batching por longitud.

### Alternativa: truncated BPTT

- segmentos consecutivos del mismo episodio;
- pasar hidden state del segmento anterior;
- `detach()` entre segmentos;
- `start_flag = 1` solo en el comienzo real del episodio.

### Alternativa: burn-in

- reconstruir hidden state con una sección inicial;
- calcular loss solo en los pasos posteriores.

Crear pruebas que confirmen que dos segmentos consecutivos producen el mismo hidden state que una ejecución continua.

## 3.4 `reset()` completo

`StudentAgent.reset()` debe reiniciar:

```text
hidden_state
previous_action
start_flag
timestep interno
buffers de historial
estado del router, si existe
```

Probar que ejecutar un episodio después de `reset()` produce el mismo inicio que crear una nueva instancia.

## 3.5 Convención de acciones y dinámica

Verificar en todos los datasets y en evaluación:

```text
0 = up
1 = down
2 = right
3 = left
4 = stay
5 = interact
```

Confirmar también:

- `old_dynamics`;
- formato de observación;
- dimensión real;
- `agent_index`;
- role swap;
- si existen datasets con otra convención.

Cualquier archivo incompatible debe quedar marcado y excluido en el manifest, no corregido silenciosamente.

## 3.6 Métricas reales de sopa

Verificar que `mean_soups_proxy` coincida con las sopas entregadas por el entorno.

No inferir sopas solo dividiendo reward si la configuración puede cambiar.

Guardar por episodio:

```text
soups_delivered
sparse_return
official_score
first_soup_timestep
last_soup_timestep
timeouts
```

Usar la fórmula oficial del proyecto:

```text
score =
10000 * soups
+ 10 * (horizon - last_soup_timestep)
+ (horizon - first_soup_timestep)
- penalty
```

Si no hay sopa, score = 0.

## 3.7 Evaluación PyTorch directa vs NumPy exportado

Evaluar el mismo checkpoint de dos formas:

```text
A. modelo PyTorch directo;
B. modelo NumPy usado por StudentAgent.
```

Usar exactamente las mismas seeds, layouts, partners y roles.

Los resultados deben ser equivalentes. Si no lo son, no entrenar todavía.

---

# 4. Fase 2 — Diagnóstico de comportamiento

Crear:

```text
configs/watch_option_a_success.yaml
configs/watch_option_a_failure_greedy.yaml
configs/watch_option_a_forced.yaml
scripts/trace_episode.py
reports/diagnostics/failure_catalog.csv
reports/diagnostics/behavior_analysis.md
```

Analizar al menos:

```text
5 episodios exitosos
5 fallos con greedy_full_task
5 fallos en forced_coordination
5 fallos con stay o random_motion
```

Para cada episodio registrar:

- timestep;
- observación resumida;
- objeto sostenido;
- posición de ambos agentes si está disponible;
- acción del ego;
- acción del partner;
- probabilidad/logits;
- hidden norm;
- repeticiones;
- interacciones;
- sopa o progreso;
- motivo de finalización.

Clasificar cada fallo en una taxonomía:

```text
deadlock físico
duplicación de tarea
stay prolongado
loop de 2–4 acciones
interacción nula repetida
objeto incorrecto en mano
ignora plato
ignora sopa lista
no entrega
conflicto de rol
desorientación por role swap
drift del hidden state
estado fuera de distribución
otro
```

El reporte debe contar cuántos fallos pertenecen a cada categoría.

No proponer la siguiente arquitectura sin esta evidencia.

---

# 5. Fase 3 — Reauditoría y corrección del dataset

Actualizar:

```text
training/datasets.py
scripts/audit_datasets.py
reports/dataset_inventory_a2.csv
reports/dataset_quality_a2.csv
artifacts/shared/split_manifest_a2.json
```

## 5.1 Metadata obligatoria por episodio

Cada episodio debe tener:

```text
episode_id
group_id
layout
partner_type
source_type
agent_index
role_swap
seed
num_steps
soups
official_score_proxy o real
first_soup_timestep
last_soup_timestep
percent_stay
percent_interact
longest_repeated_action_run
changed_state_rate
has_custom_layout
quality_tier
```

`source_type` debe distinguir al menos:

```text
human
greedy_teacher
other_teacher
```

## 5.2 No mezclar duplicados

Una trayectoria con `.npz` y `.pkl` del mismo stem debe contarse una sola vez.

Prioridad:

```text
.npz para tensores
.metadata.json para contexto
.pkl solo para información faltante
```

## 5.3 Quality tiers

No borrar automáticamente episodios de cero sopas. Clasificarlos:

```text
Tier A:
  varias sopas, score alto, comportamiento activo

Tier B:
  al menos una sopa, progreso consistente

Tier C:
  cero sopas pero navegación/interacciones útiles o estados raros

Tier D:
  inactividad, loops improductivos, archivo incompleto o comportamiento corrupto
```

Uso inicial:

```text
A: peso alto
B: peso normal
C: peso bajo o dataset auxiliar de cobertura
D: excluir del BC principal
```

La ponderación exacta debe ser configurable y registrada.

## 5.4 Balance de acciones

No eliminar indiscriminadamente `stay`; esperar puede ser una acción correcta.

Distinguir:

```text
stay productivo:
  espera por cocción, cede paso, evita choque, espera handoff

stay improductivo:
  repetición prolongada sin cambio de estado ni progreso
```

Aplicar una combinación configurable de:

- class weights con límites;
- downsampling solo de `stay` improductivo;
- sampling balanceado por episodio/layout/partner;
- focal loss opcional.

No medir éxito por accuracy global.

## 5.5 Pesos por calidad temporal

Probar una variante de loss ponderada por retorno futuro:

```text
weight_t = episode_quality_weight * (1 + lambda * normalized_return_to_go_t)
```

Así, las acciones cercanas a trayectorias que terminan en sopa reciben mayor importancia.

Comparar contra CE estándar mediante evaluación online.

---

# 6. Fase 4 — Teacher rollouts dirigidos

Auditar los 60 rollouts teacher existentes.

Crear:

```text
reports/teacher_rollouts_audit.csv
```

Incluir:

```text
layout
partner
role
seed
soups
score
zero_soup
source_checkpoint
```

No utilizar rollouts teacher fallidos con el mismo peso que los exitosos.

Generar nuevos rollouts únicamente para celdas críticas:

```text
cramped_room + greedy_full_task
coordination_ring + stay
coordination_ring + random_motion
forced_coordination + cada partner relevante
```

Antes de guardar una trayectoria como teacher, comprobar que el teacher realmente obtiene sopa.

Requisitos:

```text
guardar solo teacher_good para BC principal;
guardar teacher_bad separado para análisis;
preservar source_type y partner_type;
no usar información privilegiada en el input del alumno;
no duplicar rollouts equivalentes.
```

Si el greedy no resuelve un layout, no usarlo como teacher allí. Evaluar otro teacher o usar RL dirigido.

---

# 7. Fase 5 — A2.1: GRU-BC corregida

Crear una nueva línea sin sobrescribir A1:

```text
models/option_a2_gru_policy.py
training/train_option_a2_gru_bc.py
configs/train_option_a2.yaml
configs/evaluate_option_a2.yaml
artifacts/option_a2/
reports/option_a2/
```

## 7.1 Arquitectura base

Mantener inicialmente una arquitectura controlada:

```text
obs normalizada
+ agent_index one-hot
+ previous_action one-hot
+ start flag
    ↓
MLP encoder
    ↓
GRU
    ↓
action head de 6 acciones
```

No aumentar tamaño hasta demostrar que la implementación y el dataset están correctos.

Primera corrida:

```text
hidden_size = 128
episodios completos o TBPTT correcto
```

Segunda corrida opcional, solo si la primera está sana:

```text
hidden_size = 256
```

## 7.2 Reducir exposure bias

Comparar:

### Baseline

Teacher forcing con `action[t-1]` real.

### Variante

Scheduled sampling gradual:

```text
inicio: casi siempre acción humana previa
final: mezclar acción predicha previa
```

Registrar la probabilidad usada por época.

No aplicar scheduled sampling antes de pasar todas las pruebas de secuencia.

## 7.3 Loss

Comparar de manera controlada:

```text
A2.1a: CE estándar balanceada
A2.1b: CE + quality/return-to-go weighting
A2.1c: focal loss opcional
```

No combinar todas las técnicas en la primera corrida.

## 7.4 Selección de checkpoint

No seleccionar solo por `val_loss`.

Cada 2–5 épocas ejecutar una evaluación online compacta:

```text
2–3 layouts
3 partners
2 seeds
role swap
```

Orden de selección:

```text
1. menor zero-soup rate
2. mayor soups promedio
3. mayor score oficial
4. menor timeout rate
5. val_loss como desempate
```

Guardar:

```text
best_by_val_loss.pt
best_by_zero_soup.pt
best_by_official_score.pt
```

Evaluar los tres al final.

## 7.5 Gates para aceptar A2.1

A2.1 debe mejorar A1 bajo el mismo protocolo.

Gate mínimo:

```text
zero-soup rate < A1
mean soups > A1
smoke final con greedy_full_task entrega al menos una sopa
sin degradación grave en las celdas donde A1 obtenía 5 sopas
latencia p95 < 100 ms
```

No declarar éxito por una loss menor.

---

# 8. Fase 6 — A2.2: especialistas y router

Solo implementar si A2.1 continúa mostrando resultados bimodales por partner.

No usar promedio simple de logits como primera opción.

## 8.1 Experto S: autosuficiente/robusto

Entrenar principalmente con:

```text
stay
random_motion
partners de baja utilidad
datos humanos donde el humano completa el pipeline
```

Objetivo:

```text
completar sopa sin depender del partner
evitar bloqueos
recuperarse de estados raros
```

## 8.2 Experto G: complementario a greedy

Entrenar principalmente con:

```text
greedy_full_task
greedy sticky
greedy con random actions
demos humanas coordinadas
```

Objetivo:

```text
no duplicar la tarea del partner
priorizar la subtarea complementaria
ceder y desbloquear
aprovechar sopa lista
```

## 8.3 Experto F opcional

Solo si hay datos suficientes:

```text
forced_coordination
handoffs
small corridors
roles separados
```

## 8.4 Router

Implementar primero hard routing.

Cuando el partner es conocido por configuración y las reglas lo permiten:

```text
partner declarado → experto correspondiente
```

Cuando el partner es desconocido:

```text
historial de observaciones/acciones
    ↓
router recurrente pequeño
    ↓
selección de experto
```

Evitar cambiar de experto en cada timestep. Usar:

```text
hysteresis
mínimo de pasos por experto
confidence threshold
```

Comparar:

```text
mejor experto individual
router
promedio de logits, solo como ablation
```

Aceptar el router solo si supera consistentemente a ambos expertos.

---

# 9. Fase 7 — Option B reducida y con criterio de parada

No ejecutar FCP/MEP/COLE completo.

Option B solo continúa como:

```text
A2 warm-start
    ↓
PPO reducido
    ↓
partner pool pequeña
```

Partner pool inicial:

```text
greedy_full_task
greedy sticky
greedy con ruido
random_motion
stay
1–2 checkpoints históricos de A2
```

## 9.1 Curriculum

Orden sugerido:

```text
1. cramped_room + greedy
2. greedy sticky/random
3. coordination_ring
4. forced_coordination
5. mezcla de layouts y partners
```

## 9.2 Recompensa

Mantener reward final de sopa como objetivo principal.

Se permite shaping temporal para:

```text
ingrediente útil en olla
plato recogido cuando hay sopa lista
sopa recogida
entrega
desbloqueo o progreso
```

Documentar exactamente el shaping y comprobar que no produce reward hacking.

## 9.3 Rendimiento

Usar entornos vectorizados si la infraestructura lo permite.

Registrar:

```text
steps por segundo
uso CPU
uso GPU
tiempo por millón de pasos
```

## 9.4 Gate de muerte de Option B

Después de 1–3 millones de pasos efectivos, detener B si no cumple:

```text
reduce zero-soup rate respecto a A2;
mejora score oficial;
resuelve al menos una celda crítica adicional;
no destruye las celdas buenas de A2.
```

No continuar solo porque la reward de entrenamiento aumenta.

---

# 10. Protocolo de evaluación común

Crear o actualizar:

```text
scripts/run_evaluation_matrix.py
reports/evaluation_matrix_a2.csv
reports/evaluation_summary_a2.md
```

Comparar bajo condiciones idénticas:

```text
GreedyFullTask baseline
A0 MLP
A1 GRU actual
A2.1 GRU corregida
A2.2 expertos/router, si existe
Option B reducida, si existe
ensemble simple solo como ablation
```

## 10.1 Matriz mínima

Para desarrollo:

```text
layouts críticos
× stay
× random_motion
× greedy_full_task
× sticky/random greedy si está disponible
× 5 seeds
× role swap
```

Para candidatos finales:

```text
todos los layouts disponibles
× todos los partners disponibles
× mínimo 20 seeds en celdas críticas
× role swap
```

## 10.2 Reportar desagregado

No mostrar únicamente promedios globales.

Reportar:

```text
por layout
por partner
por seed
por agent_index
con/sin role swap
known layout
unseen layout
custom layout
```

Métricas:

```text
mean soups
median soups
zero-soup rate
official score mean/median
first soup timestep
last soup timestep
timeouts
latency mean/p95/max
```

## 10.3 Intervalos

Para candidatos finales, calcular bootstrap 95 % CI o al menos desviación estándar.

---

# 11. Integración final

La política ganadora debe exportarse a:

```text
artifacts/final_policy.npz
artifacts/final_policy_config.json
```

Si hay expertos/router:

```text
artifacts/final/
  solo_policy.npz
  greedy_policy.npz
  forced_policy.npz        # opcional
  router_policy.npz
  final_policy_config.json
```

`policies/template.py` debe:

```text
cargar pesos una sola vez
normalizar exactamente igual que training
resetear estado recurrente
devolver entero 0..5
no leer datasets
no entrenar
no escribir archivos
mantener p95 < 100 ms
```

Ejecutar smoke final:

```bash
.venv\Scripts\python.exe -m src.evaluate --config configs\evaluate_final.yaml
```

El smoke no puede terminar con cero sopas en todas las partidas.

---

# 12. Reporte final obligatorio

Al terminar, volver a escribir el reporte de estado.

## 12.1 Preservar el reporte anterior

Guardar el reporte previo como:

```text
reports/baseline_before_a2/CURRENT_STATUS_OPTION_A_BEFORE_A2.md
```

## 12.2 Crear el nuevo reporte principal

Crear o reemplazar:

```text
CURRENT_STATUS_OPTION_A.md
```

También crear:

```text
reports/FINAL_CORRECTION_REPORT_A2.md
```

Ambos deben contener resultados ejecutados, no planes futuros.

## 12.3 Estructura obligatoria del reporte

```markdown
# Estado actual después de correcciones A2

## 1. Resumen ejecutivo
- qué se corrigió;
- qué sigue fallando;
- si el agente es competitivo o no.

## 2. Bugs encontrados
- evidencia;
- archivo y línea;
- corrección aplicada;
- test que lo valida.

## 3. Integridad del modelo
- PyTorch vs NPZ;
- previous_action;
- hidden state;
- reset;
- action mapping;
- métricas de sopa.

## 4. Cambios en dataset
- episodios por tier;
- exclusiones;
- weights;
- teacher rollouts aceptados/rechazados;
- splits finales.

## 5. Modelos entrenados
Para cada modelo:
- arquitectura;
- hiperparámetros;
- epochs/steps;
- GPU/CPU;
- tiempo;
- checkpoint;
- criterio de selección.

## 6. Resultados offline
- train/val loss;
- accuracy por acción;
- matriz de confusión;
- métricas de secuencia.

## 7. Resultados online
- tabla por layout × partner × rol;
- sopas;
- zero-soup rate;
- score oficial;
- timeouts;
- latencia.

## 8. Comparación
- A0 vs A1 vs A2.1 vs A2.2 vs B;
- mejoras y regresiones;
- intervalos o variabilidad.

## 9. Análisis visual de fallos
- taxonomía;
- ejemplos;
- frecuencia por tipo.

## 10. Estado de Option B
- pasos entrenados;
- mejora real o no;
- decisión: continuar o detener;
- justificación cuantitativa.

## 11. Modelo recomendado
- checkpoint elegido;
- razón;
- riesgos conocidos;
- comando exacto de evaluación.

## 12. Veredicto honesto
- listo para competir: sí/no;
- siguiente acción de mayor impacto.
```

## 12.4 Archivos anexos obligatorios

```text
reports/diagnostics/integrity_report.md
reports/diagnostics/behavior_analysis.md
reports/dataset_quality_a2.csv
reports/teacher_rollouts_audit.csv
reports/evaluation_matrix_a2.csv
reports/evaluation_summary_a2.md
reports/FINAL_CORRECTION_REPORT_A2.md
```

---

# 13. Respuesta final que Codex debe entregar al usuario

Al finalizar, Codex debe responder con:

1. resumen de lo ejecutado;
2. bugs encontrados y corregidos;
3. modelos entrenados;
4. mejor resultado antes y después;
5. si Option B fue detenida o continúa;
6. ruta exacta del mejor checkpoint;
7. comando exacto para evaluarlo;
8. rutas de todos los reportes;
9. limitaciones que aún quedan.

No afirmar que algo funciona sin incluir la métrica o la prueba que lo demuestra.

---

# 14. Prompt corto para iniciar esta guía

Usar este mensaje con Codex:

```text
Trabaja únicamente dentro de
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked

Sigue completa y secuencialmente la guía
GUIA_CORRECCIONES_OPTION_A2.md.

Empieza por las pruebas de integridad y no entrenes un nuevo modelo hasta que
PyTorch vs NPZ, previous_action, hidden state, reset, action mapping y métricas
hayan sido verificadas.

Luego implementa A2.1, evalúala con el protocolo común y crea A2.2 solamente
si los resultados muestran dependencia bimodal del partner. Option B debe
continuar únicamente como PPO warm-start reducido y debe detenerse si no supera
los gates definidos.

Al terminar, conserva el reporte anterior, vuelve a generar
CURRENT_STATUS_OPTION_A.md y crea reports/FINAL_CORRECTION_REPORT_A2.md con
todos los resultados reales, comandos, checkpoints, regresiones y limitaciones.
No entregues solo un plan: ejecuta las pruebas y reporta los resultados.
```
