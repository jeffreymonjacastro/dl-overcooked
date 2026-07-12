# Guía de implementación para dos agentes Overcooked-AI: opciones [A] y [B]

## Cómo usar esta guía con Codex

Esta guía está diseñada para que dos personas trabajen en paralelo sobre el mismo proyecto, pero entrenen enfoques distintos y comparables.

Prompt mínimo para la primera persona:

```text
Usa la guía CODEX_GUIA_ENTRENAMIENTO_AB.md con la opción [A].
Trabaja únicamente dentro de:
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
Implementa, ejecuta, evalúa y documenta la opción [A] completa. No implementes la opción [B].
```

Prompt mínimo para la segunda persona:

```text
Usa la guía CODEX_GUIA_ENTRENAMIENTO_AB.md con la opción [B].
Trabaja únicamente dentro de:
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
Implementa, ejecuta, evalúa y documenta la opción [B] completa. No implementes la opción [A].
```

Después de recibir los dos checkpoints:

```text
Usa la guía CODEX_GUIA_ENTRENAMIENTO_AB.md en modo [ENSEMBLE].
Integra los artefactos finales de [A] y [B], calibra el ensemble y compáralo contra ambos modelos individuales.
```

---

# 1. Objetivo real

El objetivo no es maximizar la accuracy offline ni reproducir un paper completo. El objetivo es obtener el mayor score posible en la competencia:

```text
score = 10000 * numero_de_sopas
      + 10 * (horizon - timestep_ultima_sopa)
      + (horizon - timestep_primera_sopa)
      - penalizacion
```

La prioridad de selección debe ser, en este orden:

1. Mayor número promedio de sopas.
2. Menor proporción de episodios con cero sopas.
3. Mayor score oficial promedio.
4. Mejor robustez por layout, partner, seed y role swap.
5. Latencia de inferencia segura y ausencia de timeouts.

No declarar que un modelo es mejor solo porque tiene menor loss de entrenamiento o mayor accuracy de acciones.

---

# 2. Restricciones del repositorio

## 2.1 Directorio de trabajo

Trabaja únicamente dentro de:

```text
C:\Users\USER\Desktop\DL_FINAL\dl-overcooked
```

Puedes crear, modificar, ejecutar y eliminar archivos únicamente dentro de esa carpeta. No accedas a credenciales, repositorios, carpetas personales ni recursos fuera de ella.

## 2.2 Partes estables que no deben modificarse

Trata como infraestructura oficial y casi intocable:

```text
src/
  evaluate.py
  run_game.py
  runner.py
  environment.py
  observations.py
  policy_loader.py
  policy_wrappers.py
  demonstrations.py
  constants.py

policies/
  basic_policies.py
  human_keyboard_policy.py

configs/
  evaluate.yaml
  play.yaml
  collect_demonstrations.yaml

data/
  datasets originales
```

No modificar `src/` para facilitar el entrenamiento. Si se detecta un bloqueo real, documentarlo primero en un reporte y crear preferentemente un adapter nuevo en `training/`.

## 2.3 Partes permitidas

Se permite crear o modificar:

```text
training/
models/
scripts/
artifacts/
reports/
configs/evaluate_option_*.yaml
configs/train_option_*.yaml
policies/template.py
```

`policies/template.py` debe contener solo inferencia. No debe entrenar, explorar datasets ni escribir archivos.

---

# 3. Contrato común del agente final

Verifica en el código real, pero se espera una interfaz equivalente a:

```python
class StudentAgent:
    def __init__(self, config=None):
        ...

    def reset(self):
        ...

    def act(self, obs):
        return action_index
```

Acciones esperadas:

```text
0 = north/up
1 = south/down
2 = east/right
3 = west/left
4 = stay
5 = interact
```

La observación actual suele contener:

```python
obs["obs"]          # vector, normalmente shape (96,)
obs["agent_index"]  # 0 o 1
```

No asumirlo sin verificarlo mediante una ejecución real. La observación usada durante entrenamiento debe coincidir exactamente con la disponible durante evaluación.

El agente entregado debe:

- devolver un entero entre 0 y 5;
- limpiar memoria temporal en `reset()`;
- no depender de rutas absolutas;
- no escribir archivos durante inferencia;
- cargar pesos una sola vez en `__init__`;
- mantener latencia p95 claramente inferior al límite oficial, normalmente cercano a 100 ms;
- funcionar en CPU aunque se haya entrenado en GPU.

---

# 4. Fundamento técnico de las dos opciones

Las opciones son complementarias, no duplicadas.

## Opción [A]: recurrente y centrada en demostraciones humanas

**Nombre recomendado:** `Human-Aware Recurrent Behavioral Cloning`.

Objetivo: aprovechar la gran colección multi-grupo de demostraciones humanas y aprender una política recurrente capaz de usar el historial para inferir patrones del compañero.

Inspiración académica:

- Behavioral Cloning y human-proxy policies usadas en human-AI coordination.
- Uso de recurrencia para integrar historia de observaciones.
- Conclusión de OvercookedV2: muchas fallas en Overcooked original provienen de mala cobertura de estados; por ello se añaden rollouts, role swaps y estados visitados por políticas imperfectas.

Esta no es una reproducción exacta de OvercookedV2: nuestro starter usa una observación featurizada de aproximadamente 96 dimensiones, no necesariamente el grid multicanal del paper. La adaptación debe declararse explícitamente.

## Opción [B]: actor-critic y entrenamiento FCP-style contra una población

**Nombre recomendado:** `BC-Warmstarted FCP-style PPO`.

Objetivo: partir de una policy supervisada razonable y después optimizarla on-policy contra partners variados y congelados.

Inspiración académica:

- Fictitious Co-Play: población de partners formada por agentes entrenados con distintas seeds y checkpoints históricos.
- ZSC-Eval: evaluación con partners diversos, parallel partner sampling, centralized critic cuando sea posible y entropy decay.
- COLE-inspired sampling: aumentar la frecuencia de partners contra los cuales el ego-agent obtiene peor rendimiento.
- OvercookedV2: ampliar cobertura de estados mediante rollouts variados, seeds, roles y partners diferentes.

Esta opción debe describirse como `FCP-style` o `FCP-inspired` salvo que replique realmente todas las etapas y presupuestos del paper. No afirmar que es FCP exacto si se usa una población reducida o un número menor de steps.

---

# 5. Arquitectura común de carpetas

Codex debe adaptar los nombres a la estructura real, pero el resultado debe aproximarse a:

```text
dl-overcooked/
  src/                              # existente, no tocar
  data/                             # original, no sobrescribir

  policies/
    template.py                     # agente final y selector de modo
    basic_policies.py               # existente, no tocar

  training/
    datasets.py                     # loader común
    splits.py                       # splits por episodio/grupo/layout
    normalization.py                # estadísticas solo de train
    losses.py                       # weighted CE/focal opcional
    train_option_a_gru_bc.py
    train_option_b_bc_warmstart.py
    train_option_b_fcp_ppo.py
    partner_pool.py
    rollout_buffer.py
    evaluate_checkpoints.py

  models/
    common.py
    option_a_gru_policy.py
    option_b_actor_critic.py
    ensemble.py

  scripts/
    audit_datasets.py
    build_split_manifest.py
    collect_teacher_rollouts.py
    run_evaluation_matrix.py
    export_policy.py
    benchmark_inference.py

  configs/
    train_option_a.yaml
    train_option_b.yaml
    evaluate_option_a.yaml
    evaluate_option_b.yaml
    evaluate_ensemble.yaml
    evaluate_final.yaml

  artifacts/
    shared/
      normalization.json
      split_manifest.json
      dataset_schema.json
    option_a/
      best_checkpoint.pt
      final_policy.npz
      final_policy_config.json
    option_b/
      bc_warmstart.pt
      best_checkpoint.pt
      partner_pool_manifest.json
      final_policy.npz
      final_policy_config.json
    ensemble/
      ensemble_config.json

  reports/
    dataset_inventory.csv
    dataset_summary.md
    split_summary.md
    option_a_training.csv
    option_a_evaluation.csv
    option_a_report.md
    option_b_training.csv
    option_b_evaluation.csv
    option_b_report.md
    ensemble_evaluation.csv
    final_comparison.md
    implementation_deviations.md
```

No crear archivos vacíos solo por cumplir esta estructura. Cada archivo creado debe tener una responsabilidad real.

---

# 6. Fase común obligatoria antes de [A] o [B]

Ambas ramas deben ejecutar primero la misma fase común. Si otra persona ya la implementó correctamente, reutilizarla sin duplicarla.

## 6.1 Verificación del starter

Ejecutar una evaluación mínima con agentes existentes y guardar:

- comando usado;
- layout;
- partner;
- seed;
- role;
- estructura exacta de `obs`;
- mapping real de acciones;
- límite de tiempo por acción;
- salida generada por el runner.

Crear:

```text
reports/starter_smoke_test.md
```

No continuar si el starter no ejecuta una partida completa.

## 6.2 Auditoría recursiva del dataset

El loader debe buscar de forma recursiva:

```python
Path("data").rglob("*.npz")
```

No asumir nombres como `grabaciones/`, porque existen variantes como:

```text
grabaciones/
Grabaciones/
demonstrations/
demostrations/
data/
20 grabaciones/
20 grabaciones completas/
```

Reglas:

1. `.npz` es la fuente principal para entrenamiento.
2. `.metadata.json` aporta layout, partner, seed, role y contexto.
3. `.pkl` se usa solo como respaldo si falta información.
4. No contar `.npz` y `.pkl` del mismo episodio como dos partidas.
5. No modificar ni normalizar archivos dentro de `data/`.
6. No asumir que todos los episodios tienen 250 pasos ni obs de 96 sin validarlo.

Crear `reports/dataset_inventory.csv` con al menos:

```text
episode_id
source_group
source_path
stem
format
obs_shape
num_steps
action_min
action_max
layout
layout_type
partner
seed
agent_index
role_swap
reward_sum
num_deliveries
first_delivery_t
last_delivery_t
stay_ratio
interact_ratio
has_npz
has_pkl
has_metadata_json
has_layout_file
quality_flags
usable_for_bc
usable_for_environment_rollout
```

No asumir que `reward / 20 = sopas`; obtener entregas desde summaries o eventos reales si existen. Si no existe una forma confiable, marcar `num_deliveries` como desconocido.

## 6.3 Clasificación de layouts y episodios

Clasificar cada episodio:

```text
Tier 1: layout built-in reconocido; sirve para BC y rollouts.
Tier 2: layout custom con archivo .layout válido; sirve para BC y rollouts.
Tier 3: layout custom sin .layout; sirve para BC, pero no para reproducir el entorno.
Tier 4: archivo incompatible, corrupto o con mapping desconocido; cuarentena.
```

No descartar automáticamente episodios con cero sopas. Pueden contener navegación, bloqueos y estados de recuperación. Deben recibir menor peso, no necesariamente peso cero.

## 6.4 Deduplicación

Deduplicar por:

1. misma ruta base/stem;
2. metadata de episode id y seed;
3. hash de `obs + actions` cuando sea necesario.

Guardar duplicados detectados en:

```text
reports/dataset_duplicates.csv
```

## 6.5 Splits sin fuga de información

Nunca dividir por timestep. La unidad es el episodio completo.

Crear un manifest reproducible con seed fija:

```text
artifacts/shared/split_manifest.json
```

Debe incluir cuatro evaluaciones:

```text
train:
  grupos y layouts usados para optimización

validation_seen_layout:
  grupos/personas no vistos, layouts vistos

validation_unseen_layout:
  layouts completos no usados en train

validation_combined:
  grupos no vistos + layouts no vistos
```

Además, reservar un `internal_test` que no se use para elegir hiperparámetros.

Después de seleccionar arquitectura e hiperparámetros, se permite un retraining final con más datos, pero se debe conservar un test final independiente.

## 6.6 Normalización

Calcular `mean/std` solo con observaciones de train. Guardar:

```text
artifacts/shared/normalization.json
```

Aplicar un mínimo epsilon a desviaciones cercanas a cero. No normalizar IDs categóricos como `agent_index` con los mismos estadísticos de las features continuas.

## 6.7 Sampling balanceado

El dataset colectivo puede estar dominado por ciertos grupos, layouts, partners o por la acción `stay`.

Implementar al menos:

- muestreo balanceado por grupo;
- muestreo balanceado por layout;
- muestreo balanceado por partner cuando exista metadata;
- class weights o focal loss para evitar colapso a `stay`;
- pesos de calidad por episodio con límites máximos y mínimos.

Ejemplo inicial, ajustable tras auditoría:

```text
>= 3 entregas: peso 1.50
2 entregas:    peso 1.25
1 entrega:     peso 1.00
0 entregas:    peso 0.40
unknown:       peso 0.75
```

No permitir que un solo grupo o una sola partida de alto score domine el entrenamiento.

## 6.8 Baseline mínimo compartido

Antes de las opciones avanzadas, entrenar un MLP-BC pequeño como prueba del pipeline:

```text
input: obs normalizada + agent_index one-hot
MLP: 128 -> 128
output: 6 logits
loss: weighted cross entropy
```

Este baseline no es la entrega final. Sirve para confirmar:

- carga de datos;
- mapping de acciones;
- guardado y carga de checkpoints;
- integración con `StudentAgent`;
- evaluación end-to-end.

Debe ser capaz de sobreajustar un mini-dataset deliberadamente. Si no puede, hay un error de pipeline.

---

# 7. Opción [A]: Human-Aware Recurrent Behavioral Cloning

Codex debe ejecutar esta sección únicamente cuando se solicite opción `[A]`.

## 7.1 Hipótesis

Una observación aislada puede ser ambigua: la acción apropiada depende de lo que el compañero viene haciendo. Una GRU puede conservar historia y distinguir, por ejemplo:

- partner que ayuda con ingredientes;
- partner que se mueve de forma aleatoria;
- partner con sticky actions;
- partner que permanece quieto;
- cambio de rol durante el episodio.

La opción [A] se apoya principalmente en demostraciones humanas de muchos grupos y en cobertura de estados, no en un entrenamiento RL largo.

## 7.2 Entradas disponibles

Usar únicamente información reproducible en evaluación:

```text
obs actual normalizada
agent_index one-hot
acción propia anterior one-hot
flag de inicio de episodio
contador de timestep normalizado, solo si StudentAgent puede mantenerlo internamente
```

No usar layout ID, partner ID, reward futuro, acción real del compañero o metadata que no esté disponible durante deployment.

## 7.3 Arquitectura recomendada

Versión inicial:

```text
obs_dim ~= 96
+ agent_index one-hot: 2
+ previous_action one-hot: 6
+ start flag: 1

Linear(input_dim, 256)
LayerNorm(256)
GELU
Dropout(0.10)
Linear(256, 128)
GELU
GRU(input_size=128, hidden_size=128, num_layers=1)
LayerNorm(128)
Action head: Linear(128, 6)
```

Variantes permitidas para ablation:

```text
hidden_size: 64, 128, 256
sequence_length: 16, 32, 64
1 o 2 capas GRU
Dropout: 0.0 a 0.2
```

No hacer una búsqueda enorme. Priorizar pocas variantes bien evaluadas.

## 7.4 Construcción de secuencias

No mezclar pasos de episodios diferentes.

Crear ventanas preservando orden temporal:

```text
[episode t=0..31]
[episode t=16..47]
...
```

Aplicar padding y mask solo cuando sea necesario. El hidden state debe reiniciarse en:

- inicio de episodio;
- `done=True`;
- llamada a `StudentAgent.reset()`;
- role swap que implique un nuevo episodio.

## 7.5 Loss

Loss principal:

```text
weighted cross entropy(action_logits, human_action)
```

Combinar:

```text
peso_total = peso_clase * peso_episodio * peso_balance_grupo_layout_partner
```

Opcionales, solo si mejoran validación:

- label smoothing pequeño: 0.01–0.05;
- focal loss;
- auxiliary head para predecir `next_reward_event` si puede construirse sin leakage.

No optimizar accuracy global como métrica principal.

## 7.6 Cobertura de estados y datos teacher

La conclusión práctica de OvercookedV2 es que la mala cobertura de estados puede causar fallas fuertes en cross-play. Aplicar esa idea de forma compatible:

1. Entrenar `A0 = GRU-BC human-only`.
2. Ejecutar A0 en layouts reproducibles contra `stay`, `random_motion` y `greedy_full_task`.
3. Guardar estados visitados donde ocurra al menos una condición:
   - cero sopas;
   - repetición excesiva de acciones;
   - muchos `stay` consecutivos;
   - loops detectados;
   - baja confianza del modelo;
   - choque/bloqueo persistente, si el entorno lo expone.
4. Si `greedy_full_task` puede consultar el mismo estado durante recolección, guardar su acción como label teacher.
5. Marcar claramente `source=human` o `source=greedy_teacher`.
6. Reentrenar con teacher batches limitados inicialmente a 20–35%.

Esto es `DAgger-lite` o `teacher relabeling`, no afirmar que es DAgger exacto si no se consulta al teacher en cada estado online.

Si el greedy requiere raw state/MDP y no puede ejecutarse sobre estados guardados, recolectar rollouts completos del greedy mediante el runner oficial. No inventar labels a partir del vector de 96 features.

## 7.7 Experimentos mínimos de [A]

Ejecutar:

```text
A0: MLP-BC baseline
A1: GRU-BC human-only
A2: GRU-BC + balance de clase/grupo/layout
A3: GRU-BC + teacher/state-coverage, si es técnicamente posible
```

Seleccionar por evaluación en entorno, no por loss.

## 7.8 Archivos esperados de [A]

```text
models/option_a_gru_policy.py
training/train_option_a_gru_bc.py
configs/train_option_a.yaml
configs/evaluate_option_a.yaml
artifacts/option_a/best_checkpoint.pt
artifacts/option_a/final_policy.npz
artifacts/option_a/final_policy_config.json
reports/option_a_training.csv
reports/option_a_evaluation.csv
reports/option_a_report.md
```

`final_policy_config.json` debe incluir:

```json
{
  "option": "A",
  "model_type": "gru_bc",
  "obs_dim": 96,
  "num_actions": 6,
  "agent_index_encoding": "one_hot",
  "previous_action": true,
  "hidden_size": 128,
  "num_layers": 1,
  "normalization_path": "artifacts/shared/normalization.json",
  "action_mapping": {
    "0": "north",
    "1": "south",
    "2": "east",
    "3": "west",
    "4": "stay",
    "5": "interact"
  }
}
```

Adaptar valores a lo realmente entrenado.

## 7.9 Criterio de aceptación de [A]

[A] se considera válida solo si:

- supera o iguala al MLP-BC en sopas promedio;
- reduce episodios de cero sopas en validation_combined;
- funciona en ambos agent indices;
- `reset()` elimina correctamente el hidden state;
- inferencia CPU p95 cumple holgadamente el límite;
- genera evaluación reproducible con seeds fijas.

---

# 8. Opción [B]: BC-Warmstarted FCP-style PPO

Codex debe ejecutar esta sección únicamente cuando se solicite opción `[B]`.

## 8.1 Hipótesis

Behavioral Cloning proporciona una inicialización útil, pero sufre distribution shift: un pequeño error lleva a estados no vistos. PPO permite optimizar directamente el rendimiento en el entorno. Entrenar contra una población diversa reduce la dependencia de un único compañero.

## 8.2 Etapa B0: warm-start supervisado

Entrenar un actor MLP con los datos humanos:

```text
input = stack de las últimas K observaciones normalizadas
      + agent_index one-hot
      + previous_action one-hot

K inicial: 4
MLP: 512 -> 256 -> 128
LayerNorm + ReLU/GELU
Actor head: 128 -> 6 logits
```

El stack temporal permite contexto corto sin depender de una GRU, produciendo un modelo complementario a [A].

Guardar:

```text
artifacts/option_b/bc_warmstart.pt
```

## 8.3 Actor-critic

Durante PPO:

```text
actor:
  usa únicamente información disponible al StudentAgent

critic:
  preferentemente centralized critic con estado completo durante entrenamiento,
  solo si el environment adapter lo expone sin modificar el contrato de evaluación
```

Si no hay estado global accesible, usar critic sobre la misma observación del actor y documentar la desviación.

El critic nunca debe ser necesario en `policies/template.py`.

## 8.4 Construcción de la partner pool

Partner pool mínima:

```text
stay
random_motion
greedy_full_task
greedy_full_task + random_action_prob bajo
greedy_full_task + random_action_prob medio
greedy_full_task + random_action_prob alto
greedy_full_task + sticky actions en niveles disponibles
```

Verificar wrappers reales. No crear ruido duplicado si ya existe.

Partner pool FCP-style adicional:

1. Entrenar varias policies con seeds diferentes en self-play o co-play.
2. Guardar checkpoints tempranos, medios y finales.
3. Congelar esos checkpoints.
4. Añadirlos a la pool como partners.

Presupuesto escalado inicial:

```text
4 seeds
3 checkpoints por seed: early, mid, final
12 partners aprendidos
```

Si el tiempo es insuficiente:

```text
3 seeds x 2 checkpoints = 6 partners
```

No reducir la diversidad a varias copias del mismo checkpoint.

## 8.5 Human proxies opcionales

Si la infraestructura lo permite sin bloquear el avance, entrenar 2–4 proxies simples mediante BC:

```text
proxy autosuficiente
proxy colaborativo
proxy lento/pasivo
proxy errático o de baja calidad
```

Los clusters deben surgir de estadísticas observadas, no de nombres inventados. Si no hay suficiente señal para clusterizar, usar proxies por grupos de rendimiento y partner de recolección.

Los human proxies son partners congelados; no son el ego-agent final.

## 8.6 Environment adapter para PPO

Crear un adapter en `training/`, no modificar `src/runner.py` para convertirlo en trainer.

Debe permitir:

```text
reset(layout, seed, role, partner)
step(ego_action)
observation para actor
estado global opcional para critic
reward
terminated
truncated
info
```

Manejar correctamente time-limit truncation. No tratar un horizon truncado como terminal natural al calcular bootstrap returns.

Si el environment oficial no permite vectorización, implementar workers multiproceso o rollouts secuenciales primero. Priorizar corrección antes que throughput.

## 8.7 PPO recomendado

Configuración inicial, ajustable tras smoke tests:

```text
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
learning_rate: 3e-4 con warmup corto y decay
entropy_coef: iniciar 0.02–0.05 y decaer
value_coef: 0.5
max_grad_norm: 0.5
ppo_epochs: 4
minibatch_size: 256 o 512
rollout_steps totales por update: según throughput
```

No copiar hiperparámetros ciegamente. Registrar cada cambio.

Inicializar el actor con `bc_warmstart.pt`. El critic puede inicializarse aleatoriamente.

## 8.8 Sampling de partners

Primera etapa:

```text
uniform sampling entre categorías de partners
```

Luego implementar sampling adaptativo COLE-inspired:

1. Evaluar periódicamente score por partner.
2. Calcular una dificultad normalizada.
3. Aumentar moderadamente la probabilidad de partners de bajo score.
4. Mantener un mínimo de probabilidad para todos los partners.

Ejemplo:

```text
p_i = (1 - epsilon) * softmax(-beta * normalized_score_i)
    + epsilon / N
```

Valores iniciales:

```text
epsilon: 0.20
beta: 1.0
```

No permitir que el entrenamiento colapse a un único partner difícil.

## 8.9 Layout, seed y role sampling

Por episodio, muestrear:

```text
layout
partner
seed
agent_index/role
```

Balancear built-in y custom reproducibles. Los episodios Tier 3 sin `.layout` no pueden usarse para rollouts, aunque sí hayan servido en BC warm-start.

## 8.10 State coverage

Aplicar cobertura mediante:

- partners heterogéneos;
- varias seeds;
- role swap;
- noise/sticky actions;
- checkpoints históricos;
- rollouts on-policy del ego-agent;
- custom layouts reproducibles.

Implementar reset desde estados guardados únicamente si el entorno soporta serializar/restaurar estados con seguridad. Si no lo soporta, no modificar `src/environment.py` de forma riesgosa; la diversidad on-policy ya proporciona cobertura adicional.

## 8.11 Available action masks

ZSC-Eval reporta máscaras básicas para evitar golpes repetidos contra counters e interacciones nulas. En este proyecto:

- usar action masks solo si pueden derivarse correctamente de la información disponible;
- no prohibir acciones que podrían ser útiles para coordinación;
- evaluar con y sin máscara;
- documentar exactamente las reglas.

No convertir una heurística fuerte en un supuesto oculto del modelo.

## 8.12 Checkpoints históricos

Guardar snapshots del ego-agent a intervalos regulares. Algunos pueden añadirse a la pool como partners congelados para aumentar diversidad.

Mantener manifest:

```text
artifacts/option_b/partner_pool_manifest.json
```

Con:

```text
partner_id
type
seed
training_step
checkpoint_path
layouts_seen
source
skill_estimate
behavior_statistics
```

## 8.13 Experimentos mínimos de [B]

Ejecutar:

```text
B0: MLP-BC warm-start
B1: PPO self-play o contra un único greedy
B2: PPO contra partner pool uniforme
B3: PPO partner pool + adaptive sampling
B4: PPO partner pool + human proxies, si están disponibles
```

La comparación B1 vs B2 es obligatoria para demostrar el aporte de la población.

## 8.14 Archivos esperados de [B]

```text
models/option_b_actor_critic.py
training/train_option_b_bc_warmstart.py
training/train_option_b_fcp_ppo.py
training/partner_pool.py
configs/train_option_b.yaml
configs/evaluate_option_b.yaml
artifacts/option_b/bc_warmstart.pt
artifacts/option_b/best_checkpoint.pt
artifacts/option_b/partner_pool_manifest.json
artifacts/option_b/final_policy.npz
artifacts/option_b/final_policy_config.json
reports/option_b_training.csv
reports/option_b_evaluation.csv
reports/option_b_report.md
```

## 8.15 Criterio de aceptación de [B]

[B] se considera válida solo si:

- supera su propio BC warm-start en sopas promedio;
- partner-pool training supera o mejora la robustez de single-partner PPO;
- no colapsa contra `random_motion` o `stay`;
- mantiene rendimiento al cambiar agent index;
- no depende del critic en inferencia;
- cumple el límite de latencia en CPU;
- no produce timeouts técnicos.

---

# 9. Integración con `policies/template.py`

El mismo `StudentAgent` debe poder cargar distintos modos mediante config:

```yaml
config:
  mode: option_a
  model_config_path: artifacts/option_a/final_policy_config.json
  checkpoint_path: artifacts/option_a/final_policy.npz
```

```yaml
config:
  mode: option_b
  model_config_path: artifacts/option_b/final_policy_config.json
  checkpoint_path: artifacts/option_b/final_policy.npz
```

```yaml
config:
  mode: ensemble
  ensemble_config_path: artifacts/ensemble/ensemble_config.json
```

Responsabilidades de `StudentAgent`:

```text
__init__:
  leer config
  cargar normalización
  cargar pesos
  crear buffers temporales

reset:
  limpiar hidden state de A
  limpiar deque temporal de B
  limpiar previous_action
  reiniciar timestep interno

act:
  validar obs
  extraer vector y agent_index
  normalizar
  ejecutar forward
  elegir acción
  actualizar previous_action
  devolver int 0..5
```

No debe importar scripts de entrenamiento.

## Formato de checkpoint

Durante entrenamiento se permite `.pt`. Para entrega:

- verificar si PyTorch está garantizado en el evaluador;
- si no está garantizado, exportar pesos a `.npz` y usar inferencia NumPy;
- no usar ONNX salvo que la dependencia esté confirmada;
- guardar arquitectura y normalización en JSON.

---

# 10. Protocolo de evaluación obligatorio

## 10.1 Matriz

Evaluar cada modelo en:

```text
layouts reproducibles
x partners
x seeds
x agent_index/role
```

Partners mínimos:

```text
stay
random_motion
greedy_full_task
greedy + sticky
greedy + random actions
```

Durante desarrollo usar al menos 10 episodios por celda. Para candidatos finales, 30–50 episodios por celda cuando el tiempo lo permita.

## 10.2 Métricas

Guardar:

```text
mean_soups
median_soups
zero_soup_rate
official_score_mean
official_score_std
first_soup_t_mean
last_soup_t_mean
timeouts
invalid_actions
latency_mean_ms
latency_p95_ms
latency_max_ms
```

También guardar resultados desagregados por layout, partner, seed y role.

## 10.3 Selección robusta

Usar un ranking lexicográfico:

```text
1. mean_soups
2. -zero_soup_rate
3. official_score_mean
4. peor cuartil de score entre partners/layouts
5. latency_p95
```

El promedio puede esconder fallas completas. Reportar también:

- peor layout;
- peor partner;
- intervalos de confianza bootstrap cuando sea viable;
- cross-play entre modelos propios si se pueden usar como partners.

## 10.4 Score oficial

Implementar la fórmula oficial en un único helper probado mediante tests. No duplicarla en varios scripts.

Si no hay sopa:

```text
score = 0
```

La penalización debe tomarse del runner real; no inventar timeouts.

---

# 11. Modo [ENSEMBLE]

Ejecutar solo después de contar con artefactos válidos de [A] y [B].

## 11.1 Objetivo

[A] aporta historia larga y comportamiento humano. [B] aporta optimización on-policy y robustez frente a una población de partners. El ensemble solo se conserva si mejora evaluación real.

## 11.2 Interfaz común

Cada modelo debe exponer:

```python
logits = model.logits(obs, agent_index)
```

[A] mantiene hidden state GRU. [B] mantiene un deque de las últimas K observaciones.

## 11.3 Ensemble inicial

Usar promedio ponderado de logits calibrados:

```text
z = w * (logits_A / T_A) + (1 - w) * (logits_B / T_B)
action = argmax(z)
```

Calibrar `T_A`, `T_B` y `w` únicamente en validation, mediante una búsqueda pequeña:

```text
w ∈ {0.0, 0.1, ..., 1.0}
T_A, T_B ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
```

Seleccionar por sopas/score en entorno, no por NLL offline solamente.

## 11.4 Alternativas posteriores

Solo si average logits no mejora:

1. confidence gating;
2. gating model pequeño entrenado con validación;
3. reglas de desempate simples.

No usar gating por layout ID si este no estará disponible en layouts desconocidos.

No usar greedy como fallback en inferencia si `StudentAgent` no recibe raw state/MDP.

## 11.5 Criterio de aceptación del ensemble

Conservar ensemble únicamente si:

- supera al mejor individual en `mean_soups` o `official_score_mean`;
- no incrementa significativamente `zero_soup_rate`;
- mejora o mantiene el peor cuartil;
- p95 de latencia permanece bajo el límite;
- `reset()` reinicia correctamente ambos modelos.

Si no mejora, entregar el mejor modelo individual. Un ensemble no es automáticamente mejor.

---

# 12. Distribución recomendada del trabajo

Debido al tipo de cómputo:

```text
Opción [A]: ideal para Kaggle
- entrenamiento supervisado intensivo en GPU
- fácil de pausar y exportar
- pocas dependencias del simulador durante la mayor parte del proceso

Opción [B]: ideal para la máquina local persistente
- PPO genera muchos rollouts y puede ser CPU-bound
- requiere sesiones largas, depuración del entorno y checkpoints frecuentes
- se beneficia de control total del proceso y almacenamiento persistente
```

La asignación es recomendada, no obligatoria.

## Contrato de intercambio entre personas

Ambas ramas deben entregar:

```text
final_policy.npz
final_policy_config.json
normalization.json o referencia al común
evaluation.csv
report.md
commit hash o lista exacta de archivos modificados
comando exacto de entrenamiento
comando exacto de evaluación
```

No compartir solo un checkpoint sin su arquitectura y normalización.

---

# 13. Orden de ejecución y gates

## Gate 0 — Starter

```text
El juego ejecuta un episodio con políticas existentes.
```

## Gate 1 — Data

```text
Auditoría completa, deduplicación, split manifest y normalización.
```

## Gate 2 — Mini overfit

```text
El baseline puede sobreajustar un subconjunto minúsculo.
```

## Gate 3 — Integración

```text
StudentAgent carga un checkpoint y completa una evaluación.
```

## Gate 4A / 4B — Entrenamiento

```text
La opción correspondiente mejora respecto a su baseline.
```

## Gate 5 — Robustez

```text
Evaluación multi-layout, multi-partner, multi-seed y role swap.
```

## Gate 6 — Export y latencia

```text
Artefacto final portable, CPU-safe y sin timeouts.
```

## Gate 7 — Ensemble

```text
Solo después de aprobar A y B de forma independiente.
```

No saltar gates. Cuando un gate falla, corregir antes de añadir más complejidad.

---

# 14. Tests mínimos

Crear tests para:

```text
loader recursivo
deduplicación
split por episodio
normalización sin leakage
action mapping
shape de modelos
reset de GRU/deque
serialización y recarga
misma acción antes/después de export
score oficial
latencia
StudentAgent retorna int 0..5
```

Prueba crítica:

```text
cargar checkpoint -> correr episodio -> destruir agente -> recargar -> misma seed -> comportamiento reproducible
```

---

# 15. Reporte que Codex debe devolver

Codex no debe terminar diciendo únicamente “implementado”. Debe guardar y responder con:

## Para [A]

```text
reports/option_a_report.md
```

Debe contener:

- archivos creados/modificados;
- esquema real de los datos;
- arquitectura exacta;
- loss y sampling usados;
- cantidad de episodios/transiciones;
- splits;
- comando de entrenamiento;
- tiempo de entrenamiento y hardware;
- mejor checkpoint;
- tabla de evaluación;
- fallas observadas;
- diferencias respecto a los papers;
- próximos experimentos recomendados.

## Para [B]

```text
reports/option_b_report.md
```

Debe contener además:

- environment adapter;
- composición exacta de partner pool;
- número de steps PPO;
- política de sampling;
- centralized/decentralized critic;
- entropy schedule;
- throughput de rollouts;
- estabilidad del entrenamiento;
- comparación single-partner vs partner-pool.

## Para [ENSEMBLE]

```text
reports/final_comparison.md
```

Debe comparar:

```text
Greedy baseline
MLP-BC baseline
Option A
Option B
Ensemble A+B
```

Y recomendar un único agente final basándose en resultados medidos.

---

# 16. Reglas de honestidad técnica

1. No afirmar que se implementó FCP, HSP, MEP, COLE u OvercookedV2 de forma fiel si solo se tomó una idea.
2. Usar nombres como `FCP-style`, `COLE-inspired` o `OvercookedV2-inspired state coverage` cuando corresponda.
3. No atribuir mejoras a una técnica sin una ablation.
4. No utilizar data del test para ajustar pesos, thresholds o ensemble.
5. No ocultar episodios con cero sopas.
6. No seleccionar por una sola seed.
7. No reportar solo el mejor run; incluir dispersión.
8. No cambiar el starter oficial para obtener métricas artificialmente mejores.

---

# 17. Criterio final de entrega

La entrega final debe quedar así:

```text
configs/evaluate_final.yaml
policies/template.py
artifacts/final_policy.npz
artifacts/final_policy_config.json
```

Opcionalmente:

```text
artifacts/ensemble/ensemble_config.json
```

La entrega debe correr mediante el flujo oficial, por ejemplo:

```bash
python -m src.evaluate --config configs/evaluate_final.yaml
```

Antes de cerrar:

1. ejecutar desde un proceso limpio;
2. verificar rutas relativas;
3. probar CPU;
4. medir p95/max latency;
5. probar ambos roles;
6. probar al menos tres seeds;
7. confirmar que no se carga ningún dataset en inferencia;
8. confirmar que no se requiere conexión a internet;
9. guardar el reporte final y todos los comandos reproducibles.

---

# 18. Decisión recomendada

La estrategia de mayor probabilidad de éxito es mantener dos líneas realmente distintas:

```text
[A] GRU-BC humana + cobertura de estados/teacher
[B] BC warm-start + PPO contra partner pool FCP-style
```

Después:

```text
comparar A vs B
probar ensemble calibrado
entregar ensemble solo si mejora
```

Esto aprovecha:

- la gran cantidad de demostraciones humanas;
- la diversidad de grupos, layouts y partners;
- la capacidad de una política recurrente para interpretar historia;
- la corrección on-policy de PPO;
- la robustez de entrenar contra una población;
- la cobertura de estados señalada por OvercookedV2;
- el contrato estable de `StudentAgent` sin romper el starter.
