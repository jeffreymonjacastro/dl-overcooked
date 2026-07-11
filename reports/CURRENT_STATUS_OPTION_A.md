# Estado actual de Option A

## Resumen corto

La opcion [A] fue implementada y entrenada, pero el resultado actual **no esta
listo para competir**.

El agente:

- carga correctamente desde `policies/template.py`;
- usa pesos exportados en `artifacts/final_policy.npz`;
- corre mediante el flujo oficial `src.evaluate`;
- tiene latencia segura;
- logra sopas en algunas combinaciones de layout/partner;
- falla completamente en otras combinaciones importantes.

La situacion no es que el codigo no funcione. El codigo funciona. El problema es
que la politica aprendida aun no generaliza bien.

## Que tenemos implementado

### Infraestructura

Se agrego una linea completa para Option A:

```text
scripts/audit_datasets.py
scripts/collect_teacher_rollouts.py
scripts/run_evaluation_matrix.py
scripts/benchmark_inference.py
training/datasets.py
training/train_option_a_gru_bc.py
models/option_a_gru_policy.py
```

Tambien se agregaron configs:

```text
configs/train_option_a.yaml
configs/train_option_a_mlp.yaml
configs/evaluate_option_a.yaml
configs/evaluate_option_a_mlp.yaml
configs/evaluate_final.yaml
```

Y se modifico:

```text
policies/template.py
```

No se modifico `src/` ni la data original.

## Entorno

Se creo un entorno local:

```text
.venv/
```

Se instalo PyTorch CUDA y se verifico la GPU:

```text
GPU: NVIDIA GeForce RTX 4080
torch: 2.11.0+cu128
CUDA: disponible
```

El entrenamiento uso `device: cuda`.

## Data y auditoria

Se auditaron los `.npz` de `data/` de forma recursiva.

Archivos generados:

```text
reports/dataset_inventory.csv
reports/dataset_duplicates.csv
reports/dataset_summary.md
artifacts/shared/split_manifest.json
artifacts/shared/normalization.json
artifacts/shared/dataset_schema.json
```

Splits actuales:

```text
train: 415 episodios
validation_seen_layout: 51 episodios
validation_unseen_layout: 12 episodios
validation_combined: 9 episodios
internal_test: 41 episodios
```

La normalizacion fue calculada solo con train.

## Modelos entrenados

### A0: MLP baseline

Modelo simple usado para confirmar que el pipeline funciona.

Artefactos:

```text
artifacts/option_a/best_mlp_checkpoint.pt
artifacts/option_a/mlp_policy.npz
artifacts/option_a/mlp_policy_config.json
```

Resultado:

```text
best_val_loss: 1.414982589689928
```

### A1: GRU-BC humana + teacher

Modelo recurrente principal de Option A.

Arquitectura:

```text
obs normalizada
+ agent_index one-hot
+ previous_action one-hot
+ start flag
-> MLP encoder
-> GRU hidden_size=128
-> action head 6 acciones
```

Artefactos:

```text
artifacts/option_a/best_checkpoint.pt
artifacts/option_a/final_policy.npz
artifacts/option_a/final_policy_config.json
artifacts/final_policy.npz
artifacts/final_policy_config.json
```

Entrenamiento:

```text
train_episodes: 475
val_episodes: 72
best_val_loss: 1.3776483345937784
device: cuda
```

Incluye:

```text
datos humanos + 60 rollouts teacher greedy
```

## Evaluacion

Se evaluo con una matriz compacta:

```text
3 layouts
x 4 partners
x 3 seeds
x role swap activado
```

Archivos:

```text
reports/option_a_evaluation.csv
reports/option_a_mlp_evaluation.csv
```

Comparacion promedio:

| Modelo | Mean soups proxy | Zero soup rate proxy | Score proxy |
|---|---:|---:|---:|
| A0 MLP | 0.1111 | 0.9444 | 1159.78 |
| A1 GRU + teacher | 1.1944 | 0.6806 | 12231.46 |

Conclusion:

```text
La GRU mejora claramente al MLP baseline.
Pero sigue fallando demasiado para considerarse agente final fuerte.
```

## Donde funciona

La GRU tuvo buen rendimiento en algunas celdas:

```text
cramped_room + random_motion:
  mean_soups_proxy: 5.0
  zero_soup_rate_proxy: 0.0

coordination_ring + greedy_full_task:
  mean_soups_proxy: 5.0
  zero_soup_rate_proxy: 0.0
```

Esto indica que:

```text
la integracion funciona;
la politica aprendio algo util;
el modelo puede producir secuencias que entregan sopas.
```

## Donde falla

Fallas importantes:

```text
forced_coordination: casi todo cero
coordination_ring + stay/random_motion: cero
cramped_room + deterministic greedy_full_task: cero en smoke final
```

Smoke final:

```bash
.venv\Scripts\python.exe -m src.evaluate --config configs\evaluate_final.yaml
```

Resultado:

```text
num_rollouts: 10
mean_return_sparse: 0.0
```

Esto es grave porque el config final por defecto no obtiene sopas.

## Latencia

Benchmark:

```text
latency_mean_ms: 6.2248
latency_p95_ms: 7.1659
latency_max_ms: 37.8408
```

Esto esta bien. El problema no es velocidad.

## Diagnostico

El proyecto esta en este estado:

```text
Pipeline: funciona
Entrenamiento GPU: funciona
Export a NumPy: funciona
StudentAgent: funciona
Evaluacion: funciona
Rendimiento: insuficiente
```

La causa probable no es una sola. Hay varias:

1. El BC aprende acciones humanas promedio, pero no necesariamente una politica
   que maximiza sopas.
2. Hay muchos estados donde quedarse quieto o moverse mal recibe mucho peso.
3. La data mezcla grupos, partners y layouts con calidades muy distintas.
4. La GRU fue entrenada pocas epocas.
5. Los rollouts teacher fueron pocos y algunos tambien obtuvieron cero.
6. Algunas celdas requieren coordinacion fuerte, no solo imitacion.

## Que significa esto

No estamos en cero. Tenemos una base real:

```text
datos auditados
splits
normalizacion
modelo recurrente
export final
runner oficial
reportes
latencia medida
matriz de evaluacion
```

Pero no hay que vender este checkpoint como solucion final. Es un primer agente
funcional, no un agente competitivo.

## Proximo paso recomendado

Prioridad alta:

```text
1. Revisar visualmente el comportamiento con rendering window.
2. Entrenar mas epocas la GRU.
3. Cambiar weighting para castigar menos las trayectorias malas.
4. Agregar mas teacher rollouts exitosos por layout.
5. Evaluar por layout/partner despues de cada cambio.
```

Prioridad media:

```text
6. Probar seq_len=64.
7. Probar hidden_size=256.
8. Entrenar solo con episodios de mayor calidad y comparar.
9. Separar modelos por tipo de partner o usar ensemble.
```

Si queremos una mejora rapida, atacaria asi:

```text
1. Crear config visual para mirar que hace el agente.
2. Generar teacher rollouts exitosos en forced_coordination y cramped_room.
3. Reentrenar GRU 30-50 epocas con early stopping.
4. Comparar contra el checkpoint actual.
```

## Comando para ver el agente

Actualmente no se abrio ventana porque los configs usan:

```yaml
rendering:
  mode: none
```

Para verlo hay que crear o usar un config con:

```yaml
rendering:
  mode: window
  fps: 5
```

Ejemplo futuro:

```bash
.venv\Scripts\python.exe -m src.evaluate --config configs\watch_option_a.yaml
```

## Veredicto

Estado actual:

```text
Implementacion completa: si
Entrenamiento terminado: si
GPU usada: si
Agente integrable: si
Agente competitivo: no todavia
Problema principal: rendimiento/robustez, no integracion
```

