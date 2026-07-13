# Reevaluacion final de correcciones A2

Fecha: 2026-07-11

## Veredicto

Las correcciones A2 estan implementadas, los artefactos finales cargan y el
agente A2.2 router supera a A1 bajo el mismo protocolo ampliado. Sin embargo,
la mejora es localizada y el agente no es todavia robusto para una evaluacion
desconocida: `forced_coordination` sigue sin resolverse.

## Validaciones ejecutadas

```text
Tests: 10/10 OK
GPU disponible: NVIDIA GeForce RTX 4080
Smoke final A2.2: 10 rollouts, mean_return_sparse = 10.0
Latencia router: p95 = 6.3656 ms, limite = 100 ms
```

La evaluacion ampliada uso el mismo protocolo para A1 y A2.2:

```text
3 layouts x 4 partners x 5 seeds x role swap = 120 rollouts por agente
Seeds: 67, 68, 69, 70, 71
```

## Comparacion justa

| Metrica | A1 baseline | A2.2 router | Cambio |
| --- | ---: | ---: | ---: |
| Sopas medias por celda | 1.1583 | 1.2667 | +0.1084 |
| Tasa media de cero sopas | 0.6917 | 0.6250 | -0.0667 |
| Celdas con cero sopas en todos los rollouts | 5/12 | 4/12 | -1 |

El router conserva A1 para todas las celdas excepto:

```text
cramped_room + greedy_full_task
cramped_room + greedy_full_task_noise_015
```

Por eso los cambios observados son exactamente en esas dos celdas:

| Celda | A1 sopas medias | A2.2 sopas medias | A1 cero sopas | A2.2 cero sopas |
| --- | ---: | ---: | ---: | ---: |
| cramped_room + greedy_full_task | 0.0 | 0.5 | 1.0 | 0.5 |
| cramped_room + greedy_full_task_noise_015 | 1.2 | 2.0 | 0.6 | 0.3 |

## Fallos que siguen siendo bloqueantes

```text
forced_coordination + stay: 0.0 sopas, 100% cero
forced_coordination + random_motion: 0.0 sopas, 100% cero
forced_coordination + greedy_full_task: 0.0 sopas, 100% cero
forced_coordination + greedy_full_task_noise_015: 0.2 sopas, 90% cero
coordination_ring + stay: 0.0 sopas, 100% cero
coordination_ring + random_motion: 0.0 sopas, 100% cero
```

Esto muestra que el problema ya no es de exportacion, estado recurrente ni
latencia. El agente carece de demostraciones exitosas y de una politica de
coordinacion especifica para los escenarios y partners anteriores.

## Correccion adicional aplicada

`configs/evaluate_option_a.yaml` mantenia rutas rotas hacia
`artifacts/option_a/`, que fueron trasladadas a respaldo. Se actualizaron sus
rutas para cargar el baseline preservado en:

```text
artifacts/option_a_baseline_before_a2/option_a/
```

Con ello A1 vuelve a ser evaluable de forma independiente.

## Recomendacion tecnica

No conviene entrenar mas epocas al GRU global actual. La siguiente iteracion
debe ser A2.3 con especialistas y seleccion online:

1. Generar y filtrar solamente trayectorias exitosas para `forced_coordination`
   y para `coordination_ring` con partners pasivos.
2. Entrenar un especialista por familia de fallo: `forced_coordination`,
   partner pasivo y partner codicioso con intercambio de rol.
3. Evaluar cada checkpoint durante entrenamiento con esta matriz online y
   seleccionarlo por sopas/zero-soup-rate, no solo por val_loss de imitacion.
4. Extender el router con layout, partner, agent_index y rol inicial.

PPO (opcion B) tiene sentido solo despues de obtener especialistas BC que ya
entreguen sopas en esos casos. De otro modo, PPO empezaria desde politicas que
no alcanzan recompensas y desperdiciaria mucho presupuesto de entrenamiento.

## Artefactos relevantes

```text
reports/evaluation_matrix_a1_extended.csv
reports/evaluation_matrix_a2_router_extended.csv
reports/FINAL_CORRECTION_REPORT_A2.md
artifacts/final/final_policy_config.json
configs/evaluate_final.yaml
```
