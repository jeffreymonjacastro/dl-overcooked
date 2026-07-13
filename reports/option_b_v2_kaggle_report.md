# Option B v2 Kaggle Report

Fecha local: 2026-07-12

## Resumen

Se ejecuto el notebook de `kaggle/v2` como kernel remoto para avanzar hacia el objetivo de lograr minimo 3 entregas por escenario detectado en `data/`.

- Kernel: `jeffreyamc/overcooked-option-b-v2`
- Ultima version remota ejecutada: version 6
- Estado remoto final: `COMPLETE`
- Outputs descargados en: `kaggle/v2/outputs/`
- Estado interno del notebook: `complete`
- Paridad PyTorch -> NumPy: `true`, error maximo `4.76837158203125e-07`

El objetivo final aun no se cumplio. La mejor evidencia actual es:

- Mejor corrida hasta ahora: version 5, con `31/50` layouts evaluados alcanzando 3+ entregas al menos una vez con algun partner evaluado.
- Ultima corrida: version 6, con `26/50` layouts evaluados alcanzando 3+ entregas al menos una vez.
- `0/50` layouts pasan el criterio estricto de minimo 3 en todas las evaluaciones.
- En la mejor corrida quedan `19/50` layouts evaluados que todavia no alcanzan 3 entregas ni una vez.
- `7/57` escenarios del catalogo no pasan preflight.

## Cambios implementados

- `training/partner_pool.py`: se agrego `NeuralPartner`; el pool base usa clones BC, no `GreedyHumanModel`.
- `training/layout_catalog.py`: se agrego un catalogo de layouts desde metadata de `data/`, con 57 escenarios y 528 archivos `.npz`.
- `training/datasets.py`: se corrigio la calidad de episodios para contar entregas reales por reward positivo, no `reward / 20`.
- `scripts/build_notebook.py`: se compila un notebook autocontenido para Kaggle con BC warm-start, PPO, export NumPy, paridad y evaluacion.
- `scripts/build_notebook.py`: se agrego `ScriptedGreedyPartner` para cebolla/tomate como teacher temporal.
- `scripts/build_notebook.py`: version 5 agrega `evaluate_policy_snapshot()` y selecciona checkpoint por cobertura evaluada, no solo por episodios vistos durante PPO.
- `configs/train_option_b.yaml`: usa `layout_source: all_from_data`, `layout_sampling: adaptive_coverage`, `scripted_partner_ingredients: [onion, tomato]`, `eval_checkpoint_freq_steps: 200000`, `layout_focus_ids` para los layouts faltantes y `shaped_reward_coef: 1.0`.
- `kaggle/v2/input/main.ipynb`: notebook compilado para el kernel remoto.

## Outputs descargados

Archivos principales:

- `bc_warmstart.pt`
- `best_checkpoint_by_soups.pt`
- `best_checkpoint_eval_summary.json`
- `train_checkpoint_by_coverage.pt`
- `final_policy.npz`
- `final_policy_config.json`
- `normalization.json`
- `layout_manifest.json`
- `option_b_bc_training.csv`
- `option_b_ppo_training.csv`
- `option_b_evaluation.csv`
- `scenario_delivery_audit.csv`
- `data_demonstration_delivery_audit.csv`
- `partner_pool_manifest.json`
- `parity_check.json`
- `run_summary.json`
- `overcooked-option-b-v2.log`

## Resultado remoto version 6

`run_summary.json` reporta:

- `bc_val_loss`: `1.1186353742772883`
- `valid_layout_count`: `50`
- `ppo_best_train_coverage3`: `13`
- `ppo_best_eval_pass3`: `0`
- `ppo_best_eval_reach3`: `25`
- `ppo_best_eval_mean_deliveries`: `0.7066666666666667`
- `parity_ok`: `true`
- `eval_summary.n_eps`: `900`
- `eval_summary.n_ok`: `900`
- `eval_summary.mean_deliveries`: `0.7022222222222222`
- `eval_summary.zero_delivery_rate`: `0.79`
- `eval_summary.layouts_reaching_min3_any_partner`: `26`
- `eval_summary.layouts_passing_min3`: `0`

Comparacion contra iteraciones previas:

| version | cambio principal | mean deliveries eval | layouts reaching 3 once | layouts passing strict min3 |
|---|---|---:|---:|---:|
| v3 | entregas reales + shaping 1.0 | 0.16 | n/a | 0 |
| v4 | BC clones + scripted teachers + adaptive coverage | 0.17 | 3 | 0 |
| v5 | checkpoint por evaluacion multi-partner | 0.928 | 31 | 0 |
| v6 | foco 8x en los 19 layouts faltantes | 0.702 | 26 | 0 |

## Auditoria por escenario

Archivo generado: `kaggle/v2/outputs/scenario_delivery_audit.csv`.

Resultado:

- Layouts evaluados: `50`
- Episodios OK: `900`
- Layouts con minimo >= 3 en todas las evaluaciones: `0`
- Layouts que alcanzaron 3 al menos una vez en version 6: `26`
- Layouts que no alcanzaron 3 ni una vez en version 6: `24`
- Errores de evaluacion: `0`

Escenarios evaluados que todavia no alcanzaban 3 entregas en la mejor corrida, version 5:

- `custom_easy_coop`
- `custom_hallway`
- `duelo_1v1`
- `guillermo_custom_02`
- `guillermo_custom_03`
- `guillermo_custom_04`
- `guillermo_custom_05`
- `mini_reto`
- `onion_hard_1`
- `salinas_custom_02`
- `salinas_custom_03`
- `tomato_easy_2`
- `tomato_hard_3`
- `asymmetric_advantages`
- `counter_circuit`
- `counter_circuit_o_1order`
- `cramped_corridor`
- `forced_coordination`
- `small_corridor`

Mejores layouts por promedio en la version 6:

| layout | mean | max | best partner |
|---|---:|---:|---|
| tomato_easy_1 | 2.000 | 6 | eval_scripted_tomato |
| soup_coordination | 1.722 | 5 | eval_scripted_tomato |
| esquina_l | 1.667 | 5 | eval_scripted_onion |
| onion_easy_1 | 1.667 | 5 | eval_scripted_onion |
| onion_easy_3 | 1.667 | 5 | eval_scripted_onion |
| simple_o | 1.556 | 8 | eval_scripted_onion |

## Layouts excluidos

Archivo fuente: `kaggle/v2/outputs/layout_manifest.json`.

El catalogo detecta 57 escenarios. El preflight de Overcooked-AI valido 50 y excluyo 7:

- `chavez_room`: `FileNotFoundError`
- `diagonal_run`: `FileNotFoundError`
- `jamcy_room`: `FileNotFoundError`
- `m_room`: `FileNotFoundError`
- `maze_kitchen`: `FileNotFoundError`
- `salinas_custom_04`: timeout de preflight
- `salinas_custom_05`: timeout de preflight

## Auditoria de demostraciones

Archivo generado: `kaggle/v2/outputs/data_demonstration_delivery_audit.csv`.

Resultado sobre `data/` agrupado por escenario del catalogo:

- Escenarios detectados: `57`
- Escenarios cuyo mejor archivo tiene menos de 3 entregas: `20`

Ejemplos:

- `guillermo_custom_02`: max `0`
- `guillermo_custom_03`: max `0`
- `guillermo_custom_04`: max `0`
- `guillermo_custom_05`: max `0`
- `counter_circuit`: max `1`
- `simple_tomato`: max `1`
- `forced_coordination`: max `2`
- `small_corridor`: max `2`
- `tutorial_2`: max `2`

## Diagnostico

La version 5 confirma que el enfoque con teacher scripted y checkpoint por evaluacion mueve la aguja: se paso de `3/50` a `31/50` layouts alcanzando 3 entregas al menos una vez.

La version 6 prueba que focalizar 8x los 19 layouts faltantes no mejora la cobertura global; bajo a `26/50`. Esto sugiere que no basta con muestrear mas esos layouts: varios necesitan un teacher o demostraciones mejores.

Lo que sigue impidiendo completar el objetivo:

- Varios layouts tienen geometria o coordinacion donde el teacher simple tampoco genera trayectorias suficientes.
- En 20 escenarios la data no contiene demostraciones con 3 entregas, por lo que BC/FCP parte con poca o ninguna senal exitosa.
- 7 escenarios no son reproducibles en el preflight actual.
- El criterio estricto de minimo 3 en todas las combinaciones de partner/rol/seed sigue en `0/50`.

## Siguiente iteracion recomendada

La siguiente version debe atacar los 19 layouts restantes directamente:

1. Crear un curriculum/filtro `target_layout_ids` con los 19 layouts que no alcanzan 3.
2. Entrenar una fase adicional solo sobre esos layouts, no sobre los 50.
3. Evaluar y guardar checkpoint por `layouts_reaching_min3_any_partner` sobre esos 19.
4. Para layouts como `guillermo_custom_02-05`, `counter_circuit`, `forced_coordination` y `cramped_corridor`, mejorar el teacher scripted o generar nuevas demostraciones exitosas.
5. Resolver los 7 layouts excluidos antes de declarar cobertura de todo `data/`.

## Estado del objetivo

No completado.

Evidencia:

- `kaggle/v2/outputs/run_summary.json`: notebook completo.
- `kaggle/v2/outputs/parity_check.json`: paridad correcta.
- `kaggle/v2/outputs/scenario_delivery_audit.csv`: 26 layouts alcanzan 3 alguna vez en la ultima corrida, 0 pasan el criterio estricto.
- `kaggle/v2/outputs/data_demonstration_delivery_audit.csv`: 20 escenarios no tienen demostracion con 3 entregas.
