# Backup v2: Kaggle v5 FCP-style PPO

Este backup congela la mejor corrida verificada de `overcooked-option-b-v2`: Kaggle version 5.

## Estructura

- `configs/evaluate_final.yaml`: evaluacion CPU apuntando a este backup.
- `configs/play_student.yaml`: juego interactivo humano vs agente v5.
- `policies/template.py`: `StudentAgent` con inferencia NumPy.
- `weights/`: pesos NumPy, normalizacion y checkpoints PyTorch.
- `results/`: CSV/JSON descargados de Kaggle v5.
- `reports/v5_summary.md`: resumen de resultados.

## Resultado

- `31/50` layouts evaluables alcanzan 3+ entregas al menos una vez.
- `mean_deliveries`: `0.9277777777777778`.
- `parity_ok`: `true`.

## Evaluar

Desde la raiz del repo:

```powershell
.venv\Scripts\python -m src.evaluate --config backups\v2\configs\evaluate_final.yaml
```

## Jugar con el agente

Desde la raiz del repo:

```powershell
.venv\Scripts\python -m src.run_game --config backups\v2\configs\play_student.yaml
```

Si el agente se queda demasiado pasivo en `agent_0`, prueba la variante con el agente como `agent_1`:

```powershell
.venv\Scripts\python -m src.run_game --config backups\v2\configs\play_student_as_agent1.yaml
```

Controles del humano (`agent_1`):

- Mover: flechas o `WASD`
- Interactuar: `Space`, `E` o `Enter`

Para cambiar escenario, edita `environment.layout_name` en `backups/v2/configs/play_student.yaml`.
