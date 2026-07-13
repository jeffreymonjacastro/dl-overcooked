# try_A_end

Snapshot minimo de la version A2.2 al cierre de la iteracion.

Contenido preservado:

- `artifacts/final/`: router final y pesos exportados (`a1_policy.npz`, `a2_policy.npz`) con configs.
- `artifacts/shared/normalization.json`: normalizacion requerida por las configs de los modelos.
- `configs/evaluate_final.yaml`: evaluacion smoke del agente final.
- `configs/evaluate_option_a2_router.yaml`: configuracion de matriz/router usada en la reevaluacion.
- `configs/evaluate_option_a.yaml`: baseline A1 usado para comparacion.
- `reports/FINAL_REEVALUATION_A2.md`: reporte principal para entregar/defender.
- `reports/evaluation_matrix_a1_extended.csv`: matriz extendida del baseline A1.
- `reports/evaluation_matrix_a2_router_extended.csv`: matriz extendida del router A2.2.
- `reports/option_a2_router_latency.json`: latencia medida del router.

Resumen tecnico:

- A2.2 router mejora a A1 bajo el protocolo ampliado, pero la mejora es pequena.
- El resultado no es robusto todavia; `forced_coordination` sigue siendo el fallo principal.
- Este paquete es suficiente para revisar la version final A2.2 sin conservar salidas pesadas o diagnosticos intermedios.

Para reproducir esta version desde la raiz del repo limpio hay que restaurar
primero el codigo archivado en `added_code/` sobre las mismas rutas relativas
del proyecto. En particular, `added_code/policies/template.py` contiene el
loader modificado que entiende `option_a2_router`.

Luego se puede ejecutar:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --config legacy\try_A_end\configs\evaluate_final.yaml
```
