# Guía de Desarrollo y Estado del Proyecto (AGENTS.md)

Este archivo sirve como el **manual de desarrollo y directrices de trabajo** para cualquier asistente de inteligencia artificial o desarrollador que trabaje en este repositorio.

---

## 1. Estado Actual del Proyecto (Option B)

El proyecto cuenta actualmente con la **Opción B (BC-Warmstarted FCP-Style PPO)** implementada, entrenada y evaluada con éxito:
* **Behavior Cloning (BC) Warmstart**: El modelo base fue pre-entrenado durante 30 épocas en los datos colectivos humanos (Tiers A y B), logrando una val loss de `1.132`.
* **FCP-style PPO**: La política se entrenó durante `150,000` pasos contra un pool de compañeros (`stay`, `random`, `greedy`, y variantes epsilon-greedy), logrando una tasa de éxito promedio de **3.95 sopas por episodio** (picos de 6 sopas) en Cramped Room.
* **Respaldo de Versión**: Se encuentra una versión independiente y 100% autoevaluable en `backups/v1/`.

---

## 2. Estructura de Almacenamiento Estándar

Para mantener el repositorio limpio y organizado, se deben respetar estrictamente las siguientes carpetas para guardar cada recurso:

* **`artifacts/`**: Almacena únicamente los pesos finales activos y las configuraciones que lee el agente en tiempo de ejecución:
  * `artifacts/final_policy.npz` (Pesos de inferencia NumPy).
  * `artifacts/final_policy_config.json` (Configuración del modelo).
  * `artifacts/shared/normalization.json` (Parámetros de normalización).
* **`reports/`**: Destinada exclusivamente a los **reportes de estado y análisis en formato Markdown (.md)** (ej. `reports/option_b_report.md`, `reports/final_comparison.md`).
* **`results/`** (o subcarpetas dentro de backups): Almacena los **archivos de datos crudos y métricas en formato CSV** (ej. csv de entrenamiento BC, csv de entrenamiento PPO, logs de evaluación).
* **`outputs/`**: Destinada a guardar los archivos de salida generados por el evaluador local `src/evaluate.py` (ej. `outputs/option_b_evaluation/episodes.csv` que detalla los resultados por seed y rol).
* **`backups/`**: Contiene checkpoints históricos congelados y autoevaluables. Cada subcarpeta (ej. `backups/v1/`) debe tener su propia estructura interna (`weights/`, `configs/`, `policies/`, `reports/`, `results/`).

---

## 3. Flujo de Trabajo con la Skill de Kaggle

El entrenamiento de modelos pesados de Deep Learning se realiza de manera remota en Kaggle utilizando aceleración por GPU. Para iterar en experimentos de entrenamiento, sigue este flujo secuencial:

### Paso 1: Modificar el compilador del Notebook
Toda la lógica de entrenamiento de modelos, datasets y preprocesamiento se escribe en archivos normales en la carpeta `training/` y `models/`. Luego, edita [scripts/build_notebook.py](file:///c:/Users/jeffr/GitHub/dl-overcooked/scripts/build_notebook.py) para definir qué celdas de código se inyectarán en el notebook final.

### Paso 2: Compilar el Jupyter Notebook
Genera el notebook `main.ipynb` compilando las celdas locales ejecutando:
```bash
python scripts/build_notebook.py
```
*Verifica que no haya SyntaxError en la sintaxis de las listas de celdas.*

### Paso 3: Push a Kaggle
Empuja la versión al kernel remoto de Kaggle forzando el uso de la GPU Nvidia T4:
```bash
kaggle kernels push -p kaggle\v1\input --accelerator NvidiaTeslaT4
```

### Paso 4: Monitorear Ejecución y Logs
* Consulta el estado del kernel:
  ```bash
  kaggle kernels status jeffreyamc/overcooked-option-b-v1
  ```
* Extrae los logs en tiempo real para verificar errores o ver el progreso del entrenamiento:
  ```bash
  kaggle kernels logs jeffreyamc/overcooked-option-b-v1
  ```

### Paso 5: Descargar Outputs
Una vez el estado sea `COMPLETE`, descarga todos los pesos, csvs y jsons generados por el kernel en Kaggle:
```bash
kaggle kernels output jeffreyamc/overcooked-option-b-v1 -p kaggle\v1\output
```
*Después de descargar, mueve cada archivo a su carpeta correspondiente (`artifacts/`, `reports/` o `results/`).*

---

## 4. Tareas y Restricciones del Asistente de IA

Cuando el usuario solicite implementar mejoras o correr nuevos experimentos:

1. **Integridad de Código**:
   * No modifiques archivos en `src/` o los datasets crudos en `data/` a menos que se trate de un bug técnico confirmado por logs de error.
   * Preserva los comentarios y docstrings originales en `policies/template.py`.
2. **Prueba de Paridad**:
   * Al exportar un nuevo modelo `.npz`, siempre ejecuta una prueba comparativa para asegurar que la salida (logits) del paso forward de NumPy sea idéntica a la salida de PyTorch (diferencia máxima menor a $1 \times 10^{-5}$).
3. **Manejo de Incompatibilidades**:
   * Mantén los parches de compatibilidad de NumPy 2.x en el muestreador de acciones y de PyTorch 2.6 en la deserialización de checkpoints.
4. **Verificación Local**:
   * Nunca reportes que un modelo funciona sin haber corrido antes una evaluación en CPU usando:
     ```bash
     .venv\Scripts\python -m src.evaluate --config configs\evaluate_final.yaml
     ```
