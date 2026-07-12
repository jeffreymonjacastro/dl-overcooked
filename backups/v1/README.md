# Backup v1: BC-Warmstarted FCP-Style PPO

Este directorio contiene todo lo necesario para ejecutar la versión entrenada de nuestro agente autónomo de la **Opción B (BC-Warmstarted FCP-Style PPO)** en Overcooked-AI.

---

## 1. Estructura de Archivos
* **`configs/`**:
  * `evaluate_final.yaml`: Archivo de configuración para evaluar el agente con las rutas relativas apuntando a esta carpeta de backup.
  * `train_option_b.yaml`: Archivo de configuración que contiene los hiperparámetros exactos utilizados durante el entrenamiento de PPO.
* **`policies/template.py`**: El código del agente (`StudentAgent`) con la arquitectura de inferencia optimizada en NumPy puro (mantiene un buffer de historial, normaliza las observaciones y realiza el paso forward).
* **`reports/`**:
  * `option_b_report.md`: Reporte científico detallado del entrenamiento de PPO, del pool de compañeros y las dinámicas.
  * `final_comparison.md`: Comparación cuantitativa de Option B frente a los baselines (stay, random, greedy).
  * `CURRENT_STATUS_OPTION_A.md`: Resumen histórico inicial de Option A.
* **`results/`**:
  * `option_b_evaluation.csv`: Resultados crudos detallados de la matriz de evaluación.
  * `option_b_bc_training.csv`: Métricas por época del entrenamiento Behavior Cloning (warmstart).
  * `option_b_ppo_training.csv`: Métricas por paso de la optimización con PPO.
* **`weights/`**:
  * `final_policy.npz`: Los pesos de la red neuronal exportados en formato NumPy.
  * `final_policy_config.json`: Archivo de configuración del modelo.
  * `normalization.json`: Estadísticas de normalización (media y desviación estándar) calculadas sobre el dataset de entrenamiento.
  * `best_checkpoint_by_soups.pt`: Checkpoint de PyTorch completo por si se desea reanudar el entrenamiento de PPO.

---

## 2. Lo que se implementó
1. **Behavior Cloning Warmstart**: Pre-entrenamos el modelo Actor-Critic ($512 \times 256 \times 128$) utilizando aprendizaje supervisado sobre demostraciones humanas de alta calidad (Tiers A y B), logrando una val loss de $1.132$ antes de iniciar PPO.
2. **Entrenamiento On-Policy (PPO)**: Corrimos $150,000$ pasos de PPO contra una población diversa de compañeros (**FCP-Style Partner Pool**) para aprender alineación robusta y evitar el comportamiento rígido de los agentes clásicos.
3. **Paso Forward en NumPy Puro**: Diseñamos el modelo para correr inferencia directa en CPU usando NumPy sin requerir dependencias pesadas de PyTorch, garantizando una latencia menor a $1\text{ ms}$ y eliminando penalizaciones por timeout.
4. **Optimizaciones Clave**:
   * **Entorno y Partner Caching**: Cacheamos todas las 24 combinaciones de mapas y planeadores al inicio, lo que aceleró el entrenamiento en **1000x** (reduciendo el tiempo a solo $6.3$ minutos).
   * **NumPy 2.x Action Mismatch Fix**: Solucionamos la incompatibilidad del muestreador del paquete `overcooked_ai_py` en NumPy 2.0+ mediante un monkeypatch en memoria.
   * **PyTorch 2.6 Weights Unpickler Fix**: Añadimos compatibilidad para la carga segura de checkpoints con variables complejas.

---

## 3. Resultados y Mejoras obtenidas
* **Cramped Room (Con Greedy Partner)**:
  * **Antes (BC)**: 2.5 sopas promedio.
  * **Después (PPO)**: **4.67 sopas promedio** (Máximo: **6.0 sopas**, score de **60,615**).
* **Cramped Room (Con Random Partner)**:
  * **Antes (BC)**: 0.0 sopas.
  * **Después (PPO)**: **1.67 sopas promedio** (Máximo: **4.0 sopas**), demostrando que el agente es capaz de terminar platos a pesar de que el compañero se mueva de forma caótica.
* **Coordination Ring (Con Greedy Partner)**:
  * **Antes (BC)**: 0.5 sopas promedio.
  * **Después (PPO)**: **1.17 sopas promedio** (Máximo: **4.0 sopas** en el Rol 0).

---

## 4. Instrucción exacta para ejecutar la evaluación
Para correr la evaluación utilizando esta versión de respaldo, ejecuta la siguiente línea de comando desde la raíz del proyecto:

```bash
.venv\Scripts\python -m src.evaluate --config backups\v1\configs\evaluate_final.yaml
```
