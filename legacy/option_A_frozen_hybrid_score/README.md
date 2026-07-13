# Option A Frozen Hybrid Score Snapshot

Fecha: 2026-07-12

Este snapshot congela la version Option A/hybrid que alcanzo el objetivo de score:

- 3.2250 sopas promedio en matriz 5 seeds + role swap.
- 3.1944 sopas promedio en simulacion 3 seeds sin role swap.
- Baseline A2.2: 1.2500 sopas promedio.

Politica final:

```text
builtin: hybrid_official_score
config: configs/evaluate_final.yaml
```

Contenido:

- configs/: configs necesarias para correr final y comparar baselines.
- artifacts/final/: pesos A1/A2 preservados.
- artifacts/shared/normalization.json: normalizacion para A1/A2.
- code_overlay/: archivos de codigo que implementan/calculan esta version.
- reports/: evidencia final y matrices.
- CURRENT_STATUS_OPTION_A.md: resumen corto.

Comandos de prueba desde la raiz del repo:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.evaluate --config configs\evaluate_final.yaml
.\.venv\Scripts\python.exe scripts\run_evaluation_matrix.py --config configs\evaluate_hybrid_official.yaml --output reports\phase_hybrid_official_matrix.csv --episodes 5
```

No se incluye .venv porque es pesado y regenerable.
