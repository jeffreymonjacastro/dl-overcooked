# Competition Adaptation Changes Report

Fecha: 2026-07-13

## Proposito

Este reporte resume los cambios hechos para adaptar el agente a los escenarios
de competencia. El criterio principal fue score-first: primero asegurar sopas y
pasar los escenarios revelados, luego optimizar generalizacion.

La regla de trabajo para futuras fases es:

```text
No reemplazar lo que ya funciona en los tres escenarios revelados.
Agregar mejoras como rutas, wrappers o fallbacks especificos.
```

## Cambios principales

### 1. Politica portfolio score-first

Archivo:

```text
policies/score_first_portfolio_policy.py
```

Funcion:

```text
Seleccionar una estrategia segun layout y partner.
```

Rutas importantes:

```text
asymmetric_advantages + greedy_full_task -> GreedyHumanModel
counter_circuit + greedy_full_task -> RecipeAwareGreedyPolicy
forced_coordination -> adaptive_competition
cramped_room + random_motion -> hybrid
coordination_ring + random_motion -> greedy_local
default -> adaptive_competition
```

Impacto:

```text
Permite especializar sin destruir la ruta general.
Fue clave para pasar asymmetric_advantages y counter_circuit.
```

### 2. Soporte para acciones sticky

Archivo:

```text
src/policy_wrappers.py
```

Cambio:

```text
Se agrego StickyActionWrapper.
wrap_agent ahora acepta sticky_action_prob.
```

Impacto:

```text
Permite evaluar correctamente escenarios donde el companero tiene sticky actions.
Fue necesario para reproducir los escenarios 2 y 3.
```

### 3. Politica recipe-aware

Archivo:

```text
policies/basic_policies.py
```

Cambio:

```text
Se agrego RecipeAwareGreedyPolicy.
```

Funcion:

```text
Leer la receta del MDP y elegir ingredientes segun la orden real, no solo onion.
```

Impacto:

```text
Mejoro counter_circuit, que contiene recetas mixtas onion/tomato.
Evita depender de una politica onion-only en mapas multi-receta.
```

### 4. Registro de nuevas politicas

Archivo:

```text
src/policy_loader.py
```

Cambio:

```text
Registro de recipe_aware_greedy.
Registro de score_first_portfolio.
Registro de adaptive_competition y shortppo candidate.
```

Impacto:

```text
Permite correr las politicas desde YAML o scripts sin modificar el runner base.
```

### 5. Evaluador de escenarios revelados

Archivo:

```text
scripts/evaluate_revealed_scenarios.py
```

Funcion:

```text
Ejecutar exactamente los tres escenarios revelados.
Calcular sopas, score oficial, grupos aprobados y reportes CSV/MD.
```

Outputs:

```text
reports/REVEALED_SCENARIOS_EVAL_REPORT.md
reports/revealed_scenarios_results_score_first_portfolio.csv
reports/revealed_scenarios_groups_score_first_portfolio.csv
```

Impacto:

```text
Da una forma reproducible de demostrar el exito en competencia.
```

### 6. Score oficial centralizado

Archivo:

```text
scripts/score_official.py
```

Funcion:

```text
Calcular score por intento usando sopas, tiempos de entrega y penalizaciones.
```

Impacto:

```text
Evita que training, evaluacion y reportes usen metricas distintas.
```

### 7. Evaluador amplio de layouts

Archivo:

```text
scripts/evaluate_layout_sweep.py
```

Funcion:

```text
Evaluar la politica actual en muchos layouts con timeout por layout.
Separar resultados OK, LOW y omitidos/fallidos.
```

Outputs:

```text
reports/LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO.md
reports/layout_sweep_score_first_portfolio.csv
reports/layout_sweep_omitted.csv
```

Impacto:

```text
Sirve para diagnosticar generalizacion sin tocar la version exitosa.
```

## Cambios que no deben hacerse ahora

No se recomienda:

```text
Reemplazar score_first_portfolio por una politica nueva no validada.
Cambiar globalmente greedy_full_task a receta mixta si puede danar onion layouts.
Cambiar configs/evaluate_final.yaml sin una razon clara.
Borrar rutas especificas que hicieron pasar los escenarios revelados.
```

## Direccion de mejoras futuras

Las mejoras deben ser incrementales:

1. Agregar rutas por layout solo para mapas que hoy fallan.
2. Mejorar handoff en layouts de coordinacion estricta.
3. Extender recipe-aware a mapas tomate/multi-receta sin tocar las rutas
   reveladas.
4. Evitar planners gigantes o agregar timeout/fallback para mapas grandes.
5. Mantener tests y evaluacion revelada antes de aceptar cualquier cambio.

Checklist minimo antes de cambiar algo:

```text
python scripts/evaluate_revealed_scenarios.py --policy score_first_portfolio --seeds 67-96 --group-size 3
python -m unittest discover -s tests -v
```

Si una mejora baja algun escenario revelado, no debe promoverse.
