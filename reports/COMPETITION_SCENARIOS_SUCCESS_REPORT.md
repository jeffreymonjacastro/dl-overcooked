# Competition Scenarios Success Report

Fecha: 2026-07-13

## Objetivo

Evaluar la version actual del agente contra los tres escenarios revelados de la
competencia, sin entrenamiento nuevo, usando la politica actual:

```text
score_first_portfolio
```

## Protocolo

```text
seeds = 67..96
episodes = 30 por escenario
group_size = 3
groups = 10 por escenario
horizon = 250
role_swap = false
partner = greedy_full_task
```

Ruido aplicado:

```text
Escenario 1: sin sticky, sin random
Escenario 2: sticky_action_prob = 0.10
Escenario 3: sticky_action_prob = 0.10, random_action_prob = 0.15
```

## Resultado por escenario

| Escenario | Layout | Condicion | Sopas promedio | Score promedio | Grupos aprobados | Resultado |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `asymmetric_advantages` | `greedy_full_task` | 5.9333 | 59754.27 | 10/10 | Exitoso |
| 2 | `coordination_ring` | `greedy_full_task` + sticky | 4.9000 | 49736.90 | 10/10 | Exitoso |
| 3 | `counter_circuit` | `greedy_full_task` + sticky + random | 5.3667 | 54149.23 | 10/10 | Exitoso |

## Interpretacion

El resultado es positivo porque los tres escenarios pasan todos sus grupos.
Ademas, el promedio de sopas queda muy por encima del minimo pedido:

```text
Escenario 1: supera ampliamente el requisito de entregar al menos una sopa.
Escenario 2: supera el requisito de dos sopas promedio por grupo.
Escenario 3: supera el requisito de dos sopas promedio por grupo.
```

El caso mas importante es `counter_circuit`, porque combina sticky actions,
random actions y recetas mixtas. La version anterior onion-only no era suficiente
para ese tipo de mapa, pero la ruta actual recipe-aware permite sostener el
score.

## Evidencia por grupos

Resumen de aprobacion:

```text
asymmetric_advantages: 10/10 grupos aprobados
coordination_ring: 10/10 grupos aprobados
counter_circuit: 10/10 grupos aprobados
```

Archivo fuente con detalle por seed y grupo:

```text
reports/REVEALED_SCENARIOS_EVAL_REPORT.md
reports/revealed_scenarios_results_score_first_portfolio.csv
reports/revealed_scenarios_groups_score_first_portfolio.csv
```

## Conclusion

Para los escenarios revelados, la version actual es apta para demostrar al
profesor. No se recomienda cambiar la politica base antes de la demostracion.
Cualquier mejora futura debe ser compatible con estas rutas exitosas y no debe
alterar el comportamiento que ya paso 10/10 grupos.
