# Project General Status Report

Fecha: 2026-07-13

## Resumen ejecutivo

El proyecto ya tiene una version funcional para los tres escenarios revelados de
competencia. La politica recomendada para mostrar esos resultados es:

```text
score_first_portfolio
```

El punto mas importante es que esta version ya cumple los tres tasks pedidos.
Por eso, las mejoras futuras deben hacerse encima de esta base, con rutas
especificas o fallbacks nuevos, sin reemplazar la logica que ya funciona para:

```text
asymmetric_advantages
coordination_ring
counter_circuit
```

## Evidencia principal

Reporte principal de escenarios revelados:

```text
reports/REVEALED_SCENARIOS_EVAL_REPORT.md
```

Resultados:

| Escenario | Layout | Ruido | Sopas promedio | Score promedio | Grupos aprobados |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `asymmetric_advantages` | sin ruido | 5.9333 | 59754.27 | 10/10 |
| 2 | `coordination_ring` | sticky=0.10 | 4.9000 | 49736.90 | 10/10 |
| 3 | `counter_circuit` | sticky=0.10, random=0.15 | 5.3667 | 54149.23 | 10/10 |

Conclusion: la version actual no solo entrega una sopa; supera el minimo de los
escenarios revelados y mantiene margen alto.

## Barrido amplio de layouts

Tambien se hizo un sweep amplio sobre los layouts oficiales y algunos layouts de
`configs/data`:

```text
reports/LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO.md
reports/layout_sweep_score_first_portfolio.csv
reports/layout_sweep_omitted.csv
```

Resultado general:

```text
44 layouts evaluados
18 layouts >= 2 sopas promedio
26 layouts < 2 sopas promedio
6 layouts omitidos/fallidos/timeout
```

Layouts con buen desempeno relevante:

| Layout | Sopas promedio | Score promedio |
| --- | ---: | ---: |
| `asymmetric_advantages` | 5.900 | 59401.90 |
| `asymmetric_advantages_tomato` | 3.000 | 32079.00 |
| `coordination_ring` | 8.000 | 80258.00 |
| `counter_circuit` | 5.000 | 50955.00 |
| `cramped_room` | 6.200 | 62428.10 |
| `five_by_five` | 6.000 | 60384.00 |
| `scenario2` | 6.000 | 60399.00 |
| `scenario3` | 4.000 | 41035.00 |

## Causas probables de los casos bajos

Los layouts con bajo score no invalidan el exito en los tres escenarios de
competencia. Sirven para diagnosticar que la politica actual esta especializada
en resolver los layouts revelados y algunos layouts simples, pero no es todavia
un agente universal.

Principales causas probables:

1. Coordinacion estricta o handoff obligatorio.
   Layouts como `forced_coordination`, `soup_coordination`, `schelling`,
   `pipeline` y similares requieren entregas indirectas, bloqueos o decisiones
   de espera coordinada. La politica actual no tiene una estrategia robusta de
   handoff general para todos esos mapas.

2. Recetas tomate o multi-receta.
   Algunos layouts como `simple_tomato`, `cramped_room_tomato` o variantes con
   ordenes especiales fallan porque varias rutas base siguen siendo
   onion-focused. Se agrego una ruta recipe-aware para `counter_circuit`, pero
   no conviene cambiar globalmente todo a tomate/multi-receta porque podria
   danar los tres escenarios que ya pasan.

3. Layouts personalizados no alineados con la ruta actual.
   `chavez_room`, `jamcy_room`, `m_room` y `diagonal_run` aparecen como archivos
   validos, pero la politica actual no tiene reglas especificas para sus
   geometria, accesos, fuentes, ollas y puntos de entrega.

4. Costos de planner o mapas demasiado grandes.
   `corridor` y `you_shall_not_pass` hicieron timeout. `multiplayer_schelling`
   intento crear una matriz enorme de planner, por lo que no es practico con el
   enfoque actual.

5. Layouts incompatibles o configuracion interna invalida.
   `maze_kitchen` fallo por incompatibilidad de `recipe_values` con
   `<ingredient>_value`; `tutorial_1` fallo por configuracion de receta;
   `cramped_room_single` fallo por suposicion de dos agentes.

## Decision de proyecto

La version actual debe considerarse baseline congelado para competencia:

```text
policy = score_first_portfolio
config recomendada = configs/evaluate_best_current.yaml
script de escenarios = scripts/evaluate_revealed_scenarios.py
```

No se recomienda reemplazar la politica completa antes de entregar o mostrar.
Las mejoras deben ser incrementales, por ejemplo:

```text
if layout == nuevo_layout_problematico:
    usar ruta especializada
else:
    mantener score_first_portfolio actual
```

Esto protege los tres escenarios que ya funcionan.
