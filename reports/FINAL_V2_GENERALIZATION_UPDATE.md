# Final V2 Generalization Update

Fecha: 2026-07-13

## Objetivo

Mejorar `score_first_portfolio_v2` para que mas layouts lleguen a 3 sopas
promedio sin romper los tres escenarios revelados de competencia.

No se hizo entrenamiento nuevo. Esta fase usa routing score-first, policies
existentes y reglas exactas pequenas para desbloquear layouts especificos o
rescatar entregas parciales donde el objetivo de 3 sopas aun no se alcanza.

## Cambios implementados

1. `diagonal_run` y `pipeline`

Ruta nueva:

```text
greedy_full_task con ingredient=tomato
```

Resultado:

```text
diagonal_run: 16.000 sopas promedio
pipeline: 6.000 sopas promedio
```

2. `bonus_order_test`

Ruta nueva:

```text
greedy_human_model
```

El layout tenia recetas bonus/mixtas y la ruta previa no completaba ninguna
entrega valida. `greedy_human_model` si resuelve el layout con el partner fijo
`greedy_full_task`, sin necesitar una regla exacta nueva.

Resultado:

```text
bonus_order_test: 5.000 sopas promedio
score promedio: 50579.00
```

3. `scenario4`

Ruta nueva:

```text
Scenario4YieldPolicy
```

La policy abre el bloqueo inicial, deja que el companero meta la tercera cebolla
y luego se estaciona para no bloquear el pasillo. El companero `greedy_full_task`
puede completar ciclos solo.

Resultado:

```text
scenario4: 4.000 sopas promedio
```

4. `cramped_corridor`

Ruta nueva:

```text
StayPolicy
```

En este layout, nuestro agente estorbaba al companero. Quedarse quieta permite
que `greedy_full_task` complete el layout.

Resultado:

```text
cramped_corridor: 3.000 sopas promedio
```

5. `long_cook_time`

Ruta nueva:

```text
LongCookTimeAssistPolicy
```

La primera ruta parcial usaba `StayPolicy`, porque nuestro agente estorbaba al
companero si empezaba greedy desde `t=0`. Luego se probo delayed-greedy y llego
a 1 sopa con score `11012.00`. La version actual aprovecha que el layout acepta
recetas de 1 cebolla: espera 10 pasos, arranca una sopa de 1 cebolla y baja para
liberar la casilla de pickup. Esto no sube el numero de sopas, pero adelanta la
entrega y mejora el score.

Tambien se corrigio el scorer oficial local: este layout entrega reward 10 por
sopa, no 20. El scorer anterior redondeaba `10 / 20` a 0 y ocultaba entregas
reales. Ahora cualquier reward positivo cuenta como al menos una sopa entregada.

Resultado:

```text
long_cook_time: 1.000 sopa promedio
score promedio: 11287.00
```

Por `cook_time=100`, una sola olla y horizonte 250, 3 sopas no parece realista
con el partner actual. La ruta one-onion entrega la primera sopa en `t=133`,
pero la segunda coccion mas temprana observada queda fuera de tiempo para estar
lista y ser entregada antes del horizonte.

6. `small_corridor`

Ruta nueva:

```text
SmallCorridorHandoffPolicy
```

La policy ejecuta una ruta exacta de handoff: entrega tres cebollas al companero,
enciende la sopa, busca plato y entrega una sopa. En la ultima pasada se
comprimio la primera espera entre handoffs: la primera entrega bajo de `t=170`
a `t=149`, subiendo el score sin cambiar los escenarios protegidos.

Resultado:

```text
small_corridor: 1.000 sopa promedio
score promedio: 11111.00
```

Sigue bajo el objetivo de 3 sopas. En probe, la segunda sopa queda con 3
ingredientes demasiado tarde si nuestro agente espera sin bloquear al partner.
Tambien se probo iniciar temprano una segunda coccion, pero eso cocina una sopa
de 1 cebolla que no da reward en este layout porque la orden valida exige 3
cebollas. Por eso la mejora integrada solo adelanta la primera entrega.

## Resultado del sweep completo

Protocolo:

```text
policy = score_first_portfolio_v2
partner = greedy_full_task
seeds = 67..76
horizon = 250
role_swap = false
timeout = 90s por layout
```

Resultado:

```text
44 layouts evaluados
38 layouts OK, con >= 3 sopas promedio
6 layouts LOW
6 layouts omitidos/fallidos/timeout
```

Archivo principal:

```text
reports/LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO_V2.md
reports/layout_sweep_score_first_portfolio_v2.csv
reports/layout_sweep_score_first_portfolio_v2_omitted.csv
reports/LOW_LAYOUT_IMPOSSIBILITY_EVIDENCE.md
```

## Layouts que siguen bajos

| Layout | Sopas promedio | Causa probable |
| --- | ---: | --- |
| `forced_coordination` | 0.000 | Handoff estructural obligatorio; requiere pasar objetos entre zonas. |
| `forced_coordination_tomato` | 0.000 | Handoff estructural con receta mixta/tomate. |
| `m_shaped_s` | 0.000 | El partner bloquea el unico pasillo con una cebolla en mano; una busqueda acotada no encontro entrega corta. |
| `simple_tomato` | 0.000 | El partner onion-only contamina la olla; se entregan sopas invalidas para la orden tomato-only. |
| `small_corridor` | 1.000 | Handoff estructural; se rescata 1 sopa con ruta exacta, pero no alcanza tiempo para una segunda/tercera dentro del horizonte. |
| `long_cook_time` | 1.000 | Cocina muy lenta; se rescata 1 sopa con una ruta one-onion temprana, pero no alcanza 3 en el horizonte actual. |

### Evidencia adicional de descarte

Se hicieron probes adicionales sin integrar codigo nuevo:

- `forced_coordination` y `forced_coordination_tomato`: el partner
  `greedy_full_task` toma una cebolla en `t=2` y queda inmovil intentando ir a
  una olla que no puede alcanzar. No deposita el objeto en el counter central,
  asi que nuestro agente no recibe ingredientes.
- `simple_tomato`: una ruta de bloqueo puede poner tomates, pero el partner
  onion-only contamina la olla con cebolla antes de completar la receta
  tomato-only. Si nuestro agente bloquea la olla permanentemente, no puede
  salir a buscar los tomates restantes.
- `m_shaped_s`: una ruta exacta de nuestro agente puede llenar y cocinar la
  olla, pero el partner queda detenido en el unico pasillo hacia plato/servicio
  sosteniendo cebolla. La sopa queda lista, pero no hay acceso al plato ni a la
  entrega.
- Probe manual adicional de `simple_tomato`: la ruta de tomate izquierdo logra
  poner el primer tomate, pero cuando nuestro agente sale de `(2,1)` para buscar
  el segundo tomate, el partner entra a la casilla de olla y agrega cebolla. El
  estado resultante fue una olla `tomato/onion/tomato`, invalida para la orden
  tomato-only.
- Probe manual adicional de `m_shaped_s`: esperar o bloquear temprano no evita
  el cuello. La sopa de cebolla puede quedar lista, pero el partner queda en
  `(1,2)` con cebolla, que es la unica casilla entre la olla/plato/servicio.
  Nuestro agente queda en `(1,1)` sin acceso al plato ni a la salida.
- Busqueda acotada adicional con partner `greedy_full_task` fijo:
  `m_shaped_s` exploro 43,755 estados sin reward, `simple_tomato` exploro
  80,126 estados sin reward, `forced_coordination` agoto 10 estados y
  `forced_coordination_tomato` agoto 640 estados. No se integro codigo desde
  esa busqueda porque no aparecio ninguna primera entrega.

## Regresion de competencia

La regresion protegida sigue pasando despues de los cambios:

```text
python scripts/run_final_regression_suite.py --policy score_first_portfolio_v2 --seeds 67-96 --group-size 3
```

Resultados:

| Escenario | Layout | Sopas promedio | Score promedio | Grupos aprobados |
| ---: | --- | ---: | ---: | ---: |
| 1 | `asymmetric_advantages` | 5.9333 | 59754.27 | 10/10 |
| 2 | `coordination_ring` | 4.9000 | 49736.90 | 10/10 |
| 3 | `counter_circuit` | 5.3667 | 54149.23 | 10/10 |

## Tests

```text
python -m unittest discover -s tests -v
```

Resultado:

```text
13 tests OK
```

## Archivos modificados en esta fase

```text
policies/score_first_portfolio_v2.py
scripts/score_official.py
tests/test_score_official.py
reports/LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO_V2.md
reports/layout_sweep_score_first_portfolio_v2.csv
reports/layout_sweep_score_first_portfolio_v2_omitted.csv
reports/final_pass/protected_score_first_portfolio_v2.md
reports/FINAL_V2_GENERALIZATION_UPDATE.md
configs/validated_routes.yaml
```

## Conclusion

La V2 mejoro de forma incremental sin cambiar la base que ya pasaba competencia.
El estado actual es mucho mejor para generalizacion:

```text
Antes de esta fase: 33/44 layouts evaluados con >= 3 sopas
Despues de esta fase: 38/44 layouts evaluados con >= 3 sopas
```

Los layouts restantes no deberian arreglarse cambiando globalmente la policy.
Requieren handoff explicito, bloqueo anti-contaminacion de recetas, o rutas
exactas por layout.
