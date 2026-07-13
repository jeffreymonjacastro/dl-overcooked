# Low Layout Impossibility Evidence

Fecha: 2026-07-13

Este reporte complementa el sweep de `score_first_portfolio_v2`. No cambia
codigo ni resultados; documenta por que algunos layouts no llegaron a 3 sopas
con el partner fijo `greedy_full_task`.

## Resumen

| Layout | Resultado actual | Evidencia principal |
| --- | ---: | --- |
| `forced_coordination` | 0 sopas | Componentes desconectados: ingredientes/platos y ollas/servicio estan en lados distintos sin feature compartida. |
| `forced_coordination_tomato` | 0 sopas | Componentes desconectados: ingredientes/servicio y ollas/plato estan en lados distintos sin feature compartida. |
| `simple_tomato` | 0 sopas | El partner onion-only contamina la olla tomato-only cuando nuestro agente sale a buscar tomates. |
| `m_shaped_s` | 0 sopas | La sopa puede quedar lista, pero el partner bloquea la unica casilla hacia plato/servicio con cebolla en mano. |
| `long_cook_time` | 1 sopa | `cook_time=100`, una sola olla y horizonte 250; se adelanto la primera entrega a `t=133`, pero no alcanza para tres ciclos. |
| `small_corridor` | 1 sopa | Handoff estrecho; primera entrega a `t=149`, pero la segunda sopa valida no alcanza a cocinarse/entregarse dentro del horizonte. |

## Evidencia Estructural

### `forced_coordination`

Componentes caminables:

```text
componente izquierdo: [(1,1), (1,2), (1,3)]
componente derecho:   [(3,1), (3,2), (3,3)]
```

Acceso por componente:

```text
izquierdo: onion, dish
derecho: pot, serve
```

El layout no tiene una feature adyacente a ambos componentes que permita pasar
objetos entre lados. El partner fijo empieza en el lado izquierdo; nuestro
agente empieza en el derecho. Sin transferencia de ingredientes/platos, no hay
forma de completar una sopa.

### `forced_coordination_tomato`

Componentes caminables:

```text
componente izquierdo: [(1,1), (1,2), (1,3)]
componente derecho:   [(3,1), (3,2), (3,3)]
```

Acceso por componente:

```text
izquierdo: onion, tomato, serve
derecho: pot, dish
```

Tampoco hay feature compartida entre componentes. Aunque un lado tiene
ingredientes y servicio, el otro tiene olla y plato. Con el partner fijo, no hay
canal fisico para transferir objetos entre ambos lados.

## Evidencia De Probes

### `simple_tomato`

Probe manual:

```text
ruta: tomar tomate izquierdo -> poner tomate en olla -> repetir
estado observado: olla tomato/onion/tomato
resultado: 0 reward
```

La unica forma de impedir contaminacion es ocupar la casilla de olla `(2,1)`.
Pero si nuestro agente se queda alli, no puede salir a buscar los tomates
restantes. Cuando sale, el partner `greedy_full_task` entra y agrega cebolla.

### `m_shaped_s`

Probe manual:

```text
resultado parcial: olla llena/cocinada/lista
bloqueo: partner en (1,2) con cebolla
resultado: sin acceso a dish ni serve
```

El problema no es cocinar la sopa: eso si se puede lograr. El problema es que
el partner queda ocupando el unico pasillo hacia plato/servicio, y nuestro
agente queda del lado de la olla sin poder completar pickup/delivery.

## Busqueda Acotada

Tambien se ejecuto una busqueda de estados con el partner `greedy_full_task`
fijo. No se integro codigo porque no encontro ninguna primera entrega:

```text
m_shaped_s: 43,755 estados unicos sin reward
simple_tomato: 80,126 estados unicos sin reward
forced_coordination: 10 estados, agotado sin reward
forced_coordination_tomato: 640 estados, agotado sin reward
```

## Conclusion

No conviene cambiar globalmente `score_first_portfolio_v2` para estos casos:
las mejoras globales arriesgan romper los escenarios protegidos que ya pasan.
Los casos restantes requieren cambiar partner/layout/reglas, o una policy con
control de ambos agentes. Con el protocolo actual, la V2 debe conservar las
rutas parciales y el reporte debe declarar estas limitaciones.
