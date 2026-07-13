# Forced Coordination Blocker Audit

Fecha: 2026-07-12

## Resumen

El objetivo literal de llegar a `>= 2` sopas en todos los escenarios no queda
demostrado porque dos celdas siguen en cero:

```text
forced_coordination + random_motion = 0.0000
forced_coordination + stay = 0.0000
```

Esto no se debe a un fallo simple del router. Es una limitacion estructural del
layout con esos partners.

## Layout

`forced_coordination` separa los recursos en dos componentes caminables:

```text
[['X', 'X', 'X', 'P', 'X'],
 ['O', ' ', 'X', ' ', 'P'],
 ['O', ' ', 'X', ' ', 'X'],
 ['D', ' ', 'X', ' ', 'X'],
 ['X', 'X', 'X', 'S', 'X']]
```

Recursos:

```text
left side: onions, dishes
right side: pots, serve
handoff counters: middle X column
valid walkable positions:
  left = (1,1), (1,2), (1,3)
  right = (3,1), (3,2), (3,3)
```

Para producir sopa, algun agente del lado izquierdo debe poner ingredientes y
dishes en counters de handoff usando `INTERACT`. Si eso no ocurre, el agente del
lado derecho no puede obtener los objetos necesarios.

## Evidencia por partner

### forced_coordination + stay

`stay` nunca se mueve ni interactua:

```text
timestep 0..18:
action_0 = stay
action_1 = stay
reward = 0
```

Por tanto no hay handoff posible.

### forced_coordination + random_motion

`random_motion` solo emite movimientos y `stay`; no emite `INTERACT`.

Primeros pasos observados:

```text
action_1 = left, right, right, up, right, right, left, down, ...
reward = 0
```

Como no hay `INTERACT`, no puede recoger onions/dishes ni dejarlos en handoff.

### forced_coordination + greedy_full_task

El resultado promedio es `3.0` sopas por role swap:

```text
role_swap False: 0 sopas
role_swap True: 6 sopas
```

Cuando el agente propio queda en el lado cooker y el partner `greedy_full_task`
queda en el lado supplier, el partner toma una onion y queda bloqueado intentando
alcanzar un pot inaccesible. No deposita la onion en el handoff counter.

Episodios:

```text
seed 67: 0 / 6 sopas segun rol
seed 68: 0 / 6 sopas segun rol
seed 69: 0 / 6 sopas segun rol
```

## Busqueda de mejoras

Se evaluaron con el mismo protocolo:

```text
hybrid_official_score
adaptive_competition
adaptive_competition_shortppo
score_first_portfolio
greedy_full_task
greedy_human_model
```

Ninguno supera a `adaptive_competition` en las celdas bloqueadas:

```text
forced_coordination + random_motion = 0.0000
forced_coordination + stay = 0.0000
```

## Conclusion

Con el protocolo actual, el objetivo completo no es alcanzable solo modificando
el agente propio. Se necesita al menos una de estas condiciones externas:

```text
1. cambiar el partner para que pueda hacer handoff;
2. cambiar el protocolo para no exigir forced + stay/random_motion;
3. controlar ambos agentes;
4. modificar el layout.
```

Mientras esas condiciones no cambien, la mejor version recomendada sigue siendo:

```text
adaptive_competition
configs/evaluate_best_current.yaml
```
