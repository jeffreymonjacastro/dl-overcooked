# Scenario 4 — Random Motion Evaluation Report

Fecha: 2026-07-13

## Protocolo

```text
layout_file = configs/layouts/scenario_4.layout
policy = score_first_portfolio_v2
partner = random_motion
horizon = 400
seeds = 67, 68, 69
role_swap = false
```

Configuración ejecutada:

```text
configs/evaluate_scenario_4.yaml
```

Comandos:

```powershell
uv run python -m src.evaluate --config configs/evaluate_scenario_4.yaml
uv run python scripts/score_official.py --output-dir outputs/scenario_4_random_motion_v2 --horizon 400
```

Con `random_motion`, el portfolio V2 no activa la ruta exacta
`Scenario4YieldPolicy`, que está reservada para `greedy_full_task`. La ruta
efectiva de esta evaluación es `adaptive_default`.

## Resultados por episodio

| Episodio | Seed | Sopas | Primera sopa | Última sopa | Score oficial |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 67 | 10 | 62 | 391 | 100428.00 |
| 1 | 68 | 10 | 62 | 375 | 100588.00 |
| 2 | 69 | 10 | 62 | 381 | 100528.00 |

## Promedios

```text
sopas promedio por episodio = 10.0000
score oficial promedio       = 100514.67
primera sopa promedio        = 62.00
última sopa promedio         = 382.33
episodios con cero sopas     = 0/3
```

El score por episodio se calculó con:

```text
10000 * soups
+ 10 * (400 - last_soup_timestep)
+ (400 - first_soup_timestep)
- min(100 * timeouts, 5000)
```

## Interpretación frente al criterio del profesor

El requisito mínimo era entregar al menos una sopa en promedio. El resultado
observado fue:

```text
10.0000 sopas promedio
```

Por lo tanto, esta ejecución supera el requisito mínimo asociado a 12 puntos.
Los puntajes de 14, 15 o 16 dependen del puesto comparativo contra los demás
equipos; ese puesto no puede determinarse únicamente con esta evaluación local.

## Veredicto

```text
PROMEDIO >= 1 SOPA: SÍ
RESULTADO EN LOS 3 SEEDS: 10, 10, 10 SOPAS
ZERO-RATE: 0%
```
