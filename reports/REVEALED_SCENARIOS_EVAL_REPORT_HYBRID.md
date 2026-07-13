# Revealed Scenarios Evaluation Report

No training was run. This evaluation uses the current policy as-is.

## Protocol

```text
policy = hybrid_official_score
seeds = 67..96 (30 episodes)
group_size = 3
role_swap = false
horizon = 250
sticky_action_prob = 0.10 where sticky is requested
random_action_prob = 0.15 where random actions are requested
```

## Summary

| Scenario | Layout | Noise | Mean soups | Score mean | Groups passed | Base points |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `asymmetric_advantages` | sticky=0.0, random=0.0 | 1.0000 | 12211.00 | 10/10 | 6 |
| 2 | `coordination_ring` | sticky=0.1, random=0.0 | 4.9000 | 49736.90 | 10/10 | 9 |
| 3 | `counter_circuit` | sticky=0.1, random=0.15 | 0.0000 | 0.00 | 0/10 | 0 |

## Groups

### Scenario 1 - `asymmetric_advantages`

| Group | Seeds | Soups | Mean soups | Mean score | Passed |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 67|68|69 | 1|1|1 | 1.0000 | 12211.00 | True |
| 2 | 70|71|72 | 1|1|1 | 1.0000 | 12211.00 | True |
| 3 | 73|74|75 | 1|1|1 | 1.0000 | 12211.00 | True |
| 4 | 76|77|78 | 1|1|1 | 1.0000 | 12211.00 | True |
| 5 | 79|80|81 | 1|1|1 | 1.0000 | 12211.00 | True |
| 6 | 82|83|84 | 1|1|1 | 1.0000 | 12211.00 | True |
| 7 | 85|86|87 | 1|1|1 | 1.0000 | 12211.00 | True |
| 8 | 88|89|90 | 1|1|1 | 1.0000 | 12211.00 | True |
| 9 | 91|92|93 | 1|1|1 | 1.0000 | 12211.00 | True |
| 10 | 94|95|96 | 1|1|1 | 1.0000 | 12211.00 | True |

### Scenario 2 - `coordination_ring`

| Group | Seeds | Soups | Mean soups | Mean score | Passed |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 67|68|69 | 3|6|8 | 5.6667 | 57453.67 | True |
| 2 | 70|71|72 | 6|5|6 | 5.6667 | 57442.00 | True |
| 3 | 73|74|75 | 3|8|4 | 5.0000 | 51008.67 | True |
| 4 | 76|77|78 | 8|3|1 | 4.0000 | 41184.67 | True |
| 5 | 79|80|81 | 8|0|0 | 2.6667 | 26738.67 | True |
| 6 | 82|83|84 | 0|6|6 | 4.0000 | 40478.00 | True |
| 7 | 85|86|87 | 6|6|8 | 6.6667 | 67240.33 | True |
| 8 | 88|89|90 | 3|7|7 | 5.6667 | 57277.33 | True |
| 9 | 91|92|93 | 7|3|7 | 5.6667 | 57346.67 | True |
| 10 | 94|95|96 | 2|4|6 | 4.0000 | 41199.00 | True |

### Scenario 3 - `counter_circuit`

| Group | Seeds | Soups | Mean soups | Mean score | Passed |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 67|68|69 | 0|0|0 | 0.0000 | 0.00 | False |
| 2 | 70|71|72 | 0|0|0 | 0.0000 | 0.00 | False |
| 3 | 73|74|75 | 0|0|0 | 0.0000 | 0.00 | False |
| 4 | 76|77|78 | 0|0|0 | 0.0000 | 0.00 | False |
| 5 | 79|80|81 | 0|0|0 | 0.0000 | 0.00 | False |
| 6 | 82|83|84 | 0|0|0 | 0.0000 | 0.00 | False |
| 7 | 85|86|87 | 0|0|0 | 0.0000 | 0.00 | False |
| 8 | 88|89|90 | 0|0|0 | 0.0000 | 0.00 | False |
| 9 | 91|92|93 | 0|0|0 | 0.0000 | 0.00 | False |
| 10 | 94|95|96 | 0|0|0 | 0.0000 | 0.00 | False |

