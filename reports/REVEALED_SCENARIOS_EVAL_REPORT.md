# Revealed Scenarios Evaluation Report

No training was run. This evaluation uses the current policy as-is.

## Protocol

```text
policy = score_first_portfolio
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
| 1 | `asymmetric_advantages` | sticky=0.0, random=0.0 | 5.9333 | 59754.27 | 10/10 | 6 |
| 2 | `coordination_ring` | sticky=0.1, random=0.0 | 4.9000 | 49736.90 | 10/10 | 9 |
| 3 | `counter_circuit` | sticky=0.1, random=0.15 | 5.3667 | 54149.23 | 10/10 | 11 |

## Groups

### Scenario 1 - `asymmetric_advantages`

| Group | Seeds | Soups | Mean soups | Mean score | Passed |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 67|68|69 | 6|6|6 | 6.0000 | 60401.67 | True |
| 2 | 70|71|72 | 6|6|6 | 6.0000 | 60382.00 | True |
| 3 | 73|74|75 | 6|6|6 | 6.0000 | 60381.33 | True |
| 4 | 76|77|78 | 5|6|6 | 5.6667 | 57143.33 | True |
| 5 | 79|80|81 | 6|6|6 | 6.0000 | 60332.33 | True |
| 6 | 82|83|84 | 7|6|6 | 6.3333 | 63701.00 | True |
| 7 | 85|86|87 | 6|6|7 | 6.3333 | 63701.67 | True |
| 8 | 88|89|90 | 6|6|2 | 4.6667 | 47486.33 | True |
| 9 | 91|92|93 | 6|6|6 | 6.0000 | 60365.67 | True |
| 10 | 94|95|96 | 6|6|7 | 6.3333 | 63647.33 | True |

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
| 1 | 67|68|69 | 10|5|4 | 6.3333 | 63865.00 | True |
| 2 | 70|71|72 | 4|4|4 | 4.0000 | 40497.67 | True |
| 3 | 73|74|75 | 7|7|4 | 6.0000 | 60428.33 | True |
| 4 | 76|77|78 | 4|7|4 | 5.0000 | 50502.00 | True |
| 5 | 79|80|81 | 7|10|10 | 9.0000 | 90360.33 | True |
| 6 | 82|83|84 | 2|0|4 | 2.0000 | 20642.00 | True |
| 7 | 85|86|87 | 7|5|5 | 5.6667 | 57412.67 | True |
| 8 | 88|89|90 | 5|10|0 | 5.0000 | 50427.33 | True |
| 9 | 91|92|93 | 6|7|5 | 6.0000 | 60500.33 | True |
| 10 | 94|95|96 | 7|7|0 | 4.6667 | 46856.67 | True |

