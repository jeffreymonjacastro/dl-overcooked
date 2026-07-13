# Revealed Scenarios Evaluation Report

No training was run. This evaluation uses the current policy as-is.

## Protocol

```text
policy = greedy_human_model
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
| 2 | `coordination_ring` | sticky=0.1, random=0.0 | 4.6667 | 47562.43 | 10/10 | 9 |
| 3 | `counter_circuit` | sticky=0.1, random=0.15 | 0.0000 | 0.00 | 0/10 | 0 |

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
| 1 | 67|68|69 | 6|6|2 | 4.6667 | 47457.67 | True |
| 2 | 70|71|72 | 6|2|2 | 3.3333 | 34617.33 | True |
| 3 | 73|74|75 | 6|7|2 | 5.0000 | 50813.00 | True |
| 4 | 76|77|78 | 6|5|5 | 5.3333 | 53870.00 | True |
| 5 | 79|80|81 | 5|6|2 | 4.3333 | 44349.00 | True |
| 6 | 82|83|84 | 6|2|2 | 3.3333 | 34732.00 | True |
| 7 | 85|86|87 | 6|6|2 | 4.6667 | 47632.33 | True |
| 8 | 88|89|90 | 6|7|6 | 6.3333 | 63680.67 | True |
| 9 | 91|92|93 | 4|6|6 | 5.3333 | 54023.33 | True |
| 10 | 94|95|96 | 1|6|6 | 4.3333 | 44449.00 | True |

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

