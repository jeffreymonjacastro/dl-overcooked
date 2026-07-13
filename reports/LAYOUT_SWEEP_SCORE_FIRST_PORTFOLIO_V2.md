# Layout Sweep - Score First Portfolio

No training was run. This evaluates the current policy as-is.

## Protocol

```text
policy = score_first_portfolio_v2
partner = greedy_full_task
seeds = 67..76 (10 episodes per completed layout)
horizon = 250
role_swap = false
noise = none
timeout_seconds_per_layout = 90
```

## Summary

| Layout | Source | Mean soups | Mean score | P(>=1 soup) | Zero rate | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `asymmetric_advantages` | builtin | 5.900 | 59401.90 | 1.00 | 0.00 | OK |
| `asymmetric_advantages_tomato` | builtin | 3.000 | 32079.00 | 1.00 | 0.00 | OK |
| `bonus_order_test` | builtin | 5.000 | 50579.00 | 1.00 | 0.00 | OK |
| `bottleneck` | builtin | 4.000 | 40347.00 | 1.00 | 0.00 | OK |
| `centre_objects` | builtin | 5.000 | 50505.00 | 1.00 | 0.00 | OK |
| `centre_pots` | builtin | 8.000 | 80526.00 | 1.00 | 0.00 | OK |
| `chavez_room` | file | 8.000 | 80733.00 | 1.00 | 0.00 | OK |
| `coordination_ring` | builtin | 8.000 | 80258.00 | 1.00 | 0.00 | OK |
| `counter_circuit` | builtin | 5.000 | 50955.00 | 1.00 | 0.00 | OK |
| `counter_circuit_o_1order` | builtin | 3.000 | 31066.00 | 1.00 | 0.00 | OK |
| `cramped_corridor` | builtin | 3.000 | 31298.00 | 1.00 | 0.00 | OK |
| `cramped_room` | builtin | 6.200 | 62428.10 | 1.00 | 0.00 | OK |
| `cramped_room_o_3orders` | builtin | 5.000 | 50450.00 | 1.00 | 0.00 | OK |
| `cramped_room_tomato` | builtin | 8.000 | 80569.00 | 1.00 | 0.00 | OK |
| `diagonal_run` | file | 16.000 | 160519.00 | 1.00 | 0.00 | OK |
| `five_by_five` | builtin | 6.000 | 60384.00 | 1.00 | 0.00 | OK |
| `forced_coordination` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `forced_coordination_tomato` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `inverse_marshmallow_experiment` | builtin | 18.000 | 180323.00 | 1.00 | 0.00 | OK |
| `jamcy_room` | file | 11.200 | 113088.00 | 1.00 | 0.00 | OK |
| `large_room` | builtin | 5.100 | 51423.60 | 1.00 | 0.00 | OK |
| `long_cook_time` | builtin | 1.000 | 11287.00 | 1.00 | 0.00 | LOW |
| `marshmallow_experiment` | builtin | 24.000 | 240737.00 | 1.00 | 0.00 | OK |
| `marshmallow_experiment_coordination` | builtin | 12.000 | 121036.00 | 1.00 | 0.00 | OK |
| `mdp_test` | builtin | 4.800 | 49888.80 | 1.00 | 0.00 | OK |
| `m_room` | file | 10.000 | 100205.60 | 1.00 | 0.00 | OK |
| `m_shaped_s` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `pipeline` | builtin | 6.000 | 60729.00 | 1.00 | 0.00 | OK |
| `scenario1_s` | builtin | 4.000 | 40347.00 | 1.00 | 0.00 | OK |
| `scenario2` | builtin | 6.000 | 60399.00 | 1.00 | 0.00 | OK |
| `scenario2_s` | builtin | 5.600 | 56372.30 | 1.00 | 0.00 | OK |
| `scenario3` | builtin | 4.000 | 41035.00 | 1.00 | 0.00 | OK |
| `scenario4` | builtin | 4.000 | 40537.00 | 1.00 | 0.00 | OK |
| `schelling` | builtin | 7.700 | 77313.10 | 1.00 | 0.00 | OK |
| `schelling_s` | builtin | 9.100 | 91325.50 | 1.00 | 0.00 | OK |
| `simple_o` | builtin | 12.000 | 120407.00 | 1.00 | 0.00 | OK |
| `simple_o_t` | builtin | 12.000 | 120407.00 | 1.00 | 0.00 | OK |
| `simple_tomato` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `small_corridor` | builtin | 1.000 | 11111.00 | 1.00 | 0.00 | LOW |
| `soup_coordination` | builtin | 14.000 | 140336.00 | 1.00 | 0.00 | OK |
| `tutorial_0` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `tutorial_2` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `tutorial_3` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `unident` | builtin | 5.900 | 59281.70 | 1.00 | 0.00 | OK |

## Layouts Below 2 Soups

| Layout | Mean soups | Mean score | Zero rate | Likely reason |
| --- | ---: | ---: | ---: | --- |
| `forced_coordination` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `forced_coordination_tomato` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `m_shaped_s` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `simple_tomato` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `long_cook_time` | 1.000 | 11287.00 | 0.00 | It delivers soups but too slowly; this layout needs layout-specific routing optimization. |
| `small_corridor` | 1.000 | 11111.00 | 0.00 | It delivers soups but too slowly; this layout needs layout-specific routing optimization. |

## Omitted Or Timed Out

| Layout | Reason | Path |
| --- | --- | --- |
| `corridor` | timeout after 90s; current planner/policy is too slow here | `` |
| `cramped_room_single` | IndexError: tuple index out of range | `` |
| `maze_kitchen` | ValueError: 'recipe_values' incompatible with '<ingredient>_value' | `C:\Users\USER\Desktop\DL_FINAL\dl-overcooked\configs\layouts\maze_kitchen.layout` |
| `multiplayer_schelling` | MemoryError: Unable to allocate 337. GiB for an array with shape (212520, 212520) and data type float64 | `` |
| `tutorial_1` | ValueError: Recipe class must be configured before recipes can be created | `` |
| `you_shall_not_pass` | timeout after 90s; current planner/policy is too slow here | `` |
