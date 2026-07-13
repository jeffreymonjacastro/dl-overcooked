# Layout Sweep - Score First Portfolio

No training was run. This evaluates the current policy as-is.

## Protocol

```text
policy = score_first_portfolio
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
| `bonus_order_test` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `bottleneck` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `centre_objects` | builtin | 1.000 | 12343.00 | 1.00 | 0.00 | LOW |
| `centre_pots` | builtin | 8.000 | 80526.00 | 1.00 | 0.00 | OK |
| `chavez_room` | file | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `coordination_ring` | builtin | 8.000 | 80258.00 | 1.00 | 0.00 | OK |
| `counter_circuit` | builtin | 5.000 | 50955.00 | 1.00 | 0.00 | OK |
| `counter_circuit_o_1order` | builtin | 3.000 | 31066.00 | 1.00 | 0.00 | OK |
| `cramped_corridor` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `cramped_room` | builtin | 6.200 | 62428.10 | 1.00 | 0.00 | OK |
| `cramped_room_o_3orders` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `cramped_room_tomato` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `diagonal_run` | file | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `five_by_five` | builtin | 6.000 | 60384.00 | 1.00 | 0.00 | OK |
| `forced_coordination` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `forced_coordination_tomato` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `inverse_marshmallow_experiment` | builtin | 18.000 | 180323.00 | 1.00 | 0.00 | OK |
| `jamcy_room` | file | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `large_room` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `long_cook_time` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `marshmallow_experiment` | builtin | 24.000 | 240737.00 | 1.00 | 0.00 | OK |
| `marshmallow_experiment_coordination` | builtin | 12.000 | 121036.00 | 1.00 | 0.00 | OK |
| `mdp_test` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `m_room` | file | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `m_shaped_s` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `pipeline` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `scenario1_s` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `scenario2` | builtin | 6.000 | 60399.00 | 1.00 | 0.00 | OK |
| `scenario2_s` | builtin | 1.000 | 12332.00 | 1.00 | 0.00 | LOW |
| `scenario3` | builtin | 4.000 | 41035.00 | 1.00 | 0.00 | OK |
| `scenario4` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `schelling` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `schelling_s` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `simple_o` | builtin | 12.000 | 120407.00 | 1.00 | 0.00 | OK |
| `simple_o_t` | builtin | 12.000 | 120407.00 | 1.00 | 0.00 | OK |
| `simple_tomato` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `small_corridor` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `soup_coordination` | builtin | 0.000 | 0.00 | 0.00 | 1.00 | LOW |
| `tutorial_0` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `tutorial_2` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `tutorial_3` | builtin | 18.000 | 180664.00 | 1.00 | 0.00 | OK |
| `unident` | builtin | 1.000 | 12200.00 | 1.00 | 0.00 | LOW |

## Layouts Below 2 Soups

| Layout | Mean soups | Mean score | Zero rate | Likely reason |
| --- | ---: | ---: | ---: | --- |
| `bonus_order_test` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `bottleneck` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `chavez_room` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `cramped_corridor` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `cramped_room_o_3orders` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `cramped_room_tomato` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `diagonal_run` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `forced_coordination` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `forced_coordination_tomato` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `jamcy_room` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `large_room` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `long_cook_time` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `m_room` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `m_shaped_s` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `mdp_test` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `pipeline` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `scenario1_s` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `scenario4` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `schelling` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `schelling_s` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `simple_tomato` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `small_corridor` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `soup_coordination` | 0.000 | 0.00 | 1.00 | 0 soups in every measured seed; current routing/partner pairing cannot complete this layout. |
| `centre_objects` | 1.000 | 12343.00 | 0.00 | It delivers soups but too slowly; this layout needs layout-specific routing optimization. |
| `scenario2_s` | 1.000 | 12332.00 | 0.00 | It delivers soups but too slowly; this layout needs layout-specific routing optimization. |
| `unident` | 1.000 | 12200.00 | 0.00 | It delivers soups but too slowly; this layout needs layout-specific routing optimization. |

## Omitted Or Timed Out

| Layout | Reason | Path |
| --- | --- | --- |
| `corridor` | timeout after 90s; current planner/policy is too slow here | `` |
| `cramped_room_single` | IndexError: tuple index out of range | `` |
| `maze_kitchen` | ValueError: 'recipe_values' incompatible with '<ingredient>_value' | `C:\Users\USER\Desktop\DL_FINAL\dl-overcooked\configs\layouts\maze_kitchen.layout` |
| `multiplayer_schelling` | MemoryError: Unable to allocate 337. GiB for an array with shape (212520, 212520) and data type float64 | `` |
| `tutorial_1` | ValueError: Recipe class must be configured before recipes can be created | `` |
| `you_shall_not_pass` | timeout after 90s; current planner/policy is too slow here | `` |
