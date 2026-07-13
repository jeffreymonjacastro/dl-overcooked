# Low Layout Triage

| Layout | Class | Priority | Mean soups | Reason |
| --- | --- | ---: | ---: | --- |
| `bonus_order_test` | A_competition_like_and_solvable_recipe | 1 | 0.000 | Likely recipe/order mismatch or partner contamination. |
| `bottleneck` | A_competition_like_bottleneck | 1 | 0.000 | Likely deadlock, narrow corridor, or yield/replan issue. |
| `chavez_room` | A_competition_like_custom_geometry | 1 | 0.000 | Custom geometry not covered by current route table. |
| `cramped_corridor` | A_competition_like_bottleneck | 1 | 0.000 | Likely deadlock, narrow corridor, or yield/replan issue. |
| `cramped_room_o_3orders` | A_competition_like_and_solvable_recipe | 1 | 0.000 | Likely recipe/order mismatch or partner contamination. |
| `cramped_room_tomato` | A_competition_like_and_solvable_recipe | 1 | 0.000 | Likely recipe/order mismatch or partner contamination. |
| `diagonal_run` | A_competition_like_custom_geometry | 1 | 0.000 | Custom geometry not covered by current route table. |
| `jamcy_room` | A_competition_like_custom_geometry | 1 | 0.000 | Custom geometry not covered by current route table. |
| `large_room` | A_competition_like_custom_geometry | 1 | 0.000 | Custom geometry not covered by current route table. |
| `m_room` | A_competition_like_custom_geometry | 1 | 0.000 | Custom geometry not covered by current route table. |
| `m_shaped_s` | A_competition_like_bottleneck | 1 | 0.000 | Likely deadlock, narrow corridor, or yield/replan issue. |
| `scenario1_s` | A_competition_like_bottleneck | 1 | 0.000 | Likely deadlock, narrow corridor, or yield/replan issue. |
| `scenario4` | A_competition_like_bottleneck | 1 | 0.000 | Likely deadlock, narrow corridor, or yield/replan issue. |
| `simple_tomato` | A_competition_like_and_solvable_recipe | 1 | 0.000 | Likely recipe/order mismatch or partner contamination. |
| `centre_objects` | B_slow_but_working | 2 | 1.000 | Already delivers soups but below target; good candidate for specialist routing. |
| `scenario2_s` | B_slow_but_working | 2 | 1.000 | Already delivers soups but below target; good candidate for specialist routing. |
| `unident` | B_slow_but_working | 2 | 1.000 | Already delivers soups but below target; good candidate for specialist routing. |
| `forced_coordination` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `forced_coordination_tomato` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `pipeline` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `schelling` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `schelling_s` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `small_corridor` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `soup_coordination` | C_structurally_forced_handoff | 3 | 0.000 | Needs handoff, role symmetry, or partner cooperation. |
| `long_cook_time` | C_unknown_low_layout | 4 | 0.000 | Needs manual inspection before promotion. |
| `mdp_test` | C_unknown_low_layout | 4 | 0.000 | Needs manual inspection before promotion. |
| `corridor` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: timeout after 90s; current planner/policy is too slow here |
| `cramped_room_single` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: IndexError: tuple index out of range |
| `maze_kitchen` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: ValueError: 'recipe_values' incompatible with '<ingredient>_value' |
| `multiplayer_schelling` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: MemoryError: Unable to allocate 337. GiB for an array with shape (212520, 212520) and data type float64 |
| `tutorial_1` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: ValueError: Recipe class must be configured before recipes can be created |
| `you_shall_not_pass` | D_or_E_planner_pathological_or_invalid | 5 | 0.000 | Timeout, memory blow-up, invalid recipe config, or incompatible test layout. Detail: timeout after 90s; current planner/policy is too slow here |
