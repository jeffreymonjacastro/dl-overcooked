# Official Score Calculation Examples

Formula:

```text
if soups == 0:
    score = 0
else:
    penalty = min(100 * timeouts, 5000)
    score = 10000 * soups
          + 10 * (horizon - last_soup_timestep)
          + (horizon - first_soup_timestep)
          - penalty
```

Example 1, zero soups:

```text
soups = 0
horizon = 250
score = 0
```

Example 2, one soup:

```text
soups = 1
horizon = 250
first_soup_timestep = 100
last_soup_timestep = 100
timeouts = 0
penalty = 0
score = 10000 * 1 + 10 * (250 - 100) + (250 - 100) - 0
score = 11650
```

Example 3, three soups with timeouts:

```text
soups = 3
horizon = 250
first_soup_timestep = 50
last_soup_timestep = 210
timeouts = 7
penalty = min(100 * 7, 5000) = 700
score = 10000 * 3 + 10 * (250 - 210) + (250 - 50) - 700
score = 29900
```

In this repo, `steps.csv` does not expose a dedicated soup-delivery event. The
validated proxy used by `scripts/score_official.py` is sparse positive reward:
Overcooked-AI gives `20` sparse reward per delivered soup, so `reward / 20`
counts deliveries at that timestep.
