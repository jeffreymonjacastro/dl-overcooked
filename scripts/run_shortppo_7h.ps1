$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "C:\Users\USER\Desktop\DL_FINAL\dl-overcooked"

New-Item -ItemType Directory -Force -Path "reports\shortppo" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts\shortppo" | Out-Null

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\collect_shortppo_macro_dataset.py
.\.venv\Scripts\python.exe training\ppo_macro.py --config configs\train_macro_ppo_7h.yaml --steps 200
.\.venv\Scripts\python.exe scripts\evaluate_shortppo_candidate.py
