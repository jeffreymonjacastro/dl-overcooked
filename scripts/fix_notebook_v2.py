import json
import pathlib

def main():
    nb_path = pathlib.Path("kaggle/v1/input/main.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    fixed_install = False
    fixed_ppo = False
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # 1. Update install cell: remove force-reinstall of numpy since we're monkeypatching it
            if "Reinstalling numpy" in source:
                print("Found install cell, updating...")
                cell["source"] = [
                    "progress['stage'] = 'install'; save_progress()\n",
                    "\n",
                    "def run_cmd(cmds, quiet=True):\n",
                    "    r = subprocess.run(cmds, capture_output=True, text=True)\n",
                    "    if not quiet or r.returncode != 0:\n",
                    "        if r.stdout: print('  OUT:', r.stdout[-300:])\n",
                    "        if r.stderr: print('  ERR:', r.stderr[-300:])\n",
                    "    return r.returncode == 0\n",
                    "\n",
                    "OVERCOOKED_INSTALLED = False\n",
                    "\n",
                    "# Attempt 1: PyPI install (relaxed dependencies)\n",
                    "print('[1] Trying PyPI install of overcooked-ai...')\n",
                    "if run_cmd(['pip', 'install', 'overcooked-ai', '--no-deps', '-q']):\n",
                    "    run_cmd(['pip', 'install', 'PyYAML>=6.0', 'scipy', 'dill', 'flask', '-q'])\n",
                    "    try:\n",
                    "        import numpy as np\n",
                    "        np.Inf = np.inf\n",
                    "        import overcooked_ai_py; OVERCOOKED_INSTALLED = True; print('  OK')\n",
                    "    except Exception as e: print(f'  Import failed: {e}')\n",
                    "\n",
                    "# Attempt 2: GitHub source\n",
                    "if not OVERCOOKED_INSTALLED:\n",
                    "    print('[2] Trying GitHub source...')\n",
                    "    if run_cmd(['pip', 'install', 'git+https://github.com/HumanCompatibleAI/overcooked_ai.git', '--no-deps', '-q']):\n",
                    "        run_cmd(['pip', 'install', 'PyYAML>=6.0', 'scipy', 'dill', '-q'])\n",
                    "        try:\n",
                    "            import numpy as np\n",
                    "            np.Inf = np.inf\n",
                    "            import overcooked_ai_py; OVERCOOKED_INSTALLED = True; print('  OK')\n",
                    "        except Exception as e: print(f'  Import failed: {e}')\n",
                    "\n",
                    "print(f'OVERCOOKED_INSTALLED: {OVERCOOKED_INSTALLED}')\n",
                    "progress['overcooked_available'] = OVERCOOKED_INSTALLED; save_progress()\n"
                ]
                fixed_install = True
            
            # 2. Update PPO import cell: add monkeypatch right before import
            if "from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv" in source:
                print("Found PPO import cell, updating...")
                cell["source"] = [
                    "progress['stage']='ppo_training'; save_progress()\n",
                    "\n",
                    "if OVERCOOKED_INSTALLED:\n",
                    "    try:\n",
                    "        import numpy as np\n",
                    "        np.Inf = np.inf\n",
                    "        np.bool = bool\n",
                    "        np.int = int\n",
                    "        np.float = float\n",
                    "        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv\n",
                    "        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld\n",
                    "        from overcooked_ai_py.agents.agent import GreedyHumanModel, RandomAgent, StayAgent\n",
                    "        from overcooked_ai_py.mdp.actions import Action\n",
                    "        PPO_ENV_OK = True\n",
                    "        print('overcooked_ai_py environment ready')\n",
                    "    except Exception as e:\n",
                    "        PPO_ENV_OK = False\n",
                    "        print(f'overcooked_ai_py import failed: {e}. PPO phase skipped.')\n",
                    "else:\n",
                    "    PPO_ENV_OK = False\n",
                    "    print('overcooked not installed. Skipping PPO phase.')\n",
                    "\n",
                    "progress['ppo_env_ok'] = PPO_ENV_OK; save_progress()\n"
                ]
                fixed_ppo = True
                
    if fixed_install and fixed_ppo:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully with NumPy monkeypatch.")
    else:
        print(f"Error: fixed_install={fixed_install}, fixed_ppo={fixed_ppo}")

if __name__ == "__main__":
    main()
