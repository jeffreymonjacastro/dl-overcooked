import json
import pathlib

def main():
    nb_path = pathlib.Path("kaggle/v1/input/main.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    fixed_layouts = False
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Find the cell that defines LAYOUTS for PPO
            if "LAYOUTS = ['cramped_room'" in source and "counter_circuit" in source:
                print("Found PPO layouts cell, updating...")
                # Re-define this cell to exclude counter_circuit
                cell["source"] = [
                    "if PPO_ENV_OK:\n",
                    "    # ---- Environment builder ----\n",
                    "    # Exclude counter_circuit because GreedyHumanModel does not support tomato recipes\n",
                    "    LAYOUTS = ['cramped_room', 'coordination_ring', 'forced_coordination', 'asymmetric_advantages']\n",
                    "    HORIZON = 250\n",
                    "\n",
                    "    def build_env(layout, seed=42):\n",
                    "        mdp = OvercookedGridworld.from_layout_name(layout)\n",
                    "        return OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)\n",
                    "\n",
                    "    def featurize(env, state, ai):\n",
                    "        return env.featurize_state_mdp(state)[ai].astype(np.float32)\n",
                    "\n",
                    "    class EpsGreedy:\n",
                    "        def __init__(self, base, eps, seed=42):\n",
                    "            self.base=base; self.eps=eps; self.rng=np.random.default_rng(seed)\n",
                    "        def action(self, s):\n",
                    "            if self.rng.random()<self.eps:\n",
                    "                return Action.INDEX_TO_ACTION[int(self.rng.integers(0,6))], {}\n",
                    "            return self.base.action(s)\n",
                    "        def reset(self): pass\n",
                    "        def set_agent_index(self, i): self.base.set_agent_index(i)\n",
                    "        def set_mdp(self, m): self.base.set_mdp(m)\n",
                    "\n",
                    "    def make_partner(ptype, env, seed=42):\n",
                    "        if ptype=='stay': return StayAgent()\n",
                    "        if ptype=='random': return RandomAgent(all_actions=True)\n",
                    "        base = GreedyHumanModel(env.mlam)\n",
                    "        if ptype=='greedy': return base\n",
                    "        eps = {'greedy10':0.1,'greedy25':0.25,'greedy40':0.40}.get(ptype,0.25)\n",
                    "        return EpsGreedy(base, eps, seed)\n",
                    "\n",
                    "    PTYPES = ['stay','random','greedy','greedy10','greedy25','greedy40']\n",
                    "    PSCORES = {p: [] for p in PTYPES}\n",
                    "\n",
                    "    def sample_partner_cole(eps=0.2, beta=1.0):\n",
                    "        avg = np.array([np.mean(PSCORES[p]) if PSCORES[p] else .5 for p in PTYPES])\n",
                    "        rng_s = avg.max()-avg.min()\n",
                    "        n = (avg-avg.min())/rng_s if rng_s>0 else np.full(len(avg),.5)\n",
                    "        x=-beta*n; x=x-x.max(); sm=np.exp(x)/np.exp(x).sum()\n",
                    "        probs=(1-eps)*sm+eps/len(PTYPES); probs/=probs.sum()\n",
                    "        return np.random.choice(PTYPES, p=probs)\n",
                    "\n",
                    "    print('PPO environment setup done')\n",
                    "    print(f'Partner pool: {PTYPES}')\n"
                ]
                fixed_layouts = True
                break
                
    if fixed_layouts:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully to exclude counter_circuit.")
    else:
        print("Error: PPO layouts cell not found!")

if __name__ == "__main__":
    main()
