import json
import os
import pathlib
import sys
import textwrap

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.layout_catalog import build_layout_catalog

KAGGLE_VERSION = os.environ.get("KAGGLE_VERSION", "v1")
NB_PATH = ROOT / "kaggle" / KAGGLE_VERSION / "input" / "main.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def build() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "train_option_b.yaml").read_text(encoding="utf-8"))
    catalog = build_layout_catalog(ROOT / "data")
    for spec in catalog["specs"]:
        spec.pop("example_paths", None)
    catalog["data_root"] = "data"
    cfg_json = json.dumps(cfg, indent=2, sort_keys=True)
    catalog_json = json.dumps(catalog, indent=2, sort_keys=True)

    cells = [
        md(
            """
            # Option B: BC-Warmstarted FCP-style PPO with BC Clones

            PPO trains against neural BC clones and frozen PPO snapshots, with tomato layouts enabled.
            """
        ),
        code(
            f"""
            import csv, copy, json, pathlib, subprocess, time, traceback, zipfile
            from collections import deque
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader

            TRAIN_CFG = json.loads({json.dumps(cfg_json)})
            LAYOUT_CATALOG = json.loads({json.dumps(catalog_json)})
            OUTPUT_DIR = pathlib.Path('/kaggle/working')
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            progress = {{'status': 'running', 'stage': 'init', 'artifacts': []}}

            def save_progress():
                (OUTPUT_DIR / 'run_summary.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')

            save_progress()
            print(f'Device: {{DEVICE}}')
            if torch.cuda.is_available():
                print(f'GPU: {{torch.cuda.get_device_name(0)}}')
            print('Config total_steps:', TRAIN_CFG['ppo']['total_steps'])
            print('Catalog layouts:', LAYOUT_CATALOG['layout_count'], 'weights:', LAYOUT_CATALOG['weight_sum'])
            """
        ),
        md("## Instalar Overcooked-AI"),
        code(
            """
            progress['stage'] = 'install_overcooked'; save_progress()

            def run_cmd(cmd):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print('CMD failed:', ' '.join(cmd))
                    if r.stdout: print(r.stdout[-500:])
                    if r.stderr: print(r.stderr[-500:])
                return r.returncode == 0

            OVERCOOKED_INSTALLED = False
            for cmd in [
                ['pip', 'install', 'overcooked-ai', '--no-deps', '-q'],
                ['pip', 'install', 'git+https://github.com/HumanCompatibleAI/overcooked_ai.git', '--no-deps', '-q'],
            ]:
                if run_cmd(cmd):
                    run_cmd(['pip', 'install', 'PyYAML>=6.0', 'scipy', 'dill', 'flask', '-q'])
                    try:
                        np.Inf = np.inf
                        from overcooked_ai_py.mdp.actions import Action, Direction
                        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
                        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
                        OVERCOOKED_INSTALLED = True
                        break
                    except Exception as exc:
                        print('Overcooked import failed:', repr(exc))

            progress['overcooked_available'] = OVERCOOKED_INSTALLED
            save_progress()
            print('OVERCOOKED_INSTALLED:', OVERCOOKED_INSTALLED)
            """
        ),
        md("## Cargar demostraciones"),
        code(
            """
            progress['stage'] = 'load_data'; save_progress()

            for zip_path in pathlib.Path('/kaggle/input').rglob('*.zip'):
                print('Extracting:', zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall('/kaggle/working')

            candidates = [pathlib.Path('/kaggle/working/data'), pathlib.Path('/kaggle/working')]
            candidates += [p for p in pathlib.Path('/kaggle/input').glob('*') if p.is_dir()]
            DATA_DIR, npz_files = None, []
            for p in candidates:
                if p.exists():
                    found = sorted(p.rglob('*.npz'))
                    print(f'{p}: {len(found)} npz')
                    if len(found) > len(npz_files):
                        DATA_DIR, npz_files = p, found
            if not npz_files:
                raise FileNotFoundError('No .npz demonstration files found in Kaggle inputs')

            OBS_DIM = int(np.load(npz_files[0], allow_pickle=True)['obs'].shape[1])
            print('Using DATA_DIR:', DATA_DIR)
            print('Distinct npz paths:', len(npz_files), 'OBS_DIM:', OBS_DIM)

            def quality_tier(rewards, actions):
                deliveries = int((rewards > 0).sum())
                stay = float((actions == 4).mean()) if len(actions) else 1.0
                max_run = cur = 1
                for i in range(1, len(actions)):
                    cur = cur + 1 if actions[i] == actions[i - 1] else 1
                    max_run = max(max_run, cur)
                if deliveries >= 2: return 'A', 1.5
                if deliveries >= 1: return 'B', 1.0
                if int((actions == 5).sum()) > 5 and stay < 0.8 and max_run < 50: return 'C', 0.1
                if stay > 0.9 or max_run > 100: return 'D', 0.0
                return 'C', 0.4

            def load_episodes(files):
                episodes = []
                bad = 0
                for fp in sorted(files):
                    try:
                        d = np.load(fp, allow_pickle=True)
                        obs = d['obs'].astype(np.float32)
                        actions = d['actions'].astype(np.int64)
                        rewards = d.get('rewards', np.zeros(len(actions), dtype=np.float32)).astype(np.float32)
                        epids = d.get('episode_ids', np.zeros(len(obs), dtype=int))
                        agent_idxs = d.get('agent_indices', np.zeros(len(obs), dtype=int))
                        for eid in np.unique(epids):
                            mask = epids == eid
                            if int(mask.sum()) < 5:
                                continue
                            tier, weight = quality_tier(rewards[mask], actions[mask])
                            if tier == 'D':
                                continue
                            episodes.append({
                                'obs': obs[mask],
                                'actions': actions[mask],
                                'rewards': rewards[mask],
                                'agent_index': int(agent_idxs[mask][0]) if agent_idxs[mask].size else 0,
                                'quality_tier': tier,
                                'quality_weight': weight,
                                'deliveries': int((rewards[mask] > 0).sum()),
                                'source_path': str(fp),
                            })
                    except Exception as exc:
                        bad += 1
                        print('Bad npz:', fp, repr(exc))
                return episodes, bad

            episodes, bad_files = load_episodes(npz_files)
            if not episodes:
                raise RuntimeError('No usable episodes after quality filtering')
            tiers = {}
            for ep in episodes:
                tiers[ep['quality_tier']] = tiers.get(ep['quality_tier'], 0) + 1
            print('Episodes:', len(episodes), 'bad_files:', bad_files, 'tiers:', tiers)

            rng = np.random.default_rng(TRAIN_CFG['bc_warmstart']['seed'])
            idx = rng.permutation(len(episodes)).tolist()
            n_train = int(len(idx) * TRAIN_CFG['bc_warmstart']['train_frac'])
            n_val = int(len(idx) * TRAIN_CFG['bc_warmstart']['val_frac'])
            train_eps = [episodes[i] for i in idx[:n_train]]
            val_eps = [episodes[i] for i in idx[n_train:n_train + n_val]]

            all_train_obs = np.concatenate([e['obs'] for e in train_eps], axis=0)
            NORM_MEAN = all_train_obs.mean(axis=0).astype(np.float32)
            NORM_STD = np.maximum(all_train_obs.std(axis=0), 1e-8).astype(np.float32)
            (OUTPUT_DIR / 'normalization.json').write_text(
                json.dumps({'mean': NORM_MEAN.tolist(), 'std': NORM_STD.tolist()}, indent=2),
                encoding='utf-8',
            )
            progress['artifacts'].append('normalization.json'); save_progress()
            """
        ),
        md("## Modelo y dataloaders BC"),
        code(
            """
            class MLP(nn.Module):
                def __init__(self, layer_sizes, use_layer_norm=True, dropout=0.0):
                    super().__init__()
                    layers = []
                    for i in range(len(layer_sizes) - 1):
                        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                        if i != len(layer_sizes) - 2:
                            if use_layer_norm:
                                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
                            layers.append(nn.ReLU())
                            if dropout > 0:
                                layers.append(nn.Dropout(dropout))
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)

            class BCWarmstartActor(nn.Module):
                def __init__(self, obs_dim=96, k_stack=4, num_actions=6, hidden_sizes=(512, 256, 128), dropout=0.1):
                    super().__init__()
                    self.obs_dim = obs_dim; self.k_stack = k_stack; self.num_actions = num_actions
                    input_dim = k_stack * obs_dim + 2 + num_actions
                    self.encoder = MLP([input_dim] + list(hidden_sizes), dropout=dropout)
                    self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)
                    for m in self.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)
                    nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
                def forward(self, stack_obs, agent_index, prev_action):
                    ah = F.one_hot(agent_index.long(), 2).float()
                    ph = F.one_hot(prev_action.long(), self.num_actions).float()
                    return self.actor_head(self.encoder(torch.cat([stack_obs, ah, ph], dim=-1)))

            class ActorCritic(nn.Module):
                def __init__(self, obs_dim=96, k_stack=4, num_actions=6, hidden_sizes=(512, 256, 128), dropout=0.05):
                    super().__init__()
                    self.obs_dim = obs_dim; self.k_stack = k_stack; self.num_actions = num_actions
                    input_dim = k_stack * obs_dim + 2 + num_actions
                    layers = [input_dim] + list(hidden_sizes)
                    self.actor_encoder = MLP(layers, dropout=dropout)
                    self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)
                    self.critic_encoder = MLP(layers, dropout=dropout)
                    self.critic_head = nn.Linear(hidden_sizes[-1], 1)
                    for m in self.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)
                    nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
                def _input(self, stack_obs, agent_index, prev_action):
                    return torch.cat([
                        stack_obs,
                        F.one_hot(agent_index.long(), 2).float(),
                        F.one_hot(prev_action.long(), self.num_actions).float(),
                    ], dim=-1)
                def actor_logits(self, stack_obs, agent_index, prev_action):
                    return self.actor_head(self.actor_encoder(self._input(stack_obs, agent_index, prev_action)))
                def forward(self, stack_obs, agent_index, prev_action):
                    x = self._input(stack_obs, agent_index, prev_action)
                    return self.actor_head(self.actor_encoder(x)), self.critic_head(self.critic_encoder(x)).squeeze(-1)
                def load_from_bc(self, bc):
                    self.actor_encoder.load_state_dict(bc.encoder.state_dict())
                    self.actor_head.load_state_dict(bc.actor_head.state_dict())

            K_STACK = int(TRAIN_CFG['bc_warmstart']['k_stack'])
            HS = tuple(TRAIN_CFG['bc_warmstart']['hidden_sizes'])

            class BCDataset(Dataset):
                def __init__(self, eps, mean, std, k_stack=4):
                    self.samples = []
                    all_actions = np.concatenate([e['actions'] for e in eps])
                    counts = np.maximum(np.bincount(all_actions, minlength=6).astype(float), 1.0)
                    action_w = np.clip((counts.sum() / (6 * counts)) / (counts.sum() / (6 * counts)).mean(), 0.5, 3.0)
                    for ep in eps:
                        obs = (ep['obs'] - mean) / std
                        actions = ep['actions']; ai = int(ep['agent_index']); qw = float(ep['quality_weight'])
                        for t in range(len(actions)):
                            frames = [obs[max(0, t - j)] for j in range(k_stack - 1, -1, -1)]
                            target = int(actions[t])
                            prev = int(actions[t - 1]) if t > 0 else 0
                            weight = float(np.clip(qw * action_w[target], 0.05, 5.0))
                            self.samples.append((np.concatenate(frames).astype(np.float32), ai, prev, target, weight))
                def __len__(self): return len(self.samples)
                def __getitem__(self, i):
                    st, ai, prev, target, weight = self.samples[i]
                    return (
                        torch.from_numpy(st),
                        torch.tensor(ai, dtype=torch.long),
                        torch.tensor(prev, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                        torch.tensor(weight, dtype=torch.float32),
                    )

            train_ds = BCDataset(train_eps, NORM_MEAN, NORM_STD, K_STACK)
            val_ds = BCDataset(val_eps, NORM_MEAN, NORM_STD, K_STACK)
            batch = int(TRAIN_CFG['bc_warmstart']['batch_size'])
            train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
            val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
            print('BC samples:', len(train_ds), 'val:', len(val_ds))
            """
        ),
        md("## Entrenar BC warm-start"),
        code(
            """
            progress['stage'] = 'bc_warmstart'; save_progress()
            bc_cfg = TRAIN_CFG['bc_warmstart']
            bc_model = BCWarmstartActor(OBS_DIM, K_STACK, 6, HS, dropout=float(bc_cfg['dropout'])).to(DEVICE)
            opt_bc = optim.Adam(bc_model.parameters(), lr=float(bc_cfg['lr']), weight_decay=float(bc_cfg['weight_decay']))
            sched = optim.lr_scheduler.CosineAnnealingLR(opt_bc, T_max=int(bc_cfg['epochs']), eta_min=1e-5)
            best_val = float('inf'); patience = 0; history = []

            for epoch in range(1, int(bc_cfg['epochs']) + 1):
                t0 = time.time()
                bc_model.train(); tr_loss = tr_ok = tr_n = 0
                for st, ai, prev, target, weight in train_dl:
                    st, ai, prev, target, weight = st.to(DEVICE), ai.to(DEVICE), prev.to(DEVICE), target.to(DEVICE), weight.to(DEVICE)
                    logits = bc_model(st, ai, prev)
                    ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=float(bc_cfg['label_smoothing']))
                    loss = (ce * weight).sum() / weight.sum().clamp(1.0)
                    opt_bc.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(bc_model.parameters(), 1.0)
                    opt_bc.step()
                    tr_loss += loss.item() * len(target); tr_ok += int((logits.argmax(-1) == target).sum()); tr_n += len(target)
                sched.step()

                bc_model.eval(); vl_loss = vl_ok = vl_n = 0
                with torch.no_grad():
                    for st, ai, prev, target, weight in val_dl:
                        st, ai, prev, target, weight = st.to(DEVICE), ai.to(DEVICE), prev.to(DEVICE), target.to(DEVICE), weight.to(DEVICE)
                        logits = bc_model(st, ai, prev)
                        ce = F.cross_entropy(logits, target, reduction='none')
                        loss = (ce * weight).sum() / weight.sum().clamp(1.0)
                        vl_loss += loss.item() * len(target); vl_ok += int((logits.argmax(-1) == target).sum()); vl_n += len(target)

                row = {
                    'epoch': epoch,
                    'train_loss': tr_loss / max(tr_n, 1),
                    'train_acc': tr_ok / max(tr_n, 1),
                    'val_loss': vl_loss / max(vl_n, 1),
                    'val_acc': vl_ok / max(vl_n, 1),
                    'elapsed_s': time.time() - t0,
                }
                history.append(row)
                print(f"BC epoch {epoch:02d}: train={row['train_loss']:.4f} val={row['val_loss']:.4f} acc={row['val_acc']:.3f}")
                if row['val_loss'] < best_val:
                    best_val = row['val_loss']; patience = 0
                    torch.save({
                        'model_state_dict': bc_model.state_dict(),
                        'obs_dim': OBS_DIM,
                        'k_stack': K_STACK,
                        'hidden_sizes': list(HS),
                        'val_loss': best_val,
                    }, OUTPUT_DIR / 'bc_warmstart.pt')
                else:
                    patience += 1
                    if patience >= int(bc_cfg['patience']):
                        print('BC early stop')
                        break

            with open(OUTPUT_DIR / 'option_b_bc_training.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader(); w.writerows(history)
            ck = torch.load(OUTPUT_DIR / 'bc_warmstart.pt', map_location=DEVICE, weights_only=False)
            bc_model.load_state_dict(ck['model_state_dict']); bc_model.eval()
            progress['bc_val_loss'] = float(ck['val_loss'])
            progress['artifacts'].extend(['bc_warmstart.pt', 'option_b_bc_training.csv'])
            save_progress()
            """
        ),
        md("## Preparar entornos y partners neuronales"),
        code(
            """
            progress['stage'] = 'ppo_env_preflight'; save_progress()
            PPO_ENV_OK = bool(OVERCOOKED_INSTALLED)

            def clean_grid(grid):
                rows = [r.rstrip('\\n') for r in str(grid).split('\\n')]
                rows = [r.strip() for r in rows if r.strip()]
                if not rows:
                    raise ValueError('empty layout grid')
                if len({len(r) for r in rows}) != 1:
                    raise ValueError('layout rows must have equal width')
                return rows

            def build_mdp_from_spec(spec):
                if spec['type'] == 'custom':
                    params = dict(spec['custom_layout_dict'])
                    grid = clean_grid(params.pop('grid'))
                    params.setdefault('layout_name', spec['layout_name'])
                    return OvercookedGridworld.from_grid(
                        layout_grid=grid,
                        base_layout_params=params,
                        params_to_overwrite={'old_dynamics': True},
                    )
                return OvercookedGridworld.from_layout_name(spec['layout_name'], old_dynamics=True)

            def build_env_from_spec(spec):
                return OvercookedEnv.from_mdp(build_mdp_from_spec(spec), horizon=int(spec.get('horizon') or 250), info_level=0)

            class NeuralPartner:
                def __init__(self, model, mean, std, env, agent_index=1, temperature=1.0, seed=42):
                    self.model = model
                    self.model.eval()
                    self.mean = mean; self.std = std; self.env = env
                    self.agent_index = int(agent_index)
                    self.temperature = max(float(temperature), 1e-6)
                    self.rng = np.random.default_rng(seed)
                    self.reset()
                def reset(self):
                    self.prev_action = 0
                    self.stack = deque([np.zeros(OBS_DIM, np.float32)] * K_STACK, maxlen=K_STACK)
                def set_agent_index(self, agent_index):
                    if int(agent_index) != self.agent_index:
                        self.agent_index = int(agent_index)
                        self.reset()
                def set_env(self, env):
                    if env is not self.env:
                        self.env = env
                        self.reset()
                def set_mdp(self, mdp):
                    self.mdp = mdp
                def action(self, state):
                    obs = self.env.featurize_state_mdp(state)[self.agent_index].astype(np.float32)
                    self.stack.append((obs - self.mean) / self.std)
                    st = np.concatenate(list(self.stack)).astype(np.float32)
                    with torch.no_grad():
                        out = self.model(
                            torch.from_numpy(st).unsqueeze(0).to(DEVICE),
                            torch.tensor([self.agent_index], dtype=torch.long, device=DEVICE),
                            torch.tensor([self.prev_action], dtype=torch.long, device=DEVICE),
                        )
                        logits = out[0] if isinstance(out, tuple) else out
                        probs = torch.softmax(logits[0] / self.temperature, dim=-1).cpu().numpy()
                    ai = int(self.rng.choice(len(probs), p=probs))
                    self.prev_action = ai
                    return Action.INDEX_TO_ACTION[ai], {}

            class ScriptedGreedyPartner:
                def __init__(self, env, agent_index=1, ingredient='onion', mode='balanced', seed=42):
                    self.env = env
                    self.mdp = env.mdp
                    self.agent_index = int(agent_index)
                    self.ingredient = ingredient if ingredient in {'onion', 'tomato'} else 'onion'
                    self.mode = mode if mode in {'balanced', 'runner', 'server'} else 'balanced'
                def reset(self):
                    pass
                def set_agent_index(self, agent_index):
                    self.agent_index = int(agent_index)
                def set_env(self, env):
                    self.env = env
                    self.mdp = env.mdp
                def set_mdp(self, mdp):
                    self.mdp = mdp
                def action(self, state):
                    try:
                        target = self._target(state)
                        if target is None:
                            return Action.STAY, {'policy_name': 'scripted_greedy', 'mode': self.mode, 'target': None}
                        return self._move_or_interact(state, target), {'policy_name': 'scripted_greedy', 'mode': self.mode, 'target': target}
                    except Exception as exc:
                        return Action.STAY, {'policy_name': 'scripted_greedy', 'mode': self.mode, 'fallback': True, 'error': repr(exc)}
                def _target(self, state):
                    p = state.players[self.agent_index]
                    held = p.held_object
                    pot_states = self.mdp.get_pot_states(state)
                    if held is not None:
                        if held.name == 'soup':
                            return self._nearest(p.position, self.mdp.get_serving_locations())
                        if held.name == 'dish':
                            ready = list(pot_states.get('ready', []))
                            if ready:
                                return self._nearest(p.position, ready)
                            waiting = list(pot_states.get('cooking', [])) + list(pot_states.get(f'{Recipe.MAX_NUM_INGREDIENTS}_items', []))
                            return self._nearest(p.position, waiting)
                        if held.name in {'onion', 'tomato'}:
                            return self._nearest(p.position, self._pots_accepting(pot_states))
                        return None
                    ready = list(pot_states.get('ready', []))
                    if self.mode == 'server':
                        if ready:
                            counter_dishes = self._counter_objects(state, 'dish')
                            return self._nearest(p.position, counter_dishes or self.mdp.get_dish_dispenser_locations())
                        waiting = list(pot_states.get('cooking', [])) + list(pot_states.get(f'{Recipe.MAX_NUM_INGREDIENTS}_items', []))
                        if waiting:
                            return self._nearest(p.position, self.mdp.get_dish_dispenser_locations())
                        pots = self._pots_accepting(pot_states)
                        if pots:
                            counter_ingredients = self._counter_objects(state, self.ingredient)
                            return self._nearest(p.position, counter_ingredients or self._ingredient_locations())
                        return None
                    if self.mode == 'runner':
                        pots = self._pots_accepting(pot_states)
                        if pots:
                            counter_ingredients = self._counter_objects(state, self.ingredient)
                            return self._nearest(p.position, counter_ingredients or self._ingredient_locations())
                        if ready:
                            return self._nearest(p.position, self.mdp.get_dish_dispenser_locations())
                        full = list(pot_states.get(f'{Recipe.MAX_NUM_INGREDIENTS}_items', []))
                        if full:
                            return self._nearest(p.position, full)
                        cooking = list(pot_states.get('cooking', []))
                        if cooking:
                            return self._nearest(p.position, self.mdp.get_dish_dispenser_locations())
                        return None
                    if ready:
                        counter_dishes = self._counter_objects(state, 'dish')
                        return self._nearest(p.position, counter_dishes or self.mdp.get_dish_dispenser_locations())
                    pots = self._pots_accepting(pot_states)
                    if pots:
                        counter_ingredients = self._counter_objects(state, self.ingredient)
                        return self._nearest(p.position, counter_ingredients or self._ingredient_locations())
                    full = list(pot_states.get(f'{Recipe.MAX_NUM_INGREDIENTS}_items', []))
                    if full:
                        return self._nearest(p.position, full)
                    cooking = list(pot_states.get('cooking', []))
                    if cooking:
                        return self._nearest(p.position, self.mdp.get_dish_dispenser_locations())
                    return None
                def _ingredient_locations(self):
                    locs = self.mdp.get_tomato_dispenser_locations() if self.ingredient == 'tomato' else self.mdp.get_onion_dispenser_locations()
                    return list(locs)
                def _pots_accepting(self, pot_states):
                    pots = list(pot_states.get('empty', []))
                    for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
                        pots.extend(list(pot_states.get(f'{k}_items', [])))
                    return pots
                def _counter_objects(self, state, name):
                    return [obj.position for obj in state.objects.values() if obj.name == name]
                def _move_or_interact(self, state, target):
                    p = state.players[self.agent_index]
                    if self._adjacent(p.position, target):
                        direction = self._direction(p.position, target)
                        return Action.INTERACT if p.orientation == direction else direction
                    nxt = self._next_step(state, target)
                    if nxt is None:
                        return Action.STAY
                    return Action.determine_action_for_change_in_pos(p.position, nxt)
                def _next_step(self, state, target):
                    start = state.players[self.agent_index].position
                    valid = set(self.mdp.get_valid_player_positions())
                    blocked = {pl.position for i, pl in enumerate(state.players) if i != self.agent_index}
                    goals = [p for p in self._adjacent_positions(target) if p in valid and p not in blocked]
                    if not goals:
                        goals = [p for p in self._adjacent_positions(target) if p in valid]
                    if not goals:
                        return None
                    queue = deque([(start, [start])])
                    seen = {start}
                    while queue:
                        pos, path = queue.popleft()
                        if pos in goals:
                            return path[1] if len(path) > 1 else None
                        for d in Direction.ALL_DIRECTIONS:
                            nxt = Action.move_in_direction(pos, d)
                            if nxt not in valid or (nxt in blocked and nxt not in goals) or nxt in seen:
                                continue
                            seen.add(nxt)
                            queue.append((nxt, path + [nxt]))
                    return None
                def _nearest(self, origin, positions):
                    positions = list(positions)
                    if not positions:
                        return None
                    return min(positions, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))
                def _adjacent_positions(self, pos):
                    return [Action.move_in_direction(pos, d) for d in Direction.ALL_DIRECTIONS]
                def _adjacent(self, a, b):
                    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                def _direction(self, a, b):
                    direction = (b[0] - a[0], b[1] - a[1])
                    if direction not in Direction.ALL_DIRECTIONS:
                        raise ValueError(f'not adjacent: {a}->{b}')
                    return direction

            def time_limit(seconds):
                try:
                    import contextlib, signal
                    @contextlib.contextmanager
                    def _ctx():
                        def handler(signum, frame):
                            raise TimeoutError(f'layout preflight timed out after {seconds}s')
                        old = signal.signal(signal.SIGALRM, handler)
                        signal.alarm(int(seconds))
                        try:
                            yield
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old)
                    return _ctx()
                except Exception:
                    import contextlib
                    return contextlib.nullcontext()

            layout_manifest = []
            ENV_CACHE = {}
            VALID_LAYOUTS = []
            if PPO_ENV_OK:
                timeout_s = int(TRAIN_CFG['ppo'].get('layout_preflight_timeout_s', 60))
                for spec in LAYOUT_CATALOG['specs']:
                    row = {k: spec.get(k) for k in ['id', 'type', 'layout_name', 'hash', 'weight', 'horizon', 'example_paths']}
                    try:
                        with time_limit(timeout_s):
                            env = build_env_from_spec(spec)
                            env.reset(regen_mdp=False)
                            obs = env.featurize_state_mdp(env.state)
                            if len(obs) != 2 or int(obs[0].shape[0]) != OBS_DIM:
                                raise ValueError(f'obs shape mismatch: {[getattr(o, "shape", None) for o in obs]}')
                            env.step([Action.STAY, Action.STAY])
                        ENV_CACHE[spec['id']] = env
                        VALID_LAYOUTS.append(spec)
                        row['status'] = 'ok'
                    except Exception as exc:
                        row['status'] = 'excluded'
                        row['error'] = repr(exc)
                    layout_manifest.append(row)
                PPO_ENV_OK = len(VALID_LAYOUTS) > 0

            (OUTPUT_DIR / 'layout_manifest.json').write_text(json.dumps({
                'catalog_layout_count': LAYOUT_CATALOG['layout_count'],
                'catalog_weight_sum': LAYOUT_CATALOG['weight_sum'],
                'valid_layout_count': len(VALID_LAYOUTS),
                'layouts': layout_manifest,
            }, indent=2), encoding='utf-8')
            progress['ppo_env_ok'] = PPO_ENV_OK
            progress['valid_layout_count'] = len(VALID_LAYOUTS)
            progress['artifacts'].append('layout_manifest.json')
            save_progress()
            print('Valid PPO layouts:', len(VALID_LAYOUTS), 'of', LAYOUT_CATALOG['layout_count'])
            """
        ),
        md("## PPO FCP contra BC clones e históricos"),
        code(
            """
            progress['stage'] = 'ppo_train'; save_progress()
            ppo_model = ActorCritic(OBS_DIM, K_STACK, 6, HS, dropout=0.05).to(DEVICE)
            ppo_model.load_from_bc(bc_model)
            opt_ppo = optim.Adam(ppo_model.parameters(), lr=float(TRAIN_CFG['ppo']['learning_rate']))
            ppo_hist = []
            partner_specs = [
                {'id': f"bc_clone_t{str(t).replace('.', '_')}", 'type': 'bc_clone', 'temperature': float(t), 'model': bc_model, 'snapshot_step': None}
                for t in TRAIN_CFG['ppo'].get('bc_partner_temperatures', [0.75, 1.0, 1.5])
            ]
            partner_specs.extend([
                {'id': f'scripted_{mode}_{ing}', 'type': 'scripted_greedy', 'ingredient': ing, 'mode': mode, 'temperature': 1.0, 'model': None, 'snapshot_step': None}
                for ing in TRAIN_CFG['ppo'].get('scripted_partner_ingredients', [])
                for mode in TRAIN_CFG['ppo'].get('scripted_partner_modes', ['balanced'])
            ])
            PSCORES = {p['id']: [] for p in partner_specs}

            def manifest_partners():
                return [{k: v for k, v in p.items() if k != 'model'} for p in partner_specs]

            def add_historical(step, score):
                frozen = copy.deepcopy(ppo_model).to(DEVICE).eval()
                for param in frozen.parameters():
                    param.requires_grad_(False)
                pid = f'historical_ppo_step{step}'
                partner_specs.append({'id': pid, 'type': 'historical_ppo', 'temperature': 1.0, 'model': frozen, 'snapshot_step': int(step), 'mean_deliveries_at_add': float(score)})
                PSCORES[pid] = []
                torch.save({'step': int(step), 'model_state_dict': frozen.state_dict(), 'mean_deliveries': float(score)}, OUTPUT_DIR / f'ppo_partner_step{step}.pt')

            def softmax_np(x):
                x = np.asarray(x, dtype=np.float64); x = x - x.max()
                e = np.exp(x)
                return e / e.sum()

            def sample_partner():
                eps = float(TRAIN_CFG['ppo']['adaptive_epsilon'])
                beta = float(TRAIN_CFG['ppo']['adaptive_beta'])
                ids = [p['id'] for p in partner_specs]
                avg = np.array([np.mean(PSCORES[i]) if PSCORES[i] else 0.5 for i in ids], dtype=np.float64)
                if TRAIN_CFG['ppo']['partner_sampling'] == 'adaptive_cole' and len(ids) > 1:
                    span = avg.max() - avg.min()
                    norm = (avg - avg.min()) / span if span > 0 else np.ones_like(avg) * 0.5
                    probs = (1 - eps) * softmax_np(-beta * norm) + eps / len(ids)
                    probs = probs / probs.sum()
                    idx = int(np.random.choice(len(ids), p=probs))
                else:
                    idx = int(np.random.randint(len(ids)))
                return partner_specs[idx]

            layout_weights = np.array([max(1, int(s.get('weight', 1))) for s in VALID_LAYOUTS], dtype=np.float64)
            focus_ids = set(TRAIN_CFG['ppo'].get('layout_focus_ids', []))
            if focus_ids:
                focus_mult = float(TRAIN_CFG['ppo'].get('layout_focus_multiplier', 1.0))
                for i, spec in enumerate(VALID_LAYOUTS):
                    if spec['id'] in focus_ids:
                        layout_weights[i] *= focus_mult
            layout_probs = layout_weights / layout_weights.sum() if len(layout_weights) else np.array([])
            LAYOUT_SCORES = {s['id']: [] for s in VALID_LAYOUTS}
            def sample_layout_spec():
                if TRAIN_CFG['ppo'].get('layout_sampling') == 'adaptive_coverage':
                    beta = float(TRAIN_CFG['ppo'].get('layout_deficit_beta', 2.0))
                    recent = np.array([
                        np.mean(LAYOUT_SCORES[s['id']][-10:]) if LAYOUT_SCORES[s['id']] else 0.0
                        for s in VALID_LAYOUTS
                    ], dtype=np.float64)
                    deficit = np.maximum(0.0, 3.0 - recent)
                    probs = layout_weights * (1.0 + beta * deficit)
                    probs = probs / probs.sum()
                    return VALID_LAYOUTS[int(np.random.choice(len(VALID_LAYOUTS), p=probs))]
                return VALID_LAYOUTS[int(np.random.choice(len(VALID_LAYOUTS), p=layout_probs))]

            def featurize(env, state, role):
                return env.featurize_state_mdp(state)[role].astype(np.float32)
            def make_stack(dq):
                return np.concatenate(list(dq)).astype(np.float32)
            def make_partner(pspec, env, role, seed):
                if pspec['type'] == 'scripted_greedy':
                    return ScriptedGreedyPartner(env, agent_index=1 - role, ingredient=pspec.get('ingredient', 'onion'), mode=pspec.get('mode', 'balanced'), seed=seed)
                return NeuralPartner(pspec['model'], NORM_MEAN, NORM_STD, env, agent_index=1 - role, temperature=pspec['temperature'], seed=seed)
            def entropy_coef(step):
                p = TRAIN_CFG['ppo']; t = min(step / max(int(p['entropy_decay_steps']), 1), 1.0)
                return float(p['entropy_coef_start']) + (float(p['entropy_coef_end']) - float(p['entropy_coef_start'])) * t

            def eval_partner_specs():
                specs = []
                for name in TRAIN_CFG['ppo'].get('eval_checkpoint_partners', ['bc_clone']):
                    if name == 'bc_clone':
                        specs.append({'id': 'eval_bc_clone', 'type': 'bc_clone', 'temperature': 1.0, 'model': bc_model})
                    elif name.startswith('scripted_'):
                        parts = name.split('_')
                        if len(parts) == 2:
                            mode, ingredient = 'balanced', parts[1]
                        else:
                            mode, ingredient = parts[1], parts[2]
                        specs.append({'id': f'eval_scripted_{mode}_{ingredient}', 'type': 'scripted_greedy', 'ingredient': ingredient, 'mode': mode, 'temperature': 1.0, 'model': None})
                return specs

            def evaluate_policy_snapshot(eval_model, step_label, csv_path=None, seeds=None, partners=None):
                seeds = list(seeds if seeds is not None else TRAIN_CFG['ppo'].get('eval_checkpoint_seeds', [101]))
                partners = list(partners if partners is not None else eval_partner_specs())
                eval_model.eval()
                rows = []
                for spec in VALID_LAYOUTS:
                    for pspec in partners:
                        for seed in seeds:
                            for role in [0, 1]:
                                try:
                                    env = build_env_from_spec(spec)
                                    env.reset(regen_mdp=False); state = env.state
                                    partner = make_partner(pspec, env, role, seed=seed + role)
                                    dq = deque([np.zeros(OBS_DIM, np.float32)] * K_STACK, maxlen=K_STACK)
                                    dq.append((featurize(env, state, role) - NORM_MEAN) / NORM_STD)
                                    prev = 0; ep_return = 0.0; deliveries = 0; done = False; steps = 0
                                    while not done:
                                        st = make_stack(dq)
                                        with torch.no_grad():
                                            if hasattr(eval_model, 'actor_logits'):
                                                logits = eval_model.actor_logits(
                                                    torch.from_numpy(st).unsqueeze(0).to(DEVICE),
                                                    torch.tensor([role], dtype=torch.long, device=DEVICE),
                                                    torch.tensor([prev], dtype=torch.long, device=DEVICE),
                                                )
                                            else:
                                                logits = eval_model(
                                                    torch.from_numpy(st).unsqueeze(0).to(DEVICE),
                                                    torch.tensor([role], dtype=torch.long, device=DEVICE),
                                                    torch.tensor([prev], dtype=torch.long, device=DEVICE),
                                                )
                                            ac = int(logits.argmax(-1).item())
                                        partner.set_env(env); partner.set_agent_index(1 - role); partner.set_mdp(env.mdp)
                                        pa, _ = partner.action(state)
                                        joint = [Action.INDEX_TO_ACTION[ac], pa] if role == 0 else [pa, Action.INDEX_TO_ACTION[ac]]
                                        state, reward, done, info = env.step(joint)
                                        ep_return += float(reward)
                                        deliveries += int(np.sum(np.asarray(info.get('sparse_r_by_agent', [reward, 0.0]), dtype=np.float32)) > 0)
                                        prev = ac
                                        dq.append((featurize(env, state, role) - NORM_MEAN) / NORM_STD)
                                        steps += 1
                                    rows.append({
                                        'step': step_label,
                                        'layout_id': spec['id'],
                                        'layout_name': spec['layout_name'],
                                        'layout_type': spec['type'],
                                        'seed': seed,
                                        'role': role,
                                        'partner': pspec['id'],
                                        'deliveries': deliveries,
                                        'soups': deliveries,
                                        'sparse_return': ep_return,
                                        'steps': steps,
                                    })
                                except Exception as exc:
                                    rows.append({
                                        'step': step_label,
                                        'layout_id': spec['id'],
                                        'layout_name': spec['layout_name'],
                                        'layout_type': spec['type'],
                                        'seed': seed,
                                        'role': role,
                                        'partner': pspec['id'],
                                        'deliveries': np.nan,
                                        'soups': np.nan,
                                        'sparse_return': np.nan,
                                        'steps': 0,
                                        'error': repr(exc),
                                    })
                good = [r for r in rows if 'error' not in r and not np.isnan(float(r['deliveries']))]
                by_layout = {}
                for r in good:
                    by_layout.setdefault(r['layout_id'], []).append(float(r['deliveries']))
                best_by_layout = {k: max(v) for k, v in by_layout.items()}
                min_by_layout = {k: min(v) for k, v in by_layout.items()}
                summary = {
                    'n_eps': len(rows),
                    'n_ok': len(good),
                    'mean_deliveries': float(np.mean([r['deliveries'] for r in good])) if good else None,
                    'zero_delivery_rate': float(np.mean([r['deliveries'] < 1 for r in good])) if good else None,
                    'layouts_reaching_min3_any_partner': int(sum(v >= 3 for v in best_by_layout.values())),
                    'layouts_passing_min3': int(sum(v >= 3 for v in min_by_layout.values())),
                    'best_deliveries_by_layout': {k: float(v) for k, v in sorted(best_by_layout.items())},
                    'min_deliveries_by_layout': {k: float(v) for k, v in sorted(min_by_layout.items())},
                }
                if csv_path is not None:
                    fieldnames = sorted({k for r in rows for k in r.keys()})
                    with open(csv_path, 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader(); w.writerows(rows)
                return rows, summary

            if PPO_ENV_OK:
                p = TRAIN_CFG['ppo']
                GAMMA = float(p['gamma']); LAM = float(p['gae_lambda']); CLIP = float(p['clip_range'])
                VC = float(p['value_coef']); MG = float(p['max_grad_norm'])
                PPO_EP = int(p['ppo_epochs']); MB = int(p['minibatch_size']); RS = int(p['rollout_steps'])
                TOTAL = int(p['total_steps']); HIST_FREQ = int(p['add_historical_checkpoint_freq'])
                EVAL_FREQ = int(p.get('eval_checkpoint_freq_steps', 100000))
                SHAPED_COEF = float(p.get('shaped_reward_coef', 0.0))
                HIST_MIN = float(p.get('historical_min_mean_deliveries', 0.0))
                rb_st = np.zeros((RS, K_STACK * OBS_DIM), np.float32)
                rb_ai = np.zeros(RS, np.int64); rb_pa = np.zeros(RS, np.int64); rb_ac = np.zeros(RS, np.int64)
                rb_rw = np.zeros(RS, np.float32); rb_dn = np.zeros(RS, np.float32); rb_vl = np.zeros(RS, np.float32); rb_lp = np.zeros(RS, np.float32)

                gstep = 0; next_hist = HIST_FREQ; next_eval = EVAL_FREQ
                ep_deliveries = []; ep_sparse_returns = []; best_deliveries = -1e9; best_coverage3 = -1
                best_eval_score = (-1, -1, -1.0)
                cur_spec = sample_layout_spec(); cur_ps = sample_partner(); cur_role = int(np.random.randint(0, 2))
                env = ENV_CACHE[cur_spec['id']]; env.reset(regen_mdp=False); state = env.state
                partner = make_partner(cur_ps, env, cur_role, seed=gstep + 123)
                dq = deque([np.zeros(OBS_DIM, np.float32)] * K_STACK, maxlen=K_STACK)
                dq.append((featurize(env, state, cur_role) - NORM_MEAN) / NORM_STD)
                prev_a = 0; ep_sparse_r = 0.0; ep_shaped_r = 0.0; ep_delivery_count = 0; t0 = time.time()
                print(f'PPO start: total_steps={TOTAL:,}, layouts={len(VALID_LAYOUTS)}, partners={len(partner_specs)}')

                while gstep < TOTAL:
                    ppo_model.eval()
                    for si in range(RS):
                        st_np = make_stack(dq)
                        with torch.no_grad():
                            logits, value = ppo_model(
                                torch.from_numpy(st_np).unsqueeze(0).to(DEVICE),
                                torch.tensor([cur_role], dtype=torch.long, device=DEVICE),
                                torch.tensor([prev_a], dtype=torch.long, device=DEVICE),
                            )
                            dist = torch.distributions.Categorical(logits=logits)
                            ac = dist.sample(); lp = dist.log_prob(ac)
                        ac_i = int(ac.item())
                        partner.set_env(env); partner.set_agent_index(1 - cur_role); partner.set_mdp(env.mdp)
                        partner_action, _ = partner.action(state)
                        ego_action = Action.INDEX_TO_ACTION[ac_i]
                        joint_action = [ego_action, partner_action] if cur_role == 0 else [partner_action, ego_action]
                        next_state, reward, done, info = env.step(joint_action)
                        sparse_reward = float(reward)
                        shaped_reward = float(np.sum(np.asarray(info.get('shaped_r_by_agent', [0.0, 0.0]), dtype=np.float32)))
                        delivery_event = int(np.sum(np.asarray(info.get('sparse_r_by_agent', [0.0, 0.0]), dtype=np.float32)) > 0)
                        train_reward = sparse_reward + SHAPED_COEF * shaped_reward
                        rb_st[si] = st_np; rb_ai[si] = cur_role; rb_pa[si] = prev_a; rb_ac[si] = ac_i
                        rb_rw[si] = train_reward; rb_dn[si] = float(done); rb_vl[si] = float(value.item()); rb_lp[si] = float(lp.item())
                        ep_sparse_r += sparse_reward; ep_shaped_r += shaped_reward; ep_delivery_count += delivery_event
                        prev_a = ac_i; state = next_state
                        dq.append((featurize(env, state, cur_role) - NORM_MEAN) / NORM_STD)

                        if done:
                            ep_sparse_returns.append(ep_sparse_r); ep_deliveries.append(ep_delivery_count)
                            LAYOUT_SCORES[cur_spec['id']].append(ep_delivery_count)
                            LAYOUT_SCORES[cur_spec['id']] = LAYOUT_SCORES[cur_spec['id']][-20:]
                            PSCORES[cur_ps['id']].append(ep_delivery_count)
                            PSCORES[cur_ps['id']] = PSCORES[cur_ps['id']][-20:]
                            cur_spec = sample_layout_spec(); cur_ps = sample_partner(); cur_role = int(np.random.randint(0, 2))
                            env = ENV_CACHE[cur_spec['id']]; env.reset(regen_mdp=False); state = env.state
                            partner = make_partner(cur_ps, env, cur_role, seed=gstep + si + 456)
                            dq = deque([np.zeros(OBS_DIM, np.float32)] * K_STACK, maxlen=K_STACK)
                            dq.append((featurize(env, state, cur_role) - NORM_MEAN) / NORM_STD)
                            prev_a = 0; ep_sparse_r = 0.0; ep_shaped_r = 0.0; ep_delivery_count = 0

                    with torch.no_grad():
                        _, last_value = ppo_model(
                            torch.from_numpy(make_stack(dq)).unsqueeze(0).to(DEVICE),
                            torch.tensor([cur_role], dtype=torch.long, device=DEVICE),
                            torch.tensor([prev_a], dtype=torch.long, device=DEVICE),
                        )
                    adv = np.zeros(RS, np.float32); gae = 0.0; last_done = bool(rb_dn[RS - 1])
                    for t in reversed(range(RS)):
                        next_non_terminal = 1.0 - (float(last_done) if t == RS - 1 else rb_dn[t + 1])
                        next_value = float(last_value.item()) if t == RS - 1 else rb_vl[t + 1]
                        delta = rb_rw[t] + GAMMA * next_value * next_non_terminal - rb_vl[t]
                        gae = delta + GAMMA * LAM * next_non_terminal * gae
                        adv[t] = gae
                    returns = adv + rb_vl
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    ppo_model.train(); pg_losses = []; v_losses = []; ents = []
                    ec = entropy_coef(gstep)
                    for _ in range(PPO_EP):
                        order = np.random.permutation(RS)
                        for start in range(0, RS, MB):
                            batch_idx = order[start:start + MB]
                            logits, values = ppo_model(
                                torch.from_numpy(rb_st[batch_idx]).to(DEVICE),
                                torch.from_numpy(rb_ai[batch_idx]).to(DEVICE),
                                torch.from_numpy(rb_pa[batch_idx]).to(DEVICE),
                            )
                            dist = torch.distributions.Categorical(logits=logits)
                            new_lp = dist.log_prob(torch.from_numpy(rb_ac[batch_idx]).to(DEVICE))
                            entropy = dist.entropy().mean()
                            ratio = torch.exp(new_lp - torch.from_numpy(rb_lp[batch_idx]).to(DEVICE))
                            mb_adv = torch.from_numpy(adv[batch_idx]).to(DEVICE)
                            mb_ret = torch.from_numpy(returns[batch_idx]).to(DEVICE)
                            pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                            vl = F.mse_loss(values, mb_ret)
                            loss = pg + VC * vl - ec * entropy
                            opt_ppo.zero_grad(); loss.backward()
                            nn.utils.clip_grad_norm_(ppo_model.parameters(), MG)
                            opt_ppo.step()
                            pg_losses.append(float(pg.item())); v_losses.append(float(vl.item())); ents.append(float(entropy.item()))

                    gstep += RS
                    if ep_deliveries:
                        mean_deliveries = float(np.mean(ep_deliveries[-20:]))
                        zero_rate = float(np.mean(np.array(ep_deliveries[-20:]) < 1.0))
                        layout_best = {sid: (max(vals) if vals else 0) for sid, vals in LAYOUT_SCORES.items()}
                        coverage3 = int(sum(v >= 3 for v in layout_best.values()))
                    else:
                        mean_deliveries = 0.0
                        zero_rate = 1.0
                        coverage3 = 0
                    if gstep >= next_hist and gstep < TOTAL:
                        if mean_deliveries >= HIST_MIN:
                            add_historical(gstep, mean_deliveries)
                        next_hist += HIST_FREQ
                    if ep_deliveries:
                        row = {
                            'step': int(gstep),
                            'mean_deliveries': mean_deliveries,
                            'zero_delivery_rate': zero_rate,
                            'layout_train_coverage3': coverage3,
                            'partners': len(partner_specs),
                            'pg_loss': float(np.mean(pg_losses)),
                            'value_loss': float(np.mean(v_losses)),
                            'entropy': float(np.mean(ents)),
                            'elapsed_s': time.time() - t0,
                        }
                        ppo_hist.append(row)
                        print(f"Step {gstep:8d}: deliveries={mean_deliveries:.3f} coverage3={coverage3}/{len(VALID_LAYOUTS)} zero={zero_rate:.3f} partners={len(partner_specs)}")
                        if (coverage3, mean_deliveries) > (best_coverage3, best_deliveries):
                            best_coverage3 = coverage3
                            best_deliveries = mean_deliveries
                            torch.save({
                                'step': int(gstep),
                                'model_state_dict': ppo_model.state_dict(),
                                'mean_deliveries': best_deliveries,
                                'layout_train_coverage3': best_coverage3,
                                'obs_dim': OBS_DIM,
                                'k_stack': K_STACK,
                                'hidden_sizes': list(HS),
                            }, OUTPUT_DIR / 'train_checkpoint_by_coverage.pt')
                            progress['ppo_best_deliveries'] = best_deliveries
                            progress['ppo_best_train_coverage3'] = best_coverage3
                        if EVAL_FREQ > 0 and (gstep >= next_eval or gstep >= TOTAL):
                            _, eval_summary = evaluate_policy_snapshot(ppo_model, int(gstep))
                            eval_score = (
                                int(eval_summary['layouts_passing_min3']),
                                int(eval_summary['layouts_reaching_min3_any_partner']),
                                float(eval_summary['mean_deliveries'] or 0.0),
                            )
                            print(f"Eval {gstep:8d}: pass3={eval_score[0]}/{len(VALID_LAYOUTS)} reach3={eval_score[1]}/{len(VALID_LAYOUTS)} mean={eval_score[2]:.3f}")
                            if eval_score > best_eval_score:
                                best_eval_score = eval_score
                                torch.save({
                                    'step': int(gstep),
                                    'model_state_dict': ppo_model.state_dict(),
                                    'eval_summary': eval_summary,
                                    'obs_dim': OBS_DIM,
                                    'k_stack': K_STACK,
                                    'hidden_sizes': list(HS),
                                }, OUTPUT_DIR / 'best_checkpoint_by_soups.pt')
                                (OUTPUT_DIR / 'best_checkpoint_eval_summary.json').write_text(json.dumps(eval_summary, indent=2), encoding='utf-8')
                                progress['ppo_best_eval_pass3'] = eval_score[0]
                                progress['ppo_best_eval_reach3'] = eval_score[1]
                                progress['ppo_best_eval_mean_deliveries'] = eval_score[2]
                            while gstep >= next_eval:
                                next_eval += EVAL_FREQ
                        with open(OUTPUT_DIR / 'option_b_ppo_training.csv', 'w', newline='') as f:
                            w = csv.DictWriter(f, fieldnames=list(row.keys()))
                            w.writeheader(); w.writerows(ppo_hist)
                        (OUTPUT_DIR / 'partner_pool_manifest.json').write_text(
                            json.dumps({'partners': manifest_partners()}, indent=2),
                            encoding='utf-8',
                        )
                        save_progress()

                progress['artifacts'].extend(['best_checkpoint_by_soups.pt', 'best_checkpoint_eval_summary.json', 'train_checkpoint_by_coverage.pt', 'option_b_ppo_training.csv', 'partner_pool_manifest.json'])
                save_progress()
            else:
                print('PPO skipped: no valid Overcooked layouts after preflight')
            """
        ),
        md("## Exportar política y verificar paridad"),
        code(
            """
            progress['stage'] = 'export'; save_progress()
            ppo_ckpt = OUTPUT_DIR / 'best_checkpoint_by_soups.pt'
            if ppo_ckpt.exists():
                ck = torch.load(ppo_ckpt, map_location=DEVICE, weights_only=False)
                ppo_model.load_state_dict(ck['model_state_dict'])
                ppo_model.eval()
                export_model = ppo_model
                params = {f'actor_encoder.{n}': p.detach().cpu().numpy() for n, p in ppo_model.actor_encoder.named_parameters()}
                params.update({f'actor_head.{n}': p.detach().cpu().numpy() for n, p in ppo_model.actor_head.named_parameters()})
                model_type = 'fcp_ppo_mlp'
            else:
                bc_model.eval()
                export_model = bc_model
                params = {f'actor_encoder.{n}': p.detach().cpu().numpy() for n, p in bc_model.encoder.named_parameters()}
                params.update({f'actor_head.{n}': p.detach().cpu().numpy() for n, p in bc_model.actor_head.named_parameters()})
                model_type = 'bc_warmstart_mlp'

            np.savez_compressed(OUTPUT_DIR / 'final_policy.npz', **params)
            cfg_out = {
                'option': 'B',
                'model_type': model_type,
                'obs_dim': OBS_DIM,
                'k_stack': K_STACK,
                'num_actions': 6,
                'hidden_sizes': list(HS),
                'normalization_path': 'artifacts/shared/normalization.json',
                'agent_index_encoding': 'one_hot',
                'previous_action': True,
                'obs_stack_k': K_STACK,
            }
            (OUTPUT_DIR / 'final_policy_config.json').write_text(json.dumps(cfg_out, indent=2), encoding='utf-8')

            exported = np.load(OUTPUT_DIR / 'final_policy.npz')
            def ln_np(x, w, b, eps=1e-5):
                return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b
            def np_forward(st, ai, prev):
                ah = np.zeros(2, np.float32); ah[int(ai)] = 1.0
                ph = np.zeros(6, np.float32); ph[int(prev)] = 1.0
                x = np.concatenate([st.astype(np.float32).ravel(), ah, ph])
                li = 0
                for i in range(len(HS)):
                    W = exported[f'actor_encoder.net.{li}.weight']; b = exported[f'actor_encoder.net.{li}.bias']
                    x = x @ W.T + b; li += 1
                    if i != len(HS) - 1:
                        x = ln_np(x, exported[f'actor_encoder.net.{li}.weight'], exported[f'actor_encoder.net.{li}.bias']); li += 1
                        x = np.maximum(0.0, x); li += 1
                        li += 1
                return x @ exported['actor_head.weight'].T + exported['actor_head.bias']

            max_err = 0.0; matches = 0; checks = 20
            for _ in range(checks):
                st = np.random.randn(K_STACK * OBS_DIM).astype(np.float32)
                ai = int(np.random.randint(0, 2)); prev = int(np.random.randint(0, 6))
                with torch.no_grad():
                    if hasattr(export_model, 'actor_logits'):
                        pt = export_model.actor_logits(
                            torch.from_numpy(st).unsqueeze(0).to(DEVICE),
                            torch.tensor([ai], dtype=torch.long, device=DEVICE),
                            torch.tensor([prev], dtype=torch.long, device=DEVICE),
                        ).cpu().numpy()[0]
                    else:
                        pt = export_model(
                            torch.from_numpy(st).unsqueeze(0).to(DEVICE),
                            torch.tensor([ai], dtype=torch.long, device=DEVICE),
                            torch.tensor([prev], dtype=torch.long, device=DEVICE),
                        ).cpu().numpy()[0]
                nf = np_forward(st, ai, prev)
                err = float(np.abs(pt - nf).max())
                max_err = max(max_err, err)
                matches += int(np.argmax(pt) == np.argmax(nf))
            parity_ok = bool(max_err < 1e-5 and matches == checks)
            (OUTPUT_DIR / 'parity_check.json').write_text(json.dumps({
                'max_abs_error': max_err,
                'action_match': matches,
                'checks': checks,
                'parity_ok': parity_ok,
            }, indent=2), encoding='utf-8')
            print(f'Parity max_err={max_err:.3e}, match={matches}/{checks}, ok={parity_ok}')
            progress['parity_ok'] = parity_ok
            progress['artifacts'].extend(['final_policy.npz', 'final_policy_config.json', 'parity_check.json'])
            save_progress()
            """
        ),
        md("## Evaluar contra partners de checkpoint"),
        code(
            """
            progress['stage'] = 'evaluation'; save_progress()
            if PPO_ENV_OK:
                eval_model = ppo_model if (OUTPUT_DIR / 'best_checkpoint_by_soups.pt').exists() else bc_model
                seeds = [42, 43, 44]
                _, summary = evaluate_policy_snapshot(eval_model, 'final', OUTPUT_DIR / 'option_b_evaluation.csv', seeds=seeds)
                (OUTPUT_DIR / 'eval_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                progress['eval_summary'] = summary
                progress['artifacts'].extend(['option_b_evaluation.csv', 'eval_summary.json'])
            else:
                print('Evaluation skipped: no valid PPO env')
            save_progress()
            """
        ),
        md("## Finalizar"),
        code(
            """
            try:
                if not (OUTPUT_DIR / 'partner_pool_manifest.json').exists():
                    (OUTPUT_DIR / 'partner_pool_manifest.json').write_text(
                        json.dumps({'partners': manifest_partners() if 'partner_specs' in globals() else []}, indent=2),
                        encoding='utf-8',
                    )
                    progress['artifacts'].append('partner_pool_manifest.json')
                progress['status'] = 'complete'
                progress['stage'] = 'done'
            except Exception as exc:
                progress['status'] = 'failed'
                progress['error'] = repr(exc)
                progress['traceback'] = traceback.format_exc()
            finally:
                save_progress()
                print('Output files:')
                for path in sorted(OUTPUT_DIR.iterdir()):
                    if path.is_file():
                        print(f'  {path.name}: {path.stat().st_size / 1024:.1f} KB')
                print('Run status:', progress['status'])
            """
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Jupyter notebook built successfully at {NB_PATH}")


if __name__ == "__main__":
    build()
