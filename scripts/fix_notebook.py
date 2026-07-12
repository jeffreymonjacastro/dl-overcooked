import json
import pathlib

def main():
    nb_path = pathlib.Path("kaggle/v1/input/main.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    fixed_np = False
    fixed_install = False
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Fix np_forward cell
            if "def np_forward(st, ai, pa):" in source:
                print("Found np_forward cell, updating...")
                cell["source"] = [
                    "exported = np.load(OUTPUT_DIR/'final_policy.npz')\n",
                    "\n",
                    "def ln_np(x, w, b, eps=1e-5):\n",
                    "    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True)\n",
                    "    return (x-m)/np.sqrt(v+eps)*w+b\n",
                    "\n",
                    "def np_forward(st, ai, pa):\n",
                    "    ah=np.zeros(2,np.float32); ah[ai]=1.\n",
                    "    ph=np.zeros(6,np.float32); ph[pa]=1.\n",
                    "    x=np.concatenate([st.flatten(),ah,ph])\n",
                    "    li=0\n",
                    "    num_layers = len(HS)\n",
                    "    for i in range(num_layers):\n",
                    "        is_last = (i == num_layers - 1)\n",
                    "        W=exported[f'actor_encoder.net.{li}.weight']; b=exported[f'actor_encoder.net.{li}.bias']\n",
                    "        x=x@W.T+b; li+=1\n",
                    "        if not is_last:\n",
                    "            x=ln_np(x,exported[f'actor_encoder.net.{li}.weight'],exported[f'actor_encoder.net.{li}.bias']); li+=1\n",
                    "            x=np.maximum(0.,x); li+=1; li+=1  # ReLU + Dropout skip\n",
                    "    return x@exported['actor_head.weight'].T+exported['actor_head.bias']\n",
                    "\n",
                    "max_err=0; match_=0; N=10\n",
                    "for _ in range(N):\n",
                    "    ts=np.random.randn(K_STACK*OBS_DIM).astype(np.float32)\n",
                    "    tai=np.random.randint(0,2); tpa=np.random.randint(0,6)\n",
                    "    with torch.no_grad():\n",
                    "        if PPO_ENV_OK and ppo_ckpt.exists():\n",
                    "            pt_l=ppo_model.actor_logits(torch.from_numpy(ts).unsqueeze(0).to(DEVICE),\n",
                    "                                        torch.tensor([tai],device=DEVICE),\n",
                    "                                        torch.tensor([tpa],device=DEVICE)).cpu().numpy()[0]\n",
                    "        else:\n",
                    "            pt_l=bc_model(torch.from_numpy(ts).unsqueeze(0).to(DEVICE),\n",
                    "                          torch.tensor([tai],device=DEVICE),\n",
                    "                          torch.tensor([tpa],device=DEVICE)).cpu().numpy()[0]\n",
                    "    np_l=np_forward(ts,tai,tpa)\n",
                    "    err=np.abs(pt_l-np_l).max(); max_err=max(max_err,err)\n",
                    "    if np.argmax(pt_l)==np.argmax(np_l): match_+=1\n",
                    "\n",
                    "print(f'Parity: max_err={max_err:.2e}, action_match={match_}/{N}')\n",
                    "ok=max_err<=1e-4 and match_==N\n",
                    "(OUTPUT_DIR/'parity_check.json').write_text(json.dumps({'max_abs_error':float(max_err),'match_rate':match_/N,'parity_ok':ok},indent=2))\n",
                    "progress['parity_ok']=ok; save_progress()\n",
                    "print('Parity OK:', ok)\n"
                ]
                fixed_np = True
            
            # Fix install cell
            if "progress['stage'] = 'install';" in source or "progress['stage']='install';" in source:
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
                    "# Force reinstall numpy < 1.24 to avoid np.Inf/NumPy 2.x compatibility issues\n",
                    "print('Reinstalling numpy<1.24...')\n",
                    "run_cmd(['pip', 'install', 'numpy<1.24', '--force-reinstall', '-q'])\n",
                    "\n",
                    "OVERCOOKED_INSTALLED = False\n",
                    "\n",
                    "# Attempt 1: PyPI install of overcooked-ai (relaxed dependencies)\n",
                    "print('[1] Trying PyPI install of overcooked-ai...')\n",
                    "if run_cmd(['pip', 'install', 'overcooked-ai', '--no-deps', '-q']):\n",
                    "    run_cmd(['pip', 'install', 'PyYAML>=6.0', 'scipy', 'dill', 'flask', '-q'])\n",
                    "    try:\n",
                    "        import overcooked_ai_py; OVERCOOKED_INSTALLED = True; print('  OK')\n",
                    "    except Exception as e: print(f'  Import failed: {e}')\n",
                    "\n",
                    "# Attempt 2: GitHub source\n",
                    "if not OVERCOOKED_INSTALLED:\n",
                    "    print('[2] Trying GitHub source...')\n",
                    "    if run_cmd(['pip', 'install', 'git+https://github.com/HumanCompatibleAI/overcooked_ai.git', '--no-deps', '-q']):\n",
                    "        run_cmd(['pip', 'install', 'PyYAML>=6.0', 'scipy', 'dill', '-q'])\n",
                    "        try:\n",
                    "            import overcooked_ai_py; OVERCOOKED_INSTALLED = True; print('  OK')\n",
                    "        except Exception as e: print(f'  Import failed: {e}')\n",
                    "\n",
                    "print(f'OVERCOOKED_INSTALLED: {OVERCOOKED_INSTALLED}')\n",
                    "progress['overcooked_available'] = OVERCOOKED_INSTALLED; save_progress()\n"
                ]
                fixed_install = True
                
    if fixed_np and fixed_install:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully with both fixes.")
    else:
        print(f"Error: fixed_np={fixed_np}, fixed_install={fixed_install}")

if __name__ == "__main__":
    main()
