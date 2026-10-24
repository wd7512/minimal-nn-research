"""Optuna over PPO hypers. Short-budget trials from the 9.33 winner, best eval100 wins."""
import os
import subprocess
import sys

import optuna
import torch

PY = "/Users/williamdennis/repos/minimal-nn-competition/.venv/bin/python"
SCRIPT = "/Users/williamdennis/.hermes/cache/scratch/ppo_obs.py"
SEEDLIST = ",".join(map(str, range(300)))
INIT = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/models/snake-obstacles-8x8/obs-fb/model.pt"


def objective(trial):
    env = dict(os.environ)
    env.update({
        "PPO_H": "512",
        "PPO_INIT": INIT,
        "PPO_STEPS": "300000",
        "PPO_NSTEP": "2048",
        "PPO_EVAL_EVERY": "25",
        "PPO_LR": str(trial.suggest_float("lr", 3e-5, 1e-3, log=True)),
        "PPO_CLIP": str(trial.suggest_float("clip", 0.1, 0.4)),
        "PPO_GAMMA": str(trial.suggest_float("gamma", 0.99, 0.999)),
        "PPO_ENT": str(trial.suggest_float("ent", 1e-4, 5e-2, log=True)),
        "PPO_SEEDLIST": SEEDLIST,
        "PPO_OUT": f"/tmp/ppo-obs/trial-{trial.number}.pt",
    })
    r = subprocess.run([PY, SCRIPT], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"trial {trial.number} FAILED\n{r.stderr[-2000:]}", flush=True)
        raise optuna.TrialPruned()
    try:
        d = torch.load(env["PPO_OUT"], map_location="cpu", weights_only=False)
        s = float(d["score"])
    except Exception:
        raise optuna.TrialPruned()
    print(f"trial {trial.number}: score={s:.2f} params={trial.params}", flush=True)
    return s


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=16, n_jobs=6)
    print("BEST", study.best_value, study.best_params, flush=True)
