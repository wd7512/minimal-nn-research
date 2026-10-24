"""Optuna round 2: narrowed around trial-5, warm-started from ppo-512 (16.73)."""
import os
import subprocess

import optuna
import torch

PY = "/Users/williamdennis/repos/minimal-nn-competition/.venv/bin/python"
SCRIPT = "/Users/williamdennis/.hermes/cache/scratch/ppo_obs.py"
SEEDLIST = ",".join(map(str, range(300)))
INIT = "/tmp/ppo-obs/best-full.pt"


def objective(trial):
    env = dict(os.environ)
    env.update({
        "PPO_H": "512",
        "PPO_INIT": INIT,
        "PPO_STEPS": "300000",
        "PPO_NSTEP": "2048",
        "PPO_EVAL_EVERY": "25",
        "PPO_LR": str(trial.suggest_float("lr", 5e-5, 5e-4, log=True)),
        "PPO_CLIP": str(trial.suggest_float("clip", 0.08, 0.25)),
        "PPO_GAMMA": str(trial.suggest_float("gamma", 0.992, 0.997)),
        "PPO_ENT": str(trial.suggest_float("ent", 1e-3, 2e-2, log=True)),
        "PPO_SEEDLIST": SEEDLIST,
        "PPO_OUT": f"/tmp/ppo-obs/r2-trial-{trial.number}.pt",
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
    study.optimize(objective, n_trials=12, n_jobs=4)
    print("BEST", study.best_value, study.best_params, flush=True)
