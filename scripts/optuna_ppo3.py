"""Optuna round 3: optimizer (adam vs muon) x LR, warm-started from 18.71."""
import os
import subprocess

import optuna
import torch

PY = "/Users/williamdennis/repos/minimal-nn-competition/.venv/bin/python"
SCRIPT = "/Users/williamdennis/.hermes/cache/scratch/ppo_obs.py"
SEEDLIST = ",".join(map(str, range(300)))
INIT = "/tmp/ppo-obs/r2-plain.pt"


def objective(trial):
    env = dict(os.environ)
    env.update({
        "PPO_H": "512",
        "PPO_INIT": INIT,
        "PPO_STEPS": "300000",
        "PPO_NSTEP": "2048",
        "PPO_EVAL_EVERY": "25",
        "PPO_OPT": trial.suggest_categorical("opt", ["adam", "muon"]),
        "PPO_LR": str(trial.suggest_float("lr", 1e-5, 5e-2, log=True)),
        "PPO_CLIP": "0.12",
        "PPO_GAMMA": "0.994",
        "PPO_ENT": "0.006",
        "PPO_SEEDLIST": SEEDLIST,
        "PPO_OUT": f"/tmp/ppo-obs/r3-trial-{trial.number}.pt",
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
