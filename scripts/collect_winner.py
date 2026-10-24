"""Collect (obs, action) pairs from a winner MLP's own rollouts for BC distillation."""
import os
import sys

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/submit-eval/src")
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeObstacles8x8Benchmark  # noqa: E402

CKPT = os.environ.get("CW_CKPT", "/tmp/ppo-obs/r2-plain.pt")
OUT = os.environ.get("CW_OUT", "/tmp/ppo-obs/pairs-r2.pt")
NSEEDS = int(os.environ.get("CW_NSEEDS", "300"))

BM = SnakeObstacles8x8Benchmark()
d = torch.load(CKPT, map_location="cpu", weights_only=False)
H = d["H"] if isinstance(d["H"], list) else [d["H"]]
sd = {k: d[k] for k in d if k not in ("H", "score")}


def forward(o):
    h = o
    for i in range(len(H)):
        h = torch.relu(h @ sd[f"{2*i}.weight"].t() + sd[f"{2*i}.bias"])
    return h @ sd[f"{2*len(H)}.weight"].t() + sd[f"{2*len(H)}.bias"]


@torch.no_grad()
def main():
    all_o, all_a = [], []
    for seed in range(NSEEDS):
        env = BM._environment(seed, None)
        obs = env.reset()
        last_len, since_eat = len(env.snake), 0
        while True:
            o = torch.tensor(obs, dtype=torch.float32)
            a = int(torch.argmax(forward(o)))
            all_o.append(o)
            all_a.append(a)
            obs, _, done = env.step(a)
            if len(env.snake) > last_len:
                last_len = len(env.snake)
                since_eat = 0
            else:
                since_eat += 1
                if since_eat > 128:
                    break
            if done:
                break
    torch.save({"obs": torch.stack(all_o), "act": torch.tensor(all_a), "H": H}, OUT)
    print(f"saved {len(all_a)} pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
