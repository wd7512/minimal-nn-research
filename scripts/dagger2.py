"""DAgger: aggregate teacher relabels of student-visited states (persistent counts).

Env: MODEL (net checkpoint to roll out), STATE (counts dict), EXPORT (majority pairs).
 agreggate -> train (snake_train) -> eval -> repeat with new MODEL.
"""
import logging
import os
import sys
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dagger2")

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import (  # noqa: E402
    obstacles_for_seed, find_closed_tour)
from teacher import teacher_action  # noqa: E402

MODEL = os.environ["MODEL"]
STATE = os.environ.get("STATE", "/Users/williamdennis/.hermes/cache/scratch/snake16/agg_counts.pt")
EXPORT = os.environ.get("EXPORT", "/Users/williamdennis/.hermes/cache/scratch/snake16/agg.pt")
HIDDEN = int(os.environ.get("HIDDEN", "512"))


def qkey(obs):
    # Exact float32 key: consistent between torch training tensors and
    # torch.tensor(obs, dtype=torch.float32) at eval. No rounding.
    return tuple(torch.tensor(obs, dtype=torch.float32).tolist())


if os.path.exists(STATE):
    agg = torch.load(STATE, weights_only=False)
    logger.info("loaded state unique %d", len(agg))
else:
    agg = defaultdict(Counter)
    d = torch.load("/Users/williamdennis/.hermes/cache/scratch/snake16/rawTeacher.pt",
                    map_location="cpu", weights_only=True)
    for i in range(len(d["raw_x"])):
        agg[qkey(d["raw_x"][i].tolist())][int(d["raw_y"][i])] += 1
    logger.info("built from raw unique %d", len(agg))

TOURS = {}
for s in range(100):
    t = find_closed_tour(8, obstacles_for_seed(8, s))
    TOURS[s] = {t[i]: t[(i + 1) % len(t)] for i in range(len(t))}

sd = torch.load(MODEL, map_location="cpu", weights_only=True)
layers = [torch.nn.Linear(24, HIDDEN), torch.nn.ReLU()]
model = torch.nn.Sequential(*layers, torch.nn.Linear(HIDDEN, 4))
model.load_state_dict(sd)
model.eval()


def rollout_collect(seed):
    blocked = obstacles_for_seed(8, seed)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    out, done, last_len, since_eat = [], False, 1, 0
    with torch.no_grad():
        while not done:
            a = model(torch.tensor(obs, dtype=torch.float32)[None]).argmax(1).item()
            e2 = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                                  blocked=blocked, loop_on_full=True)
            e2.snake = list(env.snake)
            e2.food = env.food
            e2.direction = env.direction
            lab = teacher_action(e2, Counter(), TOURS[seed])
            out.append((qkey(obs), lab))
            obs, _, done = env.step(a)
            if len(env.snake) > last_len:
                last_len, since_eat = len(env.snake), 0
            else:
                since_eat += 1
                if since_eat > 2 * 8 * 8:
                    done = True
    return out


new = 0
for s in range(100):
    for o, lab in rollout_collect(s):
        agg[o][lab] += 1
        new += 1
logger.info("added %d labels, unique %d", new, len(agg))
torch.save(dict(agg), STATE)
Xu = torch.tensor([list(o) for o in agg], dtype=torch.float32)
yu = torch.tensor([c.most_common(1)[0][0] for c in agg.values()], dtype=torch.int64)
torch.save((Xu, yu), EXPORT)
logger.info("saved state + export (unique %d)", len(Xu))
