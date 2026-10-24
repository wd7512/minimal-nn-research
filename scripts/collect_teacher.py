"""Collect (24-obs -> BFS-teacher action) pairs on snake-obstacles-8x8."""
import logging
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("collectT")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed, find_closed_tour
from teacher import teacher_action

import os

OUT = os.environ.get("OUT", "/Users/williamdennis/.hermes/cache/scratch/snake16/pairsTeacher.pt")
USE_TOUR = os.environ.get("USE_TOUR", "1") == "1"
px, py, lengths, won = [], [], [], 0
for seed in range(100):
    blocked = obstacles_for_seed(8, seed)
    target = 64 - len(blocked)
    tour = find_closed_tour(8, blocked)
    tpos = {c: i for i, c in enumerate(tour)}
    succ = {tour[i]: tour[(i + 1) % len(tour)] for i in range(len(tour))}
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    counts = Counter()
    tguide = succ if USE_TOUR else None
    done, last_len, since_eat = False, 1, 0
    while not done:
        a = teacher_action(env, counts, tguide)
        px.append(tuple(obs))
        py.append(a)
        obs, _, done = env.step(a)
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * 8 * 8:
                done = True
    lengths.append(len(env.snake))
    won += bool(getattr(env, "won", False))

import statistics
logger.info("teacher fills=%d/100 won=%d mean=%.2f",
            sum(1 for sd, L in enumerate(lengths) if L == 64 - len(obstacles_for_seed(8, sd))),
            won, statistics.fmean(lengths))

X = torch.tensor(px, dtype=torch.float32)
y = torch.tensor(py, dtype=torch.int64)
logger.info("raw pairs: %d", len(X))
ux, inv, cnt = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
logger.info("unique: %d dup-extra: %d", len(ux), int((cnt - 1).sum()))
conf = sum(1 for i in range(len(ux)) if int((inv == i).sum()) > 1 and int(torch.unique(y[inv == i]).numel()) > 1)
logger.info("conflicts: %d ceiling=%.6f", conf, (len(y) - conf) / len(y))
maj = torch.empty(len(ux), dtype=torch.int64)
for i in range(len(ux)):
    lab = y[inv == i]
    vals, c = torch.unique(lab, return_counts=True)
    maj[i] = vals[c.argmax()]
torch.save((ux, maj), OUT)
if os.environ.get("SAVE_RAW"):
    torch.save({"raw_x": X, "raw_y": y}, os.environ["SAVE_RAW"])
    logger.info("saved raw")
logger.info("saved")
