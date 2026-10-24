"""Collect (head-6 obs -> action) pairs from the cycle-44 teacher on 16x16, all 100 seeds."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("collect16")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

INDICES = (0, 1, 2, 3, 4, 5)

teacher = torch.nn.Sequential(torch.nn.Linear(6, 44), torch.nn.ReLU(), torch.nn.Linear(44, 4))
teacher.load_state_dict(torch.load(
    "/Users/williamdennis/.hermes/cache/scratch/snake16/cycle44.pt",
    map_location="cpu", weights_only=True))
teacher.eval()

pairs_x, pairs_y = [], []
lengths = []
with torch.no_grad():
    for seed in range(100):
        env = SnakeEnvironment(size=16, seed=seed, max_moves=65536,
                               input_indices=INDICES, blocked=(), loop_on_full=False)
        obs = env.reset()
        done = False
        ticks = 0
        last_len = len(env.snake)
        since_eat = 0
        while not done:
            a = teacher(torch.tensor(obs, dtype=torch.float32)[None]).argmax(1).item()
            pairs_x.append(obs)
            pairs_y.append(a)
            obs, _, done = env.step(a)
            ticks += 1
            if len(env.snake) > last_len:
                last_len = len(env.snake)
                since_eat = 0
            else:
                since_eat += 1
                if since_eat > 2 * 16 * 16:
                    done = True
        lengths.append(len(env.snake))

import statistics
logger.info("fills: %d/100 mean_len=%.2f", sum(1 for L in lengths if L == 256), statistics.fmean(lengths))

X = torch.tensor(pairs_x, dtype=torch.float32)
y = torch.tensor(pairs_y, dtype=torch.int64)
logger.info("raw pairs: %d", len(X))
ux, inv, cnt = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
logger.info("unique: %d dup-extra: %d", len(ux), int((cnt - 1).sum()))
conf = 0
for i in range(len(ux)):
    m = (inv == i)
    if int(m.sum()) > 1 and int(torch.unique(y[m]).numel()) > 1:
        conf += 1
logger.info("conflicted groups: %d ceiling=%.6f", conf, (len(y) - conf) / len(y))
torch.save((ux, torch.stack([y[inv == i][0] for i in range(len(ux))])), "placeholder")
# rebuild majority labels properly
maj = torch.empty(len(ux), dtype=torch.int64)
for i in range(len(ux)):
    lab = y[inv == i]
    vals, c = torch.unique(lab, return_counts=True)
    maj[i] = vals[c.argmax()]
torch.save((ux, maj), "/Users/williamdennis/.hermes/cache/scratch/snake16/pairs16.pt")
logger.info("saved pairs16.pt")
