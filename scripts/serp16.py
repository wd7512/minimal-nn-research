"""Analytic serpentine-cycle teacher for 16x16 (column-0 return spine)."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("serp16")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

SIZE = 16
# DIRECTIONS: 0=up(0,-1) 1=right(1,0) 2=down(0,1) 3=left(-1,0)  [x=col, y=row]
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Build the cycle: row 0 left-to-right, then serpentine columns 15..1, spine col 0 up.
path = [(x, 0) for x in range(SIZE)]
for k, col in enumerate(range(SIZE - 1, 0, -1)):
    rows = range(1, SIZE) if k % 2 == 0 else range(SIZE - 1, 0, -1)
    last = path[-1]
    first = (col, rows[0] if not isinstance(rows, range) else list(rows)[0])
    path += [(col, r) for r in rows]
path.append((0, SIZE - 1))
path += [(0, y) for y in range(SIZE - 2, 0, -1)]

assert len(path) == SIZE * SIZE, len(path)
assert len(set(path)) == SIZE * SIZE, "not simple"
succ = {path[i]: path[(i + 1) % len(path)] for i in range(len(path))}
# successor must be adjacent
for a, b in succ.items():
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)


def action_for(head):
    nx, ny = succ[head]
    d = (nx - head[0], ny - head[1])
    return DIRS.index(d)


import os

INDICES = tuple(int(i) for i in os.environ.get("INDICES", "0,1,2,3,4,5").split(","))
OUT = os.environ.get("OUT", "/Users/williamdennis/.hermes/cache/scratch/snake16/pairs16serp.pt")
pairs_x, pairs_y, lengths = [], [], []
for seed in range(100):
    env = SnakeEnvironment(size=SIZE, seed=seed, max_moves=65536,
                           input_indices=INDICES, blocked=(), loop_on_full=False)
    obs = env.reset()
    done, ticks, last_len, since_eat = False, 0, 1, 0
    while not done:
        a = action_for(env.snake[0])
        pairs_x.append(obs)
        pairs_y.append(a)
        obs, _, done = env.step(a)
        ticks += 1
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * SIZE * SIZE:
                done = True
    lengths.append(len(env.snake))

import statistics
logger.info("fills: %d/100 mean_len=%.2f", sum(1 for L in lengths if L == SIZE * SIZE), statistics.fmean(lengths))

X = torch.tensor(pairs_x, dtype=torch.float32)
y = torch.tensor(pairs_y, dtype=torch.int64)
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
logger.info("saved")
