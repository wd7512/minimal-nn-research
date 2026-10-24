"""Transposed serpentine teacher (long vertical runs, spine row) + distill check."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("serp16T")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

SIZE = 16
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Column-spine path from serp16, transposed: (x,y) -> (y,x) gives row-spine cycle.
col_path = [(x, 0) for x in range(SIZE)]
for k, col in enumerate(range(SIZE - 1, 0, -1)):
    rows = list(range(1, SIZE)) if k % 2 == 0 else list(range(SIZE - 1, 0, -1))
    col_path += [(col, r) for r in rows]
col_path.append((0, SIZE - 1))
col_path += [(0, y) for y in range(SIZE - 2, 0, -1)]
path = [(y, x) for (x, y) in col_path]
# Traverse reversed so the move from the spawn cell (8,8) matches the
# engine's initial heading (right): the transposed map sends up->left,
# a reversal the engine would ignore.
path = list(reversed(path))

assert len(path) == SIZE * SIZE and len(set(path)) == SIZE * SIZE
succ = {path[i]: path[(i + 1) % len(path)] for i in range(len(path))}
for a, b in succ.items():
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)


def action_for(head):
    nx, ny = succ[head]
    return DIRS.index((nx - head[0], ny - head[1]))


INDICES = (0, 1, 2, 3, 4, 5)
pairs_x, pairs_y, lengths = [], [], []
for seed in range(100):
    env = SnakeEnvironment(size=SIZE, seed=seed, max_moves=65536,
                           input_indices=INDICES, blocked=(), loop_on_full=False)
    obs = env.reset()
    done, last_len, since_eat = False, 1, 0
    while not done:
        a = action_for(env.snake[0])
        pairs_x.append(obs)
        pairs_y.append(a)
        obs, _, done = env.step(a)
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * SIZE * SIZE:
                done = True
    lengths.append(len(env.snake))

import statistics
logger.info("fills: %d/100 mean=%.2f", sum(1 for L in lengths if L == SIZE * SIZE), statistics.fmean(lengths))
X = torch.tensor(pairs_x, dtype=torch.float32)
y = torch.tensor(pairs_y, dtype=torch.int64)
ux, inv, cnt = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
logger.info("unique: %d", len(ux))
conf = sum(1 for i in range(len(ux)) if int((inv == i).sum()) > 1 and int(torch.unique(y[inv == i]).numel()) > 1)
logger.info("conflicts: %d", conf)
maj = torch.empty(len(ux), dtype=torch.int64)
for i in range(len(ux)):
    lab = y[inv == i]
    vals, c = torch.unique(lab, return_counts=True)
    maj[i] = vals[c.argmax()]
torch.save((ux, maj), "/Users/williamdennis/.hermes/cache/scratch/snake16/pairs16serpT.pt")
logger.info("saved")
