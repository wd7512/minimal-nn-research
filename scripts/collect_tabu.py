"""Tabu-escape teacher: BFS+safety normally, least-visited safe move when looping."""
import logging
import sys
from collections import Counter, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tabu")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed  # noqa: E402
from teacher import teacher_action, DX  # noqa: E402

REV = {0: 2, 2: 0, 1: 3, 3: 1}


def neighbors(cell):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < 8 and 0 <= ny < 8:
            yield (nx, ny)


def bfs_path(start, goal_test, blocked):
    if goal_test(start):
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in neighbors(c):
            if n in prev or n in blocked:
                continue
            prev[n] = c
            if goal_test(n):
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(n)
    return None


def safe_cands(env):
    head = env.snake[0]
    food = env.food
    body = set(env.snake[1:])
    blocked = set(env.blocked)
    tail = env.snake[-1]
    rev = ({v: k for k, v in DX.items()}[env.direction] + 2) % 4
    out = []
    for a, (dx, dy) in DX.items():
        if a == rev:
            continue
        cell = (head[0] + dx, head[1] + dy)
        if not (0 <= cell[0] < 8 and 0 <= cell[1] < 8):
            continue
        if cell in blocked:
            continue
        eats = food is not None and cell == food
        if cell in body and not (cell == tail and not eats):
            continue
        occ = blocked | (body | {head}) - ({tail} if not eats else set())
        tail_ok = bfs_path(cell, lambda c: c == tail, occ) is not None
        if tail_ok:
            out.append((a, cell, eats))
    return out


def tabu_escape(env, counts):
    cands = safe_cands(env)
    if not cands:
        return 0
    food = env.food
    best, best_key = None, None
    for a, cell, eats in cands:
        nb = list(env.snake)
        nb.insert(0, cell)
        if not eats:
            nb.pop()
        k = (tuple(nb), food if not eats else None)
        c = counts.get(k, 0)
        kk = (c, 0 if eats else 1, a)
        if best_key is None or kk < best_key:
            best, best_key = a, kk
    for a, cell, eats in cands:
        nb = list(env.snake)
        nb.insert(0, cell)
        if not eats:
            nb.pop()
        counts[(tuple(nb), food if not eats else None)] += 0  # touch
    return best


px, py, lengths, won = [], [], [], 0
for seed in range(100):
    blocked = obstacles_for_seed(8, seed)
    target = 64 - len(blocked)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    counts = Counter()
    done, last_len, since_eat = False, 1, 0
    while not done:
        key = (tuple(env.snake), env.food)
        if counts[key] >= 3:
            a = tabu_escape(env, counts)
        else:
            a = teacher_action(env, counts, None)
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
logger.info("fills=%d/100 won=%d mean=%.2f",
            sum(1 for sd, L in enumerate(lengths) if L == 64 - len(obstacles_for_seed(8, sd))),
            won, statistics.fmean(lengths))
X = torch.tensor(px, dtype=torch.float32)
y = torch.tensor(py, dtype=torch.int64)
ux, inv, cnt = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
logger.info("raw=%d unique=%d", len(X), len(ux))
conf = sum(1 for i in range(len(ux)) if int((inv == i).sum()) > 1 and int(torch.unique(y[inv == i]).numel()) > 1)
logger.info("conflicts=%d ceiling=%.6f", conf, (len(y) - conf) / len(y))
maj = torch.empty(len(ux), dtype=torch.int64)
for i in range(len(ux)):
    lab = y[inv == i]
    vals, c = torch.unique(lab, return_counts=True)
    maj[i] = vals[c.argmax()]
torch.save((ux, maj), "/Users/williamdennis/.hermes/cache/scratch/snake16/pairsTabu.pt")
logger.info("saved")
