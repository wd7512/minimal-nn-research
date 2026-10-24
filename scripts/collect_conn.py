"""Connectivity-preserving greedy teacher: eat while keeping free space connected."""
import logging
import sys
from collections import Counter, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("conngreedy")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment
from minimal_nn_competition.benchmarks.snake.engine import DIRECTIONS
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed

DX = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
REV = {v: k for k, v in {0: 2, 2: 0, 1: 3, 3: 1}.items()}


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


def connected(free):
    """All free cells in one component?"""
    free = set(free)
    if not free:
        return True
    stack = [next(iter(free))]
    seen = set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        for n in neighbors(c):
            if n in free and n not in seen:
                stack.append(n)
    return seen == free


def teacher_action(env, counts):
    head = env.snake[0]
    food = env.food
    body = set(env.snake[1:])
    blocked = set(env.blocked)
    tail = env.snake[-1]
    rev = ({0: 2, 2: 0, 1: 3, 3: 1})[({v: k for k, v in DX.items()}[env.direction])]
    cands = []
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
        cands.append((a, cell, eats))
    if not cands:
        return 0
    key = (tuple(env.snake), food)
    counts[key] += 1
    stalled = counts[key] >= 4
    scored = []
    for a, cell, eats in cands:
        nb = set(body)
        nb.add(head)
        if not eats:
            nb.discard(tail)
        occ = blocked | nb
        tail_ok = bfs_path(cell, lambda c: c == tail, occ) is not None
        food_ok = True
        if food is not None and not eats:
            occ2 = set(occ)
            food_ok = bfs_path(cell, lambda c: c == food, occ2) is not None
        free = {(x, y) for x in range(8) for y in range(8)} - blocked - nb
        free.add(cell)
        conn = connected(free)
        dist = abs(cell[0] - food[0]) + abs(cell[1] - food[1]) if food else 0
        tier = (0 if eats and tail_ok and conn else
                1 if tail_ok and food_ok and conn else
                2 if tail_ok and conn else
                3 if tail_ok else 4)
        scored.append((tier, -1 if eats else dist, a))
    scored.sort()
    if stalled:
        eat_now = [s for s in scored if s[1] == -1]
        if eat_now:
            return eat_now[0][2]
        return scored[0][2]
    return scored[0][2]


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
        a = teacher_action(env, counts)
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
torch.save((ux, maj), "/Users/williamdennis/.hermes/cache/scratch/snake16/pairsConn.pt")
logger.info("saved")
