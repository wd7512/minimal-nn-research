"""Greedy first-divergence repair of the tour-majority table (dense length objective)."""
import logging
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("repair")

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.engine import DIRECTIONS  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import (  # noqa: E402
    obstacles_for_seed, find_closed_tour)
from table_eval import fallback_action  # noqa: E402

OUT = "/Users/williamdennis/.hermes/cache/scratch/snake16/table_repair.pt"


def qkey(obs):
    return tuple(round(float(v), 6) for v in obs)


# tour oracle per seed for divergence detection
ORACLE = {}
for s in range(100):
    blocked = obstacles_for_seed(8, s)
    tour = find_closed_tour(8, blocked)
    pos = {c: i for i, c in enumerate(tour)}
    ORACLE[s] = (tour, pos)


def oracle_action(seed, head):
    tour, pos = ORACLE[seed]
    nxt = tour[(pos[head] + 1) % len(tour)]
    return DIRECTIONS.index((nxt[0] - head[0], nxt[1] - head[1]))


def rollout(seed, table, cap_moves=16384):
    blocked = obstacles_for_seed(8, seed)
    target = 64 - len(blocked)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=cap_moves, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    done, ticks, last_len, since_eat = False, 0, 1, 0
    divs = []
    while not done:
        o = qkey(obs)
        a = table.get(o)
        if a is None:
            a = fallback_action(env)
        if not divs and a != oracle_action(seed, env.snake[0]):
            divs.append((o, oracle_action(seed, env.snake[0]), a))
        obs, _, done = env.step(a)
        ticks += 1
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * 8 * 8:
                done = True
    won = bool(getattr(env, "won", False))
    ok = len(env.snake) == target and won
    return ok, len(env.snake), ticks, divs


def full_eval(table):
    passed, total, fails = 0, 0, []
    for s in range(100):
        ok, ln, _, _ = rollout(s, table)
        passed += ok
        total += ln
        if not ok:
            fails.append(s)
    return passed, total, fails


X, y = torch.load("/Users/williamdennis/.hermes/cache/scratch/snake16/pairsTeacher.pt",
                   map_location="cpu", weights_only=True)
by_obs = {}
for i in range(len(X)):
    by_obs.setdefault(qkey(X[i].tolist()), []).append(int(y[i]))
table = {o: Counter(v).most_common(1)[0][0] for o, v in by_obs.items()}
logger.info("table size %d", len(table))

passed, total, fails = full_eval(table)
logger.info("start: %d/100 total=%d", passed, total)
okseeds = set(range(100)) - set(fails)
torch.save(table, OUT)

for it in range(200):
    improved = False
    _, _, fails = full_eval(table)
    for s in fails:
        ok, ln, _, divs = rollout(s, table)
        if not divs:
            continue
        o, oa, cur = divs[0]
        for a in [oa] + [x for x in range(4) if x not in (cur, oa)]:
            table[o] = a
            p2, t2, f2 = full_eval(table)
            ok2 = set(range(100)) - set(f2)
            if (p2, t2) > (passed, total) and okseeds.issubset(ok2):
                passed, total = p2, t2
                okseeds = ok2
                logger.info("iter %d: seed %d o-fix a=%d -> %d/100 total=%d", it, s, a, p2, t2)
                torch.save(table, OUT)
                improved = True
                break
            table[o] = cur
        if improved:
            break
    if not improved:
        logger.info("iter %d: plateau at %d/100 total=%d", it, passed, total)
        break
logger.info("FINAL %d/100 total=%d", passed, total)
