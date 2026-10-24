"""Majority-table closed-loop eval with quantized keys (consistent build/lookup)."""
import sys
from collections import Counter

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed  # noqa: E402

sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")
from table_eval import fallback_action  # noqa: E402


def qkey(obs):
    return tuple(round(float(v), 6) for v in obs)


X, y = torch.load("/Users/williamdennis/.hermes/cache/scratch/snake16/pairsobs.pt",
                   map_location="cpu", weights_only=True)
by_obs = {}
for i in range(len(X)):
    by_obs.setdefault(qkey(X[i].tolist()), []).append(int(y[i]))
table = {o: Counter(v).most_common(1)[0][0] for o, v in by_obs.items()}
print(f"table size {len(table)}")

passed, moves, fails = 0, 0, []
for seed in range(100):
    blocked = obstacles_for_seed(8, seed)
    target = 64 - len(blocked)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    done, ticks, last_len, since_eat = False, 0, 1, 0
    while not done:
        a = table.get(qkey(obs))
        if a is None:
            a = fallback_action(env)
        obs, _, done = env.step(a)
        ticks += 1
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * 8 * 8:
                done = True
    won = bool(getattr(env, "won", False))
    moves += ticks
    if len(env.snake) == target and won:
        passed += 1
    else:
        fails.append((seed, len(env.snake), ticks, won))
print(f"majority table: {passed}/100 moves={moves}")
print("fails:", fails[:20])
