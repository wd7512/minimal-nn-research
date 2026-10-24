"""Failure-mode analysis of the pure-BFS teacher: cap-loop vs dead-end."""
import sys
from collections import Counter

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import (  # noqa: E402
    obstacles_for_seed, find_closed_tour)
from teacher import teacher_action  # noqa: E402

ends = []
for seed in range(100):
    blocked = obstacles_for_seed(8, seed)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    counts = Counter()
    tour = find_closed_tour(8, blocked)
    succ = {tour[i]: tour[(i + 1) % len(tour)] for i in range(len(tour))}
    done, last_len, since_eat = False, 1, 0
    while not done:
        a = teacher_action(env, counts, None)
        obs, _, done = env.step(a)
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * 8 * 8:
                done = True
    ends.append((seed, len(env.snake), "CAP" if since_eat > 128 else "END"))
caps = [e for e in ends if e[2] == "CAP"]
print(f"caps={len(caps)} ends={100-len(caps)}", flush=True)
print("cap lengths:", sorted(e[1] for e in caps), flush=True)
print("end lengths:", sorted(e[1] for e in ends if e[2] == "END"), flush=True)
