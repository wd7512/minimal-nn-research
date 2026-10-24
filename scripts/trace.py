"""First-divergence trace: teacher vs memorized net on one seed."""
import sys
from collections import Counter

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import (  # noqa: E402
    obstacles_for_seed, find_closed_tour)
from teacher import teacher_action  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
model_path = sys.argv[2]

blocked = obstacles_for_seed(8, seed)
tour = find_closed_tour(8, blocked)
succ = {tour[i]: tour[(i + 1) % len(tour)] for i in range(len(tour))}

sd = torch.load(model_path, map_location="cpu", weights_only=True)
model = torch.nn.Sequential(torch.nn.Linear(24, 512), torch.nn.ReLU(), torch.nn.Linear(512, 4))
model.load_state_dict(sd)
model.eval()

env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                       blocked=blocked, loop_on_full=True)
obs = env.reset()
counts = Counter()
with torch.no_grad():
    for t in range(40):
        ta = teacher_action(env, counts, succ)
        na = model(torch.tensor(obs, dtype=torch.float32)[None]).argmax(1).item()
        flag = "" if ta == na else "  <-- DIVERGE"
        print(f"t={t} head={env.snake[0]} food={env.food} teacher={ta} net={na}{flag}",
              flush=True)
        if ta != na:
            break
        obs, _, done = env.step(ta)
        if done:
            print("teacher done")
            break
