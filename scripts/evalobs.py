"""Score a saved obstacle MLP on the dev snake-obstacles-8x8 benchmark."""
import statistics
import sys

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed  # noqa: E402

path, hidden = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256
sd = torch.load(path, map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
model = torch.nn.Sequential(torch.nn.Linear(24, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 4))
model.load_state_dict(sd)
model.eval()

lengths, passed = [], 0
with torch.no_grad():
    for seed in range(100):
        blocked = obstacles_for_seed(8, seed)
        target = 64 - len(blocked)
        env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                               blocked=blocked, loop_on_full=True)
        obs = env.reset()
        done, last_len, since_eat = False, 1, 0
        while not done:
            a = model(torch.tensor(obs, dtype=torch.float32)[None]).argmax(1).item()
            obs, _, done = env.step(a)
            if len(env.snake) > last_len:
                last_len, since_eat = len(env.snake), 0
            else:
                since_eat += 1
                if since_eat > 2 * 8 * 8:
                    done = True
        lengths.append(len(env.snake))
        if len(env.snake) == target and getattr(env, "won", False):
            passed += 1
print(f"{path}: passed={passed}/100 mean={statistics.fmean(lengths):.2f} min={min(lengths)}")
