"""Error map for a saved best.pt on serpentine pairs."""
import sys

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402

X, y = torch.load(
    "/Users/williamdennis/.hermes/cache/scratch/snake16/pairs16serp.pt",
    map_location="cpu",
    weights_only=True,
)
model = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
model.load_state_dict(
    torch.load(sys.argv[1], map_location="cpu", weights_only=True)
)
model.eval()
names = ["up", "right", "down", "left"]
with torch.no_grad():
    pred = model(X).argmax(1)
wrong = (pred != y).nonzero().flatten().tolist()
print(f"wrong {len(wrong)}/256")
env = SnakeEnvironment(
    size=16, seed=0, max_moves=8, input_indices=(0, 1, 2, 3, 4, 5), blocked=(), loop_on_full=False
)
obs_of = {}
for yy in range(16):
    for xx in range(16):
        env.snake = [(xx, yy)]
        obs_of[tuple(env.observation())] = (xx, yy)
Xl = [tuple(v) for v in X.tolist()]
grid = [["." for _ in range(16)] for _ in range(16)]
obs_list = []
for yy in range(16):
    for xx in range(16):
        env.snake = [(xx, yy)]
        obs_list.append(((xx, yy), torch.tensor(env.observation(), dtype=torch.float32)))
for i in wrong:
    xi = X[i]
    cell = None
    for c, o in obs_list:
        if torch.allclose(xi, o, atol=1e-6):
            cell = c
            break
    print(f"pair={i} cell={cell} true={names[y[i]]} pred={names[pred[i]]}")
    if cell:
        xx, yy = cell
        grid[yy][xx] = "?"
print("map (?=wrong cell):")
for row in grid:
    print("".join(row))
