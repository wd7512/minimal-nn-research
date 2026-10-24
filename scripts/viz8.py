"""Visualize an 8x8 winner's direction map + dump its cycle."""
import sys

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402

WHICH = sys.argv[1] if len(sys.argv) > 1 else "hamiltonian-3-h6"
cfg_path = f"/tmp/{WHICH}.json"
pt_path = f"/tmp/{WHICH}.pt"

import json

cfg = json.load(open(cfg_path))
print(cfg.get("description"))
model = torch.nn.Sequential(
    torch.nn.Linear(cfg["input_size"], cfg["hidden_sizes"][0]),
    torch.nn.ReLU(),
    torch.nn.Linear(cfg["hidden_sizes"][0], cfg["output_size"]),
)
model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
model.eval()
names = ["up", "right", "down", "left"]
glyph = {"up": "^", "right": ">", "down": "v", "left": "<"}
env = SnakeEnvironment(
    size=8, seed=0, max_moves=8, input_indices=tuple(cfg.get("input_indices", [0, 1, 2, 3, 4, 5])),
    blocked=(), loop_on_full=False,
)
arrows = []
with torch.no_grad():
    for yy in range(8):
        row = ""
        for xx in range(8):
            env.snake = [(xx, yy)]
            a = model(torch.tensor(env.observation(), dtype=torch.float32)[None]).argmax(1).item()
            row += glyph[names[a]]
        arrows.append(row)
print("\n".join(arrows))
