"""Record (obs -> action-taken) pairs along a table+fallback policy's own rollouts."""
import sys

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import obstacles_for_seed  # noqa: E402
from table_eval import fallback_action  # noqa: E402

TABLE_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
OUT = sys.argv[2]


def qkey(obs):
    return tuple(torch.tensor(obs, dtype=torch.float32).tolist())


table = torch.load(TABLE_PATH, weights_only=True) if TABLE_PATH else {}
if isinstance(table, dict) and table and not isinstance(next(iter(table.values())), int):
    raise SystemExit("table file must map obs->action int")

px, py, total = [], [], 0
for seed in range(100):
    blocked = obstacles_for_seed(8, seed)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    done, last_len, since_eat = False, 1, 0
    while not done:
        o = qkey(obs)
        a = table.get(o)
        if a is None:
            a = fallback_action(env)
        px.append(o)
        py.append(a)
        obs, _, done = env.step(a)
        if len(env.snake) > last_len:
            last_len, since_eat = len(env.snake), 0
        else:
            since_eat += 1
            if since_eat > 2 * 8 * 8:
                done = True
    total += len(env.snake)
print(f"policy mean={total/100:.2f}", flush=True)
X = torch.tensor(px, dtype=torch.float32)
y = torch.tensor(py, dtype=torch.int64)
ux, inv, _ = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
print(f"raw={len(X)} unique={len(ux)}", flush=True)
conf = sum(1 for i in range(len(ux)) if int((inv == i).sum()) > 1 and int(torch.unique(y[inv == i]).numel()) > 1)
print(f"conflicts={conf} ceiling={(len(y)-conf)/len(y):.6f}", flush=True)
maj = torch.empty(len(ux), dtype=torch.int64)
for i in range(len(ux)):
    lab = y[inv == i]
    vals, c = torch.unique(lab, return_counts=True)
    maj[i] = vals[c.argmax()]
torch.save((ux, maj), OUT)
print(f"saved {OUT}", flush=True)
