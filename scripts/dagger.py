"""DAgger for obstacles: student rollouts relabeled by the full-state teacher."""
import logging
import sys
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dagger")

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/grind-obstacles")

from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment  # noqa: E402
from minimal_nn_competition.benchmarks.snake.layout import (  # noqa: E402
    obstacles_for_seed, find_closed_tour)
from teacher import teacher_action  # noqa: E402

AGG = "/Users/williamdennis/.hermes/cache/scratch/snake16/agg.pt"
MODEL = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/models/snake-obstacles-8x8/obs-512/model.pt"

torch.manual_seed(0)
agg = defaultdict(Counter)  # qkey -> Counter(action)
X0, y0 = torch.load("/Users/williamdennis/.hermes/cache/scratch/snake16/pairsTeacher.pt",
                     map_location="cpu", weights_only=True)


def qkey(obs):
    return tuple(round(float(v), 6) for v in obs)


for i in range(len(X0)):
    agg[qkey(X0[i].tolist())][int(y0[i])] += 1
logger.info("base unique %d", len(agg))

TOURS = {}
for s in range(100):
    t = find_closed_tour(8, obstacles_for_seed(8, s))
    TOURS[s] = {t[i]: t[(i + 1) % len(t)] for i in range(len(t))}

sd = torch.load(MODEL, map_location="cpu", weights_only=True)
model = torch.nn.Sequential(torch.nn.Linear(24, 512), torch.nn.ReLU(), torch.nn.Linear(512, 4))
model.load_state_dict(sd)
model.eval()


def rollout_collect(seed):
    """Roll out net; return list of (qkey, teacher_label)."""
    blocked = obstacles_for_seed(8, seed)
    env = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                           blocked=blocked, loop_on_full=True)
    obs = env.reset()
    out, done, last_len, since_eat = [], False, 1, 0
    with torch.no_grad():
        while not done:
            a = model(torch.tensor(obs, dtype=torch.float32)[None]).argmax(1).item()
            # relabel current full state with teacher
            e2 = SnakeEnvironment(size=8, seed=seed, max_moves=16384, input_indices=None,
                                  blocked=blocked, loop_on_full=True)
            e2.snake = list(env.snake)
            e2.food = env.food
            e2.direction = env.direction
            lab = teacher_action(e2, Counter(), TOURS[seed])
            out.append((qkey(obs), lab))
            obs, _, done = env.step(a)
            if len(env.snake) > last_len:
                last_len, since_eat = len(env.snake), 0
            else:
                since_eat += 1
                if since_eat > 2 * 8 * 8:
                    done = True
    return out


for rnd in range(1, 7):
    new = 0
    for s in range(100):
        for o, lab in rollout_collect(s):
            if lab is None:
                continue
            agg[o][lab] += 1
            new += 1
    logger.info("round %d: +%d labels, unique %d", rnd, new, len(agg))
    Xu = torch.tensor([list(o) for o in agg], dtype=torch.float32)
    yu = torch.tensor([c.most_common(1)[0][0] for c in agg.values()], dtype=torch.int64)
    torch.save((Xu, yu), AGG)
    logger.info("round %d: saved agg (unique %d)", rnd, len(Xu))
    break  # train separately
