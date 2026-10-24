"""GA evolving MLP weights directly on closed-loop obstacle-snake score."""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import torch

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/submit-eval/src")
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeObstacles8x8Benchmark  # noqa: E402

H = int(os.environ.get("GA_H", "16"))
P = int(os.environ.get("GA_POP", "64"))
G = int(os.environ.get("GA_GENS", "150"))
K = int(os.environ.get("GA_SEEDS", "16"))
OUT = os.environ.get("GA_OUT", "/tmp/ga-obs/best.pt")
SEED0 = int(os.environ.get("GA_SEED", "0"))
INIT = os.environ.get("GA_INIT", "")  # optional .pt to seed population

def parse_seeds(spec, k):
    if spec:
        out = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out
    return list(range(0, 100, 100 // k))[:k]


BM = SnakeObstacles8x8Benchmark()
SUBSEEDS = parse_seeds(os.environ.get("GA_SEEDLIST", ""), K)

def fit_one(args):
    genome, seeds = args
    w1, b1, w2, b2 = decode(genome)
    with torch.no_grad():
        total = 0
        for seed in seeds:
            env = BM._environment(seed, None)
            obs = env.reset()
            for _ in range(512):
                o = torch.tensor(obs, dtype=torch.float32)
                h = torch.relu(o @ w1 + b1)
                a = int(torch.argmax(h @ w2 + b2))
                obs, _, done = env.step(a)
                if done:
                    break
            total += len(env.snake)
    return total / len(seeds)


WORKERS = int(os.environ.get("GA_WORKERS", "8"))
sizes = [a * b if isinstance(a, int) and not isinstance(b, tuple) else 0 for a, b in []]
sizes = [24 * H, H, H * 4, 4]
N = sum(sizes)


def decode(genome):
    t = torch.tensor(genome, dtype=torch.float32)
    i0 = 24 * H
    i1 = i0 + H
    i2 = i1 + H * 4
    w1 = t[0:i0].reshape(24, H)
    b1 = t[i0:i1]
    w2 = t[i1:i2].reshape(H, 4)
    b2 = t[i2:i2 + 4]
    return w1, b1, w2, b2


@torch.no_grad()
def fitness(genome):
    w1, b1, w2, b2 = decode(genome)
    total = 0
    for seed in SUBSEEDS:
        env = BM._environment(seed, None)
        obs = env.reset()
        n = 0
        for _ in range(512):
            o = torch.tensor(obs, dtype=torch.float32)
            h = torch.relu(o @ w1 + b1)
            a = int(torch.argmax(h @ w2 + b2))
            obs, _, done = env.step(a)
            n += 1
            if done:
                break
        total += len(env.snake)
    return total / len(SUBSEEDS)


def rand_genome(scale=0.5):
    return [random.gauss(0, scale) for _ in range(N)]


def main():
    random.seed(SEED0)
    pop = []
    if INIT and os.path.exists(INIT):
        obj = torch.load(INIT, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "genome" in obj:
            base = list(obj["genome"])
        else:
            sd = obj
            base = sd["0.weight"].t().reshape(-1).tolist() + sd["0.bias"].tolist() + sd["2.weight"].t().reshape(-1).tolist() + sd["2.bias"].tolist()
        assert len(base) == N, (len(base), N)
        pop.append(base)
    while len(pop) < P:
        pop.append(rand_genome())
    best = None
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for g in range(G):
            scores = list(pool.map(fit_one, [(x, SUBSEEDS) for x in pop]))
            scored = sorted(zip(scores, pop), key=lambda t: -t[0])
            scored = [(s, x) for s, x in scored]
            if best is None or scored[0][0] > best[0]:
                best = (scored[0][0], list(scored[0][1]))
                torch.save({"genome": best[1], "score": best[0], "H": H}, OUT)
            print(f"gen {g}: best={scored[0][0]:.2f} med={scored[P//2][0]:.2f} t={time.time()-t0:.0f}s", flush=True)
            elite = [x for _, x in scored[:4]]
            nxt = list(elite)
            while len(nxt) < P:
                a, b = max(random.sample(scored, 3))[1], max(random.sample(scored, 3))[1]
                mask = [random.random() < 0.5 for _ in range(N)]
                kid = [aa if m else bb for aa, bb, m in zip(a, b, mask)]
                kid = [v + random.gauss(0, 0.15) if random.random() < 0.3 else v for v in kid]
                nxt.append(kid)
            pop = nxt
    print("DONE", best[0], flush=True)


if __name__ == "__main__":
    main()
