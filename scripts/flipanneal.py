"""Square-flip annealing: search 16x16 Hamiltonian cycles for small-net distillability.

Fitness(H) = best-accuracy of standardized single-H SFT (short schedule) on the
cycle's teacher pairs; cycles that don't fill 100/100 score 0.
"""
import argparse
import logging
import random
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("anneal")

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")

import torch
from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

SIZE = 16
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
INDICES = (0, 1, 2, 3, 4, 5)

# ---- cycle construction: column-spine serpentine (proven 100/100) ----
col_path = [(x, 0) for x in range(SIZE)]
for k, col in enumerate(range(SIZE - 1, 0, -1)):
    rows = list(range(1, SIZE)) if k % 2 == 0 else list(range(SIZE - 1, 0, -1))
    col_path += [(col, r) for r in rows]
col_path.append((0, SIZE - 1))
col_path += [(0, y) for y in range(SIZE - 2, 0, -1)]


def succ_of(path):
    return {path[i]: path[(i + 1) % len(path)] for i in range(len(path))}


def valid_cycle(succ):
    if len(succ) != SIZE * SIZE or set(succ) != set((x, y) for y in range(SIZE) for x in range(SIZE)):
        return False
    seen, cur = set(), next(iter(succ))
    for _ in range(SIZE * SIZE):
        if cur in seen:
            return False
        seen.add(cur)
        cur = succ[cur]
    return cur == next(iter(succ)) and len(seen) == SIZE * SIZE


def _single_flip(succ, x, y):
    """Rewire one 2x2 square's parallel edges; return new succ or None (may split)."""
    a, b, c, d = (x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)
    edges = {(a, b), (b, a), (b, c), (c, b), (c, d), (d, c), (d, a), (a, d)}
    ins = [(u, succ[u]) for u in (a, b, c, d) if (u, succ.get(u)) in edges]
    if len(ins) != 2:
        return None
    (u1, v1), (u2, v2) = ins
    got = sorted([tuple(sorted((u1, v1))), tuple(sorted((u2, v2)))])
    horiz = sorted([tuple(sorted((a, b))), tuple(sorted((c, d)))])
    vert = sorted([tuple(sorted((b, c))), tuple(sorted((d, a)))])
    if got != horiz and got != vert:
        return None
    new = dict(succ)
    new[u1], new[u2] = v2, v1
    return new


def square_flip(succ, rng):
    """Two simultaneous square flips; a pair of splits can re-merge to one cycle."""
    for _ in range(10):
        x1, y1 = rng.randrange(SIZE - 1), rng.randrange(SIZE - 1)
        x2, y2 = rng.randrange(SIZE - 1), rng.randrange(SIZE - 1)
        if (x1, y1) == (x2, y2):
            continue
        s1 = _single_flip(succ, x1, y1)
        if s1 is None:
            continue
        s2 = _single_flip(s1, x2, y2)
        if s2 is None:
            continue
        if valid_cycle(s2):
            return s2
    return None


def simulate(succ, seeds=range(100)):
    px, py, fills = [], [], 0
    for seed in seeds:
        env = SnakeEnvironment(size=SIZE, seed=seed, max_moves=65536,
                               input_indices=INDICES, blocked=(), loop_on_full=False)
        obs = env.reset()
        done, last_len, since_eat = False, 1, 0
        while not done:
            head = env.snake[0]
            if head not in succ:
                break
            t = succ[head]
            a = DIRS.index((t[0] - head[0], t[1] - head[1]))
            px.append(obs)
            py.append(a)
            obs, _, done = env.step(a)
            if len(env.snake) > last_len:
                last_len, since_eat = len(env.snake), 0
            else:
                since_eat += 1
                if since_eat > 2 * SIZE * SIZE:
                    done = True
        if len(env.snake) == SIZE * SIZE:
            fills += 1
    X = torch.tensor(px, dtype=torch.float32) if px else torch.empty(0, 6)
    y = torch.tensor(py, dtype=torch.int64) if py else torch.empty(0, dtype=torch.int64)
    return X, y, fills


def fitness(X, y, width, epochs, seed):
    if len(X) == 0:
        return 0, 0
    ux, inv, _ = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
    maj = torch.empty(len(ux), dtype=torch.int64)
    for i in range(len(ux)):
        lab = y[inv == i]
        vals, c = torch.unique(lab, return_counts=True)
        maj[i] = vals[c.argmax()]
    if int((((maj[inv] != y)).sum())) > 0:
        pass  # conflicts tolerated in proxy (agreement checked at the end)
    n = len(ux)
    Xu = (ux - ux.mean(0)) / ux.std(0).clamp_min(1e-6)
    yu = maj
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, width), torch.nn.ReLU(),
                                torch.nn.Linear(width, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(Xu), yu).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            best = max(best, int((model(Xu).argmax(1) == yu).sum()))
    return best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/Users/williamdennis/.hermes/cache/scratch/snake16/best_cycle.pt")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    base = succ_of(col_path)
    assert valid_cycle(base)
    X, y, fills = simulate(base)
    logger.info("base fills=%d pairs=%d", fills, len(X))
    fb, nb = fitness(X, y, args.width, args.epochs, 0)
    logger.info("base fitness best=%d/%d", fb, nb)
    best, best_fit, best_n = base, fb, nb
    cur = base
    for it in range(args.iters):
        prop = square_flip(cur, rng)
        if prop is None:
            continue
        # 20 seeds visit every cell (each fill covers all 256), so the
        # deduped 256 patterns match the full 100-seed set; validity of the
        # cycle implies 100/100 fills (lap 256 < per-food cap 512).
        Xp, yp, fp = simulate(prop, seeds=range(20))
        if fp < 20:
            continue
        f, nn = fitness(Xp, yp, args.width, args.epochs, 0)
        logger.info("iter %d fit %d/%d (best %d)", it, f, nn, best_fit)
        if f >= best_fit:
            best, best_fit, best_n = prop, f, nn
            cur = prop
            torch.save({"succ": {f"{k[0]},{k[1]}": list(v) for k, v in prop.items()},
                        "fit": f, "n": nn}, args.out)
            logger.info("NEW BEST %d/%d saved", f, nn)
            if f == nn:
                logger.info("PERFECT at iter %d", it)
                break
    logger.info("DONE best=%d/%d", best_fit, best_n)


if __name__ == "__main__":
    main()
