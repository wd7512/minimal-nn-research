"""PPO on obstacle snake. Shared trunk + policy/value heads, running-norm folded back at save."""
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/submit-eval/src")
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeObstacles8x8Benchmark  # noqa: E402

HLIST = [int(x) for x in os.environ.get("PPO_H", "256,256").split(",")]
INIT_PT = os.environ.get("PPO_INIT", "")  # state_dict to load policy trunk from
STEPS = int(os.environ.get("PPO_STEPS", "1500000"))
NSTEP = int(os.environ.get("PPO_NSTEP", "1024"))
EPOCHS = 4
MINIB = 256
GAMMA = float(os.environ.get("PPO_GAMMA", "0.995"))
LAM = 0.95
CLIP = float(os.environ.get("PPO_CLIP", "0.2"))
LR = float(os.environ.get("PPO_LR", "3e-4"))
ENT = float(os.environ.get("PPO_ENT", "0.01"))
SHAPE_K = float(os.environ.get("PPO_SHAPE_K", "0.0"))  # >0: reward approaching food
OUT = os.environ.get("PPO_OUT", "/tmp/ppo-obs/best.pt")
EVAL_EVERY = int(os.environ.get("PPO_EVAL_EVERY", "10"))
TRAIN_SEEDS = [int(x) for x in os.environ.get("PPO_SEEDLIST", "").split(",") if x] or list(range(300))

BM = SnakeObstacles8x8Benchmark()
EVAL_SEEDS = list(range(100))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], 24
        for h in HLIST:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.pi = nn.Linear(d, 4)
        self.v = nn.Linear(d, 1)
        if INIT_PT:
            sd = torch.load(INIT_PT, map_location="cpu", weights_only=True)
            n = len(HLIST)
            assert sd["0.weight"].shape[0] == HLIST[0], "INIT hidden mismatch"
            with torch.no_grad():
                self.trunk[0].weight.copy_(sd["0.weight"])
                self.trunk[0].bias.copy_(sd["0.bias"])
                for i in range(1, n):
                    self.trunk[2 * i].weight.copy_(sd[f"{2*i}.weight"])
                    self.trunk[2 * i].bias.copy_(sd[f"{2*i}.bias"])
                self.pi.weight.copy_(sd[f"{2*n}.weight"])
                self.pi.bias.copy_(sd[f"{2*n}.bias"])

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


@torch.no_grad()
def eval_mean(net, mu, sigma):
    tot = 0
    for seed in EVAL_SEEDS:
        env = BM._environment(seed, None)
        obs = env.reset()
        last_length, since_eat = len(env.snake), 0
        while True:
            o = (torch.tensor(obs, dtype=torch.float32) - mu) / sigma
            a = int(torch.argmax(net.pi(net.trunk(o))))
            obs, _, done = env.step(a)
            if len(env.snake) > last_length:
                last_length = len(env.snake)
                since_eat = 0
            else:
                since_eat += 1
                if since_eat > 128:
                    break
            if done:
                break
        tot += len(env.snake)
    return tot / len(EVAL_SEEDS)


def _zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["buf"] = torch.zeros_like(p) if p.ndim == 2 else None

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if p.ndim == 2:
                    buf = self.state[p]["buf"]
                    buf.mul_(group["momentum"]).add_(g)
                    upd = _zeropower_via_newtonschulz5(buf if not group["nesterov"] else g + group["momentum"] * buf)
                    upd = upd * max(p.size(0), p.size(1)) ** 0.5
                    p.add_(upd, alpha=-group["lr"])
                else:
                    p.add_(g, alpha=-group["lr"])


OPT = os.environ.get("PPO_OPT", "adam")


def main():
    torch.manual_seed(0)
    random.seed(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    net = Net()
    opt = Muon(net.parameters(), lr=LR) if OPT == "muon" else torch.optim.Adam(net.parameters(), lr=LR)
    mu = torch.zeros(24)
    var = torch.ones(24)
    n_obs = 0
    if not INIT_PT:
        # init norm stats from random play (skipped for winner-INIT: weights expect raw obs)
        env0 = BM._environment(0, None)
        o = env0.reset()
        for _ in range(2000):
            t = torch.tensor(o, dtype=torch.float32)
            n_obs += 1
            d = t - mu
            mu += d / n_obs
            var += d * (t - mu)
            o, _, done = env0.step(random.randrange(4))
            if done:
                o = env0.reset()
    sigma = torch.sqrt(var / max(n_obs, 1)).clamp_min(1e-3)

    seed_idx = 0
    env = BM._environment(TRAIN_SEEDS[0], None)
    obs = env.reset()
    last_len, since_eat = len(env.snake), 0
    best = 0.0
    t0 = time.time()
    nupd = STEPS // NSTEP
    for upd in range(nupd):
        Ob, Ac, Re, Dn, Lp, Vl = [], [], [], [], [], []
        ep_len = 0
        for _ in range(NSTEP):
            o = (torch.tensor(obs, dtype=torch.float32) - mu) / sigma
            logits, val = net(o.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            if SHAPE_K > 0:
                fx, fy = env.food
                hx, hy = env.head_cell % 8, env.head_cell // 8
                d0 = abs(hx - fx) + abs(hy - fy)
            nobs, r, done = env.step(int(a))
            if SHAPE_K > 0:
                hx, hy = env.head_cell % 8, env.head_cell // 8
                d1 = abs(hx - fx) + abs(hy - fy)
                r = r + SHAPE_K * (d0 - d1)
            if len(env.snake) > last_len:
                last_len = len(env.snake)
                since_eat = 0
            else:
                since_eat += 1
                if since_eat > 128:
                    done = True
            Ob.append(o)
            Ac.append(int(a))
            Re.append(float(r))
            Dn.append(float(done))
            Lp.append(dist.log_prob(a))
            Vl.append(float(val))
            obs = nobs
            ep_len += 1
            if done:
                seed_idx = (seed_idx + 1) % len(TRAIN_SEEDS)
                env = BM._environment(TRAIN_SEEDS[seed_idx], None)
                obs = env.reset()
                last_len, since_eat = len(env.snake), 0
        Ob = torch.stack(Ob)
        Ac = torch.tensor(Ac)
        Re = torch.tensor(Re)
        Dn = torch.tensor(Dn)
        old_lp = torch.stack(Lp).detach()
        Vl = torch.tensor(Vl)
        with torch.no_grad():
            adv, gae = torch.zeros(NSTEP), 0.0
            for t in reversed(range(NSTEP)):
                mask = 1.0 - Dn[t]
                # bootstrap with next value (0 if terminal handled by mask on last)
                nxt = Vl[t + 1] if t + 1 < NSTEP else 0.0
                delta = Re[t] + GAMMA * mask * nxt - Vl[t]
                gae = delta + GAMMA * LAM * mask * gae
                adv[t] = gae
        ret = adv + Vl
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = torch.randperm(NSTEP)
        for e in range(EPOCHS):
            for i in range(0, NSTEP, MINIB):
                b = idx[i:i + MINIB]
                logits, val = net(Ob[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(Ac[b])
                ratio = torch.exp(lp - old_lp[b])
                pg = -torch.min(ratio * adv[b], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]).mean()
                vv = 0.5 * ((ret[b] - val) ** 2).mean()
                ent = -dist.entropy().mean()
                opt.zero_grad()
                (pg + vv - ENT * ent).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        # anneal lr
        for pg_ in opt.param_groups:
            pg_["lr"] = LR * (1 - upd / nupd)
        if (upd + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                s = eval_mean(net, mu, sigma)
            print(f"upd {upd+1}/{nupd}: eval100={s:.2f} t={time.time()-t0:.0f}s", flush=True)
            if s > best:
                best = s
                # fold norm into first linear: y = W((x-mu)/sigma)+b
                with torch.no_grad():
                    l0 = net.trunk[0]
                    W = l0.weight / sigma.unsqueeze(0)
                    b = l0.bias - (l0.weight / sigma.unsqueeze(0) * mu.unsqueeze(0)).sum(1)
                    sd = {"0.weight": W, "0.bias": b}
                    for i in range(1, len(HLIST)):
                        sd[f"{2*i}.weight"] = net.trunk[2 * i].weight
                        sd[f"{2*i}.bias"] = net.trunk[2 * i].bias
                    sd[f"{2*len(HLIST)}.weight"] = net.pi.weight
                    sd[f"{2*len(HLIST)}.bias"] = net.pi.bias
                    sd["H"] = list(HLIST)
                    sd["score"] = s
                torch.save(sd, OUT)
    print("DONE best=", best, flush=True)


if __name__ == "__main__":
    main()
