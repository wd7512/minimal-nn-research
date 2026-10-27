"""Shrink an MLP to its smallest perfect-memorisation shape.

Pattern (established on Fashion-MNIST, 70k samples, 100% train-fit target):
  1. FIRST layer shrinks SEQUENTIALLY, one neuron at a time. Each step prunes
     the weakest neuron (smallest |weights|_1 + |bias|) and WARM-STARTS the
     next shape from the survivors. Sequential descent is what makes the
     warm start valid: neighbouring widths share almost all their weights.
  2. Every LATER layer starts at a fixed high width and shrinks by BISECTION.
     Each midpoint attempt is warm-started by pruning the best passing model
     down to the midpoint width. Bisection is valid here because the first
     layer already carries the representation; later layers only need sizing.
  3. The last layer may bisect all the way to 0 (layer eliminated).

Usage:
  python shrink_mlp.py <teacher.pt> <n_start> <m_start> <k_start> [--out DIR]
      --data PATH (default: fashion-mnist.pt next to this script's data dir)
      --max-ep N (default 10000) --patience N (default 3000)

Checkpoints: best<N>_<M>_<K>.pt per passing shape, final.json at the end.
"""
import sys
import torch
from pathlib import Path
import torch.nn as nn

SCRATCH_FALLBACK = Path.home() / ".hermes/cache/scratch/shrink_out"
DATA_DEFAULT = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/data/fashion-mnist.pt"
MAX_EP = 10000
PATIENCE = 3000
LOG_EVERY = 100


def build(widths):
    layers = []
    prev = 784
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    layers.append(nn.Linear(prev, 10))
    return nn.Sequential(*layers)


def key(i):
    return f"{2 * i}.weight", f"{2 * i}.bias"


def prune_idx(sd, layer):
    wk, bk = key(layer)
    W, b = sd[wk], sd[bk]
    return int((W.abs().sum(1) + b.abs()).argmin().item())


def drop_neuron(sd, layer, idx, widths):
    """Return (new_sd, new_widths) with neuron idx removed from `layer`.
    Later layers lose the corresponding input column; K=widths[-1] going to
    0 drops the layer entirely (fresh output head)."""
    new_widths = list(widths)
    out = dict(sd)
    wk, bk = key(layer)
    keep = [i for i in range(widths[layer]) if i != idx]
    out[wk] = sd[wk][keep]
    out[bk] = sd[bk][keep]
    if layer + 1 < len(widths):
        nwk, nbk = key(layer + 1)
        out[nwk] = sd[nwk][:, keep]
        out[nbk] = sd[nbk]
    else:
        new_widths = widths[:-1]  # last layer eliminated -> fresh head
        out = {k: v for k, v in out.items() if not k.startswith(str(2 * layer))}
    return out, new_widths


class Shrinker:
    def __init__(self, data, device, out):
        XP, yP = torch.load(data, map_location="cpu", weights_only=True)
        self.Xn = (XP.float() / 255.0).to(device)
        self.yn = yP.to(device)
        self.device = device
        self.out = out
        self.lf = nn.CrossEntropyLoss()

    def evaluate(self, model):
        model.eval()
        tc = 0
        with torch.no_grad():
            for b in torch.arange(len(self.Xn), device=self.device).split(8192):
                tc += (model(self.Xn[b]).argmax(1) == self.yn[b]).sum().item()
        return tc / len(self.Xn)

    def train_to_perfect(self, model, tag, max_ep=MAX_EP, lr=3e-3):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        bs = 4096
        best = self.evaluate(model)
        best_state = None
        if best == 1.0:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        since = 0
        ep = 0
        while ep < max_ep:
            ep += 1
            model.train()
            perm = torch.randperm(len(self.Xn), device=self.device)
            tot = 0
            for b in perm.split(bs):
                opt.zero_grad(set_to_none=True)
                loss = self.lf(model(self.Xn[b]), self.yn[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if ep % LOG_EVERY == 0 or ep == 1:
                acc = self.evaluate(model)
                if acc > best:
                    best = acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    since = 0
                else:
                    since += LOG_EVERY
                print(f"[{tag}] epoch {ep}/{max_ep} acc {acc:.6f} best {best:.6f}", flush=True)
                if best == 1.0 or since >= PATIENCE:
                    break
        return best, best_state

    def load_into(self, widths, sd):
        """Warm-start: copy every layer present in both; fresh-init the rest."""
        model = build(widths).to(self.device)
        own = model.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v.to(self.device))
        return model

    def attempt(self, widths, sd, tag):
        model = self.load_into(widths, sd)
        best, bs = self.train_to_perfect(model, tag)
        if best == 1.0:
            name = "best" + "_".join(map(str, widths)) + ".pt"
            torch.save(bs, self.out / name)
        return best, bs

    def shrink_first_sequential(self, widths, sd):
        """Stage 1: first layer down one neuron at a time, warm-started."""
        while widths[0] > 1:
            idx = prune_idx(sd, 0)
            sd2, w2 = drop_neuron(sd, 0, idx, widths)
            tag = "N" + "-".join(map(str, w2))
            print(f"pruning L0 neuron {idx} -> {w2}", flush=True)
            best, bs = self.attempt(w2, sd2, tag)
            if best == 1.0:
                widths, sd = w2, bs
            else:
                print(f"{w2} failed; smallest passing N = {widths[0]}", flush=True)
                break
        return widths, sd

    def bisect_layer(self, widths, sd, layer):
        """Stage 2+: bisect `layer` from its width down to 1 (then try 0 for last)."""
        lo, hi = 1, widths[layer]
        seed = sd
        while lo < hi:
            mid = (lo + hi) // 2
            sd0, w0 = seed, list(widths)
            cur = widths[layer]
            while cur > mid:
                idx = prune_idx(sd0, layer)
                sd0, w0 = drop_neuron(sd0, layer, idx, w0)
                cur -= 1
            tag = ("N" if layer == 0 else f"L{layer}") + "-".join(map(str, w0))
            best, bs = self.attempt(w0, sd0, tag)
            if best == 1.0:
                hi = mid
                seed = bs
                widths = w0
            else:
                lo = mid + 1
        print(f"layer {layer} settled at {hi}", flush=True)
        return widths, seed


def main():
    teacher = sys.argv[1]
    widths = list(map(int, sys.argv[2:5]))
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else SCRATCH_FALLBACK
    out.mkdir(parents=True, exist_ok=True)
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else DATA_DEFAULT
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    s = Shrinker(data, device, out)

    t = torch.load(teacher, map_location="cpu")
    # seed first layer: prune teacher rows by output-column norm to n_start
    W2, n = t["2.weight"], widths[0]
    keep = sorted(range(W2.shape[1]), key=lambda i: W2[:, i].abs().sum().item(), reverse=True)[:n]
    sd = {"0.weight": t["0.weight"][keep], "0.bias": t["0.bias"][keep]}
    if len(widths) > 1:
        sd["2.weight"] = t["2.weight"][:, keep] if t["2.weight"].shape[1] == len(keep) else None
    model = s.load_into(widths, {k: v for k, v in sd.items() if v is not None})
    best, bs = s.train_to_perfect(model, "seed-" + "-".join(map(str, widths)))
    if best != 1.0:
        print("seed shape failed; abort", flush=True)
        sys.exit(1)
    sd = bs

    widths, sd = s.shrink_first_sequential(widths, sd)
    for layer in range(1, len(widths)):
        widths, sd = s.bisect_layer(widths, sd, layer)
    # last-layer elimination: bisect stops at 1, now try dropping it entirely
    if len(widths) > 1:
        last = len(widths) - 1
        idx = prune_idx(sd, last)
        sd0, w0 = drop_neuron(sd, last, idx, widths)
        best, bs = s.attempt(w0, sd0, "K0-" + "-".join(map(str, w0)))
        if best == 1.0:
            widths, sd = w0, bs
            print("final layer eliminated", flush=True)
    print(f"FINAL {widths}", flush=True)
    (out / "final.json").write_text(str(widths))


if __name__ == "__main__":
    main()
