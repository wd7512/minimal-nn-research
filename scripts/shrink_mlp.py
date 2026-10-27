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
import copy
import torch
from pathlib import Path
import torch.nn as nn

SCRATCH_FALLBACK = Path.home() / ".hermes/cache/scratch/shrink_out"
DATA_DEFAULT = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/data/fashion-mnist.pt"
MAX_EP = 10000
PATIENCE = 3000
LOG_EVERY = 100


def build(widths, in_dim=784, out_dim=10):
    layers = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def key(i):
    return f"{2 * i}.weight", f"{2 * i}.bias"


def prune_idx(sd, layer):
    wk, bk = key(layer)
    W, b = sd[wk], sd[bk]
    return int((W.abs().sum(1) + b.abs()).argmin().item())


def drop_neuron(sd, layer, idx, widths):
    """Return (new_sd, new_widths) with neuron idx removed from `layer`.
    Removes exactly one neuron; widths[layer] shrinks by one. Never
    eliminates layers (use eliminate_last for that)."""
    new_widths = list(widths)
    new_widths[layer] -= 1
    assert new_widths[layer] >= 1, f"layer {layer} would go to 0; use eliminate_last"
    out = dict(sd)
    wk, bk = key(layer)
    assert sd[wk].shape[0] == widths[layer], f"{wk} dim {sd[wk].shape[0]} != widths {widths}"
    keep = [i for i in range(widths[layer]) if i != idx]
    out[wk] = sd[wk][keep]
    out[bk] = sd[bk][keep]
    if layer + 1 < len(widths):
        nwk, nbk = key(layer + 1)
        out[nwk] = sd[nwk][:, keep]
    return out, new_widths


def eliminate_last(sd, widths):
    """Drop the final hidden layer entirely; head re-inits fresh in load_into."""
    assert len(widths) > 1
    new_widths = widths[:-1]
    drop = {str(2 * (len(widths) - 1)), str(2 * len(widths))}
    out = {k: v for k, v in sd.items() if k.split(".")[0] not in drop}
    return out, new_widths


class Shrinker:
    def __init__(self, data, device, out, data_format="pt", task="classify", tol=0.1,
                 in_dim=784, out_dim=10):
        torch.set_default_dtype(torch.float32)  # pin everything to float32
        if data_format == "chess":
            sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/src")
            from minimal_nn_competition.benchmarks.chess_eval.benchmark import load_tensors
            X, y = load_tensors(data)
            self.Xn = X.float().to(device)
            self.yn = y.float().to(device)
        else:
            XP, yP = torch.load(data, map_location="cpu", weights_only=True)
            self.Xn = (XP.float() / 255.0).to(device)
            self.yn = yP.to(device)
        self.device = device
        self.out = out
        self.task = task
        self.tol = tol
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lf = nn.MSELoss() if task == "regress" else nn.CrossEntropyLoss()

    def score(self, model):
        """Fraction of samples within tolerance (exact match for classify).

        Always evaluated on CPU in float32, matching the official evaluator,
        so MPS rounding can't flip borderline samples at the gate.
        """
        mc = copy.deepcopy(model).cpu().float().eval()
        good = 0
        total = len(self.Xn)
        with torch.no_grad():
            for b in torch.arange(total).split(8192):
                xb = self.Xn[b].cpu()
                yb = self.yn[b].cpu()
                if self.task == "regress":
                    err = (mc(xb).squeeze(1) - yb).abs()
                    good += (err <= self.tol).sum().item()
                else:
                    good += (mc(xb).argmax(1) == yb).sum().item()
        return good / total

    def evaluate(self, model):
        return self.score(model)

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
                pred = model(self.Xn[b])
                if self.task == "regress":
                    pred = pred.squeeze(1)
                loss = self.lf(pred, self.yn[b])
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
        model = build(widths, self.in_dim, self.out_dim).to(self.device).float()
        own = model.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v.to(self.device))
        return model

    def attempt(self, widths, sd, tag, lrs=(3e-3, 1e-3, 3e-4, 1e-4), max_ep=MAX_EP):
        """Train a shape to perfect, retrying with smaller LRs on failure."""
        model = self.load_into(widths, sd)
        best, bs = 0.0, None
        for lr in lrs:
            best, bs = self.train_to_perfect(model, f"{tag}-lr{lr}", max_ep=max_ep, lr=lr)
            if best == 1.0:
                name = "best" + "_".join(map(str, widths)) + ".pt"
                torch.save(bs, self.out / name)
                return best, bs
            print(f"[{tag}] lr {lr} failed at {best:.6f}; retrying", flush=True)
            model = self.load_into(widths, sd)  # fresh start, same warm weights
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
    rest = sys.argv[2:]
    widths = []
    for a in rest:
        if a.startswith("--"):
            break
        widths.append(int(a))
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else SCRATCH_FALLBACK
    out.mkdir(parents=True, exist_ok=True)
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else DATA_DEFAULT
    fmt = sys.argv[sys.argv.index("--format") + 1] if "--format" in sys.argv else "pt"
    task = sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else "classify"
    in_dim = int(sys.argv[sys.argv.index("--in-dim") + 1]) if "--in-dim" in sys.argv else 784
    out_dim = int(sys.argv[sys.argv.index("--out-dim") + 1]) if "--out-dim" in sys.argv else 10
    identity_k = "--identity-k" in sys.argv
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    s = Shrinker(data, device, out, data_format=fmt, task=task,
                 in_dim=in_dim, out_dim=out_dim)

    t = torch.load(teacher, map_location="cpu")
    # fix at float32: cast any non-float32 checkpoint tensors on load
    for k, v in list(t.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype != torch.float32:
            print(f"casting {k} from {v.dtype} to float32", flush=True)
            t[k] = v.float()
    # seed first layer: prune teacher rows by output-column norm to n_start
    W2, n = t["2.weight"], widths[0]
    keep = sorted(range(W2.shape[1]), key=lambda i: W2[:, i].abs().sum().item(), reverse=True)[:n]
    sd = {"0.weight": t["0.weight"][keep], "0.bias": t["0.bias"][keep]}
    if len(widths) > 1:
        sd["2.weight"] = t["2.weight"][:, keep] if t["2.weight"].shape[1] == len(keep) else None
        sd["2.bias"] = t["2.bias"] if "2.bias" in t and t["2.bias"].shape[0] == t["2.weight"].shape[0] else None
    if identity_k and len(widths) == 3 and "4.weight" in t and t["4.weight"].shape == (out_dim, widths[1]):
        # init the fresh K layer as a pass-through of the first K neurons, so the
        # 3-layer seed starts near the 2-layer teacher instead of from scratch
        m, k = widths[1], widths[2]
        w3 = torch.zeros(k, m)
        for i in range(min(k, m)):
            w3[i, i] = 1.0
        sd["4.weight"] = w3
        sd["4.bias"] = torch.zeros(k)
        sd["6.weight"] = t["4.weight"][:, :k].clone()
        sd["6.bias"] = t["4.bias"].clone()
        print(f"identity-K init from teacher head (K={k})", flush=True)
    model = s.load_into(widths, {k: v for k, v in sd.items() if v is not None})
    seed_tag = "seed-" + "-".join(map(str, widths))
    # resume: a previous run may already have proven this exact shape
    seed_file = out / ("best" + "_".join(map(str, widths)) + ".pt")
    if seed_file.exists():
        try:
            prev = torch.load(seed_file, map_location="cpu")
            m0 = s.load_into(widths, prev)
            if s.evaluate(m0) == 1.0:
                print(f"resuming from proven {seed_file.name}", flush=True)
                sd = {k: v.cpu().clone() for k, v in m0.state_dict().items()}
                best, bs = 1.0, sd
            else:
                best, bs = s.attempt(widths, {k: v for k, v in sd.items() if v is not None}, seed_tag)
        except Exception as e:
            print(f"resume failed ({e}); retraining seed", flush=True)
            best, bs = s.attempt(widths, {k: v for k, v in sd.items() if v is not None}, seed_tag)
    else:
        best, bs = s.attempt(widths, {k: v for k, v in sd.items() if v is not None}, seed_tag)
    if best != 1.0 or bs is None:
        print("seed shape failed; abort", flush=True)
        sys.exit(1)
    sd = bs

    widths, sd = s.shrink_first_sequential(widths, sd)
    for layer in range(1, len(widths)):
        widths, sd = s.bisect_layer(widths, sd, layer)
    # last-layer elimination: bisect stops at 1, now try dropping it entirely
    if len(widths) > 1:
        sd0, w0 = eliminate_last(sd, widths)
        best, bs = s.attempt(w0, sd0, "K0-" + "-".join(map(str, w0)))
        if best == 1.0:
            widths, sd = w0, bs
            print("final layer eliminated", flush=True)
    print(f"FINAL {widths}", flush=True)
    (out / "final.json").write_text(str(widths))


if __name__ == "__main__":
    main()
