import sys; sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition")
import torch, json
from pathlib import Path
import torch.nn as nn

SCRATCH = Path("/Users/williamdennis/.hermes/cache/scratch/reduce_nm")
SCRATCH.mkdir(parents=True, exist_ok=True)
DATA = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/data/fashion-mnist.pt"
TEACHER = "/Users/williamdennis/repos/minimal-nn-competition/models/fashion-mnist/mlp-32/model.pt"
MAX_EP = 10000
PATIENCE = 3000
LOG_EVERY = 100

def build(widths):
    layers = []; prev = 784
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]; prev = w
    layers.append(nn.Linear(prev, 10))
    return nn.Sequential(*layers)

device = "mps" if torch.backends.mps.is_available() else "cpu"
XP, yP = torch.load(DATA, map_location="cpu", weights_only=True)
Xn = (XP.float() / 255.0).to(device); yn = yP.to(device)
lf = nn.CrossEntropyLoss()

def train_to_perfect(model, tag, max_ep=MAX_EP, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bs = 4096; best = 0; best_state = None; since = 0
    for epoch in range(max_ep):
        model.train(); perm = torch.randperm(len(Xn), device=device); tot = 0
        for b in perm.split(bs):
            opt.zero_grad(set_to_none=True)
            loss = lf(model(Xn[b]), yn[b]); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            model.eval(); tc = 0
            with torch.no_grad():
                for b in torch.arange(len(Xn), device=device).split(8192):
                    tc += (model(Xn[b]).argmax(1) == yn[b]).sum().item()
            acc = tc / len(Xn)
            if acc > best:
                best = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                since = 0
            else:
                since += LOG_EVERY
            print(f"[{tag}] epoch {epoch+1}/{max_ep} loss {tot/len(Xn):.4f} acc {acc:.6f} best {best:.6f}", flush=True)
            if best == 1.0:
                break
            if since >= PATIENCE:
                print(f"[{tag}] plateaued, stopping early", flush=True)
                break
    return best, best_state

def prune_first_layer(sd, n_in):
    # sd keys: '0.weight' (N,784), '0.bias' (N,), '2.weight' (M,N), ...
    W1 = sd["0.weight"]; b1 = sd["0.bias"]
    score = W1.abs().sum(1) + b1.abs()
    drop = int(score.argmin().item())
    keep = [i for i in range(n_in) if i != drop]
    print(f"pruning neuron {drop} (score {score[drop].item():.4f})", flush=True)
    return keep

# ---- Stage 0: build [31,128] warm-started from teacher pruned to 31 ----
teacher_sd = torch.load(TEACHER, map_location="cpu")
W1t = teacher_sd["0.weight"]; b1t = teacher_sd["0.bias"]; W2t = teacher_sd["2.weight"]
score = W2t.abs().sum(0)
drop0 = int(score.argmin().item())
keep31 = [i for i in range(32) if i != drop0]
print(f"teacher prune idx {drop0}", flush=True)
model = build([31, 128]).to(device)
with torch.no_grad():
    model[0].weight.copy_(W1t[keep31]); model[0].bias.copy_(b1t[keep31])
best, bs = train_to_perfect(model, "N31-M128")
results = {}
if best == 1.0:
    results[31] = bs
    torch.save(bs, SCRATCH / "best31_128.pt")
    cur_sd, cur_n = bs, 31
else:
    print("even [31,128] failed; abort", flush=True)
    sys.exit(1)

# ---- Stage 1: reduce N one at a time, M fixed at 128 ----
while True:
    keep = prune_first_layer(cur_sd, cur_n)
    nxt = cur_n - 1
    if nxt < 1:
        break
    model = build([nxt, 128]).to(device)
    with torch.no_grad():
        model[0].weight.copy_(cur_sd["0.weight"][keep])
        model[0].bias.copy_(cur_sd["0.bias"][keep])
        model[2].weight.copy_(cur_sd["2.weight"][:, keep])
        model[2].bias.copy_(cur_sd["2.bias"])
    best, bs = train_to_perfect(model, f"N{nxt}-M128")
    if best == 1.0:
        results[nxt] = bs
        torch.save(bs, SCRATCH / f"best{nxt}_128.pt")
        cur_sd, cur_n = bs, nxt
    else:
        print(f"N={nxt} failed to reach 100%; smallest passing N = {cur_n}", flush=True)
        break

json.dump({"passing_N": sorted(results)}, open(SCRATCH / "stage1.json", "w"))
print("STAGE1 DONE smallest passing N:", cur_n, flush=True)

# ---- Stage 2: bisect M for smallest passing N ----
lo, hi = 1, 128
best_m_sd = None
while lo < hi:
    mid = (lo + hi) // 2
    model = build([cur_n, mid]).to(device)
    with torch.no_grad():
        model[0].weight.copy_(cur_sd["0.weight"])
        model[0].bias.copy_(cur_sd["0.bias"])
    best, bs = train_to_perfect(model, f"N{cur_n}-M{mid}")
    if best == 1.0:
        hi = mid
        torch.save(bs, SCRATCH / f"best{cur_n}_{mid}.pt")
        best_m_sd = (mid, bs)
    else:
        lo = mid + 1
print(f"STAGE2 DONE smallest M for N={cur_n}: {lo}", flush=True)
json.dump({"N": cur_n, "min_M": lo}, open(SCRATCH / "stage2.json", "w"))
