"""Hard-sample focus for chess-100k: start from the 99.982% teacher and train with
extra weight on borderline samples (|err| > 0.08), gated on the CPU float32 score.
Usage: python chess100k_focus.py [--lr 1e-4] [--epochs 2000]"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-research/scripts")
from shrink_mlp import Shrinker, build

LR = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 1e-4
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 2000
OUT = Path("/Users/williamdennis/.hermes/cache/scratch/chess100k/focus")
OUT.mkdir(parents=True, exist_ok=True)

s = Shrinker(
    "/Users/williamdennis/repos/minimal-nn-competition/src/minimal_nn_competition/benchmarks/chess_eval/positions-100k.jsonl",
    "mps" if torch.backends.mps.is_available() else "cpu", OUT,
    data_format="chess", task="regress", in_dim=769, out_dim=1)
model = build([512, 512], 769, 1).to(s.device).float()
t = torch.load("/Users/williamdennis/repos/minimal-nn-competition/models/chess-eval-100k/mlp-512-512/model.pt",
               map_location="cpu")
model.load_state_dict({k: v for k, v in t.items()})
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
lf = nn.MSELoss()
best = s.evaluate(model)
print(f"start acc {best:.6f}", flush=True)
since = 0
bs = 4096
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(s.Xn), device=s.device)
    tot = 0
    for b in perm.split(bs):
        opt.zero_grad(set_to_none=True)
        pred = model(s.Xn[b]).squeeze(1)
        yb = s.yn[b]
        with torch.no_grad():
            hard = (pred - yb).abs() > 0.08
        base = lf(pred, yb)
        loss = base if hard.sum() == 0 else base + 5.0 * lf(pred[hard], yb[hard])
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    if ep % 20 == 0 or ep == 1:
        acc = s.evaluate(model)
        if acc > best:
            best = acc
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       OUT / "best512_512.pt")
            since = 0
        else:
            since += 20
        print(f"[focus] epoch {ep}/{EPOCHS} acc {acc:.6f} best {best:.6f}", flush=True)
        if best == 1.0 or since >= 500:
            break
print(f"FOCUS DONE best={best}", flush=True)
