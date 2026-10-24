"""Memorise distilled snake pairs (tiny: full-batch, best-state kept).

Usage: python snake_train.py <pairs.pt> <out-dir> <h1> [h2 ...] [--epochs N] [--seed S] [--lr F]
Writes config.json + model.pt ONLY on 100% fit. Config overridden by env:
  BENCHMARK, INDICES (comma list), SIZE (board size, unused, informational).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("snake_train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("widths", nargs="+", type=int)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--loss", default="ce", choices=["ce", "mse"])
    ap.add_argument("--batch", type=int, default=0, help="minibatch size, 0 = full batch")
    ap.add_argument("--standardize", action="store_true", help="train on standardized inputs, fold back on save")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    X, y = torch.load(args.pairs, map_location="cpu", weights_only=True)
    X, y = X.float(), y.long()
    n, in_size = X.shape
    nclass = int(y.max()) + 1
    mu, sigma = None, None
    if args.standardize:
        mu = X.mean(0)
        sigma = X.std(0).clamp_min(1e-6)
        X = (X - mu) / sigma
    logger.info("pairs=%d in=%d classes=%d widths=%s", n, in_size, nclass, args.widths)

    layers, prev = [], in_size
    for w in args.widths:
        layers += [torch.nn.Linear(prev, w), torch.nn.ReLU()]
        prev = w
    layers.append(torch.nn.Linear(prev, nclass))
    model = torch.nn.Sequential(*layers)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("params=%d", n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce_fn = torch.nn.CrossEntropyLoss()
    def loss_fn(out, tgt):
        if args.loss == "mse":
            return torch.nn.functional.mse_loss(out, torch.nn.functional.one_hot(tgt, nclass).float())
        return ce_fn(out, tgt)

    best, best_state = -1, None
    for epoch in range(args.epochs):
        model.train()
        if args.batch and args.batch < n:
            perm = torch.randperm(n)
            for idx in perm.split(args.batch):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(X[idx]), y[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                loss = loss_fn(model(X), y)
        else:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            correct = (model(X).argmax(1) == y).sum().item()
        if correct > best:
            best = correct
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 500 == 0 or correct == n:
            logger.info("epoch %d loss %.4f acc %.4f best %d/%d", epoch + 1, loss.item(), correct / n, best, n)
        if best == n:
            logger.info("FULL FIT at epoch %d", epoch + 1)
            break
    logger.info("BEST %d/%d", best, n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_state = best_state
    if args.standardize and best_state is not None:
        # Fold (x-mu)/sigma into the first linear layer: exact same function.
        save_state = dict(best_state)
        w = save_state["0.weight"] / sigma.unsqueeze(0)
        save_state["0.weight"] = w
        save_state["0.bias"] = save_state["0.bias"] - (w * mu.unsqueeze(0)).sum(1)
    torch.save(save_state, args.out_dir / "best.pt")
    if best == n:
        torch.save(save_state, args.out_dir / "model.pt")
        cfg = {
            "benchmark": os.environ.get("BENCHMARK", "snake-simple-16x16"),
            "name": args.out_dir.name,
            "author": "hermes",
            "architecture": "mlp",
            "input_size": in_size,
            "output_size": nclass,
            "hidden_sizes": list(args.widths),
            "activation": "relu",
            "input_indices": [int(i) for i in os.environ.get("INDICES", "0,1,2,3,4,5").split(",")],
        }
        (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        logger.info("WROTE %s", args.out_dir)
    return 0 if best == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
