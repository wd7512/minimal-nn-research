"""Memorisation trainer: wide MLP to 100% on the full 70k cache, best-state kept.

Usage: python mem_train.py <mnist|fashion-mnist> <out-dir> <h1> [h2 ...] [--epochs N] [--seed S]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mem_train")


def build_mlp(widths, in_size=784, out_size=10):
    import torch.nn as nn

    layers = []
    prev = in_size
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["mnist", "fashion-mnist"])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("widths", nargs="+", type=int)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("device=%s", dev)

    data_path = args.data or Path(
        f"/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/data/{args.benchmark}.pt"
    )
    X, y = torch.load(data_path, map_location="cpu", weights_only=True)
    X = (X.float() / 255.0).to(dev)
    y = y.flatten().long().to(dev)
    n = len(X)
    logger.info("samples=%d mean=%.4f", n, X.mean().item())

    model = build_mlp(args.widths).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("widths=%s params=%d", args.widths, n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_correct = -1
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for idx in perm.split(args.batch):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            # chunked full-cache eval to stay light on MPS memory
            correct = 0
            for idx in torch.arange(n, device=dev).split(8192):
                correct += (model(X[idx]).argmax(1) == y[idx]).sum().item()
        acc = correct / n
        if correct > best_correct:
            best_correct = correct
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        logger.info(
            "epoch %d/%d loss %.4f acc %.6f best %d/%d",
            epoch + 1, args.epochs, tot / n, acc, best_correct, n,
        )
        if best_correct == n:
            logger.info("FULL FIT at epoch %d", epoch + 1)
            break

    logger.info("BEST %d/%d = %.6f", best_correct, n, best_correct / n)
    if best_correct == n:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.out_dir / "model.pt")
        cfg = {
            "benchmark": args.benchmark,
            "name": args.out_dir.name,
            "author": "hermes",
            "architecture": "mlp",
            "input_size": 784,
            "output_size": 10,
            "hidden_sizes": list(args.widths),
            "activation": "relu",
            "normalisation": True,
        }
        (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        logger.info("WROTE %s", args.out_dir)
    return 0 if best_correct == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
