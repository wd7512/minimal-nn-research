"""Build combined 70k caches (train+test) for the dev-branch evaluators."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch

SRC = "/Users/williamdennis/repos/minimal-nn-competition/data"
DST = "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/data"

import os

os.makedirs(DST, exist_ok=True)

for name in ["mnist", "fashion-mnist"]:
    tr_x, tr_y = torch.load(f"{SRC}/{name}-train.pt", map_location="cpu", weights_only=True)
    te_x, te_y = torch.load(f"{SRC}/{name}-test.pt", map_location="cpu", weights_only=True)
    X = torch.cat([tr_x, te_x])
    y = torch.cat([tr_y.flatten(), te_y.flatten()])
    torch.save((X, y), f"{DST}/{name}.pt")
    logging.info("%s: %s labels=%s", name, tuple(X.shape), tuple(y.shape))
