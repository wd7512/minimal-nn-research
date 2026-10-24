# Fashion-MNIST — 27,835 params, passing

Same bar and method as MNIST. Fashion needs ~2.5x the capacity (harder textures, same 70k).

## Run log

| Widths | Epochs | Best / 70000 | Verdict |
|---|---|---|---|
| 128 | 400 | 70000 (ep 267) | pass |
| 64 | 500 | 69996 | near-miss (4 short) |
| 64 | 1200 | 70000 (ep 773) | pass, 50,890 |
| 32 | 1200 | 69017 | hard wall |
| 48 | 1200 | 70000 (ep 916) | pass, 38,170 |
| 40 | 1200 | 69998 | near-miss (2 short) |
| 40 | 2000 | 70000 (ep 1324) | pass, 31,810 |
| 36 | 2000 | 70000 (ep 1550) | pass, 28,630 |
| 34 | 2000 | 69865 / 69342 (2 seeds) | wall (both seeds) |
| 32-32 | 2000 | 69809 | wall (pairs worse than single-34) |
| 34-24 | 2000 | 69692 | wall |
| 35 | 2000 | 70000 (ep 1706) | PASS, 27,835 |

Minimum: **single-35, 27,835 params** (`models/fashion-mnist/mlp-35`, PR #93).

## Finding

Same pattern as MNIST: 2-short near-miss at 1200 epochs → full fit at 2000. Pairs ([32,32], [34,24]) underperform single-34 — when one width fails, splitting capacity across two layers does not save it here (unlike MNIST's 16-16 vs 16-wide wall, where depth helped).

## Reproduce

```bash
V=<venv python>
BENCHMARK=fashion-mnist $V scripts/mem_train.py fashion-mnist models/fashion-mnist/mlp-35 35 --epochs 2000 --seed 0
PYTHONPATH=src $V -m minimal_nn_competition evaluate fashion-mnist models/fashion-mnist/mlp-35 --data-path data/fashion-mnist.pt
```
