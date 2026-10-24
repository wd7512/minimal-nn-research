# MNIST — 11,350 params, passing

Bar: 100% on all 70k (train+test combined; dedupe confirmed 70,000 unique, 0 label conflicts, so 100% is attainable).

## Method

`scripts/mem_train.py`: AdamW, cosine-to-zero, batch 256, lr 3e-3, zero weight decay, keep best state, normalised inputs. Wide-to-narrow sweep, extending the schedule on near-misses before upsizing.

## Run log (single hidden layer unless noted)

| Widths | Epochs | Best / 70000 | Verdict |
|---|---|---|---|
| 512-256 | 150 | 70000 (fit ep 73) | pass, 535,818 |
| 256 | 200 | 70000 (ep 33) | pass, 203,530 |
| 128 | 250 | 70000 (ep 25) | pass, 101,770 |
| 64 | 400 | 70000 (ep 40) | pass, 50,890 |
| 32 | 500 | 70000 (ep 97) | pass, 25,450 |
| 16 | 600 | 69373 | wall (3x LR probe: 69391 — capacity, not LR) |
| 24 | 600 | 70000 (ep 360) | pass, 19,090 |
| 20 | 700 | 70000 (ep 384) | pass, 15,910 |
| 20-10 | 700 | 70000 (ep 550) | pass, 16,020 (no gain) |
| 16-16 | 800 | 70000 (ep 506) | pass, 13,002 |
| 16-8 | 800 | 69963 / 69924 / 69608 (3 seeds) | wall |
| 16-10 | 800 | 69519 | wall |
| 14-14 | 800 | 69945 | near-miss |
| 14-14 | 2000 | 70000 (ep 1651) | PASS, 11,350 |
| 12-12 | 1600 | 69331 / 68949 | wall |
| 13-13 | 1600 | 68978 | wall |
| 14-12 | 1600 | 69903 / 69495 | wall |

Minimum: **[14,14], 11,350 params** (`models/mnist/mlp-14-14`, PR #92).

## Finding

Near-misses (69,903–69,963) break through with 2–3x longer schedules, not wider nets. Single-16 is a hard capacity wall (LR-independent). Two narrow layers beat one wide layer at equal params in the 13–16 regime.

## Reproduce

```bash
V=<venv python>; W=<worktree>
$V scripts/build_caches.py            # data/mnist.pt (70000,784)
BENCHMARK=mnist $V scripts/mem_train.py mnist models/mnist/mlp-14-14 14 14 --epochs 2000 --seed 0
PYTHONPATH=src $V -m minimal_nn_competition evaluate mnist models/mnist/mlp-14-14 --data-path data/mnist.pt
```
