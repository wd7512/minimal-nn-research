# Snake Simple 16x16 — no smaller model found (floor documented)

Bar: fill all 256 cells on all 100 seeds. Standing record: `cycle-44` (head-6 MLP [44], 488 params, passing). Goal was a smaller passing net. Result: **no passing net below 44 width found**; best train-fit (width-36, 241/256) scores 3.79 closed-loop.

## What was tried

Teacher: analytic serpentine Hamiltonian cycle (column-0 spine; spawn (8,8) heading right requires the reversed path — `scripts/serp16.py`, `serp16T.py`). Teacher fills 100/100, 256 unique obs states, 0 conflicts. Student: distill (obs head-6 → tour action) via `scripts/snake_train.py`.

| Config | Fit / 256 | Closed-loop |
|---|---|---|
| cycle-8 / 16 (clone cycle-44) | no fit | — |
| serp-8 (seeds 0-4) | 162–168 | — |
| serp-16-mse / -mb32 | 168 / 182 | — |
| serp-32 (seeds 0-3) | 172–192 | — |
| serp-32-4 / serpT-8b | 182 / 160–165 | — |
| serp-8f (8-feat inputs) | 167–168 | — |
| serp-40 | 180–191 | — |
| serp-16-std / serp-32-std (input standardisation, folded back) | 196 / 236+228 | — |
| serp-36-std | **241** | **3.79 (submitted, PR #94)** |
| serp-40-std | 198 (worse than 32!) | — |
| serp-32-std long (20k epochs) | 206 (regressed) | — |
| LR 3e-3 / 1e-3 probes | 167–217 (worse) | — |
| Square-flip annealing, width-12 target | 180 (flat landscape) | — |

Submission: `models/snake-simple-16x16/serp-36` (400 params, failing) — documents the floor.

## Findings

1. Width scaling inverts: 36 > 32 > 40. Blind width scaling is dead.
2. Standardisation + fold-back is the only big win (+~40 patterns at width-32).
3. Long schedules HURT here (236 @ 8k → 206 @ 20k) — opposite of the image tasks. Noise-dominated, not progress-limited.
4. Train-fit does not predict closed-loop (241/256 fit → 3.79 score). Same distribution-shift disease as obstacles.
5. Annealing the cycle geometry barely moves distillability (178 → 180/256 at width-12).

## Reproduce

```bash
V=<venv python>
$V scripts/serp16.py                                   # pairs16serp.pt (teacher pairs)
BENCHMARK=snake-simple-16x16 INDICES=0,1,2,3,4,5 \
  $V scripts/snake_train.py pairs16serp.pt out/ 36 --epochs 8000 --seed 0 --standardize
```
