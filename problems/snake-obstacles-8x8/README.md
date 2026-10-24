# Snake Obstacles 8x8 — stuck at 9.33 mean (structure documented)

Bar: fill board-minus-obstacles on all 100 seeds (~61 cells). Layouts have verified per-seed Hamiltonian tours (`find_closed_tour`). Best submission: **9.33 mean** (`models/snake-obstacles-8x8/obs-fb`, 14,852 params, failing, PR #95).

## Why it is hard (verified)

- Obs (24 feats) has NO obstacle map — rays sense blockers only as distances. Same obs demands different actions on different seeds: **1,238/35,698 conflicted groups** on 100-seed tour pairs.
- Any memorised net (even 100% train fit, width-512) dies at the first conflict (~tick 5, score ~2–3). No recovery.
- Union of blocked cells over seeds = 55/64: no universal tour exists. Per-seed tours are mandatory.
- Tour teacher: 61.06 mean, 100/100 fills — unattainable by a layout-blind student.

## What was tried (all scored by official closed-loop eval)

| Approach | Policy mean | Student score |
|---|---|---|
| Tour teacher (oracle) | 61.06 (100/100) | — |
| Pure-BFS + tail-safety teacher | 21.21 (0/100) | — |
| Connectivity-preserving greedy | 10.75 | — |
| Safe-greedy fallback only | 16.38 | — |
| Majority table (tour) | 6.29 | — |
| Table + first-divergence repair, no-regression | 9.54, 2/100 | — |
| Memorise tour pairs, h512, 100% fit | — | ~2.5 |
| DAgger (teacher-relabel student states), h512 | — | 1.95 → 3.4 (round 1→2; +~1/round, too slow) |
| Memorise BFS-teacher rollouts, h512 | — | 8.17 |
| **Memorise safe-greedy rollouts, h512, 100% fit** | — | **9.33 (submitted)** |
| Tabu (least-visited-safe) escape teacher | 23.55, 1 fill | 7.29 |

Failure analysis of the BFS teacher: 82/100 failures are per-food CAP loops (alive but looping), only 18 die outright — yet tabu escape only lifts 21.2 → 23.6.

## Findings

1. Memorise a policy's OWN rollout distribution (self-consistent), never an oracle's idealized one.
2. DAgger majority-vote dilutes early critical decisions; repair's targeted choices are better but single-obs moves plateau fast (2/100).
3. Train accuracy is uncorrelated with closed-loop score. Select ONLY on official eval; keep a seed lottery (3 seeds evaluated, best kept).
4. Retrains can regress (2.46 → 1.95) despite higher fit — closed-loop basins differ.

## Reproduce (best line)

```bash
V=<venv python>
$V scripts/collect_policy.py none pairsFallback.pt   # safe-greedy rollout pairs
BENCHMARK=snake-obstacles-8x8 INDICES=0..23 \
  $V scripts/snake_train.py pairsFallback.pt out/ 512 --epochs 4000 --seed 0 --batch 256 --lr 0.003
```
