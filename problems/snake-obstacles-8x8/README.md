# Snake Obstacles 8x8 — 24.64 mean via closed-loop PPO (was 9.33)

Bar: fill board-minus-obstacles on all 100 seeds (~61 cells). Layouts have verified per-seed Hamiltonian tours (`find_closed_tour`). Best submission: **24.64 mean** (`models/snake-obstacles-8x8/ppo-512-256`, 145,156 params, failing, PR #115).

Score line: 9.33 → 13.0 → 16.73 → 18.71 → 23.76 → 24.64.

## Why it is hard (verified)

- Obs (24 feats) has NO obstacle map — rays sense blockers only as distances. Same obs demands different actions on different seeds: **1,238/35,698 conflicted groups** on 100-seed tour pairs.
- Any memorised net (even 100% train fit, width-512) dies at the first conflict (~tick 5, score ~2–3). No recovery.
- Union of blocked cells over seeds = 55/64: no universal tour exists. Per-seed tours are mandatory.
- Tour teacher: 61.06 mean, 100/100 fills — unattainable by a layout-blind student.

## What was tried (all scored by official closed-loop eval)

| Approach | Score |
|---|---|
| Tour teacher (oracle) | 61.06 (100/100) |
| Pure-BFS + tail-safety teacher | 21.21 (0/100) |
| Connectivity-preserving greedy | 10.75 |
| Safe-greedy fallback only | 16.38 |
| Majority table (tour) | 6.29 |
| Table + first-divergence repair, no-regression | 9.54, 2/100 |
| Memorise tour pairs, h512, 100% fit | ~2.5 |
| DAgger (teacher-relabel student states), h512 | 1.95 → 3.4 |
| Memorise BFS-teacher rollouts, h512 | 8.17 |
| **Memorise safe-greedy rollouts, h512, 100% fit** | **9.33 (PR #95)** |
| Tabu (least-visited-safe) escape teacher | 23.55 teacher / 7.29 student |
| GA evolve weights directly on closed-loop mean (pop 192, 7 workers) | 7.38 (retired — PPO lapped it) |
| PPO from scratch, [256,256], 1.5M steps, seeds 0–299 | 5.75 |
| PPO warm-started from 9.33, Optuna hypers (300k-step trials) | 13.00 |
| → full 3M-step run, same hypers | 16.73 (PR #113) |
| PPO round 2 from 16.73 (plain / shaped / low-LR) | 18.71 / 18.39 / 18.21 |
| Adam vs Muon in Optuna (12 trials from 18.71) | Muon 18.69 ≈ Adam 17.90 — no signal, dropped |
| Distil 18.71 → [1024] (99.65% imitation → 18.5) → PPO on eval-100 | 22.48 |
| Distil 18.71 → [512,256] (50441/50442 → 18.71 exact) → PPO on eval-100 | 23.76 (PR #115) |
| PPO round 2 from 23.76, 10M steps (plain / shaped) | **24.64** / 24.56 |

Failure analysis of the BFS teacher: 82/100 failures are per-food CAP loops (alive but looping), only 18 die outright — yet tabu escape only lifts 21.2 → 23.6.

## Findings

1. Memorise a policy's OWN rollout distribution (self-consistent), never an oracle's idealized one.
2. DAgger majority-vote dilutes early critical decisions; repair's targeted choices are better but single-obs moves plateau fast (2/100).
3. Train accuracy is uncorrelated with closed-loop score. Select ONLY on official eval.
4. Closed-loop methods (GA/PPO) beat distillation — no distribution shift.
5. Warm-starting PPO from the previous best works repeatedly (9.33→16.73→18.71); each round needs ~5–10M steps.
6. Eval seeds are fixed (0–99), so training on eval-100 only is legitimate deliberate overfit (same as image memorisation). 300-seed training dilutes capacity.
7. Capacity, not tuning: round-2 variants (shaping, hypers) converge to the same ceiling; bigger nets break it ([512]→[512,256]: 18.71→23.76).
8. PPO protocol details that matter: mirror the official eval EXACTLY (128-tick per-food cap) in rollout termination and eval; normalise with running stats for from-scratch runs but keep raw obs when warm-starting supervised weights; greedy argmax eval with best-checkpointing.

## Reproduce (best line)

```bash
V=<venv python>
# 1. distil winner into big net (BC on winner's own rollouts)
CW_CKPT=winner.pt CW_OUT=pairs-w.pt CW_NSEEDS=100 $V scripts/collect_winner.py
$V scripts/snake_train.py pairs-w-tuple.pt dist/ 512 256 --epochs 2000 --seed 0
# 2. PPO warm-started, eval seeds only
PPO_H=512,256 PPO_INIT=dist/model.pt PPO_STEPS=10000000 PPO_NSTEP=2048 \
  PPO_LR=1.45e-4 PPO_CLIP=0.12 PPO_GAMMA=0.994 PPO_ENT=0.0128 \
  PPO_SEEDLIST=0..99 PPO_OUT=best.pt $V scripts/ppo_obs.py
```

Optuna studies: `scripts/optuna_ppo.py` (round 1, from 9.33), `optuna_ppo2.py` (narrowed, from 16.73), `optuna_ppo3.py` (adam-vs-muon, from 18.71). GA baseline: `scripts/ga_obs.py`.
