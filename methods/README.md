# Methods

## memorisation-sweep.md

Static classification memorisation (MNIST/Fashion): AdamW, cosine-to-zero, batch 256, lr 3e-3, **zero weight decay**, best-state checkpointing, normalised inputs. Start wide (512-256), halve, extend schedule on near-misses (≤0.1% short) before upsizing. Fine-grain the floor (24→20→[20,10]→[16,16]→[14,14]). Diagnose stalls with a 3x-LR probe: unchanged best = capacity wall, not LR. See `scripts/mem_train.py`, `scripts/build_caches.py`.

## snake-distillation.md

Analytic serpentine teacher (`scripts/serp16.py`, column-0 spine; reverse the path so spawn (8,8) matches the engine's initial heading right). Distill head-6 obs → tour action (`scripts/snake_train.py`: CE/MSE, minibatch, `--standardize` with exact fold-back into the first linear so the submission arch is unchanged, always saves `best.pt`, writes `model.pt`+config only on 100% fit). Standardisation is the only large win; width scaling inverts past 36; long schedules regress.

## dagger.md

DAgger loop for closed-loop tasks (`scripts/dagger2.py`): rollout student → relabel visited states with the full-state teacher (rebuild env per state via settable `snake`/`food`/`direction`) → aggregate COUNTS (persistent dict, not majority) → retrain → select on official eval. Rebuild the aggregate from raw pairs with exact float32 keys. Expect ~+1 mean/round on obstacles; keep a seed lottery.

## repair-search.md

First-divergence table repair (`scripts/repair2.py`): quantized-key majority table, per-seed oracle actions, screen candidates on the failing seed's length, confirm on global (passed, total) with no-regression on passing seeds. Log to file (not pipe) for visibility. Plateaus fast on obstacles (2/100) — single-obs moves can't build detours.

## tabu-teacher.md

When a greedy teacher CAP-loops (alive but stuck), replace its stall fallback with least-visited-safe escape: track `(snake, food)` visit counts, at count ≥ 3 pick the tail-safe move minimizing resulting-state visits (`scripts/collect_tabu.py`). Lifted BFS 21.2 → 23.6 mean on obstacles. Diagnose first with a failure-mode split (cap-loop vs dead-end, `scripts/failmode.py`).

## eval-hygiene.md

- Train accuracy ≠ closed-loop score. Every selection decision uses the official evaluator.
- `model.pt` is only written on 100% fit; copy `best.pt` for near-miss evals (and write the config by hand — benchmark, name, author, arch, input_size, hidden_sizes, activation, input_indices).
- Key tables by exact float32 obs tuples (`tuple(torch.tensor(obs, dtype=torch.float32).tolist())`). Rounding to N decimals silently merges/splits states and zeroes table hit rates with no error.
- Image evals need `--data-path` to the combined 70k cache; snake evals take none.
- Background long runs with notify; never pipe blind (`| tail`) on jobs whose progress you need — log to file.
