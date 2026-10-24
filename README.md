# minimal-nn-research

Research and experimentation log for the [minimal-nn-competition](https://wd7512.github.io/minimal-nn-competition/) — how small can a network be while exactly memorising a task, and where does memorisation break.

The competition repo is vendored as a submodule (`vendor/minimal-nn-competition`, pinned to `origin/dev` at time of work). Everything here runs against it; model submissions live there as `models/<benchmark>/<name>/{config.json,model.pt}`.

## Scoreboard (official evaluator, 2026-10-24)

| Problem | Bar | Submission | Params | Result |
|---|---|---|---|---|
| MNIST | 100% of 70k | `models/mnist/mlp-14-14` (PR #92) | 11,350 | PASS |
| Fashion-MNIST | 100% of 70k | `models/fashion-mnist/mlp-35` (PR #93) | 27,835 | PASS |
| Snake 16x16 | fill 256 cells, all 100 seeds | `models/snake-simple-16x16/serp-36` (PR #94) | 400 | fail, 3.79 |
| Snake obstacles 8x8 | fill board-minus-obstacles, all 100 seeds | `models/snake-obstacles-8x8/obs-fb` (PR #95) | 14,852 | fail, 9.33 |

Standing record for snake 16x16 remains `cycle-44` (488 params, passing).

## Layout

- `problems/<name>/` — per-problem notes: the bar, what was tried, run tables, the key finding, how to reproduce.
- `methods/` — cross-cutting techniques that worked (or instructively failed).
- `scripts/` — the exact training/collection/eval scripts used, copied verbatim from the work sessions.
- `vendor/minimal-nn-competition` — the competition + submission repo (submodule).

## Reproduce

```bash
git clone --recurse-submodules <this-repo>
cd minimal-nn-research
git -C vendor/minimal-nn-competition checkout <pinned-dev-sha>  # see .gitmodules log
# then follow the per-problem READMEs; they assume the venv + data caches described there
```

## Headline lessons

1. On the 70k memorisation tasks, the recipe is: zero weight decay, long cosine-to-zero schedules, best-state checkpointing, halve widths, extend the schedule on near-misses before upsizing. Train accuracy = eval score (static task).
2. On closed-loop (snake) tasks, train accuracy is UNCORRELATED with score. First-divergence distribution shift kills memorised policies. Always select on official closed-loop eval, never on fit.
3. Memorise a policy's OWN rollout distribution (self-consistent), not an oracle's idealized one.
4. Exact float32 obs tuples as table keys. Rounding silently breaks lookups (zero matches, no error).
