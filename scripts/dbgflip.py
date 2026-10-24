"""Debug: count flippable 2x2 squares in the serpentine cycle."""
import random
import sys

sys.path.insert(0, "/Users/williamdennis/.hermes/cache/scratch/snake16")
from flipanneal import SIZE, col_path, square_flip, succ_of, valid_cycle  # noqa: E402

base = succ_of(col_path)
print("valid:", valid_cycle(base))

rng = random.Random(0)
ok, none = 0, 0
for _ in range(300):
    r = square_flip(base, rng)
    if r is None:
        none += 1
    else:
        ok += 1
print(f"flips={ok} none={none}")

# manual count over all squares
sys.path.insert(0, "/Users/williamdennis/repos/minimal-nn-competition/.worktrees/sweep-mnist/src")
count = 0
for x in range(SIZE - 1):
    for y in range(SIZE - 1):
        a, b, c, d = (x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)
        edges = {(a, b), (b, a), (b, c), (c, b), (c, d), (d, c), (d, a), (a, d)}
        ins = [(u, base[u]) for u in (a, b, c, d) if (u, base.get(u)) in edges]
        if len(ins) == 2:
            (u1, v1), (u2, v2) = ins
            got = sorted([tuple(sorted((u1, v1))), tuple(sorted((u2, v2)))])
            horiz = sorted([tuple(sorted((a, b))), tuple(sorted((c, d)))])
            vert = sorted([tuple(sorted((b, c))), tuple(sorted((d, a)))])
            if got == horiz or got == vert:
                count += 1
print("flippable squares:", count)
