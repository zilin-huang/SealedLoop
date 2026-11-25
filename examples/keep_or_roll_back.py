"""Decide whether a candidate policy replaces the current one.

Both policies are scored on the same evaluation items by a judge that the
improvement loop never sees. The test reads the paired scores one item at a
time and stops at the first decision.

    python examples/keep_or_roll_back.py
"""
import numpy as np

from sealedloop import GateConfig, gate_path

rng = np.random.default_rng(0)
n_items = 100

# Success (1) or failure (0) of each policy on each item. Replace these with
# the scores of your own judge; any score with a fixed range works.
current = (rng.random(n_items) < 0.60).astype(float)
candidates = {
    "better": np.maximum(current, (rng.random(n_items) < 0.40).astype(float)),
    "same": current.copy(),
    "worse": np.minimum(current, (rng.random(n_items) < 0.50).astype(float)),
}

cfg = GateConfig(alpha=0.05, budget=n_items, min_n=10, score_range=(0.0, 1.0))

for name, candidate in candidates.items():
    reading = gate_path(candidate, current, cfg)
    used = reading.stopped_at if reading.stopped_at is not None else reading.n
    print(f"{name:7s} {'keep' if reading.keep else 'roll back':9s} "
          f"({reading.decision}, {used} of {n_items} items, "
          f"gain in [{reading.lo:+.3f}, {reading.hi:+.3f}])")
