"""Is the test-like bootstrap SD right? Compare, on a saved run's cross-fitted OOF labels:
  (a) resampling.replicate_indices (with replacement, valid cell counts)  -- what report() uses
  (b) the same cell counts drawn WITHOUT replacement from the pool           -- a pseudo-test draw
  PYTHONPATH=. uv run python redteam/calibration/04_boot_vs_subsample.py [RUN] [POLICY]
"""
import sys

import numpy as np

from alta.validation import policies
from alta.validation.data import load_oof
from alta.validation.resampling import cell_ids, replicate_indices, replicate_scores, target_counts

run = sys.argv[1] if len(sys.argv) > 1 else "emb_bge_large_lr"
spec = sys.argv[2] if len(sys.argv) > 2 else "variety+ukg"
d = load_oof(run)
pred, _ = policies.crossfit(spec, d)
B = 4000
boot = replicate_scores(d, pred, replicate_indices(d, B=B, seed=0))

rng = np.random.default_rng(1)
cells, counts = cell_ids(d), target_counts(d)
idx = {0: [], 1: []}
for _ in range(B):
    for v in (0, 1):
        idx[v].append(np.concatenate([rng.choice(np.flatnonzero(cells == c), size=k, replace=False)
                                      for c, k in sorted(counts.items()) if c // 8 == v and k]))
sub = replicate_scores(d, pred, {v: np.stack(idx[v]) for v in (0, 1)})
print(f"run {run}, policy {spec}")
for k in boot:
    print(f"{k:8s} boot mean {boot[k].mean():.4f} sd {boot[k].std():.4f} | "
          f"no-replacement mean {sub[k].mean():.4f} sd {sub[k].std():.4f}")
