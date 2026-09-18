"""Run all pseudo-leaderboard draws in parallel (CPU only); results -> redteam/calibration/out/.

  PYTHONPATH=. uv run python redteam/calibration/01_run_draws.py [N_RANDOM] [N_JOBS]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from joblib import Parallel, delayed

import nest

n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 60
n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
jobs = [("valid", s) for s in range(5)] + [("random", s) for s in range(n_random)]
t0 = time.time()
for i, d in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(nest.run_draw)(k, s) for k, s in jobs)):
    print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(jobs)} done: {d}", flush=True)
