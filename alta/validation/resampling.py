"""Test-like resampling: what would these predictions score on a hidden test set?

valid.csv is an exact stratified sample of the pool (every variety x source x sentiment x
sarcasm cell within +-2 rows of proportional), so the hidden test set very likely is too.
We therefore draw replicate "test sets" with fixed per-cell counts (valid's counts by
default), sampling rows with replacement within each cell from the honest OOF labels.

Paired comparisons reuse the same replicate indices for both systems, which removes the
shared test-sampling noise and leaves only the noise in their difference.
"""

from __future__ import annotations

import numpy as np

from alta.validation.data import TASKS, EvalData
from alta.validation.metrics import macro_f1_counts


def cell_ids(d: EvalData) -> np.ndarray:
    """16 cells: variety x source x sentiment x sarcasm."""
    return d.variety * 8 + d.source * 4 + d.y["sent"] * 2 + d.y["sarc"]


def target_counts(d: EvalData, n_test: int | None = None) -> dict[int, int]:
    """Per-cell counts of a test-like set: valid's counts, or pool proportions scaled to n_test."""
    cells = cell_ids(d)
    if n_test is None:
        m = d.origin == "valid"
        ids, cnt = np.unique(cells[m], return_counts=True)
        return dict(zip(ids.tolist(), cnt.tolist()))
    ids, cnt = np.unique(cells, return_counts=True)
    return {int(i): int(round(c * n_test / len(d))) for i, c in zip(ids, cnt)}


def replicate_indices(d: EvalData, B: int = 2000, seed: int = 0,
                     n_test: int | None = None) -> dict[int, np.ndarray]:
    """variety -> (B, n_variety) row indices of replicate test sets."""
    rng = np.random.default_rng(seed)
    cells = cell_ids(d)
    counts = target_counts(d, n_test)
    out = {}
    for v in (0, 1):
        parts = []
        for c, k in sorted(counts.items()):
            if c // 8 != v or k == 0:
                continue
            members = np.flatnonzero(cells == c)
            parts.append(members[rng.integers(0, len(members), size=(B, k))])
        out[v] = np.concatenate(parts, axis=1)
    return out


def replicate_scores(d: EvalData, pred: dict[str, np.ndarray],
                     idx: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Component and final scores for every replicate."""
    out = {}
    for t in TASKS:
        for v, vn in ((0, "AU"), (1, "UK")):
            y = d.y[t][idx[v]].astype(bool)
            h = pred[t][idx[v]].astype(bool)
            out[f"{t}-{vn}"] = macro_f1_counts((y & h).sum(1), (~y & h).sum(1),
                                               (y & ~h).sum(1), (~y & ~h).sum(1))
    out["score"] = (np.minimum(out["sent-AU"], out["sent-UK"])
                    + np.minimum(out["sarc-AU"], out["sarc-UK"])) / 2
    return out


def summarise(x: np.ndarray, reference: float | None = None) -> dict[str, float]:
    s = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
         "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95))}
    if reference is not None:
        s["p_above_ref"] = float((x > reference).mean())
    return s


def paired(d: EvalData, pred_a: dict, pred_b: dict, B: int = 2000, seed: int = 0,
           n_test: int | None = None) -> dict[str, dict]:
    """Distribution of (A - B) on shared test-like replicates, per component and final score."""
    idx = replicate_indices(d, B, seed, n_test)
    ra, rb = replicate_scores(d, pred_a, idx), replicate_scores(d, pred_b, idx)
    out = {}
    for k in ra:
        diff = ra[k] - rb[k]
        out[k] = {**summarise(diff), "p_a_better": float((diff > 0).mean()),
                  "p_tie": float((diff == 0).mean())}
    return out


def binding_rates(reps: dict[str, np.ndarray]) -> dict[str, float]:
    """How often each variety is the min (binding) one, per task."""
    return {"sent: AU binds": float((reps["sent-AU"] <= reps["sent-UK"]).mean()),
            "sarc: UK binds": float((reps["sarc-UK"] <= reps["sarc-AU"]).mean())}
