"""deberta under THREAD-GROUPED folds vs the standard folds (paper table).

enc_deb_l_thr_s{13,14,15} (trained/evaluated with data/external/context/thread_folds.parquet) vs
enc_deb_l_s{13,14,15} (project folds). Policy segment+ukg cross-fitted over each run's OWN folds.
Prints per-seed and seed-averaged components, per-fold scores (per each run's own folds), and the paired
P(standard better) on the whole pool (both cover every row once). Writes results/paper/thread_folds.csv.

  uv run python -m experiments.encoders.thread_compare
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import copy

import numpy as np
import polars as pl

from alta.common import ROOT, load_pool
from alta.validation import policies
from alta.validation.data import load_oof
from alta.validation.metrics import components
from alta.validation.resampling import paired

SEEDS = (13, 14, 15)
ARMS = {"standard": ("enc_deb_l_s{}", None),
        "thread": ("enc_deb_l_thr_s{}", ROOT / "data/external/context/thread_folds.parquet")}


def with_folds(d, pool, folds_file):
    if folds_file is None:
        return d
    f = pool.select("uid").join(pl.read_parquet(folds_file), on="uid", how="left", maintain_order="left")
    d = copy.copy(d)
    d.fold = f["thread_fold"].to_numpy().astype(np.int8)
    return d


def evaluate(d) -> tuple[dict, list[float], dict]:
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    per_fold = [components(d.subset(d.fold == k), {t: v[d.fold == k] for t, v in pred.items()})["score"]
                for k in range(5)]
    return components(d, pred), per_fold, pred


def main() -> None:
    pool = load_pool()
    rows, avg = [], {}
    for arm, (pat, ff) in ARMS.items():
        ds = [with_folds(load_oof(pat.format(s), pool), pool, ff) for s in SEEDS]
        for s, d in zip(SEEDS, ds):
            c, pf, _ = evaluate(d)
            rows.append({"arm": arm, "seed": str(s), **c, "fold_scores": " ".join(f"{x:.3f}" for x in pf),
                         "fold_sd": float(np.std(pf, ddof=1))})
        a = copy.copy(ds[0])
        a.p = {t: np.mean([d.p[t] for d in ds], axis=0) for t in a.p}
        c, pf, pred = evaluate(a)
        avg[arm] = (a, pred)
        rows.append({"arm": arm, "seed": "mean3", **c, "fold_scores": " ".join(f"{x:.3f}" for x in pf),
                     "fold_sd": float(np.std(pf, ddof=1))})
    df = pl.DataFrame(rows)
    out = ROOT / "results" / "paper" / "thread_folds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4, fmt_str_lengths=60):
        print(df)
    (sa, pa), (ta, pt) = avg["standard"], avg["thread"]
    assert (sa.uid == ta.uid).all()
    pr = paired(sa, pa, pt)
    print("paired P(standard-folds arm scores higher than thread-folds arm), test-sized replicates:",
          {k: round(v["p_a_better"], 3) for k, v in pr.items()})
    print(out)


if __name__ == "__main__":
    main()
