"""Paper analysis: the lead's score under standard vs link_id thread-grouped folds.

For the same member set, the weight-free logit mean is built from (a) the members' standard OOF and
(b) their thread-fold OOF (runs saved as MEMBER__thr); segment+ukg is cross-fitted on the matching folds
(standard `fold` vs `thread_fold` from data/external/context/thread_folds.parquet). Reports score,
components, per-fold scores, and the gap; also each member alone. Writes results/paper/thread_gap.csv.

  uv run python experiments/classical/thread_gap.py enc_deb_l_x3 cls_e5me5_lr llm_q35_9b_probe
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from common import ROOT
from alta.common import load_pool
from alta.validation import policies
from alta.validation.data import load_oof
from alta.validation.metrics import components

OUT = ROOT / "results" / "paper"
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def evaluate(d, folds):
    d = d.with_p(d.p)
    d.fold = folds.astype(np.int8)
    pred, _ = policies.crossfit("segment+ukg", d)
    comp = components(d, pred)
    per = []
    for k in range(5):
        m = folds == k
        per.append(components(d.subset(m), {t: v[m] for t, v in pred.items()})["score"])
    return comp, per


def main(members: list[str]) -> None:
    pool = load_pool()
    tf = pool.select("uid").join(pl.read_parquet(ROOT / "data/external/context/thread_folds.parquet"), on="uid",
                                 how="left", maintain_order="left")["thread_fold"].to_numpy()
    std = pool["fold"].to_numpy()
    rows = []
    for scheme, suffix, folds in (("standard", "", std), ("thread", "__thr", tf)):
        ds = [load_oof(m + suffix, pool) for m in members]
        blend = ds[0].with_p({t: 1 / (1 + np.exp(-np.mean([logit(d.p[t]) for d in ds], 0))) for t in ("sent", "sarc")})
        for name, d in [("blend(" + "+".join(members) + ")", blend)] + list(zip(members, ds)):
            comp, per = evaluate(d, folds)
            rows.append({"model": name, "scheme": scheme, **{k: round(v, 4) for k, v in comp.items()},
                         **{f"fold{k}": round(v, 4) for k, v in enumerate(per)}, "fold_sd": round(float(np.std(per)), 4)})
    res = pl.DataFrame(rows)
    gap = res.pivot(on="scheme", index="model", values="score").with_columns(
        (pl.col("thread") - pl.col("standard")).round(4).alias("gap"))
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_csv(OUT / "thread_gap.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=250, fmt_str_lengths=60):
        print(res)
        print(gap)


if __name__ == "__main__":
    main(sys.argv[1:])
