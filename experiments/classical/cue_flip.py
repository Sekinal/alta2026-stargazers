"""Cue-flip data for the paper figure: AUC of label-free zero-shot cues for GOLD sarcasm within gold-negative Reddit
comments, per variety, with row-bootstrap 95% CIs (1,000 resamples). Output: results/paper/cue_flip.csv

  uv run python experiments/classical/cue_flip.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT, pool

CUES = [("zs_nli__mocking", "mocking", "derision"), ("zs_nli__contempt", "contempt", "derision"),
        ("zs_nli__angry", "anger", "derision"), ("zs_nli__cynical", "cynicism", "derision"),
        ("zs_nli__joking", "playful joking", "irony"), ("zs_nli__praise", "praise", "irony"),
        ("zs_nli__ironic_praise", "ironic praise", "irony"),
        ("zs_affect__emo7_joy", "joy", "irony")]


def main() -> None:
    df = pool()
    store = ROOT / "experiments" / "features" / "store"
    F = df.select("uid").join(pl.read_parquet(store / "zs_nli__pool.parquet"), on="uid", how="left",
                              maintain_order="left").join(pl.read_parquet(store / "zs_affect__pool.parquet"),
                                                          on="uid", how="left", maintain_order="left")
    y = df["sarcasm"].to_numpy()
    rng = np.random.default_rng(2026)
    rows = []
    for v in ("en-AU", "en-UK"):
        m = np.flatnonzero(((df["variety"] == v) & (df["source"] == "reddit") & (df["sentiment"] == 0)).to_numpy())
        boots = [rng.choice(m, len(m)) for _ in range(1000)]
        for col, label, group in CUES:
            x = F[col].to_numpy()
            auc = roc_auc_score(y[m], x[m])
            bs = [roc_auc_score(y[b], x[b]) for b in boots]
            rows.append({"variety": v, "cue": label, "group": group, "auc": round(auc, 4),
                         "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4),
                         "n": len(m), "n_sarc": int(y[m].sum())})
    out = pl.DataFrame(rows)
    out.write_csv(ROOT / "results" / "paper" / "cue_flip.csv")
    print(out)


if __name__ == "__main__":
    main()
