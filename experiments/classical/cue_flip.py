"""Cue-flip data for the paper figure: AUC of label-free zero-shot cues for GOLD sarcasm among Reddit comments that a
label-free sentiment model (SiEBERT) calls negative, per variety, with row-bootstrap 95% CIs (1,000 resamples).
Filtering on SiEBERT rather than on the gold sentiment label avoids conditioning on the same annotator's labels
(collider bias). Also reports the unconditional AUCs (all Reddit comments) and per-subreddit AUCs.
Output: results/paper/cue_flip.csv, results/paper/cue_flip_subreddit.csv

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
        m = np.flatnonzero(((df["variety"] == v) & (df["source"] == "reddit")).to_numpy()
                           & (F["zs_affect__sent_siebert"].to_numpy() < 0))
        m_all = np.flatnonzero(((df["variety"] == v) & (df["source"] == "reddit")).to_numpy())
        boots = [rng.choice(m, len(m)) for _ in range(1000)]
        for col, label, group in CUES:
            x = F[col].to_numpy()
            auc = roc_auc_score(y[m], x[m])
            bs = [roc_auc_score(y[b], x[b]) for b in boots]
            rows.append({"variety": v, "cue": label, "group": group, "auc": round(auc, 4),
                         "auc_all": round(float(roc_auc_score(y[m_all], x[m_all])), 4),
                         "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4),
                         "n": len(m), "n_sarc": int(y[m].sum())})
    out = pl.DataFrame(rows)
    out.write_csv(ROOT / "results" / "paper" / "cue_flip.csv")
    print(out)
    ctx = pl.read_parquet(ROOT / "data" / "external" / "context" / "pool_context.parquet").select("uid", "subreddit")
    G = df.select("uid", "variety", "source", "sarcasm").join(ctx, on="uid", how="left").with_columns(
        **{c: F[c] for c, _, _ in CUES}, sieb=F["zs_affect__sent_siebert"])
    G = G.filter((pl.col("source") == "reddit") & (pl.col("sieb") < 0) & pl.col("subreddit").is_not_null())
    sub = []
    for (v, sr), g in G.group_by(["variety", "subreddit"]):
        k = int(g["sarcasm"].sum())
        if g.height < 60 or k < 15 or k > g.height - 15:
            continue
        sub.append({"variety": v, "subreddit": sr, "n": g.height, "sarc_rate": round(k / g.height, 3),
                    **{lab: round(float(roc_auc_score(g["sarcasm"], g[c])), 3) for c, lab, _ in CUES}})
    sub = pl.DataFrame(sub).sort("variety", "n", descending=[False, True])
    sub.write_csv(ROOT / "results" / "paper" / "cue_flip_subreddit.csv")
    print(sub)


if __name__ == "__main__":
    main()
