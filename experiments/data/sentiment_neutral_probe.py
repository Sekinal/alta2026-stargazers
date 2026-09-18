"""How would the task's annotators label external 'neutral' text? Fit TF-IDF + LR on the pool's reddit
sentiment labels and score each neutral bucket of the external corpora. Reference: the same model's
out-of-fold predicted-positive rate on pool reddit (which equals ~ the gold positive rate).

    uv run --with scikit-learn python experiments/data/sentiment_neutral_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402

EXT = ROOT / "data" / "external" / "sentiment"


def model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))


def main() -> None:
    pool = load_pool().filter(pl.col("source") == "reddit")
    x, y = pool["text"].to_list(), pool["sentiment"].to_numpy()
    oof = np.zeros(len(x))
    folds = pool["fold"].to_numpy()
    for f in np.unique(folds):
        m = model().fit([t for t, g in zip(x, folds) if g != f], y[folds != f])
        oof[folds == f] = m.predict_proba([t for t, g in zip(x, folds) if g == f])[:, 1]
    print(f"pool reddit: gold pos {y.mean():.3f}  oof mean P(pos) {oof.mean():.3f}  oof >0.5 {(oof > .5).mean():.3f}")
    for v in ("en-AU", "en-UK"):
        k = (pool["variety"] == v).to_numpy()
        print(f"  {v}: gold pos {y[k].mean():.3f}  oof mean P(pos) {oof[k].mean():.3f}")
    m = model().fit(x, y)
    for f in sorted(EXT.glob("*__neutral.parquet")):
        df = pl.read_parquet(f)
        grp = df.with_columns(pl.col("orig_label").str.split("|").list.first().alias("g")) if "go_emotions" in f.name \
            else df.with_columns(pl.col("orig_label").str.split(":").list.first().alias("g"))
        grp = grp.with_columns(pl.Series("p", m.predict_proba(grp["text"].to_list())[:, 1]))
        out = grp.group_by("g").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"),
                                    (pl.col("p") > 0.5).mean().round(3).alias("frac_pos")).sort("n", descending=True)
        print(f"\n{f.name}  (all: mean P(pos) {grp['p'].mean():.3f}, >0.5 {(grp['p'] > .5).mean():.3f})")
        print(out.head(12))
    for name in ("go_emotions", "tweet_eval_sentiment", "qut_ausvotes2025"):
        b = pl.read_parquet(EXT / f"{name}__full.parquet")
        p, t = m.predict_proba(b["text"].to_list())[:, 1], b["sentiment"].to_numpy()
        print(f"{name} binary: pool-model P(pos)>0.5 on pos {np.mean(p[t == 1] > .5):.3f}, "
              f"on neg {np.mean(p[t == 0] > .5):.3f}, AUC {roc_auc_score(t, p):.3f}")


if __name__ == "__main__":
    main()
