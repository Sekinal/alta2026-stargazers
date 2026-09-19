"""Annotator surrogate (paper analysis): which interpretable signals predict each variety's SARCASM labels?

Reddit rows only, one L2 logistic regression (standardised features, fixed C) per variety and block:
  A  thread metadata : subreddit one-hot (+ unmatched level), is_top_level, matched          (context file)
  B  topic/polarity  : 12 NMF topics of word TF-IDF (unsupervised, fit on all Reddit texts), politics-entity
                       counts (AU/UK), surface polarity (VADER compound, pos/neg rates), log length
  C  pragmatic cues  : fe-linguistic ling_pragmatic (28 features)
  ALL = A+B+C;  ref: gold negative sentiment alone (a reference, not a surrogate input)
Outputs (results/paper/, no raw text):
  surrogate_auc.csv       5-fold AUC (standard folds) per variety x block, fold SD
  surrogate_coef.csv      coefficients of the full-data fit with 95% bootstrap CIs (500 resamples)
  surrogate_transfer.csv  train on one variety's Reddit rows, AUC on the other's
  surrogate_topics.csv    top words per NMF topic (aggregate vocabulary only)

  uv run python experiments/classical/surrogate.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import ROOT, pool
from context import load_ctx
from features import handcrafted

OUT = ROOT / "results" / "paper"
STORE = ROOT / "experiments" / "features" / "store"
C = 0.1
N_BOOT = 500
SEED = 2026


def blocks(d: pl.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    c = load_ctx(d)
    subs = sorted(x for x in c.filter(pl.col("matched"))["subreddit"].unique().to_list()
                  if c.filter(pl.col("subreddit") == x).height >= 20)
    A = [(c["subreddit"] == s).fill_null(False).cast(pl.Float64).to_numpy() for s in subs]
    A += [(~c["matched"]).cast(pl.Float64).to_numpy(),
          (c["matched"] & c["is_top_level"].fill_null(False)).cast(pl.Float64).to_numpy()]
    a_names = [f"sub={s}" for s in subs] + ["unmatched", "top_level"]

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, stop_words="english", sublinear_tf=True)
    X = tf.fit_transform(d["norm"].to_list())
    nmf = NMF(12, random_state=SEED, init="nndsvda", max_iter=500)
    T = nmf.fit_transform(X)
    vocab = np.array(tf.get_feature_names_out())
    top = [", ".join(vocab[np.argsort(-w)[:8]]) for w in nmf.components_]
    pd_ = pl.read_parquet(STORE / "ling_polarity__pool.parquet")
    pol = d.select("uid").join(pd_, on="uid", how="left", maintain_order="left")
    H, hn = handcrafted(d)
    B = [T[:, i] for i in range(12)] + [H[:, hn.index("pol_au")], H[:, hn.index("pol_uk")],
                                        pol["ling_polarity__compound"].to_numpy(),
                                        pol["ling_polarity__pos_rate"].to_numpy(),
                                        pol["ling_polarity__neg_rate"].to_numpy(), H[:, hn.index("log_words")]]
    b_names = [f"topic{i}" for i in range(12)] + ["pol_entities_au", "pol_entities_uk", "vader_compound",
                                                  "pos_rate", "neg_rate", "log_words"]
    pr = d.select("uid").join(pl.read_parquet(STORE / "ling_pragmatic__pool.parquet"), on="uid", how="left",
                              maintain_order="left")
    c_names = [x for x in pr.columns if x != "uid"]
    Cm = pr.select(c_names).to_numpy()
    out = {"A_thread": (np.column_stack(A), a_names), "B_topic_polarity": (np.column_stack(B), b_names),
           "C_pragmatic": (Cm, [x.replace("ling_pragmatic__", "") for x in c_names])}
    out["ALL"] = (np.hstack([v[0] for v in out.values()]), sum((v[1] for v in out.values()), []))
    pl.DataFrame({"topic": [f"topic{i}" for i in range(12)], "top_words": top}).write_csv(OUT / "surrogate_topics.csv")
    return out


def fit(X, y):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], m.coef_.ravel()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pool().filter(pl.col("source").cast(pl.String) == "reddit")
    B = blocks(df)
    var = df["variety"].cast(pl.String).to_numpy()
    y = df["sarcasm"].to_numpy()
    neg = 1 - df["sentiment"].to_numpy()
    fold = df["fold"].to_numpy()
    auc_rows, coef_rows, tr_rows = [], [], []
    rng = np.random.default_rng(SEED)
    for v in ("en-AU", "en-UK"):
        m = var == v
        auc_rows.append({"variety": v, "block": "ref_gold_negative", "cv_auc": roc_auc_score(y[m], neg[m]),
                         "fold_sd": float(np.std([roc_auc_score(y[m & (fold == k)], neg[m & (fold == k)]) for k in range(5)])),
                         "n": int(m.sum()), "n_pos": int(y[m].sum())})
        for name, (X, names) in B.items():
            p = np.zeros(len(y))
            for k in range(5):
                tr, te = m & (fold != k), m & (fold == k)
                p[te] = fit(X[tr], y[tr])[0](X[te])
            per = [roc_auc_score(y[m & (fold == k)], p[m & (fold == k)]) for k in range(5)]
            auc_rows.append({"variety": v, "block": name, "cv_auc": roc_auc_score(y[m], p[m]),
                             "fold_sd": float(np.std(per)), "n": int(m.sum()), "n_pos": int(y[m].sum())})
            _, coef = fit(X[m], y[m])
            idx = np.flatnonzero(m)
            boots = np.array([fit(X[b], y[b])[1] for b in (rng.choice(idx, len(idx)) for _ in range(N_BOOT))])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for f, c_, l_, h_ in zip(names, coef, lo, hi):
                coef_rows.append({"variety": v, "block": name, "feature": f, "coef_std": c_, "ci_lo": l_, "ci_hi": h_,
                                  "ci_excludes_0": bool(l_ > 0 or h_ < 0)})
            # cross-variety transfer: fit on this variety, score the other
            o = (var != v)
            pred, _ = fit(X[m], y[m])
            tr_rows.append({"train": v, "test": "en-UK" if v == "en-AU" else "en-AU", "block": name,
                            "transfer_auc": roc_auc_score(y[o], pred(X[o]))})
    auc = pl.DataFrame(auc_rows)
    coefs = pl.DataFrame(coef_rows)
    trans = pl.DataFrame(tr_rows)
    auc.write_csv(OUT / "surrogate_auc.csv")
    coefs.write_csv(OUT / "surrogate_coef.csv")
    trans.write_csv(OUT / "surrogate_transfer.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(auc)
        print(trans)
        for v in ("en-AU", "en-UK"):
            print(v, coefs.filter(pl.col("variety") == v, pl.col("block") == "ALL")
                  .sort(pl.col("coef_std").abs(), descending=True).head(15))
    print(pl.read_csv(OUT / "surrogate_topics.csv"))


if __name__ == "__main__":
    main()
