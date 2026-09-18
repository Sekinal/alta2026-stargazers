"""Does the other variety help? Learning curves + probe-direction similarity + negativity rule.

For each target variety V and training fraction f of V's training folds, fit a bge-large linear
probe on (f of V) alone vs (f of V) + all of the other variety's training folds; score OOF on V.
"""

import warnings

import numpy as np
import polars as pl
from lets_plot import aes, facet_wrap, geom_line, geom_point, ggplot, labs, scale_color_manual, scale_linetype_manual
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from _shared import CACHE, emb, my_theme, pool, save
from alta.common import N_FOLDS, VARIETIES

warnings.filterwarnings("ignore")
FRACS = (0.25, 0.5, 0.75, 1.0)
C = 10.0


def features(df, X):
    return np.hstack([X, (df["source"] == "reddit").to_numpy()[:, None].astype(np.float32)])


def main():
    df = pool()
    X = features(df, emb("bge_large"))
    var, fold = df["variety"].to_numpy(), df["fold"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for task in ("sarcasm", "sentiment"):
        y = df[task].to_numpy()
        for v in VARIETIES:
            for frac in FRACS:
                for mode in ("own only", "own + other variety"):
                    pred = np.full(len(df), np.nan)
                    for f in range(N_FOLDS):
                        own = np.where((fold != f) & (var == v))[0]
                        own = rng.choice(own, int(len(own) * frac), replace=False)
                        tr = own if mode == "own only" else np.concatenate([own, np.where((fold != f) & (var != v))[0]])
                        te = (fold == f) & (var == v)
                        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
                        pred[te] = clf.predict_proba(X[te])[:, 1]
                    m = var == v
                    s = f1_score(y[m], pred[m] >= 0.5, average="macro")
                    rows.append({"task": task, "variety": v, "frac": frac, "mode": mode, "macroF1": s})
                    print(rows[-1])
    lc = pl.DataFrame(rows)
    lc.write_parquet(CACHE / "learning_curves.parquet")
    p = (ggplot(lc, aes("frac", "macroF1", color="variety", linetype="mode")) + my_theme()
         + geom_line(size=1.2) + geom_point(size=3)
         + facet_wrap("task", scales="free_y")
         + scale_color_manual(values={"en-AU": "#1b9e77", "en-UK": "#7570b3"})
         + scale_linetype_manual(values={"own only": "solid", "own + other variety": "dashed"})
         + labs(title="Learning curves: does adding the other variety's data help?",
                subtitle="bge-large linear probe, OOF macro-F1 on the target variety; x = fraction of the target variety's own training rows",
                x="fraction of own-variety training data", y="macro-F1 (target variety)", color="", linetype=""))
    save(p, "learning_curves", 1200, 560)

    # probe direction similarity: sarcasm weight vectors fit on AU-reddit vs UK-reddit
    Xe = emb("bge_large")
    w = {}
    for task in ("sarcasm", "sentiment"):
        for v in VARIETIES:
            m = ((var == v) & (df["source"] == "reddit").to_numpy())
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xe[m], df[task].to_numpy()[m])
            w[(task, v)] = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        print(f"cosine(AU-reddit, UK-reddit) {task} probe weights: {w[(task, 'en-AU')] @ w[(task, 'en-UK')]:.3f}")
    print(f"cosine(sarcasm, sentiment) AU: {w[('sarcasm', 'en-AU')] @ w[('sentiment', 'en-AU')]:.3f} "
          f"UK: {w[('sarcasm', 'en-UK')] @ w[('sentiment', 'en-UK')]:.3f}")

    # negativity rule: sarcasm := reddit & negative (gold sentiment) -> how far does it get?
    for v in VARIETIES:
        d = df.filter(pl.col("variety") == v)
        rule = ((d["source"] == "reddit") & (d["sentiment"] == 0)).to_numpy()
        r = d.filter(pl.col("source") == "reddit")
        print(f"{v}: rule 'reddit & negative' sarcasm macro-F1 = {f1_score(d['sarcasm'], rule, average='macro'):.3f}; "
              f"P(sarc|reddit,neg)={r.filter(pl.col('sentiment') == 0)['sarcasm'].mean():.3f} "
              f"P(sarc|reddit,pos)={r.filter(pl.col('sentiment') == 1)['sarcasm'].mean():.3f} "
              f"P(neg|reddit,sarc)={1 - r.filter(pl.col('sarcasm') == 1)['sentiment'].mean():.3f}")


if __name__ == "__main__":
    main()
