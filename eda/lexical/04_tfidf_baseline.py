"""Interpretable lexical baseline: TF-IDF (word 1-2g + char_wb 2-5g) + LogisticRegression, provided 5 folds.

Variants (vectorisers are fit on the training folds only, then shared by all variants of that fold):
  global       one model per task on everything
  global+seg   + one-hot source x variety indicators (segment-specific intercepts)
  augment      Daume (2007) feature augmentation by variety: [X, X*isAU, X*isUK] + segment indicators
  per-variety  separate model per variety
  per-segment  separate model per source x variety (google-UK sarcasm has 1 positive -> predicts ~0)
  AU->UK / UK->AU  train on one variety only, predict the other (cross-variety transfer)
Scoring: alta.common.score, (a) fixed threshold 0.5, (b) best per-variety threshold chosen ON THE SAME
OOF predictions (optimistic upper bound), (c) cross-fitted per-variety threshold: for fold f the threshold
is tuned on the OOF rows of the other 4 folds (honest estimate of what threshold tuning buys).
Saves the best variant by (c) with alta.common.save_preds('tfidf_lr', ...).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp
from lets_plot import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from _shared import INK2, TABLES, TOKEN_PATTERN, VAR_COL, pool_with_norm, save, th
from alta.common import macro_f1, save_preds, score

VARIANTS = ["global", "global+seg", "augment", "per-variety", "per-segment", "AU->UK / UK->AU"]


def plot_scores(res: pl.DataFrame) -> None:
    long = res.unpivot(index=["variant", "mode"], on=["sent-AU", "sent-UK", "sarc-AU", "sarc-UK"],
                       variable_name="component", value_name="f1").with_columns(
        pl.col("component").str.slice(0, 4).replace({"sent": "sentiment", "sarc": "sarcasm"}).alias("task"),
        pl.col("component").str.slice(5).replace({"AU": "en-AU", "UK": "en-UK"}).alias("variety"))
    vo = {v: i for i, v in enumerate(VARIANTS)}
    long = long.with_columns(pl.col("variant").replace_strict(vo).alias("vo"))
    tot = res.select("variant", "mode", "score").with_columns(pl.col("variant").replace_strict(vo).alias("vo"))
    p = (
        ggplot(long, aes(y=as_discrete("variant", order_by="vo", order=-1)))
        + geom_point(aes(x="f1", color="variety", shape="task"), size=4.2,
                     tooltips=layer_tooltips().line("@variant · @mode").line("@task @variety macro-F1 @f1")
                     .format("@f1", ".3f"))
        + geom_point(aes(x="score"), data=tot, size=6, color="#0b0b0b",
                     tooltips=layer_tooltips().line("official score @score").format("@score", ".3f"))
        + scale_color_manual(values=VAR_COL) + scale_shape_manual(values={"sentiment": 16, "sarcasm": 17})
        + facet_wrap("mode", ncol=1)
        + labs(title="TF-IDF + LogReg baseline: out-of-fold macro-F1 per component",
               subtitle="5-fold OOF over train+valid · black dot = official score (min over varieties, averaged over tasks)"
                        "\ntuned = threshold chosen on the OOF itself (optimistic); cross-fit = tuned on the other 4 folds (honest)",
               x="macro-F1")
        + th("top") + theme(axis_title_y=element_blank())
    )
    save(p, "tfidf_lr_scores", w=1100, h=950)


if "--plot-only" in sys.argv:
    plot_scores(pl.read_csv(TABLES / "tfidf_lr_scores.csv"))
    sys.exit()

pool = pool_with_norm()
N = pool.height
TASKS = ("sentiment", "sarcasm")
Y = {t: pool[t].to_numpy() for t in TASKS}
folds = pool["fold"].to_numpy()
is_au = (pool["variety"] == "en-AU").to_numpy()
seg_codes = pool["segment"].to_numpy()
SEGS = sorted(set(seg_codes))
S = sp.csr_matrix(np.stack([(seg_codes == s).astype(float) for s in SEGS], 1))
C = {"sentiment": 4.0, "sarcasm": 2.0}


def fit_predict(Xtr, ytr, Xte, c):
    if ytr.min() == ytr.max() or ytr.sum() < 3:  # degenerate cell (google-UK sarcasm)
        return np.full(Xte.shape[0], ytr.mean())
    m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear")
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


oof = {v: {t: np.zeros(N) for t in TASKS} for v in VARIANTS}

for f in range(5):
    tr, te = folds != f, folds == f
    texts = pool["norm"].to_list()
    wv = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, max_features=60_000)
    cv = TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                         max_features=150_000)
    tr_txt = [t for t, k in zip(texts, tr) if k]
    wv.fit(tr_txt), cv.fit(tr_txt)
    X = sp.hstack([wv.transform(texts), cv.transform(texts)]).tocsr()
    XS = sp.hstack([X, S]).tocsr()
    au = sp.diags(is_au.astype(float))
    XA = sp.hstack([X, au @ X, sp.diags((~is_au).astype(float)) @ X, S]).tocsr()
    for t in TASKS:
        y, c = Y[t], C[t]
        oof["global"][t][te] = fit_predict(X[tr], y[tr], X[te], c)
        oof["global+seg"][t][te] = fit_predict(XS[tr], y[tr], XS[te], c)
        oof["augment"][t][te] = fit_predict(XA[tr], y[tr], XA[te], c)
        for vm in (is_au, ~is_au):
            oof["per-variety"][t][te & vm] = fit_predict(X[tr & vm], y[tr & vm], X[te & vm], c)
            # cross-variety: train on the OTHER variety, predict this one
            oof["AU->UK / UK->AU"][t][te & vm] = fit_predict(X[tr & ~vm], y[tr & ~vm], X[te & vm], c)
        for s in SEGS:
            sm = seg_codes == s
            oof["per-segment"][t][te & sm] = fit_predict(X[tr & sm], y[tr & sm], X[te & sm], c)
    print(f"fold {f} done  (features: {X.shape[1]:,})", flush=True)

# all variants' OOF probabilities, for later inspection / ensembling
pool.select("uid").with_columns(
    pl.Series(f"{v}|{t}", oof[v][t]) for v in VARIANTS for t in TASKS
).write_parquet(TABLES / "tfidf_oof_all_variants.parquet")


def tuned_threshold(p, y):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [macro_f1(y, (p >= g).astype(int)) for g in grid]
    return float(grid[int(np.argmax(f1s))])


rows = []
for v in VARIANTS:
    for mode in ("thr=0.5", "thr=tuned/variety", "thr=cross-fit/variety"):
        df = pool.select("uid", "variety", "source", "sentiment", "sarcasm")
        preds, thr = {}, {}
        for t in TASKS:
            p = oof[v][t]
            pr = np.zeros(N, int)
            for vm, vn in ((is_au, "AU"), (~is_au, "UK")):
                if mode == "thr=cross-fit/variety":
                    ths = []
                    for f in range(5):
                        fit, app = vm & (folds != f), vm & (folds == f)
                        ths.append(tuned_threshold(p[fit], Y[t][fit]))
                        pr[app] = (p[app] >= ths[-1]).astype(int)
                    thr[f"{t[:4]}-{vn}"] = round(float(np.mean(ths)), 3)
                    continue
                th_ = 0.5 if mode == "thr=0.5" else tuned_threshold(p[vm], Y[t][vm])
                thr[f"{t[:4]}-{vn}"] = round(th_, 3)
                pr[vm] = (p[vm] >= th_).astype(int)
            preds[t] = pr
        df = df.with_columns(pl.Series("sentiment_pred", preds["sentiment"]), pl.Series("sarcasm_pred", preds["sarcasm"]))
        sc = score(df)
        # per-source breakdown of sarcasm macro-F1 on reddit (where the task actually lives)
        red = df.filter(pl.col("source") == "reddit")
        sc_red = score(red)
        rows.append({"variant": v, "mode": mode, **{k: round(val, 4) for k, val in sc.items()},
                     "reddit_sarc-AU": round(sc_red["sarc-AU"], 4), "reddit_sarc-UK": round(sc_red["sarc-UK"], 4),
                     "thresholds": json.dumps(thr)})
res = pl.DataFrame(rows)
res.write_csv(TABLES / "tfidf_lr_scores.csv")
with pl.Config(tbl_rows=30, tbl_width_chars=250, fmt_str_lengths=120):
    print(res.drop("thresholds"))
    print(res.filter(pl.col("mode") != "thr=0.5").select("variant", "thresholds"))

# Save the best variant (by cross-fitted-threshold official score).
best = res.filter(pl.col("mode") == "thr=cross-fit/variety", pl.col("variant") != "AU->UK / UK->AU").sort(
    "score", descending=True).row(0, named=True)
bv = best["variant"]
out = pool.select("uid").with_columns(pl.Series("p_sent", oof[bv]["sentiment"]), pl.Series("p_sarc", oof[bv]["sarcasm"]))
path = save_preds("tfidf_lr", out, meta={
    "desc": "TF-IDF word1-2g(60k)+char_wb2-5g(150k), LR liblinear class_weight=balanced, C=" + json.dumps(C),
    "variant": bv, "score_thr0.5": res.filter(pl.col("variant") == bv, pl.col("mode") == "thr=0.5")["score"][0],
    "score_crossfit_thr": best["score"],
    "score_tuned_thr_optimistic": res.filter(pl.col("variant") == bv, pl.col("mode") == "thr=tuned/variety")["score"][0],
    "thresholds_crossfit_mean": json.loads(best["thresholds"])})
print(f"\nsaved {bv} -> {path}")

plot_scores(res)
