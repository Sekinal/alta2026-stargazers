"""Weighted log-odds ratios with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008).

Contrasts (all within a source x variety cell, so topic/domain differences don't leak in):
  * sarcasm 1 vs 0 within reddit (AU, UK)            [google AU has 73 sarcastic rows: also shown]
  * sentiment 1 vs 0 within reddit and within google (AU, UK)
Counts are DOCUMENT frequencies (binary per text) so a single long rant can't dominate a term.
Prior: the cell's own pooled document frequencies (both classes) rescaled to a fixed total ALPHA0.
(A pool-wide prior is NOT used: with unequal class sizes it gives spurious non-zero z to terms that
never occur in the cell, e.g. google-only tokens inside a reddit contrast.)
Outputs: tables/logodds_*.csv, plots/logodds_*.png|html, plots/transfer_*.png|html.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from lets_plot import *
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

from _shared import (INK2, SARC_COL, SENT_COL, TABLES, TOKEN_PATTERN, VAR_COL, pool_with_norm,
                     save, th)

ALPHA0 = 20_000.0
MIN_DF = 5
TOP = 15

pool = pool_with_norm()
cv = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), binary=True, min_df=MIN_DF,
                     lowercase=False)
X = cv.fit_transform(pool["norm"].to_list()).tocsr()
vocab = np.array(cv.get_feature_names_out())
print(f"vocab (1-2gram, min_df={MIN_DF}): {len(vocab):,}")


def log_odds(mask_i: np.ndarray, mask_j: np.ndarray) -> pl.DataFrame:
    """delta = log-odds(w | i) - log-odds(w | j), z = delta / sd  (Monroe et al. eq. 16/18)."""
    yi = np.asarray(X[mask_i].sum(0)).ravel().astype(float)
    yj = np.asarray(X[mask_j].sum(0)).ravel().astype(float)
    bg = yi + yj
    alpha = np.maximum(bg / bg.sum() * ALPHA0, 1e-3)  # floor only matters for absent terms (df_cell=0)
    ni, nj = yi.sum(), yj.sum()
    li = np.log((yi + alpha) / (ni + ALPHA0 - yi - alpha))
    lj = np.log((yj + alpha) / (nj + ALPHA0 - yj - alpha))
    delta = li - lj
    var = 1.0 / (yi + alpha) + 1.0 / (yj + alpha)
    return pl.DataFrame({
        "term": vocab, "delta": delta, "z": delta / np.sqrt(var),
        "df_pos": yi.astype(int), "df_neg": yj.astype(int),
        "rate_pos": yi / mask_i.sum(), "rate_neg": yj / mask_j.sum(),
    })


src = pool["source"].to_numpy()
var_ = pool["variety"].to_numpy()
CONTRASTS = [  # (name, task, source, variety)
    ("sarc_reddit_AU", "sarcasm", "reddit", "en-AU"),
    ("sarc_reddit_UK", "sarcasm", "reddit", "en-UK"),
    ("sarc_google_AU", "sarcasm", "google", "en-AU"),
    ("sent_reddit_AU", "sentiment", "reddit", "en-AU"),
    ("sent_reddit_UK", "sentiment", "reddit", "en-UK"),
    ("sent_google_AU", "sentiment", "google", "en-AU"),
    ("sent_google_UK", "sentiment", "google", "en-UK"),
]
res: dict[str, pl.DataFrame] = {}
for name, task, s, v in CONTRASTS:
    y = pool[task].to_numpy()
    cell = (src == s) & (var_ == v)
    df = log_odds(cell & (y == 1), cell & (y == 0))
    df = df.with_columns(pl.lit(name).alias("contrast"), (pl.col("df_pos") + pl.col("df_neg")).alias("df_cell"))
    df = df.filter(pl.col("df_cell") >= 3)
    res[name] = df
    df.sort("z", descending=True).write_csv(TABLES / f"logodds_{name}.csv")
    top = df.sort("z", descending=True)
    print(f"\n== {name}  (n1={int((cell & (y == 1)).sum())}, n0={int((cell & (y == 0)).sum())})")
    print("  +", ", ".join(f"{t}({z:.1f})" for t, z in top.head(20).select("term", "z").iter_rows()))
    print("  -", ", ".join(f"{t}({z:.1f})" for t, z in top.tail(20).reverse().select("term", "z").iter_rows()))


# ---------------------------------------------------------------- lollipop plots
def lolli(name: str, title: str, pos_lab: str, neg_lab: str, colours: dict, var: str):
    d = res[name]
    d = pl.concat([d.sort("z", descending=True).head(TOP), d.sort("z").head(TOP)]).with_columns(
        pl.when(pl.col("z") > 0).then(pl.lit(pos_lab)).otherwise(pl.lit(neg_lab)).alias("side"),
        (pl.col("term") + pl.lit("  ")).alias("lab"),
    )
    return (
        ggplot(d)
        + geom_vline(xintercept=0, color=INK2, size=0.5)
        + geom_segment(aes(xend="z", y=as_discrete("lab", order_by="z", order=1), yend="lab", color="side"),
                       x=0, size=1.1)
        + geom_point(aes(x="z", y="lab", color="side"), size=3.6,
                     tooltips=layer_tooltips().line("@term").line("z = @z").line("delta = @delta")
                     .line(f"docs {pos_lab}: @df_pos").line(f"docs {neg_lab}: @df_neg").format("@z", ".2f")
                     .format("@delta", ".2f"))
        + scale_color_manual(values={pos_lab: colours[pos_lab], neg_lab: colours[neg_lab]})
        + labs(title=title, x="z-score of weighted log-odds")
        + th("bottom") + theme(legend_title=element_blank(), axis_title_y=element_blank())
        + theme(panel_grid_major_y=element_blank(), plot_title=element_text(color=VAR_COL[var], size=13,
                                                                             face="bold"))
    )


def lolli_pair(prefix: str, task_title: str, pos_lab: str, neg_lab: str, colours: dict, src_: str):
    a = lolli(f"{prefix}_{src_}_AU", f"en-AU · {src_}", pos_lab, neg_lab, colours, "en-AU")
    b = lolli(f"{prefix}_{src_}_UK", f"en-UK · {src_}", pos_lab, neg_lab, colours, "en-UK")
    g = gggrid([a, b], ncol=2) + ggtitle(
        f"{task_title}: most discriminative uni/bigrams ({src_})",
        f"Monroe et al. weighted log-odds, informative Dirichlet prior; top {TOP} each way; document frequencies")
    return g


save(lolli_pair("sarc", "Sarcasm", "sarcastic", "not sarcastic", SARC_COL, "reddit"),
     "logodds_sarcasm_reddit", w=1200, h=860)
save(lolli_pair("sent", "Sentiment", "positive", "negative", SENT_COL, "reddit"),
     "logodds_sentiment_reddit", w=1200, h=860)
save(lolli_pair("sent", "Sentiment", "positive", "negative", SENT_COL, "google"),
     "logodds_sentiment_google", w=1200, h=860)
save(lolli("sarc_google_AU", "Sarcasm in en-AU google reviews (73 sarcastic vs 1003 not)", "sarcastic",
           "not sarcastic", SARC_COL, "en-AU"), "logodds_sarcasm_google_AU", w=620, h=860)


# ---------------------------------------------------------------- AU vs UK transfer
def transfer(a: str, b: str, min_df: int = 8):
    """Correlate per-term delta between AU and UK over terms reasonably frequent in BOTH cells."""
    j = (res[a].select("term", pl.col("delta").alias("d_au"), pl.col("z").alias("z_au"),
                       pl.col("df_cell").alias("n_au"))
         .join(res[b].select("term", pl.col("delta").alias("d_uk"), pl.col("z").alias("z_uk"),
                             pl.col("df_cell").alias("n_uk")), on="term")
         .filter((pl.col("n_au") >= min_df) & (pl.col("n_uk") >= min_df)))
    r = stats.pearsonr(j["d_au"], j["d_uk"])[0]
    rho = stats.spearmanr(j["d_au"], j["d_uk"])[0]
    # Of the top-50 AU cues (by |z|), how many point the same way in UK?
    top_au = j.sort(pl.col("z_au").abs(), descending=True).head(50)
    agree = float((np.sign(top_au["d_au"]) == np.sign(top_au["d_uk"])).mean())
    top_uk = j.sort(pl.col("z_uk").abs(), descending=True).head(50)
    agree_uk = float((np.sign(top_uk["d_au"]) == np.sign(top_uk["d_uk"])).mean())
    return j, r, rho, agree, agree_uk


def split_half(task: str, s_: str, v: str, terms: pl.Series, reps: int = 20) -> float:
    """Noise ceiling: mean Pearson r of delta between two random halves of the SAME cell, over `terms`.
    Halves have ~half the data, so this is a conservative (low) ceiling for the full-cell AU-vs-UK r."""
    rng = np.random.default_rng(0)
    y = pool[task].to_numpy()
    cell = np.flatnonzero((src == s_) & (var_ == v))
    keep = np.isin(vocab, terms.to_numpy())
    rs = []
    for _ in range(reps):
        h = np.zeros(len(pool), bool)
        h[rng.permutation(cell)[: len(cell) // 2]] = True
        c = np.isin(np.arange(len(pool)), cell)
        d1 = log_odds(h & (y == 1), h & (y == 0))["delta"].to_numpy()[keep]
        d2 = log_odds(c & ~h & (y == 1), c & ~h & (y == 0))["delta"].to_numpy()[keep]
        rs.append(stats.pearsonr(d1, d2)[0])
    return float(np.mean(rs))


summary = []
for task, a, b, title in [
    ("sarcasm", "sarc_reddit_AU", "sarc_reddit_UK", "Sarcasm cues, reddit"),
    ("sentiment", "sent_reddit_AU", "sent_reddit_UK", "Sentiment cues, reddit"),
    ("sentiment", "sent_google_AU", "sent_google_UK", "Sentiment cues, google"),
]:
    j, r, rho, agree, agree_uk = transfer(a, b)
    src_ = a.split("_")[1]
    sh_au = split_half(task, src_, "en-AU", j["term"])
    sh_uk = split_half(task, src_, "en-UK", j["term"])
    summary.append({"contrast": title, "n_terms": j.height, "pearson_delta": r, "spearman_delta": rho,
                    "sign_agree_top50_AU": agree, "sign_agree_top50_UK": agree_uk,
                    "splithalf_r_AU": sh_au, "splithalf_r_UK": sh_uk})
    j = j.with_columns((pl.col("z_au") ** 2 + pl.col("z_uk") ** 2).sqrt().alias("mag"))
    lab = pl.concat([
        j.sort("mag", descending=True).head(18),
        j.sort("z_au", descending=True).head(5), j.sort("z_uk", descending=True).head(5),
        j.sort("z_au").head(4), j.sort("z_uk").head(4),
    ]).unique("term")
    lim = float(max(j["z_au"].abs().max(), j["z_uk"].abs().max())) * 1.05
    p = (
        ggplot(j, aes("z_au", "z_uk"))
        + geom_hline(yintercept=0, color=INK2, size=0.4) + geom_vline(xintercept=0, color=INK2, size=0.4)
        + geom_abline(slope=1, intercept=0, color=INK2, linetype="dashed", size=0.4)
        + geom_point(color="#8a8984", alpha=0.35, size=1.6,
                     tooltips=layer_tooltips().line("@term").line("z AU @z_au").line("z UK @z_uk")
                     .format("@z_au", ".2f").format("@z_uk", ".2f"))
        + geom_point(data=lab, color="#4a3aa7" if task == "sarcasm" else "#1baf7a", size=2.6)
        + geom_text_repel(data=lab, mapping=aes(label="term"), size=6, color="#0b0b0b", seed=1,
                          max_overlaps=40)
        + coord_fixed(xlim=[-lim, lim], ylim=[-lim, lim])
        + labs(title=f"{title}: do AU cues transfer to UK?",
               subtitle=f"{j.height} terms with ≥8 docs in both varieties · Pearson r(delta) = {r:.2f}, "
                        f"Spearman ρ = {rho:.2f} · top-50 AU cues same sign in UK: {agree:.0%}\n"
                        f"noise ceiling (split-half r within variety): AU {sh_au:.2f}, UK {sh_uk:.2f}",
               x="z-score in en-AU (positive = sarcastic/positive)", y="z-score in en-UK")
        + th()
    )
    save(p, f"transfer_{a.split('_', 1)[0]}_{a.split('_')[1]}", w=900, h=900)

summary = pl.DataFrame(summary)
print("\n", summary)
summary.write_csv(TABLES / "transfer_summary.csv")
