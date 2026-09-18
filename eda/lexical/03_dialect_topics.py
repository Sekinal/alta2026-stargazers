"""Dialect markers, lexical separability of en-AU vs en-UK, and reddit topics with label rates.

1. Hand-picked AU / UK dialect markers (spelling, slang, politics, places): % of texts per variety x source.
2. How separable are AU vs UK lexically, within each source? CV'd TF-IDF + LR (provided folds), ROC-AUC,
   plus Monroe log-odds AU vs UK to show WHAT separates them.
3. NMF topics on reddit (both varieties): topic share by variety and sarcasm / sentiment rate per topic.
Outputs: tables/dialect_*.csv, tables/topics_*.csv, plots/dialect_*.png, plots/topics_*.png
"""

from __future__ import annotations

import numpy as np
import polars as pl
from lets_plot import *
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _shared import (INK2, SARC_COL, TABLES, TOKEN_PATTERN, VAR_COL, pool_with_norm, save, th, wilson)

pool = pool_with_norm()

# ---------------------------------------------------------------- 1. dialect markers
DIALECT = {  # name: (expected variety, regex on lower-cased normalised text)
    "arvo / servo / bottle-o": ("AU", r"\b(?:arvo|servo|bottle-?o|brekkie|maccas|bogan\w*|avo)\b"),
    "reckon": ("AU", r"\breckon\w*\b"),
    "mate": ("both", r"\bmate(?:s)?\b"),
    "heaps": ("AU", r"\bheaps\b"),
    "keen / stoked": ("AU", r"\b(?:keen|stoked|ripper|fair dinkum|no worries|suss)\b"),
    "bloody": ("both", r"\bbloody\b"),
    "innit / proper / bruv": ("UK", r"\b(?:innit|bruv|blud|proper|mardy|gutted|knackered|chuffed|skint)\b"),
    "quid / pounds / £": ("UK", r"\b(?:quid|pounds?|pence|fiver|tenner)\b|£"),
    "$ / dollars": ("AU", r"\$|\bdollars?\b|\bbucks\b"),
    "cheers / ta": ("UK", r"\b(?:cheers|ta)\b"),
    "lovely / brilliant": ("UK", r"\b(?:lovely|brilliant)\b"),
    "bloke / wanker / arse": ("both", r"\b(?:blokes?|wank\w*|arse\w*)\b"),
    "Tories / Labour / Reform / NHS": ("UK", r"\b(?:tor(?:y|ies)|labour|reform|nhs|sunak|starmer|rishi|brexit|farage|"
                                               r"westminster|council tax|boris)\b"),
    "Labor / LNP / Dutton / Albo": ("AU", r"\b(?:labor|lnp|liberals?|dutton|albo\w*|greens|centrelink|medicare|"
                                          r"negative gearing|murdoch)\b"),
    "UK places": ("UK", r"\b(?:london|manchester|birmingham|scotland|wales|england|britain|british|uk|leeds|"
                         r"liverpool|glasgow|bristol)\b"),
    "AU places": ("AU", r"\b(?:sydney|melbourne|brisbane|perth|adelaide|canberra|victoria|nsw|qld|queensland|"
                         r"tasmania|australia\w*|aussie\w*)\b"),
    "-our spelling (colour…)": ("both", r"\b(?:colour|favour|flavour|behaviour|neighbour|honour|labour)\w*\b"),
    "-ise spelling": ("both", r"\b\w+(?:ise|ised|ising|isation)\b"),
    "-or / -ize (US)": ("US", r"\b(?:color|favor|flavor|behavior|neighbor|honor)\w*\b|\b\w+(?:ize|ized|izing|ization)\b"),
    "housing / rent / cost of living": ("both", r"\b(?:rent\w*|housing|mortgage\w*|landlords?|cost of living|"
                                                r"house prices?|interest rates?)\b"),
    "immigration / migrants": ("both", r"\b(?:immigra\w*|migra\w*|asylum|refugees?|boats?)\b"),
}
rows = []
for (src, var), d in pool.group_by("source", "variety"):
    for name, (exp, rx) in DIALECT.items():
        k = int(d["norm"].str.contains(rx).sum())
        rows.append({"source": src, "variety": var, "marker": name, "expected": exp, "k": k, "n": d.height})
dial = pl.DataFrame(rows)
lo, hi = wilson(dial["k"].to_numpy(), dial["n"].to_numpy())
dial = dial.with_columns((pl.col("k") / pl.col("n") * 100).alias("pct"), pl.Series("lo", lo * 100),
                         pl.Series("hi", hi * 100))
dial.write_csv(TABLES / "dialect_markers.csv")
wide = dial.pivot(on="variety", index=["source", "marker"], values="pct").sort("source", "marker")
with pl.Config(tbl_rows=60, tbl_width_chars=200):
    print(wide)

order = {m: i for i, m in enumerate(DIALECT)}
dd = dial.with_columns(pl.col("marker").replace_strict(order).alias("ord"),
                       (pl.col("marker") + pl.lit("  [") + pl.col("expected") + pl.lit("]")).alias("lab"))
seg = dd.pivot(on="variety", index=["source", "marker", "lab", "ord"], values="pct")
p = (
    ggplot(dd, aes(y=as_discrete("lab", order_by="ord", order=-1)))
    + geom_segment(aes(x="en-AU", xend="en-UK", yend="lab"), data=seg, color="#cfcdc6", size=2.2)
    + geom_point(aes(x="pct", color="variety"), size=3.8,
                 tooltips=layer_tooltips().line("@marker · @variety").line("@k / @n texts = @pct %")
                 .format("@pct", ".1f"))
    + scale_color_manual(values=VAR_COL)
    + scale_x_continuous(trans="sqrt", breaks=[0, 1, 2, 5, 10, 20])
    + facet_wrap("source", ncol=2)
    + labs(title="Dialect and topic markers: en-AU vs en-UK",
           subtitle="% of texts containing the marker (sqrt scale) · [expected variety] from the marker's origin",
           x="% of texts (sqrt scale)")
    + th("top") + theme(axis_title_y=element_blank())
)
save(p, "dialect_markers", w=1200, h=800)

# Marker usage vs labels: do AU politics markers carry the AU sarcasm signal?
lab_rows = []
for (var,), d in pool.filter(pl.col("source") == "reddit").group_by("variety"):
    for name, (_, rx) in DIALECT.items():
        has = d["norm"].str.contains(rx)
        for flag, sub in ((True, d.filter(has)), (False, d.filter(~has))):
            lab_rows.append({"variety": var, "marker": name, "has": flag, "n": sub.height,
                             "sarc_rate": sub["sarcasm"].mean() if sub.height else None,
                             "pos_rate": sub["sentiment"].mean() if sub.height else None})
ml = pl.DataFrame(lab_rows).filter(pl.col("has"), pl.col("n") >= 15).sort("variety", "sarc_rate", descending=[False, True])
ml.write_csv(TABLES / "dialect_marker_label_rates.csv")
with pl.Config(tbl_rows=60, tbl_width_chars=200):
    print("\nreddit label rates among texts containing each marker:\n", ml)

# ---------------------------------------------------------------- 2. separability AU vs UK
def cv_auc(sub: pl.DataFrame, analyzer: str, ngr: tuple[int, int]) -> float:
    y = (sub["variety"] == "en-AU").to_numpy().astype(int)
    oof = np.zeros(sub.height)
    for f in range(5):
        tr, te = (sub["fold"] != f).to_numpy(), (sub["fold"] == f).to_numpy()
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=2, sublinear_tf=True,
                              token_pattern=TOKEN_PATTERN if analyzer == "word" else None, lowercase=False)
        Xtr = vec.fit_transform(sub.filter(pl.Series(tr))["norm"].to_list())
        Xte = vec.transform(sub.filter(pl.Series(te))["norm"].to_list())
        oof[te] = LogisticRegression(C=4, max_iter=2000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, oof))


sep = []
for src in ("google", "reddit"):
    sub = pool.filter(pl.col("source") == src)
    for name, an, ngr in (("word 1-2g", "word", (1, 2)), ("char 2-5g", "char_wb", (2, 5))):
        sep.append({"source": src, "features": name, "roc_auc_AU_vs_UK": cv_auc(sub, an, ngr)})
sep = pl.DataFrame(sep)
print("\nLexical separability of variety within source (5-fold OOF ROC-AUC):\n", sep)
sep.write_csv(TABLES / "dialect_separability.csv")

# What separates them: Monroe log-odds AU vs UK within each source (cell-pooled prior, as in 01).
cv = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 1), binary=True, min_df=5, lowercase=False)
X = cv.fit_transform(pool["norm"].to_list()).tocsr()
vocab = np.array(cv.get_feature_names_out())
lo_rows = []
for src in ("google", "reddit"):
    m = (pool["source"] == src).to_numpy()
    au, uk = m & (pool["variety"] == "en-AU").to_numpy(), m & (pool["variety"] == "en-UK").to_numpy()
    yi, yj = np.asarray(X[au].sum(0)).ravel() + 0.0, np.asarray(X[uk].sum(0)).ravel() + 0.0
    a0 = 20_000.0
    alpha = np.maximum((yi + yj) / (yi + yj).sum() * a0, 1e-3)
    delta = np.log((yi + alpha) / (yi.sum() + a0 - yi - alpha)) - np.log((yj + alpha) / (yj.sum() + a0 - yj - alpha))
    z = delta / np.sqrt(1 / (yi + alpha) + 1 / (yj + alpha))
    df = pl.DataFrame({"term": vocab, "z": z, "df_AU": yi.astype(int), "df_UK": yj.astype(int)}).filter(
        pl.col("df_AU") + pl.col("df_UK") >= 5).with_columns(pl.lit(src).alias("source"))
    lo_rows.append(pl.concat([df.sort("z", descending=True).head(20), df.sort("z").head(20)]))
    df.sort("z", descending=True).write_csv(TABLES / f"dialect_logodds_{src}.csv")
top = pl.concat(lo_rows).with_columns(
    pl.when(pl.col("z") > 0).then(pl.lit("en-AU")).otherwise(pl.lit("en-UK")).alias("variety"))
for src in ("google", "reddit"):
    t = top.filter(pl.col("source") == src)
    print(f"\n{src} AU>UK:", ", ".join(t.filter(pl.col("z") > 0)["term"].to_list()))
    print(f"{src} UK>AU:", ", ".join(t.filter(pl.col("z") < 0)["term"].to_list()))


def varlolli(src: str):
    t = top.filter(pl.col("source") == src)
    return (
        ggplot(t)
        + geom_vline(xintercept=0, color=INK2, size=0.5)
        + geom_segment(aes(xend="z", y=as_discrete("term", order_by="z", order=1), yend="term", color="variety"),
                       x=0, size=1.1)
        + geom_point(aes(x="z", y="term", color="variety"), size=3.3,
                     tooltips=layer_tooltips().line("@term").line("z @z").line("docs AU @df_AU / UK @df_UK")
                     .format("@z", ".1f"))
        + scale_color_manual(values=VAR_COL)
        + labs(title=src, x="z (log-odds AU vs UK)")
        + th("bottom") + theme(axis_title_y=element_blank(), legend_title=element_blank(),
                               panel_grid_major_y=element_blank())
    )


auc_txt = " · ".join(f"{r['source']} {r['features']}: AUC {r['roc_auc_AU_vs_UK']:.3f}" for r in sep.iter_rows(named=True))
save(gggrid([varlolli("google"), varlolli("reddit")], ncol=2)
     + ggtitle("What separates en-AU from en-UK text (top 20 unigrams each way)", auc_txt),
     "dialect_logodds", w=1200, h=1000)

# ---------------------------------------------------------------- 3. reddit NMF topics
K = 12
red = pool.filter(pl.col("source") == "reddit")
tf = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, stop_words="english", min_df=5, max_df=0.3,
                     sublinear_tf=True)
Xr = tf.fit_transform(red["norm"].to_list())
nmf = NMF(n_components=K, init="nndsvda", random_state=0, max_iter=600)
W = nmf.fit_transform(Xr)
terms = np.array(tf.get_feature_names_out())
tops = [", ".join(terms[np.argsort(-c)[:7]]) for c in nmf.components_]
dom = W.argmax(1)
dom = np.where(W.max(1) < 1e-6, -1, dom)  # empty after stop-word removal
red = red.with_columns(pl.Series("topic", dom))
tt = (red.filter(pl.col("topic") >= 0).group_by("topic", "variety")
      .agg(pl.len().alias("n"), pl.col("sarcasm").sum().alias("k_sarc"), pl.col("sentiment").sum().alias("k_pos"))
      .with_columns(pl.col("topic").map_elements(lambda t: f"T{t:02d}: {tops[t]}", return_dtype=pl.String)
                    .alias("label")))
tot = red.group_by("variety").len().rename({"len": "N"})
tt = tt.join(tot, on="variety").with_columns((pl.col("n") / pl.col("N") * 100).alias("share"),
                                             (pl.col("k_sarc") / pl.col("n") * 100).alias("sarc_pct"),
                                             (pl.col("k_pos") / pl.col("n") * 100).alias("pos_pct"))
slo, shi = wilson(tt["k_sarc"].to_numpy(), tt["n"].to_numpy())
tt = tt.with_columns(pl.Series("s_lo", slo * 100), pl.Series("s_hi", shi * 100)).sort("topic", "variety")
tt.write_csv(TABLES / "topics_reddit.csv")
with pl.Config(tbl_rows=60, tbl_width_chars=250, fmt_str_lengths=90):
    print(tt.select("label", "variety", "n", "share", "sarc_pct", "pos_pct"))

ordr = tt.group_by("label").agg(pl.col("n").sum().alias("tot"))
tt = tt.join(ordr, on="label")
p1 = (
    ggplot(tt, aes(y=as_discrete("label", order_by="tot", order=1), x="share", fill="variety"))
    + geom_bar(stat="identity", position=position_dodge(0.8), width=0.75,
               tooltips=layer_tooltips().line("@label").line("@variety: @n texts (@share %)").format("@share", ".1f"))
    + scale_fill_manual(values=VAR_COL)
    + labs(title="Topic share", x="% of the variety's reddit texts")
    + th("bottom") + theme(axis_title_y=element_blank(), legend_title=element_blank())
)
p2 = (
    ggplot(tt, aes(y=as_discrete("label", order_by="tot", order=1), color="variety"))
    + geom_errorbar(aes(xmin="s_lo", xmax="s_hi"), width=0, size=0.9, alpha=0.6)
    + geom_point(aes(x="sarc_pct"), size=3.5,
                 tooltips=layer_tooltips().line("@label").line("@variety sarcasm @k_sarc/@n = @sarc_pct %")
                 .line("positive @pos_pct %").format("@sarc_pct", ".0f").format("@pos_pct", ".0f"))
    + scale_color_manual(values=VAR_COL)
    + labs(title="Sarcasm rate within topic (Wilson 95% CI)", x="% sarcastic")
    + th("bottom") + theme(axis_title_y=element_blank(), axis_text_y=element_blank(), legend_title=element_blank())
)
save(gggrid([p1, p2], ncol=2, widths=[1.6, 1])
     + ggtitle(f"Reddit topics (NMF, k={K}, TF-IDF, both varieties) and their sarcasm rates",
               "Each text assigned to its dominant topic; bars = topic share per variety"),
     "topics_reddit", w=1400, h=750)
