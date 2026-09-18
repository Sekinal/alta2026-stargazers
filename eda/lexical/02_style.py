"""Stylistic markers vs labels, and the "sarcasm inverts sentiment" surface-polarity test.

Run: uv run --with vaderSentiment python 02_style.py   (VADER lexicon only; no labelled data involved)

1. Rate of each marker (regex on RAW text) per label within each source x variety cell, Wilson 95% CIs,
   and log2 lift = rate(label=1) / rate(label=0) with a Katz log-ratio 95% CI.
2. Surface polarity (VADER lexicon + compound) of sarcastic-negative vs plain-negative vs positive texts.
Outputs: tables/style_rates.csv, tables/surface_polarity.csv, plots/style_*.png, plots/surface_*.png
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from lets_plot import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from _shared import (INK2, SARC_COL, SENT_COL, TABLES, VAR_COL, pool_with_norm, save, th, wilson)

pool = pool_with_norm()

W = lambda *ws: r"(?i)\b(?:" + "|".join(ws) + r")\b"  # noqa: E731
MARKERS: dict[str, str] = {
    # punctuation / typography
    "exclamation !": r"!",
    "multiple !!": r"!!",
    "question ?": r"\?",
    "ellipsis ...": r"\.\.\.|…",
    "ALL-CAPS word (4+)": r"\b[A-Z]{4,}\b",
    "short \"quoted\" span": r"[\"“][^\"”\n]{1,40}[\"”]",
    "reddit >quote": r"(?m)^\s*>",
    "emoji": r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]",
    "/s tag": r"(?:^|\s)/s\b",
    # interjections / sarcasm formulae
    "oh / ah": W("oh", "ohh+", "ah", "ahh+"),
    "wow": W("wow"),
    "yeah right / sure": r"(?i)\byeah,? right\b|\bsure(?:ly)?\b",
    "great": W("great"),
    "lovely / brilliant": W("lovely", "brilliant"),
    "thanks / cheers": W("thanks?", "thank you", "cheers", "ta"),
    "what a / how (adj)": r"(?i)\bwhat an? \w+|\bhow (?:dare|convenient|surprising|shocking)\b",
    "of course / obviously": r"(?i)\bof course\b|\bobviously\b|\bclearly\b",
    "shocked / surprise": r"(?i)\bshock(?:ed|ing|er)?\b|\bsurpris(?:e|ed|ing)\b|\bwho (?:would have|could have|knew)\b",
    # intensity
    "intensifier": W("so", "very", "really", "absolutely", "totally", "literally", "extremely", "incredibly",
                     "super", "such"),
    "profanity": r"(?i)\b(?:fuck\w*|shit\w*|crap\w*|bullshit|damn\w*|arse\w*|cunt\w*|wank\w*|piss\w*|twat\w*|"
                 r"bastard\w*|bloody|bollocks)\b",
    "laughter": r"(?i)\b(?:lol+|lmf?ao|rofl|ha(?:ha)+h?|hehe+)\b",
    # reddit-isms / entities
    "hashtag": r"(?:^|\s)#\w+",
    "URL": r"https?://|www\.",
    "@ / u/ mention": r"(?:^|\s)@\w+|(?:^|[\s(])/?u/\w+",
    "number": r"\d",
    "price ($ £ € quid)": r"[$£€]\s?\d|\d+\s?(?:quid|bucks|dollars|pounds|k\b)",
    # person / negation
    "1st person sg (I/me/my)": r"(?i)\b(?:i|me|my|mine|i'm|i've|i'd|i'll)\b",
    "1st person pl (we/us/our)": W("we", "us", "our", "ours", "we're", "we've"),
    "2nd person (you/your)": W("you", "your", "yours", "you're", "you've", "ya"),
    "negation": r"(?i)\b(?:not|no|never|nothing|nobody|none)\b|n't\b",
}
GROUPS = {  # for ordering the lift plot
    "punctuation": ["exclamation !", "multiple !!", "question ?", "ellipsis ...", "ALL-CAPS word (4+)",
                    "short \"quoted\" span", "reddit >quote", "emoji", "/s tag"],
    "interjection": ["oh / ah", "wow", "yeah right / sure", "great", "lovely / brilliant", "thanks / cheers",
                     "what a / how (adj)", "of course / obviously", "shocked / surprise"],
    "intensity": ["intensifier", "profanity", "laughter"],
    "entity": ["hashtag", "URL", "@ / u/ mention", "number", "price ($ £ € quid)"],
    "person": ["1st person sg (I/me/my)", "1st person pl (we/us/our)", "2nd person (you/your)", "negation"],
}
ORDER = [m for g in GROUPS.values() for m in g]
MGROUP = {m: g for g, ms in GROUPS.items() for m in ms}

feat = pool.select("uid", "source", "variety", "segment", "sentiment", "sarcasm",
                   *[pl.col("text").str.contains(rx).alias(m) for m, rx in MARKERS.items()])

# ---------------------------------------------------------------- rates + lift
rows = []
for task in ("sarcasm", "sentiment"):
    for (seg,), d in feat.group_by("segment", maintain_order=True):
        n1, n0 = int((d[task] == 1).sum()), int((d[task] == 0).sum())
        if min(n1, n0) < 30:  # google UK sarcasm (1 positive) is skipped
            continue
        for m in MARKERS:
            k1 = int(d.filter(pl.col(task) == 1)[m].sum())
            k0 = int(d.filter(pl.col(task) == 0)[m].sum())
            a, b = k1 + 0.5, k0 + 0.5  # Haldane correction for zero cells
            ll = np.log2((a / (n1 + 0.5)) / (b / (n0 + 0.5)))
            se = np.sqrt(1 / a - 1 / (n1 + 0.5) + 1 / b - 1 / (n0 + 0.5)) / np.log(2)
            rows.append({"task": task, "segment": seg, "source": seg.split(" · ")[0],
                         "variety": seg.split(" · ")[1], "marker": m, "group": MGROUP[m],
                         "n1": n1, "n0": n0, "k1": k1, "k0": k0, "rate1": k1 / n1, "rate0": k0 / n0,
                         "log2_lift": ll, "lo": ll - 1.96 * se, "hi": ll + 1.96 * se})
rates = pl.DataFrame(rows)
rates.write_csv(TABLES / "style_rates.csv")

with pl.Config(tbl_rows=200, tbl_width_chars=200):
    print(rates.filter(pl.col("source") == "reddit", pl.col("task") == "sarcasm")
          .select("variety", "marker", "k1", "rate1", "rate0", "log2_lift", "lo", "hi")
          .sort("marker", "variety"))


def lift_plot(task: str, source: str, title: str, sub: str):
    # markers seen in < 5 texts of the cell (e.g. emoji/URLs, which the dataset has stripped) are dropped
    d = rates.filter(pl.col("task") == task, pl.col("source") == source, pl.col("k1") + pl.col("k0") >= 5)
    kept = [m for m in ORDER if m in set(d["marker"].to_list())]
    # numeric y with a gap between marker groups (lets-plot dodge misaligns errorbars and points)
    ypos, y, prev = {}, 0.0, None
    for m in kept:
        y -= 1.0 + (0.8 if prev is not None and MGROUP[m] != prev else 0.0)
        ypos[m], prev = y, MGROUP[m]
    off = {"en-AU": 0.17, "en-UK": -0.17}
    d = d.with_columns(
        pl.col("lo").clip(-5, 5), pl.col("hi").clip(-5, 5), pl.col("log2_lift").clip(-5, 5),
        (pl.col("marker").replace_strict(ypos) + pl.col("variety").replace_strict(off)).alias("y"),
        (pl.col("rate1") * 100).round(1).alias("pct1"), (pl.col("rate0") * 100).round(1).alias("pct0"))
    heads = pl.DataFrame([{"group": g, "y": max(ypos[m] for m in kept if MGROUP[m] == g) + 0.75}
                          for g in GROUPS if any(MGROUP[m] == g for m in kept)])
    return (
        ggplot(d)
        + geom_vline(xintercept=0, color=INK2, size=0.5)
        + geom_errorbar(aes(y="y", xmin="lo", xmax="hi", color="variety"), width=0.0, size=0.9, alpha=0.55)
        + geom_point(aes(x="log2_lift", y="y", color="variety"), size=3.5,
                     tooltips=layer_tooltips().line("@marker · @variety").line("rate label=1: @pct1 % (@k1/@n1)")
                     .line("rate label=0: @pct0 % (@k0/@n0)").line("log2 lift @log2_lift")
                     .format("@log2_lift", ".2f"))
        + geom_text(aes(y="y", label="group"), data=heads, x=-5, hjust=0, size=6, fontface="bold", color=INK2)
        + scale_color_manual(values=VAR_COL, name="variety")
        + scale_y_continuous(breaks=list(ypos.values()), labels=list(ypos.keys()))
        + scale_x_continuous(limits=[-5, 5])
        + labs(title=title, subtitle=sub, x="log2( rate in label=1 / rate in label=0 )  ·  95% CI, clipped at ±5")
        + th("top") + theme(axis_title_y=element_blank(), panel_grid_major_y=element_blank())
    )


save(lift_plot("sarcasm", "reddit", "Stylistic markers: sarcastic vs non-sarcastic reddit",
               "log2 lift per variety; right of 0 = more frequent in sarcastic texts"),
     "style_lift_sarcasm_reddit", w=950, h=1100)
save(lift_plot("sentiment", "reddit", "Stylistic markers: positive vs negative reddit",
               "log2 lift per variety; right of 0 = more frequent in positive texts"),
     "style_lift_sentiment_reddit", w=950, h=1100)
save(lift_plot("sentiment", "google", "Stylistic markers: positive vs negative google reviews",
               "log2 lift per variety; right of 0 = more frequent in positive reviews"),
     "style_lift_sentiment_google", w=950, h=1100)

# Raw rates (with Wilson CIs) for the markers most associated with sarcasm, per reddit variety.
key = ["/s tag", "oh / ah", "yeah right / sure", "great", "thanks / cheers", "lovely / brilliant",
       "of course / obviously", "shocked / surprise", "what a / how (adj)", "short \"quoted\" span", "question ?",
       "exclamation !", "profanity", "1st person sg (I/me/my)", "2nd person (you/your)", "1st person pl (we/us/our)"]
long = pl.concat([
    rates.filter(pl.col("task") == "sarcasm", pl.col("source") == "reddit", pl.col("marker").is_in(key))
    .select("variety", "marker", pl.lit(lab).alias("label"), pl.col(k).alias("k"), pl.col(n).alias("n"))
    for lab, k, n in (("sarcastic", "k1", "n1"), ("not sarcastic", "k0", "n0"))
])
lo, hi = wilson(long["k"].to_numpy(), long["n"].to_numpy())
long = long.with_columns((pl.col("k") / pl.col("n") * 100).alias("pct"), pl.Series("lo", lo * 100),
                         pl.Series("hi", hi * 100),
                         pl.col("marker").replace_strict({m: i for i, m in enumerate(key)}).alias("ord"))
long = long.with_columns((-pl.col("ord") + pl.when(pl.col("label") == "sarcastic").then(-0.16).otherwise(0.16))
                         .alias("y"))
p = (
    ggplot(long)
    + geom_errorbar(aes(y="y", xmin="lo", xmax="hi", color="label"), width=0, size=0.9)
    + geom_point(aes(x="pct", y="y", color="label"), size=3.2,
                 tooltips=layer_tooltips().line("@marker · @label").line("@k / @n = @pct %").format("@pct", ".1f"))
    + scale_color_manual(values=SARC_COL)
    + scale_y_continuous(breaks=[-i for i in range(len(key))], labels=key)
    + facet_wrap("variety", ncol=2)
    + labs(title="How often each marker appears in reddit texts, by sarcasm label",
           subtitle="% of texts containing the marker · Wilson 95% CI", x="% of texts")
    + th("top") + theme(axis_title_y=element_blank(), legend_title=element_blank(),
                        panel_grid_major_y=element_blank())
)
save(p, "style_rates_sarcasm_reddit", w=1100, h=760)

# ---------------------------------------------------------------- surface polarity (VADER)
sia = SentimentIntensityAnalyzer()
lex = sia.lexicon
tok = re.compile(r"[a-z][a-z']*")


def surface(t: str) -> tuple[float, int, int]:
    words = tok.findall(t.lower())
    pos = sum(1 for w in words if lex.get(w, 0) >= 1.0)
    neg = sum(1 for w in words if lex.get(w, 0) <= -1.0)
    return sia.polarity_scores(t)["compound"], pos, neg


vals = [surface(t) for t in pool["text"].to_list()]
pool = pool.with_columns(pl.Series("compound", [v[0] for v in vals]), pl.Series("n_poslex", [v[1] for v in vals]),
                         pl.Series("n_neglex", [v[2] for v in vals]))
pool = pool.with_columns(
    pl.when((pl.col("sarcasm") == 1) & (pl.col("sentiment") == 0)).then(pl.lit("sarcastic · negative"))
    .when((pl.col("sarcasm") == 0) & (pl.col("sentiment") == 0)).then(pl.lit("plain · negative"))
    .when((pl.col("sarcasm") == 0) & (pl.col("sentiment") == 1)).then(pl.lit("plain · positive"))
    .otherwise(pl.lit("sarcastic · positive")).alias("grp"))
GRP_ORDER = ["plain · positive", "plain · negative", "sarcastic · negative", "sarcastic · positive"]
GRP_COL = {"plain · positive": SENT_COL["positive"], "plain · negative": SENT_COL["negative"],
           "sarcastic · negative": SARC_COL["sarcastic"], "sarcastic · positive": "#e87ba4"}

sp = (pool.group_by("segment", "variety", "source", "grp")
      .agg(pl.len().alias("n"),
           (pl.col("n_poslex") > 0).sum().alias("k_poslex"),
           (pl.col("compound") > 0.05).sum().alias("k_surfpos"),
           ((pl.col("n_poslex") > 0) & (pl.col("n_neglex") > 0)).sum().alias("k_mixed"),
           pl.col("compound").mean().alias("mean_compound"),
           pl.col("n_poslex").mean().alias("mean_poslex"))
      .sort("segment", "grp"))
sp.write_csv(TABLES / "surface_polarity.csv")
with pl.Config(tbl_rows=50, tbl_width_chars=200):
    print(sp.with_columns((pl.col(c) / pl.col("n")).round(3).alias(c.replace("k_", "p_"))
                          for c in ("k_poslex", "k_surfpos", "k_mixed")))

# Bar chart: % with VADER compound > 0.05, per group, reddit + google AU.
metric = []
for lab, col in (("≥1 positive-lexicon word", "k_poslex"), ("VADER compound > 0.05", "k_surfpos"),
                 ("both positive & negative words", "k_mixed")):
    d = sp.filter(pl.col("n") >= 20)
    lo, hi = wilson(d[col].to_numpy(), d["n"].to_numpy())
    metric.append(d.select("segment", "grp", "n", pl.lit(lab).alias("metric"),
                           (pl.col(col) / pl.col("n") * 100).alias("pct"))
                  .with_columns(pl.Series("lo", lo * 100), pl.Series("hi", hi * 100)))
metric = pl.concat(metric).filter(pl.col("segment") != "google · en-UK").with_columns(
    pl.col("grp").replace_strict({g: i for i, g in enumerate(GRP_ORDER)}).alias("gord"))
p = (
    ggplot(metric, aes(x=as_discrete("grp", order_by="gord", order=1), y="pct", fill="grp"))
    + geom_bar(stat="identity", width=0.72, color="#fcfcfb", size=0.6,
               tooltips=layer_tooltips().line("@segment · @grp").line("@metric: @pct % (n=@n)")
               .format("@pct", ".1f"))
    + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.18, color=INK2, size=0.6)
    + geom_text(aes(label="pct", y="hi"), label_format=".0f", nudge_y=6, size=7, color=INK2)
    + scale_fill_manual(values=GRP_COL)
    + facet_grid(x="segment", y="metric")
    + scale_y_continuous(limits=[0, 110], breaks=[0, 25, 50, 75, 100])
    + labs(title="Does sarcasm hide behind positive words?",
           subtitle="Surface polarity by label group (VADER lexicon) · Wilson 95% CI · groups with n ≥ 20",
           y="% of texts")
    + th("top") + theme(axis_text_x=element_blank(), axis_title_x=element_blank(), legend_title=element_blank(),
                        panel_grid_major_x=element_blank())
)
save(p, "surface_polarity_bars", w=1150, h=820)

# Density of VADER compound per group (reddit only).
dd = pool.filter(pl.col("source") == "reddit", pl.col("grp") != "sarcastic · positive")
p = (
    ggplot(dd, aes("compound", fill="grp", color="grp"))
    + geom_density(alpha=0.25, size=0.9, adjust=0.8)
    + scale_fill_manual(values=GRP_COL, limits=GRP_ORDER[:3]) + scale_color_manual(values=GRP_COL, limits=GRP_ORDER[:3])
    + facet_wrap("variety", ncol=1)
    + labs(title="Surface polarity of reddit texts by label group",
           subtitle="VADER compound (−1 very negative … +1 very positive); en-UK sarcastic negatives shift toward positive; en-AU ones look like plain negatives",
           x="VADER compound", y="density")
    + th("top") + theme(legend_title=element_blank())
)
save(p, "surface_compound_density", w=900, h=700)

# How much would a lexicon-only sentiment rule be fooled by sarcasm? error rate of compound>0.05 as "positive".
fool = (pool.filter(pl.col("sentiment") == 0)
        .group_by("segment", "sarcasm").agg(pl.len().alias("n"), (pl.col("compound") > 0.05).mean().alias("p_fooled"))
        .sort("segment", "sarcasm"))
print("\nNegative texts that VADER calls positive (compound > 0.05):\n", fool)
fool.write_csv(TABLES / "vader_fooled.csv")
