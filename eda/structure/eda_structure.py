"""EDA angle: structure, labels, train/valid shift, duplicates, label noise and the metric.

Run from the project root:  uv run python eda/structure/eda_structure.py
Outputs: eda/structure/plots/*.png|html, eda/structure/tables/*.csv, eda/structure/examples/*.txt
and a console log (tee it to eda/structure/output.txt).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lets_plot import *  # noqa: F403
from scipy import stats
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alta.common import load_pool, score  # noqa: E402

LetsPlot.setup_html()
OUT = Path(__file__).resolve().parent
PLOTS, TABLES, EXAMPLES = OUT / "plots", OUT / "tables", OUT / "examples"
for d in (PLOTS, TABLES, EXAMPLES):
    d.mkdir(parents=True, exist_ok=True)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(160)
pl.Config.set_tbl_width_chars(220)
RNG = np.random.default_rng(2026)

# ---------------------------------------------------------------- styling
VAR_COL = {"en-AU": "#2a78d6", "en-UK": "#eb6834"}
JOINT_COL = {
    "neg · not sarcastic": "#4a3aa7",
    "neg · sarcastic": "#e87ba4",
    "pos · not sarcastic": "#1baf7a",
    "pos · sarcastic": "#eda100",
}
SENT_COL = {"negative": "#4a3aa7", "positive": "#1baf7a"}
SARC_COL = {"not sarcastic": "#b5b3ad", "sarcastic": "#e87ba4"}
INK, INK2 = "#0b0b0b", "#52514e"


def theme_base():
    return (
        theme_minimal2()
        + theme(
            plot_title=element_text(size=16, face="bold", color=INK),
            plot_subtitle=element_text(size=12, color=INK2),
            plot_caption=element_text(size=10, color=INK2),
            axis_title=element_text(size=12, color=INK2),
            axis_text=element_text(size=11, color=INK2),
            legend_title=element_text(size=11, color=INK2),
            legend_text=element_text(size=11, color=INK),
            strip_text=element_text(size=12, face="bold", color=INK),
            panel_grid_major=element_line(color="#e6e5e0", size=0.4),
            panel_grid_minor=element_blank(),
            legend_position="top",
            plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
        )
    )


def save(p, name: str, w: int = 900, h: int = 520):
    p = p + ggsize(w, h)
    ggsave(p, f"{name}.png", path=str(PLOTS), scale=2)
    ggsave(p, f"{name}.html", path=str(PLOTS))
    print(f"  [plot] {name}.png")


def section(title: str):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def mf1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


# ---------------------------------------------------------------- data + derived columns
pool = load_pool().with_columns(
    pl.col("source", "variety", "origin").cast(pl.String),
)
pool = pool.with_columns(
    (pl.col("variety") + " · " + pl.col("source")).alias("segment"),
    pl.col("text").str.len_chars().alias("n_chars"),
    pl.col("text").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n_words"),
    # sentences ~ runs of text ended by . ! ? or newline
    pl.col("text").str.count_matches(r"[.!?]+(\s|$)|\n+").clip(lower_bound=1).alias("n_sents"),
    pl.col("text").str.count_matches("\n").alias("n_newlines"),
    pl.when(pl.col("sentiment") == 1).then(pl.lit("pos")).otherwise(pl.lit("neg")).alias("_s"),
    pl.when(pl.col("sarcasm") == 1).then(pl.lit("sarcastic")).otherwise(pl.lit("not sarcastic")).alias("sarc_lab"),
    pl.when(pl.col("sentiment") == 1).then(pl.lit("positive")).otherwise(pl.lit("negative")).alias("sent_lab"),
).with_columns((pl.col("_s") + " · " + pl.col("sarc_lab")).alias("joint")).drop("_s")
SEG_ORDER = ["en-AU · google", "en-AU · reddit", "en-UK · google", "en-UK · reddit"]
print(f"pool rows: {pool.height}")

# ================================================================ 1. composition
section("1. COMPOSITION")
comp = (
    pool.group_by("variety", "source", "origin")
    .agg(
        pl.len().alias("n"),
        pl.col("sentiment").mean().round(3).alias("pos_rate"),
        pl.col("sarcasm").mean().round(3).alias("sarc_rate"),
        pl.col("sarcasm").sum().alias("n_sarc"),
    )
    .sort("variety", "source", "origin")
)
print(comp)
comp.write_csv(TABLES / "composition.csv")

by_var = pool.group_by("variety", "origin").agg(
    pl.len().alias("n"), pl.col("sentiment").mean().round(3).alias("pos_rate"),
    pl.col("sarcasm").sum().alias("n_sarc"), pl.col("sarcasm").mean().round(3).alias("sarc_rate"),
    (pl.col("source") == "reddit").mean().round(3).alias("reddit_share"),
).sort("variety", "origin")
print(by_var)

joint = (
    pool.group_by("segment", "joint").agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum().over("segment")).alias("share"))
    .sort("segment", "joint")
)
print(joint.pivot(on="joint", index="segment", values="n").sort("segment"))
joint.write_csv(TABLES / "joint_labels.csv")

# Plot 01: tile map of counts
tiles = pool.group_by("variety", "source", "origin").agg(pl.len().alias("n"), pl.col("sarcasm").sum().alias("ns"))
tiles = tiles.with_columns(
    (pl.col("n").cast(pl.String) + "\n(" + pl.col("ns").cast(pl.String) + " sarcastic)").alias("lab"),
    pl.col("origin").replace({"train": "train.csv", "valid": "valid.csv"}),
)
p = (
    ggplot(tiles, aes("source", "variety", fill="n"))
    + geom_tile(color="#fcfcfb", size=2, width=0.96, height=0.96)
    + geom_text(aes(label="lab"), size=7, color=INK, fontface="bold")
    + facet_wrap("origin")
    + scale_fill_gradient(low="#dbe8f8", high="#6ea4e6", name="rows")
    + labs(
        title="Where the 6,288 labelled rows live",
        subtitle="Rows per variety × source; the valid split mirrors train at ~1:7.3 in every cell",
        x="source", y="variety",
    )
    + theme_base() + theme(panel_grid_major=element_blank(), legend_position="none")
)
save(p, "01_composition_tiles", 900, 420)

# Plot 02: joint label distribution per segment (100% stacked bars)
jl = joint.with_columns(
    pl.when(pl.col("share") >= 0.04)
    .then(pl.col("n").cast(pl.String))
    .otherwise(pl.lit("")).alias("lab")
)
p = (
    ggplot(jl, aes(x="segment", y="share", fill="joint"))
    + geom_bar(stat="identity", position=position_stack(), color="#fcfcfb", size=0.8, width=0.7)
    + geom_text(aes(label="lab"), position=position_stack(vjust=0.5), color="white", size=6, fontface="bold")
    + scale_fill_manual(values=JOINT_COL, name="")
    + scale_y_continuous(format=".0%")
    + scale_x_discrete(limits=list(reversed(SEG_ORDER)))
    + coord_flip()
    + labs(
        title="Joint (sentiment, sarcasm) label mix by segment",
        subtitle="Sarcasm ≈ negative sentiment and ≈ reddit; en-UK google has 1 sarcastic row in 2,065, en-AU google 73 in 1,076",
        x="", y="share of rows (numbers = row counts)",
    )
    + theme_base()
)
save(p, "02_joint_labels", 1000, 440)

# ================================================================ 2. length & formatting
section("2. LENGTH & FORMATTING")
len_stats = (
    pool.group_by("segment")
    .agg(
        pl.len().alias("n"),
        pl.col("n_chars").median().alias("chars_med"),
        pl.col("n_chars").quantile(0.1).alias("chars_p10"),
        pl.col("n_chars").quantile(0.9).alias("chars_p90"),
        pl.col("n_chars").max().alias("chars_max"),
        pl.col("n_words").median().alias("words_med"),
        pl.col("n_sents").median().alias("sents_med"),
        (pl.col("n_newlines") > 0).mean().round(3).alias("has_newline"),
        (pl.col("n_words") > 400).mean().round(4).alias("gt400w"),
        (pl.col("n_chars") < 20).sum().alias("lt20c"),
    )
    .sort("segment")
)
print(len_stats)
len_stats.write_csv(TABLES / "length_stats.csv")

# formatting markers
fmt = pool.with_columns(
    pl.col("text").str.contains(r"https?://|www\.").alias("url"),
    pl.col("text").str.contains(r"&amp;|&gt;|&lt;|&#").alias("html_entity"),
    pl.col("text").str.contains(r"(^|\n)\s*>").alias("reddit_quote"),
    pl.col("text").str.contains(r"\*\*|\[.+\]\(.+\)|~~").alias("markdown"),
    pl.col("text").str.contains(r"\b[A-Z]{3,}\b").alias("caps_word"),
    pl.col("text").str.contains(r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]").alias("emoji"),
    pl.col("text").str.contains(r"\.\.\.|…").alias("ellipsis"),
    pl.col("text").str.contains(r"\?").alias("question"),
    pl.col("text").str.contains(r"!").alias("exclaim"),
    (pl.col("n_newlines") > 0).alias("newline"),
    # PTB-style tokenisation left in the google reviews: "could n't", "( finally)", "' avocado goodness '"
    pl.col("text").str.contains(r"\b(ca|do|does|wo|is|was|were|did|could|would|should|are|have|has|had|ai) n't\b").alias("spaced_nt"),
    pl.col("text").str.contains(r"\( \w|\w \)|\(\s\w").alias("spaced_paren"),
    pl.col("text").str.contains(r"(?i)\(translated by google\)|\(original\)").alias("google_translated"),
)
FMT_COLS = ["spaced_nt", "spaced_paren", "newline", "url", "html_entity", "reddit_quote", "markdown", "caps_word", "emoji", "ellipsis",
            "question", "exclaim", "google_translated"]
fmt_seg = fmt.group_by("segment").agg([pl.col(c).mean().round(3) for c in FMT_COLS]).sort("segment")
print(fmt_seg)
fmt_seg.write_csv(TABLES / "formatting_by_segment.csv")
# formatting marker -> label rates within reddit (does a marker shift labels?)
rows = []
for seg in SEG_ORDER:
    d = fmt.filter(pl.col("segment") == seg)
    for c in FMT_COLS:
        k = d[c].sum()
        if k < 10 or k > d.height - 10:
            continue
        on, off = d.filter(pl.col(c)), d.filter(~pl.col(c))
        rows.append(dict(segment=seg, marker=c, n_with=k,
                         sarc_with=on["sarcasm"].mean(), sarc_without=off["sarcasm"].mean(),
                         pos_with=on["sentiment"].mean(), pos_without=off["sentiment"].mean()))
fmt_lab = pl.DataFrame(rows).with_columns(pl.selectors.float().round(3))
print(fmt_lab)
fmt_lab.write_csv(TABLES / "formatting_label_rates.csv")

# Plot 03: char-length distributions (log scale) by segment
p = (
    ggplot(pool, aes(x="n_chars", color="variety", fill="variety"))
    + geom_density(alpha=0.15, size=1.0)
    + facet_wrap("source", ncol=1)
    + scale_x_log10(breaks=[3, 10, 30, 100, 300, 1000, 3000, 8000])
    + scale_color_manual(values=VAR_COL) + scale_fill_manual(values=VAR_COL)
    + labs(
        title="Text length by source and variety",
        subtitle="Google reviews are ~2× longer than reddit (median ~300 vs ~150 chars); en-AU reddit has a long tail to 8.5k chars",
        x="characters (log scale)", y="density", color="", fill="",
    )
    + theme_base()
)
save(p, "03_length_density", 950, 560)

# Plot 04: label rate by length quintile within segment
lq = pool.with_columns(
    pl.col("n_chars").rank("ordinal").over("segment").alias("_r"),
    pl.len().over("segment").alias("_n"),
).with_columns(((pl.col("_r") - 1) * 5 // pl.col("_n") + 1).alias("len_q"))
lrate = (
    lq.group_by("segment", "variety", "source", "len_q")
    .agg(
        pl.col("sentiment").mean().alias("positive sentiment"),
        pl.col("sarcasm").mean().alias("sarcasm"),
        pl.col("n_chars").median().alias("med_chars"),
        pl.len().alias("n"),
    )
    .unpivot(index=["segment", "variety", "source", "len_q", "med_chars", "n"], variable_name="label", value_name="rate")
    .sort("segment", "label", "len_q")
)
p = (
    ggplot(lrate, aes("len_q", "rate", color="variety"))
    + geom_line(size=1.2) + geom_point(size=3.5, shape=21, fill="#fcfcfb", stroke=1.5)
    + facet_grid(x="source", y="label", scales="free_y")
    + scale_color_manual(values=VAR_COL)
    + scale_y_continuous(format=".0%")
    + scale_x_continuous(breaks=[1, 2, 3, 4, 5], labels=["Q1\nshortest", "Q2", "Q3", "Q4", "Q5\nlongest"])
    + labs(
        title="Does length predict the label within a segment?",
        subtitle="Label rate per within-segment length quintile. Longer texts are more often negative everywhere; longer en-UK reddit posts are less often sarcastic",
        x="length quintile (chars, within segment)", y="label rate", color="",
    )
    + theme_base()
)
save(p, "04_label_rate_by_length", 1050, 620)

# AUC of length for each label within segment
auc_rows = []
for seg in SEG_ORDER:
    d = pool.filter(pl.col("segment") == seg)
    for lab in ("sentiment", "sarcasm"):
        y = d[lab].to_numpy()
        if 5 <= y.sum() <= len(y) - 5:
            x = np.log(d["n_chars"].to_numpy())
            a = roc_auc_score(y, x)
            # Mann-Whitney p-value
            pv = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
            auc_rows.append(dict(segment=seg, label=lab, n_pos=int(y.sum()), auc_len=round(a, 3), p=float(f"{pv:.2g}")))
auc_len = pl.DataFrame(auc_rows)
print("AUC of log(char length) for label within segment (0.5 = no signal; <0.5 = shorter => positive class)")
print(auc_len)
auc_len.write_csv(TABLES / "length_auc.csv")

# Plot 05: formatting markers by segment
n_nonascii = pool["text"].str.count_matches(r"[^\x00-\x7F]").sum()
print(f"non-ASCII characters in the whole pool: {n_nonascii} (texts are ASCII-only: emoji/accents stripped upstream)")
fl = fmt_seg.unpivot(index="segment", variable_name="marker", value_name="share")
present = fl.group_by("marker").agg(pl.col("share").max()).filter(pl.col("share") > 0)["marker"].to_list()
fl = fl.filter(pl.col("marker").is_in(present)).with_columns(
    pl.col("segment").str.split(" · ").list.get(0).alias("variety"),
    pl.col("segment").str.split(" · ").list.get(1).alias("source"),
)
p = (
    ggplot(fl, aes(x="marker", y="share", fill="variety"))
    + geom_bar(stat="identity", position=position_dodge(width=0.8), width=0.75, color="#fcfcfb", size=0.5)
    + facet_wrap("source", ncol=1)
    + scale_fill_manual(values=VAR_COL)
    + scale_y_continuous(format=".0%")
    + scale_x_discrete(limits=[c for c in FMT_COLS if c in present])
    + labs(
        title="Formatting markers by segment",
        subtitle="Google text carries tokeniser artefacts (spaced n't in ~30%); newlines and '>' quotes only in reddit; URLs, emoji and non-ASCII are absent",
        x="", y="share of rows", fill="",
    )
    + theme_base() + theme(axis_text_x=element_text(angle=30, hjust=1))
)
save(p, "05_formatting_markers", 1050, 620)

# ================================================================ 3. train vs valid
section("3. TRAIN vs VALID")
tv_rows = []
for seg in SEG_ORDER + ["ALL"]:
    d = pool if seg == "ALL" else pool.filter(pl.col("segment") == seg)
    tr, va = d.filter(pl.col("origin") == "train"), d.filter(pl.col("origin") == "valid")
    ks = stats.ks_2samp(tr["n_chars"].to_numpy(), va["n_chars"].to_numpy())
    tv_rows.append(dict(
        segment=seg, n_train=tr.height, n_valid=va.height, ratio=round(tr.height / va.height, 2),
        pos_train=round(tr["sentiment"].mean(), 4), pos_valid=round(va["sentiment"].mean(), 4),
        sarc_train=round(tr["sarcasm"].mean(), 4), sarc_valid=round(va["sarcasm"].mean(), 4),
        medlen_train=tr["n_chars"].median(), medlen_valid=va["n_chars"].median(), ks_p_len=round(ks.pvalue, 3),
    ))
tv = pl.DataFrame(tv_rows)
print(tv)
tv.write_csv(TABLES / "train_vs_valid.csv")
# joint-cell counts train vs valid: is valid an exact stratified draw?
cell = (
    pool.group_by("variety", "source", "sentiment", "sarcasm", "origin").len()
    .pivot(on="origin", index=["variety", "source", "sentiment", "sarcasm"], values="len").fill_null(0)
    .with_columns(
        (pl.col("train") + pl.col("valid")).alias("total"),
    )
    .with_columns(
        (pl.col("total") * 757 / 6288).round(1).alias("valid_expected_if_proportional"),
    )
    .sort("variety", "source", "sentiment", "sarcasm")
)
print("Joint cells: valid counts vs proportional expectation (stratified sampling => near-exact match)")
print(cell)
cell.write_csv(TABLES / "train_valid_cells.csv")
dev = (cell["valid"] - cell["valid_expected_if_proportional"]).abs()
print(f"max |valid - expected| over {cell.height} cells = {dev.max():.2f}; "
      f"under simple random sampling we'd expect deviations of ~sqrt(n*p) e.g. {np.sqrt(1545 * 757 / 6288):.1f} in the biggest cell")

# Plot 06: train vs valid rates dot plot
tvp = pl.concat([
    tv.filter(pl.col("segment") != "ALL").select("segment", pl.lit("positive sentiment rate").alias("metric"),
                                                 pl.col("pos_train").alias("train"), pl.col("pos_valid").alias("valid")),
    tv.filter(pl.col("segment") != "ALL").select("segment", pl.lit("sarcasm rate").alias("metric"),
                                                 pl.col("sarc_train").alias("train"), pl.col("sarc_valid").alias("valid")),
]).unpivot(index=["segment", "metric"], variable_name="split", value_name="rate")
p = (
    ggplot(tvp, aes(x="rate", y="segment", color="split", shape="split"))
    + geom_point(size=6, stroke=1.6, alpha=0.9)
    + facet_wrap("metric", ncol=2, scales="free_x")
    + scale_color_manual(values={"train": "#4a3aa7", "valid": "#eda100"})
    + scale_shape_manual(values={"train": 1, "valid": 4})
    + scale_x_continuous(format=".0%")
    + scale_y_discrete(limits=list(reversed(SEG_ORDER)))
    + labs(
        title="valid.csv is a stratified sample of the same distribution as train.csv",
        subtitle="Every joint label cell matches proportional allocation within ±2 rows (the only visible gap: en-AU google sarcasm, 8 expected vs 7); length KS p > 0.07",
        x="rate", y="", color="", shape="",
    )
    + theme_base()
)
save(p, "06_train_vs_valid", 1000, 420)

# ================================================================ 4. duplicates, near-duplicates, short texts
section("4. DUPLICATES / NEAR-DUPLICATES / SHORT TEXTS")


def norm(s: pl.Expr) -> pl.Expr:
    return s.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


pool = pool.with_columns(norm(pl.col("text")).alias("norm"))
for key in ("text", "norm"):
    g = (
        pool.group_by(key)
        .agg(
            pl.len().alias("n"),
            pl.col("variety").unique().alias("varieties"),
            pl.col("source").unique().alias("sources"),
            pl.col("origin").unique().alias("origins"),
            pl.col("sentiment").n_unique().alias("sent_vals"),
            pl.col("sarcasm").n_unique().alias("sarc_vals"),
            pl.col("sentiment").alias("sents"),
            pl.col("sarcasm").alias("sarcs"),
        )
        .filter(pl.col("n") > 1)
        .sort("n", descending=True)
    )
    print(f"\n[{key}] duplicate groups: {g.height}, rows involved: {g['n'].sum()}, "
          f"label-conflict groups: sent {(g['sent_vals'] > 1).sum()}, sarc {(g['sarc_vals'] > 1).sum()}; "
          f"cross-variety groups: {(g['varieties'].list.len() > 1).sum()}; "
          f"cross train/valid groups: {(g['origins'].list.len() > 1).sum()}")
    print(g.select(pl.col(key).str.slice(0, 70), "n", "varieties", "sources", "origins", "sents", "sarcs"))
    g.with_columns(pl.col(c).cast(pl.List(pl.String)).list.join("|") for c in ("varieties", "sources", "origins", "sents", "sarcs")) \
        .write_csv(TABLES / f"dups_{key}.csv")

# near duplicates with char 3-5gram tf-idf cosine
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
X = vec.fit_transform(pool["norm"].to_list())
pairs = []
for s in range(0, X.shape[0], 800):
    sim = (X[s:s + 800] @ X.T).toarray()
    ii, jj = np.nonzero(sim >= 0.8)
    for i, j in zip(ii + s, jj):
        if i < j:
            pairs.append((i, j, float(sim[i - s, j])))
    del sim
nd = pl.DataFrame(pairs, schema=["i", "j", "cos"], orient="row")
cols = ["uid", "variety", "source", "origin", "sentiment", "sarcasm", "text", "norm"]
pi = pool.select(cols).with_row_index("i")
nd = (
    nd.join(pi, on="i").join(pi.rename({c: f"{c}_b" for c in cols}).rename({"i": "j"}), on="j")
    .filter(pl.col("norm") != pl.col("norm_b"))  # exclude exact normalised dups (reported above)
    .sort("cos", descending=True)
)
print(f"\nnear-duplicate pairs (char-tfidf cosine >= 0.8, not identical after normalisation): {nd.height}; "
      f"sent-conflict {(nd['sentiment'] != nd['sentiment_b']).sum()}, sarc-conflict {(nd['sarcasm'] != nd['sarcasm_b']).sum()}, "
      f"cross-variety {(nd['variety'] != nd['variety_b']).sum()}, cross-origin {(nd['origin'] != nd['origin_b']).sum()}")
print(nd.select("cos", "variety", "variety_b", "origin", "origin_b", "sentiment", "sentiment_b", "sarcasm", "sarcasm_b",
                pl.col("text").str.slice(0, 60), pl.col("text_b").str.slice(0, 60)).head(25))
nd.drop("norm", "norm_b").write_csv(TABLES / "near_duplicates.csv")
del X

# short texts
pool = pool.with_columns(
    pl.when(pl.col("n_chars") < 20).then(pl.lit("< 20 chars"))
    .when(pl.col("n_chars") < 50).then(pl.lit("20-49 chars"))
    .otherwise(pl.lit("≥ 50 chars")).alias("len_band")
)
short = pool.group_by("segment", "len_band").agg(
    pl.len().alias("n"), pl.col("sentiment").mean().round(3).alias("pos_rate"), pl.col("sarcasm").mean().round(3).alias("sarc_rate")
).sort("segment", "len_band")
print(short)
short.write_csv(TABLES / "short_texts.csv")
with open(EXAMPLES / "short_texts.txt", "w") as f:
    for r in pool.filter(pl.col("n_chars") < 20).sort("segment", "joint").iter_rows(named=True):
        f.write(f"{r['uid']}\t{r['segment']}\tsent={r['sentiment']} sarc={r['sarcasm']}\t{r['text']!r}\n")

sb = pool.group_by("len_band", "source", "joint").len().with_columns(
    (pl.col("len") / pl.col("len").sum().over("len_band", "source")).alias("share"),
    pl.len().over("len_band", "source").alias("_k"),
)
nlab = pool.group_by("len_band", "source").len().with_columns(("n=" + pl.col("len").cast(pl.String)).alias("lab"))
p = (
    ggplot(sb, aes(x="len_band", y="share", fill="joint"))
    + geom_bar(stat="identity", position=position_stack(), width=0.7, color="#fcfcfb", size=0.8)
    + geom_text(aes(x="len_band", label="lab"), data=nlab, y=1.05, inherit_aes=False, size=5, color=INK2)
    + facet_wrap("source")
    + scale_fill_manual(values=JOINT_COL, name="")
    + scale_x_discrete(limits=["< 20 chars", "20-49 chars", "≥ 50 chars"])
    + scale_y_continuous(format=".0%", breaks=[0, 0.25, 0.5, 0.75, 1])
    + labs(
        title="Very short texts carry labels too — and they are mostly guessable",
        subtitle="Label mix by length band; short reddit replies are more often positive ('Thank you!', 'Good point!') and less often sarcastic",
        x="", y="share of rows",
    )
    + theme_base()
)
save(p, "07_short_texts", 1000, 520)

# ================================================================ 5. label-noise hints
section("5. LABEL-NOISE HINTS (examples written to examples/)")


def dump(df: pl.DataFrame, name: str, header: str):
    with open(EXAMPLES / name, "w") as f:
        f.write(header + f"\n{df.height} rows\n\n")
        for r in df.iter_rows(named=True):
            f.write(f"--- {r['uid']} | {r['segment']} | {r['origin']} | sent={r['sentiment']} sarc={r['sarcasm']}\n{r['text']}\n\n")
    print(f"  {name}: {df.height} rows")


dump(pool.filter((pl.col("sarcasm") == 1) & (pl.col("sentiment") == 1)), "sarc1_sent1.txt", "Sarcastic AND positive sentiment")
dump(pool.filter((pl.col("sarcasm") == 1) & (pl.col("source") == "google")), "google_sarcastic.txt", "Google reviews labelled sarcastic")
dump(pool.filter((pl.col("source") == "google") & (pl.col("sentiment") == 0) & (pl.col("n_chars") < 200))
     .sample(40, seed=1), "google_negative_short_sample.txt", "Random short google negatives (check for mislabelled positives)")
dump(pool.filter((pl.col("source") == "reddit") & (pl.col("variety") == "en-UK") & (pl.col("sarcasm") == 1))
     .sample(40, seed=1), "uk_reddit_sarcastic_sample.txt", "Random en-UK reddit sarcastic")
dump(pool.filter((pl.col("source") == "reddit") & (pl.col("variety") == "en-AU") & (pl.col("sarcasm") == 1))
     .sample(40, seed=1), "au_reddit_sarcastic_sample.txt", "Random en-AU reddit sarcastic")
# google star-like hints: strongly positive words but label 0, or strongly negative words and label 1
POSW = r"(?i)\b(amazing|excellent|fantastic|highly recommend|best|awesome|delicious|perfect|wonderful|lovely)\b"
NEGW = r"(?i)\b(worst|terrible|awful|disgusting|rude|never again|avoid|horrible|waste)\b"
sus = pool.filter(pl.col("source") == "google").with_columns(
    pl.col("text").str.count_matches(POSW).alias("npos"), pl.col("text").str.count_matches(NEGW).alias("nneg")
)
dump(sus.filter((pl.col("sentiment") == 0) & (pl.col("npos") >= 2) & (pl.col("nneg") == 0)), "google_neg_with_praise.txt",
     "Google sentiment=0 with >=2 praise words and no negative words")
dump(sus.filter((pl.col("sentiment") == 1) & (pl.col("nneg") >= 2) & (pl.col("npos") == 0)), "google_pos_with_complaints.txt",
     "Google sentiment=1 with >=2 complaint words and no praise words")

# ================================================================ 6. the metric
section("6. THE METRIC")
valid = pool.filter(pl.col("origin") == "valid")
print("valid per variety:", valid.group_by("variety").agg(pl.len(), pl.col("sarcasm").sum(), pl.col("sentiment").sum()).sort("variety"))

# 6a. trivial structural predictors
g, r, au, uk = (pl.col("source") == "google"), (pl.col("source") == "reddit"), (pl.col("variety") == "en-AU"), (pl.col("variety") == "en-UK")
RULES_SENT = {
    "all 0": pl.lit(0), "all 1": pl.lit(1),
    "google→1, reddit→0": g.cast(pl.Int64),
}
RULES_SARC = {
    "all 0": pl.lit(0), "all 1": pl.lit(1),
    "reddit→1": r.cast(pl.Int64),
    "AU-reddit→1 only": (r & au).cast(pl.Int64),
    "reddit→1, but predicted neg only": r.cast(pl.Int64),  # placeholder, overwritten below
}
RULES_SARC.pop("reddit→1, but predicted neg only")


def eval_rules(df: pl.DataFrame, tag: str) -> pl.DataFrame:
    rows = []
    for sn, se in RULES_SENT.items():
        for cn, ce in RULES_SARC.items():
            d = df.with_columns(se.alias("sentiment_pred"), ce.alias("sarcasm_pred"))
            s = score(d)
            rows.append(dict(eval=tag, sent_rule=sn, sarc_rule=cn, **{k: round(v, 4) for k, v in s.items()}))
    return pl.DataFrame(rows).sort("score", descending=True)


RULES_SARC["reddit & gold-neg→1 (oracle)"] = (r & (pl.col("sentiment") == 0)).cast(pl.Int64)
rules_pool = eval_rules(pool, "pool")
rules_valid = eval_rules(valid, "valid")
print(rules_pool)
print(rules_valid.head(5))
pl.concat([rules_pool, rules_valid]).write_csv(TABLES / "trivial_rules.csv")

# Oracle-prior: best possible predictor that uses ONLY segment (any per-segment constant). Also random-at-prior expectation.
best = {}
for task in ("sentiment", "sarcasm"):
    for v in ("en-AU", "en-UK"):
        d = pool.filter(pl.col("variety") == v)
        bestv, bestcfg = -1, None
        for cg in (0, 1):
            for cr in (0, 1):
                pred = np.where(d["source"].to_numpy() == "google", cg, cr)
                m = mf1(d[task].to_numpy(), pred)
                if m > bestv:
                    bestv, bestcfg = m, (cg, cr)
        best[(task, v)] = (round(bestv, 4), bestcfg)
print("best per-segment constant (google, reddit) per task×variety:", best)
# random predictions at the per-segment prior (expected, by simulation)
sim = []
for _ in range(200):
    d = pool.with_columns(
        pl.Series("u1", RNG.random(pool.height)), pl.Series("u2", RNG.random(pool.height))
    ).with_columns(
        (pl.col("u1") < pl.col("sentiment").mean().over("segment")).cast(pl.Int64).alias("sentiment_pred"),
        (pl.col("u2") < pl.col("sarcasm").mean().over("segment")).cast(pl.Int64).alias("sarcasm_pred"),
    )
    sim.append(score(d))
simdf = pl.DataFrame(sim)
print("random-at-segment-prior (mean over 200 sims):", simdf.mean().with_columns(pl.all().round(4)))

# 6b. a reference classifier (tf-idf + LR, OOF on the fixed folds) to anchor noise/threshold analysis
print("\nfitting reference tf-idf + LR classifier (OOF on 5 folds)...")
seg_tok = ("__" + pool["variety"] + "_" + pool["source"] + "__ ").to_list()
texts = [a + t for a, t in zip(seg_tok, pool["text"].to_list())]
oof = {"sentiment": np.zeros(pool.height), "sarcasm": np.zeros(pool.height)}
folds = pool["fold"].to_numpy()
for k in range(5):
    tr, te = folds != k, folds == k
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=150_000)
    ttr = [t for t, m in zip(texts, tr) if m]
    tte = [t for t, m in zip(texts, te) if m]
    Xtr = hstack([vw.fit_transform(ttr), vc.fit_transform(ttr)]).tocsr()
    Xte = hstack([vw.transform(tte), vc.transform(tte)]).tocsr()
    for lab in oof:
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
        clf.fit(Xtr, pool[lab].to_numpy()[tr])
        oof[lab][te] = clf.predict_proba(Xte)[:, 1]
pool = pool.with_columns(pl.Series("p_sent", oof["sentiment"]), pl.Series("p_sarc", oof["sarcasm"]))
ref = pool.with_columns(
    (pl.col("p_sent") >= 0.5).cast(pl.Int64).alias("sentiment_pred"),
    (pl.col("p_sarc") >= 0.5).cast(pl.Int64).alias("sarcasm_pred"),
)
s_ref = score(ref)
print("reference LR @0.5, pool OOF:", {k: round(v, 4) for k, v in s_ref.items()})
s_ref_v = score(ref.filter(pl.col("origin") == "valid"))
print("reference LR @0.5, valid rows:", {k: round(v, 4) for k, v in s_ref_v.items()})
for v in ("en-AU", "en-UK"):
    d = pool.filter(pl.col("variety") == v)
    print(f"  AUC {v}: sent {roc_auc_score(d['sentiment'], d['p_sent']):.3f}, sarc {roc_auc_score(d['sarcasm'], d['p_sarc']):.3f}; "
          f"sarc within reddit {roc_auc_score(d.filter(pl.col('source') == 'reddit')['sarcasm'], d.filter(pl.col('source') == 'reddit')['p_sarc']):.3f}")
# hard rule on top: sarcasm = 0 for en-UK google
ref_rule = ref.with_columns(
    pl.when((pl.col("variety") == "en-UK") & (pl.col("source") == "google")).then(0).otherwise(pl.col("sarcasm_pred")).alias("sarcasm_pred")
)
print("reference LR + rule(UK google → sarc 0):", {k: round(v, 4) for k, v in score(ref_rule).items()})
ref_negrule = ref.with_columns(((pl.col("source") == "reddit") & (pl.col("sentiment_pred") == 0)).cast(pl.Int64).alias("sarcasm_pred"))
print("sarcasm := reddit & LR-predicted negative sentiment:", {k: round(v, 4) for k, v in score(ref_negrule).items()})
print("reference LR false-positive sarcasm on UK google:", ref.filter((pl.col("segment") == "en-UK · google"))["sarcasm_pred"].sum())


# per-variety tuned thresholds (optimistic, tuned on the same OOF)
def best_thr(y, p, grid=np.linspace(0.05, 0.95, 91)):
    f = [mf1(y, (p >= t).astype(int)) for t in grid]
    i = int(np.argmax(f))
    return float(grid[i]), f[i]


thr = {}
for v in ("en-AU", "en-UK"):
    d = pool.filter(pl.col("variety") == v)
    thr[(v, "sent")] = best_thr(d["sentiment"].to_numpy(), d["p_sent"].to_numpy())
    thr[(v, "sarc")] = best_thr(d["sarcasm"].to_numpy(), d["p_sarc"].to_numpy())
print("per-variety tuned thresholds (thr, macro-F1):", {k: (round(a, 2), round(b, 4)) for k, (a, b) in thr.items()})

# Plot 09: threshold curves on OOF reference classifier
grid = np.linspace(0.02, 0.98, 97)
tc = []
for v in ("en-AU", "en-UK"):
    d = pool.filter(pl.col("variety") == v)
    for task, lab, col in (("sentiment", "sentiment", "p_sent"), ("sarcasm", "sarcasm", "p_sarc")):
        y, pr = d[lab].to_numpy(), d[col].to_numpy()
        for t in grid:
            tc.append(dict(variety=v, task=task, thr=float(t), macro_f1=mf1(y, (pr >= t).astype(int)),
                           pred_rate=float((pr >= t).mean())))
tc = pl.DataFrame(tc)
tcbest = tc.group_by("variety", "task").agg(pl.all().sort_by("macro_f1").last())
p = (
    ggplot(tc, aes("thr", "macro_f1", color="variety"))
    + geom_line(size=1.3)
    + geom_point(data=tcbest, size=5, shape=21, fill="#fcfcfb", stroke=2)
    + geom_vline(xintercept=0.5, linetype="dashed", color="#9a9994", size=0.6)
    + facet_wrap("task", ncol=2)
    + scale_color_manual(values=VAR_COL)
    + labs(
        title="Macro-F1 vs decision threshold (reference tf-idf + LR, out-of-fold)",
        subtitle="Circles = best threshold per variety. Sarc-UK peaks lowest and at a different threshold than sarc-AU → tune thresholds per variety",
        x="threshold on P(positive class)", y="macro-F1", color="",
    )
    + theme_base()
)
save(p, "09_threshold_curves_oof", 1050, 500)

# 6c. bootstrap noise: resample test sets like valid (stratified: exact valid cell counts) from the pooled OOF preds
print("\nbootstrap: test sets with valid's exact joint-cell counts, drawn from pool OOF predictions")
cells = valid.group_by("variety", "source", "sentiment", "sarcasm").len()
idx_by_cell = {}
pool_i = pool.with_row_index("ri")
for c in cells.iter_rows(named=True):
    ids = pool_i.filter(
        (pl.col("variety") == c["variety"]) & (pl.col("source") == c["source"])
        & (pl.col("sentiment") == c["sentiment"]) & (pl.col("sarcasm") == c["sarcasm"])
    )["ri"].to_numpy()
    idx_by_cell[(c["variety"], c["source"], c["sentiment"], c["sarcasm"])] = (ids, c["len"])
yS, yC = pool["sentiment"].to_numpy(), pool["sarcasm"].to_numpy()
isAU = (pool["variety"] == "en-AU").to_numpy()


def boot_scores(pS: np.ndarray, pC: np.ndarray, B: int = 2000, stratified: bool = True, n_au=371, n_uk=386) -> pl.DataFrame:
    res = []
    au_idx, uk_idx = np.nonzero(isAU)[0], np.nonzero(~isAU)[0]
    for _ in range(B):
        if stratified:
            ix = np.concatenate([RNG.choice(ids, n, replace=True) for ids, n in idx_by_cell.values()])
        else:
            ix = np.concatenate([RNG.choice(au_idx, n_au), RNG.choice(uk_idx, n_uk)])
        a = isAU[ix]
        row = {}
        for task, y, pp in (("sent", yS, pS), ("sarc", yC, pC)):
            for v, m in (("AU", a), ("UK", ~a)):
                row[f"{task}-{v}"] = mf1(y[ix][m], pp[ix][m])
        row["score"] = (min(row["sent-AU"], row["sent-UK"]) + min(row["sarc-AU"], row["sarc-UK"])) / 2
        row["sent_binds"] = "sent-AU" if row["sent-AU"] < row["sent-UK"] else "sent-UK"
        row["sarc_binds"] = "sarc-AU" if row["sarc-AU"] < row["sarc-UK"] else "sarc-UK"
        res.append(row)
    return pl.DataFrame(res)


predS = (pool["p_sent"].to_numpy() >= 0.5).astype(int)
predC = (pool["p_sarc"].to_numpy() >= 0.5).astype(int)
bs = boot_scores(predS, predC, stratified=True)
bi = boot_scores(predS, predC, stratified=False)
comp_cols = ["sent-AU", "sent-UK", "sarc-AU", "sarc-UK", "score"]
summ = []
for tag, b in (("stratified (valid-like cells)", bs), ("iid per variety", bi)):
    for c in comp_cols:
        x = b[c].to_numpy()
        summ.append(dict(bootstrap=tag, component=c, mean=round(x.mean(), 4), sd=round(x.std(), 4),
                         p05=round(np.quantile(x, 0.05), 4), p95=round(np.quantile(x, 0.95), 4)))
summ = pl.DataFrame(summ)
print(summ)
summ.write_csv(TABLES / "bootstrap_noise.csv")
for tag, b in (("stratified", bs), ("iid", bi)):
    print(f"  [{tag}] binding frequency:", b["sent_binds"].value_counts().to_dicts(), b["sarc_binds"].value_counts().to_dicts())


# Plot 08: structural baselines vs a bag-of-words model, per component
sim_mean = simdf.mean().row(0, named=True)
base_rows = [
    ("constant: sent=1, sarc=0", rules_pool.filter((pl.col("sent_rule") == "all 1") & (pl.col("sarc_rule") == "all 0")).row(0, named=True)),
    ("random at segment prior", sim_mean),
    ("source rule: google→pos, reddit→neg & sarcastic",
     rules_pool.filter((pl.col("sent_rule") == "google→1, reddit→0") & (pl.col("sarc_rule") == "reddit→1")).row(0, named=True)),
    ("tf-idf + LR (OOF, thr 0.5)", s_ref),
]
PRED_ORDER = [n for n, _ in base_rows]
b8 = pl.DataFrame([dict(predictor=n, component=c, macro_f1=float(r[c])) for n, r in base_rows for c in comp_cols]).with_columns(
    pl.col("macro_f1").round(2).cast(pl.String).alias("lab"))
p = (
    ggplot(b8, aes(x="component", y="macro_f1", fill="predictor"))
    + geom_bar(stat="identity", position=position_dodge(width=0.84), width=0.8, color="#fcfcfb", size=0.6)
    + geom_text(aes(label="lab", group="predictor"), position=position_dodge(width=0.84), vjust=-0.6, size=4.2, color=INK)
    + scale_fill_manual(values=dict(zip(PRED_ORDER, ["#dcdad3", "#a9a7a0", "#63625d", "#1baf7a"])), limits=PRED_ORDER, name="")
    + scale_x_discrete(limits=comp_cols)
    + scale_y_continuous(limits=[0, 1.0])
    + guides(fill=guide_legend(nrow=2))
    + labs(
        title="What source/variety alone buys under the official metric",
        subtitle="Full pool (train+valid). The source rule alone reaches 0.64 and ties bag-of-words on sarc-UK (0.59): UK sarcasm is barely learned beyond 'is it reddit?'",
        x="", y="macro-F1",
    )
    + theme_base()
)
save(p, "08_trivial_rules", 1100, 560)

# Threshold-selection noise: tune the sarc-UK threshold on a valid-sized sample, apply it to the whole pool
grid_t = np.linspace(0.1, 0.8, 71)
uk_mask = ~(pool["variety"] == "en-AU").to_numpy()
yC_all, pC_all = pool["sarcasm"].to_numpy(), pool["p_sarc"].to_numpy()
pool_curve = np.array([mf1(yC_all[uk_mask], (pC_all[uk_mask] >= t).astype(int)) for t in grid_t])
uk_cells = [(ids, n) for (v, *_), (ids, n) in idx_by_cell.items() if v == "en-UK"]
chosen = []
for _ in range(300):
    ix = np.concatenate([RNG.choice(ids, n, replace=True) for ids, n in uk_cells])
    f = [mf1(yC_all[ix], (pC_all[ix] >= t).astype(int)) for t in grid_t]
    chosen.append(int(np.argmax(f)))
chosen = np.array(chosen)
print(f"sarc-UK threshold tuned on valid-sized samples: median {np.median(grid_t[chosen]):.2f}, "
      f"5-95% [{np.quantile(grid_t[chosen], 0.05):.2f}, {np.quantile(grid_t[chosen], 0.95):.2f}]; pool F1 at chosen thr: "
      f"mean {pool_curve[chosen].mean():.4f} vs pool-optimal {pool_curve.max():.4f} (loss {pool_curve.max() - pool_curve[chosen].mean():.4f})")

# paired bootstrap: how big must a real improvement be to show up on a valid-sized test? compare the LR vs LR with
# 10% of sarcasm errors flipped to correct (a genuinely better model) → fraction of test sets where better model wins.
err = np.nonzero(predC != yC)[0]
predC_better = predC.copy()
fix = RNG.choice(err, size=int(0.10 * len(err)), replace=False)
predC_better[fix] = yC[fix]
wins, deltas = 0, []
for _ in range(1000):
    ix = np.concatenate([RNG.choice(ids, n, replace=True) for ids, n in idx_by_cell.values()])
    m = ~isAU[ix]
    d = mf1(yC[ix][m], predC_better[ix][m]) - mf1(yC[ix][m], predC[ix][m])
    deltas.append(d)
    wins += d > 0
print(f"paired bootstrap: model fixing 10% of sarcasm errors: mean Δsarc-UK {np.mean(deltas):+.4f}, "
      f"better-model-wins {wins / 10:.1f}% of valid-sized test sets, ties {np.mean(np.array(deltas) == 0) * 100:.1f}%")

# Plot 10: bootstrap distributions per component
bl = bs.select(comp_cols).unpivot(variable_name="component", value_name="macro_f1").with_columns(
    pl.when(pl.col("component").str.ends_with("AU")).then(pl.lit("en-AU"))
    .when(pl.col("component").str.ends_with("UK")).then(pl.lit("en-UK")).otherwise(pl.lit("final score")).alias("variety")
)
sl = summ.filter(pl.col("bootstrap").str.starts_with("strat")).with_columns(
    ("sd " + pl.col("sd").round(3).cast(pl.String) + "\n90% [" + pl.col("p05").round(2).cast(pl.String) + ", "
     + pl.col("p95").round(2).cast(pl.String) + "]").alias("lab")
)
p = (
    ggplot(bl, aes(x="component", y="macro_f1", fill="variety", color="variety"))
    + geom_violin(alpha=0.25, size=0.8, scale="width", width=0.8, quantiles=[0.05, 0.5, 0.95], quantile_linetype="dashed")
    + geom_text(aes(x="component", label="lab"), data=sl, y=0.44, inherit_aes=False, size=5, color=INK2)
    + scale_fill_manual(values={**VAR_COL, "final score": "#52514e"}, name="")
    + scale_color_manual(values={**VAR_COL, "final score": "#52514e"}, name="")
    + scale_x_discrete(limits=comp_cols)
    + labs(
        title="How noisy is a valid-sized test set? (2,000 bootstrap draws)",
        subtitle="Fixed reference classifier; test sets with valid's exact label-cell counts (~371 AU / 386 UK rows, 30 UK sarcastic). sarc-UK is the widest",
        x="", y="macro-F1",
    )
    + theme_base()
)
save(p, "10_bootstrap_noise", 1050, 560)

# 6d. binormal simulation: macro-F1 vs predicted-positive rate for different prevalences / AUCs
print("\nbinormal simulation (population-level macro-F1 vs threshold)")
Phi = stats.norm.cdf
sims, opt = [], []
for prev, plabel in ((0.078, "7.8% (en-UK sarcasm)"), (0.297, "29.7% (en-AU sarcasm)"), (0.5, "50% (balanced)")):
    for auc in (0.80, 0.90, 0.95):
        dprime = np.sqrt(2) * stats.norm.ppf(auc)
        ts = np.linspace(-3, 5, 801)
        tpr, fpr = 1 - Phi(ts - dprime), 1 - Phi(ts)
        TP, FP = prev * tpr, (1 - prev) * fpr
        FN, TN = prev - TP, (1 - prev) - FP
        f1p = 2 * TP / (2 * TP + FP + FN)
        f1n = 2 * TN / (2 * TN + FN + FP)
        macro = (f1p + f1n) / 2
        rate = TP + FP
        for t, m, rr in zip(ts, macro, rate):
            if 0.005 <= rr <= 0.7:
                sims.append(dict(prevalence=plabel, auc=f"AUC {auc:.2f}", pred_rate=float(rr), macro_f1=float(m)))
        i = int(np.nanargmax(macro))
        # width of the rate window within 0.01 of the optimum
        ok = rate[macro >= macro[i] - 0.01]
        opt.append(dict(prevalence=plabel, auc=auc, best_macro_f1=round(float(macro[i]), 4),
                        best_pred_rate=round(float(rate[i]), 4), rate_window_1pt=(round(float(ok.min()), 3), round(float(ok.max()), 3)),
                        f1_at_prior_threshold=round(float(macro[np.argmin(np.abs(rate - prev))]), 4)))
opt = pl.DataFrame(opt)
print(opt)
opt.with_columns(pl.col("rate_window_1pt").cast(pl.List(pl.String)).list.join("–")).write_csv(TABLES / "binormal_threshold_sim.csv")
sims = pl.DataFrame(sims)
opt_pts = opt.with_columns(pl.format("AUC {}", pl.col("auc").round(2).cast(pl.String).str.pad_end(4, "0")).alias("auc")).rename(
    {"best_pred_rate": "pred_rate", "best_macro_f1": "macro_f1"})
prev_lines = pl.DataFrame({"prevalence": ["7.8% (en-UK sarcasm)", "29.7% (en-AU sarcasm)", "50% (balanced)"], "x": [0.078, 0.297, 0.5]})
p = (
    ggplot(sims, aes("pred_rate", "macro_f1", color="auc"))
    + geom_vline(aes(xintercept="x"), data=prev_lines, linetype="dashed", color="#9a9994", size=0.6)
    + geom_line(size=1.3)
    + geom_point(data=opt_pts, size=4.5, shape=21, fill="#fcfcfb", stroke=2)
    + facet_wrap("prevalence", ncol=3, order=0)
    + scale_color_manual(values={"AUC 0.80": "#b9c9ec", "AUC 0.90": "#6f93db", "AUC 0.95": "#2a4f9e"}, name="")
    + scale_x_continuous(format=".0%")
    + labs(
        title="Rare classes: macro-F1 is capped lower and the sweet spot is narrow",
        subtitle="Binormal simulation. Dashed = true prevalence; circles = optimum. At 7.8% prevalence AUC 0.90 peaks at 0.74; the optimum always sits at predicted-rate ≈ prevalence",
        x="fraction of rows predicted positive (sets the threshold)", y="macro-F1 (population)",
    )
    + theme_base()
)
save(p, "11_threshold_sim_prevalence", 1150, 480)

# 6e. sarc-UK decomposition: google rows are free true negatives
d = pool.filter(pl.col("variety") == "en-UK")
yuk, puk = d["sarcasm"].to_numpy(), (d["p_sarc"].to_numpy() >= 0.5).astype(int)
isr = (d["source"] == "reddit").to_numpy()
print(f"\nsarc-UK macro-F1 all rows {mf1(yuk, puk):.4f} vs reddit-only {mf1(yuk[isr], puk[isr]):.4f}; "
      f"UK reddit sarcasm prevalence {yuk[isr].mean():.3f}")
d = pool.filter(pl.col("variety") == "en-AU")
yau, pau = d["sarcasm"].to_numpy(), (d["p_sarc"].to_numpy() >= 0.5).astype(int)
isr = (d["source"] == "reddit").to_numpy()
print(f"sarc-AU macro-F1 all rows {mf1(yau, pau):.4f} vs reddit-only {mf1(yau[isr], pau[isr]):.4f}; google-only sarcasm AUC "
      f"{roc_auc_score(yau[~isr], d['p_sarc'].to_numpy()[~isr]):.3f} (n_pos={yau[~isr].sum()})")

# 6f. sentiment: which variety binds and why (per segment accuracy of reference model)
seg_acc = ref.group_by("segment").agg(
    (pl.col("sentiment_pred") == pl.col("sentiment")).mean().round(4).alias("sent_acc"),
    (pl.col("sarcasm_pred") == pl.col("sarcasm")).mean().round(4).alias("sarc_acc"),
).sort("segment")
print(seg_acc)
print("\ndone.")
