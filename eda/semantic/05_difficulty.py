"""Difficulty analysis on the ens3 OOF probabilities (run 04_noise.py first).

Calibration per variety, macro-F1-vs-threshold curves per variety and fold, nested threshold
tuning, per-fold score spread, hard UK sarcasm examples, and error rates by surface slices.
"""

import numpy as np
import polars as pl
from lets_plot import (
    aes, facet_grid, facet_wrap, geom_abline, geom_line, geom_path, geom_point, geom_segment, geom_vline, ggplot,
    labs, scale_color_manual, scale_size, coord_fixed, scale_x_continuous, scale_y_continuous, geom_text,
    geom_bar, position_dodge, geom_errorbar, guides,
)
from sklearn.metrics import f1_score, precision_recall_curve

from _shared import CACHE, VARIETY_COLORS, my_theme, pool, save, segment
from alta.common import N_FOLDS, PREDS, VARIETIES, score

GRID = np.round(np.linspace(0.05, 0.95, 91), 3)
TASKS = {"sentiment": "p_sent", "sarcasm": "p_sarc"}


def load() -> pl.DataFrame:
    p = pl.read_parquet(PREDS / "emb_ens3_lr.parquet").filter(pl.col("split") == "oof").drop("split")
    df = pool()
    return df.with_columns(segment(df).alias("segment")).join(p, on="uid", how="left")


def mf1(y, p, t):
    return f1_score(y, p >= t, average="macro", zero_division=0)


def calibration(df: pl.DataFrame) -> None:
    rows = []
    for t, pc in TASKS.items():
        for v in VARIETIES:
            d = df.filter(pl.col("variety") == v)
            y, p = d[t].to_numpy(), d[pc].to_numpy()
            bins = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                m = bins == b
                if m.sum() >= 10:
                    rows.append({"task": t, "variety": v, "pred": p[m].mean(), "obs": y[m].mean(), "n": int(m.sum())})
            print(f"calibration {t} {v}: mean p={p.mean():.3f} base rate={y.mean():.3f} "
                  f"brier={np.mean((p - y) ** 2):.4f}")
    c = pl.DataFrame(rows)
    p = (ggplot(c, aes("pred", "obs", color="variety")) + my_theme()
         + geom_abline(slope=1, intercept=0, linetype="dashed", color="#999999")
         + geom_line(size=1.1) + geom_point(aes(size="n"), alpha=0.85)
         + facet_wrap("task", ncol=2) + scale_color_manual(values=VARIETY_COLORS)
         + scale_size(range=[2, 7], guide="none")
         + scale_x_continuous(limits=[0, 1]) + scale_y_continuous(limits=[0, 1])
         + labs(title="Reliability of the ens3 linear probe (class-balanced LR)",
                subtitle="10 equal-width bins (>= 10 rows, point size = rows); below the diagonal = P(positive) over-estimated",
                x="mean predicted probability", y="observed positive rate", color=""))
    save(p, "calibration", 1200, 620)


def thresholds(df: pl.DataFrame) -> None:
    """Macro-F1 vs threshold per variety (overall + each fold), and nested threshold tuning."""
    curves, opt_rows = [], []
    for t, pc in TASKS.items():
        for v in VARIETIES:
            d = df.filter(pl.col("variety") == v)
            y, p, f = d[t].to_numpy(), d[pc].to_numpy(), d["fold"].to_numpy()
            for fold in [-1, *range(N_FOLDS)]:
                m = np.ones(len(y), bool) if fold == -1 else f == fold
                vals = [mf1(y[m], p[m], g) for g in GRID]
                curves += [{"task": t, "variety": v, "fold": "all" if fold == -1 else str(fold),
                            "thr": g, "f1": s} for g, s in zip(GRID, vals)]
                opt_rows.append({"task": t, "variety": v, "fold": fold, "best_thr": GRID[int(np.argmax(vals))],
                                 "best_f1": max(vals), "f1@0.5": mf1(y[m], p[m], 0.5)})
    cur = pl.DataFrame(curves)
    opt = pl.DataFrame(opt_rows)
    with pl.Config(tbl_rows=40, float_precision=3):
        print(opt.filter(pl.col("fold") == -1))
        print(opt.filter(pl.col("fold") >= 0).group_by("task", "variety").agg(
            pl.col("best_thr").mean().alias("thr_mean"), pl.col("best_thr").std().alias("thr_sd"),
            pl.col("best_thr").min().alias("thr_min"), pl.col("best_thr").max().alias("thr_max"),
            pl.col("f1@0.5").std().alias("f1_sd_across_folds")).sort("task", "variety"))
    opt.write_parquet(CACHE / "thresholds.parquet")

    allc = cur.filter(pl.col("fold") == "all")
    best = opt.filter(pl.col("fold") == -1).rename({"best_thr": "thr", "best_f1": "f1"})
    p = (ggplot() + my_theme()
         + geom_line(aes("thr", "f1", color="variety", group="fold"),
                     data=cur.filter(pl.col("fold") != "all").with_columns(pl.concat_str("variety", "fold").alias("fold")),
                     size=0.5, alpha=0.35)
         + geom_line(aes("thr", "f1", color="variety"), data=allc, size=1.6)
         + geom_point(aes("thr", "f1", color="variety"), data=best, size=5, shape=21, fill="white", stroke=2)
         + geom_vline(xintercept=0.5, linetype="dotted", color="#777777")
         + facet_wrap("task", scales="free_y") + scale_color_manual(values=VARIETY_COLORS)
         + labs(title="Macro-F1 vs decision threshold, per variety",
                subtitle="thick = all OOF rows, thin = each of the 5 folds, ring = F1-optimal threshold",
                x="threshold on P(positive)", y="macro-F1 within variety", color=""))
    save(p, "threshold_curves", 1200, 560)

    # nested: pick per-variety threshold on 4 folds, apply to the 5th
    res = {"0.5": [], "global": [], "per_variety": []}
    for fold in range(N_FOLDS):
        tr, te = df.filter(pl.col("fold") != fold), df.filter(pl.col("fold") == fold)
        thr = {}
        for t, pc in TASKS.items():
            g = GRID[int(np.argmax([mf1(tr[t].to_numpy(), tr[pc].to_numpy(), x) for x in GRID]))]
            thr[(t, "global")] = g
            for v in VARIETIES:
                d = tr.filter(pl.col("variety") == v)
                thr[(t, v)] = GRID[int(np.argmax([mf1(d[t].to_numpy(), d[pc].to_numpy(), x) for x in GRID]))]
        for mode in res:
            def th(t, v):
                return 0.5 if mode == "0.5" else thr[(t, "global")] if mode == "global" else thr[(t, v)]
            te2 = te.with_columns(
                pl.when(pl.col("variety") == "en-AU").then(pl.col("p_sent") >= th("sentiment", "en-AU"))
                .otherwise(pl.col("p_sent") >= th("sentiment", "en-UK")).cast(pl.Int64).alias("sentiment_pred"),
                pl.when(pl.col("variety") == "en-AU").then(pl.col("p_sarc") >= th("sarcasm", "en-AU"))
                .otherwise(pl.col("p_sarc") >= th("sarcasm", "en-UK")).cast(pl.Int64).alias("sarcasm_pred"))
            res[mode].append(score(te2))
    for mode, lst in res.items():
        keys = lst[0].keys()
        print(f"nested thresholds [{mode}]:",
              {k: f"{np.mean([r[k] for r in lst]):.3f}±{np.std([r[k] for r in lst]):.3f}" for k in keys})

    # PR curves for sarcasm per variety (reddit only: UK google has a single positive)
    rows = []
    for v in VARIETIES:
        d = df.filter((pl.col("variety") == v) & (pl.col("source") == "reddit"))
        pr, rc, th = precision_recall_curve(d["sarcasm"].to_numpy(), d["p_sarc"].to_numpy())
        rows += [{"variety": v, "precision": a, "recall": b} for a, b in zip(pr, rc)]
        base = d["sarcasm"].mean()
        print(f"PR sarcasm reddit {v}: base rate {base:.3f}")
    p = (ggplot(pl.DataFrame(rows), aes("recall", "precision", color="variety")) + my_theme()
         + geom_path(size=1.3) + scale_color_manual(values=VARIETY_COLORS)
         + scale_x_continuous(limits=[0, 1]) + scale_y_continuous(limits=[0, 1])
         + labs(title="Sarcasm precision-recall on Reddit, by variety",
                subtitle="ens3 linear probe OOF; UK sarcasm is rarer (21.7% vs 42.0%) and far less separable",
                x="recall", y="precision", color=""))
    save(p, "pr_sarcasm_reddit", 850, 650)


def hard_examples(df: pl.DataFrame) -> None:
    d = df.filter((pl.col("variety") == "en-UK") & (pl.col("source") == "reddit"))
    with pl.Config(tbl_rows=20, fmt_str_lengths=200, tbl_width_chars=260, float_precision=2):
        print("UK-reddit FN (sarcastic, lowest p):")
        print(d.filter(pl.col("sarcasm") == 1).sort("p_sarc").select("uid", "sentiment", "p_sarc", "text").head(15))
        print("UK-reddit FP (not sarcastic, highest p):")
        print(d.filter(pl.col("sarcasm") == 0).sort("p_sarc", descending=True).select("uid", "sentiment", "p_sarc", "text").head(15))
    # does the model use negativity as a proxy for sarcasm?
    r = (df.filter(pl.col("source") == "reddit")
         .group_by("variety", "sentiment", "sarcasm")
         .agg(pl.len().alias("n"), (pl.col("p_sarc") >= 0.5).mean().alias("pred_sarc_rate"),
              pl.col("p_sarc").mean().alias("mean_p")).sort("variety", "sentiment", "sarcasm"))
    with pl.Config(tbl_rows=20, float_precision=3):
        print(r)


def slices(df: pl.DataFrame) -> None:
    """Sarcasm base rate and OOF macro-F1 on Reddit slices (surface cues the context-less text has)."""
    d = df.filter(pl.col("source") == "reddit").with_columns(
        pl.col("text").str.split(" ").list.len().alias("words"),
        pl.col("text").str.contains(r"(?m)^\s*(>|&gt;)").alias("quotes_parent"),
        pl.col("text").str.contains(r"\"[^\"]{3,}\"|“[^”]{3,}”").alias("has_quotes"),
        pl.col("text").str.contains(r"\?").alias("question"),
        pl.col("text").str.contains(r"(?i)(^|\s)/s\b").alias("slash_s"),
    ).with_columns(
        pl.when(pl.col("words") <= 10).then(pl.lit("a) <=10 words"))
        .when(pl.col("words") <= 30).then(pl.lit("b) 11-30"))
        .when(pl.col("words") <= 80).then(pl.lit("c) 31-80"))
        .otherwise(pl.lit("d) >80")).alias("len_bin"))
    rows = []
    for col in ("len_bin", "quotes_parent", "has_quotes", "question", "slash_s"):
        for (v, val), g in d.group_by("variety", col):
            if g.height < 15:
                continue
            y, p = g["sarcasm"].to_numpy(), g["p_sarc"].to_numpy()
            rows.append({"slice": col, "value": str(val), "variety": v, "n": g.height,
                         "sarc_rate": y.mean(), "acc": ((p >= 0.5) == y).mean(),
                         "macroF1": mf1(y, p, 0.5) if 0 < y.sum() < len(y) else float("nan")})
    s = pl.DataFrame(rows).sort("slice", "value", "variety")
    with pl.Config(tbl_rows=60, float_precision=3):
        print(s)
    s.write_parquet(CACHE / "slices.parquet")
    lb = s.filter(pl.col("slice") == "len_bin").with_columns(
        ("n=" + pl.col("n").cast(pl.String)).alias("n_lab"), pl.lit(0.04).alias("y0"))
    p = (ggplot(lb, aes("value", "macroF1", fill="variety")) + my_theme()
         + geom_bar(stat="identity", position=position_dodge(0.8), width=0.75)
         + geom_text(aes(label="n_lab", y="y0", group="variety"), position=position_dodge(0.8), size=7,
                     color="white")
         + scale_color_manual(values=VARIETY_COLORS) + scale_y_continuous(limits=[0, 1])
         + labs(title="Reddit sarcasm: OOF macro-F1 by text length",
                subtitle="ens3 linear probe, threshold 0.5; n = Reddit rows in the bin",
                x="words", y="macro-F1 (sarcasm)", fill=""))
    from lets_plot import scale_fill_manual
    p += scale_fill_manual(values=VARIETY_COLORS)
    save(p, "sarcasm_by_length", 900, 560)


def main():
    df = load()
    calibration(df)
    thresholds(df)
    hard_examples(df)
    slices(df)


if __name__ == "__main__":
    main()
