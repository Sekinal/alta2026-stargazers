"""Label noise / ambiguity: kNN label agreement, confident learning, conflicting near-duplicates.

Reference OOF probabilities = mean of the best linear probes on bge_large, bge_base and mpnet
(saved as preds/emb_ens3_lr). Outputs tables to cache/ and plots to plots/.
"""

import numpy as np
import polars as pl
from lets_plot import (
    aes, coord_flip, geom_bar, geom_errorbar, geom_hline, geom_point, geom_text, ggplot, labs,
    position_dodge, scale_color_manual, scale_fill_manual, scale_y_continuous, facet_wrap,
)
from sklearn.neighbors import NearestNeighbors

from _shared import CACHE, VARIETY_COLORS, emb, my_theme, pool, save, segment
from alta.common import PREDS, save_preds, score

K = 10
TASKS = {"sentiment": "p_sent", "sarcasm": "p_sarc"}


def ensemble(df: pl.DataFrame) -> pl.DataFrame:
    ps = [pl.read_parquet(PREDS / f"emb_{m}_lr.parquet").filter(pl.col("split") == "oof")
          for m in ("bge_large", "bge_base", "mpnet")]
    p = pl.concat(ps).group_by("uid").agg(pl.col("p_sent", "p_sarc").mean())
    save_preds("emb_ens3_lr", p, meta={"members": ["bge_large", "bge_base", "mpnet"], "design": "best per model"})
    return df.join(p, on="uid", how="left")


def _knn(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X[idx])
    _, nb = nn.kneighbors(X[idx])
    return idx[nb[:, 1:]]  # drop self


def knn_agreement(df: pl.DataFrame, X: np.ndarray) -> pl.DataFrame:
    """Share of the K nearest neighbours sharing each label (neighbours searched within the same
    variety x source segment, so chance = p^2 + (1-p)^2 exactly), plus variety purity of the
    neighbourhood within the same source (how separable AU and UK are)."""
    out = {f"knn_{t}": np.zeros(len(df)) for t in TASKS}
    out["knn_same_variety"] = np.zeros(len(df))
    seg = df["segment"].to_numpy()
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        nb = _knn(X, idx)
        for t in TASKS:
            y = df[t].to_numpy()
            out[f"knn_{t}"][idx] = (y[nb] == y[idx][:, None]).mean(1)
    src, var = df["source"].to_numpy(), df["variety"].to_numpy()
    for s in np.unique(src):
        idx = np.where(src == s)[0]
        nb = _knn(X, idx)
        out["knn_same_variety"][idx] = (var[nb] == var[idx][:, None]).mean(1)
    return df.with_columns(pl.Series(k, v) for k, v in out.items())


def confident_learning(df: pl.DataFrame) -> pl.DataFrame:
    """Binary confident learning (Northcutt et al. 2021), thresholds computed per segment.

    t_c = mean predicted P(c) over rows labelled c. A row labelled y is flagged when the model's
    probability of the OTHER class clears that class's threshold (and exceeds its own-class margin).
    """
    parts = []
    for seg_df in df.partition_by("segment"):
        cols = {}
        for t, pc in TASKS.items():
            y = seg_df[t].to_numpy()
            p1 = seg_df[pc].to_numpy()
            if y.sum() < 5 or (1 - y).sum() < 5:
                cols[f"cl_{t}"] = np.zeros(len(y), dtype=bool)
                continue
            t1, t0 = p1[y == 1].mean(), (1 - p1)[y == 0].mean()
            above1, above0 = p1 >= t1, (1 - p1) >= t0
            # when both clear their threshold, CL assigns the class with the larger margin
            guess1 = above1 & (~above0 | (p1 - t1 >= (1 - p1) - t0))
            guess0 = above0 & ~guess1
            cols[f"cl_{t}"] = ((y == 0) & guess1) | ((y == 1) & guess0)
        parts.append(seg_df.select("uid").with_columns(pl.Series(k, v) for k, v in cols.items()))
    return df.join(pl.concat(parts), on="uid", how="left")  # keep row order aligned with X


def near_duplicates(df: pl.DataFrame, X: np.ndarray, thr: float = 0.95) -> pl.DataFrame:
    """All pairs among 5-NN with cosine >= thr, flagged whether byte-identical."""
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
    dist, nb = nn.kneighbors(X)
    rows = []
    text = df["text"].to_list()
    for i in range(len(df)):
        for d, j in zip(dist[i, 1:], nb[i, 1:]):
            if j > i and 1 - d >= thr:
                rows.append((i, int(j), 1 - d, text[i] == text[j]))
    pairs = pl.DataFrame(rows, schema=["i", "j", "cos", "identical"], orient="row")
    a = df.select("uid", "segment", "sentiment", "sarcasm", "text").with_row_index("i")
    return (pairs.with_columns(pl.col("i", "j").cast(pl.UInt32))
            .join(a, on="i").join(a.rename(lambda c: c + "_b").rename({"i_b": "j"}), on="j"))


def main():
    df = pool().with_columns(segment(pool()).alias("segment"))
    df = ensemble(df)
    ens = df.with_columns((pl.col("p_sent") >= 0.5).cast(pl.Int64).alias("sentiment_pred"),
                          (pl.col("p_sarc") >= 0.5).cast(pl.Int64).alias("sarcasm_pred"))
    print("ens3 OOF score:", {k: round(v, 4) for k, v in score(ens).items()})
    print("ens3 valid-only:", {k: round(v, 4) for k, v in score(ens.filter(pl.col("origin") == "valid")).items()})

    X = emb("bge_large")
    df = knn_agreement(df, X)
    df = confident_learning(df)
    df.drop("text").write_parquet(CACHE / "noise_table.parquet")

    # ---- per-segment summary: observed kNN agreement vs chance, CL flag rate
    summ = (df.group_by("segment").agg(
        pl.len().alias("n"),
        pl.col("sarcasm").mean().alias("sarc_rate"),
        pl.col("sentiment").mean().alias("pos_rate"),
        pl.col("knn_sentiment").mean().alias("knn_sent"),
        pl.col("knn_sarcasm").mean().alias("knn_sarc"),
        pl.col("knn_sarcasm").filter(pl.col("sarcasm") == 1).mean().alias("knn_sarc_pos"),
        pl.col("knn_same_variety").mean().alias("knn_same_var"),
        pl.col("cl_sentiment").mean().alias("cl_sent_rate"),
        pl.col("cl_sarcasm").mean().alias("cl_sarc_rate"),
        pl.col("cl_sentiment").sum().alias("cl_sent_n"),
        pl.col("cl_sarcasm").sum().alias("cl_sarc_n"),
    ).with_columns(
        (pl.col("pos_rate") ** 2 + (1 - pl.col("pos_rate")) ** 2).alias("chance_sent"),
        (pl.col("sarc_rate") ** 2 + (1 - pl.col("sarc_rate")) ** 2).alias("chance_sarc"),
    ).sort("segment"))
    with pl.Config(tbl_cols=20, float_precision=3, tbl_width_chars=250):
        print(summ)
    summ.write_parquet(CACHE / "noise_summary.parquet")

    # kNN variety purity vs the chance level for that source (share of each variety within source)
    vs = df.group_by("source", "variety").agg(pl.len().alias("n"), pl.col("knn_same_variety").mean())
    vs = vs.with_columns((pl.col("n") / pl.col("n").sum().over("source")).alias("chance"))
    print(vs.sort("source", "variety"))

    # ---- plot: kNN agreement vs chance per segment & task
    long = pl.concat([
        summ.select("segment", pl.lit("sentiment").alias("task"), pl.col("knn_sent").alias("observed"),
                    pl.col("chance_sent").alias("chance")),
        summ.select("segment", pl.lit("sarcasm").alias("task"), pl.col("knn_sarc").alias("observed"),
                    pl.col("chance_sarc").alias("chance")),
    ]).with_columns(pl.col("segment").str.slice(0, 2).alias("variety"))
    long = long.with_columns(("en-" + pl.col("variety")).alias("variety"),
                             pl.col("observed").map_elements(lambda x: f"{x:.2f}", return_dtype=pl.String).alias("lab"))
    p = (ggplot(long) + my_theme()
         + geom_errorbar(aes(x="segment", ymin="chance", ymax="observed", color="variety"), width=0, size=1.4)
         + geom_point(aes(x="segment", y="chance"), shape=124, size=8, color="#555555")
         + geom_point(aes(x="segment", y="observed", color="variety"), size=5)
         + geom_text(aes(x="segment", y="observed", label="lab"), nudge_y=0.012, hjust=0, size=7)
         + facet_wrap("task") + coord_flip()
         + scale_color_manual(values=VARIETY_COLORS)
         + scale_y_continuous(limits=[0.45, 1.05])
         + labs(title="How often do a text's 10 nearest neighbours share its label?",
                subtitle="dot = observed (bge-large, neighbours within the same variety x source segment); bar = chance given the segment base rate",
                x="", y="mean kNN label agreement (k = 10)", color=""))
    save(p, "knn_agreement", 1200, 520)

    # ---- suspected mislabels: CL-flagged rows ranked by confidence in the other class
    text = pool().select("uid", "text")
    for t, pc in TASKS.items():
        sus = (df.filter(pl.col(f"cl_{t}"))
               .with_columns(pl.when(pl.col(t) == 1).then(1 - pl.col(pc)).otherwise(pl.col(pc)).alias("p_other"))
               .sort("p_other", descending=True)
               .join(text, on="uid", how="left", suffix="_t")
               .select("uid", "segment", t, "p_other", f"knn_{t}", "text"))
        sus.write_parquet(CACHE / f"suspects_{t}.parquet")
        with pl.Config(tbl_rows=25, fmt_str_lengths=160, tbl_width_chars=260, float_precision=2):
            print(t, "flagged:", sus.height)
            print(sus.head(25))

    # ---- conflicting near-duplicates
    nd = near_duplicates(df, X)
    nd.write_parquet(CACHE / "near_duplicates.parquet")
    conf = nd.filter((pl.col("sentiment") != pl.col("sentiment_b")) | (pl.col("sarcasm") != pl.col("sarcasm_b")))
    print("near-dup pairs (cos>=.95):", nd.height, "identical:", nd["identical"].sum(),
          "label-conflicting:", conf.height, "of which identical text:", conf["identical"].sum())
    with pl.Config(tbl_rows=30, fmt_str_lengths=120, tbl_width_chars=300, float_precision=3):
        print(conf.select("cos", "identical", "segment", "sentiment", "sarcasm", "sentiment_b", "sarcasm_b",
                          "text", "text_b").head(30))


if __name__ == "__main__":
    main()
