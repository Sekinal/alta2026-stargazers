"""2-D UMAP maps of the embeddings, coloured by source / variety / sentiment / sarcasm / joint label.

Usage: uv run python eda/semantic/03_umap.py [model ...]   (default: bge_large mpnet)
"""

import sys

import numpy as np
import polars as pl
import umap
from lets_plot import (
    aes, facet_grid, facet_wrap, geom_point, ggplot, guide_legend, guides, labs,
    scale_color_manual, theme, element_blank,
)

from _shared import (
    CACHE, JOINT_COLORS, LABEL_COLORS, SENT_COLORS, SOURCE_COLORS, VARIETY_COLORS, emb,
    my_theme, pool, save,
)

NO_AXES = theme(axis_text=element_blank(), axis_title=element_blank(), axis_ticks=element_blank())


def coords(model: str) -> pl.DataFrame:
    path = CACHE / f"umap_{model}.npy"
    if path.exists():
        xy = np.load(path)
    else:
        xy = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=0).fit_transform(emb(model))
        np.save(path, xy)
    df = pool()
    return df.with_columns(
        pl.Series("x", xy[:, 0]), pl.Series("y", xy[:, 1]),
        pl.when(pl.col("sentiment") == 1).then(pl.lit("pos")).otherwise(pl.lit("neg")).alias("sent"),
        pl.col("sarcasm").cast(pl.String).alias("sarc"),
        pl.concat_str(
            pl.when(pl.col("sentiment") == 1).then(pl.lit("pos")).otherwise(pl.lit("neg")),
            pl.when(pl.col("sarcasm") == 1).then(pl.lit("sarc")).otherwise(pl.lit("not sarc")),
            separator=" / ",
        ).alias("joint"),
    )


def base(d, color, sub, title):
    return (ggplot(d, aes("x", "y", color=color)) + my_theme() + NO_AXES
            + labs(title=title, subtitle=sub, color=""))


def main(models: list[str]) -> None:
    for m in models:
        d = coords(m)
        sub = f"UMAP (cosine, n_neighbors=30) of {m} sentence embeddings, n = {d.height}"
        big = guides(color=guide_legend(override_aes={"size": 5, "alpha": 1}))

        p = (base(d, "source", sub, "Google reviews and Reddit posts occupy different regions")
             + geom_point(size=1.0, alpha=0.5) + scale_color_manual(values=SOURCE_COLORS) + big)
        save(p, f"umap_{m}_source", 900, 750)

        p = (base(d, "variety", sub + "; faceted by source", "AU and UK interleave in Google reviews but drift apart on Reddit (topic)")
             + geom_point(size=1.0, alpha=0.5) + scale_color_manual(values=VARIETY_COLORS)
             + facet_wrap("source", scales="free", ncol=2) + big)
        save(p, f"umap_{m}_variety", 1400, 650)

        # draw positives on top so the minority class is visible
        ds = d.sort("sarcasm")
        p = (base(ds, "sarc", sub + "; red = sarcastic", "Where does sarcasm live?")
             + geom_point(data=ds.filter(pl.col("sarcasm") == 0), size=0.9, alpha=0.35)
             + geom_point(data=ds.filter(pl.col("sarcasm") == 1), size=1.3, alpha=0.8)
             + scale_color_manual(values=LABEL_COLORS, labels={"0": "not sarcastic", "1": "sarcastic"})
             + facet_grid("source", "variety", scales="free") + big)
        save(p, f"umap_{m}_sarcasm", 1300, 1100)

        p = (base(d, "sent", sub, "Sentiment is the dominant axis inside each source")
             + geom_point(size=0.9, alpha=0.5) + scale_color_manual(values=SENT_COLORS)
             + facet_grid("source", "variety", scales="free") + big)
        save(p, f"umap_{m}_sentiment", 1300, 1100)

        dj = d.sort("sarcasm")
        p = (base(dj, "joint", sub, "Joint label: AU-reddit sarcasm forms a region, UK-reddit sarcasm is mixed into negative posts")
             + geom_point(size=1.0, alpha=0.55) + scale_color_manual(values=JOINT_COLORS)
             + facet_grid("source", "variety", scales="free") + big)
        save(p, f"umap_{m}_joint", 1300, 1100)
        print("done", m)


if __name__ == "__main__":
    main(sys.argv[1:] or ["bge_large", "mpnet"])
