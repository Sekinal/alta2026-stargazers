"""Build cleaned AU/UK Reddit corpora from the raw Arctic Shift pulls (fetch_reddit_arctic.py).

Source: Arctic Shift (https://arctic-shift.photon-reddit.com), Reddit comments + submissions,
2022-01 .. 2025-12. Terms: Reddit User Agreement / Data API terms; research use, not redistributed.
Posts become "title\\nselftext"; removed/deleted bodies, bots and AutoModerator are dropped.

    uv run --with py3langid python experiments/data/build_reddit.py --raw /workspace/raw/reddit \
        --out /workspace/raw/clean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indomain_common import BOT_AUTHOR, BOT_TEXT, approx_tokens, clean_text, finalise  # noqa: E402

LICENSE = "Reddit content via Arctic Shift; Reddit User Agreement, research use only"
DROP_BODY = ["[deleted]", "[removed]", "[ Removed by Reddit ]", "[removed by reddit]", ""]


def load(raw: Path) -> pl.DataFrame:
    comments, posts = [], []
    for p in sorted(raw.glob("*_comments.parquet")):
        comments.append(pl.read_parquet(p).select(
            pl.col("body").alias("text"), "author", "variety", pl.col("subreddit").alias("community"),
            pl.col("created_utc").cast(pl.Int64), pl.col("score").cast(pl.Int64),
            pl.col("parent_id").cast(pl.String), pl.col("link_id").cast(pl.String), pl.lit("comment").alias("kind")))
    for p in sorted(raw.glob("*_posts.parquet")):
        d = pl.read_parquet(p)
        body = pl.col("selftext").fill_null("")
        body = pl.when(body.str.strip_chars().is_in(DROP_BODY)).then(pl.lit("")).otherwise(body)
        posts.append(d.select(
            pl.concat_str([pl.col("title").fill_null(""), body], separator="\n").str.strip_chars().alias("text"),
            "author", "variety", pl.col("subreddit").alias("community"),
            pl.col("created_utc").cast(pl.Int64), pl.col("score").cast(pl.Int64),
            pl.lit(None, pl.String).alias("parent_id"), pl.concat_str([pl.lit("t3_"), pl.col("id")]).alias("link_id"),
            pl.lit("post").alias("kind")))
    return pl.concat(comments + posts, how="vertical_relaxed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/raw/reddit", type=Path)
    ap.add_argument("--out", default="/workspace/raw/clean", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.raw)
    n0 = df.height
    df = df.filter(~pl.col("text").str.strip_chars().is_in(DROP_BODY),
                   ~pl.col("author").fill_null("").str.contains(BOT_AUTHOR),
                   ~pl.col("text").str.contains(BOT_TEXT))
    n_bot = df.height
    df = df.with_columns(clean_text(pl.col("text"), reddit=True).alias("text"),
                         pl.lit("reddit").alias("source"), pl.lit("arctic_shift").alias("dataset"),
                         pl.lit(LICENSE).alias("license")).drop("author")
    stats = {}
    for v, tag in [("en-AU", "au"), ("en-UK", "uk")]:
        g, st = finalise(df.filter(pl.col("variety") == v))
        st["tokens"] = approx_tokens(g)
        st["time_range"] = [g["created_utc"].min(), g["created_utc"].max()]
        st["per_community"] = {k: n for k, n in g.group_by("community").len().sort("len", descending=True).rows()}
        stats[f"reddit_{tag}"] = st
        g.write_parquet(args.out / f"reddit_{tag}.parquet")
        print(f"reddit_{tag}: {g.height} rows, ~{st['tokens'] / 1e6:.1f}M tokens", st, flush=True)
    stats["raw_rows"], stats["after_deleted_bot_filter"] = n0, n_bot
    (args.out / "reddit_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
