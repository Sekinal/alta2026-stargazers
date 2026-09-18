"""Balanced sarcasm pre-SFT sample from the external in-community /s Reddit sets (+ gold EPIC).

  uv run python experiments/llm/build_presft.py --n 12000 --out experiments/llm/cache/presft_v1.parquet

Output columns: text, variety, source (= reddit), sarcasm, dataset, subreddit. The texts are used
in exactly the task prompt format (sarc_v2, variety slot filled), without parent_text. UK politics
subreddits are sampled first; half of each variety is positive. The sets were already deduplicated
against the pool by the data agent; exact duplicates of pool texts are dropped again here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402

EXT = ROOT / "data" / "external" / "sarcasm"
UK_POLITICS = {"ukpolitics", "unitedkingdom", "LabourUK", "tories", "Scotland", "scotland"}
EPIC_VARIETY = {"australia": "en-AU", "CasualUK": "en-UK", "britishproblems": "en-UK"}
_S = re.compile(r"\s*/s\b", re.I)


def clean(df: pl.DataFrame) -> pl.DataFrame:
    return (df.with_columns(pl.col("text").map_elements(lambda t: _S.sub("", t).strip(),
                                                         return_dtype=pl.String))
              .filter(pl.col("text").str.len_chars().is_between(8, 1500)))


def sample(df: pl.DataFrame, n: int, seed: int, prefer: set[str] | None = None) -> pl.DataFrame:
    """n/2 positives + n/2 negatives; `prefer` subreddits get up to 60% of each half."""
    parts = []
    for lab in (1, 0):
        d = df.filter(pl.col("sarcasm") == lab).sample(fraction=1.0, shuffle=True, seed=seed)
        k = n // 2
        if prefer:
            pr = d.filter(pl.col("subreddit").is_in(list(prefer))).head(int(0.6 * k))
            rest = d.filter(~pl.col("subreddit").is_in(list(prefer))).head(k - pr.height)
            parts += [pr, rest]
        else:
            parts.append(d.head(k))
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "experiments/llm/cache/presft_v1.parquet"))
    a = ap.parse_args()
    cols = ["text", "sarcasm", "subreddit", "dataset"]
    uk = clean(pl.read_parquet(EXT / "reddit_s_uk.parquet").select(cols))
    au = clean(pl.read_parquet(EXT / "reddit_s_au.parquet").select(cols))
    epic = clean(pl.read_parquet(EXT / "epic_irony.parquet")
                 .filter(pl.col("subreddit").is_in(list(EPIC_VARIETY))).select(cols))
    n_uk, n_au = int(a.n * 0.55), int(a.n * 0.35)
    out = pl.concat([
        sample(uk, n_uk, a.seed, UK_POLITICS).with_columns(pl.lit("en-UK").alias("variety")),
        sample(au, n_au, a.seed).with_columns(pl.lit("en-AU").alias("variety")),
        epic.with_columns(pl.col("subreddit").replace_strict(EPIC_VARIETY).alias("variety")),
    ]).with_columns(pl.lit("reddit").alias("source"))
    pool_texts = set(load_pool()["text"].to_list())
    out = (out.filter(~pl.col("text").is_in(list(pool_texts)))
              .unique("text", keep="first")
              .sample(fraction=1.0, shuffle=True, seed=a.seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(out.group_by("variety", "sarcasm", "dataset").len().sort("variety", "dataset", "sarcasm"))
    print(out.filter(pl.col("variety") == "en-UK")["subreddit"].value_counts(sort=True).head(8))


if __name__ == "__main__":
    main()
