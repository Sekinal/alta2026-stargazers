"""Shared cleaning / filtering for the unlabelled in-domain pretraining corpus (reddit + reviews).

Output schema (one parquet per variety x source):
    text, variety, source, community, created_utc, score, parent_id, link_id, dataset, license, kind
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl

SOURCES = {  # recorded in MANIFEST.json
    "arctic_shift": {"url": "https://arctic-shift.photon-reddit.com/api (comments/search, posts/search)",
                     "license": "Reddit User Agreement / Data API terms; research use, not redistributed",
                     "time_range": "2022-01-01..2025-12-31", "variety": "en-AU, en-UK (by subreddit)"},
    "googlelocal": {"url": "https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal/",
                    "license": "UCSD McAuley lab, research/non-commercial use",
                    "time_range": "~2007..2014", "variety": "place address ends in United Kingdom/UK/Australia"},
    "booking": {"url": "https://huggingface.co/datasets/Booking-com/accommodation-reviews",
                "license": "CC BY-SA 4.0 (README: non-commercial)", "time_range": "2023",
                "variety": "accommodation_country"},
    "nhs_gp": {"url": "https://huggingface.co/datasets/janduplessis886/england-nhs-gp-reviews",
               "license": "Open Government Licence v3.0 (NHS.uk)", "time_range": "2022..2024", "variety": "en-UK"},
    "airbnb": {"url": "https://insideairbnb.com/get-the-data/ (Sydney, Melbourne reviews.csv.gz, 2026-06-16)",
               "license": "CC BY 4.0", "time_range": "2022-01-01..2026-06 (sampled 250k)", "variety": "en-AU"},
}
MAX_WORDS = 512
MIN_WORDS = 5
N_PROC = int(os.environ.get("INDOMAIN_PROCS", "6"))
SCHEMA = {
    "text": pl.String, "variety": pl.String, "source": pl.String, "community": pl.String,
    "created_utc": pl.Int64, "score": pl.Int64, "parent_id": pl.String, "link_id": pl.String,
    "dataset": pl.String, "license": pl.String, "kind": pl.String,
}
BOT_AUTHOR = r"(?i)^(automoderator|.*bot|bot.*|.*_bot_.*|reddit-?.*|sneakpeekbot|remindmebot|b0trank)$"
BOT_TEXT = r"(?i)(i am a bot|i'm a bot|this action was performed automatically|beep boop|\^\^i'm \^\^a \^\^bot)"


def normalise(col: pl.Expr) -> pl.Expr:
    """Dedupe key: lowercase, alphanumerics only, whitespace collapsed (shared with dedupe_indomain)."""
    return (col.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}\s]", "")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def clean_text(col: pl.Expr, reddit: bool) -> pl.Expr:
    """Unescape HTML, strip markdown links/URLs/quote lines, collapse whitespace. Keeps '/s'."""
    t = (col.str.replace_all("&amp;", "&", literal=True).str.replace_all("&lt;", "<", literal=True)
         .str.replace_all("&gt;", ">", literal=True).str.replace_all("&quot;", '"', literal=True)
         .str.replace_all("&#39;", "'", literal=True).str.replace_all("&#x27;", "'", literal=True)
         .str.replace_all(r"&#x200[bB];|&nbsp;|\u200b", " ")
         .str.replace_all(r"(?i)<br\s*/?>", "\n"))
    if reddit:
        t = (t.str.replace_all(r"(?m)^\s*>.*$", "")                   # quoted parent text
             .str.replace_all(r"!\[[^\]]*\]\([^)]*\)", "")              # images
             .str.replace_all(r"\[([^\]]*)\]\([^)]*\)", "${1}")         # [text](url) -> text
             .str.replace_all(r"(?m)^#{1,6}\s*", "")                    # headers
             .str.replace_all(r"\*\*|__|~~|(?m)^\s*[*-]\s+", "")        # bold / strike / bullets
             .str.replace_all(r"\\([_*\[\]()~>#-])", "${1}"))           # markdown escapes
    t = (t.str.replace_all(r"https?://\S+|www\.\S+", "")
         .str.replace_all(r"[ \t\r\f\v]+", " ").str.replace_all(r" ?\n ?", "\n")
         .str.replace_all(r"\n{3,}", "\n\n").str.strip_chars())
    return t


def cap_words(df: pl.DataFrame, max_words: int = MAX_WORDS) -> pl.DataFrame:
    """Truncate texts longer than max_words (whitespace tokens); short texts keep their newlines."""
    n = pl.col("text").str.count_matches(r"\S+")
    long = pl.col("text").str.split(" ").list.head(max_words).list.join(" ")
    # split on single spaces keeps newlines inside tokens; approximate but cheap
    return df.with_columns(pl.when(n > max_words).then(long).otherwise(pl.col("text")).alias("text"))


def _langid_chunk(texts: list[str]) -> list[str]:
    import py3langid as langid
    return [langid.classify(t[:1000])[0] if t else "" for t in texts]


def english_mask(texts: pl.Series, procs: int = N_PROC) -> pl.Series:
    """py3langid label == 'en' (parallel over processes)."""
    vals = texts.to_list()
    step = max(1, len(vals) // (procs * 8) + 1)
    chunks = [vals[i:i + step] for i in range(0, len(vals), step)]
    with ProcessPoolExecutor(procs) as ex:
        labels = [x for part in ex.map(_langid_chunk, chunks) for x in part]
    return pl.Series(labels) == "en"


def finalise(df: pl.DataFrame, check_lang: bool = True) -> tuple[pl.DataFrame, dict]:
    """Common tail: min/max length, English, exact within-corpus dedupe. Returns (df, stats)."""
    stats = {"in": df.height}
    df = df.filter(pl.col("text").is_not_null())
    df = df.filter(pl.col("text").str.count_matches(r"\S+") >= MIN_WORDS)
    stats["after_min_words"] = df.height
    df = cap_words(df)
    if check_lang and df.height:
        df = df.filter(english_mask(df["text"]))
    stats["after_lang"] = df.height
    df = df.with_columns(normalise(pl.col("text")).alias("_norm"))
    df = df.filter(pl.col("_norm").str.len_chars() > 0).unique("_norm", keep="first", maintain_order=True)
    stats["after_exact_dedupe"] = df.height
    return df.drop("_norm").select([pl.col(c).cast(t) for c, t in SCHEMA.items()]), stats


def approx_tokens(df: pl.DataFrame) -> int:
    """~1.3 subword tokens per whitespace word (deberta-v3 / qwen BPE on English web text)."""
    return int(df.select(pl.col("text").str.count_matches(r"\S+").sum()).item() * 1.3)
