"""Shared output schema, /s stripping and writer for external sarcasm corpora.

Every dataset lands in data/external/sarcasm/<dataset>.parquet (git-ignored; not ours to redistribute)
with the columns in SCHEMA, deduplicated against the labelled pool, and gets an entry in
data/external/sarcasm/MANIFEST.json.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "sarcasm"
RAW = ROOT / "data" / "external" / "raw"
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = {
    "text": pl.String, "sarcasm": pl.Int8, "sentiment": pl.Int8, "source": pl.String,
    "subreddit": pl.String, "variety_guess": pl.String, "dataset": pl.String, "license": pl.String,
    "label_quality": pl.String, "parent_text": pl.String, "year": pl.Int16,
}

# "/s" as a sarcasm tag: standalone token, optionally bracketed, not a path/URL/unit ("w/s", "m/s", "/sarcasm" ok)
_S_TAG = re.compile(r"(?i)(?<![\w/.:])[(\[]?\s*/(?:s|sarc|sarcasm)\b[)\]]?(?![\w/])")
# /s referred to rather than used: "forgot the /s", "no /s", "is this /s?", "needs a /s"
_S_META = re.compile(r"(?i)\b(?:no|not|without|forgot|forget|forgetting|need|needs|needed|missing|add|added|"
                     r"adding|put|use|using|drop|dropped|implied|obvious|include|included)\s+(?:the\s+|a\s+|an\s+|your\s+|my\s+)?"
                     r"[(\[]?/s\b|/s\s*\?|/s\s+(?:tag|needed|is|was)\b")


def has_s_tag(text: str) -> bool:
    return bool(_S_TAG.search(text or ""))


_TAG = r"(?<![\w/.:])/(?:s|sarc|sarcasm)\b"
# tag at the end of a line, optionally bracketed, followed only by punctuation/emoji
_S_END = re.compile(r"(?mi)[(\[]?\s*" + _TAG + r"[)\]]?[^\w\n]*$")
# "(/s obviously)", "(/s if it isn't clear)" closing a line
_S_PAREN = re.compile(r"(?mi)[(\[]\s*" + _TAG + r"[^)\]\n]{0,40}[)\]][^\w\n]*$")
# a short line that starts with the tag: "/s - if it isn't obvious."
_S_LINE = re.compile(r"(?mi)^\s*[(\[]?\s*" + _TAG + r"[^\n]{0,40}$")


def strip_s_tag(text: str) -> str | None:
    """Remove a closing /s tag; None if /s is talked about, used mid-sentence, or only quoted."""
    if not text or _S_META.search(text):
        return None
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(">"))
    body = re.sub(r"\^+\(([^)]*)\)", r"\1", body).replace("^", "")  # reddit superscript
    out = _S_PAREN.sub("", body)
    out = _S_END.sub("", out)
    out = _S_LINE.sub("", out)
    if out == body or _S_TAG.search(out):  # no closing tag, or another /s left mid-text
        return None
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    out = re.sub(r"\s+([.!?,])$", r"\1", out)
    return out or None


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        req = urllib.request.Request(url, headers={"User-Agent": "alta-research/0.1"})
        with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return dest


def finalize(df: pl.DataFrame, **const) -> pl.DataFrame:
    """Add constant columns, fill missing schema columns with nulls, cast and order."""
    for k, v in const.items():
        df = df.with_columns(pl.lit(v).alias(k))
    for k, t in SCHEMA.items():
        if k not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=t).alias(k))
    df = df.select([pl.col(k).cast(t) for k, t in SCHEMA.items()])
    df = df.with_columns(pl.col("text").str.strip_chars(), pl.col("parent_text").str.strip_chars())
    return df.filter(pl.col("text").str.len_chars() > 0, pl.col("sarcasm").is_not_null())


def write(df: pl.DataFrame, name: str, *, url: str, license: str, notes: str = "",
          matcher=None, dedupe_within: bool = True) -> dict:
    """Dedupe against the pool, write parquet, update MANIFEST.json; returns the manifest entry."""
    from dedupe import Matcher, dedupe_frame, reference_texts

    OUT.mkdir(parents=True, exist_ok=True)
    n0 = df.height
    if dedupe_within:  # identical text with conflicting labels: keep the majority label, one copy
        df = (df.with_columns(pl.col("sarcasm").mean().over("text").alias("_m"))
              .filter((pl.col("_m") != 0.5))
              .with_columns((pl.col("_m") > 0.5).cast(pl.Int8).alias("sarcasm"))
              .unique("text", keep="first", maintain_order=True).drop("_m"))
    n_within = n0 - df.height
    matcher = matcher or Matcher(reference_texts())
    df, stats = dedupe_frame(df, matcher)
    df.write_parquet(OUT / f"{name}.parquet")
    entry = {
        "dataset": name, "rows": df.height, "positives": int(df["sarcasm"].sum()),
        "license": license, "url": url,
        "dedupe_pool_drops": stats["dropped"], "dedupe_pool_drops_pos": stats["dropped_pos"],
        "dedupe_parent_hits": stats.get("parent_hits"), "within_dup_or_conflict_drops": n_within,
        "sources": df["source"].value_counts().sort("count", descending=True).rows(),
        "variety": df["variety_guess"].value_counts().sort("count", descending=True).rows(),
        "label_quality": df["label_quality"].unique().to_list(),
        "has_parent": int(df["parent_text"].is_not_null().sum()),
        "has_sentiment": int(df["sentiment"].is_not_null().sum()),
        "notes": notes,
    }
    man = OUT / "MANIFEST.json"
    m = json.loads(man.read_text()) if man.exists() else {}
    m[name] = entry
    man.write_text(json.dumps(m, indent=1, default=str))
    print(json.dumps({k: entry[k] for k in ("dataset", "rows", "positives", "dedupe_pool_drops")}), flush=True)
    return entry
