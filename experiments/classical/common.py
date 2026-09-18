"""Shared helpers for experiments/classical: pool arrays, text normalisation, quick honest scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import TASKS, pool_eval  # noqa: E402
from alta.validation.metrics import components  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
LABEL = {"sent": "sentiment", "sarc": "sarcasm"}


def normalise(s: pl.Expr) -> pl.Expr:
    """Lower-case, unify quotes, placeholder URLs/users/subreddits, re-join google tokeniser splits."""
    return (
        s.str.replace_all(r"[‘’ʼ`]", "'")
        .str.replace_all(r"[“”]", '"')
        .str.replace_all(r"https?://\S+|www\.\S+", " xxurl ")
        .str.replace_all(r"(?:^|\s)/?u/[A-Za-z0-9_-]+", " xxuser ")
        .str.replace_all(r"(?:^|\s)/?r/[A-Za-z0-9_]+", " xxsubreddit ")
        .str.replace_all(r"(\w) n't\b", "${1}n't")
        .str.replace_all(r"(\w) '(s|re|ve|ll|d|m)\b", "${1}'${2}")
        .str.to_lowercase()
    )


def frame(df: pl.DataFrame) -> pl.DataFrame:
    """Adds norm text and integer segment codes (variety*2 + source; 0 AU-google .. 3 UK-reddit)."""
    return df.with_columns(
        normalise(pl.col("text")).alias("norm"),
        ((pl.col("variety").cast(pl.String) == "en-UK").cast(pl.Int8) * 2
         + (pl.col("source").cast(pl.String) == "reddit").cast(pl.Int8)).alias("seg"),
    )


def pool() -> pl.DataFrame:
    return frame(load_pool())


_EVAL = None


def quick(p_sent: np.ndarray, p_sarc: np.ndarray, specs=("fixed+ukg", "prevtuned+ukg")) -> dict:
    """Honest (cross-fitted policy) scores for pool-ordered OOF probabilities."""
    global _EVAL
    if _EVAL is None:
        _EVAL = pool_eval()
    d = _EVAL.with_p({"sent": np.asarray(p_sent, float), "sarc": np.asarray(p_sarc, float)})
    out = {}
    for spec in specs:
        pred, _ = policies.crossfit(spec, d)
        c = components(d, pred)
        out[spec] = {k: round(float(v), 4) for k, v in c.items()}
    return out


def fmt(res: dict) -> str:
    return " | ".join(f"{s}: {r['score']:.4f} (sAU {r['sent-AU']:.3f} sUK {r['sent-UK']:.3f} "
                      f"cAU {r['sarc-AU']:.3f} cUK {r['sarc-UK']:.3f})" for s, r in res.items())
