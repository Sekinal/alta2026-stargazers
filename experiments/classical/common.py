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


THREAD_ENV = "CLS_THREAD_FOLDS"  # diagnostic: set to 1 to regroup the pool's folds by Reddit thread


def thread_folds(df: pl.DataFrame, seed: int = 2026) -> pl.Series:
    """Folds grouped by thread (Reddit link_id = submission id; unmatched rows and google rows are their own
    group), stratified roughly by segment: groups are shuffled within segment and each goes to the fold
    with the fewest rows of that segment so far. Label-free."""
    import os  # noqa: F401
    ctx = pl.read_parquet(ROOT / "data" / "external" / "context" / "pool_context.parquet")
    d = df.select("uid", "seg").join(ctx.select("uid", "matched", "link_id"),
                                      on="uid", how="left", maintain_order="left")
    d = d.with_columns(pl.when(pl.col("matched").fill_null(False) & pl.col("link_id").is_not_null())
                       .then(pl.col("link_id")).otherwise(pl.col("uid")).alias("grp"))
    g = d.group_by("grp", maintain_order=True).agg(pl.col("seg").mode().first(), pl.len().alias("n"))
    rng = np.random.default_rng(seed)
    g = g.with_columns(pl.Series("r", rng.random(g.height))).sort("seg", "r")
    load = {}
    fold_of = {}
    for grp, seg, n in g.select("grp", "seg", "n").iter_rows():
        counts = [load.get((seg, k), 0) for k in range(5)]
        k = int(np.argmin(counts))
        load[(seg, k)] = counts[k] + n
        fold_of[grp] = k
    return pl.Series("fold", [fold_of[x] for x in d["grp"].to_list()], dtype=pl.Int8)


def thread_mode() -> bool:
    import os
    return os.environ.get(THREAD_ENV) == "1"


def pool() -> pl.DataFrame:
    df = frame(load_pool())
    if thread_mode():
        df = df.with_columns(thread_folds(df))
    return df


def save_kw(run: str, df: pl.DataFrame) -> tuple[str, dict]:
    """Run name and folds_used for save_preds: thread-fold diagnostics are saved as RUN__thr, unmeasured."""
    if thread_mode():
        return run + "__thr", {}
    return run, {"folds_used": df.select("uid", "fold")}


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
