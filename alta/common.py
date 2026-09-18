"""Shared data, CV folds, metric and prediction I/O for all ALTA 2026 experiments.

Protocol
--------
* ``load_pool()`` returns train.csv + valid.csv concatenated (6288 rows) with a stable
  ``uid`` column and a ``fold`` column (0..4). Folds are stratified on
  variety x source x sentiment x sarcasm and exact-duplicate texts share a fold.
* Every model writes ONE parquet per run to ``preds/<run_name>.parquet`` via
  ``save_preds``. It contains out-of-fold probabilities for every pool row
  (``split == "oof"``) and, optionally, probabilities for the test rows (``split == "test"``).
  Test probabilities must be the AVERAGE OF THE 5 FOLD MODELS, not a refit on the full pool:
  red-team pseudo-leaderboards found fold-averaging ties refit for linear probes and beats it by
  ~0.009 (with half the seed variance) for seed-dependent models. Columns: uid, split, p_sent, p_sarc.
* ``score`` reproduces the official metric (verified against scoring/evaluate.py).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDS = ROOT / "preds"
N_FOLDS = 5
SEED = 2026
VARIETIES = ("en-AU", "en-UK")


CACHE = DATA / "pool.parquet"


def _read(path: Path, origin: str) -> pl.DataFrame:
    """uid = origin initial + zero-padded row number, e.g. t00042 / v00007 / x00123."""
    tag = {"train": "t", "valid": "v", "test": "x"}[origin]
    return (
        pl.read_csv(path)
        .with_row_index("row")
        .with_columns(
            pl.lit(origin).alias("origin"),
            (pl.lit(tag) + pl.col("row").cast(pl.String).str.zfill(5)).alias("uid"),
        )
    )


def _assign_folds(df: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    """Stratified group folds: groups = identical text, strata = source/variety/labels."""
    rng = np.random.default_rng(seed)
    groups = (
        df.group_by("text", maintain_order=True)
        .agg(
            pl.concat_str(
                pl.col("variety").first(),
                pl.col("source").first(),
                pl.col("sentiment").max().cast(pl.String),
                pl.col("sarcasm").max().cast(pl.String),
                separator="|",
            ).alias("stratum")
        )
        .with_columns(pl.Series("r", rng.random(df["text"].n_unique())))
        .sort("stratum", "r")
    )
    # Round-robin within each stratum after shuffling.
    groups = groups.with_columns(
        (pl.int_range(pl.len()).over("stratum") % N_FOLDS).alias("fold")
    )
    return df.join(groups.select("text", "fold"), on="text", how="left")


def load_pool(refresh: bool = False) -> pl.DataFrame:
    """train + valid with uid, origin ('train'|'valid') and fold columns (parquet-cached)."""
    if CACHE.exists() and not refresh:
        return pl.read_parquet(CACHE)
    pool = pl.concat([_read(DATA / "train.csv", "train"), _read(DATA / "valid.csv", "valid")])
    pool = _assign_folds(pool).drop("row").with_columns(
        pl.col("source", "variety", "origin").cast(pl.Categorical),
        pl.col("sentiment", "sarcasm", "fold").cast(pl.Int8),
    )
    pool.write_parquet(CACHE)
    return pool


def fold_hash(pool: pl.DataFrame | None = None) -> str:
    """Fingerprint of the uid -> fold assignment; stored with every run to detect fold drift."""
    pool = load_pool() if pool is None else pool
    s = pool.sort("uid").select(pl.concat_str("uid", pl.col("fold").cast(pl.String), separator=":"))
    return hashlib.sha1("|".join(s.to_series().to_list()).encode()).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_test(path: str | Path) -> pl.DataFrame:
    """Unlabelled test CSV, row order preserved (the scorer relies on it)."""
    return _read(Path(path), "test").drop("row")


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def score(df: pl.DataFrame, sent: str = "sentiment_pred", sarc: str = "sarcasm_pred",
          gold_sent: str = "sentiment", gold_sarc: str = "sarcasm") -> dict[str, float]:
    """Official metric on a frame holding gold + hard predictions."""
    out: dict[str, float] = {}
    for task, p, g in (("sent", sent, gold_sent), ("sarc", sarc, gold_sarc)):
        for v in VARIETIES:
            d = df.filter(pl.col("variety") == v)
            out[f"{task}-{v[3:]}"] = macro_f1(d[g].to_numpy(), d[p].to_numpy())
    out["score"] = (min(out["sent-AU"], out["sent-UK"]) + min(out["sarc-AU"], out["sarc-UK"])) / 2
    return out


def save_preds(run: str, oof: pl.DataFrame, test: pl.DataFrame | None = None,
               meta: dict | None = None) -> Path:
    """oof/test: frames with uid, p_sent, p_sarc (probability of the positive class)."""
    PREDS.mkdir(exist_ok=True)
    cols = ["uid", "p_sent", "p_sarc"]
    parts = [oof.select(cols).with_columns(pl.lit("oof").alias("split"))]
    if test is not None:
        parts.append(test.select(cols).with_columns(pl.lit("test").alias("split")))
    out = pl.concat(parts).with_columns(pl.col("p_sent", "p_sarc").cast(pl.Float64))
    path = PREDS / f"{run}.parquet"
    out.write_parquet(path)
    meta = {**(meta or {}), "run": run, "fold_hash": fold_hash(), "git": git_commit()}
    (PREDS / f"{run}.json").write_text(json.dumps(meta, indent=2))
    return path


def evaluate_oof(run: str, thr_sent: float = 0.5, thr_sarc: float = 0.5) -> dict[str, dict]:
    """Score a saved run on the full OOF pool and on the valid.csv subset, fixed thresholds."""
    pool = load_pool()
    p = pl.read_parquet(PREDS / f"{run}.parquet").filter(pl.col("split") == "oof")
    df = pool.join(p, on="uid", how="inner").with_columns(
        (pl.col("p_sent") >= thr_sent).cast(pl.Int64).alias("sentiment_pred"),
        (pl.col("p_sarc") >= thr_sarc).cast(pl.Int64).alias("sarcasm_pred"),
    )
    assert df.height == pool.height, "OOF predictions must cover every pool row"
    return {"pool": score(df), "valid": score(df.filter(pl.col("origin") == "valid"))}


if __name__ == "__main__":
    pool = load_pool(refresh=True)
    print(pool.group_by("fold").agg(pl.len(), pl.col("sarcasm").mean(), pl.col("sentiment").mean()).sort("fold"))
    print(pool.group_by("fold", "variety").agg(pl.col("sarcasm").sum()).sort("fold", "variety"))
    assert pool["uid"].n_unique() == pool.height
