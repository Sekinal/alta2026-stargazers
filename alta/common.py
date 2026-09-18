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


SOURCES = ("google", "reddit")


def _read(path: Path, origin: str) -> pl.DataFrame:
    """uid = origin initial + zero-padded row number, e.g. t00042 / v00007 / x00123.

    Labels are read as nullable Int64 so a test file with empty label columns parses; unknown
    source/variety values raise instead of silently mapping to a default segment.
    """
    tag = {"train": "t", "valid": "v", "test": "x"}[origin]
    df = pl.read_csv(path, schema_overrides={"source": pl.String, "variety": pl.String, "text": pl.String,
                                             "sentiment": pl.Int64, "sarcasm": pl.Int64})
    missing = {"source", "variety", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if any("\r" in c for c in df.columns):
        raise ValueError(f"{path}: CR-only line endings (old Mac format); convert to LF/CRLF first")
    blank = df.select(pl.all_horizontal(pl.all().is_null())).to_series()
    if blank.any():  # blank lines: pandas skips them, polars yields all-null rows
        import warnings
        warnings.warn(f"{path}: dropped {int(blank.sum())} blank line(s)")
        df = df.filter(~blank)
    if df.height == 0:
        raise ValueError(f"{path}: no rows")
    if df["source"].null_count() or df["variety"].null_count():
        raise ValueError(f"{path}: null source/variety values")
    if df.height >= 100_000:
        raise ValueError(f"{path}: {df.height} rows exceeds 5-digit uids")
    bad_src = set(df["source"].unique().to_list()) - set(SOURCES)
    bad_var = set(df["variety"].unique().to_list()) - set(VARIETIES)
    if bad_src or bad_var:
        raise ValueError(f"{path}: unknown source {bad_src or '-'} / variety {bad_var or '-'}")
    if df["text"].null_count():
        df = df.with_columns(pl.col("text").fill_null(""))
    return df.with_row_index("row").with_columns(
        pl.lit(origin).alias("origin"),
        (pl.lit(tag) + pl.col("row").cast(pl.String).str.zfill(5)).alias("uid"),
    )


def frame_fingerprint(df: pl.DataFrame) -> str:
    """Order-sensitive hash of (source, variety, text) rows: binds predictions to one exact file."""
    rows = df.select(pl.concat_str(pl.col("source").cast(pl.String), pl.col("variety").cast(pl.String),
                                   pl.col("text"), separator="\x1f")).to_series().to_list()
    return hashlib.sha1("\x1e".join(rows).encode()).hexdigest()[:16]


def _csv_sha(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


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
    src_sha = _csv_sha(DATA / "train.csv", DATA / "valid.csv")
    stamp = CACHE.with_suffix(".sha")
    if CACHE.exists() and not refresh and stamp.exists() and stamp.read_text() == src_sha:
        return pl.read_parquet(CACHE)
    pool = pl.concat([_read(DATA / "train.csv", "train"), _read(DATA / "valid.csv", "valid")])
    pool = _assign_folds(pool).drop("row").with_columns(
        pl.col("source", "variety", "origin").cast(pl.Categorical),
        pl.col("sentiment", "sarcasm", "fold").cast(pl.Int8),
    )
    pool.write_parquet(CACHE)
    stamp.write_text(src_sha)
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


def _validate_probs(df: pl.DataFrame, what: str) -> None:
    if df["uid"].n_unique() != df.height:
        raise ValueError(f"{what}: duplicated uids")
    for c in ("p_sent", "p_sarc"):
        x = df[c].cast(pl.Float64).to_numpy()
        if df[c].null_count() or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f"{what}.{c}: probabilities must be finite and in [0, 1]")


def save_preds(run: str, oof: pl.DataFrame, test: pl.DataFrame | None = None,
               meta: dict | None = None, test_csv: str | Path | None = None) -> Path:
    """oof/test: frames with uid, p_sent, p_sarc (probability of the positive class).

    Test predictions must name the file they were made for (``test_csv``): its fingerprint and
    row count are stored so a submission can never pair them with a different/reordered file.
    """
    PREDS.mkdir(exist_ok=True)
    cols = ["uid", "p_sent", "p_sarc"]
    _validate_probs(oof.select(cols), f"{run} oof")
    parts = [oof.select(cols).with_columns(pl.lit("oof").alias("split"))]
    test_meta = {}
    if test is not None:
        if test_csv is None:
            raise ValueError("save_preds: pass test_csv=<path of the test CSV> with test predictions")
        tframe = load_test(test_csv)
        _validate_probs(test.select(cols), f"{run} test")
        if set(test["uid"].to_list()) != set(tframe["uid"].to_list()):
            raise ValueError(f"{run}: test uids do not match the {tframe.height} rows of {test_csv}")
        test_meta = {"test_csv": str(test_csv), "test_sha": frame_fingerprint(tframe), "n_test": tframe.height}
        parts.append(test.select(cols).with_columns(pl.lit("test").alias("split")))
    out = pl.concat(parts).with_columns(pl.col("p_sent", "p_sarc").cast(pl.Float64))
    path = PREDS / f"{run}.parquet"
    out.write_parquet(path)
    meta = {**(meta or {}), **test_meta, "run": run, "fold_hash": fold_hash(), "git": git_commit()}
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
    if df.height != pool.height:
        raise ValueError("OOF predictions must cover every pool row")
    return {"pool": score(df), "valid": score(df.filter(pl.col("origin") == "valid"))}


if __name__ == "__main__":
    pool = load_pool(refresh=True)
    print(pool.group_by("fold").agg(pl.len(), pl.col("sarcasm").mean(), pl.col("sentiment").mean()).sort("fold"))
    print(pool.group_by("fold", "variety").agg(pl.col("sarcasm").sum()).sort("fold", "variety"))
    assert pool["uid"].n_unique() == pool.height
