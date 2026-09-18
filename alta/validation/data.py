"""Evaluation data: gold labels, segments, folds and a run's OOF probabilities as numpy arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from alta.common import PREDS, VARIETIES, _validate_probs, frame_fingerprint, load_pool

TASKS = ("sent", "sarc")
SOURCES = ("google", "reddit")


@dataclass
class EvalData:
    """Row-aligned arrays for one set of rows (pool OOF or test).

    variety/source are int codes (0 = en-AU / google, 1 = en-UK / reddit).
    y is None for unlabelled (test) rows. p maps task -> probability array.
    """

    uid: np.ndarray
    variety: np.ndarray
    source: np.ndarray
    fold: np.ndarray
    origin: np.ndarray
    n_words: np.ndarray
    p: dict[str, np.ndarray]
    y: dict[str, np.ndarray] | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.uid)

    def subset(self, mask: np.ndarray) -> EvalData:
        return EvalData(
            uid=self.uid[mask], variety=self.variety[mask], source=self.source[mask],
            fold=self.fold[mask], origin=self.origin[mask], n_words=self.n_words[mask],
            p={t: v[mask] for t, v in self.p.items()},
            y=None if self.y is None else {t: v[mask] for t, v in self.y.items()},
            meta=self.meta,
        )

    def with_p(self, p: dict[str, np.ndarray], **meta) -> EvalData:
        return EvalData(self.uid, self.variety, self.source, self.fold, self.origin,
                        self.n_words, p, self.y, {**self.meta, **meta})

    def segment(self) -> np.ndarray:
        """0 AU-google, 1 AU-reddit, 2 UK-google, 3 UK-reddit."""
        return self.variety * 2 + self.source


SEGMENT_NAMES = ("AU-google", "AU-reddit", "UK-google", "UK-reddit")


def _frame_to_eval(df: pl.DataFrame, labelled: bool, meta: dict) -> EvalData:
    df = df.with_columns(
        pl.col("text").str.split(" ").list.len().alias("n_words"),
    )
    return EvalData(
        uid=df["uid"].to_numpy(),
        variety=(df["variety"].cast(pl.String) == VARIETIES[1]).to_numpy().astype(np.int8),
        source=(df["source"].cast(pl.String) == SOURCES[1]).to_numpy().astype(np.int8),
        fold=df["fold"].to_numpy().astype(np.int8) if "fold" in df.columns
        else np.full(df.height, -1, np.int8),
        origin=df["origin"].cast(pl.String).to_numpy(),
        n_words=df["n_words"].to_numpy(),
        p={"sent": df["p_sent"].to_numpy().astype(np.float64),
           "sarc": df["p_sarc"].to_numpy().astype(np.float64)},
        y={"sent": df["sentiment"].to_numpy().astype(np.int8),
           "sarc": df["sarcasm"].to_numpy().astype(np.int8)} if labelled else None,
        meta=meta,
    )


def read_run(run: str) -> tuple[pl.DataFrame, dict]:
    path = PREDS / f"{run}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no predictions for run '{run}' at {path}")
    meta_path = PREDS / f"{run}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pl.read_parquet(path), meta


def load_oof(run: str, pool: pl.DataFrame | None = None) -> EvalData:
    """OOF predictions of a run joined to the pool, in pool order."""
    pool = load_pool() if pool is None else pool
    preds, meta = read_run(run)
    oof = preds.filter(pl.col("split") == "oof").select("uid", "p_sent", "p_sarc")
    _validate_probs(oof, f"{run} oof")
    df = pool.join(oof, on="uid", how="left", maintain_order="left")
    missing = df["p_sent"].null_count() + df["p_sarc"].null_count()
    if missing:
        raise ValueError(f"run '{run}' is missing OOF predictions for {missing} pool cells")
    return _frame_to_eval(df, labelled=True, meta={**meta, "run": run})


def load_test_preds(run: str, test: pl.DataFrame) -> EvalData:
    """Test predictions of a run joined to the test frame, in test (submission) order."""
    preds, meta = read_run(run)
    tp = preds.filter(pl.col("split") == "test").select("uid", "p_sent", "p_sarc")
    if tp.height == 0:
        raise ValueError(f"run '{run}' has no test predictions")
    _validate_probs(tp, f"{run} test")
    want = frame_fingerprint(test)
    if meta.get("test_sha") != want or meta.get("n_test") != test.height:
        raise ValueError(f"run '{run}' test predictions were made for a different test file "
                         f"(sha {meta.get('test_sha')}, n {meta.get('n_test')}) than this one "
                         f"(sha {want}, n {test.height}); re-run its predict_test")
    if set(tp["uid"].to_list()) != set(test["uid"].to_list()):
        raise ValueError(f"run '{run}' test uids do not match the test file")
    df = test.join(tp, on="uid", how="left", maintain_order="left")
    return _frame_to_eval(df, labelled=False, meta={**meta, "run": run})


def pool_eval(pool: pl.DataFrame | None = None) -> EvalData:
    """Pool with dummy probabilities (for label-only analyses and tests)."""
    pool = load_pool() if pool is None else pool
    df = pool.with_columns(pl.lit(0.5).alias("p_sent"), pl.lit(0.5).alias("p_sarc"))
    return _frame_to_eval(df, labelled=True, meta={})


def list_runs() -> list[str]:
    return sorted(p.stem for p in Path(PREDS).glob("*.parquet"))
