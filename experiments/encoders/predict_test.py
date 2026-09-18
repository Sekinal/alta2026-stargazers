"""Test-set predictions for a kept encoder run: average the 5 fold checkpoints' probabilities.

Loads models/<run>/fold{k}/, predicts every row of the test CSV (columns source, variety,
text; label columns empty) and re-saves preds/<run>.parquet with the existing OOF rows plus
split == "test" rows keyed by alta.common.load_test uids. Several runs can be given; each is
processed and saved separately.

  flock .gpu.lock uv run python -m experiments.encoders.predict_test --run enc_x --test data/test.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import json
import sys

import numpy as np
import polars as pl
import torch

from alta.common import N_FOLDS, PREDS, load_pool, load_test, save_preds
from experiments.encoders.common import MODELS, load_fold, predict


def predict_run(run: str, test: pl.DataFrame, test_csv: str) -> None:
    ps, pc = np.zeros(test.height), np.zeros(test.height)
    for k in range(N_FOLDS):
        model, tok, cfg = load_fold(MODELS / run / f"fold{k}")
        a, b = predict(model, tok, test, cfg["max_len"], not cfg.get("no_prefix", False))
        ps += a / N_FOLDS
        pc += b / N_FOLDS
        del model
        torch.cuda.empty_cache()
        print(f"{run} fold{k}: mean p_sent={a.mean():.3f} p_sarc={b.mean():.3f}", flush=True)
    old = pl.read_parquet(PREDS / f"{run}.parquet")
    oof = old.filter(pl.col("split") == "oof")
    meta = json.loads((PREDS / f"{run}.json").read_text())
    meta = {k: v for k, v in meta.items() if k not in ("run", "fold_hash", "git") and not k.startswith("test")}
    meta["test_rows"] = test.height
    tdf = test.select("uid").with_columns(pl.Series("p_sent", ps), pl.Series("p_sarc", pc))
    print(save_preds(run, oof, tdf, meta=meta, test_csv=test_csv,
                     folds_used=load_pool().select("uid", "fold")))  # train.py uses the pool folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    test = load_test(a.test)
    missing = {"source", "variety", "text"} - set(test.columns)
    if missing:
        print(f"test CSV lacks {missing}", file=sys.stderr)
        return 1
    for run in a.run:
        predict_run(run, test, a.test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
