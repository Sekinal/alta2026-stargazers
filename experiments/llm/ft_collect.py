"""Assemble the 5 fold outputs of ft.py into a saved run.

  uv run python experiments/llm/ft_collect.py --run llm_q8_ft [--test PATH]

OOF: fold k rows come from the adapter trained without fold k. Test: mean of the 5 fold margins,
then sigmoid. en-UK google sarcasm (not trained on) keeps the model's score, which the ukg rule
overrides anyway.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import CACHE, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import N_FOLDS, load_pool, load_test, save_preds  # noqa: E402

MODEL_DIR = ROOT / "models"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def collect(run: str, test_path: str = ""):
    d = ROOT / "experiments" / "llm" / "out" / run
    pool = load_pool()
    oof = pl.concat([pl.read_parquet(d / f"fold{k}.parquet") for k in range(N_FOLDS)])
    oof = pool.select("uid").join(oof, on="uid", how="left")
    assert oof["m_sarc"].null_count() == 0, "missing folds"
    oof = oof.select("uid", pl.col("m_sent").map_batches(sigmoid).alias("p_sent"),
                     pl.col("m_sarc").map_batches(sigmoid).alias("p_sarc"))
    test_df = None
    if test_path:
        test = load_test(test_path)
        parts = [pl.read_parquet(d / f"fold{k}_test.parquet") for k in range(N_FOLDS)]
        for p in parts:
            assert p["uid"].to_list() == test["uid"].to_list()
        ms = np.mean([p["m_sent"].to_numpy() for p in parts], 0)
        mc = np.mean([p["m_sarc"].to_numpy() for p in parts], 0)
        test_df = test.select("uid").with_columns(pl.Series("p_sent", sigmoid(ms)),
                                                   pl.Series("p_sarc", sigmoid(mc)))
    args = json.loads((MODEL_DIR / run / "fold0" / "train_args.json").read_text())
    meta = {"model": args["model"], "kind": "ft", "prompt_sarc": "sarc_v2", "prompt_sent": "sent_v1",
            "shots": 0, "seed": args["seed"], "train_args": args,
            "recipe": {"kind": "ft", "model": args["model"]},
            "description": f"QLoRA verbalizer fine-tune, 5-fold cross-fitted; adapters in models/{run}"}
    save_preds(run, oof, test_df, meta, test_csv=test_path or None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", default="")
    a = ap.parse_args()
    collect(a.run, a.test)
    print(json.dumps({"saved": a.run}))
