"""Assemble per-fold outputs of ft.py into one cross-fitted run (optionally seed-averaged).

  uv run python experiments/llm/ft_collect.py --run llm_ft_R1pre_cv --members llm_ft_R1pre_s0 llm_ft_R1pre_s1 \
      [--test experiments/llm/fake_test.csv]

Member M's fold k output lives in experiments/llm/out/<M>/fold0.parquet for k = 0 and in
experiments/llm/out/<M>_f<k>/fold<k>.parquet otherwise (ladder.sh naming). OOF: the fold-k rows
come from the adapters trained without fold k, margins averaged over members (seeds). Test: mean
margin over all members x folds, then sigmoid. With ALTA_CANARY_SEED set, save_preds stores the
result as <run>__canary<seed>.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import ROOT, test_split  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import N_FOLDS, load_pool, load_test, save_preds  # noqa: E402

OUT = ROOT / "experiments" / "llm" / "out"


def fold_dir(member: str, k: int):
    return OUT / (member if k == 0 else f"{member}_f{k}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def collect(run: str, members: list[str], test_path: str = "", desc: str = ""):
    pool = load_pool()
    parts = []
    for k in range(N_FOLDS):
        fk = [pl.read_parquet(fold_dir(m, k) / f"fold{k}.parquet") for m in members]
        base = fk[0].select("uid")
        for f in fk[1:]:
            assert f["uid"].to_list() == base["uid"].to_list()
        parts.append(base.with_columns(
            pl.Series("m_sent", np.mean([f["m_sent"].to_numpy() for f in fk], 0)),
            pl.Series("m_sarc", np.mean([f["m_sarc"].to_numpy() for f in fk], 0))))
    oof = pool.select("uid").join(pl.concat(parts), on="uid", how="left")
    assert oof["m_sarc"].null_count() == 0, "missing folds"
    oof = oof.select("uid", pl.col("m_sent").map_batches(sigmoid).alias("p_sent"),
                     pl.col("m_sarc").map_batches(sigmoid).alias("p_sarc"))
    test_df = None
    if test_path:
        test = load_test(test_path)
        tp = [pl.read_parquet(fold_dir(m, k) / f"fold{k}_{test_split(test_path)}.parquet")
              for m in members for k in range(N_FOLDS)]
        for p in tp:
            assert p["uid"].to_list() == test["uid"].to_list()
        test_df = test.select("uid").with_columns(
            pl.Series("p_sent", sigmoid(np.mean([p["m_sent"].to_numpy() for p in tp], 0))),
            pl.Series("p_sarc", sigmoid(np.mean([p["m_sarc"].to_numpy() for p in tp], 0))))
    stats = {}
    for m in members:
        for k in range(N_FOLDS):
            f = fold_dir(m, k) / f"fold{k}_stats.json"
            if f.exists():
                stats[f"{m}/fold{k}"] = json.loads(f.read_text())["args"]
    meta = {"model": "qwen3.5-4b", "kind": "ft", "members": members, "shots": 0,
            "seed": [s.get("seed") for s in stats.values()][:1],
            "prompt_sarc": "sarc_v2", "prompt_sent": "sent_v1", "fold_args": stats,
            "recipe": {"kind": "ft", "members": members},
            "description": desc or f"Qwen3.5-4B bf16 LoRA (r=64, rsLoRA) verbalizer fine-tune, 5-fold cross-fitted, "
                                   f"margins averaged over {len(members)} seed(s); adapters in models/<member>[_f<k>]"}
    save_preds(run, oof, test_df, meta, test_csv=test_path or None,
               folds_used=pool.select("uid", "fold"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--desc", default="")
    a = ap.parse_args()
    collect(a.run, a.members, a.test, a.desc)
    print(json.dumps({"saved": a.run}))
