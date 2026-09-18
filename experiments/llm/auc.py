"""Per-segment AUCs for saved runs or raw cache tags (works on partial caches too).

  uv run python experiments/llm/auc.py run emb_ens3_lr llm_q8_zs_v1
  uv run python experiments/llm/auc.py tag qwen3-8b/sarc_v1 qwen3-8b/sent_v1
"""

from __future__ import annotations

import sys

import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import CACHE, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import PREDS, load_pool  # noqa: E402

SEGS = [("en-AU", "reddit"), ("en-UK", "reddit"), ("en-AU", "google"), ("en-UK", "google")]


def seg_aucs(df: pl.DataFrame, cols: dict[str, str]) -> dict[str, float]:
    out = {}
    for task, (col, gold) in cols.items():
        for v, s in SEGS:
            d = df.filter((pl.col("variety") == v) & (pl.col("source") == s))
            if d[gold].n_unique() < 2 or d[gold].sum() < 5:
                continue
            out[f"{task}-{v[3:]}-{s[:3]}"] = roc_auc_score(d[gold], d[col])
    return out


def main():
    mode, names = sys.argv[1], sys.argv[2:]
    pool = load_pool()
    rows = []
    for name in names:
        if mode == "run":
            p = pl.read_parquet(PREDS / f"{name}.parquet").filter(pl.col("split") == "oof")
            df = pool.join(p, on="uid")
            r = seg_aucs(df, {"sarc": ("p_sarc", "sarcasm"), "sent": ("p_sent", "sentiment")})
        else:
            parts = sorted((CACHE / name / "pool").glob("part_*.parquet"))
            df = pool.join(pl.concat([pl.read_parquet(x) for x in parts]), on="uid")
            task = "sarc" if "/sarc" in name else "sent"
            gold = "sarcasm" if task == "sarc" else "sentiment"
            r = seg_aucs(df, {task: ("prob", gold)})
            r["n"] = df.height
        rows.append({"name": name, **{k: round(v, 3) for k, v in r.items()}})
    with pl.Config(tbl_cols=30, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
