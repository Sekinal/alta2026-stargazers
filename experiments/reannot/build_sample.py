"""Build the blind re-annotation sample (experiments/prereg/reannotation.md).

Writes (git-ignored, contain raw text) experiments/reannot/work/batch_<i>.jsonl with {"item", "variety", "text"},
and (no text, committed) experiments/reannot/sample.csv with item -> uid, stratum, gold, lead probability and the
lead's segment+ukg decision.

  uv run python experiments/reannot/build_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import PREDS, load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import pool_eval  # noqa: E402

HERE = Path(__file__).resolve().parent
LEAD = "stk_v5r"


def main() -> None:
    pool = load_pool()
    oof = pool.select("uid").join(pl.read_parquet(PREDS / f"{LEAD}.parquet").filter(pl.col("split") == "oof"),
                                  on="uid", how="left", maintain_order="left")
    d = pool_eval().with_p({"sent": oof["p_sent"].to_numpy(), "sarc": oof["p_sarc"].to_numpy()})
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    df = pool.with_columns(oof["p_sent"], oof["p_sarc"], pl.Series("lead_sent", pred["sent"]),
                           pl.Series("lead_sarc", pred["sarc"]))
    rng = np.random.default_rng(2026)
    parts = []
    for v in ("en-AU", "en-UK"):
        r = df.filter((pl.col("variety") == v) & (pl.col("source") == "reddit"))
        u_idx = rng.choice(r.height, 150, replace=False)
        U = r[u_idx].with_columns(pl.lit("U").alias("stratum"))
        rest = r.filter(~pl.col("uid").is_in(U["uid"].to_list()))
        dis = rest.filter((pl.col("p_sarc") - pl.col("sarcasm")).abs() > 0.8)
        D = dis[rng.choice(dis.height, min(50, dis.height), replace=False)].with_columns(pl.lit("D").alias("stratum"))
        print(v, "D pool size", dis.height)
        parts += [U, D]
    s = pl.concat(parts)
    s = s[rng.permutation(s.height)].with_row_index("item")
    s = s.with_columns(pl.format("i{}", pl.col("item")).alias("item"))
    s.select("item", "uid", "variety", "stratum", "sentiment", "sarcasm", "p_sent", "p_sarc", "lead_sent", "lead_sarc") \
        .write_csv(HERE / "sample.csv")
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    for b in range(0, s.height, 100):
        with open(work / f"batch_{b // 100}.jsonl", "w") as f:
            for it, v, t in s[b:b + 100].select("item", "variety", "text").iter_rows():
                f.write(json.dumps({"item": it, "variety": v, "text": t}) + "\n")
    print(s.group_by("variety", "stratum").len().sort("variety", "stratum"))


if __name__ == "__main__":
    main()
