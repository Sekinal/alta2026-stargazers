"""Blind batches of ALL Reddit rows of one variety for the reader feature (experiments/prereg/reader_feature.md).

  uv run python experiments/reannot/build_full.py --variety en-UK [--test data/test.csv]
Writes experiments/reannot/work/full_<split>_<variety>_<i>.jsonl (git-ignored: raw text) with item = uid.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool, load_test  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variety", required=True)
ap.add_argument("--test", default="")
a = ap.parse_args()
df = load_test(a.test) if a.test else load_pool()
split = "test" if a.test else "pool"
r = df.filter((pl.col("variety") == a.variety) & (pl.col("source") == "reddit"))
r = r[np.random.default_rng(7).permutation(r.height)]
out = ROOT / "experiments" / "reannot" / "work"
out.mkdir(exist_ok=True)
for b in range(0, r.height, 100):
    with open(out / f"full_{split}_{a.variety}_{b // 100}.jsonl", "w") as f:
        for u, v, t in r[b:b + 100].select("uid", "variety", "text").iter_rows():
            f.write(json.dumps({"item": u, "variety": v, "text": t}) + "\n")
print(split, a.variety, r.height, "rows ->", (r.height + 99) // 100, "batches")
