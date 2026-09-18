"""Print schema/label balance of external sarcasm sets (for pre-SFT planning)."""
import json
from pathlib import Path

import polars as pl

D = Path("data/external/sarcasm")
print(json.dumps(json.loads((D / "MANIFEST.json").read_text()), indent=1)[:4000])
for f in sorted(D.glob("*.parquet")):
    d = pl.read_parquet(f)
    print("\n==", f.name, d.shape, d.columns)
    for c in ("label", "sarcasm", "variety", "subreddit", "source"):
        if c in d.columns:
            print(d[c].value_counts(sort=True).head(12))
    with pl.Config(fmt_str_lengths=120, tbl_width_chars=200):
        print(d.head(3))
