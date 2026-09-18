"""Assemble the recommended sentiment pre-fine-tuning mix (target weak spot: AU-reddit sentiment).

    uv run python experiments/data/sentiment_mix.py [--scale 1.0] [--out data/external/sentiment/mix_v1.parquet]

Rows per dataset are upper bounds (taken label-balanced where the file allows). Reddit and short conversational
text are over-weighted; reviews are kept for robustness, not because review sentiment needs help. Excluded:
QUT election stance labels (poor proxy for this task's sentiment) and google_local_2018_au_uk (held until it is
deduped against the test file; see MANIFEST). Writes the composition to MANIFEST.json under "_mix_v1".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external" / "sentiment"
SEED = 0

# file -> max rows. __full keeps the natural label ratio (only used where it is close to balanced or small).
MIX = {
    "go_emotions__full": 40_000,        # human-labelled Reddit comments
    "tweet_eval_sentiment__full": 35_000,  # human-labelled tweets, short and conversational
    "sentiment140": 30_000,             # distant labels, very short casual text
    "dynasent": 40_000,                 # hard, human-validated review sentences
    "sst2": 25_000,                     # short phrases
    "rotten_tomatoes": 10_000,
    # google_local_2018_au_uk excluded: overlaps BESSTIE's Google source (test-label leak risk), see MANIFEST
    "yelp_polarity": 20_000,
    "amazon_polarity": 20_000,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(EXT / "mix_v1.parquet"))
    a = ap.parse_args()
    parts, comp = [], {}
    for name, n in MIX.items():
        df = pl.read_parquet(EXT / f"{name}.parquet")
        n = min(int(n * a.scale), df.height)
        per = n // 2
        k = [df.filter(pl.col("sentiment") == s) for s in (0, 1)]
        # balanced if both labels can fill half; otherwise take what the minority allows and fill with the other
        take0 = min(per, k[0].height)
        take1 = min(n - take0, k[1].height)
        parts += [k[0].sample(take0, seed=SEED), k[1].sample(take1, seed=SEED)]
        comp[name] = {"rows": take0 + take1, "pos": take1}
        print(f"{name:30s} {take0 + take1:7d}  pos {take1 / max(take0 + take1, 1):.2f}")
    mix = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED)
    mix.write_parquet(a.out)
    print(f"total {mix.height}  pos {mix['sentiment'].mean():.3f}  ->  {a.out}")
    by_source = dict(mix.group_by("source").len().sort("len", descending=True).iter_rows())
    print(by_source)
    mpath = EXT / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["_mix_v1"] = {"file": str(Path(a.out).relative_to(ROOT)), "rows": mix.height,
                           "pos_rate": round(float(mix["sentiment"].mean()), 4), "by_source": by_source,
                           "composition": comp, "excluded": {
                               "google_local_2018_au_uk": "hold_until_test_dedupe (overlaps BESSTIE Google source)",
                               "qut_ausvotes2025": "stance labels are not this task's sentiment (probe AUC 0.56)",
                               "imdb": "long movie reviews, low relevance"}}
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
