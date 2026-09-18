"""Compose a run from other runs' columns: sentiment from one run, sarcasm per variety from others.

No fitting: each output probability is copied from the named member run for that task x variety,
so the composed OOF is as cross-fitted as its members. Test rows are composed too when every
member has test predictions for the same test file.

  uv run python -m experiments.encoders.compose NAME --sent RUN --sarc-au RUN --sarc-uk RUN
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import json

import polars as pl

from alta.common import PREDS, load_pool, load_test, save_preds


def _read(run: str) -> tuple[pl.DataFrame, dict]:
    return pl.read_parquet(PREDS / f"{run}.parquet"), json.loads((PREDS / f"{run}.json").read_text())


def compose(name: str, sent: str, sarc_au: str, sarc_uk: str) -> None:
    runs = {"sent": sent, "sarc_au": sarc_au, "sarc_uk": sarc_uk}
    data = {k: _read(r) for k, r in runs.items()}
    shas = {m.get("test_sha") for _, m in data.values()}
    test_csv = next(iter(data.values()))[1].get("test_csv") if len(shas) == 1 and None not in shas else None

    def build(split: str, frame: pl.DataFrame) -> pl.DataFrame:
        cols = {k: p.filter(pl.col("split") == split).select("uid", pl.col("p_sent").alias(f"s_{k}"),
                                                               pl.col("p_sarc").alias(f"c_{k}"))
                for k, (p, _) in data.items()}
        out = frame.select("uid", pl.col("variety").cast(pl.String))
        for k, c in cols.items():
            assert c.height == out.height, f"{runs[k]} {split}: {c.height} rows, expected {out.height}"
            out = out.join(c, on="uid", how="left", maintain_order="left")
        return out.select(
            "uid", pl.col("s_sent").alias("p_sent"),
            pl.when(pl.col("variety") == "en-UK").then(pl.col("c_sarc_uk")).otherwise(pl.col("c_sarc_au"))
            .alias("p_sarc"))

    oof = build("oof", load_pool())
    test = build("test", load_test(test_csv)) if test_csv else None
    meta = {"model": "compose", "members": runs,
            "description": f"sentiment from {sent}; sarcasm en-AU from {sarc_au}, en-UK from {sarc_uk}"}
    print(save_preds(name, oof, test, meta=meta, test_csv=test_csv,
                     folds_used=load_pool().select("uid", "fold")))  # members use the pool folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--sent", required=True)
    ap.add_argument("--sarc-au", required=True)
    ap.add_argument("--sarc-uk", required=True)
    a = ap.parse_args()
    compose(a.name, a.sent, a.sarc_au, a.sarc_uk)


if __name__ == "__main__":
    main()
