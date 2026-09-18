"""SARC 2.0 main (Khodak et al. 2018): reddit comments self-labelled with /s (tag removed by the authors).

  uv run --with pyahocorasick python experiments/data/sarc_build.py

Reads data/external/raw/sarc/main/{comments.json.bz2, {train,test}-unbalanced.csv.bz2} (downloaded from
https://nlp.cs.princeton.edu/old/SARC/2.0/main/) and writes
  sarc_main.parquet   every labelled response in the unbalanced sequence files (train + test), with the
                      comment it replies to as parent_text and subreddit / year; is_balanced-style
                      subsampling is left to the consumer.
  sarc_ukau.parquet   the subset from UK / AU subreddits (see arctic_fetch.SUBS plus a few older ones).
"""

from __future__ import annotations

import bz2
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arctic_fetch import VARIETY  # noqa: E402
from dedupe import Matcher, reference_texts  # noqa: E402
from ext_common import RAW, finalize, write  # noqa: E402

D = RAW / "sarc" / "main"
URL = "https://nlp.cs.princeton.edu/old/SARC/2.0/"
LICENSE = "research use; no explicit licence (Princeton release; CC BY 2.0 per the CreativeLang HF mirror)"
EXTRA_VARIETY = {"ukpolitics": "en-UK", "unitedkingdom": "en-UK", "britishproblems": "en-UK", "UKPersonalFinance": "en-UK",
                 "australia": "en-AU", "AustralianPolitics": "en-AU", "straya": "en-AU", "circlejerkaustralia": "en-AU",
                 "AFL": "en-AU", "nrl": "en-AU", "Cricket": "other", "PremierLeague": "en-UK", "soccer": "other",
                 "reddevils": "en-UK", "Gunners": "en-UK", "LiverpoolFC": "en-UK", "chelseafc": "en-UK",
                 "coys": "en-UK", "canada": "other", "newzealand": "other", "ireland": "other", "india": "other"}


def main() -> None:
    print("loading comments.json ...", flush=True)
    with bz2.open(D / "comments.json.bz2", "rt") as f:
        comments = json.load(f)
    ids = list(comments)
    com = pl.DataFrame({
        "id": ids,
        "ctext": [comments[i].get("text") for i in ids],
        "csub": [comments[i].get("subreddit") for i in ids],
        "cutc": [comments[i].get("created_utc") for i in ids],
    }, schema={"id": pl.String, "ctext": pl.String, "csub": pl.String, "cutc": pl.Int64})
    del comments
    print("comments:", com.height, flush=True)

    rows = []
    for split in ("train", "test"):
        with bz2.open(D / f"{split}-unbalanced.csv.bz2", "rt") as f:
            for line in f:
                anc, resp, lab = line.rstrip("\n").split("|")
                parent = anc.split()[-1]
                for r, l in zip(resp.split(), lab.split()):
                    rows.append((r, parent, int(l)))
    lab = pl.DataFrame(rows, schema={"id": pl.String, "pid": pl.String, "sarcasm": pl.Int8}, orient="row")
    del rows
    print("labelled responses:", lab.height, "positives:", int(lab["sarcasm"].sum()), flush=True)

    df = (lab.join(com, on="id", how="inner")
          .join(com.select(pl.col("id").alias("pid"), pl.col("ctext").alias("parent_text")), on="pid", how="left"))
    var = {**EXTRA_VARIETY, **VARIETY}
    df = df.select(
        pl.col("ctext").alias("text"), "sarcasm", "parent_text", pl.col("csub").alias("subreddit"),
        pl.from_epoch("cutc").dt.year().alias("year"),
        pl.col("csub").replace_strict(var, default="unknown").alias("variety_guess"),
    )
    df = finalize(df, dataset="sarc_main", source="reddit", license=LICENSE, label_quality="self-labelled")
    matcher = Matcher(reference_texts())
    write(df, "sarc_main", url=URL + "main/", license=LICENSE, matcher=matcher,
          notes="SARC 2.0 main, train+test unbalanced sequence files (2009-2017, ~0.5-1% /s). /s already stripped "
                "by the authors. parent_text = the comment/post title replied to. Subsample negatives as needed.")
    uk = df.filter(pl.col("variety_guess").is_in(["en-UK", "en-AU"]))
    write(uk.with_columns(pl.lit("sarc_ukau").alias("dataset")), "sarc_ukau", url=URL + "main/", license=LICENSE,
          matcher=matcher, notes="Subset of sarc_main from UK/AU subreddits (subreddit-based variety_guess).")


if __name__ == "__main__":
    main()
