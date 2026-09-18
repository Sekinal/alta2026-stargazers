"""Build cleaned AU/UK place-review corpora from the raw downloads (download_reviews.sh).

Sources (all ungated):
  googlelocal  UCSD Google Local 2018 (He, Kang, McAuley 2017; Pasricha & McAuley 2018)
               https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal/  -- research use only.
               Places filtered by address ending in 'United Kingdom'/'UK' or 'Australia'. 2007-2014.
  booking      Booking.com accommodation reviews (HF Booking-com/accommodation-reviews), CC BY-SA 4.0
               (README says non-commercial). English reviews, 2023. accommodation_country filter.
  nhs          NHS.uk GP surgery reviews (HF janduplessis886/england-nhs-gp-reviews), Open Government
               Licence v3.0. England, 2022-2024. UK only.
  airbnb       Inside Airbnb reviews.csv.gz (Sydney, Melbourne; scrape 2026-06), CC BY 4.0. AU only
               (fills the AU gap); restricted to --airbnb-since and capped with --airbnb-cap.

    uv run --with py3langid python experiments/data/build_reviews.py --raw /workspace/raw/reviews \
        --out /workspace/raw/clean
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indomain_common import approx_tokens, clean_text, finalise  # noqa: E402

LIC = {
    "googlelocal": "UCSD Google Local 2018; research/non-commercial use (McAuley lab terms)",
    "booking": "CC BY-SA 4.0 (Booking.com README: non-commercial)",
    "nhs": "Open Government Licence v3.0 (NHS.uk reviews via HF janduplessis886)",
    "airbnb": "CC BY 4.0 (Inside Airbnb)",
}
URL = {
    "googlelocal": "https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal/reviews.clean.json.gz",
    "booking": "https://huggingface.co/datasets/Booking-com/accommodation-reviews",
    "nhs": "https://huggingface.co/datasets/janduplessis886/england-nhs-gp-reviews",
    "airbnb": "https://insideairbnb.com/get-the-data/",
}
PLACE_ID = re.compile(r"'gPlusPlaceId': u?'(\d+)'")


def googlelocal(raw: Path) -> pl.DataFrame:
    country = {}
    with gzip.open(raw / "googlelocal/places.clean.json.gz", "rt") as f:
        for line in f:
            if "Kingdom" not in line and "Australia" not in line and "UK" not in line:
                continue
            d = ast.literal_eval(line)
            last = ", ".join(d.get("address") or []).strip()
            if last.endswith("United Kingdom") or last.endswith(", UK"):
                country[d["gPlusPlaceId"]] = "en-UK"
            elif last.endswith("Australia"):
                country[d["gPlusPlaceId"]] = "en-AU"
    print(f"googlelocal places: {sum(v == 'en-UK' for v in country.values())} UK, "
          f"{sum(v == 'en-AU' for v in country.values())} AU", flush=True)
    rows = []
    with gzip.open(raw / "googlelocal/reviews.clean.json.gz", "rt") as f:
        for line in f:
            m = PLACE_ID.search(line)
            if not m or m.group(1) not in country:
                continue
            d = ast.literal_eval(line)
            if not d.get("reviewText"):
                continue
            cats = d.get("categories") or []
            rows.append({"text": d["reviewText"], "variety": country[m.group(1)],
                         "community": cats[0] if cats else None, "created_utc": d.get("unixReviewTime"),
                         "link_id": m.group(1)})
    return pl.DataFrame(rows, schema_overrides={"created_utc": pl.Int64}).with_columns(
        pl.lit("googlelocal").alias("dataset"))


def booking(raw: Path) -> pl.DataFrame:
    df = pl.concat([pl.read_csv(raw / f"booking/{s}.csv", infer_schema_length=0) for s in ("train", "val")])
    df = df.filter(pl.col("accommodation_country").is_in(["United Kingdom", "Australia"]))
    parts = [pl.col(c).fill_null("").str.strip_chars() for c in ("review_title", "review_positive", "review_negative")]
    return df.select(
        pl.concat_str(parts, separator="\n").str.strip_chars().alias("text"),
        pl.when(pl.col("accommodation_country") == "Australia").then(pl.lit("en-AU")).otherwise(pl.lit("en-UK"))
        .alias("variety"),
        pl.col("accommodation_type").alias("community"),
        pl.lit(None, pl.Int64).alias("created_utc"),  # only check-in month is given; reviews are from 2023
        pl.col("accommodation_id").alias("link_id"),
        pl.lit("booking").alias("dataset"))


def nhs(raw: Path) -> pl.DataFrame:
    df = pl.read_csv(raw / "nhs/england-nhs-gp-reviews.csv", infer_schema_length=0)
    return df.select(
        pl.concat_str([pl.col("title").fill_null(""), pl.col("comment").fill_null("")], separator="\n")
        .str.strip_chars().alias("text"),
        pl.lit("en-UK").alias("variety"), pl.lit("GP surgery").alias("community"),
        pl.col("visited_date").str.to_date("%B %Y", strict=False).cast(pl.Datetime("ms"))
        .dt.epoch("s").alias("created_utc"),
        pl.col("surgeryname").alias("link_id"), pl.lit("nhs_gp").alias("dataset"))


def airbnb(raw: Path, since: str, cap: int, seed: int = 2026) -> pl.DataFrame:
    dfs = []
    for p in sorted((raw / "airbnb").glob("australia_*.csv.gz")):
        with gzip.open(p, "rb") as f:
            df = pl.read_csv(f.read(), infer_schema_length=0)
        dfs.append(df.with_columns(pl.lit(p.name.split("_")[1].split(".")[0]).alias("city")))
    df = pl.concat(dfs).filter(pl.col("date") >= since, pl.col("comments").str.count_matches(r"\S+") >= 12)
    if df.height > cap:
        df = df.sample(cap, seed=seed)
    return df.select(
        pl.col("comments").alias("text"), pl.lit("en-AU").alias("variety"),
        pl.concat_str([pl.lit("airbnb "), pl.col("city")]).alias("community"),
        pl.col("date").str.to_date(strict=False).cast(pl.Datetime("ms")).dt.epoch("s").alias("created_utc"),
        pl.col("listing_id").alias("link_id"), pl.lit("airbnb").alias("dataset"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/raw/reviews", type=Path)
    ap.add_argument("--out", default="/workspace/raw/clean", type=Path)
    ap.add_argument("--airbnb-since", default="2022-01-01")
    ap.add_argument("--airbnb-cap", type=int, default=250_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts, stats = [], {}
    for name, fn in [("nhs", nhs), ("booking", booking), ("googlelocal", googlelocal),
                     ("airbnb", lambda r: airbnb(r, args.airbnb_since, args.airbnb_cap))]:
        df = fn(args.raw).with_columns(clean_text(pl.col("text"), reddit=False).alias("text"),
                                       pl.lit(LIC[name]).alias("license"), pl.lit("review").alias("source"),
                                       pl.lit("review").alias("kind"), pl.lit(None, pl.Int64).alias("score"),
                                       pl.lit(None, pl.String).alias("parent_id"))
        for v, g in df.group_by("variety"):
            g, st = finalise(g)
            st["tokens"] = approx_tokens(g)
            stats[f"{name}/{v[0]}"] = st
            print(name, v[0], st, flush=True)
            parts.append(g)
    allr = pl.concat(parts)
    for v, tag in [("en-AU", "au"), ("en-UK", "uk")]:
        g = allr.filter(pl.col("variety") == v)
        g.write_parquet(args.out / f"reviews_{tag}.parquet")
        print(f"reviews_{tag}: {g.height} rows, ~{approx_tokens(g) / 1e6:.1f}M tokens", flush=True)
    (args.out / "reviews_stats.json").write_text(json.dumps({"stats": stats, "urls": URL, "licenses": LIC}, indent=1))


if __name__ == "__main__":
    main()
