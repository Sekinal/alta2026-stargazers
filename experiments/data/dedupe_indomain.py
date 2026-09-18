"""Drop in-domain pretraining texts that overlap our labelled data (pool) or the test set.

A corpus text is dropped if its normalised form (lowercase, alphanumerics only; whitespace is removed
too, so tokenised "do n't" matches "don't", as in dedupe_sentiment.py)
  * equals any reference text, or
  * contains a reference text of >= 40 normalised chars, or
  * is >= 40 normalised chars and is contained in a reference text.
References: alta.common.load_pool() texts + every --against CSV/parquet ('text' column) + data/test.csv
if it exists. Re-run in place once the test CSV is released:

    uv run python experiments/data/dedupe_indomain.py --in data/external/indomain --against data/test.csv

Updates <out>/MANIFEST.json (rows, ~tokens, time range, cumulative dedupe drops per file).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alta.common import load_pool  # noqa: E402
from indomain_common import SOURCES, approx_tokens, normalise  # noqa: E402

MIN_CONTAIN = 40
FILES = ("reddit_au", "reddit_uk", "reviews_au", "reviews_uk")


def dedupe_key(col: pl.Expr) -> pl.Expr:
    return normalise(col).str.replace_all(" ", "", literal=True)


def references(against: list[Path]) -> tuple[list[str], list[str], list[str]]:
    texts = load_pool()["text"].to_list()
    test = ROOT / "data" / "test.csv"
    paths = against + ([test] if test.exists() and test.resolve() not in [p.resolve() for p in against] else [])
    for p in paths:
        texts += (pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, infer_schema_length=0))["text"].to_list()
    norms = pl.Series(texts).drop_nulls().to_frame("t").select(dedupe_key(pl.col("t"))).to_series()
    norms = norms.filter(norms.str.len_chars() > 0).unique().to_list()
    return norms, [r for r in norms if len(r) >= MIN_CONTAIN], ["pool"] + [str(p) for p in paths]


def overlap_flags(norm: pl.Series, ref_all: list[str], ref_long: list[str]) -> pl.DataFrame:
    grams = pl.Series(list({r[i:i + MIN_CONTAIN] for r in ref_long for i in range(len(r) - MIN_CONTAIN + 1)}), dtype=pl.String)
    df = norm.to_frame("n").with_columns(
        pl.col("n").is_in(pl.Series(ref_all).implode()).alias("eq"),
        pl.col("n").str.contains_any(ref_long).alias("contains_ref"),
        ((pl.col("n").str.len_chars() >= MIN_CONTAIN)
         & pl.col("n").str.slice(0, MIN_CONTAIN).is_in(grams.implode())).alias("_cand"))
    # verify "contained in a reference" candidates exactly (few rows)
    cand = df.with_row_index().filter(pl.col("_cand"))
    hits = {i for i, n in cand.select("index", "n").iter_rows() if any(n in r for r in ref_long)}
    return df.with_columns(pl.Series("in_ref", [i in hits for i in range(df.height)])).drop("_cand")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "data/external/indomain")
    ap.add_argument("--out", type=Path, default=None, help="default: same as --in (in place)")
    ap.add_argument("--against", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    out = args.out or args.inp
    out.mkdir(parents=True, exist_ok=True)
    ref_all, ref_long, ref_names = references(args.against)
    print(f"references: {len(ref_all)} unique normalised texts, {len(ref_long)} >= {MIN_CONTAIN} chars", flush=True)
    man_path = out / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"files": {}}
    for name in FILES:
        src = args.inp / f"{name}.parquet"
        if not src.exists():
            print(f"skip {name}: missing")
            continue
        t0 = time.time()
        df = pl.read_parquet(src)
        flags = overlap_flags(df.select(dedupe_key(pl.col("text"))).to_series(), ref_all, ref_long)
        drop = flags.select(pl.any_horizontal("eq", "contains_ref", "in_ref")).to_series()
        run = {"refs": ref_names, "n_ref": len(ref_all),
               "dropped": int(drop.sum()), "equal": int(flags["eq"].sum()),
               "contains_ref": int(flags["contains_ref"].sum()), "contained_in_ref": int(flags["in_ref"].sum()),
               "time": time.strftime("%Y-%m-%d %H:%M")}
        kept = df.filter(~drop)
        tmp = out / f"{name}.parquet.tmp"
        kept.write_parquet(tmp)
        tmp.rename(out / f"{name}.parquet")
        entry = manifest["files"].setdefault(name, {})
        entry.setdefault("dedupe_runs", []).append(run)
        ts = kept["created_utc"].drop_nulls()
        entry.update({
            "path": str(out / f"{name}.parquet"), "rows": kept.height, "approx_tokens": approx_tokens(kept),
            "dedupe_dropped_total": sum(r["dropped"] for r in entry["dedupe_runs"]),
            "time_range_utc": [time.strftime("%Y-%m-%d", time.gmtime(ts.min())),
                               time.strftime("%Y-%m-%d", time.gmtime(ts.max()))] if len(ts) else None,
            "by_dataset": {k: {"rows": n, "approx_tokens": approx_tokens(kept.filter(pl.col("dataset") == k)),
                               "license": kept.filter(pl.col("dataset") == k)["license"][0]}
                           for k, n in kept.group_by("dataset").len().rows()},
        })
        print(f"{name}: {df.height} -> {kept.height} ({run}) in {time.time() - t0:.0f}s", flush=True)
    manifest["sources"] = SOURCES
    for st in sorted(args.inp.glob("*_stats.json")):  # build-time filter counts from build_*.py
        manifest.setdefault("build_stats", {})[st.stem] = json.loads(st.read_text())
    manifest["updated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    man_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
