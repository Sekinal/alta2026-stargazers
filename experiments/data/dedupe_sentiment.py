"""Deduplicate external text against the competition texts (pool, and later the test CSV).

A candidate text is dropped if, after compact normalisation (NFKC, lowercase, keep only [a-z0-9],
so "do n't" == "don't" and punctuation/spacing differences vanish), it
  - equals a reference text, or
  - contains a reference text of >= MIN_LEN normalised chars, or
  - is contained in a reference text and itself has >= MIN_LEN normalised chars.

Library use:
    idx = RefIndex(reference_texts)
    keep = ~idx.flag(candidate_texts)          # numpy bool array

CLI (re-run against the test file once released; rewrites files in place, prints drop counts):
    PYTHONPATH=. uv run --with pyahocorasick python experiments/data/dedupe_sentiment.py \
        --against data/test.csv data/external/*/*.parquet [--no-pool] [--dry-run]
--against takes a CSV or parquet with a `text` column and may be repeated; the pool is always included
unless --no-pool. Every text-bearing column present in a file (TEXT_COLUMNS, e.g. `parent_text` context in the
sarcasm sets) is checked; a row is dropped if any of them hits.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import ahocorasick
import numpy as np
import polars as pl

MIN_LEN = 40
TEXT_COLUMNS = ("text", "parent_text", "context", "title", "submission_title", "selftext")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str | None) -> str:
    if not s:
        return ""
    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", s).lower())


class RefIndex:
    def __init__(self, texts: list[str], min_len: int = MIN_LEN):
        self.min_len = min_len
        self.refs = sorted({n for n in map(norm, texts) if n})
        self.exact = set(self.refs)
        # reference-in-candidate: Aho-Corasick over long reference strings
        self.ac = ahocorasick.Automaton()
        for i, r in enumerate(self.refs):
            if len(r) >= min_len:
                self.ac.add_word(r, i)
        self.ac.make_automaton()
        # candidate-in-reference: every min_len window of each reference -> reference ids
        self.win: dict[str, list[int]] = {}
        for i, r in enumerate(self.refs):
            for j in range(len(r) - min_len + 1):
                self.win.setdefault(r[j : j + min_len], []).append(i)

    def hit(self, n: str) -> bool:
        if not n:
            return False
        if n in self.exact:
            return True
        if len(n) < self.min_len:
            return False
        if next(self.ac.iter(n), None) is not None:
            return True
        cands = self.win.get(n[: self.min_len])
        return bool(cands) and any(n in self.refs[i] for i in cands)

    def flag(self, texts) -> np.ndarray:
        return np.fromiter((self.hit(norm(t)) for t in texts), dtype=bool, count=len(texts))


def pool_texts() -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alta.common import load_pool

    return load_pool()["text"].to_list()


def read_texts(path: str) -> list[str]:
    df = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path, infer_schema_length=0)
    return df["text"].drop_nulls().to_list()


def build_index(against: tuple[str, ...] = (), use_pool: bool = True) -> RefIndex:
    texts = pool_texts() if use_pool else []
    for p in against:
        texts += read_texts(p)
    return RefIndex(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--against", "--ref-csv", dest="against", action="append", default=[],
                    help="extra reference file (CSV or parquet with a text column), e.g. data/test.csv; repeatable")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    idx = build_index(tuple(a.against), use_pool=not a.no_pool)
    print(f"reference texts: {len(idx.refs)}", file=sys.stderr)
    for f in map(Path, a.files):
        df = pl.read_parquet(f)
        cols = [c for c in TEXT_COLUMNS if c in df.columns and df.schema[c] == pl.String]
        bad = np.zeros(df.height, dtype=bool)
        for c in cols:
            bad |= idx.flag(df[c].to_list())
        print(f"{f.name}: {int(bad.sum())} / {df.height} dropped (checked {', '.join(cols)})")
        if bad.any() and not a.dry_run:
            df.filter(pl.Series(~bad)).write_parquet(f)


if __name__ == "__main__":
    main()
