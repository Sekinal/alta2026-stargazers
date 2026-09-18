"""Drop external texts that overlap our labelled pool (or a test CSV).

A row is dropped when its normalised text (lowercase, letters/digits only, whitespace collapsed)
  * equals a reference text, or
  * contains a reference text, or is contained in one, with the shorter side >= 40 normalised chars.
Both ``text`` and ``parent_text`` are checked (a pool text used as context would leak it too).

Library use:   keep_mask(texts, refs) -> np.ndarray[bool]
CLI (rewrites the parquets in place, prints drop counts as JSON):
  uv run --with pyahocorasick python experiments/data/dedupe.py data/external/sarcasm/*.parquet
  uv run --with pyahocorasick python experiments/data/dedupe.py --test data/test.csv data/external/sarcasm/*.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ahocorasick
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alta.common import load_pool, load_test  # noqa: E402

MIN_SUB = 40


def normalise(s: pl.Expr) -> pl.Expr:
    return (s.fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ")
            .str.strip_chars())


def norm_list(texts: list[str] | pl.Series) -> list[str]:
    return pl.DataFrame({"t": pl.Series(texts, dtype=pl.String)}).select(normalise(pl.col("t")))["t"].to_list()


class Matcher:
    """Exact / containment matcher against a fixed set of normalised reference texts."""

    def __init__(self, refs: list[str]):
        refs = [r for r in refs if r]
        self.exact = set(refs)
        # external text contains a reference: Aho-Corasick over references >= MIN_SUB chars
        self.ac = ahocorasick.Automaton()
        for r in {r for r in refs if len(r) >= MIN_SUB}:
            self.ac.add_word(r, len(r))
        self.has_ac = len(self.ac) > 0
        if self.has_ac:
            self.ac.make_automaton()
        # reference contains external text: any external text of length >= MIN_SUB inside a reference
        # starts with one of the reference's MIN_SUB-char windows; verify candidates with `in`.
        self.windows: dict[str, list[int]] = {}
        self.refs = [r for r in refs if len(r) >= MIN_SUB]
        for i, r in enumerate(self.refs):
            for j in range(len(r) - MIN_SUB + 1):
                self.windows.setdefault(r[j:j + MIN_SUB], []).append(i)

    def hit(self, t: str) -> bool:
        if not t:
            return False
        if t in self.exact:
            return True
        if len(t) >= MIN_SUB:
            if self.has_ac:
                for _ in self.ac.iter(t):
                    return True
            for i in self.windows.get(t[:MIN_SUB], ()):
                if t in self.refs[i]:
                    return True
        return False


def reference_texts(test_paths: list[str] | None = None) -> list[str]:
    texts = load_pool()["text"].to_list()
    for p in test_paths or []:
        texts += load_test(p)["text"].to_list()
    return norm_list(texts)


def keep_mask(texts, matcher: Matcher) -> np.ndarray:
    return np.array([not matcher.hit(t) for t in norm_list(texts)], dtype=bool)


def dedupe_frame(df: pl.DataFrame, matcher: Matcher) -> tuple[pl.DataFrame, dict]:
    keep = keep_mask(df["text"], matcher)
    stats = {"text_hits": int((~keep).sum())}
    if "parent_text" in df.columns:
        pk = keep_mask(df["parent_text"], matcher)
        stats["parent_hits"] = int((keep & ~pk).sum())
        keep &= pk
    stats["dropped"] = int((~keep).sum())
    stats["dropped_pos"] = int(((~keep) & (df["sarcasm"].to_numpy() == 1)).sum()) if "sarcasm" in df.columns else None
    return df.filter(pl.Series(keep)), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--test", action="append", default=[], help="also dedupe against this test CSV")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    matcher = Matcher(reference_texts(a.test))
    report = {}
    for f in a.files:
        df = pl.read_parquet(f)
        out, stats = dedupe_frame(df, matcher)
        stats["rows_before"], stats["rows_after"] = df.height, out.height
        report[Path(f).stem] = stats
        if not a.dry_run and out.height != df.height:
            out.write_parquet(f)
        print(json.dumps({Path(f).stem: stats}), flush=True)


if __name__ == "__main__":
    main()
