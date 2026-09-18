"""Sentence-level zero-shot scores (label-free): within-text incongruity of sentiment and irony.

Each text is split into sentences (regex on .!? and newlines; up to 24 kept, the rest merged into the
last); SiEBERT sentiment and DeBERTa-v3-large TweetEval-irony run on every sentence (max 128 tokens).
Per text and model: max, min, mean, sd, range, first, last; for sentiment also the share of clearly
positive / negative sentences, a mixed-polarity flag and whole-text minus max-sentence polarity.
Single-sentence texts get sd 0 and range 0; n_sent is included.

  flock .gpu.lock uv run python experiments/features/build_zs_sentlvl.py --input pool|TEST.csv --out PARQUET
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np

from build_zs_clf import MODELS
from fe_common import cached, classifier_logits, default_out, load_input, write_set

NAME = "zs_sentlvl"
MAX_SENT = 24
SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)|\s*\n+\s*")


def split(text: str) -> list[str]:
    parts = [p.strip() for p in SPLIT.split(text) if p and p.strip() and re.search(r"\w", p)]
    parts = parts or [text.strip() or "."]
    if len(parts) > MAX_SENT:
        parts = parts[:MAX_SENT - 1] + [" ".join(parts[MAX_SENT - 1:])]
    return parts


def agg(v: np.ndarray, key: str) -> dict[str, float]:
    return {f"{key}_max": v.max(), f"{key}_min": v.min(), f"{key}_mean": v.mean(), f"{key}_sd": v.std(),
            f"{key}_range": v.max() - v.min(), f"{key}_first": v[0], f"{key}_last": v[-1]}


def build(inp: str, out: str | None) -> None:
    t0 = time.time()
    df = load_input(inp)
    sents = [split(t) for t in df["text"].to_list()]
    flat = [s for ss in sents for s in ss]
    owner = np.repeat(np.arange(df.height), [len(ss) for ss in sents])
    print(f"{len(flat)} sentences", flush=True)
    scores = {}
    for key in ("sent_siebert", "irony_deb"):
        scores[key] = cached(f"sent_{key}_n{len(flat)}", df,
                   lambda: _per_sentence(MODELS[key][0], flat), n_rows=len(flat))
    whole = cached("clf_sent_siebert", df, lambda: classifier_logits(MODELS["sent_siebert"][0], df["text"].to_list()))
    whole = whole[:, 1] - whole[:, 0]
    rows = []
    for i in range(df.height):
        m = owner == i
        s, r = scores["sent_siebert"][m], scores["irony_deb"][m]
        row = {**agg(s, "sent"), **agg(r, "irony"), "n_sent": np.log1p(m.sum()),
               "sent_frac_pos": (s > 2).mean(), "sent_frac_neg": (s < -2).mean(),
               "sent_mixed": float((s > 2).any() and (s < -2).any()),
               "sent_whole_minus_max": whole[i] - s.max()}
        rows.append(row)
    feats = {k: np.array([r[k] for r in rows], np.float32) for k in rows[0]}
    desc = {k: "" for k in feats}
    for k in feats:
        model = "SiEBERT sentiment log-odds" if k.startswith("sent") else "TweetEval-irony DeBERTa log-odds"
        desc[k] = f"{model}, per-sentence aggregate: {k.split('_', 1)[1]}"
    desc["n_sent"] = "log(1 + n sentences)"
    desc["sent_frac_pos"] = "share of sentences with sentiment log-odds > 2"
    desc["sent_frac_neg"] = "share of sentences with sentiment log-odds < -2"
    desc["sent_mixed"] = "1 if the text has both a clearly positive and a clearly negative sentence"
    desc["sent_whole_minus_max"] = "whole-text sentiment log-odds minus the most positive sentence's"
    write_set(NAME, df, feats, out or default_out(NAME, df), desc,
              [MODELS["sent_siebert"][0], MODELS["irony_deb"][0]], t0, {"max_sent": MAX_SENT})


def _per_sentence(hf_id: str, flat: list[str]) -> np.ndarray:
    L = classifier_logits(hf_id, flat, max_len=128, bs=64)
    return L[:, 1] - L[:, 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.input, a.out)
