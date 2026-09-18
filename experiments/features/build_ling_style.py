"""Feature set `ling_style`: register, structure and function-word profile.

Theory: function words and punctuation carry stance and register independently of topic
(Pennebaker's LIWC function-word findings; stylometry), and are the part of a text least tied to
this dataset's topics, so they should transfer. Sarcastic reddit replies are short, conversational,
second/third-person oriented and punctuation-heavy; sincere advice/reviews are longer, first-person
and descriptive. Readability and lexical diversity capture planning/effort. All computed on the
normalised text (tokeniser artefacts undone) and as rates, so formatting changes do not move them.

Does not duplicate experiments/classical/features.py (log_chars, log_words, n_sent, first_person,
second_person, digits, split_nt are there).

  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_style.py [--eval | --input TEST.csv]
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import polars as pl

from _fe import main
from _lex import sentences, tokens

SET = "ling_style"

FW = {
    "fw_i": r"\b(?:i|me|my|mine|myself|i'm|i've|i'd|i'll)\b",
    "fw_we": r"\b(?:we|us|our|ours|ourselves|we're|we've|we'd|we'll)\b",
    "fw_he_she": r"\b(?:he|she|him|her|his|hers|he's|she's)\b",
    "fw_they": r"\b(?:they|them|their|theirs|themselves|they're|they've|they'd|they'll)\b",
    "fw_it_deictic": r"\b(?:it|it's|this|that|that's|these|those)\b",
    "fw_article": r"\b(?:a|an|the)\b",
    "fw_prep": r"\b(?:in|on|at|of|for|with|to|from|by|about|into|over|after|before|under|between|through)\b",
    "fw_conj": r"\b(?:and|or|but|so|because|although|though|while|whereas|if|unless)\b",
    "fw_negation": r"\b(?:not|no|never|nothing|nobody|none|nor|\w+n't|cannot)\b",
    "fw_modal": r"\b(?:can|could|would|should|might|may|must|shall|will|won't|wouldn't|shouldn't|couldn't)\b",
    "fw_aux_past": r"\b(?:was|were|had|did)\b",
    "fw_quant": r"\b(?:all|some|any|many|much|more|less|few|every|each|lot|lots)\b",
}

DESCRIPTIONS = {
    "mean_sent_len": "mean tokens per sentence, log1p",
    "sd_sent_len": "SD of tokens per sentence, log1p",
    "mean_word_len": "mean characters per token",
    "long_word_rate": "tokens with >= 7 characters / tokens",
    "flesch": "Flesch reading ease (vowel-group syllable estimate), clipped to [-50, 120] / 100",
    "ttr": "moving-average type-token ratio (window 25; plain TTR if shorter)",
    "hapax_rate": "tokens occurring once / tokens",
    "comma_rate": "commas per token",
    "punct_entropy": "Shannon entropy of the punctuation-character distribution (bits)",
    "punct_rate": "punctuation characters per token",
    "paren_rate": "parentheses/brackets per token",
    "newline_rate": "line breaks per sentence (paragraphing/lists)",
    "list_lines": "lines starting with a bullet/number marker, log1p",
    "contraction_rate": "contractions (n't, 's, 're, 'm, 've, 'll, 'd) per token",
    "number_rate": "numeric tokens (incl. prices, percentages) per token",
    "cap_start_share": "share of sentences starting with a capital letter",
    **{k: f"{k[3:]} function words per token" for k in FW},
}


def syllables(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and len(w) > 3 else 0))


def row(text: str) -> dict:
    lo = text.lower()
    toks = tokens(text)
    n = max(len(toks), 1)
    sents = sentences(text)
    sl = np.array([len(tokens(s)) for s in sents], float)
    syl = sum(syllables(w) for w in toks)
    flesch = 206.835 - 1.015 * (n / len(sents)) - 84.6 * (syl / n) if toks else 0.0
    if len(toks) > 25:
        ttr = float(np.mean([len(set(toks[i:i + 25])) / 25 for i in range(len(toks) - 24)]))
    else:
        ttr = len(set(toks)) / n
    cnt = Counter(toks)
    punct = [c for c in text if c in ".,;:!?\"'()[]-–—…/*&%$#@"]
    pc = Counter(punct)
    ent = -sum(v / len(punct) * math.log2(v / len(punct)) for v in pc.values()) if punct else 0.0
    return {
        "mean_sent_len": np.log1p(sl.mean()),
        "sd_sent_len": np.log1p(sl.std()),
        "mean_word_len": float(np.mean([len(w) for w in toks])) if toks else 0.0,
        "long_word_rate": sum(len(w) >= 7 for w in toks) / n,
        "flesch": float(np.clip(flesch, -50, 120)) / 100,
        "ttr": ttr,
        "hapax_rate": sum(v == 1 for v in cnt.values()) / n,
        "comma_rate": text.count(",") / n,
        "punct_entropy": ent,
        "punct_rate": len(punct) / n,
        "paren_rate": len(re.findall(r"[()\[\]]", text)) / n,
        "newline_rate": text.count("\n") / len(sents),
        "list_lines": np.log1p(len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", text))),
        "contraction_rate": len(re.findall(r"\b\w+(?:n't|'s|'re|'m|'ve|'ll|'d)\b", lo)) / n,
        "number_rate": len(re.findall(r"[$£€]?\d+(?:[.,]\d+)?%?", text)) / n,
        "cap_start_share": sum(bool(re.match(r"^\W*[A-Z]", s)) for s in sents) / len(sents),
        **{k: len(re.findall(p, lo)) / n for k, p in FW.items()},
    }


def build(df: pl.DataFrame) -> pl.DataFrame:
    rows = [row(t) for t in df["text"].to_list()]
    return pl.DataFrame({"uid": df["uid"]}).hstack(pl.DataFrame(rows).select(list(DESCRIPTIONS)))


if __name__ == "__main__":
    main(SET, build, DESCRIPTIONS)
