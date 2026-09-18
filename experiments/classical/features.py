"""Handcrafted meta-features (no labels involved): length, punctuation, markers, entities, segment.

Used by gbdt.py (as model inputs) and stack.py (as stacker meta-features). Deterministic per text.
"""

from __future__ import annotations

import numpy as np
import polars as pl

PROFANITY = r"\b(?:fuck\w*|shit\w*|cunt\w*|bastard\w*|bullshit|crap\w*|damn\w*|bloody|arse\w*|ass(?:hole)?s?|wank\w*|bollocks|piss\w*|dick\w*|twat\w*|prick\w*)\b"
POL_AU = r"\b(?:labor|lnp|liberals?|dutton|albo\w*|albanese|greens|morrison|scomo|one nation|hanson|coalition|nationals|abbott|turnbull|murdoch)\b"
POL_UK = r"\b(?:tor(?:y|ies)|labour|reform|nhs|farage|starmer|sunak|brexit|boris|johnson|truss|lib dems?|snp|corbyn|westminster)\b"
SARC_PHRASES = r"\b(?:oh|ah|boy|wow|sure|yeah right|can'?t wait|i'?m sure|to be fair|be fair|totally|clearly|obviously|shocking|what a surprise|who knew|genius|lol|lmao|haha\w*)\b"
THANKS = r"\b(?:thanks?|thank you|cheers|ta)\b"


def handcrafted(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """df needs text, source, variety. Returns (n x k float32 matrix, names)."""
    t = pl.col("text")
    lo = t.str.to_lowercase()
    exprs = {
        "log_chars": (t.str.len_chars() + 1).log(),
        "log_words": (t.str.split(" ").list.len() + 1).log(),
        "n_sent": t.str.count_matches(r"[.!?]+(?:\s|$)"),
        "q_marks": t.str.count_matches(r"\?"),
        "excl": t.str.count_matches("!"),
        "ellipsis": t.str.count_matches(r"\.\.\.|…"),
        "quotes": t.str.count_matches(r"\"|''|\s'\w"),
        "caps_words": t.str.count_matches(r"\b[A-Z]{2,}\b"),
        "caps_ratio": t.str.count_matches(r"[A-Z]") / (t.str.len_chars() + 1),
        "slash_s": lo.str.contains(r"(?:^|\s)/s\b").cast(pl.Int8),
        "starts_oh": lo.str.contains(r"^\W*(?:oh|ah|wow|yeah|sure)\b").cast(pl.Int8),
        "sarc_phr": lo.str.count_matches(SARC_PHRASES),
        "profanity": lo.str.count_matches(PROFANITY),
        "pol_au": lo.str.count_matches(POL_AU),
        "pol_uk": lo.str.count_matches(POL_UK),
        "thanks": lo.str.count_matches(THANKS),
        "first_person": lo.str.count_matches(r"\b(?:i|me|my|mine|we|our)\b") / (t.str.split(" ").list.len() + 1),
        "second_person": lo.str.count_matches(r"\b(?:you|your|you're|ya)\b") / (t.str.split(" ").list.len() + 1),
        "digits": t.str.count_matches(r"\d"),
        "split_nt": t.str.count_matches(r"\w n't\b"),
        "is_uk": (pl.col("variety").cast(pl.String) == "en-UK").cast(pl.Int8),
        "is_reddit": (pl.col("source").cast(pl.String) == "reddit").cast(pl.Int8),
    }
    out = df.select(**exprs).cast(pl.Float32)
    return out.to_numpy(), out.columns
