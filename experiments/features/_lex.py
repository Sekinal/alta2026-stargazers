"""Tokenisation, sentence/clause splitting and sentiment lexicons shared by the linguistic builders.

Lexicons are general-purpose resources built on other data (no ALTA labels):
  VADER 3.3.2 (Hutto & Gilbert 2014): 7.5k valence-rated tokens, booster and negation lists.
  NRC EmoLex (Mohammad & Turney 2013, via the nrclex 4.1.0 package data): 6.5k words -> 8 emotions
  + positive/negative; covers "negative situation" words VADER lacks (delay, tax, late).
Run builders with ``uv run --with vaderSentiment --with nrclex python ...``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from functools import lru_cache

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
CLAUSE_SPLIT = re.compile(r"[,;:]\s+|\s[-–—]+\s|\s(?=\b(?:but|although|though|however|yet|whereas|except)\b)", re.I)
NRC_EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in "!?"]


def sentences(text: str) -> list[str]:
    s = [x.strip() for x in SENT_SPLIT.split(text) if x and x.strip()]
    return s or [text]


def clauses(text: str) -> list[str]:
    out = []
    for s in sentences(text):
        out += [c.strip() for c in CLAUSE_SPLIT.split(s) if c and c.strip() and re.search(r"[a-zA-Z]", c)]
    return out or [text]


@lru_cache(maxsize=1)
def vader():
    import vaderSentiment.vaderSentiment as v
    return v.SentimentIntensityAnalyzer(), set(v.BOOSTER_DICT), set(v.NEGATE) | {"never", "no", "nothing", "nobody", "none"}


@lru_cache(maxsize=1)
def nrc() -> dict[str, frozenset]:
    path = importlib.util.find_spec("nrclex").submodule_search_locations[0] + "/data/nrc_en.json"
    return {w: frozenset(e) for w, e in json.load(open(path)).items()}


def nrc_lookup(w: str) -> frozenset:
    """Exact match, then a crude suffix strip (cancelled -> cancel, delays -> delay)."""
    d = nrc()
    if w in d:
        return d[w]
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""),
                     ("ly", ""), ("led", "l")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            r = w[: -len(suf)] + rep
            if r in d:
                return d[r]
    return frozenset()


def valence(w: str) -> float:
    """VADER token valence (0 if not in the lexicon)."""
    return vader()[0].lexicon.get(w, 0.0)
