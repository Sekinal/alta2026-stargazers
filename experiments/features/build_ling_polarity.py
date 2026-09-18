"""Feature set `ling_polarity`: sentiment incongruity / contrast (label-free).

Theory: sarcasm is often a positive surface expression about a negative situation (Riloff et al. 2013;
Joshi et al. 2015 explicit/implicit incongruity), so the signal is not overall polarity (which the
embeddings capture) but its internal *spread*: positive words next to negative-situation words,
polarity flips between clauses, strong positives inside net-negative sentences, intensified or
negated positives. VADER supplies rated valence (with negation/booster rules), NRC EmoLex supplies
negative-situation and emotion words VADER lacks.

  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_polarity.py [--eval]
  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_polarity.py --input TEST.csv
"""

from __future__ import annotations

import numpy as np
import polars as pl

from _fe import main
from _lex import NRC_EMOTIONS, clauses, nrc_lookup, sentences, tokens, vader, valence

SET = "ling_polarity"
DESCRIPTIONS = {
    "compound": "VADER compound of the whole text",
    "sent_max": "max VADER compound over sentences",
    "sent_min": "min VADER compound over sentences",
    "sent_range": "sent_max - sent_min (polarity spread across sentences)",
    "clause_flips": "number of polarity sign changes between consecutive clauses (|compound| > 0.05)",
    "clause_contrast": "1 if one clause has compound > 0.3 and another < -0.3",
    "last_minus_first": "compound(last sentence) - compound(first sentence)",
    "pos_rate": "VADER positive tokens / tokens",
    "neg_rate": "VADER negative tokens / tokens",
    "nrc_neg_rate": "NRC negative (incl. negative-situation) words / tokens",
    "pos_share": "positive / (positive + negative) lexicon tokens (VADER pos, VADER or NRC neg), 0.5 if none",
    "both_polar": "1 if the text has both a VADER-positive and a (VADER or NRC) negative word",
    "strong_pos": "count of strong positive words (VADER valence >= 2.5, e.g. brilliant, love), log1p",
    "strong_pos_in_neg_sent": "positive words inside sentences whose other words are net negative, log1p",
    "pos_near_neg": "positive word with a negative word within 4 tokens (Riloff positive-sentiment + negative-situation), log1p",
    "boosted_pos": "booster (so/very/absolutely...) directly before a positive word, log1p",
    "negated_pos": "negator within 3 tokens before a positive word (not great), log1p",
    "lex_vs_vader": "mean raw token valence (no rules) scaled - VADER compound: negation/contrast rules at work",
    "pos_excl_sent": "sentences with positive compound ending in '!', log1p",
    "pos_trailing": "sentences with positive compound ending in '...' or '?', log1p",
    **{f"emo_{e}": f"NRC {e} words / tokens" for e in NRC_EMOTIONS},
    "joy_anger_contrast": "min(joy, anger+disgust) word counts: co-present opposite emotions, log1p",
}


def row(text: str) -> dict:
    sia, boosters, negate = vader()
    toks = tokens(text)
    n = max(len(toks), 1)
    val = np.array([valence(w) for w in toks]) if toks else np.zeros(0)
    emo = [nrc_lookup(w) for w in toks]
    pos = val > 0
    neg_v = val < 0
    neg_any = neg_v | np.array([("negative" in e) and v <= 0 for e, v in zip(emo, val)], bool) if toks else neg_v
    sents = sentences(text)
    sc = [sia.polarity_scores(s)["compound"] for s in sents]
    cc = [sia.polarity_scores(c)["compound"] for c in clauses(text)]
    signs = [np.sign(c) for c in cc if abs(c) > 0.05]
    flips = sum(a != b for a, b in zip(signs, signs[1:]))
    # positive words in net-negative sentences
    pin = 0
    pe = pt = 0
    for s, c in zip(sents, sc):
        st = tokens(s)
        sv = [valence(w) for w in st]
        rest_neg = [sum(sv) - v for v in sv]
        pin += sum(1 for v, r in zip(sv, rest_neg) if v > 0 and r < -0.5)
        ss = s.rstrip()
        if c > 0.05 and ss.endswith("!"):
            pe += 1
        if c > 0.05 and (ss.endswith("...") or ss.endswith("…") or ss.endswith("?")):
            pt += 1
    near = 0
    idx_neg = np.flatnonzero(neg_any)
    for i in np.flatnonzero(pos):
        if idx_neg.size and np.min(np.abs(idx_neg - i)) <= 4:
            near += 1
    boosted = sum(1 for i in range(1, len(toks)) if pos[i] and toks[i - 1] in boosters)
    negated = sum(1 for i in range(len(toks)) if pos[i] and any(t in negate or t.endswith("n't")
                                                                  for t in toks[max(0, i - 3):i]))
    ec = {e: sum(e in x for x in emo) for e in NRC_EMOTIONS}
    comp = sia.polarity_scores(text)["compound"]
    raw = float(val.sum()) / np.sqrt(val @ val + 15) if toks else 0.0  # VADER's normalisation, no rules
    npos, nneg = int(pos.sum()), int(neg_any.sum())
    return {
        "compound": comp, "sent_max": max(sc), "sent_min": min(sc), "sent_range": max(sc) - min(sc),
        "clause_flips": flips,
        "clause_contrast": float(any(c > 0.3 for c in cc) and any(c < -0.3 for c in cc)),
        "last_minus_first": sc[-1] - sc[0],
        "pos_rate": npos / n, "neg_rate": int(neg_v.sum()) / n,
        "nrc_neg_rate": sum("negative" in e for e in emo) / n,
        "pos_share": npos / (npos + nneg) if npos + nneg else 0.5,
        "both_polar": float(npos > 0 and nneg > 0),
        "strong_pos": np.log1p(int((val >= 2.5).sum())),
        "strong_pos_in_neg_sent": np.log1p(pin),
        "pos_near_neg": np.log1p(near), "boosted_pos": np.log1p(boosted), "negated_pos": np.log1p(negated),
        "lex_vs_vader": raw - comp,
        "pos_excl_sent": np.log1p(pe), "pos_trailing": np.log1p(pt),
        **{f"emo_{e}": ec[e] / n for e in NRC_EMOTIONS},
        "joy_anger_contrast": np.log1p(min(ec["joy"], ec["anger"] + ec["disgust"])),
    }


def build(df: pl.DataFrame) -> pl.DataFrame:
    rows = [row(t) for t in df["text"].to_list()]
    return pl.DataFrame({"uid": df["uid"]}).hstack(pl.DataFrame(rows).select(list(DESCRIPTIONS)))


if __name__ == "__main__":
    main(SET, build, DESCRIPTIONS, {"resources": {"vaderSentiment": "3.3.2", "nrclex (NRC EmoLex data)": "4.1.0"}})
