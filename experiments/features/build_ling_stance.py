"""Feature set `ling_stance`: evaluation targets (who/what the text is about) and target x polarity.

Theory: sarcasm and negative sentiment in online political/social talk are aimed at targets of
public grievance (governments, parties, politicians, institutions, corporations, landlords); an
ironic *positive* statement about such a target is the prototypical sarcastic move ("Great job,
Government"). Target mentions alone are topical (the embeddings see them); the new information is
the product "positive surface polarity x grievance target", which a linear stacker cannot form.

Lists are GENERAL categories, not mined from the labels: the major parties and current/recent
leaders of both countries (as named by the task lead), generic political/institutional nouns,
corporate/economic actors, and grievance topics (cost of living, housing, tax). Documented below.

  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_stance.py [--eval | --input TEST.csv]
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from _fe import main
from _lex import tokens, vader, valence

SET = "ling_stance"

# public political actors, both varieties (parties, leaders), plus generic political nouns
POLITICAL = (r"\b(?:tor(?:y|ies)|conservatives?|labou?r|lnp|liberals?|coalition|nationals|greens?|one nation|reform uk|"
             r"lib ?dems?|snp|dutton|albo|albanese|morrison|scomo|abbott|turnbull|hanson|sunak|starmer|truss|boris|"
             r"johnson|farage|corbyn|theresa may|cameron|badenoch|braverman|politicians?|polies|mps?|senators?|ministers?|"
             r"prime minister|premier|chancellor|treasurer|cabinet|parliament|westminster|canberra|government|govt|"
             r"gov|election|voters?|lefties|leftists?|right[- ]wing|woke|leftie|lefty)\b")
INSTITUTION = (r"\b(?:nhs|ato|hmrc|centrelink|ndis|council|councils|councillors?|police|cops|coppers|courts?|judges?|"
               r"bbc|abc|media|news ?corp|murdoch|the sun|daily mail|universit(?:y|ies)|schools?|hospitals?|"
               r"public service|civil service|department|dept|regulators?|ofcom|ofgem|ofwat|accc|immigration|border force|"
               r"royal family|monarchy|the palace)\b")
CORPORATE = (r"\b(?:compan(?:y|ies)|corporations?|corporates?|ceos?|executives?|shareholders?|profits?|billionaires?|"
             r"millionaires?|the rich|landlords?|real estate|developers?|estate agents?|banks?|bankers?|energy compan(?:y|ies)|"
             r"supermarkets?|airlines?|telcos?|water compan(?:y|ies)|big business|big tech|investors?|management|bosses?)\b")
GRIEVANCE = (r"\b(?:tax(?:es|ed)?|rent|rents|housing|mortgages?|cost of living|prices?|inflation|bills?|wages?|salar(?:y|ies)|"
             r"cuts?|funding|strikes?|debt|interest rates?|pension|immigration|migrants?|crisis|shortage|waiting lists?|"
             r"potholes?|delays?|fees?|fines?|scandal|corruption|lies|lying|promises?)\b")
PRAISE_OF_TARGET = r"\b(?:well done|good job|great job|nice one|bravo|genius|thanks?|thank you|cheers|love (?:it|that|how)|so glad|can'?t wait)\b"

DESCRIPTIONS = {
    "political": "political actors (parties, leaders, government, MPs, election...) per token, sqrt",
    "institution": "public institutions (NHS, ATO, council, police, courts, media, universities...) per token, sqrt",
    "corporate": "corporate/economic actors (companies, CEOs, landlords, banks, billionaires...) per token, sqrt",
    "grievance": "grievance topics (tax, rent, prices, bills, cuts, delays, crisis...) per token, sqrt",
    "any_target": "1 if any political/institutional/corporate target is mentioned",
    "they_rate": "third-person plural pronouns (they/them/their) per token: othering of an out-group",
    "target_x_pos": "any_target x VADER positive-token share (positive talk about a grievance target)",
    "target_x_compound": "any_target x VADER compound",
    "grievance_x_pos": "grievance-topic present x VADER positive-token share",
    "praise_x_target": "praise/thanks formulas (well done, genius, thanks, can't wait) in a text with a target, log1p",
    "target_pos_window": "positive words within 5 tokens of a target mention, log1p",
    "target_neg_window": "negative words within 5 tokens of a target mention, log1p",
}


def _hits(pattern: str, lo: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(pattern, lo)]


def row(text: str) -> dict:
    sia, _, _ = vader()
    lo = text.lower()
    toks = tokens(text)
    n = max(len(toks), 1)
    counts = {k: len(_hits(p, lo)) for k, p in (("political", POLITICAL), ("institution", INSTITUTION),
                                                ("corporate", CORPORATE), ("grievance", GRIEVANCE))}
    tgt = counts["political"] + counts["institution"] + counts["corporate"] > 0
    val = [valence(w) for w in toks]
    pos_share = sum(v > 0 for v in val) / n
    comp = sia.polarity_scores(text)["compound"]
    # token positions of targets: map char offsets to token index via running token spans
    spans = [m.span() for m in re.finditer(r"[a-z]+(?:'[a-z]+)?", lo)]
    starts = np.array([s for s, _ in spans]) if spans else np.zeros(0, int)
    tpos = []
    for p in (POLITICAL, INSTITUTION, CORPORATE):
        for s, _ in _hits(p, lo):
            if len(starts):
                tpos.append(int(np.searchsorted(starts, s, side="right") - 1))
    pw = nw = 0
    if tpos:
        tp = np.array(tpos)
        for i, v in enumerate(val):
            if v != 0 and np.min(np.abs(tp - i)) <= 5:
                pw += v > 0
                nw += v < 0
    return {
        **{k: np.sqrt(counts[k] / n) for k in ("political", "institution", "corporate", "grievance")},
        "any_target": float(tgt),
        "they_rate": len(re.findall(r"\b(?:they|them|their|theirs|they're|they've|they'll)\b", lo)) / n,
        "target_x_pos": float(tgt) * pos_share,
        "target_x_compound": float(tgt) * comp,
        "grievance_x_pos": float(counts["grievance"] > 0) * pos_share,
        "praise_x_target": np.log1p(len(re.findall(PRAISE_OF_TARGET, lo))) if tgt else 0.0,
        "target_pos_window": np.log1p(pw),
        "target_neg_window": np.log1p(nw),
    }


def build(df: pl.DataFrame) -> pl.DataFrame:
    rows = [row(t) for t in df["text"].to_list()]
    return pl.DataFrame({"uid": df["uid"]}).hstack(pl.DataFrame(rows).select(list(DESCRIPTIONS)))


if __name__ == "__main__":
    main(SET, build, DESCRIPTIONS, {"resources": {"vaderSentiment": "3.3.2"},
                                    "lists": {"POLITICAL": POLITICAL, "INSTITUTION": INSTITUTION,
                                              "CORPORATE": CORPORATE, "GRIEVANCE": GRIEVANCE,
                                              "PRAISE_OF_TARGET": PRAISE_OF_TARGET}})
