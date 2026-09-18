"""Feature set `ling_pragmatic`: pragmatic and prosodic-in-text markers of ironic/evaluative stance.

Theory: irony is signalled by markers that frame an utterance as not meant literally (Attardo 2000;
Burgers et al. 2012 "verbal irony markers"): interjections and short evaluative exclamatives
("Oh great.", "Brilliant."), hyperbole / extreme-case formulations (Pomerantz 1986), mock-certainty
epistemic adverbs ("of course", "clearly", "I'm sure"), expectation-violation / echoic markers
("again", "as usual", "who knew"; Kreuz & Glucksberg 1989, Sperber & Wilson echoic mention),
rhetorical and tag questions, scare quotes, typographic prosody (elongation, repeated punctuation,
emphatic caps), imperatives and counterfactual wishes. All lists are closed-class or general
English, not tuned on this dataset. Counts are per sentence or log1p so length is not re-encoded.

Does not duplicate experiments/classical/features.py (q_marks, excl, ellipsis, caps_words, /s,
starts_oh, sarc_phr, second_person are there); these are finer-grained or normalised variants.

  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_pragmatic.py [--eval | --input TEST.csv]
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from _fe import main
from _lex import sentences, tokens, valence

SET = "ling_pragmatic"

INTERJ = r"(?:oh+|ah+|wow+|gosh|golly|boy|whoa|ooh+|aw+|yay|woo+|hmm+|ha|huh|well well|yeah|yes|sure|right|ok(?:ay)?|cool|welp|phew|ugh|meh|pfft|bah)"
EVAL_EXCL = r"(?:great|brilliant|lovely|fantastic|perfect|nice|amazing|wonderful|superb|excellent|awesome|splendid|marvellous|marvelous|genius|classic|charming|delightful|good (?:one|job|stuff)|well done|nice one|love it|how nice|how lovely)"
EXTREME = r"\b(?:always|never|every(?:one|body|thing|where)?|no ?one|nobody|nothing|all|entire(?:ly)?|whole|completely|totally|absolutely|literally|forever|ever|best|worst|most|least|million|billion|thousand|infinite(?:ly)?|perfect(?:ly)?|utter(?:ly)?)\b"
SUPERLATIVE = r"\b(?:\w{3,}est|best|worst|most|least)\b"
INTENS = r"\b(?:so|very|really|such|too|extremely|incredibly|truly|highly|super|mega|bloody|damn|totally|absolutely|utterly|completely|simply|just|quite|remarkably)\b"
CERTAINTY = r"\b(?:of course|clearly|obviously|surely|certainly|definitely|naturally|evidently|undoubtedly|no doubt|i'?m sure|i am sure|i bet|must be|of course)\b"
HEDGE = r"\b(?:i think|i guess|i suppose|maybe|perhaps|probably|possibly|apparently|seemingly|seems?|kind of|sort of|kinda|sorta|allegedly|supposedly|i reckon|reckon)\b"
EXPECT = r"\b(?:again|as usual|as always|yet again|once again|for once|who knew|who would(?:'ve| have) (?:thought|guessed)|what a surprise|surprise surprise|shock(?:ing|er)?|colou?r me|never saw|saw that coming|in other news|how convenient|how surprising|imagine that|fancy that|funny how|funny that)\b"
COUNTERFACT = r"\b(?:if only|i wish|would(?:'ve| have) been|could(?:'ve| have) been|should(?:'ve| have)|imagine if|what if)\b"
RHET_START = r"^(?:why|how|what|who|where|when|is|isn't|are|aren't|do|don't|does|doesn't|did|didn't|can|can't|could|couldn't|would|wouldn't|should|shouldn't|will|won't|has|hasn't|have|haven't|was|wasn't)\b"
TAG_Q = r"(?:,\s*(?:isn't it|aren't they|don't they|doesn't it|didn't they|wasn't it|won't they|right|eh|innit|no|yeah|huh)\s*\?)"
IMPER = r"^(?:just |please |go on,? |oh,? )?(?:imagine|try|look|remember|tell|let|let's|go|get|stop|check|enjoy|keep|give|make|take|think|wait|watch|vote|pay|blame|thank|bring|consider|pretend|picture|guess|don't|do|never|shut|calm|relax|ask|read|see)\b"
LAUGH = r"\b(?:lol+|lmao+|lmfao|rofl|ha(?:ha)+h?|he(?:he)+|kek|heh)\b"
SCARE = r"(?:\"[^\"\n]{1,30}\"|(?<![a-z])'[^'\n]{1,30}'(?![a-z]))"

DESCRIPTIONS = {
    "interj_initial": "share of sentences starting with an interjection/discourse particle (oh, wow, yeah, sure, well well...)",
    "eval_exclamative": "short (<= 4 token) sentences that are a positive evaluative exclamative (Brilliant. / Oh great!), log1p",
    "what_a": "'what a/an X' and 'how ADJ' exclamatives, log1p",
    "pos_one_word_sent": "one- or two-token sentences whose tokens are VADER-positive, log1p",
    "extreme_case": "extreme-case formulations / hyperbole words per token (always, never, everyone, literally, million...)",
    "superlative": "superlatives per token (-est, best, worst, most)",
    "intensifier": "intensifiers per token (so, very, really, bloody, totally...)",
    "certainty": "mock-certainty epistemic markers per sentence (of course, clearly, obviously, I'm sure, must be)",
    "hedge": "hedges per sentence (I think, maybe, apparently, sort of, reckon)",
    "expectation": "expectation-violation / echoic markers, log1p (again, as usual, who knew, what a surprise, how convenient)",
    "counterfactual": "counterfactual/wish constructions, log1p (if only, I wish, should have, imagine if)",
    "q_sent_share": "share of sentences ending in '?'",
    "rhetorical_q": "questions opening with wh-word or auxiliary, per sentence",
    "neg_q": "questions containing a negated auxiliary (isn't, don't, why can't...), log1p",
    "tag_q": "tag questions (', isn't it?', ', right?', ', eh?'), log1p",
    "scare_quote": "short (<= 30 char) quoted spans: echoic mention / scare quotes, log1p",
    "reply_quote": "lines starting with '>' (quoting another comment), log1p",
    "trailing_ellipsis": "text ends in '...' or '…'",
    "multi_punct": "runs of repeated/mixed ! and ? (!!, ?!, ???), log1p",
    "elongation": "words with a letter repeated 3+ times (sooo, yesss), log1p",
    "caps_emph": "all-caps words of >= 4 letters (emphatic, not acronyms), log1p",
    "excl_sent_share": "share of sentences ending in '!'",
    "imperative": "share of sentences opening with an imperative verb/let's/please",
    "laughter": "laughter tokens (lol, haha, lmao, heh), log1p",
    "starts_lower": "text starts with a lowercase letter (casual register)",
    "no_final_punct": "text ends without . ! ? (casual register)",
    "you_subject": "'you' + verb/auxiliary constructions per token (you must, you think, you're)",
    "we_they": "we/us/our AND they/them/their both present (in-group vs out-group framing)",
}


def row(text: str) -> dict:
    lo = text.lower()
    toks = tokens(text)
    n = max(len(toks), 1)
    sents = sentences(text)
    ns = len(sents)
    s_lo = [s.lower().strip(" \"'*-(") for s in sents]
    ev = 0
    pos1 = 0
    for s in s_lo:
        st = tokens(s)
        if len(st) <= 4 and re.match(rf"^(?:{INTERJ}[,!. ]+)?(?:that'?s |that is |this is |just |how |so |oh so |really )?{EVAL_EXCL}\b", s):
            ev += 1
        if 1 <= len(st) <= 2 and all(valence(w) > 0 for w in st):
            pos1 += 1
    qs = [s for s in s_lo if s.rstrip().endswith("?")]
    return {
        "interj_initial": sum(bool(re.match(rf"^{INTERJ}\b", s)) for s in s_lo) / ns,
        "eval_exclamative": np.log1p(ev),
        "what_a": np.log1p(len(re.findall(r"\bwhat an? \w+|\bhow (?:very |so )?\w+(?:ly)? (?:is|was|are|of)\b|\bhow \w+[!]", lo))),
        "pos_one_word_sent": np.log1p(pos1),
        "extreme_case": len(re.findall(EXTREME, lo)) / n,
        "superlative": len(re.findall(SUPERLATIVE, lo)) / n,
        "intensifier": len(re.findall(INTENS, lo)) / n,
        "certainty": len(re.findall(CERTAINTY, lo)) / ns,
        "hedge": len(re.findall(HEDGE, lo)) / ns,
        "expectation": np.log1p(len(re.findall(EXPECT, lo))),
        "counterfactual": np.log1p(len(re.findall(COUNTERFACT, lo))),
        "q_sent_share": len(qs) / ns,
        "rhetorical_q": sum(bool(re.match(RHET_START, q)) for q in qs) / ns,
        "neg_q": np.log1p(sum(bool(re.search(r"\b\w+n't\b|\bnot\b", q)) for q in qs)),
        "tag_q": np.log1p(len(re.findall(TAG_Q, lo))),
        "scare_quote": np.log1p(len(re.findall(SCARE, text))),
        "reply_quote": np.log1p(len(re.findall(r"(?m)^\s*>", text))),
        "trailing_ellipsis": float(bool(re.search(r"(?:\.\.\.|…)\W*$", text))),
        "multi_punct": np.log1p(len(re.findall(r"[!?]{2,}", text))),
        "elongation": np.log1p(len(re.findall(r"\b\w*([a-z])\1\1\w*\b", lo))),
        "caps_emph": np.log1p(len(re.findall(r"\b[A-Z]{4,}\b", text)) if not text.isupper() else 0),
        "excl_sent_share": sum(s.rstrip().endswith("!") for s in sents) / ns,
        "imperative": sum(bool(re.match(IMPER, s)) for s in s_lo) / ns,
        "laughter": np.log1p(len(re.findall(LAUGH, lo))),
        "starts_lower": float(bool(re.match(r"^\W*[a-z]", text))),
        "no_final_punct": float(not re.search(r"[.!?…)\"']\s*$", text)),
        "you_subject": len(re.findall(r"\byou(?:'re|'ll|'d|'ve)?\s+(?:are|were|must|should|can|could|will|would|think|know|want|need|have|do|don't|really|clearly|obviously)\b|\byou(?:'re|'ll|'d|'ve)\b", lo)) / n,
        "we_they": float(bool(re.search(r"\b(?:we|us|our)\b", lo)) and bool(re.search(r"\b(?:they|them|their)\b", lo))),
    }


def build(df: pl.DataFrame) -> pl.DataFrame:
    rows = [row(t) for t in df["text"].to_list()]
    return pl.DataFrame({"uid": df["uid"]}).hstack(pl.DataFrame(rows).select(list(DESCRIPTIONS)))


if __name__ == "__main__":
    main(SET, build, DESCRIPTIONS, {"resources": {"vaderSentiment": "3.3.2 (valence for pos_one_word_sent)"}})
