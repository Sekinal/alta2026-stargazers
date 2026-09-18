"""Feature set `ling_core`: a compact, theory-chosen subset of the four ling_* sets (15 features).

Small on purpose: the 28-feature sets overfit a per-segment stacker on UK reddit (1,143 rows, 248
sarcastic). One or two features per theory, picked a priori (by construct, not per-feature label
screening):
  incongruity (Riloff 2013)     pos_share, sent_range, clause_contrast, pos_near_neg, strong_pos_in_neg_sent
  irony markers (Burgers 2012)  interj_initial, eval_exclamative, certainty, expectation, rhetorical_q, scare_quote
  stance/target                 target_x_pos, they_rate
  register                      fw_i, mean_sent_len

  uv run --with vaderSentiment --with nrclex python experiments/features/build_ling_core.py [--eval | --input TEST.csv]
"""

from __future__ import annotations

import polars as pl

import build_ling_polarity as pol
import build_ling_pragmatic as prag
import build_ling_stance as sta
import build_ling_style as sty
from _fe import main

SET = "ling_core"
PICK = {
    pol: ("pos_share", "sent_range", "clause_contrast", "pos_near_neg", "strong_pos_in_neg_sent"),
    prag: ("interj_initial", "eval_exclamative", "certainty", "expectation", "rhetorical_q", "scare_quote"),
    sta: ("target_x_pos", "they_rate"),
    sty: ("fw_i", "mean_sent_len"),
}
DESCRIPTIONS = {c: f"{m.SET}: {m.DESCRIPTIONS[c]}" for m, cols in PICK.items() for c in cols}


def build(df: pl.DataFrame) -> pl.DataFrame:
    texts = df["text"].to_list()
    rows = [{c: v for m, cols in PICK.items() for c, v in m.row(t).items() if c in cols} for t in texts]
    return pl.DataFrame({"uid": df["uid"]}).hstack(pl.DataFrame(rows).select(list(DESCRIPTIONS)))


if __name__ == "__main__":
    main(SET, build, DESCRIPTIONS, {"resources": {"vaderSentiment": "3.3.2", "nrclex (NRC EmoLex data)": "4.1.0"}})
