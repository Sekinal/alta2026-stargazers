"""Builder for zs_nli_x: extra NLI hypotheses (same model as zs_nli) aimed at deadpan / understated
sarcasm and at what non-sarcastic reddit posts do instead (facts, genuine questions, experiences).

  flock .gpu.lock uv run python experiments/features/build_zs_nli_x.py --input pool|TEST.csv --out PARQUET
"""

import argparse
import time

import numpy as np

from build_zs_nli import MODEL, score_hypotheses
from fe_common import default_out, load_input, write_set

NAME = "zs_nli_x"
CONCEPTS = {
    "passive_aggr": ["The author is being passive-aggressive.", "The author is making a snide remark."],
    "disbelief": ["The author is expressing disbelief.", "The author finds something ridiculous."],
    "fake_enthusiasm": ["The author pretends to be enthusiastic.", "The author's excitement is fake."],
    "understatement": ["The author is using understatement.",
                       "The author says something mild but means something strong."],
    "deadpan": ["The author makes a dry, deadpan remark.", "The author uses dry humour."],
    "mock_quote": ["The author repeats someone's words in order to mock them.",
                   "The author imitates someone to make them look foolish."],
    "literal": ["The text should be taken literally.", "The author's words match their true feelings."],
    "factual": ["The author is stating a fact.", "This text gives factual information."],
    "genuine_q": ["The author asks a genuine question.", "The author wants to know the answer to a question."],
    "personal": ["The author describes a personal experience.", "The author tells a story about their own life."],
    "disagree": ["The author disagrees with someone.", "The author is arguing against an opinion."],
    "criticise_gov": ["The author criticises the government.", "The author criticises politicians."],
    "venting": ["The author is venting.", "The author is letting off steam."],
}


def build(inp, out):
    t0 = time.time()
    df = load_input(inp)
    S = score_hypotheses(df, [h for hs in CONCEPTS.values() for h in hs])
    feats = {c: np.mean([S[h] for h in hs], 0) for c, hs in CONCEPTS.items()}
    feats["fake_minus_literal"] = feats["fake_enthusiasm"] - feats["literal"]
    desc = {c: f"mean entailment log-odds over {len(hs)} paraphrases: {hs}" for c, hs in CONCEPTS.items()}
    desc["fake_minus_literal"] = "fake_enthusiasm - literal"
    write_set(NAME, df, feats, out or default_out(NAME, df), desc, [MODEL], t0, {"hypotheses": CONCEPTS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.input, a.out)
