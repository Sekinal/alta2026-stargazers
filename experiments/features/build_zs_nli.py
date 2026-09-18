"""Zero-shot NLI hypothesis scores (label-free): entailment log-odds of pragmatic hypotheses.

Premise = review/post text (head+tail truncated to MAX_LEN tokens), hypothesis = a statement about the
author. Each concept has several paraphrases; the feature is the mean entailment log-odds over them
(robust to wording). Contrast features: praise - complain, sincere - opposite.
Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (MNLI/FEVER/ANLI/... + synthetic data; released
before BESSTIE). Per-hypothesis scores are cached, so adding hypotheses only scores the new ones.

  flock .gpu.lock uv run python experiments/features/build_zs_nli.py --input pool|TEST.csv --out PARQUET
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np

from fe_common import CACHE, default_out, _config, fingerprint, free_gpu, head_tail, load_input, write_set

NAME = "zs_nli"
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MAX_LEN = 256
CONCEPTS = {
    "opposite": ["The author means the opposite of what they literally say.",
                 "This text is sarcastic.", "The writer is being ironic."],
    "mocking": ["The author is mocking someone.", "The author is ridiculing something.",
                "This text makes fun of someone."],
    "sincere": ["The author is being sincere.", "The author means exactly what they say.",
                "This is a genuine, straightforward statement."],
    "complain": ["The author is complaining.", "The author is unhappy about something.", "This is a complaint."],
    "praise": ["The author is praising something.", "The author is happy with something.",
               "This is a compliment."],
    "ironic_praise": ["This is ironic praise.",
                      "The author pretends to be positive but is actually negative.",
                      "The compliment in this text is not meant seriously."],
    "joking": ["The author is joking.", "This text is meant to be funny.", "The author is being playful."],
    "angry": ["The author is angry.", "The author is frustrated.", "The author is annoyed."],
    "politics": ["This is about politics.", "This text discusses the government or politicians.",
                 "This text is about a political party."],
    "exaggerate": ["The author is exaggerating.", "This text contains an exaggeration.",
                   "The author overstates things for effect."],
    "rhetorical": ["The author asks a rhetorical question.", "The author asks a question they do not expect answered."],
    "positive": ["The author feels positive.", "This text expresses a positive opinion."],
    "negative": ["The author feels negative.", "This text expresses a negative opinion."],
    "advice": ["The author is asking for advice or information.", "The author is giving helpful advice."],
    "contempt": ["The author is contemptuous.", "The author looks down on someone."],
    "cynical": ["The author is cynical.", "The author expects things to go badly."],
}
DESC = {
    "opposite": "says the opposite of what it means / sarcastic / ironic",
    "mocking": "mocks or ridicules someone", "sincere": "sincere, means what it says",
    "complain": "complaint", "praise": "praise / compliment", "ironic_praise": "positive surface, negative intent",
    "joking": "joking / playful", "angry": "angry / frustrated", "politics": "about politics",
    "exaggerate": "exaggeration / hyperbole", "rhetorical": "rhetorical question",
    "positive": "positive opinion", "negative": "negative opinion", "advice": "asks for or gives advice",
    "contempt": "contempt / condescension", "cynical": "cynical / pessimistic",
}


def score_hypotheses(df, hyps: list[str], bs: int = 64, model_id: str = MODEL) -> dict[str, np.ndarray]:
    """hypothesis -> entailment log-odds per row (cached per model x hypothesis).

    2-class models: logit_ent - logit_not; 3-class (MNLI) models: logit_ent - logsumexp(others).
    """
    MODEL = model_id
    fp = fingerprint(df)
    key = lambda h: CACHE / f"nli_{hashlib.sha1((MODEL + h).encode()).hexdigest()[:12]}__{fp}.npy"
    out = {h: np.load(key(h)) for h in hyps if key(h).exists()}
    todo = [h for h in hyps if h not in out]
    if not todo:
        return out
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_grad_enabled(False)
    cfg = _config(MODEL)
    ent = {v.lower(): int(k) for k, v in cfg.id2label.items()}["entailment"]
    tok = AutoTokenizer.from_pretrained(MODEL, config=cfg)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=cfg, dtype=torch.float16).cuda().eval()
    texts, n_tok = head_tail(df["text"].to_list(), tok, MAX_LEN - 32)
    order = np.argsort(-n_tok)
    for h in todo:
        t0 = time.time()
        s = np.zeros(len(texts), np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], [h] * len(idx), padding=True, truncation="only_first",
                      max_length=MAX_LEN, return_tensors="pt").to("cuda")
            lg = model(**enc).logits.float()
            rest = torch.cat([lg[:, :ent], lg[:, ent + 1:]], 1)
            s[idx] = (lg[:, ent] - torch.logsumexp(rest, 1)).cpu().numpy()
        assert np.isfinite(s).all()
        np.save(key(h), s)
        out[h] = s
        print(f"[nli] {time.time() - t0:.0f}s  {h}", flush=True)
    del model
    free_gpu()
    return out


def build(inp: str, out: str | None, name: str = NAME, model_id: str = MODEL) -> None:
    t0 = time.time()
    df = load_input(inp)
    S = score_hypotheses(df, [h for hs in CONCEPTS.values() for h in hs], model_id=model_id)
    feats = {c: np.mean([S[h] for h in hs], 0) for c, hs in CONCEPTS.items()}
    feats["praise_minus_complain"] = feats["praise"] - feats["complain"]
    feats["opposite_minus_sincere"] = feats["opposite"] - feats["sincere"]
    desc = {c: f"mean entailment log-odds over {len(hs)} paraphrases: {DESC[c]}" for c, hs in CONCEPTS.items()}
    desc["praise_minus_complain"] = "praise - complain (surface polarity via NLI)"
    desc["opposite_minus_sincere"] = "opposite - sincere (irony contrast)"
    write_set(name, df, feats, out or default_out(name, df), desc, [model_id], t0,
              {"hypotheses": CONCEPTS, "max_len": MAX_LEN})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.input, a.out)
