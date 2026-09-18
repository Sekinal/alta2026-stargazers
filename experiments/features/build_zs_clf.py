"""Zero-shot outputs of pretrained classifiers trained on OTHER data (none on BESSTIE), as features.

Sets (each its own parquet; build_<set>.py are thin wrappers around this file):
  zs_irony   irony / sarcasm / humour detectors
  zs_affect  sentiment (two domains) and emotion
  zs_tox     toxicity, hate, offensive, politeness, formality
  zs_topic   tweet topic (multi-label)
Output per model: binary -> logit margin (log-odds of the positive class); single-label multi-class
-> log-softmax per class (+ expected stars for the 5-star model); multi-label -> raw logits.
Texts over 512 tokens are head+tail truncated (3/4 head, 1/4 tail).

  flock .gpu.lock uv run python experiments/features/build_zs_clf.py --set zs_irony --input pool \
      --out experiments/features/store/zs_irony__pool.parquet
"""

from __future__ import annotations

import argparse
import string
import time

import numpy as np

from fe_common import default_out, cached, classifier_logits, load_input, write_set

# key -> (hf id, kind, class names or None, preprocess, what it is)
MODELS = {
    "irony_deb": ("Elron/deberta-v3-large-irony", "bin", None, None,
                  "DeBERTa-v3-large fine-tuned on TweetEval irony: log-odds ironic"),
    "sarc_dlg": ("hallisky/sarcasm-classifier-gpt4-data", "bin", None, None,
                 "RoBERTa-large sarcasm classifier on GPT-4 dialogue data: log-odds more sarcastic"),
    "sarc_head": ("helinivan/english-sarcasm-detector", "bin", None, "headline",
                  "BERT sarcasm detector trained on news headlines (lowercased, no punctuation): log-odds sarcastic"),
    "sarc_reddit": ("ScandinavianMrT/distilbert-SARC", "bin", None, None,
                    "DistilBERT fine-tuned on SARC (self-annotated Reddit sarcasm, /s): log-odds sarcastic"),
    "isarcasm": ("w11wo/deberta-v3-base-isarcasm", "bin", None, None,
                 "DeBERTa-v3-base fine-tuned on iSarcasmEval (author-labelled tweets): log-odds sarcastic"),
    "humor": ("Humor-Research/humor-detection-comb-23", "bin", None, None,
              "RoBERTa humour detector (combined joke datasets): log-odds humorous"),
    "sent_siebert": ("siebert/sentiment-roberta-large-english", "bin", None, None,
                     "SiEBERT RoBERTa-large sentiment (15 English datasets): log-odds positive"),
    "stars": ("nlptown/bert-base-multilingual-uncased-sentiment", "multi",
              ["s1", "s2", "s3", "s4", "s5"], None, "product-review star rating (1-5): log-softmax per star"),
    "emo7": ("j-hartmann/emotion-english-distilroberta-base", "multi",
             ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"], None,
             "Ekman-7 emotion (DistilRoBERTa): log-softmax per emotion"),
    "tox": ("unitary/unbiased-toxic-roberta", "multilabel",
            ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"],
            None, "Jigsaw unbiased toxicity RoBERTa: logit per attribute (identity heads dropped)"),
    "hate": ("cardiffnlp/twitter-roberta-base-hate-latest", "bin", None, None,
             "Twitter RoBERTa hate speech: log-odds hate"),
    "offensive": ("cardiffnlp/twitter-roberta-base-offensive", "bin", None, None,
                  "Twitter RoBERTa offensive (TweetEval): log-odds offensive"),
    "polite": ("Intel/polite-guard", "multi", ["polite", "somewhat_polite", "neutral", "impolite"], None,
               "Intel polite-guard BERT: log-softmax per politeness class"),
    "formal": ("s-nlp/roberta-base-formality-ranker", "bin", None, None,
               "RoBERTa formality ranker (GYAFC + Pavlick): log-odds formal"),
    "topic": ("cardiffnlp/tweet-topic-21-multi", "multilabel",
              ["arts_culture", "business", "celebrity", "daily_life", "family", "fashion", "film_tv",
               "fitness_health", "food_dining", "gaming", "learning", "music", "news_social_concern",
               "hobbies", "relationships", "science_tech", "sports", "travel", "youth_student"],
              None, "TweetTopic-21 multi-label topic: logit per topic"),
}
# repos whose own tokenizer files are broken (humor: every text encodes to <s></s>)
TOKENIZER = {"humor": "FacebookAI/roberta-large"}
SETS = {
    "zs_irony": ["irony_deb", "sarc_dlg", "sarc_head", "humor", "sarc_reddit", "isarcasm"],
    "zs_affect": ["sent_siebert", "stars", "emo7"],
    "zs_tox": ["tox", "hate", "offensive", "polite", "formal"],
    "zs_topic": ["topic"],
}


def prep(texts: list[str], how: str | None) -> list[str]:
    if how == "headline":
        tbl = str.maketrans("", "", string.punctuation)
        return [t.lower().translate(tbl).strip() for t in texts]
    return texts


def features(key: str, df) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    hf, kind, names, how, desc = MODELS[key]
    L = cached(f"clf_{key}", df, lambda: classifier_logits(hf, prep(df["text"].to_list(), how),
                                                                    tok_id=TOKENIZER.get(key)))
    out, d = {}, {}
    if kind == "bin":
        out[key] = L[:, 1] - L[:, 0]
        d[key] = desc
    elif kind == "multi":
        lp = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
        for i, n in enumerate(names):
            out[f"{key}_{n}"] = lp[:, i]
            d[f"{key}_{n}"] = f"{desc} [{n}]"
        if key == "stars":
            out["stars_mean"] = (np.exp(lp) * np.arange(1, 6)).sum(1)
            d["stars_mean"] = "expected star rating (1-5)"
    else:
        for i, n in enumerate(names):
            out[f"{key}_{n}"] = L[:, i]
            d[f"{key}_{n}"] = f"{desc} [{n}]"
    return out, d


def build(set_name: str, inp: str, out: str | None) -> None:
    t0 = time.time()
    df = load_input(inp)
    feats, desc = {}, {}
    for key in SETS[set_name]:
        f, d = features(key, df)
        feats.update(f)
        desc.update(d)
    out = out or default_out(set_name, df)
    write_set(set_name, df, feats, out, desc, [MODELS[k][0] for k in SETS[set_name]], t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.set, a.input, a.out)


if __name__ == "__main__":
    main()
