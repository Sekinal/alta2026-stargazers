"""Perturbation / prompt-paraphrase robustness of a fine-tuned holdout adapter.

  CUDA_VISIBLE_DEVICES=1 uv run python experiments/llm/robust.py --run llm_ft_R1_s0 [--fold 0]

Re-scores the held-out fold with models/<run>/fold<k> under realistic shifts and reports, per
condition, the within-holdout segment+ukg score, sarc-UK, sent-AU, UK-reddit sarcasm AUC and
AU-reddit sentiment AUC, plus the drop from the clean condition. Writes
experiments/llm/out/<run>/robust_fold<k>.parquet.
"""

from __future__ import annotations

import argparse
import random
import re
import string
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import llm_common  # noqa: E402
from ft import OUT, PROMPTS, Head, predict  # noqa: E402
from llm_common import ROOT, load_model, verbalizer_ids  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import _frame_to_eval  # noqa: E402
from alta.validation.metrics import components  # noqa: E402

UK2AU = {  # spelling is shared (both -our/-ise); swap slang/lexicon instead
    "mate": "cobber", "football": "footy", "lorry": "truck", "crisps": "chips", "flat": "unit",
    "petrol station": "servo", "afternoon": "arvo", "pub": "pub", "Tories": "Liberals",
    "Tory": "Liberal", "Labour": "Labor", "NHS": "Medicare", "Westminster": "Canberra",
    "council": "shire", "quid": "bucks", "pounds": "dollars", "£": "$", "Brexit": "the referendum",
    "London": "Sydney", "Manchester": "Melbourne",
}
AU2UK = {v: k for k, v in UK2AU.items() if v not in ("pub",)}


def swap_words(text: str, table: dict[str, str]) -> str:
    for a, b in sorted(table.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![\w$£]){re.escape(a)}(?!\w)", b, text)
    return text


def typos(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        r = rng.random()
        if ch.isalpha() and r < rate:
            op = rng.randrange(3)
            if op == 0:
                continue  # deletion
            if op == 1:
                out.append(ch + ch)  # duplication
                continue
            out.append(rng.choice(string.ascii_lowercase))  # substitution
            continue
        out.append(ch)
    return "".join(out)


_NT_ADD = re.compile(r"(\w)n't\b")


def perturb(df: pl.DataFrame, cond: str, seed: int = 0) -> pl.DataFrame:
    rng = random.Random(seed)
    texts, var = df["text"].to_list(), df["variety"].cast(pl.String).to_list()
    src = df["source"].cast(pl.String).to_list()
    if cond == "lower":
        texts = [t.lower() for t in texts]
    elif cond == "nopunct":
        texts = [re.sub(r"[^\w\s']", "", t) for t in texts]
    elif cond == "typos3":
        texts = [typos(t, 0.03, rng) for t in texts]
    elif cond == "dialect_swap":
        texts = [swap_words(t, UK2AU if v == "en-UK" else AU2UK) for t, v in zip(texts, var)]
    elif cond == "nt_artefacts_add":  # reddit gets the google tokeniser artefacts
        texts = [_NT_ADD.sub(r"\1 n't", t) if s == "reddit" else t for t, s in zip(texts, src)]
    elif cond == "nt_artefacts_raw":  # google keeps its artefacts (normalise() is bypassed)
        pass
    return df.with_columns(pl.Series("text", texts))


PARAPHRASES = {
    "para1": ("Does this text use sarcasm? Reply Yes or No.",
              "What is the overall sentiment of this text, positive or negative? Reply Positive or Negative."),
    "para2": ("Would a careful annotator label this text as sarcastic? Answer Yes or No.",
              "Would a careful annotator label the writer's attitude as positive or negative? "
              "Answer Positive or Negative."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-text-tokens", type=int, default=192)
    a = ap.parse_args()
    from peft import PeftModel

    pool = load_pool()
    hold = pool.filter(pl.col("fold") == a.fold)
    tok, model = load_model(a.model)
    ids = {}
    for task in ("sarc", "sent"):
        p, n = verbalizer_ids(tok, task)
        ids[task] = (p[0], n[0])
    uniq = sorted({i for pn in ids.values() for i in pn})
    head = Head(model.lm_head.weight[uniq].detach().clone(),
                {t: (uniq.index(p), uniq.index(n)) for t, (p, n) in ids.items()}).to(0)
    model.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    inner = model.model
    model = PeftModel.from_pretrained(model, ROOT / "models" / a.run / f"fold{a.fold}")
    model.eval()

    orig_norm, orig_msg, orig_desc = llm_common.normalise, llm_common.user_message, dict(llm_common.SOURCE_DESC)
    conds = ["clean", "lower", "nopunct", "typos3", "dialect_swap", "nt_artefacts_add",
             "nt_artefacts_raw", "no_slot", "para1", "para2"]
    rows = []
    for cond in conds:
        df = perturb(hold, cond)
        llm_common.normalise = (lambda t, s: t) if cond == "nt_artefacts_raw" else orig_norm
        llm_common.SOURCE_DESC.clear()
        llm_common.SOURCE_DESC.update(
            {k: "a short text from the internet" for k in orig_desc} if cond == "no_slot" else orig_desc)
        if cond in PARAPHRASES:
            qs, qn = PARAPHRASES[cond]

            def msg(prompt, variety, source, text, shots=None, _qs=qs, _qn=qn):
                m = orig_msg(prompt, variety, source, text, shots)
                m = m.replace("Is the text sarcastic? Answer Yes or No.", _qs)
                return m.replace("Is the overall sentiment of the text positive or negative? "
                                 "Answer Positive or Negative.", _qn)
            llm_common.user_message = msg
        else:
            llm_common.user_message = orig_msg
        m = {t: predict(inner, head, tok, df, t, a.max_text_tokens) for t in ("sarc", "sent")}
        sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
        ev = _frame_to_eval(hold.with_columns(pl.Series("p_sent", sig(m["sent"])),
                                              pl.Series("p_sarc", sig(m["sarc"]))), True, {})
        pred, _ = policies.crossfit(policies.DEFAULT_POLICY, ev)
        c = components(ev, pred)
        from sklearn.metrics import roc_auc_score
        ukr = (ev.variety == 1) & (ev.source == 1)
        aur = (ev.variety == 0) & (ev.source == 1)
        rows.append({"cond": cond, **{k: c[k] for k in ("score", "sent-AU", "sent-UK", "sarc-AU", "sarc-UK")},
                     "auc_sarc_ukr": roc_auc_score(ev.y["sarc"][ukr], ev.p["sarc"][ukr]),
                     "auc_sent_aur": roc_auc_score(ev.y["sent"][aur], ev.p["sent"][aur])})
        print(rows[-1], flush=True)
    res = pl.DataFrame(rows)
    base = res.filter(pl.col("cond") == "clean")
    res = res.with_columns((pl.col("score") - base["score"][0]).alias("d_score"),
                           (pl.col("sarc-UK") - base["sarc-UK"][0]).alias("d_sarc_uk"),
                           (pl.col("auc_sarc_ukr") - base["auc_sarc_ukr"][0]).alias("d_auc_sarc_ukr"))
    res.write_parquet(OUT / a.run / f"robust_fold{a.fold}.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=220):
        print(res.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
