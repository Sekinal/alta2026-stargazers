"""Causal-LM features (label-free): per-token surprisal statistics and cloze log-odds.

Model: Qwen/Qwen2.5-1.5B (base, not instruction-tuned), fp16.
1. Surprisal of the text given a neutral prefix ("Online comment:\\n"): mean, sd, max, p90, mean of the
   top-5 tokens, mean over the last third minus the first two thirds (a surprising ending), n tokens.
2. Cloze: after the text, log P(answer A) - log P(answer B) for a few question frames, e.g.
   "Is this comment sarcastic? Answer:" " Yes" vs " No". Paraphrased frames are averaged per concept.
Texts are head+tail truncated to 384 tokens.

  flock .gpu.lock uv run python experiments/features/build_zs_lm.py --input pool|TEST.csv --out PARQUET
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fe_common import cached, default_out, free_gpu, head_tail, load_input, write_set

NAME = "zs_lm"
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_TOK = 384
PREFIX = "Online comment:\n"
# concept -> [(frame appended after the text, answer A, answer B)]; feature = mean log P(A) - log P(B)
CLOZE = {
    "sarcastic": [("\n\nQuestion: Is the comment above sarcastic? Answer:", " Yes", " No"),
                  ("\n\nQuestion: Does the author mean the opposite of what they say? Answer:", " Yes", " No"),
                  ("\n\nThe tone of the comment above is", " sarcastic", " sincere")],
    "positive": [("\n\nQuestion: Is the comment above positive or negative? Answer:", " positive", " negative"),
                 ("\n\nOverall, the author's opinion is", " positive", " negative"),
                 ("\n\nQuestion: Is the author happy? Answer:", " Yes", " No")],
    "joking": [("\n\nQuestion: Is the author joking? Answer:", " Yes", " No")],
    "angry": [("\n\nQuestion: Is the author angry? Answer:", " Yes", " No")],
}


def surprisal_stats(tok, model, texts: list[str], bs: int = 4) -> np.ndarray:
    import torch
    pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
    ids = [pre + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = np.argsort([-len(x) for x in ids])
    out = np.zeros((len(texts), 7), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
            m[r, :len(ids[i])] = 1
        x, m = x.cuda(), m.cuda()
        lg = model(input_ids=x, attention_mask=m).logits[:, :-1]
        nll = torch.nn.functional.cross_entropy(lg.float().transpose(1, 2), x[:, 1:], reduction="none").cpu().numpy()
        for r, i in enumerate(idx):
            v = nll[r, len(pre) - 1:len(ids[i]) - 1]  # text tokens only
            if len(v) == 0:
                v = np.zeros(1)
            k = max(1, (2 * len(v)) // 3)
            tail = v[k:].mean() - v[:k].mean() if len(v) >= 3 else 0.0
            out[i] = [v.mean(), v.std(), v.max(), np.quantile(v, 0.9), np.sort(v)[-5:].mean(), tail,
                      np.log1p(len(v))]
    return out


def cloze(tok, model, texts: list[str], frame: str, a: str, b: str, bs: int = 16) -> np.ndarray:
    import torch
    ta, tb = tok(a, add_special_tokens=False)["input_ids"][0], tok(b, add_special_tokens=False)["input_ids"][0]
    fr = tok(frame, add_special_tokens=False)["input_ids"]
    ids = [tok(PREFIX + t, add_special_tokens=False)["input_ids"] + fr for t in texts]
    order = np.argsort([-len(x) for x in ids])
    out = np.zeros(len(texts), np.float32)
    tok.padding_side = "left"
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):  # left padding: the next-token position is the last column
            x[r, L - len(ids[i]):] = torch.tensor(ids[i])
            m[r, L - len(ids[i]):] = 1
        pos = (m.cumsum(1) - 1).clamp(min=0)
        lg = model(input_ids=x.cuda(), attention_mask=m.cuda(), position_ids=pos.cuda(),
                   logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(lg, -1)
        out[idx] = (lp[:, ta] - lp[:, tb]).cpu().numpy()
    return out


def compute(df) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    texts, _ = head_tail(df["text"].to_list(), tok, MAX_TOK)
    cols = [surprisal_stats(tok, model, texts)]
    for frames in CLOZE.values():
        for f, a, b in frames:
            t0 = time.time()
            cols.append(cloze(tok, model, texts, f, a, b)[:, None])
            print(f"[cloze] {time.time() - t0:.0f}s {f.strip()}", flush=True)
    del model
    free_gpu()
    return np.hstack(cols)


def build(inp: str, out: str | None) -> None:
    t0 = time.time()
    df = load_input(inp)
    M = cached(f"lm_qwen25_1p5b_v1", df, lambda: compute(df))
    names = ["surp_mean", "surp_sd", "surp_max", "surp_p90", "surp_top5", "surp_tail_gap", "log_ntok"]
    feats = {n: M[:, i] for i, n in enumerate(names)}
    j = len(names)
    for c, frames in CLOZE.items():
        feats[f"cloze_{c}"] = M[:, j:j + len(frames)].mean(1)
        j += len(frames)
    desc = {"surp_mean": "mean token surprisal (nats) given a neutral prefix", "surp_sd": "sd of token surprisal",
            "surp_max": "max token surprisal", "surp_p90": "90th percentile token surprisal",
            "surp_top5": "mean of the 5 most surprising tokens",
            "surp_tail_gap": "mean surprisal of the last third minus the first two thirds",
            "log_ntok": "log(1 + n tokens)"}
    desc.update({f"cloze_{c}": f"mean log P(A)-log P(B) over {len(fr)} cloze frames: {[x[0].strip() for x in fr]}"
                 for c, fr in CLOZE.items()})
    write_set(NAME, df, feats, out or default_out(NAME, df), desc, [MODEL], t0,
              {"cloze": {c: [list(x) for x in fr] for c, fr in CLOZE.items()}, "prefix": PREFIX})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.input, a.out)
