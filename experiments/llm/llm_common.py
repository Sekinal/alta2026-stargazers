"""Shared pieces for the LLM runs: model loading, prompts, verbalizer scoring, caching.

Scoring = one forward pass per (row, prompt). From the last prompt position we keep
  * the log-probabilities of the answer verbalizers (e.g. "Yes" vs "No"), and
  * the hidden state of a few layers (features for probes / the stacker).
The answer probability is softmax over the two verbalizer classes only (logsumexp over
case variants), never sampled text.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache"
LOGS = HERE / "logs"

MODELS = {
    "qwen3-8b": "unsloth/Qwen3-8B-bnb-4bit",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-9b-nf4": "Qwen/Qwen3.5-9B",  # 4-bit NF4 on the fly (bitsandbytes)
}

_NT = re.compile(r"(\w) n't\b")
_OPEN = re.compile(r"\( ")
_CLOSE = re.compile(r" ([).,!?;:])")


def normalise(text: str, source: str) -> str:
    """Undo the google tokeniser artefacts ("could n't", "( x", "x .")."""
    if source == "google":
        text = _NT.sub(r"\1n't", text)
        text = _OPEN.sub("(", text)
        text = _CLOSE.sub(r"\1", text)
    return text


# ------------------------------------------------------------------------------ prompts
SOURCE_DESC = {
    ("en-AU", "reddit"): "a Reddit comment from an Australian community (Australian English)",
    ("en-UK", "reddit"): "a Reddit comment from a British community (UK English)",
    ("en-AU", "google"): "a Google Maps review of an Australian place (Australian English)",
    ("en-UK", "google"): "a Google Maps review of a UK place (UK English)",
}

SARC_DEF_STRICT = (
    "Sarcasm means the writer says something they do not literally mean, usually the opposite, "
    "in order to mock, criticise or show contempt. Typical forms: mock praise or fake enthusiasm "
    "(\"Oh great\", \"Can't wait\", \"Well done\"), deadpan statements that are obviously absurd, "
    "feigned agreement (\"I'm sure that will work\"), mock outrage, and ironic rhetorical questions. "
    "A text that is merely angry, negative, rude or critical but says exactly what it means is "
    "NOT sarcastic."
)
SARC_DEF_LOOSE = (
    "Sarcasm here is understood broadly: the writer uses irony, mockery, ridicule, cynical or "
    "derisive jabs, snark, or says something they do not literally mean, usually to criticise "
    "or show contempt (for example cynical remarks about politicians, parties or government). "
    "A sincere, neutral, informative or friendly text is NOT sarcastic."
)
SARC_DEF_UK = SARC_DEF_STRICT + (
    " British sarcasm is often dry and understated: a flat statement or a set phrase "
    "(\"oh\", \"can't wait\", \"I'm sure\", \"to be fair\", \"what could go wrong\") "
    "that pretends to approve of something absurd, especially in UK politics "
    "(Tories, Labour, the NHS, Brexit)."
)

SYSTEM = "You are an expert annotator of English online text. Answer with a single word."


def _truncate(tok, text: str, max_tokens: int) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = max_tokens // 2
    return tok.decode(ids[:head]) + " [...] " + tok.decode(ids[-(max_tokens - head):])


def user_message(prompt: str, variety: str, source: str, text: str,
                 shots: list[tuple[str, str]] | None = None) -> str:
    """shots: list of (text, answer) exemplars, already truncated."""
    desc = SOURCE_DESC[(variety, source)]
    if prompt.startswith("sarc"):
        if prompt == "sarc_v1":
            definition = SARC_DEF_STRICT
        elif prompt == "sarc_v2":  # variety-specific definitions
            definition = SARC_DEF_UK if variety == "en-UK" else SARC_DEF_LOOSE
        else:
            raise ValueError(prompt)
        q = "Is the text sarcastic? Answer Yes or No."
        head = f"{definition}\n\nThe text is {desc}."
    elif prompt.startswith("sent"):
        q = ("Is the overall sentiment of the text positive or negative? "
             "Answer Positive or Negative.")
        head = (f"The text is {desc}. Decide whether the writer's overall sentiment is positive "
                "or negative. Sarcastic praise counts as negative.")
    else:
        raise ValueError(prompt)
    parts = [head]
    if shots:
        parts.append("Examples:")
        for i, (t, a) in enumerate(shots, 1):
            parts.append(f"Example {i}:\n<text>\n{t}\n</text>\nAnswer: {a}")
        parts.append("Now the text to label.")
    parts.append(f"<text>\n{text}\n</text>\n\n{q}")
    return "\n\n".join(parts)


VERBALIZERS = {
    "sarc": (["Yes", "yes", "YES"], ["No", "no", "NO"]),
    "sent": (["Positive", "positive", "POS"], ["Negative", "negative", "NEG"]),
}


def build_prompts(tok, df: pl.DataFrame, prompt: str, max_text_tokens: int = 256,
                  shots: list[list[tuple[str, str]]] | None = None) -> list[str]:
    out = []
    for i, (src, var, txt) in enumerate(df.select("source", "variety", "text").iter_rows()):
        t = _truncate(tok, normalise(txt, str(src)), max_text_tokens)
        msg = user_message(prompt, str(var), str(src), t, shots[i] if shots else None)
        chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]
        s = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        out.append(s)
    return out


def verbalizer_ids(tok, task: str) -> tuple[list[int], list[int]]:
    pos, neg = VERBALIZERS[task]
    ids = []
    for words in (pos, neg):
        cur = []
        for w in words:
            t = tok(w, add_special_tokens=False)["input_ids"]
            if len(t) >= 1:
                cur.append(t[0])
        ids.append(sorted(set(cur)))
    assert not set(ids[0]) & set(ids[1]), ids
    return ids[0], ids[1]


# ------------------------------------------------------------------------------ model
def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if name.endswith("-nf4"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head"])
    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map={"": 0}, dtype=torch.bfloat16, low_cpu_mem_usage=True, **kw)
    model.eval()
    return tok, model


class Scorer:
    """Keeps only the verbalizer rows of lm_head on the GPU; hooks capture chosen layers."""

    def __init__(self, tok, model, layers: tuple[int, ...]):
        self.tok, self.model = tok, model
        self.base = model.model
        self.head = {}
        for task in VERBALIZERS:
            p, n = verbalizer_ids(tok, task)
            self.head[task] = (p, n)
        all_ids = sorted({i for p, n in self.head.values() for i in p + n})
        self.id_index = {t: k for k, t in enumerate(all_ids)}
        self.W = model.lm_head.weight[all_ids].detach().clone()
        model.lm_head = torch.nn.Identity()  # free the full-vocab head
        torch.cuda.empty_cache()
        self.layers = layers
        self._cap = {}
        blocks = self.base.layers
        for li in layers:
            blocks[li].register_forward_hook(self._hook(li))

    def _hook(self, li):
        def f(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._cap[li] = h[:, -1, :].float().cpu()
        return f

    @torch.no_grad()
    def __call__(self, prompts: list[str], task: str):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(0) for k, v in enc.items()}
        self._cap = {}
        out = self.base(**enc, use_cache=False)
        last = out.last_hidden_state[:, -1, :]
        logits = last.float() @ self.W.float().T
        lp = torch.log_softmax(logits, dim=-1)  # over the verbalizer subset only
        p, n = self.head[task]
        lpos = torch.logsumexp(lp[:, [self.id_index[i] for i in p]], dim=-1)
        lneg = torch.logsumexp(lp[:, [self.id_index[i] for i in n]], dim=-1)
        prob = torch.sigmoid(lpos - lneg).cpu().numpy()
        hid = np.stack([self._cap[li].numpy() for li in self.layers], 1).astype(np.float16)
        return prob, (lpos - lneg).cpu().numpy(), hid


def batches_by_budget(lengths: np.ndarray, max_tokens: int, max_bs: int):
    """Indices sorted by length, grouped so bs * maxlen <= max_tokens."""
    order = np.argsort(-lengths, kind="stable")
    cur: list[int] = []
    for i in order:
        L = lengths[i]
        if cur and (len(cur) + 1) * lengths[cur[0]] > max_tokens or len(cur) >= max_bs:
            yield cur
            cur = []
        cur.append(int(i))
    if cur:
        yield cur


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
