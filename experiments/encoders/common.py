"""Shared pieces for the fine-tuned encoder runs: text prep, model, batching, inference.

Model: one pretrained encoder, first-token (CLS) pooling, a sentiment logit and one sarcasm
logit per variety (the sarcasm head is picked by the row's variety). The input is prefixed
with the segment, e.g. "[en-UK] [reddit] text", and google tokeniser artefacts are undone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT / "models"
CACHE = HERE / "cache"

_NT = re.compile(r"(\w) n't\b")
_OPEN = re.compile(r"\( ")
_CLOSE = re.compile(r" ([).,!?;:])")


def normalise(text: str, source: str) -> str:
    if source == "google":
        text = _NT.sub(r"\1n't", text)
        text = _OPEN.sub("(", text)
        text = _CLOSE.sub(r"\1", text)
    return text


def model_inputs(df: pl.DataFrame, prefix: bool = True) -> list[str]:
    out = []
    for src, var, txt in df.select("source", "variety", "text").iter_rows():
        t = normalise(txt, str(src))
        out.append(f"[{var}] [{src}] {t}" if prefix else t)
    return out


def encode(tok, texts: list[str], max_len: int) -> list[list[int]]:
    """Token ids with special tokens; texts longer than max_len keep the head and the tail."""
    ids = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    body = max_len - 2
    out = []
    for x in ids:
        if len(x) > body:
            head = body // 2
            x = x[:head] + x[-(body - head):]
        out.append([tok.cls_token_id] + x + [tok.sep_token_id])
    return out


def collate(batch_ids: list[list[int]], pad_id: int):
    n = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), n), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return ids, mask


class MultiTask(nn.Module):
    def __init__(self, encoder, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sent = nn.Linear(h, 1)
        self.sarc = nn.Linear(h, 2)  # column 0 = en-AU, 1 = en-UK

    def forward(self, ids, mask, variety):
        hs = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        z = self.drop(hs[:, 0])
        sent = self.sent(z).squeeze(-1)
        sarc = self.sarc(z).gather(1, variety[:, None]).squeeze(-1)
        return sent, sarc


def build(model_name: str, dropout: float = 0.1) -> MultiTask:
    enc = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    return MultiTask(enc, dropout)


def save_fold(model: MultiTask, tok, out: Path, cfg: dict) -> None:
    """bf16 weights of encoder + heads, tokenizer and the run config."""
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(out)
    tok.save_pretrained(out)
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out / "weights.pt")
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))


def load_fold(path: Path, device="cuda") -> tuple[MultiTask, object, dict]:
    cfg = json.loads((path / "run_config.json").read_text())
    enc = AutoModel.from_config(AutoConfig.from_pretrained(path))
    model = MultiTask(enc)
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(path)
    return model.to(device).eval(), tok, cfg


@torch.no_grad()
def predict(model: MultiTask, tok, df: pl.DataFrame, max_len: int, prefix: bool = True,
            bs: int = 32, device="cuda") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids = encode(tok, model_inputs(df, prefix), max_len)
    var = (df["variety"].cast(pl.String) == "en-UK").cast(pl.Int64).to_numpy()
    order = np.argsort([len(x) for x in ids])
    ps, pc = np.zeros(len(ids)), np.zeros(len(ids))
    for s in range(0, len(ids), bs):
        idx = order[s : s + bs]
        x, m = collate([ids[i] for i in idx], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = model(x.to(device), m.to(device), torch.as_tensor(var[idx], device=device))
        ps[idx] = torch.sigmoid(a.float()).cpu().numpy()
        pc[idx] = torch.sigmoid(b.float()).cpu().numpy()
    return ps, pc


def robustness_preds(model: MultiTask, tok, run: str, fold: int, max_len: int, prefix: bool = True) -> int:
    """Perturbation suite (experiments/robustness): predict every perturbed copy of fold `fold` with
    THIS fold-k model (it never saw those rows) and save it. drop_prefix = same text, prefix off.
    Returns the number of kinds scored (0 if the holdout dir for this fold does not exist)."""
    from alta.common import load_test
    from experiments.robustness.perturb import save_rob_preds

    d = ROOT / "experiments" / "robustness" / f"holdout_f{fold}_s2026"
    if not (d / "manifest.json").exists():
        return 0
    kinds = json.loads((d / "manifest.json").read_text())["kinds"]
    for kind in kinds:
        df = load_test(d / f"{kind}.csv")
        ps, pc = predict(model, tok, df, max_len, prefix and kind != "drop_prefix")
        save_rob_preds(run, kind, df.select("uid").with_columns(pl.Series("p_sent", ps), pl.Series("p_sarc", pc)),
                       d, protocol="fold_model", meta={"model_fold": fold})
    return len(kinds)
