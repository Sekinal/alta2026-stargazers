"""Stage 0: domain-adaptive masked-LM pretraining (DAPT, Gururangan et al. 2020) on in-domain text.

Corpus (data/external/indomain, already deduplicated against the pool): sample ~--tokens tokens,
60% Reddit / 40% reviews, AU:UK 50:50, UK politics subreddits up-weighted x1.5. Each document gets
the stage-2 prefix ("[en-UK] [reddit] ...", reviews tagged [google] like our Google reviews),
documents are packed into --max_len blocks, 15% dynamic masking, one pass, linear decay.
A wall-clock cap (--max_minutes) shrinks the schedule after measuring throughput on the first
steps, so the LR still decays to 0 at the (earlier) end. Saves the MLM model to --out; the
fine-tuning script loads its encoder with --model <out>.

  uv run python -m experiments.encoders.dapt --out models/dapt_deb_l --tokens 40e6 --max_minutes 90
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from experiments.encoders.common import ROOT, normalise

EXT = ROOT / "data" / "external" / "indomain"
UK_POLITICS = {"ukpolitics", "unitedkingdom", "LabourUK", "tories", "Scotland", "SNP", "brexit",
               "GreenAndPleasant", "uklabour"}
SHARES = {("reddit", "en-AU"): 0.30, ("reddit", "en-UK"): 0.30, ("review", "en-AU"): 0.20,
          ("review", "en-UK"): 0.20}
FILES = {("reddit", "en-AU"): "reddit_au", ("reddit", "en-UK"): "reddit_uk",
         ("review", "en-AU"): "reviews_au", ("review", "en-UK"): "reviews_uk"}


def sample_corpus(tokens: float, seed: int) -> list[str]:
    """Documents per stratum until its token share is met (tokens ~ chars / 4.2)."""
    rng = np.random.default_rng(seed)
    docs = []
    for key, share in SHARES.items():
        src, var = key
        d = pl.read_parquet(EXT / f"{FILES[key]}.parquet", columns=["text", "community"])
        d = d.filter(pl.col("text").str.len_chars() >= 20)
        w = np.ones(d.height)
        if key == ("reddit", "en-UK"):
            w[d["community"].is_in(list(UK_POLITICS)).to_numpy()] = 1.5
        order = rng.choice(d.height, size=d.height, replace=False, p=w / w.sum())
        est = (d["text"].str.len_chars().to_numpy()[order] / 4.2).cumsum()
        n = int(np.searchsorted(est, share * tokens)) + 1
        tag = "reddit" if src == "reddit" else "google"
        texts = d["text"].gather(order[:n]).to_list()
        docs += [f"[{var}] [{tag}] {normalise(t, tag)}" for t in texts]
        print(f"{key}: {n} docs (~{est[min(n, len(est)) - 1] / 1e6:.1f}M tokens)", flush=True)
    rng.shuffle(docs)
    return docs


def pack(tok, docs: list[str], max_len: int) -> np.ndarray:
    """[CLS] doc [SEP] doc [SEP] ... in blocks of max_len (last partial block dropped)."""
    body = max_len - 1
    stream: list[int] = []
    for s in range(0, len(docs), 10000):
        for x in tok(docs[s:s + 10000], add_special_tokens=False)["input_ids"]:
            stream += x + [tok.sep_token_id]
    n = len(stream) // body
    arr = np.asarray(stream[: n * body], dtype=np.int32).reshape(n, body)
    return np.concatenate([np.full((n, 1), tok.cls_token_id, dtype=np.int32), arr], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=40e6)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32, help="effective batch (micro-batch = bs / accum)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max_minutes", type=float, default=90)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    t_prep = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    blocks = pack(tok, sample_corpus(a.tokens, a.seed), a.max_len)
    print(f"{len(blocks)} blocks x {a.max_len} = {blocks.size / 1e6:.1f}M tokens, prep {time.time() - t_prep:.0f}s",
          flush=True)
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    dev = "cuda"
    model = AutoModelForMaskedLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    nodecay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    plan = {"total": len(blocks) // a.bs if a.max_steps is None else min(a.max_steps, len(blocks) // a.bs)}
    plan["warm"] = max(1, int(a.warmup * plan["total"]))

    def lr_at(step: int) -> float:  # reads the (possibly shrunk) plan
        if step < plan["warm"]:
            return (step + 1) / plan["warm"]
        return max(0.0, (plan["total"] - step) / max(1, plan["total"] - plan["warm"]))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    order = np.random.default_rng(a.seed).permutation(len(blocks))
    model.train()
    t0, run_loss = time.time(), 0.0
    step = 0
    while step < plan["total"]:
        idx = order[step * a.bs:(step + 1) * a.bs]
        for micro in np.array_split(idx, a.accum):
            batch = coll([{"input_ids": torch.as_tensor(blocks[i], dtype=torch.long)} for i in micro])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
            (out.loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss += out.loss.item()
        step += 1
        if step == 100:  # throughput check: shrink the plan to fit the wall-clock cap
            per_step = (time.time() - t0) / step
            fit = int(a.max_minutes * 60 / per_step)
            print(f"throughput {a.bs * a.max_len / per_step:.0f} tok/s, {per_step:.3f}s/step; "
                  f"planned {plan['total']} steps, {fit} fit in {a.max_minutes} min", flush=True)
            if fit < plan["total"]:
                plan["total"] = fit  # warmup unchanged (already done or nearly)
        if step % 200 == 0 or step == plan["total"]:
            n = step % 200 or 200
            print(f"step {step}/{plan['total']} loss={run_loss / n:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"t={time.time() - t0:.0f}s", flush=True)
            run_loss = 0.0

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "dapt_config.json").write_text(json.dumps(
        {**vars(a), "steps": step, "tokens_seen": step * a.bs * a.max_len, "blocks": len(blocks),
         "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"saved {out}: {step} steps, {step * a.bs * a.max_len / 1e6:.1f}M tokens, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
