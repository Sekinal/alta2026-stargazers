"""Step B stage 1: train the multi-task model on teacher soft pseudo-labels, then save the encoder.

Same inputs ("[variety] [source] text", google artefacts undone), same heads (sentiment + per-variety
sarcasm) and soft-label BCE for both tasks, one pass over data/external/pseudo/fold<k>.parquet
(capped by --max_minutes, the LR schedule shrinks to fit after a throughput check). Stage 2 is the
unchanged gold fine-tune started from the saved encoder with fresh heads:

  uv run python -m experiments.encoders.pretrain_pseudo --fold 0 --out models/pseudo_f0_deb_l
  uv run python -m experiments.encoders.train --model models/pseudo_f0_deb_l --folds 0 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import json
import random
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer

from experiments.encoders.common import ROOT, build, collate, encode, model_inputs

PSEUDO = ROOT / "data" / "external" / "pseudo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="pseudo-labels from the fold-k teacher")
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max_minutes", type=float, default=90)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows")
    ap.add_argument("--tasks", default="both", choices=["both", "sarc"],
                    help="sarc: sarcasm soft labels only (sentiment loss off in stage 1)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    df = pl.read_parquet(PSEUDO / f"fold{a.fold}.parquet").sample(fraction=1.0, shuffle=True, seed=a.seed)
    if a.limit:
        df = df.head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = encode(tok, model_inputs(df), a.max_len)
    ys = df["p_sent"].to_numpy().astype(np.float32)
    yc = df["p_sarc"].to_numpy().astype(np.float32)
    var = (df["variety"] == "en-UK").cast(pl.Int64).to_numpy()

    dev = "cuda"
    model = build(a.model).to(dev)
    enc = list(model.encoder.named_parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": [p for _, p in enc if p.ndim > 1], "weight_decay": 0.01},
                             {"params": [p for _, p in enc if p.ndim <= 1], "weight_decay": 0.0},
                             {"params": heads, "lr": 1e-4, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    perm = np.random.permutation(len(ids))
    chunk = a.bs * 50
    perm = np.concatenate([c[np.argsort([len(ids[i]) for i in c])]
                           for c in np.array_split(perm, max(1, len(perm) // chunk))])
    batches = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
    random.shuffle(batches)
    plan = {"total": len(batches)}
    plan["warm"] = max(1, int(a.warmup * plan["total"]))

    def lr_at(step: int) -> float:  # reads the (possibly shrunk) plan
        if step < plan["warm"]:
            return (step + 1) / plan["warm"]
        return max(0.0, (plan["total"] - step) / max(1, plan["total"] - plan["warm"]))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run_loss, step = time.time(), 0.0, 0
    while step < plan["total"]:
        idx = batches[step]
        x, m = collate([ids[i] for i in idx], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ls, lc = model(x.to(dev), m.to(dev), torch.as_tensor(var[idx], device=dev))
        loss = bce(lc.float(), torch.as_tensor(yc[idx], device=dev))
        if a.tasks == "both":
            loss = loss + bce(ls.float(), torch.as_tensor(ys[idx], device=dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        step += 1
        if step == 300:  # throughput check: shrink the plan to fit the wall-clock cap
            fit = int(a.max_minutes * 60 / ((time.time() - t0) / step))
            print(f"{(time.time() - t0) / step:.3f}s/step; planned {plan['total']}, {fit} fit in {a.max_minutes} min",
                  flush=True)
            plan["total"] = min(plan["total"], fit)
        if step % 1000 == 0 or step == plan["total"]:
            n = step % 1000 or 1000
            print(f"step {step}/{plan['total']} loss={run_loss / n:.4f} t={time.time() - t0:.0f}s", flush=True)
            run_loss = 0.0

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "stage1_config.json").write_text(json.dumps(
        {**vars(a), "rows_total": len(ids), "rows_seen": step * a.bs, "steps": step,
         "teacher": df["teacher"][0], "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"saved {out}: {step} steps, {step * a.bs} rows, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
