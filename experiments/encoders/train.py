"""Cross-fitted fine-tuning of a multi-task encoder (sentiment + variety-specific sarcasm).

For each requested fold k: train on folds != k for a FIXED number of epochs, predict fold k,
write experiments/encoders/cache/<run>/fold{k}.parquet (and, with --save, the bf16 checkpoint
to models/<run>/fold{k}/). Nothing about fold k's labels feeds back into its predictions;
--pilot only logs per-epoch held-out diagnostics (used once, on fold 0, to fix the epoch count).
When all 5 fold files exist, --collect writes preds/<run>.parquet via alta.common.save_preds.

  flock .gpu.lock uv run python -m experiments.encoders.train --run enc_x --model M --folds 0,1
  uv run python -m experiments.encoders.train --run enc_x --collect
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import bitsandbytes as bnb
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from alta.common import N_FOLDS, load_pool, save_preds, score
from experiments.encoders.common import (CACHE, MODELS, build, collate, encode, model_inputs,
                                         predict, save_fold)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def diagnostics(df: pl.DataFrame, ps, pc) -> str:
    """Held-out AUCs within the hard segments and the 0.5-threshold score (pilot only)."""
    d = df.with_columns(pl.Series("p_sent", ps), pl.Series("p_sarc", pc))
    out = []
    for name, v, t, col in (("sentAUr", "en-AU", "sentiment", "p_sent"), ("sarcAUr", "en-AU", "sarcasm", "p_sarc"),
                            ("sarcUKr", "en-UK", "sarcasm", "p_sarc")):
        s = d.filter((pl.col("variety") == v) & (pl.col("source") == "reddit"))
        out.append(f"{name}={roc_auc_score(s[t], s[col]):.3f}")
    d = d.with_columns((pl.col("p_sent") >= .5).cast(pl.Int64).alias("sentiment_pred"),
                       ((pl.col("p_sarc") >= .5) & ~((pl.col("variety") == "en-UK") & (pl.col("source") == "google")))
                       .cast(pl.Int64).alias("sarcasm_pred"))
    sc = score(d)
    out += [f"{k}={v:.3f}" for k, v in sc.items()]
    return " ".join(out)


def run_fold(a, pool: pl.DataFrame, k: int) -> None:
    seed_all(a.seed + k)
    dev = "cuda"
    tr, te = pool.filter(pl.col("fold") != k), pool.filter(pl.col("fold") == k)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = encode(tok, model_inputs(tr, not a.no_prefix), a.max_len)
    y_sent = tr["sentiment"].to_numpy().astype(np.float32)
    y_sarc = tr["sarcasm"].to_numpy().astype(np.float32)
    y_sarc = y_sarc * (1 - a.ls) + a.ls / 2
    var = (tr["variety"].cast(pl.String) == "en-UK").cast(pl.Int64).to_numpy()

    model = build(a.model, a.dropout).to(dev)
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_decay = [p for n, p in model.encoder.named_parameters() if p.ndim > 1]
    enc_nodecay = [p for n, p in model.encoder.named_parameters() if p.ndim <= 1]
    opt_cls = bnb.optim.AdamW8bit if a.optim == "adamw8bit" else torch.optim.AdamW
    opt = opt_cls([
        {"params": enc_decay, "lr": a.lr, "weight_decay": a.wd},
        {"params": enc_nodecay, "lr": a.lr, "weight_decay": 0.0},
        {"params": head, "lr": a.head_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = math.ceil(len(ids) / (a.bs * a.accum))
    total = steps_per_epoch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    bce = torch.nn.BCEWithLogitsLoss()
    bce_each = torch.nn.BCEWithLogitsLoss(reduction="none")
    # Optionally train the sarcasm heads on one variety only (the other's sarcasm loss is masked).
    sarc_mask = np.ones(len(ids), dtype=np.float32) if a.sarc_only is None \
        else (tr["variety"].cast(pl.String) == a.sarc_only).cast(pl.Float32).to_numpy()

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(ids))
        # Length-bucketed batches (shuffle within chunks of 50 batches) to cut padding.
        chunk = a.bs * 50
        perm = np.concatenate([c[np.argsort([len(ids[i]) for i in c])]
                               for c in np.array_split(perm, max(1, len(perm) // chunk))])
        batches = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
        random.shuffle(batches)
        run_loss, n = 0.0, 0
        for bi, idx in enumerate(batches):
            x, m = collate([ids[i] for i in idx], tok.pad_token_id)
            v = torch.as_tensor(var[idx], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls, lc = model(x.to(dev), m.to(dev), v)
            sm = torch.as_tensor(sarc_mask[idx], device=dev)
            l_sarc = (bce_each(lc.float(), torch.as_tensor(y_sarc[idx], device=dev)) * sm).sum() / sm.sum().clamp(min=1)
            loss = bce(ls.float(), torch.as_tensor(y_sent[idx], device=dev)) + a.sarc_w * l_sarc
            (loss / a.accum).backward()
            run_loss += loss.item(); n += 1
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        msg = f"fold {k} epoch {ep + 1}/{a.epochs} loss={run_loss / n:.4f} t={time.time() - t0:.0f}s"
        if a.pilot:
            ps, pc = predict(model, tok, te, a.max_len, not a.no_prefix)
            msg += " | " + diagnostics(te, ps, pc)
        print(msg, flush=True)

    ps, pc = predict(model, tok, te, a.max_len, not a.no_prefix)
    out_dir = CACHE / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    te.select("uid").with_columns(pl.Series("p_sent", ps), pl.Series("p_sarc", pc)) \
        .write_parquet(out_dir / f"fold{k}.parquet")
    cfg = {k2: v2 for k2, v2 in vars(a).items() if k2 not in ("folds", "collect")}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"fold {k} done in {time.time() - t0:.0f}s | " + diagnostics(te, ps, pc), flush=True)
    if a.save:
        save_fold(model, tok, MODELS / a.run / f"fold{k}", cfg)
    del model, opt
    torch.cuda.empty_cache()


def collect(run: str) -> None:
    d = CACHE / run
    parts = [d / f"fold{k}.parquet" for k in range(N_FOLDS)]
    missing = [p.name for p in parts if not p.exists()]
    assert not missing, f"missing {missing}"
    oof = pl.concat([pl.read_parquet(p) for p in parts])
    cfg = json.loads((d / "config.json").read_text())
    meta = {"model": cfg["model"], "hyperparameters": cfg, "epochs": cfg["epochs"], "seed": cfg["seed"],
            "description": cfg.get("desc", ""),
            "checkpoints": str(MODELS / run) if (MODELS / run).exists() else None}
    print(save_preds(run, oof, meta=meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ls", type=float, default=0.0, help="label smoothing for sarcasm targets")
    ap.add_argument("--sarc_w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_prefix", action="store_true")
    ap.add_argument("--sarc_only", default=None, choices=["en-AU", "en-UK"],
                    help="sarcasm loss only on this variety's rows")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--desc", default="")
    a = ap.parse_args()
    if a.collect:
        collect(a.run)
        return
    pool = load_pool()
    for k in map(int, a.folds.split(",")):
        run_fold(a, pool, k)


if __name__ == "__main__":
    main()
