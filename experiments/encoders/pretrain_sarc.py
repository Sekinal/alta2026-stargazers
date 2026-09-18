"""Stage 1: supervised sarcasm pre-fine-tuning on external data, then save the encoder.

Mix (data/external/sarcasm, already deduplicated against the pool):
  * reddit_s_uk / reddit_s_au (in-community weak "/s" labels, tag stripped): all positives,
    negatives subsampled so positives are ~--pos_rate of these rows; variety known.
  * gold sets (epic_irony, isarcasmeval_en, semeval2018_irony, figlang2020_reddit/_twitter,
    reddit_irony_wallace2014): all rows. Variety tag only where known (epic_irony UK/AU).
No parent_text (our task has no context). Input format matches stage 2: "[en-UK] [reddit] text";
unknown variety -> "[twitter] text". Sarcasm head per variety as in stage 2 (unknown -> UK head).
One epoch; only the encoder is kept (stage 2 starts with fresh heads):

  uv run python -m experiments.encoders.pretrain_sarc --out models/pft_sarc_deb_l
  uv run python -m experiments.encoders.train --model models/pft_sarc_deb_l ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from experiments.encoders.common import ROOT, build, collate, encode, normalise

EXT = ROOT / "data" / "external" / "sarcasm"
GOLD = ("epic_irony", "isarcasmeval_en", "semeval2018_irony", "figlang2020_reddit",
        "figlang2020_twitter", "reddit_irony_wallace2014")


def load_mix(pos_rate: float, seed: int, limit: int | None) -> pl.DataFrame:
    cols = ["text", "sarcasm", "source", "variety", "dataset"]
    parts = []
    for name, var in (("reddit_s_uk", "en-UK"), ("reddit_s_au", "en-AU")):
        d = pl.read_parquet(EXT / f"{name}.parquet").select(
            "text", "sarcasm", "source", pl.lit(var).alias("variety"), pl.lit(name).alias("dataset"))
        pos, neg = d.filter(pl.col("sarcasm") == 1), d.filter(pl.col("sarcasm") == 0)
        n_neg = min(neg.height, int(pos.height * (1 - pos_rate) / pos_rate))
        parts += [pos, neg.sample(n_neg, seed=seed)]
    for name in GOLD:
        d = pl.read_parquet(EXT / f"{name}.parquet")
        parts.append(d.select(
            "text", "sarcasm", "source",
            pl.when(pl.col("variety_guess").is_in(["en-UK", "en-AU"])).then(pl.col("variety_guess"))
            .otherwise(pl.lit(None, pl.String)).alias("variety"),
            pl.lit(name).alias("dataset")))
    mix = pl.concat([p.select(cols) for p in parts]).filter(pl.col("text").str.len_chars() > 0)
    mix = mix.with_columns(pl.col("sarcasm").cast(pl.Float32)).sample(fraction=1.0, shuffle=True, seed=seed)
    return mix.head(limit) if limit else mix


def inputs(mix: pl.DataFrame) -> list[str]:
    out = []
    for txt, src, var in mix.select("text", "source", "variety").iter_rows():
        src = "reddit" if src == "reddit" else "twitter"
        t = normalise(txt, src)
        out.append(f"[{var}] [{src}] {t}" if var else f"[{src}] {t}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos_rate", type=float, default=0.38)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows of the shuffled mix")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    mix = load_mix(a.pos_rate, a.seed, a.limit)
    print(mix.group_by("dataset").agg(pl.len(), pl.col("sarcasm").mean()).sort("dataset"), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = encode(tok, inputs(mix), a.max_len)
    y = mix["sarcasm"].to_numpy()
    var = (mix["variety"].fill_null("en-UK") == "en-UK").cast(pl.Int64).to_numpy()

    dev = "cuda"
    model = build(a.model).to(dev)
    params = [{"params": [p for n, p in model.encoder.named_parameters() if p.ndim > 1], "weight_decay": 0.01},
              {"params": [p for n, p in model.encoder.named_parameters() if p.ndim <= 1], "weight_decay": 0.0},
              {"params": list(model.sarc.parameters()), "lr": 1e-4, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    # length-bucketed batches, shuffled
    perm = np.random.permutation(len(ids))
    chunk = a.bs * 50
    perm = np.concatenate([c[np.argsort([len(ids[i]) for i in c])]
                           for c in np.array_split(perm, max(1, len(perm) // chunk))])
    batches = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
    random.shuffle(batches)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * len(batches)), len(batches))
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run_loss = time.time(), 0.0
    for bi, idx in enumerate(batches):
        x, m = collate([ids[i] for i in idx], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, lc = model(x.to(dev), m.to(dev), torch.as_tensor(var[idx], device=dev))
        loss = bce(lc.float(), torch.as_tensor(y[idx], device=dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        if (bi + 1) % 500 == 0 or bi == len(batches) - 1:
            n = (bi % 500) + 1
            print(f"step {bi + 1}/{len(batches)} loss={run_loss / n:.4f} t={time.time() - t0:.0f}s", flush=True)
            run_loss = 0.0

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "stage1_config.json").write_text(json.dumps(
        {**vars(a), "rows": mix.height, "pos_rate_actual": float(y.mean()),
         "per_dataset": mix.group_by("dataset").len().to_dicts(), "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"saved {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
