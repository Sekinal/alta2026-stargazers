"""QLoRA fine-tuning of the verbalizer classifier, proper 5-fold cross-fitting.

For fold k: train LoRA adapters on pool rows with fold != k (fixed schedule, no early stopping),
then score fold k (all rows, both tasks) and optionally a test CSV. The loss is 2-way cross-entropy
over the verbalizer logits ("Yes"/"No", "Positive"/"Negative") at the answer position of the
same chat prompt used zero-shot, so training starts from the zero-shot solution.

  flock .gpu.lock uv run python experiments/llm/ft.py --run llm_q8_ft --fold 0 [--test PATH]
  flock .gpu.lock uv run python experiments/llm/ft.py --run llm_q8_ft --predict-only --fold 0 --test PATH

Outputs: models/<run>/fold<k>/ (adapter only), experiments/llm/out/<run>/fold<k>.parquet (OOF rows of
fold k, written as soon as the fold finishes; a finished fold is skipped on re-run, so a killed
spot job resumes) and fold<k>_test.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import CACHE, ROOT, build_prompts, load_model, verbalizer_ids  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import load_pool, load_test  # noqa: E402

MODEL_DIR = ROOT / "models"
OUT = ROOT / "experiments" / "llm" / "out"
PROMPTS = {"sarc": "sarc_v2", "sent": "sent_v1"}


def training_rows(pool: pl.DataFrame, task: str, rows: str) -> pl.DataFrame:
    if task == "sarc":
        # UK google has 1 sarcastic row in 2065: nothing to learn there
        keep = (pl.col("source") == "reddit") | (pl.col("variety") == "en-AU")
    else:
        keep = pl.lit(True) if rows == "all" else (pl.col("source") == "reddit")
    return pool.filter(keep)


class Head(torch.nn.Module):
    """Last-position hidden state -> verbalizer margin (pos logit - neg logit), per task."""

    def __init__(self, W: torch.Tensor, ids: dict[str, tuple[int, int]]):
        super().__init__()
        self.register_buffer("W", W.float())
        self.ids = ids

    def forward(self, h: torch.Tensor, task: str) -> torch.Tensor:
        p, n = self.ids[task]
        return h.float() @ (self.W[p] - self.W[n])


def encode(tok, prompts):
    return tok(prompts, add_special_tokens=False)["input_ids"]


def pad_batch(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        att[i, n - len(s):] = 1
    return ids.to(0), att.to(0)


@torch.no_grad()
def predict(inner, head, tok, df, task, max_text_tokens, max_batch_tokens=6000):
    prompts = build_prompts(tok, df, PROMPTS[task], max_text_tokens)
    seqs = encode(tok, prompts)
    order = np.argsort([-len(s) for s in seqs])
    out = np.zeros(len(seqs))
    i = 0
    while i < len(order):
        L = len(seqs[order[i]])
        bs = max(1, min(32, max_batch_tokens // L))
        idx = order[i:i + bs]
        ids, att = pad_batch([seqs[j] for j in idx], tok.pad_token_id)
        h = inner(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[:, -1]
        out[idx] = head(h, task).cpu().numpy()
        i += bs
    return out  # margin (logit of the positive class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tasks", default="sarc,sent")
    ap.add_argument("--sent-rows", default="reddit", choices=["reddit", "all"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-text-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0, help="for timing only")
    ap.add_argument("--test", default="")
    ap.add_argument("--predict-only", action="store_true")
    a = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    tasks = a.tasks.split(",")
    pool = load_pool()
    out_dir = OUT / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    adir = MODEL_DIR / a.run / f"fold{a.fold}"
    if not a.predict_only and (out_dir / f"fold{a.fold}.parquet").exists() and (
            not a.test or (out_dir / f"fold{a.fold}_test.parquet").exists()):
        print(f"fold {a.fold} already done; skipping", flush=True)
        return

    tok, model = load_model(a.model)
    tok.padding_side = "left"
    ids = {}
    for task in ("sarc", "sent"):
        p, n = verbalizer_ids(tok, task)
        ids[task] = (p[0], n[0])  # canonical capitalised verbalizers ("Yes"/"No", ...)
    uniq = sorted({i for pn in ids.values() for i in pn})
    W = model.lm_head.weight[uniq].detach().clone()
    head = Head(W, {t: (uniq.index(p), uniq.index(n)) for t, (p, n) in ids.items()}).to(0)
    model.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    inner = model.model  # LoRA is injected in place, so this stays the adapted backbone

    if a.predict_only:
        model = PeftModel.from_pretrained(model, adir)
    else:
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

        # ---------------------------------------------------------------- training data
        train = pool.filter(pl.col("fold") != a.fold)
        examples = []  # (token ids, task, label)
        for task in tasks:
            d = training_rows(train, task, a.sent_rows)
            y = d["sarcasm" if task == "sarc" else "sentiment"].to_numpy()
            for s, lab in zip(encode(tok, build_prompts(tok, d, PROMPTS[task], a.max_text_tokens)), y):
                examples.append((s, task, int(lab)))
        rng.shuffle(examples)
        if a.limit_train:
            examples = examples[:a.limit_train]
        # batches: same task, similar length (chunk of 50*bs sorted by length), shuffled
        batches = []
        n_ep = math.ceil(a.epochs)
        for ep in range(n_ep):
            ex = list(examples)
            rng.shuffle(ex)
            if ep == n_ep - 1 and a.epochs % 1:
                ex = ex[:int(len(ex) * (a.epochs % 1))]
            for task in tasks:
                et = [e for e in ex if e[1] == task]
                for c in range(0, len(et), 50 * a.bs):
                    chunk = sorted(et[c:c + 50 * a.bs], key=lambda e: len(e[0]))
                    batches += [chunk[i:i + a.bs] for i in range(0, len(chunk), a.bs)]
        rng.shuffle(batches)
        steps = math.ceil(len(batches) / a.accum)
        print(f"fold {a.fold}: {len(examples)} examples, {len(batches)} batches, {steps} steps; "
              f"load {time.time() - t0:.0f}s", flush=True)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
        model.train()
        t1, run_loss = time.time(), []
        for bi, batch in enumerate(batches):
            ids_, att = pad_batch([e[0] for e in batch], tok.pad_token_id)
            y = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=0)
            h = inner(input_ids=ids_, attention_mask=att, use_cache=False).last_hidden_state[:, -1]
            margin = head(h, batch[0][1])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, y) / a.accum
            loss.backward()
            run_loss.append(loss.item() * a.accum)
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if (bi + 1) % 100 == 0:
                el = time.time() - t1
                print(f"  batch {bi + 1}/{len(batches)} loss {np.mean(run_loss[-100:]):.4f} "
                      f"{(bi + 1) * a.bs / el:.2f} ex/s, eta {el / (bi + 1) * (len(batches) - bi - 1) / 60:.1f} min, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        print(f"train time {(time.time() - t1) / 60:.1f} min", flush=True)
        model.save_pretrained(adir)
        (adir / "train_args.json").write_text(json.dumps(vars(a), indent=2))

    # ---------------------------------------------------------------- predictions
    model.eval()
    t2 = time.time()
    targets = []
    if not a.predict_only:
        targets.append(("pool", pool.filter(pl.col("fold") == a.fold)))
    if a.test:
        targets.append(("test", load_test(a.test)))
    for split, df in targets:
        res = {"uid": df["uid"].to_list()}
        for task in ("sarc", "sent"):
            res[f"m_{task}"] = predict(inner, head, tok, df, task, a.max_text_tokens)
        name = f"fold{a.fold}.parquet" if split == "pool" else f"fold{a.fold}_test.parquet"
        pl.DataFrame(res).write_parquet(out_dir / name)
    print(f"predict time {(time.time() - t2) / 60:.1f} min; total {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
