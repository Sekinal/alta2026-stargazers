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
from llm_common import CACHE, MODELS, ROOT, build_prompts, load_model, verbalizer_ids  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import load_pool, load_test, save_preds  # noqa: E402

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


class LinearHead(torch.nn.Module):
    """Trainable linear head on the last-position hidden state, one logit per task
    (AstroCLIMB-style; zero-initialised, so training starts from p = 0.5)."""

    def __init__(self, hidden: int, tasks=("sarc", "sent")):
        super().__init__()
        self.tasks = list(tasks)
        self.lin = torch.nn.Linear(hidden, len(self.tasks))
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, h: torch.Tensor, task: str) -> torch.Tensor:
        return self.lin(h.float())[:, self.tasks.index(task)]


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
def predict(inner, head, tok, df, task, max_text_tokens, max_batch_tokens=3000):
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


def load_presft(a) -> pl.DataFrame:
    d = pl.read_parquet(a.presft)
    return d.head(a.presft_limit) if a.presft_limit else d


def make_batches(examples, epochs, tasks, bs, rng):
    """Same-task batches of similar length (chunks of 50*bs sorted by length), shuffled."""
    batches = []
    n_ep = math.ceil(epochs)
    for ep in range(n_ep):
        ex = list(examples)
        rng.shuffle(ex)
        if ep == n_ep - 1 and epochs % 1:
            ex = ex[:int(len(ex) * (epochs % 1))]
        for task in tasks:
            et = [e for e in ex if e[1] == task]
            for c in range(0, len(et), 50 * bs):
                chunk = sorted(et[c:c + 50 * bs], key=lambda e: len(e[0]))
                batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def train_stage(model, inner, head, tok, batches, params, lr, a) -> dict:
    """One optimisation stage: fresh optimizer, 6% warm-up then linear decay, fixed length."""
    steps = math.ceil(len(batches) / a.accum)
    if a.riemannian:
        from peft.optimizers import create_riemannian_optimizer
        opt = create_riemannian_optimizer(model, torch.optim.AdamW, lr=lr, weight_decay=a.wd)
    else:
        groups = [{"params": params, "lr": lr}]
        hp = [p for p in head.parameters() if p.requires_grad]
        if hp:
            groups.append({"params": hp, "lr": a.head_lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=a.wd)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
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
            torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (bi + 1) % 100 == 0:
            el = time.time() - t1
            print(f"  batch {bi + 1}/{len(batches)} loss {np.mean(run_loss[-100:]):.4f} "
                  f"{(bi + 1) * a.bs / el:.2f} ex/s, eta {el / (bi + 1) * (len(batches) - bi - 1) / 60:.1f} min, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del opt
    return {"minutes": (time.time() - t1) / 60, "steps": steps,
            "loss_curve": [float(np.mean(run_loss[i:i + 100])) for i in range(0, len(run_loss), 100)]}


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
    # adapter ladder (R1 LoRA+rsLoRA, R2 EVA, R2b PiSSA, R3 +DoRA, R4 +Riemannian)
    ap.add_argument("--alpha", type=float, default=0, help="default 2r (or 16 with --rslora)")
    ap.add_argument("--rslora", action="store_true")
    ap.add_argument("--init", default="gaussian", choices=["gaussian", "eva", "pissa_niter_16"])
    ap.add_argument("--eva-rho", type=float, default=2.0)
    ap.add_argument("--eva-tau", type=float, default=0.99)
    ap.add_argument("--eva-batches", type=int, default=32, help="EVA init batches of 8 (training folds)")
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--riemannian", action="store_true")
    ap.add_argument("--targets", default="all-linear")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--presft", default="", help="parquet (text, variety, source, sarcasm): sarcasm "
                    "pre-SFT stage on external data before the task stage, same adapter")
    ap.add_argument("--presft-lr", type=float, default=0)
    ap.add_argument("--presft-limit", type=int, default=0)
    ap.add_argument("--backend", default="hf", choices=["hf", "unsloth"])
    ap.add_argument("--head", default="verbalizer", choices=["verbalizer", "linear"])
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--pseudo", default="", help="parquet of soft pseudo-labels (text, variety, source, "
                    "p_sent, p_sarc): adaptation stage before the gold stage, soft-label BCE")
    ap.add_argument("--pseudo-n", type=int, default=60000, help="rows sampled from --pseudo")
    ap.add_argument("--pseudo-lr", type=float, default=0)
    ap.add_argument("--pseudo-tasks", default="sarc,sent")
    ap.add_argument("--save-run", action="store_true",
                    help="save the held-out fold as a partial run preds/<run>.parquet (screening)")
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

    if a.backend == "unsloth":
        # Unsloth (isolated venv): bf16 16-bit LoRA, its own gradient checkpointing and kernels
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(MODELS.get(a.model, a.model), dtype=torch.bfloat16,
                                                       load_in_4bit=False, max_seq_length=1024)
        tok = getattr(tok, "tokenizer", tok)  # Qwen3.5 loads as a VL processor
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        tok, model = load_model(a.model)
    tok.padding_side = "left"
    ids = {}
    for task in ("sarc", "sent"):
        p, n = verbalizer_ids(tok, task)
        ids[task] = (p[0], n[0])  # canonical capitalised verbalizers ("Yes"/"No", ...)
    uniq = sorted({i for pn in ids.values() for i in pn})
    W = model.lm_head.weight[uniq].detach().clone()
    head = Head(W, {t: (uniq.index(p), uniq.index(n)) for t, (p, n) in ids.items()}).to(0)
    if a.predict_only and (adir / "head.pt").exists():
        a.head = "linear"  # the adapter was trained with a separate linear head
    if a.head == "linear":
        head = LinearHead(model.config.get_text_config().hidden_size).to(0)
        if a.predict_only:
            head.load_state_dict(torch.load(adir / "head.pt", map_location="cuda:0"))
    model.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    inner = model.model  # LoRA is injected in place, so this stays the adapted backbone

    if a.predict_only:
        model = PeftModel.from_pretrained(model, adir)
    else:
        from peft import EvaConfig, initialize_lora_eva_weights
        targets = (a.targets if a.targets == "all-linear" else
                   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        alpha = a.alpha or (16.0 if a.rslora else 2.0 * a.r)
        cfg = LoraConfig(r=a.r, lora_alpha=alpha, lora_dropout=a.dropout, bias="none",
                         target_modules=targets, use_rslora=a.rslora, use_dora=a.dora,
                         init_lora_weights=True if a.init == "gaussian" else a.init,
                         eva_config=EvaConfig(rho=a.eva_rho, tau=a.eva_tau, use_label_mask=False)
                         if a.init == "eva" else None)
        t_init = time.time()
        if a.backend == "unsloth":
            assert a.init == "gaussian" and not a.dora, "unsloth path: plain LoRA only"
            names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                            if isinstance(m, torch.nn.Linear)} - {"lm_head"})
            model = FastLanguageModel.get_peft_model(
                model, r=a.r, lora_alpha=alpha, lora_dropout=a.dropout, target_modules=names,
                use_rslora=a.rslora, bias="none", use_gradient_checkpointing="unsloth",
                random_state=a.seed)
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model = get_peft_model(model, cfg)
        train = pool.filter(pl.col("fold") != a.fold)  # training folds only (also for EVA init)
        stats = {"args": vars(a), "alpha": alpha}
        if a.init == "eva":
            # activation SVDs from the TRAINING folds only (never the held-out fold / test)
            ev = []
            if a.presft:  # pre-SFT stage comes first: estimate activations on the external data
                ev += encode(tok, build_prompts(tok, load_presft(a), PROMPTS["sarc"], a.max_text_tokens))
            else:
                for task in tasks:
                    d = training_rows(train, task, a.sent_rows)
                    ev += encode(tok, build_prompts(tok, d, PROMPTS[task], a.max_text_tokens))
            ev = [ev[i] for i in rng.permutation(len(ev))]

            eva_loader = []
            for i in range(0, min(len(ev), 8 * a.eva_batches), 8):
                ids_, att = pad_batch(ev[i:i + 8], tok.pad_token_id)
                eva_loader.append({"input_ids": ids_, "attention_mask": att})

            # PEFT's EVA cycles over the loader until every layer's top components are stable
            # (cos >= tau); for 248 layers at ~30 s/batch that never finishes in budget. Cap it:
            # after one pass over the eva_batches distinct batches, mark all hooks converged so the
            # rank allocation uses the components estimated so far.
            import peft.tuners.lora.eva as eva_mod
            n_seen = {"n": 0}
            orig_call = eva_mod.SVDHook.__call__

            conv_at_cap = {}

            def capped_call(self, m, inp, out):
                if n_seen["n"] > len(eva_loader):
                    # disclose how far the real convergence criterion had got at the cap
                    conv_at_cap.setdefault(self.name, float(self.converged.float().mean()))
                    self.converged = torch.ones_like(self.converged)
                    return
                orig_call(self, m, inp, out)

            def counting_forward(m, inputs):
                n_seen["n"] += 1
                return eva_mod.forward_fn_dict(m, inputs)

            eva_mod.SVDHook.__call__ = capped_call
            with torch.no_grad():
                initialize_lora_eva_weights(model, eva_loader, forward_fn=counting_forward,
                                            show_progress_bar=True)
            eva_mod.SVDHook.__call__ = orig_call
            stats["eva_iters"] = n_seen["n"]
            cv = np.array(list(conv_at_cap.values()) or [1.0])
            stats["eva_converged_at_cap"] = {"layers_still_running": len(cv),
                                             "mean_frac_components_converged": float(cv.mean()),
                                             "frac_layers_fully_converged": float((cv >= 1).mean())}
            del eva_loader
            ranks = model.peft_config["default"].rank_pattern or {}
            rv = np.array(list(ranks.values()) or [a.r])
            stats["eva_ranks"] = {"n_layers": len(ranks), "min": int(rv.min()), "median": float(np.median(rv)),
                                  "max": int(rv.max()), "counts": {int(k): int(v) for k, v in zip(*np.unique(rv, return_counts=True))},
                                  "by_module": {m: float(np.mean([v for k, v in ranks.items() if k.endswith(m)] or [0]))
                                                for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                          "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                                                          "out_proj")}}
            print("EVA ranks", json.dumps(stats["eva_ranks"]), flush=True)
        stats["init_s"] = time.time() - t_init
        model.print_trainable_parameters()

        # ---------------------------------------------------------------- training data
        examples = []  # (token ids, task, label)
        for task in tasks:
            d = training_rows(train, task, a.sent_rows)
            y = d["sarcasm" if task == "sarc" else "sentiment"].to_numpy()
            for s, lab in zip(encode(tok, build_prompts(tok, d, PROMPTS[task], a.max_text_tokens)), y):
                examples.append((s, task, int(lab)))
        rng.shuffle(examples)
        if a.limit_train:
            examples = examples[:a.limit_train]
        batches = make_batches(examples, a.epochs, tasks, a.bs, rng)
        pre_batches = None
        if a.presft:
            d = load_presft(a)
            pre_ex = [(s_, "sarc", int(lab)) for s_, lab in zip(
                encode(tok, build_prompts(tok, d, PROMPTS["sarc"], a.max_text_tokens)), d["sarcasm"].to_numpy())]
            pre_batches = make_batches(pre_ex, 1.0, ["sarc"], a.bs, rng)
            stats["presft_examples"] = len(pre_ex)
        ps_batches = None
        if a.pseudo:
            d = pl.read_parquet(a.pseudo)
            d = d.sample(min(a.pseudo_n, d.height), seed=a.seed)
            ps_ex = []
            for task, col in (("sarc", "p_sarc"), ("sent", "p_sent")):
                if task not in a.pseudo_tasks.split(","):
                    continue
                dt = d.filter(pl.col(col).is_not_null())
                if task == "sarc":  # same rows as the gold sarcasm task (no UK google)
                    dt = dt.filter((pl.col("source") == "reddit") | (pl.col("variety") == "en-AU"))
                ps_ex += [(s_, task, float(y)) for s_, y in zip(
                    encode(tok, build_prompts(tok, dt, PROMPTS[task], a.max_text_tokens)), dt[col].to_numpy())]
            ps_batches = make_batches(ps_ex, 1.0, a.pseudo_tasks.split(","), a.bs, rng)
            stats["pseudo_examples"] = len(ps_ex)
        print(f"fold {a.fold}: {len(examples)} examples, {len(batches)} batches; "
              f"load {time.time() - t0:.0f}s", flush=True)

        params = [p for p in model.parameters() if p.requires_grad]
        model.train()
        t1 = time.time()
        if pre_batches:
            print(f"pre-SFT: {len(pre_batches)} batches", flush=True)
            stats["presft"] = train_stage(model, inner, head, tok, pre_batches, params,
                                          a.presft_lr or a.lr, a)
        if ps_batches:
            print(f"pseudo-label adaptation: {len(ps_batches)} batches", flush=True)
            stats["pseudo"] = train_stage(model, inner, head, tok, ps_batches, params,
                                          a.pseudo_lr or a.lr, a)
        stats["task"] = train_stage(model, inner, head, tok, batches, params, a.lr, a)
        stats.update(train_min=(time.time() - t1) / 60, n_examples=len(examples),
                     peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                     loss_curve=stats["task"]["loss_curve"])
        print(f"train time {stats['train_min']:.1f} min", flush=True)
        model.save_pretrained(adir)
        if a.head == "linear":
            torch.save(head.state_dict(), adir / "head.pt")
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
        if split == "pool" and a.save_run:
            sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
            part = pl.DataFrame({"uid": res["uid"], "p_sent": sig(res["m_sent"]),
                                 "p_sarc": sig(res["m_sarc"])})
            save_preds(a.run, part, meta={
                "model": a.model, "kind": "ft-holdout", "holdout_fold": a.fold, "seed": a.seed,
                "prompt_sarc": PROMPTS["sarc"], "prompt_sent": PROMPTS["sent"], "shots": 0,
                "description": f"Qwen LoRA verbalizer fine-tune, trained on folds != {a.fold}, "
                               f"predicts fold {a.fold} only (screening run)", "train_stats": stats})
    if not a.predict_only:
        (out_dir / f"fold{a.fold}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"predict time {(time.time() - t2) / 60:.1f} min; total {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
