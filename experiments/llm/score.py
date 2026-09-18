"""Zero-/few-shot LLM scoring of pool (or test) rows; resumable, chunked cache.

  flock .gpu.lock uv run python experiments/llm/score.py --model qwen3-8b --prompt sarc_v1 \
      [--rows reddit] [--limit 200] [--shots knn4] [--test data/test.csv] [--max-minutes 80]

Writes cache/<model>/<prompt>[_<shots>]/<split>/part_XXXX.parquet (uid, prob, margin) and
part_XXXX.npy (last-position hidden states of --layers, float16). Already-scored uids are
skipped, so an interrupted pass resumes. Rows are processed longest-first.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import (CACHE, CONTEXT_POOL, Scorer, Timer, add_context, batches_by_budget,  # noqa: E402
                        build_prompts, load_model)
from shots import select_shots  # noqa: E402

sys.path.insert(0, str(CACHE.parents[2]))
from alta.common import load_pool, load_test  # noqa: E402


def done_uids(out_dir) -> set[str]:
    if not out_dir.exists():
        return set()
    parts = sorted(out_dir.glob("part_*.parquet"))
    return set(pl.concat([pl.read_parquet(p) for p in parts])["uid"]) if parts else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--shots", default="", help="e.g. knn4 (kNN exemplars from other folds)")
    ap.add_argument("--rows", default="all", choices=["all", "reddit", "uk_reddit"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test", default="")
    ap.add_argument("--context", default="", help="context parquet (uid-keyed); default: pool context")
    ap.add_argument("--layers", default="", help="default: n/2, 3n/4, n-1")
    ap.add_argument("--max-text-tokens", type=int, default=256)
    ap.add_argument("--max-batch-tokens", type=int, default=24000)
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--max-minutes", type=float, default=80)
    a = ap.parse_args()

    task = "sarc" if a.prompt.startswith("sarc") else "sent"
    split = "test" if a.test else "pool"
    df = load_test(a.test) if a.test else load_pool()
    if a.rows == "reddit":
        df = df.filter(pl.col("source") == "reddit")
    elif a.rows == "uk_reddit":
        df = df.filter((pl.col("source") == "reddit") & (pl.col("variety") == "en-UK"))
    if a.limit:
        df = df.sample(a.limit, seed=0)
    tag = a.prompt + (f"_{a.shots}" if a.shots else "")
    out_dir = CACHE / a.model / tag / split
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_uids(out_dir)
    df = df.filter(~pl.col("uid").is_in(list(done)))
    print(f"{tag} {split}: {len(done)} done, {df.height} to score", flush=True)
    if df.height == 0:
        return

    clock = Timer()
    tok, model = load_model(a.model)
    n_layers = len(model.model.layers)
    layers = (tuple(int(x) for x in a.layers.split(",")) if a.layers
              else (n_layers // 2, 3 * n_layers // 4, n_layers - 1))
    print(f"layers {layers} of {n_layers}", flush=True)
    scorer = Scorer(tok, model, layers)
    print(f"loaded in {clock():.0f}s, vram {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
          flush=True)

    shots = select_shots(a.shots, df, task, tok, a.max_text_tokens, split) if a.shots else None
    if a.prompt.endswith("c"):  # context-aware prompt: thread title / body / parent for matched rows
        df = add_context(tok, df, a.context or CONTEXT_POOL)
        print(f"context for {df['context'].is_not_null().sum()} of {df.height} rows", flush=True)
    prompts = build_prompts(tok, df, a.prompt, a.max_text_tokens, shots)
    lengths = np.array([len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]])
    print(f"prompt tokens: mean {lengths.mean():.0f} max {lengths.max()}", flush=True)
    uids = df["uid"].to_list()

    n_part = len(list(out_dir.glob("part_*.parquet")))
    buf_u, buf_p, buf_m, buf_h = [], [], [], []
    t_start, n_done, tok_done = clock(), 0, 0

    def flush():
        nonlocal n_part, buf_u, buf_p, buf_m, buf_h
        if not buf_u:
            return
        pl.DataFrame({"uid": buf_u, "prob": np.concatenate(buf_p),
                      "margin": np.concatenate(buf_m)}).write_parquet(
            out_dir / f"part_{n_part:04d}.parquet")
        np.save(out_dir / f"part_{n_part:04d}.npy", np.concatenate(buf_h))
        n_part += 1
        buf_u, buf_p, buf_m, buf_h = [], [], [], []

    for idx in batches_by_budget(lengths, a.max_batch_tokens, a.max_bs):
        prob, margin, hid = scorer([prompts[i] for i in idx], task)
        buf_u += [uids[i] for i in idx]
        buf_p.append(prob); buf_m.append(margin); buf_h.append(hid)
        n_done += len(idx); tok_done += int(lengths[idx].sum())
        if len(buf_u) >= a.save_every:
            flush()
            el = clock() - t_start
            print(f"{n_done}/{len(uids)} rows, {n_done / el:.1f} rows/s, {tok_done / el:.0f} tok/s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        if clock() > a.max_minutes * 60:
            print("time budget reached; stopping (resume by re-running)", flush=True)
            break
    flush()
    el = clock() - t_start
    print(f"done {n_done} rows in {el:.0f}s ({n_done / max(el, 1e-9):.2f} rows/s, "
          f"{tok_done / max(el, 1e-9):.0f} tok/s), peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
