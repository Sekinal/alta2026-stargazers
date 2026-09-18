"""Encode every pool text with a few sentence-transformers models; cache to cache/emb_<model>.npy.

Texts longer than the model's max length are head+tail truncated (first 3/4 and last 1/4 of the
token budget) so that the closing line of a long reddit rant / review, often where the verdict or
the sarcastic punchline sits, survives. Usage: uv run python eda/semantic/01_embed.py [model ...]
"""

import gc
import json
import sys

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from _shared import CACHE, MODELS, pool


def head_tail(texts: list[str], tok, max_len: int) -> tuple[list[str], np.ndarray]:
    budget = max_len - 2  # [CLS]/[SEP]
    out, n_tok = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok.append(len(ids))
        if len(ids) > budget:
            h = int(budget * 0.75)
            ids = ids[:h] + ids[-(budget - h):]
            t = tok.decode(ids, skip_special_tokens=True)
        out.append(t)
    return out, np.array(n_tok)


def main(names: list[str]) -> None:
    df = pool()
    texts = df["text"].to_list()
    stats_path = CACHE / "token_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for name in names:
        hf_id, max_len = MODELS[name]
        model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = max_len
        tr, n_tok = head_tail(texts, model.tokenizer, max_len)
        # re-tokenising decoded text can drift by a token or two; the model truncates the rest
        over = n_tok > max_len - 2
        stats[name] = {
            "max_len": max_len,
            "n_over": int(over.sum()),
            "n_over_by_source": {
                s: int(over[(df["source"] == s).to_numpy()].sum()) for s in ("google", "reddit")
            },
            "tok_p50": float(np.median(n_tok)), "tok_p95": float(np.quantile(n_tok, 0.95)),
            "tok_p99": float(np.quantile(n_tok, 0.99)), "tok_max": int(n_tok.max()),
        }
        print(name, stats[name])
        # sort by length to minimise padding; small batches keep VRAM/RAM low
        order = np.argsort(n_tok)
        e = model.encode([tr[i] for i in order], batch_size=32, normalize_embeddings=True,
                         show_progress_bar=True, convert_to_numpy=True)
        emb = np.empty_like(e, dtype=np.float32)
        emb[order] = e.astype(np.float32)
        np.save(CACHE / f"emb_{name}.npy", emb)
        del model, e
        gc.collect()
        torch.cuda.empty_cache()
    stats_path.write_text(json.dumps(stats, indent=2))
    pl.DataFrame({"uid": df["uid"]}).write_parquet(CACHE / "uid_order.parquet")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
