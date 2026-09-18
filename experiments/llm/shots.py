"""Leak-free few-shot exemplar selection.

Specs:
  knnN   N nearest labelled neighbours (bge-large cosine) from the same variety x source
  knnbN  N/2 nearest positives + N/2 nearest negatives (class-balanced), same segment

For a pool row in fold k, candidates come only from folds != k (exact-duplicate texts share a
fold, so a row never sees itself). Test rows may use the whole pool. Exemplars are ordered from
least to most similar, so the most similar one sits next to the query.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from llm_common import CACHE, ROOT, _truncate, normalise

EMB = ROOT / "eda" / "semantic" / "cache"
LABEL = {"sarc": ("sarcasm", "Yes", "No"), "sent": ("sentiment", "Positive", "Negative")}


def pool_embeddings() -> tuple[pl.DataFrame, np.ndarray]:
    import sys
    sys.path.insert(0, str(ROOT))
    from alta.common import load_pool
    order = pl.read_parquet(EMB / "uid_order.parquet")["uid"]
    emb = np.load(EMB / "emb_bge_large.npy")
    pool = load_pool()
    assert pool["uid"].to_list() == order.to_list()
    return pool, emb


def test_embeddings(df: pl.DataFrame) -> np.ndarray:
    """bge-large embeddings for unseen rows (CPU; cached per uid set)."""
    import hashlib
    key = hashlib.sha1("|".join(df["text"].to_list()).encode()).hexdigest()[:12]
    path = CACHE / f"emb_bge_large_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    m.max_seq_length = 512
    e = m.encode(df["text"].to_list(), batch_size=16, normalize_embeddings=True,
                 convert_to_numpy=True).astype(np.float32)
    np.save(path, e)
    return e


def select_shots(spec: str, df: pl.DataFrame, task: str, tok, max_text_tokens: int,
                 split: str) -> list[list[tuple[str, str]]]:
    m = re.fullmatch(r"knn(b?)(\d+)", spec)
    assert m, spec
    balanced, n = m.group(1) == "b", int(m.group(2))
    col, yes, no = LABEL[task]
    pool, emb = pool_embeddings()
    row_of = {u: i for i, u in enumerate(pool["uid"].to_list())}
    if split == "pool":
        q_emb = emb[[row_of[u] for u in df["uid"].to_list()]]
        q_fold = df["fold"].to_numpy()
    else:
        q_emb = test_embeddings(df)
        q_fold = np.full(df.height, -1)
    p_fold = pool["fold"].to_numpy()
    p_seg = (pool["variety"].cast(pl.String) + "|" + pool["source"].cast(pl.String)).to_numpy()
    q_seg = (df["variety"].cast(pl.String) + "|" + df["source"].cast(pl.String)).to_numpy()
    p_y = pool[col].to_numpy()
    p_txt = [normalise(t, str(s)) for t, s in pool.select("text", "source").iter_rows()]
    short = max(48, max_text_tokens // 2)  # exemplars get half the query's token budget
    cache: dict[int, str] = {}

    def text(j):
        if j not in cache:
            cache[j] = _truncate(tok, p_txt[j], short)
        return cache[j]

    out = []
    for i in range(df.height):
        ok = (p_seg == q_seg[i]) & (p_fold != q_fold[i])
        sims = emb @ q_emb[i]
        sims = np.where(ok, sims, -np.inf)
        if balanced:
            picks = []
            for lab in (1, 0):
                s = np.where(p_y == lab, sims, -np.inf)
                k = min(n // 2, int(np.isfinite(s).sum()))
                picks += list(np.argsort(-s)[:k])
            picks = sorted(picks, key=lambda j: sims[j])
        else:
            picks = list(np.argsort(-sims)[:n])[::-1]
        out.append([(text(j), yes if p_y[j] == 1 else no) for j in picks])
    return out
