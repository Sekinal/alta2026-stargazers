"""Frozen text features on the GPU: sentence embeddings and classifier-head outputs, cached to cache/.

Two kinds of extractors:
  st:<name>   sentence-transformers model -> L2-normalised embedding (optionally with a prompt)
  clf:<name>  HF sequence classifier -> [logits..., mean-pooled last hidden state]
Texts over the length budget are head+tail truncated (first 3/4, last 1/4 of the tokens), as in
eda/semantic/01_embed.py, so the closing line of long posts survives.

Output: cache/emb_<name>__<tag>.npy, rows aligned with load_pool() (tag 'pool') or with
load_test(PATH) (tag = 'test-' + sha1 of the test file). Usage:
  flock .gpu.lock uv run python experiments/classical/embed.py [--test PATH] name [name ...]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool, load_test  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)

Q3_SENT = "Instruct: Classify the sentiment of the given online review or comment as positive or negative\nQuery: "
Q3_SARC = "Instruct: Decide whether the given online review or comment is sarcastic or ironic\nQuery: "
# name -> (kind, HF id, max_len, prompt)
MODELS = {
    "bge_large": ("st", "BAAI/bge-large-en-v1.5", 512, None),
    "bge_base": ("st", "BAAI/bge-base-en-v1.5", 512, None),
    "mpnet": ("st", "sentence-transformers/all-mpnet-base-v2", 384, None),
    "e5_large": ("st", "intfloat/e5-large-v2", 512, "query: "),
    "gte_large": ("st", "thenlper/gte-large", 512, None),
    "mxbai": ("st", "mixedbread-ai/mxbai-embed-large-v1", 512, None),
    "qwen3_sent": ("st", "Qwen/Qwen3-Embedding-0.6B", 512, Q3_SENT),
    "qwen3_sarc": ("st", "Qwen/Qwen3-Embedding-0.6B", 512, Q3_SARC),
    "qwen3_4b_sent": ("st", "Qwen/Qwen3-Embedding-4B", 512, Q3_SENT),
    "qwen3_4b_sarc": ("st", "Qwen/Qwen3-Embedding-4B", 512, Q3_SARC),
    "qwen3_8b_sent": ("st", "Qwen/Qwen3-Embedding-8B", 512, Q3_SENT),
    "qwen3_8b_sarc": ("st", "Qwen/Qwen3-Embedding-8B", 512, Q3_SARC),
    "me5_sent": ("st", "intfloat/multilingual-e5-large-instruct", 512,
                 "Instruct: Classify the sentiment of the given online review or comment as positive or negative\nQuery: "),
    "me5_sarc": ("st", "intfloat/multilingual-e5-large-instruct", 512,
                 "Instruct: Decide whether the given online review or comment is sarcastic or ironic\nQuery: "),
    "e5_base": ("st", "intfloat/e5-base-v2", 512, "query: "),
    "tw_irony": ("clf", "cardiffnlp/twitter-roberta-base-irony", 512, None),
    "tw_sent": ("clf", "cardiffnlp/twitter-roberta-base-sentiment-latest", 512, None),
    "go_emo": ("clf", "SamLowe/roberta-base-go_emotions", 512, None),
}


def tag_for(test: Path | None) -> str:
    if test is None:
        return "pool"
    return "test-" + hashlib.sha1(test.read_bytes()).hexdigest()[:10]


def cache_path(name: str, test: Path | None = None) -> Path:
    return CACHE / f"emb_{name}__{tag_for(test)}.npy"


def load(name: str, test: Path | None = None) -> np.ndarray:
    p = cache_path(name, test)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run experiments/classical/embed.py {'--test ' + str(test) if test else ''} {name}")
    return np.load(p)


def head_tail(texts: list[str], tok, budget: int) -> tuple[list[str], np.ndarray]:
    out, n_tok = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok.append(len(ids))
        if len(ids) > budget:
            h = int(budget * 0.75)
            t = tok.decode(ids[:h] + ids[-(budget - h):], skip_special_tokens=True)
        out.append(t)
    return out, np.array(n_tok)


def embed_st(hf_id: str, max_len: int, prompt: str | None, texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = max_len
    budget = max_len - 2 - (len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"]) if prompt else 0)
    tr, n_tok = head_tail(texts, model.tokenizer, budget)
    order = np.argsort(n_tok)
    e = model.encode([tr[i] for i in order], batch_size=8, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True, prompt=prompt)
    out = np.empty_like(e, dtype=np.float32)
    out[order] = e.astype(np.float32)
    del model
    return out


def embed_clf(hf_id: str, max_len: int, texts: list[str]) -> np.ndarray:
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, torch_dtype=torch.float16).cuda().eval()
    tr, n_tok = head_tail(texts, tok, max_len - 2)
    order = np.argsort(n_tok)
    rows = [None] * len(texts)
    for s in range(0, len(order), 16):
        idx = order[s:s + 16]
        b = tok([tr[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to("cuda")
        o = model(**b, output_hidden_states=True)
        h = o.hidden_states[-1].float()
        m = b["attention_mask"].unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1)
        feats = torch.cat([o.logits.float(), mean], 1).cpu().numpy()
        for j, i in enumerate(idx):
            rows[i] = feats[j]
    del model
    return np.stack(rows).astype(np.float32)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--test", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ctx", type=Path, help="context parquet (uid, parent_text, submission_title, ...): embed "
                    "the recovered Reddit context instead of the text; saved as emb_<name>_ctx__<tag>.npy")
    a = ap.parse_args()
    df = load_pool() if a.test is None else load_test(a.test)
    texts = df["text"].to_list()
    suffix = ""
    if a.ctx is not None:
        from context import context_text
        texts = context_text(df, a.ctx)
        suffix = "_ctx"
    for name in a.names:
        out = cache_path(name + suffix, a.test)
        if out.exists() and not a.force:
            print("cached", out)
            continue
        kind, hf_id, max_len, prompt = MODELS[name]
        if suffix:  # empty context (unmatched rows) -> a single space; the flag lives in the features
            texts = [t if t.strip() else " " for t in texts]
        e = embed_st(hf_id, max_len, prompt, texts) if kind == "st" else embed_clf(hf_id, max_len, texts)
        assert e.shape[0] == len(texts) and np.isfinite(e).all(), name
        np.save(out, e)
        print("saved", out, e.shape, flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
