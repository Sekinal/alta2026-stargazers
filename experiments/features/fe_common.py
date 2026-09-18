"""Shared helpers for model-derived (zero-shot, label-free) feature builders.

Builders call ``load_input`` (``--input pool`` or a test CSV), compute features from text/source/
variety only, and ``write_set`` the parquet plus a sidecar json next to it. Raw per-model outputs
are cached in experiments/features/cache/ keyed by model + input fingerprint, so re-running a
builder (e.g. after changing the aggregation) does not touch the GPU.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import frame_fingerprint, load_pool, load_test  # noqa: E402
from _fe import normalise  # noqa: E402  (shared with fe-linguistic: undo google tokeniser splits)

HERE = Path(__file__).resolve().parent
STORE = HERE / "store"
CACHE = HERE / "cache"
STORE.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)


def load_input(arg: str) -> pl.DataFrame:
    """'pool' -> the 6288-row pool (labels dropped); otherwise an unlabelled test CSV.

    `text` is normalised ("did n't" -> "didn't", " ?" -> "?") so pretrained models see natural
    text in both sources; `tag` names the default output (pool | test-<fingerprint>).
    """
    df = load_pool() if arg == "pool" else load_test(arg)
    tag = "pool" if arg == "pool" else f"test-{frame_fingerprint(df)[:10]}"
    return df.select("uid", "source", "variety", pl.col("text").map_elements(normalise, return_dtype=pl.String),
                     pl.lit(tag).alias("tag"))


def default_out(name: str, df: pl.DataFrame) -> Path:
    return STORE / f"{name}__{df['tag'][0]}.parquet"


def fingerprint(df: pl.DataFrame) -> str:
    h = hashlib.sha1()
    for u, t in zip(df["uid"].to_list(), df["text"].to_list()):
        h.update(f"{u}\x00{t}\x01".encode())
    return h.hexdigest()[:10]


def cached(key: str, df: pl.DataFrame, fn, n_rows: int | None = None) -> np.ndarray:
    """np.ndarray cache keyed by (key, input fingerprint); n_rows defaults to one row per text."""
    p = CACHE / f"{key}__{fingerprint(df)}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    arr = np.asarray(fn(), dtype=np.float32)
    assert arr.shape[0] == (n_rows or df.height) and np.isfinite(arr).all(), key
    np.save(p, arr)
    print(f"[cache] {key}: {arr.shape} in {time.time() - t0:.0f}s", flush=True)
    return arr


def free_gpu() -> None:
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def head_tail(texts: list[str], tok, budget: int) -> tuple[list[str], np.ndarray]:
    """Truncate to `budget` tokens keeping the first 3/4 and last 1/4 (closing line survives)."""
    out, n_tok = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok.append(len(ids))
        if len(ids) > budget:
            h = int(budget * 0.75)
            t = tok.decode(ids[:h] + ids[-(budget - h):], skip_special_tokens=True)
        out.append(t)
    return out, np.array(n_tok)


def _config(hf_id: str):
    """AutoConfig, repairing old configs whose id2label values are ints (rejected by transformers 5)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    d = json.loads(Path(hf_hub_download(hf_id, "config.json")).read_text())
    if d.get("id2label"):
        d["id2label"] = {str(k): str(v) for k, v in d["id2label"].items()}
        d["label2id"] = {v: int(k) for k, v in d["id2label"].items()}
    return AutoConfig.for_model(**d)


def classifier_logits(hf_id: str, texts: list[str], max_len: int = 512, bs: int = 32,
                      pairs: list[str] | None = None, tok_id: str | None = None) -> np.ndarray:
    """Logits of an HF sequence classifier (fp16, length-sorted batches). `pairs`: second segment;
    `tok_id`: tokenizer repo for models shipped without working tokenizer files."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_grad_enabled(False)
    cfg = _config(hf_id)
    tok = AutoTokenizer.from_pretrained(tok_id) if tok_id else AutoTokenizer.from_pretrained(hf_id, config=cfg)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, config=cfg, dtype=torch.float16).cuda().eval()
    extra = 0 if pairs is None else max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in pairs)
    tr, n_tok = head_tail(texts, tok, max_len - 4 - extra)
    order = np.argsort(-n_tok)
    out = np.zeros((len(texts), model.config.num_labels), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        a = [tr[i] for i in idx]
        b = None if pairs is None else [pairs[i] for i in idx]
        enc = tok(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        out[idx] = model(**enc).logits.float().cpu().numpy()
    if (out.std(0) < 1e-3).any():  # e.g. a broken tokenizer mapping every text to the same ids
        raise RuntimeError(f"{hf_id}: constant outputs, check tokenizer/weights")
    del model
    free_gpu()
    return out


def write_set(name: str, df: pl.DataFrame, feats: dict[str, np.ndarray], out: str | Path,
              describe: dict[str, str], models: list[str], t0: float, extra: dict | None = None) -> None:
    cols = {f"{name}__{k}": np.asarray(v, np.float32) for k, v in feats.items()}
    for k, v in cols.items():
        assert v.shape == (df.height,) and np.isfinite(v).all(), k
    frame = pl.DataFrame({"uid": df["uid"], **cols})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    print(f"wrote {out} {frame.shape}", flush=True)
    if out.name == f"{name}__pool.parquet":
        side = out.with_name(f"{name}.json")
        old = json.loads(side.read_text()) if side.exists() else {}
        old.update({"set": name, "n_features": len(cols),
                    "features": {f"{name}__{k}": describe.get(k, describe.get(k.split("__")[0], ""))
                                 for k in feats},
                    "models": models, "runtime_s": round(time.time() - t0, 1),
                    "builder": f"experiments/features/build_{name}.py", **(extra or {})})
        side.write_text(json.dumps(old, indent=2))
