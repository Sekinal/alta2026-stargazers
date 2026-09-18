"""Recovered Reddit context (data/external/context/*.parquet, label-free): descriptives, a subreddit-prior
model, and cheap context features for the stacker.

  uv run python experiments/classical/context.py describe
  uv run python experiments/classical/context.py prior      # saves cls_subreddit_prior (5-fold OOF)
  uv run python experiments/classical/context.py features   # writes store/ctx_meta__pool.parquet
  uv run python experiments/classical/context.py features --test TEST.csv --ctx TEST_CONTEXT.parquet

Context text for embeddings: the parent comment when the row is a reply, else submission title + selftext.
Unmatched rows get an empty context, a `matched`=0 flag and their own subreddit level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from common import save_kw, LABEL, ROOT, fmt, pool, quick
from alta.common import load_test, save_preds

CTX = ROOT / "data" / "external" / "context" / "pool_context.parquet"
STORE = ROOT / "experiments" / "features" / "store"
MIN_SUB = 20  # subreddits with fewer pool rows share the level "other"


def load_ctx(df: pl.DataFrame, path: Path = CTX) -> pl.DataFrame:
    c = pl.read_parquet(path)
    out = df.select("uid").join(c, on="uid", how="left", maintain_order="left")
    return out.with_columns(pl.col("matched").fill_null(False).cast(pl.Boolean))


def context_text(df: pl.DataFrame, path: Path = CTX) -> list[str]:
    c = load_ctx(df, path)
    txt = []
    for r in c.iter_rows(named=True):
        if not r["matched"]:
            txt.append("")
        elif not r.get("is_top_level") and (r.get("parent_text") or "").strip():
            txt.append(r["parent_text"])
        else:
            txt.append(((r.get("submission_title") or "") + "\n" + (r.get("submission_selftext") or "")).strip())
    return txt


def sub_level(vocab: list[str]) -> pl.Expr:
    return (pl.when(~pl.col("matched")).then(pl.lit("__unmatched"))
            .when(pl.col("subreddit").is_in(vocab)).then(pl.col("subreddit"))
            .otherwise(pl.lit("__other")))


def pool_vocab() -> list[str]:
    c = load_ctx(pool())
    vc = c.filter(pl.col("matched")).group_by("subreddit").len().filter(pl.col("len") >= MIN_SUB)
    return sorted(vc["subreddit"].to_list())


def ctx_features(df: pl.DataFrame, path: Path, vocab: list[str]) -> pl.DataFrame:
    c = load_ctx(df, path).with_columns(sub_level(vocab).alias("lvl"))
    levels = vocab + ["__other", "__unmatched"]
    feats = c.select(
        "uid",
        *[(pl.col("lvl") == l).cast(pl.Float32).alias(f"ctx_meta__sub_{l}") for l in levels],
        pl.col("matched").cast(pl.Float32).alias("ctx_meta__matched"),
        (pl.col("matched") & pl.col("is_top_level").fill_null(False)).cast(pl.Float32).alias("ctx_meta__top_level"),
        pl.when(pl.col("matched")).then(pl.col("score").fill_null(0).cast(pl.Float64).clip(0, None).log1p())
        .otherwise(0.0).cast(pl.Float32).alias("ctx_meta__log_score"),
        pl.when(pl.col("matched")).then((pl.col("score").fill_null(0) < 0).cast(pl.Float32))
        .otherwise(0.0).cast(pl.Float32).alias("ctx_meta__neg_score"),
    )
    return feats


def describe() -> None:
    df = pool()
    c = load_ctx(df).with_columns(pl.col("is_top_level").fill_null(False))
    d = df.select("uid", pl.col("variety").cast(pl.String), pl.col("source").cast(pl.String), "sentiment", "sarcasm") \
        .join(c, on="uid").filter(pl.col("source") == "reddit")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(d.group_by("variety", "matched").agg(pl.len(), pl.col("sarcasm").mean(), pl.col("sentiment").mean())
              .sort("variety", "matched"))
        print(d.filter(pl.col("matched")).group_by("variety", "is_top_level")
              .agg(pl.len(), pl.col("sarcasm").mean(), pl.col("sentiment").mean()).sort("variety", "is_top_level"))
        for v in ("en-AU", "en-UK"):
            print(v, d.filter(pl.col("matched"), pl.col("variety") == v).group_by("subreddit")
                  .agg(pl.len(), pl.col("sarcasm").mean().round(3), pl.col("sentiment").mean().round(3))
                  .sort("len", descending=True).head(15))


def prior() -> None:
    """Smoothed subreddit rate per task x variety, fitted on training folds: p = (k + a*m)/(n + a)."""
    df = pool()
    vocab = pool_vocab()
    c = load_ctx(df).with_columns(sub_level(vocab).alias("lvl"))
    d = df.select("uid", "fold", "seg", "sentiment", "sarcasm").with_columns(c["lvl"])
    fold = d["fold"].to_numpy()
    oof = {t: np.zeros(d.height) for t in ("sent", "sarc")}
    a = 10.0
    for t in oof:
        lab = LABEL[t]
        for k in range(5):
            tr = d.filter(pl.col("fold") != k)
            seg_rate = tr.group_by("seg").agg(pl.col(lab).mean().alias("m"))
            rates = tr.group_by("seg", "lvl").agg(pl.col(lab).sum().alias("k"), pl.len().alias("n"))
            te = d.filter(pl.col("fold") == k).join(seg_rate, on="seg", how="left") \
                .join(rates, on=["seg", "lvl"], how="left").with_columns(pl.col("k", "n").fill_null(0))
            p = ((te["k"] + a * te["m"]) / (te["n"] + a)).to_numpy()
            oof[t][fold == k] = p
    res = quick(oof["sent"], oof["sarc"], specs=("segment+ukg",))
    print("subreddit prior", fmt(res))
    name, kw = save_kw("cls_subreddit_prior", df)
    save_preds(name, pl.DataFrame({"uid": d["uid"], "p_sent": oof["sent"], "p_sarc": oof["sarc"]}), **kw,
               meta={"kind": "ctx_prior", "script": "experiments/classical/context.py prior",
                     "desc": f"smoothed (a={a}) rate by segment x subreddit level (>= {MIN_SUB} rows, other, unmatched)",
                     "quick": res})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["describe", "prior", "features"])
    ap.add_argument("--test", type=Path)
    ap.add_argument("--ctx", type=Path, default=CTX)
    a = ap.parse_args()
    if a.cmd == "describe":
        describe()
    elif a.cmd == "prior":
        prior()
    else:
        vocab = pool_vocab()
        if a.test is None:
            f = ctx_features(pool(), a.ctx, vocab)
            out = STORE / "ctx_meta__pool.parquet"
        else:
            from alta.common import frame_fingerprint
            t = load_test(a.test)
            f = ctx_features(t, a.ctx, vocab)
            out = STORE / f"ctx_meta__test-{frame_fingerprint(t)[:10]}.parquet"
        f.write_parquet(out)
        (STORE / "ctx_meta.json").write_text(json.dumps(
            {"set": "ctx_meta", "features": f.columns[1:], "vocab_min_rows": MIN_SUB, "vocab": vocab,
             "builder": "experiments/classical/context.py features"}, indent=2))
        print(out, f.shape)


if __name__ == "__main__":
    main()
