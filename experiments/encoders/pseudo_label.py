"""Step A: soft pseudo-labels for unlabelled in-domain text from fold-k teacher checkpoints.

Leakage rule: the teacher for the fold-k screen must not have seen fold k, so it is the average
of the FOLD-k checkpoints of the given runs (each trained on the other 4 folds).
Sample (data/external/indomain, deduplicated against the pool): --n_reddit Reddit comments
(AU:UK 50:50, UK politics subreddits up-weighted x1.5) + --n_reviews reviews (AU:UK 50:50).
Reviews get source "google" (the only review source in our task), so the "[variety] [source]"
prefix built at training time matches. Output columns: text (raw), variety, source, p_sent,
p_sarc, teacher, community, plus model_input (the exact prefixed, normalised teacher input).

  uv run python -m experiments.encoders.pseudo_label --fold 0 --runs enc_deb_l_s13 enc_deb_l_s14 enc_deb_l_s15
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import time

import numpy as np
import polars as pl
import torch

from experiments.encoders.common import MODELS, ROOT, load_fold, model_inputs, predict
from experiments.encoders.dapt import UK_POLITICS

EXT = ROOT / "data" / "external" / "indomain"
OUT = ROOT / "data" / "external" / "pseudo"


def sample(n_reddit: int, n_reviews: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for fname, var, src, n in (("reddit_au", "en-AU", "reddit", n_reddit // 2),
                               ("reddit_uk", "en-UK", "reddit", n_reddit - n_reddit // 2),
                               ("reviews_au", "en-AU", "google", n_reviews // 2),
                               ("reviews_uk", "en-UK", "google", n_reviews - n_reviews // 2)):
        d = pl.read_parquet(EXT / f"{fname}.parquet", columns=["text", "community"])
        d = d.filter(pl.col("text").str.len_chars().is_between(20, 3000)).unique("text", keep="first",
                                                                              maintain_order=True)
        w = np.ones(d.height)
        if fname == "reddit_uk":
            w[d["community"].is_in(list(UK_POLITICS)).to_numpy()] = 1.5
        idx = rng.choice(d.height, size=min(n, d.height), replace=False, p=w / w.sum())
        parts.append(d[np.sort(idx)].with_columns(pl.lit(var).alias("variety"), pl.lit(src).alias("source")))
        print(f"{fname}: {len(idx)} of {d.height}", flush=True)
    return pl.concat(parts).select("text", "variety", "source", "community")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--n_reddit", type=int, default=300_000)
    ap.add_argument("--n_reviews", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--part", default=None, help="label with a subset of the teachers (e.g. one per GPU) into "
                    "fold<k>.part-<part>.parquet; combine with --merge")
    ap.add_argument("--merge", nargs="+", default=None, help="part names to average (weighted by #teachers)")
    ap.add_argument("--texts_from", default="fold0.parquet",
                    help="label exactly the texts of this existing pseudo file (same sample for every fold); "
                         "'' = draw a new sample")
    a = ap.parse_args()
    if a.merge:
        merge(a.fold, a.merge)
        return

    if a.texts_from and (OUT / a.texts_from).exists():
        df = pl.read_parquet(OUT / a.texts_from, columns=["text", "variety", "source", "community"])
        print(f"labelling the {df.height} texts of {a.texts_from}", flush=True)
    else:
        df = sample(a.n_reddit, a.n_reviews, a.seed)
    ps, pc = np.zeros(df.height), np.zeros(df.height)
    for run in a.runs:
        t0 = time.time()
        model, tok, cfg = load_fold(MODELS / run / f"fold{a.fold}")
        s, c = predict(model, tok, df, cfg["max_len"], not cfg.get("no_prefix", False), bs=a.bs)
        ps += s / len(a.runs)
        pc += c / len(a.runs)
        del model
        torch.cuda.empty_cache()
        dt = time.time() - t0
        print(f"{run} fold{a.fold}: {df.height} rows in {dt:.0f}s ({df.height / dt:.0f} rows/s); "
              f"mean p_sent={s.mean():.3f} p_sarc={c.mean():.3f}", flush=True)
    if a.part:
        OUT.mkdir(parents=True, exist_ok=True)
        df.select("text", "variety", "source", "community").with_columns(
            pl.Series("p_sent", ps), pl.Series("p_sarc", pc), pl.lit(len(a.runs)).alias("n_teachers"),
            pl.lit("/".join(a.runs)).alias("runs")).write_parquet(OUT / f"fold{a.fold}.part-{a.part}.parquet")
        print(f"saved part {a.part}", flush=True)
        return
    write(df, ps, pc, a.fold, a.runs)


def write(df: pl.DataFrame, ps, pc, fold: int, runs: list[str]) -> None:
    a_runs, a_fold = runs, fold
    teacher = f"{'/'.join(a_runs)} fold{a_fold} (mean of {len(a_runs)} fold-{a_fold} checkpoints)"
    out = df.with_columns(pl.Series("p_sent", ps), pl.Series("p_sarc", pc), pl.lit(teacher).alias("teacher"),
                          pl.Series("model_input", model_inputs(df)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fold{a_fold}.parquet"
    out.write_parquet(path)
    print(out.group_by("variety", "source").agg(pl.len(), pl.col("p_sent").mean(), pl.col("p_sarc").mean(),
                                                (pl.col("p_sarc") > 0.5).mean().alias("sarc_rate"))
          .sort("variety", "source"))
    print(f"saved {path}", flush=True)


def merge(fold: int, parts: list[str]) -> None:
    """Average the parts' probabilities (weighted by their number of teachers); rows must align."""
    ds = [pl.read_parquet(OUT / f"fold{fold}.part-{p}.parquet") for p in parts]
    for d in ds[1:]:
        assert d["text"].equals(ds[0]["text"]), "parts label different samples"
    w = np.array([d["n_teachers"][0] for d in ds], dtype=float)
    ps = sum(wi * d["p_sent"].to_numpy() for wi, d in zip(w, ds)) / w.sum()
    pc = sum(wi * d["p_sarc"].to_numpy() for wi, d in zip(w, ds)) / w.sum()
    runs = [r for d in ds for r in d["runs"][0].split("/")]
    write(ds[0].select("text", "variety", "source", "community"), ps, pc, fold, runs)
    for p in parts:
        (OUT / f"fold{fold}.part-{p}.parquet").unlink()


if __name__ == "__main__":
    main()
