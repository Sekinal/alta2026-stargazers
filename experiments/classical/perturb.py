"""Perturbation robustness (selection charter item 4) for embedding-probe members and their stack.

1) make: writes cache/pert_<kind>.csv = the pool with perturbed text (row order = load_pool()).
   kinds: lower   lowercase + strip punctuation except apostrophes
          typo    ~3% of letters get a random adjacent swap / drop / duplicate (seeded)
          dialect AU<->UK spelling/slang swaps (-ise<->-ize, colour<->color, mate<->buddy, ...)
   then embed them on the GPU:  embed.py --test cache/pert_<kind>.csv <names>
2) eval: for every fold k, the fold-k member models (fit on clean folds != k) score the PERTURBED
   fold-k rows; the fold-k stacker (fit on clean OOF of folds != k) combines them. Reports the
   headline score clean vs perturbed.

  uv run python experiments/classical/perturb.py make
  uv run python experiments/classical/perturb.py eval --stack stk_final
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import polars as pl

from common import CACHE, LABEL, fmt, pool, quick
from alta.common import PREDS, SEED, load_pool

KINDS = ("lower", "typo", "dialect")
SWAPS = [("colour", "color"), ("favour", "favor"),
         ("centre", "center"), ("theatre", "theater"), ("realise", "realize"), ("mate", "buddy"),
         ("rubbish", "garbage"), ("arvo", "afternoon"), ("brekkie", "breakfast"), ("reckon", "think"),
         ("bloody", "really"), ("shop", "store"), ("holiday", "vacation"), ("queue", "line")]


def perturb(text: str, kind: str, rng: np.random.Generator) -> str:
    if kind == "lower":
        return re.sub(r"[^\w\s']", "", text.lower())
    if kind == "typo":
        out = []
        for ch in text:
            if ch.isalpha() and rng.random() < 0.03:
                op = rng.integers(3)
                if op == 0 and out:
                    out[-1], ch = ch, out[-1]
                elif op == 1:
                    continue
                else:
                    out.append(ch)
            out.append(ch)
        return "".join(out)
    if kind == "dialect":
        d = {**{a: b for a, b in SWAPS}, **{b: a for a, b in SWAPS}}
        out = []
        for w in re.split(r"(\W+)", text):
            lw = w.lower()
            if lw in d:
                r = d[lw]
            elif len(lw) >= 7 and re.search(r"is(e|ed|es|ing|ation)$", lw):
                r = re.sub(r"is(e|ed|es|ing|ation)$", r"iz\1", lw)
            elif len(lw) >= 7 and re.search(r"iz(e|ed|es|ing|ation)$", lw):
                r = re.sub(r"iz(e|ed|es|ing|ation)$", r"is\1", lw)
            else:
                out.append(w)
                continue
            out.append(r.capitalize() if w[:1].isupper() else r)
        return "".join(out)
    raise ValueError(kind)


def make() -> None:
    df = load_pool()
    rng = np.random.default_rng(SEED)
    for kind in KINDS:
        t = [perturb(x, kind, rng) for x in df["text"].to_list()]
        out = df.select(pl.col("source").cast(pl.String), pl.col("variety").cast(pl.String)).with_columns(
            pl.Series("text", t), pl.lit(None, pl.String).alias("sentiment"), pl.lit(None, pl.String).alias("sarcasm"))
        path = CACHE / f"pert_{kind}.csv"
        out.write_csv(path)
        changed = np.mean([a != b for a, b in zip(t, df["text"].to_list())])
        print(path, f"changed rows {changed:.1%}")


def eval_stack(stack_run: str) -> None:
    import embed
    import probe
    import stack
    meta = json.loads((PREDS / f"{stack_run}.json").read_text())
    cfg, members = meta["config"], meta["members"]
    df = pool()
    seg, fold = df["seg"].to_numpy(), df["fold"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    mcfg = {}
    for m in members:
        mm = json.loads((PREDS / f"{m}.json").read_text())
        if mm.get("kind") != "probe":
            sys.exit(f"{m}: only probe members supported (kind={mm.get('kind')})")
        mcfg[m] = mm["config"]
    M_clean = stack.load_members(members, df["uid"].to_numpy())
    for kind in KINDS:
        path = CACHE / f"pert_{kind}.csv"
        # member OOF on perturbed held-out rows, fold models fitted on clean training folds
        M = {m: {t: np.zeros(df.height) for t in ("sent", "sarc")} for m in members}
        for m, c in mcfg.items():
            X, Xp = probe.load_features(c["emb"]), probe.load_features(c["emb"], path)
            for k in range(5):
                tr, te = fold != k, fold == k
                p = probe.predict(c, X[tr], {t: v[tr] for t, v in y.items()}, seg[tr], Xp[te], seg[te])
                for t in p:
                    M[m][t][te] = stack.logit(p[t])
        out = {}
        for t in ("sent", "sarc"):
            Xc, mono = stack.design(M_clean, members, t, cfg, seg, None)
            Xq, _ = stack.design(M, members, t, cfg, seg, None)
            out[t] = np.zeros(df.height)
            for k in range(5):
                tr, te = fold != k, fold == k
                out[t][te] = stack.fit_predict(cfg, Xc[tr], y[t][tr], seg[tr] // 2, fold[tr], Xq[te], seg[te] // 2, mono)
        clean = pl.read_parquet(PREDS / f"{stack_run}.parquet").filter(pl.col("split") == "oof")
        clean = df.select("uid").join(clean, on="uid", how="left", maintain_order="left")
        spec = ("segment+ukg",)
        print(f"{kind:8s} clean     {fmt(quick(clean['p_sent'].to_numpy(), clean['p_sarc'].to_numpy(), spec))}")
        print(f"{kind:8s} perturbed {fmt(quick(out['sent'], out['sarc'], spec))}")
        for m in members:
            mc = df.select("uid").join(pl.read_parquet(PREDS / f"{m}.parquet").filter(pl.col("split") == "oof"),
                                       on="uid", how="left", maintain_order="left")
            a = quick(mc["p_sent"].to_numpy(), mc["p_sarc"].to_numpy(), spec)["segment+ukg"]["score"]
            b = quick(1 / (1 + np.exp(-M[m]["sent"])), 1 / (1 + np.exp(-M[m]["sarc"])), spec)["segment+ukg"]["score"]
            print(f"    {m:24s} clean {a:.4f} perturbed {b:.4f} ({b - a:+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "eval"])
    ap.add_argument("--stack")
    a = ap.parse_args()
    make() if a.cmd == "make" else eval_stack(a.stack)


if __name__ == "__main__":
    main()
