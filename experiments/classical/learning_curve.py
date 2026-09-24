"""Learning curves of the two frozen-feature lead members (e5+me5 LR probe, Qwen3.5-9B NF4 probe).

For each outer fold k the probe is trained on a nested, segment-stratified fraction of the other folds'
rows (draw 0 uses exactly the subsets of experiments/encoders/train.py --train_frac) and predicts fold k,
giving a full OOF run per (fraction, draw). Scored with the official metric under the cross-fitted
segment+ukg policy, plus threshold-free AUCs on the hard Reddit cells. Output: results/paper/learning_curve_probes.csv

  uv run python experiments/classical/learning_curve.py [--members e5,llm] [--draws 3]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import LABEL, ROOT, pool, quick
import probe as P

sys.path.insert(0, str(ROOT / "experiments" / "llm"))
from make_run import fit_predict, load_multi, segment_dummies  # noqa: E402

FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
OUT = ROOT / "results" / "paper" / "learning_curve_probes.csv"


def subset(df_tr: pl.DataFrame, k: int, frac: float, draw: int) -> np.ndarray:
    """Boolean mask over df_tr rows: nested within (k, draw) as frac grows; draw 0 = the encoder subsets."""
    if frac >= 1:
        return np.ones(df_tr.height, bool)
    r = np.random.default_rng(2026 + k + 1000 * draw).random(df_tr.height)
    q = (df_tr.select("variety", "source").with_columns(pl.Series("r", r))
         .with_columns((pl.col("r").rank("ordinal").over("variety", "source")
                        / pl.len().over("variety", "source")).alias("q")))["q"].to_numpy()
    return q <= frac


def e5_member(df, fold, seg, y, frac, draw):
    X = P.load_features("e5_large+me5_sarc")
    cfg = {"clf": "lr", "sent": "pooled", "sarc": "segment"}
    out = {t: np.zeros(df.height) for t in ("sent", "sarc")}
    for k in range(5):
        tr_idx = np.flatnonzero(fold != k)
        tr_idx = tr_idx[subset(df[tr_idx], k, frac, draw)]
        te = fold == k
        p = P.predict(cfg, X[tr_idx], {t: y[t][tr_idx] for t in y}, seg[tr_idx], X[te], seg[te])
        for t in out:
            out[t][te] = p[t]
    return out


_LLM = {}


def llm_member(df, fold, seg, y, frac, draw):
    """Same features / C / grouping as llm_q35_9b_probe (make_run.build, kind=probe)."""
    if not _LLM:
        uids = df["uid"].to_list()
        for t, tag in (("sarc", "sarc_v2"), ("sent", "sent_v1")):
            _, margin, hid = load_multi("qwen3.5-9b-nf4", tag, "pool", uids, 1)
            _LLM[t] = np.concatenate([hid, margin, segment_dummies(df)], 1)
    C = 0.002
    out = {t: np.zeros(df.height) for t in ("sent", "sarc")}
    ukg = seg == 2  # en-UK google: sarcasm prior only, as in the member
    var = seg // 2
    for k in range(5):
        tr_all = np.flatnonzero(fold != k)
        tr_idx = tr_all[subset(df[tr_all], k, frac, draw)]
        te = fold == k
        out["sent"][te] = fit_predict(_LLM["sent"][tr_idx], y["sent"][tr_idx], [_LLM["sent"][te]], C)[0]
        for v in (0, 1):
            m_tr = tr_idx[(var[tr_idx] == v) & ~ukg[tr_idx]]
            m_te = te & (var == v) & ~ukg
            out["sarc"][m_te] = fit_predict(_LLM["sarc"][m_tr], y["sarc"][m_tr], [_LLM["sarc"][m_te]], C)[0]
        out["sarc"][te & ukg] = 1e-4
    return out


def aucs(df, seg, y, p) -> dict:
    r = {}
    for name, t, s in (("auc_sent_AUr", "sent", 1), ("auc_sarc_AUr", "sarc", 1), ("auc_sarc_UKr", "sarc", 3)):
        m = seg == s
        r[name] = round(float(roc_auc_score(y[t][m], p[t][m])), 4)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="e5,llm")
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    df = pool()
    fold, seg = df["fold"].to_numpy(), df["seg"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    rows = []
    for mem in a.members.split(","):
        fn = e5_member if mem == "e5" else llm_member
        for frac in FRACS:
            for draw in range(1 if frac >= 1 else a.draws):
                p = fn(df, fold, seg, y, frac, draw)
                res = quick(p["sent"], p["sarc"], specs=("segment+ukg",))["segment+ukg"]
                row = {"member": mem, "frac": frac, "draw": draw, "n_train": int(round(frac * (fold != 0).sum())),
                       **res, **aucs(df, seg, y, p)}
                print(row, flush=True)
                rows.append(row)
    out = pl.DataFrame(rows)
    if OUT.exists():
        old = pl.read_csv(OUT)
        out = pl.concat([old.filter(~pl.col("member").is_in(out["member"].unique().to_list())), out], how="diagonal")
    out.write_csv(OUT)
    print(out.group_by("member", "frac").agg(pl.col("score").mean().round(4).alias("score_mean"),
                                             pl.col("score").std().round(4).alias("score_sd"),
                                             pl.col("auc_sarc_AUr").mean().round(4), pl.col("auc_sarc_UKr").mean().round(4))
          .sort("member", "frac"))


if __name__ == "__main__":
    main()
