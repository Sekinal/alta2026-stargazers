"""CONTRACT quick evaluation of a feature set in store/ (writes the result into store/<set>.json).

(1) alone: per task x variety L2-LR on [standardised set + is_reddit], cross-fitted on the shared folds.
(2) incremental: per task x variety L2-LR on [cls_e5me5_lr OOF logits (both tasks) + is_reddit]
    with vs without the set; per-fold deltas and a paired test-like bootstrap p (A = with set).
Scores are honest (policy cross-fitted) under segment+ukg. Also reported: within-slice AUC and
policy macro-F1 on the two worst slices, UK-reddit sarcasm and AU-reddit sentiment.

  uv run python experiments/features/fe_eval.py SET [SET ...] [--base cls_e5me5_lr] [--C 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import PREDS, load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import pool_eval  # noqa: E402
from alta.validation.metrics import components, macro_f1, per_fold  # noqa: E402
from alta.validation.resampling import paired  # noqa: E402

STORE = Path(__file__).resolve().parent / "store"
TASKS = ("sent", "sarc")
LABEL = {"sent": "sentiment", "sarc": "sarcasm"}
SPEC = policies.DEFAULT_POLICY
EPS = 1e-5


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_set(names: list[str], uids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    order = pl.DataFrame({"uid": uids})
    mats, cols = [], []
    for n in names:
        f = pl.read_parquet(STORE / f"{n}__pool.parquet")
        d = order.join(f, on="uid", how="left", maintain_order="left")
        c = [x for x in d.columns if x != "uid"]
        if d.select(pl.any_horizontal(pl.col(c).is_null())).to_series().any():
            raise ValueError(f"{n}: pool uids missing")
        mats.append(d.select(c).to_numpy().astype(np.float64))
        cols += c
    return np.hstack(mats), cols


def crossfit(X: np.ndarray, y: np.ndarray, var: np.ndarray, fold: np.ndarray, C: float) -> np.ndarray:
    p = np.zeros(len(y))
    for k in np.unique(fold):
        for v in (0, 1):
            tr, te = (fold != k) & (var == v), (fold == k) & (var == v)
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def summary(d, p: dict) -> dict:
    dd = d.with_p(p)
    pred, _ = policies.crossfit(SPEC, dd)
    c = components(dd, pred)
    seg = dd.segment()
    ukr, aur = seg == 3, seg == 1
    return {
        "score": round(c["score"], 4), **{k: round(v, 4) for k, v in c.items() if k != "score"},
        "per_fold": [round(x["score"], 4) for x in per_fold(dd, pred)],
        "ukr_sarc_auc": round(float(roc_auc_score(dd.y["sarc"][ukr], p["sarc"][ukr])), 4),
        "ukr_sarc_f1": round(macro_f1(dd.y["sarc"][ukr], pred["sarc"][ukr]), 4),
        "aur_sent_auc": round(float(roc_auc_score(dd.y["sent"][aur], p["sent"][aur])), 4),
        "aur_sent_f1": round(macro_f1(dd.y["sent"][aur], pred["sent"][aur]), 4),
        **{f"ll_{t}_{vn}": round(float(log_loss(dd.y[t][dd.variety == v], np.clip(p[t][dd.variety == v], EPS, 1 - EPS))), 4)
           for t in TASKS for v, vn in ((0, "AU"), (1, "UK"))},
        "ll_ukr_sarc": round(float(log_loss(dd.y["sarc"][ukr], np.clip(p["sarc"][ukr], EPS, 1 - EPS))), 4),
        "_pred": pred,
    }


def evaluate(names: list[str], base_run: str, C: float) -> dict:
    pool = load_pool()
    d = pool_eval(pool)
    uids = pool["uid"].to_numpy()
    assert (d.uid == uids).all()
    var, fold, reddit = d.variety, d.fold, d.source.astype(float)[:, None]
    y = {t: pool[LABEL[t]].to_numpy() for t in TASKS}
    X, cols = load_set(names, uids)
    cols_b = []
    for run in base_run.split(","):  # comma-separated base runs: all their logits enter the base
        b = pl.DataFrame({"uid": uids}).join(pl.read_parquet(PREDS / f"{run}.parquet").filter(pl.col("split") == "oof"),
                                             on="uid", how="left", maintain_order="left")
        cols_b += [logit(b["p_sent"].to_numpy()), logit(b["p_sarc"].to_numpy())]
    B = np.column_stack(cols_b + [reddit])

    alone = summary(d, {t: crossfit(np.hstack([X, reddit]), y[t], var, fold, C) for t in TASKS})
    base = summary(d, {t: crossfit(B, y[t], var, fold, C) for t in TASKS})
    withs = summary(d, {t: crossfit(np.hstack([B, X]), y[t], var, fold, C) for t in TASKS})
    res = paired(d, withs.pop("_pred"), base.pop("_pred"), B=2000)
    alone.pop("_pred")
    delta = {k: round(withs[k] - base[k], 4) for k in withs if k != "per_fold"}
    delta["per_fold"] = [round(a - c, 4) for a, c in zip(withs["per_fold"], base["per_fold"])]
    seg = d.segment()
    uni = {}
    for j, c in enumerate(cols):  # univariate within-slice AUC (diagnostic only)
        uni[c] = {"ukr_sarc": round(float(roc_auc_score(y["sarc"][seg == 3], X[seg == 3, j])), 3),
                  "aur_sarc": round(float(roc_auc_score(y["sarc"][seg == 1], X[seg == 1, j])), 3),
                  "aur_sent": round(float(roc_auc_score(y["sent"][seg == 1], X[seg == 1, j])), 3)}
    return {"univariate_auc": uni, "eval_C": C, "base_run": base_run, "n_features": len(cols), "alone": alone, "base": base,
            "with_set": withs, "delta": delta,
            "p_with_better": {k: round(v["p_a_better"], 3) for k, v in res.items()}}


def show(name: str, r: dict) -> None:
    a, dl = r["alone"], r["delta"]
    print(f"== {name} ({r['n_features']} feats, C={r['eval_C']})")
    print(f"  alone: {a['score']:.4f} sAU {a['sent-AU']:.3f} cUK {a['sarc-UK']:.3f} | "
          f"UKr sarc AUC {a['ukr_sarc_auc']:.3f}  AUr sent AUC {a['aur_sent_auc']:.3f}")
    print(f"  base: UKr sarc AUC {r['base']['ukr_sarc_auc']:.3f} F1 {r['base']['ukr_sarc_f1']:.3f} | "
          f"AUr sent AUC {r['base']['aur_sent_auc']:.3f} F1 {r['base']['aur_sent_f1']:.3f}")
    top = sorted(r["univariate_auc"].items(), key=lambda kv: -abs(kv[1]["ukr_sarc"] - 0.5))[:6]
    print("  top UKr-sarc univariate AUC: " + ", ".join(f"{k.split('__', 1)[1]} {v['ukr_sarc']:.3f}" for k, v in top))
    top = sorted(r["univariate_auc"].items(), key=lambda kv: -abs(kv[1]["aur_sent"] - 0.5))[:4]
    print("  top AUr-sent univariate AUC: " + ", ".join(f"{k.split('__', 1)[1]} {v['aur_sent']:.3f}" for k, v in top))
    print(f"  base {r['base']['score']:.4f} -> with {r['with_set']['score']:.4f}: "
          f"delta {dl['score']:+.4f} per-fold {dl['per_fold']} p={r['p_with_better']['score']}")
    print(f"  delta sAU {dl['sent-AU']:+.4f} cUK {dl['sarc-UK']:+.4f} cAU {dl['sarc-AU']:+.4f} | "
          f"UKr sarc AUC {dl['ukr_sarc_auc']:+.4f} F1 {dl['ukr_sarc_f1']:+.4f} | "
          f"AUr sent AUC {dl['aur_sent_auc']:+.4f} F1 {dl['aur_sent_f1']:+.4f}")
    print("  delta log-loss (neg = better): " + " ".join(f"{k[3:]} {v:+.4f}" for k, v in dl.items() if k.startswith("ll_")),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--base", default="cls_e5me5_lr")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--joint", action="store_true", help="evaluate the sets together (not written)")
    a = ap.parse_args()
    groups = [a.sets] if a.joint else [[s] for s in a.sets]
    for g in groups:
        r = evaluate(g, a.base, a.C)
        show("+".join(g), r)
        if not a.joint and a.base == "cls_e5me5_lr":
            side = STORE / f"{g[0]}.json"
            meta = json.loads(side.read_text()) if side.exists() else {}
            meta["quick_eval"] = r
            side.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
