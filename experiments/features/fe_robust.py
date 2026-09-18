"""Perturbation robustness of feature sets on the fold-0 holdout (experiments/robustness protocol).

Features for each perturbed holdout CSV are built by the set's builder into cache/rob/<set>__<kind>.parquet
(see logs/run_rob.sh). Two read-outs per kind:
  * feature stability: median / min Spearman between clean and perturbed values, over features;
  * downstream drop: LR per task x variety on [set features + is_reddit] trained on pool folds 1-4
    (never on the holdout rows), segment+ukg thresholds fitted on the CLEAN holdout predictions and
    applied unchanged to the perturbed ones; score and the two worst slices vs clean.

  uv run python experiments/features/fe_robust.py zs_irony zs_nli [--joint]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import load_pool  # noqa: E402
from alta.validation import policies  # noqa: E402
from alta.validation.data import pool_eval  # noqa: E402
from alta.validation.metrics import components, macro_f1  # noqa: E402

HERE = Path(__file__).resolve().parent
ROB = HERE / "cache" / "rob"
HOLD = ROOT / "experiments" / "robustness" / "holdout_f0_s2026"
KINDS = ("clean", "lowercase", "strip_punct", "typo_3pct", "au_to_uk", "google_artefacts_on")
TASKS = ("sent", "sarc")
LABEL = {"sent": "sentiment", "sarc": "sarcasm"}


def feats(sets: list[str], kind: str | None, uids: list[str]) -> np.ndarray:
    """kind None -> pool store; else the perturbed holdout (x-uids mapped back to pool uids)."""
    order = pl.DataFrame({"uid": uids})
    mats = []
    for s in sets:
        if kind is None:
            f = pl.read_parquet(HERE / "store" / f"{s}__pool.parquet")
        else:
            f = pl.read_parquet(ROB / f"{s}__{kind}.parquet")
            xmap = {f"x{i:05d}": u for i, u in enumerate(json.loads((HOLD / "manifest.json").read_text())["uids"])}
            f = f.with_columns(pl.col("uid").replace_strict(xmap))
        d = order.join(f, on="uid", how="left", maintain_order="left")
        mats.append(d.drop("uid").to_numpy().astype(np.float64))
    X = np.hstack(mats)
    assert np.isfinite(X).all()
    return X


def run(sets: list[str]) -> list[dict]:
    pool = load_pool()
    d = pool_eval(pool)
    hold_uids = json.loads((HOLD / "manifest.json").read_text())["uids"]
    tr = (d.fold != 0)
    h = np.isin(d.uid, hold_uids)
    hu = d.uid[h].tolist()  # pool order
    dh = d.subset(h)
    Xp = feats(sets, None, d.uid.tolist())
    red = d.source.astype(float)[:, None]
    models = {}
    for t in TASKS:
        y = pool[LABEL[t]].to_numpy()
        for v in (0, 1):
            m = tr & (d.variety == v)
            X = np.hstack([Xp[m], red[m]])
            sc = StandardScaler().fit(X)
            models[t, v] = (sc, LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X), y[m]))
    rows, clean_X, params = [], None, None
    for kind in KINDS:
        if not all((ROB / f"{s}__{kind}.parquet").exists() for s in sets):
            continue
        Xk = feats(sets, kind, hu)
        p = {}
        for t in TASKS:
            p[t] = np.zeros(len(hu))
            for v in (0, 1):
                m = dh.variety == v
                sc, clf = models[t, v]
                p[t][m] = clf.predict_proba(sc.transform(np.hstack([Xk[m], dh.source[m, None].astype(float)])))[:, 1]
        dk = dh.with_p(p)
        if kind == "clean":
            clean_X, params = Xk, policies.fit("segment+ukg", dk)
        pred = policies.apply(params, dk)
        c = components(dk, pred)
        seg = dk.segment()
        rho = [spearmanr(clean_X[:, j], Xk[:, j])[0] for j in range(Xk.shape[1])]
        rows.append({"kind": kind, "score": round(c["score"], 4), "sent-AU": round(c["sent-AU"], 4),
                     "sarc-UK": round(c["sarc-UK"], 4),
                     "ukr_sarc_auc": round(float(roc_auc_score(dk.y["sarc"][seg == 3], p["sarc"][seg == 3])), 4),
                     "ukr_sarc_f1": round(macro_f1(dk.y["sarc"][seg == 3], pred["sarc"][seg == 3]), 4),
                     "aur_sent_f1": round(macro_f1(dk.y["sent"][seg == 1], pred["sent"][seg == 1]), 4),
                     "feat_rho_median": round(float(np.median(rho)), 3), "feat_rho_min": round(float(np.min(rho)), 3)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--joint", action="store_true")
    a = ap.parse_args()
    for g in ([a.sets] if a.joint else [[s] for s in a.sets]):
        rows = run(g)
        print(f"== {'+'.join(g)} (fold-0 holdout, LR on set alone trained on folds 1-4)")
        print(pl.DataFrame(rows))
        if not a.joint:
            side = HERE / "store" / f"{g[0]}.json"
            meta = json.loads(side.read_text())
            meta["robustness_f0"] = rows
            side.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
