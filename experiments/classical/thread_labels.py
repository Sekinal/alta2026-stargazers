"""Transductive same-thread label features on top of a base run (default: the lead's OOF blend).

The official split is by comment, so ~80% of held-out Reddit rows share a submission (link_id) with
labelled pool rows. For a row, the features are built from the LABELS of the other labelled rows in its
thread (and of its parent comment, when that is a pool row):
  thr_has   1 if any labelled thread-mate exists
  thr_mean  shrunk mean label of thread-mates: (sum + a*prior) / (n + a), centred on the prior
  par_lab   parent row's label (centred), 0 if the parent is not labelled
per task (sent/sarc, both used for each task). Google rows / unmatched rows: all zeros.

Honesty (nested cross-fitting): the level-2 model that predicts fold k is fitted on rows of folds j != k
whose features use labels from folds not in {j, k}; fold-k rows' features use labels from folds != k.
At test time the features use all pool labels. The base logit enters as a feature (fixed per row).

  uv run python experiments/classical/thread_labels.py --base stk_v5 --run thr_v5 [--a 2] [--C 1]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from common import LABEL, ROOT, fmt, pool, quick, save_kw
from alta.common import PREDS, save_preds

TASKS = ("sent", "sarc")
EPS = 1e-5
CTX = ROOT / "data" / "external" / "context" / "pool_context.parquet"


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def thread_keys(uids: np.ndarray, ctx_path=CTX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """link_id, own comment id, parent comment id per uid ("" when unknown)."""
    ctx = pl.read_parquet(ctx_path)
    d = pl.DataFrame({"uid": uids}).join(
        ctx.select("uid", "matched", "link_id", "comment_id", "parent_id"), on="uid", how="left",
        maintain_order="left")
    ok = d["matched"].fill_null(False).to_numpy()
    f = lambda c: np.where(ok, d[c].fill_null("").to_numpy().astype(str), "")  # noqa: E731
    par = np.char.replace(f("parent_id").astype(str), "t1_", "")
    return f("link_id"), f("comment_id"), par


def features(link, cid, par, src_mask, y, var, q_rows, a: float) -> np.ndarray:
    """Features for rows q_rows from labels of rows src_mask (excluding the row itself).
    y: dict task -> labels; var: variety codes (priors per variety over the source rows)."""
    n = len(link)
    out = np.zeros((n, 6))
    for ti, t in enumerate(TASKS):
        prior = {v: y[t][src_mask & (var == v)].mean() for v in (0, 1)}
        # thread sums over source rows
        sums, cnts = {}, {}
        for i in np.flatnonzero(src_mask & (link != "")):
            sums[link[i]] = sums.get(link[i], 0.0) + y[t][i]
            cnts[link[i]] = cnts.get(link[i], 0) + 1
        lab_of = {cid[i]: y[t][i] for i in np.flatnonzero(src_mask & (cid != ""))}
        for i in np.flatnonzero(q_rows & (link != "")):
            s, c = sums.get(link[i], 0.0), cnts.get(link[i], 0)
            if src_mask[i]:  # never use the row's own label
                s, c = s - y[t][i], c - 1
            pr = prior[var[i]]
            if c > 0:
                out[i, ti * 3 + 0] = 1.0
                out[i, ti * 3 + 1] = (s + a * pr) / (c + a) - pr
            if par[i] in lab_of and par[i] != cid[i]:
                out[i, ti * 3 + 2] = lab_of[par[i]] - pr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stk_v5")
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", type=float, default=2.0, help="shrinkage pseudo-count toward the variety prior")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--no-parent", action="store_true")
    a = ap.parse_args()
    df = pool()
    uids, seg, fold = df["uid"].to_numpy(), df["seg"].to_numpy(), df["fold"].to_numpy()
    var, reddit = seg // 2, seg % 2 == 1
    y = {t: df[LABEL[t]].to_numpy().astype(float) for t in TASKS}
    p = pl.DataFrame({"uid": uids}).join(pl.read_parquet(PREDS / f"{a.base}.parquet").filter(pl.col("split") == "oof"),
                                         on="uid", how="left", maintain_order="left")
    base = {t: logit(p[f"p_{t}"].to_numpy()) for t in TASKS}
    link, cid, par = thread_keys(uids)
    drop = [2, 5] if a.no_parent else []

    oof = {t: base[t].copy() for t in TASKS}
    coefs = []
    for k in range(5):
        te = fold == k
        F_te = features(link, cid, par, fold != k, y, var, te, a.a)
        F_tr = np.zeros((len(uids), 6))
        for j in range(5):
            if j == k:
                continue
            F_tr += features(link, cid, par, (fold != k) & (fold != j), y, var, fold == j, a.a)
        F_te, F_tr = np.delete(F_te, drop, 1), np.delete(F_tr, drop, 1)
        tr = fold != k
        for t in TASKS:
            for v in (0, 1):
                m_tr, m_te = tr & reddit & (var == v), te & reddit & (var == v)
                if t == "sarc" and y[t][m_tr].sum() < 5:
                    continue
                X_tr = np.column_stack([base[t][m_tr], F_tr[m_tr]])
                X_te = np.column_stack([base[t][m_te], F_te[m_te]])
                clf = LogisticRegression(C=a.C, max_iter=2000).fit(X_tr, y[t][m_tr])
                oof[t][m_te] = logit(clf.predict_proba(X_te)[:, 1])
                coefs.append((k, t, v, np.round(clf.coef_[0], 3).tolist()))
    cov = {v: float(np.mean(features(link, cid, par, fold != 0, y, var, (fold == 0) & reddit & (var == v), a.a)[
        (fold == 0) & reddit & (var == v), 0])) for v in (0, 1)}
    print("fold-0 reddit coverage (has labelled thread-mate): AU %.2f UK %.2f" % (cov[0], cov[1]))
    for c in coefs[:8]:
        print(c)
    pr = {t: 1 / (1 + np.exp(-oof[t])) for t in TASKS}
    res0 = quick(1 / (1 + np.exp(-base["sent"])), 1 / (1 + np.exp(-base["sarc"])), specs=("segment+ukg",))
    res = quick(pr["sent"], pr["sarc"], specs=("segment+ukg",))
    print("base ", fmt(res0))
    print(a.run, fmt(res))
    name, kw = save_kw(a.run, df)
    save_preds(name, **kw, oof=pl.DataFrame({"uid": uids, "p_sent": pr["sent"], "p_sarc": pr["sarc"]}),
               meta={"kind": "thread_labels", "script": "experiments/classical/thread_labels.py",
                     "base": a.base, "config": vars(a), "quick": res,
                     "description": f"{a.base} + same-thread label features (nested cross-fitted LR per task x variety, reddit rows)"})


if __name__ == "__main__":
    main()
