"""Combine runs into new runs (saved like any other run, so they get the same validation).

blend  simple average of probabilities (mean) or of logits (logit); no fitting.
stack  per task x variety logistic regression on member logits + source indicator.
       OOF is cross-fitted (stacker for fold k never sees fold k labels); test predictions
       come from a stacker fitted on the full OOF pool, applied to members' test probabilities.
       Caveat: a policy cross-fitted on top of stacked OOF sees second-order leakage
       (fold j's stacker saw fold k), which is small with a 2-10 feature stacker.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from alta.common import load_pool, save_preds
from alta.validation.data import TASKS, load_oof, read_run

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _member_frames(runs: list[str]) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Wide frames uid -> p_sent_<run>, p_sarc_<run> for oof and (if all have it) test."""
    oof, test, have_test = None, None, True
    for r in runs:
        preds, _ = read_run(r)
        for split in ("oof", "test"):
            part = preds.filter(pl.col("split") == split).select(
                "uid", pl.col("p_sent").alias(f"p_sent_{r}"), pl.col("p_sarc").alias(f"p_sarc_{r}"))
            if split == "oof":
                oof = part if oof is None else oof.join(part, on="uid", how="inner")
            elif part.height == 0:
                have_test = False
            else:
                test = part if test is None else test.join(part, on="uid", how="inner")
    return oof, (test if have_test else None)


def blend(runs: list[str], name: str, method: str = "mean") -> str:
    oof, test = _member_frames(runs)

    def combine(frame):
        out = frame.select("uid")
        for t in ("sent", "sarc"):
            m = np.column_stack([frame[f"p_{t}_{r}"].to_numpy() for r in runs])
            p = m.mean(1) if method == "mean" else 1 / (1 + np.exp(-_logit(m).mean(1)))
            out = out.with_columns(pl.Series(f"p_{t}", p))
        return out

    save_preds(name, combine(oof), None if test is None else combine(test),
               meta={"kind": "blend", "method": method, "members": runs})
    return name


def stack(runs: list[str], name: str, C: float = 1.0) -> str:
    pool = load_pool()
    members = [load_oof(r, pool) for r in runs]
    base = members[0]
    _, test = _member_frames(runs)

    def features(ds_p, variety, source):
        return np.column_stack([_logit(p) for p in ds_p] + [source.astype(float)])

    oof_p = {t: np.zeros(len(base)) for t in TASKS}
    test_p = None
    if test is not None:
        test_seg = pl.DataFrame({"uid": test["uid"]}).join(
            _test_segments(), on="uid", how="left")
        test_p = {t: np.zeros(test.height) for t in TASKS}
    for t in TASKS:
        for v in (0, 1):
            mv = base.variety == v
            X = features([m.p[t] for m in members], base.variety, base.source)
            y = base.y[t]
            if y[mv].sum() < 5:  # degenerate (never happens per variety, kept for safety)
                oof_p[t][mv] = X[mv, 0]
                continue
            for k in np.unique(base.fold):
                tr, te = mv & (base.fold != k), mv & (base.fold == k)
                clf = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr])
                oof_p[t][te] = clf.predict_proba(X[te])[:, 1]
            if test_p is not None:
                clf = LogisticRegression(C=C, max_iter=1000).fit(X[mv], y[mv])
                tv = (test_seg["variety"] == v).to_numpy()
                Xt = np.column_stack([_logit(test[f"p_{t}_{r}"].to_numpy()) for r in runs]
                                     + [test_seg["source"].to_numpy().astype(float)])
                test_p[t][tv] = clf.predict_proba(Xt[tv])[:, 1]
    oof = pl.DataFrame({"uid": base.uid, "p_sent": oof_p["sent"], "p_sarc": oof_p["sarc"]})
    tst = None if test_p is None else pl.DataFrame(
        {"uid": test["uid"], "p_sent": test_p["sent"], "p_sarc": test_p["sarc"]})
    save_preds(name, oof, tst, meta={"kind": "stack", "members": runs, "C": C})
    return name


def _test_segments() -> pl.DataFrame:
    from alta.common import DATA, load_test
    t = load_test(DATA / "test.csv")
    return t.select("uid", (pl.col("variety") == "en-UK").cast(pl.Int8).alias("variety"),
                    (pl.col("source") == "reddit").cast(pl.Int8).alias("source"))
