"""Level-2 stacking over runs in preds/, cross-fitted on the shared folds, with honest member selection.

Level-2 inputs for task t: logit(p_t) of every member; optionally ('xtask') logit(p_other) of every
member (sarcasm is ~97% negative, so sentiment evidence helps sarcasm and vice versa); segment
indicators; optionally ('meta') a few handcrafted features from features.py.
Level-2 models:
  lr    one L2 logistic regression per task x variety (source flag inside), C fixed
  lgbm  one LightGBM per task on all rows (segment as feature), small trees, monotone increasing in
        the task's own member logits; rounds chosen by inner CV over the training folds only
Cross-fitting: the stacker that predicts fold k is fitted on the OOF rows of the other 4 folds.
Member selection (--select): greedy forward selection by a smooth criterion (mean log-loss over the
four task x variety components of the cross-fitted stacker). To get an honest score for the whole
procedure, selection is NESTED: for outer fold k it runs on the other 4 folds only (inner
cross-fitting by their fold ids), then the chosen stacker predicts fold k. The final member set
(stored in meta, used for test) is the selection on all 5 folds.
Test: the 5 fold stackers applied to the members' test predictions (split == 'test'), averaged.

  uv run python experiments/classical/stack.py --run stk_lr_x --l2 lr --xtask RUN [RUN ...]
  uv run python experiments/classical/stack.py --run stk_sel --l2 lr --xtask --select RUN [RUN ...]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common import LABEL, ROOT, fmt, pool, quick
from features import handcrafted
from alta.common import CANARY_ENV, PREDS, fold_hash, save_preds

warnings.filterwarnings("ignore")
EPS = 1e-5
TASKS = ("sent", "sarc")
META_COLS = ("log_words", "q_marks", "excl", "quotes", "caps_ratio", "sarc_phr", "profanity",
             "pol_au", "pol_uk", "second_person")
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  num_threads=4, verbose=-1, seed=2026)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_members(runs: list[str], uids: np.ndarray, split: str = "oof") -> dict[str, dict[str, np.ndarray]]:
    """run -> task -> logits aligned with uids."""
    out = {}
    order = pl.DataFrame({"uid": uids})
    fh = fold_hash()
    seed = os.environ.get(CANARY_ENV)
    for r in runs:
        if seed is not None and split == "oof":
            # canary: the whole chain must be shuffled, so members must be their own canary runs
            r_src = f"{r}__canary{seed}"
            if not (PREDS / f"{r_src}.parquet").exists():
                raise FileNotFoundError(f"canary needs {r_src}: run {r}'s script with {CANARY_ENV}={seed}")
        else:
            r_src = r
        mh = json.loads((PREDS / f"{r_src}.json").read_text()).get("fold_hash")
        if split == "oof" and mh is not None and mh != fh:
            raise ValueError(f"{r}: OOF made on different folds ({mh} != {fh})")
        p = pl.read_parquet(PREDS / f"{r_src}.parquet").filter(pl.col("split") == split)
        d = order.join(p, on="uid", how="left", maintain_order="left")
        if d["p_sent"].null_count() or d["p_sarc"].null_count():
            raise ValueError(f"{r}: missing {split} predictions")
        out[r] = {t: logit(d[f"p_{t}"].to_numpy()) for t in TASKS}
    return out


STORE = ROOT / "experiments" / "features" / "store"


def load_feats(sets: list[str], uids: np.ndarray, test_path=None) -> np.ndarray | None:
    """Label-free feature sets from experiments/features/store (CONTRACT.md), standardised with the
    POOL mean/sd (unsupervised; the same transform is applied to test rows)."""
    if not sets:
        return None
    from alta.common import frame_fingerprint, load_test
    order = pl.DataFrame({"uid": uids})
    out = []
    for s_ in sets:
        pool_f = pl.read_parquet(STORE / f"{s_}__pool.parquet")
        cols = [c for c in pool_f.columns if c != "uid"]
        ref = pool_f.select(cols).to_numpy().astype(np.float64)
        mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0) + 1e-9
        src = pool_f if test_path is None else pl.read_parquet(
            STORE / f"{s_}__test-{frame_fingerprint(load_test(test_path))[:10]}.parquet")
        d = order.join(src, on="uid", how="left", maintain_order="left")
        if d.select(cols).null_count().sum_horizontal().item():
            raise ValueError(f"feature set {s_}: missing rows or NaNs")
        out.append((d.select(cols).to_numpy() - mu) / sd)
    return np.hstack(out)


def design(M: dict, runs: list[str], t: str, cfg: dict, seg: np.ndarray, meta: np.ndarray | None,
           extra: np.ndarray | None = None):
    """Feature matrix for task t; returns (X, monotone flags). extra = standardised feature-set columns."""
    other = "sarc" if t == "sent" else "sent"
    cols = [M[r][t] for r in runs]
    mono = [1] * len(runs)
    if cfg["xtask"]:
        cols += [M[r][other] for r in runs]
        mono += [0] * len(runs)
    if cfg["l2"] == "lgbm":
        cols += [seg.astype(float)]
    elif cfg["l2"] == "xrfm":
        cols += [(seg == g).astype(float) for g in range(4)]
        mono += [0] * 3
    else:
        cols += [(seg % 2).astype(float)]
    mono += [0]
    if cfg["meta"] and meta is not None:
        cols += list(meta.T)
        mono += [0] * meta.shape[1]
    if extra is not None:
        cols += list(extra.T)
        mono += [0] * extra.shape[1]
    return np.column_stack(cols), mono


def _lgb_rounds(X, y, fold, mono) -> int:
    """Inner CV over the training folds: mean best iteration."""
    best = []
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        m = lgb.train({**LGB_PARAMS, "monotone_constraints": mono}, lgb.Dataset(X[tr], y[tr]), 1000,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        best.append(m.best_iteration or 1000)
    return int(np.mean(best))


XRFM_GRID = [(bw, reg) for bw in (1.0, 4.0) for reg in (1e-2, 1.0)]


def _xrfm(X, y, fold, Xte) -> np.ndarray:
    """xRFM level 2 (run with `uv run --with xrfm`): standardised features; (bandwidth, reg) chosen by
    inner CV over the training folds (mean AUC), diagonal M to keep it regularised."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from probe import XRFMProbe
    auc = np.zeros(len(XRFM_GRID))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        sc = StandardScaler().fit(X[tr])
        for i, (bw, reg) in enumerate(XRFM_GRID):
            m = XRFMProbe(bw, reg, True).fit(sc.transform(X[tr]), y[tr])
            auc[i] += roc_auc_score(y[va], m.predict_proba(sc.transform(X[va]))[:, 1])
    bw, reg = XRFM_GRID[int(np.argmax(auc))]
    sc = StandardScaler().fit(X)
    return XRFMProbe(bw, reg, True).fit(sc.transform(X), y).predict_proba(sc.transform(Xte))[:, 1]


def fit_predict(cfg, X, y, var, fold, Xte, var_te, mono) -> np.ndarray:
    p = np.zeros(len(Xte))
    if cfg["l2"] == "lr":
        for v in (0, 1):
            m, mt = var == v, var_te == v
            if not mt.any():
                continue
            if y[m].sum() < 5:
                p[mt] = y[m].mean()
                continue
            clf = LogisticRegression(C=cfg["C"], max_iter=2000).fit(X[m], y[m])
            p[mt] = clf.predict_proba(Xte[mt])[:, 1]
        return p
    if cfg["l2"] == "xrfm":
        return _xrfm(X, y, fold, Xte)
    n = _lgb_rounds(X, y, fold, mono)
    m = lgb.train({**LGB_PARAMS, "monotone_constraints": mono}, lgb.Dataset(X, y), n)
    return m.predict(Xte)


FEATS: dict = {}  # task -> standardised feature-set matrix for the pool (set in main)


def crossfit(cfg, runs, M, y, seg, fold, meta, rows=None) -> dict[str, np.ndarray]:
    """Cross-fitted stacker probabilities for `rows` (bool mask; default all) by their fold ids."""
    rows = np.ones(len(seg), bool) if rows is None else rows
    out = {t: np.full(len(seg), np.nan) for t in TASKS}
    var = seg // 2
    for t in TASKS:
        X, mono = design(M, runs, t, cfg, seg, meta, FEATS.get(t))
        for k in np.unique(fold[rows]):
            tr, te = rows & (fold != k), rows & (fold == k)
            out[t][te] = fit_predict(cfg, X[tr], y[t][tr], var[tr], fold[tr], X[te], var[te], mono)
    return out


def criterion(p, y, seg, rows) -> float:
    """Mean log-loss over the task x variety components (lower is better)."""
    ll = []
    for t in TASKS:
        for v in (0, 1):
            m = rows & (seg // 2 == v)
            ll.append(log_loss(y[t][m], np.clip(p[t][m], EPS, 1 - EPS), labels=[0, 1]))
    return float(np.mean(ll))


def forward_select(cfg, cands, M, y, seg, fold, meta, rows, tol=5e-4, max_k=8) -> list[str]:
    chosen, best = [], np.inf
    while len(chosen) < max_k:
        trial = {}
        for r in cands:
            if r in chosen:
                continue
            p = crossfit(cfg, chosen + [r], M, y, seg, fold, meta, rows)
            trial[r] = criterion(p, y, seg, rows)
        if not trial:
            break
        r, c = min(trial.items(), key=lambda kv: kv[1])
        if c > best - tol:
            break
        chosen.append(r)
        best = c
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--run", required=True)
    ap.add_argument("--l2", default="lr", choices=["lr", "lgbm", "xrfm"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--xtask", action="store_true")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--feats-sent", default="", help="comma-separated feature sets (experiments/features/store)")
    ap.add_argument("--feats-sarc", default="")
    a = ap.parse_args()
    cfg = {"l2": a.l2, "C": a.C, "xtask": a.xtask, "meta": a.meta,
           "feats": {"sent": [x for x in a.feats_sent.split(",") if x], "sarc": [x for x in a.feats_sarc.split(",") if x]}}
    df = pool()
    uids, seg, fold = df["uid"].to_numpy(), df["seg"].to_numpy(), df["fold"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in TASKS}
    M = load_members(a.runs, uids)
    for t in TASKS:
        FEATS[t] = load_feats(cfg["feats"][t], uids)
    meta = None
    if a.meta:
        H, names = handcrafted(df)
        meta = H[:, [names.index(c) for c in META_COLS]]
    extra = {}
    if a.select:
        oof = {t: np.zeros(len(seg)) for t in TASKS}
        picks = []
        for k in range(5):
            inner = fold != k
            chosen = forward_select(cfg, a.runs, M, y, seg, fold, meta, inner)
            picks.append(chosen)
            print(f"outer fold {k}: {chosen}", flush=True)
            for t in TASKS:
                X, mono = design(M, chosen, t, cfg, seg, meta, FEATS.get(t))
                te = fold == k
                oof[t][te] = fit_predict(cfg, X[inner], y[t][inner], seg[inner] // 2, fold[inner],
                                         X[te], seg[te] // 2, mono)
        members = forward_select(cfg, a.runs, M, y, seg, fold, meta, np.ones(len(seg), bool))
        print("final selection on all folds:", members)
        extra = {"candidates": a.runs, "nested_picks": picks}
    else:
        members = a.runs
        oof = crossfit(cfg, members, M, y, seg, fold, meta)
    res = quick(oof["sent"], oof["sarc"])
    print(a.run, cfg, fmt(res), flush=True)
    save_preds(a.run, folds_used=df.select("uid", "fold"), oof=pl.DataFrame({"uid": uids, "p_sent": oof["sent"], "p_sarc": oof["sarc"]}),
               meta={"kind": "stack", "script": "experiments/classical/stack.py", "config": cfg,
                     "members": members, **extra, "quick": res,
                     "test_protocol": "average of the 5 fold stackers applied to members' test preds"})


def predict_test(run: str, test_df: pl.DataFrame, test_path=None) -> dict[str, np.ndarray]:
    """Average of the 5 fold stackers (each fitted on the other folds' OOF rows) on the members' test preds."""
    meta = json.loads((PREDS / f"{run}.json").read_text())
    cfg, members = meta["config"], meta["members"]
    df = pool()
    seg, fold = df["seg"].to_numpy(), df["fold"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in TASKS}
    M = load_members(members, df["uid"].to_numpy())
    Mt = load_members(members, test_df["uid"].to_numpy(), split="test")
    seg_t = test_df["seg"].to_numpy()
    meta_tr = meta_te = None
    if cfg["meta"]:
        H, names = handcrafted(df)
        Ht, _ = handcrafted(test_df)
        idx = [names.index(c) for c in META_COLS]
        meta_tr, meta_te = H[:, idx], Ht[:, idx]
    out = {}
    for t in TASKS:
        fs = cfg.get("feats", {}).get(t, [])
        X, mono = design(M, members, t, cfg, seg, meta_tr, load_feats(fs, df["uid"].to_numpy()))
        Xt, _ = design(Mt, members, t, cfg, seg_t, meta_te, load_feats(fs, test_df["uid"].to_numpy(), test_path))
        ps = []
        for k in range(5):
            tr = fold != k
            ps.append(fit_predict(cfg, X[tr], y[t][tr], seg[tr] // 2, fold[tr], Xt, seg_t // 2, mono))
        out[t] = np.mean(ps, 0)
    return out


if __name__ == "__main__":
    main()
