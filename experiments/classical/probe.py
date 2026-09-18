"""Probes on frozen features from embed.py: LR / linear SVM / small MLP, per design, inner-CV C.

Features are one or more cached embeddings, concatenated ('+'-joined names), standardised on the
training rows. For each task x model cell the regularisation strength is chosen by 3-fold inner CV
(ROC AUC) on the training rows of the outer fold. Designs as in tfidf.py (pooled / variety /
segment); UK-google sarcasm (1 positive) gets the smoothed prevalence.

OOF: uv run python experiments/classical/probe.py --emb bge_large --clf lr --sent segment --sarc segment --run cls_bge_large_lr
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import embed
from common import LABEL, fmt, pool, quick
from alta.common import SEED, save_preds

warnings.filterwarnings("ignore")
GRID = {"lr": (0.001, 0.003, 0.01, 0.03), "svm": (0.0003, 0.001, 0.003), "mlp": (1.0, 3.0, 10.0),
        # xRFM (Beaglehole et al. 2025; pip xrfm): (bandwidth, ridge reg, diagonal M); run with `uv run --with xrfm`
        "xrfm": tuple((bw, reg, diag) for bw in (0.5, 1.0, 2.0, 4.0) for reg in (1e-3, 1e-2, 1e-1, 1.0)
                      for diag in (True, False))}


class XRFMProbe:
    """sklearn-style wrapper: 20% of the rows it is given (never the outer held-out fold) are xRFM's
    internal validation split, used by xRFM to pick the best of its 5 RFM (AGOP) iterations.
    Features are scaled by 1/sqrt(d) so pairwise distances are O(1) for the bandwidth grid."""

    def __init__(self, bw, reg, diag):
        self.bw, self.reg, self.diag = bw, reg, diag

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        from xrfm import xRFM
        self.s = 1 / np.sqrt(X.shape[1])
        A, V, ya, yv = train_test_split(X * self.s, y, test_size=0.2, stratify=y, random_state=SEED)
        p = {"model": {"kernel": "l2_high_dim", "exponent": 1.0, "bandwidth": self.bw, "diag": self.diag,
                       "bandwidth_mode": "constant"},
             "fit": {"reg": self.reg, "iters": 5, "return_best_params": True, "early_stop_rfm": True,
                     "get_agop_best_model": True}}
        self.m = xRFM(rfm_params=p, max_leaf_size=60000, device="cpu", tuning_metric="auc", verbose=False,
                      n_threads=4, random_state=SEED)
        self.m.fit(A.astype(np.float32), ya.astype(np.float32), V.astype(np.float32), yv.astype(np.float32))
        return self

    def predict_proba(self, X):
        p = np.asarray(self.m.predict_proba((X * self.s).astype(np.float32)))
        p = p[:, 1] if p.ndim == 2 else p.ravel()
        return np.column_stack([1 - p, p])


def load_features(names: str, test=None) -> np.ndarray:
    return np.hstack([embed.load(n, test) for n in names.split("+")]).astype(np.float32)


def _make(clf: str, c: float):
    if clf == "lr":
        return LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    if clf == "xrfm":
        return XRFMProbe(*c)
    if clf == "svm":
        return CalibratedClassifierCV(LinearSVC(C=c, class_weight="balanced", max_iter=5000), cv=3)
    return MLPClassifier((256,), alpha=c, early_stopping=False, max_iter=60, learning_rate_init=1e-3,
                         batch_size=128, random_state=SEED)


def _fit(clf, X, y, c):
    if clf == "mlp":  # no class weights in sklearn's MLP: oversample the minority class
        rng = np.random.default_rng(SEED)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        idx = np.concatenate([big, rng.choice(small, len(big), replace=True)])
        return _make(clf, c).fit(X[idx], y[idx])
    return _make(clf, c).fit(X, y)


def choose(clf, X, y):
    grid = GRID[clf]
    if len(grid) == 1:
        return grid[0]
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    auc = np.zeros(len(grid))
    for tr, te in skf.split(X, y):
        for i, c in enumerate(grid):
            auc[i] += roc_auc_score(y[te], _fit(clf, X[tr], y[tr], c).predict_proba(X[te])[:, 1])
    return grid[int(np.argmax(auc))]


def cells(design: str, seg: np.ndarray):
    if design == "pooled":
        yield np.ones(len(seg), bool)
    elif design == "variety":
        for v in (0, 1):
            yield seg // 2 == v
    elif design == "segment":
        for s in range(4):
            yield seg == s
    else:
        raise ValueError(design)


def with_seg(X, seg, design):
    """Pooled models get segment one-hots, variety models a source flag (scaled like the rest)."""
    if design == "pooled":
        return np.hstack([X, np.stack([(seg == s) for s in range(4)], 1).astype(np.float32) * 3])
    if design == "variety":
        return np.hstack([X, (seg % 2)[:, None].astype(np.float32) * 3])
    return X


def predict(cfg: dict, Xtr, ytr: dict, segtr, Xte, segte, log: list | None = None) -> dict[str, np.ndarray]:
    out = {}
    for t in ("sent", "sarc"):
        d = cfg[t]
        p = np.full(len(segte), np.nan)
        for m_tr, m_te in zip(cells(d, segtr), cells(d, segte)):
            if not m_te.any():
                continue
            y = ytr[t][m_tr]
            if y.sum() < 5 or y.sum() > len(y) - 5:
                p[m_te] = (y.sum() + 0.5) / (len(y) + 1.0)
                continue
            sc = StandardScaler().fit(Xtr[m_tr])
            A = with_seg(sc.transform(Xtr[m_tr]), segtr[m_tr], d)
            B = with_seg(sc.transform(Xte[m_te]), segte[m_te], d)
            c = choose(cfg["clf"], A, y)
            if log is not None:
                log.append(f"{t}:{c}")
            p[m_te] = _fit(cfg["clf"], A, y, c).predict_proba(B)[:, 1]
        out[t] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--clf", default="lr", choices=list(GRID))
    ap.add_argument("--sent", default="segment")
    ap.add_argument("--sarc", default="segment")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = {"emb": a.emb, "clf": a.clf, "sent": a.sent, "sarc": a.sarc}
    df = pool()
    X = load_features(a.emb)
    fold, seg = df["fold"].to_numpy(), df["seg"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    oof = {t: np.zeros(df.height) for t in y}
    log: list = []
    t0 = time.time()
    for k in range(5):
        tr, te = fold != k, fold == k
        p = predict(cfg, X[tr], {t: v[tr] for t, v in y.items()}, seg[tr], X[te], seg[te], log)
        for t in oof:
            oof[t][te] = p[t]
    assert all(np.isfinite(v).all() for v in oof.values())
    res = quick(oof["sent"], oof["sarc"])
    print(f"{a.run} {cfg} {time.time() - t0:.0f}s", fmt(res), flush=True)
    save_preds(a.run, folds_used=df.select("uid", "fold"), oof=pl.DataFrame({"uid": df["uid"], "p_sent": oof["sent"], "p_sarc": oof["sarc"]}),
               meta={"kind": "probe", "script": "experiments/classical/probe.py", "config": cfg,
                     "grid": GRID[a.clf], "chosen_inner_cv": log, "test_protocol": "average of the 5 fold models (predict_test.py)",
                     "quick": res})


if __name__ == "__main__":
    main()
