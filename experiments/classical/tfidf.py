"""TF-IDF linear models: LR, calibrated linear SVM, NB-LR and NB-SVM (Wang & Manning 2012 style).

Features: word 1-3grams + char_wb 2-6grams, sublinear tf, fitted on the training rows only.
Designs (per task): pooled (+segment one-hots), variety (+source one-hot), segment, augment
(Daume 2007: [X, X*AU, X*UK] + segment one-hots).
NB variants scale every feature by the log-count ratio r = log(p/|p|_1) - log(q/|q|_1) computed on
the training rows of that model (binarised counts, alpha=1), then fit LR / SVM on X * r.
C is chosen per task x cell by 3-fold inner CV (ROC AUC) on the training rows of each outer fold.
SVM probabilities: CalibratedClassifierCV(LinearSVC, cv=3, sigmoid), fitted on training rows only.

OOF: uv run python experiments/classical/tfidf.py --model lr --sent pooled --sarc variety --run cls_tfidf_lr
Test predictions: predict_test.py averages the 5 fold models built with predict().
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from common import CACHE, LABEL, fmt, pool, quick
from alta.common import SEED, save_preds

warnings.filterwarnings("ignore")
TOKEN_PATTERN = r"(?u)(?:/s\b|[a-z][a-z0-9]*(?:'[a-z]+)?|\d+|[!?]+|\.\.\.)"
C_GRID = {"lr": (1.0, 4.0, 16.0), "nblr": (1.0, 4.0, 16.0), "svm": (0.03, 0.1, 0.3), "nbsvm": (0.03, 0.1, 0.3)}


def vectorise(train_texts: list[str], texts: list[str]) -> sp.csr_matrix:
    wv = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 3), min_df=2,
                         sublinear_tf=True, max_features=100_000, dtype=np.float32)
    cv = TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(2, 6), min_df=3,
                         sublinear_tf=True, max_features=200_000, dtype=np.float32)
    wv.fit(train_texts)
    cv.fit(train_texts)
    return sp.hstack([wv.transform(texts), cv.transform(texts)]).tocsr()


def nb_ratio(X: sp.csr_matrix, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    B = (X > 0).astype(np.float32)
    p = alpha + np.asarray(B[y == 1].sum(0)).ravel()
    q = alpha + np.asarray(B[y == 0].sum(0)).ravel()
    return (np.log(p / p.sum()) - np.log(q / q.sum())).astype(np.float32)


def _clf(model: str, C: float):
    if model in ("lr", "nblr"):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear")
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", max_iter=5000), cv=3, method="sigmoid")


def _fit(model: str, X, y, C):
    if model.startswith("nb"):
        r = nb_ratio(X, y)
        clf = _clf(model, C).fit(X @ sp.diags(r), y)
        return lambda Z: clf.predict_proba(Z @ sp.diags(r))[:, 1]
    clf = _clf(model, C).fit(X, y)
    return lambda Z: clf.predict_proba(Z)[:, 1]


def choose_c(model: str, X, y) -> float:
    grid = C_GRID[model]
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    aucs = np.zeros(len(grid))
    for tr, te in skf.split(np.zeros(len(y)), y):
        for i, C in enumerate(grid):
            aucs[i] += roc_auc_score(y[te], _fit(model, X[tr], y[tr], C)(X[te]))
    return grid[int(np.argmax(aucs))]


def fit_predict(model: str, X, y, Xte, log: list | None = None, tag: str = "") -> np.ndarray:
    if y.sum() < 5 or y.sum() > len(y) - 5:  # UK-google sarcasm: 1 positive -> prevalence
        return np.full(Xte.shape[0], (y.sum() + 0.5) / (len(y) + 1.0))
    C = choose_c(model, X, y)
    if log is not None:
        log.append((tag, C))
    return _fit(model, X, y, C)(Xte)


def design_cells(design: str, X, seg):
    """Yields (row mask over all rows, feature matrix) per model cell."""
    n = X.shape[0]
    onehot = sp.csr_matrix(np.stack([(seg == s) for s in range(4)], 1).astype(np.float32))
    if design == "pooled":
        yield np.ones(n, bool), sp.hstack([X, onehot]).tocsr()
    elif design == "augment":
        au = sp.diags((seg < 2).astype(np.float32))
        uk = sp.diags((seg >= 2).astype(np.float32))
        yield np.ones(n, bool), sp.hstack([X, au @ X, uk @ X, onehot]).tocsr()
    elif design == "variety":
        src = sp.csr_matrix((seg % 2).astype(np.float32)[:, None])
        Xs = sp.hstack([X, src]).tocsr()
        for v in (0, 1):
            yield seg // 2 == v, Xs
    elif design == "segment":
        for s in range(4):
            yield seg == s, X
    else:
        raise ValueError(design)


def predict(cfg: dict, train: pl.DataFrame, test: pl.DataFrame, log: list | None = None,
            cache_key: str | None = None) -> dict[str, np.ndarray]:
    """Fit on `train` (frames from common.frame with labels), return task -> P(positive) for `test`."""
    texts = train["norm"].to_list() + test["norm"].to_list()
    path = CACHE / f"tfidf_{cache_key}.npz" if cache_key else None
    if path is not None and path.exists():
        X = sp.load_npz(path).tocsr()
    else:
        X = vectorise(train["norm"].to_list(), texts)
        if path is not None:
            sp.save_npz(path, X)
    seg = np.concatenate([train["seg"].to_numpy(), test["seg"].to_numpy()])
    is_tr = np.arange(len(texts)) < train.height
    out = {}
    for t in ("sent", "sarc"):
        y_all = np.concatenate([train[LABEL[t]].to_numpy(), np.zeros(test.height, np.int8)])
        p = np.full(len(texts), np.nan)
        for mask, Xd in design_cells(cfg[t], X, seg):
            tr, te = mask & is_tr, mask & ~is_tr
            if te.any():
                p[te] = fit_predict(cfg["model"], Xd[tr], y_all[tr], Xd[te], log, f"{t}")
        out[t] = p[~is_tr]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C_GRID))
    ap.add_argument("--sent", default="pooled")
    ap.add_argument("--sarc", default="variety")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = {"model": a.model, "sent": a.sent, "sarc": a.sarc}
    df = pool()
    fold = df["fold"].to_numpy()
    oof = {t: np.zeros(df.height) for t in ("sent", "sarc")}
    log: list = []
    t0 = time.time()
    for k in range(5):
        tr, te = df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k)
        p = predict(cfg, tr, te, log, cache_key=f"pool_fold{k}")
        for t in oof:
            oof[t][fold == k] = p[t]
        print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)
    assert all(np.isfinite(v).all() for v in oof.values())
    res = quick(oof["sent"], oof["sarc"])
    print(a.run, cfg, fmt(res))
    save_preds(a.run, folds_used=df.select("uid", "fold"), oof=pl.DataFrame({"uid": df["uid"], "p_sent": oof["sent"], "p_sarc": oof["sarc"]}),
               meta={"kind": "tfidf", "script": "experiments/classical/tfidf.py", "config": cfg,
                     "features": "word1-3g(100k,min_df2)+char_wb2-6g(200k,min_df3), sublinear tf, fit on train rows",
                     "C_grid": C_GRID[a.model], "C_chosen_inner_cv": [f"{t}:{c}" for t, c in log],
                     "test_protocol": "average of the 5 fold models (predict_test.py)", "quick": res})


if __name__ == "__main__":
    main()
