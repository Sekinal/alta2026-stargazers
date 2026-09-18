"""Shared helpers for the statistical red-team scripts (cheap embedding models, EvalData builders)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from alta.common import load_pool
from alta.validation import policies
from alta.validation.data import TASKS, EvalData, pool_eval
from alta.validation.metrics import components

CACHE = ROOT / "eda" / "semantic" / "cache"
YCOL = {"sent": "sentiment", "sarc": "sarcasm"}


def emb(model: str) -> np.ndarray:
    return np.load(CACHE / f"emb_{model}.npy")


def base_eval() -> EvalData:
    return pool_eval(load_pool())


def design(d: EvalData, X: np.ndarray) -> np.ndarray:
    seg = d.segment()
    onehot = np.stack([(seg == s).astype(np.float32) for s in range(4)], axis=1)
    return np.hstack([X, onehot])


def lr(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def fit_predict(X, y, Xte_list, C=1.0, model=None):
    clf = (model or (lambda: lr(C)))()
    clf.fit(X, y)
    return [clf.predict_proba(Z)[:, 1] for Z in Xte_list]


def oof_and_test(X, y, fold, train_mask, test_mask, C=1.0, model=None):
    """OOF probs on train_mask rows (their folds), plus per-fold test probs (n_folds, n_test)."""
    oof = np.full(len(y), np.nan)
    tests = []
    for k in np.unique(fold[train_mask]):
        tr = train_mask & (fold != k)
        te = train_mask & (fold == k)
        a, b = fit_predict(X[tr], y[tr], [X[te], X[test_mask]], C, model)
        oof[te] = a
        tests.append(b)
    return oof, np.array(tests)


def score_policy(d: EvalData, spec: str) -> float:
    pred, _ = policies.crossfit(spec, d)
    return components(d, pred)["score"]
