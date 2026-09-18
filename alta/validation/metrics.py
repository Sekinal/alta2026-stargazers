"""Vectorised metrics with exactly the official semantics.

The official scorer is sklearn ``f1_score(average="macro")`` per task x variety, which
averages over the labels present in y_true OR y_pred (zero_division=0). ``macro_f1_counts``
reproduces that from confusion counts, so it can be evaluated for thousands of bootstrap
replicates or candidate thresholds at once.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from alta.validation.data import SEGMENT_NAMES, TASKS, EvalData

COMPONENTS = ("sent-AU", "sent-UK", "sarc-AU", "sarc-UK")


def macro_f1_counts(tp, fp, fn, tn) -> np.ndarray:
    tp, fp, fn, tn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn, tn))
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fp + fn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
    has_pos = (tp + fn + fp) > 0  # label 1 in y_true or y_pred
    has_neg = (tn + fp + fn) > 0  # label 0 in y_true or y_pred
    n = has_pos.astype(float) + has_neg.astype(float)
    return np.where(n > 0, (f1_pos * has_pos + f1_neg * has_neg) / np.maximum(n, 1), 0.0)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = y.astype(bool), pred.astype(bool)
    return float(macro_f1_counts((y & pred).sum(), (~y & pred).sum(),
                                 (y & ~pred).sum(), (~y & ~pred).sum()))


def components(d: EvalData, pred: dict[str, np.ndarray]) -> dict[str, float]:
    """The four official components and the final score."""
    out = {}
    for t in TASKS:
        for v, vn in ((0, "AU"), (1, "UK")):
            m = d.variety == v
            out[f"{t}-{vn}"] = macro_f1(d.y[t][m], pred[t][m])
    out["score"] = (min(out["sent-AU"], out["sent-UK"]) + min(out["sarc-AU"], out["sarc-UK"])) / 2
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return err / len(y)


def _safe(fn, y, p) -> float:
    return float(fn(y, p)) if 0 < y.sum() < len(y) else float("nan")


def detail(d: EvalData, pred: dict[str, np.ndarray]) -> list[dict]:
    """Per task x variety x (all | segment) diagnostics: F1s, rates, ranking and calibration."""
    rows = []
    seg = d.segment()
    for t in TASKS:
        y, p, yhat = d.y[t], d.p[t], pred[t]
        groups = [(f"{t}-AU", d.variety == 0), (f"{t}-UK", d.variety == 1)]
        groups += [(f"{t}:{SEGMENT_NAMES[s]}", seg == s) for s in range(4)]
        for name, m in groups:
            yy, pp, hh = y[m], p[m], yhat[m].astype(bool)
            tp = int((yy & hh).sum())
            rows.append({
                "group": name, "n": int(m.sum()), "n_pos": int(yy.sum()),
                "prevalence": float(yy.mean()), "pred_rate": float(hh.mean()),
                "macro_f1": macro_f1(yy, hh),
                "precision": tp / max(int(hh.sum()), 1), "recall": tp / max(int(yy.sum()), 1),
                "auc": _safe(roc_auc_score, yy, pp), "ap": _safe(average_precision_score, yy, pp),
                "brier": float(np.mean((pp - yy) ** 2)), "ece": ece(yy, pp),
            })
    return rows


def per_fold(d: EvalData, pred: dict[str, np.ndarray]) -> list[dict]:
    return [{"fold": int(k), **components(d.subset(d.fold == k), {t: v[d.fold == k] for t, v in pred.items()})}
            for k in np.unique(d.fold)]


def length_slices(d: EvalData, pred: dict[str, np.ndarray],
                  edges=(0, 10, 30, 80, 10**9)) -> list[dict]:
    """Reddit-only macro-F1 by word-count bin, per task x variety (where the hard cases live)."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for v, vn in ((0, "AU"), (1, "UK")):
            m = (d.source == 1) & (d.variety == v) & (d.n_words > lo) & (d.n_words <= hi)
            row = {"words": f"{lo + 1}-{hi}" if hi < 10**9 else f">{lo}", "variety": vn, "n": int(m.sum())}
            for t in TASKS:
                row[t] = macro_f1(d.y[t][m], pred[t][m]) if m.any() else float("nan")
            rows.append(row)
    return rows
