"""Decision policies: probabilities -> hard labels, fitted on labelled rows, applied elsewhere.

A policy spec is ``base[+rule...]``:

bases
  fixed       threshold 0.5
  global      one threshold per task, maximising that task's min over varieties
  variety     one threshold per task x variety, maximising that component
  segment     one threshold per task x variety x source (coordinate ascent on the component)
  prev        transductive prevalence matching: in every task x segment of the rows being
              labelled, predict the top q fraction as positive, q = prevalence in the fit rows.
              Uses no labels of the labelled rows; valid because test is stratified like valid.
  prevtuned   prev, with a per task x variety multiplier on q tuned on the fit rows

rules
  ukg         sarcasm = 0 for en-UK google (1 positive in 2065 pool rows)
  neg         sentiment = 0 wherever sarcasm is predicted (98% of sarcastic rows are negative)

``crossfit`` gives honest estimates: fit on 4 folds' OOF rows, apply to the 5th.
"""

from __future__ import annotations

import numpy as np

from alta.validation.data import TASKS, EvalData
from alta.validation.metrics import macro_f1_counts

BASES = ("fixed", "global", "variety", "segment", "prev", "prevtuned")
RULES = ("ukg", "neg")
DEFAULT_POLICIES = ("fixed", "fixed+ukg", "global+ukg", "variety+ukg", "segment+ukg",
                    "prev+ukg", "prevtuned+ukg", "variety+ukg+neg", "prev+ukg+neg")
MULTIPLIERS = np.round(np.arange(0.5, 2.01, 0.05), 2)


# ---------------------------------------------------------------- threshold sweeps
def sweep(y: np.ndarray, p: np.ndarray, fixed=(0, 0, 0, 0)):
    """All distinct cut points of p: returns (thresholds, macro-F1) with extra fixed counts.

    Prediction rule everywhere is ``p >= threshold``.
    """
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order].astype(np.int64)
    n, pos = len(ps), int(ys.sum())
    # k = number predicted positive; only cut where the next value is strictly smaller.
    ks = np.concatenate([[0], np.nonzero(np.diff(ps) < 0)[0] + 1, [n]]) if n else np.array([0])
    ks = np.unique(ks)
    tp = np.concatenate([[0], np.cumsum(ys)])[ks]
    fp = ks - tp
    fn = pos - tp
    tn = n - pos - fp
    f = macro_f1_counts(tp + fixed[0], fp + fixed[1], fn + fixed[2], tn + fixed[3])
    thr = np.empty(len(ks))
    for i, k in enumerate(ks):
        if k == 0:
            thr[i] = np.nextafter(ps[0], np.inf) if n else 0.5
        elif k == n:
            thr[i] = ps[-1]
        else:
            thr[i] = (ps[k - 1] + ps[k]) / 2
    return thr, f


def _counts(y, pred):
    y, pred = y.astype(bool), pred.astype(bool)
    return ((y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum(), (~y & ~pred).sum())


def best_threshold(y, p, fixed=(0, 0, 0, 0)) -> float:
    thr, f = sweep(y, p, fixed)
    best = np.flatnonzero(f >= f.max() - 1e-12)
    return float(thr[best[len(best) // 2]])  # middle of the optimal plateau


def _f1_at(y, p, thresholds) -> np.ndarray:
    """macro-F1 at each of many thresholds (vectorised via sorting)."""
    order = np.argsort(p)
    ps, ys = p[order], y[order].astype(np.int64)
    n, pos = len(ps), int(ys.sum())
    k_neg = np.searchsorted(ps, thresholds, side="left")  # rows predicted negative
    cum = np.concatenate([[0], np.cumsum(ys)])
    fn = cum[k_neg]
    tp = pos - fn
    fp = (n - k_neg) - tp
    tn = n - pos - fp
    return macro_f1_counts(tp, fp, fn, tn)


# ---------------------------------------------------------------- fitting
def _fit_base(base: str, d: EvalData) -> dict:
    params: dict = {"base": base}
    if base == "fixed":
        return params
    if base == "global":
        for t in TASKS:
            grid = np.unique(np.quantile(d.p[t], np.linspace(0, 1, 801)))
            f = np.minimum(*[_f1_at(d.y[t][d.variety == v], d.p[t][d.variety == v], grid) for v in (0, 1)])
            best = np.flatnonzero(f >= f.max() - 1e-12)
            params[t] = float(grid[best[len(best) // 2]])
        return params
    if base == "variety":
        for t in TASKS:
            params[t] = {v: best_threshold(d.y[t][d.variety == v], d.p[t][d.variety == v]) for v in (0, 1)}
        return params
    if base == "segment":
        for t in TASKS:
            params[t] = {}
            for v in (0, 1):
                mv = d.variety == v
                start = best_threshold(d.y[t][mv], d.p[t][mv])
                thr = {s: start for s in (0, 1)}
                for _ in range(3):  # coordinate ascent over the two sources
                    for s in (0, 1):
                        m_s, m_o = mv & (d.source == s), mv & (d.source == 1 - s)
                        fixed = _counts(d.y[t][m_o], d.p[t][m_o] >= thr[1 - s])
                        thr[s] = best_threshold(d.y[t][m_s], d.p[t][m_s], fixed)
                params[t].update({(v, s): thr[s] for s in (0, 1)})
        return params
    if base in ("prev", "prevtuned"):
        seg = d.segment()
        for t in TASKS:
            params[t] = {"q": {s: float(d.y[t][seg == s].mean()) for s in range(4)},
                         "mult": {0: 1.0, 1: 1.0}}
            if base == "prevtuned":
                for v in (0, 1):
                    mv = d.variety == v
                    scores = []
                    for mult in MULTIPLIERS:
                        pred = _prevalence_pred(d, t, params[t]["q"], {v: mult, 1 - v: 1.0})
                        scores.append(macro_f1_counts(*_counts(d.y[t][mv], pred[mv])))
                    scores = np.asarray(scores)
                    best = np.flatnonzero(scores >= scores.max() - 1e-12)
                    params[t]["mult"][v] = float(MULTIPLIERS[best[len(best) // 2]])
        return params
    raise ValueError(f"unknown policy base '{base}'")


def _prevalence_pred(d: EvalData, t: str, q: dict, mult: dict) -> np.ndarray:
    pred = np.zeros(len(d), dtype=bool)
    seg = d.segment()
    for s in range(4):
        idx = np.flatnonzero(seg == s)
        k = int(round(min(1.0, q[s] * mult[s // 2]) * len(idx)))
        if k:
            top = idx[np.argsort(-d.p[t][idx], kind="stable")[:k]]
            pred[top] = True
    return pred


def _apply_base(params: dict, d: EvalData) -> dict[str, np.ndarray]:
    base = params["base"]
    pred = {}
    for t in TASKS:
        p = d.p[t]
        if base == "fixed":
            pred[t] = p >= 0.5
        elif base == "global":
            pred[t] = p >= params[t]
        elif base == "variety":
            pred[t] = p >= np.where(d.variety == 0, params[t][0], params[t][1])
        elif base == "segment":
            thr = np.array([params[t][(v, s)] for v, s in zip(d.variety, d.source)])
            pred[t] = p >= thr
        else:
            pred[t] = _prevalence_pred(d, t, params[t]["q"], params[t]["mult"])
    return pred


def parse(spec: str) -> tuple[str, list[str]]:
    base, *rules = spec.split("+")
    if base not in BASES or any(r not in RULES for r in rules):
        raise ValueError(f"bad policy '{spec}'; bases {BASES}, rules {RULES}")
    return base, rules


def fit(spec: str, d: EvalData) -> dict:
    base, rules = parse(spec)
    return {**_fit_base(base, d), "rules": rules, "spec": spec}


def apply(params: dict, d: EvalData) -> dict[str, np.ndarray]:
    pred = _apply_base(params, d)
    if "ukg" in params["rules"]:
        pred["sarc"] = pred["sarc"] & ~((d.variety == 1) & (d.source == 0))
    if "neg" in params["rules"]:
        pred["sent"] = pred["sent"] & ~pred["sarc"]
    return {t: v.astype(np.int8) for t, v in pred.items()}


def crossfit(spec: str, d: EvalData) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Honest labels for every row: policy fitted on the other folds, applied to this fold."""
    pred = {t: np.zeros(len(d), dtype=np.int8) for t in TASKS}
    fitted = []
    for k in np.unique(d.fold):
        test = d.fold == k
        params = fit(spec, d.subset(~test))
        out = apply(params, d.subset(test))
        for t in TASKS:
            pred[t][test] = out[t]
        fitted.append(params)
    return pred, fitted
