"""Leak-proof CV harness and permutation canary.

Red-team (redteam/stats/04, 10, 11) showed that leaky pipelines pass every output-only check:
supervised preprocessing on all labels (+0.043), in-sample predictions (+0.037), valid-only
leaks, and runs trained on other folds all looked honest. Two defences:

1. ``cv_fit_predict``: the harness owns the fold loop. ``fit_predict`` only ever receives the
   training folds WITH labels and the held-out fold WITHOUT labels; the folds used are returned
   so ``save_preds(..., folds_used=...)`` can prove which folds produced the OOF.
2. Permutation canary: re-run the SAME pipeline with ``ALTA_CANARY_SEED`` set, which makes
   ``alta.common.load_pool`` return labels shuffled within each segment. Because the shuffle
   happens at the only label source, it also catches leaks outside the harness (global label
   lookups, preprocessing fitted once on all rows). An honest pipeline then scores at the
   no-information floor (~0.64 under segment+ukg: the segment alone predicts sentiment); a
   leaky one scores above it.

In-process (cheap models)::

    oof, folds_used = cv_fit_predict(fit_predict)
    result = canary(lambda: cv_fit_predict(fit_predict)[0], seeds=3)
    save_preds(run, oof, meta={..., "canary": result}, folds_used=folds_used)

Script-based (any model)::

    for s in 0 1 2: ALTA_CANARY_SEED=$s uv run python train.py --run RUN   # saves RUN__canary$s
    uv run python -m alta.validate canary RUN        # scores them, writes the result into RUN's meta

For expensive models (fine-tuned encoders/LLMs) one seed is acceptable; say so in meta.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Callable

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from alta.common import CANARY_ENV, PREDS, load_pool
from alta.validation import policies
from alta.validation.data import TASKS, EvalData, _frame_to_eval, read_run
from alta.validation.metrics import components

FitPredict = Callable[[pl.DataFrame, pl.DataFrame], dict]
LABELS = ("sentiment", "sarcasm")
CANARY_MARGIN = 0.02    # FAIL if shuffled-label score exceeds the noise floor by more than this
CANARY_AUC_SE = 3.5     # ... or if any within-segment AUC on shuffled labels is > 0.5 + 3.5 SE
#                         (3.5 not 3: up to 48 tests over 3 seeds keeps false alarms ~1%)


def cv_fit_predict(fit_predict: FitPredict, pool: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the fixed folds; returns (oof frame uid/p_sent/p_sarc, folds_used uid/fold)."""
    pool = load_pool() if pool is None else pool
    parts = []
    for k in sorted(pool["fold"].unique().to_list()):
        train = pool.filter(pl.col("fold") != k)
        test = pool.filter(pl.col("fold") == k).drop(*LABELS)
        out = fit_predict(train, test)
        for t in TASKS:
            p = np.asarray(out[t], dtype=np.float64)
            if p.shape != (test.height,):
                raise ValueError(f"fit_predict returned {p.shape} for {t}, expected ({test.height},)")
        parts.append(pl.DataFrame({"uid": test["uid"], "p_sent": out["sent"], "p_sarc": out["sarc"]}))
    return pl.concat(parts), pool.select("uid", "fold")


@contextmanager
def canary_labels(seed: int):
    old = os.environ.get(CANARY_ENV)
    os.environ[CANARY_ENV] = str(seed)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(CANARY_ENV, None)
        else:
            os.environ[CANARY_ENV] = old


def _eval(pool: pl.DataFrame, oof: pl.DataFrame) -> EvalData:
    df = pool.join(oof, on="uid", how="left", maintain_order="left")
    if df["p_sent"].null_count():
        raise ValueError("canary OOF does not cover the pool")
    return _frame_to_eval(df, labelled=True, meta={})


def _headline(d: EvalData) -> float:
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    return components(d, pred)["score"]


def noise_floor(seeds: int = 5) -> float:
    """Headline score of uniform-noise probabilities (what 'no information beyond segment' earns)."""
    pool = load_pool()
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        oof = pl.DataFrame({"uid": pool["uid"], "p_sent": rng.random(pool.height), "p_sarc": rng.random(pool.height)})
        vals.append(_headline(_eval(pool, oof)))
    return float(np.mean(vals))


def score_canary(oofs: dict[int, pl.DataFrame]) -> dict:
    """oofs: canary seed -> OOF produced under that seed. Scored against the shuffled labels."""
    floor = noise_floor()
    scores, z = [], []
    for seed, oof in sorted(oofs.items()):
        with canary_labels(seed):
            shuffled = load_pool()
        d = _eval(shuffled, oof)
        scores.append(_headline(d))
        seg = d.segment()
        for t in TASKS:  # within-segment AUC vs 0.5 (shuffling removed all within-segment signal)
            for g in range(4):
                m = seg == g
                y, p = d.y[t][m], d.p[t][m]
                n1, n0 = int(y.sum()), int((1 - y).sum())
                if n1 >= 10 and n0 >= 10:
                    se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
                    z.append((roc_auc_score(y, p) - 0.5) / se)
    mean = float(np.mean(scores))
    zmax = float(np.max(z)) if z else 0.0
    return {"passed": bool(mean <= floor + CANARY_MARGIN and zmax <= CANARY_AUC_SE),
            "shuffled_scores": [round(x, 4) for x in scores], "mean": round(mean, 4),
            "floor": round(floor, 4), "margin": CANARY_MARGIN, "max_auc_z": round(zmax, 2),
            "seeds": sorted(oofs)}


def canary(run_pipeline: Callable[[], pl.DataFrame], seeds: int = 3) -> dict:
    """In-process canary: ``run_pipeline()`` must produce OOF using labels from load_pool()."""
    oofs = {}
    for s in range(seeds):
        with canary_labels(s):
            oofs[s] = run_pipeline().select("uid", "p_sent", "p_sarc")
    return score_canary(oofs)


def canary_from_runs(run: str) -> dict:
    """Score saved RUN__canary<seed> runs and record the result in RUN's meta json."""
    seeds = sorted(int(p.stem.rsplit("__canary", 1)[1]) for p in PREDS.glob(f"{run}__canary*.parquet"))
    if not seeds:
        raise FileNotFoundError(f"no {run}__canary<seed> runs; run the pipeline with {CANARY_ENV}=<seed>")
    oofs = {s: read_run(f"{run}__canary{s}")[0].filter(pl.col("split") == "oof") for s in seeds}
    result = score_canary(oofs)
    meta_path = PREDS / f"{run}.json"
    meta = json.loads(meta_path.read_text())
    meta["canary"] = result
    meta_path.write_text(json.dumps(meta, indent=2))
    return result
