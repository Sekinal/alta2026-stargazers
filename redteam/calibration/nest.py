"""Nested pseudo-leaderboard: does the validation stack predict held-out (test-like) scores?

One *draw* = hold out a valid-sized pseudo-test set T with exactly valid.csv's per-cell counts
(variety x source x sentiment x sarcasm), keep the rest R. On R we rebuild everything the stack
would see in reality: fresh stratified-group 5-fold CV (alta.common._assign_folds, draw seed),
a valid-sized stratified "dev" subset (origin == "valid", what the organisers' dev board sees),
OOF probabilities for every model variant, cross-fitted policies, the test-like bootstrap and
paired comparisons. T predictions come from (a) a model refit on all of R and (b) the average of
the 5 CV fold models; the policy is fitted on all of R's OOF (as make_test_submission does).

Draw kinds
  valid       T = the real valid.csv, R = train.csv (fold seed 2026 + k for k = 0..4)
  random      T = stratified random draw from the pool (duplicate texts never go to T)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from alta.common import ROOT, _assign_folds, load_pool
from alta.validation import policies
from alta.validation.data import TASKS, EvalData
from alta.validation.metrics import components
from alta.validation.resampling import replicate_indices, replicate_scores

warnings.filterwarnings("ignore")

EMB = ROOT / "eda" / "semantic" / "cache"
OUT = Path(__file__).resolve().parent / "out"
YCOL = {"sent": "sentiment", "sarc": "sarcasm"}

# name -> (features, design, C); designs follow eda/semantic/02_probes.py
EMB_MODELS = {
    "bgeL_seg_C10": ("bge_large", "per_segment", 10.0),
    "bgeL_seg_C1": ("bge_large", "per_segment", 1.0),
    "bgeL_var_C10": ("bge_large", "per_variety", 10.0),
    "bgeL_pool_C10": ("bge_large", "pooled", 10.0),
    "bgeB_seg_C10": ("bge_base", "per_segment", 10.0),
    "mpnet_seg_C3": ("mpnet", "per_segment", 3.0),
    "minilm_pool_C10": ("minilm", "pooled", 10.0),
}
BLENDS = {"blend3": ("bgeL_seg_C10", "bgeB_seg_C10", "mpnet_seg_C3")}
TFIDF_C = {"sent": 4.0, "sarc": 2.0}
POLICIES = ("fixed", "fixed+ukg", "global+ukg", "variety+ukg", "segment+ukg", "prev+ukg",
            "prevtuned+ukg", "variety+ukg+neg", "prev+ukg+neg", "prevtuned+ukg+neg",
            "segment+ukg+neg")
B = 2000

_URL = r"https?://\S+|www\.\S+"
TOKEN_PATTERN = r"(?u)(?:/s\b|[a-z][a-z0-9]*(?:'[a-z]+)?|\d+)"


def base_pool() -> pl.DataFrame:
    pool = load_pool().with_columns(pl.col("source", "variety", "origin").cast(pl.String))
    norm = (pl.col("text").str.replace_all(r"[‘’ʼ`]", "'").str.replace_all(r"[“”]", '"')
            .str.replace_all(_URL, " xxurl ")
            .str.replace_all(r"(?:^|\s)/?u/[A-Za-z0-9_-]+", " xxuser ")
            .str.replace_all(r"(?:^|\s)/?r/[A-Za-z0-9_]+", " xxsubreddit ").str.to_lowercase())
    return pool.with_columns(
        norm.alias("norm"),
        (pl.col("variety") + "|" + pl.col("source") + "|" + pl.col("sentiment").cast(pl.String)
         + "|" + pl.col("sarcasm").cast(pl.String)).alias("cell"),
        pl.col("text").is_duplicated().alias("dup"),
    ).with_row_index("i")


def _stratified(df: pl.DataFrame, counts: dict[str, int], rng) -> np.ndarray:
    """Row indices (column i) with exactly counts[cell] rows per cell, no duplicated texts."""
    out = []
    for cell, k in counts.items():
        cand = df.filter((pl.col("cell") == cell) & ~pl.col("dup"))["i"].to_numpy()
        out.append(rng.choice(cand, size=k, replace=False))
    return np.sort(np.concatenate(out))


def make_draw(pool: pl.DataFrame, kind: str, seed: int) -> dict:
    rng = np.random.default_rng(10_000 + seed)
    counts = dict(pool.filter(pl.col("origin") == "valid").group_by("cell").len().iter_rows())
    if kind == "valid":
        t_idx = pool.filter(pl.col("origin") == "valid")["i"].to_numpy()
    else:
        t_idx = _stratified(pool, counts, rng)
    is_t = np.zeros(pool.height, bool)
    is_t[t_idx] = True
    R = pool.filter(~pl.Series(is_t))
    dev = _stratified(R, counts, rng)
    R = _assign_folds(R.drop("fold"), seed=2026 + seed).sort("i")
    R = R.with_columns(pl.when(pl.col("i").is_in(dev)).then(pl.lit("valid")).otherwise(pl.lit("train"))
                       .alias("origin"))
    T = pool.filter(pl.Series(is_t)).with_columns(pl.lit(-1).alias("fold"), pl.lit("test").alias("origin"))
    return {"R": R, "T": T}


def to_eval(df: pl.DataFrame, p: dict[str, np.ndarray]) -> EvalData:
    return EvalData(
        uid=df["uid"].to_numpy(),
        variety=(df["variety"] == "en-UK").to_numpy().astype(np.int8),
        source=(df["source"] == "reddit").to_numpy().astype(np.int8),
        fold=df["fold"].to_numpy().astype(np.int8),
        origin=df["origin"].to_numpy(),
        n_words=df["text"].str.split(" ").list.len().to_numpy(),
        p=p,
        y={t: df[YCOL[t]].to_numpy().astype(np.int8) for t in TASKS},
    )


# ---------------------------------------------------------------- models
def _lr(X, y, Xte_list, C, solver="lbfgs"):
    if len(np.unique(y)) < 2 or (solver == "liblinear" and y.sum() < 3):
        return [np.full(Xte.shape[0], float(y.mean())) for Xte in Xte_list]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver=solver).fit(X, y)
    return [clf.predict_proba(Xte)[:, 1] for Xte in Xte_list]


def _groups(df: pl.DataFrame, design: str) -> np.ndarray:
    if design == "per_segment":
        return (df["variety"] + "|" + df["source"]).to_numpy()
    if design == "per_variety":
        return df["variety"].to_numpy()
    return np.zeros(df.height, dtype="<U1").astype(object)


def _onehot(df: pl.DataFrame) -> np.ndarray:
    seg = (df["variety"] + "|" + df["source"]).to_numpy()
    return np.stack([(seg == s).astype(np.float32) for s in
                     ("en-AU|google", "en-AU|reddit", "en-UK|google", "en-UK|reddit")], 1)


def emb_model(R, T, feats: str, design: str, C: float, seed_perturb: int | None = None):
    """Returns oof (R), refit (T), foldavg (T) probabilities per task."""
    X = np.load(EMB / f"emb_{feats}.npy")
    XR, XT = X[R["i"].to_numpy()], X[T["i"].to_numpy()]
    del X
    if design != "per_segment":
        XR, XT = np.hstack([XR, _onehot(R)]), np.hstack([XT, _onehot(T)])
    gR, gT = _groups(R, design), _groups(T, design)
    fold = R["fold"].to_numpy()
    out = {k: {t: np.zeros(n) for t in TASKS} for k, n in
           (("oof", R.height), ("refit", T.height), ("foldavg", T.height))}
    for t in TASKS:
        y = R[YCOL[t]].to_numpy()
        for g in np.unique(gR):
            mR, mT = gR == g, gT == g
            for k in range(5):
                tr, te = mR & (fold != k), mR & (fold == k)
                p_te, p_T = _lr(XR[tr], y[tr], [XR[te], XT[mT]], C)
                out["oof"][t][te] = p_te
                out["foldavg"][t][mT] += p_T / 5
            out["refit"][t][mT] = _lr(XR[mR], y[mR], [XT[mT]], C)[0]
    return out


def _tfidf(fit_txt):
    wv = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, max_features=60_000)
    cv = TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(2, 5), min_df=3,
                         sublinear_tf=True, max_features=150_000)
    wv.fit(fit_txt), cv.fit(fit_txt)
    return lambda txt: sp.hstack([wv.transform(txt), cv.transform(txt)]).tocsr()


def tfidf_model(R, T):
    """Per-segment TF-IDF + LR (the saved tfidf_lr variant)."""
    txtR, txtT = R["norm"].to_list(), T["norm"].to_list()
    gR, gT = _groups(R, "per_segment"), _groups(T, "per_segment")
    fold = R["fold"].to_numpy()
    out = {k: {t: np.zeros(n) for t in TASKS} for k, n in
           (("oof", R.height), ("refit", T.height), ("foldavg", T.height))}
    for k in list(range(5)) + ["all"]:
        tr = fold != k if k != "all" else np.ones(R.height, bool)
        vec = _tfidf([s for s, m in zip(txtR, tr) if m])
        XR, XT = vec(txtR), vec(txtT)
        for t in TASKS:
            y = R[YCOL[t]].to_numpy()
            for g in np.unique(gR):
                mR, mT = gR == g, gT == g
                if k == "all":
                    out["refit"][t][mT] = _lr(XR[mR], y[mR], [XT[mT]], TFIDF_C[t], "liblinear")[0]
                    continue
                te = mR & (fold == k)
                p_te, p_T = _lr(XR[mR & tr], y[mR & tr], [XR[te], XT[mT]], TFIDF_C[t], "liblinear")
                out["oof"][t][te] = p_te
                out["foldavg"][t][mT] += p_T / 5
    return out


def all_models(R, T) -> dict[str, dict]:
    res = {name: emb_model(R, T, *spec) for name, spec in EMB_MODELS.items()}
    res["tfidf_seg"] = tfidf_model(R, T)
    for name, members in BLENDS.items():
        res[name] = {k: {t: np.mean([res[m][k][t] for m in members], 0) for t in TASKS}
                     for k in ("oof", "refit", "foldavg")}
    return res


# ---------------------------------------------------------------- evaluation of one draw
def evaluate(draw_id: str, R, T, models: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, reps_all = [], {}
    idx = None
    for m, pr in models.items():
        dR = to_eval(R, pr["oof"])
        if idx is None:
            idx = replicate_indices(dR, B=B, seed=0)  # same as report(): valid-count replicates
        dev = dR.origin == "valid"
        for spec in POLICIES:
            cf, _ = policies.crossfit(spec, dR)
            c = components(dR, cf)
            cdev = components(dR.subset(dev), {t: cf[t][dev] for t in TASKS})
            fold_scores = [components(dR.subset(dR.fold == k), {t: cf[t][dR.fold == k] for t in TASKS})["score"]
                           for k in range(5)]
            reps = replicate_scores(dR, cf, idx)["score"]
            reps_all[(m, spec)] = reps
            params = policies.fit(spec, dR)
            row = {"draw": draw_id, "model": m, "policy": spec, "cv": c["score"],
                   **{f"cv_{k}": v for k, v in c.items() if k != "score"},
                   "dev": cdev["score"], "fold_sd": float(np.std(fold_scores, ddof=1)),
                   "boot_mean": float(reps.mean()), "boot_sd": float(reps.std(ddof=1)),
                   "boot_q05": float(np.quantile(reps, 0.05)), "boot_q95": float(np.quantile(reps, 0.95))}
            for mode in ("refit", "foldavg"):
                dT = to_eval(T, pr[mode])
                ct = components(dT, policies.apply(params, dT))
                row[f"test_{mode}"] = ct["score"]
                row.update({f"test_{mode}_{k}": v for k, v in ct.items() if k != "score"})
            rows.append(row)
    cand = pl.DataFrame(rows)
    # all pairs: bootstrap p_a_better vs the realised pseudo-test difference
    keys = list(reps_all)
    tab = {(r["model"], r["policy"]): r for r in rows}
    pr_rows = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            diff = reps_all[ka] - reps_all[kb]
            ra, rb = tab[ka], tab[kb]
            pr_rows.append({"draw": draw_id, "model_a": ka[0], "policy_a": ka[1], "model_b": kb[0],
                            "policy_b": kb[1], "p_a_better": float((diff > 0).mean()),
                            "p_tie": float((diff == 0).mean()), "boot_diff": float(diff.mean()),
                            "cv_diff": ra["cv"] - rb["cv"], "dev_diff": ra["dev"] - rb["dev"],
                            "test_refit_diff": ra["test_refit"] - rb["test_refit"],
                            "test_foldavg_diff": ra["test_foldavg"] - rb["test_foldavg"]})
    return cand, pl.DataFrame(pr_rows)


def run_draw(kind: str, seed: int) -> str:
    OUT.mkdir(exist_ok=True)
    draw_id = f"{kind}{seed:02d}"
    path = OUT / f"cand_{draw_id}.parquet"
    if path.exists():
        return draw_id
    pool = base_pool()
    d = make_draw(pool, kind, seed)
    models = all_models(d["R"], d["T"])
    cand, pairs = evaluate(draw_id, d["R"], d["T"], models)
    pairs.write_parquet(OUT / f"pairs_{draw_id}.parquet")
    cand.write_parquet(path)
    return draw_id
