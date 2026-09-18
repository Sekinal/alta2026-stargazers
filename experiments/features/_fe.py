"""Shared plumbing for feature-set builders: input loading, text normalisation, CLI, quick evaluation.

Builders define ``build(df) -> pl.DataFrame`` (uid + raw feature columns) and a ``DESCRIPTIONS``
dict, then call ``main(SET, build, DESCRIPTIONS)``. Features never see labels: ``load_input``
drops ``sentiment``/``sarcasm`` before the builder is called. Evaluation (``--eval``) is the quick
check of CONTRACT.md and runs label-using models only through ``alta.validation.harness``.

  uv run python experiments/features/build_<set>.py                      # pool -> store/<set>__pool.parquet
  uv run python experiments/features/build_<set>.py --input data/test.csv --out store/<set>__test.parquet
  uv run python experiments/features/build_<set>.py --eval               # pool + quick evaluation + sidecar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alta.common import frame_fingerprint, load_pool, load_test  # noqa: E402

STORE = Path(__file__).resolve().parent / "store"
BASE_RUN = "cls_e5me5_lr"
LABELS = ("sentiment", "sarcasm")


# ------------------------------------------------------------------ text normalisation
_CONTR = re.compile(r"(\w) (n't|'s|'re|'ve|'ll|'d|'m)\b", re.I)
_OPEN = re.compile(r"([(\[]) ")
_CLOSE = re.compile(r" ([)\]])")
_PUNCT = re.compile(r"(\w) ([?!.,;:])(?=\s|$)")


def normalise(text: str) -> str:
    """Undo the google tokeniser artefacts ("did n't" -> "didn't", "( x" -> "(x", "x ?" -> "x?").

    Features are computed on the normalised text so they do not depend on the formatting of a
    source; reddit text (which has no artefacts) is unchanged apart from whitespace runs.
    """
    t = _CONTR.sub(r"\1\2", text)
    t = _OPEN.sub(r"\1", t)
    t = _CLOSE.sub(r"\1", t)
    t = _PUNCT.sub(r"\1\2", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def load_input(path: str | None) -> pl.DataFrame:
    """uid, source, variety, text (normalised) and text_raw. Labels are dropped here."""
    df = load_pool() if path is None else load_test(path)
    df = df.drop([c for c in LABELS if c in df.columns])
    return df.select("uid", pl.col("source").cast(pl.String), pl.col("variety").cast(pl.String),
                     pl.col("text").alias("text_raw"),
                     pl.col("text").map_elements(normalise, return_dtype=pl.String).alias("text"))


# ------------------------------------------------------------------ CLI
def main(set_name: str, build: Callable[[pl.DataFrame], pl.DataFrame], descriptions: dict[str, str],
         extra_meta: dict | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="'pool' (default) or a test CSV")
    ap.add_argument("--out", default=None, help="output parquet")
    ap.add_argument("--eval", action="store_true", help="pool only: run the quick evaluation, write sidecar")
    a = ap.parse_args()
    if a.input == "pool":
        a.input = None
    t0 = time.time()
    df = load_input(a.input)
    feats = build(df)
    missing = set(descriptions) ^ set(feats.columns[1:])
    if missing:
        raise ValueError(f"descriptions and features disagree: {sorted(missing)}")
    feats = feats.select("uid", *[pl.col(c).cast(pl.Float32).alias(f"{set_name}__{c}") for c in descriptions])
    if feats.height != df.height or not (feats["uid"] == df["uid"]).all():
        raise ValueError("builder changed rows/order")
    arr = feats.drop("uid").to_numpy()
    if not np.isfinite(arr).all():
        raise ValueError("non-finite feature values")
    if a.input is None:
        out = Path(a.out) if a.out else STORE / f"{set_name}__pool.parquet"
    else:
        out = Path(a.out) if a.out else STORE / f"{set_name}__test-{frame_fingerprint(load_test(a.input))[:10]}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(out)
    runtime = time.time() - t0
    print(f"wrote {out} ({feats.height} rows x {feats.width - 1} features, {runtime:.1f}s)")
    if a.eval:
        if a.input is not None:
            raise SystemExit("--eval only on the pool")
        ev = evaluate(feats)
        side = {"set": set_name, "builder": f"experiments/features/build_{set_name}.py",
                "n_features": feats.width - 1, "runtime_s": round(runtime, 1),
                "text": "computed on normalise(text): google tokeniser artefacts undone",
                "features": {f"{set_name}__{k}": v for k, v in descriptions.items()},
                **(extra_meta or {}), "evaluation": ev}
        (STORE / f"{set_name}.json").write_text(json.dumps(side, indent=2))
        print(json.dumps(ev, indent=2))


# ------------------------------------------------------------------ quick evaluation
def _prep(X: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise on training rows; clip to +-5 SD so heavy tails do not dominate."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return np.clip((X - mu) / sd, -5, 5), np.clip((Xte - mu) / sd, -5, 5)


def _lr_per_segment(X, y, seg, Xte, seg_te, C, n_free: int = 0) -> np.ndarray:
    """n_free leading columns (base logits) are scaled x10 after standardising: ~100x weaker penalty,
    so C shrinks only the feature set and not the base model's own evidence."""
    from sklearn.linear_model import LogisticRegression
    p = np.zeros(len(Xte))
    for v in (0, 1, 2, 3):
        m, mt = seg == v, seg_te == v
        if not mt.any():
            continue
        if y[m].sum() < 5 or X.shape[1] == 0:
            p[mt] = y[m].mean()
            continue
        Xs, Xts = _prep(X[m], Xte[mt])
        Xs[:, :n_free] *= 10
        Xts[:, :n_free] *= 10
        p[mt] = LogisticRegression(C=C, max_iter=3000).fit(Xs, y[m]).predict_proba(Xts)[:, 1]
    return p


def _base_logits(uids: np.ndarray) -> dict[str, np.ndarray]:
    from alta.common import PREDS
    p = pl.read_parquet(PREDS / f"{BASE_RUN}.parquet").filter(pl.col("split") == "oof")
    d = pl.DataFrame({"uid": uids}).join(p, on="uid", how="left", maintain_order="left")
    out = {}
    for t in ("sent", "sarc"):
        x = np.clip(d[f"p_{t}"].to_numpy(), 1e-5, 1 - 1e-5)
        out[t] = np.log(x / (1 - x))
    return out


def _seg(df: pl.DataFrame) -> np.ndarray:
    """0 AU-google, 1 AU-reddit, 2 UK-google, 3 UK-reddit."""
    return ((df["variety"].cast(pl.String) == "en-UK").to_numpy().astype(int) * 2
            + (df["source"].cast(pl.String) == "reddit").to_numpy().astype(int))


def make_fit_predict(feats: pl.DataFrame | None, with_base: bool, C: float):
    """fit_predict for cv_fit_predict: per task x segment LR on [base sent+sarc logits] + set."""
    def fp(train: pl.DataFrame, test: pl.DataFrame) -> dict:
        def X(df):
            cols = [np.zeros((df.height, 0))]
            if with_base:
                b = _base_logits(df["uid"].to_numpy())
                cols += [b["sent"][:, None], b["sarc"][:, None]]
            if feats is not None:
                f = df.select("uid").join(feats, on="uid", how="left", maintain_order="left").drop("uid")
                cols.append(f.to_numpy().astype(float))
            return np.hstack(cols)
        Xtr, Xte = X(train), X(test)
        seg, seg_te = _seg(train), _seg(test)
        nf = 2 if with_base else 0
        return {"sent": _lr_per_segment(Xtr, train["sentiment"].to_numpy(), seg, Xte, seg_te, C, nf),
                "sarc": _lr_per_segment(Xtr, train["sarcasm"].to_numpy(), seg, Xte, seg_te, C, nf)}
    return fp


def _slice_stats(d, p, pred) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score
    from alta.validation.metrics import macro_f1
    out = {}
    for name, t, v in (("UKreddit_sarc", "sarc", 1), ("AUreddit_sent", "sent", 0),
                       ("AUreddit_sarc", "sarc", 0)):
        m = (d.source == 1) & (d.variety == v)
        y = d.y[t][m]
        out[name] = {"auc": round(float(roc_auc_score(y, p[t][m])), 4),
                     "logloss": round(float(log_loss(y, np.clip(p[t][m], 1e-5, 1 - 1e-5))), 4),
                     "macroF1": round(macro_f1(y, pred[t][m]), 4)}
    return out


def _score(oof: pl.DataFrame):
    from alta.validation import policies
    from alta.validation.harness import _eval
    from alta.validation.metrics import components, per_fold
    d = _eval(load_pool(), oof)
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    return d, pred, components(d, pred), per_fold(d, pred)


def _r(x: dict) -> dict:
    return {k: round(float(v), 4) for k, v in x.items()}


def _fold_slice_auc(d, name: str) -> list[float]:
    from sklearn.metrics import roc_auc_score
    t, v = {"UKreddit_sarc": ("sarc", 1), "AUreddit_sent": ("sent", 0)}[name]
    out = []
    for k in np.unique(d.fold):
        m = (d.source == 1) & (d.variety == v) & (d.fold == k)
        out.append(float(roc_auc_score(d.y[t][m], d.p[t][m])))
    return out


def evaluate(feats: pl.DataFrame, C_alone: float = 0.1, C_grid=(0.01, 0.03, 0.1), C_main: float = 0.03,
             canary_seeds: int = 3) -> dict:
    """(1) set alone per segment; (2) per segment LR on base logits (+ set) for each C; paired comparison.

    F1-based deltas carry ~0.005 of threshold noise (per-fold recalibration moves the cross-fitted
    thresholds), so the within-slice AUC / log-loss deltas are the more sensitive read-out.
    """
    from alta.validation.harness import canary, cv_fit_predict
    from alta.validation.resampling import paired
    res = {"policy": "segment+ukg", "model": "LR per task x segment on standardised (clipped) features "
           "(segment_only arm = segment priors)", "base_run": BASE_RUN}
    for name, fp in (("segment_only", make_fit_predict(None, False, C_alone)),
                     ("alone", make_fit_predict(feats, False, C_alone))):
        oof, _ = cv_fit_predict(fp)
        d, pred, comp, folds = _score(oof)
        res[name] = {"C": C_alone, **_r(comp), "per_fold": [round(f["score"], 4) for f in folds],
                     "slices": _slice_stats(d, d.p, pred)}
    res["alone"]["canary"] = canary(lambda: cv_fit_predict(make_fit_predict(feats, False, C_alone))[0],
                                    seeds=canary_seeds)
    res["increment"] = {}
    for C in C_grid:
        runs = {}
        for name, f in (("base", None), ("base_plus_set", feats)):
            oof, _ = cv_fit_predict(make_fit_predict(f, True, C))
            d, pred, comp, folds = _score(oof)
            runs[name] = (d, pred, comp, folds, _slice_stats(d, d.p, pred))
        (da, pa, ca, fa, sa), (db, pb, cb, fb, sb) = runs["base_plus_set"], runs["base"]
        pr = paired(da, pa, pb, B=2000)
        inc = {
            "delta": {k: round(ca[k] - cb[k], 4) for k in ca},
            "per_fold_delta": [round(x["score"] - y["score"], 4) for x, y in zip(fa, fb)],
            "p_set_better": {k: round(v["p_a_better"], 3) for k, v in pr.items()},
            "slice_delta": {s: {m: round(sa[s][m] - sb[s][m], 4) for m in ("auc", "logloss", "macroF1")} for s in sb},
            "per_fold_auc_delta": {s: [round(x - y, 4) for x, y in zip(_fold_slice_auc(da, s), _fold_slice_auc(db, s))]
                                   for s in ("UKreddit_sarc", "AUreddit_sent")},
            "mean_component_logloss_delta": round(_mean_ll(da, da.p) - _mean_ll(db, db.p), 5),
        }
        if C == C_main:
            res["base"] = {"C": C, **_r(cb), "per_fold": [round(f["score"], 4) for f in fb], "slices": sb}
            res["base_plus_set"] = {"C": C, **_r(ca), "per_fold": [round(f["score"], 4) for f in fa], "slices": sa}
        res["increment"][f"C={C}"] = inc
    return res


def _mean_ll(d, p) -> float:
    from sklearn.metrics import log_loss
    ll = []
    for t in ("sent", "sarc"):
        for v in (0, 1):
            m = d.variety == v
            ll.append(log_loss(d.y[t][m], np.clip(p[t][m], 1e-5, 1 - 1e-5), labels=[0, 1]))
    return float(np.mean(ll))
