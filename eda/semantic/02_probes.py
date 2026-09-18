"""Linear probes (LogisticRegression) on frozen sentence embeddings, provided 5 folds.

Training-set designs compared (all evaluated OOF on the same held-out rows):
  pooled        one model on all rows; segment one-hot (variety x source) appended to features
  per_variety   separate models for en-AU and en-UK
  per_source    separate models for google and reddit
  per_segment   separate models for each variety x source cell
  cross_variety train on the OTHER variety only (AU->UK, UK->AU): pure transfer
Outputs: cache/probe_oof.parquet (long format), cache/probe_scores.parquet, preds/emb_<model>_lr.
"""

import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from _shared import CACHE, MODELS, emb, pool
from alta.common import N_FOLDS, VARIETIES, save_preds

warnings.filterwarnings("ignore")
TASKS = ("sentiment", "sarcasm")


def fit_predict(X, y, Xte, C):
    if len(np.unique(y)) < 2:  # e.g. UK-google sarcasm has a single positive
        return np.full(len(Xte), float(y.mean()))
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit(X, y)
    return clf.predict_proba(Xte)[:, 1]


def oof_probs(df: pl.DataFrame, X: np.ndarray, design: str, C: float) -> dict[str, np.ndarray]:
    fold = df["fold"].to_numpy()
    var = df["variety"].to_numpy()
    src = df["source"].to_numpy()
    seg = np.char.add(np.char.add(var.astype(str), "|"), src.astype(str))
    onehot = np.stack([(seg == s).astype(np.float32) for s in np.unique(seg)], axis=1)
    out = {t: np.full(len(df), np.nan) for t in TASKS}
    for t in TASKS:
        y = df[t].to_numpy()
        for f in range(N_FOLDS):
            tr, te = fold != f, fold == f
            if design == "pooled":
                Xa = np.hstack([X, onehot])
                out[t][te] = fit_predict(Xa[tr], y[tr], Xa[te], C)
                continue
            if design == "cross_variety":
                for v in VARIETIES:
                    m_tr, m_te = tr & (var != v), te & (var == v)
                    Xa = np.hstack([X, (src == "reddit")[:, None].astype(np.float32)])
                    out[t][m_te] = fit_predict(Xa[m_tr], y[m_tr], Xa[m_te], C)
                continue
            key = {"per_variety": var, "per_source": src, "per_segment": seg}[design]
            for k in np.unique(key):
                m_tr, m_te = tr & (key == k), te & (key == k)
                Xa = X if design == "per_segment" else np.hstack([X, onehot])
                out[t][m_te] = fit_predict(Xa[m_tr], y[m_tr], Xa[m_te], C)
    return out


def scores(df: pl.DataFrame, p: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-variety macro-F1 at 0.5 and at the per-variety F1-optimal threshold (optimistic)."""
    var = df["variety"].to_numpy()
    r = {}
    for t, short in zip(TASKS, ("sent", "sarc")):
        y = df[t].to_numpy()
        for v in VARIETIES:
            m = var == v
            r[f"{short}-{v[3:]}"] = f1_score(y[m], p[t][m] >= 0.5, average="macro")
            grid = np.linspace(0.05, 0.95, 91)
            r[f"{short}-{v[3:]}*"] = max(f1_score(y[m], p[t][m] >= g, average="macro") for g in grid)
    r["score"] = (min(r["sent-AU"], r["sent-UK"]) + min(r["sarc-AU"], r["sarc-UK"])) / 2
    r["score*"] = (min(r["sent-AU*"], r["sent-UK*"]) + min(r["sarc-AU*"], r["sarc-UK*"])) / 2
    return r


def main():
    df = pool()
    rows, long = [], []
    for model in MODELS:
        X = emb(model)
        # pick C on the pooled design (cheap), then reuse for all designs
        best_c, best = None, -1.0
        for C in (0.3, 1.0, 3.0, 10.0):
            s = scores(df, oof_probs(df, X, "pooled", C))["score"]
            print(f"{model} pooled C={C}: {s:.4f}")
            if s > best:
                best_c, best = C, s
        for design in ("pooled", "per_variety", "per_source", "per_segment", "cross_variety"):
            p = oof_probs(df, X, design, best_c)
            s = scores(df, p)
            print(model, design, {k: round(v, 3) for k, v in s.items()})
            rows.append({"model": model, "design": design, "C": best_c, **s})
            long.append(pl.DataFrame({"model": model, "design": design, "uid": df["uid"],
                                      "p_sent": p["sentiment"], "p_sarc": p["sarcasm"]}))
    res = pl.DataFrame(rows)
    res.write_parquet(CACHE / "probe_scores.parquet")
    long = pl.concat(long)
    long.write_parquet(CACHE / "probe_oof.parquet")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=3, tbl_width_chars=250):
        print(res.select("model", "design", "C", "sent-AU", "sent-UK", "sarc-AU", "sarc-UK",
                         "score", "sarc-AU*", "sarc-UK*", "score*"))

    # best design per model -> preds/ (by threshold-0.5 score, the honest number)
    for model in MODELS:
        best = res.filter(pl.col("model") == model).sort("score", descending=True).row(0, named=True)
        oof = long.filter((pl.col("model") == model) & (pl.col("design") == best["design"]))
        path = save_preds(f"emb_{model}_lr", oof.select("uid", "p_sent", "p_sarc"),
                          meta={"design": best["design"], "C": best["C"], "score@0.5": best["score"]})
        print("saved", path, best["design"])


if __name__ == "__main__":
    main()
