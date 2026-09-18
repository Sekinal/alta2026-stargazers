"""Gradient-boosted trees on dense features (lane 3a): handcrafted + frozen-classifier logits +
SVD of TF-IDF + PCA of a sentence embedding. One model per task on all rows, segment as feature.

All unsupervised projections (TF-IDF vocabulary/idf, SVD, PCA) are fitted on the training rows of
the outer fold. Boosting rounds are chosen by inner CV over the training folds (mean best iteration),
never on the held-out fold.

  uv run python experiments/classical/gbdt.py --algo lgbm --emb e5_large --run cls_gbdt_lgbm
"""

from __future__ import annotations

import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, TruncatedSVD

import embed
from common import CACHE, LABEL, fmt, pool, quick
from features import handcrafted
from tfidf import vectorise
from alta.common import SEED, save_preds

warnings.filterwarnings("ignore")
CLF_LOGITS = {"tw_irony": 2, "tw_sent": 3, "go_emo": 28}  # leading logit columns of the clf features
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30,
           feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
           num_threads=4, verbose=-1, seed=SEED)


def nested_probe_logits(train: pl.DataFrame, test: pl.DataFrame, emb_tr, emb_te) -> tuple[np.ndarray, np.ndarray]:
    """Supervised projection: LR-probe logits (config of cls_e5me5_lr) for sentiment and sarcasm.
    Training rows get INNER-CV logits (probe fitted on the other training folds only); the held-out rows
    get the probe fitted on all training rows. No logit ever comes from a model that saw its row."""
    import probe
    cfg = {"clf": "lr", "sent": "pooled", "sarc": "segment"}
    y = {t: train[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    seg, fold = train["seg"].to_numpy(), train["fold"].to_numpy()
    inner = np.zeros((train.height, 2))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        p = probe.predict(cfg, emb_tr[tr], {t: v[tr] for t, v in y.items()}, seg[tr], emb_tr[te], seg[te])
        inner[te] = np.column_stack([p["sent"], p["sarc"]])
    p = probe.predict(cfg, emb_tr, y, seg, emb_te, test["seg"].to_numpy())
    outer = np.column_stack([p["sent"], p["sarc"]])
    lg = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))
    return lg(inner), lg(outer)


def dense_features(train: pl.DataFrame, test: pl.DataFrame, emb_tr, emb_te, logit_tr, logit_te,
                   X_tfidf=None, mode: str = "pca") -> tuple[np.ndarray, np.ndarray]:
    """mode pca: handcrafted + segment + clf logits + SVD64(TF-IDF) + PCA64(emb)
            raw: handcrafted + segment + clf logits + ALL embedding dims
            logit: handcrafted + segment + clf logits + nested LR-probe logits (supervised projection)"""
    Htr, _ = handcrafted(train)
    Hte, _ = handcrafted(test)
    seg_tr, seg_te = train["seg"].to_numpy()[:, None], test["seg"].to_numpy()[:, None]
    A, B = [Htr, seg_tr, logit_tr], [Hte, seg_te, logit_te]
    if mode == "pca":
        if X_tfidf is None:
            X_tfidf = vectorise(train["norm"].to_list(), train["norm"].to_list() + test["norm"].to_list())
        svd = TruncatedSVD(64, random_state=SEED).fit(X_tfidf[:train.height])
        S = svd.transform(X_tfidf)
        pca = PCA(64, random_state=SEED).fit(emb_tr)
        A += [S[:train.height], pca.transform(emb_tr)]
        B += [S[train.height:], pca.transform(emb_te)]
    elif mode == "raw":
        A.append(emb_tr)
        B.append(emb_te)
    elif mode == "logit":
        a, b = nested_probe_logits(train, test, emb_tr, emb_te)
        A.append(a)
        B.append(b)
    else:
        raise ValueError(mode)
    return np.hstack(A).astype(np.float32), np.hstack(B).astype(np.float32)


def fit_predict(algo, A, y, fold, B, colsample: float = 0.5) -> np.ndarray:
    """Fixed, regularised hyperparameters; boosting rounds = mean best iteration of inner CV over the
    training folds (early stopping only ever sees training rows)."""
    best = []
    if algo == "lgbm":
        prm = {**LGB, "feature_fraction": colsample}
        for k in np.unique(fold):
            tr, va = fold != k, fold == k
            m = lgb.train(prm, lgb.Dataset(A[tr], y[tr]), 2000, valid_sets=[lgb.Dataset(A[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            best.append(m.best_iteration or 2000)
        return lgb.train(prm, lgb.Dataset(A, y), int(np.mean(best))).predict(B)
    if algo == "xgb":
        import xgboost as xgb
        prm = dict(objective="binary:logistic", eval_metric="logloss", eta=0.03, max_depth=4,
                   min_child_weight=5, subsample=0.8, colsample_bytree=colsample, reg_lambda=5.0,
                   tree_method="hist", nthread=4, seed=SEED)
        for k in np.unique(fold):
            tr, va = fold != k, fold == k
            m = xgb.train(prm, xgb.DMatrix(A[tr], y[tr]), 2000, evals=[(xgb.DMatrix(A[va], y[va]), "va")],
                          early_stopping_rounds=100, verbose_eval=False)
            best.append(m.best_iteration + 1)
        m = xgb.train(prm, xgb.DMatrix(A, y), int(np.mean(best)))
        return m.predict(xgb.DMatrix(B))
    from catboost import CatBoostClassifier
    kw = dict(learning_rate=0.05, depth=6, l2_leaf_reg=5, thread_count=4, random_seed=SEED, verbose=0,
              rsm=colsample, allow_writing_files=False)
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        m = CatBoostClassifier(iterations=2000, od_type="Iter", od_wait=100, **kw)
        m.fit(A[tr], y[tr], eval_set=(A[va], y[va]), use_best_model=True)
        best.append(m.get_best_iteration() + 1)
    return CatBoostClassifier(iterations=int(np.mean(best)), **kw).fit(A, y).predict_proba(B)[:, 1]


def logits(names, test=None) -> np.ndarray:
    parts = [embed.load(n, test)[:, :k] for n, k in CLF_LOGITS.items() if n in names]
    if not parts:
        return np.zeros((len(embed.load("e5_large", test)), 0), np.float32)
    return np.hstack(parts)


def main() -> None:
    import scipy.sparse as sp
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="lgbm", choices=["lgbm", "cat", "xgb"])
    ap.add_argument("--features", default="pca", choices=["pca", "raw", "logit"])
    ap.add_argument("--colsample", type=float, default=0.5)
    ap.add_argument("--emb", default="e5_large")
    ap.add_argument("--clf", default="tw_irony,tw_sent,go_emo")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    df = pool()
    E = np.hstack([embed.load(n) for n in a.emb.split("+")])
    L = logits(a.clf.split(","))
    fold = df["fold"].to_numpy()
    y = {t: df[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    oof = {t: np.zeros(df.height) for t in y}
    t0 = time.time()
    for k in range(5):
        tr, te = fold != k, fold == k
        path = CACHE / f"tfidf_pool_fold{k}.npz"
        Xt = sp.load_npz(path).tocsr() if path.exists() and a.features == "pca" else None
        A, B = dense_features(df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k),
                              E[tr], E[te], L[tr], L[te], Xt, mode=a.features)
        for t in y:
            oof[t][te] = fit_predict(a.algo, A, y[t][tr], fold[tr], B, a.colsample)
        print(f"fold {k} {time.time() - t0:.0f}s", flush=True)
    res = quick(oof["sent"], oof["sarc"])
    print(a.run, fmt(res))
    save_preds(a.run, folds_used=df.select("uid", "fold"), oof=pl.DataFrame({"uid": df["uid"], "p_sent": oof["sent"], "p_sarc": oof["sarc"]}),
               meta={"kind": "gbdt", "script": "experiments/classical/gbdt.py", "config": vars(a),
                     "features": dense_features.__doc__,
                     "params": "see fit_predict (fixed); rounds by inner CV on training folds",
                     "test_protocol": "average of the 5 fold models (predict_test.py)", "quick": res})


if __name__ == "__main__":
    main()
