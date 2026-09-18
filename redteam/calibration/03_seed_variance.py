"""Training-seed noise: the same MLP probe (bge_large, pooled + segment one-hot) trained with
different random seeds on R = train.csv, T = valid.csv. Measures seed SD of CV and test scores,
and how often the stack's paired comparison 'finds' a difference between two seeds of one config.

  PYTHONPATH=. uv run python redteam/calibration/03_seed_variance.py [N_SEEDS] [N_JOBS]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nest  # noqa: E402  (sets BLAS threads)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from alta.validation.data import TASKS  # noqa: E402


def mlp_model(R, T, seed: int):
    X = np.load(nest.EMB / "emb_bge_large.npy")
    XR = np.hstack([X[R["i"].to_numpy()], nest._onehot(R)])
    XT = np.hstack([X[T["i"].to_numpy()], nest._onehot(T)])
    fold = R["fold"].to_numpy()
    out = {k: {t: np.zeros(n) for t in TASKS} for k, n in
           (("oof", R.height), ("refit", T.height), ("foldavg", T.height))}

    def fit(Xa, y):
        return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=300,
                             random_state=seed).fit(Xa, y)

    for t in TASKS:
        y = R[nest.YCOL[t]].to_numpy()
        for k in range(5):
            clf = fit(XR[fold != k], y[fold != k])
            out["oof"][t][fold == k] = clf.predict_proba(XR[fold == k])[:, 1]
            out["foldavg"][t] += clf.predict_proba(XT)[:, 1] / 5
        out["refit"][t] = fit(XR, y).predict_proba(XT)[:, 1]
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pool = nest.base_pool()
    d = nest.make_draw(pool, "valid", 0)
    R, T = d["R"], d["T"]
    res = Parallel(n_jobs=n_jobs)(delayed(mlp_model)(R, T, s) for s in range(n_seeds))
    models = {f"mlp_s{s}": r for s, r in enumerate(res)}
    cand, pairs = nest.evaluate("seed", R, T, models)
    cand.write_parquet(nest.OUT / "seed_cand.parquet")
    pairs.write_parquet(nest.OUT / "seed_pairs.parquet")

    cfg = pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=4)
    cfg.__enter__()
    print(cand.group_by("policy").agg(
        pl.col("cv").mean(), pl.col("cv").std().alias("sd_cv_seed"),
        pl.col("dev").std().alias("sd_dev_seed"),
        pl.col("test_foldavg").mean(), pl.col("test_foldavg").std().alias("sd_fa_seed"),
        pl.col("test_refit").mean(), pl.col("test_refit").std().alias("sd_rf_seed"),
        pl.col("boot_sd").mean()).sort("cv", descending=True))
    same = pairs.filter(pl.col("policy_a") == pl.col("policy_b")).with_columns(
        pl.max_horizontal("p_a_better", 1 - pl.col("p_a_better") - pl.col("p_tie")).alias("p"))
    print("seed-vs-seed pairs (same config, same policy):", same.height)
    print(same.select((pl.col("p") >= 0.8).mean().alias("p>=0.8"), (pl.col("p") >= 0.9).mean().alias("p>=0.9"),
                      (pl.col("p") >= 0.95).mean().alias("p>=0.95"), pl.col("cv_diff").abs().mean().alias("|cv diff|"),
                      pl.col("cv_diff").abs().max().alias("max |cv diff|")))
    best = cand.sort("cv", descending=True).group_by("model").first().sort("model")
    print(best.select("model", "policy", "cv", "test_refit", "test_foldavg"))
