"""Holdout simulation used by attacks 1b/3/4/5 (generates a cache; analysis in 03_*.py).

Each rep r: draw a hidden-test-like holdout H from the pool with exactly valid.csv's 16 cell
counts (757 rows, no text shared with the rest). On the remaining pool P' (pool folds kept),
train every config 5-fold: OOF on P', each fold model predicts H (-> fold-averaged test probs,
like save_preds' test split), plus a refit on all of P' predicting H.

Output: SCRATCH/holdout_sim.npz-like per-rep npz files.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from _lib import *  # noqa: F403
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

from alta.validation.resampling import cell_ids, target_counts

OUT = Path(os.environ.get("RT_CACHE", "/tmp/rt_stats_cache"))
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {f"{m}_C{C}": (m, "lr", C) for m in ("bge_large", "bge_base", "mpnet", "minilm")
           for C in (0.1, 1.0, 10.0)}
CONFIGS |= {"bge_base_mlp": ("bge_base", "mlp", None), "mpnet_mlp": ("mpnet", "mlp", None)}


def make_model(kind, C, seed):
    if kind == "lr":
        return lambda: lr(C)
    return lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=60,
                                 random_state=seed, early_stopping=False)


def holdout(d, rng, texts):
    cells = cell_ids(d)
    h = np.zeros(len(d), bool)
    for c, k in target_counts(d).items():
        idx = np.flatnonzero(cells == c)
        h[rng.choice(idx, k, replace=False)] = True
    shared = np.isin(texts, texts[h]) & ~h  # rows in P' duplicating an H text are dropped
    return h, ~h & ~shared


def run_config(name, X, d, h, keep, rep):
    m, kind, C = CONFIGS[name]
    out = {}
    for t in TASKS:
        y = d.y[t]
        oof, tests = oof_and_test(X, y, d.fold, keep, h, C,
                                  make_model(kind, C, 1000 * rep + (t == "sarc")))
        refit = fit_predict(X[keep], y[keep], [X[h]], C, make_model(kind, C, 99 + rep))[0]
        out[t] = (oof[keep], tests, refit)
    return name, out


def main(reps):
    d = base_eval()
    texts = load_pool()["text"].to_numpy()
    Xs = {m: design(d, emb(m)) for m in ("bge_large", "bge_base", "mpnet", "minilm")}
    for rep in reps:
        path = OUT / f"rep{rep:02d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng(1000 + rep)
        h, keep = holdout(d, rng, texts)
        res = Parallel(n_jobs=7)(delayed(run_config)(n, Xs[CONFIGS[n][0]], d, h, keep, rep)
                                 for n in CONFIGS)
        arrs = {"h": h, "keep": keep}
        for name, out in res:
            for t, (oof, tests, refit) in out.items():
                arrs[f"{name}|{t}|oof"] = oof
                arrs[f"{name}|{t}|tests"] = tests
                arrs[f"{name}|{t}|refit"] = refit
        np.savez_compressed(path, **arrs)
        print("rep", rep, "done", flush=True)


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    main(range(a, b))
