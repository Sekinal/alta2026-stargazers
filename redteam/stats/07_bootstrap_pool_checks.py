"""Attack 3/6 on the pool only (no retraining):

* Jensen gap: headline pool score (min of pooled components) vs E[score] over test-like replicates.
* fold_sd vs what pure sampling noise predicts at fold size (bootstrap with n_test=fold size).
* transductive policies: bootstrap with labels frozen (what report.py does) vs re-applying the
  fitted prev policy on each replicate (what happens on the real test).
* ties in probabilities (prev policy breaks ties by row order, thresholds by >=).
"""

from _lib import *  # noqa: F403
from alta.validation.data import list_runs, load_oof
from alta.validation.metrics import per_fold
from alta.validation.resampling import replicate_indices, replicate_scores

B = 2000


def main():
    rows = []
    for run in [r for r in list_runs() if not r.startswith("rt_")]:
        d = load_oof(run)
        for spec in ("fixed+ukg", "variety+ukg", "prev+ukg", "prevtuned+ukg"):
            pred, _ = policies.crossfit(spec, d)
            pool_score = components(d, pred)["score"]
            idx = replicate_indices(d, B=B, seed=0)
            reps = replicate_scores(d, pred, idx)["score"]
            idx_f = replicate_indices(d, B=B, seed=1, n_test=len(d) // 5)
            reps_f = replicate_scores(d, pred, idx_f)["score"]
            folds = [f["score"] for f in per_fold(d, pred)]
            row = {"run": run, "policy": spec, "pool": pool_score, "boot_mean": reps.mean(),
                   "jensen_gap": pool_score - reps.mean(), "boot_sd": reps.std(ddof=1),
                   "fold_sd": np.std(folds, ddof=1),
                   "boot_sd@fold_n*fpc": reps_f.std(ddof=1) * np.sqrt(0.8)}
            if spec.startswith("prev"):
                # re-apply the transductive policy (fitted on the full pool) on every replicate
                params = policies.fit(spec, d)
                sc = []
                for b in range(400):
                    sel = np.concatenate([idx[0][b], idx[1][b]])
                    sub = d.subset(sel)
                    sc.append(components(sub, policies.apply(params, sub))["score"])
                sc = np.array(sc)
                row |= {"reapply_mean": sc.mean(), "reapply_sd": sc.std(ddof=1)}
            rows.append(row)
        ties = {t: 1 - len(np.unique(d.p[t])) / len(d) for t in TASKS}
        print(run, "tie fraction", {k: round(v, 4) for k, v in ties.items()}, flush=True)
    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250, float_precision=4):
        print(df)
        print(df.select(pl.col("jensen_gap").mean(), (pl.col("fold_sd") / pl.col("boot_sd@fold_n*fpc")).mean().alias("fold_sd/boot_pred"),
                        (pl.col("reapply_sd") / pl.col("boot_sd")).mean().alias("reapply_sd/boot_sd"),
                        (pl.col("reapply_mean") - pl.col("boot_mean")).mean().alias("reapply-boot mean")))


if __name__ == "__main__":
    main()
