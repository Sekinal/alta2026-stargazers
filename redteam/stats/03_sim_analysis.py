"""Analysis of the holdout simulation (02_holdout_sim.py cache).

Per rep and config, P' = pool minus holdout H (757 valid-like rows):
  * crossfit every default policy on P' OOF -> 'reported' scores (what report.py would print)
  * test-like bootstrap on P' (n_test=757)  -> predicted mean / sd / 90% interval
  * truth: policy fitted on all P' OOF applied to H predictions from (a) the fold-averaged
    5 fold models (what save_preds' test split holds), (b) one fold model, (c) a refit on all P'
Outputs calibration of the bootstrap, winner's curse of policy and run selection, calibration of
paired p_a_better, and the OOF -> fold-averaged-test shift.
"""

import itertools
import os
import sys
from pathlib import Path

from _lib import *  # noqa: F403
from alta.validation.metrics import per_fold
from alta.validation.resampling import paired, replicate_indices, replicate_scores

CACHE = Path(os.environ.get("RT_CACHE", "/tmp/rt_stats_cache"))
SPECS = policies.DEFAULT_POLICIES
PAIR_SPECS = ("fixed+ukg", "variety+ukg")
B = 1000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def analyse_rep(path, base):
    z = np.load(path)
    h, keep = z["h"], z["keep"]
    configs = sorted({k.split("|")[0] for k in z.files if "|" in k})
    dP, dH = base.subset(keep), base.subset(h)
    idx = replicate_indices(dP, B=B, seed=0, n_test=757)
    rows, preds = [], {}
    for c in configs:
        oof = {t: z[f"{c}|{t}|oof"] for t in TASKS}
        tests = {t: z[f"{c}|{t}|tests"] for t in TASKS}
        refit = {t: z[f"{c}|{t}|refit"] for t in TASKS}
        dp = dP.with_p(oof)
        variants = {"avg": dH.with_p({t: tests[t].mean(0) for t in TASKS}),
                    "single": dH.with_p({t: tests[t][0] for t in TASKS}),
                    "refit": dH.with_p(refit)}
        for spec in SPECS:
            pred, _ = policies.crossfit(spec, dp)
            preds[(c, spec)] = pred
            rep_sc = replicate_scores(dp, pred, idx)["score"]
            params = policies.fit(spec, dp)
            row = {"path": path.name, "config": c, "policy": spec,
                   "reported": components(dp, pred)["score"],
                   "fold_sd": float(np.std([f["score"] for f in per_fold(dp, pred)], ddof=1)),
                   "boot_mean": rep_sc.mean(), "boot_sd": rep_sc.std(ddof=1),
                   "q05": np.quantile(rep_sc, 0.05), "q95": np.quantile(rep_sc, 0.95)}
            for v, dv in variants.items():
                row[f"H_{v}"] = components(dv, policies.apply(params, dv))["score"]
            if spec == "variety+ukg":
                for t in TASKS:
                    thr = np.where(dp.variety == 0, params[t][0], params[t][1])
                    thr_h = np.where(dH.variety == 0, params[t][0], params[t][1])
                    row[f"{t}_rate_oof"] = float((dp.p[t] >= thr).mean())
                    for v, dv in variants.items():
                        row[f"{t}_rate_{v}"] = float((dv.p[t] >= thr_h).mean())
                    row[f"{t}_abslogit_oof"] = float(np.abs(logit(dp.p[t])).mean())
                    row[f"{t}_abslogit_avg"] = float(np.abs(logit(variants["avg"].p[t])).mean())
                    row[f"{t}_abslogit_single"] = float(np.abs(logit(variants["single"].p[t])).mean())
            rows.append(row)
    pairs = []
    for spec in PAIR_SPECS:
        hs = {r["config"]: r["H_avg"] for r in rows if r["policy"] == spec}
        for a, b in itertools.combinations(configs, 2):
            res = paired(dP, preds[(a, spec)], preds[(b, spec)], B=500, seed=1, n_test=757)["score"]
            outcome = 0.5 if hs[a] == hs[b] else float(hs[a] > hs[b])
            pairs.append({"path": path.name, "policy": spec, "a": a, "b": b,
                          "p_a_better": res["p_a_better"], "mean_diff": res["mean"], "sd_diff": res["sd"],
                          "H_diff": hs[a] - hs[b], "a_wins_H": outcome})
    return rows, pairs


def main():
    base = base_eval()
    files = sorted(CACHE.glob("rep*.npz"))
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    rows, pairs = [], []
    for f in files:
        r, p = analyse_rep(f, base)
        rows += r
        pairs += p
        print(f.name, flush=True)
    df = pl.DataFrame(rows)
    pr = pl.DataFrame(pairs)
    df.write_parquet(CACHE / "sim_rows.parquet")
    pr.write_parquet(CACHE / "sim_pairs.parquet")
    print("reps:", len(files))


if __name__ == "__main__":
    main()
