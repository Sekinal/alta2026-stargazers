"""Follow-ups on the draws: which interval would have had 90% coverage, real-valid coverage,
and fold-seed sensitivity of paired verdicts.
  PYTHONPATH=. uv run python redteam/calibration/05_interval_fixes.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nest import OUT  # noqa: E402

cand = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("cand_*.parquet"))])
pairs = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("pairs_*.parquet"))])
rnd = cand.filter(pl.col("draw").str.starts_with("random"))
val = cand.filter(pl.col("draw").str.starts_with("valid"))
N_T, N_R = 757, 5531
SEED_SD = 0.005  # sd of a candidate's cv over fold seeds (02_analyse section 7)

print("coverage of central 90% intervals, random draws (all 99 candidates x 60 draws)")
rows = []
for centre in ("boot_mean", "cv"):
    for k in (1.0, 1.1, 1.2, 1.3):
        for mode in ("foldavg", "refit"):
            z = (rnd[f"test_{mode}"] - rnd[centre]) / (k * rnd["boot_sd"])
            rows.append({"centre": centre, "sd": f"{k} x boot_sd", "mode": mode,
                         "cover90": float((z.abs() <= 1.645).mean()), "below": float((z < -1.645).mean()),
                         "above": float((z > 1.645).mean())})
    sd = np.sqrt(rnd["boot_sd"] ** 2 * (1 + N_T / N_R) + SEED_SD ** 2)
    for mode in ("foldavg", "refit"):
        z = (rnd[f"test_{mode}"] - rnd[centre]) / sd
        rows.append({"centre": centre, "sd": "sqrt(boot^2(1+nT/nPool)+seed^2)", "mode": mode,
                     "cover90": float((z.abs() <= 1.645).mean()), "below": float((z < -1.645).mean()),
                     "above": float((z > 1.645).mean())})
with pl.Config(tbl_rows=40, float_precision=3, tbl_width_chars=160):
    print(pl.DataFrame(rows))
    print("\nreal valid.csv as test (5 fold seeds x 99 candidates): coverage of the stack's boot interval")
    print(val.select(((pl.col("test_foldavg") >= pl.col("boot_q05")) & (pl.col("test_foldavg") <= pl.col("boot_q95"))).mean().alias("cover_fa"),
                     (pl.col("test_foldavg") > pl.col("boot_q95")).mean().alias("above"),
                     (pl.col("test_foldavg") - pl.col("cv")).mean().alias("mean test-cv"),
                     (pl.col("test_foldavg") - pl.col("boot_mean")).mean().alias("mean test-bootmean")))
    print(val.group_by("model").agg((pl.col("test_foldavg") - pl.col("cv")).mean().alias("valid - cv")).sort("valid - cv"))

    print("\nfold-seed sensitivity of paired verdicts on R = train.csv (model pairs at fixed+ukg and at segment+ukg)")
    vp = pairs.filter(pl.col("draw").str.starts_with("valid"), pl.col("policy_a") == pl.col("policy_b"),
                      pl.col("policy_a").is_in(["fixed+ukg", "segment+ukg"]))
    g = vp.group_by("model_a", "model_b", "policy_a").agg(
        pl.col("cv_diff").std().alias("sd_seed_cvdiff"), pl.col("cv_diff").mean().alias("mean_cvdiff"),
        (pl.col("p_a_better") >= 0.9).sum().alias("n_seeds_A_sig"), (pl.col("p_a_better") <= 0.1).sum().alias("n_seeds_B_sig"))
    print(g.select(pl.col("sd_seed_cvdiff").mean().alias("mean sd over fold seeds of cv diff"),
                   ((pl.col("n_seeds_A_sig") > 0) & (pl.col("n_seeds_A_sig") < 5)).mean().alias("A 'sig' on some seeds only"),
                   ((pl.col("n_seeds_B_sig") > 0) & (pl.col("n_seeds_B_sig") < 5)).mean().alias("B 'sig' on some seeds only"),
                   ((pl.col("n_seeds_A_sig") > 0) & (pl.col("n_seeds_B_sig") > 0)).sum().alias("sig both ways")))
