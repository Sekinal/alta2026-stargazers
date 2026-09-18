"""Aggregate the pseudo-leaderboard draws: bias, coverage, selection, pair calibration, policies.

  PYTHONPATH=. uv run python redteam/calibration/02_analyse.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nest import OUT  # noqa: E402

DEFAULT = ("fixed", "fixed+ukg", "global+ukg", "variety+ukg", "segment+ukg", "prev+ukg",
           "prevtuned+ukg", "variety+ukg+neg", "prev+ukg+neg")  # policies.DEFAULT_POLICIES
MODES = ("refit", "foldavg")

cand = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("cand_*.parquet"))])
pairs = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("pairs_*.parquet"))])
cand = cand.with_columns(pl.col("draw").str.starts_with("random").alias("rnd"))
rnd = cand.filter("rnd")
n_draws = rnd["draw"].n_unique()
cfg = pl.Config(tbl_rows=120, tbl_cols=20, tbl_width_chars=220, float_precision=4,
                tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True)
cfg.__enter__()


def h(t):
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


print(f"draws: {cand['draw'].n_unique()} ({n_draws} random, "
      f"{cand.filter(~pl.col('rnd'))['draw'].n_unique()} real-valid fold seeds); "
      f"candidates/draw: {cand.filter(pl.col('draw') == cand['draw'][0]).height}")

# ------------------------------------------------------------------ 1. bias and coverage
h("1a. Bias of each estimate vs pseudo-test (random draws; all model x policy candidates)")
for mode in MODES:
    t = f"test_{mode}"
    print(rnd.select(
        pl.lit(mode).alias("test preds"),
        (pl.col("cv") - pl.col(t)).mean().alias("cv-test"),
        (pl.col("boot_mean") - pl.col(t)).mean().alias("bootmean-test"),
        (pl.col("dev") - pl.col(t)).mean().alias("dev-test"),
        (pl.col("cv") - pl.col(t)).std().alias("sd(cv-test)"),
        (pl.col("dev") - pl.col(t)).std().alias("sd(dev-test)"),
        pl.col("boot_sd").mean().alias("mean boot_sd"),
        ((pl.col(t) >= pl.col("boot_q05")) & (pl.col(t) <= pl.col("boot_q95"))).mean().alias("cover90"),
        (pl.col(t) < pl.col("boot_q05")).mean().alias("below"),
        (pl.col(t) > pl.col("boot_q95")).mean().alias("above"),
        ((pl.col(t) - pl.col("boot_mean")) / pl.col("boot_sd")).std().alias("sd(z)"),
    ))

h("1b. Bias by model (policy = each model's report pick, refit / foldavg)")
pick = (cand.filter(pl.col("policy").is_in(DEFAULT))
        .sort("cv", descending=True).group_by("draw", "model").first())
print(pick.filter("rnd").group_by("model").agg(
    pl.col("policy").mode().first().alias("modal pick"),
    pl.col("cv").mean(), pl.col("test_refit").mean(), pl.col("test_foldavg").mean(),
    (pl.col("cv") - pl.col("test_refit")).mean().alias("bias_refit"),
    (pl.col("cv") - pl.col("test_foldavg")).mean().alias("bias_foldavg"),
    ((pl.col("test_foldavg") >= pl.col("boot_q05")) & (pl.col("test_foldavg") <= pl.col("boot_q95")))
    .mean().alias("cover90_fa"),
    ((pl.col("test_refit") >= pl.col("boot_q05")) & (pl.col("test_refit") <= pl.col("boot_q95")))
    .mean().alias("cover90_rf"),
).sort("cv", descending=True))

h("1c. Winner's curse: the overall CV-best candidate per draw (what we'd report as 'our score')")
best = cand.sort("cv", descending=True).group_by("draw").first()
best_def = cand.filter(pl.col("policy").is_in(DEFAULT)).sort("cv", descending=True).group_by("draw").first()
for name, b in (("all 11 policies", best), ("report DEFAULT_POLICIES", best_def)):
    b = b.filter("rnd")
    print(name, b.select(
        pl.len().alias("n"), pl.col("cv").mean(), pl.col("test_refit").mean(), pl.col("test_foldavg").mean(),
        (pl.col("cv") - pl.col("test_refit")).mean().alias("bias_rf"),
        (pl.col("cv") - pl.col("test_foldavg")).mean().alias("bias_fa"),
        ((pl.col("cv") - pl.col("test_foldavg")).std() / pl.len().sqrt()).alias("se_fa"),
        ((pl.col("test_foldavg") >= pl.col("boot_q05")) & (pl.col("test_foldavg") <= pl.col("boot_q95")))
        .mean().alias("cover90_fa"),
    ))
    print(b.group_by("model", "policy").len().sort("len", descending=True).head(8))

h("1d. Does the bootstrap SD match the real spread of the pseudo-test score? (per candidate, over random draws)")
spread = rnd.group_by("model", "policy").agg(
    pl.col("test_foldavg").std().alias("sd_test_fa"), pl.col("test_refit").std().alias("sd_test_rf"),
    (pl.col("test_foldavg") - pl.col("cv")).std().alias("sd(test_fa-cv)"),
    pl.col("boot_sd").mean().alias("boot_sd"), pl.col("cv").std().alias("sd_cv"))
print(spread.select(pl.exclude("model", "policy").mean()))
print(spread.select(pl.exclude("model", "policy").mean()).select(
    (pl.col("sd(test_fa-cv)") / pl.col("boot_sd")).alias("inflation needed for honest CI")))

# ------------------------------------------------------------------ truth per candidate
truth = rnd.group_by("model", "policy").agg(
    pl.col("test_refit").mean().alias("true_rf"), pl.col("test_foldavg").mean().alias("true_fa"),
    pl.col("cv").mean().alias("mean_cv"))
h("Population truth: mean pseudo-test score per model (foldavg), by policy")
print(truth.pivot(on="policy", index="model", values="true_fa").sort("model"))
print("\nrefit:")
print(truth.pivot(on="policy", index="model", values="true_rf").sort("model"))

# ------------------------------------------------------------------ 2. selection
h("2. Selection accuracy (random draws)")


def selection(frame: pl.DataFrame, label: str, by: str = "cv"):
    rows = []
    for d, g in frame.group_by("draw"):
        g = g.join(truth, on=["model", "policy"])
        for mode, tt in (("refit", "true_rf"), ("foldavg", "true_fa")):
            t = f"test_{mode}"
            chosen = g.sort(by, descending=True).row(0, named=True)
            rows.append({"mode": mode,
                         "hit_draw": chosen[t] >= g[t].max() - 1e-12,
                         "hit_truth": chosen[tt] >= g[tt].max() - 1e-12,
                         "regret_draw": g[t].max() - chosen[t],
                         "regret_truth": g[tt].max() - chosen[tt],
                         "rho_draw": spearmanr(g[by], g[t]).statistic,
                         "rho_truth": spearmanr(g[by], g[tt]).statistic})
    print(label, f"(select by {by}, {frame.group_by('draw', 'model', 'policy').len().height // max(1, frame['draw'].n_unique())} candidates)")
    print(pl.DataFrame(rows).group_by("mode").agg(pl.all().mean()).sort("mode"))


selection(rnd, "all model x policy")
selection(rnd, "all model x policy", by="dev")
selection(rnd.filter(pl.col("policy") == "fixed+ukg"), "models at fixed+ukg (compare default)")
selection(rnd.filter(pl.col("policy") == "fixed+ukg"), "models at fixed+ukg", by="dev")
selection(pick.filter("rnd"), "models at report-picked policy")
selection(pick.filter("rnd"), "models at report-picked policy", by="dev")
for m in rnd["model"].unique().sort():
    selection(rnd.filter(pl.col("model") == m, pl.col("policy").is_in(DEFAULT)), f"policies within {m}")

h("2b. Report's policy pick vs fixed defaults (test score of the pick minus alternative, mean over draws x models)")
alt = rnd.filter(pl.col("policy").is_in(["fixed+ukg", "variety+ukg", "prev+ukg", "prevtuned+ukg", "segment+ukg"]))
j = pick.filter("rnd").select("draw", "model", "policy", "test_refit", "test_foldavg").join(
    alt.select("draw", "model", pl.col("policy").alias("alt"), pl.col("test_refit").alias("a_rf"),
               pl.col("test_foldavg").alias("a_fa")), on=["draw", "model"])
print(j.group_by("alt").agg((pl.col("test_foldavg") - pl.col("a_fa")).mean().alias("pick-alt (fa)"),
                            (pl.col("test_refit") - pl.col("a_rf")).mean().alias("pick-alt (rf)"),
                            ((pl.col("test_foldavg") - pl.col("a_fa")) > 0).mean().alias("pick wins (fa)"),
                            ).sort("alt"))

# ------------------------------------------------------------------ 3. policies on pseudo-test
h("3. Policies judged on pseudo-test (mean over random draws x models; delta vs fixed+ukg)")
ref = rnd.filter(pl.col("policy") == "fixed+ukg").select("draw", "model", pl.col("test_foldavg").alias("r_fa"),
                                                         pl.col("test_refit").alias("r_rf"), pl.col("cv").alias("r_cv"))
pol = rnd.join(ref, on=["draw", "model"]).group_by("policy").agg(
    pl.col("cv").mean().alias("cv"), pl.col("test_foldavg").mean().alias("test_fa"),
    pl.col("test_refit").mean().alias("test_rf"),
    (pl.col("cv") - pl.col("r_cv")).mean().alias("d_cv"),
    (pl.col("test_foldavg") - pl.col("r_fa")).mean().alias("d_test_fa"),
    ((pl.col("test_foldavg") - pl.col("r_fa")).std() / pl.len().sqrt()).alias("se_fa"),
    (pl.col("test_refit") - pl.col("r_rf")).mean().alias("d_test_rf"),
    (pl.col("test_foldavg") - pl.col("test_refit")).mean().alias("fa-rf"),
    ((pl.col("test_foldavg") - pl.col("test_refit")) > 0).mean().alias("P(fa>rf)"),
).sort("test_fa", descending=True)
print(pol)
print("\nper model, delta test_fa vs fixed+ukg:")
print(rnd.join(ref, on=["draw", "model"]).group_by("model", "policy").agg(
    (pl.col("test_foldavg") - pl.col("r_fa")).mean().alias("d")).pivot(on="policy", index="model", values="d"))
print("\nrank of policy by test_fa within (draw, model): mean rank (1 = best)")
print(rnd.with_columns(pl.col("test_foldavg").rank(descending=True).over("draw", "model").alias("rk"))
      .group_by("policy").agg(pl.col("rk").mean()).sort("rk"))

# ------------------------------------------------------------------ 4. refit vs foldavg
h("4. Refit vs fold-average (model at report-picked policy), random draws")
print(pick.filter("rnd").group_by("model").agg(
    (pl.col("test_foldavg") - pl.col("test_refit")).mean().alias("fa-rf"),
    ((pl.col("test_foldavg") - pl.col("test_refit")) > 0).mean().alias("P(fa>rf)"),
    (pl.col("cv") - pl.col("test_refit")).abs().mean().alias("|cv-rf|"),
    (pl.col("cv") - pl.col("test_foldavg")).abs().mean().alias("|cv-fa|")).sort("model"))

# ------------------------------------------------------------------ 5. pair calibration
h("5. Paired comparisons: does p_a_better predict the sign of the pseudo-test difference?")
pr = pairs.filter(pl.col("draw").str.starts_with("random")).with_columns(
    pl.when(pl.col("p_a_better") >= 0.5).then(pl.col("p_a_better")).otherwise(1 - pl.col("p_a_better") - pl.col("p_tie")).alias("p"),
    pl.when(pl.col("p_a_better") >= 0.5).then(1).otherwise(-1).alias("sgn"),
    pl.when(pl.col("model_a") == pl.col("model_b")).then(pl.lit("policy pair (same model)"))
    .when(pl.col("policy_a") == pl.col("policy_b")).then(pl.lit("model pair (same policy)"))
    .otherwise(pl.lit("mixed")).alias("kind"),
)
tr = truth.select("model", "policy", "true_fa", "true_rf")
pr = pr.join(tr.rename({"model": "model_a", "policy": "policy_a", "true_fa": "ta_fa", "true_rf": "ta_rf"}),
             on=["model_a", "policy_a"]).join(
    tr.rename({"model": "model_b", "policy": "policy_b", "true_fa": "tb_fa", "true_rf": "tb_rf"}),
    on=["model_b", "policy_b"])
for mode, tt in (("foldavg", "fa"), ("refit", "rf")):
    d = pl.col(f"test_{mode}_diff") * pl.col("sgn")
    dt = (pl.col(f"ta_{tt}") - pl.col(f"tb_{tt}")) * pl.col("sgn")
    tab = pr.filter(pl.col(f"test_{mode}_diff") != 0).with_columns(
        pl.col("p").cut([0.6, 0.7, 0.8, 0.9, 0.95, 0.99], left_closed=True).alias("bin")
    ).group_by("kind", "bin").agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p"),
                                  (d > 0).mean().alias("sign_right_draw"),
                                  (dt > 0).mean().alias("sign_right_truth")).sort("kind", "bin")
    print(f"test preds: {mode}")
    print(tab)
    print(pr.filter(pl.col(f"test_{mode}_diff") != 0).with_columns(
        pl.col("p").cut([0.8, 0.9, 0.95], left_closed=True).alias("thr")).group_by("thr").agg(
        pl.len(), pl.col("p").mean(), (d > 0).mean().alias("sign_right_draw"),
        (dt > 0).mean().alias("sign_right_truth")).sort("thr"))

# ------------------------------------------------------------------ 6. dev subset vs pool
h("6. valid-sized dev subset vs pool OOF as the yardstick")
print("pick disagreement rate (dev-best != cv-best):")
for label, fr in (("model x policy", rnd), ("models @fixed+ukg", rnd.filter(pl.col("policy") == "fixed+ukg")),
                  ("models @report pick", pick.filter("rnd"))):
    a = fr.sort("cv", descending=True).group_by("draw").first().select("draw", pl.col("model").alias("m1"), pl.col("policy").alias("p1"))
    b = fr.sort("dev", descending=True).group_by("draw").first().select("draw", pl.col("model").alias("m2"), pl.col("policy").alias("p2"))
    x = a.join(b, on="draw")
    print(f"  {label}: {((x['m1'] != x['m2']) | (x['p1'] != x['p2'])).mean():.2f}")
print(rnd.select((pl.col("dev") - pl.col("test_foldavg")).std().alias("sd(dev-test)"),
                 (pl.col("cv") - pl.col("test_foldavg")).std().alias("sd(cv-test)"),
                 pl.corr("dev", "test_foldavg").alias("r(dev,test)"), pl.corr("cv", "test_foldavg").alias("r(cv,test)")))

# ------------------------------------------------------------------ 7. real valid.csv, fold seeds
h("7. Real valid.csv as the test set (R = train.csv), 5 fold seeds")
val = cand.filter(~pl.col("rnd"))
vp = val.filter(pl.col("policy").is_in(DEFAULT)).sort("cv", descending=True).group_by("draw", "model").first()
print(vp.group_by("model").agg(
    pl.col("policy").unique().sort().str.join(",").alias("picked policies over seeds"),
    pl.col("cv").mean(), pl.col("cv").std().alias("sd_cv(seed)"),
    pl.col("test_foldavg").mean(), pl.col("test_foldavg").std().alias("sd_fa(seed)"),
    pl.col("test_refit").mean(), pl.col("test_refit").std().alias("sd_rf(seed)"),
    ((pl.col("test_foldavg") >= pl.col("boot_q05")) & (pl.col("test_foldavg") <= pl.col("boot_q95"))).mean().alias("cover")
).sort("cv", descending=True))
print("CV-best candidate per fold seed:")
print(val.sort("cv", descending=True).group_by("draw").first().select(
    "draw", "model", "policy", "cv", "boot_q05", "boot_q95", "test_refit", "test_foldavg").sort("draw"))
print("sd over fold seeds of each candidate's cv, averaged:",
      val.group_by("model", "policy").agg(pl.col("cv").std()).select(pl.col("cv").mean()).item())
