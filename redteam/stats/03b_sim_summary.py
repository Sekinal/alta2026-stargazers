"""Summaries of 03_sim_analysis.py outputs (see that file for the design)."""

import os
from pathlib import Path

from _lib import *  # noqa: F403

CACHE = Path(os.environ.get("RT_CACHE", "/tmp/rt_stats_cache"))
CFG = dict(tbl_rows=80, tbl_cols=30, tbl_width_chars=260, float_precision=4, fmt_str_lengths=40)


def main():
    df = pl.read_parquet(CACHE / "sim_rows.parquet")
    pr = pl.read_parquet(CACHE / "sim_pairs.parquet")
    n_rep = df["path"].n_unique()
    print(f"reps={n_rep} configs={df['config'].n_unique()}")
    df = df.with_columns(((pl.col("H_avg") - pl.col("boot_mean")) / pl.col("boot_sd")).alias("z"),
                         pl.col("H_avg").is_between(pl.col("q05"), pl.col("q95")).alias("cov90"),
                         (pl.col("reported") - pl.col("H_avg")).alias("rep_minus_H"))
    with pl.Config(**CFG):
        print("\n== 3. bootstrap calibration (truth = fold-averaged test on H), by policy")
        print(df.group_by("policy").agg(
            pl.len(), pl.col("rep_minus_H").mean().alias("bias(reported-H)"),
            (pl.col("boot_mean") - pl.col("H_avg")).mean().alias("bias(boot-H)"),
            pl.col("boot_sd").mean(), pl.col("H_avg").std().alias("sd(H) all"),
            pl.col("z").mean().alias("mean z"), pl.col("z").std().alias("sd z"),
            pl.col("cov90").mean().alias("coverage90")).sort("policy"))
        # SD of H across reps for a fixed config vs its mean boot_sd: the honest check
        per_cfg = df.group_by("config", "policy").agg(
            pl.col("H_avg").std().alias("sd_H_across_reps"), pl.col("boot_sd").mean().alias("boot_sd"),
            pl.col("reported").std().alias("sd_reported_across_reps"),
            pl.col("fold_sd").mean().alias("mean_fold_sd"))
        print(per_cfg.group_by("policy").agg(pl.col("sd_H_across_reps").mean(), pl.col("boot_sd").mean(),
                                             pl.col("sd_reported_across_reps").mean(),
                                             pl.col("mean_fold_sd").mean()).sort("policy"))
        print("by config (fixed+ukg / variety+ukg):")
        print(df.filter(pl.col("policy").is_in(["fixed+ukg", "variety+ukg"])).group_by("config", "policy").agg(
            pl.col("reported").mean(), pl.col("H_avg").mean(), pl.col("H_single").mean(), pl.col("H_refit").mean(),
            pl.col("z").mean().alias("mean z"), pl.col("z").std().alias("sd z"), pl.col("cov90").mean()).sort("config", "policy"))

        print("\n== 5. OOF-fitted policy on fold-averaged vs single-model vs refit test predictions")
        print(df.group_by("policy").agg(
            (pl.col("H_avg") - pl.col("H_single")).mean().alias("avg-single"),
            (pl.col("H_avg") - pl.col("H_refit")).mean().alias("avg-refit"),
            (pl.col("reported") - pl.col("H_avg")).mean().alias("reported-avg")).sort("policy"))
        v = df.filter(pl.col("policy") == "variety+ukg")
        print(v.group_by(pl.col("config").str.contains("mlp").alias("mlp")).agg(
            *[pl.col(c).mean() for c in v.columns if "rate" in c or "abslogit" in c]))
        print(v.group_by("config").agg(
            *[pl.col(c).mean() for c in ("sarc_rate_oof", "sarc_rate_avg", "sarc_rate_refit",
                                           "sarc_abslogit_oof", "sarc_abslogit_avg", "sarc_abslogit_single")],
            (pl.col("H_avg") - pl.col("H_refit")).mean().alias("avg-refit")).sort("config"))

        print("\n== 1a. selecting the best policy (report.py) : reported max vs its H score")
        best = df.sort("reported", descending=True).group_by("path", "config").first()
        fx = df.filter(pl.col("policy") == "fixed+ukg").select("path", "config", pl.col("H_avg").alias("H_fixed"),
                                                              pl.col("reported").alias("rep_fixed"))
        best = best.join(fx, on=["path", "config"])
        print(best.select((pl.col("reported") - pl.col("H_avg")).mean().alias("best: reported-H"),
                          (pl.col("rep_fixed") - pl.col("H_fixed")).mean().alias("fixed+ukg: reported-H"),
                          (pl.col("H_avg") - pl.col("H_fixed")).mean().alias("H gain of chosen policy over fixed+ukg"),
                          (pl.col("reported") - pl.col("rep_fixed")).mean().alias("reported gain over fixed+ukg")))
        mean_pol = df.group_by("path", "config").agg(pl.col("H_avg").mean().alias("H_mean_policy"),
                                                     pl.col("H_avg").max().alias("H_oracle_policy"))
        print(best.join(mean_pol, on=["path", "config"]).select(
            (pl.col("H_avg") - pl.col("H_mean_policy")).mean().alias("chosen - avg policy on H"),
            (pl.col("H_oracle_policy") - pl.col("H_avg")).mean().alias("oracle - chosen on H")))

        print("\n== 1b. board: best config per rep (by reported best-policy score)")
        board = best.sort("reported", descending=True).group_by("path").agg(
            pl.col("config").first().alias("winner"), pl.col("reported").first().alias("win_reported"),
            pl.col("H_avg").first().alias("win_H"), pl.col("H_avg").max().alias("best_H"),
            (pl.col("reported") - pl.col("H_avg")).mean().alias("avg_gap"),
            pl.corr("reported", "H_avg", method="spearman").alias("spearman"))
        print(board.select((pl.col("win_reported") - pl.col("win_H")).mean().alias("winner reported-H"),
                           pl.col("avg_gap").mean().alias("avg config reported-H"),
                           (pl.col("best_H") - pl.col("win_H")).mean().alias("regret vs H-best"),
                           pl.col("spearman").mean(), pl.col("winner").value_counts().alias("winners")))
        # top-2 on the board: how often does H agree with the board order?
        top = best.sort("reported", descending=True).group_by("path").agg(pl.col("reported").head(2), pl.col("H_avg").head(2))
        agree = [(h[0] > h[1]) for h in top["H_avg"].to_list()]
        gaps = [r[0] - r[1] for r in top["reported"].to_list()]
        print(f"board #1 beats #2 on H in {np.mean(agree):.2f} of reps; mean reported gap {np.mean(gaps):.4f}")

        print("\n== 3b. paired p_a_better calibration (truth: A beats B on H, fold-averaged)")
        pr = pr.with_columns(pl.col("p_a_better").cut([0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]).alias("bin"))
        print(pr.group_by("bin").agg(pl.len(), pl.col("p_a_better").mean(), pl.col("a_wins_H").mean()).sort("p_a_better"))
        close = pr.filter(pl.col("mean_diff").abs() < 0.01)
        print("pairs with |mean diff|<0.01:")
        print(close.group_by("bin").agg(pl.len(), pl.col("p_a_better").mean(), pl.col("a_wins_H").mean()).sort("p_a_better"))
        # 'decisive' rule used by compare(): p>=0.8 or <=0.2
        dec = pr.filter((pl.col("p_a_better") >= 0.8) | (pl.col("p_a_better") <= 0.2))
        wrong = dec.filter(((pl.col("p_a_better") >= 0.8) & (pl.col("a_wins_H") < 0.5)) |
                           ((pl.col("p_a_better") <= 0.2) & (pl.col("a_wins_H") > 0.5)))
        print(f"decisive calls: {dec.height}/{pr.height}, wrong on H: {wrong.height / max(dec.height, 1):.3f}")
        # SD of H_diff vs predicted SD of diff -> is paired sd calibrated
        print(pr.with_columns((pl.col("H_diff") - pl.col("mean_diff")).alias("err")).group_by("policy").agg(
            pl.col("err").std().alias("sd(H_diff - boot_mean_diff)"), pl.col("sd_diff").mean().alias("mean paired boot sd"),
            ((pl.col("H_diff") - pl.col("mean_diff")) / pl.col("sd_diff")).std().alias("sd of z_pair")))


if __name__ == "__main__":
    main()
