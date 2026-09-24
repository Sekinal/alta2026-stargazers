"""Ensemble-size curves from the 10 deberta-v3-large seeds (enc_deb_l_s13..17 and the retrains enc_deb_l_r13..17,
same recipe): for n = 1..10 seeds, the probability-mean of up to 30 random n-subsets, alone and inside the lead
(logit mean with cls_e5me5_lr and llm_q35_9b_probe). Official metric, cross-fitted segment+ukg; also mean pairwise
disagreement of seeds. Output: results/paper/ensemble_curve.csv

  uv run python experiments/classical/ensemble_curve.py
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT, pool, quick
from stack import load_members

SEEDS = [f"enc_deb_l_s{s}" for s in range(13, 18)] + [f"enc_deb_l_r{s}" for s in range(13, 18)]
OTHERS = ["cls_e5me5_lr", "llm_q35_9b_probe"]
OUT = ROOT / "results" / "paper" / "ensemble_curve.csv"


def sig(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    df = pool()
    uids, seg = df["uid"].to_numpy(), df["seg"].to_numpy()
    M = load_members(SEEDS + OTHERS, uids)
    Pr = {t: np.stack([sig(M[r][t]) for r in SEEDS]) for t in ("sent", "sarc")}
    y_sarc = df["sarcasm"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for n in range(1, len(SEEDS) + 1):
        combos = list(itertools.combinations(range(len(SEEDS)), n))
        if len(combos) > 30:
            combos = [combos[i] for i in rng.choice(len(combos), 30, replace=False)]
        for c in combos:
            p = {t: Pr[t][list(c)].mean(0) for t in Pr}
            alone = quick(p["sent"], p["sarc"], specs=("segment+ukg",))["segment+ukg"]
            lg = {t: np.mean([np.log(np.clip(p[t], 1e-6, 1 - 1e-6) / (1 - np.clip(p[t], 1e-6, 1 - 1e-6)))]
                             + [M[r][t] for r in OTHERS], 0) for t in p}
            lead = quick(sig(lg["sent"]), sig(lg["sarc"]), specs=("segment+ukg",))["segment+ukg"]
            m = seg == 1
            rows.append({"n": n, "combo": "+".join(SEEDS[i].split("_")[-1] for i in c),
                         "deb_score": alone["score"], "deb_sarc_AU": alone["sarc-AU"], "deb_sent_AU": alone["sent-AU"],
                         "lead_score": lead["score"], "lead_sarc_AU": lead["sarc-AU"], "lead_sent_AU": lead["sent-AU"],
                         "deb_auc_sarc_AUr": round(float(roc_auc_score(y_sarc[m], p["sarc"][m])), 4)})
        print(n, len(combos), flush=True)
    d = pl.DataFrame(rows)
    d.write_csv(OUT)
    # seed disagreement at the 0.5 threshold (label flips between two single seeds)
    flips = [np.mean((Pr["sarc"][i] >= .5) != (Pr["sarc"][j] >= .5)) for i, j in itertools.combinations(range(10), 2)]
    print("mean pairwise sarcasm label-flip rate between single seeds:", round(float(np.mean(flips)), 4))
    print(d.group_by("n").agg(pl.col("deb_score").mean().round(4).alias("deb_mean"), pl.col("deb_score").std().round(4).alias("deb_sd"),
                              pl.col("lead_score").mean().round(4).alias("lead_mean"), pl.col("lead_score").std().round(4).alias("lead_sd"),
                              pl.col("deb_auc_sarc_AUr").mean().round(4)).sort("n"))


if __name__ == "__main__":
    main()
