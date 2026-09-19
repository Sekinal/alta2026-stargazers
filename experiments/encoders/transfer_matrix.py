"""Cross-variety transfer matrix for deberta-v3-large (paper table).

Rows: training data {en-AU only, en-UK only, both} (enc_deb_l_trainAU_s*, enc_deb_l_trainUK_s*, enc_deb_l_s*);
columns: evaluation variety {AU, UK}. Metrics per cell, mean +- SD over seeds 13/14/15, full 5-fold OOF:
  sentiment macro-F1 and AUC on all rows of the eval variety;
  sarcasm macro-F1 and AUC on the eval variety's Reddit rows (UK google has 1 sarcastic row).
Macro-F1 uses the shipped policy (segment+ukg) cross-fitted over the project folds, per run.

  uv run python -m experiments.encoders.transfer_matrix   -> results/paper/transfer_matrix.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from alta.common import ROOT, load_pool
from alta.validation import policies
from alta.validation.data import load_oof
from alta.validation.metrics import macro_f1

ARMS = {"AU": "enc_deb_l_trainAU_s{}", "UK": "enc_deb_l_trainUK_s{}", "both": "enc_deb_l_s{}"}
SEEDS = (13, 14, 15)


def cells(run: str, pool) -> list[dict]:
    d = load_oof(run, pool)
    pred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)
    out = []
    for v, vn in ((0, "AU"), (1, "UK")):
        m_all = d.variety == v
        m_red = m_all & (d.source == 1)
        out.append({"eval": vn,
                    "sent_f1": macro_f1(d.y["sent"][m_all], pred["sent"][m_all]),
                    "sent_auc": roc_auc_score(d.y["sent"][m_all], d.p["sent"][m_all]),
                    "sarc_reddit_f1": macro_f1(d.y["sarc"][m_red], pred["sarc"][m_red]),
                    "sarc_reddit_auc": roc_auc_score(d.y["sarc"][m_red], d.p["sarc"][m_red])})
    return out


def main() -> None:
    pool = load_pool()
    rows = []
    for train, pat in ARMS.items():
        for s in SEEDS:
            for c in cells(pat.format(s), pool):
                rows.append({"train": train, "seed": s, **c})
    df = pl.DataFrame(rows)
    metrics = ["sent_f1", "sent_auc", "sarc_reddit_f1", "sarc_reddit_auc"]
    table = (df.group_by("train", "eval", maintain_order=True)
             .agg([pl.col(m).mean().alias(m) for m in metrics] + [pl.col(m).std().alias(f"{m}_sd") for m in metrics]
                  + [pl.len().alias("n_seeds")]))
    out = ROOT / "results" / "paper" / "transfer_matrix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    df.write_csv(out.with_name("transfer_matrix_per_seed.csv"))
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=3):
        print(table)
    print(out)


if __name__ == "__main__":
    main()
