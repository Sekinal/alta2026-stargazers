"""Perturbation drops per ARM: mean over the arm's seeds of d_score (and d_sarc-UK / d_sent-AU) per kind.

Each seed's fold-0 robustness predictions (written by train.py / robustness.py with the fold-k
model) are reported with experiments/robustness/perturb.py (thresholds fitted on clean), then averaged.

  uv run python -m experiments.encoders.robust_arms base=enc_deb_l_s13,enc_deb_l_s14,enc_deb_l_s15 pft=...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import contextlib
import io

import numpy as np

from experiments.robustness import perturb
from experiments.robustness.perturb import robustness_report

# Thresholds are fitted on the clean holdout predictions only; hide preds/ so the optional
# "thresholds from the other folds' OOF" read-out is skipped (screening runs have no OOF there).
perturb.PREDS = Path("/nonexistent")

KINDS = ("lowercase", "strip_punct", "typo_3pct", "au_to_uk", "uk_to_au", "to_us", "google_artefacts_on",
         "google_artefacts_off", "whitespace_noise", "drop_prefix")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=run1,run2,...")
    a = ap.parse_args()
    print("| arm | metric | clean | " + " | ".join(KINDS) + " | worst |")
    print("|" + "---|" * (len(KINDS) + 4))
    for spec in a.arms:
        name, runs = spec.split("=", 1)
        per = []
        for r in runs.split(","):
            with contextlib.redirect_stdout(io.StringIO()):
                per.append({row["kind"]: row for row in robustness_report(r)})
        for m in ("score", "sarc-UK", "sent-AU"):
            clean = np.mean([p["clean"][m] for p in per])
            drops = [np.mean([p[k][f"d_{m}"] for p in per]) for k in KINDS]
            print(f"| {name} | d_{m} | {clean:.4f} | " + " | ".join(f"{x:+.4f}" for x in drops)
                  + f" | {min(drops):+.4f} |")


if __name__ == "__main__":
    main()
