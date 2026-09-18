"""Perturbation suite for a saved run's fold-k checkpoint (models/<run>/fold<k>/), then the report.

  uv run python -m experiments.encoders.robustness --run enc_deb_l_s13 [--fold 0]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import subprocess

from experiments.encoders.common import MODELS, ROOT, load_fold, robustness_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    for run in a.run:
        model, tok, cfg = load_fold(MODELS / run / f"fold{a.fold}")
        n = robustness_preds(model, tok, run, a.fold, cfg["max_len"], not cfg.get("no_prefix", False))
        print(f"{run}: {n} kinds scored with the fold-{a.fold} model", flush=True)
        subprocess.run([sys.executable, str(ROOT / "experiments/robustness/perturb.py"), "report", "--run", run,
                        "--dir", str(ROOT / f"experiments/robustness/holdout_f{a.fold}_s2026")], check=False)


if __name__ == "__main__":
    main()
