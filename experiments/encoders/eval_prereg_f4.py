"""Evaluate the pre-registered recipe R_deb vs stk_v5 on a holdout fold (experiments/prereg/fold4_pseudo_sarc.md).

R_deb: sentiment = stk_v5's (logit mean of enc_deb_l_x5, cls_e5me5_lr, llm_q35_9b_probe);
sarcasm = logit mean of [0.5 * mean(enc_deb_l_x5) + 0.5 * mean(student seeds), cls_e5me5_lr, llm_q35_9b_probe].
Policy segment+ukg cross-fitted within the holdout; success = p_a_better >= 0.8 (n_test = fold size)
AND sarc-AU(R_deb) >= sarc-AU(stk_v5). Nothing is fitted on the holdout labels except the policy.

  uv run python -m experiments.encoders.eval_prereg_f4 --fold 4 --students enc_pseudosarc_deb_l_f4_s13 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse
import copy

import numpy as np

from alta.common import load_pool
from alta.validation import policies
from alta.validation.metrics import components
from alta.validation.report import load_holdout
from alta.validation.resampling import paired

LEAD, DEB, OTHERS = "stk_v5", "enc_deb_l_x5", ("cls_e5me5_lr", "llm_q35_9b_probe")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--students", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    pool = load_pool()
    h = {r: load_holdout(r, pool, a.fold) for r in (LEAD, DEB, *OTHERS, *a.students)}
    uid = h[LEAD].uid
    assert all((d.uid == uid).all() for d in h.values()), "holdout rows misaligned"

    # sanity: stk_v5 on the holdout == logit mean of its members (so R_deb differs only in the deberta sarcasm)
    for t in ("sent", "sarc"):
        rebuilt = sigmoid(np.mean([logit(h[r].p[t]) for r in (DEB, *OTHERS)], axis=0))
        print(f"stk_v5 {t}: max |rebuilt - saved| = {np.abs(rebuilt - h[LEAD].p[t]).max():.2e}")

    student = np.mean([h[r].p["sarc"] for r in a.students], axis=0)
    deb_sarc = 0.5 * h[DEB].p["sarc"] + 0.5 * student
    r = copy.copy(h[LEAD])
    r.p = {"sent": h[LEAD].p["sent"],
           "sarc": sigmoid(np.mean([logit(deb_sarc)] + [logit(h[o].p["sarc"]) for o in OTHERS], axis=0))}
    pa, _ = policies.crossfit(policies.DEFAULT_POLICY, r)
    pb, _ = policies.crossfit(policies.DEFAULT_POLICY, h[LEAD])
    ca, cb = components(r, pa), components(h[LEAD], pb)
    pr = paired(h[LEAD], pa, pb, B=a.B, n_test=len(uid))
    print(f"\n# Pre-registered R_deb vs {LEAD}, holdout fold {a.fold} ({len(uid)} rows), policy "
          f"{policies.DEFAULT_POLICY} cross-fitted in the holdout; students: {', '.join(a.students)}\n")
    print("| metric | R_deb | stk_v5 | diff | p_a_better |")
    print("|---|---|---|---|---|")
    for k in ("sent-AU", "sent-UK", "sarc-AU", "sarc-UK", "score"):
        print(f"| {k} | {ca[k]:.4f} | {cb[k]:.4f} | {ca[k] - cb[k]:+.4f} | {pr[k]['p_a_better']:.3f} |")
    ok = pr["score"]["p_a_better"] >= 0.8 and ca["sarc-AU"] >= cb["sarc-AU"]
    print(f"\nSuccess criterion (p >= 0.8 AND sarc-AU not lower): {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
