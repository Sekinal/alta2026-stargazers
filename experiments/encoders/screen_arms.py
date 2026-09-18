"""Compare multi-seed ARMS on one holdout fold (screening protocol), with the generalization read-outs.

Per arm: each seed's holdout score (mean +- SD over seeds), the seed-averaged arm (mean of the
seeds' probabilities on the holdout rows), its components, the worst slices (UK-reddit sarcasm,
AU-reddit sentiment, short <= 10 words and long > 80 words: macro-F1 per task pooled over
varieties), and the paired P(first arm beats this arm) on holdout-sized replicates. The decision
policy is segment+ukg cross-fitted WITHIN the holdout, exactly like `alta.validate screen`.

  uv run python -m experiments.encoders.screen_arms --fold 0 \
      base=enc_deb_l_s13,enc_deb_l_s14,enc_deb_l_s15 pft=enc_pftsarc_deb_l_s13,...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # also runnable as a plain script

import argparse

import numpy as np

from alta.common import load_pool
from alta.validation import policies
from alta.validation.metrics import components, macro_f1
from alta.validation.report import load_holdout
from alta.validation.resampling import paired

SPEC = policies.DEFAULT_POLICY


def slices(d, pred) -> dict[str, float]:
    uk_r = (d.variety == 1) & (d.source == 1)
    au_r = (d.variety == 0) & (d.source == 1)
    short, long_ = d.n_words <= 10, d.n_words > 80
    return {
        "sarc_UKreddit": macro_f1(d.y["sarc"][uk_r], pred["sarc"][uk_r]),
        "sent_AUreddit": macro_f1(d.y["sent"][au_r], pred["sent"][au_r]),
        "short_sent": macro_f1(d.y["sent"][short], pred["sent"][short]),
        "short_sarc": macro_f1(d.y["sarc"][short], pred["sarc"][short]),
        "long_sent": macro_f1(d.y["sent"][long_], pred["sent"][long_]),
        "long_sarc": macro_f1(d.y["sarc"][long_], pred["sarc"][long_]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=run1,run2,...  (first arm = reference)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--stack_with", default="", help="comma-separated runs: logit-mean each arm with these "
                    "(e.g. the other stk members) to read the arm's value inside the stack")
    a = ap.parse_args()
    pool = load_pool()
    res = {}
    for spec in a.arms:
        name, runs = spec.split("=", 1)
        ds = [load_holdout(r, pool, a.fold) for r in runs.split(",")]
        seed_scores = [components(d, policies.crossfit(SPEC, d)[0])["score"] for d in ds]
        avg = ds[0]
        for d in ds[1:]:
            assert (d.uid == avg.uid).all()
        avg.p = {t: np.mean([d.p[t] for d in ds], axis=0) for t in avg.p}
        if a.stack_with:
            others = [load_holdout(r, pool, a.fold) for r in a.stack_with.split(",")]
            for o in others:
                assert (o.uid == avg.uid).all()
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            avg.p = {t: 1 / (1 + np.exp(-np.mean([lg(avg.p[t])] + [lg(o.p[t]) for o in others], axis=0)))
                     for t in avg.p}
        pred = policies.crossfit(SPEC, avg)[0]
        res[name] = (avg, pred, seed_scores)
    ref = next(iter(res))
    print(f"# arms on holdout fold {a.fold} ({len(res[ref][0])} rows), policy {SPEC} cross-fitted in the holdout\n")
    print("| arm | seeds | seed mean | seed SD | arm score | sent-AU | sent-UK | sarc-AU | sarc-UK | "
          "sarc UK-reddit | sent AU-reddit | short sent | short sarc | long sent | long sarc | P(ref better) |")
    print("|" + "---|" * 16)
    for name, (d, pred, ss) in res.items():
        c, s = components(d, pred), slices(d, pred)
        p = "" if name == ref else f"{paired(res[ref][0], res[ref][1], pred, B=a.B, n_test=len(d))['score']['p_a_better']:.3f}"
        print(f"| {name} | {len(ss)} | {np.mean(ss):.4f} | {np.std(ss, ddof=1) if len(ss) > 1 else float('nan'):.4f} "
              f"| {c['score']:.4f} | {c['sent-AU']:.3f} | {c['sent-UK']:.3f} | {c['sarc-AU']:.3f} | {c['sarc-UK']:.3f} | "
              + " | ".join(f"{v:.3f}" for v in s.values()) + f" | {p} |")
    print("\nP(ref better) < 0.8 = tie (and > 0.2 means the arm is not better either).")


if __name__ == "__main__":
    main()
