"""Attack 2a: build cheating runs and see whether check_run / report catch them.

All cheats derive from the same honest model (LR on bge_base, pooled design, C=1).
Runs are saved as preds/rt_stats_*.parquet and removed at the end (unless --keep).
"""

import sys

from _lib import *  # noqa: F403
from sklearn.model_selection import KFold

from alta.common import PREDS, save_preds
from alta.validation.checks import check_run
from alta.validation.report import evaluate_policies

RNG = np.random.default_rng(0)


def honest(X, d, fold):
    out = {}
    for t in TASKS:
        p = np.zeros(len(d))
        for k in np.unique(fold):
            tr = fold != k
            p[~tr] = fit_predict(X[tr], d.y[t][tr], [X[~tr]])[0]
        out[t] = p
    return out


def main():
    d = base_eval()
    X = design(d, emb("bge_base"))
    valid = d.origin == "valid"
    oof = honest(X, d, d.fold)
    insample = {t: fit_predict(X, d.y[t], [X])[0] for t in TASKS}
    # model trained on all train rows + all valid rows; used for valid rows only
    cheats = {
        "honest": oof,
        "insample_all": insample,
        "insample_valid_only": {t: np.where(valid, insample[t], oof[t]) for t in TASKS},
    }
    part = RNG.random(len(d)) < 0.25
    cheats["insample_25pct"] = {t: np.where(part, insample[t], oof[t]) for t in TASKS}
    # label nudging: p <- p + eps*(y-0.5) (logit space), a "soft" leak
    for eps in (0.5, 1.0):
        lg = {t: np.log(np.clip(oof[t], 1e-6, 1 - 1e-6) / (1 - np.clip(oof[t], 1e-6, 1 - 1e-6))) for t in TASKS}
        cheats[f"logit_nudge_{eps}"] = {t: 1 / (1 + np.exp(-(lg[t] + eps * (2 * d.y[t] - 1)))) for t in TASKS}
    # plain KFold (no groups, no stratification): different folds, still OOF
    kf = np.zeros(len(d), np.int8)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=1).split(X)):
        kf[te] = k
    cheats["other_folds_kfold"] = honest(X, d, kf)

    rows = []
    for name, p in cheats.items():
        run = f"rt_stats_{name}"
        save_preds(run, pl.DataFrame({"uid": d.uid, "p_sent": p["sent"], "p_sarc": p["sarc"]}),
                   meta={"cheat": name})
        chk = check_run(run)
        pol = evaluate_policies(run)
        best = next(r for r in pol if r["policy"] == "segment+ukg")  # report headline policy
        fixed = next(r for r in pol if r["policy"] == "fixed+ukg")
        flagged = [m for lvl, m in chk if lvl != "OK"]
        rows.append({"run": name, "fixed": fixed["score"], "segment+ukg": best["score"],
                     "valid_score": best["valid_score"],
                     "valid-pool gap": best["valid_score"] - best["score"],
                     "flags": " | ".join(flagged) or "-"})
        print(rows[-1], flush=True)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=300, float_precision=4, fmt_str_lengths=200):
        print(pl.DataFrame(rows))
    if "--keep" not in sys.argv:
        for name in cheats:
            for ext in ("parquet", "json"):
                (PREDS / f"rt_stats_{name}.{ext}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
