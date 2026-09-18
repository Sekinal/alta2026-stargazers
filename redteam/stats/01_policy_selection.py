"""Attack 1a: optimism of reporting max over the 9 cross-fitted policies.

For every run in preds/ (non rt_ ones):
  reported  = max_j crossfit_score(policy_j)            (what report.py headlines)
  nested    = outer fold k: choose policy by 4-fold crossfit on the other 4 folds only,
              fit it on those 4 folds, apply to fold k; pooled score over all rows
  split     = winner's curse on halves: pick best policy on a random stratified half A of the
              (already cross-fitted) labels, measure it on half B; optimism = E[A_max - B_chosen]
Also a null run (pure-noise probabilities) to show the chance floor of each policy.
"""

from _lib import *  # noqa: F403
from alta.validation.data import load_oof, list_runs
from alta.validation.resampling import cell_ids

SPECS = policies.DEFAULT_POLICIES


def crossfit_all(d):
    return {s: policies.crossfit(s, d)[0] for s in SPECS}


def nested(d):
    pred = {t: np.zeros(len(d), np.int8) for t in TASKS}
    chosen = []
    for k in np.unique(d.fold):
        inner = d.subset(d.fold != k)
        sc = {s: components(inner, policies.crossfit(s, inner)[0])["score"] for s in SPECS}
        best = max(sc, key=sc.get)
        chosen.append(best)
        out = policies.apply(policies.fit(best, inner), d.subset(d.fold == k))
        for t in TASKS:
            pred[t][d.fold == k] = out[t]
    return components(d, pred)["score"], chosen


def split_optimism(d, preds, R=300, seed=0):
    rng = np.random.default_rng(seed)
    cells = cell_ids(d)
    opt = []
    for _ in range(R):
        a = np.zeros(len(d), bool)
        for c in np.unique(cells):
            idx = np.flatnonzero(cells == c)
            a[rng.choice(idx, len(idx) // 2, replace=False)] = True
        da, db = d.subset(a), d.subset(~a)
        sa = {s: components(da, {t: p[t][a] for t in TASKS})["score"] for s, p in preds.items()}
        best = max(sa, key=sa.get)
        sb = components(db, {t: preds[best][t][~a] for t in TASKS})["score"]
        opt.append(sa[best] - sb)
    return float(np.mean(opt)), float(np.std(opt) / np.sqrt(R))


def main():
    rows = []
    runs = [r for r in list_runs() if not r.startswith("rt_")]
    base = base_eval()
    rng = np.random.default_rng(7)
    null = base.with_p({t: rng.random(len(base)) for t in TASKS})
    for name in runs + ["NULL(uniform p)"]:
        d = null if name.startswith("NULL") else load_oof(name)
        preds = crossfit_all(d)
        sc = {s: components(d, p)["score"] for s, p in preds.items()}
        best = max(sc, key=sc.get)
        ns, chosen = nested(d)
        so, se = split_optimism(d, preds, R=200)
        rows.append({"run": name, "reported_max": sc[best], "best_policy": best,
                     "fixed": sc["fixed"], "mean_policy": float(np.mean(list(sc.values()))),
                     "nested": ns, "max_minus_nested": sc[best] - ns,
                     "nested_choices": ",".join(chosen), "half_split_optimism": so, "se": se})
        print(rows[-1], flush=True)
        if name.startswith("NULL"):
            print({s: round(v, 4) for s, v in sc.items()})
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=250, float_precision=4, fmt_str_lengths=120):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
