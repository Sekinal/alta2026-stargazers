"""Attack 2c: second-order leakage when a policy is cross-fitted on stacked OOF.

current : ensemble.stack OOF (stacker for fold k fitted on folds != k), then policies.crossfit
nested  : outer fold k -> inner stacked OOF on folds != k built with stackers that never see
          fold k; the policy is fitted on that inner OOF and applied to the outer stacked fold k.
Run with the real members, and with extra pure-noise members (more stacker capacity).
"""

from _lib import *  # noqa: F403
from alta.common import PREDS
from alta.validation import ensemble
from alta.validation.data import load_oof

MEMBERS = ["emb_bge_large_lr", "emb_bge_base_lr", "emb_mpnet_lr", "emb_minilm_lr", "tfidf_lr"]
SPECS = ("fixed+ukg", "variety+ukg", "segment+ukg", "prevtuned+ukg")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_oof(F, d, rows, C=1.0):
    """Mirror of ensemble.stack restricted to `rows`: per task x variety, cross-fitted over folds."""
    out = {t: np.full(len(d), np.nan) for t in TASKS}
    for t in TASKS:
        X = np.column_stack([logit(f) for f in F[t]] + [d.source.astype(float)])
        for v in (0, 1):
            mv = rows & (d.variety == v)
            for k in np.unique(d.fold[rows]):
                tr, te = mv & (d.fold != k), mv & (d.fold == k)
                clf = LogisticRegression(C=C, max_iter=1000).fit(X[tr], d.y[t][tr])
                out[t][te] = clf.predict_proba(X[te])[:, 1]
    return out


def evaluate(F, d, C=1.0):
    cur = stack_oof(F, d, np.ones(len(d), bool), C)
    dc = d.with_p(cur)
    res = {}
    for spec in SPECS:
        pc, _ = policies.crossfit(spec, dc)
        nested = {t: np.zeros(len(d), np.int8) for t in TASKS}
        for k in np.unique(d.fold):
            inner_rows = d.fold != k
            inner = stack_oof(F, d, inner_rows, C)
            di = d.with_p({t: np.nan_to_num(inner[t]) for t in TASKS}).subset(inner_rows)
            params = policies.fit(spec, di)
            out = policies.apply(params, dc.subset(d.fold == k))
            for t in TASKS:
                nested[t][d.fold == k] = out[t]
        res[spec] = (components(d, pc)["score"], components(d, nested)["score"])
    return cur, res


def main():
    d0 = base_eval()
    members = [load_oof(r) for r in MEMBERS]
    F = {t: [m.p[t] for m in members] for t in TASKS}
    # sanity: our mirror equals ensemble.stack
    ensemble.stack(MEMBERS, "rt_stats_stack")
    ref = load_oof("rt_stats_stack")
    cur, res = evaluate(F, d0)
    print("mirror max abs diff vs ensemble.stack:",
          max(float(np.abs(cur[t] - ref.p[t]).max()) for t in TASKS))
    for ext in ("parquet", "json"):
        (PREDS / f"rt_stats_stack.{ext}").unlink(missing_ok=True)
    rows = [{"members": "5 real", "policy": s, "current": a, "nested": b, "leak": a - b} for s, (a, b) in res.items()]
    rng = np.random.default_rng(3)
    for n_noise, C in ((20, 1.0), (60, 1.0), (60, 100.0)):
        Fn = {t: F[t] + [rng.random(len(d0)) for _ in range(n_noise)] for t in TASKS}
        _, res = evaluate(Fn, d0, C)
        rows += [{"members": f"5 real + {n_noise} noise, C={C}", "policy": s, "current": a, "nested": b,
                  "leak": a - b} for s, (a, b) in res.items()]
        print(rows[-4:], flush=True)
    with pl.Config(tbl_rows=50, tbl_width_chars=200, float_precision=4, fmt_str_lengths=60):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
