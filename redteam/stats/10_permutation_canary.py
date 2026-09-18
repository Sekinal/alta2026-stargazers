"""Proposed leakage detector: rerun a training pipeline on labels permuted within segment.

An honest OOF pipeline then scores at the 'segment prior' floor (text carries no signal);
any pipeline that lets a row's own label reach its prediction scores above it. Compared with
check_run's static canaries (score > 0.93, UK-reddit AUC > 0.95), which the in-sample cheat passes.
"""

from _lib import *  # noqa: F403


def permute_within_segment(d, rng):
    y = {t: d.y[t].copy() for t in TASKS}
    seg = d.segment()
    for s in range(4):
        idx = np.flatnonzero(seg == s)
        perm = rng.permutation(idx)
        for t in TASKS:
            y[t][idx] = d.y[t][perm]
    return y


def main():
    d0 = base_eval()
    X = design(d0, emb("bge_base"))
    for r in range(3):
        y = permute_within_segment(d0, np.random.default_rng(r))
        d = EvalData(d0.uid, d0.variety, d0.source, d0.fold, d0.origin, d0.n_words, d0.p, y)
        honest, leaky = {}, {}
        for t in TASKS:
            honest[t] = np.zeros(len(d))
            for k in range(5):
                tr = d.fold != k
                honest[t][~tr] = fit_predict(X[tr], y[t][tr], [X[~tr]])[0]
            leaky[t] = fit_predict(X, y[t], [X])[0]
            part = np.random.default_rng(10 + r).random(len(d)) < 0.25
        partial = {t: np.where(part, leaky[t], honest[t]) for t in TASKS}
        for name, p in (("honest", honest), ("insample_all", leaky), ("insample_25pct", partial)):
            dp = d.with_p(p)
            res = {s: round(score_policy(dp, s), 4) for s in ("fixed+ukg", "segment+ukg", "prevtuned+ukg")}
            print(f"perm{r} {name:15s}", res, flush=True)


if __name__ == "__main__":
    main()
