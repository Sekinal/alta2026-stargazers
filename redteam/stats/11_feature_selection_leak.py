"""Attack 2d: the classic subtle leak, supervised feature selection fitted on ALL pool labels
before an otherwise honest 5-fold CV. Saved as a run and passed through check_run; then the
same pipeline on within-segment permuted labels (the proposed canary) exposes it.
"""

from _lib import *  # noqa: F403
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from alta.common import PREDS, save_preds
from alta.validation.checks import check_run

K = 3000


def run(Xall, d, y, leaky):
    out = {}
    for t in TASKS:
        p = np.zeros(len(d))
        sel_all = SelectKBest(chi2, k=K).fit(Xall, y[t]) if leaky else None
        for k in range(5):
            tr = d.fold != k
            sel = sel_all or SelectKBest(chi2, k=K).fit(Xall[tr], y[t][tr])
            X = sel.transform(Xall)
            p[~tr] = lr(4.0).fit(X[tr], y[t][tr]).predict_proba(X[~tr])[:, 1]
        out[t] = p
    return out


def perm_labels(d, rng):
    y = {t: d.y[t].copy() for t in TASKS}
    seg = d.segment()
    for s in range(4):
        idx = np.flatnonzero(seg == s)
        perm = rng.permutation(idx)
        for t in TASKS:
            y[t][idx] = d.y[t][perm]
    return y


def main():
    d = base_eval()
    texts = load_pool()["text"].to_list()
    Xall = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    for leaky in (False, True):
        name = f"rt_stats_fs_{'leaky' if leaky else 'honest'}"
        p = run(Xall, d, d.y, leaky)
        save_preds(name, pl.DataFrame({"uid": d.uid, "p_sent": p["sent"], "p_sarc": p["sarc"]}))
        flags = [m for lvl, m in check_run(name) if lvl != "OK"]
        dp = d.with_p(p)
        print(name, {s: round(score_policy(dp, s), 4) for s in ("fixed+ukg", "segment+ukg")}, "flags:", flags or "-")
        for ext in ("parquet", "json"):
            (PREDS / f"{name}.{ext}").unlink(missing_ok=True)
        for r in range(2):
            y = perm_labels(d, np.random.default_rng(r))
            dperm = EvalData(d.uid, d.variety, d.source, d.fold, d.origin, d.n_words, d.p, y)
            pp = run(Xall, dperm, y, leaky)
            print(f"   permuted-label canary {r}:",
                  {s: round(score_policy(dperm.with_p(pp), s), 4) for s in ("fixed+ukg", "segment+ukg")})


if __name__ == "__main__":
    main()
