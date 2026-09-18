"""Attack 4: transductive prevalence policies when the test set is not a valid-like stratified sample.

Policies are fitted on the full OOF pool (as make_test_submission does) and applied to test-like
draws from the pool OOF (with replacement) under several sampling scenarios. prev/prevtuned are
re-applied on every draw (they label the top-q of each segment of the rows they see); threshold
policies are row-wise. Reports mean score and the loss vs the stratified scenario.
"""

from _lib import *  # noqa: F403
from alta.validation.data import load_oof
from alta.validation.resampling import cell_ids, target_counts

SPECS = ("fixed+ukg", "variety+ukg", "segment+ukg", "prev+ukg", "prevtuned+ukg")
R = 300


def draws(d, scenario, rng):
    cells = cell_ids(d)
    base = target_counts(d)  # valid's cell counts (757)
    n = sum(base.values())
    for _ in range(R):
        if scenario == "stratified(valid counts)":
            counts = base
        elif scenario == "simple random n=757":
            yield rng.integers(0, len(d), n)
            continue
        elif scenario.startswith("small stratified"):
            m = int(scenario.split("n=")[1])
            counts = {c: rng.binomial(k, m / n) for c, k in base.items()}
        else:
            seg, factor = scenario.split(" x")[0].split(": ")[1], float(scenario.split(" x")[1])
            v, s = {"UK-reddit": (1, 1), "AU-reddit": (0, 1)}[seg]
            counts = dict(base)
            # scale sarcastic cells of that segment, keep segment size fixed by removing/adding non-sarc
            for sent in (0, 1):
                c1, c0 = v * 8 + s * 4 + sent * 2 + 1, v * 8 + s * 4 + sent * 2
                new = int(round(base.get(c1, 0) * factor))
                counts[c0] = max(0, base.get(c0, 0) - (new - base.get(c1, 0)))
                counts[c1] = new
        parts = [np.flatnonzero(cells == c)[rng.integers(0, (cells == c).sum(), k)]
                 for c, k in counts.items() if k > 0]
        yield np.concatenate(parts)


def main():
    scenarios = ["stratified(valid counts)", "simple random n=757", "small stratified n=200",
                 "sarc prev shift: UK-reddit x1.5", "sarc prev shift: UK-reddit x0.67",
                 "sarc prev shift: AU-reddit x1.3", "sarc prev shift: AU-reddit x0.7"]
    rows = []
    for run in ("emb_ens3_lr", "emb_bge_large_lr", "tfidf_lr"):
        d = load_oof(run)
        params = {s: policies.fit(s, d) for s in SPECS}
        for sc in scenarios:
            rng = np.random.default_rng(0)
            res = {s: [] for s in SPECS}
            for sel in draws(d, sc, rng):
                sub = d.subset(sel)
                for s in SPECS:
                    res[s].append(components(sub, policies.apply(params[s], sub))["score"])
            rows.append({"run": run, "scenario": sc, **{s: float(np.mean(v)) for s, v in res.items()}})
        print(run, flush=True)
    df = pl.DataFrame(rows)
    ref = df.filter(pl.col("scenario") == "stratified(valid counts)").drop("scenario")
    delta = df.join(ref, on="run", suffix="_ref").select(
        "run", "scenario", *[(pl.col(s) - pl.col(f"{s}_ref")).alias(f"d_{s}") for s in SPECS])
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250, float_precision=4, fmt_str_lengths=40):
        print(df)
        print(delta)


if __name__ == "__main__":
    main()
