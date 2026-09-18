"""Tests for the validation stack. Run: uv run pytest -q"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import f1_score

import alta.common as common
import alta.validation.data as vdata
import alta.validation.submission as vsub
from alta.validation import policies
from alta.validation.data import TASKS, pool_eval
from alta.validation.metrics import components, macro_f1, macro_f1_counts
from alta.validation.resampling import cell_ids, paired, replicate_scores, target_counts, replicate_indices

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def pool():
    return common.load_pool()


@pytest.fixture(scope="module")
def d(pool):
    """Pool with informative-but-noisy synthetic probabilities."""
    e = pool_eval(pool)
    rng = np.random.default_rng(1)
    p = {t: 1 / (1 + np.exp(-(2.5 * (e.y[t] - 0.5) * 2 + rng.normal(0, 1.6, len(e))))) for t in TASKS}
    return e.with_p(p)


# ----------------------------------------------------------------- metric
@pytest.mark.parametrize("trial", range(200))
def test_macro_f1_matches_sklearn(trial):
    rng = np.random.default_rng(trial)
    n = int(rng.integers(1, 40))
    y = rng.random(n) < rng.random()
    p = rng.random(n) < rng.random()
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def test_macro_f1_degenerate_cases():
    for y, p in [([0, 0], [0, 0]), ([1, 1], [1, 1]), ([0, 0], [1, 1]), ([1, 0], [0, 0])]:
        y, p = np.array(y, bool), np.array(p, bool)
        assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def test_components_match_common_score(pool, d):
    pred = {t: (d.p[t] >= 0.5).astype(np.int8) for t in TASKS}
    df = pool.with_columns(pl.Series("sentiment_pred", pred["sent"]), pl.Series("sarcasm_pred", pred["sarc"]))
    ref = common.score(df)
    ours = components(d, pred)
    for k in ("sent-AU", "sent-UK", "sarc-AU", "sarc-UK", "score"):
        assert ours[k] == pytest.approx(ref[k], abs=1e-12)


def test_matches_official_scorer(tmp_path, pool, d):
    """End to end against the organisers' evaluate.py on valid.csv."""
    if not vsub.SCORER.exists():
        pytest.skip("scoring program not unpacked")
    valid = pool.filter(pl.col("origin") == "valid").sort("uid")
    m = d.origin == "valid"
    order = np.argsort(d.uid[m])
    pred = {t: (d.p[t][m][order] >= 0.5).astype(np.int64) for t in TASKS}
    ans = valid.select(pl.col("source", "variety").cast(pl.String), "text").with_columns(
        pl.Series("sentiment", pred["sent"]), pl.Series("sarcasm", pred["sarc"]))
    ans.write_csv(tmp_path / "answer.csv")
    official = vsub.official_score(tmp_path / "answer.csv", common.DATA / "valid.csv")
    ours = components(d.subset(m), {t: d.p[t][m] >= 0.5 for t in TASKS})
    assert official["score"] == pytest.approx(ours["score"], abs=6e-5)  # scorer rounds to 4 dp
    assert official["f1-sarcasm-en-UK"] == pytest.approx(ours["sarc-UK"], abs=6e-5)


# ----------------------------------------------------------------- folds
def test_folds_are_group_disjoint_and_stable(pool):
    assert pool.group_by("text").agg(pl.col("fold").n_unique()).filter(pl.col("fold") > 1).height == 0
    assert common.fold_hash(pool) == common.fold_hash(common.load_pool(refresh=True))


def test_folds_are_stratified(pool):
    cells = pool.group_by("variety", "source", "sentiment", "sarcasm", "fold").len()
    spread = cells.group_by("variety", "source", "sentiment", "sarcasm").agg(
        (pl.col("len").max() - pl.col("len").min()).alias("spread"))
    # round-robin gives +-1 per stratum; multi-row duplicate groups can add 1 more
    assert spread["spread"].max() <= 3


# ----------------------------------------------------------------- policies
def test_sweep_matches_brute_force():
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.2
    p = np.round(rng.random(300), 2)  # ties on purpose
    thr, f = policies.sweep(y, p)
    for t, fv in zip(thr, f):
        assert fv == pytest.approx(macro_f1(y, p >= t))
    brute = max(macro_f1(y, p >= t) for t in np.unique(np.concatenate([p, [1.1]])))
    assert f.max() == pytest.approx(brute)


@pytest.mark.parametrize("spec", policies.DEFAULT_POLICIES)
def test_crossfit_never_sees_held_out_labels(d, spec):
    """Scrambling fold k's labels must not change fold k's predictions."""
    pred, _ = policies.crossfit(spec, d)
    k = 2
    rng = np.random.default_rng(7)
    y2 = {t: d.y[t].copy() for t in TASKS}
    for t in TASKS:
        m = d.fold == k
        y2[t][m] = rng.permutation(y2[t][m]) ^ (rng.random(m.sum()) < 0.3)
    d2 = vdata.EvalData(d.uid, d.variety, d.source, d.fold, d.origin, d.n_words, d.p, y2)
    pred2, _ = policies.crossfit(spec, d2)
    for t in TASKS:
        assert (pred[t][d.fold == k] == pred2[t][d.fold == k]).all()


def test_prevalence_policy_matches_rates(d):
    params = policies.fit("prev", d)
    pred = policies.apply(params, d)
    seg = d.segment()
    for t in TASKS:
        for s in range(4):
            m = seg == s
            assert abs(pred[t][m].mean() - d.y[t][m].mean()) <= 1 / m.sum() + 1e-12


def test_ukg_rule(d):
    pred = policies.apply(policies.fit("fixed+ukg", d), d)
    assert pred["sarc"][(d.variety == 1) & (d.source == 0)].sum() == 0


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        policies.parse("bogus+ukg")


# ----------------------------------------------------------------- resampling
def test_bootstrap_keeps_valid_cell_counts(d):
    idx = replicate_indices(d, B=5, seed=0)
    counts = target_counts(d)
    cells = cell_ids(d)
    for v in (0, 1):
        for b in range(5):
            got = dict(zip(*np.unique(cells[idx[v][b]], return_counts=True)))
            want = {c: k for c, k in counts.items() if c // 8 == v}
            assert {int(c): int(k) for c, k in got.items()} == want


def test_bootstrap_reproducible_and_centred(d):
    pred = {t: (d.p[t] >= 0.5).astype(np.int8) for t in TASKS}
    a = replicate_scores(d, pred, replicate_indices(d, B=300, seed=5))
    b = replicate_scores(d, pred, replicate_indices(d, B=300, seed=5))
    assert np.array_equal(a["score"], b["score"])
    full = components(d, pred)["score"]
    assert abs(a["score"].mean() - full) < 0.02


def test_paired_identical_is_zero(d):
    pred = {t: (d.p[t] >= 0.5).astype(np.int8) for t in TASKS}
    res = paired(d, pred, pred, B=100)
    assert res["score"]["mean"] == 0 and res["score"]["p_a_better"] == 0


# ----------------------------------------------------------------- run files + submission
@pytest.fixture()
def fake_run(tmp_path, monkeypatch, d):
    monkeypatch.setattr(common, "PREDS", tmp_path)
    monkeypatch.setattr(vdata, "PREDS", tmp_path)
    monkeypatch.setattr(vsub, "SUBMISSIONS", tmp_path / "subs")
    oof = pl.DataFrame({"uid": d.uid, "p_sent": d.p["sent"], "p_sarc": d.p["sarc"]})
    common.save_preds("fake", oof)
    return "fake"


def test_check_run_passes_and_catches_problems(fake_run, tmp_path):
    from alta.validation.checks import FAIL, check_run
    assert not [m for lvl, m in check_run(fake_run) if lvl == FAIL]
    preds = pl.read_parquet(tmp_path / "fake.parquet")
    preds.head(100).write_parquet(tmp_path / "fake.parquet")
    assert any(lvl == FAIL for lvl, _ in check_run(fake_run))


def test_fold_drift_detected(fake_run, tmp_path):
    import json
    from alta.validation.checks import FAIL, check_run
    meta = json.loads((tmp_path / "fake.json").read_text())
    meta["fold_hash"] = "000000000000"
    (tmp_path / "fake.json").write_text(json.dumps(meta))
    assert any(lvl == FAIL and "fold" in m for lvl, m in check_run(fake_run))


def test_dev_submission_roundtrip(fake_run, d):
    if not vsub.SCORER.exists():
        pytest.skip("scoring program not unpacked")
    z = vsub.make_dev_submission(fake_run, "variety+ukg")
    res = vsub.check_submission(z, common.DATA / "valid.csv", common.DATA / "valid.csv")
    assert not [m for lvl, m in res if lvl == "FAIL"], res
    pred, _ = policies.crossfit("variety+ukg", d)
    m = d.origin == "valid"
    ours = components(d.subset(m), {t: pred[t][m] for t in TASKS})["score"]
    official = [m for lvl, m in res if "official scorer ran" in m][0]
    assert f"{ours:.4f}" in official


def test_headline_policy_is_default_without_strong_evidence(d):
    from alta.validation.report import choose_policy
    rows = [{"policy": policies.DEFAULT_POLICY, "score": 0.80},
            {"policy": "prev+ukg", "score": 0.803}]  # gain below SWITCH_MIN_GAIN
    assert choose_policy(d, rows, B=50)[0] == policies.DEFAULT_POLICY


def test_neg_rule_not_in_defaults():
    assert all("neg" not in p for p in policies.DEFAULT_POLICIES)
    assert policies.DEFAULT_POLICY in policies.DEFAULT_POLICIES
