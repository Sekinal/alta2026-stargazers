"""Regression tests for the statistics red-team findings (redteam/stats/). Run: uv run pytest -q"""

from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
import pytest
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

import alta.common as common
import alta.validation.data as vdata
import alta.validation.submission as vsub
from alta.validation import harness
from alta.validation.checks import FAIL, WARN, check_run

HV = HashingVectorizer(n_features=2**12, alternate_sign=False, norm="l2", ngram_range=(1, 1))


def _fit(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    Xtr, Xte = HV.transform(train["text"].to_list()), HV.transform(test["text"].to_list())
    out = {}
    for t, c in (("sent", "sentiment"), ("sarc", "sarcasm")):
        clf = LogisticRegression(C=2.0, max_iter=300).fit(Xtr, train[c].to_numpy())
        out[t] = clf.predict_proba(Xte)[:, 1]
    return out


def honest():
    return harness.cv_fit_predict(_fit)[0]


def leaky_in_sample():
    """Classic leak outside the harness: fit on ALL pool labels, predict every row."""
    pool = common.load_pool()
    X = HV.transform(pool["text"].to_list())
    out = {"uid": pool["uid"]}
    for t, c in (("p_sent", "sentiment"), ("p_sarc", "sarcasm")):
        out[t] = LogisticRegression(C=2.0, max_iter=300).fit(X, pool[c].to_numpy()).predict_proba(X)[:, 1]
    return pl.DataFrame(out)


@pytest.fixture(scope="module")
def canaries():
    return {"honest": harness.canary(honest, seeds=2), "leaky": harness.canary(leaky_in_sample, seeds=2)}


def test_canary_passes_honest_pipeline(canaries):
    r = canaries["honest"]
    assert r["passed"], r
    assert abs(r["mean"] - r["floor"]) < 0.03


def test_canary_flags_leak_outside_the_harness(canaries):
    r = canaries["leaky"]
    assert not r["passed"], r


def test_canary_env_shuffles_only_labels_and_restores():
    base = common.load_pool()
    with harness.canary_labels(0):
        sh = common.load_pool()
    assert os.environ.get(common.CANARY_ENV) is None
    assert sh["uid"].to_list() == base["uid"].to_list() and sh["text"].to_list() == base["text"].to_list()
    assert not sh["sarcasm"].equals(base["sarcasm"])
    for (v, s), g in base.group_by("variety", "source"):  # per-segment label rates preserved
        g2 = sh.filter((pl.col("variety") == v) & (pl.col("source") == s))
        assert g["sarcasm"].sum() == g2["sarcasm"].sum() and g["sentiment"].sum() == g2["sentiment"].sum()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for mod in (common, vdata, harness):
        monkeypatch.setattr(mod, "PREDS", tmp_path)
    return tmp_path


def _probe_run(name, push_valid=0.0, push_all=0.0, **kw):
    pool = common.load_pool()
    rng = np.random.default_rng(0)
    out = {"uid": pool["uid"]}
    valid = (pool["origin"] == "valid").to_numpy()
    for t, c in (("p_sent", "sentiment"), ("p_sarc", "sarcasm")):
        y = pool[c].to_numpy()
        logit = 1.2 * (2 * y - 1) + rng.normal(0, 1.5, pool.height)
        logit = logit + push_all * (2 * y - 1) + push_valid * valid * (2 * y - 1)
        out[t] = 1 / (1 + np.exp(-logit))
    common.save_preds(name, pl.DataFrame(out), **kw)
    return name


def test_save_preds_refuses_other_folds(env):
    pool = common.load_pool()
    wrong = pool.select("uid", ((pl.col("fold") + 1) % 5).alias("fold"))
    with pytest.raises(ValueError, match="different fold"):
        _probe_run("r", folds_used=wrong)
    _probe_run("ok", folds_used=pool.select("uid", "fold"))
    assert json.loads((env / "ok.json").read_text())["folds_measured"] is True


def test_canary_run_cannot_shadow_real_run(env):
    with harness.canary_labels(1):
        _probe_run("r")
    assert (env / "r__canary1.parquet").exists() and not (env / "r.parquet").exists()


def test_valid_only_leak_warns(env):
    _probe_run("honest")
    _probe_run("leak", push_valid=1.5)
    msgs = lambda r: [m for lvl, m in check_run(r) if lvl == WARN and "valid-vs-pool" in m]
    assert not msgs("honest")
    assert msgs("leak")


def test_leak_score_check_uses_headline_policy(env):
    _probe_run("big", push_all=2.5)
    assert any(lvl == WARN and "implausibly high" in m for lvl, m in check_run("big"))


def test_failed_canary_fails_check(env):
    _probe_run("r")
    meta = json.loads((env / "r.json").read_text())
    meta["canary"] = {"passed": False, "mean": 0.7, "floor": 0.64, "margin": 0.02, "max_auc_z": 9}
    (env / "r.json").write_text(json.dumps(meta))
    assert any(lvl == FAIL and "canary" in m for lvl, m in check_run("r"))


def test_shift_diagnostic_flags_non_valid_like_test(env, tmp_path):
    v = pl.read_csv(common.DATA / "valid.csv")
    ukr = v.filter((pl.col("variety") == "en-UK") & (pl.col("source") == "reddit"))
    skewed = pl.concat([v, ukr, ukr]).with_columns(pl.lit(None, pl.Int64).alias("sentiment"),
                                                   pl.lit(None, pl.Int64).alias("sarcasm"))
    path = tmp_path / "skewed.csv"
    skewed.write_csv(path)
    t = common.load_test(path)
    rng = np.random.default_rng(0)
    _probe_run("r", test=pl.DataFrame({"uid": t["uid"], "p_sent": rng.random(t.height),
                                       "p_sarc": rng.random(t.height)}), test_csv=path)
    res = vsub.shift_diagnostics("r", path)
    assert any(lvl == "WARN" and "UK/reddit share" in m for lvl, m in res)
