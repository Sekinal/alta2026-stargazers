"""Regression tests for the software red-team findings (redteam/software/). Run: uv run pytest -q"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import polars as pl
import pytest

import alta.common as common
import alta.validation.data as vdata
import alta.validation.ensemble as vens
import alta.validation.report as vrep
import alta.validation.submission as vsub
from alta.validation import policies

MESSY = ['he said "no, thanks"\nthen left', "NA", "", "tab\there, comma, done", "emoji \U0001f600 ok",
         "line\r\nbreak", "None"]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated preds/submissions/results dirs."""
    for mod in (common, vdata, vens):
        monkeypatch.setattr(mod, "PREDS", tmp_path / "preds")
    (tmp_path / "preds").mkdir()
    monkeypatch.setattr(vsub, "SUBMISSIONS", tmp_path / "subs")
    monkeypatch.setattr(vrep, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(vrep, "LEDGER", tmp_path / "results" / "ledger.csv")
    return tmp_path


def _fake_test(tmp_path, seed=0, n=None, crlf_bom=True) -> tuple:
    """valid.csv shuffled, labels blank, messy texts injected; returns (path, gold frame in test order)."""
    v = pl.read_csv(common.DATA / "valid.csv").sample(fraction=1.0, shuffle=True, seed=seed)
    if n:
        v = v.head(n)
    texts = v["text"].to_list()
    for i, t in enumerate(MESSY):
        texts[i * 7] = t
    v = v.with_columns(pl.Series("text", texts))
    blank = v.with_columns(pl.lit(None, pl.Int64).alias("sentiment"), pl.lit(None, pl.Int64).alias("sarcasm"))
    path = tmp_path / f"test_{seed}_{n}.csv"
    body = blank.write_csv()
    if crlf_bom:
        body = "﻿" + body.replace("\n", "\r\n")
    path.write_bytes(body.encode())
    return path, v


def _save_run(name, test_csv, rng_seed=0):
    pool = common.load_pool()
    rng = np.random.default_rng(rng_seed)
    y = {t: pool[c].to_numpy() for t, c in (("sent", "sentiment"), ("sarc", "sarcasm"))}
    p = {t: np.clip(0.5 + 0.35 * (2 * y[t] - 1) + rng.normal(0, 0.25, pool.height), 0, 1) for t in y}
    oof = pl.DataFrame({"uid": pool["uid"], "p_sent": p["sent"], "p_sarc": p["sarc"]})
    test = None
    if test_csv is not None:
        t = common.load_test(test_csv)
        test = pl.DataFrame({"uid": t["uid"], "p_sent": rng.random(t.height), "p_sarc": rng.random(t.height)})
    common.save_preds(name, oof, test, test_csv=test_csv)
    return name


# 1 ---------------------------------------------------------------- test split bound to its file
def test_test_split_bound_to_file(env):
    path, _ = _fake_test(env, seed=1)
    _save_run("r", path)
    other, _ = _fake_test(env, seed=2)           # same rows, different order
    shorter, _ = _fake_test(env, seed=1, n=500)
    for f in (other, shorter):
        with pytest.raises(ValueError, match="different test file|uids"):
            vsub.make_test_submission("r", "segment+ukg", f)


def test_save_preds_requires_test_csv(env):
    with pytest.raises(ValueError, match="test_csv"):
        pool = common.load_pool()
        oof = pl.DataFrame({"uid": pool["uid"], "p_sent": 0.5, "p_sarc": 0.5})
        common.save_preds("x", oof, pl.DataFrame({"uid": ["x00000"], "p_sent": [0.5], "p_sarc": [0.5]}))


# 2 ---------------------------------------------------------------- submit refuses broken runs
@pytest.mark.parametrize("breakage", ["nan_test", "nan_oof", "dup_oof", "dup_test", "stale_folds"])
def test_submit_refuses_failing_run(env, breakage):
    path, _ = _fake_test(env)
    _save_run("r", path)
    f = env / "preds" / "r.parquet"
    df = pl.read_parquet(f)
    meta = json.loads((env / "preds" / "r.json").read_text())
    if breakage == "nan_test":
        df = df.with_columns(pl.when((pl.col("split") == "test") & (pl.int_range(pl.len()) % 50 == 0))
                             .then(float("nan")).otherwise(pl.col("p_sarc")).alias("p_sarc"))
    elif breakage == "nan_oof":
        df = df.with_columns(pl.when(pl.int_range(pl.len()) == 3).then(float("nan"))
                             .otherwise(pl.col("p_sent")).alias("p_sent"))
    elif breakage == "dup_oof":
        df = pl.concat([df, df.filter(pl.col("split") == "oof").head(10)])
    elif breakage == "dup_test":
        df = pl.concat([df, df.filter(pl.col("split") == "test").head(3)])
    else:
        meta["fold_hash"] = "deadbeef0000"
    df.write_parquet(f)
    (env / "preds" / "r.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        vsub.make_test_submission("r", "segment+ukg", path)
    with pytest.raises(ValueError):
        vsub.make_dev_submission("r", "segment+ukg")


# 3 ---------------------------------------------------------------- unknown source/variety
@pytest.mark.parametrize("col,val", [("source", "Reddit"), ("variety", " en-UK"), ("variety", "en-GB"), ("source", None)])
def test_unknown_variety_or_source_rejected(tmp_path, col, val):
    v = pl.read_csv(common.DATA / "valid.csv").head(20)
    v = v.with_columns(pl.when(pl.int_range(pl.len()) == 5).then(pl.lit(val)).otherwise(pl.col(col)).alias(col))
    path = tmp_path / "t.csv"
    v.write_csv(path)
    with pytest.raises(ValueError):
        common.load_test(path)


# 4/5 ---------------------------------------------------------------- every policy, exact row order
@pytest.mark.parametrize("spec", policies.DEFAULT_POLICIES)
def test_test_submission_every_policy(env, spec):
    path, _ = _fake_test(env)
    _save_run("r", path)
    z = vsub.make_test_submission("r", spec, path)
    res = vsub.check_submission(z, path)
    assert not [m for lvl, m in res if lvl == "FAIL"], res


def test_test_submission_row_order_and_official_score(env):
    """Test probabilities = the dev OOF for the same rows: the shuffled test answer must score
    exactly like our metric on those rows, via the official scorer on a messy file."""
    if not vsub.SCORER.exists():
        pytest.skip("scoring program not unpacked")
    path, gold = _fake_test(env, seed=3)
    pool = common.load_pool()
    rng = np.random.default_rng(0)
    y = {t: pool[c].to_numpy() for t, c in (("sent", "sentiment"), ("sarc", "sarcasm"))}
    p = {t: np.clip(0.5 + 0.3 * (2 * y[t] - 1) + rng.normal(0, 0.3, pool.height), 0, 1) for t in y}
    oof = pl.DataFrame({"uid": pool["uid"], "p_sent": p["sent"], "p_sarc": p["sarc"]})
    # map each test row back to its valid.csv row via the untouched rows' text
    t = common.load_test(path)
    valid = pool.filter(pl.col("origin") == "valid").select("uid", "text", "source", "variety")
    orig = pl.read_csv(common.DATA / "valid.csv").with_row_index("r")
    key = gold.join(orig.select("r", "text", "source", "variety"), on=["text", "source", "variety"], how="left",
                    maintain_order="left")
    ok = key["r"].is_not_null().to_numpy() & np.array([tx not in MESSY for tx in gold["text"].to_list()])
    vuid = np.array([f"v{int(r):05d}" if r is not None else "" for r in key["r"].to_list()])
    pmap = dict(zip(oof["uid"].to_list(), zip(p["sent"], p["sarc"])))
    tp = np.array([pmap.get(u, (0.5, 0.5)) for u in vuid])
    test = pl.DataFrame({"uid": t["uid"], "p_sent": tp[:, 0], "p_sarc": tp[:, 1]})
    common.save_preds("r", oof, test, test_csv=path)
    z = vsub.make_test_submission("r", "fixed+ukg", path)
    truth = env / "truth.csv"
    gold.with_columns(pl.col("sentiment").fill_null(0), pl.col("sarcasm").fill_null(0)).write_csv(truth)
    res = vsub.check_submission(z, path, truth)
    assert not [m for lvl, m in res if lvl == "FAIL"], res
    ans = pl.read_csv(env / "subs" / "test_r_fixed+ukg" / "answer.csv")
    assert ans["text"].to_list() == t["text"].to_list()
    exp_sent = (tp[:, 0] >= 0.5).astype(int)
    assert (ans["sentiment"].to_numpy()[ok] == exp_sent[ok]).all()
    _ = valid


# 6 ---------------------------------------------------------------- CSV edge cases
def test_load_test_blank_lines_and_cr_only(tmp_path):
    v = pl.read_csv(common.DATA / "valid.csv").head(10)
    body = v.write_csv()
    lines = body.splitlines()
    (tmp_path / "blank.csv").write_text("\n".join(lines[:4] + [""] + lines[4:]) + "\n\n")
    with pytest.warns(UserWarning, match="blank"):
        assert common.load_test(tmp_path / "blank.csv").height == 10
    (tmp_path / "cr.csv").write_bytes(body.replace("\n", "\r").encode())
    with pytest.raises(ValueError):
        common.load_test(tmp_path / "cr.csv")
    (tmp_path / "empty.csv").write_text("source,variety,text,sentiment,sarcasm\n")
    with pytest.raises(ValueError):
        common.load_test(tmp_path / "empty.csv")


# 7/8 ---------------------------------------------------------------- stack / blend
def test_stack_uses_bound_test_file_and_rejects_mismatch(env):
    path, _ = _fake_test(env)
    _save_run("a", path, 1)
    _save_run("b", path, 2)
    vens.stack(["a", "b"], "s")
    meta = json.loads((env / "preds" / "s.json").read_text())
    assert meta["test_csv"] == str(path)
    other, _ = _fake_test(env, seed=9)
    _save_run("c", other, 3)
    with pytest.raises(ValueError, match="different test files"):
        vens.stack(["a", "c"], "s2")


def test_blend_refuses_member_name_existing_run_and_stale_folds(env):
    _save_run("a", None, 1)
    _save_run("b", None, 2)
    with pytest.raises(ValueError, match="members"):
        vens.blend(["a", "a", "b"], "a")
    vens.blend(["a", "b"], "ab")
    with pytest.raises(ValueError, match="exists"):
        vens.blend(["a", "b"], "ab")
    meta = json.loads((env / "preds" / "b.json").read_text())
    meta["fold_hash"] = "000000000000"
    (env / "preds" / "b.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fold_hash"):
        vens.blend(["a", "b"], "ab2")


# 9 ---------------------------------------------------------------- ledger types
def test_ledger_roundtrip_numeric_like_code(env, monkeypatch):
    rows = [{"policy": policies.DEFAULT_POLICY, "score": 0.8, "sent-AU": 0.9, "sent-UK": 0.9,
             "sarc-AU": 0.7, "sarc-UK": 0.7, "fold_sd": 0.01, "valid_score": 0.8}]
    reps = {"score": np.array([0.8, 0.81])}
    codes = ["5e12345"] + ["1234567"] * 120 + ["abc1234-dirty"]
    for c in codes:
        monkeypatch.setattr(vrep, "git_commit", lambda c=c: c)
        vrep.append_ledger("r", rows, reps, policies.DEFAULT_POLICY)
    led = vrep.read_ledger()
    assert led["code"].to_list() == codes
    assert "abc1234-dirty" in vrep.board()


# 10 ---------------------------------------------------------------- CLI exit codes
def test_report_cli_nonzero_on_fail(env):
    _save_run("r", None)
    f = env / "preds" / "r.parquet"
    pl.read_parquet(f).head(100).write_parquet(f)
    from alta.validate import main
    with pytest.raises(SystemExit) as e:
        main(["report", "r", "--no-write"])
    assert e.value.code == 1


def test_cli_rejects_small_B():
    with pytest.raises(SystemExit) as e:
        from alta.validate import main
        main(["report", "whatever", "--B", "1"])
    assert e.value.code == 2


# 11 ---------------------------------------------------------------- pool cache invalidation
def test_pool_cache_invalidated_on_csv_change(tmp_path, monkeypatch):
    for f in ("train.csv", "valid.csv"):
        (tmp_path / f).write_bytes((common.DATA / f).read_bytes())
    monkeypatch.setattr(common, "DATA", tmp_path)
    monkeypatch.setattr(common, "CACHE", tmp_path / "pool.parquet")
    a = common.load_pool()
    tr = pl.read_csv(tmp_path / "train.csv")
    tr.with_columns((1 - pl.col("sarcasm")).alias("sarcasm")).write_csv(tmp_path / "train.csv")
    b = common.load_pool()
    assert a["sarcasm"].sum() != b["sarcasm"].sum()


# 12 ---------------------------------------------------------------- optional: real Codabench image
def test_scorer_docker_equivalence(env):
    try:
        ok = subprocess.run(["docker", "image", "inspect", "codalab/codalab-legacy:py37"],
                            capture_output=True).returncode == 0
    except OSError:
        ok = False
    if not ok or not vsub.SCORER.exists():
        pytest.skip("codalab-legacy:py37 image not available")
    _save_run("r", None)
    z = vsub.make_dev_submission("r", "segment+ukg")
    ours = [m for lvl, m in vsub.check_submission(z, common.DATA / "valid.csv", common.DATA / "valid.csv")
            if "official scorer ran" in m][0]
    d = env / "docker"
    (d / "in" / "res").mkdir(parents=True)
    (d / "in" / "ref").mkdir(parents=True)
    (d / "in" / "res" / "answer.csv").write_bytes((env / "subs" / "dev_r_segment+ukg" / "answer.csv").read_bytes())
    (d / "in" / "ref" / "truth.csv").write_bytes((common.DATA / "valid.csv").read_bytes())
    (d / "prog").mkdir()
    (d / "prog" / "evaluate.py").write_bytes(vsub.SCORER.read_bytes())
    import os
    subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{d}:/w",
                    "codalab/codalab-legacy:py37",
                    "python", "/w/prog/evaluate.py", "/w/in", "/w/out"], check=True, capture_output=True)
    score = [l for l in (d / "out" / "scores.txt").read_text().splitlines() if l.startswith("score")][0]
    assert score.split(":")[1].strip() in ours
