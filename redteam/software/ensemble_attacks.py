"""Attack 3b: blend/stack mis-joins and provenance.

Temporarily places the fake test at data/test.csv for the stack scenarios and removes it
afterwards (there is no data/test.csv in the real tree). All runs are rt_sw_* and deleted.

Usage: PYTHONPATH=. uv run python redteam/software/ensemble_attacks.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from attacks import BASE, FAKE, FAKE_TEST, fails, run, test_preds_for  # noqa: E402

from alta.common import DATA, PREDS, load_test, save_preds  # noqa: E402
from alta.validation import ensemble  # noqa: E402
from alta.validation.checks import check_run  # noqa: E402
from alta.validation.data import load_test_preds, read_run  # noqa: E402

TEST_CSV = DATA / "test.csv"
assert not TEST_CSV.exists(), "refusing to run: data/test.csv exists"


def member(name, oof=BASE, tp=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda d: d.with_columns(pl.col("p_sent", "p_sarc").map_batches(  # noqa: E731
        lambda s: pl.Series(np.clip(s.to_numpy() + rng.normal(0, noise, len(s)), 0, 1))))
    save_preds(name, f(oof), None if tp is None else f(tp))
    return name


@run("blend_overwrites_member")
def e_overwrite():
    member("rt_sw_m1", tp=test_preds_for(FAKE_TEST), noise=0.05, seed=1)
    before = read_run("rt_sw_m1")[0]["p_sent"].to_numpy().copy()
    member("rt_sw_m2", tp=test_preds_for(FAKE_TEST), noise=0.2, seed=2)
    ensemble.blend(["rt_sw_m1", "rt_sw_m2"], "rt_sw_m1")
    after, meta = read_run("rt_sw_m1")
    return ("SILENT", f"member rt_sw_m1 replaced by the blend without warning "
                      f"(max |dp|={np.abs(after['p_sent'].to_numpy() - before).max():.3f}, meta kind={meta.get('kind')})")


@run("blend_launders_fold_hash")
def e_launder():
    member("rt_sw_old", noise=0.05, seed=3)
    mp = PREDS / "rt_sw_old.json"
    m = json.loads(mp.read_text()); m["fold_hash"] = "000000000000"; mp.write_text(json.dumps(m))
    member("rt_sw_new", noise=0.05, seed=4)
    ensemble.blend(["rt_sw_old", "rt_sw_new"], "rt_sw_bl")
    return ("SILENT", f"member FAILs {fails(check_run('rt_sw_old'))[:1]}; blend FAILs {fails(check_run('rt_sw_bl'))}")


@run("stack_without_data_test_csv")
def e_stack_notest():
    member("rt_sw_s1", tp=test_preds_for(FAKE_TEST), noise=0.05, seed=5)
    member("rt_sw_s2", tp=test_preds_for(FAKE_TEST), noise=0.1, seed=6)
    ensemble.stack(["rt_sw_s1", "rt_sw_s2"], "rt_sw_st")
    return "OK", "stacked"


@run("stack_test_segments_from_other_file")
def e_stack_other():
    """Members made for fake/test.csv (order A); data/test.csv is a same-size file in order B."""
    import subprocess
    other = Path(__file__).parent / "fake_other"
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_fake.py"), "--order-seed", "1",
                    "--run", "rt_sw_tmp", "--out", str(other)], check=True, capture_output=True)
    shutil.copy(other / "test.csv", TEST_CSV)
    try:
        member("rt_sw_s1", tp=test_preds_for(FAKE_TEST), noise=0.05, seed=5)
        member("rt_sw_s2", tp=test_preds_for(FAKE_TEST), noise=0.1, seed=6)
        ensemble.stack(["rt_sw_s1", "rt_sw_s2"], "rt_sw_st")
        d = load_test_preds("rt_sw_st", load_test(FAKE_TEST))
        d_ok = None
        shutil.copy(FAKE_TEST, TEST_CSV)
        ensemble.stack(["rt_sw_s1", "rt_sw_s2"], "rt_sw_st_ok")
        d_ok = load_test_preds("rt_sw_st_ok", load_test(FAKE_TEST))
        diff = np.abs(d.p["sarc"] - d_ok.p["sarc"])
        return ("SILENT", f"stack used data/test.csv for variety/source: {int((diff > 0.05).sum())} of {len(diff)} "
                          f"test p_sarc differ by >0.05 from the stack built against the right file")
    finally:
        TEST_CSV.unlink(missing_ok=True)
        shutil.rmtree(other, ignore_errors=True)


@run("stack_members_test_for_longer_file")
def e_stack_extra():
    """Members' test split has uids beyond data/test.csv -> null segments in stack."""
    t = load_test(FAKE_TEST).head(500).select("source", "variety", "text", "sentiment", "sarcasm")
    t.write_csv(TEST_CSV)
    try:
        member("rt_sw_s1", tp=test_preds_for(FAKE_TEST), noise=0.05, seed=5)
        member("rt_sw_s2", tp=test_preds_for(FAKE_TEST), noise=0.1, seed=6)
        ensemble.stack(["rt_sw_s1", "rt_sw_s2"], "rt_sw_st")
        tp = read_run("rt_sw_st")[0].filter(pl.col("split") == "test")
        return ("SILENT", f"stack test split has {tp.height} rows; rows beyond data/test.csv: "
                          f"p_sent==0 exactly in {int((tp['p_sent'] == 0).sum())} rows")
    finally:
        TEST_CSV.unlink(missing_ok=True)


@run("polars_left_join_order_without_maintain_order")
def e_join_order():
    """ensemble.stack aligns test_seg to test by a left join WITHOUT maintain_order."""
    for n in (757, 50_000, 500_000):
        uid = pl.Series("uid", [f"x{i:07d}" for i in np.random.default_rng(0).permutation(n)])
        left = pl.DataFrame({"uid": uid})
        right = pl.DataFrame({"uid": uid.sort(), "v": np.arange(n)})
        j = left.join(right, on="uid", how="left")
        if not (j["uid"] == uid).all():
            return "SILENT", f"left-join order changed at n={n}"
    return "INFO", "order preserved up to n=500k in polars 1.44 (undocumented; latent)"


if __name__ == "__main__":
    try:
        for s in (e_overwrite, e_launder, e_stack_notest, e_stack_other, e_stack_extra, e_join_order):
            s()
    finally:
        TEST_CSV.unlink(missing_ok=True)
        for p in PREDS.glob("rt_sw_*"):
            if p.stem != "rt_sw_e2e":
                p.unlink()
