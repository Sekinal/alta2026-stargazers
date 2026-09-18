"""Attacks 3-4: mis-joins and degenerate inputs on the submit path (in-process).

Every scenario builds a fake run preds/rt_sw_<name> (deleted at the end), calls the real code
and reports CRASH (exception), SILENT (ran, but the answer is provably wrong / garbage) or OK.
Submissions go to a temp dir (submission.SUBMISSIONS is redirected).

Usage: PYTHONPATH=. uv run python redteam/software/attacks.py [scenario ...]
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import polars as pl

import alta.validation.submission as vsub
from alta.common import DATA, PREDS, load_pool, load_test, save_preds
from alta.validation import policies
from alta.validation.checks import FAIL, check_run
from alta.validation.data import load_oof, load_test_preds

HERE = Path(__file__).parent
FAKE = HERE / "fake"
TMP = Path(tempfile.mkdtemp(prefix="rt_sw_"))
vsub.SUBMISSIONS = TMP / "subs"
POOL = load_pool()
BASE = pl.read_parquet(PREDS / "emb_ens3_lr.parquet").filter(pl.col("split") == "oof").select(
    "uid", "p_sent", "p_sarc")
FAKE_TEST = FAKE / "test.csv"
RESULTS: list[tuple[str, str, str]] = []


def test_preds_for(test_csv: Path) -> pl.DataFrame:
    """Honest test preds for a fake test built by make_fake (by text lookup in valid)."""
    t = load_test(test_csv)
    v = POOL.filter(pl.col("origin") == "valid").select("uid", "text").join(BASE, on="uid")
    # texts were mutated for some rows by make_fake; fall back to 0.5 there
    return t.select("uid", "text").join(v.drop("uid"), on="text", how="left", maintain_order="left").unique(
        "uid", keep="first", maintain_order=True).select(
        "uid", pl.col("p_sent").fill_null(0.5), pl.col("p_sarc").fill_null(0.5))


def answer(z: Path) -> pl.DataFrame:
    return pl.read_csv(z.parent / "answer.csv")


def run(name: str):
    def deco(fn):
        def wrapped():
            try:
                verdict, msg = fn()
            except Exception as e:  # noqa: BLE001
                verdict, msg = "CRASH", f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                if "-v" in sys.argv:
                    traceback.print_exc()
            RESULTS.append((name, verdict, msg))
            print(f"[{verdict:6}] {name}: {msg}", flush=True)
        wrapped.name = name
        return wrapped
    return deco


def fails(res):
    return [m for lvl, m in res if lvl == FAIL]


# --------------------------------------------------------------------------- mis-joins
@run("stale_test_split_other_order")
def a_stale():
    """Run's test split made for test order A; submitted against a same-size test file in order B."""
    other = TMP / "other"
    import subprocess
    subprocess.run([sys.executable, str(HERE / "make_fake.py"), "--order-seed", "1", "--run",
                    "rt_sw_tmp_other", "--out", str(other)], check=True, capture_output=True)
    for p in PREDS.glob("rt_sw_tmp_other.*"):
        p.unlink()
    save_preds("rt_sw_stale", BASE, test_preds_for(FAKE_TEST))  # made for seed-0 order
    z = vsub.make_test_submission("rt_sw_stale", "fixed+ukg", other / "test.csv")
    res = vsub.check_submission(z, other / "test.csv", other / "truth.csv")
    good = vsub.check_submission(
        vsub.make_test_submission("rt_sw_stale", "fixed+ukg", FAKE_TEST, name="good"), FAKE_TEST,
        FAKE / "truth.csv")
    s_bad = [m for _, m in res if "official" in m][0]
    s_good = [m for _, m in good if "official" in m][0]
    return ("SILENT" if not fails(res) else "OK",
            f"no FAIL; {s_bad.split('{')[0].strip()} vs right file {s_good.split('{')[0].strip()}")


@run("test_split_longer_than_test")
def a_longer():
    tp = test_preds_for(FAKE_TEST)
    t2 = TMP / "short_test.csv"
    load_test(FAKE_TEST).head(500).select("source", "variety", "text", "sentiment", "sarcasm").write_csv(t2)
    save_preds("rt_sw_longer", BASE, tp)  # 757 test preds
    z = vsub.make_test_submission("rt_sw_longer", "fixed+ukg", t2)
    res = vsub.check_submission(z, t2)
    return ("SILENT" if not fails(res) else "OK",
            f"757-row test split silently truncated to a 500-row test file, FAILs={fails(res)}")


@run("duplicate_test_uid")
def a_dup_test():
    tp = test_preds_for(FAKE_TEST)
    tp = pl.concat([tp, tp.head(3).with_columns(pl.lit(0.99).alias("p_sent"))])
    save_preds("rt_sw_duptest", BASE, tp)
    lv = [(l, m) for l, m in check_run("rt_sw_duptest") if l != "OK"]
    z = vsub.make_test_submission("rt_sw_duptest", "fixed+ukg", FAKE_TEST)
    return "SILENT", f"check_run non-OK={lv}; submission built: {z.name}"


@run("nan_in_test_probs")
def a_nan_test():
    tp = test_preds_for(FAKE_TEST).with_columns(
        pl.when(pl.int_range(pl.len()) < 200).then(float("nan")).otherwise(pl.col("p_sarc")).alias("p_sarc"))
    save_preds("rt_sw_nantest", BASE, tp)
    z = vsub.make_test_submission("rt_sw_nantest", "fixed+ukg", FAKE_TEST)
    res = vsub.check_submission(z, FAKE_TEST, FAKE / "truth.csv")
    a = answer(z)
    return ("SILENT" if not fails(res) else "OK",
            f"200 NaN p_sarc -> answer built, FAILs={fails(res)}, sarcasm=1 in first 200 rows: "
            f"{a.head(200)['sarcasm'].sum()}; check_run FAILs={fails(check_run('rt_sw_nantest'))}")


@run("nan_in_test_probs_prev_policy")
def a_nan_prev():
    tp = test_preds_for(FAKE_TEST).with_columns(
        pl.when(pl.int_range(pl.len()) % 3 == 0).then(float("nan")).otherwise(pl.col("p_sarc")).alias("p_sarc"))
    save_preds("rt_sw_nanprev", BASE, tp)
    z = vsub.make_test_submission("rt_sw_nanprev", "prev+ukg", FAKE_TEST)
    a = answer(z)
    nan_rows = a.with_row_index().filter(pl.col("index") % 3 == 0)
    return ("SILENT" if nan_rows["sarcasm"].sum() else "INFO",
            f"prev policy: {nan_rows['sarcasm'].sum()} of {nan_rows.height} NaN-prob rows labelled sarcastic "
            "(argsort(-nan) ordering)")


@run("nan_in_oof_submit")
def a_nan_oof():
    oof = BASE.with_columns(pl.when(pl.int_range(pl.len()) < 50).then(float("nan"))
                            .otherwise(pl.col("p_sent")).alias("p_sent"))
    save_preds("rt_sw_nanoof", oof, test_preds_for(FAKE_TEST))
    out = {}
    for spec in ("variety+ukg", "global+ukg", "segment+ukg", "prevtuned+ukg"):
        params = policies.fit(spec, load_oof("rt_sw_nanoof"))
        out[spec] = {str(k): v for k, v in params["sent"].items()} if isinstance(params["sent"], dict) else params["sent"]
    z = vsub.make_test_submission("rt_sw_nanoof", "variety+ukg", FAKE_TEST)
    a = answer(z)
    return ("SILENT", f"fitted sent thresholds with 50 NaN OOF probs: {json.dumps(out, default=str)[:300]}; "
                      f"answer sentiment rate {a['sentiment'].mean():.3f}")


@run("dup_oof_uid_submit")
def a_dup_oof():
    oof = pl.concat([BASE, BASE.head(2000)])
    save_preds("rt_sw_dupoof", oof, test_preds_for(FAKE_TEST))
    d = load_oof("rt_sw_dupoof")
    z = vsub.make_test_submission("rt_sw_dupoof", "variety+ukg", FAKE_TEST)
    return "SILENT", f"load_oof returned {len(d)} rows for a {POOL.height}-row pool; test submission built"


@run("stale_fold_hash_submit")
def a_stale_hash():
    save_preds("rt_sw_hash", BASE, test_preds_for(FAKE_TEST))
    mp = PREDS / "rt_sw_hash.json"
    meta = json.loads(mp.read_text()); meta["fold_hash"] = "deadbeef0000"; mp.write_text(json.dumps(meta))
    z = vsub.make_test_submission("rt_sw_hash", "variety+ukg", FAKE_TEST)
    return "SILENT", f"check_run FAILs {fails(check_run('rt_sw_hash'))} but submit built {z.name} anyway"


# --------------------------------------------------------------------------- test file content
def _write_test(frame: pl.DataFrame, name: str) -> Path:
    p = TMP / f"{name}.csv"
    frame.select("source", "variety", "text", "sentiment", "sarcasm").write_csv(p)
    return p


@run("variety_source_case_whitespace")
def a_case():
    t = load_test(FAKE_TEST)
    t2 = t.with_columns(pl.when(pl.col("source") == "reddit").then(pl.lit("Reddit")).otherwise(pl.col("source"))
                        .alias("source"),
                        pl.when(pl.col("variety") == "en-UK").then(pl.lit("en-UK ")).otherwise(pl.col("variety"))
                        .alias("variety"))
    p = _write_test(t2, "case")
    save_preds("rt_sw_case", BASE, test_preds_for(FAKE_TEST))
    z = vsub.make_test_submission("rt_sw_case", "variety+ukg", p)
    res = vsub.check_submission(z, p)
    a = answer(z)
    return ("SILENT" if not fails(res) else "OK",
            f"'Reddit'/'en-UK ' coded as google/AU: all UK sarcasm={a.filter(pl.col('variety')=='en-UK ')['sarcasm'].sum()}, "
            f"FAILs={fails(res)}")


@run("blank_line_in_test")
def a_blank():
    raw = FAKE_TEST.read_bytes()
    lines = raw.split(b"\r\n")
    p = TMP / "blank.csv"
    p.write_bytes(b"\r\n".join(lines[:5] + [b""] + lines[5:]) + b"\r\n\r\n")
    save_preds("rt_sw_blank", BASE, test_preds_for(p))
    t = load_test(p)
    z = vsub.make_test_submission("rt_sw_blank", "prev+ukg", p)
    res = vsub.check_submission(z, p)
    return ("SILENT" if not fails(res) else "OK",
            f"load_test rows {t.height} (null-variety rows {t['variety'].null_count()}), answer built, FAILs={fails(res)}")


@run("empty_test")
def a_empty():
    p = TMP / "empty.csv"
    p.write_text("source,variety,text,sentiment,sarcasm\n")
    save_preds("rt_sw_empty", BASE, test_preds_for(FAKE_TEST))
    z = vsub.make_test_submission("rt_sw_empty", "prev+ukg", p)
    res = vsub.check_submission(z, p)
    return ("OK" if fails(res) else "SILENT", f"built empty answer; check: {[(l, m[:80]) for l, m in res][:3]}")


@run("test_without_uk_google")
def a_no_ukg():
    t = load_test(FAKE_TEST).filter(~((pl.col("variety") == "en-UK") & (pl.col("source") == "google")))
    p = _write_test(t, "noukg")
    save_preds("rt_sw_noukg", BASE, test_preds_for(p))
    out = {}
    for spec in ("prev+ukg", "prevtuned+ukg"):
        z = vsub.make_test_submission("rt_sw_noukg", spec, p, name=spec)
        out[spec] = answer(z).height
    return "OK", f"0-row segment handled: {out}"


# --------------------------------------------------------------------------- degenerate probabilities
def _degenerate(name, fn):
    oof = BASE.with_columns(fn(pl.col("p_sent")).alias("p_sent"), fn(pl.col("p_sarc")).alias("p_sarc"))
    tp = test_preds_for(FAKE_TEST)
    tp = tp.with_columns(fn(pl.col("p_sent")).alias("p_sent"), fn(pl.col("p_sarc")).alias("p_sarc"))
    save_preds(f"rt_sw_{name}", oof, tp)
    out = {}
    d = load_oof(f"rt_sw_{name}")
    for spec in policies.DEFAULT_POLICIES:
        try:
            pred, _ = policies.crossfit(spec, d)
            z = vsub.make_test_submission(f"rt_sw_{name}", spec, FAKE_TEST, name=f"{name}_{spec}")
            a = answer(z)
            out[spec] = f"{a['sentiment'].mean():.2f}/{a['sarcasm'].mean():.2f}"
        except Exception as e:  # noqa: BLE001
            out[spec] = f"CRASH {type(e).__name__}: {str(e)[:60]}"
    lv = sorted({l for l, _ in check_run(f"rt_sw_{name}")})
    crash = any(v.startswith("CRASH") for v in out.values())
    return ("CRASH" if crash else "INFO", f"check_run levels {lv}; test pos rates sent/sarc {out}")


@run("all_constant_0.5")
def a_const():
    return _degenerate("const", lambda c: c * 0 + 0.5)


@run("all_zero")
def a_zero():
    return _degenerate("zero", lambda c: c * 0)


@run("all_one")
def a_one():
    return _degenerate("one", lambda c: c * 0 + 1)


@run("hard_labels_0_1")
def a_hard():
    return _degenerate("hard", lambda c: (c >= 0.5).cast(pl.Float64))


@run("segment_policy_test_submit")
def a_segment():
    save_preds("rt_sw_seg", BASE, test_preds_for(FAKE_TEST))
    z = vsub.make_test_submission("rt_sw_seg", "segment+ukg", FAKE_TEST)
    return "OK", str(z)


@run("source_capitalised")
def a_source_case():
    t = load_test(FAKE_TEST).with_columns(pl.col("source").str.to_titlecase())  # Google / Reddit
    p = _write_test(t, "srccase")
    save_preds("rt_sw_src", BASE, test_preds_for(FAKE_TEST))
    z = vsub.make_test_submission("rt_sw_src", "fixed+ukg", p)
    truth = pl.read_csv(FAKE / "truth.csv").with_columns(pl.col("source").str.to_titlecase())
    truth.write_csv(TMP / "srccase_truth.csv")
    res = vsub.check_submission(z, p, TMP / "srccase_truth.csv")
    a = answer(z)
    s = [m for _, m in res if "official" in m][0].split("{")[0]
    return ("SILENT" if not fails(res) else "OK",
            f"source 'Reddit' -> coded google, ukg zeroes UK-reddit sarcasm: UK sarcasm positives="
            f"{a.filter(pl.col('variety') == 'en-UK')['sarcasm'].sum()}; {s}; FAILs={fails(res)}")


SCENARIOS = [a_source_case, a_segment, a_stale, a_longer, a_dup_test, a_nan_test, a_nan_prev, a_nan_oof, a_dup_oof, a_stale_hash,
             a_case, a_blank, a_empty, a_no_ukg, a_const, a_zero, a_one, a_hard]

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    try:
        for s in SCENARIOS:
            if not want or s.name in want:
                s()
    finally:
        for p in PREDS.glob("rt_sw_*"):
            if p.stem != "rt_sw_e2e":
                p.unlink()
        shutil.rmtree(TMP, ignore_errors=True)
