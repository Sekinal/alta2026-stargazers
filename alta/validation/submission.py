"""Build and verify Codabench submissions (zip containing answer.csv at its root).

Two paths:
* ``make_dev_submission``: public-test (valid.csv) answer from the run's honest cross-fitted
  OOF labels. valid rows are in the CV pool, so fold models must never predict them in-sample.
* ``make_test_submission``: final test answer from the run's ``test`` split, with the policy
  fitted on the full OOF pool.

``check_submission`` validates the zip against the test CSV and runs the official scorer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from alta.common import DATA, ROOT, frame_fingerprint, load_pool, load_test
from alta.validation import policies
from alta.validation.data import load_oof, load_test_preds

SUBMISSIONS = ROOT / "submissions"
COLUMNS = ["source", "variety", "text", "sentiment", "sarcasm"]
SCORER = ROOT / "scoring" / "evaluate.py"


def _write(frame: pl.DataFrame, pred: dict, name: str, info: dict) -> Path:
    out_dir = SUBMISSIONS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ans = frame.select("source", "variety", "text").with_columns(
        pl.Series("sentiment", pred["sent"].astype(np.int64)),
        pl.Series("sarcasm", pred["sarc"].astype(np.int64)),
    ).with_columns(pl.col("source", "variety").cast(pl.String))
    csv = out_dir / "answer.csv"
    ans.write_csv(csv)
    with zipfile.ZipFile(out_dir / "answer.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, arcname="answer.csv")
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    return out_dir / "answer.zip"


def _jsonable(o):
    """Policy params -> JSON: tuple keys (variety, source) become 'v|s', numpy scalars -> Python."""
    if isinstance(o, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _require_clean(run: str, pool) -> None:
    from alta.validation.checks import FAIL, check_run
    fails = [m for level, m in check_run(run, pool) if level == FAIL]
    if fails:
        raise ValueError(f"run '{run}' failed integrity checks, refusing to build a submission: {fails}")


def make_dev_submission(run: str, spec: str, name: str | None = None) -> Path:
    pool = load_pool()
    _require_clean(run, pool)
    d = load_oof(run, pool)
    pred, _ = policies.crossfit(spec, d)
    valid = d.origin == "valid"
    order = np.argsort(d.uid[valid])  # v00000, v00001, ... = valid.csv order
    frame = pool.filter(pl.col("origin") == "valid").sort("uid")
    if not (frame["uid"].to_numpy() == d.uid[valid][order]).all():
        raise RuntimeError("valid rows misaligned between pool and OOF")
    vp = {t: pred[t][valid][order] for t in pred}
    return _write(frame, vp, name or f"dev_{run}_{spec}",
                  {"run": run, "policy": spec, "phase": "dev", "source": "crossfit OOF"})


def make_test_submission(run: str, spec: str, test_csv: Path, name: str | None = None) -> Path:
    pool = load_pool()
    _require_clean(run, pool)
    test = load_test(test_csv)
    params = policies.fit(spec, load_oof(run, pool))
    pred = policies.apply(params, load_test_preds(run, test))
    return _write(test, pred, name or f"test_{run}_{spec}",
                  {"run": run, "policy": spec, "phase": "test", "test_csv": str(test_csv),
                   "test_sha": frame_fingerprint(test),
                   "params": _jsonable(params)})


def shift_diagnostics(run: str, test_csv: Path) -> list[tuple[str, str]]:
    """Is the test set valid-like? Prevalence policies force predicted rates to the pool
    prevalence, so the answer's rates can't reveal a shift; use segment+ukg's rates instead."""
    out: list[tuple[str, str]] = []
    pool = load_pool()
    test = load_test(test_csv)
    valid = pool.filter(pl.col("origin") == "valid")
    if test.height < 400:
        out.append(("WARN", f"test has {test.height} rows (< 400): prevalence policies get noisy; prefer segment+ukg"))
    seg = lambda f: f.select((pl.col("variety").cast(pl.String) + "/" + pl.col("source").cast(pl.String)).alias("seg"))
    vshare = seg(valid).group_by("seg").len().with_columns((pl.col("len") / valid.height).alias("p"))
    tcount = seg(test).group_by("seg").len()
    for r in vshare.join(tcount, on="seg", how="left", suffix="_t").iter_rows(named=True):
        n_t = r["len_t"] or 0
        sd = np.sqrt(r["p"] * (1 - r["p"]) / test.height)
        z = (n_t / test.height - r["p"]) / sd
        out.append(("WARN" if abs(z) > 3 else "OK", f"segment {r['seg']} share {n_t / test.height:.3f} vs valid {r['p']:.3f} (z {z:+.1f})"))
    params = policies.fit(policies.DEFAULT_POLICY, load_oof(run, pool))
    td = load_test_preds(run, test)
    pred = policies.apply(params, td)
    prev = pool.group_by("variety", "source").agg(pl.col("sentiment", "sarcasm").mean())
    for v, vn in ((0, "en-AU"), (1, "en-UK")):
        for src_i, src in ((0, "google"), (1, "reddit")):
            m = (td.variety == v) & (td.source == src_i)
            if not m.any():
                continue
            row = prev.filter((pl.col("variety").cast(pl.String) == vn) & (pl.col("source").cast(pl.String) == src))
            for t, col in (("sent", "sentiment"), ("sarc", "sarcasm")):
                rate, q = float(pred[t][m].mean()), float(row[col][0])
                sd = np.sqrt(max(q * (1 - q), 1e-4) / m.sum()) + 0.03  # + model rate error
                z = (rate - q) / sd
                out.append(("WARN" if abs(z) > 3 else "OK",
                            f"{vn}/{src} {col}: {policies.DEFAULT_POLICY} rate {rate:.3f} vs pool prevalence {q:.3f} (z {z:+.1f})"))
    return out


def official_score(answer_csv: Path, truth_csv: Path) -> dict[str, float]:
    """Run the organisers' evaluate.py in its own environment (pandas, as on Codabench)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in" / "res").mkdir(parents=True)
        (tmp / "in" / "ref").mkdir(parents=True)
        shutil.copy(answer_csv, tmp / "in" / "res" / "answer.csv")
        shutil.copy(truth_csv, tmp / "in" / "ref" / "truth.csv")
        subprocess.run(
            ["uv", "run", "--no-project", "--with", "pandas", "--with", "scikit-learn",
             "python", str(SCORER), str(tmp / "in"), str(tmp / "out")],
            check=True, capture_output=True, text=True,
        )
        scores = {}
        for line in (tmp / "out" / "scores.txt").read_text().splitlines():
            k, v = line.split(":")
            scores[k.strip()] = float(v)
        return scores


def check_submission(zip_path: Path, test_csv: Path, truth_csv: Path | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if names != ["answer.csv"]:
            out.append(("FAIL", f"zip must contain exactly answer.csv at its root, has {names}"))
            return out
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            ans = pl.read_csv(Path(tmp) / "answer.csv")
            answer_csv = Path(tmp) / "answer.csv"
            test = load_test(test_csv)  # same normalisation as the submission builder
            if ans.columns != COLUMNS:
                out.append(("FAIL", f"columns {ans.columns} != {COLUMNS}"))
                return out
            if ans.height != test.height:
                out.append(("FAIL", f"{ans.height} rows, test has {test.height}"))
                return out
            for c in ("source", "variety", "text"):
                if not ans[c].cast(pl.String).eq_missing(test[c].cast(pl.String)).all():
                    out.append(("FAIL", f"column {c} differs from the test file (row order must match)"))
            for c in ("sentiment", "sarcasm"):
                vals = set(ans[c].unique().to_list())
                if not vals <= {0, 1} or ans[c].null_count():
                    out.append(("FAIL", f"{c} must be 0/1 with no nulls, found {vals}"))
            if not any(level == "FAIL" for level, _ in out):
                out.append(("OK", f"format: {ans.height} rows, columns and row order match the test file"))
            # predicted rates vs the pool prevalences (the test set is stratified like valid)
            pool = load_pool().with_columns(pl.col("source", "variety").cast(pl.String))
            prev = pool.group_by("variety", "source").agg(pl.col("sentiment", "sarcasm").mean())
            got = ans.group_by("variety", "source").agg(pl.col("sentiment", "sarcasm").mean())
            cmp = got.join(prev, on=["variety", "source"], suffix="_pool").sort("variety", "source")
            for r in cmp.iter_rows(named=True):
                for c in ("sentiment", "sarcasm"):
                    a, b = r[c], r[f"{c}_pool"]
                    level = "WARN" if abs(a - b) > max(0.06, 0.35 * b) else "OK"
                    out.append((level, f"{r['variety']}/{r['source']} {c} rate {a:.3f} vs pool prevalence {b:.3f}"))
            for v in ("en-AU", "en-UK"):
                for c in ("sentiment", "sarcasm"):
                    if ans.filter(pl.col("variety") == v)[c].n_unique() < 2:
                        out.append(("WARN", f"{v} {c}: only one class predicted (macro-F1 capped at ~0.5)"))
            # official scorer must run on it (against the truth if we have it)
            ref = truth_csv
            if ref is None:
                ref = Path(tmp) / "dummy_truth.csv"
                test.select(pl.col("source", "variety").cast(pl.String), "text").with_columns(
                    pl.lit(0).alias("sentiment"), pl.lit(0).alias("sarcasm")).write_csv(ref)
            try:
                s = official_score(answer_csv, ref)
                out.append(("OK", "official scorer ran" + (f": score {s['score']:.4f} " + str(s) if truth_csv else "")))
            except subprocess.CalledProcessError as e:
                out.append(("FAIL", f"official scorer crashed: {e.stderr[-500:]}"))
    return out


DEFAULT_TEST = DATA / "test.csv"
