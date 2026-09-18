"""Integrity checks for a run's prediction file. Each check yields (level, message)."""

from __future__ import annotations

import numpy as np
import polars as pl

from alta.common import fold_hash, load_pool
from alta.validation.data import SEGMENT_NAMES, TASKS, load_oof, read_run
from alta.validation.metrics import components, per_fold

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def check_run(run: str, pool: pl.DataFrame | None = None) -> list[tuple[str, str]]:
    pool = load_pool() if pool is None else pool
    out: list[tuple[str, str]] = []
    preds, meta = read_run(run)

    # --- schema
    expected = {"uid", "split", "p_sent", "p_sarc"}
    if set(preds.columns) != expected:
        out.append((FAIL, f"columns {sorted(preds.columns)} != {sorted(expected)}"))
        return out
    bad_split = set(preds["split"].unique().to_list()) - {"oof", "test"}
    if bad_split:
        out.append((FAIL, f"unknown split values {bad_split}"))

    # --- OOF coverage
    oof = preds.filter(pl.col("split") == "oof")
    pool_uids, oof_uids = set(pool["uid"].to_list()), oof["uid"].to_list()
    if len(oof_uids) != len(set(oof_uids)):
        out.append((FAIL, f"{len(oof_uids) - len(set(oof_uids))} duplicated OOF uids"))
    missing, extra = pool_uids - set(oof_uids), set(oof_uids) - pool_uids
    if missing or extra:
        out.append((FAIL, f"OOF coverage: {len(missing)} pool rows missing, {len(extra)} unknown uids"))
    else:
        out.append((OK, f"OOF covers all {len(pool_uids)} pool rows exactly once"))

    # --- probability sanity
    for split, part in (("oof", oof), ("test", preds.filter(pl.col("split") == "test"))):
        for c in ("p_sent", "p_sarc"):
            x = part[c].to_numpy()
            if len(x) == 0:
                continue
            if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                out.append((FAIL, f"{split}.{c}: values outside [0, 1] or non-finite"))
            elif len(np.unique(x)) < 20:
                out.append((WARN, f"{split}.{c}: only {len(np.unique(x))} distinct values (hard labels?)"))

    # --- test split integrity
    test_part = preds.filter(pl.col("split") == "test")
    if test_part.height:
        tu = test_part["uid"].to_list()
        if len(tu) != len(set(tu)):
            out.append((FAIL, f"{len(tu) - len(set(tu))} duplicated test uids"))
        if not test_part["uid"].str.contains(r"^x\d{5}$").all():
            out.append((FAIL, "test uids must look like x00000 (from alta.common.load_test)"))
        if "test_sha" not in meta or meta.get("n_test") != test_part.height:
            out.append((FAIL, "test split lacks a fingerprint of its test file or has the wrong row count "
                              "(save it with save_preds(..., test_csv=PATH))"))
        else:
            out.append((OK, f"test split: {test_part.height} rows bound to {meta.get('test_csv')} ({meta['test_sha']})"))

    # --- provenance
    if "fold_hash" not in meta:
        out.append((WARN, "no fold_hash in meta (saved before provenance tracking); cannot prove fold match"))
    elif meta["fold_hash"] != fold_hash(pool):
        out.append((FAIL, f"fold_hash {meta['fold_hash']} != current {fold_hash(pool)}: folds changed"))
    else:
        out.append((OK, "fold_hash matches current folds"))
    if "git" not in meta:
        out.append((WARN, "no git commit recorded in meta"))
    if not meta.get("folds_measured"):
        out.append((WARN, "folds not measured: fold_hash only proves the pool's folds, not the ones the model "
                          "used; train through alta.validation.harness.cv_fit_predict and pass folds_used"))
    can = meta.get("canary")
    if can is None:
        out.append((WARN, "no permutation canary in meta (alta.validation.harness.canary); leaks from "
                          "supervised preprocessing / in-sample predictions are invisible without it"))
    elif not can.get("passed"):
        out.append((FAIL, f"permutation canary FAILED: shuffled-label score {can.get('mean')} vs floor "
                          f"{can.get('floor')} (+{can.get('margin')}), max AUC z {can.get('max_auc_z')}"))
    else:
        out.append((OK, f"permutation canary passed ({can.get('mean')} vs floor {can.get('floor')})"))

    if any(level == FAIL for level, _ in out):
        return out

    # --- leakage / instability canaries on the OOF itself
    d = load_oof(run, pool)
    pred = {t: (d.p[t] >= 0.5).astype(np.int8) for t in TASKS}
    folds = per_fold(d, pred)
    spread = max(f["score"] for f in folds) - min(f["score"] for f in folds)
    out.append((WARN if spread > 0.06 else OK, f"per-fold score spread {spread:.3f} (threshold 0.5)"))
    from alta.validation import policies
    from alta.validation.resampling import replicate_indices, replicate_scores
    hpred, _ = policies.crossfit(policies.DEFAULT_POLICY, d)  # the operating point we ship
    s = components(d, hpred)["score"]
    if s > 0.93:
        out.append((WARN, f"headline score {s:.3f} is implausibly high for this task: check for fold leakage"))
    # valid rows leaked (e.g. trained on valid): valid subset beats pool by far more than sampling allows
    valid = d.origin == "valid"
    sv = components(d.subset(valid), {t: v[valid] for t, v in hpred.items()})["score"]
    boot_sd = float(replicate_scores(d, hpred, replicate_indices(d, B=300, seed=1))["score"].std(ddof=1))
    zv = (sv - s) / (boot_sd * np.sqrt(1 - valid.sum() / len(d)))
    out.append((WARN if zv > 2 else OK, f"valid-vs-pool gap z = {zv:.2f} (valid {sv:.3f}, pool {s:.3f}; "
                                         "> 2 suggests valid rows leaked into training)"))
    uk_reddit = (d.variety == 1) & (d.source == 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(d.y["sarc"][uk_reddit], d.p["sarc"][uk_reddit])
    if auc > 0.95:
        out.append((WARN, f"UK-reddit sarcasm AUC {auc:.3f} is implausibly high: check for leakage"))

    # --- test vs OOF distribution shift (fold-averaged test probs are often sharper/flatter)
    test = preds.filter(pl.col("split") == "test")
    if test.height:
        seg_oof = d.segment()
        test_rates = _test_segment_rates(test, meta.get("test_csv"))
        for t in TASKS:
            for s, name in enumerate(SEGMENT_NAMES):
                if name not in test_rates:
                    continue
                r_oof = float((d.p[t][seg_oof == s] >= 0.5).mean())
                r_test = test_rates[name][t]
                if abs(r_test - r_oof) > max(0.05, 0.3 * r_oof):
                    out.append((WARN, f"{t} {name}: P(p>=0.5) test {r_test:.3f} vs OOF {r_oof:.3f}; "
                                      "prefer a prevalence policy"))
    return out


def _test_segment_rates(test_preds: pl.DataFrame, test_csv: str | None) -> dict:
    """Segments come from the test file the predictions are bound to (meta test_csv)."""
    from pathlib import Path

    from alta.common import load_test
    if not test_csv or not Path(test_csv).exists():
        return {}
    path = test_csv
    t = load_test(path).join(test_preds, on="uid", how="inner")
    out = {}
    for (v, s), g in t.group_by("variety", "source"):
        name = f"{v[3:]}-{s}"
        out[name] = {"sent": float((g["p_sent"] >= 0.5).mean()), "sarc": float((g["p_sarc"] >= 0.5).mean())}
    return out


def format_checks(checks: list[tuple[str, str]]) -> str:
    return "\n".join(f"  [{level:4}] {msg}" for level, msg in checks)
