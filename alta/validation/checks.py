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

    # --- provenance
    if "fold_hash" not in meta:
        out.append((WARN, "no fold_hash in meta (saved before provenance tracking); cannot prove fold match"))
    elif meta["fold_hash"] != fold_hash(pool):
        out.append((FAIL, f"fold_hash {meta['fold_hash']} != current {fold_hash(pool)}: folds changed"))
    else:
        out.append((OK, "fold_hash matches current folds"))
    if "git" not in meta:
        out.append((WARN, "no git commit recorded in meta"))

    if any(level == FAIL for level, _ in out):
        return out

    # --- leakage / instability canaries on the OOF itself
    d = load_oof(run, pool)
    pred = {t: (d.p[t] >= 0.5).astype(np.int8) for t in TASKS}
    folds = per_fold(d, pred)
    spread = max(f["score"] for f in folds) - min(f["score"] for f in folds)
    out.append((WARN if spread > 0.06 else OK, f"per-fold score spread {spread:.3f} (threshold 0.5)"))
    s = components(d, pred)["score"]
    if s > 0.93:
        out.append((WARN, f"OOF score {s:.3f} is implausibly high for this task: check for fold leakage"))
    uk_reddit = (d.variety == 1) & (d.source == 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(d.y["sarc"][uk_reddit], d.p["sarc"][uk_reddit])
    if auc > 0.95:
        out.append((WARN, f"UK-reddit sarcasm AUC {auc:.3f} is implausibly high: check for leakage"))

    # --- test vs OOF distribution shift (fold-averaged test probs are often sharper/flatter)
    test = preds.filter(pl.col("split") == "test")
    if test.height:
        seg_oof = d.segment()
        test_rates = _test_segment_rates(test)
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


def _test_segment_rates(test_preds: pl.DataFrame) -> dict:
    """Needs the test frame to know segments; uids of test rows are x-prefixed row numbers."""
    from alta.common import DATA, load_test
    path = DATA / "test.csv"
    if not path.exists():
        return {}
    t = load_test(path).join(test_preds, on="uid", how="inner")
    out = {}
    for (v, s), g in t.group_by("variety", "source"):
        name = f"{v[3:]}-{s}"
        out[name] = {"sent": float((g["p_sent"] >= 0.5).mean()), "sarc": float((g["p_sarc"] >= 0.5).mean())}
    return out


def format_checks(checks: list[tuple[str, str]]) -> str:
    return "\n".join(f"  [{level:4}] {msg}" for level, msg in checks)
