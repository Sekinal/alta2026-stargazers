"""Full validation report for a run, a results ledger, and paired comparisons."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from alta.common import ROOT, fold_hash, git_commit, load_pool
from alta.validation import policies
from alta.validation.checks import FAIL, check_run, format_checks
from alta.validation.data import TASKS, load_oof
from alta.validation.metrics import components, detail, length_slices, per_fold
from alta.validation.resampling import (binding_rates, paired, replicate_scores, summarise,
                                        replicate_indices)

RESULTS = ROOT / "results"
LEDGER = RESULTS / "ledger.csv"
BASELINE_VALID = 0.7740  # organisers' distilbert baseline on valid.csv
B_DEFAULT = 2000


def _fmt(rows: list[dict], digits: int = 3) -> str:
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col(pl.Float64).round(digits))
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=200, tbl_hide_dataframe_shape=True,
                   tbl_formatting="MARKDOWN", tbl_hide_column_data_types=True, fmt_str_lengths=40):
        return str(df)


def evaluate_policies(run: str, specs=policies.DEFAULT_POLICIES, pool=None) -> list[dict]:
    d = load_oof(run, pool)
    valid = d.origin == "valid"
    rows = []
    for spec in specs:
        pred, _ = policies.crossfit(spec, d)
        c = components(d, pred)
        cv = components(d.subset(valid), {t: pred[t][valid] for t in TASKS})
        folds = [f["score"] for f in per_fold(d, pred)]
        rows.append({"policy": spec, **c, "fold_sd": float(np.std(folds, ddof=1)),
                     "valid_score": cv["score"]})
    return rows


def report(run: str, specs=policies.DEFAULT_POLICIES, B: int = B_DEFAULT, write: bool = True,
           plots: bool = False) -> str:
    pool = load_pool()
    lines = [f"# Validation report: `{run}`", "",
             f"generated {dt.datetime.now():%Y-%m-%d %H:%M}, code {git_commit()}, folds {fold_hash(pool)}", ""]
    checks = check_run(run, pool)
    lines += ["## Integrity checks", "```", format_checks(checks), "```", ""]
    if any(level == FAIL for level, _ in checks):
        text = "\n".join(lines + ["**FAILED integrity checks: no scores reported.**"])
        print(text)
        return text

    d = load_oof(run, pool)
    rows = evaluate_policies(run, specs, pool)
    best = max(rows, key=lambda r: r["score"])
    lines += ["## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)",
              "Components/score are over all 6,288 pool rows; `valid_score` is the valid.csv subset "
              "of the same honest labels; `fold_sd` is the SD of the per-fold score.", "",
              _fmt(rows), ""]

    spec = best["policy"]
    pred, fitted = policies.crossfit(spec, d)
    idx = replicate_indices(d, B=B, seed=0)
    reps = replicate_scores(d, pred, idx)
    boot = [{"metric": k, **summarise(v, BASELINE_VALID if k == "score" else None)} for k, v in reps.items()]
    lines += [f"## Test-like bootstrap for best policy `{spec}` (B={B}, valid-sized stratified replicates)",
              "Expected leaderboard distribution from test-set sampling alone "
              f"(p_above_ref = P(score > baseline {BASELINE_VALID})).", "",
              _fmt(boot), "", "binding varieties: "
              + ", ".join(f"{k} {v:.1%}" for k, v in binding_rates(reps).items()), ""]
    lines += ["## Per-fold scores", "", _fmt(per_fold(d, pred)), ""]
    lines += ["## Diagnostics by component and segment", "", _fmt(detail(d, pred)), ""]
    lines += ["## Reddit slices by length (macro-F1)", "", _fmt(length_slices(d, pred)), ""]

    text = "\n".join(lines)
    if write:
        (RESULTS / "reports").mkdir(parents=True, exist_ok=True)
        (RESULTS / "reports" / f"{run}.md").write_text(text)
        append_ledger(run, rows, reps)
        if plots:
            _plots(run, rows, reps)
    print(text)
    return text


def append_ledger(run: str, rows: list[dict], reps: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    best = max(rows, key=lambda r: r["score"])
    new = pl.DataFrame([{
        "time": now, "run": run, "code": git_commit(), "folds": fold_hash(),
        "policy": r["policy"], "score": r["score"], "sent_AU": r["sent-AU"], "sent_UK": r["sent-UK"],
        "sarc_AU": r["sarc-AU"], "sarc_UK": r["sarc-UK"], "fold_sd": r["fold_sd"],
        "valid_score": r["valid_score"], "is_best": r is best,
        "boot_mean": float(reps["score"].mean()) if r is best else None,
        "boot_sd": float(reps["score"].std(ddof=1)) if r is best else None,
    } for r in rows])
    if LEDGER.exists():
        new = pl.concat([pl.read_csv(LEDGER), new], how="diagonal_relaxed")
    new.write_csv(LEDGER)


def board() -> str:
    if not LEDGER.exists():
        return "ledger is empty"
    led = pl.read_csv(LEDGER).filter(pl.col("is_best"))
    latest = led.sort("time").group_by("run").last().sort("score", descending=True)
    return _fmt(latest.select("run", "policy", "score", "sent_AU", "sarc_UK", "fold_sd",
                              "valid_score", "boot_sd", "code", "time").to_dicts())


def compare(run_a: str, run_b: str, spec: str = "fixed+ukg", B: int = B_DEFAULT) -> str:
    pool = load_pool()
    da, db = load_oof(run_a, pool), load_oof(run_b, pool)
    assert (da.uid == db.uid).all()
    pa, _ = policies.crossfit(spec, da)
    pb, _ = policies.crossfit(spec, db)
    res = paired(da, pa, pb, B=B)
    fa, fb = per_fold(da, pa), per_fold(db, pb)
    fold_diff = np.array([x["score"] - y["score"] for x, y in zip(fa, fb)])
    ca, cb = components(da, pa), components(db, pb)
    lines = [f"# Paired comparison: `{run_a}` (A) vs `{run_b}` (B), policy `{spec}`", "",
             _fmt([{"metric": k, "A": ca[k], "B": cb[k], **v} for k, v in res.items()]), "",
             f"per-fold score differences A-B: {np.round(fold_diff, 4).tolist()} "
             f"(A wins {int((fold_diff > 0).sum())}/5)", "",
             "Read: p_a_better is the share of test-like replicates where A outscores B. "
             "Treat < 0.8 as no evidence of a difference."]
    text = "\n".join(lines)
    print(text)
    return text


def _plots(run: str, rows: list[dict], reps: dict) -> None:
    from lets_plot import (LetsPlot, aes, geom_density, geom_vline, ggplot, ggsave, ggsize, labs,
                           theme_minimal2)
    LetsPlot.setup_html()
    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    long = pl.concat([pl.DataFrame({"metric": k, "value": v}) for k, v in reps.items()])
    p = (ggplot(long, aes("value", fill="metric"))
         + geom_density(alpha=0.35, size=0.4)
         + geom_vline(xintercept=BASELINE_VALID, linetype="dashed")
         + labs(title=f"{run}: test-like bootstrap distribution",
                subtitle="valid-sized stratified replicates; dashed = organisers' baseline",
                x="macro-F1 / score", y="density")
         + theme_minimal2() + ggsize(900, 450))
    ggsave(p, f"{run}_bootstrap.png", path=str(out), scale=1.5)
