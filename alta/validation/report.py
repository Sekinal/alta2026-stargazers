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
# The test-sampling bootstrap ignores the CV estimate's own error, the policy refit and the
# final-model difference; red-team pseudo-leaderboards found its 90% band covers only ~82%.
# Inflating its SD by 1.2 restored 0.906 (fold-average) / 0.911 (refit) coverage.
INTERVAL_INFLATION = 1.2
# A non-default policy becomes the headline only with strong paired evidence.
SWITCH_MIN_GAIN, SWITCH_MIN_P = 0.005, 0.95


def choose_policy(d, rows: list[dict], B: int = B_DEFAULT) -> tuple[str, str]:
    """DEFAULT_POLICY unless another policy beats it by >= SWITCH_MIN_GAIN with p >= SWITCH_MIN_P."""
    default = policies.DEFAULT_POLICY
    base = next((r for r in rows if r["policy"] == default), None)
    if base is None:
        return max(rows, key=lambda r: r["score"])["policy"], "default not evaluated: argmax"
    pred_default, _ = policies.crossfit(default, d)
    best, why = default, "default policy"
    for r in sorted(rows, key=lambda r: -r["score"]):
        if r["policy"] == default or r["score"] - base["score"] < SWITCH_MIN_GAIN:
            continue
        pred, _ = policies.crossfit(r["policy"], d)
        p = paired(d, pred, pred_default, B=B)["score"]["p_a_better"]
        if p >= SWITCH_MIN_P:
            best, why = r["policy"], f"beats {default} by {r['score'] - base['score']:+.4f}, p={p:.3f}"
            break
    return best, why


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
        raise SystemExit(1)

    d = load_oof(run, pool)
    rows = evaluate_policies(run, specs, pool)
    spec, why = choose_policy(d, rows, B)
    head = next(r for r in rows if r["policy"] == spec)
    lines += ["## Decision policies (cross-fitted: fitted on 4 folds, applied to the 5th)",
              "Components/score are over all 6,288 pool rows. `fold_sd` is the SD of the per-fold "
              "score. `valid_score` (valid.csv subset) is a DIAGNOSTIC only: selecting on it has ~2-3x "
              "the regret of the pool score. Policy differences within a run are mostly noise; "
              f"the headline uses `{policies.DEFAULT_POLICY}` unless another policy wins by "
              f">= {SWITCH_MIN_GAIN} with paired p >= {SWITCH_MIN_P}.", "",
              _fmt(rows), "", f"**Headline policy: `{spec}`** ({why}); "
              f"**score {head['score']:.4f}**", ""]

    pred, fitted = policies.crossfit(spec, d)
    idx = replicate_indices(d, B=B, seed=0)
    reps = replicate_scores(d, pred, idx)
    boot = [{"metric": k, "cv": head[k] if k != "score" else head["score"],
             **summarise(v, BASELINE_VALID if k == "score" else None)} for k, v in reps.items()]
    sd = float(reps["score"].std(ddof=1)) * INTERVAL_INFLATION
    lo, hi = head["score"] - 1.645 * sd, head["score"] + 1.645 * sd
    lines += [f"## Expected test score for `{spec}`",
              f"**Point estimate {head['score']:.4f} (pool CV; red-team: ~unbiased for the chosen system). "
              f"90% interval [{lo:.4f}, {hi:.4f}]** = CV +- 1.645 x {INTERVAL_INFLATION} x bootstrap SD.", "",
              f"Test-sampling-only bootstrap (B={B}, valid-sized stratified replicates; its q05/q95 are too "
              "narrow on their own, and boot mean runs ~0.0015 pessimistic):", "",
              _fmt(boot), "", "binding varieties: "
              + ", ".join(f"{k} {v:.1%}" for k, v in binding_rates(reps).items()), ""]
    lines += ["## Per-fold scores", "", _fmt(per_fold(d, pred)), ""]
    lines += ["## Diagnostics by component and segment", "", _fmt(detail(d, pred)), ""]
    lines += ["## Reddit slices by length (macro-F1)", "", _fmt(length_slices(d, pred)), ""]

    text = "\n".join(lines)
    if write:
        (RESULTS / "reports").mkdir(parents=True, exist_ok=True)
        (RESULTS / "reports" / f"{run}.md").write_text(text)
        append_ledger(run, rows, reps, spec)
        if plots:
            _plots(run, rows, reps)
    print(text)
    return text


def append_ledger(run: str, rows: list[dict], reps: dict, headline: str) -> None:
    """One row per policy; is_best marks the headline policy (not the argmax)."""
    RESULTS.mkdir(exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    best = next(r for r in rows if r["policy"] == headline)
    new = pl.DataFrame([{
        "time": now, "run": run, "code": git_commit(), "folds": fold_hash(),
        "policy": r["policy"], "score": r["score"], "sent_AU": r["sent-AU"], "sent_UK": r["sent-UK"],
        "sarc_AU": r["sarc-AU"], "sarc_UK": r["sarc-UK"], "fold_sd": r["fold_sd"],
        "valid_score": r["valid_score"], "is_best": r is best,
        "boot_mean": float(reps["score"].mean()) if r is best else None,
        "boot_sd": float(reps["score"].std(ddof=1)) if r is best else None,
    } for r in rows])
    if LEDGER.exists():
        new = pl.concat([read_ledger(), new.with_columns(pl.col(STRING_COLS).cast(pl.String))],
                        how="diagonal_relaxed")
    new.write_csv(LEDGER)


STRING_COLS = ("time", "run", "code", "folds", "policy")


def read_ledger() -> pl.DataFrame:
    """Commit ids like 5e12345 / 1234567 must stay strings (type inference would mangle them)."""
    return pl.read_csv(LEDGER, schema_overrides={c: pl.String for c in STRING_COLS})


def board() -> str:
    if not LEDGER.exists():
        return "ledger is empty"
    led = read_ledger().filter(pl.col("is_best"))
    latest = led.sort("time").group_by("run").last().sort("score", descending=True)
    return _fmt(latest.select("run", "policy", "score", "sent_AU", "sarc_UK", "fold_sd",
                              "valid_score", "boot_sd", "code", "time").to_dicts())


def compare(run_a: str, run_b: str, spec: str = policies.DEFAULT_POLICY, B: int = B_DEFAULT) -> str:
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
             "Read: p_a_better is the share of test-like replicates where A outscores B. Calibration "
             "(red-team): for model-vs-model pairs p 0.85 / 0.93 / 0.97 won the real test 81% / 90% / 97% "
             "of the time; policy-vs-policy pairs are overconfident (0.92 -> 86%). Treat < 0.8 as no "
             "evidence. Seed noise: two seeds of the SAME config reach p >= 0.9 in 18% of pairs, so for "
             "seed-dependent models compare seed-averaged / fold-averaged runs (>= 3 seeds per arm)."]
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
