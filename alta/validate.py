"""Validation CLI.

  uv run python -m alta.validate check RUN
  uv run python -m alta.validate report RUN [--policies a,b] [--B 2000] [--plots] [--no-write]
  uv run python -m alta.validate compare RUN_A RUN_B [--policy segment+ukg]
  uv run python -m alta.validate board
  uv run python -m alta.validate blend NAME RUN [RUN ...] [--method mean|logit]
  uv run python -m alta.validate stack NAME RUN [RUN ...]
  uv run python -m alta.validate submit RUN --policy P (--dev | --test data/test.csv)
  uv run python -m alta.validate check-submission ZIP --test CSV [--truth CSV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alta.validation import ensemble, policies, report, submission
from alta.validation.checks import FAIL, check_run, format_checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="alta.validate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check"); s.add_argument("run")
    s = sub.add_parser("report"); s.add_argument("run")
    s.add_argument("--policies", default=",".join(policies.DEFAULT_POLICIES))
    s.add_argument("--B", type=int, default=report.B_DEFAULT)
    s.add_argument("--plots", action="store_true"); s.add_argument("--no-write", action="store_true")
    s = sub.add_parser("compare"); s.add_argument("run_a"); s.add_argument("run_b")
    s.add_argument("--policy", default=policies.DEFAULT_POLICY)
    s.add_argument("--B", type=int, default=report.B_DEFAULT)
    sub.add_parser("board")
    for name in ("blend", "stack"):
        s = sub.add_parser(name); s.add_argument("name"); s.add_argument("runs", nargs="+")
        s.add_argument("--force", action="store_true", help="overwrite an existing run of that name")
        if name == "blend":
            s.add_argument("--method", default="mean", choices=["mean", "logit"])
    s = sub.add_parser("submit"); s.add_argument("run"); s.add_argument("--policy", required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true"); g.add_argument("--test", type=Path)
    s = sub.add_parser("check-submission"); s.add_argument("zip", type=Path)
    s.add_argument("--test", type=Path, required=True); s.add_argument("--truth", type=Path)
    a = ap.parse_args(argv)
    if getattr(a, "B", 2) < 2:
        ap.error("--B must be >= 2")

    if a.cmd == "check":
        res = check_run(a.run)
        print(format_checks(res))
        return 1 if any(level == FAIL for level, _ in res) else 0
    if a.cmd == "report":
        report.report(a.run, a.policies.split(","), B=a.B, write=not a.no_write, plots=a.plots)
    elif a.cmd == "compare":
        report.compare(a.run_a, a.run_b, a.policy, a.B)
    elif a.cmd == "board":
        print(report.board())
    elif a.cmd == "blend":
        print(ensemble.blend(a.runs, a.name, a.method, force=a.force))
    elif a.cmd == "stack":
        print(ensemble.stack(a.runs, a.name, force=a.force))
    elif a.cmd == "submit":
        z = (submission.make_dev_submission(a.run, a.policy) if a.dev
             else submission.make_test_submission(a.run, a.policy, a.test))
        test_csv = submission.DATA / "valid.csv" if a.dev else a.test
        truth = submission.DATA / "valid.csv" if a.dev else None
        res = submission.check_submission(z, test_csv, truth)
        print(z)
        print(format_checks(res))
        return 1 if any(level == FAIL for level, _ in res) else 0
    elif a.cmd == "check-submission":
        res = submission.check_submission(a.zip, a.test, a.truth)
        print(format_checks(res))
        return 1 if any(level == FAIL for level, _ in res) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
