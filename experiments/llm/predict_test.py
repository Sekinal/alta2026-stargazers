"""Test-set probabilities for a kept LLM run; re-saves preds/<run>.parquet with the test split.

Run it under the GPU lock (it scores the test rows):
  flock .gpu.lock uv run python experiments/llm/predict_test.py --run llm_q8_probe_v2 --test data/test.csv

Recipes (read from preds/<run>.json):
  zs / probe  score the test rows with the run's prompts (few-shot tags use exemplars from the
              whole pool), then rebuild the run; probe = mean of the 5 cross-fitted fold models
  ft          score the test rows with each fold adapter (models/<run>/fold<k>), average margins
Test uids come from alta.common.load_test(PATH). The OOF part is rebuilt identically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_common import HERE, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from alta.common import N_FOLDS, PREDS  # noqa: E402


def split_tag(tag: str) -> list[str]:
    """'sarc_v1_knnb4' -> ['--prompt', 'sarc_v1', '--shots', 'knnb4']"""
    parts = tag.split("_")
    out = ["--prompt", "_".join(parts[:2])]
    if len(parts) > 2:
        out += ["--shots", "_".join(parts[2:])]
    return out


def sh(args: list[str]):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    meta = json.loads((PREDS / f"{a.run}.json").read_text())
    r = meta["recipe"]
    if r["kind"] in ("zs", "probe"):
        for tag in (r["sarc"], r["sent"]):
            sh([str(HERE / "score.py"), "--model", r["model"], *split_tag(tag), "--test", a.test])
        sys.path.insert(0, str(HERE))
        from make_run import build
        keep = {k: v for k, v in meta.items() if k not in ("run", "fold_hash", "git")}
        build(a.run, r["kind"], r["model"], r["sarc"], r["sent"], r["layer"], r["C"], a.test, keep)
    elif r["kind"] == "ft":
        for k in range(N_FOLDS):
            sh([str(HERE / "ft.py"), "--run", a.run, "--model", r["model"], "--fold", str(k),
                "--predict-only", "--test", a.test,
                "--max-text-tokens", str(meta["train_args"]["max_text_tokens"])])
        sh([str(HERE / "ft_collect.py"), "--run", a.run, "--test", a.test])
    else:
        raise ValueError(r["kind"])
    print(f"saved test predictions into preds/{a.run}.parquet")


if __name__ == "__main__":
    main()
