"""Stage, scan and upload the reproducibility release to the Hugging Face Hub.

  uv run --with huggingface_hub --with pyahocorasick python scripts/hf_release.py stage   # -> .hf_stage/ (hard links)
  uv run --with huggingface_hub --with pyahocorasick python scripts/hf_release.py scan    # refuses on any text hit
  HF_TOKEN=... uv run --with huggingface_hub --with pyahocorasick python scripts/hf_release.py upload --repo USER/NAME

What is published: every git-tracked file (curated to hold no competition text), run predictions, label-free caches and
feature stores, results (minus the remote mirror), the paper build, submission metadata (not answer files), and hf/README.md
as the model card. Never published: data/, baseline/, scoring/, raw-text CSVs, recovered Reddit context, external corpora,
re-annotation work files and the gold-bearing sample.csv. The scan checks every staged text-like file and every string
column of every parquet against all competition texts (train, valid, test) and the external pre-SFT corpus.
Model checkpoints are uploaded separately from the GPU box (see upload_models).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".hf_stage"

ARTIFACT_DIRS = ["preds", "experiments/classical/cache", "experiments/encoders/cache", "experiments/llm/cache",
                 "experiments/llm/out", "experiments/features/cache", "experiments/features/store",
                 "experiments/encoders/logs", "experiments/llm/logs", "experiments/features/logs",
                 "experiments/classical/logs", "results", "redteam/calibration/out", "paper"]
EXCLUDE = ["data/*", "baseline/*", "scoring/*", "results/remote/*", "*.csv.bak",
           "experiments/**/*.csv", "experiments/reannot/work/*", "experiments/reannot/sample.csv",
           "experiments/llm/cache/presft_v1.parquet", "experiments/llm/cache/locks/*",
           "experiments/robustness/holdout_*", "eda/*/examples/*", "eda/*/output.txt",
           "eda/structure/tables/dups_text.csv", "eda/structure/tables/near_duplicates.csv",
           "eda/structure/tables/dups_norm.csv",
           "submissions/*/answer.csv", "submissions/*/answer.zip", "*/__pycache__/*", "*.pyc",
           "paper/main.aux", "paper/main.bbl", "paper/main.blg", "paper/main.fdb_latexmk", "paper/main.fls",
           "paper/main.log", "paper/main.out", ".hf_stage/*"]
TEXT_EXT = {".csv", ".json", ".jsonl", ".md", ".txt", ".tex", ".py", ".sh", ".log", ".yaml", ".yml", ".toml", ".bib",
            ".html", ".tsv"}
# columns known to hold identifiers/tags only (still text-matched; exempt from the free-text heuristic)
SAFE_STR_COLS = {"uid", "split", "run", "item", "member", "variety", "source", "segment", "seg", "policy", "task",
                 "metric", "pair", "rates", "combo", "kind", "name", "arm", "group", "code", "folds", "time", "fold_hash",
                 "stratum", "label", "tag", "prompt", "model", "subreddit", "link_id", "comment_id", "parent_id",
                 "is_best", "spec", "component", "v", "f",
                 "draw", "cond"}  # draw = resampling seed tag, cond = perturbation name


def excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) or rel.startswith(p.rstrip("*")) and p.endswith("/*") for p in EXCLUDE)


def files() -> list[str]:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    out = set(tracked)
    for d in ARTIFACT_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file():
                out.add(str(p.relative_to(ROOT)))
    for p in (ROOT / "submissions").glob("*/info.json"):
        out.add(str(p.relative_to(ROOT)))
    return sorted(r for r in out if not excluded(r) and (ROOT / r).is_file())


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    fl = files()
    for r in fl:
        dst = STAGE / ("PROJECT_README.md" if r == "README.md" else "README.md" if r == "hf/README.md" else r)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(ROOT / r, dst)
    size = sum((STAGE / p).stat().st_size for p in [str(x.relative_to(STAGE)) for x in STAGE.rglob("*") if x.is_file()])
    print(f"staged {len(fl)} files, {size / 1e9:.2f} GB in {STAGE}")


def texts() -> list[str]:
    import polars as pl
    out = set()
    for f in ("data/train.csv", "data/valid.csv", "data/test.csv"):
        if (ROOT / f).exists():
            out |= set(pl.read_csv(ROOT / f)["text"].drop_nulls().to_list())
    ext = ROOT / "experiments/llm/cache/presft_v1.parquet"
    if ext.exists():
        out |= set(pl.read_parquet(ext)["text"].drop_nulls().to_list())
    # 40-char windows from the middle of each text: robust to tokenisation/whitespace edits at the ends
    keys = set()
    for t in out:
        t = " ".join(t.split())
        if len(t) >= 40:
            m = len(t) // 2
            keys.add(t[m - 20:m + 20])
    return sorted(keys)


def scan() -> int:
    import ahocorasick
    import numpy as np
    import polars as pl
    A = ahocorasick.Automaton()
    for i, k in enumerate(texts()):
        A.add_word(k, i)
    A.make_automaton()
    print(f"scanning against {len(A)} text keys")
    bad = []
    for p in sorted(STAGE.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(STAGE))
        try:
            if p.suffix in TEXT_EXT:
                s = " ".join(p.read_text(errors="ignore").split())
                if next(A.iter(s), None) is not None:
                    bad.append((rel, "text match"))
            elif p.suffix == ".parquet":
                d = pl.read_parquet(p)
                for c, dt in d.schema.items():
                    if dt not in (pl.String, pl.Categorical, pl.Enum):
                        continue
                    vals = [v for v in d[c].cast(pl.String).drop_nulls().unique().to_list()]
                    s = " ".join(" ".join(vals).split())
                    if next(A.iter(s), None) is not None:
                        bad.append((rel, f"text match in column {c!r}"))
                        break
                    # free-text-looking values (whatever the column is called) need a human look
                    if c not in SAFE_STR_COLS and any(len(v) > 60 and v.count(" ") >= 6 for v in vals):
                        bad.append((rel, f"free-text-like column {c!r}"))
                        break
            elif p.suffix == ".npy":
                np.load(p, mmap_mode="r", allow_pickle=False)  # object arrays (could hold strings) raise here
            elif p.suffix == ".npz":
                with np.load(p, allow_pickle=False) as z:
                    for k in z.files:
                        if z[k].dtype.kind in "OUS" and k != "format":  # scipy sparse format tag, e.g. b'csr'
                            bad.append((rel, f"string array {k!r}"))
        except Exception as e:  # noqa: BLE001
            bad.append((rel, f"unreadable/unsafe: {type(e).__name__}: {str(e)[:80]}"))
    for b in bad:
        print("FLAG", *b)
    print(f"{len(bad)} flagged files")
    return len(bad)


def upload(repo: str) -> None:
    from huggingface_hub import HfApi
    if scan():
        sys.exit("refusing to upload: flagged files (see above)")
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, folder_path=STAGE, repo_type="model")
    print(json.dumps({"repo": repo, "private": api.repo_info(repo).private}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "scan", "upload"])
    ap.add_argument("--repo", default="Thermostatic/alta2026-thermostatic")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage()
    elif a.cmd == "scan":
        sys.exit(1 if scan() else 0)
    else:
        upload(a.repo)
