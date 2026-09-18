"""Test predictions for a saved classical run (cls_*) or stack (stk_*), re-saved into preds/RUN.parquet.

  uv run python experiments/classical/predict_test.py --run RUN --test PATH

Protocol (per run kind, read from preds/RUN.json):
  tfidf / probe / gbdt   AVERAGE OF THE 5 FOLD MODELS: for fold k the exact OOF procedure (fit on
                         folds != k, inner-CV hyperparameters) is rerun and applied to the test rows.
                         This is the project protocol (alta/common.py): test probabilities then have
                         the same distribution as the OOF ones the decision policy is fitted on
                         (red-team: ties refit for linear probes, beats it for noisier models).
  stack                  members' test predictions are produced first (recursively for cls_/stk_
                         members; other agents' members must already carry split == 'test' rows),
                         then the 5 fold level-2 models (each fitted on the OOF rows of the other
                         folds, as for the OOF) are applied to them and averaged.
  blend                  mean of the members' test probabilities.
Missing test embeddings are extracted on the GPU (under the .gpu.lock) via embed.py.
The OOF rows of the run are kept unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

import embed
from common import HERE, LABEL, ROOT, frame, pool
from alta.common import PREDS, frame_fingerprint, load_test, save_preds


def ensure_embeddings(names: list[str], test: Path) -> None:
    missing = [n for n in names if not embed.cache_path(n, test).exists()]
    if missing:
        subprocess.run(["flock", str(ROOT / ".gpu.lock"), "uv", "run", "python", str(HERE / "embed.py"),
                        "--test", str(test), *missing], cwd=ROOT, check=True)


def has_test(run: str, test_path: Path, n: int) -> bool:
    """Test rows present AND bound to this exact test file (alta's fingerprint in the run's meta)."""
    p = pl.read_parquet(PREDS / f"{run}.parquet")
    meta = json.loads((PREDS / f"{run}.json").read_text())
    return (p.filter(pl.col("split") == "test").height == n
            and meta.get("test_sha") == frame_fingerprint(load_test(test_path)))


def fold_average(fit_fold) -> dict[str, np.ndarray]:
    """fit_fold(k) -> task -> test probabilities of the fold-k model; returns their mean."""
    ps = [fit_fold(k) for k in range(5)]
    return {t: np.mean([p[t] for p in ps], 0) for t in ("sent", "sarc")}


def predict(run: str, test_path: Path, force: bool = False) -> None:
    meta = json.loads((PREDS / f"{run}.json").read_text())
    test = frame(load_test(test_path))
    if has_test(run, test_path, test.height) and not force:
        print(f"{run}: test predictions already present")
        return
    kind = meta.get("kind")
    df = pool()
    y = {t: df[LABEL[t]].to_numpy() for t in ("sent", "sarc")}
    fold = df["fold"].to_numpy()
    if kind == "tfidf":
        import tfidf
        p = fold_average(lambda k: tfidf.predict(meta["config"], df.filter(pl.col("fold") != k), test))
    elif kind == "probe":
        import probe
        cfg = meta["config"]
        names = cfg["emb"].split("+")
        ensure_embeddings(names, test_path)
        X, Xt, seg = probe.load_features(cfg["emb"]), probe.load_features(cfg["emb"], test_path), df["seg"].to_numpy()
        p = fold_average(lambda k: probe.predict(cfg, X[fold != k], {t: v[fold != k] for t, v in y.items()},
                                                 seg[fold != k], Xt, test["seg"].to_numpy()))
    elif kind == "gbdt":
        import gbdt
        cfg = meta["config"]
        clf = cfg["clf"].split(",")
        ensure_embeddings([c for c in clf if c in gbdt.CLF_LOGITS], test_path)
        names = cfg["emb"].split("+")
        ensure_embeddings(names, test_path)
        E = np.hstack([embed.load(n) for n in names])
        Et = np.hstack([embed.load(n, test_path) for n in names])
        L, Lt = gbdt.logits(clf), gbdt.logits(clf, test_path)
        mode, colsample = cfg.get("features", "pca"), cfg.get("colsample", 0.5)

        def fit_fold(k):
            tr = fold != k
            A, B = gbdt.dense_features(df.filter(pl.col("fold") != k), test, E[tr], Et, L[tr], Lt, mode=mode)
            return {t: gbdt.fit_predict(cfg["algo"], A, y[t][tr], fold[tr], B, colsample) for t in y}
        p = fold_average(fit_fold)
    elif kind in ("stack", "blend"):
        for m in meta["members"]:
            if not has_test(m, test_path, test.height) or (force and m.startswith(("cls_", "stk_"))):
                if not m.startswith(("cls_", "stk_")):
                    sys.exit(f"member {m} has no test predictions: run its owner's predict_test first")
                predict(m, test_path, force)
        if kind == "stack":
            import stack
            p = stack.predict_test(run, test, test_path)
        else:
            import stack
            M = stack.load_members(meta["members"], test["uid"].to_numpy(), split="test")
            p = {}
            for t in ("sent", "sarc"):
                P = np.column_stack([1 / (1 + np.exp(-M[m][t])) for m in meta["members"]])
                p[t] = P.mean(1) if meta.get("method", "mean") == "mean" else \
                    1 / (1 + np.exp(-np.column_stack([M[m][t] for m in meta["members"]]).mean(1)))
    else:
        sys.exit(f"{run}: unsupported kind {kind!r}")
    for t, v in p.items():
        assert len(v) == test.height and np.isfinite(v).all(), (run, t)
    oof = pl.read_parquet(PREDS / f"{run}.parquet").filter(pl.col("split") == "oof")
    tst = pl.DataFrame({"uid": test["uid"], "p_sent": p["sent"], "p_sarc": p["sarc"]})
    for k in ("fold_hash", "git", "test_csv", "test_sha", "n_test", "folds_measured"):
        meta.pop(k, None)
    # OOF rows are unchanged; every script here produced them on the project folds (fold != k)
    save_preds(run, oof, tst, meta={**meta, "test_protocol": "average of the 5 fold models"},
               test_csv=test_path, folds_used=df.select("uid", "fold"))
    print(f"{run}: saved {test.height} test predictions "
          f"(mean p_sent {p['sent'].mean():.3f}, p_sarc {p['sarc'].mean():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="recompute even if test rows exist")
    a = ap.parse_args()
    predict(a.run, a.test.resolve(), a.force)


if __name__ == "__main__":
    main()
