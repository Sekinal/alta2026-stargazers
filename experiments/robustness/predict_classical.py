"""Fold-k-model robustness predictions for classical tfidf / probe runs (reference implementation).

For every kind CSV of a holdout directory: fit the run's exact OOF procedure on the pool folds
!= k (clean text) and predict the perturbed holdout rows, then save with protocol "fold_model".
Probe runs need embeddings of each CSV; missing ones are extracted via experiments/classical/embed.py
under .gpu.lock (same as predict_test.py).

  uv run python experiments/robustness/predict_classical.py --run cls_e5me5_lr [--dir DIR] [--kinds a,b]
  uv run python experiments/robustness/perturb.py report --run cls_e5me5_lr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "classical"))
sys.path.insert(0, str(HERE))
from common import LABEL, frame, pool  # noqa: E402
from perturb import KINDS, _manifest, holdout_dir, save_rob_preds  # noqa: E402
from alta.common import PREDS, load_test  # noqa: E402


def predict_kind(run: str, meta: dict, csv: Path, fold: int) -> dict[str, np.ndarray]:
    df = pool()
    tr = df["fold"].to_numpy() != fold
    test = frame(load_test(csv))
    if meta["kind"] == "tfidf":
        import tfidf
        return tfidf.predict(meta["config"], df.filter(pl.col("fold") != fold), test)
    if meta["kind"] == "probe":
        import probe
        from predict_test import ensure_embeddings
        cfg = meta["config"]
        ensure_embeddings(cfg["emb"].split("+"), csv)
        X, Xt = probe.load_features(cfg["emb"]), probe.load_features(cfg["emb"], csv)
        y = {t: df[LABEL[t]].to_numpy()[tr] for t in ("sent", "sarc")}
        return probe.predict(cfg, X[tr], y, df["seg"].to_numpy()[tr], Xt, test["seg"].to_numpy())
    raise SystemExit(f"{run}: kind {meta['kind']!r} not supported here (tfidf / probe only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--kinds", default=None, help="comma list (default: all except drop_prefix)")
    a = ap.parse_args()
    d = a.dir or holdout_dir()
    man = _manifest(d)
    meta = json.loads((PREDS / f"{a.run}.json").read_text())
    kinds = a.kinds.split(",") if a.kinds else [k for k in KINDS if k != "drop_prefix"]
    for kind in kinds:
        csv = (d / f"{kind}.csv").resolve()
        p = predict_kind(a.run, meta, csv, man["fold"])
        test = load_test(csv)
        out = pl.DataFrame({"uid": test["uid"], "p_sent": p["sent"], "p_sarc": p["sarc"]})
        path = save_rob_preds(a.run, kind, out, d, protocol="fold_model")
        print(f"{a.run} {kind}: {path}")
        if kind == "clean":  # the fold-k model on clean text must reproduce the OOF
            oof = pl.read_parquet(PREDS / f"{a.run}.parquet").filter(pl.col("split") == "oof")
            chk = pl.read_parquet(path).join(oof, on="uid", suffix="_oof")
            err = max(float((chk["p_sent"] - chk["p_sent_oof"]).abs().max()),
                      float((chk["p_sarc"] - chk["p_sarc_oof"]).abs().max()))
            print(f"  clean vs OOF max |dp| = {err:.2e}")


if __name__ == "__main__":
    main()
