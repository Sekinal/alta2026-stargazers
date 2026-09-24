"""Pre-registered LLM-reader sarcasm feature for en-UK Reddit (experiments/prereg/reader_feature.md).

UK Reddit sarcasm: cross-fitted LR on [lead logit, reader label]; everything else = the lead. Saves run
<lead>_rdr (OOF + test when test labels and the lead's test split exist).

  uv run python experiments/reannot/reader_feature.py --labels DIR [--lead stk_v5r] [--test data/test.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alta.common import PREDS, load_pool, load_test, save_preds  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def labels(d: Path, split: str, variety: str) -> pl.DataFrame:
    rows = []
    for f in sorted(d.glob(f"full_{split}_{variety}_*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return pl.DataFrame(rows).select(pl.col("item").alias("uid"), pl.col("sarcasm").alias("reader_sarc"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead", default="stk_v5r")
    ap.add_argument("--test", default="")
    a = ap.parse_args()
    d = Path(a.labels)
    pool = load_pool()
    lead = pl.read_parquet(PREDS / f"{a.lead}.parquet")
    oof = pool.select("uid", "fold", "variety", "source", "sarcasm").join(
        lead.filter(pl.col("split") == "oof"), on="uid", how="left", maintain_order="left")
    lab = labels(d, "pool", "en-UK")
    oof = oof.join(lab, on="uid", how="left", maintain_order="left")
    m = ((oof["variety"] == "en-UK") & (oof["source"] == "reddit")).to_numpy()
    assert oof.filter(pl.Series(m))["reader_sarc"].null_count() == 0, "missing reader labels for UK reddit pool rows"
    X = np.column_stack([logit(oof["p_sarc"].to_numpy()), oof["reader_sarc"].fill_null(0).to_numpy()])
    y, fold = oof["sarcasm"].to_numpy(), oof["fold"].to_numpy()
    p = oof["p_sarc"].to_numpy().copy()
    models = []
    for k in range(5):
        tr, te = m & (fold != k), m & (fold == k)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
        models.append(clf)
        print(f"fold {k}: coef lead {clf.coef_[0][0]:.3f} reader {clf.coef_[0][1]:.3f}")
    out = oof.select("uid", "p_sent").with_columns(pl.Series("p_sarc", p))
    tdf = None
    if a.test:
        test = load_test(a.test)
        lt = test.select("uid", "variety", "source").join(lead.filter(pl.col("split") == "test"), on="uid",
                                                           how="left", maintain_order="left")
        lt = lt.join(labels(d, "test", "en-UK"), on="uid", how="left", maintain_order="left")
        mt = ((lt["variety"] == "en-UK") & (lt["source"] == "reddit")).to_numpy()
        assert lt.filter(pl.Series(mt))["reader_sarc"].null_count() == 0, "missing reader labels for UK reddit test rows"
        Xt = np.column_stack([logit(lt["p_sarc"].to_numpy()), lt["reader_sarc"].fill_null(0).to_numpy()])
        pt = lt["p_sarc"].to_numpy().copy()
        pt[mt] = np.mean([c.predict_proba(Xt[mt])[:, 1] for c in models], 0)
        tdf = lt.select("uid", "p_sent").with_columns(pl.Series("p_sarc", pt))
    name = f"{a.lead}_rdr"
    meta = {"kind": "reader_feature", "script": "experiments/reannot/reader_feature.py", "base": a.lead,
            "description": f"{a.lead} + LLM-reader sarcasm label for en-UK Reddit (cross-fitted LR), prereg reader_feature.md",
            "canary": {"passed": True, "status": "inherits the lead's members; reader labels are label-free"}}
    print(save_preds(name, out, tdf, meta=meta, test_csv=a.test or None, folds_used=pool.select("uid", "fold")))


if __name__ == "__main__":
    main()
