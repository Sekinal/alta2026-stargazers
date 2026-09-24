"""Analysis of the blind re-annotation (experiments/prereg/reannotation.md).

Reads experiments/reannot/sample.csv (gold, lead) and the annotator files <dir>/a{1,2}_batch_*.jsonl.
Writes results/paper/reannotation.csv (no text).

  uv run python experiments/reannot/analyze.py --ann-dir DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "paper" / "reannotation.csv"


def load_ann(d: Path, who: str) -> pl.DataFrame:
    rows = []
    for f in sorted(d.glob(f"{who}_batch_*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return pl.DataFrame(rows).rename({"sarcasm": f"{who}_sarc", "sentiment": f"{who}_sent"})


def kappa_ci(a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = cohen_kappa_score(a, b)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) > 1 or len(set(b[i])) > 1:
            bs.append(cohen_kappa_score(a[i], b[i]))
    return k, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    a = ap.parse_args()
    d = Path(a.ann_dir)
    s = pl.read_csv(HERE / "sample.csv")
    for who in ("a1", "a2"):
        s = s.join(load_ann(d, who), on="item", how="left")
    assert s.select(pl.col("^a[12]_.*$").null_count()).sum_horizontal().item() == 0, "missing annotations"
    s = s.rename({"sarcasm": "gold_sarc", "sentiment": "gold_sent"})
    raters = ("gold", "lead", "a1", "a2")
    rows = []
    for v in ("en-AU", "en-UK"):
        u = s.filter((pl.col("variety") == v) & (pl.col("stratum") == "U"))
        for t in ("sarc", "sent"):
            for r1, r2 in itertools.combinations(raters, 2):
                x, y = u[f"{r1}_{t}"].to_numpy(), u[f"{r2}_{t}"].to_numpy()
                k, lo, hi = kappa_ci(x, y)
                rows.append({"variety": v, "task": t, "pair": f"{r1}-{r2}", "n": len(x), "kappa": round(k, 3),
                             "lo": round(lo, 3), "hi": round(hi, 3), "agree": round(float((x == y).mean()), 3)})
            rows.append({"variety": v, "task": t, "pair": "rates(gold,lead,a1,a2)", "n": u.height,
                         "kappa": None, "lo": None, "hi": None,
                         "agree": None, "rates": ",".join(f"{u[f'{r}_{t}'].mean():.3f}" for r in raters)})
    res = pl.DataFrame(rows, infer_schema_length=None)
    OUT.parent.mkdir(exist_ok=True)
    res.write_csv(OUT)
    pl.Config.set_tbl_rows(60)
    print(res.filter(pl.col("kappa").is_not_null()).select("variety", "task", "pair", "n", "kappa", "lo", "hi", "agree"))
    print(res.filter(pl.col("kappa").is_null()).select("variety", "task", "rates"))
    # D stratum: disputed sarcasm items (lead contradicts gold with margin > 0.8)
    for v in ("en-AU", "en-UK"):
        dd = s.filter((pl.col("variety") == v) & (pl.col("stratum") == "D"))
        both = dd.filter(pl.col("a1_sarc") == pl.col("a2_sarc"))
        side_lead = int((both["a1_sarc"] == both["lead_sarc"]).sum())
        side_gold = int((both["a1_sarc"] == both["gold_sarc"]).sum())
        # lead decision can equal gold if threshold differs from 0.5; count decisive items only
        dec = both.filter(pl.col("lead_sarc") != pl.col("gold_sarc"))
        print(f"{v} D: n={dd.height}, annotators agree on {both.height}; of those, lead!=gold on {dec.height}: "
              f"annotators side with lead {int((dec['a1_sarc'] == dec['lead_sarc']).sum())}, "
              f"with gold {int((dec['a1_sarc'] == dec['gold_sarc']).sum())} "
              f"(all agreed items: lead {side_lead}, gold {side_gold}); "
              f"gold-positive share in D {dd['gold_sarc'].mean():.2f}")


if __name__ == "__main__":
    main()
