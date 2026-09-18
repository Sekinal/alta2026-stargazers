"""Attack 2b: degenerate answers under the real Codabench scorer image vs our metric.

Builds answers from fake/truth.csv (all-0, all-1, one-class per variety, float labels,
random) and compares docker (py3.7 / pandas 0.24 / sklearn 0.21) scores with metrics.components.

Usage: PYTHONPATH=. uv run python redteam/software/scorer_degenerate.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from scorer_compat import score  # noqa: E402

from alta.validation.metrics import macro_f1  # noqa: E402

HERE = Path(__file__).parent
TRUTH = HERE / "fake/truth.csv"


def ours(ans: pl.DataFrame, truth: pl.DataFrame) -> float:
    c = {}
    for t in ("sentiment", "sarcasm"):
        for v in ("en-AU", "en-UK"):
            m = (truth["variety"] == v).to_numpy()
            c[t, v] = macro_f1(truth[t].to_numpy()[m], ans[t].to_numpy()[m].astype(float) > 0.5)
    return (min(c["sentiment", "en-AU"], c["sentiment", "en-UK"]) + min(c["sarcasm", "en-AU"], c["sarcasm", "en-UK"])) / 2


def main():
    truth = pl.read_csv(TRUTH)
    rng = np.random.default_rng(0)
    uk = truth["variety"] == "en-UK"
    variants = {
        "all0": truth.with_columns(pl.lit(0).alias("sentiment"), pl.lit(0).alias("sarcasm")),
        "all1": truth.with_columns(pl.lit(1).alias("sentiment"), pl.lit(1).alias("sarcasm")),
        "uk_sarc_all0": truth.with_columns(pl.when(uk).then(0).otherwise(pl.col("sarcasm")).alias("sarcasm")),
        "random": truth.with_columns(pl.Series("sentiment", rng.integers(0, 2, truth.height)),
                                     pl.Series("sarcasm", rng.integers(0, 2, truth.height))),
        "float_labels": truth.with_columns(pl.col("sentiment", "sarcasm").cast(pl.Float64)),
        "perfect": truth,
    }
    out = HERE / "out_degenerate"
    out.mkdir(exist_ok=True)
    for name, a in variants.items():
        p = out / f"{name}.csv"
        a.write_csv(p)
        print(f"{name:14} ours {ours(a, truth):.4f} | docker {score('docker_py37', p, TRUTH)} | "
              f"modern {score('modern', p, TRUTH)}", flush=True)


if __name__ == "__main__":
    main()
