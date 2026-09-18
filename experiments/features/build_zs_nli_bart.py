"""Builder for zs_nli_bart: the zs_nli hypotheses scored by a second NLI model, facebook/bart-large-mnli
(MNLI only; different architecture and pretraining), for robustness to the NLI model.

  flock .gpu.lock uv run python experiments/features/build_zs_nli_bart.py --input pool|TEST.csv --out PARQUET
"""

import argparse

from build_zs_nli import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build(a.input, a.out, name="zs_nli_bart", model_id="facebook/bart-large-mnli")
