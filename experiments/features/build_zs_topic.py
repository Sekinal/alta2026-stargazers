"""Builder for the zs_topic feature set (see build_zs_clf.py).

  flock .gpu.lock uv run python experiments/features/build_zs_topic.py --input pool|TEST.csv --out PARQUET
"""

import argparse

from build_zs_clf import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    build("zs_topic", a.input, a.out)
