#!/bin/bash
# Build every zero-shot feature set (fe-models lane) for one input: 'pool' or a test CSV.
# Outputs go to experiments/features/store/<set>__{pool|test-<fingerprint>}.parquet.
#   bash experiments/features/build_zs_all.sh data/test.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
IN=${1:-pool}
for b in zs_irony zs_affect zs_tox zs_topic zs_nli zs_nli_bart zs_nli_x zs_sentlvl zs_lm; do
  [ -f experiments/features/build_$b.py ] || continue
  flock .gpu.lock uv run python -u experiments/features/build_$b.py --input "$IN"
done
