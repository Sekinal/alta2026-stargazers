#!/usr/bin/env bash
# Rebuild the stk_v5 lead's GPU members on a fresh box (checkpoints were lost with the old box).
# Same recipe as enc_deb_l_s13..17, new run names (enc_deb_l_r13..17) so the original OOF stays for comparison.
#   bash experiments/encoders/queue_retrain.sh data/test.csv
cd /workspace/ALTA
TEST=${1:-data/test.csv}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
set -o pipefail
uv run python experiments/llm/predict_test.py --run llm_q35_9b_probe --test $TEST || echo "STAGE FAILED: llm probe"
uv run python experiments/classical/embed.py --test $TEST e5_large me5_sarc || echo "STAGE FAILED: embed"
FT="--model microsoft/deberta-v3-large --epochs 4 --lr 1e-5 --bs 16 --optim adamw --no_robust --save"
for s in 13 14 15 16 17; do
  R=enc_deb_l_r$s
  uv run python -m experiments.encoders.train --run $R $FT --seed $s \
    --desc "deberta-v3-large multitask (sent + per-variety sarc heads), segment prefix, CLS, 4 epochs, seed $s (retrain of enc_deb_l_s$s)" \
    && uv run python -m experiments.encoders.train --run $R --collect \
    && uv run python experiments/encoders/predict_test.py --run $R --test $TEST || echo "STAGE FAILED: $R"
done
echo QUEUE_DONE
