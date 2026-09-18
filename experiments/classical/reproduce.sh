#!/usr/bin/env bash
# Reproduce the classical lane's final runs (OOF), their canaries, and test predictions.
#   experiments/classical/reproduce.sh [TEST_CSV]
# GPU steps (embed.py) need ~10 GB for Qwen3-Embedding-4B; everything else is CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/classical
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

# 1. frozen features for the pool (cached in $C/cache)
uv run python $C/embed.py e5_large me5_sarc qwen3_4b_sarc qwen3_4b_sent tw_sent tw_irony go_emo

# 2. members (OOF on the project folds, hyperparameters by inner CV)
probe() { uv run python $C/probe.py --emb "$1" --sent pooled --sarc segment --run "$2"; }
members() {
  probe e5_large+me5_sarc cls_e5me5_lr
  probe qwen3_4b_sarc cls_qwen3_4b_sarc_lr
  probe qwen3_4b_sent cls_qwen3_4b_sent_lr
  probe tw_sent cls_tw_sent_lr
  uv run python $C/tfidf.py --model lr --sent pooled --sarc variety --run cls_tfidf_lr  # builds the fold TF-IDF cache used by gbdt
  uv run python $C/gbdt.py --algo lgbm --emb e5_large --run cls_gbdt_lgbm
}
M5="cls_e5me5_lr cls_qwen3_4b_sarc_lr cls_qwen3_4b_sent_lr cls_tw_sent_lr cls_gbdt_lgbm"
stackit() { uv run python $C/stack.py --run stk_cls5 --l2 lr --C 0.1 --xtask $M5; }

# 3. real runs, then 3 permutation-canary seeds of the whole chain (members + stack)
members; stackit
for s in 0 1 2; do export ALTA_CANARY_SEED=$s; members; stackit; done
unset ALTA_CANARY_SEED
for r in $M5 stk_cls5; do uv run python -m alta.validate canary $r; uv run python -m alta.validate report $r > /dev/null; done

# 4. test predictions (fold-averaged members, then fold-averaged stacker)
if [[ $# -ge 1 ]]; then
  uv run python $C/predict_test.py --run stk_cls5 --test "$1"
  uv run python -m alta.validate check stk_cls5
fi
