#!/bin/bash
# Re-save kept runs with folds_used / zero-shot canary meta and fake-test predictions (cached scores).
export OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TEST=${1:-experiments/llm/fake_test.csv}
for r in llm_q35_4b_zs llm_q35_4b_probe llm_q35_4b_probe_L1 llm_q35_4b_probe_L2 llm_q35_9b_zs llm_q35_9b_probe; do
  uv run python experiments/llm/predict_test.py --run $r --test $TEST
  uv run python -m alta.validate check $r
done
