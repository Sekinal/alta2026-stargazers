#!/bin/bash
# After PID exits: 9B fake-test predictions + meta refresh.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
while kill -0 $1 2>/dev/null; do sleep 30; done
for r in llm_q35_9b_zs llm_q35_9b_probe; do
  uv run python experiments/llm/predict_test.py --run $r --test experiments/llm/fake_test.csv
  uv run python -m alta.validate check $r
done
