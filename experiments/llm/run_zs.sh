#!/bin/bash
# Zero-shot scoring passes over all pool rows (resumable). Usage: run_zs.sh MODEL PROMPT [PROMPT...]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
model=$1; shift
for p in "$@"; do
  uv run python experiments/llm/score.py --model $model --prompt $p ${EXTRA:-}
done
