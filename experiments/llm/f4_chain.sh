#!/bin/bash
# GPU 1: after ctx_chain (ctx probe + A16psu s1), the pre-registered fold-4 confirmation of A16psu s0.
cd /workspace/ALTA
export PATH=$HOME/.local/bin:$PATH CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
while pgrep -f "bash experiments/llm/[c]tx_chain.sh" >/dev/null; do sleep 30; done
while [ ! -f data/external/pseudo/fold4.parquet ]; do sleep 60; done
TEST=experiments/llm/fake_test.csv experiments/llm/ladder.sh A16psu 0 4 >> logs/ladder_A16psu_s0_f4.log 2>&1
