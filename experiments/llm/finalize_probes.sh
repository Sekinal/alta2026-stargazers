#!/bin/bash
# Rebuild the kept Qwen3.5-4B runs with test predictions for TEST, then 3-seed permutation canaries.
#   CUDA_VISIBLE_DEVICES=1 experiments/llm/finalize_probes.sh experiments/llm/fake_test.csv
set -e
TEST=$1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
M="--model qwen3.5-4b --sarc sarc_v2 --sent sent_v1"
# restore the C=0.05 probes other agents stacked on (identical to the originals)
uv run python experiments/llm/make_run.py --run llm_q35_4b_probe_L1 --kind probe $M --layer 1 --C 0.05 \
  --desc "Qwen3.5-4B cross-fitted LR probe (C=0.05) on layer-24 hidden state + margin (sarc_v2/sent_v1)"
uv run python experiments/llm/make_run.py --run llm_q35_4b_probe_L2 --kind probe $M --layer 2 --C 0.05 \
  --desc "Qwen3.5-4B cross-fitted LR probe (C=0.05) on layer-31 hidden state + margin (sarc_v2/sent_v1)"
for r in llm_q35_4b_zs llm_q35_4b_probe llm_q35_4b_probe_L1 llm_q35_4b_probe_L2; do
  uv run python experiments/llm/predict_test.py --run $r --test $TEST
done
for s in 0 1 2; do
  ALTA_CANARY_SEED=$s uv run python experiments/llm/make_run.py --run llm_q35_4b_probe --kind probe $M --layer 1 --C 0.005
  ALTA_CANARY_SEED=$s uv run python experiments/llm/make_run.py --run llm_q35_4b_probe_L1 --kind probe $M --layer 1 --C 0.05
  ALTA_CANARY_SEED=$s uv run python experiments/llm/make_run.py --run llm_q35_4b_probe_L2 --kind probe $M --layer 2 --C 0.05
done
for r in llm_q35_4b_probe llm_q35_4b_probe_L1 llm_q35_4b_probe_L2; do
  uv run python -m alta.validate canary $r
done
for r in llm_q35_4b_zs llm_q35_4b_probe llm_q35_4b_probe_L1 llm_q35_4b_probe_L2; do
  uv run python -m alta.validate check $r
done
