#!/bin/bash
# CPU: final Qwen3.5-9B (NF4) probe run + validation + 3-seed canary.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
M="--model qwen3.5-9b-nf4 --sarc sarc_v2 --sent sent_v1 --layer 1 --C 0.002"
D="Qwen3.5-9B NF4: cross-fitted LR probe (C=0.002) on layer-24 last-token hidden state + verbalizer margin of sarc_v2/sent_v1 prompts; sarcasm per variety"
uv run python experiments/llm/make_run.py --run llm_q35_9b_probe --kind probe $M --desc "$D"
uv run python experiments/llm/make_run.py --run llm_q35_9b_zs --kind zs --model qwen3.5-9b-nf4 --sarc sarc_v2 --sent sent_v1 \
  --desc "Qwen3.5-9B NF4 zero-shot verbalizer probabilities (sarc_v2 / sent_v1)"
for s in 0 1 2; do ALTA_CANARY_SEED=$s uv run python experiments/llm/make_run.py --run llm_q35_9b_probe --kind probe $M --desc "$D"; done
uv run python -m alta.validate canary llm_q35_9b_probe
uv run python -m alta.validate report llm_q35_9b_probe
uv run python -m alta.validate compare llm_q35_9b_probe llm_q35_4b_probe
uv run python -m alta.validate compare llm_q35_9b_probe emb_ens3_lr
