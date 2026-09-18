#!/bin/bash
# GPU 1 chain: after A16psu s0, context-aware 9B probe (scores, runs, canary, fake test), then A16psu s1
# (s1 first if the context file is not there yet).
cd /workspace/ALTA
export PATH=$HOME/.local/bin:$PATH CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
while pgrep -f "[l]lm_ft_A16psu_s0" >/dev/null; do sleep 30; done
S1_DONE=0
if [ ! -f data/external/context/pool_context.parquet ]; then   # context not ready: s1 first
  experiments/llm/ladder.sh A16psu 1 0 >> logs/ladder_A16psu_s1_f0.log 2>&1; S1_DONE=1
  while [ ! -f data/external/context/pool_context.parquet ]; do sleep 60; done
fi
for p in sarc_v2c sent_v1c; do
  uv run python experiments/llm/score.py --model qwen3.5-9b-nf4 --prompt $p --max-batch-tokens 8000
done
M="--model qwen3.5-9b-nf4 --sarc sarc_v2c --sent sent_v1c --layer 1 --C 0.002"
D="Qwen3.5-9B NF4 probe as llm_q35_9b_probe, prompts extended with the recovered Reddit thread context (subreddit, title, body/parent) for matched rows"
uv run python experiments/llm/make_run.py --run llm_q35_9b_ctx_probe --kind probe $M --desc "$D"
for s in 0 1 2; do ALTA_CANARY_SEED=$s uv run python experiments/llm/make_run.py --run llm_q35_9b_ctx_probe --kind probe $M --desc "$D"; done
uv run python -m alta.validate canary llm_q35_9b_ctx_probe
uv run python experiments/llm/predict_test.py --run llm_q35_9b_ctx_probe --test experiments/llm/fake_test.csv \
  --context data/external/context/valid_context_standin.parquet
uv run python -m alta.validate check llm_q35_9b_ctx_probe
uv run python -m alta.validate compare llm_q35_9b_ctx_probe llm_q35_9b_probe
uv run python experiments/llm/ctx_report.py llm_q35_9b_probe llm_q35_9b_ctx_probe
[ $S1_DONE = 1 ] || experiments/llm/ladder.sh A16psu 1 0 >> logs/ladder_A16psu_s1_f0.log 2>&1
