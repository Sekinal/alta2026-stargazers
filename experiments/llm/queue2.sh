#!/bin/bash
# After queue.sh: pre-SFT smoke, R1pre x 2 seeds, robustness of R1 s0. Idempotent.
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
cd /workspace/ALTA
while pgrep -f "[q]ueue.sh" >/dev/null; do sleep 30; done
if [ ! -f experiments/llm/out/llm_tmp_presft_smoke/fold0.parquet ]; then
  uv run python experiments/llm/ft.py --run llm_tmp_presft_smoke --model qwen3.5-4b --fold 0 --r 64 --rslora \
    --presft experiments/llm/cache/presft_v1.parquet --presft-limit 64 --limit-train 64 >> logs/presft_smoke.log 2>&1 || exit 1
fi
for s in 0 1; do experiments/llm/ladder.sh R1pre $s >> logs/ladder_R1pre_s$s.log 2>&1; done
[ -f experiments/llm/out/llm_ft_R1_s0/robust_fold0.parquet ] || \
  uv run python experiments/llm/robust.py --run llm_ft_R1_s0 >> logs/robust_R1_s0.log 2>&1
