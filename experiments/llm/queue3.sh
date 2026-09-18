#!/bin/bash
# Replaces queue.sh/queue2.sh: finish R1 s1 (running), R2 x2, pre-SFT R1pre x2, robustness, EVA tau probe.
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
cd /workspace/ALTA
while pgrep -f "[f]t.py --run llm_ft_R1_s1" >/dev/null; do sleep 30; done
experiments/llm/ladder.sh R1 1 >> logs/ladder_R1_s1.log 2>&1   # no-op if finished
for s in 0 1; do experiments/llm/ladder.sh R2 $s >> logs/ladder_R2_s$s.log 2>&1; done
if [ ! -f experiments/llm/out/llm_tmp_presft_smoke/fold0.parquet ]; then
  uv run python experiments/llm/ft.py --run llm_tmp_presft_smoke --model qwen3.5-4b --fold 0 --r 64 --rslora \
    --presft experiments/llm/cache/presft_v1.parquet --presft-limit 64 --limit-train 64 >> logs/presft_smoke.log 2>&1
fi
for s in 0 1; do experiments/llm/ladder.sh R1pre $s >> logs/ladder_R1pre_s$s.log 2>&1; done
[ -f experiments/llm/out/llm_ft_R1_s0/robust_fold0.parquet ] || \
  uv run python experiments/llm/robust.py --run llm_ft_R1_s0 >> logs/robust_R1_s0.log 2>&1
# EVA convergence probe at tau=0.95 (init only; tiny training)
[ -f experiments/llm/out/llm_tmp_eva_tau95/fold0_stats.json ] || \
  uv run python experiments/llm/ft.py --run llm_tmp_eva_tau95 --model qwen3.5-4b --fold 0 --r 64 --rslora \
    --targets all-linear --init eva --eva-tau 0.95 --limit-train 16 --sent-rows reddit >> logs/eva_tau95.log 2>&1
