#!/bin/bash
# Adapter ladder on holdout fold F (train on the other folds), Qwen3.5-4B bf16, all-linear, r=64.
#   ladder.sh RUNG SEED [FOLD]     RUNG in R1 R2 R2b R3 R4 R1pre
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
rung=$1; seed=$2; fold=${3:-0}
common="--model qwen3.5-4b --fold $fold --seed $seed --epochs 1 --bs 8 --accum 2 --lr 1e-4 --wd 0.01
        --dropout 0.05 --r 64 --rslora --targets all-linear --sent-rows reddit --max-text-tokens 192 --save-run"
case $rung in
  R1)  extra="" ;;
  R2)  extra="--init eva" ;;
  R2b) extra="--init pissa_niter_16" ;;
  R3)  extra="--init eva --dora" ;;
  R4)  extra="--init eva --dora --riemannian" ;;
  R1pre) extra="--presft experiments/llm/cache/presft_v1.parquet" ;;  # R1 + sarcasm pre-SFT stage
  R1precheap) extra="--presft experiments/llm/cache/presft_v1.parquet --presft-limit 400 --epochs 0.25" ;;  # canary
esac
suffix=$([ "$fold" = "0" ] && echo "" || echo "_f$fold")
# TEST=path adds fold-model test predictions (fold<k>_test.parquet)
uv run python experiments/llm/ft.py --run llm_ft_${rung}_s${seed}${suffix} $common $extra ${TEST:+--test $TEST}
