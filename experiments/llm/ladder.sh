#!/bin/bash
# Adapter ladder on holdout fold F (train on the other folds), Qwen3.5-4B bf16, all-linear, r=64.
#   ladder.sh RUNG SEED [FOLD]     RUNG in R1 R2 R2b R3 R4 R1pre A16 A64 A16ps
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
rung=$1; seed=$2; fold=${3:-0}
common="--model qwen3.5-4b --fold $fold --seed $seed --epochs 1 --bs 8 --accum 2 --lr 1e-4 --wd 0.01
        --dropout 0.05 --targets all-linear --sent-rows reddit --max-text-tokens 192 --save-run"
R64="--r 64 --rslora"
PSEUDO=${PSEUDO:-data/external/pseudo/fold$fold.parquet}   # teacher trained without this fold
UNS="--backend unsloth"; PY_UNS=/workspace/venv-unsloth/bin/python
PY="uv run python"
case $rung in
  R1)  extra="" ;;
  R1u)   PY=$PY_UNS; extra="$UNS" ;;                      # R1 through Unsloth (parity check)
  A16u)  PY=$PY_UNS; R64=""; extra="$UNS --r 16 --alpha 32 --head linear --head-lr 3e-4" ;;
  A16psu) PY=$PY_UNS; R64=""; extra="$UNS --r 16 --alpha 32 --head linear --head-lr 3e-4 --pseudo $PSEUDO --pseudo-n ${PSEUDO_N:-40000} --pseudo-tasks sarc" ;;
  # AstroCLIMB-style: bf16 LoRA r16/alpha32 + separate linear head (head lr 3e-4, LoRA 1e-4)
  A16)   R64=""; extra="--r 16 --alpha 32 --head linear --head-lr 3e-4" ;;
  A64)   extra="--head linear --head-lr 3e-4" ;;
  A16ps) R64=""; extra="--r 16 --alpha 32 --head linear --head-lr 3e-4 --pseudo $PSEUDO --pseudo-n ${PSEUDO_N:-40000} --pseudo-tasks sarc" ;;
  R2)  extra="--init eva" ;;
  R2b) extra="--init pissa_niter_16" ;;
  R3)  extra="--init eva --dora" ;;
  R4)  extra="--init eva --dora --riemannian" ;;
  R1pre) extra="--presft experiments/llm/cache/presft_v1.parquet" ;;  # R1 + sarcasm pre-SFT stage
  R1precheap) extra="--presft experiments/llm/cache/presft_v1.parquet --presft-limit 400 --epochs 0.25" ;;  # canary
esac
suffix=$([ "$fold" = "0" ] && echo "" || echo "_f$fold")
# TEST=path adds fold-model test predictions (fold<k>_test.parquet)
$PY experiments/llm/ft.py --run llm_ft_${rung}_s${seed}${suffix} $common $R64 $extra ${TEST:+--test $TEST}
