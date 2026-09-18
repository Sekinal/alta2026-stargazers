#!/bin/bash
# GPU worker over a shared job list (claims jobs atomically with mkdir; safe to run one per GPU
# and to relaunch after a reboot: finished work is skipped by ft.py / robust.py).
#   CUDA_VISIBLE_DEVICES=0 experiments/llm/worker.sh experiments/llm/jobs_cv.txt
# Job lines: "RUNG SEED FOLD" | "PREDTEST RUN FOLD" | "ROBUST RUN" | "CANARY RUNG SEED FOLD" | "CMD ..."
cd /workspace/ALTA
export PYTHONUNBUFFERED=1 TEST=experiments/llm/fake_test.csv
jobs=$1; locks=experiments/llm/cache/locks; mkdir -p $locks logs
while read -r line; do
  [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
  key=$(echo "$line" | tr ' /' '__')
  mkdir $locks/$key 2>/dev/null || continue          # claimed by another worker (or done)
  echo "$(date -u +%H:%M) gpu$CUDA_VISIBLE_DEVICES start $line"
  set -- $line
  case $1 in
    PREDTEST) uv run python experiments/llm/ft.py --run $2 --model qwen3.5-4b --fold $3 --predict-only \
                --test $TEST --max-text-tokens 192 >> logs/predtest_$2_$3.log 2>&1 ;;
    ROBUST)   uv run python experiments/llm/robust.py --run $2 >> logs/robust_$2.log 2>&1 ;;
    CANARY)   ALTA_CANARY_SEED=0 experiments/llm/ladder.sh $2 $3 $4 >> logs/canary_$2_s$3_f$4.log 2>&1 ;;
    CMD)      shift; eval "$@" >> logs/worker_cmd.log 2>&1 ;;
    *)        experiments/llm/ladder.sh $1 $2 $3 >> logs/ladder_$1_s$2_f$3.log 2>&1 ;;
  esac
  rc=$?
  echo "$(date -u +%H:%M) gpu$CUDA_VISIBLE_DEVICES done rc=$rc $line"
  [ $rc -ne 0 ] && rmdir $locks/$key                  # failed: let it be retried
done < "$jobs"
