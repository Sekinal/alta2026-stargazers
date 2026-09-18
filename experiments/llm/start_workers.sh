#!/bin/bash
# GPU1 worker starts when queue3 is finished; GPU0 worker when GPU0 has been idle (<2 GB) for 2 min.
cd /workspace/ALTA
export PATH=$HOME/.local/bin:$PATH
( while pgrep -f "[q]ueue3.sh" >/dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=1 experiments/llm/worker.sh experiments/llm/jobs_cv.txt >> logs/worker_gpu1.log 2>&1 ) &
( idle=0; while [ $idle -lt 4 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$used" -lt 2000 ] && idle=$((idle+1)) || idle=0; sleep 30; done
  CUDA_VISIBLE_DEVICES=0 experiments/llm/worker.sh experiments/llm/jobs_cv.txt >> logs/worker_gpu0.log 2>&1 ) &
wait
