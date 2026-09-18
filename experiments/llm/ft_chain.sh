#!/bin/bash
# Run ft.py for the given folds sequentially after PID (if given) exits. Finished folds are skipped.
#   ft_chain.sh RUN WAIT_PID "FOLDS" [extra ft.py args]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
run=$1; wait_pid=$2; folds=$3; shift 3
while [ -n "$wait_pid" ] && kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
for k in $folds; do
  uv run python experiments/llm/ft.py --run $run --fold $k "$@"
done
