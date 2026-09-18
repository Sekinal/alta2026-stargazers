#!/bin/bash
# CPU: probe variants on cached Qwen3.5-4B hidden states (temporary runs llm_tmp_*).
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
for L in 0 1 2; do for C in 0.002 0.005 0.01; do
  uv run python experiments/llm/make_run.py --run llm_tmp_L${L}_C$C --kind probe --model qwen3.5-4b \
    --sarc sarc_v2 --sent sent_v1 --layer $L --C $C > /dev/null
done; done
uv run python experiments/llm/auc.py run $(ls preds | grep llm_tmp_ | grep parquet | sed 's/.parquet//')
