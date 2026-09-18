#!/bin/bash
# CPU: multi-prompt probe variants on cached Qwen3.5-4B features.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
for C in 0.002 0.005; do
  uv run python experiments/llm/make_run.py --run llm_tmp_v12_C$C --kind probe --model qwen3.5-4b \
    --sarc sarc_v2+sarc_v1 --sent sent_v1+sarc_v2 --layer 1 --C $C > /dev/null
done
uv run python experiments/llm/auc.py run llm_q35_4b_probe $(ls preds | grep llm_tmp_ | grep parquet | sed 's/.parquet//')
