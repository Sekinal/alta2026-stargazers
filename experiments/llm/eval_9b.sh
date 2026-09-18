#!/bin/bash
# CPU: Qwen3.5-9B (NF4) zero-shot and probe variants, compared with the 4B runs.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
M="--model qwen3.5-9b-nf4 --sarc sarc_v2 --sent sent_v1"
uv run python experiments/llm/make_run.py --run llm_tmp_9b_zs --kind zs $M
for L in 0 1 2; do for C in 0.002 0.005; do
  uv run python experiments/llm/make_run.py --run llm_tmp_9b_L${L}_C$C --kind probe $M --layer $L --C $C
done; done
uv run python experiments/llm/auc.py run llm_q35_4b_zs llm_q35_4b_probe $(ls preds | grep llm_tmp_9b | grep parquet | sed 's/.parquet//')
uv run python experiments/llm/auc.py tag qwen3.5-9b-nf4/sarc_v1 qwen3.5-4b/sarc_v1
