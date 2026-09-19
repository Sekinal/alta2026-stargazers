#!/usr/bin/env bash
# Paper runs (remote). Usage: bash experiments/encoders/queue_paper.sh transfer|thread
# transfer (GPU 0): deberta trained on ONE variety's rows (standard folds), predicting all held-out rows.
# thread   (GPU 1): the enc_deb_l config under thread-grouped folds (data/external/context/thread_folds.parquet).
cd /workspace/ALTA
FT="--model microsoft/deberta-v3-large --epochs 4 --lr 1e-5 --bs 16 --optim adamw --no_robust"
if [ "$1" = transfer ]; then
  export CUDA_VISIBLE_DEVICES=0
  for v in AU UK; do for s in 13 14 15; do
    R=enc_deb_l_train${v}_s$s
    uv run python -m experiments.encoders.train --run $R $FT --seed $s --train_variety en-$v \
      --desc "transfer matrix: enc_deb_l config trained on en-$v rows only, predicting all held-out rows, seed $s" \
      && uv run python -m experiments.encoders.train --run $R --collect || echo "STAGE FAILED: $R"
  done; done
else
  export CUDA_VISIBLE_DEVICES=1
  while [ ! -f data/external/context/thread_folds.parquet ]; do sleep 60; done
  for s in 13 14 15; do
    R=enc_deb_l_thr_s$s
    uv run python -m experiments.encoders.train --run $R $FT --seed $s \
      --folds_file data/external/context/thread_folds.parquet --folds_col thread_fold \
      --desc "enc_deb_l config under THREAD-GROUPED folds (thread_folds.parquet), seed $s" \
      && uv run python -m experiments.encoders.train --run $R --collect || echo "STAGE FAILED: $R"
  done
fi
echo QUEUE_DONE
