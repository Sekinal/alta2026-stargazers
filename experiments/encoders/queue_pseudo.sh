#!/usr/bin/env bash
# GPU-0 queue: pseudo-label adaptation, fold-0 screen (remote).
#   A: fold-0 teacher (fold-0 checkpoints of enc_deb_l_s13/14/15, never saw fold 0) labels ~400k in-domain texts
#   B: stage 1 = student on the soft labels (<= 90 min); stage 2 = unchanged gold fine-tune on folds 1-4, seeds 13/14/15
#   scripts/remote_run.sh enc_pseudo "bash experiments/encoders/queue_pseudo.sh"
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ALTA
[ -f data/external/pseudo/fold0.parquet ] || \
  uv run python -m experiments.encoders.pseudo_label --fold 0 --runs enc_deb_l_s13 enc_deb_l_s14 enc_deb_l_s15 \
  || { echo "STAGE FAILED: pseudo_label"; exit 1; }
echo PSEUDO_LABELS_DONE
[ -f models/pseudo_f0_deb_l/config.json ] || \
  uv run python -m experiments.encoders.pretrain_pseudo --fold 0 --out models/pseudo_f0_deb_l --max_minutes 90 \
  || { echo "STAGE FAILED: pretrain_pseudo"; exit 1; }
for s in 13 14 15; do
  R=enc_pseudo_deb_l_s$s
  uv run python -m experiments.encoders.train --run $R --model models/pseudo_f0_deb_l --epochs 4 --lr 1e-5 --bs 16 \
    --optim adamw --folds 0 --seed $s \
    --desc "stage1 soft pseudo-labels (~400k in-domain texts, fold-0 teacher = enc_deb_l_s13-15 fold0) -> deberta-v3-large gold FT, seed $s" \
    && uv run python -m experiments.encoders.train --run $R --collect --folds 0 || echo "STAGE FAILED: $R"
done
echo QUEUE_DONE
