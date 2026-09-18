#!/usr/bin/env bash
# GPU-0 queue: pseudo-label adaptation with SARCASM soft labels only in stage 1 (fold-0 screen).
# Follow-up to queue_pseudo.sh: both-task pseudo-labels improved reddit sarcasm AUC but cost AU-reddit sentiment.
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ALTA
[ -f models/pseudo_sarc_f0_deb_l/config.json ] || \
  uv run python -m experiments.encoders.pretrain_pseudo --fold 0 --tasks sarc --out models/pseudo_sarc_f0_deb_l --max_minutes 90 \
  || { echo "STAGE FAILED: pretrain_pseudo sarc"; exit 1; }
for s in 13 14 15; do
  R=enc_pseudosarc_deb_l_s$s
  uv run python -m experiments.encoders.train --run $R --model models/pseudo_sarc_f0_deb_l --epochs 4 --lr 1e-5 --bs 16 \
    --optim adamw --folds 0 --seed $s \
    --desc "stage1 SARCASM-only soft pseudo-labels (~400k in-domain texts, fold-0 teacher = enc_deb_l_s13-15 fold0) -> deberta-v3-large gold FT, seed $s" \
    && uv run python -m experiments.encoders.train --run $R --collect --folds 0 || echo "STAGE FAILED: $R"
done
echo QUEUE_DONE
