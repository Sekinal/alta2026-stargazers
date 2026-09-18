#!/usr/bin/env bash
# Fold-4 confirmation, GPU-1 half: third teacher for fold4.parquet, fold-1/2 labels for a possible 5-fold
# while GPU 0 trains the stage-1 student, then the seed-15 gold fine-tune.
export CUDA_VISIBLE_DEVICES=1
cd /workspace/ALTA
P=data/external/pseudo
[ -f $P/fold4.parquet ] || uv run python -m experiments.encoders.pseudo_label --fold 4 --runs enc_deb_l_s15 --part b || exit 1
for k in 1 2; do
  [ -f $P/fold$k.parquet ] || uv run python -m experiments.encoders.pseudo_label --fold $k \
    --runs enc_deb_l_s13 enc_deb_l_s14 enc_deb_l_s15 || echo "STAGE FAILED: labels fold$k"
done
while [ ! -f models/pseudo_sarc_f4_deb_l/stage1_config.json ]; do sleep 30; done  # written last
sleep 10
R=enc_pseudosarc_deb_l_f4_s15
uv run python -m experiments.encoders.train --run $R --model models/pseudo_sarc_f4_deb_l --epochs 4 --lr 1e-5 --bs 16 \
  --optim adamw --folds 4 --seed 15 \
  --desc "fold-4 confirmation: stage1 SARCASM-only soft pseudo-labels (fold-4 teacher = enc_deb_l_s13-15 fold4) -> gold FT on folds 0-3, seed 15" \
  && uv run python -m experiments.encoders.train --run $R --collect --folds 4 || echo "STAGE FAILED: $R"
echo QUEUE_DONE
