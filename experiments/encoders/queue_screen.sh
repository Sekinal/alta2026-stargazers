#!/usr/bin/env bash
# Fold-0 screens (seeds 13-15, paired with enc_deb_l_s13..15) of generalization arms, then the lead retrain.
#   bash experiments/encoders/queue_screen.sh
cd /workspace/ALTA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BASE="--epochs 4 --bs 16 --optim adamw --no_robust --folds 0"
DEB=microsoft/deberta-v3-large
declare -A ARM=(
  [ema]="--model $DEB --lr 1e-5 --ema 0.999"
  [llrdema]="--model $DEB --lr 1.5e-5 --llrd 0.9 --ema 0.999"
  [nli]="--model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --lr 1e-5"
  [electra]="--model google/electra-large-discriminator --lr 1e-5"
  [mbert]="--model answerdotai/ModernBERT-large --lr 2e-5"
)
for arm in ema llrdema nli electra mbert; do
  for s in 13 14 15; do
    R=enc_scr_${arm}_s$s
    uv run python -m experiments.encoders.train --run $R $BASE ${ARM[$arm]} --seed $s \
      --desc "fold-0 screen arm $arm (${ARM[$arm]}), seed $s" \
      && uv run python -m experiments.encoders.train --run $R --folds 0 --collect || echo "STAGE FAILED: $R"
  done
done
echo SCREEN_DONE
bash experiments/encoders/queue_retrain.sh data/test.csv
