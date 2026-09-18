#!/usr/bin/env bash
# GPU-0 queue for the pretraining screens (remote). Each stage is independent: a failure is logged
# and the queue moves on. Screens: fold 0 held out (train on folds 1-4), seeds 13/14/15, same
# stage-2 config as enc_deb_l (4 epochs, lr 1e-5, bs 16), compared with enc_deb_l_s13/14/15 fold 0.
#   scripts/remote_run.sh enc_queue "bash experiments/encoders/queue_pretrain.sh"
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ALTA
while pgrep -f "encoders.[t]rain" > /dev/null; do sleep 30; done

FT="--epochs 4 --lr 1e-5 --bs 16 --optim adamw --folds 0"
screen() {  # screen NAME INIT_MODEL DESC
  for s in 13 14 15; do
    uv run python -m experiments.encoders.train --run "$1_s$s" --model "$2" $FT --seed $s --desc "$3, seed $s" \
      && uv run python -m experiments.encoders.train --run "$1_s$s" --collect --folds 0 \
      || echo "STAGE FAILED: $1_s$s"
  done
}

# (ii) sarcasm pre-FT -> FT
[ -f models/pft_sarc_deb_l/config.json ] || \
  uv run python -m experiments.encoders.pretrain_sarc --out models/pft_sarc_deb_l || echo "STAGE FAILED: pft_sarc"
[ -f models/pft_sarc_deb_l/config.json ] && screen enc_pftsarc_deb_l models/pft_sarc_deb_l \
  "stage1 sarcasm pre-FT (reddit /s UK+AU + gold irony sets, 1 epoch) -> deberta-v3-large stage-2 FT"

# (i) DAPT -> FT
[ -f models/dapt_deb_l/config.json ] || \
  uv run python -m experiments.encoders.dapt --out models/dapt_deb_l --tokens 40e6 --max_minutes 90 || echo "STAGE FAILED: dapt"
[ -f models/dapt_deb_l/config.json ] && screen enc_dapt_deb_l models/dapt_deb_l \
  "stage0 DAPT (MLM, in-domain AU/UK reddit+reviews) -> deberta-v3-large stage-2 FT"

# (iii) DAPT -> sarcasm pre-FT -> FT
[ -f models/dapt_deb_l/config.json ] && { [ -f models/dapt_pft_sarc_deb_l/config.json ] || \
  uv run python -m experiments.encoders.pretrain_sarc --model models/dapt_deb_l --out models/dapt_pft_sarc_deb_l \
  || echo "STAGE FAILED: dapt_pft_sarc"; }
[ -f models/dapt_pft_sarc_deb_l/config.json ] && screen enc_daptpft_deb_l models/dapt_pft_sarc_deb_l \
  "stage0 DAPT -> stage1 sarcasm pre-FT -> deberta-v3-large stage-2 FT"
echo QUEUE_DONE
