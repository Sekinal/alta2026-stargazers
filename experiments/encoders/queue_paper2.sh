#!/usr/bin/env bash
# Paper analyses (remote, 1 GPU): deberta learning curves and architecture ablations, all 5 folds (full OOF),
# enc_deb_l recipe unless stated. bash experiments/encoders/queue_paper2.sh
cd /workspace/ALTA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FT="--model microsoft/deberta-v3-large --epochs 4 --lr 1e-5 --bs 16 --optim adamw --no_robust"
run() {  # name seed desc args...
  local R=$1 s=$2 d=$3; shift 3
  uv run python -m experiments.encoders.train --run $R $FT --seed $s --desc "$d, seed $s" "$@" \
    && uv run python -m experiments.encoders.train --run $R --collect || echo "STAGE FAILED: $R"
}
for s in 13 14; do
  for f in 25 50 75; do
    run enc_deb_l_lc${f}_s$s $s "learning curve: enc_deb_l recipe on a nested segment-stratified ${f}% of the training folds" --train_frac 0.$f
  done
done
for s in 13 14; do
  run enc_deb_l_nopfx_s$s $s "ablation: no [variety] [source] input prefix" --no_prefix
  run enc_deb_l_sentonly_s$s $s "ablation: sentiment loss only (sarcasm head untrained)" --sarc_w 0
  run enc_deb_l_sarconly_s$s $s "ablation: sarcasm loss only (sentiment head untrained)" --sent_w 0
  run enc_deb_l_shsarc_s$s $s "ablation: one sarcasm head shared by both varieties" --shared_sarc
done
echo QUEUE_DONE
