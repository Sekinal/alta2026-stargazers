#!/usr/bin/env bash
# Fold-4 confirmation (pre-registered: experiments/prereg/fold4_pseudo_sarc.md), GPU-0 half.
# Same procedure as the fold-0 screen: ONE stage-1 student (seed 13, sarcasm-only soft BCE) on the
# fold-4 teacher's labels, then the unchanged gold fine-tune on folds 0-3 with seeds 13/14/15 (s15 on GPU 1).
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ALTA
P=data/external/pseudo
[ -f $P/fold4.parquet ] || {
  uv run python -m experiments.encoders.pseudo_label --fold 4 --runs enc_deb_l_s13 enc_deb_l_s14 --part a || exit 1
  while [ ! -f $P/fold4.part-b.parquet ]; do sleep 30; done
  uv run python -m experiments.encoders.pseudo_label --fold 4 --merge a b || exit 1
}
uv run python -c "
import polars as pl
a, b = pl.read_parquet('$P/fold0.parquet'), pl.read_parquet('$P/fold4.parquet')
assert a['text'].equals(b['text']), 'fold4 sample differs from fold0'
print('fold4.parquet: same 400k texts as fold0.parquet; teacher:', b['teacher'][0])
print('corr p_sarc fold0 vs fold4 teacher:', pl.DataFrame({'a': a['p_sarc'], 'b': b['p_sarc']}).corr()[0, 1])
"
echo F4_LABELS_DONE
[ -f models/pseudo_sarc_f4_deb_l/config.json ] || \
  uv run python -m experiments.encoders.pretrain_pseudo --fold 4 --tasks sarc --out models/pseudo_sarc_f4_deb_l --max_minutes 90 \
  || { echo "STAGE FAILED: stage1"; exit 1; }
for s in 13 14; do
  R=enc_pseudosarc_deb_l_f4_s$s
  uv run python -m experiments.encoders.train --run $R --model models/pseudo_sarc_f4_deb_l --epochs 4 --lr 1e-5 --bs 16 \
    --optim adamw --folds 4 --seed $s \
    --desc "fold-4 confirmation: stage1 SARCASM-only soft pseudo-labels (fold-4 teacher = enc_deb_l_s13-15 fold4) -> gold FT on folds 0-3, seed $s" \
    && uv run python -m experiments.encoders.train --run $R --collect --folds 4 || echo "STAGE FAILED: $R"
done
echo QUEUE_DONE
