#!/bin/bash
# Idempotent GPU-1 queue (safe to relaunch after a reboot: finished runs are skipped by ft.py).
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
cd /workspace/ALTA
for s in 0 1; do for r in R1 R4; do
  experiments/llm/ladder.sh $r $s >> logs/ladder_${r}_s$s.log 2>&1
done; done
for r in llm_q35_9b_zs llm_q35_9b_probe; do
  grep -q '"test_sha"' preds/$r.json 2>/dev/null && grep -q '"folds_measured": true\|n/a zero-shot' preds/$r.json && continue
  uv run python experiments/llm/predict_test.py --run $r --test experiments/llm/fake_test.csv >> logs/fixmeta9b.log 2>&1
done
