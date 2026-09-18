#!/bin/bash
# Pass 1: zero-shot sarcasm (strict + variety-specific) and sentiment on all pool rows.
cd /home/ieqr/Desktop/research/ALTA
for p in sarc_v1 sent_v1 sarc_v2; do
  flock .gpu.lock uv run python experiments/llm/score.py --model qwen3-8b --prompt $p --max-minutes 80 2>&1 \
    | grep --line-buffered -v -E "Loading weights|warning|include|define|note:|^ *\||\^"
done
