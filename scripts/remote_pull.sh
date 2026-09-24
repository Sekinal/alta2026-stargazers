#!/usr/bin/env bash
# Pull outputs remote -> local: run predictions, results, experiment caches/logs/out.
# Model checkpoints are NOT pulled here (large); use remote_pull_models.sh RUN.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/remote.env
SSH="ssh -p $REMOTE_PORT -o ConnectTimeout=15 -o ServerAliveInterval=10"
fails=()
rsync -azu -e "$SSH" "$REMOTE_HOST:$REMOTE_DIR/preds/" preds/ 2>/dev/null || fails+=("preds")
# runs saved on the laptop go up too (newest file wins in both directions)
rsync -azu -e "$SSH" preds/ "$REMOTE_HOST:$REMOTE_DIR/preds/" 2>/dev/null || fails+=("preds")
rsync -az -e "$SSH" --exclude="remote/" "$REMOTE_HOST:$REMOTE_DIR/results/" results/remote/ 2>/dev/null || fails+=("results")
rsync -az -e "$SSH" --include='*/' --include='cache/***' --include='logs/***' --include='out/***' \
  --exclude='*' "$REMOTE_HOST:$REMOTE_DIR/experiments/" experiments/ 2>/dev/null || fails+=("experiments")
if [[ ${#fails[@]} -eq 0 ]]; then echo "$(date +%H:%M:%S) pulled"; else echo "$(date +%H:%M:%S) PULL FAILED: ${fails[*]}" >&2; exit 1; fi
