#!/usr/bin/env bash
# Push code (never outputs) local -> remote. Safe to run any time.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/remote.env
rsync -az -e "ssh -p $REMOTE_PORT" \
  --include='*/' --include='*.py' --include='*.sh' --include='*.toml' --include='*.lock' \
  --include='*.json' --include='*.yaml' --include='*.txt' --include='*.md' \
  --exclude='.venv/' --exclude='.git/' --exclude='models/' --exclude='preds/' --exclude='results/' \
  --exclude='*/cache/' --exclude='*/logs/' --exclude='*/out/' --exclude='__pycache__/' --exclude='*' \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"
# data files are needed too (small); push only if changed
rsync -az -e "ssh -p $REMOTE_PORT" data/ "$REMOTE_HOST:$REMOTE_DIR/data/"
