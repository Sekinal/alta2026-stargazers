#!/usr/bin/env bash
# Pull the checkpoints of one kept run: scripts/remote_pull_models.sh RUN
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/remote.env
mkdir -p models
rsync -az --partial -e "ssh -p $REMOTE_PORT" "$REMOTE_HOST:$REMOTE_DIR/models/$1" models/
