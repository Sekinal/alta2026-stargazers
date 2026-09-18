#!/usr/bin/env bash
# Run a command on the remote, detached (survives ssh drops), logging to logs/<name>.log.
#   scripts/remote_run.sh NAME "uv run python experiments/x/train.py --fold 0"
# Check:  scripts/remote_run.sh --status   |   tail:  ssh ... tail -f /workspace/ALTA/logs/NAME.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/remote.env
SSH="ssh -p $REMOTE_PORT $REMOTE_HOST"
if [[ "${1:-}" == "--status" ]]; then
  $SSH "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; \
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; \
        ps -eo pid,etime,pcpu,rss,args | grep -E '[u]v run|[p]ython' | cut -c1-200; free -g | head -2"
  exit 0
fi
name="$1"; shift
$SSH "cd $REMOTE_DIR && mkdir -p logs && export PATH=\$HOME/.local/bin:\$PATH && \
      nohup setsid bash -c $(printf '%q' "$*") > logs/$name.log 2>&1 < /dev/null & echo started $name pid \$!"
