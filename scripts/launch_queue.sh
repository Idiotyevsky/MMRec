#!/usr/bin/env bash
# Fully detach a queue from the calling shell so that it survives the parent
# process (or the terminal) going away.
#
#   bash scripts/launch_queue.sh <gpu_id> <queue_file>
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: launch_queue.sh <gpu_id> <queue_file>}"
QUEUE="${2:?usage: launch_queue.sh <gpu_id> <queue_file>}"
NAME="$(basename "$QUEUE" .txt)"
LOG="/tmp/opencode/${NAME}_gpu${GPU}.log"

mkdir -p /tmp/opencode
setsid nohup bash scripts/run_suite.sh "$GPU" "$QUEUE" >"$LOG" 2>&1 </dev/null &
disown || true
echo "launched $NAME on gpu $GPU (pid $!) -> $LOG"
