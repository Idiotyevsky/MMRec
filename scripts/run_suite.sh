#!/usr/bin/env bash
# Launch a small experiment queue on one GPU.
#
#   bash scripts/run_suite.sh <gpu_id> <queue_file>
#
# Queue file format (one job per line, blank lines and '#' comments ignored):
#
#   tag|config.yaml|override1@override2@...
#
# Extra overrides are separated by '@' so that values may contain spaces and
# quotes without any shell word-splitting.  Jobs in one queue run sequentially;
# different queues run on different GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: run_suite.sh <gpu_id> <queue_file>}"
QUEUE="${2:?usage: run_suite.sh <gpu_id> <queue_file>}"

mkdir -p results/logs
while IFS='|' read -r tag config extra; do
  tag="$(echo "$tag" | xargs)"
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac

  # Guard against orchestrating the same experiment twice: a tag that already
  # has a finished run, or that is currently training, is skipped.
  # The glob must match a run id (`<tag>_<date>-<time>_<hash>`), not just the
  # prefix: `${tag}_*` let tag `sasrec` match `sasrec_itemdrop_*` and silently
  # skipped a job whose run did not exist.
  if compgen -G "results/runs/${tag}_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-*/metrics.json" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] SKIP $tag (already has metrics.json)"
    continue
  fi
  if pgrep -f "scripts/train.py .*--tag ${tag}( |\$)" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] SKIP $tag (already running)"
    continue
  fi

  args=()
  if [ -n "${extra:-}" ]; then
    IFS='@' read -r -a args <<< "$extra"
  fi

  log="results/logs/${tag}.log"
  echo "[$(date '+%F %T')] START $tag on gpu $GPU" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/train.py \
      --config "$config" --tag "$tag" "${args[@]}" >>"$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] END $tag rc=$rc" | tee -a "$log"
done < "$QUEUE"
echo "[$(date '+%F %T')] queue $QUEUE finished"
