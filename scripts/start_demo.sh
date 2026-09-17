#!/usr/bin/env bash
# Start the ShortRec demo: FastAPI backend + Vite frontend.
#
#   bash scripts/start_demo.sh              # cpu backend on :8000, frontend on :5173
#   bash scripts/start_demo.sh --device cuda
#
# The backend serves on 127.0.0.1:8000 and the frontend proxies /api to it, so
# only the frontend URL needs to be opened.
set -uo pipefail
cd "$(dirname "$0")/.."

DEVICE="cpu"
API_PORT=8000
WEB_PORT=5173
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --api-port) API_PORT="$2"; shift 2;;
    --web-port) WEB_PORT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

mkdir -p /tmp/opencode
API_LOG=/tmp/opencode/shortrec_api.log
WEB_LOG=/tmp/opencode/shortrec_web.log

echo "==> checking demo prerequisites"
if ! python scripts/prepare_demo.py; then
  echo
  echo "Demo prerequisites are not ready (see above)."
  echo "At minimum you need a processed dataset and one trained ranker:"
  echo "  python -m src.data.preprocess --out data/processed/base"
  echo "  python -m src.data.preprocess --out data/processed/cold10 --cold-ratio 0.1 --cold-seed 42"
  echo "  python scripts/train.py --config configs/sasrec.yaml"
  echo "  python scripts/train.py --config configs/mm_sasrec_concat.yaml"
  exit 1
fi

echo "==> starting API on :${API_PORT} (device=${DEVICE})"
setsid nohup python -m src.serving.app --port "$API_PORT" --device "$DEVICE" \
  > "$API_LOG" 2>&1 </dev/null &
API_PID=$!
disown || true

for i in $(seq 1 60); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "    API ready (pid ${API_PID})"
    break
  fi
  sleep 1
  if [ "$i" = 60 ]; then
    echo "    API did not become ready; last log lines:" >&2
    tail -20 "$API_LOG" >&2
    exit 1
  fi
done

if [ ! -d frontend/node_modules ]; then
  echo "==> installing frontend dependencies"
  (cd frontend && npm install --no-audit --no-fund) || exit 1
fi

echo "==> starting frontend on :${WEB_PORT}"
cd frontend
setsid nohup npm run dev -- --port "$WEB_PORT" --host 127.0.0.1 \
  > "$WEB_LOG" 2>&1 </dev/null &
WEB_PID=$!
disown || true

for i in $(seq 1 60); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${WEB_PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

cat <<EOF

ShortRec is running:
  UI       http://127.0.0.1:${WEB_PORT}
  API      http://127.0.0.1:${API_PORT}
  API docs http://127.0.0.1:${API_PORT}/docs

  logs: ${API_LOG}
        ${WEB_LOG}

Stop with:
  kill ${API_PID} ${WEB_PID}
EOF
