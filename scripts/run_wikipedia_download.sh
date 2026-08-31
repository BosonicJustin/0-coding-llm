#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/coding_model_from_scratch}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/coding-model-venv/bin/python}"
LOG_DIR="$DATA_ROOT/logs"
LOG_FILE="$LOG_DIR/wikipedia_collector.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Starting/resuming English Wikipedia collection" | tee -a "$LOG_FILE"

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_wikipedia.py" \
  --root "$DATA_ROOT" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --dataset-config 20231101.en \
  --cache-dir /tmp/wikipedia-cache \
  --checkpoint-bytes 1000000000 \
  --checkpoint-documents 100000 \
  --compression-level 3 \
  --compression-threads 4 \
  --min-free-gb 50 \
  --log-every-documents 1000 \
  2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] Wikipedia collector exited with status $exit_code" | tee -a "$LOG_FILE"
exit "$exit_code"
