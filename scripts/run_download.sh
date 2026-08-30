#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/coding_model_from_scratch"
DATA_ROOT="/workspace/dataset"
PYTHON_BIN="/opt/coding-model-venv/bin/python"
LOG_DIR="$DATA_ROOT/logs"
LOG_FILE="$LOG_DIR/collector.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS=2

echo "[$(date --iso-8601=seconds)] Starting/resuming parallel Stack v3 collection" | tee -a "$LOG_FILE"

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_stack_v3_parallel.py" \
  --root "$DATA_ROOT" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --cache-dir /tmp/huggingface-cache \
  --workers 8 \
  --token-batch-size 64 \
  --checkpoint-bytes 100000000 \
  --checkpoint-repos 10000 \
  --compression-level 3 \
  --compression-threads 2 \
  --min-free-gb 50 \
  --log-every-repos 1000 \
  2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] Collector exited with status $exit_code" | tee -a "$LOG_FILE"
exit "$exit_code"
