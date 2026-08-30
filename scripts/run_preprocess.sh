#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/coding_model_from_scratch"
DATA_ROOT="/workspace/dataset"
PYTHON_BIN="/opt/coding-model-venv/bin/python"
LOG_DIR="$DATA_ROOT/logs"
LOG_FILE="$LOG_DIR/preprocess.log"
RUN_FINGERPRINT_AUDIT_INDEX="${RUN_FINGERPRINT_AUDIT_INDEX:-0}"
PREPROCESS_MAX_DOCUMENT_BYTES="${PREPROCESS_MAX_DOCUMENT_BYTES:-16777216}"
PREPROCESS_MAX_BATCH_BYTES="${PREPROCESS_MAX_BATCH_BYTES:-67108864}"
PREPROCESS_MAX_INFLIGHT_BYTES="${PREPROCESS_MAX_INFLIGHT_BYTES:-536870912}"
PREPROCESS_MAX_MANIFEST_MEMBER_BYTES="${PREPROCESS_MAX_MANIFEST_MEMBER_BYTES:-8589934592}"
PREPROCESS_MAX_MANIFEST_LINE_BYTES="${PREPROCESS_MAX_MANIFEST_LINE_BYTES:-1048576}"
PREPROCESS_SCRATCH_MIN_FREE_GB="${PREPROCESS_SCRATCH_MIN_FREE_GB:-10}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$RUN_FINGERPRINT_AUDIT_INDEX" != "0" && "$RUN_FINGERPRINT_AUDIT_INDEX" != "1" ]]; then
  echo "RUN_FINGERPRINT_AUDIT_INDEX must be 0 or 1" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Starting/resuming streaming preprocessing" | tee -a "$LOG_FILE"

# The final curation plan trusts Stack v3 Train's upstream code near-deduplication.
# Fail closed on every launch if this volume is not the exact reviewed release.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/curation_policy.py" \
  --root "$DATA_ROOT" | tee -a "$LOG_FILE"

set +e
nice -n 5 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/preprocess_raw_stream.py" \
  --root "$DATA_ROOT" \
  --workers 24 \
  --analysis-batch-size 64 \
  --max-document-bytes "$PREPROCESS_MAX_DOCUMENT_BYTES" \
  --max-analysis-batch-bytes "$PREPROCESS_MAX_BATCH_BYTES" \
  --max-inflight-bytes "$PREPROCESS_MAX_INFLIGHT_BYTES" \
  --max-manifest-member-bytes "$PREPROCESS_MAX_MANIFEST_MEMBER_BYTES" \
  --max-manifest-line-bytes "$PREPROCESS_MAX_MANIFEST_LINE_BYTES" \
  --scratch-root /tmp/coding-model-preprocess \
  --scratch-min-free-gb "$PREPROCESS_SCRATCH_MIN_FREE_GB" \
  --index-mode deferred \
  --status-interval-seconds 300 \
  --poll-seconds 30 \
  --min-free-gb 50 \
  --log-every-documents 10000 \
  --require-closed-collection \
  --once \
  2>&1 | tee -a "$LOG_FILE"

audit_exit_code=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] Archive audit exited with status $audit_exit_code" | tee -a "$LOG_FILE"
if [[ "$audit_exit_code" -ne 0 ]]; then
  exit "$audit_exit_code"
fi

# Defense in depth: never print "complete" based only on process exit. Rebuild
# the exact quota/raw/report inventory, reconcile any legacy report+error state,
# and require zero remaining archive errors.
set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/preprocess_raw_stream.py" \
  --root "$DATA_ROOT" \
  --scratch-root /tmp/coding-model-preprocess \
  --status --require-complete --require-closed-collection --skip-dedup-status \
  2>&1 | tee -a "$LOG_FILE"
coverage_exit_code=${PIPESTATUS[0]}
set -e
if [[ "$coverage_exit_code" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] Raw audit coverage gate failed with status $coverage_exit_code" | tee -a "$LOG_FILE"
  exit "$coverage_exit_code"
fi

# Production curation independently revalidates every immutable fingerprint and
# performs the authoritative exact/normalized canonicalization in its own
# journal. Rebuilding the legacy aggregate telemetry index is therefore
# optional and disabled by default; on network storage it duplicates the most
# expensive database work without changing selected training data.
if [[ "$RUN_FINGERPRINT_AUDIT_INDEX" != "1" ]]; then
  echo "[$(date --iso-8601=seconds)] Audit complete; optional fingerprint telemetry indexing skipped" | tee -a "$LOG_FILE"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] Starting/resuming batched fingerprint indexing" | tee -a "$LOG_FILE"
set +e
nice -n 10 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/preprocess_raw_stream.py" \
  --root "$DATA_ROOT" \
  --scratch-root /tmp/coding-model-preprocess \
  --scratch-min-free-gb "$PREPROCESS_SCRATCH_MIN_FREE_GB" \
  --index-mode only \
  --index-batch-size 10000 \
  --status-interval-seconds 300 \
  --poll-seconds 30 \
  --min-free-gb 50 \
  --log-every-documents 10000 \
  --require-closed-collection \
  --once \
  2>&1 | tee -a "$LOG_FILE"

index_exit_code=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] Fingerprint indexing exited with status $index_exit_code" | tee -a "$LOG_FILE"
exit "$index_exit_code"
