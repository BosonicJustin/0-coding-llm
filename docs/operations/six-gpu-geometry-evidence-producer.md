# Six-GPU geometry evidence producer

`scripts/qualify_training_geometry.py` is the launch-day producer for the
strict `pretraining-accepted-geometry` receipt. It measures the complete
`(6,32)`, `(12,16)`, `(24,8)` equal-effective-batch grid with compile disabled
and enabled, selects only among threshold-passing candidates, and then repeats
the selected recipe against the immutable final training order. It never uses
the trainer's compute-only step timer as throughput authority.

The grid is a diagnostic selection stage. The final-order soak is the authority
used for training. Both stages run exactly six ranks with BF16, deterministic
algorithms, activation checkpointing, fused AdamW, document-boundary masking,
validation, offline W&B, two graceful checkpoints, and a real six-rank process
restart/resume. The external supervisor starts its monotonic interval after
five optimizer updates and ends after at least 100 measured updates. Its token
delta comes from the trainer's checkpoint-authoritative
`consumed_input_tokens` counter.

## Safety boundary

Run the existing RunPod pod qualifier first. The geometry producer accepts only
that exact authenticated hardware contract. Before every run or resume it
re-hashes the frozen source, interpreter, hardware receipt, baseline receipt,
tokenizer, and data manifests; checks six homogeneous CUDA devices, native BF16,
NCCL availability, PyTorch/CUDA version identity, and all-six-GPU memory; and
refuses a nested `torchrun` environment.

Every plan, candidate result, grid result, and final receipt is write-once and
has an exact `.sha256` sidecar. A nonblocking lock prevents two producers from
sharing one output root. An exact rerun returns an already authenticated
terminal result. A different input, path, source hash, setting, or interpreter
requires a new output root.

An operating-system interruption after the first phase's durable journal can
resume on the same Linux boot. Unjournaled output, a stale-boot monotonic origin,
a started/incomplete second phase, a mutated checkpoint/log, or an immutable
`FAILURE.json` is poisoned and requires a new output root. Never delete only a
journal, stop file, result, or sidecar to force a retry.

The supervisor redacts inherited secret values from durable logs. Do not put a
credential in a path, run name, W&B project, or command argument. W&B remains
offline for qualification.

## Required one-GPU baseline

Six-GPU scaling efficiency cannot be derived from six-GPU measurements alone.
Provide a separately measured, authenticated one-GPU baseline for every grid
candidate on one GPU from the same qualified pod. These one-GPU recipes use the
per-rank microbatch and the same accumulation:

| Candidate | One-GPU global rows | Accumulation | Compile |
| --- | ---: | ---: | --- |
| `g6-a32-compile0` | 1 | 32 | false |
| `g6-a32-compile1` | 1 | 32 | true |
| `g12-a16-compile0` | 2 | 16 | false |
| `g12-a16-compile1` | 2 | 16 | true |
| `g24-a8-compile0` | 4 | 8 | false |
| `g24-a8-compile1` | 4 | 8 | true |

Each baseline must use external `time.monotonic_ns`, authoritative starting and
ending consumed-input-token counters, and the same end-to-end scope across
validation, checkpoint, offline-W&B logging, stop, and one-rank resume. It must
cover at least 100 update-aligned steps. Do not copy the trainer's
`perf/input_tokens_per_second` field and do not estimate a missing value.

Generate the deliberately non-runnable schema template:

```bash
PROJECT_ROOT=/workspace/0-coding-llm
PYTHON_BIN=/opt/coding-model-train-venv/bin/python
HARDWARE_RECEIPT=/workspace/run-authority/pod/hardware.json
SEQUENCE_LENGTH=4096

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/qualify_training_geometry.py baseline-template \
  --hardware-contract "$HARDWARE_RECEIPT" \
  --sequence-length "$SEQUENCE_LENGTH" \
  > /workspace/run-authority/geometry/single-gpu-baselines.TEMPLATE.json
```

Replace every placeholder with observed integers/exact decimal throughput,
remove `instructions` and `sequence_length_for_alignment`, and change status to
`pass`. The final draft has exactly `format`, `format_version`, `status`,
`hardware_contract_sha256`, and `candidates`. Let the producer validate every
counter/rate and atomically publish the immutable receipt and exact sidecar:

```bash
BASELINE_DRAFT=/workspace/run-authority/geometry/single-gpu-baselines.DRAFT.json
BASELINE=/workspace/run-authority/geometry/single-gpu-baselines.json
test ! -e "$BASELINE"
test ! -e "$BASELINE.sha256"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/qualify_training_geometry.py" seal-baselines \
  --draft "$BASELINE_DRAFT" \
  --output "$BASELINE" \
  --hardware-contract "$HARDWARE_RECEIPT" \
  --sequence-length "$SEQUENCE_LENGTH"
```

The grid preflight rejects a missing candidate, extra field, wrong hardware
hash, non-update-aligned counters, non-positive interval, or throughput that
differs from `token_delta * 1e9 / elapsed_wall_time_ns` by more than one part
per million. A SHA sidecar authenticates bytes; it does not make invented
measurements valid scientific evidence.

## Grid preflight and run

Use packed training manifests on local read-only NVMe when possible. Keep the
qualification root on a durable volume because it contains the two checkpoint
generations and logs for all six candidates. The six candidate lineages need
roughly 185 GB for mature 1.3B latest/previous checkpoints; reserve at least
250 GiB for the grid, logs, compiler artifacts, and explicit headroom. The
final soak needs another roughly 35–50 GB if retained beside the grid.

```bash
PROJECT_ROOT=/workspace/0-coding-llm
PYTHON_BIN=/opt/coding-model-train-venv/bin/python
PACKED=/local-nvme/training-corpus/packed/train
VALIDATION_ORDER=/local-nvme/training-corpus/orders/validation/manifest.json
TOKENIZER=/local-nvme/training-corpus/tokenizer/starcoder2
HARDWARE_RECEIPT=/workspace/run-authority/pod/hardware.json
BASELINES=/workspace/run-authority/geometry/single-gpu-baselines.json
GRID_ROOT=/workspace/run-authority/geometry/grid-20260901
GRID_DRIVER=/workspace/run-authority/geometry/grid-20260901.driver
CHECKPOINT_GENERATION_BYTES=17179869184

# Re-establish the exact deterministic environment frozen by the pod receipt.
export CUDA_VISIBLE_DEVICES="$(jq -jr \
  '.qualification.host.environment.cuda_visible_devices | join(",")' \
  "$HARDWARE_RECEIPT")"
export OMP_NUM_THREADS="$(jq -er \
  '.qualification.host.environment.required.OMP_NUM_THREADS' \
  "$HARDWARE_RECEIPT")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
unset LOCAL_RANK RANK WORLD_SIZE NCCL_P2P_DISABLE NCCL_SHM_DISABLE

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/qualify_training_geometry.py preflight-grid \
  --output-root "$GRID_ROOT" \
  --python-packed-manifest "$PACKED/python/manifest.json" \
  --other-code-packed-manifest "$PACKED/other_code/manifest.json" \
  --english-packed-manifest "$PACKED/english/manifest.json" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --tokenizer "$TOKENIZER" \
  --hardware-contract "$HARDWARE_RECEIPT" \
  --single-gpu-baselines "$BASELINES" \
  --checkpoint-generation-bytes "$CHECKPOINT_GENERATION_BYTES" \
  > "$GRID_DRIVER.preflight.json"
```

`preflight-grid` performs no GPU action and creates no output root. Inspect its
six phase-one and six resume commands. Then start one detached producer:

```bash
tmux new-session -d -s geometry-grid -c "$PROJECT_ROOT" \
  "'$PYTHON_BIN' -u scripts/qualify_training_geometry.py run-grid \
    --output-root '$GRID_ROOT' \
    --python-packed-manifest '$PACKED/python/manifest.json' \
    --other-code-packed-manifest '$PACKED/other_code/manifest.json' \
    --english-packed-manifest '$PACKED/english/manifest.json' \
    --validation-order-manifest '$VALIDATION_ORDER' \
    --tokenizer '$TOKENIZER' \
    --hardware-contract '$HARDWARE_RECEIPT' \
    --single-gpu-baselines '$BASELINES' \
    --checkpoint-generation-bytes '$CHECKPOINT_GENERATION_BYTES' \
    > '$GRID_DRIVER.stdout.json' 2> '$GRID_DRIVER.stderr.log'"
```

The producer builds a deterministic 113-update 40/40/20 diagnostic order for
each geometry. It proves that all six encoded `order.bin` payloads are
byte-identical before training. Monitor without altering state:

```bash
"$PYTHON_BIN" scripts/qualify_training_geometry.py status \
  --output-root "$GRID_ROOT"
tail -n 20 "$GRID_ROOT"/candidates/*/logs/phase-*.log
tmux has-session -t geometry-grid
```

Success requires `$GRID_ROOT/GRID-RESULT.json` and its sidecar with
`status: pass`. Failed thresholds are immutable evidence; use a fresh root for
a changed recipe or retry.

## Build the selected final order

Read the accepted geometry; do not copy a throughput value into the final
receipt. Build the immutable 52.58B-token order with that exact geometry:

```bash
GLOBAL_ROWS=$(jq -er '.accepted_candidate.global_microbatch_rows' \
  "$GRID_ROOT/GRID-RESULT.json")
ACCUMULATION=$(jq -er '.accepted_candidate.gradient_accumulation_steps' \
  "$GRID_ROOT/GRID-RESULT.json")
FINAL_ORDER=/local-nvme/training-corpus/orders/train-final
test ! -e "$FINAL_ORDER"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_training_order.py" \
  --python-manifest "$PACKED/python/manifest.json" \
  --other-code-manifest "$PACKED/other_code/manifest.json" \
  --english-manifest "$PACKED/english/manifest.json" \
  --output "$FINAL_ORDER" \
  --seed 1234 \
  --global-microbatch-rows "$GLOBAL_ROWS" \
  --gradient-accumulation-steps "$ACCUMULATION" \
  --expected-total-input-tokens 52580000000
```

Certify the final order/data using the existing corpus qualification route
before treating it as launch data.

## Final-order preflight and accepted receipt

The final stage starts from step zero on the exact production order, measures
the selected candidate across validation/checkpoint/offline-W&B/stop/resume,
and publishes the strict receipt accepted by `pretrain.geometry_evidence`.

```bash
FINAL_ROOT=/workspace/run-authority/geometry/final-soak-20260901
FINAL_DRIVER=/workspace/run-authority/geometry/final-soak-20260901.driver

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/qualify_training_geometry.py" preflight-final \
  --output-root "$FINAL_ROOT" \
  --train-order-manifest "$FINAL_ORDER/manifest.json" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --tokenizer "$TOKENIZER" \
  --hardware-contract "$HARDWARE_RECEIPT" \
  --single-gpu-baselines "$BASELINES" \
  --grid-result "$GRID_ROOT/GRID-RESULT.json" \
  --checkpoint-generation-bytes "$CHECKPOINT_GENERATION_BYTES" \
  > "$FINAL_DRIVER.preflight.json"

tmux new-session -d -s geometry-final-soak -c "$PROJECT_ROOT" \
  "'$PYTHON_BIN' -u scripts/qualify_training_geometry.py run-final \
    --output-root '$FINAL_ROOT' \
    --train-order-manifest '$FINAL_ORDER/manifest.json' \
    --validation-order-manifest '$VALIDATION_ORDER' \
    --tokenizer '$TOKENIZER' \
    --hardware-contract '$HARDWARE_RECEIPT' \
    --single-gpu-baselines '$BASELINES' \
    --grid-result '$GRID_ROOT/GRID-RESULT.json' \
    --checkpoint-generation-bytes '$CHECKPOINT_GENERATION_BYTES' \
    > '$FINAL_DRIVER.stdout.json' 2> '$FINAL_DRIVER.stderr.log'"
```

GO only when both
`$FINAL_ROOT/accepted-geometry.json` and
`$FINAL_ROOT/accepted-geometry.json.sha256` exist, status is `pass`, and the
later run-authority builder accepts that exact path. Preserve the complete grid
and final roots until the experiment record is archived; their JSON evidence
contains hashes of the logs, W&B files, commands, and both checkpoint
generations.
