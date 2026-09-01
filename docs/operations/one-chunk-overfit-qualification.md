# One-chunk overfit and exact-resume qualification

This is a bounded correctness gate, not useful training. It repeatedly presents
one immutable global batch through the production `CausalLM`, DDP wrapper,
block-diagonal packed-document mask, token-normalized `Trainer`, atomic
two-generation checkpoint publisher, and rank-zero W&B adapter.

A passing CPU run qualifies only the harness. CUDA, FlexAttention, BF16, NCCL,
GPU memory, and GPU restart behavior pass only when the corresponding GPU run
passes on the final pod.

## Frozen pass criteria

The final process exits zero only when all of these hold:

- the packed chunk contains at least one in-row document transition;
- every transition has a masked next-token target and a reset position ID;
- all losses are finite and final loss is below initial loss;
- `final_loss / initial_loss <= 0.10`;
- final token cross-entropy is at most `0.50` (perplexity at most 1.649);
- when an uninterrupted reference is supplied, model, optimizer, train state,
  every rank's RNG state, trajectory configuration, data identity, tokenizer
  identities, and world size have identical semantic SHA-256 digests.

The final-loss threshold means the geometric mean probability assigned to the
correct supervised token is at least `exp(-0.5)`, about 60.7%. This is a real
memorization criterion, not merely a downward-sloping loss curve.

`checkpoint.pt` is published with the production atomic writer. Scheduled
updates rotate the prior complete generation to `checkpoint.previous.pt`.
`result.json` is atomically set to `status: running` before model allocation, a
partial phase ends as `status: checkpointed`, and only a completed final phase
can say `status: passed`.

## CPU harness smoke

This disposable synthetic fixture still uses the real packed writer, order,
mmap/collator, boundary labels, model, trainer, and checkpoint code:

```bash
python scripts/overfit_single_chunk.py \
  --device cpu \
  --model-size tiny \
  --batch-size 2 \
  --steps 100 \
  --wandb-mode disabled \
  --output-dir /tmp/overfit-cpu-smoke
```

Its result cannot be cited as CUDA qualification.

## Single-GPU real-data qualification

Use a new output root and the final GPU-local train order. The first run is an
uninterrupted reference. The second and third commands are separate processes:
they stop at step 50, reload the atomic checkpoint, and compare the final
trajectory to the reference.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN=/workspace/venv/bin/python
PROJECT_ROOT=/workspace/0-coding-llm
TRAIN_ORDER=/local-nvme/packed-v1/orders/train/manifest.json
TOKENIZER_ROOT=/network-volume/tokenizer/starcoder2
QUAL_ROOT=/network-volume/audits/one-chunk-single-gpu-v1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  --order-manifest "$TRAIN_ORDER" \
  --tokenizer "$TOKENIZER_ROOT" \
  --model-size tiny \
  --device cuda \
  --parameter-dtype float32 \
  --precision bfloat16 \
  --no-activation-checkpointing \
  --batch-size 1 \
  --steps 100 \
  --checkpoint-every 10 \
  --learning-rate 3e-3 \
  --required-loss-ratio 0.10 \
  --required-final-loss 0.50 \
  --wandb-mode disabled \
  --output-dir "$QUAL_ROOT/control"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  --order-manifest "$TRAIN_ORDER" \
  --tokenizer "$TOKENIZER_ROOT" \
  --model-size tiny \
  --device cuda \
  --parameter-dtype float32 \
  --precision bfloat16 \
  --no-activation-checkpointing \
  --batch-size 1 \
  --steps 100 \
  --stop-after-step 50 \
  --checkpoint-every 10 \
  --learning-rate 3e-3 \
  --required-loss-ratio 0.10 \
  --required-final-loss 0.50 \
  --wandb-mode offline \
  --wandb-run-name one-chunk-single-gpu-resume \
  --output-dir "$QUAL_ROOT/resumed"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  --order-manifest "$TRAIN_ORDER" \
  --tokenizer "$TOKENIZER_ROOT" \
  --model-size tiny \
  --device cuda \
  --parameter-dtype float32 \
  --precision bfloat16 \
  --no-activation-checkpointing \
  --batch-size 1 \
  --steps 100 \
  --checkpoint-every 10 \
  --learning-rate 3e-3 \
  --required-loss-ratio 0.10 \
  --required-final-loss 0.50 \
  --wandb-mode offline \
  --wandb-run-name one-chunk-single-gpu-resume \
  --output-dir "$QUAL_ROOT/resumed" \
  --resume "$QUAL_ROOT/resumed/checkpoint.pt" \
  --exact-reference-checkpoint "$QUAL_ROOT/control/checkpoint.pt"
```

Offline W&B is initialized only on rank zero. Its run ID is checkpoint-bound
and reused by the resume process. Use `--wandb-mode disabled` in both resumed
phases if W&B is intentionally outside this gate.

## Six-GPU 1.3B qualification

Run the same three fresh-process phases on the final six-GPU pod. `--batch-size`
is the global fixed-batch row count and must divide six; each rank receives the
same contiguous slice production DDP uses. Six rows is the smallest value.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN=/workspace/venv/bin/python
PROJECT_ROOT=/workspace/0-coding-llm
TRAIN_ORDER=/local-nvme/packed-v1/orders/train/manifest.json
TOKENIZER_ROOT=/network-volume/tokenizer/starcoder2
QUAL_ROOT=/network-volume/audits/one-chunk-six-gpu-1.3b-v1
COMMON=(
  --order-manifest "$TRAIN_ORDER"
  --tokenizer "$TOKENIZER_ROOT"
  --model-size 1.3b
  --device cuda
  --parameter-dtype float32
  --precision bfloat16
  --activation-checkpointing
  --batch-size 6
  --steps 100
  --checkpoint-every 10
  --learning-rate 3e-4
  --required-loss-ratio 0.10
  --required-final-loss 0.50
)

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  "${COMMON[@]}" \
  --wandb-mode disabled \
  --output-dir "$QUAL_ROOT/control"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  "${COMMON[@]}" \
  --stop-after-step 50 \
  --wandb-mode offline \
  --wandb-run-name one-chunk-six-gpu-resume \
  --output-dir "$QUAL_ROOT/resumed"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$PROJECT_ROOT/scripts/overfit_single_chunk.py" \
  "${COMMON[@]}" \
  --wandb-mode offline \
  --wandb-run-name one-chunk-six-gpu-resume \
  --output-dir "$QUAL_ROOT/resumed" \
  --resume "$QUAL_ROOT/resumed/checkpoint.pt" \
  --exact-reference-checkpoint "$QUAL_ROOT/control/checkpoint.pt"
```

The final check is mechanical:

```bash
jq -e '
  .status == "passed" and
  .world_size == 6 and
  .packed_boundaries.in_row_document_boundaries > 0 and
  .loss_ratio <= .required_loss_ratio and
  .final_loss <= .required_final_loss and
  .exact_resume.requested == true and
  .exact_resume.exact_match == true and
  (.exact_resume.mismatches | length) == 0
' "$QUAL_ROOT/resumed/result.json" >/dev/null
```

If the fixed six-row prefix has no in-row boundary, the harness fails before
optimization. Select a larger global batch divisible by six; do not remove the
boundary criterion. If 100 steps fail the frozen memorization thresholds,
preserve the failed evidence and investigate before changing the recipe. Any
changed steps, learning rate, batch, precision, model, or world size requires
rerunning all three phases under a new output root.

This gate feeds the accepted-geometry evidence. It does not replace CUDA model
parity, soak, preemption, validation, storage, or immutable long-run authority
qualification.
