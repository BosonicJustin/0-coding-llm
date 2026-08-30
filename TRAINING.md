# Native PyTorch pre-training harness

`pretrain/train.py` is the optimizer-side boundary of the pre-training stack.
It consumes only immutable packed batches from `pretrain.data`; raw archives,
text filtering, tokenization, and packing never occur in the hot training path.

## Correctness invariants

- Packed rows remain attention-isolated by `document_ids`; physical batch rows
  are independent in the model as well.
- Each microbatch backpropagates `loss_sum`, not its own mean loss. After a full
  accumulation window, gradients are divided once by the exact global number
  of supervised tokens. This prevents rows with many masked document-boundary
  targets from receiving too much weight.
- DDP accumulation suppresses intermediate gradient synchronizations. The last
  microbatch performs the all-reduce, and the trainer accounts for DDP's
  gradient averaging when applying the global token denominator.
- Production DDP uses the default dynamic reducer (`static_graph=False`). On
  supported PyTorch builds, `static_graph=True` combined with `no_sync()`
  accumulation can fail inside the reducer on the first synchronized backward.
  A real two-process Gloo regression test exercises this exact wrapper.
- Gradient norm is measured after token normalization and before clipping.
- Checkpoints are permitted only at clean optimizer-step boundaries. A partial
  accumulation window is never presented as durable progress.
- The loader resumes from `completed_microbatches`, which is equivalent to the
  sampler's `start_global_microbatch`. Prefetched but unconsumed batches do
  not change checkpoint progress.
- The production train order v4 freezes global microbatch rows, gradient
  accumulation, effective optimizer-update rows, and the exact full-update
  token prefix. The trainer and sampler reject any geometry or cursor mismatch
  before consuming data. Held-out validation order v4 is intentionally
  unfrozen and evaluation uses accumulation 1 over its complete-microbatch
  prefix at the training global microbatch size.

## Checkpoints and exact resume

Every atomic checkpoint contains:

- raw model parameters (independent of DDP or `torch.compile` wrappers);
- optimizer state;
- optimizer-step, microbatch, input-token, and supervised-token counters;
- Python, NumPy, CPU torch, and the assigned CUDA device's RNG state for each
  rank. A worker never seeds, captures, or restores peer GPUs;
- model configuration, parameter dtypes, optimizer-trajectory configuration,
  world size, Python/NumPy/torch/CUDA runtime identity, model/trainer source
  hashes, immutable data identity, the exact tokenizer-manifest SHA-256, and a
  canonical SHA-256 of the complete token-to-ID vocabulary.

Resume rejects a different dataset/order, model, dtype, world size, torch
version, LR schedule, accumulation factor, or optimizer hyperparameters. Log
and checkpoint frequencies may change because they do not alter parameters.
Atomic publication uses file and parent-directory `fsync` plus same-directory
hard links and `os.replace`, so an interruption cannot turn a partial write
into `last.pt`. The new latest is published and directory-synced before the old
latest hard link replaces `<name>.previous.pt`; this retains a known-readable
generation without a second 15 GB copy operation. Resume from the previous
generation if loading the latest one fails.

Exactness means resuming on the same software and device topology. The CPU test
compares every model and optimizer tensor after uninterrupted versus
interrupted/resumed training with exact equality. The chosen CUDA container
still needs the equivalent bitwise gate before a long run.

Checkpoint format v5 stores one rank-local CUDA RNG device index and state per
worker and binds the model to both tokenizer identities. This avoids the
multi-GPU-unsafe `get_rng_state_all()` behavior and prevents a same-size but
semantically different vocabulary from being substituted on resume or export.
Earlier checkpoint formats are intentionally rejected for exact resume rather
than ambiguously migrated.

`tests/test_train_distributed.py` contains real, two-process Gloo correctness
gates. The first runs the production DDP wrapper with two accumulated
microbatches and different supervised-token counts on each rank, then proves
synchronized replicas, optimizer state, counters, global metrics, and
numerical equivalence to a single-process global-batch reference. The second
runs an uninterrupted two-update format-v4 order alongside a step-one atomic
checkpoint; it terminates both workers, starts two fresh workers, restores the
rank RNG states and immutable-order cursor, and requires bit-for-bit identical
final models, optimizer state, counters, metrics, RNG states, and next-row
position. These tests need permission to bind a localhost Gloo socket; a
restricted sandbox may block them even though they use `file://` rendezvous.

The harness supports FP32 and BF16 autocast. FP16 is intentionally rejected:
summed-token accumulation needs a carefully tested dynamic-loss-scaling and
overflow-replay policy, while the intended training GPUs support BF16's wider
exponent range. An unvalidated FP16 flag would be worse than a clear failure.

Only load checkpoints produced by this experiment. PyTorch optimizer/RNG
checkpoints use pickle and are not safe when obtained from an untrusted source.

## Single-packed-chunk overfit gate

The default command is entirely local, creates a disposable synthetic corpus,
loads one batch through the real shard/order/mmap/collator path, freezes that
batch by content hash, and asserts that its loss falls by at least 50%:

```bash
python scripts/overfit_single_chunk.py
```

Artifacts are written to `runs/overfit-single-chunk/`:

- `checkpoint.pt`: a resumable full-state checkpoint;
- `result.json`: initial/final loss, ratio, model configuration, batch hash,
  parameter count, and data identity.

Exercise one fixed batch from the real packed corpus on a GPU with the small
debug architecture first:

```bash
python scripts/overfit_single_chunk.py \
  --order-manifest /local-nvme/train/order-seed-1234/manifest.json \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --model-size tiny \
  --device cuda \
  --precision bfloat16 \
  --steps 100
```

Then run the same gate with the exact architecture:

```bash
python scripts/overfit_single_chunk.py \
  --order-manifest /local-nvme/train/order-seed-1234/manifest.json \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --model-size 1.3b \
  --device cuda \
  --parameter-dtype float32 \
  --precision bfloat16 \
  --batch-size 1 \
  --steps 100
```

The script reuses the same loaded tensors at every step; it does not repeatedly
ask the sampler for the next row. Its checksum-verified diagnostic prefix loader
may select a smaller batch than the order's frozen global microbatch without
creating a training cursor or weakening the production sampler contract. The
fixed-batch SHA-256 is stored in the
checkpoint, so resume fails if even one tensor value or shape changes.

## Pre-training entry point

The raw entry point below is useful for controlled development and GPU gates.
Use the fail-closed production launcher later in this section for a long run.

Single process:

```bash
python -m pretrain.train \
  --order-manifest /local-nvme/train/order-seed-1234/manifest.json \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --model-size 1.3b \
  --device cuda \
  --precision bfloat16 \
  --checkpoint /workspace/dataset/checkpoints/pretrain-last.pt
```

The planned production topology is one six-GPU RunPod pod. Replicated DDP uses
one process per local GPU:

```bash
torchrun --standalone --nproc-per-node=6 -m pretrain.train \
  --order-manifest /local-nvme/train/order-seed-1234/manifest.json \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --model-size 1.3b \
  --device cuda \
  --precision bfloat16 \
  --checkpoint /workspace/dataset/checkpoints/pretrain-last.pt
```

The example intentionally keeps the authoritative checkpoint on the persistent
RunPod network volume. Do not keep the only multi-day checkpoint on ephemeral
pod-local NVMe. A future asynchronous local-to-network replica manager may
reduce checkpoint stalls, but it must confirm the durable copy before deleting
or advancing the previous durable checkpoint. Offline W&B files should live on
persistent storage for the same reason.

`global-microbatch-rows` is the number of packed rows across all ranks in one
forward/backward pass. Effective update rows are
`global_microbatch_rows * gradient_accumulation_steps`. These values and the
resulting optimizer-update count are immutable fields of the order manifest.
All three CLI flags are optional: omission uses the frozen manifest values;
supplying a different value fails closed.

Those frozen values must come from a memory/throughput smoke on the actual GPU
topology before final order publication. No fixed microbatch, accumulation, or
step count is recommended here. The production examples intentionally omit all
three so the trainer consumes the finalized order manifest's complete contract.
For the planned six-GPU pod, global microbatch rows must be divisible by six and
the accepted checkpoint trajectory is bound to `world_size=6`. Keep packed hot
data on pod-local NVMe when practical, but publish checkpoints and audit
evidence to the mounted network volume.

To resume, pass the same trajectory settings plus:

```bash
--resume /workspace/dataset/checkpoints/pretrain-last.pt
```

If `--checkpoint` is omitted on resume, subsequent checkpoints overwrite the
resume path atomically. Supplying `--checkpoint` explicitly redirects future
checkpoints, which is useful only when intentional. A fresh pre-training run
must provide `--checkpoint`; there is deliberately no ephemeral local default,
and an existing latest/previous generation is never overwritten unless it was
opened through `--resume`.

The sampler exposes exactly the manifest-authorized complete-update prefix. A
checkpoint cursor must be accumulation-aligned, and final trainer input and
supervised-token counters must equal the manifest before the run is accepted.

## Production preflight and launcher

`scripts/launch_pretraining.py` is the production entry point. It renders and,
only with `--execute`, runs the exact validated-Python distributed-launch
argument vector. It fails before model allocation unless all of the following
are proven:

- the train order is format v4 with frozen optimizer geometry;
- the explicit `--tokenizer` directory has a valid source manifest, every
  declared file matches its recorded size and SHA-256, its manifest SHA-256
  matches both orders, and its canonical token-to-ID mapping has the declared
  vocabulary size;
- the distinct validation order is format v4, has split `validation`, is
  intentionally unfrozen, and can supply the requested number of complete
  global microbatches at the training microbatch size;
- both orders and all referenced packed manifests and shard payloads exist on
  the declared local-data device, their metadata identities agree, and every
  payload has its recorded byte size; the local-data path is an explicitly
  read-only mount so no writer can change an mmap payload after preflight;
- visible CUDA devices exactly equal `--nproc-per-node`, NCCL is available,
  every device supports BF16, and the frozen global microbatch is divisible by
  world size;
- the checkpoint is under an explicit durable root on a different filesystem
  device from the local packed copy, and at least twice the measured checkpoint
  generation upper bound is free;
- the checkpoint directory actually supports the trainer's required file
  `fsync`, directory `fsync`, nonblocking `flock`, hard-link, and atomic-replace
  operations; and
- enabled W&B mode has its optional dependency, durable output path, and, for
  online mode, a configured API credential.

The launch-time data check is deliberately metadata-only: it reads JSON
manifests and stats payloads, but reads zero bytes from `order.bin`, token
shards, or start shards. Execution additionally requires two immutable evidence
receipts made by `scripts/certify_pretraining_data.py`. The certifier runs the
full order and packed checksum/semantic validators, inventories every exact
local file with its declared digest and stat identity, records validator source
and runtime identity plus completion time, and writes an exact SHA-256 sidecar.
The launcher reopens and binds those receipts to the current local files. A
copy, remount, payload metadata change, or validator-code change requires fresh
certification. A dry run may show missing evidence; `--execute` may not.

The interpreter is part of that identity: receipts and preflight reports record
the resolved Python executable and its SHA-256. The command is always
`<that-python> -m torch.distributed.run ...`; a different `torchrun` earlier on
`PATH` cannot select an uncertified Python/PyTorch/CUDA environment. On execute,
the wrapper replaces itself with that process. This preserves the launcher PID,
so scheduler or container signals reach torchrun instead of leaving workers
orphaned behind an exited wrapper.

The production wrapper always passes `--deterministic` and launches with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. A conflicting pre-existing setting is an
error. Nondeterministic performance experiments must bypass this production
wrapper and use the raw entry point explicitly.

Prepare unique durable directories and first render a dry run on the final GPU
pod. `CHECKPOINT_GENERATION_BYTES` must be a measured conservative upper bound,
not the current size of an early checkpoint:

```bash
export GPU_PACKED_SOURCE=/local-nvme/packed-v1-staged
export GPU_PACKED=/local-nvme/packed-v1-ro
export DATA_ROOT=/workspace/dataset
export RUN_ID=pretrain-v1
export TRAIN_ORDER="$GPU_PACKED/orders/train/manifest.json"
export VALIDATION_ORDER="$GPU_PACKED/orders/validation/manifest.json"
export CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$RUN_ID"
export AUDIT_DIR="$DATA_ROOT/audits/$RUN_ID"
export CHECKPOINT_GENERATION_BYTES=18000000000
export CHECKPOINT_EVERY=1000
export EVAL_EVERY=1000
export EVAL_BATCHES=32
export TRAIN_VALIDATION_EVIDENCE="$AUDIT_DIR/train-full-validation.json"
export HELDOUT_VALIDATION_EVIDENCE="$AUDIT_DIR/validation-full-validation.json"

mkdir -p "$CHECKPOINT_DIR" "$AUDIT_DIR"

# After rsync/copy is complete and every writer is stopped, expose the final
# local copy at the exact launch path through a read-only Linux bind mount.
sudo mkdir -p "$GPU_PACKED"
sudo mount --bind "$GPU_PACKED_SOURCE" "$GPU_PACKED"
sudo mount -o remount,bind,ro "$GPU_PACKED"
findmnt -no TARGET,FSTYPE,OPTIONS --target "$GPU_PACKED"
export GLOBAL_MICROBATCH_ROWS="$(jq -er \
  '.training_consumption.frozen_global_microbatch_rows' "$TRAIN_ORDER")"

python scripts/certify_pretraining_data.py "$TRAIN_ORDER" \
  --expected-split train \
  --local-data-root "$GPU_PACKED" \
  --output "$TRAIN_VALIDATION_EVIDENCE"

python scripts/certify_pretraining_data.py "$VALIDATION_ORDER" \
  --expected-split validation \
  --local-data-root "$GPU_PACKED" \
  --global-microbatch-rows "$GLOBAL_MICROBATCH_ROWS" \
  --output "$HELDOUT_VALIDATION_EVIDENCE"

python scripts/launch_pretraining.py \
  --train-order-manifest "$TRAIN_ORDER" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --local-data-root "$GPU_PACKED" \
  --durable-checkpoint-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT_DIR/last.pt" \
  --checkpoint-generation-bytes "$CHECKPOINT_GENERATION_BYTES" \
  --resume-generation none \
  --nproc-per-node 6 \
  --workers 4 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --eval-at-start \
  --train-data-evidence "$TRAIN_VALIDATION_EVIDENCE" \
  --validation-data-evidence "$HELDOUT_VALIDATION_EVIDENCE" \
  --wandb-mode offline \
  --wandb-project coding-model-from-scratch \
  --wandb-run-name "$RUN_ID" \
  --preflight-report "$AUDIT_DIR/preflight-dry-run.json" \
  --dry-run \
  -- \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 100 \
  --weight-decay 0.1 \
  --max-grad-norm 1.0 \
  --seed 1234
```

Review the emitted JSON and shell-rendered command. Then repeat the same
arguments with a new immutable report path and replace `--dry-run` with
`--execute`. Options owned by the launcher—including data, geometry, device,
precision, validation, checkpoint, W&B, and determinism flags—are rejected
after `--`; only ordinary trainer trajectory options belong there. The wrapper
never passes `--verify-packed-payloads` to every rank.

Do not replace the read-only bind mount with ordinary file permissions. A
writable mount leaves a time-of-check/time-of-use window between evidence
verification and worker mmap. Any remount, snapshot replacement, recopy, or
payload change invalidates the receipts; create new receipt paths after the
final read-only view is in place.

For exact resume, rerun the identical command with a new report path and:

```bash
--resume-generation latest
```

Latest is always the first choice. If trusted-checkpoint loading itself fails,
rerun preflight explicitly with:

```bash
--resume-generation previous
```

That loads `last.previous.pt` and continues publishing successful checkpoints
to canonical `last.pt`. A new run refuses any existing latest or previous
generation; resume never happens implicitly. Only load experiment-owned
checkpoints because PyTorch optimizer/RNG state uses pickle.

Resume currently mmap-loads the selected checkpoint directly from the durable
checkpoint filesystem. The production wrapper does not copy or claim to verify
a local-NVMe resume staging path, and it rejects attempts to override its
durable `--resume`/`--checkpoint` pair after `--`. Benchmark this read on the
real volume. A future local-read/durable-write optimization needs an explicit
trainer contract and source-to-local SHA-256 proof; a generic redirected resume
path is not that contract.

## Metrics and W&B

Rank zero emits JSON metrics:

- `train/loss` and `train/token_loss`: global token-normalized cross-entropy;
- `train/loss_sum`: global summed cross-entropy for the update window;
- `train/perplexity`, `train/learning_rate`, and `train/grad_norm`;
- cumulative `train/rows`, `train/input_tokens`, `train/supervised_tokens`
  (also emitted as `train/loss_tokens`), and `train/microbatches`;
- input/supervised token throughput and seconds per optimizer step.

At the configured cadence it also emits global and per-domain
`validation/loss`, perplexity, row/token counts, and throughput. Evaluation
all-reduces summed loss and supervised-token counts, does not update model or
optimizer state, preserves training RNG state, and uses accumulation 1 over the
complete-microbatch prefix of the intentionally unfrozen validation order.

W&B is optional and disabled by default. Disabled mode does not import the
package or access the network. Install the separate optional requirement and
use offline mode for a network-free run:

```bash
pip install -r requirements-wandb.txt
python -m pretrain.train ... --wandb-mode offline
```

Online logging requires the explicit `--wandb-mode online` flag.
The generated W&B run ID is stored inside the same atomic model checkpoint. On
resume the CLI reuses that ID automatically; `--wandb-run-id` can override it
deliberately. Coupling it to the model generation avoids attaching old weights
to a newer sidecar after a crash, and keeps a resumed optimizer run on one W&B
history instead of silently creating disconnected charts.

For the 1.284B FP32-parameter AdamW run, one mature checkpoint is approximately
15.4 GB (about 5.1 GB of weights plus 10.3 GB of moments, before container
overhead). The CLI default is therefore 1,000 optimizer steps, not 100. Benchmark
checkpoint time on the real network volume, choose a cadence from measured
recovery cost, and budget roughly two full generations for `last` + `previous`.

## Required GPU gates before the long run

The CPU suite proves optimizer math, loader compatibility, loss reduction,
atomic checkpoint structure, lazy W&B behavior, and exact CPU resume. Before
spending the full token budget, run these gates in the frozen GPU container:

1. After copying to local NVMe, stop writers, create the read-only local view,
   and certify both the final train and validation orders using
   `scripts/certify_pretraining_data.py` exactly as shown above. Preserve both
   JSON receipts and exact `.sha256` sidecars under the durable audit directory.
   The production launcher rejects execute mode without both receipts and the
   same read-only mount identities.

   ```bash
   findmnt -no TARGET,FSTYPE,OPTIONS --target "$GPU_PACKED"
   pushd "$AUDIT_DIR"
   sha256sum -c "$TRAIN_VALIDATION_EVIDENCE.sha256"
   sha256sum -c "$HELDOUT_VALIDATION_EVIDENCE.sha256"
   popd
   ```

   Do not pass `--verify-packed-payloads` to every distributed rank after this
   certification, because each rank would re-read the entire corpus.
2. Run the real-data tiny-model overfit gate.
3. Run the 1.3B fixed-batch overfit gate and inspect the loss curve and gradient
   norm for at least 100 steps.
4. Verify FlexAttention document isolation and forward/backward on CUDA.
5. Compare uninterrupted and resumed CUDA runs from the same checkpoint with
   exact model/optimizer tensor checks.
6. Benchmark worker count, pinned-memory transfer, `torch.compile`, batch size,
   accumulation, and checkpoint latency on the selected GPU/filesystem.
7. Measure peak memory. The 1.3B entry point defaults complete-transformer-block
   activation checkpointing on, but the current DDP path still replicates
   parameters and optimizer state. If that does not meet the target
   batch/context, add native FSDP and sharded checkpoints before the long run;
   do not reduce context or silently change optimizer precision merely to make
   the smoke test fit.
8. Exercise the reserved, contamination-clean, unfrozen validation order with
   the pinned `eval_every` and `eval_batches`; verify global/per-domain metrics,
   RNG preservation, uninterrupted-versus-resumed equality, and DDP summed-loss
   and supervised-token reductions. Never use MBPP or any final test benchmark
   for training-time validation.
