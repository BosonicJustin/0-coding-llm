# Six-GPU pre-training launch qualification

This is the short GO/NO-GO path for the first production pre-training run. The
longer operational procedure remains in
[production-runbook.md](production-runbook.md); this page is the launch-day
checklist.

## Frozen strategy

- one RunPod node, exactly six homogeneous CUDA devices and six DDP ranks;
- the 1.3B model with FP32 parameters and optimizer state, BF16 autocast, and
  explicit whole-block activation checkpointing;
- the immutable packed train order and a distinct held-out validation order;
- packed document boundaries isolated by the model attention mask; physical
  batch rows are also independent;
- rank-zero-only W&B and checkpoint publication, with all ranks participating
  in reductions and checkpoint RNG collection;
- local read-only packed data and a separate durable checkpoint filesystem.

FSDP is not implemented in the accepted path. Do not start the long run on a
GPU type where replicated DDP fails the measured memory gate. An FSDP change
would require a new optimizer/checkpoint/resume design and a fresh authority.

## Accounting that must agree

For six ranks:

```text
local rows per rank per forward = global_microbatch_rows / 6
rows per optimizer update       = global_microbatch_rows * accumulation_steps
input tokens per update         = rows per update * sequence_length
completed microbatches          = completed updates * accumulation_steps
```

`global_microbatch_rows` must divide by six. The order manifest freezes these
values, the sampler gives each rank a disjoint slice of the same global order,
and every checkpoint records the global counters and one RNG state per rank.
The trainer also reconciles final global and per-domain input/supervised-token
counts with the order manifest.

## 1. CPU qualification before renting GPUs

From the exact source revision intended for the run:

```bash
python -m unittest \
  tests.test_model \
  tests.test_training_data \
  tests.test_train \
  tests.test_train_distributed \
  tests.test_launch_pretraining \
  tests.test_run_authority \
  tests.test_cuda_qualification -v
```

This covers packed-boundary isolation, six-rank sampler partitioning, global
token-normalized DDP math, atomic checkpoint rotation, fresh-process exact
resume, validation RNG preservation, rank-zero W&B behavior, launch preflight,
and authority mutation rejection. It does not prove CUDA memory, NCCL, kernel
correctness, throughput, or network-volume latency.

## 2. Prepare the final pod

1. Install the locked environment in the final container.
2. Expose exactly six homogeneous GPUs; do not rely on an extra visible device.
3. Copy packed data from the network volume to local NVMe.
4. Stop all writers and expose that final local copy through a read-only bind
   mount. Ordinary file permissions are insufficient.
5. Keep checkpoints, W&B offline files, certifications, and authority files on
   the network volume. Reserve at least two measured mature checkpoint
   generations plus operational headroom.
6. Certify train and validation with `scripts/certify_pretraining_data.py`.
   Preserve each JSON receipt and exact `.sha256` sidecar.

## 3. Run the GPU-only gates

Run these in order and stop on the first failure:

1. `scripts/qualify_cuda_model.py` on the production CUDA build. It compares
   FlexAttention with dense SDPA, tests document/row isolation, performs BF16
   backward, and takes one AdamW step.
2. Complete the separate
   [one-chunk overfit and exact-resume qualification](one-chunk-overfit-qualification.md)
   with a real packed order, the 1.3B model, 4,096-token context, and all six
   ranks. Require both frozen memorization thresholds and a bit-exact resumed
   trajectory.
3. A six-rank full-topology smoke using the intended global microbatch,
   accumulation, worker count, activation-checkpointing, compile, and fused
   optimizer decisions. Measure every GPU's allocated/reserved peak, aggregate
   input tokens/s, data-wait fraction, NCCL utilization, and minimum free VRAM.
4. A multi-step soak that crosses validation and checkpoint boundaries.
   Measure the real network checkpoint duration and verify the validation
   sample and per-domain metrics.
5. Kill and restart all six worker processes at a completed checkpoint.
   Compare model, optimizer, global cursor, per-domain counters, and rank RNG
   states with the uninterrupted deterministic trajectory.
6. Exercise graceful preemption and both `last.pt` and
   `last.previous.pt` recovery. The pod termination grace must exceed the
   launcher's checkpoint grace.
7. In offline W&B mode, confirm only rank zero creates a run directory and the
   resumed checkpoint reuses its run ID.

There is currently no automated producer for the accepted-geometry soak
receipt. Record the measured values in the strict schema documented in
[pretraining-run-authority.md](../training/pretraining-run-authority.md), write
it once with its exact SHA-256 sidecar, and let the authority builder validate
it. Automating that measurement/publisher is desirable, but hand-authored
measurement evidence must not be guessed or copied from another GPU type.

## 4. Dry-run the production launcher

The dry run has no authority yet, but it must use the final six-GPU pod and the
exact final inputs. Include the explicit launcher-owned flag:

```bash
python scripts/launch_pretraining.py \
  --train-order-manifest "$TRAIN_ORDER" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --tokenizer "$TOKENIZER_ROOT" \
  --local-data-root "$GPU_PACKED_RO" \
  --durable-checkpoint-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT_DIR/last.pt" \
  --checkpoint-generation-bytes "$CHECKPOINT_GENERATION_BYTES" \
  --resume-generation none \
  --nproc-per-node 6 \
  --model-size 1.3b \
  --activation-checkpointing \
  --workers "$TRAIN_WORKERS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --eval-at-start \
  --wandb-mode offline \
  --train-data-evidence "$TRAIN_DATA_EVIDENCE" \
  --validation-data-evidence "$VALIDATION_DATA_EVIDENCE" \
  --preflight-report "$AUDIT_DIR/launch-dry-run.json" \
  --dry-run -- \
  --learning-rate "$TRAIN_LR" \
  --min-learning-rate "$TRAIN_MIN_LR" \
  --warmup-steps "$TRAIN_WARMUP_STEPS" \
  --weight-decay "$TRAIN_WEIGHT_DECAY" \
  --beta1 0.9 --beta2 0.95 --adam-eps 1e-8 \
  --max-grad-norm "$TRAIN_MAX_GRAD_NORM" \
  --seed "$TRAIN_SEED" --log-every 1 --fused-adamw
```

Review the JSON, rendered torchrun argv, GPU profiles, DDP memory floor, storage
capabilities, evidence identities, and W&B path. A passing dry run is necessary
but not sufficient; it does not replace the GPU gates above.

## 5. Authorize and execute exactly once

Create the package lock, hardware contract, accepted-geometry receipt, training
recipe, and canonical launcher argv as described in
[pretraining-run-authority.md](../training/pretraining-run-authority.md). The
canonical argv must:

- point `--run-authority` at the intended durable output path outside Git;
- use `--execute`, exactly six ranks, and `--activation-checkpointing` before
  the `--` delimiter;
- contain every final path, cadence, W&B, optimizer, scheduler, seed, compile,
  and fused-optimizer decision explicitly.

Build the authority at that exact self-authorized path, then invoke the exact
JSON argv—do not reconstruct or reorder it. The production launcher now
revalidates the complete authority and requires the current canonical argv
digest to match before it starts torchrun.

Any changed argument, input byte, package, Git state, GPU contract, cost cap,
or resume generation requires a new canonical argv and a new write-once
authority. In particular, resume is a newly authorized launch, even when it
continues the same checkpoint trajectory.

## Final GO conditions

Start the long run only when all CPU tests pass; the final local copy is
read-only and certified; all six GPUs pass backend, memory, soak, checkpoint,
resume, validation, preemption, and W&B gates; projected time/cost is accepted;
and the execute invocation passes its self-bound immutable authority. Otherwise
stop and preserve the evidence—do not weaken context, precision, masks,
checkpointing, or validation to force a launch.
