# Immutable pretraining run authority

The production six-GPU run must have one write-once authority before it is
allowed to start. The authority is a local JSON artifact plus an adjacent
SHA-256 sidecar. It binds the scientific recipe, exact bytes of every data and
qualification input, the clean source tree, environment, launch command, and
cost envelope. Validation recollects all inputs; it does not merely check that
the authority JSON itself was not edited.

This is intentionally fail-closed. An absent field, dirty Git tree, changed
receipt, different package, stale tokenizer, non-six-GPU contract, command-line
default, or projected cost over the operator cap prevents authorization.

## What is bound

- full Git `HEAD` commit/tree object, a SHA-256 of canonical `git archive HEAD`,
  branch/origin digest metadata, and a clean
  tracked **and untracked** worktree;
- the exact Python executable, Python version, complete installed distribution
  set, and an immutable operator-created package lock;
- an operator-supplied container image digest in `sha256:<digest>` form;
- a homogeneous single-node, six-rank DDP hardware/runtime contract;
- a passing accepted-geometry receipt and sidecar, including order hashes,
  global microbatch size, accumulation, workers, compile/activation-checkpoint
  decisions, BF16/FP32 policy, six-GPU throughput, memory, checkpoint latency,
  input-wait fraction, scaling efficiency, and soak length;
- the exact passing final-corpus qualification generation, including its
  receipt and sidecar, corpus manifest and sidecar, mutation-sensitive corpus
  and tokenizer tree inventories, production 40/40/20 and token-budget policy,
  all train/validation/test evidence, and the exact qualifier source/runtime;
- fully checksummed train and validation order manifests and order payloads;
- passing full-data certification receipts and sidecars for both splits, bound
  to the current validator sources;
- tokenizer source-manifest SHA-256, semantic vocabulary SHA-256, vocabulary
  size, and all manifest-declared tokenizer files;
- the exact 1.3B model configuration and analytical parameter count
  (`1,283,557,376` parameters);
- AdamW values, warmup-cosine schedule, seed, W&B mode, deterministic policy,
  fused optimizer decision, checkpoint/eval/log cadence, and eval-at-start;
- aggregate measured throughput, projected duration, complete six-GPU pod
  hourly price, projected cost, and hard total cost cap;
- the exact launcher argv as a JSON string array, both its raw file digest and
  a canonical argv digest. Important path, geometry, recipe, and cadence values
  are cross-checked against the other inputs.

The expected hardware contract is not proof of the current CUDA host. The
launcher runtime preflight must still prove the six visible devices, BF16,
NCCL, topology, memory, and runtime immediately before `torchrun`. Likewise,
`scripts/qualify_cuda_model.py` is a separate bounded backend-correctness gate;
the accepted-geometry receipt is the full-topology throughput/memory gate.

## Required input formats

All JSON is UTF-8. Digests are lowercase SHA-256. Paths embedded by upstream
data certification must resolve to the exact current artifacts. Do not place
the output authority inside the Git worktree: creating it would make the tree
dirty on the subsequent validation.

### Package lock

Create the lock *inside the final production container on the GPU pod*, after
installing the production dependencies:

```bash
python scripts/build_pretraining_run_authority.py snapshot-package-lock \
  --output /network-volume/run-evidence/python-package-lock.json
```

The lock schema is:

```json
{
  "format": "pretraining-python-package-lock",
  "format_version": 1,
  "python": {"implementation": "cpython", "version": "3.12.11"},
  "packages": {
    "numpy": "2.2.6",
    "tokenizers": "0.21.4",
    "torch": "2.7.1+cu128"
  }
}
```

The real file contains every installed distribution, not only the abbreviated
example. The builder requires an exact installed-set match and enforces the
training dependency floors (`torch>=2.6`, `numpy>=2,<3`, and
`tokenizers>=0.21,<0.24`).

### Six-GPU hardware/runtime contract

This qualified JSON describes the one allowed topology. It is write-once and
must have an adjacent exact `.sha256` sidecar using the same two-space format
shown for the accepted-geometry receipt below. The RunPod pod qualifier
publishes both files and the authority builder rejects a missing, stale, or
malformed sidecar. Additional audit metadata is allowed, but these fields are
mandatory:

```json
{
  "format": "pretraining-six-gpu-hardware-runtime",
  "format_version": 1,
  "status": "accepted",
  "topology": "single-node",
  "world_size": 6,
  "gpu_count": 6,
  "gpu_model": "NVIDIA H100 80GB HBM3",
  "gpu_memory_bytes": 85198045184,
  "compute_capability": [9, 0],
  "multiprocessor_count": 120,
  "driver_version": "<exact value>",
  "cuda_runtime_version": "<exact value>",
  "cudnn_version": "<exact value>",
  "nccl_version": "<exact value>",
  "torch_version": "<exact installed distribution version>",
  "bf16_supported": true,
  "distributed_strategy": "ddp"
}
```

The exact values must come from the rented pod; the example GPU details are
illustrative, not defaults.

### Final corpus qualification

Pass the exact `qualification.json` produced by
`scripts/qualify_training_corpus.py`; do not copy fields into a new JSON. Its
generation directory must contain exactly `qualification.json` and
`qualification.json.sha256`, and the receipt must have `status: pass`. The
builder re-hashes the pair, corpus manifest and sidecar, every provenance
artifact, all three order manifests, and the six qualifier implementation
files. It also recomputes the qualifier's mutation-sensitive inventory for the
complete corpus and tokenizer trees, then joins the qualified train/validation
orders, tokenizer identity, model shape, EOS, vocabulary, and context length to
the other run-authority inputs.

The accepted policy permits at most the production defaults of 0.1% token
shortfall and `1e-6` absolute mixture error, requires at least eight sampled
rows per split/domain, and requires every full checksum, semantic, uniqueness,
document-index, and exact split-identity scan. A moved or copied corpus has new
filesystem identity and must be qualified at its final training location.

### Accepted geometry receipt

The full six-GPU qualification job must publish a write-once receipt and an
adjacent sidecar whose exact content is:

```text
<receipt-sha256><two spaces><receipt-filename><newline>
```

Minimum receipt schema:

```json
{
  "format": "pretraining-accepted-geometry",
  "format_version": 1,
  "status": "pass",
  "hardware_contract_sha256": "<hardware JSON SHA-256>",
  "train_order_manifest_sha256": "<train manifest SHA-256>",
  "validation_order_manifest_sha256": "<validation manifest SHA-256>",
  "accepted": {
    "global_microbatch_rows": 6,
    "gradient_accumulation_steps": 2,
    "workers": 4,
    "overfit_batch_rows": 6,
    "compile_model": false,
    "activation_checkpointing": true,
    "precision": "bfloat16",
    "parameter_dtype": "float32"
  },
  "measurements": {
    "aggregate_input_tokens_per_second": "1000",
    "peak_memory_allocated_bytes_per_gpu": 1,
    "peak_memory_reserved_bytes_per_gpu": 1,
    "minimum_free_memory_bytes_per_gpu": 1,
    "checkpoint_seconds": "<measured decimal>",
    "data_wait_fraction": "<0..1>",
    "scaling_efficiency": "<0..1>",
    "soak_steps": 100,
    "throughput_measurement": {
      "scope": "end-to-end-including-validation-checkpoint-wandb-and-resume",
      "timer": "time.monotonic_ns",
      "counter": "trainer.consumed_input_tokens",
      "start_consumed_input_tokens": 0,
      "end_consumed_input_tokens": 4915200,
      "elapsed_wall_time_ns": 4915200000000,
      "validation_events": 1,
      "checkpoint_events": 1,
      "wandb_log_events": 1,
      "resume_verified": true
    }
  }
}
```

Replace every illustrative value with a measured value. Global microbatch rows
must be divisible by six and must exactly match the frozen training order. The
external monotonic interval must include validation, checkpoint publication,
offline-W&B logging, and a verified six-rank resume. Require at least 100
optimizer updates; at least one validation, checkpoint, and W&B event; a token
delta exactly equal to `soak_steps * global_microbatch_rows *
gradient_accumulation_steps * sequence_length`; and reported throughput within
one part per million of token delta divided by elapsed wall time. Scaling
efficiency must be at least `0.70`, data-wait fraction at most `0.05`, and free
memory on every GPU at least `max(8 GiB, 10% of physical memory)`.

### Training recipe

The recipe makes values that would otherwise be defaults explicit:

```json
{
  "format": "pretraining-training-recipe",
  "format_version": 1,
  "model_size": "1.3b",
  "precision": "bfloat16",
  "parameter_dtype": "float32",
  "deterministic": true,
  "seed": 1234,
  "optimizer": {
    "name": "AdamW",
    "fused": true,
    "learning_rate": "0.0003",
    "weight_decay": "0.1",
    "beta1": "0.9",
    "beta2": "0.95",
    "epsilon": "0.00000001",
    "max_grad_norm": "1"
  },
  "schedule": {
    "name": "warmup-cosine",
    "minimum_learning_rate": "0.00003",
    "warmup_steps": 1000
  },
  "cadence": {
    "checkpoint_every": 1000,
    "eval_every": 1000,
    "eval_batches": 32,
    "eval_at_start": true,
    "log_every": 1
  },
  "wandb_mode": "offline",
  "activation_checkpointing": true,
  "compile_model": false
}
```

The builder proves cadence fits the available optimizer updates, warmup ends
before training, and evaluation fits complete validation microbatches.

### Canonical launcher argv

Store the final command as JSON, never as a shell string. This prevents quoting,
expansion, and wrapper ambiguity:

```json
[
  "/absolute/path/to/python",
  "/absolute/clean/repo/scripts/launch_pretraining.py",
  "--train-order-manifest", "/absolute/train/manifest.json",
  "--validation-order-manifest", "/absolute/validation/manifest.json",
  "--tokenizer", "/absolute/tokenizer",
  "--train-data-evidence", "/absolute/train-certification.json",
  "--validation-data-evidence", "/absolute/validation-certification.json",
  "--run-authority", "/network-volume/run-evidence/pretraining-authority.json",
  "--nproc-per-node", "6",
  "--model-size", "1.3b",
  "--activation-checkpointing",
  "--workers", "4",
  "--checkpoint-every", "1000",
  "--eval-every", "1000",
  "--eval-batches", "32",
  "--eval-at-start",
  "--wandb-mode", "offline",
  "--execute",
  "--",
  "--learning-rate", "0.0003",
  "--min-learning-rate", "0.00003",
  "--warmup-steps", "1000",
  "--weight-decay", "0.1",
  "--beta1", "0.9",
  "--beta2", "0.95",
  "--adam-eps", "0.00000001",
  "--max-grad-norm", "1",
  "--seed", "1234",
  "--log-every", "1",
  "--fused-adamw"
]
```

The real launcher also requires storage/checkpoint/preflight arguments; they
must be present in the real argv even though they are abbreviated above. The
1.3B DDP authority requires activation checkpointing. A false fused-optimizer
decision must use `--no-fused-adamw`. Compile-on must include `--compile`;
compile-off is its absence.

## Build and validate

Run from the final production environment with a clean checked-out commit:

```bash
python scripts/build_pretraining_run_authority.py build \
  --output /network-volume/run-evidence/pretraining-authority.json \
  --project-root /workspace/0-coding-llm \
  --package-lock /network-volume/run-evidence/python-package-lock.json \
  --container-image-digest sha256:<64-lowercase-hex> \
  --hardware-contract /network-volume/run-evidence/hardware.json \
  --geometry-receipt /network-volume/run-evidence/accepted-geometry.json \
  --corpus-qualification /local-data/qualification/corpus-v2/qualification.json \
  --train-order-manifest /local-data/corpus/orders/train/manifest.json \
  --validation-order-manifest /local-data/corpus/orders/validation/manifest.json \
  --train-certification /network-volume/run-evidence/train-certification.json \
  --validation-certification /network-volume/run-evidence/validation-certification.json \
  --tokenizer-root /local-data/tokenizer \
  --training-recipe /network-volume/run-evidence/training-recipe.json \
  --launcher-argv-json /network-volume/run-evidence/launcher-argv.json \
  --measured-input-tokens-per-second <exact-receipt-value> \
  --hourly-cost-usd <complete-six-GPU-pod-hourly-price> \
  --total-cost-cap-usd <operator-approved-cap>
```

Immediately before executing the canonical argv, an explicit validation is a
useful operator check:

```bash
python scripts/build_pretraining_run_authority.py validate \
  /network-volume/run-evidence/pretraining-authority.json
```

The authority and sidecar are never overwritten. Its publication path must
exactly equal canonical argv's absolute `--run-authority` value and must be
outside the Git worktree. Any change requires a new authority filename and a
fresh operator decision. `scripts/launch_pretraining.py --execute` now requires
this authority, recollects all bound inputs, and verifies that the current
canonical argv digest is the authorized digest before starting torchrun.

## Acceptance tests before renting the long-running pod

1. `python -m unittest tests.test_run_authority -v` passes.
2. Build succeeds only from the exact clean commit intended for the run.
3. Validation succeeds on the untouched authority in the final container.
4. On disposable copies, changing one byte in each of the recipe, argv, order,
   corpus qualification, corpus document index, certification, geometry,
   hardware, tokenizer, or package-lock inputs causes validation to fail;
   adding a file anywhere under the qualified corpus also fails.
5. Adding an untracked file or changing `HEAD` causes validation to fail.
6. Changing `--nproc-per-node`, a recipe value, an evidence path, or an explicit
   activation/fused flag in argv causes build to fail.
7. Setting the cost cap one cent below projected cost causes build to fail.
8. Re-running `build` at the same output path refuses to overwrite evidence.
9. The launcher CUDA preflight and backend qualification pass on the same pod;
   then the exact canonical argv—not a reconstructed shell command—is executed.
   The launcher itself reports the matching authority/argv digests.
