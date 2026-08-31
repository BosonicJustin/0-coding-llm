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

This operator-reviewed JSON describes the one allowed topology. Additional
audit metadata is allowed, but these fields are mandatory:

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
    "gradient_accumulation_steps": 32,
    "workers": 4,
    "overfit_batch_rows": 6,
    "compile_model": false,
    "activation_checkpointing": true,
    "precision": "bfloat16",
    "parameter_dtype": "float32"
  },
  "measurements": {
    "aggregate_input_tokens_per_second": "<measured decimal>",
    "peak_memory_allocated_bytes_per_gpu": 1,
    "peak_memory_reserved_bytes_per_gpu": 1,
    "minimum_free_memory_bytes_per_gpu": 1,
    "checkpoint_seconds": "<measured decimal>",
    "data_wait_fraction": "<0..1>",
    "scaling_efficiency": "<0..1>",
    "soak_steps": 100
  }
}
```

Replace every illustrative value with a measured value. Global microbatch rows
must be divisible by six and must exactly match the frozen training order.

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
  "--nproc-per-node", "6",
  "--model-size", "1.3b",
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
  "--activation-checkpointing",
  "--fused-adamw"
]
```

The real launcher also requires storage/checkpoint/preflight arguments; they
must be present in the real argv even though they are abbreviated above. False
activation-checkpointing/fused decisions must use the corresponding explicit
`--no-*` flag. Compile-on must include `--compile`; compile-off is its absence.

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
  --train-order-manifest /network-volume/packed/train/order/manifest.json \
  --validation-order-manifest /network-volume/packed/validation/order/manifest.json \
  --train-certification /network-volume/run-evidence/train-certification.json \
  --validation-certification /network-volume/run-evidence/validation-certification.json \
  --tokenizer-root /network-volume/tokenizer \
  --training-recipe /network-volume/run-evidence/training-recipe.json \
  --launcher-argv-json /network-volume/run-evidence/launcher-argv.json \
  --measured-input-tokens-per-second <exact-receipt-value> \
  --hourly-cost-usd <complete-six-GPU-pod-hourly-price> \
  --total-cost-cap-usd <operator-approved-cap>
```

Immediately before executing the canonical argv:

```bash
python scripts/build_pretraining_run_authority.py validate \
  /network-volume/run-evidence/pretraining-authority.json
```

The authority and sidecar are never overwritten. Any change requires a new
authority filename and a fresh operator decision. Until the production launcher
accepts the authority as a mandatory argument, validation and launch must be
treated as one operational step; launcher integration is deliberately outside
this isolated implementation.

## Acceptance tests before renting the long-running pod

1. `python -m unittest tests.test_run_authority -v` passes.
2. Build succeeds only from the exact clean commit intended for the run.
3. Validation succeeds on the untouched authority in the final container.
4. On disposable copies, changing one byte in each of the recipe, argv, order,
   certification, geometry, hardware, tokenizer, or package-lock inputs causes
   validation to fail.
5. Adding an untracked file or changing `HEAD` causes validation to fail.
6. Changing `--nproc-per-node`, a recipe value, an evidence path, or an explicit
   activation/fused flag in argv causes build to fail.
7. Setting the cost cap one cent below projected cost causes build to fail.
8. Re-running `build` at the same output path refuses to overwrite evidence.
9. The launcher CUDA preflight and backend qualification pass on the same pod;
   then the exact canonical argv—not a reconstructed shell command—is executed.
