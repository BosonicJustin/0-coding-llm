# RunPod six-GPU pod bootstrap and qualification

This is the environment boundary that runs **after the final pod exists but
before any GPU training job starts**. It does not install or rent anything by
itself. `scripts/qualify_runpod_pod.py` is read-only except for a bounded NCCL
scratch directory on local storage and one new immutable hardware receipt plus
its SHA-256 sidecar on the network volume.

The receipt uses the existing
`pretraining-six-gpu-hardware-runtime` v1 format. Pass that exact JSON to
`scripts/build_pretraining_run_authority.py build --hardware-contract`; no
hand-written hardware JSON is needed.

## What is and is not locked by `requirements-train.txt`

`requirements-train.txt` intentionally gives compatibility ranges, including
`torch>=2.6`; it is an installation input, not a reproducible lock. The CUDA
wheel must be selected for the final driver/GPU, and the complete installed
distribution set must then be frozen with `snapshot-package-lock`. Both the pod
receipt and final run authority verify that exact lock against the interpreter
that is actually running.

Do not reuse a package lock from another image or run. Do not install another
package after taking the lock. If the environment changes, use a new lock path,
new pod receipt path, and new run authority.

## Pod configuration before opening the shell

Configure one node with exactly six instances of one GPU model. Allocate at
least 32 GiB of `/dev/shm` and enough local NVMe for the complete packed corpus
plus explicit headroom. Mount the persistent RunPod network volume separately.
The accepted production layout is:

- writable local work/storage filesystem;
- a read-only bind mount of the final local packed-data copy on the same local
  device;
- writable network filesystem for checkpoints, W&B, package locks, pod
  receipts, certifications, and run authorities;
- local and network roots with different device IDs.

The verifier rejects overlay/tmpfs as local data, an ordinary writable data
directory, a non-network durable root, insufficient free space, small
`/dev/shm`, heterogeneous or MIG GPUs, and ambiguous device visibility.

## Exact pod-side bootstrap

Run from a fresh shell as root. Fill the path variables from the final data
publication. `TORCH_PIN` and `PYTORCH_INDEX_URL` are mandatory operator choices:
they must name one exact official PyTorch CUDA wheel suitable for the selected
GPU and driver. The commands stop instead of guessing either value.

```bash
set -euo pipefail

REPO=/workspace/0-coding-llm
VENV=/workspace/pretrain-venv
NETWORK_ROOT=/runpod-volume
LOCAL_WORK_ROOT=/workspace
LOCAL_DATA_RW=/workspace/pretraining-data
LOCAL_DATA_RO=/workspace/pretraining-data-ro
AUDIT_DIR="$NETWORK_ROOT/run-evidence/pod-v1"
WANDB_DIR="$NETWORK_ROOT/wandb"

: "${TORCH_PIN:?set an exact value such as torch==X.Y.Z}"
: "${PYTORCH_INDEX_URL:?set the matching official CUDA wheel index URL}"

test -d "$REPO/.git"
test -d "$NETWORK_ROOT"
test -d "$LOCAL_DATA_RW"
mkdir -p "$AUDIT_DIR" "$WANDB_DIR" "$LOCAL_DATA_RO"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install \
  --index-url "$PYTORCH_INDEX_URL" "$TORCH_PIN"
"$VENV/bin/python" -m pip install \
  -r "$REPO/requirements-train.txt" \
  -r "$REPO/requirements-wandb.txt"

mountpoint -q "$LOCAL_DATA_RO" || mount --bind "$LOCAL_DATA_RW" "$LOCAL_DATA_RO"
mount -o remount,bind,ro "$LOCAL_DATA_RO"
mount -o remount,size=32G /dev/shm

ulimit -n 1048576
ulimit -l unlimited
ulimit -s 8192

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE LOCAL_RANK RANK WORLD_SIZE

cd "$REPO"
git status --short
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

The bind-mount sequence is safe to repeat: an existing mount is remounted
read-only. It does not delete or replace the source copy. If `ulimit -l
unlimited`, the `/dev/shm` remount, or the clean-tree assertion fails, change
the pod/container configuration; do not weaken the qualification threshold.

The bootstrap commands install packages, so run them only when intentionally
preparing the newly rented pod. The qualification script itself never runs any
of these commands.

## Freeze the exact environment

Choose new, previously unused paths. `snapshot-package-lock` refuses to
overwrite an existing lock.

```bash
RUN_ID=pod-v1
PACKAGE_LOCK="$AUDIT_DIR/package-lock-$RUN_ID.json"
HARDWARE_RECEIPT="$AUDIT_DIR/hardware-$RUN_ID.json"

test ! -e "$PACKAGE_LOCK"
test ! -e "$HARDWARE_RECEIPT"
test ! -e "$HARDWARE_RECEIPT.sha256"

"$VENV/bin/python" scripts/build_pretraining_run_authority.py \
  snapshot-package-lock --output "$PACKAGE_LOCK"
```

The current package-lock publisher does not create a sidecar; the final run
authority hashes the lock itself. The pod hardware receipt does create and
require its exact sidecar.

## Run the fail-closed pod verifier

Set the final local paths and explicit capacity policy. Network free space must
cover the atomic-rotation peak of three mature checkpoint generations (latest,
previous, and the next temporary generation), plus logs/evidence and at least
`max(1 GiB, 10% of one generation)` headroom. Local free space is headroom
**after** the packed corpus has been
copied. The numbers below are conservative launch candidates, not inferred
facts about the eventual checkpoint size; raise them when measured artifacts
require more.

```bash
TRAIN_ORDER="$LOCAL_DATA_RO/orders/train/manifest.json"
VALIDATION_ORDER="$LOCAL_DATA_RO/orders/validation/manifest.json"
TOKENIZER_ROOT="$LOCAL_DATA_RO/tokenizer"
EVAL_BATCHES=8
MIN_NETWORK_FREE=200GiB
MIN_LOCAL_FREE=64GiB

"$VENV/bin/python" scripts/qualify_runpod_pod.py verify \
  --network-root "$NETWORK_ROOT" \
  --local-work-root "$LOCAL_WORK_ROOT" \
  --local-data-root "$LOCAL_DATA_RO" \
  --tokenizer "$TOKENIZER_ROOT" \
  --train-order-manifest "$TRAIN_ORDER" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --package-lock "$PACKAGE_LOCK" \
  --wandb-dir "$WANDB_DIR" \
  --receipt "$HARDWARE_RECEIPT" \
  --wandb-mode offline \
  --nvlink-policy require-all \
  --omp-threads "$OMP_NUM_THREADS" \
  --eval-batches "$EVAL_BATCHES" \
  --minimum-network-free-bytes "$MIN_NETWORK_FREE" \
  --minimum-local-free-bytes "$MIN_LOCAL_FREE" \
  --minimum-shm-bytes 16GiB \
  --minimum-nofile 65536 \
  --minimum-stack-bytes 8MiB

(cd "$(dirname "$HARDWARE_RECEIPT")" && \
  sha256sum -c "$(basename "$HARDWARE_RECEIPT").sha256")
```

`require-all` is the recommended policy for an NVSwitch/NVLink-connected pod:
all 15 unordered GPU pairs must be reported as `NV#`. For an intentionally
accepted PCIe topology, use `observe`; full directed CUDA peer access and the
real six-rank NCCL collective smoke still remain mandatory. `require-any`
accepts a partially linked topology. The policy is stored in the receipt. A
failed strict run must not be “repaired” by overwriting the receipt or changing
the policy at the same path—make an explicit new decision and use a new path.

The verifier performs all of these checks before publication:

1. exact six-entry `CUDA_VISIBLE_DEVICES`, CUDA/PyTorch UUID ordering, full-GPU
   (non-MIG) identity, six homogeneous names, VRAM, compute capabilities and SM
   counts, and native BF16 on every visible device;
2. driver-supported CUDA at least as new as the CUDA runtime compiled into
   PyTorch, non-empty cuDNN/NCCL versions, a compiled target for the observed
   GPU architecture, and `torch.distributed` NCCL availability;
3. symmetric 6x6 peer-access and `nvidia-smi topo -m` matrices, with the chosen
   NVLink policy;
4. an actual six-process NCCL all-reduce, all-gather, broadcast, barrier, and
   per-rank CUDA synchronization, with rank `N`, local rank `N`, and the
   expected visible GPU UUID all agreeing for every rank;
5. network/local/read-only-bind mount classification, separate network/local
   devices, explicit free-space floors, tmpfs `/dev/shm`, `RLIMIT_NOFILE`,
   `RLIMIT_STACK`, and unlimited `RLIMIT_MEMLOCK`;
6. authenticated tokenizer identity, train/validation order metadata and
   frozen six-rank geometry, exact local path containment, and the exact Python
   package lock;
7. deterministic/robustness environment values and W&B dependency/credential
   availability without recording the API key or contacting W&B.

Order inspection is metadata-only. Full payload certification receipts remain
mandatory at launch; this pod check does not re-hash hundreds of gigabytes.

## Bind the receipt into run authority

Use the exact receipt path—never copy values out by hand:

```bash
"$VENV/bin/python" scripts/build_pretraining_run_authority.py build \
  --output "$FINAL_RUN_AUTHORITY" \
  --project-root "$REPO" \
  --package-lock "$PACKAGE_LOCK" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
  --hardware-contract "$HARDWARE_RECEIPT" \
  --geometry-receipt "$GEOMETRY_RECEIPT" \
  --corpus-qualification "$CORPUS_QUALIFICATION" \
  --train-order-manifest "$TRAIN_ORDER" \
  --validation-order-manifest "$VALIDATION_ORDER" \
  --train-certification "$TRAIN_CERTIFICATION" \
  --validation-certification "$VALIDATION_CERTIFICATION" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --training-recipe "$TRAINING_RECIPE" \
  --launcher-argv-json "$LAUNCHER_ARGV_JSON" \
  --measured-input-tokens-per-second "$MEASURED_TOKENS_PER_SECOND" \
  --hourly-cost-usd "$HOURLY_COST_USD" \
  --total-cost-cap-usd "$TOTAL_COST_CAP_USD"
```

The receipt proves pod availability at its timestamp; it does not replace the
model CUDA correctness test, one-chunk overfit, full-topology geometry/memory
soak, checkpoint/resume/preemption tests, data certification, or the launcher's
immediate preflight. Run those gates from
`six-gpu-launch-qualification.md` before the long run.

## Failure and rerun rules

- Exit `0` means the receipt and sidecar were durably published. Exit `2`
  means no training is authorized.
- Receipt and sidecar paths are write-once. If one exists, the command refuses
  to overwrite or silently repair the pair. Investigate and choose a new path.
- NCCL scratch is created under the explicit local work root and removed after
  success or failure. A timed-out torchrun has its complete subprocess session
  terminated (TERM, then KILL after a bounded grace period), so orphaned NCCL
  ranks cannot survive qualification. No corpus or tokenizer bytes are modified.
- `WANDB_MODE=online` additionally requires credentials, but the verifier does
  not transmit the key or make a network login call. Qualify offline first when
  W&B reachability should not be a training dependency.
- Any package, source, driver, CUDA, GPU, mount, environment, tokenizer, or data
  path change requires a new package lock, hardware receipt, and run authority.
  The authority compares the receipt's complete installed-package identity,
  clean Git identity, qualifier/requirements hashes, and qualifier argv with
  the current production inputs rather than trusting the receipt label.
