# Prime Intellect SFT integration

This repository uses PrimeRL for fixed-dataset supervised fine-tuning. A
`verifiers` environment is **not** in the SFT data path: environments generate
or score rollouts, while ordinary SFT teacher-forces immutable target tokens.
The small coding environment under `environments/` is reserved for later
online evaluation/RL smoke tests and must never supply examples to the SFT
train or validation splits.

The reviewed upstream identities are:

- [`prime-rl@3fc28dd`](https://github.com/PrimeIntellect-ai/prime-rl/tree/3fc28ddfb354f336d1cc28e8e032f262f5aa68b2)
- [`renderers@c4772ac`](https://github.com/PrimeIntellect-ai/renderers/tree/c4772ac1321c69e83d2b4460600072911cc41a0b)
- OpenCodeInstruct
  `nvidia/OpenCodeInstruct@8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`
- StarCoder2 tokenizer
  `bigcode/starcoder2-tokenizer@9cfe60e28fd01cc1391ecd2146a34cda7534efeb`

Do not silently advance any of these pins. PrimeRL and `renderers` are moving
interfaces; a dependency update requires rerunning the renderer, config,
packing, boundary-isolation, checkpoint/resume, and GPU gates.

## Why the model takes Prime's custom Llama path

The native pretraining checkpoint is parameter-compatible with Llama but its
repository model accepts `document_ids`, which generic Hugging Face Llama does
not. `scripts/export_hf_checkpoint.py` publishes an exact HF Llama checkpoint.
PrimeRL then loads that export with `model.impl = "custom"`.

This distinction is mandatory. Prime's SFT loader packs multiple independent
conversations into a fixed physical row and emits their lengths in `seq_lens`.
At the reviewed revision, Prime passes `seq_lens` into its custom model classes
and uses variable-length attention, so packed conversations cannot attend to
one another. Its generic HF path does not receive those boundaries. Never
change the config to `impl = "hf"` while packing is enabled.

The relevant upstream implementations are PrimeRL's
[`SFTDataset`/`CatDataset`](https://github.com/PrimeIntellect-ai/prime-rl/blob/3fc28ddfb354f336d1cc28e8e032f262f5aa68b2/src/prime_rl/trainer/sft/data.py)
and its
[`forward` adapter](https://github.com/PrimeIntellect-ai/prime-rl/blob/3fc28ddfb354f336d1cc28e8e032f262f5aa68b2/src/prime_rl/trainer/model.py).

## Derived dataset contract

The certified raw snapshot is immutable and stays at:

```text
/workspace/posttraining-data/sft/opencodeinstruct/raw
```

`scripts/prepare_prime_sft.py` creates a new artifact at:

```text
/workspace/posttraining-data/sft/opencodeinstruct/derived/prime-sft-v1/
├── data/
│   ├── train-*.parquet
│   └── validation-*.parquet
├── audit/
│   ├── contamination-*.json
│   ├── rejected-*.parquet
│   ├── shard-*.json
│   └── CONTAMINATION_GROUPS.json
├── COMPLETION.json
├── DATASET_MANIFEST.json
├── IDENTITY.json
└── README.md
```

Training rows use the Hugging Face/Prime `messages` shape:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

They also retain non-sensitive provenance and quality fields. Unit tests,
execution traces, and LLM judgements are omitted from training files.

The curation algorithm is:

1. Authenticate the certified 50-shard, 5,000,000-row raw inventory and the
   exact tokenizer manifest; refuse changed bytes, symlinks, or schema drift.
2. Scan prompt, completion, and unescaped unit-test text against the frozen
   MBPP denylist. Build one global set of contaminated normalized prompt-group
   IDs before accepting any row.
3. Validate IDs, text, source enum values, score range, NUL/byte limits, and
   high-confidence credential signatures. Reject every row in a contaminated
   group. Metadata JSON parsing is intentionally disabled because those fields
   are neither published nor trained; the Arrow schema still requires strings.
4. Render with `starcoder2-coding-chat-v1` and the exact tokenizer. Drop a
   complete conversation when the causal-shifted input would exceed 4,096
   tokens; never truncate an answer.
5. Assign the normalized prompt group deterministically to train or validation
   (0.2% validation). Exact Unicode/case/whitespace variants therefore cannot
   cross splits. This is grouping, not fuzzy deduplication.
6. Write Zstandard Parquet shards and ID/reason-only rejection ledgers. Empty
   worker shards are omitted because Hugging Face Datasets 4.x cannot reliably
   discover a split containing empty Parquet shards.
7. Authenticate every worker result, aggregate examples/tokens/reasons, write
   the dataset card and manifests, then atomically rename the completed work
   directory into place. An interrupted run resumes only from checksummed
   source-shard sidecars.

The first pass is the only benchmark fingerprint pass. The second pass trusts
its checksummed global group authority after re-authenticating the raw shard,
avoiding a duplicate fingerprint scan and five million unnecessary metadata
JSON parses.

Run it on the CPU pod/network volume:

```bash
python scripts/prepare_prime_sft.py \
  --root /workspace/posttraining-data/sft/opencodeinstruct \
  --tokenizer-root /workspace/dataset/tokenizer/starcoder2 \
  --output /workspace/posttraining-data/sft/opencodeinstruct/derived/prime-sft-v1 \
  --workers 8 \
  --read-batch-size 2048 \
  --output-row-group-size 4096
```

The current policy is
`configs/posttrain/opencodeinstruct_prime_sft_v1.json`. Its minimum test score
is deliberately permissive (`0.0`) until the real score/retention distribution
has been audited. Freeze a chosen quality threshold in a **new** policy/output
version before actual SFT; do not rewrite `prime-sft-v1` after publication.

## Chat format and labels

`starcoder2-coding-chat-v1` adds no vocabulary entries. It independently
encodes ordinary ASCII role scaffolding and each message body:

```text
<|user|>\nPROMPT\n<|assistant|>\nANSWER<EOS-0>
```

Independent component encoding is part of the frozen format; it prevents a
BPE merge across the inference generation boundary and gives exact token
attribution. User text, role scaffolding, and separators are masked. Assistant
body tokens and terminal EOS ID `0` are supervised. EOS is also the generation
stop, so the model learns when to end an answer rather than always emitting
4,096 tokens.

The canonical renderer, Prime adapter, and pinned installer are documented in
`integrations/prime_intellect/README.md`.

## Export the pretrained checkpoint

Only export a trusted checkpoint; PyTorch `.pt` files are pickle containers.
The destination must not already exist.

```bash
python scripts/export_hf_checkpoint.py \
  --checkpoint /workspace/pretraining-runs/FINAL_CHECKPOINT.pt \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --output /workspace/posttraining-models/coding-llm-base-hf \
  --max-shard-size 5GB
```

The exporter rejects missing/extra keys, wrong shapes or dtypes, non-finite
weights, config disagreement, tokenizer changes, and partial publication. A
format-v5 trainer checkpoint supplies its mandatory manifest and canonical
vocabulary SHA-256 identities automatically; the source tokenizer manifest and
every file it declares must verify exactly. It
writes safetensors and `NATIVE_EXPORT_MANIFEST.json` atomically. Because the HF
export intentionally rewrites tokenizer `model_max_length`, its regenerated
`TOKENIZER_MANIFEST.json` hashes the derived files as actually published. The
input manifest is preserved byte-for-byte as `SOURCE_TOKENIZER_MANIFEST.json`
for source provenance only; it is never presented as integrity metadata for
the modified export.

## Install the reviewed Prime runtime

Use clean detached checkouts. The PrimeRL repository consumes its local
`deps/renderers` workspace package, so check that dependency out at the exact
reviewed renderer commit before applying the patch:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /opt/prime/prime-rl
git -C /opt/prime/prime-rl checkout --detach 3fc28ddfb354f336d1cc28e8e032f262f5aa68b2
git -C /opt/prime/prime-rl submodule update --init --recursive
git -C /opt/prime/prime-rl/deps/renderers fetch origin c4772ac1321c69e83d2b4460600072911cc41a0b
git -C /opt/prime/prime-rl/deps/renderers checkout --detach c4772ac1321c69e83d2b4460600072911cc41a0b

python scripts/apply_prime_renderer_patch.py \
  --renderers-checkout /opt/prime/prime-rl/deps/renderers \
  --prime-rl-checkout /opt/prime/prime-rl \
  --check-only

python scripts/apply_prime_renderer_patch.py \
  --renderers-checkout /opt/prime/prime-rl/deps/renderers \
  --prime-rl-checkout /opt/prime/prime-rl
```

The installer refuses different commits, a dirty renderer checkout, conflicting
files, or changed source/patch hashes. A repeated install must report
`already-installed`. If GitHub SSH submodule URLs are unavailable on the pod,
override those submodule URLs with their HTTPS equivalents; do not substitute
different commits.

Build/freeze the exact Linux, Python 3.12, Torch, CUDA, driver, PrimeRL, and
renderer environment only on the selected GPU image. PrimeRL's current GPU
dependency set is hardware-specific and must pass the CUDA/NCCL smoke gates
before it becomes the experiment lock.

## Six-GPU run contract

`configs/posttrain/prime_sft_6gpu.toml` is a validation-first scaffold:

- one six-GPU node, six training ranks, no inference GPU;
- custom Llama implementation and `seq_lens` boundary isolation;
- context 4,096, BF16 optimization/reduction;
- full activation checkpointing with activation CPU offloading disabled for the
  1.3B/six-GPU baseline (re-enable only if the memory smoke requires it);
- global batch 48 packed rows, microbatch one per rank, therefore eight
  gradient-accumulation rounds per optimizer step;
- assistant-only loss;
- deterministic shuffle, validation every 50 steps;
- one data-loader worker per rank for both train and validation; the pinned
  PrimeRL stream captures rank before worker initialization and higher values
  duplicate examples rather than safely sharding the stream;
- W&B plus local JSONL metrics;
- resume-capable checkpoints on the network volume (the configured checkpoint
  base omits the final `checkpoints/` component because PrimeRL appends it);
- `dry_run = true` and a placeholder 1,000-step schedule.

Authenticate W&B through `WANDB_API_KEY`; never put a credential in TOML or the
repository. Do **not** invoke `uv run sft` directly. The checked-in TOML is
byte-pinned by `configs/posttrain/prime_sft_launch_v1.json`, and the only
supported entry point is the fail-closed wrapper:

```bash
python /workspace/0-coding-llm/scripts/launch_prime_sft.py \
  --mode verify-only \
  --prime-rl-checkout /opt/prime/prime-rl \
  --cache-root /workspace/posttraining-cache/prime-sft-v1

python /workspace/0-coding-llm/scripts/launch_prime_sft.py \
  --mode prewarm \
  --prime-rl-checkout /opt/prime/prime-rl \
  --cache-root /workspace/posttraining-cache/prime-sft-v1

python /workspace/0-coding-llm/scripts/launch_prime_sft.py \
  --mode launch \
  --prime-rl-checkout /opt/prime/prime-rl \
  --cache-root /workspace/posttraining-cache/prime-sft-v1
```

`verify-only` performs no writes. It re-authenticates the curated dataset's
complete exact tree, every HF export and safetensors byte, the derived
tokenizer, and the source-tokenizer manifest plus `tokenizer.json` identity
shared by curation and model export. It also verifies both Git commits, the
installed renderer patch, and the complete TOML hash/contract. `prewarm` is the
one intentional cache-writing mode. Run it
once from a single process before Prime starts six ranks; it loads both local
HF splits and the local tokenizer, then atomically writes
`PREWARM_COMPLETE.json` bound to the dataset, model, config, split row counts,
and cache paths. `launch` repeats all gates and refuses to exec Prime without
that matching marker or its exact cache file/size inventory. This avoids six
ranks racing to build the same Arrow cache and detects a deleted or truncated
prewarm before ranks start.

The wrapper always exports explicit `HF_HOME`, `HF_DATASETS_CACHE`, and
`HF_HUB_CACHE` beneath the selected cache root. It also forces Hugging Face and
Transformers offline because every training input is local. Put the cache on
the network volume when it must survive pod replacement, or on adequately
sized local NVMe when the prewarm and training run stay on the same pod. Never
put it inside the authenticated dataset, model export, or Prime checkout. An
unwarmed filesystem must have at least the greater of 16 GiB and twice the
published data-file bytes plus 4 GiB free; a validated warm cache preserves a
4 GiB operating reserve.

The checked-in contract can only exec Prime with `dry_run = true`. Before real
training:

1. require complete raw, curated, tokenizer, model-export, and renderer
   manifests;
2. load both local HF splits and re-render sampled rows with exact length/mask
   equality;
3. run one CPU packing test proving separate `seq_lens`, reset positions, and
   assistant/EOS-only labels;
4. run one-GPU forward/backward, single-pack overfit, BF16, and memory tests;
5. run a six-rank NCCL step and prove conversation-A perturbations do not alter
   conversation-B logits;
6. measure throughput and decide activation offloading/compilation from data,
   not defaults;
7. freeze the quality threshold, effective batch, LR, warmup, epoch/step
   budget, exact environment lock, and a new unique `run.name`;
8. copy the checked config as an immutable run artifact, set
   `dry_run = false`, publish a new launch contract, and launch through the
   wrapper.

### Training approval after the quality audit

There is deliberately no automatic score threshold. Measure the real
`average_test_score` distribution, decide the retention/quality tradeoff,
publish a new curated dataset whose policy has a strictly positive
`min_average_test_score`, and retain the full audit as an immutable JSON file.
The audit has exact outer fields
`format_version=1`, `kind="opencodeinstruct_quality_audit"`,
`source_repo_id`, `source_revision`, `score_field="average_test_score"`,
`source_rows`, a non-empty `measurement` object, and timezone-aware
`created_at_utc`. The measurement object holds the reported histogram,
quantiles, retention calculations, and methodology; the launcher authenticates
all of those bytes but deliberately does not turn them into an automatic
threshold decision.

For a future non-dry run, pass a sibling `TRAINING_APPROVAL.json` through
`--training-approval`. The wrapper requires this exact v1 shape:

```json
{
  "format_version": 1,
  "kind": "prime_sft_training_approval",
  "status": "approved",
  "dataset_manifest_sha256": "...",
  "curation_identity_sha256": "...",
  "policy_sha256": "...",
  "quality_audit": {"path": "quality-audit-v2.json", "bytes": 1234, "sha256": "..."},
  "decision": {"min_average_test_score": 0.5, "accepted_train_rows": 1000000},
  "approved_at_utc": "...",
  "approval_sha256": "..."
}
```

`approval_sha256` is SHA-256 over canonical, sorted, indented JSON for the
object without that field (including the trailing newline). The selected
threshold and accepted row count must equal the authenticated derived dataset;
the audit record must authenticate a plain sibling file. The example numbers
are schema illustrations, **not** a recommended threshold or expected row
count. The current `0.0` corpus may be verified and prewarmed, but the wrapper
rejects it for a non-dry execution.

Do not select checkpoints on MBPP. MBPP is final-evaluation-only and is run
once the post-training decisions are frozen.
