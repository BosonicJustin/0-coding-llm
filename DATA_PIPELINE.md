# Stack v3 raw-code collector

This pipeline streams the pinned `HuggingFaceCode/stack-v3-train` dataset and
saves only an explicit allowlist of programming languages. It maintains two
independent buckets:

- Python: 25,718,400,000 exact StarCoder2 content tokens.
- Other programming languages: 25,718,400,000 exact StarCoder2 content tokens.

Each target is 20% above the corresponding 21.432B-token final-code budget.
Once a bucket reaches its target, the collector stops tokenizing and saving
files for that category while continuing the other category.

## What is stored

Retained source files are stored byte-for-byte in lossless `.tar.zst` archives:

```text
/workspace/dataset/
  raw/
    python/part-000000.tar.zst
    other_code/part-000000.tar.zst
  tokenizer/starcoder2/
  manifests/STACK_V3_SOURCE.json
  state/
    collector_checkpoints/
    collector_parallel/PLAN.json
    collector_parallel/worker-NN/checkpoints/
    quota_records/
    COLLECTION_COMPLETE.json
```

Every archive contains an `_manifest.jsonl` member with repository, commit,
original file path, language, license, content ID, byte count, and exact token
count for every retained file. No token IDs are stored and no training format is
chosen yet.

## Upstream deduplication boundary

The pinned `HuggingFaceCode/stack-v3-train` v3.1 source is already file-level,
cross-repository near-deduplicated. Final curation therefore does not run a
second MinHash/LSH pass over Python or other code. It still audits exact and
normalized hashes to catch acquisition/retry anomalies, removes any residual
collisions deterministically, rejects benchmark contamination independently,
and groups complete repositories for leakage-safe split assignment.

`configs/curation_policy.json` records this boundary and the exact trusted
source revision. Validate it against the downloaded source manifest with:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/curation_policy.py \
  --root /workspace/dataset \
  --policy /workspace/coding_model_from_scratch/configs/curation_policy_fast_exact_normalized.json
```

FineWeb-Edu and Wikipedia remain a possible separate local near-deduplication
domain; their overlap is not covered by Stack v3's upstream processing.
Baseline v1 deliberately defers that expensive pass because both sources are
already curated. The fast production profile deterministically keeps one
global exact-hash canonical and then one global normalized-hash canonical;
these same hashes propagate contamination. It does not perform fuzzy semantic
near-deduplication. The raw inputs are preserved so the
implemented cross-source English pass can be tested later as a controlled
corpus ablation.

Stack v3 itself is hosted in Parquet source shards, so the Hugging Face reader
must decode Parquet internally. Those source shards are streamed from Hugging
Face and are never written to the network volume. The collector creates no
Parquet output.

## Strict language policy

`configs/code_languages.json` is a fail-closed allowlist. `Python` enters the
Python bucket. Only explicitly listed programming languages enter other-code.
HTML, CSS, Markdown, JSON, YAML, XML, notebooks, text, documentation, and every
unknown language are rejected. Vendored files are rejected. Both `permissive`
and `no_license` files are accepted for this personal research experiment.

Repositories and paths visibly associated with MBPP, EvalPlus, MultiPL-E,
MBXP, MXEval, or HumanEval are quarantined during acquisition. Deeper
content-based MBPP filtering is also mandatory during acquisition. The
collector loads `configs/mbpp_denylist.json`, a non-reversible fingerprint set
generated from the canonical 974-row MBPP JSONL. It rejects canonical solution
files, embedded solution line pairs, benchmark problem/test lines, and obvious
MBPP payloads before tokenization or archiving. The denylist SHA-256 is pinned
in every source manifest and checkpoint, so a changed or missing policy fails
closed on resume. Run `scripts/audit_mbpp.py` against saved Python archives as
an independent verification step. Other evaluation suites still need deeper
content-based decontamination before final training selection.

## RunPod installation

Confirm that the network volume is mounted at `/workspace`, then install:

```bash
cd /workspace/coding_model_from_scratch
python -m venv .venv
.venv/bin/pip install -r requirements-data.txt
mkdir -p /workspace/dataset

.venv/bin/python scripts/download_tokenizer.py \
  --output /workspace/dataset/tokenizer/starcoder2
```

The tokenizer downloader resolves upstream `main` once, records the exact
40-character commit, verifies the 49,152-token vocabulary and special tokens,
and reuses that commit on every restart.

## Small pilot

Use a separate root so pilot data does not enter the production corpus:

```bash
.venv/bin/python scripts/collect_stack_v3.py \
  --root /workspace/dataset-pilot \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --cache-dir /tmp/huggingface-cache \
  --checkpoint-repos 10 \
  --max-new-repos 10 \
  --min-free-gb 50
```

Inspect the result:

```bash
find /workspace/dataset-pilot/raw -type f -maxdepth 2 -ls
.venv/bin/python scripts/quota_tracker.py \
  --root /workspace/dataset-pilot status --phase collection
```

## Parallel production run

Use the included launcher inside `tmux` so an SSH disconnect does not terminate
it and every progress record is retained in a log:

```bash
tmux new-session -d -s stack-v3 \
  /workspace/coding_model_from_scratch/scripts/run_download.sh

tail -f /workspace/dataset/logs/collector.log
```

The production launcher uses `collect_stack_v3_parallel.py`. It pins the current
Stack v3 commit, hash-orders all 8,192 source shards with a fixed seed, and runs
eight process-isolated streaming workers. On its first run it imports the old
sequential `(source shard, row)` cursor into an immutable parallel plan. Worker
zero resumes the partially consumed source shard at that exact row; workers one
through seven begin on the next seven shards. Later assignments advance by the
fixed worker count, so source shards never overlap.

Every worker has its own checkpoint directory, Hugging Face cache, rejection
ledger, and mathematically disjoint archive-index sequence. Accepted files are
tokenized in ordered batches of 64. Checkpoints are intentionally smaller than
the legacy collector—100 MB or 10,000 repositories—so aggregate quota overshoot
from concurrently open workers stays bounded. Reusing the launcher resumes each
worker from its exact cursor. `SIGINT`, `SIGTERM`, and `SIGHUP` ask every worker
to finish its current repository, close and fsync its archives, checkpoint, and
exit.

## Progress and completion

```bash
tail -n 20 /workspace/dataset/logs/collector.log

.venv/bin/python scripts/quota_tracker.py \
  --root /workspace/dataset status --phase collection
```

Worker JSON progress lines report worker ID, source shard and row, repositories
consumed, exact tokens and targets, retained throughput, free disk, and skip
counts. Supervisor `parallel_status` lines report globally committed totals and
active worker count. Progress is emitted every 1,000 repositories, at every
checkpoint, and every 15 seconds by the supervisor. Attach with
`tmux attach -t stack-v3`. Content-level benchmark rejections are written without
source text to worker-specific
`/workspace/dataset/logs/benchmark_rejections.worker-NN.jsonl` files. Separate
ledgers eliminate concurrent append races while retaining deterministic IDs.

Audit MBPP fingerprints at any checkpoint with:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/audit_mbpp.py \
  --root /workspace/dataset
```

The collector exits automatically after both buckets reach their thresholds and
writes `/workspace/dataset/state/COLLECTION_COMPLETE.json`. A complete file is
never cut in half, so each category will normally overshoot its target slightly.
This code marker alone is not the whole-corpus acceptance signal. The production
preprocess gate also requires the FineWeb-Edu and Wikipedia completion markers,
their exact source/tokenizer/benchmark identities and token totals, all four
configured collection targets, exact quota-ledger/raw archive equality, and no
hidden `.part-*` archive left by a live collector. These checks are exposed as
`STATUS.json.collection_closure` and are mandatory in `run_preprocess.sh`.

Before deleting the CPU pod, verify:

```bash
df -h /workspace
test -f /workspace/dataset/state/COLLECTION_COMPLETE.json
du -sh /workspace/dataset
```

The `/tmp/huggingface-cache` directory is disposable. The raw archives,
manifests, tokenizer, checkpoints, and quota records under `/workspace/dataset`
must remain on the network volume.

The required processing, audit, split, packing, smoke-test, and training
readiness steps after collection are specified in `PRETRAINING_CHECKLIST.md`.
Finalized archives are already being validated and fingerprinted concurrently
with acquisition as described in `STREAMING_PREPROCESS.md`.
