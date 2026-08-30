# Native PyTorch training-data contract

The code in `pretrain/data.py` is the training-side boundary between the final
curated corpus and the model. It is implemented and tested with synthetic
fixtures. It does **not** mean that the real corpus has completed filtering,
residual code duplicate handling, English near-deduplication, decontamination,
splitting, tokenization, or packing yet.

The pretraining hot path never reads raw tar archives, JSONL, SQLite, or source
text. Those remain preprocessing inputs and provenance artifacts.

## Packed row format

For context length `T`, each row stores:

- `T + 1` little-endian uint16 token IDs;
- `ceil((T + 1) / 8)` segment-start bits using NumPy little-bit order.

The first `T` tokens are model inputs. Tokens `1..T` are their next-token
targets, so the extra token supplies the final target without padding. Rows
advance by exactly `T` tokens; the look-ahead target is consequently the first
input token of the following row.

The segment-start bit is authoritative. EOS is not used to infer boundaries,
because a literal special-token string can occur in source code or prose.

At collation time, the start bits produce:

```text
input_ids       [B, T] int64
labels          [B, T] int64, with cross-segment targets set to -100
position_ids    [B, T] int64, reset to zero at each segment
document_ids    [B, T] int64, incremented at each segment
domain_ids       [B]
row_ids          [B]
sample_references [B]
```

Suppose two documents share one physical row:

```text
tokens:       A0 A1 EOS B0 B1
starts:        1  0   0  1  0
positions:     0  1   2  0  1
document IDs:  0  0   0  1  1
labels:       A1 EOS IGN B1 ...
```

The final content token of A predicts EOS. EOS does not predict the first token
of B. The model combines causal and same-document masking, so B cannot attend
to A even though both occupy one fixed row.

Each shard has paired files:

```text
shard-000000.tokens.bin
shard-000000.starts.bin
```

`manifest.json` records format version, dimensions, vocabulary and EOS IDs,
token counts, valid loss-token counts, masked boundary labels, tail accounting,
and SHA-256 checksums. `unused_as_input_tail_tokens` counts source-stream tokens
after the final complete input row; the first of these may already be stored as
the final row's lookahead. `unstored_tail_tokens` excludes that stored
lookahead, so it reports only tokens absent from all shard payloads.
`tokenizer_manifest_sha256` is mandatory, must be a
lowercase 64-character SHA-256, and pins every domain to the exact same
tokenizer artifact. The global order repeats that identity and refuses mixed
tokenizer manifests. Writers refuse non-empty destinations and publish shard
files and manifests atomically.

The full validator checks more than file sizes and hashes: it scans every
payload token for vocabulary range, requires the first segment-start bit in
every row, requires every in-row document start to be preceded by EOS, rejects
nonzero padding bits in the final start byte, and recomputes both
`valid_loss_tokens` and `masked_boundary_labels` from the encoded starts.
Shard payload names are canonical (`shard-NNNNNN.tokens.bin` and
`shard-NNNNNN.starts.bin`); traversal paths, symlinked artifacts, duplicate or
non-finite JSON values, incorrect recorded byte lengths, and unexpected files
in a fully published shard directory fail closed.

## Crash-resumable packing construction

Packing uses an atomic `.packing-journal.json` until the final manifest is
published. The journal commits all completed shard manifests and checksums,
the checksum and exact byte length of a partially open shard, carry tokens and
segment-start bits, all stream/loss counters, and the caller's opaque source
cursor. Its construction identity binds the domain, split, dimensions, EOS,
shard size, seed, tokenizer manifest, curation policy, and selected-document
manifest. Production builders must supply all three artifact hashes; the two
policy/selection hashes remain optional only so synthetic and legacy callers
can use the same writer API.

The cursor must identify the **next** document to tokenize. A builder can
checkpoint on every document by passing `source_cursor=` to `add_document`, or
checkpoint a group of documents explicitly to amortize metadata fsyncs:

```python
writer = PackedShardWriter(
    output,
    domain="python",
    split="train",
    sequence_length=4096,
    vocab_size=49152,
    eos_token_id=eos_id,
    tokenizer_manifest_sha256=tokenizer_hash,
    curation_policy_sha256=curation_hash,
    selection_manifest_sha256=selection_hash,
    construction_seed=1234,
    resume=True,
)

cursor = writer.source_cursor  # None for a new build; otherwise resume here.
for document, next_cursor in source.iter_from(cursor):
    writer.add_document(tokenizer.encode(document))
    if next_cursor["document_index"] % 10_000 == 0:
        writer.checkpoint(next_cursor)
writer.finish(source_cursor=source.end_cursor)
```

At a checkpoint, open shard files are flushed and fsynced before the journal is
atomically replaced. Rolling hashes make routine checkpoints constant-time in
the shard size. On resume, completed shards and the committed prefix of an
open shard are checksum-verified. Bytes and whole successor shards written
after the last durable cursor are rolled back, then the open hash state is
reconstructed with one bounded sequential read. Any tokenizer, policy,
selection, seed, shape, or source-format identity mismatch fails closed.

Publishing `manifest.json` is the commit point for a finished build; only then
is the journal removed. Forced-interruption tests cross several shard
boundaries and assert that resumed shard payloads and final manifest are
byte-for-byte identical to uninterrupted construction. A new writer still
refuses a non-empty destination unless `resume=True` and a valid journal or
completed manifest is present.

## Shuffle and mixture

The final train corpora are packed separately as Python, other code, and
English. Curation's exact prefixes are all materialized; they are never cut
again merely because EOS boundaries create extra packed rows. For a token cap,
order v4 chooses the largest row count at or below the cap, additionally aligned
to complete optimizer updates for train, and allocates those rows 40/40/20 by
stable largest-remainder rounding. It independently permutes every available
row ID in each domain, takes that domain's allocated prefix, then globally
shuffles the selected references without replacement. This avoids a
first-shard bias while keeping the subset deterministic.

Those weights apply to **selected model input tokens after packing**. The
52.58B train budget includes inserted EOS tokens; it does not count the
duplicated `T + 1` lookahead storage token again. The default target shortfall
is less than one row for an unfrozen capped order, or less than one effective
optimizer update for frozen train geometry. Overshooting is always rejected,
and an insufficient domain fails rather than weakening the mixture.

Boundary labels are masked, so the realized valid supervised-target mixture
can differ slightly from 40/40/20. The order manifest records expected input
weights, realized input weights, per-domain valid-target counts, their realized
weights, and the absolute input-token target/actual/delta. Validation enforces
the input mixture and absolute budget but deliberately does not force the
valid-target mixture to 40/40/20.

`rows`, `rows_per_domain`, and valid-loss counters describe the selected order.
`packed_available_*` describes all immutable packed source rows, while
`packed_surplus_*` exactly accounts the available rows that this capped run did
not select, globally and per domain. Packed surplus is not a training tail. A
capped frozen train order is selected at a full-update boundary, so its
`training_consumption.dropped_tail_rows` is zero. Choose the actual global
microbatch and accumulation only after the real GPU memory/throughput smoke;
do not infer them from an illustrative row count.

A uint64 order is only about 103 MB, so an exact offline permutation is
preferable to an approximate streaming shuffle. Order format v4 freezes the
RNG, NumPy version, seed, bit allocation, domain mapping, source-manifest
hashes, tokenizer identity, order checksum, global microbatch rows, gradient
accumulation, effective update rows, and complete-update consumption counters.
Runtime never regenerates or silently trims the order. Input and supervised
tokens are recorded exactly for the selected order, packed surplus, and any
legacy/diagnostic runtime suffix, globally and per domain. Validation proves
the 40/40/20 allocation from the immutable order.

Validation proves every selected reference is unique and inside the actual
packed source-row bounds; unselected source rows are permitted only when they
match the declared surplus. It uses a boolean seen array per domain rather than
sorting the full order and recomputes exact supervised counts from the selected
rows or their smaller complement.

Build the real order after final packed manifests exist:

```bash
python scripts/build_training_order.py \
  --python-manifest /data/final/train/python/manifest.json \
  --other-code-manifest /data/final/train/other_code/manifest.json \
  --english-manifest /data/final/train/english/manifest.json \
  --output /data/final/train/order-seed-1234 \
  --seed 1234 \
  --global-microbatch-rows ROWS_FROM_GPU_SMOKE \
  --gradient-accumulation-steps ACCUMULATION_FROM_GPU_SMOKE
```

The two uppercase values are placeholders, not recommendations. The production
materialization bridge normally creates train/validation/test orders together
when its packed stage is resumed with the GPU-smoke-derived geometry.

Run the full integrity pass before training:

```bash
python scripts/validate_training_data.py \
  /data/final/train/order-seed-1234/manifest.json
```

## Distributed ownership and resume

`DistributedBatchSampler` slices a single global microbatch into disjoint contiguous
rank portions. Every rank opens the same immutable order, and no row is repeated
between ranks. `global_microbatch_rows` must be divisible by data-parallel world
size. A production format-v4 order rejects either runtime microbatch rows or
gradient accumulation that differs from its frozen values.

The sampler is intentionally stateless with respect to prefetch. A checkpoint
stores the number of **completed global microbatches**. Its preferred
`resume_state` also binds that cursor to the order-manifest hash, `order.bin`
hash, all three packed-manifest hashes, tokenizer-manifest hash, sequence
length, global microbatch rows, accumulation, effective update rows, world size,
completed optimizer updates, exact next-order-row offset, and consumed
input-token count. Any mismatch fails closed. The native training harness
provides equivalent identity and trajectory checks around its
completed-microbatch cursor. Any batches prepared but not consumed before
interruption are discarded, preventing prefetch from silently skipping data.

For the planned six-rank run, each global microbatch is sliced into six
disjoint contiguous rank-local portions. A fresh loader process reconstructs
the same slices from the identity-bound completed-microbatch cursor; tests
prove that all six rank slices are pairwise disjoint, their union is the exact
global-order prefix, and restart emits precisely the unconsumed suffix.

The sampler exposes only the complete-update prefix recorded by the production
manifest. `training_consumption` records available and consumed microbatches,
optimizer updates, partial-tail rows, and exact consumed/dropped input and
supervised tokens, including per-domain prefix counts.

Example:

```python
from pretrain.data import create_training_dataloader, frozen_training_geometry

order_manifest = "/local-nvme/train/order-seed-1234/manifest.json"
geometry = frozen_training_geometry(order_manifest)

loader, sampler = create_training_dataloader(
    order_manifest,
    global_microbatch_rows=geometry["global_microbatch_rows"],
    gradient_accumulation_steps=geometry["gradient_accumulation_steps"],
    rank=distributed_rank,
    world_size=distributed_world_size,
    resume_state=saved_sampler_state,
    num_workers=8,
    pin_memory=True,
)

for batch in loader:
    output = model(
        batch["input_ids"].cuda(non_blocking=True),
        batch["position_ids"].cuda(non_blocking=True),
        batch["document_ids"].cuda(non_blocking=True),
        batch["labels"].cuda(non_blocking=True),
    )
```

After an optimizer step, persist a cursor with
`sampler.state_dict(completed_global_microbatches=completed_global_microbatches)`.
The native trainer resumes with `start_global_microbatch` derived from its
checkpoint; standalone loader integrations should prefer the identity-bound
sampler state.

## Serving from the RunPod volume

The network volume is the durable source. Training should asynchronously copy
the immutable packed shards and order to GPU-pod local NVMe, validate packed
payload checksums and semantics once, then memory-map the local copies. The
small `order.bin` checksum is independent and enabled by default on every
loader open; `verify_payload_checksums=False` skips only the expensive shard
scan. Even in that fast mode, dataset construction and every worker's first
mmap require exact regular-file byte lengths, so truncation, appended garbage,
and direct symlink substitution fail before a row is read. Production
collation also rechecks token range, boundary-bit padding, and the EOS/start
relationship for every loaded row. These inexpensive online checks complement,
but do not replace, the launch certificate's complete SHA-256 and semantic
scan. Do not make each worker issue random reads against network storage during
the full run.

Use large shards, persistent DataLoader workers, pinned memory, prefetching, and
non-blocking host-to-device copies. Benchmark worker counts on the real pod;
more workers are not automatically faster. The multiprocessing smoke script is:

```bash
python scripts/smoke_training_data.py --workers 8
```

The local macOS sandbox blocks PyTorch's shared-memory manager, so the full
multiworker smoke test must be run on the Linux GPU pod. Single-process loader
correctness is tested locally.

A realistic CPU/network-filesystem exercise uses actual 4,096-token rows,
validates all checksums and the exact permutation, then reports mmap loading
throughput:

```bash
python scripts/benchmark_training_loader.py \
  --temporary-root /workspace/dataset/staging \
  --sequence-length 4096 \
  --total-rows 10000 \
  --global-microbatch-rows 20 \
  --workers 8
```

The generated benchmark directory is removed after a successful or failed run
unless `--keep` is supplied.

To test the boundary with the corpus that is actually arriving on the network
volume, use:

```bash
python scripts/smoke_raw_to_training_data.py \
  --root /workspace/dataset \
  --temporary-root /workspace/dataset/staging \
  --sequence-length 4096 \
  --workers 2
```

That read-only integration smoke selects one archive with a completed
preprocessing report from each model domain, streams bounded UTF-8 members,
tokenizes them with the pinned local StarCoder2 tokenizer, pins the real
`TOKENIZER_MANIFEST.json` hash into format-v2 shards, performs full semantic and
checksum validation, constructs an order, and consumes it through spawned CPU
workers. Its outputs are disposable and removed by default. It proves format
compatibility only; it does not substitute for final filtering, global
residual code duplicate handling, English near-deduplication, decontamination,
split selection, resumable production packing,
or the provenance side index.

## SFT and RL compatibility

The model itself only consumes generic labels and segment metadata. SFT will
reuse the tokenizer and model contract but use separate immutable shards with
an additional supervised loss mask so prompt/user tokens have label `-100` and
assistant tokens contribute loss. Conversation and example boundaries remain
attention-isolated unless an explicit chat template defines them as one
sequence.

RL rollouts will not use pretraining packed rows. They will use separate prompt
datasets, per-request KV caches, generated response tokens, rewards, and rollout
metadata. Checkpoint and tokenizer identities remain shared across pretraining,
SFT, and RL, while their datasets and samplers remain independently versioned.
