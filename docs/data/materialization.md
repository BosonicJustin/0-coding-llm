# Production corpus materialization

> **Current generation-v2 boundary:** token-ID materialization has not started.
> The intended input is an all-eligible identity-v7 keep-bitmap publication,
> not the stopped v1 exact-quota selection. Packed order v4 will be the only
> authority for 40/40/20 and final model-input caps. The v5/v6 commands later in
> this document are retained as tested historical contracts and must not be run
> for v2. See [fast-generation-v2.md](fast-generation-v2.md).

`pretrain/materialize.py` and `scripts/materialize_training_corpus.py` are the
cold-path bridge from immutable raw archives plus completed curation decisions
to the binary shards consumed by pre-training. This stage does no filtering,
deduplication, benchmark detection, or split assignment. Strict curation-v5
full-near and curation-v6 fast-profile decision shards are authoritative for
those choices; the bridge validates them,
re-reads the selected raw text, tokenizes it with the pinned tokenizer, packs
it, and publishes deterministic order-v4 artifacts.

The v7 branch has the same raw-source verification, pinned tokenizer, packing,
boundary-isolation, provenance, restart, and checksum obligations. Its
selection input differs: an archive bitmap says keep/reject for each complete
source document, `selected_tokens` must equal `source_tokens`, and observed
nine-cell totals are supply rather than exact quotas. The optional authenticated
raw-token-cache adapter replaces UTF-8 decode/tokenizer replay, but no selection,
split, EOS, packing, boundary, order, or provenance authority. Its local
multi-archive byte-identity and crash/corruption contracts are green; a frozen
durable v2 selection, cache inventory, and real-archive benchmark are still
required before production launch.

## Required completed inputs

The bridge fails closed unless the curation manifest and checksum are complete
and internally consistent. Both historical v5/v6 branches require
`production_ready: true`, a complete decision-shard inventory, the zero-leakage
audit, and the pre-packing StarCoder2 token unit. The full-near v5 branch
requires `english_near_dedup_complete: true` and the complete five-file English
artifact. The fast v6 branch instead requires the exact
`fast-exact-normalized-canonical-v1` profile,
`english_near_dedup_complete: false`, status `disabled_by_fast_profile`, null
near-artifact fields, zero exact/normalized collision audits, and the pinned
semantic-near limitation. The top-level
contract must also contain `raw_archives_hashed_for_integrity: true` and
`raw_archive_payloads_parsed_by_curation: false` exactly as published by
curation. Both identity formats must embed the same
collection-completeness authority as the top-level
manifest: authority v1, `complete: true`, no pending inputs or preprocessing
errors, `legacy_dedup_index_required: false`, every source bucket present, and
matching quota/raw/report/fingerprint/completion-marker inventories. The bridge
re-hashes the marker files and pins the entire authority. Diagnostic or legacy
curation output is not accepted.

It also pins and verifies all upstream identities: raw/source manifests,
preprocessing manifest, per-archive reports and fingerprints, curation policy,
quota configuration, MBPP denylist, curation's SQLite mount/journal/storage
evidence,
decision shards, and every reused-tokenizer artifact. The near-mapping SHA must
equal `english_near_clusters_sha256`; its completeness/leakage and database
integrity audits must be clean. Its embedded calibration evidence must name the
passed, production-eligible calibration JSON and checksum sidecar under the
dataset root. The bridge reopens both regular non-symlink files, verifies exact
bytes and hashes, requires the pinned production profiles, checks the nested
identity hash and production-builder binding, and reconciles its source,
report, policy, preprocessing, benchmark, completeness, and selected-report
authorities with curation. For v5, it additionally validates the complete
embedded English near-dedup five-file artifact and mapping identity as described
below. For v6, it rejects any near artifact and preserves the exact fast profile,
audit, status, and limitation in output provenance. The loaded tokenizer must have the configured
vocabulary size and EOS ID. A changed, added, missing, malformed, internally
inconsistent, or non-regular input causes an error rather than a best-effort
build.

For full-near v5, the English wrapper is contract v2. It pins a dataset-root-relative publication
root plus exact path/byte/hash identities for `manifest.json`,
`manifest.sha256`, `clusters.jsonl.zst`, and
`operational-preflight-v1/{result.json,result.json.sha256}`. Before tokenization,
the bridge reopens all five files, rejects duplicate JSON keys, rechecks both
sidecars byte-for-byte, and requires the original manifest's identity, mapping,
audit, inputs, candidate statistics, and preflight evidence to equal the
curation copy. It recomputes the preflight identity hash and all sample-size,
rate, ETA, SQLite-growth, disk-safety, and dense-union memory formulas against
the current builder/config/calibration/policy/preprocess/benchmark/source
authorities. Missing, merely re-signed, or internally stale operational
evidence therefore fails before a raw archive is opened.

## Deterministic algorithm

For every raw archive, in canonical inventory order, the materializer:

1. Verifies the pinned report, fingerprint, decision-shard, and raw-archive
   identities. The default route streams the `.tar.zst` archive once. The
   optional v7 route instead opens one externally authorized per-archive token
   cache and validates its source/tokenizer bindings, payload hashes/dtypes,
   offsets, token range, and fingerprint alignment before writing anything.
   Both routes preserve exact manifest-index order and source totals.
2. Ignores rejected records. For every `keep` record, it decodes strict UTF-8,
   tokenizes using the pinned tokenizer, requires the recomputed full-document
   token count to equal `source_tokens`, and keeps exactly
   `token_ids[:selected_tokens]`. Thus a quota-ending partial document is
   preserved exactly; it is never silently rounded to a whole document.

   In the generation-v2 v7 branch, every keep is a complete document and
   `selected_tokens == source_tokens`; quota-ending prefixes are forbidden.
   The cached route obtains that complete content span by `manifest_index` and
   feeds it to the same writer used by the raw-tokenizer route.
3. Sends the selected prefix to one of nine independent writers:
   `{train,validation,test} x {python,other_code,english}`. Each document gets
   one EOS token. A document-start bit marks its first content token, and every
   physical row is also forced to start a new attention segment. The packed
   token row has `sequence_length + 1` values so a row supplies both inputs and
   next-token labels; the bitset supports block-diagonal causal attention and
   masks the label that would cross a document boundary.
4. Emits a compact compressed document-position record for every kept document.
   It pins the document/canonical/group IDs, language, bucket, source archive,
   source member and decision row, selected length, logical stream start/end,
   and EOS position. A row and offset are `logical_position // sequence_length`
   and `logical_position % sequence_length`; no raw text is duplicated.
5. Emits immutable `uint16` token shards and packed little-endian start-bit
   shards, with row counts, token accounting, SHA-256 checksums, and the last
   committed source cursor in each packed manifest.
6. Builds a separate deterministic global order for each split, validates all
   packed and order manifests, writes source/policy/tokenizer/fingerprint
   provenance sidecars, and finally publishes the top-level `manifest.json` and
   `manifest.sha256`.

The output layout is:

```text
packed-v1/
  packed/{train,validation,test}/{python,other_code,english}/
    shard-NNNNNN.tokens.bin
    shard-NNNNNN.starts.bin
    manifest.json
  provenance/documents/{train,validation,test}/{python,other_code,english}/
    archive-NNNNNN.jsonl.zst
    manifest.json
  orders/{train,validation,test}/
    order.bin
    manifest.json
  provenance/{source,policy,tokenizer,fingerprints}.json
  provenance/raw_token_cache.json  # present only for the cache adapter
  manifest.json
  manifest.sha256
```

## Token caps and packed surplus

Curation quotas count selected **content tokens before EOS**. Packing adds one
EOS per selected document, so the packed corpus intentionally contains more
model-input tokens than the curation quota. Order v4 is the authority for what
training or evaluation may consume:

- train selects the largest complete-optimizer-update row count at or below
  52.58B input tokens;
- validation selects whole rows at or below 0.5B input tokens;
- test selects whole rows at or below 0.5B input tokens.

Each capped order contains an exact 40% Python / 40% other-code / 20% English
row allocation (up to indivisible-row largest-remainder rounding). For each
domain, order construction independently permutes every available row and takes
the required prefix, then globally shuffles the selected references. Selection
is deterministic, without replacement, and is not biased toward the earliest
packed shards.

For generation v2, train still targets the largest optimizer-update-aligned
balanced prefix at or below 52.58B input tokens. Validation and test each use
the largest feasible balanced 40/40/20 whole-row cap at or below 0.5B. A
held-out split may therefore be smaller than 0.5B when one domain is limiting;
do not acquire extra raw data solely to fill a language-model held-out cap.

`rows`, `rows_per_domain`, `input_tokens_per_domain`, and
`valid_loss_tokens_per_domain` describe only the selected order. The manifest's
`packed_available_*` fields describe every immutable packed row, while
`packed_surplus_*` describes packed rows not selected by this run. Surplus is
auditable reusable source data, not a training tail and not permission to exceed
the target. For the frozen train geometry, the selected row count is already
optimizer-update aligned, so `training_consumption.dropped_tail_rows` should be
zero. Always gate launch on the validated order manifest's
`input_token_budget.actual_total` and `training_consumption` contract, not on
the pre-EOS curation total or directory size.

## Restart, atomicity, and byte identity

Checkpoints are committed only at raw-archive boundaries. Every split/domain
writer stores a next-archive cursor plus the checksummed document-index shard
inventory committed with that cursor. If a process dies between writer
checkpoints, some writers may be one archive ahead; resume leaves those writers
untouched and replays that archive only into lagging writers. An index shard
published just before a crash is deterministically regenerated and must compare
byte-for-byte before it is accepted. A cursor vector that differs by more than
one archive is rejected.

Open shard bytes are fsynced before an atomic writer-journal update. Completed
shards, the bridge journal, order staging directories, order files, provenance,
and final manifests use temporary paths plus rename and directory fsync. An
exclusive output lock prevents two materializers from writing concurrently.
Given identical immutable inputs, output identity, seeds, and semantic
configuration, an interrupted-and-resumed build is byte-identical to an
uninterrupted build. Tokenizer batch sizes are runtime throughput knobs and are
deliberately excluded from corpus identity; changing them must not change
bytes. Never modify or manually merge a live output directory.

## Commands

Stage 1 is safe to run before choosing GPU batch geometry. It finishes and
fully validates all nine packed outputs and document-position indexes, leaves a
durable `phase: packed` journal, and intentionally publishes neither orders nor
the top-level production manifest:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/materialize_training_corpus.py \
  --root /workspace/dataset \
  --preprocess-root /workspace/dataset/staging/preprocess \
  --selection-root /workspace/dataset/curated/selection-fast-v1 \
  --curation-policy /workspace/coding_model_from_scratch/configs/curation_policy_fast_exact_normalized.json \
  --tokenizer-root /workspace/dataset/tokenizer/starcoder2 \
  --output /workspace/dataset/final/packed-v1 \
  --stop-after-packing \
  --tokenizer-batch-documents 256 \
  --tokenizer-batch-bytes 67108864 \
  --tokenizer-max-document-bytes 67108864
```

Re-run that command to resume. The batch controls may be tuned without changing
output identity. Do not guess global microbatch rows or accumulation at this
stage.

Next, mount the packed volume on the intended GPU configuration and run the
memory/throughput smoke. Substitute its measured integer values in this stage-2
template and rerun the **same output path** without `--stop-after-packing`:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/materialize_training_corpus.py \
  --root /workspace/dataset \
  --preprocess-root /workspace/dataset/staging/preprocess \
  --selection-root /workspace/dataset/curated/selection-fast-v1 \
  --curation-policy /workspace/coding_model_from_scratch/configs/curation_policy_fast_exact_normalized.json \
  --tokenizer-root /workspace/dataset/tokenizer/starcoder2 \
  --output /workspace/dataset/final/packed-v1 \
  --global-microbatch-rows ROWS_FROM_GPU_SMOKE \
  --gradient-accumulation-steps ACCUMULATION_FROM_GPU_SMOKE
```

The placeholder values are deliberately non-authoritative. Stage 2 creates and
validates the capped orders, pins the chosen geometry, then atomically publishes
the final manifest. Once an order or final manifest is published, a mismatched
geometry/configuration is rejected. Do not use
`--diagnostic-allow-observed-mixture` or any zero token cap for the final corpus.

For a bounded end-to-end smoke test, use a distinct disposable output path so
it cannot be mistaken for the final corpus:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/materialize_training_corpus.py \
  --root /workspace/dataset \
  --selection-root /workspace/dataset/curated/selection-fast-v1 \
  --curation-policy /workspace/coding_model_from_scratch/configs/curation_policy_fast_exact_normalized.json \
  --output /workspace/dataset/smoke/materialize-one-archive \
  --stop-after-packing \
  --max-archives 1
```

This validates and commits one archive reconciliation, then reports
`complete: false` plus measured source-token, selected-content-token,
selected-stream-token-including-EOS, and raw-MiB rates.
Re-running it advances from the checkpoint; removing `--max-archives` completes
packing on that same disposable output. A smoke result is not publishable.

## Performance and capacity planning

Raw archive reading remains one sequential pass per archive. Selected documents
are accumulated into bounded batches and passed to tokenizer `encode_batch`;
output order is unchanged. Before reading a selected member, the bridge flushes
the current batch if adding that member would exceed either cap (default: 256
documents or 64 MiB of raw UTF-8 bytes). A selected member larger than
`--tokenizer-max-document-bytes` fails closed before its payload is read; the
default single-document limit is the batch-byte limit, and it may never exceed
that limit. Raise both only after a bounded smoke demonstrates sufficient RAM.
Rejected documents and selected documents already durable for every writer are
hashed in 8 MiB streaming chunks rather than materialized in memory.

The raw-byte caps are hard input bounds, not an RSS guarantee: decoded Python
strings, tokenizer-owned buffers, and token ID lists can consume several times
the raw UTF-8 size. The Rust tokenizer can use its internal worker pool. The
Python side intentionally remains one ordered archive producer; parallelism is
inside the Rust `encode_batch` call, so there is no out-of-order merge or second
raw pass. If tuning `RAYON_NUM_THREADS`, set it before launch, begin near the
pod's physical-core count, and compare measured selected-stream tokens/s;
excessive threads can compete with zstd and network I/O. Benchmark on the actual
CPU pod: zstd decode, UTF-8/tokenization, raw-volume reads, and destination-
volume writes can each become the bottleneck.

Shard output is append-heavy and sequential. On a network volume, the bridge
writes large `.part` token/start files and pays durability synchronization at
shard closes and archive checkpoint boundaries; it does not perform a random
write workload. Larger archives amortize fsync latency. Pod-local staging can
be faster, but then the complete closed corpus must be copied and revalidated,
and pod loss loses the local checkpoint. Direct network-volume output is the
safer overnight choice.

At sequence length 4096, each row stores 4097 `uint16` token IDs plus a 513-byte
start bitset: 8,707 bytes per 4,096 model-input tokens, or about 2.126 bytes per
input token. The 53.58B selected train+validation+test budget therefore needs
about 114 GB (106 GiB) for packed rows, plus about 0.105 GB for `uint64` order
references. The compressed document-position index is additional and scales
with selected document count, not tokens; its exact cost depends heavily on
source-path entropy and is typically on the order of hundreds of compressed
bytes per document, so measure the one-archive smoke and reserve several GB for
large document counts. Actual packed storage is also higher because every
curation prefix and its EOS are retained, including order-v4 surplus. Raw
archives, preprocessing, curation artifacts, and free-space headroom are
separate. Use the manifests to calculate the exact value and reserve at least
20-30% headroom.

The default 131,072-row shard is about 1.14 GB; up to nine open `.part` shards
may exist, but they are eventual output rather than a second copy. A direct
network-volume build needs no second full-corpus scratch area. A local-staging
workflow needs enough local space for the entire packed output plus headroom.
Tokenizer batches can require several times their raw UTF-8 size as Python ID
lists, and order construction holds several `uint64` arrays (roughly hundreds
of MB for about 13.1M selected rows). Budget at least 8 GiB RAM and measure peak
RSS; 16 GiB or more gives safer headroom for unusually token-dense documents or
substantial packed surplus.

For a measured sustained rate `R` model-input tokens/second, the tokenization
lower-bound ETA is `53.58e9 / R`. This excludes startup validation, filesystem
stalls, final order construction, full checksum validation, and retries:

| Sustained rate | Token-pass ETA |
| ---: | ---: |
| 0.25M tok/s | 59.5 h (2.48 d) |
| 0.5M tok/s | 29.8 h (1.24 d) |
| 1.0M tok/s | 14.9 h |
| 2.0M tok/s | 7.44 h |

Run the one-archive smoke on the intended pod and volumes, record raw bytes/s,
selected tokens/s, output bytes/s, wall time, and peak RSS, then extrapolate
from both token volume and raw compressed bytes. Use the slower estimate and
add margin for final validation; the table is conditional capacity planning,
not a promised runtime.
