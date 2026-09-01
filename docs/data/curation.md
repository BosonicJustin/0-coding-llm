# Final corpus curation and selection

> **Generation-v2 route:** exact quota selection in `curate_corpus.py` is no
> longer the intended final publication path. After the other-code top-up, v2
> must rebuild inventory, reasons, global exact/normalized canonicalization,
> and leakage-safe groups from all old plus new inputs. A separate v7 publisher
> then keeps every eligible canonical full document with per-archive bitmaps;
> packed order v4 becomes the only 40/40/20 and token-budget authority. See
> [fast-generation-v2.md](fast-generation-v2.md). The v5/v6 quota-selection
> algorithm below remains the historical implementation contract and must not
> be mistaken for the v2 launch procedure.

`scripts/curate_corpus.py` is the immutable boundary between the streaming raw
audit and the later raw-text reader/tokenizer/packer. It consumes the finalized
collection quota records and completion markers plus
`staging/preprocess/reports` and `staging/preprocess/fingerprints`. It lists and
stats raw archive paths to prove that the ledger and finalized file inventory
are identical. It never changes a raw `.tar.zst` archive; every archive across
all four buckets is read as an opaque byte stream for SHA-256 verification,
but its tar members are not parsed by curation. The
manifest records both facts explicitly; legacy `raw_archives_opened: false`
continues to mean "not opened as selection/content input."

Production selection now fails closed while collection or preprocessing is
incomplete. Before freezing the report inventory, it requires all four
collection targets and completion markers, exactly one canonical quota record,
raw archive, report, and fingerprint shard per archive, no extra raw/report/
fingerprint inputs, no hidden `.part-*` archive, and no preprocessing error
record. Per-bucket archive, document, byte, and exact-token totals must agree.
The first invocation freezes hashes of the quota-record, completion-marker,
raw-archive, report, and fingerprint inventories. A changed, added, or removed
authority is an identity mismatch on resume. Use a new output directory for a
genuinely new corpus generation.

## Algorithm

The stage is deterministic and ordered as follows. Steps 5–6 describe the full
near-deduplicated v5 profile; the fast v6 production branch is specified under
"Fast-baseline production command" below and replaces the fuzzy English
artifact with one global exact-normalized namespace.

1. Validate the pinned Stack v3.1 repository/revision/shard digest, both
   English source manifests, all tokenizer file sizes and SHA-256 values, the
   MBPP denylist identity, the preprocessing policy, the curation policy, and
   the exact-token quota configuration.
2. Reconcile all finalized collection quota records against the four collection
   targets and three completion markers. Require canonical six-digit archive
   identities, matching sources, no duplicate shard/archive IDs, exact raw-file
   coverage, no pending `.part-*` files, no preprocessing errors, and one
   canonical report and fingerprint per record. Recompute every raw archive's
   SHA-256 across all four buckets and require it to match the report before
   freezing the snapshot. Pin the per-bucket totals and every authority
   inventory SHA-256. This gate does not read or require the rebuildable legacy
   `dedup/dedup.sqlite3` audit index.
3. Hash every report and its referenced fingerprint shard. Stream every
   fingerprint row into a restart database, checking record versions, stable
   document IDs, archive/bucket/source identity, hashes, provenance, row order,
   and document/byte/token totals. Each archive is one durable transaction.
4. Apply the policy's explicit hard-reject quality hooks. MBPP hits are always
   rejected. Code without `repo_id` and English without a stable article/URL
   identity are rejected instead of being assigned unsafe singleton groups.
5. Validate the complete sibling artifact produced by
   `scripts/build_english_near_clusters.py`: `clusters.jsonl.zst`,
   `manifest.json`, and `manifest.sha256`. Curation verifies the manifest
   checksum, production-ready v1 formats, mapping hash/size/row count, exact
   builder/config/policy/preprocess/benchmark identities, source revisions,
   finalized English quota and completion evidence, report inventory, every
   referenced report/archive/fingerprint checksum, runtime and local/network
   SQLite journal evidence, database integrity, and every zero-defect
   completeness/leakage count before ingesting a mapping row. It also reopens
   the required passed calibration JSON and exact checksum sidecar under the
   dataset root, validates the pinned production gates, hashes, nested identity,
   production-builder SHA, sampling profile and sample manifest, and reconciles
   its reports/sources/policy/preprocess/benchmark/completeness authorities.
   The mapping's
   JSONL.zst format is one
   `{"doc_id": "<sha256>", "cluster_id": "..."}` row per English document.
   Duplicate, unknown, code, or missing document IDs fail closed. The builder
   validates but does not trust the compact 16-value audit sketches: it rereads
   immutable raw text for a larger LSH candidate pass and refines candidates
   with complete hashed five-word-shingle Jaccard. See
   [english-near-dedup.md](english-near-dedup.md) for the pinned algorithm and
   its statistical limits.
6. Propagate every benchmark hit through its exact and final duplicate
   cluster. Select one deterministic residual exact canonical globally. Then
   select one code canonical per normalized hash across Python and other code,
   or one English canonical per supplied near-duplicate cluster. Code does not
   enter a second MinHash/LSH pass because the pinned Stack train release is
   already cross-repository near-deduplicated.
7. Form split groups. The same code `repo_id` produces the same group ID across
   both code buckets. Surviving English documents use stable source identity;
   cross-source near duplicates have already been reduced to one canonical.
   Hash each group with the frozen policy seed into train, validation, or test.
   A group is never divided between splits.
8. Within each category/split, order canonical documents by a second seeded
   hash and select to the configured content-token quota. Each source bucket is
   scanned through `(bucket, selection_rank, doc_id)` index order. The two
   English cursors are merged with a one-row-per-bucket heap, avoiding the
   unbounded temporary B-tree created by `IN (...) ORDER BY`; `doc_id` is the
   deterministic rank-collision tiebreaker. The last selected document may
   contribute only token prefix `[0, remaining)`, which makes the content-token
   total exact without moving its repository to another split.
9. Re-read and re-hash every fingerprint, then atomically write one compressed
   immutable decision/provenance shard per input archive. A final audit requires
   zero selected content hashes, canonical clusters, source groups, or global
   code-repository groups shared by multiple splits.
10. Re-snapshot the full collection-completeness authority, re-hash every raw
    archive across all four buckets, revalidate the
    English near-dedup sibling contract, re-hash every input and output, and
    atomically publish `manifest.json` and `manifest.sha256`.
    `collection_completeness` is embedded both in the run identity and at the
    manifest top level so later packing can retain the exact corpus authority.
    The complete validated near-manifest identity, mapping identity, audit, and
    operational-preflight evidence are frozen in the curation identity. The
    English artifact wrapper is contract v2 and names a dataset-root-relative
    publication containing exactly the five closed files: mapping, manifest,
    manifest sidecar, preflight result, and preflight sidecar. Curation reopens
    both sidecars and both JSON files, recomputes the preflight identity plus
    rate/ETA/disk/union-memory formulas, and re-snapshots all five files before
    mapping ingestion and final publication. Any drift rejects the run. This is
    curation identity format v5, SQLite schema v4, and storage-preflight schema
    v2. Schema-v1/v2/v3 work can migrate only when its exact historical
    storage schema and every v5 corpus/evidence identity match. V2/v3 storage
    passes are remeasured under the stronger current capacity contract and
    linked to the old evidence hash; they are never relabeled. Older identity
    formats still fail closed and require a new output directory.

Every document has one decision record. Kept records name the split, group,
canonical, source token count, selected prefix length, quality flags, and raw
provenance. Rejected records retain deterministic reasons such as
`benchmark_cluster_contamination`, `residual_exact_duplicate`,
`residual_normalized_duplicate`, `english_near_duplicate` (full v5),
`english_normalized_duplicate` (fast v6), a policy quality reason, or
`quota_overflow`.

## All-eligible v7 publication

Generation v2 stops treating curation as the mixture sampler. Once a complete,
immutable snapshot proves reasons, canonical winners, group coverage, split
assignments, source identities, and zero leakage, the read-only v7 publisher:

1. validates the durable snapshot manifest, database checksum, checkpoint,
   source/preprocess/policy/quota identities, and SQLite query plans;
2. recomputes exact eligible document/token totals for all nine split/domain
   cells;
3. writes one restartable keep bitmap per input archive, aligned to manifest
   row order, with a 1 only for an eligible canonical document;
4. records every kept document at full source length and records no terminal
   token prefix or curation mixture claim; and
5. atomically publishes an identity-v7 manifest only after every bitmap,
   checksum, accounting total, and source-stability check passes.

Partial v1 exact-selector rows and quota progress remain audit evidence only.
They cannot affect a bitmap or be accepted as selection authority. The v7
manifest reports `selected_totals` as observed supply and keeps reference
quotas informational. The materializer preserves separate split/domain packed
streams, and deterministic order v4 later selects the actual 40/40/20 rows.

Publisher/materializer v7 qualification is still in progress. No production
command should be frozen until the fresh v2 canonical/group snapshot exists
and the producer/consumer contract suite passes.

## Token accounting boundary

The curation quotas are exact **pre-packing StarCoder2 content tokens**. They
are not the Chinchilla training-input count. EOS document boundaries add tokens
and incomplete packing tails can drop tokens. The final packed order v4
manifest's full-optimizer-update-prefix `consumed_input_tokens` is the
authoritative train/validation/test model-input budget and must be checked
before launching training.

## Restart and integrity model

`OUTPUT/.work/curation.sqlite3` is authoritative. SQLite uses full synchronous
commits; archive ingestion, each later bulk write, and every phase transition
are transactional. Schema-v4 `phase_progress` gives every archive, English
mapping load/application, benchmark-propagation pass, exact/final canonical
choice, canonical-map construction, duplicate-reason pass, group assignment,
and split/category quota an independent status, exclusive cursor,
processed-row/token counters, and committed-batch counter. A batch's table
writes, O(1) durable table counts, and cursor/counters commit atomically. A
crash after that commit resumes strictly after its key; a crash before it
replays no committed data. Full table counts occur at phase/final audit
boundaries, not on every checkpoint. The public `phase` advances to
`inventory_complete`, `canonicalized`, or `selected` only after all relevant
subphases and exact coverage/accounting/leakage checks pass.

Near-map input uses a durable row ordinal because the single compressed frame
has no safe random-seek cursor; resume replays and validates the immutable
prefix while skipping already committed rows. Every scan that drives a bulk
write uses a bounded keyset; read-only phase audits intentionally perform full
coverage scans. Quota cursors persist the exact
`(selection_rank, doc_id)` tiebreak order, and final validation proves that the
selected rows are precisely the prefix ending at that cursor, including the
terminal partial-document token count.

Before the first inventory write, a durable storage preflight uses the measured
v1 production peak: 67,824,914,432 database bytes for 51,328,930 documents.
It rounds 1,321.378 bytes/document up to 1,322, applies a separate 2x safety
factor, then adds the transaction-sidecar reserve and a 2 GB remaining floor.
At the measured v1 document count and a 100,000-row batch, the executable gate
is 144,267,290,920 bytes (134.36 GiB), versus 323,918,545,920 bytes under the
obsolete synthetic projection. The complete observed provenance—including the
4,132,940,952-byte maximum WAL—is frozen in the storage contract. Treat the
run-specific `required_free_bytes_at_measurement` as authoritative because v2's
final document count is determined by its completed report inventory.
The sidecar reserve is the larger of 256 MiB and 64 KiB per configured batch
row. Every bounded transaction records database, rollback-journal, WAL, free
space, and row-count high/low-water marks in `storage_metrics`; a pre-commit
row/sidecar/free-space violation rolls the transaction back. If a committed
WAL later exceeds its frozen bound, the output generation is permanently
marked unsafe before any cleanup checkpoint is attempted. Every later reopen
and finalization rejects that generation even if the WAL has shrunk; restart
in a new output directory with a smaller batch. WAL uses a 1,024-page
autocheckpoint and is truncated when it crosses half the frozen sidecar bound.
`--batch-size` is part of the restart identity and cannot change on resume.

SQLite temporary storage is forced to the real
`OUTPUT/.work/sqlite-tmp` directory, verified on the same filesystem as the
database and included in the free-space evidence. Thus a global sorter cannot
silently fill `/tmp` or RAM. Quota selection itself is verified to use the
streaming index without a temporary sort. The 512 MiB page cache remains a
bounded performance cache rather than the authority for pending selection.
`CHECKPOINT.json` v2 exposes the exact subphase cursors/counters and storage
watermarks, while `journal.jsonl` remains the phase/event log; both are atomic
projections of committed database state. Decision shards are written to temporary files,
fsynced, renamed, directory-fsynced, and only then recorded in the database.
On resume, every previously recorded output checksum is verified.

SQLite journal selection is fail-safe and part of the frozen run identity.
The default `--sqlite-journal-mode auto` inspects the actual output mount: WAL
is used only for a positively identified local filesystem, while NFS and other
network-like or unknown filesystems use rollback `DELETE` journaling. An
explicit `--sqlite-journal-mode wal` is rejected unless the mount is known
local; `--sqlite-journal-mode delete` is the only cross-filesystem override.
The selected mode, filesystem type, mount point/source/device, and mount options
are recorded in every checkpoint and the final manifest. Resume fails before
any database write if the mount evidence, requested policy, selected mode, or
persisted SQLite journal mode changed.

The append-heavy inventory pass intentionally has no global dedup/split
indexes. Those indexes are built once after the inventory is complete, avoiding
several random index writes for every incoming document. Fingerprint rows and
SQL inserts are processed in configurable batches. SQLite cannot resume inside
`CREATE INDEX`, so each of the five global indexes is one explicitly recorded
atomic DDL restart unit; storage/WAL projections cover that exception and the
durable cursor advances only after the complete index commits.

For maximum speed, place the output/state on pod-local NVMe. After the command
has exited successfully, verify `manifest.sha256` and the decision checksums in
the manifest, then copy only the closed published `manifest.json`,
`manifest.sha256`, and `decisions/` artifacts to the network volume. Do not copy
the live `.work/` directory, `curation.sqlite3`, `-wal`, or `-shm` files. A WAL
database copied to NFS is deliberately rejected for resume rather than silently
converted. If network-volume execution is unavoidable, auto mode selects
rollback journaling; expect substantially lower SQLite throughput.

## Fast-baseline production command

After all fingerprint reports finish, the initial speed-focused baseline skips
the separate fuzzy English near-dedup job and runs the versioned fast policy:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/curate_corpus.py \
  --root /workspace/dataset \
  --staging-root /workspace/dataset/staging/preprocess \
  --policy /workspace/coding_model_from_scratch/configs/curation_policy_fast_exact_normalized.json \
  --output /workspace/dataset/curated/selection-fast-v1 \
  --batch-size 10000 \
  --sqlite-journal-mode delete
```

This profile keeps one deterministic global canonical per exact hash and then
per normalized hash, propagates benchmark contamination across both, groups
survivors by repository/stable English source, and does not consume an English
near-map artifact. Its manifest is format v6 and explicitly records that fuzzy
near-deduplication was not performed.

Because this command runs on the network volume, its SQLite checkpoint and
closed publication are already durable. Never copy or move the live directory
while curation is running.

The curator enforces a cross-client singleton with the atomic, NFS-visible
`selection-fast-v1/.curation.cross-client-lease.json` owner record in addition
to the NFS server-visible advisory `flock`. The fully fsynced owner is published with a
no-replace hard link, so acquisition cannot expose a partial lease; clean
release atomically moves the canonical lease away before best-effort cleanup.
A hard crash leaves the complete lease behind. Use
`--recover-stale-cross-client-lease OWNER_TOKEN` only after independently
proving that no pod/process still owns the run, and pass the exact token from
the stale JSON. Recovery is serialized by the NFS advisory lock, permanently
claims that token with a no-replace hard link, rechecks the canonical owner,
archives the old record, and refuses a live same-host PID. An existing recovery
claim is a manual-review stop; never delete one automatically.

Re-run the exact same command to resume. `--max-new-archives` and
`--stop-after-phase` are controlled interruption points for tests and
operations. `--allow-missing-english-near-dedup` is diagnostic only: it uses
the residual normalized English hash as a provisional cluster and publishes
`production_ready: false`. Such an output must never feed production packing.

The approved fast policy skips only fuzzy semantic near-deduplication; it does
not retain exact or normalized-hash collisions. Semantic near duplicates may
remain and may cross source groups or splits, and that limitation is pinned in
the curation and materialization provenance. The full five-file English near
route remains available as a later controlled ablation.

The legacy parallel Stack collector did not embed `tokenizer_revision` in
`STACK_V3_SOURCE.json`. The curation manifest pins and validates the actual
tokenizer artifact and records that historical provenance limitation instead
of concealing it. Both English source manifests do contain the tokenizer
revision and must match exactly.
