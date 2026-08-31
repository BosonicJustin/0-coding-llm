# Streaming audit and fingerprint preprocessing

`scripts/preprocess_raw_stream.py` performs safe work while raw acquisition is
still running. It never changes a raw archive and never opens a hidden
`.part-*` file. An archive becomes eligible only when it has its final
`part-NNNNNN.tar.zst` name and a matching committed quota record.

## Work performed now

For every finalized archive, the pipeline:

1. stream-decompresses the complete archive and verifies its zstd/tar framing;
2. validates member order, member sizes, internal manifest rows, document
   counts, clean bytes, and exact token totals against the quota ledger;
3. calculates compressed-archive SHA-256 and per-document content SHA-256;
4. calculates a source-appropriate normalized hash for inexpensive residual
   duplicate auditing after line-ending/whitespace normalization;
5. calculates a compact bottom-k near-duplicate sketch. Because this v1 format
   was pinned before the source-policy correction, every bucket has a sketch;
   all sketches are now integrity/audit metadata only. Production English
   candidate generation rereads immutable raw text with a larger pinned
   signature instead of trusting the compact sketch;
6. independently applies the MBPP content fingerprint guard;
7. records non-destructive quality metrics and advisory flags for short,
   repetitive, minified, generated, URL-heavy, corrupted, or possibly secret
   content;
8. emits a compressed metadata/fingerprint shard containing no source text;
9. leaves a checksummed, restart-safe fingerprint shard ready for final
   curation and, optionally, batched ingestion into the rebuildable SQLite
   telemetry index.

No quality flag currently deletes or rewrites a document. The final selection
policy remains explicit and can be evaluated from these metrics after source
composition is known. The final selector independently rebuilds exact and
normalized groups and removes residual collisions in every
bucket and perform local near-duplicate clustering only for FineWeb-Edu and
Wikipedia. It must never feed Stack v3 Train code sketches into a second LSH
pass because that pinned source is already near-deduplicated upstream.

The current watcher remains byte-for-byte compatible with already completed
fingerprint shards. Omitting code sketches mid-run would change the pinned
fingerprint policy for a small CPU saving and would invalidate resumability.
The routing decision therefore lives in `configs/curation_policy.json`, not in
this immutable v1 fingerprint format.

## Outputs

```text
/workspace/dataset/staging/preprocess/
  PREPROCESS_MANIFEST.json
  STATUS.json
  reports/<bucket>/part-NNNNNN.json
  fingerprints/<bucket>/part-NNNNNN.jsonl.zst
  dedup/dedup.sqlite3                     # optional aggregate telemetry index
  errors/<bucket>/part-NNNNNN.json       # only on a failed audit
```

Reports and fingerprint shards are immutable per source archive. The policy
hash and MBPP denylist hash are pinned in `PREPROCESS_MANIFEST.json`. A policy
change must use a new fingerprint version or a new staging root.

If interrupted during an archive, temporary output is discarded and that one
archive is reprocessed. A fingerprint that was atomically published just before
a report-publication interruption is safely replaced on retry. Completed reports
are skipped. Success publication removes and directory-fsyncs any prior error
before atomically publishing the report; therefore a crash can leave either a
rediscoverable archive or a completed report, never a new report plus a stale
error. Startup durably reconciles the legacy both-present state by treating the
atomic report as authoritative. The production launcher
uses `--index-mode deferred`: archive verification and fingerprint generation
continue without waiting for SQLite on the network volume. It exits after the
closed-collection raw audit because production curation independently revalidates
and canonicalizes every fingerprint in its own journal. Deferred audit never
opens the rebuildable telemetry database, so a corrupt or locked optional index
cannot block raw-data acceptance. The legacy aggregate index is enabled only with
`RUN_FINGERPRINT_AUDIT_INDEX=1`.

Raw-audit completion means `raw_audit_complete == true`,
`audit_coverage_percent == 100`, `finalized_archives_waiting == 0`,
`archive_errors == 0`, and an exact one-to-one raw archive/quota/report/fingerprint
inventory in `report_coverage`. Missing fingerprints, orphan reports, raw archives
without quota records, quota records without raw archives, identity mismatches,
or invalid report totals all fail the raw-audit gate. This intentionally describes
the ledger that exists at the instant of the snapshot, so an empty ledger remains
diagnostically "audited" for backward compatibility.

Production acceptance additionally requires `collection_closure.complete == true`.
That versioned proof requires all four configured token targets, all three exact
completion-marker identities, a nonempty finalized ledger for every bucket, exact
ledger/raw archive equality, and zero hidden `.part-*`, malformed, unsafe, or
unowned raw inputs. `run_preprocess.sh` always enables
`--require-closed-collection`; therefore an empty or still-growing collection
returns exit `2` even if every currently finalized archive was audited. Use
`--status --require-complete --require-closed-collection --skip-dedup-status` for
a fresh machine-checkable production gate. Plain `--status` continues to return
the throttled cached snapshot for inexpensive monitoring. A
`dedup_index.available == false` status and a telemetry backlog are expected when
optional indexing is skipped and are not evidence of an incomplete corpus. English
near-deduplication and final curation enforce the report-versus-ledger gate
again before freezing their own inventories.

## Production operation

Start the low-priority watcher in a detached session:

```bash
tmux new-session -d -s preprocess \
  /workspace/coding_model_from_scratch/scripts/run_preprocess.sh
```

Monitor logs and structured status:

```bash
tail -f /workspace/dataset/logs/preprocess.log

/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/preprocess_raw_stream.py \
  --root /workspace/dataset --status
```

The audit worker count is pinned at 24 in `run_preprocess.sh` after an exact
server benchmark, and it runs at lower CPU scheduling priority than collectors.
Before starting or resuming, the launcher validates that the network volume's
Stack source manifest exactly matches the trusted deduplicated revision in
`configs/curation_policy.json`; a mismatch fails closed without touching raw or
staged data.
Documents are sent to those workers in ordered batches of at most 64, avoiding
one multiprocessing round trip per document while producing byte-identical
fingerprint shards. Count alone is not a memory bound, so three byte limits now
apply as well: 16 MiB per document, 64 MiB per submitted analysis batch, and
512 MiB across all submitted plus assembling payloads. The parent drains the
oldest future before reading another member whenever the aggregate budget would
be exceeded. Every report records the configured limits, observed payload peaks,
and submitted-batch count. Changing batching limits within the reviewed envelope
does not change record order or fingerprint bytes.

The trailing `_manifest.jsonl` has its own versioned safety envelope because it
is control-plane metadata rather than a document: at most 8 GiB declared member
size, at most 1 MiB per bounded JSONL row, and at most the exact document count
in the committed quota record. One bounded row beyond the quota count is rejected
before JSON decoding. These limits prevent a corrupt manifest from creating an
unbounded allocation or scan while leaving the fingerprint policy hash unchanged.

The per-document limit is an operational safety boundary, not a quality policy.
If a tar header declares a larger member, the preprocessor does not materialize
it. It withholds the entire archive report/fingerprint and writes a typed
`document_exceeds_operational_byte_cap` error containing the member path,
declared bytes, cap, quarantine scope, and retry guidance. It never silently
drops that document or changes the frozen curation policy. Review the source and
RAM envelope before deliberately raising the cap, then retry with
`--retry-errors`.

The large, rebuildable uncompressed metrics spool lives in pod-local
`/tmp/coding-model-preprocess`, not on the network volume. The final compressed
fingerprint is still written beside its destination and atomically renamed on
the network volume. Its SHA-256 is calculated as compressed bytes are written,
eliminating a redundant full-volume read. If requested, the optional telemetry
index independently rereads and verifies that checksum before committing.
Only one writer may operate on a staging root. `STATUS.json` is replaced
atomically at a throttled interval and at clean exit; this avoids an O(N^2)
pattern of repeatedly rereading thousands of reports and quota records while
still giving frequent progress updates. Reading the snapshot from another
shell never waits for a large SQLite transaction.

Free-space gates cover both filesystems independently. `--min-free-gb` guards
the durable dataset/staging filesystem, while `--scratch-min-free-gb` guards the
filesystem that actually backs `--scratch-root` (10 GB by default). The spool
gate is checked before each archive, again before allocating its temporary files,
and periodically while worker results are drained. Falling below either reserve
stops with a nonzero terminal result; it is never reported as successful audit
completion. Production defaults can be overridden with the reviewed
`PREPROCESS_MAX_DOCUMENT_BYTES`, `PREPROCESS_MAX_BATCH_BYTES`,
`PREPROCESS_MAX_INFLIGHT_BYTES`, `PREPROCESS_MAX_MANIFEST_MEMBER_BYTES`,
`PREPROCESS_MAX_MANIFEST_LINE_BYTES`, and `PREPROCESS_SCRATCH_MIN_FREE_GB`
environment variables passed by `run_preprocess.sh`.

Dedup status counters are maintained from the transactionally committed
`ingested_archives` summaries instead of rescanning the multi-gigabyte documents
and cluster tables after every archive. Existing databases are migrated in place
by adding and backfilling an archive token-summary column from immutable reports.
This changes neither document fingerprints nor the pinned preprocessing policy.

The following optional telemetry command can be run later without reopening raw
archives. It is not a production-corpus prerequisite:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/preprocess_raw_stream.py \
  --root /workspace/dataset --index-mode only --once
```

The indexer validates every fingerprint checksum, archive/bucket identity,
document count, and token total. Document inserts are batched, exact and
normalized hashes are aggregated per archive in memory, and SQLite performs
cluster joins/upserts in bulk. A mismatch rolls back the whole archive. The
same staging lock intentionally prevents the audit writer and indexer from
modifying shared state concurrently.

An error report means the archive failed closed and no authoritative final
report/fingerprint pair was published. It may identify source corruption or a
typed operational quarantine such as an oversize member or exhausted scratch
reserve. After diagnosing the cause, rerun with `--retry-errors`; do not silently
ignore the archive.
