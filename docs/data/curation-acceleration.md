# Curation acceleration design

## Status

No curator is running. The `selection-fast-local-v2` database described here is
stopped and retained as first-generation audit evidence; do not resume these
commands for the current corpus. The independent top-up generation must run a
fresh full canonical and leakage-safe group build, then use all-eligible bitmap
publication. See [fast-generation-v2.md](fast-generation-v2.md).

The compatible local-WAL path remains implemented as an explicit opt-in for
that future fresh output. It does not authorize modifying or replacing an
existing generation. The append-staging/parallel-decoder work is split: an
append-only staging prototype and benchmark are implemented but not integrated,
while parallel decoding remains a future optimization that should be attempted
only if a new local-WAL benchmark shows decoding is the next bottleneck.

## Why the baseline is slow

The baseline inventory writes every document into a `WITHOUT ROWID` SQLite table
keyed by a random SHA-256 document identifier while maintaining two additional
uniqueness B-trees. It commits a `synchronous=FULL` rollback-journal transaction
and publishes a complete checkpoint every bounded batch. On a network filesystem,
latency from random page writes, journal synchronization, locking, and checkpoint
metadata dominates CPU work. More decoder workers cannot remove that single-writer
storage bottleneck.

The current RunPod host reports 1 TiB of host memory, but the container cgroup is
limited to 128 GB and `/dev/shm` is limited to 60 GB. The accelerated path must
therefore use bounded RAM and adequately sized pod-local storage; it must not rely
on a fully memory-resident production database.

## Required invariants

The accelerated path must preserve all of the following:

- identical document identities, quality reasons, canonical choices, source
  groups, leakage-safe splits, quota selections, and decision records;
- deterministic results independent of worker count, completion order, batch
  size, interruption point, or resume count;
- immutable identities for policy, quotas, benchmark guard, preprocessing
  reports, fingerprints, raw archives, tokenizer, and runtime storage mode;
- bounded transactions, bounded queues, explicit disk and memory admission, and
  fail-closed handling of exhausted resources;
- a network-volume recovery authority that never claims progress newer than its
  transactionally consistent database snapshot;
- final raw-archive authentication before any production manifest is published;
- atomic publication, checksums, database integrity checks, and a retained
  rollback generation until certification completes.

## Architecture

### Source and publication storage

Raw archives, preprocessing artifacts, durable recovery snapshots, final decision
shards, and final manifests remain on the network volume. They are never mutated
by the accelerated working generation.

Production startup enforces this topology: the output must be on a positively
identified network filesystem, the SQLite work root must be on a positively
identified local filesystem, and the two must resolve to different mount/device
identities. The test suite has a private same-device override for temporary
fixtures; the CLI intentionally exposes no production bypass.

### Local working storage

SQLite databases, journals/WAL files, sorter temporaries, and bounded decoder
spools live on a separately configured pod-local filesystem. Production admission
must verify the filesystem identity and enough free space for the database,
temporary indexes, snapshots in flight, and a safety margin.

### Inventory ingestion

The separately gated append-only prototype uses this layout:

1. Reports are assigned a canonical ordinal independent of worker scheduling.
2. Workers decode and validate fingerprints into bounded local staging batches.
3. A staging row uses a sequential key and has no global secondary indexes.
4. Batch and archive cursors become durable only with their SQLite transaction.
5. Global uniqueness and canonical ordering are validated during deterministic
   bulk promotion.
6. Required indexes are bulk-built after ingestion with local temporary storage.

The production candidate keeps the existing schema but moves it to local storage,
uses larger bounded transactions, and publishes checkpoint authority only with
periodic durable snapshots.

The append-only engine passes semantic, corruption, real process-death, resume,
and interrupted-promotion tests. It is not wired into `curate_corpus.py` because
its measured local query-ready time is currently worse than the existing schema:

| Local benchmark | Existing schema | Append ingest | Append + promotion |
| --- | ---: | ---: | ---: |
| 100k rows, median of 3 | 1.486 s | 1.471 s | 2.802 s |
| 1M rows, 1 run | 24.071 s | 19.652 s | 37.270 s |

At one million rows, append-only ingestion was 1.225x faster and the database was
about 14.6% smaller, but promotion made query-ready completion 0.646x as fast as
the baseline. A target-NFS, out-of-cache benchmark may behave differently; until
then, integrating this schema would make the evidence worse rather than better.

### Recovery snapshots

Local progress and network-durable progress are distinct. A recovery publication
contains a transactionally consistent SQLite snapshot, its byte length and
SHA-256 digest, its exact database cursor, and an atomically replaced authority
manifest. A restart may discard work newer than the last published snapshot; it
must never advance the durable cursor without the matching snapshot.

Snapshots use generation-qualified names. A new snapshot is written, flushed,
authenticated, and validated before its authority manifest is published. Older
snapshots remain recoverable until the newer authority is complete.

The implementation retains exactly the configured number of complete recovery
databases (minimum and default: two). Before an older database payload is
deleted, an immutable retirement receipt binds its generation, snapshot-manifest
hash, database hash, and byte length. The small immutable manifests remain as the
hash-chain history. This bounds full database copies instead of accumulating one
per hourly snapshot. The next snapshot also performs a dynamic durable-space
check using the active database page count and page size.

Live `CHECKPOINT.json` and `journal.jsonl` projections stay in the local work
directory because they may be ahead of the last snapshot. Each durable snapshot
contains its own manifest-bound checkpoint and journal. The durable top-level
canonical database advances only through a verified atomic promotion. Therefore,
losing the local disk can discard recent work, but can never make recovery trust
an ahead cursor.

The local database uses `synchronous=FULL`, WAL, exclusive locking under the
existing cross-client lease, and a bounded 32,768-page WAL autocheckpoint. The
network-volume canonical snapshot preserves the journal mode selected for that
mount (normally rollback journaling on NFS).

### Raw integrity timing

Fingerprint curation does not read raw training text. An accelerated resume may
defer the expensive full raw-archive rehash, provided that canonical paths, sizes,
report identities, and fingerprint identities are validated first and every raw
archive is fully rehashed before final publication. A raw mismatch must invalidate
the generation before any production manifest becomes authoritative.

## Qualification gates

An accelerated implementation is eligible to replace the baseline only after:

1. fresh and resumed fixture outputs are byte-identical to the scalar baseline;
2. results are identical across supported worker counts and batch sizes;
3. forced termination before, during, and after snapshot publication recovers to
   an exact durable cursor;
4. corrupt, truncated, stale, symlinked, or mismatched snapshots fail closed;
5. local-space and cgroup-memory preflights reject unsafe configurations;
6. final raw mutation is detected before publication;
7. SQLite integrity and foreign-key checks pass after restore and finalization;
8. a representative real-data benchmark demonstrates at least a 3x end-to-end
   improvement before the live rollback generation is stopped.

The local tests cover semantic-equivalent decisions, configured WAL/locking,
snapshot corruption fallback, bounded retention, promotion retry, active-
transaction rejection, tmpfs/cgroup rejection, local-audit-ahead crash recovery,
and mandatory final raw hashing. They do not replace a target-pod benchmark.

## Capacity and expected speed

For the frozen inventory, the first-start admission requirement is exactly
323,918,545,920 bytes (about 301.67 GiB): 315,364,945,920 bytes for the projected
database including its 2x safety factor, 6,553,600,000 bytes for the bounded
100,000-row transaction sidecar, and a 2,000,000,000-byte reserve. The 320 GiB
rollout exposed 343,580,889,088 bytes free at admission, leaving a
19,662,343,168-byte margin (about 18.31 GiB). Resumes credit authenticated
database/WAL bytes already occupying that volume against the final projection;
they never require room for a second complete local database. A larger local
volume still leaves safer headroom for filesystem allocation and sorter
variability.

An observed inventory density near 1,438 bytes/document would put one real
database near 73 GB. Durable steady state then uses about 220 GB for the canonical
database plus two recovery snapshots. The more conservative unsafetied contract
projection is about 470 GB for those three copies. Canonical promotion briefly
needs one additional independent copy and now has a separate fail-closed capacity
check. A hard link is deliberately not used: mutating a later canonical database
would otherwise mutate the supposedly immutable recovery snapshot through the
shared inode. Exact free-space admission is repeated before every full snapshot,
and the network volume must also retain raw archives, fingerprints, and final
outputs.

Moving indexed random writes and WAL checkpoints off NFS should plausibly improve
the database-heavy inventory phase by 5–20x and end-to-end curation by 3–10x,
depending on decoder and snapshot time. These are planning ranges, not measured
claims. Full snapshots write one database sequentially to NFS, so benchmark the
interval and do not reduce it merely to make progress telemetry look fresher.
The qualified live rollout measured roughly three hours to publish a 31.4 GB
snapshot through copy, integrity, and SHA-256 gates. Its operational interval is
therefore six hours: fewer full-copy stalls in exchange for a larger rollback
window if both the pod and its local disk are lost.

## Benchmark matrix

Use the same frozen subset and compare:

| Variant | Work storage | Inventory layout | Batch size | Checkpointing |
| --- | --- | --- | ---: | --- |
| Baseline | network volume | indexed documents | 10,000 | every batch |
| Batched NFS | network volume | indexed documents | 100,000 | bounded |
| Local compatible | local disk | indexed documents | 100,000 | local + snapshots |
| Local staged | local disk | append staging + bulk promotion | 100,000+ | local + snapshots |

Record documents/second, source bytes/second, SQLite bytes read/written, commit
latency, checkpoint latency, peak RSS/cgroup memory, local and network bytes, final
database size, snapshot time, restore time, and total phase time. Compare semantic
tables and final files, not only aggregate counts.

## Deployment rule

Do not overwrite the baseline output. The accelerated run uses a new output and
generation identity. Preserve the baseline checkpoint and working database until
the accelerated final manifest and training-data certification have both passed.
The curator enforces this boundary: local mode refuses a canonical database that
does not already carry the exact accelerated SQLite execution identity.

## Opt-in local-work invocation

After qualification on the target pod, use a fresh output and a dedicated local
directory. The local directory must be on a positively identified local
filesystem and must pass the projected-capacity admission check.

```bash
python scripts/curate_corpus.py \
  --root /workspace/dataset \
  --staging-root /workspace/dataset/staging/preprocess \
  --policy configs/curation_policy_fast_exact_normalized.json \
  --output /workspace/dataset/curated/selection-fast-local-v2 \
  --sqlite-local-work-root /local/curation/selection-fast-local-v2 \
  --sqlite-snapshot-interval-seconds 21600 \
  --sqlite-snapshot-retention 2 \
  --defer-raw-archive-integrity-until-finalize \
  --batch-size 100000
```

The example path `/local` is illustrative. Do not use the current pod's 5 GB
container root or its 60 GB `/dev/shm`; provision enough local disk and verify its
mount identity first. Do not point the new invocation at the baseline output.

The live health monitor reads the local checkpoint while keeping its health log
and alert on durable storage:

During the narrow resume-startup or inter-archive window before the curator
publishes its next active archive, the monitor emits a warning-only
`startup_publication_grace` record for at most 60 seconds, and only while the
bound curator process is alive and all other checkpoint, journal, count, and
storage invariants pass. The condition is fatal after that bound.

```bash
python scripts/monitor_curation.py \
  --output /workspace/dataset/curated/selection-fast-local-v2 \
  --live-work-root /local/curation/selection-fast-local-v2 \
  --interval-seconds 300 \
  --stall-seconds 3600
```

These commands are operating examples, not permission to switch the production
generation. The benchmark and recovery gates remain mandatory.

For an exact-PID, lease-fenced restart of an already qualified generation,
including the tested `SIGTERM`/WAL recovery path and the required monitor delay,
follow [Controlled restart of the local-WAL curator](../operations/production-runbook.md#controlled-restart-of-the-local-wal-curator).
