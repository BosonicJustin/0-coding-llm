# Fast all-eligible curation runbook

This is the generation-v2 production route from completed preprocessing to the
immutable SQLite authority consumed by the v7 all-eligible publisher. It is a
fresh generation: never point these commands at the stopped v1 curation output
or local database.

The route preserves quality and benchmark rejection, global exact and global
normalized-hash canonicalization, stable leakage-safe group assignment, and
deterministic train/validation/test splits. It intentionally omits fuzzy English
near-deduplication, exact token-quota selection, terminal document prefixes,
legacy JSON decision emission, and periodic full SQLite copies to NFS. Mixture
and input-token budgets are enforced later by packed order v4.

The profile also omits the large `documents_selection_v2` bulk index because
that index exists only to support exact-quota ranking. It retains the content,
final-cluster, source-group, reason, primary-document, and archive-order indexes
needed by canonicalization, group assignment, and the v7 publisher.

## Measured local-storage admission

The admission contract is bound to the stopped v1 production measurement, not
the earlier synthetic 3 KiB/document estimate:

| Measurement | Exact value |
| --- | ---: |
| Inventoried documents | 51,328,930 |
| Maximum database bytes | 67,824,914,432 |
| Maximum WAL bytes | 4,132,940,952 |
| Maximum journal bytes | 0 |
| Maximum transaction rows | 100,000 |
| Committed transactions | 10,415 |

The measured database density is 1,321.378 bytes/document. Admission rounds it
up to 1,322, applies a separate 2x safety factor, adds
`max(256 MiB, batch_size * 64 KiB)` transaction-sidecar capacity, and retains a
2,000,000,000-byte free-space reserve. Existing regular SQLite-owned local
database/WAL bytes are credited only against projected database occupancy on a
resume; they can never pay for the sidecar or free-space reserves.

For 51,328,930 documents and batch size 100,000, the new first-start gate is
144,267,290,920 bytes (134.36 GiB), down from the obsolete
323,918,545,920-byte synthetic gate. For 60,000,000 documents it is
167,193,600,000 bytes (155.71 GiB). The curator computes the exact value from
the finalized v2 report inventory and persists all measurements and arithmetic
in `storage_metrics`, the checkpoint, and the snapshot provenance.

This smaller gate does not weaken runtime checks. Every transaction remains
bounded, the WAL/journal limit remains fail-closed, the 2 GB floor is checked
continuously, SQLite temp storage must share the local database filesystem, and
snapshot publication separately checks durable-volume capacity.

## Preconditions

Before launch, prove all of the following:

1. Other-code collection is closed at the v2 target and no collector is alive.
2. Incremental preprocessing is complete with no error records and exact
   raw/ledger/report/fingerprint coverage.
3. `/workspace/dataset` remains frozen; the run uses only the independent v2
   root and v2 quota configuration.
4. The output and local-work paths below do not contain another generation.
5. `/local` is a positively identified local filesystem; `/workspace` is the
   independent durable network volume.
6. The focused curation/local-store tests pass in the server data environment.

```bash
cd /workspace/0-coding-llm
/opt/coding-model-data-venv/bin/python -m unittest \
  tests.test_curation_local_store \
  tests.test_curate_corpus_local_acceleration \
  tests.test_curate_corpus

df -B1 /local /workspace
findmnt -T /local
findmnt -T /workspace
pgrep -af 'curate_corpus.py|monitor_curation.py' || true
```

Do not delete a stale lease or recovery artifact casually. Use the existing
explicit owner-token recovery procedure only after proving the old process and
pod are gone.

## Launch

Use a fresh output and local directory. The snapshot interval is accepted for
identity compatibility but periodic copies are disabled by the frozen handoff
profile; explicit exits and finalization can still force a recovery snapshot.

```bash
PROJECT_ROOT=/workspace/0-coding-llm
DATA_V2=/workspace/dataset-other-code-topup-v2
QUOTAS=$PROJECT_ROOT/configs/data_quotas_other_code_topup_v2.json
OUTPUT=$DATA_V2/curated/all-eligible-source-v2
LOCAL_WORK=/local/curation/all-eligible-source-v2
RESULT=$DATA_V2/logs/all-eligible-curation-v2.result.json
LOG=$DATA_V2/logs/all-eligible-curation-v2.stderr.log

test ! -e "$OUTPUT"
test ! -e "$LOCAL_WORK"
mkdir -p "$DATA_V2/logs" /local/curation

tmux new-session -d -s all-eligible-curation-v2 -c "$PROJECT_ROOT" \
  "/opt/coding-model-data-venv/bin/python -u scripts/curate_corpus.py \
    --root '$DATA_V2' \
    --staging-root '$DATA_V2/staging/preprocess' \
    --policy configs/curation_policy_fast_exact_normalized.json \
    --quotas '$QUOTAS' \
    --output '$OUTPUT' \
    --sqlite-local-work-root '$LOCAL_WORK' \
    --sqlite-snapshot-interval-seconds 86400 \
    --sqlite-snapshot-retention 2 \
    --defer-raw-archive-integrity-until-finalize \
    --fast-all-eligible-handoff \
    --batch-size 100000 \
    >'$RESULT' 2>'$LOG'"
```

The handoff profile never creates `selection.quota.*` progress, `selected`
rows, output decision rows, or a legacy selection manifest. During bounded
commits, the live local WAL and deterministic local checkpoint remain the
restart authority. It does not synchronously copy the whole database every
hour.

Monitor from a separate tmux session using the existing local-work-aware health
monitor. Snapshot generation count should remain zero during ordinary phase
work.

```bash
tmux new-session -d -s all-eligible-curation-v2-monitor -c "$PROJECT_ROOT" \
  "/opt/coding-model-data-venv/bin/python -u scripts/monitor_curation.py \
    --output '$OUTPUT' \
    --live-work-root '$LOCAL_WORK' \
    --interval-seconds 300 \
    --stall-seconds 3600"

tail -n 50 "$LOG"
find "$OUTPUT/.work/sqlite-snapshots-v1" -mindepth 1 -maxdepth 1 \
  -name 'snapshot-*' -print
```

## Completion gate

A successful normal run publishes exactly one immutable snapshot generation
and returns its manifest, database, and checkpoint paths in `source_snapshot`.
It deliberately does not make a second full canonical-database copy. The v7
publisher consumes the snapshot generation directly.

```bash
jq -e '.complete == true and
       .phase == "canonicalized" and
       .ready_for_all_eligible_publication == true and
       .authority.selected_documents == 0 and
       .authority.decision_archives == 0 and
       .authority.quota_subphases == 0 and
       .authority.fuzzy_near_map_rows == 0' "$RESULT"

SNAPSHOT_MANIFEST=$(jq -r '.source_snapshot.manifest_path' "$RESULT")
SNAPSHOT_DB=$(jq -r '.source_snapshot.database.path' "$RESULT")
SNAPSHOT_CHECKPOINT=$(jq -r '.source_snapshot.checkpoint.path' "$RESULT")

(cd "$(dirname "$SNAPSHOT_MANIFEST")" && \
  sha256sum -c manifest.json.sha256)

jq -e '.phase == "canonicalized" and
       .counts.selected_documents == 0 and
       .counts.output_archives == 0 and
       ([.subphases[] | select(.subphase | startswith("selection.quota."))]
        | length) == 0 and
       ([.subphases[] | select(.subphase == "selection.groups" and
                               .status == "complete")] | length) == 1' \
  "$SNAPSHOT_CHECKPOINT"

sqlite3 "file:$SNAPSHOT_DB?mode=ro&immutable=1" \
  'PRAGMA integrity_check; SELECT json_extract(value,"$") FROM metadata WHERE key="phase"; SELECT COUNT(*) FROM selected; SELECT COUNT(*) FROM output_archives;'
```

Require `integrity_check = ok`, phase `canonicalized`, and both counts zero.
Also require exactly one `all_eligible_handoff_ready` event, one complete
`canonicalized` event, complete `selection.groups` authority, and no snapshot
sidecars. The subsequent publisher independently binds the LocalSQLiteStore
manifest chain, database SHA/state, checkpoint, reports, fingerprints, and raw
archives before emitting keep bitmaps.

## Restart behavior

Rerun the exact command with the exact paths and identity after a normal Python
exception or controlled `Ctrl-C` interruption. Context exit forces one
authenticated recovery snapshot but never the redundant canonical copy. An
uncatchable process crash resumes from the live local WAL if that disk survives.
If pod-local storage is lost, recovery falls back to the newest complete durable
snapshot *when one exists*. On the first uninterrupted run there is deliberately
no periodic durable generation before finalization, so pod destruction or
`SIGKILL` before that boundary can require a fresh run; this is the explicit
throughput/recovery tradeoff. Never reuse a partially created output with
different flags, batch size, policy, quotas, mount identity, or source inventory.
