# Production pre-training runbook

This is the operator sequence for turning the frozen raw collection into the
first 1.3B pre-training run. Follow it in order. A stage is immutable once its
published checksum has been accepted; any policy, source, code, or configuration
change starts a new versioned output directory.

The current harness is **not yet approved for the 52.58B-token launch**. The
CUDA FlexAttention isolation gate, the chosen topology's memory gate, exact CUDA
resume/DDP gates, and a contamination-clean validation loop are mandatory below.
MBPP and the final test order are never used to choose hyperparameters or
checkpoints.

Canonical contracts live in [PRETRAINING_CHECKLIST.md](PRETRAINING_CHECKLIST.md),
[STREAMING_PREPROCESS.md](STREAMING_PREPROCESS.md),
[ENGLISH_NEAR_DEDUP_CALIBRATION.md](ENGLISH_NEAR_DEDUP_CALIBRATION.md),
[ENGLISH_NEAR_DEDUP.md](ENGLISH_NEAR_DEDUP.md),
[CURATION.md](CURATION.md), [MATERIALIZATION.md](MATERIALIZATION.md),
[TRAINING_DATA.md](TRAINING_DATA.md), [MODEL.md](MODEL.md), and
[TRAINING.md](TRAINING.md). This runbook orders those contracts; it does not
replace them.

## 0. Conventions and storage boundary

Run production commands under Bash and retain their stdout/stderr with the run
record. Set an explicit, never-reused `RUN_ID`:

```bash
set -Eeuo pipefail
umask 027

export PROJECT_ROOT=/workspace/coding_model_from_scratch
export DATA_ROOT=/workspace/dataset
export PYTHON_BIN=/opt/coding-model-venv/bin/python
export CPU_LOCAL=/local-nvme
export RUN_ID="${RUN_ID:?set an immutable run ID, for example corpus-v1-gpu-a}"

test -x "$PYTHON_BIN"
test -d "$PROJECT_ROOT"
test -d "$DATA_ROOT"
mkdir -p "$DATA_ROOT/audits/$RUN_ID" "$DATA_ROOT/logs/$RUN_ID"
```

Storage ownership is strict:

| Location | Contents | Authority |
| --- | --- | --- |
| Network volume, `/workspace/dataset` | raw archives; quota ledgers and completion markers; source, tokenizer, preprocessing, calibration, cluster, current fast-curation `.work`, packed, order, provenance, audit, W&B, and checkpoint artifacts | Durable source of truth |
| CPU pod local NVMe, `/local-nvme` | optional rebuildable scratch for a separately qualified future run | Never the sole copy of state required for the current network-volume run |
| Preprocessor `/tmp/coding-model-preprocess` | rebuildable uncompressed spool | Disposable |
| GPU pod local NVMe, `/local-nvme/packed-v1` | verified, read-only hot copy of the finalized packed corpus and orders | Performance copy only |

Never copy a directory while its writer is running. Never copy SQLite, `-wal`,
`-shm`, `.work`, `.part`, lock, or live journal files to make a checkpoint.
Copy only closed published artifacts into a new `.incoming` directory, validate
there, then rename on the destination filesystem. Do not use `rsync --delete`.

## 1. Freeze and audit the collection — GO/NO-GO 1

All collectors must have exited successfully. Catch preprocessing up once, then
take fresh machine-readable snapshots:

```bash
AUDIT_ROOT="$DATA_ROOT/audits/$RUN_ID/readiness"
mkdir -p "$AUDIT_ROOT"

for marker in \
  COLLECTION_COMPLETE.json \
  ENGLISH_FINEWEB_EDU_COMPLETE.json \
  ENGLISH_WIKIPEDIA_COMPLETE.json
do
  test -f "$DATA_ROOT/state/$marker"
done

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/quota_tracker.py" \
  --root "$DATA_ROOT" check --phase collection --json \
  | tee "$AUDIT_ROOT/collection-quotas.json"

PENDING_RAW="$(find "$DATA_ROOT/raw" -type f -name '.part-*' -print -quit)"
test -z "$PENDING_RAW"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/curation_policy.py" \
  --root "$DATA_ROOT" | tee "$AUDIT_ROOT/curation-policy.json"

if tmux has-session -t preprocess 2>/dev/null; then
  echo "The existing preprocess watcher is still active; monitor it and rerun this gate after it exits." >&2
  exit 3
fi

# Frozen quality/data identity is unchanged; these are operational RAM and
# local-spool safety limits. Review any override instead of raising it ad hoc.
export PREPROCESS_MAX_DOCUMENT_BYTES=16777216
export PREPROCESS_MAX_BATCH_BYTES=67108864
export PREPROCESS_MAX_INFLIGHT_BYTES=536870912
export PREPROCESS_MAX_MANIFEST_MEMBER_BYTES=8589934592
export PREPROCESS_MAX_MANIFEST_LINE_BYTES=1048576
export PREPROCESS_SCRATCH_MIN_FREE_GB=10
mkdir -p /tmp/coding-model-preprocess
df -hT /tmp/coding-model-preprocess

"$PROJECT_ROOT/scripts/run_preprocess.sh" \
  2>&1 | tee "$AUDIT_ROOT/preprocess-final-pass.log"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/preprocess_raw_stream.py" \
  --root "$DATA_ROOT" --status --require-complete \
  --require-closed-collection --skip-dedup-status \
  | tee "$AUDIT_ROOT/preprocess-status.json"

jq -e '
  .raw_audit_complete == true and
  .collection_closure.complete == true and
  .collection_closure.failure_count == 0 and
  .collection_closure.completion_markers.valid == 3 and
  .collection_closure.raw_inventory.pending_inputs == 0 and
  .collection_closure.raw_inventory.malformed_inputs == 0 and
  .collection_closure.raw_inventory.unsafe_inputs == 0 and
  .collection_closure.raw_inventory.missing_archives == 0 and
  .collection_closure.raw_inventory.extra_archives == 0 and
  .report_coverage.complete == true and
  .report_coverage.expected_archives == .report_coverage.valid_reports and
  .report_coverage.missing_reports == 0 and
  .report_coverage.unexpected_reports == 0 and
  .report_coverage.invalid_reports == 0 and
  .report_coverage.raw_archives_without_quota == 0 and
  .report_coverage.quota_records_without_raw_archive == 0 and
  .finalized_archives_waiting == 0 and
  .archive_errors == 0 and
  .audit_coverage_percent == 100 and
  ([.audited[] |
      .archives == .finalized_archives and
      .documents == .finalized_documents and
      .tokens == .finalized_tokens] | all)
' "$AUDIT_ROOT/preprocess-status.json" >/dev/null

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_mbpp.py" \
  --root "$DATA_ROOT" | tee "$AUDIT_ROOT/mbpp-audit.json"
jq -e '.findings == []' "$AUDIT_ROOT/mbpp-audit.json" >/dev/null

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m unittest discover -s tests -q \
  2>&1 | tee "$AUDIT_ROOT/cpu-tests.log"

cd "$AUDIT_ROOT"
sha256sum collection-quotas.json curation-policy.json \
  preprocess-final-pass.log preprocess-status.json mbpp-audit.json cpu-tests.log \
  > SHA256SUMS
sha256sum -c SHA256SUMS
```

`run_preprocess.sh` itself rebuilds this exact coverage inventory and exits
nonzero before printing completion if collection markers/targets are incomplete,
any hidden `.part-*` remains, or any archive error or coverage mismatch remains.
An empty or still-live ledger cannot pass merely because all currently visible
shards were audited. A member larger than the 16 MiB operational limit is an explicit typed
archive quarantine, not a silently filtered training example; review it before
changing the cap and retrying. The script independently checks the free space on
both the durable data filesystem and the filesystem actually backing the local
scratch spool. Internal manifests are separately bounded to 8 GiB per member,
1 MiB per JSONL row, and the committed quota document count before decoding.

The optional legacy fingerprint telemetry index may be unavailable or report an
indexing backlog; it is not opened by the production raw-audit gate and is not a
production prerequisite. The required gate is complete,
error-free report/fingerprint coverage. English clustering and curation
independently repeat the authoritative ledger/archive/report/fingerprint
completeness proof, including byte totals; curation also recomputes every raw
archive SHA-256 at its initial and final authority snapshots.

**GO only if:** all quota checks pass; all three completion markers exist;
there are no pending raw files or preprocessing errors; all four buckets have
exact report coverage; the MBPP audit has zero findings; the CPU suite passes;
and the exact code/environment identity is frozen in the experiment record.

## 2. Calibrate English candidate recall — GO/NO-GO 2

> **Deferred for baseline v1 (2026-08-30).** The first speed-focused corpus
> skips Stages 2 and 3. FineWeb-Edu and Wikipedia are already curated sources;
> preserve this full near-deduplication route as a later corpus ablation if the
> baseline underperforms, shows memorization, or a duplicate audit finds
> material redundancy. The baseline still performs deterministic global exact
> and normalized-hash canonicalization plus contamination propagation; it skips
> only the fuzzy/semantic English near pass.
>
> Do not use `--allow-missing-english-near-dedup` for the baseline: that is a
> diagnostic override which publishes `production_ready: false`. Stage 4 uses
> the tested production identity
> `fast-exact-normalized-canonical-v1`; it records the cross-source semantic
> near-duplicate limitation downstream.

Calibration is a bounded read-only diagnostic over real immutable English text.
It does not modify the production configuration.

```bash
CALIBRATION="$DATA_ROOT/audits/english-near-calibration-v1.json"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/calibrate_english_near_dedup.py" \
  --root "$DATA_ROOT" \
  --staging-root "$DATA_ROOT/staging/preprocess" \
  --output "$CALIBRATION"

cd "$(dirname "$CALIBRATION")"
sha256sum -c "$(basename "$CALIBRATION").sha256"
jq -e '
  .status == "pass" and
  .production_gate_eligible == true and
  .acceptance_profile == "pinned-production" and
  (.acceptance_failures | length) == 0
' "$CALIBRATION" >/dev/null
```

Exit `2` is a completed failed acceptance gate; exit `1` is invalid/incomplete
input or an integrity failure. Either is **NO-GO**. Never weaken CLI thresholds
or mutate `configs/english_near_dedup.json` to make a run pass. Version and
review a new config/output instead. The limitations of the finite synthetic
perturbation sample, correlated DOPH rows, probabilistic candidate recall, and
64-bit shingle hashing remain exactly those documented in the calibration and
near-dedup specifications.

Expected resource: bounded selection of at most eight archives and 64 eligible
documents; normally minutes to low tens of minutes on CPU and modest memory.
Record actual wall time. This is an operational gate, not corpus-wide proof.

## 3. Build English clusters on 300 GB local NVMe — GO/NO-GO 3

Use a CPU pod with at least 300 GB free local NVMe and preferably 16 GiB or more
RAM. The live SQLite/band/cache workload stays local. The builder detects the
mount and permits WAL only on positively identified local filesystems; NFS,
network-like, or unknown mounts use rollback `DELETE`. Do not force WAL on NFS.

```bash
ENGLISH_LOCAL="$CPU_LOCAL/english-near-v1"
CALIBRATION="$DATA_ROOT/audits/english-near-calibration-v1.json"

df -hT "$CPU_LOCAL"
AVAILABLE_BYTES="$(df -B1 --output=avail "$CPU_LOCAL" | tail -n 1 | tr -d ' ')"
test "$AVAILABLE_BYTES" -ge 300000000000

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_english_near_clusters.py" \
  --root "$DATA_ROOT" \
  --staging-root "$DATA_ROOT/staging/preprocess" \
  --calibration-result "$CALIBRATION" \
  --output "$ENGLISH_LOCAL" \
  --batch-size 10000 \
  --progress-interval-seconds 60 \
  --sqlite-journal-mode auto \
  2>&1 | tee "$DATA_ROOT/logs/$RUN_ID/english-near.log"

cd "$ENGLISH_LOCAL"
sha256sum -c manifest.sha256
printf '%s  clusters.jsonl.zst\n' "$(jq -er '.mapping.sha256' manifest.json)" \
  | sha256sum -c -
PREFLIGHT_RESULT="$(jq -er '.refinement_operational_preflight.result_path' manifest.json)"
PREFLIGHT_SIDECAR="$(jq -er '.refinement_operational_preflight.sidecar_path' manifest.json)"
test -f "$PREFLIGHT_RESULT" -a ! -L "$PREFLIGHT_RESULT"
test -f "$PREFLIGHT_SIDECAR" -a ! -L "$PREFLIGHT_SIDECAR"
printf '%s  %s\n' \
  "$(jq -er '.refinement_operational_preflight.result_sha256' manifest.json)" \
  "$PREFLIGHT_RESULT" | sha256sum -c -
printf '%s  %s\n' \
  "$(jq -er '.refinement_operational_preflight.sidecar_sha256' manifest.json)" \
  "$PREFLIGHT_SIDECAR" | sha256sum -c -
cmp -s "$PREFLIGHT_SIDECAR" \
  <(printf '%s  result.json\n' \
      "$(jq -er '.refinement_operational_preflight.result_sha256' manifest.json)")
jq -e '
  .manifest_version == 1 and
  .mapping_record_version == 1 and
  .production_ready == true and
  .mapping.singleton_clusters_included == true and
  .mapping.records == .completeness_and_leakage_audit.english_documents_mapped and
  .database_integrity_check == "ok" and
  .identity.calibration_evidence.contract_version == 1 and
  .identity.calibration_evidence.status == "pass" and
  .identity.calibration_evidence.production_gate_eligible == true and
  .identity.calibration_evidence.acceptance_profile == "pinned-production" and
  .identity.calibration_evidence.sampling_profile == "pinned-production" and
  (.identity.calibration_evidence.acceptance_failures | length) == 0 and
  (.identity.calibration_evidence.identity.harness_sha256 | length) == 64 and
  (.identity.calibration_evidence.identity.production_builder_sha256 | length) == 64 and
  .refinement_operational_preflight.contract_version == 1 and
  .refinement_operational_preflight.status == "pass" and
  .refinement_operational_preflight.production_gate_eligible == true and
  (.refinement_operational_preflight.failures | length) == 0 and
  .refinement_operational_preflight.sample.measured_pairs ==
    .refinement_operational_preflight.sample.expected_pairs and
  .refinement_operational_preflight.measurements.union_projected_peak_process_rss_bytes <=
    .refinement_operational_preflight.thresholds.maximum_peak_process_rss_bytes and
  .completeness_and_leakage_audit.mapping_missing_documents == 0 and
  .completeness_and_leakage_audit.mapping_unknown_documents == 0 and
  .completeness_and_leakage_audit.mapping_duplicate_documents == 0 and
  .completeness_and_leakage_audit.normalized_hashes_in_multiple_clusters == 0 and
  .completeness_and_leakage_audit.invalid_cluster_roots == 0 and
  .identity.runtime.storage.sqlite_journal_mode_selected ==
    .identity.runtime.storage.sqlite_journal_mode_actual
' manifest.json >/dev/null
```

Re-running the exact command resumes from `ENGLISH_LOCAL/.work`. For planned
stops use one of `--max-new-inventory-archives`,
`--max-new-signature-archives`, `--max-new-candidate-blocks`,
`--max-new-cache-archives`, `--max-new-refinement-pairs`,
`--max-new-union-edges`, `--max-new-union-finalization-documents`, or
`--stop-after-phase`. A signal or pod loss may repeat the current transaction
but cannot publish a partial mapping. Keep the pod/local disk until the closed
artifact has been copied and verified.

For the expected 10–13 million English documents, 24 band rows per
representative imply roughly 240–312 million disk-backed band rows before
candidate/refinement state. Plan 300 GB local NVMe and a conditional 8–36 hour
window on a capable CPU/NVMe pod. This is not a promise. Candidate generation
records an exact duplicate-block denominator and rates; at the refine boundary
the builder publishes a bounded checksummed real-cache preflight. **NO-GO**
unless that evidence passes its pinned >=1,000 pairs/s, <=72-hour projected
refinement, disk/free-space, and <=12 GiB projected/observed RSS thresholds.
The live `.work/OPERATIONAL.json` rates are useful for later phase ETAs but are
not substitutes for the pre-launch evidence. Refinement rechecks actual peak
RSS and remaining disk after every committed batch; union rechecks its
projected and measured RSS. Posting/candidate overflow and these live resource
gates fail closed rather than truncating or overcommitting.

Copy the complete closed five-file artifact to the network volume. The
preflight subtree is referenced by the manifest and is mandatory authority:

```bash
ENGLISH_NETWORK="$DATA_ROOT/staging/english-near"
ENGLISH_INCOMING="$DATA_ROOT/staging/.english-near-v1.incoming"
test ! -e "$ENGLISH_NETWORK"
mkdir -p "$ENGLISH_INCOMING"
mkdir -p "$ENGLISH_INCOMING/operational-preflight-v1"

rsync -a --partial -- \
  "$ENGLISH_LOCAL/clusters.jsonl.zst" \
  "$ENGLISH_LOCAL/manifest.json" \
  "$ENGLISH_LOCAL/manifest.sha256" \
  "$ENGLISH_INCOMING/"
rsync -a --partial -- \
  "$ENGLISH_LOCAL/operational-preflight-v1/result.json" \
  "$ENGLISH_LOCAL/operational-preflight-v1/result.json.sha256" \
  "$ENGLISH_INCOMING/operational-preflight-v1/"

cd "$ENGLISH_INCOMING"
sha256sum -c manifest.sha256
printf '%s  clusters.jsonl.zst\n' "$(jq -er '.mapping.sha256' manifest.json)" \
  | sha256sum -c -
PREFLIGHT_RESULT="$(jq -er '.refinement_operational_preflight.result_path' manifest.json)"
PREFLIGHT_SIDECAR="$(jq -er '.refinement_operational_preflight.sidecar_path' manifest.json)"
test -f "$PREFLIGHT_RESULT" -a ! -L "$PREFLIGHT_RESULT"
test -f "$PREFLIGHT_SIDECAR" -a ! -L "$PREFLIGHT_SIDECAR"
printf '%s  %s\n' \
  "$(jq -er '.refinement_operational_preflight.result_sha256' manifest.json)" \
  "$PREFLIGHT_RESULT" | sha256sum -c -
printf '%s  %s\n' \
  "$(jq -er '.refinement_operational_preflight.sidecar_sha256' manifest.json)" \
  "$PREFLIGHT_SIDECAR" | sha256sum -c -
cmp -s "$PREFLIGHT_SIDECAR" \
  <(printf '%s  result.json\n' \
      "$(jq -er '.refinement_operational_preflight.result_sha256' manifest.json)")
cd "$DATA_ROOT/staging"
mv .english-near-v1.incoming english-near
```

The complete sibling-manifest consumer contract—not merely these two hashes—is
in `ENGLISH_NEAR_DEDUP.md`; curation enforces it before reading a mapping row.

## 4. Curate and select — GO/NO-GO 4

For the fast baseline, run directly on the durable network volume. This is
slower than local NVMe but restart-safe; explicit DELETE journaling avoids WAL
on NFS. The live `selection-fast-v1` generation uses this baseline path.

An accelerated, crash-safe local-WAL path is implemented for a **fresh output**
and documented in `CURATION_ACCELERATION.md`. It requires roughly 322 GB free
at first startup with the 100,000-row batch contract (500 GB recommended), keeps
SQLite/WAL/temp state on pod-local storage, and periodically publishes verified
recovery snapshots to the network volume. The CLI refuses same-filesystem
"durable" storage and refuses converting or overwriting the baseline canonical
database. Do not switch the live generation until the target-pod equivalence,
crash-recovery, and minimum 3x end-to-end performance gates pass.

First certify rollback-journal locking, abrupt-process recovery, and a minimum
write rate on this exact mounted filesystem:

```bash
CURATION_WORK="$DATA_ROOT/curated/selection-fast-v1"
mkdir -p "$CURATION_WORK/.work"
SQLITE_PROBE="$DATA_ROOT/audits/$RUN_ID/sqlite-network-probe.json"
# Set this from: findmnt -no SOURCE --target "$CURATION_WORK/.work"
RUNPOD_NFS_SOURCE="${RUNPOD_NFS_SOURCE:?set RUNPOD_NFS_SOURCE before this probe}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/probe_sqlite_storage.py" \
  --root "$CURATION_WORK/.work" \
  --result "$SQLITE_PROBE" \
  --rows 200000 \
  --batch-rows 10000 \
  --minimum-rows-per-second 2000 \
  --minimum-index-rows-per-second 10000 \
  --require-fstype nfs4 \
  --require-source "$RUNPOD_NFS_SOURCE" \
  --require-mount-option hard \
  --require-mount-option local_lock=none

(cd "$(dirname "$SQLITE_PROBE")" && \
  sha256sum -c "$(basename "$SQLITE_PROBE").sha256")
jq --arg expected_source "$RUNPOD_NFS_SOURCE" -e '
  .status == "pass" and
  .production_gate_eligible == true and
  .failures == [] and
  .mount.fstype == "nfs4" and
  .mount.source == $expected_source and
  .mount_requirements.missing_options == [] and
  .correctness.competing_writer_excluded == true and
  .correctness.atomic_hardlink_lease_exclusion == true and
  .correctness.advisory_flock_exclusion == true and
  .correctness.hot_journal_observed == true and
  .correctness.hot_journal_magic_valid == true and
  .correctness.hot_journal_bytes > 512 and
  .correctness.main_database_changed_before_recovery == true and
  .correctness.main_database_restored_exactly == true and
  .correctness.post_crash_integrity_check == "ok" and
  .correctness.post_crash_uncommitted_rows_visible == 0 and
  .correctness.post_crash_uncommitted_payloads_visible == 0 and
  .sqlite.temp_store == "file" and
  .sqlite.temp_same_device_as_database == true and
  .measurements.index_rows_per_second >= .minimum_index_rows_per_second
' "$SQLITE_PROBE" >/dev/null
```

This is a process-crash and locking qualification, not a simulated host power
loss. It is intentionally tied to this exact RunPod network-volume identity;
if `/workspace` falls back to ephemeral root storage, the source/type gate
fails. A failed probe is NO-GO for network-volume curation.

Run the curator inside a persistent `tmux` session so an SSH/laptop disconnect
cannot terminate it. The curator also publishes a complete owner record through
an atomic hard link at
`$CURATION_WORK/.curation.cross-client-lease.json`; this prevents two pods or
two processes from writing the same database. A clean exit atomically releases
that lease. A pod/process crash deliberately leaves it in place.

```bash
AVAILABLE_BYTES="$(df --output=avail -B1 "$CURATION_WORK" | tail -n 1 | tr -d ' ')"
test "$AVAILABLE_BYTES" -ge 350000000000

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/curate_corpus.py" \
  --root "$DATA_ROOT" \
  --staging-root "$DATA_ROOT/staging/preprocess" \
  --policy "$PROJECT_ROOT/configs/curation_policy_fast_exact_normalized.json" \
  --output "$CURATION_WORK" \
  --batch-size 10000 \
  --sqlite-journal-mode delete \
  2>&1 | tee "$DATA_ROOT/logs/$RUN_ID/curation.log"

cd "$CURATION_WORK"
sha256sum -c manifest.sha256
jq -e '
  .production_ready == true and
  .english_near_dedup_complete == false and
  .english_near_dedup_status == "disabled_by_fast_profile" and
  .identity.format_version == 6 and
  .curation_profile == {
    "contract_version": 1,
    "name": "fast-exact-normalized-canonical-v1",
    "production_tier": "baseline",
    "fuzzy_near_dedup": false,
    "canonicalization": "global_exact_then_global_normalized_hash",
    "benchmark_propagation": "global_exact_and_global_normalized_hash",
    "split_grouping": "stable_repository_or_english_source",
    "known_limitations": [
      "Semantic near-duplicate documents may remain and may cross source groups or data splits."
    ]
  } and
  .identity.curation_profile == .curation_profile and
  .identity.english_near_artifact == null and
  .identity.english_near_clusters_sha256 == null and
  .identity.curation_storage_contract.contract_version == 2 and
  .identity.curation_storage_contract.projected_additional_bytes_per_document == 3072 and
  .collection_completeness.complete == true and
  .leakage_audit.content_hashes_in_multiple_splits == 0 and
  .leakage_audit.normalized_hashes_in_multiple_splits == 0 and
  .leakage_audit.content_hashes_with_multiple_selected_documents == 0 and
  .leakage_audit.normalized_hashes_with_multiple_selected_documents == 0 and
  .leakage_audit.canonical_clusters_in_multiple_splits == 0 and
  .leakage_audit.source_groups_in_multiple_splits == 0 and
  .leakage_audit.cross_bucket_code_repo_groups_in_multiple_splits == 0 and
  .fast_profile_audit.fuzzy_near_dedup_performed == false and
  .fast_profile_audit.near_map_rows == 0 and
  .fast_profile_audit.near_mapping_subphases == 0 and
  .fast_profile_audit.english_near_duplicate_reasons == 0
' manifest.json >/dev/null

while IFS=$'\t' read -r checksum relative; do
  printf '%s  %s\n' "$checksum" "$relative" | sha256sum -c -
done < <(jq -r '.decision_shards[] | [.sha256, .path] | @tsv' manifest.json)
```

Re-run the exact command to resume. `--max-new-archives` and
`--stop-after-phase` are controlled stop points. Never use
`--allow-missing-english-near-dedup` in production. Capacity and ETA are data
dependent: retain at least 350 GB genuinely free local scratch (preferably a
400+ GB device), or the same free capacity on the network volume for the
explicit NFS-safe run above, and obey the curator's emitted
`required_free_bytes_at_measurement` if it is larger. Measure committed
archives/hour during inventory, and estimate the remaining inventory as
`remaining_archives / archives_per_hour`, adding the canonicalize/select/emit
phases separately. Initial and final authority snapshots each SHA-256 every raw
archive, so also budget `2 * raw_archive_bytes / measured_sequential_hash_rate`
of network-volume reading. Quota selection performs independently indexed
per-bucket scans and a constant-size two-way English merge; it does not build a
corpus-sized in-memory English sort, and SQLite spills any unrelated temporary
work to file-backed storage. Do not reclaim local cluster state until its network
copy and this curation run have both validated.

After a hard crash, inspect the lease owner, confirm the recorded process is
gone, confirm no `curate_corpus.py` process or `tmux` session is active on any
pod mounting the volume, and confirm no database sidecar is changing. Only then
resume with the otherwise identical command plus
`--recover-stale-cross-client-lease OWNER_TOKEN`, using the exact lowercase
`owner_token` shown in the stale JSON record. Recovery holds the NFS advisory
lock, permanently publishes a no-replace claim for that exact token, rechecks
the canonical owner, and atomically archives the old record under
`.stale-curation-leases/`. It never silently deletes evidence and refuses
recovery when the recorded same-host PID is alive. An existing claim for that
token is a manual-review stop, never something to delete automatically.

The command above writes directly to the durable network publication
`$DATA_ROOT/curated/selection-fast-v1`; no second copy or rename is needed.
Only `manifest.json`, `manifest.sha256`, and `decisions/` are closed inputs to
the next stage. The live `.work/` tree remains restart state and must not be
copied or used as materializer input.

The materializer independently takes the strict curation-v6 fast-profile
branch: it revalidates the exact profile/status/limitation/audit schema,
collection authority, every decision and immutable input, and requires both
near-artifact identity fields to be null. The separate full-near branch accepts
only the curation-v5 five-file English publication contract. Its acceptance is
the final curation gate.

The frozen code revision must also make the curation producer's identity
version equal the materializer consumer's accepted version. If they differ,
this is **NO-GO** even when their isolated unit tests pass; resolve the contract
and rerun curation under a new output identity before Stage 5.

## 5. Pack first, without guessing GPU geometry — GO/NO-GO 5

Optionally run the one-archive measurement from `MATERIALIZATION.md` in a
distinct disposable output. Then run production Stage 1 directly to the
network volume so an overnight pod loss does not lose the only checkpoint:

```bash
PACKED_NETWORK="$DATA_ROOT/final/packed-v1"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_training_corpus.py" \
  --root "$DATA_ROOT" \
  --preprocess-root "$DATA_ROOT/staging/preprocess" \
  --selection-root "$DATA_ROOT/curated/selection-fast-v1" \
  --curation-policy "$PROJECT_ROOT/configs/curation_policy_fast_exact_normalized.json" \
  --tokenizer-root "$DATA_ROOT/tokenizer/starcoder2" \
  --output "$PACKED_NETWORK" \
  --stop-after-packing \
  --tokenizer-batch-documents 256 \
  --tokenizer-batch-bytes 67108864 \
  2>&1 | tee "$DATA_ROOT/logs/$RUN_ID/materialize-packing.log"

jq -e '
  .state.phase == "packed" and
  .state.completed_archives == .state.archive_count
' "$PACKED_NETWORK/.materialization-journal.json" >/dev/null
test ! -e "$PACKED_NETWORK/manifest.json"
test ! -e "$PACKED_NETWORK/manifest.sha256"
```

The missing top manifest is intentional: all nine packed manifests and
document indexes are complete and fully validated, but no order exists yet.
Rerun the exact command to resume. Batch sizes may change after measurement
without changing identity. An unhandled stop may replay the current raw archive
into lagging writers; committed archive cursors remain authoritative. Never
manually merge output directories.

At 4,096 input tokens, a packed row is 8,707 bytes. The selected 53.58B-token
budget is about 114 GB (106 GiB) of row payload plus about 0.105 GB of order
references, compressed document indexes, packed surplus, and headroom. Reserve
20–30% beyond the measured complete output. Plan at least 8 GiB RAM; 16 GiB is
safer. If the measured tokenization rate is `R` input tokens/s, the lower-bound
packing ETA is `53.58e9 / R`: about 59.5 h at 0.25M/s, 29.8 h at 0.5M/s,
14.9 h at 1M/s, or 7.44 h at 2M/s, before startup/final validation and stalls.

## 6. GPU smoke and immutable geometry — GO/NO-GO 6

Mount the network volume on the exact intended GPU type/count. Do not choose
geometry from a different topology. For each candidate, build a small,
disposable order from the completed packed manifests. Values below come from
the candidate record; they are deliberately not recommendations:

The planned first training topology is a single RunPod pod with six local GPUs,
so set `GPU_COUNT=6` for that experiment. A different count or GPU type is a new
hardware calibration and cannot silently reuse its accepted geometry or exact
resume trajectory.

```bash
export GPU_COUNT="${GPU_COUNT:?set the actual data-parallel world size}"
export CANDIDATE_GLOBAL_MICROBATCH_ROWS="${CANDIDATE_GLOBAL_MICROBATCH_ROWS:?set candidate}"
export CANDIDATE_GRADIENT_ACCUMULATION_STEPS="${CANDIDATE_GRADIENT_ACCUMULATION_STEPS:?set candidate}"
export SMOKE_OPTIMIZER_UPDATES="${SMOKE_OPTIMIZER_UPDATES:?set bounded update count}"
export SMOKE_WORKERS="${SMOKE_WORKERS:?set loader worker candidate}"
export CANDIDATE_ID="${CANDIDATE_ID:?set a unique candidate ID}"

test $((CANDIDATE_GLOBAL_MICROBATCH_ROWS % GPU_COUNT)) -eq 0
SEQUENCE_LENGTH="$(jq -er '.sequence_length' \
  "$PACKED_NETWORK/packed/train/python/manifest.json")"
SMOKE_INPUT_TOKENS=$((
  SEQUENCE_LENGTH * CANDIDATE_GLOBAL_MICROBATCH_ROWS *
  CANDIDATE_GRADIENT_ACCUMULATION_STEPS * (SMOKE_OPTIMIZER_UPDATES + 1)
))
SMOKE_ORDER="$DATA_ROOT/smoke/geometry-$RUN_ID/$CANDIDATE_ID/order"
test ! -e "$SMOKE_ORDER"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_training_order.py" \
  --python-manifest "$PACKED_NETWORK/packed/train/python/manifest.json" \
  --other-code-manifest "$PACKED_NETWORK/packed/train/other_code/manifest.json" \
  --english-manifest "$PACKED_NETWORK/packed/train/english/manifest.json" \
  --output "$SMOKE_ORDER" \
  --seed 1234 \
  --global-microbatch-rows "$CANDIDATE_GLOBAL_MICROBATCH_ROWS" \
  --gradient-accumulation-steps "$CANDIDATE_GRADIENT_ACCUMULATION_STEPS" \
  --expected-total-input-tokens "$SMOKE_INPUT_TOKENS"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_training_data.py" \
  "$SMOKE_ORDER/manifest.json" --skip-payload-checksums \
  | tee "$DATA_ROOT/audits/$RUN_ID/geometry-$CANDIDATE_ID-order.json"

test "$(jq -er '.training_consumption.optimizer_updates' \
  "$SMOKE_ORDER/manifest.json")" -ge "$SMOKE_OPTIMIZER_UPDATES"

cd "$PROJECT_ROOT"
torchrun --standalone --nproc-per-node="$GPU_COUNT" -m pretrain.train \
  --order-manifest "$SMOKE_ORDER/manifest.json" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --model-size 1.3b \
  --device cuda \
  --precision bfloat16 \
  --workers "$SMOKE_WORKERS" \
  --warmup-steps 0 \
  --checkpoint "$DATA_ROOT/checkpoints/geometry-$RUN_ID-$CANDIDATE_ID/last.pt" \
  --checkpoint-every "$SMOKE_OPTIMIZER_UPDATES" \
  --wandb-mode offline \
  --wandb-run-name "geometry-$RUN_ID-$CANDIDATE_ID"
```

The diagnostic validator may skip the already completed full payload hash; the
production validator below may not. Iterate using new candidate directories.
Measure peak allocated/reserved memory, OOM margin, aggregate input tokens/s,
data-wait time, GPU utilization, compilation behavior, and network checkpoint
latency. Accumulation changes effective optimizer batch but does not solve a
per-forward memory failure.

Publish a checksummed geometry JSON in `$DATA_ROOT/audits/$RUN_ID/` containing
the exact GPU topology/runtime, candidate order hashes, measurements, and the
accepted `global_microbatch_rows`, `gradient_accumulation_steps`, worker count,
fixed-batch overfit rows, and compile decision. Call it
`accepted-geometry.json`; never edit it in place:

```bash
GEOMETRY_RECORD="$DATA_ROOT/audits/$RUN_ID/accepted-geometry.json"
test -f "$GEOMETRY_RECORD"
sha256sum "$GEOMETRY_RECORD" > "$GEOMETRY_RECORD.sha256"
sha256sum -c "$GEOMETRY_RECORD.sha256"
```

**GO only if:** global rows divide world size; BF16 forward/backward and the
CUDA FlexAttention document-isolation test pass; loss/gradients remain finite;
memory has operational headroom; loader and checkpoint latency are acceptable;
and the 1.3B replicated model plus optimizer fits. Current DDP replicates the
model and AdamW state. If it does not fit, implement and validate native FSDP
and sharded checkpoints—do not silently reduce context length or optimizer
precision.

## 7. Construct final orders and validate on the network volume — GO/NO-GO 7

Read, do not retype, the accepted geometry. Resume the same production output
without `--stop-after-packing`:

```bash
GEOMETRY_RECORD="$DATA_ROOT/audits/$RUN_ID/accepted-geometry.json"
GLOBAL_MICROBATCH_ROWS="$(jq -er '.accepted.global_microbatch_rows' "$GEOMETRY_RECORD")"
GRADIENT_ACCUMULATION_STEPS="$(jq -er '.accepted.gradient_accumulation_steps' "$GEOMETRY_RECORD")"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_training_corpus.py" \
  --root "$DATA_ROOT" \
  --preprocess-root "$DATA_ROOT/staging/preprocess" \
  --selection-root "$DATA_ROOT/curated/selection-fast-v1" \
  --curation-policy "$PROJECT_ROOT/configs/curation_policy_fast_exact_normalized.json" \
  --tokenizer-root "$DATA_ROOT/tokenizer/starcoder2" \
  --output "$PACKED_NETWORK" \
  --global-microbatch-rows "$GLOBAL_MICROBATCH_ROWS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  2>&1 | tee "$DATA_ROOT/logs/$RUN_ID/materialize-orders.log"

cd "$PACKED_NETWORK"
sha256sum -c manifest.sha256
test ! -e .materialization-journal.json
jq -e --argjson rows "$GLOBAL_MICROBATCH_ROWS" \
          --argjson accumulation "$GRADIENT_ACCUMULATION_STEPS" '
  .source_cursor.next_archive == .source_cursor.archive_count and
  .order_configuration.frozen_global_microbatch_rows == $rows and
  .order_configuration.frozen_gradient_accumulation_steps == $accumulation
' manifest.json >/dev/null

NETWORK_VALIDATION="$DATA_ROOT/audits/$RUN_ID/final-network-validation"
mkdir -p "$NETWORK_VALIDATION"
for split in train validation test; do
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_training_data.py" \
    "$PACKED_NETWORK/orders/$split/manifest.json" \
    | tee "$NETWORK_VALIDATION/$split.json"
done
cd "$NETWORK_VALIDATION"
sha256sum train.json validation.json test.json > SHA256SUMS
sha256sum -c SHA256SUMS

jq -e --argjson rows "$GLOBAL_MICROBATCH_ROWS" \
          --argjson accumulation "$GRADIENT_ACCUMULATION_STEPS" '
  .format_version == 4 and
  .training_consumption.frozen_global_microbatch_rows == $rows and
  .training_consumption.frozen_gradient_accumulation_steps == $accumulation and
  .training_consumption.dropped_tail_rows == 0 and
  .training_consumption.consumed_input_tokens == .input_token_budget.actual_total and
  .input_token_budget.delta <= .input_token_budget.tolerance
' "$PACKED_NETWORK/orders/train/manifest.json" >/dev/null
```

Do not use `--diagnostic-allow-observed-mixture`, a zero token cap, or
`--skip-payload-checksums` for this gate. Stage 2 atomically publishes the top
manifest only after all three order-v4 manifests validate. Its train manifest,
not a hand-written value, is now authoritative for optimizer updates and exact
consumed/dropped input and supervised tokens.

## 8. Stage and revalidate the GPU-local hot copy — GO/NO-GO 8

Stop the materializer first. The network corpus remains immutable and durable.
Copy only its published tree into a new local incoming directory:

```bash
GPU_PACKED=/local-nvme/packed-v1
GPU_INCOMING=/local-nvme/.packed-v1.incoming
test ! -e "$GPU_PACKED"
mkdir -p "$GPU_INCOMING"

cd "$PACKED_NETWORK"
sha256sum -c manifest.sha256
rsync -a --partial -- \
  "$PACKED_NETWORK/packed" \
  "$PACKED_NETWORK/orders" \
  "$PACKED_NETWORK/provenance" \
  "$PACKED_NETWORK/manifest.json" \
  "$PACKED_NETWORK/manifest.sha256" \
  "$GPU_INCOMING/"

cd "$GPU_INCOMING"
sha256sum -c manifest.sha256
LOCAL_VALIDATION="$DATA_ROOT/audits/$RUN_ID/final-gpu-local-validation"
mkdir -p "$LOCAL_VALIDATION"
for split in train validation test; do
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_training_data.py" \
    "$GPU_INCOMING/orders/$split/manifest.json" \
    | tee "$LOCAL_VALIDATION/$split.json"
done
cd "$LOCAL_VALIDATION"
sha256sum train.json validation.json test.json > SHA256SUMS
sha256sum -c SHA256SUMS

cd /local-nvme
mv .packed-v1.incoming packed-v1
```

An interrupted `rsync` safely resumes into `.incoming`; never train from it.
If `GPU_PACKED` already exists, validate and use that exact version or choose a
new destination—do not overwrite it. Copy ETA is
`published_bytes / sustained_copy_bytes_per_second`, followed by a complete
local checksum scan of roughly the same byte count. Benchmark both rather than
assuming network-volume throughput.

## 9. Overfit, resume, and DDP gates — GO/NO-GO 9

Use the finalized local train order. Select the fixed diagnostic batch size
from the accepted geometry record rather than assuming a value:

```bash
TRAIN_ORDER="$GPU_PACKED/orders/train/manifest.json"
OVERFIT_BATCH_ROWS="$(jq -er '.accepted.overfit_batch_rows' "$GEOMETRY_RECORD")"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/overfit_single_chunk.py \
  --order-manifest "$TRAIN_ORDER" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --model-size tiny \
  --device cuda \
  --precision bfloat16 \
  --batch-size "$OVERFIT_BATCH_ROWS" \
  --steps 100 \
  --output-dir "$DATA_ROOT/audits/$RUN_ID/overfit-tiny"

"$PYTHON_BIN" scripts/overfit_single_chunk.py \
  --order-manifest "$TRAIN_ORDER" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --model-size 1.3b \
  --device cuda \
  --parameter-dtype float32 \
  --precision bfloat16 \
  --batch-size "$OVERFIT_BATCH_ROWS" \
  --steps 100 \
  --checkpoint-every 10 \
  --output-dir "$DATA_ROOT/audits/$RUN_ID/overfit-1.3b"

jq -e '.status == "passed" and .loss_ratio <= .required_loss_ratio' \
  "$DATA_ROOT/audits/$RUN_ID/overfit-1.3b/result.json" >/dev/null
```

For the forced-stop overfit test, start the same 1.3B command in a dedicated
output directory, wait until a scheduled `checkpoint.pt` has closed, interrupt
the process, then rerun the **identical trajectory command** with:

```bash
--resume "$DATA_ROOT/audits/$RUN_ID/overfit-resume/checkpoint.pt"
```

Do not change `--steps`, batch, seed, precision, model, or optimizer settings;
they are checkpoint identity. Compare its final model and optimizer tensors to
an uninterrupted control from the same initial seed and fixed-batch hash.

Next build a new bounded diagnostic order from the GPU-local packed manifests,
using the accepted geometry exactly. Keep it outside the immutable hot copy:

```bash
export DDP_DIAGNOSTIC_UPDATES="${DDP_DIAGNOSTIC_UPDATES:?set bounded update count}"
export DIAGNOSTIC_CHECKPOINT_EVERY="${DIAGNOSTIC_CHECKPOINT_EVERY:?set measured cadence}"
LOCAL_SEQUENCE_LENGTH="$(jq -er '.sequence_length' \
  "$GPU_PACKED/packed/train/python/manifest.json")"
LOCAL_DIAGNOSTIC_TOKENS=$((
  LOCAL_SEQUENCE_LENGTH * GLOBAL_MICROBATCH_ROWS *
  GRADIENT_ACCUMULATION_STEPS * (DDP_DIAGNOSTIC_UPDATES + 1)
))
LOCAL_DIAGNOSTIC_ORDER="/local-nvme/training-smoke/$RUN_ID/order"
test ! -e "$LOCAL_DIAGNOSTIC_ORDER"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_training_order.py" \
  --python-manifest "$GPU_PACKED/packed/train/python/manifest.json" \
  --other-code-manifest "$GPU_PACKED/packed/train/other_code/manifest.json" \
  --english-manifest "$GPU_PACKED/packed/train/english/manifest.json" \
  --output "$LOCAL_DIAGNOSTIC_ORDER" \
  --seed 1234 \
  --global-microbatch-rows "$GLOBAL_MICROBATCH_ROWS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --expected-total-input-tokens "$LOCAL_DIAGNOSTIC_TOKENS"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/validate_training_data.py" \
  "$LOCAL_DIAGNOSTIC_ORDER/manifest.json" --skip-payload-checksums \
  | tee "$DATA_ROOT/audits/$RUN_ID/ddp-diagnostic-order.json"
test "$(jq -er '.training_consumption.optimizer_updates' \
  "$LOCAL_DIAGNOSTIC_ORDER/manifest.json")" -ge "$DDP_DIAGNOSTIC_UPDATES"
```

Run both single-GPU and `torchrun` smokes through `pretrain.train`. Do not pass
`--steps`, microbatch, or accumulation to the trainer; it must read all three
from that diagnostic order.

Run an uninterrupted control and a forced-stop/resume DDP run with the same
topology and trajectory. The trainer handles `SIGHUP`, `SIGINT`, `SIGTERM`, and
`SIGUSR1`: every rank observes the request at a safe optimizer-step boundary,
participates in the required collectives, and rank zero atomically publishes a
graceful-stop checkpoint before exiting with `128 + signal`. Do not use
`SIGKILL`, and do not interrupt an in-progress checkpoint write. The atomic
`last`/`previous` generations remain rollback-safe. Resume with the same
trajectory and world size:

For the production launcher, send the signal to the supervising
`launch_pretraining.py` process. After preflight it re-execs a clean copy at the
same PID so preflight CUDA contexts do not consume training VRAM, isolates
torchrun in a separate session, and uses a shared local request file, giving
workers 30 minutes by default to finish the update and durable checkpoint. This
deliberately avoids TorchElastic's
roughly 30-second signal-shutdown fallback. Configure the pod's termination
grace to be at least `--graceful-shutdown-timeout-seconds`; otherwise the
platform can still issue `SIGKILL` before checkpoint completion. After both
the configured grace period and TorchElastic's cleanup window expire, the
supervisor kills the complete child process group to avoid orphaned ranks and
checkpoint leases.

```bash
torchrun --standalone --nproc-per-node="$GPU_COUNT" -m pretrain.train \
  --order-manifest "$LOCAL_DIAGNOSTIC_ORDER/manifest.json" \
  --tokenizer "$DATA_ROOT/tokenizer/starcoder2" \
  --model-size 1.3b \
  --device cuda \
  --precision bfloat16 \
  --workers "$(jq -er '.accepted.workers' "$GEOMETRY_RECORD")" \
  --checkpoint "$DATA_ROOT/checkpoints/ddp-gate-$RUN_ID/last.pt" \
  --checkpoint-every "$DIAGNOSTIC_CHECKPOINT_EVERY" \
  --wandb-mode offline \
  --wandb-run-name "ddp-gate-$RUN_ID"

# Resume uses the same arguments and adds:
# --resume "$DATA_ROOT/checkpoints/ddp-gate-$RUN_ID/last.pt"
```

**GO only if:** tiny and 1.3B loss fall materially; CUDA document isolation and
backward pass pass; single-GPU and DDP losses/gradients are finite; rank row
ownership is disjoint; uninterrupted versus resumed model/optimizer/cursor/RNG
states match under the documented deterministic gate; checkpoint rollback from
a deliberately interrupted write works; and measured data/communication/
checkpoint performance is acceptable. One mature replicated 1.284B FP32 AdamW
checkpoint is about 15.4 GB; reserve at least two durable generations (about
31 GB plus overhead) per active gate/run. Each DDP replica also holds FP32
parameters, gradients, and two Adam moments: 20,536,918,016 persistent bytes
per GPU before activations and workspaces. The launcher rejects the 1.3B path
below 32 GiB/device and heterogeneous visible GPUs; still require the measured
full-topology memory smoke because 32 GiB is only an admission floor.

## 10. Final launch record and launch — GO/NO-GO 10

Before launch, publish one checksummed immutable run manifest containing the
code/container/runtime hashes, all source/corpus/tokenizer/order hashes, GPU
topology, accepted geometry, optimizer/scheduler/precision/seeds, loader and
compile choices, checkpoint and validation cadence, W&B mode, measured
tokens/s, ETA, and cost. It must also pin the train/validation certification
receipts and sidecars, launch preflight reports, launcher/certifier hashes, and
the deterministic-algorithm/CUBLAS workspace contract. Read geometry and
optimizer update count from the final train order:

```bash
jq '{
  sequence_length,
  training_consumption,
  input_token_budget,
  valid_loss_tokens_per_domain,
  dataset_manifests,
  order
}' "$TRAIN_ORDER" > "$DATA_ROOT/audits/$RUN_ID/final-training-authority.json"
sha256sum "$DATA_ROOT/audits/$RUN_ID/final-training-authority.json" \
  > "$DATA_ROOT/audits/$RUN_ID/final-training-authority.json.sha256"
```

The trainer now consumes the distinct immutable validation order on a pinned
cadence, all-reduces summed loss and supervised-token count, and reports
Python, other-code, and English validation separately without advancing the
training sampler or RNG. The final test order and MBPP remain untouched. Final
training must go through `scripts/launch_pretraining.py`; invoking
`pretrain.train` or `torchrun` directly is not an approved production launch.

Expose the final local copy through a read-only bind mount, certify both exact
orders and all referenced payloads once, and preserve the receipts plus their
SHA-256 sidecars. The launcher binds those receipts to the same local file and
validator/runtime identities, enforces CUDA/BF16/topology, durable-checkpoint,
W&B, and free-space gates, and always passes the deterministic training
contract:

```bash
export TRAIN_LR="${TRAIN_LR:?read from the frozen run manifest}"
export TRAIN_MIN_LR="${TRAIN_MIN_LR:?read from the frozen run manifest}"
export TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:?read from the frozen run manifest}"
export TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:?read from the frozen run manifest}"
export TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:?read from the frozen run manifest}"
export TRAIN_SEED="${TRAIN_SEED:?read from the frozen run manifest}"
export TRAIN_WORKERS="$(jq -er '.accepted.workers' "$GEOMETRY_RECORD")"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:?read from measured network checkpoint policy}"
export EVAL_EVERY="${EVAL_EVERY:?read from the frozen run manifest}"
export EVAL_BATCHES="${EVAL_BATCHES:?read from the frozen run manifest}"
export CHECKPOINT_GENERATION_BYTES="${CHECKPOINT_GENERATION_BYTES:?measured conservative mature-generation bound}"

GPU_PACKED_SOURCE="$GPU_PACKED"
GPU_PACKED_RO=/local-nvme/packed-v1-ro
test ! -e "$GPU_PACKED_RO"
mkdir -p "$GPU_PACKED_RO"
mount --bind "$GPU_PACKED_SOURCE" "$GPU_PACKED_RO"
mount -o remount,bind,ro "$GPU_PACKED_RO"
findmnt -no TARGET,FSTYPE,OPTIONS --target "$GPU_PACKED_RO"

TRAIN_ORDER="$GPU_PACKED_RO/orders/train/manifest.json"
VALIDATION_ORDER="$GPU_PACKED_RO/orders/validation/manifest.json"
TOKENIZER_ROOT="$DATA_ROOT/tokenizer/starcoder2"
GLOBAL_MICROBATCH_ROWS="$(jq -er \
  '.training_consumption.frozen_global_microbatch_rows' "$TRAIN_ORDER")"
TRAIN_DATA_EVIDENCE="$DATA_ROOT/audits/$RUN_ID/train-data-certification.json"
VALIDATION_DATA_EVIDENCE="$DATA_ROOT/audits/$RUN_ID/validation-data-certification.json"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$RUN_ID"
mkdir -p "$CHECKPOINT_DIR"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/certify_pretraining_data.py "$TRAIN_ORDER" \
  --expected-split train \
  --local-data-root "$GPU_PACKED_RO" \
  --output "$TRAIN_DATA_EVIDENCE"
"$PYTHON_BIN" scripts/certify_pretraining_data.py "$VALIDATION_ORDER" \
  --expected-split validation \
  --local-data-root "$GPU_PACKED_RO" \
  --global-microbatch-rows "$GLOBAL_MICROBATCH_ROWS" \
  --output "$VALIDATION_DATA_EVIDENCE"

LAUNCH_COMMON=(
  --train-order-manifest "$TRAIN_ORDER"
  --validation-order-manifest "$VALIDATION_ORDER"
  --tokenizer "$TOKENIZER_ROOT"
  --local-data-root "$GPU_PACKED_RO"
  --durable-checkpoint-root "$DATA_ROOT"
  --checkpoint "$CHECKPOINT_DIR/last.pt"
  --checkpoint-generation-bytes "$CHECKPOINT_GENERATION_BYTES"
  --resume-generation none
  --nproc-per-node "$GPU_COUNT"
  --model-size 1.3b
  --workers "$TRAIN_WORKERS"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --eval-at-start
  --wandb-mode offline
  --wandb-project coding-model-from-scratch
  --wandb-run-name "$RUN_ID"
  --train-data-evidence "$TRAIN_DATA_EVIDENCE"
  --validation-data-evidence "$VALIDATION_DATA_EVIDENCE"
)

"$PYTHON_BIN" scripts/launch_pretraining.py "${LAUNCH_COMMON[@]}" \
  --preflight-report "$DATA_ROOT/audits/$RUN_ID/launch-dry-run.json" \
  --dry-run -- \
  --learning-rate "$TRAIN_LR" \
  --min-learning-rate "$TRAIN_MIN_LR" \
  --warmup-steps "$TRAIN_WARMUP_STEPS" \
  --weight-decay "$TRAIN_WEIGHT_DECAY" \
  --max-grad-norm "$TRAIN_MAX_GRAD_NORM" \
  --seed "$TRAIN_SEED"

# Review the immutable dry-run report and rendered argv, then execute once.
"$PYTHON_BIN" scripts/launch_pretraining.py "${LAUNCH_COMMON[@]}" \
  --preflight-report "$DATA_ROOT/audits/$RUN_ID/launch-execute.json" \
  --execute -- \
  --learning-rate "$TRAIN_LR" \
  --min-learning-rate "$TRAIN_MIN_LR" \
  --warmup-steps "$TRAIN_WARMUP_STEPS" \
  --weight-decay "$TRAIN_WEIGHT_DECAY" \
  --max-grad-norm "$TRAIN_MAX_GRAD_NORM" \
  --seed "$TRAIN_SEED"
```

The production launcher owns data, topology, precision, determinism,
validation, checkpoint, and W&B arguments. The train order owns steps, global
microbatch rows, and accumulation. Ordinary trainer options belong after `--`.
Resume with the exact same command and a fresh preflight report after changing
the launcher-owned generation selector to:

```bash
--resume-generation latest
```

Use `--resume-generation previous` only if loading the trusted latest
generation itself fails. A new run never resumes implicitly. Preserve the two
certification receipts, their sidecars, and both launch reports in the
immutable run record.

W&B offline files live beside the durable checkpoint. Uploading them later is
optional and must not affect training. Monitor global and per-domain training
and validation loss, LR, gradient norm, input/supervised tokens, exact mixture,
tokens/s, GPU memory/utilization, data wait, collectives, checkpoint health, and
remaining authorized tokens. Stop on non-finite values, identity/mixture drift,
corruption, failed durable checkpointing, failed validation, or unexplained
throughput collapse.

The training ETA is conditional: `52.58e9 / measured_aggregate_input_tokens_s`.
Use the slow sustained post-warmup rate from the accepted full-topology smoke,
then add measured validation and checkpoint downtime. Do not estimate from GPU
marketing throughput.
