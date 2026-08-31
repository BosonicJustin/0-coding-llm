# Experiment handoff

This file is the short operational handoff. The detailed, authoritative
procedures remain in [production-runbook.md](production-runbook.md) and
[pretraining-checklist.md](pretraining-checklist.md).

## Storage boundaries

Keep the two data lifecycles physically separate:

| Purpose | Server root | May feed pre-training? |
|---|---|---|
| Pre-training acquisition, curation, and materialization | `/workspace/dataset` | Yes, only after published-manifest validation |
| SFT/RL source downloads and derived artifacts | `/workspace/posttraining-data` | No |

Never point the pre-training curator or materializer at the post-training root.
Downloaded SFT data is quarantined until benchmark decontamination, schema
normalization, length analysis, split construction, and a separate SFT
publication manifest are complete.

## Last verified server state at 2026-08-31 16:53 UTC

- The resized CPU pod exposes a 320 GiB local overlay
  (343,597,383,680 bytes total) and the durable NFS network volume at
  `/workspace`. The exact accelerated admission gate required
  323,918,545,920 bytes free; 343,580,889,088 bytes were initially free, for a
  19,662,343,168-byte margin.
- Commit `85fb98a` and a minimal Python 3.11 runtime were deployed in the
  separate checkout `/workspace/0-coding-llm`. The 56 focused target-runtime
  curation, local-store, and monitor tests passed.
- A controlled one-archive run created the fresh accelerated generation
  `selection-fast-local-v2`, processed 100,000 documents, published and
  authenticated snapshot `snapshot-000000000001`, and passed SQLite
  `integrity_check` for both the local database and durable canonical copy.
- The full accelerated run resumed that exact local generation at 07:04 UTC in
  tmux session `curation-fast-local-v2`. Its first healthy monitor record at
  07:05 UTC reported 8 archives, 733,993 documents (1.430%), a live curator,
  no warnings, and 342,497,202,176 local bytes free. The monitor runs in
  `curation-fast-local-v2-monitor`; its first failed record was a harmless
  publication-boundary race before the curator had published its next
  running-archive subphase, followed by the healthy record.
- At 07:08 UTC the live checkpoint had advanced to 28 archives and 2,583,958
  documents (5.034%), with archive 29 running, no storage violation,
  340,704,468,992 local bytes free, a 1.75 GB database, and a bounded 641 MB
  WAL. Both curator and monitor tmux sessions remained alive.
- Commit `85fb98a` makes that monitor race bounded and explicit: a fresh
  zero-active inventory projection is warning-only for at most 60 seconds when
  the bound curator is alive and every other invariant passes; stale, dead,
  inconsistent, or over-bound states still fail. All 12 focused monitor tests
  passed on the pod, only the monitor was restarted, and its first record from
  the new code was healthy at 66 archives / 6,001,567 documents (11.692%). The
  curator was never stopped and advanced to 69 archives / 6,301,563 documents
  (12.276%) by 07:15 UTC with no violation.
- At 07:45 UTC the monitor reported 235 archives and 18,503,372 / 51,328,930
  documents inventoried (36.049%), with `other_code.000142` active, the curator
  alive as PID 368, a current checkpoint, and no warnings. The live database
  was 12,673,970,176 bytes with 329,775,230,976 local bytes free. The first
  hourly snapshot for this full-run interval is due after 08:04 UTC; the
  authenticated controlled-run snapshot remains durable as
  `snapshot-000000000001`.
- Inventory completed all 51,328,930 documents across 4,345 archives with exact
  per-source totals matching the frozen collection authority. Snapshot
  `snapshot-000000000003` published successfully at 13:35 UTC with a
  31,412,785,152-byte SQLite payload. Its synchronous NFS copy, two full
  integrity scans, and SHA-256 publication blocked processing for roughly three
  hours; do not retain this hourly full-snapshot design for the next rollout.
- The curator advanced automatically into canonicalization. At 14:03 UTC it was
  running `canonicalize.exact_choice` with a healthy current checkpoint, the
  bound curator alive as PID 368, no storage violation, and 293,162,405,888
  local bytes free.
- The monitor briefly exited at 13:54 UTC because it interpreted the legitimate
  `inventory.bulk_indexes` transition as an active archive and required the
  archive-only `expected_documents` field. The curator never stopped. The
  monitor was restarted at 14:03 UTC after the phase had advanced and
  immediately published a healthy record. The monitor now distinguishes the
  exact bulk-index subphase, requires terminal inventory accounting for it, and
  still rejects every unknown inventory subphase.
- Exact-choice canonicalization completed for all 49,463,280 groups. At 14:35
  UTC, final-choice canonicalization had committed 26,400,000 rows in 264
  atomic batches with cursor
  `88a0a01f84ece7b0edb1d53a21ce6e4c5d74945ae68fe659f066336ebbed9215`.
  The live checkpoint and journal hashes were respectively
  `1c5fb4464d2bdb567b5ed057b868c560ea462e30d13895ba7b81943faa90afbc`
  and `20d76f3801d81337ac84fe5347a74f70dd41c7ca4b568a87be6ca5328703ceb3`.
- Hourly snapshot generation four copied the then-60,043,280,384-byte database
  to NFS but remained inside its integrity pass with no published manifest.
  At 16:25 UTC the monitor was stopped, exact curator PID 368 was verified
  against its command and durable lease, and that PID received `SIGTERM` to use
  the tested WAL crash-recovery path. The local database, WAL, SHM, checkpoint,
  and journal were not moved, truncated, or deleted; their checkpoint/journal
  hashes remained unchanged after process death. Snapshot three remained the
  newest complete recovery authority.
- The same generation restarted at 16:28 UTC as PID 882. The frozen identity,
  retention two, policy, batch size, journaling mode, and deferred raw-integrity
  policy are unchanged; only the full-snapshot interval changed from 3,600 to
  21,600 seconds. The restart explicitly claimed the recorded stale lease,
  automatically removed manifestless snapshot four, authenticated snapshot
  three, hashed the existing local database and sidecars, and passed WAL,
  SQLite quick-check, and foreign-key recovery gates.
- Resume is proven rather than inferred: final-choice first reappeared at or
  beyond the exact 26,400,000-row / 264-batch floor, then advanced to
  30,200,000 rows / 302 batches by 16:53 UTC with a strictly later cursor. The
  16:52 monitor record was healthy with no warnings, the new curator and monitor
  tmux sessions were alive, no active health alert existed, local storage had
  264 GiB free, and NFS had 814 GiB free. The previous false-positive alert is
  retained under `.work/archived-health-alerts/`, not silently discarded.
- The accelerated output is durable at
  `/workspace/dataset/curated/selection-fast-local-v2`; its live SQLite/WAL
  authority is `/local/curation/selection-fast-local-v2`. Full authenticated
  snapshots are now scheduled every six hours with retention two. Snapshot
  three remains the latest complete NFS recovery point until generation four is
  republished; do not delete the pod or its local volume in that interval.

- The network-volume baseline curator was stopped deliberately with `Ctrl-C`
  through its `curation-fast-v1` tmux foreground job before the storage resize.
- Its resumable checkpoint is in `inventory`: 11,236,456 / 51,328,930 documents
  (21.891%), 113 archives completed, and `other_code/part-000020` is durable at
  30,000 / 151,510 rows, with no recorded storage violation.
- Shutdown verification found no Python `curate_corpus.py` process, no SQLite
  journal/WAL/SHM sidecar, and no cross-client lease. The canonical working
  database is 7,679,508,480 bytes and remains on the network volume.
- The old pod had 844 GB free on NFS but only 3.4 GB free local overlay and
  60 GB `/dev/shm`; that is insufficient for accelerated local-WAL curation.
  The visible unmounted NVMe devices had no corresponding `/dev` nodes inside
  the container and were not claimed, mounted, formatted, or modified.
- The accelerated implementation landed in commit `6b7decc`; the authenticated
  stop record is commit `7a6d35e`. It deliberately refuses to convert or
  overwrite the baseline output.
- The baseline checkpoint remains a rollback/resume authority. A local-WAL
  rollout will start a new output generation and must not delete
  `selection-fast-v1` until equivalence and performance gates pass.
- Quality/benchmark filtering, exact/normalized canonicalization,
  leakage-safe grouping, split assignment, and quota selection are later
  phases of that same restart-safe process.
- Tokenization is not running. It is a separate, deliberate materialization
  launch after curation publication and validation.
- Raw `nvidia/OpenCodeInstruct` at commit
  `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` is downloaded under the
  post-training root: 50 Parquet shards, 6,861,113,102 bytes, and 5,000,000
  metadata-verified rows. A streaming SHA-256 inventory completed successfully
  and published `SOURCE.json` plus `COMPLETION.json`. It is still quarantined
  and is not yet approved as SFT training input.
- The active server checkout was safely fast-forwarded through the repository
  organization changes without restarting the curator; the running curator was
  launched from the preceding `4b2e985` code state, and its curation code and
  policy were not changed by those commits. The checkout is kept clean and
  synchronized with GitHub `main`. Never mutate the preserved baseline `.work`
  tree in place.

The accelerated checkpoint and health log are the live authorities. Re-read
them together with `tmux ls`, mount identities, and `df` before intervention.

## Monitoring without mutating state

Inside the CPU pod:

```bash
tmux ls
pgrep -af 'curate_corpus.py|download_sft_dataset.py|hf download'
python3 -m json.tool \
  /local/curation/selection-fast-local-v2/CHECKPOINT.json
tail -n 2 /workspace/dataset/logs/curation-fast-local-v2-monitor.log
df -h / /workspace
```

The curation checkpoint is authoritative; the curation log may be empty because
progress commits are recorded in `CHECKPOINT.json` and SQLite.

## What follows automatically

The accelerated curator has completed inventory, bulk indexes, and exact-choice
canonicalization and is advancing the restartable final-choice subphase. It
will continue automatically through the remaining exact/normalized
canonicalization accounting, leakage-safe grouping, split assignment, quota
selection, final deferred raw-integrity verification, and atomic publication.
It must not be replaced with an ad hoc filtering process. A storage violation,
lease conflict, identity mismatch, accounting mismatch, or failed health record
after startup is a hard stop. Token materialization remains a separate launch
after final curation certification.

## What requires a separate launch

After successful curation publication:

1. Validate the curation manifest and all zero-leakage/accounting evidence.
2. Run materialization through the packing-only boundary.
3. Certify packed checksums, document-position indexes, and decoded samples.
4. Mount the packed corpus on the intended GPU topology and calibrate loader,
   microbatch, accumulation, memory, and throughput.
5. Freeze order format v4 with the measured geometry.
6. Run the CUDA/FlexAttention, BF16, single-GPU, multi-GPU, and exact-resume
   gates before the long job.

The planned training target is one six-GPU RunPod pod, not a multi-node job:
launch one NCCL process per GPU with
`torchrun --standalone --nproc-per-node=6`. Freeze only a global microbatch
divisible by six, keep the hot packed copy on pod-local NVMe when available,
and keep durable checkpoints/audits on the mounted network volume. The GPU
model and VRAM still determine whether replicated DDP fits or native FSDP must
be implemented before order v4 is frozen.

For OpenCodeInstruct, independently:

1. The raw pinned snapshot and source manifest are complete and immutable.
2. Run `scripts/prepare_prime_sft.py` to create a new derived publication; do
   not train from `raw/`.
3. The implemented policy scans prompt, answer, and unit-test text against the
   frozen MBPP denylist and propagates hits across normalized prompt groups.
   It does not claim fuzzy/semantic-paraphrase coverage, and adding any future
   final benchmark requires a new policy and output version.
4. The curator creates deterministic leakage-safe train/validation groups and
   uses the exact pinned tokenizer plus `starcoder2-coding-chat-v1`. Complete
   conversations over 4,096 causal inputs are dropped, never truncated.
5. Audit the real score/retention distribution and publish a new positive
   `min_average_test_score` policy. The current `0.0` output is inspection-only
   and is blocked by the Prime launcher for real training.
6. Export the selected format-v5 pretraining checkpoint through
   `scripts/export_hf_checkpoint.py`, prewarm the offline HF cache once, and use
   `scripts/launch_prime_sft.py`. Never invoke Prime's `sft` command directly.
7. Prime must retain custom Llama plus `seq_lens` boundary isolation. Before a
   real six-GPU run, complete the one-GPU overfit/BF16/memory and six-rank NCCL
   boundary-perturbation gates in
   [prime-sft.md](../posttraining/prime-sft.md).

## Verified code state

The model/trainer stack includes real two-process CPU/Gloo gates for DDP token
normalization and atomic checkpoint → process restart → exact immutable-order
resume. Checkpoint format v5 stores only each rank's assigned CUDA RNG state
and binds every saved generation to the authenticated tokenizer-manifest and
canonical token-to-ID vocabulary SHA-256 identities.
Remaining gates are intentionally left unchecked in
[pretraining-checklist.md](pretraining-checklist.md): some depend on the final
curated/materialized data and manual sample inspection; others require the
eventual GPU image and topology.
