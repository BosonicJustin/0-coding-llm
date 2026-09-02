# Experiment handoff

This is the short operational handoff. The canonical corpus status and identity
authority is the [pre-training corpus record](../data/final-corpus-record.md).
The completed v2 design and build history are in
[fast-generation-v2.md](../data/fast-generation-v2.md); historical incidents
and measurements remain in [experiment-log.md](../experiment/experiment-log.md).

## Storage boundaries

| Purpose | Location | State |
|---|---|---|
| Frozen pre-training generation v1 | `/workspace/dataset` | Read-only; do not resume or mutate jobs here |
| Pre-training top-up generation v2 | `/workspace/dataset-other-code-topup-v2` | Closed build origin; collection, preprocessing, curation, selection-v7, cache, and packing complete |
| Portable packed backup | `s3://transcendent-logic-data-618079239540/coding-llm/pretraining/2026-09-02-packed-v1/` | Packed-phase recovery source only; orders and top-level manifest are absent |
| Six-H100 hot path | `/root/transcendent-logic-data` | Restored to pod-local NVMe and checksum verified |
| Six-H100 checkout | `/root/0-coding-llm` | Code used for qualification and corpus consumption |
| SFT/RL source and derived artifacts | `/workspace/posttraining-data` | Quarantined from pre-training |

The selective v2 clone hard-links immutable v1 raw archives, tokenizer files,
manifests, collector state, quota records, and completed preprocess reports and
fingerprints. Closure/status files are copied to new inodes. Logs, locks,
temporary files, telemetry SQLite, curation, and packed outputs are excluded.
Never edit a hard-linked existing file in place; append new immutable files or
atomically replace only the independently copied v2 control files.

## Current state — 2026-09-02

- Collection and streaming preprocessing are complete. The closed v2 corpus
  contains 4,568 raw archives, 56,502,609 documents, and 73,992,914,797 exact
  raw tokens. All 4,568 reports and fingerprints are present with zero archive
  error records.
- Exact raw totals are 25,770,142,666 Python tokens, 35,363,570,483 other-code
  tokens, 10,287,360,535 FineWeb-Edu tokens, and 2,571,841,113 Wikipedia
  tokens.
- Corpus curation and leakage-safe group splitting are complete. Selection-v7
  published one authenticated keep/reject bitmap per raw archive, and the
  closed-world token-cache inventory is complete.
- Packing is complete for all nine `{train,validation,test} x
  {python,other_code,english}` cells. The packed journal is at phase `packed`
  with all 4,568 archives committed. Immutable token and document-start shards,
  packed manifests, tokenizer, selection, and cache authorities are present.
- The portable S3 publication contains 17,883 objects and 129,807,857,131
  bytes; its packed payload contains 13,284 files and 129,783,678,021 bytes.
  It has been restored directly to six-H100 pod-local NVMe with checksum
  verification; it did not transit through the laptop.
- Deterministic split orders and the top-level production corpus manifest are
  intentionally absent. They remain pending the six-GPU geometry decision and
  portable finalization. The packed-only artifact is recoverable input, not
  full-run launch authority.
- Raw OpenCodeInstruct remains under `/workspace/posttraining-data`; it does not
  enter either pre-training generation.

## Historical pre-top-up supply audit

Before the v2 top-up, the read-only WAL-aware audit completed successfully in
255.2 seconds. It proved 49,461,115 eligible canonical documents, 14,968,335 leakage-safe groups,
zero accounting anomalies, and these domain totals across all splits:

| Domain | Documents | Eligible content tokens |
|---|---:|---:|
| Python | 23,182,684 | 23,322,647,918 |
| Other code | 15,697,799 | 16,811,351,831 |
| English | 10,580,632 | 12,478,718,733 |

Artifact:
`/workspace/dataset/audits/supply-audit-fast-v1-20260901/supply-audit.json`

SHA-256:
`74de45bdf3438395f74f6c492f11017e6c0be6b76e0f08ec73e88a0b77169230`

These are historical v1 pre-EOS supply figures, not current packed totals. The
exact historical nine-cell table is in
[fast-generation-v2.md](../data/fast-generation-v2.md).

## Current decisions

1. Keep the v1 root frozen; v2 was built beside it.
2. The completed top-up changed only other-code acquisition. The versioned
   `configs/data_quotas_other_code_topup_v2.json` target is 35B cumulative raw
   other-code tokens; every Python and English target is unchanged.
3. After collection closed, preprocessing handled only newly added archives
   while using that same v2 quota config and reusing only immutable hard-linked
   v1 report and fingerprint shards.
4. Curation used a full fresh v2 inventory, quality/benchmark pass, global
   exact/normalized canonicalization, and leakage-safe group/split build. Do
   not reuse or patch the v1 curation database: new inputs can change global
   canonical winners.
5. Fuzzy English near-deduplication and a second code MinHash/LSH pass are
   deferred for this baseline. Retain this as a later ablation if results or
   audits justify it.
6. Exact quota selection was skipped in curation. Selection-v7 published all
   eligible canonical full documents as per-archive keep bitmaps. Treat any
   abandoned v1 `selected` rows as non-authoritative.
7. Enforce 40/40/20 and final token caps only in deterministic packed order v4.
   The first train order is one pass through 12,836,736 unique references:
   66,858 complete 192-row updates and exactly 52,579,270,656 input positions,
   with no replacement or repeats. Unselected packed rows remain available for
   later experiments. Validation/test use the largest feasible balanced
   whole-row caps no greater than 0.5B rather than downloading data solely to
   fill held-out quotas.

## Immediate operation

Do not restart any collection, preprocessing, curation, selection, cache, or
packing job. Work only from the authenticated packed restore on pod-local NVMe.
The next operation is the fail-closed six-H100 hardware and geometry
qualification for the fixed 192-row optimizer batch. A candidate is acceptable
only if memory, numerical, throughput, loader, and distributed checks pass with
the frozen packed identities. Any mismatch stops finalization.

## Remaining sequence

Advance one authenticated boundary at a time:

1. Qualify the six-H100 hardware and benchmark fixed-192-row candidates
   `(global microbatch, accumulation) = (6,32), (12,16), (24,8)`.
2. Freeze the fastest safe accepted physical geometry. Do not change the
   192-row statistical optimizer batch to make a failing candidate pass.
3. Finalize deterministic train, validation, and test order manifests from the
   existing packed rows. Require exact mixture/accounting, no replacement,
   unique selected row references, exact distributed rank partitioning, and
   the frozen 52,579,270,656-position train decision.
4. Publish and checksum the top-level corpus manifest binding orders, all nine
   packed manifests, tokenizer, source authorities, and geometry.
5. Run remaining real-data loader/attention validation and full 1.3B
   allocation, checkpoint/restart, validation, and multi-GPU soak gates.
6. Build the immutable run authority and only then launch the one-pass
   52.58B-token trajectory.

The six-H100 tiny-model overfit gate on six real packed 4,096-token rows has
passed for 1,000 steps. This validates a bounded real-data path but does not
waive the pending full-model geometry, memory, soak, validation, or resume
gates.

## Six-GPU training boundary

The intended training topology is the restored six-H100 RunPod pod with one
NCCL process per GPU (`torchrun --standalone --nproc-per-node=6`). Keep the
portable S3 packed backup, audits, and checkpoints durable; use the verified
pod-local NVMe restore for the hot path. Hardware qualification must still
prove that the implemented replicated-DDP path fits. The effective optimizer
batch is fixed initially at 192 rows; the six-GPU grid selects only its
physical microbatch/accumulation/compile geometry
before order v4 is frozen. A failed DDP gate requires stopping and
re-engineering; FSDP is not an automatic fallback in the accepted path.
