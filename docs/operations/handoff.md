# Experiment handoff

This is the short authority for current server state. The complete v2 design,
exact supply table, target calculation, and staged commands are in
[fast-generation-v2.md](../data/fast-generation-v2.md). Historical incidents
and measurements remain in [experiment-log.md](../experiment/experiment-log.md).

## Storage boundaries

| Purpose | Server root | State |
|---|---|---|
| Frozen pre-training generation v1 | `/workspace/dataset` | Read-only; do not resume or mutate jobs here |
| Pre-training top-up generation v2 | `/workspace/dataset-other-code-topup-v2` | Collector and incremental preprocessor active |
| SFT/RL source and derived artifacts | `/workspace/posttraining-data` | Quarantined from pre-training |

The selective v2 clone hard-links immutable v1 raw archives, tokenizer files,
manifests, collector state, quota records, and completed preprocess reports and
fingerprints. Closure/status files are copied to new inodes. Logs, locks,
temporary files, telemetry SQLite, curation, and packed outputs are excluded.
Never edit a hard-linked existing file in place; append new immutable files or
atomically replace only the independently copied v2 control files.

## Last verified server state — 2026-09-01 06:40:36 UTC

- The v2 clone completed successfully at 06:30:51 UTC and atomically published
  its final root. Manifest SHA-256 is
  `815c6256f0354f1b6a6cc524d96e745331c68afd02f3e72b19bb2d66ed2b3de9`;
  it records 21,181 hard links / 72,337,391,686 bytes and five copied controls.
- The other-code collector was relaunched at 06:36:50 UTC in tmux
  `stack-v3-topup-v2` using `/opt/coding-model-data-venv`. All eight workers are
  active. Committed other-code tokens advanced from 25,952,231,562 to
  26,066,335,409 while Python remained exactly 25,770,142,666. The 06:32 attempt
  failed on missing dependencies before touching data.
- The dedicated environment's 20 focused collector/clone tests passed. Its
  freeze is
  `/workspace/dataset-other-code-topup-v2/logs/data-environment.freeze.txt`,
  SHA-256
  `c84c69f333754bfbd97b3ec851ec132916466a06424043349ae273429ef81bfb`.
- Incremental archive audit/fingerprinting is active at low priority in tmux
  `preprocess-topup-v2-live`. The first new report, for other-code archive 550,
  covers 27,966 documents / 28,085,641 tokens with zero benchmark hits.
- No curator is active. The stopped first-generation local database remains at
  `/local/curation/selection-fast-local-v2/curation.sqlite3`, with its WAL and
  SHM sidecars retained. Its canonical rows and leakage-safe groups are useful
  as audited evidence, but partial exact-quota selection is not a corpus
  authority. A manifestless/incomplete NFS snapshot is not authority.
- No token-ID materialization is active or complete.
- Raw OpenCodeInstruct remains under `/workspace/posttraining-data`; it does not
  enter either pre-training generation.

## Completed supply audit

The read-only WAL-aware audit completed successfully in 255.2 seconds. It
proved 49,461,115 eligible canonical documents, 14,968,335 leakage-safe groups,
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

The exact nine-cell split/domain table is in
[fast-generation-v2.md](../data/fast-generation-v2.md).

## Current decisions

1. Keep the v1 root frozen and build v2 beside it.
2. Resume only other-code acquisition. The versioned
   `configs/data_quotas_other_code_topup_v2.json` target is 35B cumulative raw
   other-code tokens; every Python and English target is unchanged.
3. After collection closes, preprocess only newly added archives while using
   that same v2 quota config. Reuse only the immutable hard-linked v1 report
   and fingerprint shards.
4. Run a full fresh v2 inventory, quality/benchmark pass, global
   exact/normalized canonicalization, and leakage-safe group/split build. Do
   not reuse or patch the v1 curation database: new inputs can change global
   canonical winners.
5. Skip fuzzy English near-deduplication and a second code MinHash/LSH pass for
   this baseline. Retain this as a later ablation if results or audits justify
   it.
6. Skip exact quota selection in curation. Publish all eligible canonical full
   documents as per-archive keep bitmaps. Treat any abandoned v1 `selected`
   rows as non-authoritative.
7. Enforce 40/40/20 and the final token caps only in deterministic packed order
   v4. Validation/test may use the largest feasible balanced whole-row cap no
   greater than 0.5B rather than downloading data solely to fill held-out
   quotas.

## Immediate operation

Only monitor the active collector until checkpoint recovery produces stable
forward progress and confirms Python is skipped. Do not start a second
collector or preprocessing concurrently:

```bash
tmux has-session -t stack-v3-topup-v2
pgrep -af 'collect_stack_v3_parallel.py|collect_stack_v3.py'
tail -n 50 /workspace/dataset-other-code-topup-v2/logs/collector-topup-v2.log
/opt/coding-model-data-venv/bin/python \
  /workspace/0-coding-llm/scripts/quota_tracker.py \
  --root /workspace/dataset-other-code-topup-v2 \
  --config /workspace/0-coding-llm/configs/data_quotas_other_code_topup_v2.json \
  status --phase collection --json
df -h /workspace
```

The completed clone remains independently verifiable:

```bash
DATA_V2=/workspace/dataset-other-code-topup-v2
test -f "$DATA_V2/CLONE_MANIFEST.json"
(cd "$DATA_V2" && sha256sum -c CLONE_MANIFEST.sha256)
test "$(sha256sum "$DATA_V2/CLONE_MANIFEST.json" | cut -d' ' -f1)" = \
  815c6256f0354f1b6a6cc524d96e745331c68afd02f3e72b19bb2d66ed2b3de9
jq -e '.complete == true and
       .kind == "selective-hardlink-dataset-generation-clone" and
       .inventory.hardlinked_files == 21181 and
       .inventory.hardlinked_bytes == 72337391686 and
       .inventory.copied_control_files == 5 and
       .other_code_topup_plan.target_cumulative_raw_tokens == 35000000000' \
  "$DATA_V2/CLONE_MANIFEST.json" >/dev/null
```

## Remaining sequence

These are separate, ordered launches; none is automatic:

1. First qualify the already-running `stack-v3-topup-v2`: verify Python is
   skipped and only other-code totals/files advance, then let it reach 35B.
2. After the v2 collection completion marker is atomically republished, run
   `preprocess-topup-v2`. Require exact closed-collection coverage and zero
   errors.
3. Run a new full curation generation through canonicalization and leakage-safe
   groups. Freeze and authenticate its durable snapshot.
4. Publish all eligible canonical documents with the qualified v7 keep-bitmap
   publisher.
5. Run the v7-aware materializer, certify decoded samples and attention
   boundaries, then build packed order v4 after the six-GPU geometry smoke.
6. Only after packed/order certification run CUDA/FlexAttention, BF16,
   single-GPU overfit, six-rank NCCL, throughput, and exact-resume gates.

The production publisher/materializer v7 work is still being qualified. Do not
start curation publication or tokenization based only on a source checkout that
contains draft code.

## Six-GPU training boundary

The intended training topology is one six-GPU RunPod pod with one NCCL process
per GPU (`torchrun --standalone --nproc-per-node=6`). Copy the finalized,
verified packed corpus to pod-local NVMe for throughput when capacity permits;
keep network-volume corpus publications, audits, and checkpoints durable. GPU
type/VRAM and the smoke test determine whether replicated DDP fits or FSDP is
required. Do not freeze order v4 geometry before that smoke.
