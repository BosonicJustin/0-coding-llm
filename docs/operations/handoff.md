# Experiment handoff

This is the short authority for current server state. The complete v2 design,
exact supply table, target calculation, and staged commands are in
[fast-generation-v2.md](../data/fast-generation-v2.md). Historical incidents
and measurements remain in [experiment-log.md](../experiment/experiment-log.md).

## Storage boundaries

| Purpose | Server root | State |
|---|---|---|
| Frozen pre-training generation v1 | `/workspace/dataset` | Read-only; do not resume or mutate jobs here |
| Pre-training top-up generation v2 | `/workspace/dataset-other-code-topup-v2` | Collection/preprocessing closed; curation and raw-token caching active |
| SFT/RL source and derived artifacts | `/workspace/posttraining-data` | Quarantined from pre-training |

The selective v2 clone hard-links immutable v1 raw archives, tokenizer files,
manifests, collector state, quota records, and completed preprocess reports and
fingerprints. Closure/status files are copied to new inodes. Logs, locks,
temporary files, telemetry SQLite, curation, and packed outputs are excluded.
Never edit a hard-linked existing file in place; append new immutable files or
atomically replace only the independently copied v2 control files.

## Last verified server state — 2026-09-01 11:32:23 UTC

- Collection and streaming preprocessing are complete. The closed v2 corpus
  contains 4,568 raw archives, 56,502,609 documents, and 73,992,914,797 exact
  raw tokens. All 4,568 reports and fingerprints are present with zero archive
  error records.
- Exact raw totals are 25,770,142,666 Python tokens, 35,363,570,483 other-code
  tokens, 10,287,360,535 FineWeb-Edu tokens, and 2,571,841,113 Wikipedia
  tokens.
- Curation is active in tmux `all-eligible-curation-v2`. The complete
  56,502,609-document inventory and four bulk indexes finished at 11:18 UTC.
  The process remains CPU-active with a fresh checkpoint at phase
  `inventory_complete`; terminal canonicalization and the immutable final
  snapshot have not yet been published.
- The curation authority uses local WAL state at
  `/local/curation/all-eligible-source-v2` and durable output under
  `/workspace/dataset-other-code-topup-v2/curated/all-eligible-source-v2`.
  Do not copy, replace, or resume it with changed code or flags.
- Raw token caching is active in tmux `raw-token-cache-v2-16w` with 16 workers.
  It had authenticated 3,866 archive caches containing 68,955,111,064 tokens,
  or 93.19% of the closed raw-token total. Completed archive caches are
  immutable and resumable; in-flight staging is non-authoritative.
- The latest capacity check showed 205 GB free on local disk and 616 GB free on
  the network volume. Cgroup counters showed zero OOM and zero OOM-kill events.
- The server checkout remains at the source identity frozen by the running
  curator. Do not pull the newer local/GitHub `main` while curation is active.
  Sync only after the final curation result and snapshot authenticate.
- No selection-v7 publication, closed-world cache inventory, packed corpus, or
  production order has started for v2.
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

Monitor the two existing jobs without starting duplicates or changing their
source identities:

```bash
tmux has-session -t all-eligible-curation-v2
tmux has-session -t raw-token-cache-v2-16w
pgrep -af 'curate_corpus.py|cache_raw_tokens.py'
cat /local/curation/all-eligible-source-v2/CHECKPOINT.json
df -h /local /workspace
cat /sys/fs/cgroup/memory.events
```

Do not infer cache completion from aggregate token totals alone. Completion
requires exact per-archive source/report/fingerprint/tokenizer authentication,
no in-flight staging directories, and an unlocked builder lock. The completed
clone remains independently verifiable:

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

Advance one authenticated boundary at a time:

1. Let curation finish canonicalization, leakage-safe grouping, integrity
   checks, and its one immutable final SQLite snapshot. Verify the guarded
   launcher result and snapshot sidecar before using it.
2. Let raw token caching reach exact closed-world completion. Do not stop it
   merely because aggregate token counts appear close to the target.
3. After curation has exited, sync the server to the reviewed clean Git commit.
   Publish selection-v7 from the exact final snapshot.
4. Run the nine-cell packed-supply gate before any large packing write. The
   historical Python validation supply is 133 rows below the nominal 0.5B
   target, so freeze the largest feasible balanced held-out cap if the v2 gate
   confirms that shortfall. Do not change split seed or curation quota config.
5. After token caching exits and its lock is free, publish the authenticated
   closed-world cache inventory.
6. Materialize all nine packed streams with the raw-cache adapter and stop after
   packing. Validate packed shards and document indexes; a journal alone is not
   completion evidence.
7. On the final six-GPU pod, qualify hardware and run the measured geometry
   grid. Finalize deterministic orders only from the accepted measured geometry.
8. Copy to GPU-local NVMe when used for training, requalify the copied corpus,
   run full data/attention/overfit/soak/resume gates, build run authority, and
   only then launch the multi-day pre-training run.

The selection publisher, supply gate, cache-backed materializer, final corpus
qualifier, six-GPU pod qualifier, and geometry evidence producer are implemented
and tested locally. The post-curation orchestrator and unified progress reporter
remain under adversarial review and must not be deployed until committed.

## Six-GPU training boundary

The intended training topology is one six-GPU RunPod pod with one NCCL process
per GPU (`torchrun --standalone --nproc-per-node=6`). Copy the finalized,
verified packed corpus to pod-local NVMe for throughput when capacity permits;
keep network-volume corpus publications, audits, and checkpoints durable. GPU
type/VRAM and the smoke test determine whether replicated DDP fits or FSDP is
required. Do not freeze order v4 geometry before that smoke.
