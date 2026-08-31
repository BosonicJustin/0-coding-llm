# 1.3B coding-model pre-training experiment log

This document is the reproducibility record for the data and training
experiment. Times are recorded in UTC unless stated otherwise. Commands shown
here are the commands actually used or the canonical commands encoded in the
checked-in launch scripts.

## Objective and evaluation boundary

The objective is to pre-train a 1.3B-parameter decoder-only coding model from
scratch and later post-train it, targeting at least 50% on MBPP. MBPP is held
out from pre-training and post-training data. It is treated only as an eventual
evaluation set.

The nominal final token allocation is 53.58B StarCoder2-tokenizer tokens:

| Split | Python | Other code | English | Total |
|---|---:|---:|---:|---:|
| Train | 21.032B | 21.032B | 10.516B | 52.580B |
| Validation | 0.200B | 0.200B | 0.100B | 0.500B |
| Test | 0.200B | 0.200B | 0.100B | 0.500B |
| Total | 21.432B | 21.432B | 10.716B | 53.580B |

Thus code is 80% of the final corpus, split equally between Python and other
programming languages; English is 20%. Code acquisition targets 25.7184B
tokens per code bucket, providing 20% selection/deduplication headroom over the
21.432B final requirement per bucket. English acquisition is collected from
FineWeb-Edu and English Wikipedia in separate resumable jobs described below.

For the training run, this budget means processed model input slots after
packing, including inserted EOS tokens. The duplicated `T + 1` lookahead slot
stored with each row is not counted again. At context 4,096, the closest
not-over-budget train allocation is 12,836,914 rows, or 52,579,999,744 input
tokens (256 below the nominal target): 5,134,766 Python rows, 5,134,766
other-code rows, and 2,567,382 English rows. Cross-document labels are masked,
so valid supervised-target totals and their realized domain mix are tracked
separately rather than silently called the Chinchilla input-token budget.

## Immutable source and tokenizer identities

- Code source: `HuggingFaceCode/stack-v3-train`
- Resolved Stack v3 revision: `df4b205fbba4cc1c2fd1f205b10d66f730798bb9`
- Source shards: 8,192
- Deterministically ordered shard-list SHA-256:
  `d080732e8c9f2dbd3877c498e6e7a5dc96076c69e5e9ab95f7b9c51474d21408`
- Shard ordering seed: 1,307
- Tokenizer: `bigcode/starcoder2-tokenizer`
- Resolved tokenizer revision: `9cfe60e28fd01cc1391ecd2146a34cda7534efeb`
- Vocabulary: 49,152 tokens; end-of-text/BOS/EOS ID 0
- Canonical MBPP source: [google-research MBPP JSONL](https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl)
- Downloaded canonical MBPP JSONL: 974 rows, 563,743 bytes
- Canonical MBPP JSONL SHA-256:
  `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`

Raw MBPP examples are not bundled with or written into the training corpus.
The collector ships non-reversible SHA-256 fingerprints plus extracted MBPP
function-name triggers; it does not ship benchmark prompts, tests, or solutions.

## Acquisition policy

The Stack v3 source uses Parquet internally, but source shards are streamed and
not retained. Accepted source code is saved byte-for-byte as UTF-8 members in
lossless `.tar.zst` archives. Each archive contains an `_manifest.jsonl` with
provenance and exact token counts. No token IDs or Parquet output are stored.

The language policy is fail-closed:

- `Python` is the only language admitted to the Python bucket.
- Seventy-eight explicitly named programming languages are admitted to the
  other-code bucket.
- Markup, configuration, notebooks, prose, and unknown languages are rejected.
- Vendored files are rejected.
- `permissive` and `no_license` Stack v3 rows are accepted for this private,
  non-commercial research experiment.
- Once either bucket reaches 25.7184B exact tokens, that bucket is skipped while
  acquisition continues for the other bucket.
- Free-space safety threshold: 50 GB.

## Evaluation contamination controls

Repository and path markers reject MBPP, EvalPlus, MultiPL-E, MBXP, MXEval,
and HumanEval material. MBPP additionally has content-level acquisition
filtering generated from the canonical 974-row JSONL:

- exact normalized solution hashes;
- normalized adjacent-line solution hashes for embedded solutions;
- canonical problem and assertion-line hashes;
- canonical MBPP function names used only to strengthen embedded-code matches;
- explicit MBPP text markers and canonical JSON-row shape.

The generated denylist contains 967 unique exact-code hashes, 2,387 code-pair
hashes, 3,422 strong problem/test-line hashes, and 951 function names. Its file
SHA-256 is
`62ac6037a02580e9508b7f7bb5ed761d56418d81a6bad305f46da85ff7b3b3c6`.
The collector records this hash in its source manifest and every checkpoint and
refuses to resume if it differs. Python archives can be independently scanned
with `scripts/audit_mbpp.py`.

This is a strong MBPP-specific acquisition guard, not a proof that all semantic
or transformed benchmark variants are absent. A broader final-corpus
decontamination pass remains required before training.

## Infrastructure and software

On 2026-08-29, a RunPod CPU pod was attached to a persistent NFSv4 network
volume mounted at `/workspace`. At setup time the volume reported 932 GiB free.
The pod had 32 CPUs and approximately 1 TiB RAM. The container's default Python
3.8 was not used; a Python 3.13.5 environment was created at
`/opt/coding-model-venv`. Persistent artifacts live under `/workspace` while
the Hugging Face stream cache lives under disposable `/tmp` storage.

Primary installed data packages at setup were `datasets==4.8.5`,
`tokenizers==0.23.1`, `transformers==5.16.1`,
`huggingface_hub==1.29.0`, `hf-xet==1.6.0`, `pyarrow==25.0.1`, and
`zstandard==0.25.0`. The complete package list can be reproduced with
`/opt/coding-model-venv/bin/pip freeze`.

Scripts were transferred with `runpodctl send`/`runpodctl receive`. A first tar
transfer included macOS AppleDouble metadata and requested local ownership that
the NFS mount would not accept. That incomplete project directory was removed
and replaced with a checksum-verified, portable archive using
`tar --no-same-owner`. Dataset storage was not affected by that transfer issue.

## Verification history

### Initial pilot

The first end-to-end pilot streamed 10 repositories into a separate
`/workspace/dataset-pilot` root. It produced valid Python and other-code
`.tar.zst` archives with internal manifests and an exact resume checkpoint.
Results were 3,859 Python tokens and 61,748 other-code tokens. GNU tar listed
both archives successfully. Seven original unit tests passed remotely.

### First production start and MBPP guard restart

An initial production collector was started in detached `tmux` at
2026-08-29 10:21:16 UTC. Before sustained collection, it was stopped to add the
stronger MBPP content denylist requested for the evaluation boundary. The stop
created one small checkpoint and two small archives, but the shell reported
status 134 after the terminal interrupt. These pre-guard production shards are
not eligible for training and are removed before the guarded restart. This
event is retained here as part of the experiment history.

The content-guard implementation added three tests, bringing the local suite to
10 passing tests. The same suite is rerun on the pod before production restart.

### Guarded pilot and rejection-ledger restart

A second pilot used the full content guard for 100 repositories. It retained
103,496 Python tokens and 400,784 other-code tokens. An independent scan checked
107 archived Python files and found zero MBPP fingerprints. Both manifest v3
and checkpoint v3 pinned denylist SHA-256
`62ac6037a02580e9508b7f7bb5ed761d56418d81a6bad305f46da85ff7b3b3c6`.

The first guarded production observation ran for 7,175 repositories and reached
8,364,039 Python tokens plus 43,976,763 other-code tokens at 1.263 retained
MB/s. It rejected six Python files by MBPP content fingerprints. `SIGTERM`
finished the active repository, closed the archives, wrote a checkpoint, and
exited with status 0. Because the first version counted content rejections but
did not retain their provenance/reason, this small run is discarded and
restarted from repository zero. Collector v4 adds an idempotent
`logs/benchmark_rejections.jsonl` ledger with rejection reason, source cursor,
repository/commit/file/content IDs, and no source content. This change added an
eleventh unit test.

### Stable collector v4 baseline

The final ledger-enabled production run started at 2026-08-29 10:34:15 UTC.
Through repository 12,951 it retained 13,559,698 Python tokens and 101,407,612
other-code tokens. Ten content rejections were observed; all were classified as
`mbpp-embedded-code`. The ledger contained ten unique rejection IDs. At the
11,000-repository observation, retained throughput was 1.172 MB/s, the process
used about 2.3 GB RAM and 95% of one CPU, disposable `/tmp` storage remained
11% used with 4.5 GB free, and the network volume still reported 932 GiB free.

A final durability test sent `SIGTERM` at repository 12,951. Collector v4
finished the active repository, wrote checkpoint v3 and finalized both archive
parts, then exited with status 0. The checkpoint pinned collector version 4,
the exact source cursor, and the MBPP denylist hash. The launcher was restarted
at 2026-08-29 10:39:26 UTC and explicitly reported:

```text
Resuming at source shard 0, row 12,951 after 12,951 repositories
```

After 1,000 additional repositories, totals advanced to 14,046,236 Python and
110,936,280 other-code tokens. The rejection ledger remained at ten unique
rows even though one already-known rejected item was encountered again,
demonstrating idempotent rejection logging during replay/duplicate source data.
The `tmux` session, collector process, archive growth, logs, disk safety margin,
and exact-token accounting were healthy. The run was left active.

## Production operation

The canonical detached start command is:

```bash
tmux new-session -d -s stack-v3 \
  /workspace/coding_model_from_scratch/scripts/run_download.sh
```

The launcher appends stdout and stderr to:

```text
/workspace/dataset/logs/collector.log
```

Content-level benchmark exclusions are recorded without source text in:

```text
/workspace/dataset/logs/benchmark_rejections.jsonl
```

Each JSON progress record contains the current source shard and row, total
repositories consumed, exact Python/other-code tokens and targets, retained
throughput, free disk space, and per-reason skip counters. Records are emitted
every 1,000 repositories and at every checkpoint. Useful monitoring commands:

```bash
tail -f /workspace/dataset/logs/collector.log
tmux attach -t stack-v3
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/quota_tracker.py \
  --root /workspace/dataset status --phase collection
df -h /workspace
```

Checkpoints are triggered after 1 GB of accepted source, 100,000 repositories,
a bucket reaching its target, a termination request, pilot completion, or the
free-space threshold. Archives are closed and fsynced before the cursor is
committed. The final completion marker is
`/workspace/dataset/state/COLLECTION_COMPLETE.json`.

## Ongoing record

Subsequent material changes, restarts, audits, and collection milestones should
be appended here with timestamps, exact commands or script revisions, token
totals, disk use, and the reason for each decision.

### FineWeb-Edu collector implementation

On 2026-08-29, the English acquisition target was added as 12.8592B exact
StarCoder2 tokens: 20% headroom over the 10.716B final English requirement. It
was divided into 10.28736B FineWeb-Edu tokens and 2.57184B Wikipedia tokens to
preserve the planned 80/20 acquisition mixture. A
direct Hugging Face API inspection resolved `HuggingFaceFW/fineweb-edu` to
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Its `sample-100BT`
configuration contained 140 source Parquet shards and exposed text plus stable
document/Common Crawl provenance, English-language confidence, and education
scores. The card reported English and `odc-by`.

`scripts/collect_fineweb_edu.py` was added as a source-specific raw collector.
It pins and hash-orders those source files, streams one file at a time, retains
raw text in manifested `.tar.zst` archives, counts exact StarCoder2 tokens,
applies English/language-score and MBPP contamination guards, and checkpoints
an exact `(source shard, row)` cursor. It is designed to stop automatically on
the exact collection quota, free-space threshold, pilot limit, or termination
signal. The English MBPP guard and FineWeb-Edu archive/checkpoint/rejection
tests increased the complete test suite from 11 to 15 tests.

### Parallel English production collectors

The FineWeb-Edu pilot retained 32,557 exact tokens from 25 documents and then
resumed at the exact next source cursor. Its production job started in detached
`tmux` session `fineweb-edu` at 2026-08-29 11:58:00 UTC. The source manifest
pinned 140 deterministically ordered shards with shard-list SHA-256
`ddaf159bb2ba25c606b57916e68501375a26cffe54dfc014c658e716cfc7a8e3`.
At the stability observation it had processed 152,000 source rows, accepted
151,999 documents, retained 180,536,452 live exact tokens, and was sustaining
approximately 1.366 retained MB/s. One MBPP fingerprint match was rejected and
recorded without its text. The code collector continued progressing in its own
session throughout.

The second English source was implemented from `wikimedia/wikipedia`, config
`20231101.en`. The Hugging Face API resolved revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; the dataset card reports
`cc-by-sa-3.0` and `gfdl`. Its 41 Parquet source shards are streamed rather than
stored, and their deterministic ordered-list SHA-256 is
`e48c76abb08483faebe621c2fbd30a216e04ddfbee253a88b74b792e04081942`.
Accepted cleaned article text is retained byte-for-byte in
`raw/english/wikipedia` with article ID, title, URL, byte size, and exact token
count in each archive manifest. The source-specific collector has its own
2.57184B-token quota, source manifest, raw directory, checkpoints, completion
marker, rejection ledger, log, and 50 GB free-space stop.

An isolated Wikipedia pilot retained 39,702 exact tokens from 25 articles,
checkpointed, then explicitly resumed at row 25 and advanced to row 30 and
49,818 cumulative exact tokens. The full local and remote suites passed all 18
tests. Production started in detached `tmux` session `wikipedia` at
2026-08-29 12:06:17 UTC. It reached 7,768,892 live exact tokens from 7,000
articles in its first stability observation at approximately 1.149 retained
MB/s. At that point all three sessions (`stack-v3`, `fineweb-edu`, and
`wikipedia`) and their collector processes were active, while the network
volume reported approximately 930 GiB available.

Wikipedia's first production checkpoint finalized
`raw/english/wikipedia/part-000000.tar.zst` at approximately 143 MB and
atomically recorded 114,377,722 exact tokens from 100,000 articles. The quota
tracker then reported Wikipedia at 4.45% and aggregate English acquisition at
355,442,973 committed tokens (2.76%). The collector immediately opened the next
pending archive and continued running.

### Streaming integrity and dedup preprocessing

On 2026-08-29, incremental preprocessing was added so safe downstream work can
overlap acquisition. `scripts/preprocess_raw_stream.py` admits only visible
`part-NNNNNN.tar.zst` archives that also have a matching atomic quota record;
collector-owned hidden `.part-*` files are never opened. Raw content is treated
as immutable. Every admitted archive is fully decompressed and validated
against its tar manifest and quota totals, then receives archive/content hashes,
normalized hashes, bottom-k near-duplicate sketches, an independent MBPP scan,
and non-destructive quality metrics. Fingerprint outputs contain metadata and
hashes but no source text.

Exact and normalized fingerprints are ingested transactionally into a
rebuildable SQLite index, one completed fingerprint shard per transaction.
Stable document and archive identities make retries idempotent. Near-duplicate
sketch computation is performed now; final LSH candidate clustering and
representative selection remain deterministic later phases. No quality flag or
duplicate currently deletes a raw document. The pinned fingerprint-policy
SHA-256 is
`a340dc1f71a915bfec0a76509099a769eec886b2eb94c7ed31bb453f56d3d032`.

The local and remote suites increased from 18 to 22 passing tests. A real-data
pilot audited Python archive 0 in 10.634 seconds: 12,198 documents, 45,760,636
clean bytes, and 13,559,698 exact tokens. It reproduced the quota totals, found
zero archive/MBPP errors, emitted the expected fingerprint shard, passed SQLite
`PRAGMA integrity_check`, and indexed all 12,198 documents.

Production started at 2026-08-29 12:54:30 UTC in detached, low-priority `tmux`
session `preprocess` with four spawned analysis workers. A graceful `SIGTERM`
after a completed archive was used at 13:03 UTC to deploy non-blocking atomic
status snapshots; the job exited status zero and resumed in a new `preprocess`
session without reprocessing completed archives. At the first post-restart
snapshot it had audited and indexed 504,541,102 tokens from 491,574 documents
across 19 finalized archives, covering 9.099% of then-finalized source tokens.
It reported zero archive errors, 9,229 exact duplicate documents, and 9,234
normalized duplicate documents. Python preprocessing had caught up to 100% of
its then-finalized archives while the watcher continued through other-code and
English backlogs. The acquisition collectors remained active throughout.

### Parallel Stack migration and preprocessing acceleration

On 2026-08-29, local profiling showed that the CPU pod exposed 32 usable CPUs
while each sequential collector used approximately one core. Exact StarCoder2
tokenizer benchmarks over 12,000 synthetic code documents produced identical
2,989,761-token totals and improved from 6,663 documents/s with individual
`encode` calls to 19,160 documents/s with ordered batches of 32 (2.88x).
Preprocessing benchmarks over 20,000 documents improved from 2.410 seconds with
four per-document workers to 0.857 seconds with eight workers and 64-document
tasks (2.81x); the resulting fingerprint shard SHA-256 was identical.

`collect_stack_v3_parallel.py` was introduced with eight disjoint source-shard
workers, ordered token batches of 64, independent worker checkpoints, caches,
and rejection ledgers, collision-free interleaved archive indices, and a
supervisor that reports committed aggregate quotas. The legacy collector
received `SIGTERM` at 2026-08-29 19:07 UTC and exited status zero after
committing shard 52, row 8,024. Its final live totals were 1,103,731,228 Python
tokens and 12,191,623,605 other-code tokens. The immutable migration plan
imported exactly that cursor, retained the pinned Stack revision and shard-list
hash, and assigned workers to initial shards 52 through 59. New archive
numbering began at index 60 in each code bucket. Initial worker checkpoints and
quota records were inspected to verify disjoint indices and cursor continuity.

Preprocessing task IPC was changed from one process-pool call per document to
ordered 64-document calls with eight low-priority workers. While gracefully
stopping the legacy process, a separate scaling problem was identified: each
status snapshot performed full-table counts and sums over the 2.3 GB SQLite
index. Status accounting was changed to use the small transactionally
maintained per-archive summary table. Existing rows receive token summaries
from immutable fingerprint reports during an in-place schema migration. The
fingerprint policy hash, raw archives, completed reports, fingerprint shards,
and existing dedup records remain unchanged. The expanded local suite passes
27 tests, including legacy-cursor migration, no-overlap assignments,
archive-index uniqueness, idempotent worker recovery, old-database summary
migration, and byte-identical batched fingerprints.

The accelerated pipeline was deployed to the network volume on 2026-08-29.
The eight-worker Stack collector was first observed sustaining approximately
1.3--1.6 retained MB/s per worker, compared with approximately 1.3 MB/s for the
former single collector. A second graceful restart deployed a five-second
supervisor quota-poll interval, reducing repeated network-volume metadata
reads. Seven workers checkpointed normally. Worker 3 remained inside one
exceptionally large repository for about ten minutes while the other workers
were idle, so it was force-stopped only after its previous immutable checkpoint
was verified. Its two uncommitted temporary streams (22,846,127 compressed
other-code bytes and an empty Python stream) were preserved rather than deleted
under
`staging/interrupted_partials/20260829T194528Z/worker-03`. No finalized archive,
quota record, or checkpoint was modified. The collector restarted at 19:45:45
UTC with all eight workers active; worker 3 resumed from shard 71, row 18,517
and advanced another 3,000 repositories during the first minute.

The preprocessor deployment required terminating an old process that had
already committed Python archive 18 but remained blocked in the legacy
full-table status scan. There was no SQLite rollback journal, and the log
confirmed the archive transaction before termination. The upgraded job started
at 19:36:06 UTC with eight workers and 64-document tasks. Its first constant-time
status snapshots reported zero archive errors; within roughly seven minutes it
advanced from 56 previously ingested archives to 88, from approximately 4.2M
indexed documents to 4,246,196, and to 5,636,705,142 audited tokens. FineWeb-Edu
and Stack acquisition remained in independent `tmux` sessions throughout.

### Native PyTorch packed loader and model contract

On 2026-08-30, the training-side data contract was implemented independently
of the unfinished final corpus selection. `pretrain/data.py` now writes and
reads immutable uint16 packed shards. A length-`T` training row stores `T + 1`
tokens so the final input has a real look-ahead target, plus a little-endian
segment-start bitset. The bitset, not the EOS token ID, is authoritative. The
collator reconstructs reset positions and document IDs, masks only labels that
would cross segment boundaries, and reports the exact number of supervised
tokens. Literal EOS IDs inside document content therefore do not silently
create boundaries.

Train-domain manifests remain separate for Python, other code, and English.
The order builder checks the 40/40/20 fixed-row allocation and writes a complete
uint64 PCG64 permutation without replacement. At the planned 4,096-token
context, the approximately 12.84M row references require only about 103 MB, so
a fully materialized exact shuffle is simpler than a streaming shuffle buffer.
The order manifest pins the RNG, seed, NumPy version, source-manifest hashes,
domain encoding, row totals, and SHA-256. Validation uses one boolean seen array
per domain to prove every row occurs exactly once without sorting the order.

The distributed batch sampler slices one immutable global microbatch into disjoint
data-parallel rank ranges. Resume is defined by completed global microbatch
number rather than a mutable iterator cursor; prefetched but unconsumed rows are
discarded on restart. Shard readers use lazy per-worker mmaps and now expose
explicit close operations. A server benchmark revealed that persistent worker
mmaps can leave temporary `.nfs*` handles during immediate test-directory
deletion; the benchmark consequently performs explicit worker shutdown, mmap
close, garbage collection, and bounded cleanup retries. Training retains
persistent workers because production shards are immutable and are not deleted
during a run.

Order format v4 also freezes gradient accumulation and effective
optimizer-update rows. Its authoritative 52.58B accounting stops at the last
full optimizer update, not merely the last full microbatch, and records exact
consumed/dropped input and supervised tokens globally and per 40/40/20 domain.
Sampler state v2 and trainer checkpoint v2 bind that geometry and require every
resume cursor to land on an optimizer-update boundary.

`model.py` was replaced with a native PyTorch Llama implementation preserving
the exact 1,283,557,376-parameter architecture: 24 layers, 2,048 model width,
5,632 SwiGLU width, 16 query heads, four KV heads, 4,096 context, RMSNorm, RoPE,
and untied 49,152-token embeddings. CPU debugging uses an equivalent dense SDPA
document mask; CUDA training uses one FlexAttention block mask per batch, reused
by all layers. Tests prove that changing a packed document or another batch row
does not change unrelated logits, and that a loss on document B creates zero
input gradient through document A. A tiny configuration was copied parameter by
parameter into Hugging Face Llama and produced matching logits. The model
returns loss sum and supervised-token count so DDP can normalize gradients by
the global unmasked-token denominator rather than averaging biased per-rank
means.

The final source archive was transferred with `runpodctl`; local and remote
SHA-256 both equaled
`dae319e6d1642ad3c77bbec37a84486afc7928d89181f2f5e83fb0fff4f306c4`.
Critical collector, quota, denylist, and active preprocessing script hashes were
identical before overlay. The live project at
`/workspace/coding_model_from_scratch` then passed all 43 tests in a separate
PyTorch 2.13.0+cpu environment; the active data-preprocessing environment was
not modified.

A realistic network-volume benchmark built, checksummed, fully shuffled,
validated, and loaded 10,000 rows at context 4,096. Its payload was 87,070,000
bytes, representing 40,960,000 input tokens and 40,950,000 supervised tokens.
Eight spawned workers loaded all rows in 6.056 seconds, or 6,763,073 input
tokens/s, and cleanup completed with exit status zero. This validates CPU-side
format and filesystem behavior but does not replace the later local-NVMe and
CUDA end-to-end benchmark.

At the final post-deployment status check, acquisition remained complete and the
preprocessor was active in its original `tmux` session with zero errors. It had
audited 14,922,912,220 of 64,581,575,876 finalized tokens (23.107%), committed
439 archives and 12,226,197 documents, and had 3,906 finalized archives waiting.
The pinned StarCoder2 tokenizer on the volume reports vocabulary 49,152 and
shared BOS/EOS ID 0, matching the new packed-writer assumptions. No final real
training shard has been built yet: filtering, global deduplication,
decontamination, group-safe split selection, and exact-budget tokenization still
precede that step.

The data/model interface is deliberately reusable but not conflated across
phases. Future SFT shards will add prompt/assistant loss masks while reusing
tokens, positions, and document IDs. RL will use separate prompt and rollout
stores with per-request KV caches and reward metadata. Pretraining, SFT, and RL
artifacts remain independently versioned while sharing tokenizer and checkpoint
identities.

### Post-audit hardening and real-data compatibility smoke

A separate read-only audit then exercised edge cases that ordinary forward
tests did not cover. Packed and order formats advanced to version 2 before any
production shard existed. Tokenizer-manifest SHA-256 is now mandatory and
cross-checked across all domains and the order. Full validation streams the
payload to check vocabulary bounds, first/start padding bits, EOS before every
in-row document start, and independently recomputed supervised/boundary counts.
Tail accounting now distinguishes tokens unused as inputs from tokens absent
from storage, including the zero-row case. The order records and revalidates
both the 40/40/20 input mixture and the at-or-below 52.58B absolute budget,
while reporting the independently realized supervised-target mixture.

The same audit found and fixed deferred-initialization and loss-path issues.
RoPE caches are rebuilt after `meta`-device materialization; compiled
all-ignored batches remain finite; CUDA no longer performs position min/max
host synchronizations in every layer; and DDP accumulation is specified in
terms of one global supervised-token denominator for the whole optimizer
window. Labeled forwards omit logits by default. The pure-PyTorch reference
loss is split into 256-position activation-checkpointed chunks, so autograd
does not retain vocabulary-sized activation tensors; it recomputes one output
projection chunk at a time during backward. Regression tests verify gradient
parity with full loss and inspect saved tensors directly. A focused independent
re-audit reproduced all fixes and found no remaining regression in the CPU
reference path.

The hardened project archive had matching local/server SHA-256
`a00109a96820e2b8534aff0c6f60ae71eb5a113f4f274e4cbc28527f78356643`.
Before overlay, the active preprocessing script, launcher, quotas, and MBPP
denylist again matched byte-for-byte. The separate server training environment
then passed all 59 tests and 10 subtests on Linux.

`smoke_raw_to_training_data.py` performed a read-only integration check against
actual completed preprocessing reports and immutable raw archives. It sampled
six documents from Python archive 109, seven from other-code archive 109, and
eight from FineWeb-Edu archive 15; retokenized them with the pinned local
StarCoder2 tokenizer; wrote format-v2 shards at context 4,096; validated every
payload and order; and consumed all seven rows with two spawned PyTorch workers.
The result contained 28,672 input tokens and 28,654 supervised tokens. The
actual `TOKENIZER_MANIFEST.json` SHA-256 pinned into every smoke shard was
`56c718787dc6c05bc5868dc3f43e54a177d1d08a9e37306ff083b63fe7f6aadd`.
Disposable outputs were removed successfully.

The updated 10,000-row network-volume benchmark again built and validated an
87,070,000-byte payload, loaded all 40,960,000 input tokens with eight spawned
workers in 6.114 seconds (6,698,960 input tokens/s), and cleaned up. At the
latest status snapshot, preprocessing had reached 15,319,300,997 of
64,581,575,876 finalized tokens (23.721%), 441 archives, and 12,514,984
documents, with 3,904 archives waiting and zero errors. The original `tmux`
session and all eight analysis workers remained active; the volume had 870 GiB
free.

This smoke proves the current raw/archive/tokenizer boundary is compatible with
the new training format, but it is deliberately not the final corpus builder.
Global duplicate selection, quality/decontamination policy, group-safe splits,
the provenance side index, resumable production packing, and the real CUDA
FlexAttention forward/backward isolation test remain required gates.

### Source-aware deduplication policy correction

On 2026-08-30, the official Stack v3 Train documentation was rechecked. The
pinned `HuggingFaceCode/stack-v3-train` v3.1 corpus is already file-level,
cross-repository near-deduplicated. The final curation plan was corrected so
Python and other code never enter a second local MinHash/LSH pass. Exact and
normalized hashes remain a non-destructive residual/collector audit; any
residual collisions are handled deterministically before split assignment.
Benchmark decontamination and repository-safe split grouping remain mandatory
because neither follows from upstream deduplication.

FineWeb-Edu and Wikipedia still enter one local cross-source near-duplicate
pass. The source-aware routing and trusted Stack repository, commit, shard
count, and shard-list digest are pinned in `configs/curation_policy.json` and
validated by `scripts/curation_policy.py`. The preprocessing launcher runs this
validation before every future start or resume; a source mismatch fails before
the watcher opens any archive.

The running v1 fingerprint watcher was deliberately left unchanged. Its
already-pinned format computes compact sketches for every bucket; code sketches
will be retained as audit-only metadata. A local 80 MB real-code benchmark
measured 7.75 MB/s with sketches and 8.28 MB/s without them (about 7% faster in
the analysis kernel). End-to-end savings would be smaller after archive I/O and
SQLite work, so changing the policy hash mid-run was not justified. Avoiding
the later full code LSH pass is the material compute and elapsed-time saving.

### Fast postprocessing cutover

On 2026-08-30 the legacy preprocessor was stopped only after its active archive
and SQLite transaction committed. The cutover point contained 445 indexed
archives, 13,148,413 documents, and 16,030,046,311 tokens in the 7,293,890,560
byte dedup database. Its last committed archive/report/fingerprint identities
matched, and no rollback journal remained. Raw archives, completed reports,
fingerprints, and the database were preserved.

The old critical path coupled each raw-archive audit to synchronous document-by-
document updates against the large SQLite database on network storage. On the
representative `raw/other_code/part-000015.tar.zst` archive, raw analysis took
53.786 seconds, followed by several minutes of SQLite work and a transient
multi-gigabyte rollback journal. The replacement separates the work into two
restart-safe phases: a fast immutable fingerprint/report audit with indexing
deferred, followed by a batched index-only pass that never reopens raw archives.
It also uses pod-local scratch for rebuildable uncompressed metrics, hashes the
compressed fingerprint while writing it, batches document inserts and per-
archive hash-group merges, throttles global status scans, validates existing
report identities, and fsyncs parent directories around atomic publication.

Exact repeated benchmarks on that same 116,514,935-byte archive produced the
same fingerprint SHA-256 every time,
`5350e2c4ce709b37551370393f8d9d9edecd774bfb1bc7563aade1e89ef49dbf`:

| Workers | Raw elapsed | Full benchmark wall | Peak RSS |
| ---: | ---: | ---: | ---: |
| 8 | 54.926 s | 55.207 s | 251,444 KiB |
| 16 | 46.178 s | 46.490 s | 243,848 KiB |
| 24 | 42.252 s | 42.609 s | 254,972 KiB |

The 24-worker configuration was selected. It is 23.1% faster than the 8-worker
raw scan while leaving eight of the pod's 32 CPUs for decompression, writing,
and operating-system work. More importantly, deferred indexing removes the
former several-minute database stall from every archive-audit cycle.

The deployed source package SHA-256 was
`757b7ba438ad49ce655d89b9d984e477decf9ded1c35ca3e831182d6c6b05dae`.
All five deployed file hashes matched local sources. The prior live files were
saved at
`/workspace/dataset/staging/deploy_backups/postprocess-pre-fast-v7-20260830T133500Z.tgz`.
The live Linux environment passed nine focused preprocessing tests, shell
syntax validation, and the pinned source/curation-policy validation before the
new detached `preprocess` tmux session started at 10:38:33 UTC.

At the first five-minute production snapshot, the audit had reached
16,851,770,542 of 64,581,575,876 tokens (26.094%) across 1,295 archives and
13,993,989 documents, with 3,050 archives waiting and zero errors. This was an
increase of 850 completed fingerprint archives and 821,724,231 audited tokens
without changing the 445-archive dedup database. No SQLite journal appeared,
and the local scratch directory was empty between archives. The observed early
rate implies roughly three to six hours for the remaining raw audit, subject to
the larger-archive and English-source mix; the later index-only phase must be
measured separately rather than folded into that estimate.

Collection remains complete at 64.582B tokens. Downloading was not restarted:
the correct shortage decision comes only after complete quality,
decontamination, and English near-duplicate inventories quantify retained
tokens by split/domain.

A follow-up dependency audit proved that the legacy
`staging/preprocess/dedup/dedup.sqlite3` is not consumed by English near-
deduplication, final curation, packed-shard construction, order v4, loading, or
training. The production English and curation stages revalidate every
fingerprint and build independent authoritative SQLite journals. Completing the
legacy index would therefore duplicate the largest database pass merely to
obtain aggregate exact/normalized telemetry. The launcher now skips it by
default; `RUN_FINGERPRINT_AUDIT_INDEX=1` preserves an explicit rebuild path.
The partial 445-archive database remains an immutable, rebuildable audit
artifact and is not presented as production-complete.

This audit also exposed a more important correctness requirement: downstream
builders must not freeze whichever reports happen to exist. English near-
deduplication and curation now own fail-closed report-versus-finalized-ledger
gates; an external status check alone is insufficient. Required completion is
100% audited tokens, zero waiting archives, zero archive errors, and exact
per-bucket archive/document/token agreement with finalized quota records.

The audit was stopped once more only after its active archive and terminal
status committed. At that cutover it had reached 19,181,143,200 tokens
(29.701%), 1,936 reports total, and 2,409 archives waiting with zero errors.
A temporary lock guard prevented the old launcher from entering the redundant
index phase. The opt-in-index package SHA-256 was
`e783fa6c09a5606a7d8c93632f4893e122d14b9a0618f58a45d0602108a0a5a0`;
all five deployed hashes matched, nine live tests passed, and the 24-worker
audit resumed in `tmux` at 10:57:13 UTC. The dedup database remained unchanged,
and no rollback journal existed before or after the restart.

### Baseline v1 defers local English near-deduplication

On 2026-08-30 the first pre-training corpus was changed to a speed-first
baseline that skips the additional FineWeb-Edu/Wikipedia raw-text MinHash/LSH
clustering stage. FineWeb-Edu and Wikipedia are already curated sources, so the
projected 8–36 hour pass is deferred until evidence suggests its incremental
diversity benefit is worth measuring. This is an experimental choice, not a
claim that cross-source near duplicates are absent.

The baseline retains Stack v3's upstream code near-deduplication, MBPP
decontamination, repository-safe code splits, and stable-source English
grouping. Locally computed exact/normalized hashes propagate contamination and
drive a cheap deterministic two-stage canonicalization: retain one eligible
document per global byte-exact hash, then one per global normalized hash.
Edited or semantic near duplicates that do not share the normalized hash can
remain or cross internal splits, so internal validation may be optimistic. Raw
archives and fingerprints remain immutable, and the full English cluster
implementation is preserved without requiring another download or raw audit.

If MBPP is below target, internal validation does not predict external results,
generation exhibits memorized/repeated passages, or a sampled English audit
finds material redundancy, build a second deduplicated corpus and compare it at
the same architecture and training-token budget. The baseline requires an
explicit versioned curation-policy mode; the existing diagnostic
`--allow-missing-english-near-dedup` flag is not acceptable because it marks the
artifact non-production-ready. The implemented production identity is
`fast-exact-normalized-canonical-v1` (curation identity format v6); it explicitly
records that fuzzy near-deduplication was not performed.

### Fast-v6 curation production launch

On 2026-08-30 the fast-v6 curator was deployed and launched directly on the
RunPod network volume. The final source bundle SHA-256 was
`fabcb6ee34018e91f6ec2baa518e08f1e86eb6e19f43ef596f6cea6bb049dcad`.
The prior project tree was archived before replacement; the critical curator,
storage-probe, and fast-policy hashes matched local sources after deployment.
The local suite passed 218 tests with one expected macOS multiprocessing skip;
the CPU pod passed all 44 production-relevant curation/policy/probe/runbook
tests. The pod intentionally lacks PyTorch, so model, loader, materializer, and
training tests remain a mandatory GPU-pod gate. Its older SQLite also changes
query-plan wording and makes the deliberately skipped fuzzy-English test route
miss its local performance calibration; neither is used by fast-v6.

The final closed-collection gate passed with 4,345 of 4,345 reports,
51,328,930 documents, 64,581,575,876 audited content tokens, zero archive
errors, and zero closure failures. No download was resumed before curation.

Before launch, `scripts/probe_sqlite_storage.py` ran inside the exact future
`selection-fast-v1/.work` subtree and verified the expected durable mount:
the experiment's private RunPod NFSv4 network-volume source, with `hard`,
`nconnect=8`, and `local_lock=none`. The shareable runbook accepts this exact
source through `RUNPOD_NFS_SOURCE` rather than publishing the volume ID. It
measured 108,664.855 inserted rows/s
and 386,908.247 indexed rows/s. SQLite DELETE journaling, FULL synchronous
writes, file-backed temp storage beside the database, advisory-lock exclusion,
atomic hard-link lease exclusion, and competing-writer exclusion all passed. A
forced crash produced a 5,588,504-byte valid hot rollback journal, changed the
main database before recovery, then restored its exact committed SHA-256 with
integrity `ok` and zero uncommitted rows or payloads visible. The checksummed
probe result is under
`/workspace/dataset/audits/curation-fast-v1-20260830T141141Z/`.

The production process started at 14:26:48 UTC in the detached tmux session
`curation-fast-v1`, writing only to the network-volume publication
`/workspace/dataset/curated/selection-fast-v1`. Its durable owner lease and
NFS advisory lock were both present. The storage preflight projected
315,364,945,920 additional bytes for 51,328,930 documents and required
318,020,305,920 free bytes; 920,021,630,976 bytes were available, so the gate
passed. The first stable inventory snapshot had committed 579,994 documents
and 700,671,292 tokens across five complete FineWeb-Edu archives plus 80,000
rows of the sixth, with 63 committed transactions, a 386,572,288-byte maximum
database, a 77,455,304-byte maximum journal, and no storage violation.

After inventory, the same restart-safe job performs hard quality/benchmark
filtering, global exact then normalized-hash canonicalization, repository or
stable-English-source grouping, whole-group train/validation/test assignment,
and exact 40% Python / 40% other-code / 20% English quota selection. Semantic
near-deduplication remains explicitly deferred for this baseline.

### Real DDP pre-training correctness gate

On 2026-08-30 a real two-process CPU/Gloo audit exercised the production DDP
wrapper for the first time. It exposed a launch-blocking interaction between
`DistributedDataParallel(static_graph=True)` and the trainer's `no_sync()`
gradient-accumulation windows: the first accumulated synchronized backward
failed inside PyTorch's reducer. The same fixture with the default reducer
completed correctly. Production now centralizes DDP construction in
`wrap_distributed_model()` and pins `static_graph=False`.

The permanent regression test uses two accumulated microbatches with unequal
rank-local supervised-token totals, compares both replicas and optimizer state,
and verifies the globally token-normalized update against a single-process
concatenated-batch reference. This matters because mocked collectives and
single-process accumulation tests could not expose the reducer failure.

A second real-process gate now covers recovery rather than only optimizer math.
Two ranks train to step one over a frozen format-v4 synthetic order and call the
actual distributed checkpoint collective. Both processes terminate; two fresh
workers load their rank-specific RNG state and the shared immutable-order
cursor, then complete step two. The final model, AdamW state, counters, reduced
metrics, RNG state, and exhausted next-row position must match the uninterrupted
two-rank control bit for bit. This closes the CPU-testable process-restart gap;
the equivalent BF16/CUDA/NCCL gate remains mandatory on the training image.

The same audit found that `torch.manual_seed`, `get_rng_state_all`, and
`set_rng_state_all` are unsafe for one-process-per-GPU training: every worker
can initialize every visible GPU, multiplying CUDA contexts and wasting memory
before the first optimizer step. Seeding now addresses the CPU generator and
current CUDA device explicitly. Checkpoint format v4 stores exactly one local
CUDA RNG device index/state per rank and rejects a changed rank-local device on
exact resume.

Verification after both corrections:

```text
venv/bin/python -m unittest tests.test_train_distributed tests.test_train
Ran 31 tests in 2.316s
OK

venv/bin/python -m unittest discover -s tests
Ran 236 tests in 30.379s
OK (skipped=1)
```

The distributed gate needs local-socket permission because Gloo opens a
loopback transport even with a `file://` rendezvous. The remaining mandatory
hardware gates are the CUDA FlexAttention isolation/gradient smoke, bitwise
CUDA checkpoint-resume comparison, memory/throughput calibration, and a real
multi-GPU scaling run on the pinned training image.

The intended first multi-GPU gate and production run use a single RunPod pod
with six local GPUs, launched as one NCCL process per GPU with
`torchrun --standalone --nproc-per-node=6`. This is a single-node topology;
multi-node rendezvous and fault handling are outside the initial experiment.

### OpenCodeInstruct raw snapshot certification

On 2026-08-30 the separate post-training snapshot of
`nvidia/OpenCodeInstruct@8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`
was certified in place. The verifier streamed every file and accepted exactly
50 train Parquet shards, 5,000,000 metadata rows, and 6,861,113,102 compressed
data bytes. Its per-file inventory identity is
`aeec9c739bafbf17cbea36399d509eab8e2c73fea7ad99d8c147c2a8d1d2290f`.
The authoritative `SOURCE.json` SHA-256 is
`05e01330a6dbf7003e22df3442a4d4d3fd571bca01e0319c17c97bcfabc6ebf6`.

The raw snapshot remains under `/workspace/posttraining-data`, outside every
pre-training discovery root. Certification proves download identity and
integrity only; benchmark decontamination, tokenizer-length analysis, schema
normalization, split construction, and SFT publication remain pending.

The same public commit was cloned to the server in a checkout separate from
the live curator. In the server's data environment, all 10 SFT downloader tests
passed. In its PyTorch 2.13 CPU environment, 95 focused loader, trainer,
launcher, and SFT tests passed with one optional skip. No package was added to
the active curation environment.

### Prime Intellect SFT integration

On 2026-08-30 the post-training design was separated into two explicit paths.
Fixed-target supervised fine-tuning uses PrimeRL directly over an immutable
Hugging Face `messages` dataset; it does not instantiate a Verifiers
environment. A separate six-task, repository-authored Verifiers package exists
only to smoke-test later containerized evaluation/RL plumbing and is prohibited
from supplying SFT rows.

The reviewed runtime pins are PrimeRL
`3fc28ddfb354f336d1cc28e8e032f262f5aa68b2` and renderers
`c4772ac1321c69e83d2b4460600072911cc41a0b`. A typed renderer named
`starcoder2-coding-chat-v1` was added as a reproducible patch to the pinned
renderer checkout. It adds no vocabulary entries, independently encodes ASCII
role scaffolding and message bodies, masks all prompt/scaffold tokens, and
supervises assistant content plus terminal EOS ID 0. The same renderer is used
for exact 4,096-token curation checks and Prime training, preventing a hidden
template/token-count mismatch.

The OpenCodeInstruct projection is a restartable two-pass publication job. It
first authenticates the certified 50-shard/5,000,000-row raw snapshot and
builds a global authority for prompt groups matching the frozen MBPP denylist
across prompt, answer, and unit-test text. It then validates rows, propagates
group rejections, rejects high-confidence credentials, renders exact lengths,
drops rather than truncates over-context conversations, assigns normalized
prompt groups to deterministic train/validation splits, and atomically
publishes Parquet plus complete manifests and ID/reason-only rejection ledgers.
This is exact/normalized decontamination and grouping, not proof against
semantic paraphrases. The current `min_average_test_score=0.0` policy is
deliberately provisional; a real run is blocked until the observed score
distribution is audited and a new immutable positive-threshold dataset and
training approval are published.

Native format-v5 pretraining checkpoints now bind both the source tokenizer
manifest SHA-256 and a canonical token-to-ID vocabulary SHA-256. The HF exporter
requires those identities, validates the complete native state/config, maps it
to Llama names, writes only safetensors, preserves the source tokenizer
manifest, generates new integrity metadata for the derived tokenizer files,
and publishes atomically. Prime preflight additionally proves the curated data
and exported checkpoint share the exact source tokenizer manifest and
`tokenizer.json` bytes.

The checked-in Prime TOML is a six-rank dry-run scaffold: one node, six training
GPUs, custom Llama, 4,096 context, BF16, global packed-row batch 48,
microbatch one, eight accumulation rounds, assistant-only labels, and one data
worker per rank. The custom model path is mandatory because the reviewed
PrimeRL SFT packer supplies `seq_lens` for variable-length/block-diagonal
attention; the generic HF path does not preserve those boundaries. A
fail-closed launcher authenticates the complete dataset, HF export, tokenizer,
Prime/renderers checkouts, installed patch, and TOML; performs a one-process
offline HF cache prewarm; then requires the matching cache inventory before it
can exec six ranks. The checked-in contract cannot launch a real non-dry run.

No SFT curation or Prime training was launched during this implementation.
Remaining hardware gates are a one-GPU forward/backward and overfit, BF16 and
memory calibration, a six-rank NCCL step, and an explicit perturbation test
proving packed conversation boundaries remain isolated on the selected Prime
GPU image. MBPP remains final-evaluation-only and must not be used for
checkpoint or hyperparameter selection.

The final local CPU suite passed 284 tests with one platform skip, including
the real two-process Gloo checkpoint/resume gates and 39 focused Prime/SFT
integration tests. The Prime TOML also parsed successfully through the exact
pinned upstream `SFTConfig`, and the renderer patch applied and validated
against clean temporary copies of both pinned upstream checkouts.

## 2026-08-31 baseline-curator stop for local-storage upgrade

At 05:25 UTC, the network-volume `selection-fast-v1` curator was deliberately
interrupted through its tmux foreground job. The purpose was to resize the CPU
pod's local storage before qualifying the local-WAL
acceleration in commit `6b7decc`. The pod available at shutdown exposed 844 GB
free on the durable NFS volume, but only 3.4 GB of local overlay and 60 GB of
tmpfs. Read-only block-device enumeration showed host NVMe names without usable
device nodes in the container; no device was mounted, formatted, or modified.

The final durable checkpoint records inventory phase, 113 completed archives,
11,236,456 / 51,328,930 documents (21.891%), and no storage violation. The
active archive cursor is `raw/other_code/part-000020.tar.zst` at 30,000 /
151,510 rows and 25,207,482 tokens. The SQLite database is 7,679,508,480 bytes.
Shutdown verification found no remaining Python curator, SQLite journal/WAL/SHM
sidecar, or cross-client lease, so the baseline remains safely resumable.

The accelerated path is not an in-place migration. It requires a fresh output,
a separately identified local filesystem passing the exact admission gate, and
the same durable network volume for authenticated
snapshots. The baseline output must remain untouched as rollback authority until
the accelerated generation passes semantic equivalence, crash recovery, and the
minimum 3x representative performance gate.

## 2026-08-31 local-WAL qualification and full rollout

The replacement CPU pod exposed a 320 GiB local overlay
(343,597,383,680 bytes total) and the durable network volume at `/workspace`.
The frozen batch-100,000 admission calculation was 315,364,945,920 bytes for
the 2x projected database growth, 6,553,600,000 bytes for the bounded
transaction sidecar, and 2,000,000,000 bytes reserved after projection:
323,918,545,920 bytes required in total. Initial free space was
343,580,889,088 bytes, leaving a 19,662,343,168-byte margin. The local root is
therefore admitted but must not hold unrelated material during curation.

The separate server checkout `/workspace/0-coding-llm` was advanced to commit
`7a6d35e`. A minimal `/opt/coding-model-venv` used Python 3.11.13, SQLite
3.31.1, zstandard 0.25.0, and xxhash 3.8.1. The 56 focused target-runtime tests
for curation, local storage, acceleration, and monitoring passed in 23.037
seconds.

A fresh controlled generation, `selection-fast-local-v2`, was launched with
local live state under `/local/curation/selection-fast-local-v2`, durable state
under `/workspace/dataset/curated/selection-fast-local-v2`, batch size 100,000,
hourly snapshots, retention two, and deferred raw-archive hashing. The
one-archive qualification processed 100,000 FineWeb-Edu documents and exited
cleanly in approximately 49 seconds including initialization and durable
publication. Snapshot `snapshot-000000000001` authenticated against its SHA-256
sidecar; its manifest bound a 68,067,328-byte database, one archive, and 100,000
documents. `PRAGMA integrity_check` returned `ok` and counts `(1, 100000)` for
both the local database and durable canonical database. No lease or live process
remained after the controlled exit.

At 07:04 UTC, the same command without the one-archive limit resumed the exact
qualified generation in tmux session `curation-fast-local-v2`. At 07:05 UTC,
the first stable health record reported the curator alive with no warnings,
eight completed archives, 733,993 documents (1.430% of inventory), bounded WAL,
and 342,497,202,176 local bytes free. The monitor runs separately in
`curation-fast-local-v2-monitor`. Its first invocation observed a valid brief
publication boundary before the next running archive was published and exited
with a zero-running-archive alert; restarting it after the active archive
appeared immediately produced a healthy record. This record does not indicate
curator failure.

At 07:08 UTC, the checkpoint had advanced to 28 completed archives and
2,583,958 documents (5.034%), with the next archive running, no storage
violation, and 340,704,468,992 local bytes free. The live database was about
1.75 GB and its WAL about 641 MB. Both curator and monitor tmux sessions
remained alive.

Commit `85fb98a` added bounded checkpoint/journal coherence retries, required a
nonempty journal to begin at sequence one, and made the zero-active publication
window warning-only for at most 60 seconds when the bound curator process is
alive and every other invariant passes. Dead, stale, inconsistent, and
over-bound states remain fatal. All 12 focused tests passed both locally and in
the pod's minimal runtime. The server checkout was fast-forwarded and only the
monitor was restarted; the curator continued uninterrupted. The first record
from the new monitor was healthy at 66 archives / 6,001,567 documents
(11.692%). At 07:15 UTC the curator had reached 69 archives / 6,301,563
documents (12.276%) with no storage violation.

The accelerated generation was already more than 3x faster than the historical
roughly 600,000-documents/hour baseline during qualification and early rollout.
That establishes the performance gate for continuing the new generation, not
semantic equivalence of the eventual final corpus. The untouched
`selection-fast-v1` baseline remains the rollback authority until the
accelerated corpus completes, publishes, and passes final accounting, integrity,
and training-data certification.
