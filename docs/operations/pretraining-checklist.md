# Pre-training readiness checklist

This is the required path from the completed packed corpus to the first full
1.3B-parameter pre-training run. Do not train directly from raw `.tar.zst`
archives, the raw-token cache, packed-shard directory order, or the packed-only
S3 recovery artifact.

The current identity/status authority is the
[pre-training corpus record](../data/final-corpus-record.md). Generation-v2
build history is in [fast-generation-v2.md](../data/fast-generation-v2.md).
Collection, preprocessing, curation, selection-v7, raw-token caching, and
packing are complete. Exact curation-quota selection was intentionally omitted:
retain every eligible canonical full document, then enforce mixture and budgets
in packed order v4.

## 1. Preserve v1 and complete the v2 other-code top-up

- [x] Python reaches 25,718,400,000 collected StarCoder2 tokens.
- [x] Other code reaches 25,718,400,000 collected tokens.
- [x] FineWeb-Edu reaches 10,287,360,000 collected tokens.
- [x] Wikipedia reaches 2,571,840,000 collected tokens.
- [x] Every collector has written its completion marker and exited with status
  zero. A small whole-document overshoot is expected.
- [x] Save the final quota report, source manifests, resolved dataset commits,
  tokenizer manifest, collector versions, logs, and configuration hashes.
- [x] Freeze `/workspace/dataset`; no job may mutate the v1 root.
- [x] Atomically publish `/workspace/dataset-other-code-topup-v2` with a valid
  `CLONE_MANIFEST.json`: 21,181 hard links / 72,337,391,686 bytes, five copied
  controls, manifest SHA-256
  `815c6256f0354f1b6a6cc524d96e745331c68afd02f3e72b19bb2d66ed2b3de9`.
- [x] Qualify the `stack-v3-topup-v2` checkpoint recovery used during the
  build: all eight workers advanced committed other-code supply from
  25,952,231,562 to 26,066,335,409 tokens while Python remained exactly
  25,770,142,666.
- [x] Reach at least 35,000,000,000 cumulative raw other-code tokens while
  keeping Python and English targets unchanged.
- [x] Close collection only after every new `.part-*` finalized, publish the v2
  collection-completion marker, and require incremental preprocessing to pass
  closed-collection coverage with zero archive errors.

The original acquisition totaled about 64.58B audited tokens, but measured
other-code train retention was lower than the nominal headroom assumption. The
v2 35B raw other-code target provides 27.914% margin over the measured-yield
point estimate. The final train order remains capped at 52.58B input tokens.

## 2. Audit raw-corpus integrity

The streaming preprocessor begins these audit/fingerprint steps as finalized
archives arrive; see
[streaming-preprocess.md](../data/streaming-preprocess.md). After acquisition, verify that
it has caught up completely and that its error count is zero.

- [x] Enumerate every archive and record its size and SHA-256 checksum.
- [x] Stream-read every `.tar.zst`; confirm it closes cleanly and contains one
  valid `_manifest.jsonl`.
- [x] Reconcile each archive's document, byte, and exact-token totals against
  the quota records. Fail on missing or duplicate shard IDs.
- [x] Check that every manifest member exists exactly once and its stored byte
  length matches the manifest.
- [x] Confirm no hidden `.part-*` file is admitted to downstream processing.
- [x] Run the curation completeness gate and require an exact one-to-one match
  across all four buckets between collection quota records, finalized raw
  archives, preprocessing reports, and fingerprint shards. Require zero error
  records and per-bucket archive/document/byte/token totals equal to the
  collection ledgers.
- [x] Preserve the curation run's `collection_completeness`
  identity. The legacy `staging/preprocess/dedup/dedup.sqlite3` aggregate audit
  index is rebuildable and is not a production curation prerequisite.
- [x] Preserve raw archives as immutable inputs. Write every later phase to a
  new directory rather than modifying raw data.
- [x] Publish the exact v1 WAL-aware supply audit at
  `/workspace/dataset/audits/supply-audit-fast-v1-20260901/supply-audit.json`
  with SHA-256
  `74de45bdf3438395f74f6c492f11017e6c0be6b76e0f08ec73e88a0b77169230`.
- [x] Incrementally audit/fingerprint only the newly added v2 other-code
  archives using the same versioned v2 quota configuration as collection;
  require complete v2 ledger/raw/report/fingerprint identity afterward.

## 3. Define the final filtering policy before running it

- [x] Complete corpus curation under the frozen generation-v2 policy, including
  basic eligibility, benchmark rules, global exact/normalized canonicalization,
  and leakage-safe group splits.
- [x] Version the exact rules used for minimum/maximum document size, generated
  or repetitive text, corrupted Unicode, secrets, PII, minified code, vendored
  code, and any source-specific quality threshold.
- [ ] Future ablation only: decide whether to rebalance the 78-language
  `other_code` bucket. Without an
  explicit policy, high-volume languages such as C++, Java, C, and JavaScript
  will dominate it. Record per-language minimums, caps, or sampling weights.
- [ ] Future ablation only: decide whether to alter the completed English source
  composition.
- [x] Emit a reason and source identity for every filtered document. Rejected
  content itself does not need to be retained.
- [x] Produce before/after document and token accounting by split/domain and
  rejection reason. More detailed language/source tables remain optional
  analysis rather than a launch prerequisite.

## 4. Deduplicate and decontaminate

### Baseline-v1 decision: defer full English near-deduplication

The first pre-training run is a speed-focused baseline. Do **not** build the
additional FineWeb-Edu/Wikipedia raw-text MinHash/LSH clusters. These sources
are already curated, so the expected incremental benefit does not justify
delaying the initial experiment by the projected 8–36 hours. This is an
experimental choice, not a claim that cross-source near duplicates are absent.

Keep the immutable raw archives, fingerprints, and implemented cluster builder
so a deduplicated-corpus ablation can be produced later without redownloading or
re-auditing the raw corpus. The baseline still retains Stack v3's upstream code
near-deduplication, MBPP/benchmark filtering, repository-safe code splits, and
stable-source English splits. Locally computed exact/normalized hashes are used
for deterministic canonical cleanup and contamination propagation. Exactly one
eligible canonical survives each global byte-exact hash and then each global
normalized hash. This is a bounded hash pass, not fuzzy near-deduplication.

The current `--allow-missing-english-near-dedup` flag is diagnostic only and
marks output non-production-ready. The production baseline instead uses the
versioned `fast-exact-normalized-canonical-v1` policy and records the semantic
near-duplicate limitation in every downstream manifest.

- [x] Validate `configs/curation_policy_fast_exact_normalized.json` against the pinned
  `STACK_V3_SOURCE.json`. The code source must be
  `HuggingFaceCode/stack-v3-train@df4b205fbba4cc1c2fd1f205b10d66f730798bb9`.
  Reserve `configs/curation_policy.json` for the separate full-near v5
  ablation; do not substitute the diagnostic missing-near override.
- [x] Audit exact and normalized hashes across all four source buckets and
  deterministically keep one global canonical for each exact hash, followed by
  one global canonical for each normalized hash.
- [x] Do **not** run another MinHash/LSH near-deduplication pass over Python or
  other code. The pinned Stack v3 Train v3.1 source is already file-level,
  cross-repository near-deduplicated. The v1 code sketches are audit-only.
- [x] Defer FineWeb/Wikipedia near-deduplication for baseline v1. Revisit it as
  a controlled follow-up if results or duplicate audits justify the cost.
- [x] Freeze the benchmark guard used by the completed curation pass. At
  minimum keep MBPP completely out; preferably also protect HumanEval,
  EvalPlus, MultiPL-E, MBXP, MXEval, and any other evaluation planned later.
- [x] Run the frozen benchmark matching rules over the final candidates and
  preserve the authenticated benchmark-guard authority. Store only rejection
  metadata and non-reversible fingerprints.
- [ ] Run the independent MBPP audit again on the selected Python corpus and
  require zero matches.

Exact/normalized canonicalization and decontamination happen before final split
assignment. Baseline v1 skips only fuzzy/semantic English near-deduplication.
The 20% acquisition headroom remains available for quality filtering and hash
canonicalization.

## 5. Construct leakage-safe splits and packed-order budgets

Assign whole groups to one split only. For code, group by pinned source revision
and `repo_id`; for English, group by stable source/article identity. Exact and
normalized collisions have already been canonicalized globally. A later fuzzy
deduplicated ablation can use the local near-duplicate cluster instead. Never
split a known source group between train and evaluation. Use a recorded random
seed and stable source IDs. Upstream deduplication and split grouping are
distinct: the former does not make file-level random splitting safe.

The nominal rows below are reference maxima, not SQLite selection quotas:

| Split | Python | Other code | English | Total cap |
|---|---:|---:|---:|---:|
| Train | 21.032B | 21.032B | 10.516B | 52.580B |
| Validation | 0.200B | 0.200B | 0.100B | 0.500B |
| Test | 0.200B | 0.200B | 0.100B | 0.500B |
| Total | 21.432B | 21.432B | 10.716B | 53.580B |

The training budget and 40/40/20 mixture refer to model input slots after
packing, including inserted EOS tokens but excluding the duplicated lookahead
storage slot. At context 4,096, the nominal 52.58B train budget permits at most
12,836,914 whole rows (52,579,999,744 input slots). This context-derived cap is
not a training-geometry recommendation. The first-run statistical batch is now
fixed at 192 rows per update, so the selected order is exactly 12,836,736 unique
rows: 66,858 complete updates and 52,579,270,656 input positions under the
40/40/20 allocation. It is one pass through those selected references without
replacement or repeats, not through all packed-surplus rows. The pending
geometry gate chooses only the physical microbatch/accumulation decomposition.
The finalized manifest must reproduce these exact totals and report valid
supervised targets and packed surplus separately.

Generation v2 must rebuild canonical winners and groups from every old and new
archive. Its publication keeps all eligible canonical documents at full length.
Deterministic `order.bin` construction then chooses without replacement from
separate domain streams. Validation and test each use the largest feasible
balanced whole-row cap at or below 0.5B; a smaller held-out set is acceptable
when one domain is limiting and is preferable to acquiring more data solely for
the nominal held-out size.

- [x] Freeze leakage-safe train/validation/test membership in selection-v7.
- [ ] Publish and authenticate deterministic validation/test order manifests;
  once published, never reshuffle them between runs.
- [ ] Never use the language-model test set or MBPP to choose checkpoints or
  hyperparameters. Use validation for those decisions.
- [x] Verify packed supply is sufficient for the selected 40% Python / 40%
  other-code / 20% English train budget.
- [ ] Verify the finalized train order has exactly 12,836,736 unique row
  references, 66,858 complete updates, 52,579,270,656 input positions, the
  frozen 40/40/20 allocation, and no replacement or repeated row.

## 6. Freeze tokenization and sequence construction

- [x] Continue using the pinned `bigcode/starcoder2-tokenizer` revision already
  stored on the volume. Do not retrain or silently update it.
- [x] Freeze model context length at 4,096 tokens before packing.
- [x] Insert an end-of-text token between documents. Do not allow two documents
  to touch without a boundary token.
- [x] Pack tokens densely into fixed-length causal sequences. Do not pad normal
  training samples; retain carry across shard boundaries and omit only the
  incomplete tail of a finalized split/domain stream.
- [x] Prevent attention across document boundaries using explicit segment IDs;
  this policy is recorded and tested on the CPU reference backend. The CUDA
  FlexAttention implementation still requires its GPU smoke test.
- [x] Keep source, language, document-boundary, split, and document-position
  provenance in a side index even though the training payload contains token
  IDs; validate the materializer's completed document-position indexes against
  the packed shards.

## 7. Build training-ready shards

The native PyTorch packed-row format, deterministic global order, distributed
sampler, and synthetic correctness tests are implemented in `pretrain/data.py`;
see [training-data.md](../training/training-data.md). Selection-v7, the
closed-world token cache, and packing are complete. The portable S3 artifact is
packed-only and is restored at `/root/transcendent-logic-data`; orders and the
top-level corpus manifest remain pending.

- [x] Convert selected documents into large sequential binary shards, roughly
  0.5-2 GB each, plus indexes and checksums. Do not make the GPU loader open
  millions of small files inside raw tar archives.
- [x] Keep Python, other code, and English identifiable so the sampler can
  enforce the intended mixture rather than relying on directory order.
- [ ] Deterministically shuffle document or packed-sequence order with recorded
  seeds. Do not shuffle validation or test between runs.
- [ ] Verify a random sample by decoding token IDs and inspecting boundaries,
  Unicode, language labels, and source provenance.
- [ ] Write a final dataset manifest containing every shard checksum, sequence
  count, exact token count, tokenizer revision, context length, split, category,
  source composition, filtering revision, and construction seed.
- [x] Run the bridge with `--stop-after-packing`: validate all nine packed
  manifests and the checksummed per-document position indexes, and preserve its
  durable `phase: packed` journal. This stage must not guess GPU batch geometry.
- [x] Publish the portable packed-phase recovery artifact to
  `s3://transcendent-logic-data-618079239540/coding-llm/pretraining/2026-09-02-packed-v1/`
  and restore it directly to six-H100 pod-local NVMe with checksum verification.
  Do not treat this packed-only artifact as an order or top-level manifest.
- [ ] Before finalization, reauthenticate the packed journal, all nine packed
  manifests/shards, tokenizer, selection-v7, and cache inventory on local NVMe;
  any missing or mismatched identity stops the run.
- [ ] Mount the packed volume on the intended GPU topology and run the complete
  six-GPU geometry grid. Keep the effective optimizer batch fixed at 192 rows
  while measuring which global-microbatch/accumulation decomposition is safe
  and fastest.
- [ ] Build order format v4 with frozen global microbatch rows **and** gradient
  accumulation. Verify effective update rows, update count, the exact
  consumed/dropped input and supervised-token counters (including 40/40/20
  prefix accounting), and archive the full order-validator output.

## 8. Freeze the 1.3B model and optimization recipe

- [x] Use the restored six-H100 RunPod pod, one NCCL
  process per GPU (`torchrun --standalone --nproc-per-node=6`) using the
  implemented replicated-DDP path. Exact device/driver/interconnect inventory
  and full-model memory remain measured gates; a DDP memory-gate failure stops
  the launch rather than silently changing the distributed strategy.
- [x] Record the exact architecture: layer count, hidden size, attention heads,
  key/value heads, MLP size, activation, normalization, positional encoding,
  vocabulary size, context length, tied embeddings, and parameter count.
- [ ] Confirm initialization rules and numerical precision. BF16 is preferred
  when supported; FP16 requires more care with loss scaling.
- [ ] Record AdamW betas/epsilon/weight decay, peak and minimum learning rates,
  warmup tokens or steps, decay schedule, gradient clipping, dropout, and random
  seeds.
- [x] Choose the initial statistical optimizer batch: 192 packed rows, or
  786,432 input positions at sequence length 4,096. The rationale and
  controlled 96/192/384 fallback test are frozen in
  [the batch-size decision](../training/batch-size-selection.md).
- [ ] Qualify the physical global-microbatch/accumulation pair before freezing
  the order. Use the manifest's optimizer-update count and exact
  `training_consumption.consumed_input_tokens`; record the derived per-rank
  microbatch. For the planned pod, global rows must be divisible by six and
  global rows times accumulation must equal 192.
- [ ] Verify replicated-DDP optimizer-state, activation, and communication
  memory against the selected GPU configuration.

## 9. Make checkpointing genuinely resumable

- [ ] Each checkpoint includes model weights, optimizer, scheduler, scaler if
  used, RNG states, consumed-token count, optimizer step, sampler/dataloader
  position, order-manifest and order-payload hashes, all packed-manifest hashes,
  tokenizer-manifest hash, global microbatch rows, accumulation, effective
  update rows, world size, completed optimizer updates, and exact next
  order-row offset.
- [ ] Write checkpoints atomically to a dedicated network-volume directory.
  Never overwrite the last known-good checkpoint in place.
- [ ] Define retention and periodic archival policy before training.
- [ ] Perform a forced-stop test: resume and verify that the next batches,
  learning rate, and loss match an uninterrupted control run.

## 10. Run staged validation before the expensive job

- [x] Unit-test causal labels, loss masking, packed boundaries, mixture sampling,
  and exact token accounting.
- [x] Run a real two-process CPU/Gloo DDP gate through the production wrapper.
  It covers `no_sync()` accumulation, unequal per-rank supervised-token counts,
  global token normalization, replica/optimizer synchronization, and equivalence
  to a single-process global batch. It also guards the required
  `static_graph=False` reducer policy.
- [x] Verify that checkpoint RNG capture, restore, and seeding touch only each
  DDP rank's assigned CUDA device; checkpoint format v5 records the local device
  index/state plus mandatory manifest and canonical vocabulary SHA-256
  identities rather than initializing every visible GPU in every worker.
- [x] Run a real two-rank CPU process-restart gate: atomically save at a clean
  accumulation boundary, terminate both workers, restore with two fresh
  workers, resume the exact format-v4 global-order cursor from a format-v5
  trainer checkpoint, and require
  bit-for-bit equality with the uninterrupted distributed trajectory.
- [x] Make the production launcher inventory every visible GPU and reject a
  heterogeneous DDP topology. For the FP32 1.3B path, reject devices below
  32 GiB: parameters, gradients, and two Adam moments already consume
  20,536,918,016 bytes per replica before activations/workspaces.
- [x] Keep the launcher alive as a signal supervisor so preemption does not
  inherit TorchElastic's roughly 30-second worker-kill deadline. Workers poll a
  rank-shared request and checkpoint at a clean boundary; the pod termination
  grace must cover the configured supervisor timeout.
- [ ] Decode and manually inspect several complete batches.
- [x] Run the six-GPU tiny-model overfit on six real packed 4,096-token rows for
  1,000 steps; 100 steps was explicitly judged insufficient. Loss fell from
  10.80926135 to 0.00173715 (ratio 0.00016071; perplexity 1.0017387), with all
  27/27 cross-document targets masked across 33 segments and 24,549 supervised
  tokens. Preserve the latest and previous checkpoints and W&B run `4i4pvoj6`.
  This pass does not qualify the full 1.3B allocation or geometry.
- [ ] Run a short single-GPU job; require decreasing training loss, finite
  gradients, correct checkpoints, and successful resume.
- [ ] Run the full 1.3B multi-GPU geometry/scaling and soak tests; measure
  tokens/s, GPU utilization, memory headroom, local-NVMe data throughput, and
  communication overhead. Do not infer this gate from the tiny-model overfit.
- [ ] Estimate wall-clock time and GPU cost from measured throughput, not peak
  hardware specifications.
- [ ] Confirm validation loss can be computed independently for Python, other
  code, and English.

## 11. Full-run go/no-go record

Before launching, archive one immutable run manifest containing:

- dataset-manifest and shard checksums;
- source and benchmark-denylist revisions;
- tokenizer and model configuration hashes;
- code revision and complete environment/package versions;
- GPU types/count, CUDA and driver versions;
- all optimizer, scheduler, batching, precision, and distributed settings;
- random seeds, planned training tokens, expected steps, checkpoint cadence,
  validation cadence, and estimated budget.

During training monitor total and per-domain loss, learning rate, gradient norm,
tokens/s, GPU utilization, memory, numerical overflows/NaNs, data wait time, and
consumed tokens by category. Stop on non-finite loss, mixture drift, corrupted
data, checkpoint failure, or unexplained throughput collapse.

Only after every remaining fail-closed gate passes should the one-pass selected
training order start: exactly 12,836,736 unique rows, 66,858 updates, and
52,579,270,656 input positions under the 40/40/20 allocation, with no repeats.
MBPP remains evaluation-only and is not consulted during pre-training.
