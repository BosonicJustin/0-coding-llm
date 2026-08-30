# Pre-training readiness checklist

This is the required path from the current raw acquisition jobs to the first
full 1.3B-parameter pre-training run. Do not train directly from the raw
`.tar.zst` collection archives.

## 1. Finish and freeze acquisition

- [x] Python reaches 25,718,400,000 collected StarCoder2 tokens.
- [x] Other code reaches 25,718,400,000 collected tokens.
- [x] FineWeb-Edu reaches 10,287,360,000 collected tokens.
- [x] Wikipedia reaches 2,571,840,000 collected tokens.
- [x] Every collector has written its completion marker and exited with status
  zero. A small whole-document overshoot is expected.
- [x] Save the final quota report, source manifests, resolved dataset commits,
  tokenizer manifest, collector versions, logs, and configuration hashes.
- [ ] Stop the CPU download pod only after all open `.part-*` archives have
  been finalized. Keep the network volume.

Acquisition totals 64.296B tokens because it includes 20% headroom. The final
corpus is smaller: 52.58B train tokens plus 0.50B validation and 0.50B test.

## 2. Audit raw-corpus integrity

The streaming preprocessor begins these audit/fingerprint steps as finalized
archives arrive; see `STREAMING_PREPROCESS.md`. After acquisition, verify that
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

## 3. Define the final filtering policy before running it

- [ ] Version the exact rules for minimum/maximum document size, generated or
  repetitive text, corrupted Unicode, secrets, PII, minified code, vendored
  code, and any source-specific quality threshold.
- [ ] Decide how to balance the 78-language `other_code` bucket. Without an
  explicit policy, high-volume languages such as C++, Java, C, and JavaScript
  will dominate it. Record per-language minimums, caps, or sampling weights.
- [ ] Decide whether final English remains 80% FineWeb-Edu and 20% Wikipedia.
- [ ] Emit a reason and source identity for every filtered document. Rejected
  content itself does not need to be retained.
- [ ] Produce before/after document, byte, and token tables by source,
  language, and rejection reason.

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

- [ ] Validate `configs/curation_policy_fast_exact_normalized.json` against the pinned
  `STACK_V3_SOURCE.json`. The code source must be
  `HuggingFaceCode/stack-v3-train@df4b205fbba4cc1c2fd1f205b10d66f730798bb9`.
  Reserve `configs/curation_policy.json` for the separate full-near v5
  ablation; do not substitute the diagnostic missing-near override.
- [x] Audit exact and normalized hashes across all four source buckets and
  deterministically keep one global canonical for each exact hash, followed by
  one global canonical for each normalized hash.
- [ ] Do **not** run another MinHash/LSH near-deduplication pass over Python or
  other code. The pinned Stack v3 Train v3.1 source is already file-level,
  cross-repository near-deduplicated. The v1 code sketches are audit-only.
- [x] Defer FineWeb/Wikipedia near-deduplication for baseline v1. Revisit it as
  a controlled follow-up if results or duplicate audits justify the cost.
- [ ] Freeze the evaluation-suite list before the final pass. At minimum keep
  MBPP completely out; preferably also protect HumanEval, EvalPlus, MultiPL-E,
  MBXP, MXEval, and any other evaluation planned later.
- [ ] Run exact, normalized, substring, and sufficiently conservative fuzzy
  benchmark matching over the final candidates. Store only rejection metadata
  and non-reversible fingerprints.
- [ ] Run the independent MBPP audit again on the selected Python corpus and
  require zero matches.

Exact/normalized canonicalization and decontamination happen before final split
assignment. Baseline v1 skips only fuzzy/semantic English near-deduplication.
The 20% acquisition headroom remains available for quality filtering and hash
canonicalization.

## 5. Construct leakage-safe splits and exact budgets

Assign whole groups to one split only. For code, group by pinned source revision
and `repo_id`; for English, group by stable source/article identity. Exact and
normalized collisions have already been canonicalized globally. A later fuzzy
deduplicated ablation can use the local near-duplicate cluster instead. Never
split a known source group between train and evaluation. Use a recorded random
seed and stable source IDs. Upstream deduplication and split grouping are
distinct: the former does not make file-level random splitting safe.

| Split | Python | Other code | English | Total |
|---|---:|---:|---:|---:|
| Train | 21.032B | 21.032B | 10.516B | 52.580B |
| Validation | 0.200B | 0.200B | 0.100B | 0.500B |
| Test | 0.200B | 0.200B | 0.100B | 0.500B |
| Total | 21.432B | 21.432B | 10.716B | 53.580B |

The training budget and 40/40/20 mixture refer to model input slots after
packing, including inserted EOS tokens but excluding the duplicated lookahead
storage slot. At context 4,096, the nominal 52.58B train budget permits at most
12,836,914 whole rows (52,579,999,744 input slots). This context-derived cap is
not a training-geometry recommendation. Order v4 selects the largest strict
40/40/20 allocation at or below it, rounded down to a whole optimizer-update
prefix using the GPU-smoke-chosen global microbatch and gradient accumulation,
then freezes that geometry. Read the exact row/update/token count from the
finalized manifest; report valid supervised targets and packed surplus
separately.

- [ ] Make validation and test immutable after selection.
- [ ] Never use the language-model test set or MBPP to choose checkpoints or
  hyperparameters. Use validation for those decisions.
- [ ] Verify exact post-tokenization totals and the 40% Python / 40% other-code
  / 20% English training mixture.

## 6. Freeze tokenization and sequence construction

- [x] Continue using the pinned `bigcode/starcoder2-tokenizer` revision already
  stored on the volume. Do not retrain or silently update it.
- [x] Freeze model context length at 4,096 tokens before packing.
- [x] Insert an end-of-text token between documents. Do not allow two documents
  to touch without a boundary token.
- [x] Pack tokens densely into fixed-length causal sequences. Do not pad normal
  training samples; drop or mask only the incomplete tail of a finalized shard.
- [x] Prevent attention across document boundaries using explicit segment IDs;
  this policy is recorded and tested on the CPU reference backend. The CUDA
  FlexAttention implementation still requires its GPU smoke test.
- [ ] Keep source, language, document-boundary, split, and document-position
  provenance in a side index even though the training payload contains token
  IDs. Check this only after the materializer's document-position index is
  final and tested against the published shards.

## 7. Build training-ready shards

The native PyTorch packed-row format, deterministic global order, distributed
sampler, and synthetic correctness tests are implemented in `pretrain/data.py`;
see `TRAINING_DATA.md`. The boxes below remain unchecked until those tools have
been run and validated on the final selected corpus.

- [ ] Convert selected documents into large sequential binary shards, roughly
  0.5-2 GB each, plus indexes and checksums. Do not make the GPU loader open
  millions of small files inside raw tar archives.
- [ ] Keep Python, other code, and English identifiable so the sampler can
  enforce the intended mixture rather than relying on directory order.
- [ ] Deterministically shuffle document or packed-sequence order with recorded
  seeds. Do not shuffle validation or test between runs.
- [ ] Verify a random sample by decoding token IDs and inspecting boundaries,
  Unicode, language labels, and source provenance.
- [ ] Write a final dataset manifest containing every shard checksum, sequence
  count, exact token count, tokenizer revision, context length, split, category,
  source composition, filtering revision, and construction seed.
- [ ] First run the bridge with `--stop-after-packing`: validate all nine packed
  manifests and the checksummed per-document position indexes, and preserve its
  durable `phase: packed` journal. This stage must not guess GPU batch geometry.
- [ ] Mount the packed volume on the intended GPU topology and measure a
  memory/throughput smoke before choosing global microbatch rows and gradient
  accumulation.
- [ ] Build order format v4 with frozen global microbatch rows **and** gradient
  accumulation. Verify effective update rows, update count, the exact
  consumed/dropped input and supervised-token counters (including 40/40/20
  prefix accounting), and archive the full order-validator output.

## 8. Freeze the 1.3B model and optimization recipe

- [x] Plan the first production topology as one six-GPU RunPod pod, one NCCL
  process per GPU (`torchrun --standalone --nproc-per-node=6`). The exact GPU
  model, VRAM, and accepted DDP/FSDP strategy remain a measured hardware gate.
- [x] Record the exact architecture: layer count, hidden size, attention heads,
  key/value heads, MLP size, activation, normalization, positional encoding,
  vocabulary size, context length, tied embeddings, and parameter count.
- [ ] Confirm initialization rules and numerical precision. BF16 is preferred
  when supported; FP16 requires more care with loss scaling.
- [ ] Record AdamW betas/epsilon/weight decay, peak and minimum learning rates,
  warmup tokens or steps, decay schedule, gradient clipping, dropout, and random
  seeds.
- [ ] Choose global microbatch rows and gradient accumulation before freezing
  the order. Use the manifest's frozen optimizer-update count and exact
  `training_consumption.consumed_input_tokens`; record the derived per-rank
  microbatch and effective optimizer batch. For the planned pod, global rows
  must be divisible by six.
- [ ] Decide the distributed strategy and verify optimizer-state, activation,
  and communication memory against the selected GPU configuration.

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
- [ ] Decode and manually inspect several complete batches.
- [ ] Overfit one tiny batch; loss should fall sharply.
- [ ] Run a short single-GPU job; require decreasing training loss, finite
  gradients, correct checkpoints, and successful resume.
- [ ] Run a multi-GPU scaling test and measure tokens/s, GPU utilization, memory
  headroom, network-volume throughput, and communication overhead.
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

Only after these checks pass should the full 52.58B-token pre-training run
start. MBPP remains evaluation-only and is not consulted during pre-training.
