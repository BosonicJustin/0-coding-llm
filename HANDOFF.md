# Experiment handoff

This file is the short operational handoff. The detailed, authoritative
procedures remain in `PRODUCTION_RUNBOOK.md` and
`PRETRAINING_CHECKLIST.md`.

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

## Last verified live state at 2026-08-30 15:58 UTC

- `curation-fast-v1` is alive in detached `tmux` on the CPU pod.
- The curator is in `inventory`: 3,183,945 / 51,328,930 documents (6.2030%),
  34 archives have completed, and archive 35 is in progress, with no storage
  violation.
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
- The public repository is deployed separately at `/workspace/0-coding-llm`
  on commit `44a365b`; the active curator continues from its original checkout
  so deployment cannot mutate its running code.

This is the last authenticated server snapshot, not a claim about current
progress. A later connection reached the RunPod SSH gateway and the server
accepted the correct public key, but the local 1Password agent did not complete
the signing operation. Re-read `CHECKPOINT.json`, `tmux ls`, and `df` before
reporting a newer percentage or deploying another commit.

## Monitoring without mutating state

Inside the CPU pod:

```bash
tmux ls
pgrep -af 'curate_corpus.py|download_sft_dataset.py|hf download'
python3 -m json.tool \
  /workspace/dataset/curated/selection-fast-v1/.work/CHECKPOINT.json
df -h /workspace
```

The curation checkpoint is authoritative; the curation log may be empty because
progress commits are recorded in `CHECKPOINT.json` and SQLite.

## What follows automatically

The running curator advances its own bounded phases and resumes from durable
subphase cursors after an interruption. It must not be replaced with an ad hoc
filtering process. A storage violation, lease conflict, identity mismatch, or
accounting mismatch is a hard stop.

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
   boundary-perturbation gates in `PRIME_SFT.md`.

## Verified code state

The model/trainer stack includes real two-process CPU/Gloo gates for DDP token
normalization and atomic checkpoint → process restart → exact immutable-order
resume. Checkpoint format v5 stores only each rank's assigned CUDA RNG state
and binds every saved generation to the authenticated tokenizer-manifest and
canonical token-to-ID vocabulary SHA-256 identities.
Remaining gates are intentionally left unchecked in
`PRETRAINING_CHECKLIST.md`: some depend on the final curated/materialized data
and manual sample inspection; others require the eventual GPU image and
topology.
