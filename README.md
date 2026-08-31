# 0 Coding LLM

Personal research infrastructure for training a 1.284B-parameter decoder-only
coding model from scratch and evaluating it on MBPP without training on MBPP.

## Experiment contract

- Architecture: 24 layers, width 2,048, 16 query heads, 4 KV heads, SwiGLU,
  RMSNorm, RoPE, and a 49,152-token vocabulary.
- Context: 4,096 tokens with causal, block-diagonal attention across packed
  document boundaries.
- Tokenizer: pinned `bigcode/starcoder2-tokenizer` revision.
- Pre-training target: 52.58B train tokens plus immutable 0.50B validation and
  0.50B language-model test sets.
- Train mixture: 40% Python, 40% other programming languages, 20% English.
- Evaluation-isolation invariant: MBPP and every frozen final benchmark must be
  absent from training and checkpoint selection. MBPP filtering is already in
  the collection pipeline; final selected-corpus zero-match certification is
  still a pre-training gate.

The raw acquisition deliberately includes headroom. Final token budgets are
enforced only after quality filtering, contamination propagation, leakage-safe
group splitting, tokenization, and packing.

## Current status — 2026-08-31

- Raw acquisition is complete: 64.582B audited content tokens in 4,345
  finalized archives, with zero archive errors.
- Restart-safe accelerated curation is running with live SQLite/WAL state on
  pod-local storage and authenticated snapshots on the network volume. It
  performs quality/benchmark filtering, exact and normalized-hash
  canonicalization, leakage-safe grouping, split assignment, and exact mixture
  selection. The original network-volume generation remains an untouched
  rollback authority.
- Fuzzy English near-deduplication is intentionally deferred for the first
  baseline and remains available as a later controlled ablation.
- The native PyTorch model, packed loader, deterministic distributed sampler,
  materializer, checkpoint/resume harness, validation, W&B integration, and
  production launcher are implemented.
- The complete CPU suite passes 364 tests with two platform skips, including
  real two-process Gloo DDP accumulation, token-normalization, and
  process-restart gates. CUDA/NCCL qualification remains intentionally
  separate.
- CUDA FlexAttention, BF16, memory, throughput, and multi-GPU resume gates are
  still required before a long training run.
- The planned hardware topology is one six-GPU RunPod pod with one NCCL process
  per GPU; exact GPU type/VRAM and DDP versus FSDP remain smoke-test decisions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pretrain/` | Model, packed-data runtime, trainer, materializer, export, and run authority |
| `posttrain/` | SFT data contracts and Prime integration |
| `scripts/` | Auditable command-line entry points; see [scripts/README.md](scripts/README.md) |
| `configs/` | Versioned data, curation, and post-training policies |
| `docs/` | Categorized design, operations, and experiment records |
| `tests/` | CPU correctness, recovery, contract, and integration gates |

## Documentation

The complete categorized index is [docs/README.md](docs/README.md).

### Operations

- [Current operational handoff](docs/operations/handoff.md)
- [Pre-training readiness checklist](docs/operations/pretraining-checklist.md)
- [Production runbook](docs/operations/production-runbook.md)

### Pre-training

- [Data pipeline](docs/data/data-pipeline.md)
- [Curation contract](docs/data/curation.md)
- [Materialization](docs/data/materialization.md)
- [Training data and loader](docs/training/training-data.md)
- [Training harness](docs/training/training.md)
- [Model architecture](docs/training/model.md)
- [Pre-training run authority](docs/training/pretraining-run-authority.md)
- [Experiment log](docs/experiment/experiment-log.md)

### Post-training

- [SFT dataset acquisition and quarantine](docs/posttraining/posttraining-data.md)
- [Prime Intellect SFT data/model/runtime integration](docs/posttraining/prime-sft.md)
- [Prime Verifiers coding smoke environment](environments/coding_smoke/README.md)

## Local verification

Create an environment from the purpose-specific requirement files, then run:

```bash
venv/bin/python -m unittest discover -s tests
```

The real Gloo test opens a localhost transport and therefore needs local-socket
permission in restricted environments.

The checked-in GitHub Actions workflow repeats the complete CPU suite on
Python 3.12. CUDA/NCCL tests remain explicit hardware gates and are not
represented by the CPU workflow.

The current requirement files deliberately specify compatible version ranges.
Freeze the exact Python, PyTorch, CUDA, driver, and package lock only after the
six-GPU image passes the CUDA/NCCL smoke gate; that reproducible environment is
a required pre-training artifact and is not yet frozen.

## Roadmap

After pre-training, the same architecture and tokenizer can be used for SFT by
rendering instruction/response records into the 4,096-token window and masking
prompt labels. Packed SFT conversations can reuse document-boundary attention
isolation. RL environments and final benchmark execution remain separate from
pre-training data construction.

## Licensing

No license is currently granted for this repository's project source. Dataset
terms are separate from source-code terms and must be reviewed per pinned
source manifest. In particular, the downloaded OpenCodeInstruct snapshot is
CC BY 4.0 and remains quarantined until its attribution, contamination, and SFT
publication requirements in
[posttraining-data.md](docs/posttraining/posttraining-data.md) are satisfied.
