# Command-line tools

Run commands from the repository root with the Python environment appropriate
to the stage. Production commands are ordered in
[the runbook](../docs/operations/production-runbook.md); this file is a
navigation index, not an alternative procedure.

## Acquisition and source identity

- `download_tokenizer.py` — pin and authenticate the tokenizer.
- `collect_stack_v3.py`, `collect_stack_v3_parallel.py` — collect code.
- `collect_fineweb_edu.py`, `collect_wikipedia.py` — collect English sources.
- `quota_tracker.py`, `count_shard_tokens.py` — enforce and inspect quotas.
- `run_download.sh`, `run_english_download.sh`, `run_wikipedia_download.sh` —
  resumable server wrappers.

## Audit, contamination, and curation

- `preprocess_raw_stream.py`, `run_preprocess.sh` — raw archive audit and
  fingerprint preprocessing.
- `curation_policy.py`, `audit_mbpp.py`, `benchmark_guard.py` — policy and
  benchmark-isolation gates.
- `build_mbpp_denylist.py` — build the non-reversible MBPP denylist authority.
- `calibrate_english_near_dedup.py`, `build_english_near_clusters.py` — optional
  English near-duplicate workflow.
- `curate_corpus.py` — restart-safe filtering, canonicalization, grouping,
  split assignment, and quota selection.
- `monitor_curation.py` — live fail-closed health projection.
- `curation_inventory_stage.py`, `curation_local_store.py` — supporting
  curation storage modules; normally invoked through `curate_corpus.py`.

## Materialization and training data

- `cache_raw_tokens.py` — optional, curation-independent, per-archive token
  cache builder and one-archive benchmark; outputs are not directly trainable.
- `materialize_training_corpus.py` — curation-to-packed publication bridge.
- `build_training_order.py` — deterministic order-v4 construction.
- `validate_training_data.py`, `certify_pretraining_data.py` — semantic and
  full-payload certification.
- `smoke_raw_to_training_data.py`, `smoke_training_data.py` — bounded pipeline
  and loader gates.

## Pre-training

- `qualify_cuda_model.py` — CUDA FlexAttention correctness gate.
- `overfit_single_chunk.py` — packed-boundary memorization and exact-resume gate,
  supporting one process or production-style DDP under torchrun.
- `benchmark_training_loader.py` — loader and host-to-device calibration.
- `launch_pretraining.py` — production preflight, signal supervisor, and
  six-rank launcher.
- `build_pretraining_run_authority.py` — immutable environment, recipe,
  geometry, cost, and launch authorization.
- `export_hf_checkpoint.py` — authenticated native-to-Hugging-Face export.

## Post-training

- `download_sft_dataset.py` — isolated SFT source acquisition.
- `prepare_prime_sft.py` — OpenCodeInstruct decontamination and publication.
- `apply_prime_renderer_patch.py` — install the pinned renderer integration.
- `launch_prime_sft.py` — Prime SFT preflight and launcher.

## Benchmarks and storage qualification

- `benchmark_preprocess_archive.py` — raw preprocessing throughput.
- `benchmark_curation_inventory_stage.py` — inventory implementation benchmark.
- `probe_sqlite_storage.py` — SQLite filesystem and mount qualification.

Use `python scripts/<tool>.py --help` for an individual command's exact
contract. Do not invoke supporting modules directly unless their documentation
explicitly calls for it.
