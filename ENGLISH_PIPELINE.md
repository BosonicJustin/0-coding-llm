# FineWeb-Edu raw English collector

This pipeline collects the FineWeb-Edu portion of the English pre-training
corpus. It is intentionally separate from the future Wikipedia collector so
cross-source deduplication and the final 80/20 mixture can be performed later.

## Budget

The final English allocation is 10.716B exact StarCoder2 tokens: 10.516B train,
0.100B validation, and 0.100B test. The aggregate acquisition quota is 12.8592B
exact tokens, providing 20% headroom for later deduplication, decontamination,
selection, and split construction.

The planned acquisition mixture is 80% FineWeb-Edu and 20% Wikipedia. This
collector therefore stops at **10.28736B** exact FineWeb-Edu tokens. The future
Wikipedia collector has a separate 2.57184B-token quota. Both source-specific
quota records also roll up into the aggregate 12.8592B English quota.

## Pinned source

- Repository: `HuggingFaceFW/fineweb-edu`
- Configuration: `sample-100BT`
- Revision observed during implementation:
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- Source files at that revision: 140 Parquet shards
- Dataset language: English
- Dataset-card license identifier: `odc-by`

The collector resolves `main` once for a new output root and writes the exact
revision, ordered source-file list, list digest, language/license metadata,
tokenizer revision, and benchmark-denylist hash to
`manifests/FINEWEB_EDU_SOURCE.json`. Every subsequent restart reuses and
validates that identity.

FineWeb-Edu is hosted as Parquet. Source Parquet is decoded through streaming
and is not saved to the network volume. Retained text is written byte-for-byte
to lossless raw archives:

```text
/workspace/dataset/
  raw/english/fineweb_edu/part-000000.tar.zst
  manifests/FINEWEB_EDU_SOURCE.json
  state/fineweb_edu_checkpoints/checkpoint-00000001.json
  logs/fineweb_edu_collector.log
  logs/fineweb_edu_benchmark_rejections.jsonl
```

Each archive includes `_manifest.jsonl` with the FineWeb document ID, Common
Crawl dump, URL, source WARC path, language score, education score, source token
count, byte count, and exact StarCoder2 count. Token IDs are not stored.

## Filtering

- `language` must equal `en`.
- `language_score` must be at least 0.65.
- Empty documents are rejected.
- URLs visibly associated with MBPP, EvalPlus, MultiPL-E, MBXP, MXEval, or
  HumanEval are quarantined.
- The MBPP content fingerprint guard is applied to English text before
  tokenization or storage. Rejection provenance and reason are logged without
  retaining rejected text.

FineWeb-Edu already supplies its own quality and educational scores. Broader
benchmark decontamination, exact and near deduplication against Wikipedia, and
final document selection remain separate downstream phases.

## Pilot

Use a separate root so the pilot cannot enter the production quota ledger:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/collect_fineweb_edu.py \
  --root /workspace/fineweb-edu-pilot \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --cache-dir /tmp/fineweb-edu-cache \
  --checkpoint-documents 20 \
  --max-new-documents 20 \
  --log-every-documents 5 \
  --min-free-gb 50
```

## Production

Do not start a second session with the same name. Start detached:

```bash
tmux new-session -d -s fineweb-edu \
  /workspace/coding_model_from_scratch/scripts/run_english_download.sh
```

Monitor it with:

```bash
tail -f /workspace/dataset/logs/fineweb_edu_collector.log
tmux attach -t fineweb-edu
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/quota_tracker.py \
  --root /workspace/dataset status --phase collection
```

Detach from `tmux` with `Ctrl-B`, then `D`. `SIGTERM` and `SIGINT` complete the
current document, close and fsync the archive, record the exact next cursor,
and exit. The collector automatically stops after crossing 10.28736B exact
FineWeb-Edu tokens or when less than 50 GB remains free. Completion is recorded in
`state/ENGLISH_FINEWEB_EDU_COMPLETE.json`.
