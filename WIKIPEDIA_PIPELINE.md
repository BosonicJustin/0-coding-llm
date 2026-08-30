# English Wikipedia raw collector

This pipeline supplies the remaining 20% of the English acquisition corpus.
It is independent from FineWeb-Edu so source-level quotas, provenance, and
later cross-source deduplication remain explicit.

## Budget and source

- Exact StarCoder2-tokenizer stop threshold: **2.57184B tokens**.
- Repository: `wikimedia/wikipedia`.
- Configuration: `20231101.en` (cleaned English article text).
- Source revision is resolved once and pinned in
  `manifests/WIKIPEDIA_SOURCE.json`.
- The source has 41 Parquet shards. They are streamed and decoded but are not
  retained on the network volume.
- Accepted text is stored byte-for-byte in lossless raw archives under
  `raw/english/wikipedia/`, with article ID, title, URL, byte count, and exact
  token count in each archive's `_manifest.jsonl`.

The collector rejects empty articles, quarantined benchmark URLs, and content
matching the MBPP fingerprint guard. Rejection provenance is logged without
storing rejected article text.

## Detached production job

```bash
tmux new-session -d -s wikipedia \
  /workspace/coding_model_from_scratch/scripts/run_wikipedia_download.sh
```

Monitor it with:

```bash
tail -f /workspace/dataset/logs/wikipedia_collector.log
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/quota_tracker.py \
  --root /workspace/dataset status --phase collection
```

The job checkpoints every 1 GB or 100,000 source rows, stops when its exact
quota is crossed, and stops safely if network-volume free space drops below
50 GB. `SIGINT` and `SIGTERM` close and fsync the current raw archive and record
the exact next source cursor before exiting.
