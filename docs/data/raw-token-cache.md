# Curation-independent raw token cache

`pretrain/raw_token_cache.py` and `scripts/cache_raw_tokens.py` provide an
optional acceleration layer between finalized raw preprocessing and final
curation/materialization. The cache can be built while later archives are
still downloading and being fingerprinted because each archive is an
independent immutable unit.

This cache is intentionally **not directly trainable**. It has no curation
decision, split, mixture, shuffle order, EOS boundary, packing, padding, or
attention-mask authority. A future materializer may consume it only after it
joins each document ordinal to an authenticated curation publication and then
performs those missing steps.

## Per-archive schema

For report `reports/<bucket>/part-NNNNNN.json`, the output is:

```text
token-cache/raw-all-v1/archives/<bucket>/part-NNNNNN/
  tokens.u16
  offsets.u64
  manifest.json
  manifest.sha256
```

- `tokens.u16` is the concatenation of every complete document's content token
  IDs in raw internal-manifest order. Values are little-endian `uint16`.
- `offsets.u64` is little-endian `uint64`, begins with zero, contains exactly
  `documents + 1` values, and ends at the number of content tokens.
- `manifest.json` binds raw archive path/index/bucket/bytes/SHA-256, preprocess
  report path/bytes/SHA-256, fingerprint path/bytes/SHA-256, tokenizer repo and
  resolved revision, tokenizer manifest/vocabulary/EOS identities, exact
  document/byte/token totals, both binary payload hashes, manifest-index
  alignment, and builder/config identities.
- `manifest.sha256` authenticates the canonical manifest bytes. The manifest
  authenticates both payloads and all three source inputs.

There are no added special tokens. In particular, EOS is pinned as tokenizer
provenance but is absent from `tokens.u16`. There is no padding or truncation.

## Algorithm and safety boundary

For each canonical, finalized preprocess report:

1. Reject symlinks, non-regular inputs, unsafe paths, unsupported versions,
   noncanonical report/fingerprint paths, invalid counts, and raw-size drift.
2. Authenticate the complete pinned `bigcode/starcoder2-tokenizer` directory,
   its 40-character resolved revision, 49,152-token vocabulary mapping, and
   EOS identity. Explicitly disable tokenizer padding and truncation.
3. Acquire the cache-root lock. Parallel workers process different archives;
   byte and token caps keep each worker's document and tokenizer batches
   bounded.
4. Stream the raw `.tar.zst` and matching compressed fingerprint together.
   Reject non-file tar members (including symlinks), duplicate members,
   documents after the internal manifest, invalid UTF-8, content/member/order
   mismatches, corruption, and fingerprint/token/count mismatches.
5. Tokenize every document with `add_special_tokens=False`. Append full token
   IDs and one terminal offset. IDs must be integers below both 65,536 and the
   pinned vocabulary size. The tokenizer result must exactly equal the token
   count recorded by preprocessing.
6. Compare a disk-backed observed-document spool to every internal-manifest
   row. This validates exact source manifest order without retaining an
   unbounded metadata list in RAM.
7. Recheck source stat identities and SHA-256 values after the stream, verify
   archive totals, fsync both payloads, write the canonical manifest/sidecar,
   fsync the staging directory, and atomically rename it into place.
8. On restart, remove only this archive's recognizable incomplete staging
   directories. For a completed target, rehash the raw archive, report,
   fingerprint, tokens, and offsets; validate monotonic offsets and token-ID
   range; then return `verified` without retokenizing.

The output estimate is approximately two bytes per content token plus eight
bytes per document boundary. The command reserves additional free space before
workers start.

## Commands

Bounded one-archive benchmark (use a fresh output root so it measures a build,
not completed-cache verification):

```bash
venv/bin/python scripts/cache_raw_tokens.py benchmark \
  --root /workspace/dataset \
  --preprocess-root /workspace/dataset/staging/preprocess \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --report /workspace/dataset/staging/preprocess/reports/python/part-000000.json \
  --output-root /local/raw-token-cache-benchmark-v1
```

A conservative server launch that yields CPU and I/O to the live collector and
preprocessor is:

```bash
mkdir -p /workspace/dataset/logs
tmux new-session -d -s raw-token-cache-v1 \
  "cd /workspace/0-coding-llm && exec nice -n 15 ionice -c 2 -n 7 \
   venv/bin/python -u scripts/cache_raw_tokens.py build \
   --root /workspace/dataset \
   --preprocess-root /workspace/dataset/staging/preprocess \
   --tokenizer /workspace/dataset/tokenizer/starcoder2 \
   --output-root /workspace/dataset/token-cache/raw-all-v1 \
   --workers 2 >>/workspace/dataset/logs/raw-token-cache-v1.log 2>&1"
```

This is a one-shot discovery of reports that are complete when the command
starts. Run it again after preprocessing closes to cover later reports. The
default rerun deliberately rehashes completed caches rather than trusting file
names. Use a new cache root if the raw source or tokenizer generation changes;
the builder refuses to overwrite an existing cache bound to another identity.

