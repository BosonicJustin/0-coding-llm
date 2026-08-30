# Post-training dataset acquisition

The SFT corpus is kept in a dedicated post-training root. Never point this
downloader at the pre-training dataset root: it refuses known pre-training
markers and directories, and `--root` has no default.

The pinned source is
`nvidia/OpenCodeInstruct@8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`
with Hugging Face repository type `dataset`. Its closed raw inventory is:

- 50 `data/train-*.parquet` shards
- 6,861,113,102 compressed bytes
- 5,000,000 Parquet rows

## Download or resume

Install `requirements-data.txt`, choose a dedicated volume directory, and run
the same command after any interruption. `huggingface_hub.snapshot_download`
retains its resumable cache under the raw local directory. Authentication, if
the local Hugging Face setup needs it, comes from the normal environment or
credential store; the command contains no token.

```bash
SFT_ROOT=/path/to/posttraining-data/sft/opencodeinstruct

python scripts/download_sft_dataset.py \
  --root "$SFT_ROOT" \
  --max-workers 16 \
  2>&1 | tee "$SFT_ROOT.download.log"
```

The script logs download and per-shard inventory progress. It streams SHA-256
calculation, reads only Parquet metadata for row counts, and never rewrites or
converts raw shards. It publishes deterministic `SOURCE.json` and
`COMPLETION.json` only after every count passes. Re-running a complete dataset
works offline, rechecks all shards, and leaves identical authority files
untouched.

Final layout:

```text
opencodeinstruct/
├── raw/
│   └── data/train-*.parquet
├── SOURCE.json
└── COMPLETION.json
```

The raw Hugging Face resume cache may also exist under `raw/.cache/`; it is not
part of the certified 50-shard byte/row total. The pinned `README.md` and
`.gitattributes` are downloaded and checksummed as repository metadata.

## Certify the snapshot already on the server

For the existing download, no network access or redownload is needed:

```bash
python scripts/download_sft_dataset.py \
  --root /workspace/posttraining-data/sft/opencodeinstruct \
  --verify-only
```

`--verify-only` fails without publishing either authority if any shard is
missing, extra, changed during inspection, has the wrong aggregate byte or row
count, is not a regular file, or cannot be read as Parquet. If an existing
authority differs from the newly verified deterministic authority, the script
also fails rather than overwriting history; use a new root for a new source
generation.

## Quarantine before SFT

`COMPLETION.json` certifies acquisition integrity, not training suitability.
Keep this raw snapshot quarantined and immutable until a separate, versioned
post-training curation job has completed all of the following:

1. Preserve the dataset's CC-BY-4.0 attribution and review its repository card,
   provenance fields, and redistribution obligations. Record the pinned README
   checksum from `SOURCE.json` with every derived corpus.
2. Inspect the schema and reject malformed conversations, missing assistant
   targets, secrets/credentials, unsafe binary payloads, and unusable examples.
   Write transformations and rejection decisions to a new derived directory;
   never edit `raw/`.
3. Decontaminate against every frozen evaluation set—including MBPP,
   HumanEval/EvalPlus, and any later SFT/RL evaluation—using exact,
   normalized, and conservative semantic checks. Propagate contamination across
   duplicate groups before constructing train/validation splits.
4. Measure prompt, response, and combined lengths with the exact tokenizer used
   for SFT. Freeze a context-length policy before training: prefer dropping or
   explicitly structured truncation over silently cutting off an assistant
   solution, preserve EOS and loss-mask boundaries, and report retained tokens
   and examples by rejection/truncation reason.
5. Deduplicate and quality-score only under an explicit, versioned policy, then
   publish checksummed train/validation manifests. Do not point an SFT loader at
   this acquisition directory directly.
