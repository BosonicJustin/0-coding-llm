# Final training-corpus qualification

Run `scripts/qualify_training_corpus.py` after the materializer has published
all three order-v4 splits and before authorizing the multi-day pre-training
launch. Packing can finish before GPU rental, but the final train order depends
on geometry measured during a short six-GPU qualification rental. The normal
sequence is therefore: pack on the data pod, qualify candidate GPU geometry,
finalize all three orders, run this corpus gate, then authorize the long run. A
zero exit status and an authenticated `status: pass` receipt are required. The
gate never modifies the corpus.

This is deliberately a full cold-path scan, not the launcher's metadata-only
preflight. It composes the canonical readers and validators already used by
training and checks the exact bytes that will be consumed:

- authenticates the top-level corpus manifest, checksum sidecar, and every
  provenance descriptor;
- pins the qualifier source files and Python/NumPy/PyTorch/SQLite/tokenizer/
  zstandard runtime, then proves that the validator, corpus, tokenizer, and
  optional model-config identities did not change during the scan;
- requires train, validation, and test, with Python, other code, and English
  present in each split;
- authenticates the pinned tokenizer and requires its vocabulary and EOS ID
  to agree with both the packed corpus and the model configuration;
- validates every order checksum, row reference, uniqueness constraint, token
  budget, and packed-manifest binding;
- validates every token/start shard size and SHA-256, every token ID, every
  start-bit padding bit, every EOS boundary, and the exact valid/masked loss
  counters;
- reconciles every document-position shard, including its hashes, source
  identity, token totals, and continuous logical stream offsets;
- performs an exact disk-backed audit proving that neither a source
  `(archive, member)` identity nor a leakage-safe group occurs in more than
  one split;
- requires the authorized 40% Python / 40% other-code / 20% English order
  mixture and the configured 52.58B/0.5B/0.5B input-token targets within the
  explicit acceptance tolerance;
- reads a deterministic random sample from every split/domain twice and proves
  byte stability, vocabulary bounds, boundary-only loss masking, and position
  resets; and
- requires a frozen train microbatch divisible across exactly six DDP ranks,
  at least one complete optimizer update, and zero padded/dropped train rows.

The existing `validate_training_order` and `validate_packed_manifest` passes
remain authoritative for binary semantics. The qualification script composes
them rather than implementing a competing packed-data parser.

## Production command

### CPU qualification environment

Do **not** invoke this command with
`/opt/coding-model-data-venv/bin/python`. That intentionally lightweight data
environment has no PyTorch, while the canonical packed/order validators and
`PackedBatchCollator` are PyTorch-native. Reimplementing them in the qualifier
would create a second, potentially divergent data contract. Create a dedicated
CPU-only qualification environment on the CPU pod instead. Preparing this
environment and executing the corpus scan require no GPU, although the final
order geometry must already have been frozen by the short GPU qualification:

```bash
python3 -m venv /opt/coding-model-qualification-venv

/opt/coding-model-qualification-venv/bin/python -m pip install \
  -r /workspace/0-coding-llm/requirements-qualification.txt

/opt/coding-model-qualification-venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.6,<3'
```

The receipt records the exact CPU PyTorch build. The later GPU run authority
still binds the exact CUDA PyTorch build used for training. They need compatible
data/collator semantics, not an identical local-version suffix.

Use fast pod-local NVMe for the temporary exact split audit. The SQLite file is
scratch only and is deleted after the receipt is produced. The receipt must be
outside the immutable corpus root.

```bash
mkdir -p /local/corpus-qualification-scratch
mkdir -p /workspace/dataset-other-code-topup-v2/qualification

/opt/coding-model-qualification-venv/bin/python -u \
  /workspace/0-coding-llm/scripts/qualify_training_corpus.py \
  --corpus-root /workspace/dataset-other-code-topup-v2/final/packed-v2 \
  --tokenizer-root /workspace/dataset-other-code-topup-v2/tokenizer/starcoder2 \
  --output /workspace/dataset-other-code-topup-v2/qualification/corpus-v2 \
  --scratch-directory /local/corpus-qualification-scratch \
  --world-size 6 \
  --sample-rows-per-domain 8 \
  --sample-seed 20260901
```

Omitting `--model-config` binds the checked-in `ModelConfig` defaults: 49,152
vocabulary entries and a 4,096-token context for the 1.284B model. If a frozen
model-config JSON is available, pass it with `--model-config`; the receipt then
records its path, SHA-256, and exact fields. A model configuration uses the
same exact schema accepted by `pretrain.hf_export.load_model_config_json`.

Defaults bind the planned input-token authority:

| Split | Expected model-input tokens | Maximum acceptance shortfall |
|---|---:|---:|
| Train | 52,580,000,000 | 0.1% |
| Validation | 500,000,000 | 0.1% |
| Test | 500,000,000 | 0.1% |

The order manifest may declare a tighter construction tolerance. The gate
checks both: the canonical order validator enforces its declared tolerance,
and the qualification policy independently limits the final shortfall. Change
`--maximum-target-shortfall-fraction` only as an explicit experimental
decision. The default absolute mixture tolerance is `1e-6`; order-v4's exact
largest-remainder allocation is checked separately.

## Result contract

`--output` names a fresh immutable generation directory, not a JSON file. The
command builds both files in a same-filesystem staging directory, fsyncs them,
and atomically renames that directory into place under an exclusive publisher
lock. It refuses to overwrite any existing complete, torn, symlinked, or raced
generation. A crash before the rename leaves no visible final generation; a
crash after it leaves both files.

The command always attempts to publish:

- `corpus-v2/qualification.json`; and
- `corpus-v2/qualification.json.sha256`.

On success it exits zero and the JSON contains `status: pass`, the exact corpus
manifest digest, tokenizer/model identities, per-split token and mixture
accounting, every packed/index shard identity, deterministic sample receipts,
and zero source/group cross-split collisions. On any validation failure it
exits nonzero and atomically publishes `status: fail` with the failure type and
message. Never treat a failed receipt, a missing sidecar, or console output
alone as launch evidence.

After a pass, do not mutate or regenerate anything under the corpus root. A
copy to local GPU storage must be requalified at the destination, because the
receipt authenticates the exact path's bytes at inspection time. The final
run-authority builder remains a separate mandatory step: it binds this data to
the clean Git revision, package lock, container digest, six-GPU hardware
qualification, training recipe, launcher argv, and cost authority.

## Runtime and capacity

The token/start and order scans are sequential and checksum every training
payload byte. Their lower bound is storage throughput. The document-index scan
also decompresses every compact provenance record. Split identities are stored
as 32-byte keys in a temporary `WITHOUT ROWID` SQLite database using batched
upserts and split bitmasks. Repeated group IDs are coalesced inside each bounded
batch before SQLite, so large repositories do not cause one database update per
file. RAM remains bounded and the result is exact (not a Bloom-filter estimate).
Put `--scratch-directory` on pod-local NVMe with ample free space; do not put
this temporary database on the network volume.

At 4,096 tokens, the 52.58B train cap is about 12.84 million order rows. The
canonical order validator memory-maps the roughly 103 MB `uint64` order and
uses about one byte per packed-available row for exact uniqueness, plus bounded
reference arrays when reconciling selected/surplus loss totals. The packed
semantic validator processes 4,096 rows at a time (tens of MiB), and closes one
shard mmap before proceeding. The split audit uses a 256 MiB SQLite page cache
and at most `--split-identity-batch-rows` distinct pending keys (100,000 by
default); its database grows on disk with selected document count. The receipt
contains shard summaries and sample hashes, never per-document identities.
Consequently the 52.58B-token scan is designed for low-single-digit-GiB RAM,
not corpus-sized RAM. Disk capacity for the exact identity database must still
be measured from the one-chunk/full-materializer document count before launch.

Qualification can take hours on the final corpus. It is intentionally paid
once on the CPU/data pod, while other independent preparation work continues.
Do not weaken checksum, semantic, document-index, or disjointness checks merely
to shorten the gate. If it fails, preserve the receipt and repair or rematerialize
the named artifact before renting GPUs.
