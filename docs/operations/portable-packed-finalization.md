# Finalize the packed-only GPU-pod handoff

Use `scripts/finalize_portable_packed_corpus.py` when the GPU pod has the
completed `packed-v1/` publication but intentionally does not have the raw
archives, preprocess payloads, or per-archive raw-token-cache payloads. Do not
invoke `scripts/materialize_training_corpus.py` on this handoff: its constructor
correctly requires those cold-path sources and cannot authenticate a packed-only
copy.

The portable finalizer does not tokenize or pack anything. It authenticates:

- the unchanged `phase: packed` materialization journal and all nine committed
  writer cursors;
- selection-v7 plus every decision bitmap;
- the closed-world two-file raw-token-cache inventory, without requiring the
  acceleration-only cache payloads;
- the tokenizer manifest and every tokenizer file;
- the pinned curation policy, v2 quota file, and MBPP denylist;
- all nine packed manifests and their local payload size contracts;
- all nine document-index manifests, every compressed index shard checksum,
  every record, and its exact packed/cursor token accounting; and
- either the S3 restore readiness receipt or, when explicitly requested, every
  local packed token/start payload checksum and semantic invariant.

The final provenance states explicitly that absent raw/preprocess/cache
payloads were not revalidated. Their identities remain transitively bound by
selection-v7, the cache inventory, the original packed journal, and the packed
manifests. The later full corpus qualification remains mandatory and scans the
actual training bytes again.

## Locate the restored authorities

The S3 restore root contains the packed corpus, selection-v7, the StarCoder2
tokenizer, a two-file cache-inventory directory, audit files, and
`.RESTORE_READY.json`. Resolve the exact restored paths once; do not copy
metadata out of those authenticated directories or rename files inside them.

```bash
REPO=/workspace/0-coding-llm
RESTORE=/workspace/pretraining-data
CORPUS="$RESTORE/packed-v1"
SELECTION="$RESTORE/selection-v7"
TOKENIZER="$RESTORE/tokenizer/starcoder2"
CACHE_INVENTORY=/absolute/path/to/restored/two-file-cache-inventory
RESTORE_READY="$RESTORE/.RESTORE_READY.json"

test -f "$CORPUS/.materialization-journal.json"
test -f "$SELECTION/manifest.json"
test -f "$SELECTION/manifest.sha256"
test -f "$TOKENIZER/TOKENIZER_MANIFEST.json"
test -f "$CACHE_INVENTORY/manifest.json"
test -f "$CACHE_INVENTORY/manifest.sha256"
test -f "$RESTORE_READY"
```

The cache-inventory directory contains exactly `manifest.json` and
`manifest.sha256`. It is not the absent `raw-all-v1/` cache-payload tree.

## Phase 1: publish held-out orders before geometry

This mode publishes validation and test only. For each split it selects the
largest exact 40/40/20, five-row-aligned, deterministic subset no larger than
500M positions. A supply-limited held-out split is intentionally smaller than
500M. The JSON result is the authority for its exact target.

```bash
cd "$REPO"

python -u scripts/finalize_portable_packed_corpus.py \
  --mode heldout \
  --corpus-root "$CORPUS" \
  --selection-root "$SELECTION" \
  --tokenizer-root "$TOKENIZER" \
  --cache-inventory-root "$CACHE_INVENTORY" \
  --restore-ready "$RESTORE_READY"
```

Expected state after a successful held-out run:

- `orders/validation/{order.bin,manifest.json}` exists;
- `orders/test/{order.bin,manifest.json}` exists;
- `orders/train/` does not exist;
- the original packed journal is byte-for-byte unchanged; and
- no top-level `manifest.json` has been published.

The command is idempotent. It revalidates an existing order and refuses a
different seed, target, dataset binding, or partial staging/final ambiguity.

If the restore was performed without the repository restore tool and therefore
has no readiness receipt, replace `--restore-ready ...` with
`--verify-packed-payloads`. That performs another full local read/hash and is
expected to be much slower.

## Phase 2: publish train only after geometry passes

The first production run is a capped Chinchilla-style run, not a full pass over
all packed train supply. With effective optimizer batch 192 rows, the frozen
train order is:

| Contract | Value |
|---|---:|
| Nominal cap | 52,580,000,000 input positions |
| Complete updates | 66,858 |
| Rows consumed | 12,836,736 |
| Input positions consumed | 52,579,270,656 |
| Alignment shortfall | 729,344 positions |
| Repeated rows | 0 |

The order uses deterministic strict-weight row allocation, selects without
replacement, contains only complete optimizer updates, and records every
unselected packed row as reusable packed surplus. The exact measured geometry
must preserve `GLOBAL_MICROBATCH_ROWS * ACCUMULATION = 192`, and global rows
must be divisible across six ranks.

```bash
GLOBAL_MICROBATCH_ROWS=MEASURED_GLOBAL_ROWS
ACCUMULATION=MEASURED_ACCUMULATION

python -u scripts/finalize_portable_packed_corpus.py \
  --mode final \
  --corpus-root "$CORPUS" \
  --selection-root "$SELECTION" \
  --tokenizer-root "$TOKENIZER" \
  --cache-inventory-root "$CACHE_INVENTORY" \
  --restore-ready "$RESTORE_READY" \
  --global-microbatch-rows "$GLOBAL_MICROBATCH_ROWS" \
  --gradient-accumulation-steps "$ACCUMULATION"
```

Final mode first revalidates or creates the held-out orders, then creates the
train order, source/policy/tokenizer/fingerprint/cache/handoff provenance, and
the top-level corpus manifest plus sidecar. It removes the packed journal only
after the completed publication validates. A mismatched geometry, insufficient
train-domain supply, changed artifact, unknown file, torn output, or checksum
failure stops before launch authority can be created.

## Mandatory next gate

Read the exact validation/test targets from their order manifests and pass them
to `scripts/qualify_training_corpus.py`; do not assume both equal 500M.

```bash
python - "$CORPUS" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for split in ("train", "validation", "test"):
    value = json.loads((root / "orders" / split / "manifest.json").read_text())
    budget = value["input_token_budget"]
    print(split, budget["expected_total"], budget["actual_total"], budget["tolerance"])
PY
```

The full corpus qualification, one-chunk overfit/resume gate, six-rank DDP
smoke, W&B smoke, checkpoint restore, and immutable run authority must all pass
before the multi-day launch.
