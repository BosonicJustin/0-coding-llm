# Fast corpus generation v2

This is the generation-v2 design and build record. It replaced exact,
per-domain quota selection inside SQLite with a cheaper boundary: keep every
eligible canonical document, then make the packed `order.bin` files
authoritative for the 40/40/20 mixture and model-input budgets. The original
dataset generation remains frozen; generation v2 was built beside it.

For current corpus identity and launch status, use the
[pre-training corpus record](final-corpus-record.md). The operational commands
below are retained as build history and recovery documentation; they are not
instructions to restart completed collection, preprocessing, curation,
selection, token-cache, or packing jobs.

## Current outcome — 2026-09-02

- Collection, per-archive preprocessing, corpus curation, selection-v7, the
  closed-world raw-token cache, and all nine split/domain packed streams are
  complete. The packed journal is at phase `packed` with all 4,568 archives
  committed.
- The portable recovery copy is
  `s3://transcendent-logic-data-618079239540/coding-llm/pretraining/2026-09-02-packed-v1/`.
  It is deliberately a **packed-only** artifact: it contains the authenticated
  packed data and provenance needed for recovery, but no final split orders or
  top-level production corpus manifest.
- That portable artifact has been restored with checksum verification to
  six-H100 pod-local NVMe at `/root/transcendent-logic-data`. The checkout used
  to qualify and consume it is `/root/0-coding-llm`.
- Deterministic train, validation, and test orders and the final top-level
  corpus manifest remain pending six-GPU geometry qualification and portable
  finalization. No stage may retokenize documents or rewrite packed shards.
- The first training trajectory is one pass through a selected 40/40/20 order:
  exactly 12,836,736 unique packed-row references, 66,858 complete
  192-row optimizer updates, and 52,579,270,656 input positions. Selection is
  without replacement; no row repeats. Packed surplus is reserved for later
  experiments rather than appended to this run.
- Fuzzy/semantic near-deduplication remains deliberately deferred for this
  baseline. The completed corpus uses Stack v3's upstream code deduplication
  plus global exact and normalized-identical canonicalization.

The completed clone hard-links immutable raw archives, tokenizer files, source
manifests, collector restart state, quota ledgers, and completed preprocess
reports/fingerprints. It copies mutable closure/status files to independent
inodes and excludes logs, locks, temporary files, telemetry SQLite, curation,
and packed outputs. Existing hard-linked files must never be edited in place.
All new archives and derived publications are created only under v2 paths.

The completed clone authority validates with:

```bash
DATA_V2=/workspace/dataset-other-code-topup-v2
test -f "$DATA_V2/CLONE_MANIFEST.json"
(cd "$DATA_V2" && sha256sum -c CLONE_MANIFEST.sha256)
test "$(sha256sum "$DATA_V2/CLONE_MANIFEST.json" | cut -d' ' -f1)" = \
  815c6256f0354f1b6a6cc524d96e745331c68afd02f3e72b19bb2d66ed2b3de9
jq -e '.complete == true and
       .kind == "selective-hardlink-dataset-generation-clone" and
       .inventory.hardlinked_files == 21181 and
       .inventory.hardlinked_bytes == 72337391686 and
       .inventory.copied_control_files == 5 and
       .other_code_topup_plan.target_cumulative_raw_tokens == 35000000000' \
  "$DATA_V2/CLONE_MANIFEST.json" >/dev/null
test "$(sha256sum "$DATA_V2/logs/data-environment.freeze.txt" | cut -d' ' -f1)" = \
  c84c69f333754bfbd97b3ec851ec132916466a06424043349ae273429ef81bfb
```

## Historical pre-top-up supply audit

The WAL-aware, read-only audit completed in 255.2 seconds with `safe: true`,
zero accounting anomalies, 49,461,115 eligible canonical documents, and
14,968,335 leakage-safe groups. Its immutable result is:

```text
/workspace/dataset/audits/supply-audit-fast-v1-20260901/supply-audit.json
SHA-256 74de45bdf3438395f74f6c492f11017e6c0be6b76e0f08ec73e88a0b77169230
```

The sibling `.sha256` and log sidecars are in the same directory. Exact
eligible **content-token** counts and document counts are:

| Split | Domain | Documents | Content tokens |
|---|---|---:|---:|
| Train | Python | 22,757,415 | 22,914,037,162 |
| Train | Other code | 15,422,386 | 16,527,423,703 |
| Train | English | 10,382,610 | 12,244,232,284 |
| Validation | Python | 212,616 | 199,245,719 |
| Validation | Other code | 139,251 | 134,812,854 |
| Validation | English | 99,123 | 116,983,633 |
| Test | Python | 212,653 | 209,365,037 |
| Test | Other code | 136,162 | 149,115,274 |
| Test | English | 98,899 | 117,502,816 |
| **All splits** | **Python** | **23,182,684** | **23,322,647,918** |
| **All splits** | **Other code** | **15,697,799** | **16,811,351,831** |
| **All splits** | **English** | **10,580,632** | **12,478,718,733** |

These are pre-EOS source counts from the frozen v1 audit, not the completed v2
packed supply and not model-input counts. They explained the other-code top-up:
at that point, other-code train supply could not support 21.032B other-code
training tokens and limited a strict 40/40/20 corpus to about 41.32B tokens
before row/EOS effects. The completed v2 packed supply is recorded in
[the canonical corpus record](final-corpus-record.md#packed-supply).

## Why the v2 other-code target is 35B

The versioned authority is
`configs/data_quotas_other_code_topup_v2.json`. It changes only
`collection/other_code`, from 25,718,400,000 to 35,000,000,000 raw tokens.
Python, both English source targets, and the reference final budgets are
unchanged.

The calculation uses measured retained train yield rather than the nominal 20%
headroom assumption:

| Quantity | Tokens |
|---|---:|
| Audited raw other code | 25,952,231,562 |
| Eligible train other code | 16,527,423,703 |
| Required train other code | 21,032,000,000 |
| Train shortfall | 4,504,576,297 |
| Raw increment at measured 63.684% train yield | 7,073,323,057 |
| Planned raw increment to a 35B cumulative target | 9,047,768,438 |
| Headroom over the point estimate | 27.914% |
| Expected eligible-train surplus if yield repeats | 1,257,406,143 |

The target was a capacity estimate rather than a guarantee: new documents could
be filtered or lose global canonical selection. The completed v2 packed supply
now proves that the selected 52,579,270,656-position order is feasible.

## Ordered v2 pipeline record

### 1. Other-code collection — complete

The collector completed against the 35B cumulative other-code target while
Python and English remained frozen. The launch command is retained for exact
provenance; do not rerun it against the closed corpus:

```bash
PROJECT_ROOT=/workspace/0-coding-llm
DATA_V2=/workspace/dataset-other-code-topup-v2
QUOTAS="$PROJECT_ROOT/configs/data_quotas_other_code_topup_v2.json"

tmux new-session -d -s stack-v3-topup-v2 -c "$PROJECT_ROOT" \
  "env DATA_ROOT=$DATA_V2 QUOTA_CONFIG=$QUOTAS \
    PYTHON_BIN=/opt/coding-model-data-venv/bin/python \
    LOG_FILE=$DATA_V2/logs/collector-topup-v2.log \
    $PROJECT_ROOT/scripts/run_download.sh"
```

The cloned checkpoint and plan resumed at their exact shard/repository cursors.
Python was already above its unchanged target and was skipped; only
`other_code` gained archives/tokens. The collector published the v2 closure
record only after the unchanged Python/English targets and the new other-code
target were satisfied.

### 2. Incremental preprocessing — complete

After collection closed, the preprocessor verified and skipped immutable v1
archive identities and processed only newly finalized other-code archives. It
used the same v2 quota file as the collector. The historical launch command is
retained for provenance, not restart:

```bash
PROJECT_ROOT=/workspace/0-coding-llm
DATA_V2=/workspace/dataset-other-code-topup-v2
QUOTAS="$PROJECT_ROOT/configs/data_quotas_other_code_topup_v2.json"

tmux new-session -d -s preprocess-topup-v2 -c "$PROJECT_ROOT" \
  "env DATA_ROOT=$DATA_V2 QUOTA_CONFIG=$QUOTAS \
    PYTHON_BIN=/opt/coding-model-data-venv/bin/python \
    LOG_FILE=$DATA_V2/logs/preprocess-topup-v2.log \
    $PROJECT_ROOT/scripts/run_preprocess.sh"
```

Completion was accepted only with closed-collection status, exact
ledger/raw/report/fingerprint coverage, and zero errors. The optional telemetry
SQLite index was not required.

### 3. Canonicalization and leakage-safe groups — complete

Do **not** copy or incrementally patch the v1 curation SQLite database. New
other-code documents can collide with existing exact/normalized hashes and can
change which deterministic canonical wins. Therefore v2 used a fresh curation
generation over all hard-linked old inputs plus all new inputs, including full
inventory, quality/benchmark reasons, global exact/normalized canonicalization,
and leakage-safe group-to-split assignment.

Fuzzy English near-deduplication remains intentionally skipped for this first
baseline. Stack v3's upstream code near-deduplication remains the code boundary;
the local exact/normalized pass and benchmark propagation still run. Preserve
raw inputs so English fuzzy near-dedup can be tested later if results or a
duplicate audit justify it.

The production command and completion checks used for the build are frozen in
[Fast all-eligible curation runbook](fast-all-eligible-curation.md).
It uses `--fast-all-eligible-handoff`: exact quotas and legacy decisions are
never created, periodic full NFS snapshots are suppressed, and one final
immutable LocalSQLiteStore snapshot is returned for the v7 publisher. Never
point it at the stopped v1 database.

### 4. Publish all eligible canonical documents — complete

Exact quota selection inside SQLite is skipped. The v7 publisher reads only a
complete immutable curation snapshot and emits, per raw archive, a compact keep
bitmap aligned one-for-one with source manifest rows. A bit is 1 exactly when
the document is the eligible canonical assigned to a leakage-safe split; a bit
is 0 for a recorded rejection. Every kept document is complete—there is no
quota-ending token prefix and no `quota_overflow` rejection. Partial state from
the abandoned exact selector is non-authoritative.

The resulting authenticated selection-v7 manifest records all nine
split/domain document and content-token totals. It does **not** claim a
40/40/20 mixture; that remains an order-level property. All kept documents are
complete.

### 5. Token cache and packing complete; authoritative orders pending

Materialization read each kept full document from the authenticated raw-token
cache, rechecked its pinned token count, inserted EOS boundaries, and wrote
separate packed streams for Python, other code, and English in each split. The
cache, all nine packed manifests, immutable token/start shards, and packed-phase
journal are complete and authenticated. Deterministic packed order v4 is the
remaining step; it will select existing rows without replacement and globally
shuffle their references after geometry qualification.

`orders/<split>/order.bin` plus its manifest—not the raw proportions, the v7
keep counts, or directory order—will be the mixture/budget authority. The
frozen train decision is 12,836,736 unique rows, 66,858 complete updates at 192
rows per update, and exactly 52,579,270,656 input positions under the 40/40/20
allocation, with no replacement and no repeats. Validation and test each use
the largest feasible balanced 40/40/20 whole-row cap at or below 0.5B. If a
held-out domain cannot supply its nominal share, use the smaller balanced cap;
do not download more data solely to make a language-model held-out split
exactly 0.5B.

Finalization must operate only on the existing packed rows. It must not rebuild
the token cache, retokenize source documents, reorder tokens within a row,
change split membership, or rewrite any packed shard. Until the order manifests
and top-level corpus manifest are published and authenticated, the packed-only
artifact is recoverable data but is not launch authority.
