# Fast corpus generation v2

This is the current data-publication plan. It replaces exact, per-domain quota
selection inside SQLite with a much cheaper boundary: keep every eligible
canonical document, then make the packed `order.bin` files authoritative for
the 40/40/20 mixture and model-input budgets. The original dataset generation
is frozen; the v2 generation is built beside it.

## Current server state

State below was checked at **2026-09-01 06:40:36 UTC**.

- `/workspace/dataset` is the frozen v1 dataset root. No collection,
  preprocessing, curation, or token-materialization job is writing to it.
- The selective hard-link clone completed successfully at 06:30:51 UTC and
  atomically published `/workspace/dataset-other-code-topup-v2`. Its manifest
  SHA-256 is
  `815c6256f0354f1b6a6cc524d96e745331c68afd02f3e72b19bb2d66ed2b3de9`.
  It records 21,181 hard links totaling 72,337,391,686 bytes and five copied
  control files.
- The other-code collector was relaunched at 06:36:50 UTC in tmux
  `stack-v3-topup-v2` using `/opt/coding-model-data-venv`. All eight workers are
  active. Committed other-code supply advanced from 25,952,231,562 to
  26,066,335,409 tokens while Python remained exactly 25,770,142,666 tokens.
  A 06:32 launch failed on missing dependencies before touching data.
- The dedicated data environment passed 20 focused collector/clone tests. Its
  sorted `pip freeze` is
  `/workspace/dataset-other-code-topup-v2/logs/data-environment.freeze.txt`,
  SHA-256
  `c84c69f333754bfbd97b3ec851ec132916466a06424043349ae273429ef81bfb`.
  Key versions are `datasets==4.8.5`, `huggingface_hub==1.29.0`,
  `pyarrow==25.0.1`, `tokenizers==0.23.1`, and `zstandard==0.25.0`.
- Low-priority incremental preprocessing is active in tmux
  `preprocess-topup-v2-live`. It already authenticated and fingerprinted new
  archive `raw/other_code/part-000550.tar.zst` (27,966 documents and
  28,085,641 tokens) with zero benchmark hits; aggregate telemetry indexing is
  deliberately deferred.
- No curator is active. The previous local-WAL database is a stopped audit
  source, not a corpus publication: it is at
  `/local/curation/selection-fast-local-v2/curation.sqlite3` with its retained
  WAL and SHM sidecars. An unfinished NFS snapshot directory is not authority.
- **No token-ID materialization has started.** There are no production
  `tokens.bin`, packed shards, or production `order.bin` files yet.

The clone hard-links immutable raw archives, tokenizer files, source manifests,
collector restart state, quota ledgers, and completed preprocess
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

## Exact audited supply

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

These are pre-EOS source counts, not packed model-input counts. The current
other-code train supply cannot support 21.032B other-code training tokens. At a
strict 40/40/20 mixture, it limits the current train corpus to about 41.32B
tokens before row/EOS effects.

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

The target is a capacity estimate, not a guarantee: new documents can be
filtered or lose global canonical selection. The post-v2 supply audit decides
whether the final 52.58B order is feasible.

## Ordered v2 pipeline

### 1. Resume only other-code collection

This launch is now active. The command is recorded for exact recovery; do not
execute a second copy while `stack-v3-topup-v2` or its collector processes are
alive. It uses the v2 root and the same versioned quota file everywhere:

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

The cloned checkpoint and plan resume at their exact shard/repository cursors.
Python already exceeds its unchanged target, so the collector must report it as
full and skip it; only `other_code` may gain archives/tokens. Stop immediately
if Python totals or files increase. The collector atomically replaces the v2
closure record only after both unchanged Python and new other-code targets are
satisfied.

Monitor without touching v1:

```bash
tmux has-session -t stack-v3-topup-v2
pgrep -af 'collect_stack_v3_parallel.py|collect_stack_v3.py'
tail -n 50 /workspace/dataset-other-code-topup-v2/logs/collector-topup-v2.log
/opt/coding-model-data-venv/bin/python \
  /workspace/0-coding-llm/scripts/quota_tracker.py \
  --root /workspace/dataset-other-code-topup-v2 \
  --config /workspace/0-coding-llm/configs/data_quotas_other_code_topup_v2.json \
  status --phase collection --json
```

### 2. Incrementally preprocess new archives

Run this only after collection closes. The v2 clone already contains immutable
reports and fingerprints for v1 archives, so the preprocessor verifies and
skips those completed archive identities and processes only newly finalized
other-code archives. The quota path must remain the same v2 file used by the
collector.

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

Completion requires closed-collection status, exact ledger/raw/report/
fingerprint coverage, and zero errors. The optional telemetry SQLite index is
not required.

### 3. Rebuild canonicalization and leakage-safe groups

Do **not** copy or incrementally patch the v1 curation SQLite database. New
other-code documents can collide with existing exact/normalized hashes and can
change which deterministic canonical wins. Therefore v2 needs a fresh curation
generation over all hard-linked old inputs plus all new inputs, including full
inventory, quality/benchmark reasons, global exact/normalized canonicalization,
and leakage-safe group-to-split assignment.

Fuzzy English near-deduplication remains intentionally skipped for this first
baseline. Stack v3's upstream code near-deduplication remains the code boundary;
the local exact/normalized pass and benchmark propagation still run. Preserve
raw inputs so English fuzzy near-dedup can be tested later if results or a
duplicate audit justify it.

The production command for this step is intentionally not frozen here yet. It
must name a new v2 curation output/local-work generation, the v2 root and v2
quota config, and it must stop only after a complete, verified canonical/group
snapshot exists. Never point it at the stopped v1 database.

### 4. Publish all eligible canonical documents

Exact quota selection inside SQLite is skipped. The v7 publisher reads only a
complete immutable curation snapshot and emits, per raw archive, a compact keep
bitmap aligned one-for-one with source manifest rows. A bit is 1 exactly when
the document is the eligible canonical assigned to a leakage-safe split; a bit
is 0 for a recorded rejection. Every kept document is complete—there is no
quota-ending token prefix and no `quota_overflow` rejection. Partial state from
the abandoned exact selector is non-authoritative.

The resulting selection manifest records all nine split/domain document and
content-token totals. It does **not** claim a 40/40/20 mixture. Publisher and
materializer v7 qualification is still in progress, so do not launch this stage
until its production contract tests pass and the exact snapshot paths are
frozen in the run authority.

### 5. Tokenize, pack, then construct authoritative orders

Materialization reads each kept full document, rechecks its pinned token count,
tokenizes it once with the frozen StarCoder2 tokenizer, inserts EOS boundaries,
and writes separate packed streams for Python, other code, and English in each
split. Only then does deterministic packed order v4 select rows without
replacement and globally shuffle their references.

`orders/<split>/order.bin` plus its manifest—not the raw proportions, the v7
keep counts, or directory order—is the mixture/budget authority. Train targets
the largest optimizer-update-aligned whole-row prefix at or below 52.58B input
tokens with 40/40/20 row allocation. Validation and test each target the largest
feasible balanced 40/40/20 whole-row cap at or below 0.5B. If a held-out domain
cannot supply its nominal share, use the smaller balanced cap; do not download
more data solely to make a language-model held-out split exactly 0.5B.

No token-ID materialization may start before the v2 collection, incremental
preprocessing, full new canonical/group build, all-eligible publication, and
their production validations are complete.
