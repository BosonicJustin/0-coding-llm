# Data formats from source download to PyTorch batches

This document explains the physical and logical representation of the
pre-training corpus at each stage. It distinguishes source Parquet, retained
raw archives, preprocessing metadata, curation metadata, cached token IDs,
packed training rows, and the tensors seen by the model.

## End-to-end view

```text
Hugging Face Parquet -> collection -> retained raw .tar.zst archives
raw archives -> preprocessing -> reports and fingerprints
reports/fingerprints -> SQLite curation -> selection-v7
raw archives + reports/fingerprints -> authenticated raw token cache
selection-v7 + token cache -> packed token/start shards
packed shards -> split order files -> PyTorch [B, 4096]
```

The stages are deliberately separate. In particular, the SQLite database does
not contain source text or token arrays, and the training hot path does not
read Parquet, tar archives, JSONL, SQLite, or raw text.

## Current physical boundary — 2026-09-02

Collection, preprocessing, corpus-wide curation, selection-v7, the closed-world
raw-token cache, and all nine packed split/domain streams are complete. The
portable packed-phase recovery artifact is
`s3://transcendent-logic-data-618079239540/coding-llm/pretraining/2026-09-02-packed-v1/`;
it has been restored with checksum verification to
`/root/transcendent-logic-data` on the six-H100 pod.

That S3 artifact is intentionally **packed-only**. In the end-to-end diagram,
the pipeline has reached `packed token/start shards`; deterministic split-order
files and the final top-level corpus manifest remain pending geometry
qualification and finalization. Their documented shapes below describe the
contract they must satisfy, not files already present in the portable backup.
Finalization may select and shuffle existing row references only: it may not
retokenize, rewrite a packed shard, reorder tokens inside a row, or change split
membership.

The first train-order decision is one pass through 12,836,736 unique selected
row references under the 40/40/20 allocation. At 192 rows per optimizer update,
that is 66,858 complete updates and exactly 52,579,270,656 input positions.
Selection is without replacement, so no row repeats. Fuzzy/semantic near-dedup
remains a deliberately deferred baseline ablation; completed curation includes
global exact and normalized-identical canonicalization.

## 1. Source Parquet

Parquet is a typed, compressed, column-oriented table format. Hugging Face uses
it to distribute large datasets efficiently. The collectors stream pinned
source shards from:

- `HuggingFaceCode/stack-v3-train` for Python and other code;
- `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`, for educational
  English text;
- `wikimedia/wikipedia`, configuration `20231101.en`, for English Wikipedia.

The source Parquet files are not retained on the dataset volume. A collector
reads source rows, applies the configured admission rules, extracts accepted
code or text, and writes the accepted bytes directly into our raw archive
format. This is filtered extraction and repackaging, not a byte-for-byte copy
of the complete Parquet table.

For Stack v3, a source row describes a repository and contains multiple files.
Every accepted file becomes one raw archive member. For FineWeb-Edu and
Wikipedia, one accepted source row becomes one text member.

## 2. Retained raw `.tar.zst` archives

The retained raw corpus lives under paths such as:

```text
raw/python/part-000000.tar.zst
raw/other_code/part-000000.tar.zst
raw/english/fineweb_edu/part-000000.tar.zst
raw/english/wikipedia/part-000000.tar.zst
```

`tar` combines many documents into one sequential archive. Zstandard (`zst`)
compresses that archive losslessly and supports fast streaming decompression.
This avoids millions of loose filesystem files while preserving the original
UTF-8 bytes of every retained document.

A code archive has the following conceptual layout:

```text
files/<repo-id>/000000000-<content-id>.py
files/<repo-id>/000000001-<content-id>.py
...
_manifest.jsonl
```

English archives use members such as:

```text
documents/000000000-<source-id-hash>.txt
documents/000000001-<source-id-hash>.txt
...
_manifest.jsonl
```

The document members contain raw code or text, not JSON. `_manifest.jsonl` is
our own provenance index, written as the final archive member. It is not a
manifest supplied by Hugging Face. Each JSON line describes exactly one
preceding document. A code row resembles:

```json
{
  "member_path": "files/123/000000000-abc.py",
  "repo_path": "owner/project",
  "repo_id": "123",
  "commit_id": "...",
  "file_path": "src/example.py",
  "content_id": "abc",
  "language": "Python",
  "license_type": "permissive",
  "detected_licenses": ["MIT"],
  "size_bytes": 1842,
  "starcoder2_tokens": 476
}
```

The raw archives are the recoverable source of truth. Keep them through final
corpus qualification and at least the first successful pre-training run.

## 3. Per-archive preprocessing

Preprocessing describes and authenticates each document independently. For
every finalized archive it:

1. streams the Zstandard and tar layers without extracting the entire archive;
2. requires regular document members, a final internal manifest, and the
   expected member/manifest order;
3. parses and validates every `_manifest.jsonl` row;
4. checks member path, byte length, document count, clean-byte total, and pinned
   StarCoder2 token count;
5. computes content identities, normalized identities, grouping metadata,
   benchmark signals, and compact duplicate sketches;
6. publishes a checksummed report and fingerprint shard.

The raw archive is never edited. Collection ran the tokenizer to obtain the
`starcoder2_tokens` counts stored in the internal manifest. Preprocessing
authenticates and totals those pinned counts but does not retain or recompute
the complete token-ID arrays. Completed reports and fingerprints are immutable
and can be reused when the corresponding raw archive and policy identities
have not changed.

Collector-generated member names are unique by archive-global ordinal and are
expected to be safe under the pinned source schema. This preprocessing stage
does not independently prove path safety. Curation enforces unique
`(archive, member_path)` identities. The raw-token-cache builder also performs
explicit safe-path and duplicate-name validation. The direct raw materialization
fallback instead requires streamed names and order to match the authenticated
fingerprint and selection records exactly.

## 4. Corpus-wide curation

Curation combines the preprocessing metadata for the complete closed corpus.
The SQLite database stores identities, counters, source locations, grouping
keys, and decisions; it does not store raw document text or token arrays.

The current baseline performs:

1. complete document inventory and source-accounting checks;
2. basic eligibility and benchmark-denylist propagation;
3. global exact and normalized-identical canonicalization;
4. stable repository or English-source grouping;
5. leakage-safe train, validation, and test assignment at group granularity;
6. publication of an immutable authenticated SQLite snapshot.

Upstream dataset quality filtering remains the primary quality boundary. The
baseline intentionally skips expensive fuzzy semantic near-deduplication and
does not rerun a large learned quality classifier.

This stage must be recomputed after expanding the closed corpus because a new
document can collide with an existing global identity or change which copy is
canonical. Per-archive preprocessing does not need to be repeated for unchanged
archives.

## 5. Selection bitmaps

The selection-v7 publisher uses the completed immutable curation snapshot as
its decision authority. It also reopens and authenticates the snapshot-bound
raw archives, reports, fingerprints, tokenizer, policy, quota configuration,
and source authorities. For each raw archive it writes a bit aligned
one-for-one with the raw manifest rows:

```text
0 = reject this document
1 = keep this complete document
```

The bitmap payload contains only keep bits; it does not encode a split or
domain. Archive/category provenance supplies the source category, while the
materializer re-derives and verifies the leakage-safe split from authenticated
group provenance and the frozen split policy. The selection manifest records
exact document and content-token supply for all nine cells:

```text
{train, validation, test} x {python, other_code, english}
```

The bitmap does not contain source text or token IDs and does not itself impose
the final 40/40/20 training order.

## 6. Raw token cache

Token caching runs the pinned StarCoder2 tokenizer and persists the actual token
IDs so later materialization does not tokenize the same documents again. For
each raw archive, the cache stores conceptually:

```text
tokens.u16  = concatenated document token IDs, little-endian uint16
offsets.u64 = document start/end offsets, little-endian uint64
manifest.json + manifest.sha256
```

For example:

```text
document A tokens = [21, 84, 502]
document B tokens = [17, 91]

tokens  = [21, 84, 502, 17, 91]
offsets = [0, 3, 5]
```

The cache preserves raw manifest order and exact document boundaries. It does
not select documents, insert EOS, assign splits, shuffle rows, pad sequences,
or define attention. It is authenticated derived acceleration data, not the
training corpus.

## 7. Packed training rows

The production materializer joins the authenticated selection and token cache,
inserts exactly one EOS after every selected document, and writes nine
independent packed streams. A slower authenticated fallback can read raw
archives and run the tokenizer directly, so the cache is an acceleration layer
rather than a semantic requirement. No BOS token is inserted before each
document. Independent attention-segment starts are represented by a separate
bitset.

For context length `T`, one physical packed row contains:

- `T + 1` little-endian `uint16` token IDs;
- `ceil((T + 1) / 8)` little-bit-order document/segment-start bits.

For `T = 4096`, that is 4,097 token IDs plus a 513-byte start bitset, or 8,707
bytes per physical row. The extra token supplies the next-token target for the
last model-input position without padding. Bit zero is forced on for every
physical row, even when that row begins with a continuation chunk from a long
document, because attention cannot cross independently processed rows.
Adjacent rows advance by exactly `T` logical tokens, so the stored lookahead at
the end of one row is duplicated as the first model input of the next row.

Paired shard files are raw, memory-mappable binaries:

```text
packed/train/python/
  shard-000000.tokens.bin
  shard-000000.starts.bin
  manifest.json
```

An abbreviated output layout is:

```text
packed/{train,validation,test}/{python,other_code,english}/
  shard-NNNNNN.tokens.bin
  shard-NNNNNN.starts.bin
  manifest.json
orders/{train,validation,test}/
  order.bin       # pending geometry-bound finalization
  manifest.json   # pending geometry-bound finalization
provenance/documents/{train,validation,test}/{python,other_code,english}/
  archive-NNNNNN.jsonl.zst
  manifest.json
provenance/
  source.json
  policy.json
  tokenizer.json
  fingerprints.json
  raw_token_cache.json  # cache-backed route only
manifest.json     # pending top-level finalization
manifest.sha256   # pending top-level finalization
```

When finalized, `order.bin` stores little-endian `<u8` row references. The high
eight bits encode the domain ID and the low 56 bits encode the domain-local row
ID. It
globally shuffles selected physical rows while enforcing the intended 40%
Python, 40% other-code, and 20% English mixture. Shuffling never changes token
order inside one physical row, but it can change traversal order between rows,
including separate row-sized chunks originating from one long document. The
current packed-only artifact contains no authoritative `order.bin`; consumers
must fail closed rather than infer traversal order from shard or directory
order.

## 8. Runtime PyTorch tensors

For one stored token row `t[0:T+1]`, collation constructs:

```text
input_ids = t[0:T]
labels    = t[1:T+1]
```

`input_ids` are tokenizer IDs, not embeddings. They are converted from compact
on-disk `uint16` to runtime `int64`, then mapped through the model's embedding
table. `labels` are shifted next-token targets, except a label that would cross
into a new document is replaced with `-100`; PyTorch cross-entropy ignores that
position.

For two small packed documents:

```text
stored tokens:  [A0, A1, EOS, B0, B1, B2, EOS]
start bits:     [ 1,  0,   0,  1,  0,  0,   0]

input_ids:      [A0, A1, EOS,  B0, B1, B2]
labels:         [A1, EOS,-100, B1, B2, EOS]
position_ids:   [ 0,  1,   2,  0,  1,  2]
document_ids:   [ 0,  0,   0,  1,  1,  1]
```

The runtime batch has:

```text
input_ids       [B, 4096] int64
labels          [B, 4096] int64
position_ids    [B, 4096] int64
document_ids    [B, 4096] int64
domain_ids      [B] int64
row_ids         [B] int64
sample_references [B] int64
num_loss_tokens scalar int64
```

`position_ids` reset at every segment start for rotary position encoding.
`document_ids` are row-local segment identifiers used to construct the
block-diagonal causal mask. A token may attend only to earlier-or-equal tokens
with the same document ID. Documents sharing a packed row therefore do not
interact, and separate rows or elements of the batch never interact.

### Document IDs and the attention mask

Conceptually, `document_ids[b, i] = j` means that token position `i` in batch
row `b` belongs to row-local attention segment `j`. The loader reconstructs
these IDs from the start bits rather than storing a multi-byte integer for every
token. In pseudocode:

```python
document_ids = cumsum(start_flags[:T]) - 1
```

Bit zero is always a start, so the first ID in every physical row is valid. The
IDs are not corpus-wide document identifiers and may restart from zero in every
row. A continuation chunk of a document that crosses a physical-row boundary
also begins a new attention segment because separately processed rows cannot
share attention state.

For query position `q` and key position `k`, attention is allowed exactly when:

```python
allowed[b, q, k] = (
    k <= q
    and document_ids[b, k] == document_ids[b, q]
)
```

The first condition is the ordinary causal mask; the second makes the causal
matrix block diagonal by document. A representation at document-3 token 7 can
therefore use document-3 tokens 1 through 7, including the current token, but
cannot use documents 1 or 2, their EOS tokens, future document-3 tokens, or
documents 4 and 5.

The native model constructs this structural mask once in `forward` and reuses
it in every transformer layer. The CPU/debug SDPA path uses a Boolean
`[B, 1, T, T]` mask where `True` means allowed. With the production CUDA
`auto` backend, FlexAttention uses the equivalent block mask without
materializing a dense matrix. Mathematically, either route is equivalent to
setting disallowed query-key scores to negative infinity before softmax:

```python
scores = query @ key.transpose(-2, -1) / sqrt(head_dim)
scores = scores.masked_fill(~allowed, -torch.inf)
weights = torch.softmax(scores, dim=-1)
```

Disallowed keys consequently receive zero attention probability. Resetting
`position_ids` is useful for RoPE, but it does not isolate documents by itself;
the same-document attention condition is required.

### Attention masking versus loss masking

Attention masking and label masking protect different boundaries and both are
required:

- the attention mask prevents a token from reading another document;
- label value `-100` prevents a boundary-crossing next-token target from
  contributing to cross-entropy.

`-100` is an ignore sentinel, not a tokenizer ID and never an `input_ids`
value. The model counts `labels != -100` as `num_loss_tokens`, computes
cross-entropy with `ignore_index=-100`, and normalizes over only those valid
targets.

For a concrete example, let the illustrative EOS ID be `99`:

```text
document A: [10, 11]
document B: [20, 21, 22]
document C: [30, 31]

stored tokens: [10, 11, 99, 20, 21, 22, 99, 30, 31, 99]
start bits:    [ 1,  0,  0,  1,  0,  0,  0,  1,  0,  0]
input_ids:     [10, 11, 99, 20, 21, 22, 99, 30, 31]
labels:        [11, 99,-100, 21, 22, 99,-100, 31, 99]
document_ids:  [ 0,  0,  0,  1,  1,  1,  1,  2,  2]
position_ids:  [ 0,  1,  2,  0,  1,  2,  3,  0,  1]
```

The valid loss terms are:

```text
-log P(11  | A: 10)
-log P(99  | A: 10, 11)
-log P(21  | B: 20)
-log P(22  | B: 20, 21)
-log P(99  | B: 20, 21, 22)
-log P(31  | C: 30)
-log P(99  | C: 30, 31)
```

Their sum is divided by seven valid targets. The labels at the `EOS_A` and
`EOS_B` input positions are `-100` instead of the next documents' first tokens,
so the model does not learn that the next packed document follows the previous
one. Conversely, predicting each document's EOS from its last content token
remains supervised. The attention mask separately ensures that B and C cannot
use preceding documents as context for any of their valid predictions.

`row_ids` are domain-local physical-row identifiers. `sample_references` retain
the encoded order references, and `num_loss_tokens` is the exact count of
non-ignored labels used for token-normalized gradient accumulation.

When a document's final content token is retained as a model input, it learns to
predict EOS. A stream's final incomplete row can leave source tokens unused or
unstored; packed manifests record `unused_as_input_tail_tokens` and
`unstored_tail_tokens` explicitly. EOS is allowed as an ordinary token ID in
source material, so the start bitset—not EOS scanning—is the authoritative
boundary signal. The native model does not implement a generation loop;
generation clients should stop when it emits the configured EOS token or
reaches an explicit generation limit.

## Storage lifecycle

During preparation, several large representations coexist:

1. raw `.tar.zst` archives, the recoverable retained source;
2. preprocessing reports/fingerprint shards and the curation SQLite
   live/snapshot authority;
3. raw token caches, reproducible acceleration artifacts;
4. final packed binaries and provenance, the training hot-path corpus.

Provenance is qualification and preflight authority, not an iterative hot-path
payload. Once qualified, training iteration memory-maps only the order, token,
and start-bit files.

After final packed-corpus qualification, backup, and a successful training
smoke/launch, the token cache is the safest large derived layer to delete if
storage cost requires it because it can be regenerated from the retained raw
archives together with their authenticated preprocessing reports/fingerprints
and pinned tokenizer. Deleting the raw archives sacrifices the easiest path to
audit, re-curate, or regenerate the corpus and should be a deliberate later
decision.
