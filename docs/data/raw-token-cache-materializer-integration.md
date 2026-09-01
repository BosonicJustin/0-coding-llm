# Raw-token-cache materializer integration contract

This specification defines the implemented opt-in optimization in
`pretrain/materialize.py`: replace repeated StarCoder2 tokenization of v7
all-eligible documents with authenticated reads from the curation-independent
raw token cache. It does **not** authorize using a cache as packed training
data. The raw-reader/tokenizer path remains the default fallback when the two
cache arguments are omitted.

The read-only API is implemented in `pretrain/raw_token_cache_reader.py`. It
maps one archive's `manifest_index` to that complete document's content token
IDs. It knows nothing about selection, split, EOS insertion, packing, order, or
attention.

## Required cache-generation authority

Materialization does not learn trust identities from the cache directories it
is about to consume. Before enabling the adapter, publish a closed-world,
canonical cache inventory with `scripts/publish_raw_token_cache_inventory.py`.
The format and builder live in `pretrain/raw_token_cache_inventory.py`; its
canonical manifest has an exact SHA-256 sidecar. The inventory is bound to:
bound to:

- the exact v7 selection-manifest SHA-256;
- cache format `raw-document-token-cache`, format version 1, and profile
  `all-raw-documents-content-only-v1`;
- the tokenizer manifest SHA-256, vocabulary SHA-256, resolved revision,
  vocabulary size, and EOS identity from the selection authority;
- every canonical input archive exactly once, in the selection manifest's
  archive order; and
- for every archive: cache manifest path, byte count, SHA-256, and sidecar
  SHA-256, plus the raw archive, preprocess report, and fingerprint bindings
  copied from the already authenticated selection/report inventory.

There may be no missing, duplicate, extra, pending, symlinked, or differently
ordered cache archive. The cache-inventory manifest SHA-256 must become part of
the materialization journal, final packed manifest, certification report, and
resume identity. Changing it requires a new materialization generation.

For each archive, `CorpusMaterializer` constructs `RawTokenCacheAuthority` from
the certified inventory and independently compares it with the already
validated selection/report/tokenizer descriptors. It never constructs source
authority by reading and trusting the local per-archive cache manifest.

## Exact v7 join

The adapter applies only when the selection identity has format version 7,
strategy `all_eligible_canonical_documents`, and the authenticated decision
descriptor is the v1 all-eligible keep bitmap. Legacy JSON decisions and
terminal-prefix quota formats must continue through the raw-tokenization path
or fail closed.

For archive ordinal `a`:

1. Validate the existing selection report, decision-bitmap descriptor and
   payload, fingerprint descriptor, archive domain, document total, and kept
   count exactly as the current materializer does.
2. Resolve exactly one cache-inventory entry using the full raw archive path,
   not only `(bucket, part number)`. Require its archive, report, fingerprint,
   tokenizer, record, clean-byte, and content-token bindings to equal the
   current `ArchiveSpec` and selection identity.
3. Open `RawTokenCacheReader` with that external authority and the immutable
   raw, preprocess, and tokenizer roots. Opening must finish successfully
   before any writer receives a document. This authenticates:

   - canonical cache manifest and exact sidecar;
   - raw archive, preprocess report, fingerprint, and tokenizer source files;
   - read-only token and offset payload sizes, hashes, dtypes and ID range;
   - strictly increasing offsets from zero to the exact content-token total;
   - every fingerprint row's archive identity and `manifest_index`; and
   - `offset[i + 1] - offset[i] == fingerprint[i].starcoder2_tokens`, plus the
     cache builder's complete alignment digest.

4. Iterate the authenticated fingerprint and bitmap in manifest-index order.
   For each ordinal `i`, derive the same v7 decision metadata used today. The
   join key is exactly:

   ```text
   archive path + manifest_index i
   bitmap bit i (LSB0) + fingerprint row i + cache document span i
   ```

   No hash-table join, filename join, reorder, or filtering before this join is
   allowed.
5. If bitmap bit `i` is zero, preserve the current rejection/provenance checks
   and do not request tokens from the cache. If it is one:

   - require a complete-document v7 decision;
   - obtain `span = reader.document(i)`;
   - require `len(span) == decision.source_tokens ==
     fingerprint[i].starcoder2_tokens`;
   - derive leakage-safe group and split exactly as the current v7 path does;
   - send the complete content-token span to the same destination writer; and
   - emit the same document-provenance record, including
     `source_manifest_index=i` and the current logical-stream positions.

6. At the archive checkpoint, call `reader.verify_unchanged()`, close the
   reader, and commit all destination cursors and provenance shards using the
   existing archive-level journal protocol. An authentication/read failure
   must abort the archive without advancing any durable cursor.

The cache reader may be opened once per archive and iterated sequentially. Do
not reopen it per document. `TokenSpan.to_list()` provides the current writer's
plain-ID input if needed; a later measured optimization may teach the writer to
consume `TokenSpan` directly without changing output bytes.

## Authorities that must not move into the cache

The integration is correct only if the following remain byte-for-byte and
semantically identical to the existing materializer:

- **Selection:** the v7 bitmap and fingerprint-derived canonical decision,
  never the cache, determine whether a document is kept.
- **Split:** leakage-safe group derivation and the frozen v7 split thresholds
  determine the destination. Cache archive/bucket placement is not a split.
- **EOS:** cache spans contain content tokens only. `PackedShardWriter` remains
  the sole EOS insertion authority and must append exactly one boundary EOS per
  kept document.
- **Packing:** sequence length, carry/tail behavior, row construction, starts,
  positions, labels, and archive checkpointing remain `PackedShardWriter`
  responsibilities.
- **Attention boundaries:** the existing packed `starts` metadata and training
  block-mask construction remain authoritative. Cached offsets are source
  document lookup metadata; they are not an attention mask.
- **Mixture and shuffle:** packed order v4 remains the only train/evaluation
  mixture, shuffle, and absolute input-token budget authority.
- **Provenance:** fingerprint/decision metadata still supplies document IDs,
  canonical IDs, group IDs, language, split, and source member paths. The token
  cache intentionally does not duplicate those policy records.

Consequently, replacing tokenization with cache lookup must produce identical
packed payload bytes, boundary arrays, document-index shards, per-split/domain
totals, and order-v4 output for the same selection and construction settings.

## Implemented invocation

Both arguments are required together and become part of the durable journal,
resume identity, final manifest, and `provenance/raw_token_cache.json`:

```text
--raw-token-cache-root CACHE_ROOT
--raw-token-cache-inventory-root INVENTORY_ROOT
```

The cached and raw routes share every downstream writer, EOS, packing, order,
and provenance authority. The adapter has a four-archive byte-identity oracle,
payload-corruption-before-checkpoint test, inventory order-corruption test, and
mid-archive crash/resume oracle in
`tests/test_materialize_raw_token_cache.py`.

## Qualification gates

Before production use, require all of the following:

1. raw-tokenization and cache-reader materializations of a multi-archive
   fixture are byte-identical for packed tokens, starts, positions, labels,
   document provenance, journals, and final manifests except for the explicitly
   added cache-inventory provenance field;
2. kept and rejected bitmap bits, empty domain contributions, archive resume,
   and a crash after document write but before archive commit all match the
   current fault-injection oracle;
3. cache generation, manifest, sidecar, tokenizer, source, fingerprint order,
   dtype, offset, token hash, ID-range, and inventory corruption each fail
   before a durable writer cursor advances;
4. six-rank order/loader and boundary-mask qualification remains green; and
5. a real-archive benchmark shows that cache open plus sequential reading is
   materially faster than pinned tokenizer replay after accounting for the
   one-time full integrity pass.

If any gate fails, omit both cache arguments and use the existing immutable
raw-reader/tokenizer route. A cache is a replaceable acceleration artifact,
never the source-of-truth corpus.
