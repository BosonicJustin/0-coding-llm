# Cross-source English near-deduplication

`scripts/build_english_near_clusters.py` produces the complete
`doc_id -> cluster_id` mapping required by `scripts/curate_corpus.py`. It runs
FineWeb-Edu and Wikipedia in one shared duplicate domain, before any split is
assigned. Code never enters this pass.

The immutable algorithm and all thresholds/seeds are pinned in
`configs/english_near_dedup.json`. Changing that file changes the run identity
and requires a new output directory.

## Why the preprocessing sketch is not enough

The preprocessing fingerprint has only 16 bottom-k values. Worse, those
values come from at most 2,048 evenly sampled five-word shingles. It is useful
for auditing and exploratory estimates, but it has three material limitations:

- sixteen observations give a high-variance similarity estimate;
- even sampling is not alignment-stable when text is inserted or removed;
- a candidate missed by such a compact sketch cannot be recovered by a later
  exact comparison.

The production stage therefore validates every compact sketch but never uses
it as the candidate authority. Calling a cluster map derived only from those
16 values production-complete would overstate its recall.

## Algorithm

The stage is deterministic and proceeds in these durable phases:

1. Prove collection completeness before freezing the English report inventory.
   For each English source, the builder requires the authoritative
   collection-complete marker, the exact-token quota target, every immutable
   shard ledger record, and an exact one-to-one match between finalized raw
   archive indices, ledger indices, and report indices. Per-bucket archive,
   document, byte, and token totals must match. It rejects missing/unknown
   reports, pending archives, or below-target ledgers. All evidence and hashes
   are pinned in the run identity and final manifest.
2. Validate the curation policy, preprocessing manifest, benchmark-guard
   identity, tokenizer/source revisions, every report and fingerprint
   checksum, stable document ID, archive/member identity, compact sketch, and
   document/token/byte totals.
3. Require the passed production calibration JSON and its exact sibling
   `.sha256`. Recompute both hashes and require the current calibration harness,
   current production builder, production/calibration configs, policy,
   preprocessing/benchmark identities, source manifests, full report inventory,
   collection-completeness authority, and pinned sampling bounds to match.
   Fixture runs, CLI overrides, failed or stale results, and relocated evidence
   fail closed. Pin the complete evidence identity into every checkpoint and the
   final manifest.
4. Re-read every immutable raw English archive. Verify its SHA-256, tar member
   order, internal manifest, raw content SHA-256, normalized SHA-256, size and
   UTF-8 encoding for every document.
5. Compute full-raw-text densified one-permutation MinHash signatures over
   five-word shingles. Only one representative of an identical normalized
   hash enters LSH. Twenty-four pinned bands are stored in a disk-backed
   SQLite index; the 16-value audit sketch is not consulted.
6. Enumerate every posting with more than one document. The exact cardinality
   upper bound `J(A,B) <= min(|A|,|B|)/max(|A|,|B|)` removes impossible pairs
   without recall loss. Candidate pairs are unique and disk-backed.
7. If a posting or the cumulative unique-candidate count exceeds its pinned
   maximum, fail closed and write `.work/FAILURE.json`. No prefix, random
   sample, or silent truncation is allowed. Normalized duplicates were already
   collapsed before LSH, which removes the most common source of pathological
   postings.
8. Re-read the raw archives a second time and cache the complete sorted set of
   five-word shingle hashes only for documents that occur in a candidate
   pair. Each document is an independently compressed zstd frame with a
   checksummed per-archive cache. Archive membership is resolved through
   left-leading and right-leading candidate indexes; it never rescans the
   entire candidate table once per archive.
9. Before the first production refinement write, run a bounded deterministic
   10,000-pair operational preflight against the real frozen candidate table
   and real shingle-cache frames. It mirrors the selected SQLite journal mode,
   measures pairs/second and bytes/pair, projects the exact candidate-count ETA
   and disk need, and projects dense-union memory with a pinned safety factor.
   Throughput, 72-hour refinement window, free-space, or 12 GiB peak-RSS gate
   failure stops the run. The checksummed result, exact formula inputs, sample
   hash, identities, thresholds, and limitations are immutable production
   evidence; ordinary full-run telemetry cannot substitute for it.
10. Compare every candidate using complete hashed-shingle Jaccard. A near edge
   is accepted at `intersection / union >= 4/5`. Each committed batch advances
   a strict `(left_document,right_document)` cursor in the same SQLite
   transaction as the refined rows and edges. Resume uses a primary-key range
   scan, not a repeated anti-join over the processed prefix. Equal normalized
   hashes are certain edges regardless of LSH. Before work and after every
   committed batch, the long run rechecks actual peak RSS and remaining
   free-space against the immutable preflight thresholds/projection and fails
   closed if either resource gate is no longer satisfied.
11. Union accepted and normalized edges with a restart-safe SQLite union-find.
   SQLite is authoritative; a dense 8-byte-per-document parent array
   accelerates each process lifetime and is reconstructed from committed state
   after a restart. Parent flattening and final document-root assignment are
   separate bounded keyset passes; each batch advances its cursor atomically,
   so no corpus-sized parent or update list is materialized. The projected
   union allocation is gated before allocation and measured peak RSS is gated
   after allocation and every committed batch. Roots are the lowest frozen
   inventory ordinal. Every non-edge document gets a singleton cluster.
12. Audit mapping coverage, root validity, normalized-hash leakage, singleton
   counts and cross-source clusters. Atomically publish `clusters.jsonl.zst`,
   re-read and compare every output row to SQLite, then publish
   `manifest.json` and `manifest.sha256`.

The raw archives are never modified.

## Statistical boundary

This is a standard candidate-and-refine design, not a mathematical proof that
all natural-language near duplicates were found. At the configured threshold,
the ideal independent-MinHash banding model gives the recall estimate recorded
in the manifest. Densified one-permutation rows are correlated, so that number
is explicitly an estimate rather than a formal lower bound. Full-shingle
refinement removes false-positive merges from LSH but cannot recover an LSH
false negative.

The final comparison is exact over the complete set of **64-bit xxh3 shingle
hashes**, not over stored shingle strings. A 64-bit collision is possible,
although unlikely. This limitation is also recorded in every output manifest.
If the experiment requires cryptographic/string-exact guarantees, introduce a
new algorithm/config version with a wider refinement representation; do not
reinterpret an existing v1 result.

Before the expensive full run, execute the bounded candidate-recall gate in
`ENGLISH_NEAR_DEDUP_CALIBRATION.md`. A production calibration must use the
immutable real English sample, the pinned acceptance profile, and report
`production_gate_eligible == true`. A poor calibration means creating a new
pinned config (more bands/tables), not editing a run in place.

## Production run

Wait until both collection-complete markers exist and all finalized quota-ledger
archives have complete immutable fingerprint reports. Run and pass the recall
calibration first. For speed, put the output/state on pod-local NVMe; it
contains a large band index and candidate-only shingle cache. Copy the closed,
checksummed result to the network volume only after completion.

The builder does not trust the operator sequence alone. `--calibration-result`
is mandatory, must name a regular file under the immutable dataset root, and
must have the exact adjacent `<name>.sha256` emitted by the pinned harness. The
same root-relative evidence path and hashes are part of resumability, so moving
or replacing the result after a partial run is an identity mismatch.

The SQLite journal policy is pinned in the production config. At startup the
builder records the longest-prefix mount, filesystem type/source/options, and
probe method. `auto` selects WAL only for an explicit local-filesystem
allowlist (for example ext4, XFS, or local NVMe-backed overlay). NFS, other
network filesystems, and unknown filesystems use rollback-journal `DELETE`.
An explicit `--sqlite-journal-mode wal` request fails closed unless storage is
proven local; it cannot override this guard. The requested, selected, and
actual journal modes plus mount evidence appear under
`identity.runtime.storage` in every checkpoint and final manifest.

`DELETE` avoids SQLite's documented WAL/network-filesystem incompatibility,
but it is not a claim that every network filesystem has reliable locking or
failure semantics. The production recommendation remains pod-local NVMe for
live `.work`, followed by copying only the closed checksummed artifacts to the
network volume.

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/build_english_near_clusters.py \
  --root /workspace/dataset \
  --staging-root /workspace/dataset/staging/preprocess \
  --calibration-result /workspace/dataset/audits/english-near-calibration-v1.json \
  --output /local-nvme/english-near-v1 \
  --batch-size 10000 \
  --progress-interval-seconds 60
```

Re-run the exact command to resume. The SQLite database under `.work` is the
authoritative journal. `CHECKPOINT.json` and `journal.jsonl` are inspectable
projections. An existing database is opened read-first: its selected journal
mode, exact frozen identity/metadata, and phase/cursors are validated before
schema writes; journal mode is set only for a new empty database. A `DELETE`
run rejects `-wal`/`-shm` sidecars, and mount or journal-mode drift rejects the
resume rather than converting it. Never copy a live SQLite directory as a
checkpoint. `--batch-size` is operational rather than a clustering-semantic
identity before preflight; once the immutable preflight is published, its
measured batch size is enforced on every resume. Keep it unchanged throughout
a production run.

Controlled stop options exist for every expensive boundary:

- `--max-new-inventory-archives`
- `--max-new-signature-archives`
- `--max-new-candidate-blocks`
- `--max-new-cache-archives`
- `--max-new-refinement-pairs`
- `--max-new-union-edges`
- `--max-new-union-finalization-documents`
- `--stop-after-phase`

An ordinary signal or pod loss may repeat the current transaction/archive on
resume, but cannot publish a partial mapping as complete.

## Consumer manifest contract

Treat `clusters.jsonl.zst`, `manifest.json`, `manifest.sha256`, and the
`operational-preflight-v1/{result.json,result.json.sha256}` subtree as one
five-file artifact. A production consumer must fail closed unless all of these
checks pass:

1. `manifest.sha256` contains the SHA-256 of the exact `manifest.json` bytes;
   `manifest_version == 1`, `mapping_record_version == 1`, and
   `production_ready is true`.
2. `mapping.path == "clusters.jsonl.zst"`; its file SHA-256, byte size, and
   row count equal `mapping.sha256`, `mapping.bytes`, and `mapping.records`;
   `mapping.singleton_clusters_included is true`.
3. `identity.builder_sha256`, `identity.config_file_sha256`,
   `identity.config_sha256`, `identity.curation_policy_sha256`,
   `identity.preprocess_manifest_sha256`, and
   `identity.benchmark_guard_sha256` match the consumer's pinned code/config
   and current immutable inputs. The duplicate config hashes in `algorithm`
   must equal the identity hashes; `algorithm.name` must be the pinned v1
   algorithm. Record the `identity.runtime` versions as part of provenance,
   including mount detection and equality of the selected/actual SQLite
   journal modes in `identity.runtime.storage`.
4. `identity.calibration_evidence` has `contract_version == 1`, the passed gate
   literals and pinned acceptance/sampling profiles, empty failures, a
   root-relative result path/hash/size and sibling sidecar path/hash, plus the
   full nested result identity and its canonical SHA-256. Reopen the two
   regular non-symlink files, verify their exact bytes, and require the nested
   `harness_sha256`, `production_builder_sha256`, production/calibration config,
   policy, preprocessing, benchmark, source, full-report-inventory,
   collection-completeness, and sampling identities to equal the consumer's
   current authorities.
5. The top-level `refinement_operational_preflight` has exactly the v1 evidence
   contract and is not optional: root-relative result/sidecar paths, exact
   byte count and both file hashes, `status == "pass"`,
   `production_gate_eligible is true`, empty failures, canonical nested
   identity hash, pinned thresholds, sample, and measurements. Reopen both
   regular non-symlink files. Require the sidecar bytes to be exactly
   `<result_sha256>  result.json\n`, parse the result without duplicate keys,
   and require its identity/thresholds/sample/measurements to equal the
   manifest evidence. Recompute the deterministic candidate sample hash,
   batch count/size, rate and ETA, SQLite bytes-per-pair/disk/safety/free-space
   formulas, cache inventory, resource invariants, and dense-union memory
   projection. Its builder/config/calibration/report/policy/preprocess/guard,
   completeness, cache, storage, candidate, and document identities must match
   the consumer's current authorities.
6. `identity.source_manifests` hashes/revisions match the current tokenizer,
   FineWeb-Edu, and Wikipedia source manifests.
7. `identity.collection_completeness` has both English buckets. Its quota
   config and quota-record inventory hashes must match the current files. For
   each bucket, verify the completion-marker hash/content, exact-token target,
   archive-index hash, and equality of `finalized_totals` and `report_totals`.
   `quota_record_inventory_sha256` is SHA-256 over compact canonical JSON
   (`sort_keys=True`, separators `,` and `:`) of the already bucket/index-sorted
   `quota_records` list. Each bucket's `archive_indices_sha256` uses the same
   encoding over its sorted integer index list.
8. Recompute `inputs.report_inventory_sha256` from `inputs.reports` in manifest
   order by projecting each row to
   `{path: report_path, sha256: report_sha256, fingerprint_sha256: ...}` and
   hashing compact canonical JSON. Require it to equal
   `identity.report_inventory_sha256`, then verify every listed report,
   archive, and fingerprint checksum. The sum of report documents must equal
   `mapping.records`.
9. `database_integrity_check == "ok"`. In
   `completeness_and_leakage_audit`, inventory and mapped document counts must
   both equal `mapping.records`; missing, unknown, duplicate, normalized-hash
   leakage, and invalid-root counts must all be zero. Cluster and singleton
   counts must be internally possible.

Checking only the mapping hash and row coverage is insufficient: it does not
prove that the mapping was produced from the complete finalized English input
set or with the pinned algorithm.

## Handoff to final curation

After `manifest.sha256` and the mapping SHA-256 verify, use:

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/curate_corpus.py \
  --root /workspace/dataset \
  --staging-root /workspace/dataset/staging/preprocess \
  --english-near-clusters /local-nvme/english-near-v1/clusters.jsonl.zst \
  --output /workspace/dataset/curated/selection-v1 \
  --batch-size 10000
```

`curate_corpus.py` independently requires exactly one mapping row for every
English document and rejects duplicates, unknown documents, code documents,
or missing rows.
