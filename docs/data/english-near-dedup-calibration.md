# English near-dedup recall calibration

`scripts/calibrate_english_near_dedup.py` is a bounded, deterministic check of
the candidate stage pinned in `configs/english_near_dedup.json`. It does not
edit that production configuration. It publishes a JSON result and SHA-256
sidecar even when the acceptance gate fails.

The production builder deliberately does not use the preprocessing
fingerprint's 16-value sampled bottom-k sketch for candidate generation. That
sketch is too small and is not stable under insertions. Calibration instead
executes the actual full-raw-text DOPH-LSH implementation followed by the
actual complete hashed-shingle refinement.

## What it measures

The harness deterministically chooses up to four finalized archives and 32
documents per English source. Selection is SHA-256 priority sampling with a
pinned seed, so it is independent of directory order and uses bounded memory.
Only fingerprint rows with at least 96 words enter the sample. Every selected
document is then read from the immutable raw tar.zst archive and checked
against its report, fingerprint, internal manifest, content hash, normalized
hash, source identity, quota ledger, and collection-complete marker. Exact
normalized duplicates are collapsed before perturbation so they cannot inflate
the effective recall sample.

For each eligible document, seeded donor-based edits generate measured
five-word-shingle Jaccard examples in these pinned bins:

- 0.700–0.760
- 0.770–0.795
- 0.800–0.830
- 0.840–0.880
- 0.900–0.950

The edit families are append, tail truncation, and contiguous replacement.
The report records every generated pair, its exact hashed-shingle counts,
shared LSH bands, candidate decision, and refinement decision. Deterministic
low-similarity real-document controls measure candidate false positives and
refinement rejection.

The pinned production acceptance gate requires at least 48 eligible real
documents, 144 generated pairs at or above 0.8, every bin populated, zero
perturbation-generation failures, zero refinement decision errors, and a 95%
one-sided Wilson lower bound of at least 0.98 for candidate recall. The Wilson
bound, not the observed point estimate, is the acceptance statistic. This keeps
a small all-success sample from being reported as conclusive evidence of 98%
recall.

## Production calibration

Run this only after both English collectors and preprocessing have completed.
It is read-only with respect to the dataset and production config.

```bash
/opt/coding-model-venv/bin/python \
  /workspace/coding_model_from_scratch/scripts/calibrate_english_near_dedup.py \
  --root /workspace/dataset \
  --staging-root /workspace/dataset/staging/preprocess \
  --output /workspace/dataset/audits/english-near-calibration-v1.json
```

Exit status is `0` for a passed gate, `2` for a completed calibration that
failed acceptance, and `1` for invalid/incomplete inputs or an integrity
error. In every completed pass/fail case, verify both files:

```bash
cd /workspace/dataset/audits
sha256sum -c english-near-calibration-v1.json.sha256
```

A production result must have all of:

- `status == "pass"`;
- `production_gate_eligible == true`;
- `acceptance_profile == "pinned-production"`;
- no `acceptance_failures`;
- a matching `.sha256` sidecar.

The result pins both `identity.harness_sha256` and
`identity.production_builder_sha256`, plus the production config, calibration
config, seed, runtime dependency versions, full report inventory,
preprocess/source identities, complete quota/marker evidence, selected report
and raw archive hashes, and the selected-document manifest hash. Any harness or
builder edit invalidates old calibration evidence by design.

The production cluster builder requires this exact JSON path through
`--calibration-result`. It reopens the adjacent `.sha256`, revalidates the
passed/pinned profiles and all current input identities, and embeds the
root-relative result path, result/sidecar hashes, and full calibration identity
in its restart identity. A missing, failed, overridden, stale, tampered, moved,
or fixture-mode result cannot start or resume production clustering. Keep both
files on the durable dataset volume.

## Small smoke run

A fixture is JSONL (or JSONL.zst) with non-empty `doc_id`, `text`, and optional
`bucket` fields. The command below verifies the harness locally without a
server corpus:

```bash
python scripts/calibrate_english_near_dedup.py \
  --input-jsonl /path/to/small-documents.jsonl \
  --output /tmp/english-near-calibration-smoke.json \
  --minimum-documents 2 \
  --minimum-pairs 4 \
  --minimum-candidate-recall 0.50
```

Any CLI acceptance override is recorded and forces
`production_gate_eligible == false`, even if `status` is `pass`. Fixture mode
is likewise never production-gate evidence. Reducing or increasing the pinned
archive/document sampling bounds also produces a non-production result.
Overrides are for smoke and diagnosis only; they never rewrite either pinned
config.

## Statistical limits

Passing establishes observed performance only for the selected documents,
edit families, and finite seeded sample. It is not a proof of corpus-wide
recall, and synthetic perturbations do not reproduce every naturally occurring
duplicate process. DOPH rows are correlated, so the independent-MinHash curve
in the production manifest is only an idealized estimate. Perturbation pairs
also share base documents, so the Wilson interval is an operational diagnostic
gate, not a formal corpus-wide confidence guarantee. Refinement compares
complete sets of 64-bit xxh3 shingle hashes; a hash collision remains
possible. If calibration fails, create and review a new versioned production
config and rerun calibration. Never weaken or mutate the active config in
place.
