# Hugging Face base-model release workflow

This directory is a release-candidate template, not a published model. It does
not select a model-weight license and cannot pass the public-release gate until
the user resolves every marker in `README.template.md` from final evidence.

The native Hugging Face export and the Hub package are intentionally separate:

- the sealed native export is the private, exact-tree authority used for
  provenance and post-training;
- the Hub staging tree contains standard model/tokenizer files, a model card,
  generation metadata, `.gitattributes`, sanitized provenance, and its own
  complete SHA-256 manifest;
- native `.pt` checkpoints, optimizer/RNG state, logs, caches, credentials, and
  internal manifests containing local paths never enter the Hub staging tree.

## 1. Complete the model card

Copy `README.template.md` to a working location, replace every
`{{PLACEHOLDER}}` from the accepted final checkpoint/evaluation evidence, add
the user-approved `license:` field to its YAML metadata, and remove both
`RELEASE_BLOCKER` comments. Do not estimate final metrics or silently choose a
license.

For a private draft, the unmodified template may be staged. The manifest will
label it `draft`, and the tooling will refuse a public upload.

## 2. Stage without reserializing weights

Keep the staging path on the same filesystem as the sealed export to hard-link
the multi-gigabyte safetensors without copying or rewriting them:

```bash
python scripts/prepare_hf_release.py \
  --source-export /path/to/sealed-native-hf-export \
  --model-card /path/to/completed-model-card.md \
  --output /path/to/new-hf-release-candidate \
  --file-mode hardlink
```

Hard-link mode fails rather than silently copying across filesystems. Use
`--file-mode copy` only when the additional storage and I/O are intentional.
The source and destination paths must be distinct, and the destination must not
already exist.

The builder authenticates the complete sealed export, copies only the public
allowlist, removes local `name_or_path` metadata, preserves standard shard/index
bytes, writes `HF_RELEASE_MANIFEST.json`, fsyncs the tree, and atomically renames
it into place. Never edit a completed staging tree; make a new one so the
manifest remains authoritative.

## 3. Local preflight

Metadata and byte verification performs no network access and does not load all
weights:

```bash
python scripts/upload_hf_release.py preflight \
  --release-dir /path/to/new-hf-release-candidate \
  --public
```

Add `--load-smoke --device cuda:0` to instantiate the complete exported model,
check tokenizer/config identities and finite logits, and generate one greedy
token. This smoke can require roughly the full FP32 model size in host and GPU
memory.

The strict tiny CPU exporter test remains the deterministic unit oracle. The
full 1.3B CUDA FP32 SDPA export comparison observed maximum absolute logits
error `2.2411e-5` and mean absolute error `2.1518e-6`; any repeated CUDA parity
gate should use the documented implementation-noise tolerance
`atol=3e-5, rtol=1e-4` and require identical top-1 token IDs.

## 4. Upload one private commit

Install the pinned Hugging Face toolchain from `requirements-release.txt` after
installing the correct platform PyTorch wheel. Supply authentication only
through a secret environment variable:

```bash
export HF_TOKEN='set-this-through-the-host-secret-manager'

python scripts/upload_hf_release.py upload \
  --release-dir /path/to/new-hf-release-candidate \
  --repo-id OWNER/MODEL \
  --visibility private \
  --confirm-upload OWNER/MODEL
```

The command refuses an existing repository, requires the repository ID twice,
and uploads the authenticated folder in one commit. It never uploads the parent
directory. Save the returned immutable commit SHA. If repository creation
succeeds but upload fails, inspect that empty/private repository before any
retry; the script never assumes it is safe to overwrite.

## 5. Verify the immutable remote commit

Use a new directory with enough space for one complete snapshot:

```bash
python scripts/upload_hf_release.py verify-remote \
  --release-dir /path/to/new-hf-release-candidate \
  --repo-id OWNER/MODEL \
  --revision FULL_40_CHARACTER_COMMIT_SHA \
  --download-dir /path/to/new-verification-download \
  --load-smoke \
  --device cuda:0
```

This pins the exact commit, downloads its files, compares the remote manifest to
the local candidate, re-hashes every file, rejects unexpected content, and can
repeat the full-model load smoke. Only after this succeeds should the recorded
commit be announced or a separately authorized visibility change be made.
