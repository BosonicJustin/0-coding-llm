"""Build and authenticate a sanitized Hugging Face release package.

The native HF export is an internal, exact-tree artifact whose manifests contain
local provenance.  A public Hub repository has different requirements: a model
card and generation metadata are needed, while local paths, optimizer state, and
internal source manifests must not be published.  This module verifies the
sealed native export and creates a separate flat release tree without changing
or reserializing its weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .hf_export import (
    EXPORT_FORMAT,
    EXPORT_FORMAT_VERSION,
    EXPORT_MANIFEST_NAME,
    EXPORT_MANIFEST_SIDECAR_NAME,
    SOURCE_TOKENIZER_MANIFEST_NAME,
    TOKENIZER_EXPORT_FORMAT,
    TOKENIZER_MANIFEST_NAME,
)


RELEASE_FORMAT = "hugging-face-model-release"
RELEASE_FORMAT_VERSION = 1
RELEASE_MANIFEST_NAME = "HF_RELEASE_MANIFEST.json"
RELEASE_MANIFEST_SIDECAR_NAME = "HF_RELEASE_MANIFEST.sha256"
RELEASE_PROVENANCE_NAME = "RELEASE_PROVENANCE.json"
MODEL_CARD_NAME = "README.md"
GENERATION_CONFIG_NAME = "generation_config.json"
HUB_ATTRIBUTES_NAME = ".gitattributes"
PUBLICATION_BLOCKER = "RELEASE_BLOCKER"
_PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_METADATA_KEYS = frozenset(
    {"_name_or_path", "name_or_path", "merges_file", "tokenizer_file", "vocab_file"}
)
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:root|workspace|home|Users)/")
_HUB_ATTRIBUTES = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
_INTERNAL_ONLY_NAMES = frozenset(
    {
        EXPORT_MANIFEST_NAME,
        EXPORT_MANIFEST_SIDECAR_NAME,
        SOURCE_TOKENIZER_MANIFEST_NAME,
        TOKENIZER_MANIFEST_NAME,
    }
)


class ReleaseError(ValueError):
    """Raised when a source export or release package is unsafe or invalid."""


@dataclass(frozen=True)
class VerifiedSourceExport:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    public_files: tuple[str, ...]
    config: Mapping[str, Any]
    tokenizer_source: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _canonical_json(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"{label} is missing, unsafe, or not a regular file: {path}")
    return path


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must contain a JSON object")
    return value


def _flat_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ReleaseError(f"{label} must be a non-empty flat filename")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ReleaseError(f"{label} is not a flat filename: {value!r}")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReleaseError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_file_record(root: Path, record: Mapping[str, Any], *, label: str) -> str:
    name = _flat_name(record.get("path"), label=f"{label}.path")
    path = _regular_file(root / name, label=label)
    size = record.get("bytes")
    if type(size) is not int or size < 0 or path.stat().st_size != size:
        raise ReleaseError(f"{label} byte size does not match: {name}")
    expected = _sha256(record.get("sha256"), label=f"{label}.sha256")
    if _sha256_file(path) != expected:
        raise ReleaseError(f"{label} SHA-256 does not match: {name}")
    return name


def _verify_exact_tree(root: Path, expected: set[str], *, allowed_extra: set[str] | None = None) -> None:
    allowed_extra = set() if allowed_extra is None else set(allowed_extra)
    actual: set[str] = set()
    for path in root.iterdir():
        if path.name in allowed_extra:
            continue
        if path.is_symlink() or not path.is_file():
            raise ReleaseError(f"Release tree contains an unsafe or non-file entry: {path.name}")
        actual.add(path.name)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReleaseError(f"Release file inventory differs: missing={missing}, extra={extra}")


def verify_source_export(root: str | Path) -> VerifiedSourceExport:
    """Authenticate a sealed internal exporter output and select public files."""

    source = Path(root).expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ReleaseError(f"Source export is missing or unsafe: {source}")
    source = source.resolve(strict=True)
    manifest_path = _regular_file(source / EXPORT_MANIFEST_NAME, label="native export manifest")
    manifest = _load_json(manifest_path, label="native export manifest")
    if (
        manifest.get("format") != EXPORT_FORMAT
        or manifest.get("manifest_version") != EXPORT_FORMAT_VERSION
        or manifest.get("architecture") != "LlamaForCausalLM"
    ):
        raise ReleaseError("Unsupported native HF export contract")
    manifest_sha = _sha256_file(manifest_path)
    sidecar = _regular_file(
        source / EXPORT_MANIFEST_SIDECAR_NAME,
        label="native export manifest sidecar",
    )
    expected_sidecar = f"{manifest_sha}  {EXPORT_MANIFEST_NAME}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise ReleaseError("Native export manifest sidecar does not match")

    names = {EXPORT_MANIFEST_NAME, EXPORT_MANIFEST_SIDECAR_NAME}
    public: set[str] = set()
    weights = manifest.get("weights")
    if not isinstance(weights, dict) or weights.get("format") != "safetensors":
        raise ReleaseError("Native export must contain safetensors weights")
    weight_records = weights.get("files")
    if not isinstance(weight_records, list) or not weight_records:
        raise ReleaseError("Native export has no weight-shard inventory")
    tensor_names: set[str] = set()
    for index, record in enumerate(weight_records):
        if not isinstance(record, dict):
            raise ReleaseError(f"Invalid weight record {index}")
        name = _verify_file_record(source, record, label=f"weights.files[{index}]")
        if not name.endswith(".safetensors") or name in names:
            raise ReleaseError(f"Unsafe or duplicate weight filename: {name}")
        tensors = record.get("tensors")
        if not isinstance(tensors, list) or not tensors or any(
            not isinstance(item, str) or not item for item in tensors
        ):
            raise ReleaseError(f"Invalid tensor list for weight shard {name}")
        if tensor_names.intersection(tensors):
            raise ReleaseError("A tensor appears in more than one weight shard")
        tensor_names.update(tensors)
        names.add(name)
        public.add(name)

    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise ReleaseError("Native export has no non-weight inventory")
    for index, record in enumerate(file_records):
        if not isinstance(record, dict):
            raise ReleaseError(f"Invalid native non-weight record {index}")
        name = _verify_file_record(source, record, label=f"files[{index}]")
        if name in names:
            raise ReleaseError(f"Duplicate native export filename: {name}")
        names.add(name)
    _verify_exact_tree(source, names)

    index_name = weights.get("index")
    if weights.get("is_sharded"):
        if index_name != "model.safetensors.index.json" or index_name not in names:
            raise ReleaseError("Sharded native export is missing its standard weight index")
        public.add(index_name)
    elif index_name is not None:
        raise ReleaseError("Unsharded native export unexpectedly declares a weight index")

    config = _load_json(source / "config.json", label="HF model config")
    if config.get("model_type") != "llama" or config.get("architectures") != ["LlamaForCausalLM"]:
        raise ReleaseError("Native export is not a standard HF LlamaForCausalLM")
    if config.get("vocab_size") != 49_152 or config.get("max_position_embeddings") != 4_096:
        raise ReleaseError("Native export vocabulary or context differs from run-1 authority")
    public.add("config.json")
    if "native_config.json" in names:
        public.add("native_config.json")

    tokenizer_outer = manifest.get("tokenizer")
    if not isinstance(tokenizer_outer, dict):
        raise ReleaseError("Native export tokenizer record is missing")
    tokenizer_manifest = _load_json(source / TOKENIZER_MANIFEST_NAME, label="tokenizer export manifest")
    if tokenizer_manifest.get("format") != TOKENIZER_EXPORT_FORMAT:
        raise ReleaseError("Unsupported tokenizer export manifest")
    tokenizer_files = tokenizer_manifest.get("files")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise ReleaseError("Tokenizer export has no public file inventory")
    for name, record in tokenizer_files.items():
        name = _flat_name(name, label="tokenizer filename")
        if not isinstance(record, dict):
            raise ReleaseError(f"Invalid tokenizer record: {name}")
        _verify_file_record(source, {"path": name, **record}, label=f"tokenizer.files[{name}]")
        if "chat_template" in name or name.endswith(".jinja"):
            raise ReleaseError("Base-model release must not publish a chat template")
        public.add(name)

    source_manifest = _load_json(
        source / SOURCE_TOKENIZER_MANIFEST_NAME,
        label="source tokenizer manifest",
    )
    tokenizer_source = {
        key: source_manifest[key]
        for key in ("repo_id", "resolved_revision")
        if isinstance(source_manifest.get(key), str)
    }
    return VerifiedSourceExport(
        root=source,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        public_files=tuple(sorted(public)),
        config=config,
        tokenizer_source=tokenizer_source,
    )


def _scrub_local_path_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_local_path_metadata(item)
            for key, item in value.items()
            if key not in _PATH_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_scrub_local_path_metadata(item) for item in value]
    return value


def _copy_public_file(source: Path, destination: Path, *, mode: Literal["hardlink", "copy"]) -> None:
    if source.name in {"config.json", "tokenizer_config.json", "native_config.json"}:
        payload = _load_json(source, label=source.name)
        if source.name == "tokenizer_config.json" and payload.get("chat_template") not in (None, ""):
            raise ReleaseError("Base-model tokenizer_config.json must not define a chat template")
        _write_json(destination, _scrub_local_path_metadata(payload))
        return
    if mode == "hardlink":
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as exc:
            raise ReleaseError(
                f"Cannot hard-link {source.name}; choose --file-mode copy explicitly "
                "if duplicating the release bytes is acceptable"
            ) from exc
    else:
        shutil.copyfile(source, destination, follow_symlinks=False)


def _card_publication_state(card: str) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if PUBLICATION_BLOCKER in card:
        blockers.append("model card still contains RELEASE_BLOCKER")
    placeholders = sorted(set(_PLACEHOLDER.findall(card)))
    if placeholders:
        blockers.append(f"model card has unresolved placeholders: {placeholders}")
    if not re.search(r"(?m)^license:\s*[^\s#]+\s*$", card.partition("---\n")[2].partition("\n---")[0]):
        blockers.append("model card YAML has no explicit license")
    return not blockers, blockers


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _fsync_tree(root: Path) -> None:
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ReleaseError(f"Unsafe staged release entry: {path}")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_release_package(
    source_export: str | Path,
    output_dir: str | Path,
    *,
    model_card: str | Path,
    generation_config: str | Path,
    hub_attributes: str | Path,
    file_mode: Literal["hardlink", "copy"] = "hardlink",
) -> Mapping[str, Any]:
    """Create a new, sanitized Hub staging directory atomically."""

    if file_mode not in ("hardlink", "copy"):
        raise ReleaseError("file_mode must be 'hardlink' or 'copy'")
    source = verify_source_export(source_export)
    output = Path(output_dir).expanduser()
    if output.exists():
        raise ReleaseError(f"Refusing to overwrite release path: {output}")
    card_path = _regular_file(Path(model_card), label="model-card input")
    generation_path = _regular_file(Path(generation_config), label="generation-config input")
    attributes_path = _regular_file(Path(hub_attributes), label="Hub attributes input")
    card = card_path.read_text(encoding="utf-8")
    if _PRIVATE_PATH.search(card):
        raise ReleaseError("Model card contains an absolute local server/workstation path")
    public_ready, blockers = _card_publication_state(card)
    generation = _load_json(generation_path, label="generation config")
    expected_generation = {
        "bos_token_id": source.config.get("bos_token_id"),
        "eos_token_id": source.config.get("eos_token_id"),
    }
    if generation != expected_generation or generation.get("eos_token_id") != 0:
        raise ReleaseError(
            f"Generation config must exactly mirror BOS/EOS from config.json: {expected_generation}"
        )
    attributes = attributes_path.read_bytes()
    if attributes != _HUB_ATTRIBUTES:
        raise ReleaseError("Hub .gitattributes input differs from the frozen release policy")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.release-", dir=output.parent))
    published = False
    try:
        for name in source.public_files:
            _copy_public_file(source.root / name, temporary / name, mode=file_mode)
        _write_bytes(temporary / MODEL_CARD_NAME, card.encode("utf-8"))
        _write_json(temporary / GENERATION_CONFIG_NAME, generation)
        _write_bytes(temporary / HUB_ATTRIBUTES_NAME, attributes)

        source_record = source.manifest.get("source")
        checkpoint_sha = None
        if isinstance(source_record, dict):
            candidate = source_record.get("sha256")
            if isinstance(candidate, str) and _SHA256.fullmatch(candidate):
                checkpoint_sha = candidate
        if checkpoint_sha is None:
            raise ReleaseError("Native export does not authenticate its source checkpoint")
        tokenizer = source.manifest.get("tokenizer")
        provenance = {
            "format": "sanitized-hf-release-provenance",
            "format_version": 1,
            "architecture": "LlamaForCausalLM",
            "source_native_export_manifest_sha256": source.manifest_sha256,
            "source_checkpoint_sha256": checkpoint_sha,
            "tokenizer": {
                **source.tokenizer_source,
                "vocabulary_sha256": (
                    tokenizer.get("vocab_sha256") if isinstance(tokenizer, dict) else None
                ),
            },
        }
        _write_json(temporary / RELEASE_PROVENANCE_NAME, provenance)
        records = [
            _file_record(path)
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
        ]
        manifest = {
            "format": RELEASE_FORMAT,
            "format_version": RELEASE_FORMAT_VERSION,
            "publication_state": "public-ready" if public_ready else "draft",
            "publication_blockers": blockers,
            "source_native_export_manifest_sha256": source.manifest_sha256,
            "files": records,
        }
        _write_json(temporary / RELEASE_MANIFEST_NAME, manifest)
        manifest_sha = _sha256_file(temporary / RELEASE_MANIFEST_NAME)
        _write_bytes(
            temporary / RELEASE_MANIFEST_SIDECAR_NAME,
            f"{manifest_sha}  {RELEASE_MANIFEST_NAME}\n".encode("ascii"),
        )
        _fsync_tree(temporary)
        if output.exists():
            raise ReleaseError(f"Release output appeared during staging: {output}")
        os.rename(temporary, output)
        published = True
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        validate_release_package(output, require_public_ready=False)
        return manifest
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def validate_release_package(
    root: str | Path,
    *,
    require_public_ready: bool,
    allowed_extra: set[str] | None = None,
) -> Mapping[str, Any]:
    """Verify the complete release tree and optional public-release gate."""

    release = Path(root).expanduser()
    if release.is_symlink() or not release.is_dir():
        raise ReleaseError(f"Release package is missing or unsafe: {release}")
    manifest_path = _regular_file(release / RELEASE_MANIFEST_NAME, label="release manifest")
    manifest = _load_json(manifest_path, label="release manifest")
    if (
        manifest.get("format") != RELEASE_FORMAT
        or manifest.get("format_version") != RELEASE_FORMAT_VERSION
    ):
        raise ReleaseError("Unsupported HF release manifest")
    digest = _sha256_file(manifest_path)
    sidecar = _regular_file(
        release / RELEASE_MANIFEST_SIDECAR_NAME,
        label="release manifest sidecar",
    )
    if sidecar.read_text(encoding="ascii") != f"{digest}  {RELEASE_MANIFEST_NAME}\n":
        raise ReleaseError("HF release manifest sidecar does not match")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ReleaseError("HF release manifest has no file inventory")
    names = {RELEASE_MANIFEST_NAME, RELEASE_MANIFEST_SIDECAR_NAME}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ReleaseError(f"Invalid release file record {index}")
        name = _verify_file_record(release, record, label=f"files[{index}]")
        if name in names or name in _INTERNAL_ONLY_NAMES:
            raise ReleaseError(f"Duplicate or internal-only release file: {name}")
        names.add(name)
    for required in (
        MODEL_CARD_NAME,
        GENERATION_CONFIG_NAME,
        HUB_ATTRIBUTES_NAME,
        RELEASE_PROVENANCE_NAME,
        "config.json",
    ):
        if required not in names:
            raise ReleaseError(f"Release package is missing {required}")
    if not any(name.endswith(".safetensors") for name in names):
        raise ReleaseError("Release package has no safetensors weights")
    _verify_exact_tree(release, names, allowed_extra=allowed_extra)
    card = (release / MODEL_CARD_NAME).read_text(encoding="utf-8")
    public_ready, blockers = _card_publication_state(card)
    declared_state = manifest.get("publication_state")
    declared_blockers = manifest.get("publication_blockers")
    if declared_state != ("public-ready" if public_ready else "draft") or declared_blockers != blockers:
        raise ReleaseError("Model-card publication state differs from the release manifest")
    if require_public_ready and not public_ready:
        raise ReleaseError(f"Public release is blocked: {'; '.join(blockers)}")
    return manifest
