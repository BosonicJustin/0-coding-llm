#!/usr/bin/env python3
"""Report authenticated curation/cache progress and host health without writes.

This is an operational reporter, not a replacement for final corpus
qualification.  It authenticates the atomically published curation projection,
reconstructs the checkpoint-pinned report/fingerprint/raw-archive inventory
from the configured dataset generation, authenticates that generation's pinned
tokenizer, and binds every *completed* raw-token-cache manifest to the resulting
per-part authority.  It deliberately does not hash multi-gigabyte raw or token
payloads; the cache builder and closed-world cache inventory perform those full
verification passes.

The reporter never creates locks, logs, directories, snapshots, or status
files.  Its only output is JSON on stdout (or stderr for a failed inspection).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


BUCKETS = ("fineweb_edu", "other_code", "python", "wikipedia")
BUCKET_RAW_PATHS = {
    "fineweb_edu": "raw/english/fineweb_edu",
    "other_code": "raw/other_code",
    "python": "raw/python",
    "wikipedia": "raw/english/wikipedia",
}
CURATION_PHASES = {
    "inventory",
    "inventory_complete",
    "canonicalizing",
    "canonicalized",
    "selecting",
    "selected",
    "emitting",
    "emitted",
    "complete",
}
CACHE_TOP_LEVEL = {
    "format",
    "format_version",
    "profile",
    "cache_complete",
    "training_ready",
    "tokenization",
    "non_authorities",
    "source",
    "tokenizer",
    "documents",
    "payloads",
    "builder",
}
CACHE_OUTPUT_FILES = {
    "tokens.u16",
    "offsets.u64",
    "manifest.json",
    "manifest.sha256",
}
TOKENIZATION_CONTRACT = {
    "added_special_tokens": False,
    "boundary_tokens": False,
    "document_order": "raw-internal-manifest-order",
    "document_selection": "all",
    "padding": False,
    "token_payload": "document-content-only",
    "truncation": False,
}
NON_AUTHORITIES = [
    "curation",
    "document_selection",
    "eos_insertion",
    "packing",
    "shuffle_order",
    "split_assignment",
]
PART_DIRECTORY = re.compile(r"part-(\d{6})")
STAGE_DIRECTORY = re.compile(r"\.part-(\d{6})\.building-[A-Za-z0-9-]+")
INVENTORY_SUBPHASE = re.compile(
    r"inventory\.archive\.(fineweb_edu|other_code|python|wikipedia)\.(\d{6})"
)
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
LOWER_COMMIT = re.compile(r"[0-9a-f]{40}")
FAST_ALL_ELIGIBLE_HANDOFF_PROFILE = {
    "contract_version": 1,
    "name": "fast-all-eligible-publisher-handoff-v1",
    "exact_quota_selection": False,
    "decision_emission": False,
    "periodic_full_snapshots": False,
    "final_snapshot_required": True,
    "publisher": "all-eligible-identity-v7",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_SCRIPT = PROJECT_ROOT / "scripts" / "curate_corpus.py"
CACHE_SCRIPT = PROJECT_ROOT / "scripts" / "cache_raw_tokens.py"
CURATION_LEASE_FILE = ".curation.cross-client-lease.json"
CURATION_LOCK_FILE = ".curation.lock"
CACHE_LOCK_FILE = ".raw-token-cache.lock"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_CACHE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PREPROCESS_REPORT_BYTES = 8 * 1024 * 1024
MAX_TOKENIZER_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TOKENIZER_FILE_BYTES = 2 * 1024 * 1024 * 1024
CACHE_CONFIG_FIELDS = {
    "expected_vocab_size",
    "max_documents_per_archive",
    "max_document_bytes",
    "max_document_tokens",
    "tokenizer_batch_documents",
    "tokenizer_batch_bytes",
    "tokenizer_batch_tokens",
    "max_manifest_member_bytes",
    "max_json_line_bytes",
    "minimum_free_bytes",
}


class DataProgressError(RuntimeError):
    """The progress authority or a required live-health invariant failed."""


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataProgressError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(path: Path, *, label: str, maximum_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read one bounded regular file while detecting replacement or mutation."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DataProgressError(f"cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            raise DataProgressError(f"unsafe {label} size/type: {path}")
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, before.st_size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise DataProgressError(f"{label} disappeared while read: {path}") from exc
        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(current)
            or stat.S_ISLNK(current.st_mode)
        ):
            raise DataProgressError(f"{label} changed while read: {path}")
        return bytes(payload), before
    finally:
        os.close(descriptor)


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_checked_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataProgressError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataProgressError(f"{label} root must be an object")
    return payload


def _read_json(path: Path, *, label: str, maximum_bytes: int) -> tuple[dict[str, Any], os.stat_result]:
    raw, metadata = _stable_file_bytes(path, label=label, maximum_bytes=maximum_bytes)
    return _json_object(raw, label=label), metadata


def _plain_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataProgressError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    return _plain_int(value, label=label, minimum=1)


def _json_exact_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _json_exact_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_exact_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _require_exact_json(observed: Any, expected: Any, *, label: str) -> None:
    if not _json_exact_equal(observed, expected):
        raise DataProgressError(f"{label} identity mismatch")


def _require_exact_int(observed: Any, expected: int, *, label: str) -> int:
    value = _plain_int(observed, label=label)
    if value != expected:
        raise DataProgressError(f"{label} identity mismatch")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise DataProgressError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DataProgressError(f"unsafe {label}")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise DataProgressError(f"unsafe {label}")
    return value


def _stable_file_sha256(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[str, os.stat_result]:
    """Hash one bounded regular file without following a replaceable symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DataProgressError(f"cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            raise DataProgressError(f"unsafe {label} size/type: {path}")
        digest = hashlib.sha256()
        consumed = 0
        while consumed < before.st_size:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, before.st_size - consumed))
            if not chunk:
                break
            digest.update(chunk)
            consumed += len(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise DataProgressError(f"{label} disappeared while read: {path}") from exc
        if (
            consumed != before.st_size
            or os.read(descriptor, 1)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(current)
            or stat.S_ISLNK(current.st_mode)
        ):
            raise DataProgressError(f"{label} changed while read: {path}")
        return digest.hexdigest(), before
    finally:
        os.close(descriptor)


def _stable_directory_entries(path: Path, *, label: str) -> list[Path]:
    """Take one sorted directory snapshot and reject concurrent replacement."""

    before = _safe_directory(path, label=label)
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise DataProgressError(f"cannot enumerate {label}: {path}") from exc
    after = _safe_directory(path, label=label)
    if _file_identity(before) != _file_identity(after):
        raise DataProgressError(f"{label} changed while inspected: {path}")
    return entries


def _directory_inventory_snapshot(path: Path, *, label: str) -> tuple[Any, ...]:
    before = _safe_directory(path, label=label)
    entries: list[tuple[str, tuple[int, int, int, int, int, int]]] = []
    try:
        for entry in sorted(path.iterdir(), key=lambda item: item.name):
            entries.append((entry.name, _file_identity(entry.lstat())))
    except OSError as exc:
        raise DataProgressError(f"cannot snapshot {label}: {path}") from exc
    after = _safe_directory(path, label=label)
    if _file_identity(before) != _file_identity(after):
        raise DataProgressError(f"{label} changed while snapshotted: {path}")
    return (_file_identity(before), tuple(entries))


def _cache_inventory_snapshot(cache_root: Path) -> dict[str, tuple[Any, ...]]:
    snapshots = {
        str(cache_root.resolve(strict=True)): _directory_inventory_snapshot(
            cache_root, label="raw-token-cache root"
        )
    }
    archives_root = cache_root / "archives"
    if archives_root.exists() or archives_root.is_symlink():
        _safe_directory(archives_root, label="raw-token-cache archives root")
        snapshots[str(archives_root.resolve(strict=True))] = (
            _directory_inventory_snapshot(
                archives_root, label="raw-token-cache archives root"
            )
        )
        for entry in _stable_directory_entries(
            archives_root, label="raw-token-cache archives root"
        ):
            _safe_directory(entry, label="raw-token-cache bucket")
            snapshots[str(entry.resolve(strict=True))] = _directory_inventory_snapshot(
                entry, label="raw-token-cache bucket"
            )
            for target in _stable_directory_entries(
                entry, label="raw-token-cache bucket"
            ):
                if STAGE_DIRECTORY.fullmatch(target.name) is not None:
                    raise DataProgressError(
                        "in-flight cache directory prevents a coherent progress snapshot"
                    )
                _safe_directory(target, label="raw-token-cache archive entry")
                snapshots[str(target.resolve(strict=True))] = (
                    _directory_inventory_snapshot(
                        target, label="raw-token-cache archive entry"
                    )
                )
    return snapshots


def _safe_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DataProgressError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DataProgressError(f"{label} is not a real directory: {path}")
    return metadata


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DataProgressError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DataProgressError(f"{label} is not a regular non-symlink file: {path}")
    return metadata


def _progress(completed: int, total: int) -> dict[str, Any]:
    if completed < 0 or total < 1 or completed > total:
        raise DataProgressError(f"invalid progress fraction {completed}/{total}")
    percentage = completed * 100.0 / total
    if not math.isfinite(percentage):
        raise DataProgressError("non-finite progress percentage")
    return {
        "completed": completed,
        "total": total,
        "fraction": f"{completed}/{total}",
        "percent": round(percentage, 8),
        "percent_text": f"{percentage:.8f}",
    }


def _parse_journal(raw: bytes) -> tuple[int, list[dict[str, Any]]]:
    if not raw.endswith(b"\n"):
        raise DataProgressError("curation journal lacks a terminal newline")
    lines = raw.splitlines()
    expected = 1
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise DataProgressError(f"empty curation journal line {line_number}")
        event = _json_object(line, label=f"curation journal line {line_number}")
        sequence = _plain_int(event.get("sequence"), label="journal sequence", minimum=1)
        if sequence != expected:
            raise DataProgressError(
                f"curation journal sequence discontinuity: {sequence} != {expected}"
            )
        expected += 1
        events.append(event)
    return expected - 1, events


def _coherent_curation_projection(
    checkpoint_path: Path,
    journal_path: Path,
    *,
    attempts: int = 5,
) -> tuple[dict[str, Any], os.stat_result, list[dict[str, Any]]]:
    mismatch: tuple[int, int] | None = None
    for attempt in range(attempts):
        checkpoint, metadata = _read_json(
            checkpoint_path,
            label="curation checkpoint",
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        )
        journal_raw, _journal_metadata = _stable_file_bytes(
            journal_path,
            label="curation journal",
            maximum_bytes=MAX_JOURNAL_BYTES,
        )
        journal_sequence, events = _parse_journal(journal_raw)
        checkpoint_sequence = _plain_int(
            checkpoint.get("last_event_sequence"),
            label="checkpoint last_event_sequence",
        )
        if checkpoint_sequence == journal_sequence:
            return checkpoint, metadata, events
        mismatch = (checkpoint_sequence, journal_sequence)
        if attempt + 1 < attempts:
            time.sleep(0.02)
    assert mismatch is not None
    raise DataProgressError(
        "curation checkpoint/journal are incoherent after race retries: "
        f"{mismatch[0]} != {mismatch[1]}"
    )


def _collection_authority(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        raise DataProgressError("curation checkpoint identity is missing")
    completeness = identity.get("collection_completeness")
    if not isinstance(completeness, Mapping):
        raise DataProgressError("collection completeness authority is missing")
    expected_completeness_fields = {
        "format_version",
        "complete",
        "legacy_dedup_index_required",
        "pending_inputs",
        "preprocess_error_records",
        "collection_targets_exact_tokens",
        "quota_records",
        "completion_markers",
        "raw_archives",
        "reports",
        "fingerprints",
        "per_bucket",
    }
    if set(completeness) != expected_completeness_fields:
        raise DataProgressError("collection completeness authority schema mismatch")
    if (
        type(completeness.get("format_version")) is not int
        or completeness.get("format_version") != 1
        or completeness.get("complete") is not True
        or completeness.get("legacy_dedup_index_required") is not False
        or type(completeness.get("pending_inputs")) is not int
        or completeness.get("pending_inputs") != 0
        or type(completeness.get("preprocess_error_records")) is not int
        or completeness.get("preprocess_error_records") != 0
    ):
        raise DataProgressError("collection completeness authority is not closed and clean")
    per_bucket = completeness.get("per_bucket")
    if not isinstance(per_bucket, Mapping) or set(per_bucket) != set(BUCKETS):
        raise DataProgressError("collection authority bucket inventory is incomplete")
    normalized: dict[str, dict[str, int]] = {}
    for bucket in BUCKETS:
        raw = per_bucket[bucket]
        if not isinstance(raw, Mapping):
            raise DataProgressError(f"collection authority for {bucket} is malformed")
        if set(raw) != {
            "archives",
            "documents",
            "clean_bytes",
            "exact_tokens",
            "target_exact_tokens",
        }:
            raise DataProgressError(f"collection authority schema mismatch for {bucket}")
        values = {
            field: _positive_int(raw.get(field), label=f"{bucket}.{field}")
            for field in (
                "archives",
                "documents",
                "clean_bytes",
                "exact_tokens",
                "target_exact_tokens",
            )
        }
        if values["exact_tokens"] < values["target_exact_tokens"]:
            raise DataProgressError(f"collection authority target was not met for {bucket}")
        normalized[bucket] = values

    archives = sum(item["archives"] for item in normalized.values())
    documents = sum(item["documents"] for item in normalized.values())
    content_tokens = sum(item["exact_tokens"] for item in normalized.values())
    for field in ("quota_records", "raw_archives", "reports", "fingerprints"):
        descriptor = completeness.get(field)
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "count",
            "inventory_sha256",
        }:
            raise DataProgressError(f"collection {field} authority is missing")
        if _positive_int(descriptor.get("count"), label=f"{field}.count") != archives:
            raise DataProgressError(f"collection {field} count disagrees with per-bucket authority")
        _sha256(descriptor.get("inventory_sha256"), label=f"{field}.inventory_sha256")
    completion_markers = completeness.get("completion_markers")
    if not isinstance(completion_markers, Mapping) or set(completion_markers) != {
        "count",
        "inventory_sha256",
        "files",
    }:
        raise DataProgressError("collection completion marker authority is malformed")
    marker_count = _positive_int(
        completion_markers.get("count"), label="completion_markers.count"
    )
    _sha256(
        completion_markers.get("inventory_sha256"),
        label="completion_markers.inventory_sha256",
    )
    marker_files = completion_markers.get("files")
    if not isinstance(marker_files, list) or len(marker_files) != marker_count:
        raise DataProgressError("collection completion marker inventory is incomplete")
    marker_bucket_inventory: list[str] = []
    for marker in marker_files:
        if not isinstance(marker, Mapping) or set(marker) != {"path", "sha256", "buckets"}:
            raise DataProgressError("collection completion marker descriptor is malformed")
        _safe_relative(marker.get("path"), label="collection completion marker path")
        _sha256(marker.get("sha256"), label="completion marker SHA-256")
        marker_buckets = marker.get("buckets")
        if (
            not isinstance(marker_buckets, list)
            or not marker_buckets
            or any(
                not isinstance(bucket, str) or bucket not in BUCKETS
                for bucket in marker_buckets
            )
        ):
            raise DataProgressError("collection completion marker buckets are malformed")
        marker_bucket_inventory.extend(marker_buckets)
    if (
        marker_bucket_inventory != list(dict.fromkeys(marker_bucket_inventory))
        or set(marker_bucket_inventory) != set(BUCKETS)
    ):
        raise DataProgressError(
            "collection completion markers must cover every bucket exactly once"
        )
    if hashlib.sha256(_canonical_json_bytes(marker_files)).hexdigest() != completion_markers[
        "inventory_sha256"
    ]:
        raise DataProgressError("collection completion marker inventory digest mismatch")
    targets = completeness.get("collection_targets_exact_tokens")
    if not isinstance(targets, Mapping) or set(targets) != set(BUCKETS):
        raise DataProgressError("collection token targets are malformed")
    if any(
        type(targets[bucket]) is not int
        or targets[bucket] != normalized[bucket]["target_exact_tokens"]
        for bucket in BUCKETS
    ):
        raise DataProgressError("collection token targets disagree with per-bucket authority")
    return {
        "archives": archives,
        "documents": documents,
        "content_tokens": content_tokens,
        "per_bucket": normalized,
        "inventories": {
            field: dict(completeness[field])
            for field in ("quota_records", "raw_archives", "reports", "fingerprints")
        },
        "completion_markers": {
            "count": marker_count,
            "inventory_sha256": completion_markers["inventory_sha256"],
            "files": [dict(marker) for marker in marker_files],
        },
    }


def _file_under(root: Path, relative: str, *, label: str) -> Path:
    relative = _safe_relative(relative, label=f"{label} path")
    _safe_directory(root, label=f"{label} root")
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise DataProgressError(f"missing {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise DataProgressError(f"symlink in {label} path: {current}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise DataProgressError(f"{label} is not a regular file: {current}")
    return current


def _tokenizer_authority(tokenizer_root: Path) -> dict[str, Any]:
    """Authenticate the configured generation's tokenizer without importing it."""

    _safe_directory(tokenizer_root, label="tokenizer root")
    manifest_path = tokenizer_root / "TOKENIZER_MANIFEST.json"
    manifest_raw, manifest_metadata = _stable_file_bytes(
        manifest_path,
        label="tokenizer manifest",
        maximum_bytes=MAX_TOKENIZER_MANIFEST_BYTES,
    )
    manifest = _json_object(manifest_raw, label="tokenizer manifest")
    _require_exact_int(
        manifest.get("manifest_version"), 1, label="tokenizer manifest version"
    )
    if manifest.get("repo_id") != "bigcode/starcoder2-tokenizer":
        raise DataProgressError("tokenizer generation repository identity mismatch")
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or LOWER_COMMIT.fullmatch(revision) is None:
        raise DataProgressError("tokenizer generation revision is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or "tokenizer.json" not in files:
        raise DataProgressError("tokenizer manifest lacks a closed file inventory")
    declared = set(files)
    if any(
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).name != name
        or name == "TOKENIZER_MANIFEST.json"
        for name in declared
    ):
        raise DataProgressError("tokenizer manifest has an unsafe file identity")
    entries = _stable_directory_entries(tokenizer_root, label="tokenizer root")
    actual = {entry.name for entry in entries if entry.name != "TOKENIZER_MANIFEST.json"}
    if actual != declared:
        raise DataProgressError("tokenizer directory differs from its closed file inventory")
    tokenizer_raw: bytes | None = None
    file_metadata: dict[str, os.stat_result] = {}
    for name in sorted(declared):
        descriptor = files[name]
        if not isinstance(descriptor, Mapping):
            raise DataProgressError("tokenizer file descriptor is malformed")
        expected_bytes = _plain_int(
            descriptor.get("bytes"), label=f"tokenizer {name} bytes"
        )
        expected_sha = _sha256(
            descriptor.get("sha256"), label=f"tokenizer {name} SHA-256"
        )
        path = tokenizer_root / name
        digest, metadata = _stable_file_sha256(
            path, label=f"tokenizer file {name}", maximum_bytes=MAX_TOKENIZER_FILE_BYTES
        )
        if metadata.st_size != expected_bytes or digest != expected_sha:
            raise DataProgressError("tokenizer file differs from manifest authority")
        file_metadata[name] = metadata
        if name == "tokenizer.json":
            tokenizer_raw, tokenizer_metadata = _stable_file_bytes(
                path,
                label="tokenizer vocabulary",
                maximum_bytes=MAX_TOKENIZER_FILE_BYTES,
            )
            if hashlib.sha256(tokenizer_raw).hexdigest() != digest:
                raise DataProgressError("tokenizer changed between identity reads")
            file_metadata[name] = tokenizer_metadata
    assert tokenizer_raw is not None
    tokenizer_payload = _json_object(tokenizer_raw, label="tokenizer.json")
    model = tokenizer_payload.get("model")
    vocabulary = model.get("vocab") if isinstance(model, Mapping) else None
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise DataProgressError("tokenizer.json lacks a vocabulary authority")
    normalized_vocabulary: dict[str, int] = {}
    for token, identifier in vocabulary.items():
        if not isinstance(token, str) or isinstance(identifier, bool) or not isinstance(
            identifier, int
        ):
            raise DataProgressError("tokenizer vocabulary identity is malformed")
        normalized_vocabulary[token] = identifier
    added_tokens = tokenizer_payload.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise DataProgressError("tokenizer added-token authority is malformed")
    for added in added_tokens:
        if not isinstance(added, Mapping):
            raise DataProgressError("tokenizer added-token descriptor is malformed")
        token = added.get("content")
        identifier = added.get("id")
        if not isinstance(token, str) or isinstance(identifier, bool) or not isinstance(
            identifier, int
        ):
            raise DataProgressError("tokenizer added-token identity is malformed")
        previous = normalized_vocabulary.get(token)
        if previous is not None and previous != identifier:
            raise DataProgressError("tokenizer added token conflicts with model vocabulary")
        normalized_vocabulary[token] = identifier
    identifiers = list(normalized_vocabulary.values())
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(
        range(len(identifiers))
    ):
        raise DataProgressError("tokenizer IDs are not a closed contiguous vocabulary")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise DataProgressError("tokenizer manifest validation authority is missing")
    vocab_size = len(normalized_vocabulary)
    eos_token = validation.get("eos_token")
    eos_id = _plain_int(
        validation.get("eos_token_id"), label="tokenizer manifest EOS token ID"
    )
    if (
        vocab_size != 49_152
        or type(validation.get("vocab_size")) is not int
        or validation.get("vocab_size") != vocab_size
        or not isinstance(eos_token, str)
        or not eos_token
        or eos_id >= vocab_size
        or normalized_vocabulary.get(eos_token) != eos_id
    ):
        raise DataProgressError("tokenizer vocabulary/EOS authority mismatch")
    ordered = sorted(normalized_vocabulary.items(), key=lambda item: (item[1], item[0]))
    vocabulary_sha = hashlib.sha256(
        json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if _file_identity(manifest_metadata) != _file_identity(
        _regular_file(manifest_path, label="tokenizer manifest")
    ):
        raise DataProgressError("tokenizer manifest changed while inspected")
    for name, metadata in file_metadata.items():
        if _file_identity(metadata) != _file_identity(
            _regular_file(tokenizer_root / name, label="tokenizer file")
        ):
            raise DataProgressError("tokenizer file changed while inspected")
    return {
        "repo_id": "bigcode/starcoder2-tokenizer",
        "resolved_revision": revision,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "vocabulary_sha256": vocabulary_sha,
        "vocab_size": vocab_size,
        "eos_token": eos_token,
        "eos_token_id": eos_id,
        "eos_present_in_payload": False,
    }


def _generation_authority(
    generation_root: Path,
    preprocess_root: Path,
    tokenizer_root: Path,
    authority: Mapping[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Reconstruct the exact per-part authority pinned by the checkpoint."""

    _safe_directory(generation_root, label="dataset generation root")
    _safe_directory(preprocess_root, label="preprocess generation root")
    reports_root = preprocess_root / "reports"
    _safe_directory(reports_root, label="preprocess reports root")
    report_entries: list[tuple[str, Path]] = []
    bucket_entries = _stable_directory_entries(reports_root, label="preprocess reports root")
    if {entry.name for entry in bucket_entries} != set(BUCKETS):
        raise DataProgressError("preprocess report bucket inventory is incomplete or unexpected")
    for bucket_root in bucket_entries:
        if bucket_root.name not in BUCKETS:
            raise DataProgressError("unexpected preprocess report bucket")
        _safe_directory(bucket_root, label="preprocess report bucket")
        for report_path in _stable_directory_entries(
            bucket_root, label="preprocess report bucket"
        ):
            match = re.fullmatch(r"part-(\d{6})\.json", report_path.name)
            if match is None:
                raise DataProgressError("pending or malformed preprocess report identity")
            report_entries.append((bucket_root.name, report_path))

    sources: dict[tuple[str, int], dict[str, Any]] = {}
    report_identity: list[dict[str, str]] = []
    fingerprint_identity: list[dict[str, str]] = []
    raw_identity: list[dict[str, Any]] = []
    seen_archives: set[str] = set()
    seen_fingerprints: set[str] = set()
    seen_shards: set[str] = set()
    actual_totals = {
        bucket: {"archives": 0, "documents": 0, "clean_bytes": 0, "exact_tokens": 0}
        for bucket in BUCKETS
    }
    for bucket, report_path in sorted(report_entries, key=lambda item: str(item[1])):
        raw, metadata = _stable_file_bytes(
            report_path,
            label="preprocess report",
            maximum_bytes=MAX_PREPROCESS_REPORT_BYTES,
        )
        report = _json_object(raw, label="preprocess report")
        index_match = re.fullmatch(r"part-(\d{6})\.json", report_path.name)
        assert index_match is not None
        index = int(index_match.group(1))
        key = (bucket, index)
        if (
            key in sources
            or report.get("bucket") != bucket
            or type(report.get("index")) is not int
            or report.get("index") != index
        ):
            raise DataProgressError("duplicate or wrong preprocess report part identity")
        if (
            type(report.get("report_version")) is not int
            or report.get("report_version") != 1
            or type(report.get("fingerprint_version")) is not int
            or report.get("fingerprint_version") != 1
        ):
            raise DataProgressError("unsupported preprocess report identity version")
        archive = _safe_relative(report.get("archive"), label="raw archive path")
        expected_archive = f"{BUCKET_RAW_PATHS[bucket]}/part-{index:06d}.tar.zst"
        fingerprint = _safe_relative(
            report.get("fingerprint_file"), label="fingerprint path"
        )
        expected_fingerprint = f"fingerprints/{bucket}/part-{index:06d}.jsonl.zst"
        shard = report.get("quota_shard_id")
        if (
            archive != expected_archive
            or fingerprint != expected_fingerprint
            or not isinstance(shard, str)
            or not shard
            or archive in seen_archives
            or fingerprint in seen_fingerprints
            or shard in seen_shards
        ):
            raise DataProgressError("duplicate or wrong generation part identity")
        seen_archives.add(archive)
        seen_fingerprints.add(fingerprint)
        seen_shards.add(shard)
        documents = _positive_int(report.get("documents"), label="report documents")
        clean_bytes = _positive_int(report.get("clean_bytes"), label="report clean_bytes")
        tokens = _positive_int(report.get("exact_tokens"), label="report exact_tokens")
        archive_bytes = _positive_int(
            report.get("archive_compressed_bytes"), label="report archive bytes"
        )
        archive_sha = _sha256(report.get("archive_sha256"), label="report archive SHA-256")
        fingerprint_sha = _sha256(
            report.get("fingerprint_sha256"), label="report fingerprint SHA-256"
        )
        archive_path = _file_under(generation_root, archive, label="raw archive")
        fingerprint_path = _file_under(
            preprocess_root, fingerprint, label="fingerprint shard"
        )
        archive_metadata = archive_path.lstat()
        fingerprint_metadata = fingerprint_path.lstat()
        if archive_metadata.st_size != archive_bytes or fingerprint_metadata.st_size < 1:
            raise DataProgressError("generation source size differs from preprocess authority")
        report_relative = report_path.relative_to(preprocess_root).as_posix()
        report_sha = hashlib.sha256(raw).hexdigest()
        source = {
            "archive": {
                "path": archive,
                "bucket": bucket,
                "index": index,
                "bytes": archive_bytes,
                "sha256": archive_sha,
            },
            "preprocess_report": {
                "path": report_relative,
                "bytes": metadata.st_size,
                "sha256": report_sha,
                "report_version": 1,
            },
            "fingerprint": {
                "path": fingerprint,
                "bytes": fingerprint_metadata.st_size,
                "sha256": fingerprint_sha,
                "fingerprint_version": 1,
            },
        }
        sources[key] = {
            "source": source,
            "documents": {
                "records": documents,
                "clean_bytes": clean_bytes,
                "content_tokens": tokens,
            },
            "report_sha256": report_sha,
            "report_path": report_path,
            "report_metadata": metadata,
            "archive_metadata": archive_metadata,
            "fingerprint_metadata": fingerprint_metadata,
        }
        report_identity.append({"path": report_relative, "sha256": report_sha})
        fingerprint_identity.append({"path": fingerprint, "sha256": fingerprint_sha})
        raw_identity.append(
            {
                "archive": archive,
                "bytes": archive_bytes,
                "quota_shard_id": shard,
                "sha256": archive_sha,
            }
        )
        totals = actual_totals[bucket]
        totals["archives"] += 1
        totals["documents"] += documents
        totals["clean_bytes"] += clean_bytes
        totals["exact_tokens"] += tokens

    for bucket in BUCKETS:
        expected = authority["per_bucket"][bucket]
        if actual_totals[bucket] != {
            field: expected[field]
            for field in ("archives", "documents", "clean_bytes", "exact_tokens")
        }:
            raise DataProgressError("generation report totals differ from checkpoint authority")
    inventory_payloads = {
        "reports": report_identity,
        "fingerprints": fingerprint_identity,
        "raw_archives": raw_identity,
    }
    for name, payload in inventory_payloads.items():
        expected = authority["inventories"][name]
        if len(payload) != expected["count"] or hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest() != expected["inventory_sha256"]:
            raise DataProgressError(f"generation {name} inventory differs from checkpoint")

    expected_fingerprint_paths = {
        preprocess_root / item for item in seen_fingerprints
    }
    fingerprints_root = preprocess_root / "fingerprints"
    _safe_directory(fingerprints_root, label="fingerprint root")
    observed_fingerprints: set[Path] = set()
    fingerprint_buckets = _stable_directory_entries(
        fingerprints_root, label="fingerprint root"
    )
    if {entry.name for entry in fingerprint_buckets} != set(BUCKETS):
        raise DataProgressError("fingerprint bucket inventory is incomplete or unexpected")
    for bucket_root in fingerprint_buckets:
        _safe_directory(bucket_root, label="fingerprint bucket")
        for path in _stable_directory_entries(bucket_root, label="fingerprint bucket"):
            if re.fullmatch(r"part-\d{6}\.jsonl\.zst", path.name) is None:
                raise DataProgressError("pending or malformed fingerprint part identity")
            _regular_file(path, label="fingerprint shard")
            observed_fingerprints.add(path)
    if observed_fingerprints != expected_fingerprint_paths:
        raise DataProgressError("missing, wrong, or duplicate fingerprint part identity")

    for bucket in BUCKETS:
        raw_bucket = generation_root / BUCKET_RAW_PATHS[bucket]
        _safe_directory(raw_bucket, label="raw archive bucket")
        expected_paths = {
            generation_root / item
            for item in seen_archives
            if item.startswith(BUCKET_RAW_PATHS[bucket] + "/")
        }
        observed_paths: set[Path] = set()
        for path in _stable_directory_entries(raw_bucket, label="raw archive bucket"):
            if path.name.startswith(".part-"):
                raise DataProgressError("in-flight raw archive part remains")
            if path.name.startswith("part-"):
                if re.fullmatch(r"part-\d{6}\.tar\.zst", path.name) is None:
                    raise DataProgressError("malformed raw archive part identity")
                _regular_file(path, label="raw archive")
                observed_paths.add(path)
        if observed_paths != expected_paths:
            raise DataProgressError("missing, wrong, or duplicate raw archive part identity")

    for marker in authority["completion_markers"]["files"]:
        marker_path = _file_under(
            generation_root, marker["path"], label="collection completion marker"
        )
        marker_sha, _ = _stable_file_sha256(
            marker_path,
            label="collection completion marker",
            maximum_bytes=MAX_PREPROCESS_REPORT_BYTES,
        )
        if marker_sha != marker["sha256"]:
            raise DataProgressError("collection completion marker changed")

    for item in sources.values():
        archive_path = generation_root / item["source"]["archive"]["path"]
        fingerprint_path = preprocess_root / item["source"]["fingerprint"]["path"]
        if (
            _file_identity(item["report_metadata"])
            != _file_identity(
                _regular_file(item["report_path"], label="preprocess report")
            )
            or _file_identity(item["archive_metadata"])
            != _file_identity(_regular_file(archive_path, label="raw archive"))
            or _file_identity(item["fingerprint_metadata"])
            != _file_identity(_regular_file(fingerprint_path, label="fingerprint shard"))
        ):
            raise DataProgressError("generation source changed while inspected")

    tokenizer = _tokenizer_authority(tokenizer_root)
    return sources, {
        "root": str(generation_root.resolve(strict=True)),
        "preprocess_root": str(preprocess_root.resolve(strict=True)),
        "tokenizer_root": str(tokenizer_root.resolve(strict=True)),
        "reports": len(sources),
        "tokenizer": tokenizer,
    }


def _curation_progress(
    checkpoint_path: Path,
    journal_path: Path,
    generation_root: Path,
    preprocess_root: Path,
    tokenizer_root: Path,
    *,
    stale_seconds: float,
    now: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, Any],
]:
    checkpoint, metadata, events = _coherent_curation_projection(
        checkpoint_path, journal_path
    )
    _require_exact_int(
        checkpoint.get("checkpoint_version"),
        2,
        label="curation checkpoint version",
    )
    _require_exact_int(
        checkpoint.get("database_version"), 4, label="curation database version"
    )
    phase = checkpoint.get("phase")
    if phase not in CURATION_PHASES:
        raise DataProgressError(f"unknown curation phase {phase!r}")
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        raise DataProgressError("curation checkpoint identity is missing")
    fast_handoff = identity.get("fast_all_eligible_handoff")
    if fast_handoff is not None and not _json_exact_equal(
        fast_handoff, FAST_ALL_ELIGIBLE_HANDOFF_PROFILE
    ):
        raise DataProgressError("curation fast-handoff profile identity mismatch")
    terminal = phase == "complete" or (
        _json_exact_equal(fast_handoff, FAST_ALL_ELIGIBLE_HANDOFF_PROFILE)
        and phase == "canonicalized"
    )
    age = now - metadata.st_mtime
    if age < -5.0:
        raise DataProgressError("curation checkpoint modification time is in the future")
    age = max(0.0, age)
    if not terminal and age > stale_seconds:
        raise DataProgressError(
            f"active curation checkpoint is stale: {age:.3f}s > {stale_seconds:.3f}s"
        )

    authority = _collection_authority(checkpoint)
    source_authority, generation = _generation_authority(
        generation_root, preprocess_root, tokenizer_root, authority
    )
    if (
        type(identity.get("report_count")) is not int
        or identity.get("report_count") != authority["archives"]
        or identity.get("report_inventory_sha256")
        != identity["collection_completeness"]["reports"]["inventory_sha256"]
    ):
        raise DataProgressError("curation report inventory identity is inconsistent")
    counts = checkpoint.get("counts")
    if not isinstance(counts, Mapping):
        raise DataProgressError("curation counters are missing")
    archives = _plain_int(counts.get("archives"), label="curation archives")
    documents = _plain_int(counts.get("documents"), label="curation documents")
    selected = _plain_int(
        counts.get("selected_documents"), label="curation selected_documents"
    )
    output_archives = _plain_int(
        counts.get("output_archives"), label="curation output_archives"
    )
    if archives > authority["archives"] or documents > authority["documents"]:
        raise DataProgressError("curation counters exceed frozen collection authority")
    storage = checkpoint.get("storage")
    if not isinstance(storage, Mapping) or storage.get("violation") != {}:
        raise DataProgressError("curation storage authority is missing or violated")
    preflight = storage.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("status") != "pass"
        or type(preflight.get("documents_expected")) is not int
        or preflight.get("documents_expected") != authority["documents"]
    ):
        raise DataProgressError("curation storage preflight disagrees with collection authority")

    subphases = checkpoint.get("subphases")
    if not isinstance(subphases, list):
        raise DataProgressError("curation subphase projection is missing")
    inventory_rows = 0
    inventory_tokens = 0
    completed_inventory_archives = 0
    running_inventory = 0
    seen_inventory: set[tuple[str, int]] = set()
    active_subphases: list[str] = []
    for raw_subphase in subphases:
        if not isinstance(raw_subphase, Mapping):
            raise DataProgressError("curation subphase entry is malformed")
        name = raw_subphase.get("subphase")
        status_value = raw_subphase.get("status")
        if not isinstance(name, str) or status_value not in {"running", "complete"}:
            raise DataProgressError("curation subphase name/status is malformed")
        if status_value == "running":
            active_subphases.append(name)
        match = INVENTORY_SUBPHASE.fullmatch(name)
        if match is None:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in seen_inventory:
            raise DataProgressError(f"duplicate inventory subphase {name}")
        seen_inventory.add(key)
        details = raw_subphase.get("details")
        if not isinstance(details, Mapping):
            raise DataProgressError(f"inventory subphase details are missing: {name}")
        expected_rows = _positive_int(
            details.get("expected_documents"), label=f"{name}.expected_documents"
        )
        expected_tokens = _positive_int(
            details.get("expected_tokens"), label=f"{name}.expected_tokens"
        )
        expected_source = source_authority.get(key)
        if expected_source is None:
            raise DataProgressError(f"inventory subphase has no generation authority: {name}")
        expected_documents = expected_source["documents"]
        if (
            details.get("archive") != expected_source["source"]["archive"]["path"]
            or details.get("report_sha256") != expected_source["report_sha256"]
            or expected_rows != expected_documents["records"]
            or expected_tokens != expected_documents["content_tokens"]
            or type(details.get("expected_clean_bytes")) is not int
            or details.get("expected_clean_bytes") != expected_documents["clean_bytes"]
        ):
            raise DataProgressError(
                f"inventory subphase differs from generation authority: {name}"
            )
        rows = _plain_int(raw_subphase.get("processed_rows"), label=f"{name}.processed_rows")
        tokens = _plain_int(
            raw_subphase.get("processed_tokens"), label=f"{name}.processed_tokens"
        )
        _plain_int(
            raw_subphase.get("committed_batches"),
            label=f"{name}.committed_batches",
        )
        cursor = raw_subphase.get("cursor")
        clean_bytes = _plain_int(
            details.get("clean_bytes"), label=f"{name}.clean_bytes"
        )
        if (
            not isinstance(cursor, Mapping)
            or set(cursor) != {"input_rows"}
            or type(cursor.get("input_rows")) is not int
            or cursor.get("input_rows") != rows
            or clean_bytes > expected_documents["clean_bytes"]
        ):
            raise DataProgressError(f"inventory subphase cursor/bytes are invalid: {name}")
        if rows > expected_rows or tokens > expected_tokens:
            raise DataProgressError(f"inventory subphase exceeds its report authority: {name}")
        if status_value == "complete":
            if (
                rows != expected_rows
                or tokens != expected_tokens
                or clean_bytes != expected_documents["clean_bytes"]
                or type(details.get("validated_documents")) is not int
                or details.get("validated_documents") != rows
                or type(details.get("validated_tokens")) is not int
                or details.get("validated_tokens") != tokens
            ):
                raise DataProgressError(f"completed inventory subphase is incomplete: {name}")
            completed_inventory_archives += 1
        else:
            running_inventory += 1
        inventory_rows += rows
        inventory_tokens += tokens
    if len(active_subphases) > 1:
        raise DataProgressError("curation has multiple running subphases")
    if terminal and active_subphases:
        raise DataProgressError("terminal curation has a running subphase")
    if running_inventory > 1:
        raise DataProgressError("curation has multiple running inventory archives")
    if inventory_rows != documents or completed_inventory_archives != archives:
        raise DataProgressError("curation inventory subphases disagree with durable counters")
    if inventory_tokens > authority["content_tokens"]:
        raise DataProgressError("curation inventory tokens exceed collection authority")
    if phase != "inventory" and (
        inventory_rows != authority["documents"]
        or inventory_tokens != authority["content_tokens"]
        or completed_inventory_archives != authority["archives"]
    ):
        raise DataProgressError("post-inventory phase lacks complete inventory evidence")
    archive_events = [event for event in events if event.get("event") == "archive_ingested"]
    if len(archive_events) != archives:
        raise DataProgressError("curation journal/archive counter mismatch")
    journal_documents = 0
    journal_tokens = 0
    journal_archives: set[str] = set()
    for event in archive_events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise DataProgressError("archive_ingested journal payload is malformed")
        archive = payload.get("archive")
        if not isinstance(archive, str):
            raise DataProgressError("archive_ingested journal part identity is malformed")
        expected_item = next(
            (
                item
                for item in source_authority.values()
                if item["source"]["archive"]["path"] == archive
            ),
            None,
        )
        if expected_item is None or archive in journal_archives:
            raise DataProgressError("archive_ingested journal part identity is wrong/duplicate")
        journal_archives.add(str(archive))
        if (
            type(payload.get("documents")) is not int
            or payload.get("documents") != expected_item["documents"]["records"]
            or type(payload.get("tokens")) is not int
            or payload.get("tokens") != expected_item["documents"]["content_tokens"]
        ):
            raise DataProgressError("archive_ingested journal authority mismatch")
        journal_documents += _positive_int(
            payload.get("documents"), label="archive_ingested documents"
        )
        journal_tokens += _positive_int(
            payload.get("tokens"), label="archive_ingested tokens"
        )
    complete_inventory_rows = sum(
        _plain_int(subphase.get("processed_rows"), label="inventory processed_rows")
        for subphase in subphases
        if isinstance(subphase, Mapping)
        and INVENTORY_SUBPHASE.fullmatch(str(subphase.get("subphase"))) is not None
        and subphase.get("status") == "complete"
    )
    complete_inventory_tokens = sum(
        _plain_int(subphase.get("processed_tokens"), label="inventory processed_tokens")
        for subphase in subphases
        if isinstance(subphase, Mapping)
        and INVENTORY_SUBPHASE.fullmatch(str(subphase.get("subphase"))) is not None
        and subphase.get("status") == "complete"
    )
    if (
        journal_documents != complete_inventory_rows
        or journal_tokens != complete_inventory_tokens
    ):
        raise DataProgressError("curation journal totals disagree with completed inventory")

    return authority, {
        "phase": phase,
        "terminal": terminal,
        "checkpoint": str(checkpoint_path.resolve(strict=True)),
        "journal": str(journal_path.resolve(strict=True)),
        "checkpoint_mtime_utc": _utc(metadata.st_mtime),
        "checkpoint_age_seconds": round(age, 6),
        "last_event_sequence": checkpoint["last_event_sequence"],
        "active_subphase": active_subphases[0] if active_subphases else None,
        "counts": {
            "archives": _progress(archives, authority["archives"]),
            "documents": _progress(documents, authority["documents"]),
            "content_tokens": _progress(inventory_tokens, authority["content_tokens"]),
            "selected_documents": selected,
            "output_archives": output_archives,
        },
    }, source_authority, generation


def _validate_tokenizer_descriptor(value: Any) -> dict[str, Any]:
    expected = {
        "repo_id",
        "resolved_revision",
        "manifest_sha256",
        "vocabulary_sha256",
        "vocab_size",
        "eos_token",
        "eos_token_id",
        "eos_present_in_payload",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise DataProgressError("cache tokenizer descriptor schema mismatch")
    if value.get("repo_id") != "bigcode/starcoder2-tokenizer":
        raise DataProgressError("cache tokenizer repository identity mismatch")
    if not isinstance(value.get("resolved_revision"), str) or LOWER_COMMIT.fullmatch(
        value["resolved_revision"]
    ) is None:
        raise DataProgressError("cache tokenizer revision is invalid")
    _sha256(value.get("manifest_sha256"), label="tokenizer manifest SHA-256")
    _sha256(value.get("vocabulary_sha256"), label="tokenizer vocabulary SHA-256")
    vocab_size = _positive_int(value.get("vocab_size"), label="tokenizer vocab_size")
    eos_id = _plain_int(value.get("eos_token_id"), label="tokenizer eos_token_id")
    if vocab_size != 49_152 or eos_id >= vocab_size:
        raise DataProgressError("cache tokenizer vocabulary/EOS identity mismatch")
    if not isinstance(value.get("eos_token"), str) or not value["eos_token"]:
        raise DataProgressError("cache tokenizer EOS token is invalid")
    if value.get("eos_present_in_payload") is not False:
        raise DataProgressError("raw cache unexpectedly claims EOS in its payload")
    return value


def _cache_manifest(target: Path, *, bucket: str, index: int) -> tuple[dict[str, Any], os.stat_result]:
    directory_before = _safe_directory(target, label="completed cache directory")
    names = {
        entry.name
        for entry in _stable_directory_entries(target, label="completed cache directory")
    }
    if names != CACHE_OUTPUT_FILES:
        raise DataProgressError(f"completed cache file set mismatch: {target}")
    manifest_path = target / "manifest.json"
    raw, metadata = _stable_file_bytes(
        manifest_path,
        label="cache manifest",
        maximum_bytes=MAX_CACHE_MANIFEST_BYTES,
    )
    manifest = _json_object(raw, label="cache manifest")
    if raw != _canonical_json_bytes(manifest):
        raise DataProgressError(f"cache manifest is not canonical JSON: {manifest_path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar, sidecar_metadata = _stable_file_bytes(
        target / "manifest.sha256",
        label="cache manifest sidecar",
        maximum_bytes=256,
    )
    if sidecar != f"{digest}  manifest.json\n".encode("ascii"):
        raise DataProgressError(f"cache manifest sidecar mismatch: {target}")
    if set(manifest) != CACHE_TOP_LEVEL:
        raise DataProgressError(f"cache manifest top-level schema mismatch: {target}")
    if (
        manifest.get("format") != "raw-document-token-cache"
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != 1
        or manifest.get("profile") != "all-raw-documents-content-only-v1"
        or manifest.get("cache_complete") is not True
        or manifest.get("training_ready") is not False
        or not _json_exact_equal(manifest.get("tokenization"), TOKENIZATION_CONTRACT)
        or not _json_exact_equal(manifest.get("non_authorities"), NON_AUTHORITIES)
    ):
        raise DataProgressError(f"cache manifest contract mismatch: {target}")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "archive",
        "preprocess_report",
        "fingerprint",
    }:
        raise DataProgressError(f"cache source authority is malformed: {target}")
    archive = source["archive"]
    if not isinstance(archive, Mapping) or set(archive) != {
        "path",
        "bucket",
        "index",
        "bytes",
        "sha256",
    }:
        raise DataProgressError(f"cache archive authority is malformed: {target}")
    if (
        archive.get("bucket") != bucket
        or type(archive.get("index")) is not int
        or archive.get("index") != index
    ):
        raise DataProgressError(f"cache archive identity/path mismatch: {target}")
    if archive.get("path") != f"{BUCKET_RAW_PATHS[bucket]}/part-{index:06d}.tar.zst":
        raise DataProgressError(f"cache archive source path mismatch: {target}")
    _positive_int(archive.get("bytes"), label="cache archive bytes")
    _sha256(archive.get("sha256"), label="cache archive SHA-256")
    for source_name in ("preprocess_report", "fingerprint"):
        descriptor = source[source_name]
        version_field = (
            "report_version" if source_name == "preprocess_report" else "fingerprint_version"
        )
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "bytes", "sha256", version_field}
            or not isinstance(descriptor.get("path"), str)
            or type(descriptor.get(version_field)) is not int
            or descriptor.get(version_field) != 1
        ):
            raise DataProgressError(f"cache {source_name} authority is malformed")
        expected_path = (
            f"reports/{bucket}/part-{index:06d}.json"
            if source_name == "preprocess_report"
            else f"fingerprints/{bucket}/part-{index:06d}.jsonl.zst"
        )
        if descriptor.get("path") != expected_path:
            raise DataProgressError(f"cache {source_name} path identity mismatch")
        _positive_int(descriptor.get("bytes"), label=f"cache {source_name} bytes")
        _sha256(descriptor.get("sha256"), label=f"cache {source_name} SHA-256")

    tokenizer = _validate_tokenizer_descriptor(manifest.get("tokenizer"))
    documents = manifest.get("documents")
    if not isinstance(documents, Mapping) or set(documents) != {
        "records",
        "clean_bytes",
        "content_tokens",
        "alignment",
    }:
        raise DataProgressError(f"cache document descriptor is malformed: {target}")
    records = _positive_int(documents.get("records"), label="cache document records")
    clean_bytes = _positive_int(documents.get("clean_bytes"), label="cache clean bytes")
    content_tokens = _positive_int(
        documents.get("content_tokens"), label="cache content tokens"
    )
    alignment = documents.get("alignment")
    if not isinstance(alignment, Mapping) or set(alignment) != {
        "authority",
        "record_sha256",
        "offset_items",
        "offset_zero",
        "terminal_offset",
    } or (
        alignment.get("authority")
        != "manifest-index+raw-manifest+preprocess-fingerprint"
        or type(alignment.get("offset_items")) is not int
        or alignment.get("offset_items") != records + 1
        or type(alignment.get("offset_zero")) is not int
        or alignment.get("offset_zero") != 0
        or type(alignment.get("terminal_offset")) is not int
        or alignment.get("terminal_offset") != content_tokens
    ):
        raise DataProgressError(f"cache alignment authority is malformed: {target}")
    _sha256(alignment.get("record_sha256"), label="cache alignment SHA-256")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, Mapping) or set(payloads) != {"tokens", "offsets"}:
        raise DataProgressError(f"cache payload descriptor is malformed: {target}")
    tokens = payloads["tokens"]
    offsets = payloads["offsets"]
    if (
        not isinstance(tokens, Mapping)
        or set(tokens) != {
            "path",
            "dtype",
            "endianness",
            "items",
            "bytes",
            "sha256",
            "minimum_id",
            "maximum_id",
        }
        or not isinstance(offsets, Mapping)
        or set(offsets) != {
            "path",
            "dtype",
            "endianness",
            "items",
            "bytes",
            "sha256",
        }
    ):
        raise DataProgressError(f"cache payload descriptors are malformed: {target}")
    expected_token_fields = {
        "path": "tokens.u16",
        "dtype": "uint16",
        "endianness": "little",
        "items": content_tokens,
        "bytes": content_tokens * 2,
    }
    expected_offset_fields = {
        "path": "offsets.u64",
        "dtype": "uint64",
        "endianness": "little",
        "items": records + 1,
        "bytes": (records + 1) * 8,
    }
    if any(
        not _json_exact_equal(tokens.get(key), value)
        for key, value in expected_token_fields.items()
    ):
        raise DataProgressError(f"cache token payload arithmetic mismatch: {target}")
    if any(
        not _json_exact_equal(offsets.get(key), value)
        for key, value in expected_offset_fields.items()
    ):
        raise DataProgressError(f"cache offset payload arithmetic mismatch: {target}")
    _sha256(tokens.get("sha256"), label="cache token payload SHA-256")
    _sha256(offsets.get("sha256"), label="cache offset payload SHA-256")
    minimum_id = _plain_int(tokens.get("minimum_id"), label="cache minimum token ID")
    maximum_id = _plain_int(tokens.get("maximum_id"), label="cache maximum token ID")
    if minimum_id > maximum_id or maximum_id >= tokenizer["vocab_size"]:
        raise DataProgressError(f"cache token ID range is invalid: {target}")
    token_file = _regular_file(target / "tokens.u16", label="cache token payload")
    offset_file = _regular_file(target / "offsets.u64", label="cache offset payload")
    if token_file.st_size != content_tokens * 2 or offset_file.st_size != (records + 1) * 8:
        raise DataProgressError(f"cache payload size differs from manifest: {target}")
    builder = manifest.get("builder")
    if not isinstance(builder, Mapping) or set(builder) != {
        "implementation",
        "implementation_sha256",
        "config",
        "config_sha256",
        "tokenization_contract_sha256",
    }:
        raise DataProgressError(f"cache builder descriptor is malformed: {target}")
    if builder.get("implementation") != "pretrain.raw_token_cache":
        raise DataProgressError(f"cache builder implementation is invalid: {target}")
    _sha256(builder.get("implementation_sha256"), label="cache builder SHA-256")
    _sha256(
        builder.get("tokenization_contract_sha256"),
        label="cache tokenization contract SHA-256",
    )
    expected_contract_sha = hashlib.sha256(
        json.dumps(
            TOKENIZATION_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if builder.get("tokenization_contract_sha256") != expected_contract_sha:
        raise DataProgressError(f"cache tokenization contract digest mismatch: {target}")
    config = builder.get("config")
    if not isinstance(config, Mapping) or set(config) != CACHE_CONFIG_FIELDS:
        raise DataProgressError(f"cache builder config is malformed: {target}")
    for name in CACHE_CONFIG_FIELDS:
        _positive_int(config.get(name), label=f"cache builder config {name}")
    if config.get("expected_vocab_size") != 49_152:
        raise DataProgressError(f"cache builder vocabulary config mismatch: {target}")
    if builder.get("config_sha256") != hashlib.sha256(
        _canonical_json_bytes(config)
    ).hexdigest():
        raise DataProgressError(f"cache builder config digest mismatch: {target}")
    directory_after = _safe_directory(target, label="completed cache directory")
    if (
        _file_identity(directory_before) != _file_identity(directory_after)
        or _file_identity(metadata)
        != _file_identity(_regular_file(manifest_path, label="cache manifest"))
        or _file_identity(sidecar_metadata)
        != _file_identity(
            _regular_file(target / "manifest.sha256", label="cache manifest sidecar")
        )
        or _file_identity(token_file)
        != _file_identity(_regular_file(target / "tokens.u16", label="cache token payload"))
        or _file_identity(offset_file)
        != _file_identity(_regular_file(target / "offsets.u64", label="cache offset payload"))
    ):
        raise DataProgressError(f"completed cache changed while inspected: {target}")
    return {
        "bucket": bucket,
        "index": index,
        "archive": archive["path"],
        "records": records,
        "clean_bytes": clean_bytes,
        "content_tokens": content_tokens,
        "manifest_sha256": digest,
        "tokenizer": tokenizer,
        "source": {
            name: dict(descriptor) for name, descriptor in source.items()
        },
        "documents": {
            "records": records,
            "clean_bytes": clean_bytes,
            "content_tokens": content_tokens,
        },
    }, metadata


def _cache_progress(
    cache_root: Path,
    authority: Mapping[str, Any],
    source_authority: Mapping[tuple[str, int], Mapping[str, Any]],
    authoritative_tokenizer: Mapping[str, Any],
    *,
    stale_seconds: float,
    now: float,
) -> dict[str, Any]:
    _safe_directory(cache_root, label="raw-token-cache root")
    inventory_before = _cache_inventory_snapshot(cache_root)
    archives_root = cache_root / "archives"
    completed: list[dict[str, Any]] = []
    activity_mtime = cache_root.lstat().st_mtime
    tokenizer_identity: dict[str, Any] | None = None
    if archives_root.exists() or archives_root.is_symlink():
        _safe_directory(archives_root, label="raw-token-cache archives root")
        for entry in _stable_directory_entries(
            archives_root, label="raw-token-cache archives root"
        ):
            if entry.name not in BUCKETS:
                raise DataProgressError(f"unexpected cache bucket entry: {entry}")
            _safe_directory(entry, label="raw-token-cache bucket")
        for bucket in BUCKETS:
            bucket_root = archives_root / bucket
            if not bucket_root.exists():
                continue
            for target in _stable_directory_entries(
                bucket_root, label="raw-token-cache bucket"
            ):
                part_match = PART_DIRECTORY.fullmatch(target.name)
                if part_match is not None:
                    key = (bucket, int(part_match.group(1)))
                    expected = source_authority.get(key)
                    if expected is None:
                        raise DataProgressError(
                            "completed cache has no authoritative generation part identity"
                        )
                    item, metadata = _cache_manifest(
                        target, bucket=bucket, index=key[1]
                    )
                    if (
                        not _json_exact_equal(item["source"], expected["source"])
                        or not _json_exact_equal(
                            item["documents"], expected["documents"]
                        )
                        or not _json_exact_equal(
                            item["tokenizer"], authoritative_tokenizer
                        )
                    ):
                        raise DataProgressError(
                            "completed cache differs from generation/report/tokenizer authority"
                        )
                    if tokenizer_identity is None:
                        tokenizer_identity = item["tokenizer"]
                    elif tokenizer_identity != item["tokenizer"]:
                        raise DataProgressError("completed caches mix tokenizer identities")
                    completed.append(item)
                    activity_mtime = max(activity_mtime, metadata.st_mtime)
                    continue
                if STAGE_DIRECTORY.fullmatch(target.name) is not None:
                    raise DataProgressError(
                        "in-flight cache directory prevents a coherent progress snapshot"
                    )
                raise DataProgressError(f"unexpected cache archive entry: {target}")

    keys = [(item["bucket"], item["index"]) for item in completed]
    archives = [item["archive"] for item in completed]
    if len(set(keys)) != len(keys) or len(set(archives)) != len(archives):
        raise DataProgressError("duplicate completed cache identity")
    per_bucket: dict[str, dict[str, Any]] = {}
    total_archives = 0
    total_documents = 0
    total_tokens = 0
    for bucket in BUCKETS:
        items = [item for item in completed if item["bucket"] == bucket]
        observed_archives = len(items)
        observed_documents = sum(item["records"] for item in items)
        observed_tokens = sum(item["content_tokens"] for item in items)
        expected = authority["per_bucket"][bucket]
        if (
            observed_archives > expected["archives"]
            or observed_documents > expected["documents"]
            or observed_tokens > expected["exact_tokens"]
        ):
            raise DataProgressError(f"cache progress exceeds collection authority for {bucket}")
        per_bucket[bucket] = {
            "archives": _progress(observed_archives, expected["archives"]),
            "documents": _progress(observed_documents, expected["documents"]),
            "content_tokens": _progress(observed_tokens, expected["exact_tokens"]),
        }
        total_archives += observed_archives
        total_documents += observed_documents
        total_tokens += observed_tokens
    complete = (
        total_archives == authority["archives"]
        and total_documents == authority["documents"]
        and total_tokens == authority["content_tokens"]
        and set(keys) == set(source_authority)
    )
    inventory_after = _cache_inventory_snapshot(cache_root)
    if inventory_after != inventory_before:
        raise DataProgressError("raw-token-cache inventory changed during global scan")
    age = now - activity_mtime
    if age < -5.0:
        raise DataProgressError("cache activity modification time is in the future")
    age = max(0.0, age)
    if not complete and age > stale_seconds:
        raise DataProgressError(
            f"raw-token-cache publication is stale: {age:.3f}s > {stale_seconds:.3f}s"
        )
    return {
        "root": str(cache_root.resolve(strict=True)),
        "complete": complete,
        "verification_scope": (
            "checkpoint-pinned-generation+tokenizer+canonical-cache-manifest+"
            "sidecar+payload-size (no full raw/token payload rehash)"
        ),
        "activity_mtime_utc": _utc(activity_mtime),
        "activity_age_seconds": round(age, 6),
        "in_flight_directories": 0,
        "counts": {
            "archives": _progress(total_archives, authority["archives"]),
            "documents": _progress(total_documents, authority["documents"]),
            "content_tokens": _progress(total_tokens, authority["content_tokens"]),
        },
        "per_bucket": per_bucket,
        "tokenizer": authoritative_tokenizer,
    }


def _probe_processes() -> dict[str, Any]:
    root = Path("/proc")
    if not root.is_dir():
        return {"available": False, "reason": "/proc is unavailable", "processes": []}
    processes: list[dict[str, Any]] = []
    readable = 0
    skipped = 0
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            with (entry / "cmdline").open("rb") as handle:
                raw = handle.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                skipped += 1
                continue
            cwd = os.readlink(entry / "cwd")
            executable = os.readlink(entry / "exe")
            stat_raw = (entry / "stat").read_text(encoding="ascii")
            close = stat_raw.rfind(")")
            stat_fields = stat_raw[close + 2 :].split() if close >= 0 else []
            if len(stat_fields) < 20:
                raise ValueError("short proc stat")
            start_ticks = int(stat_fields[19])
            if start_ticks < 1:
                raise ValueError("invalid proc start time")
        except (OSError, UnicodeError, ValueError):
            skipped += 1
            continue
        readable += 1
        arguments = [
            item.decode("utf-8", errors="surrogateescape")
            for item in raw.rstrip(b"\0").split(b"\0")
            if item
        ]
        if arguments:
            processes.append(
                {
                    "pid": int(entry.name),
                    "argv": arguments,
                    "cwd": cwd,
                    "executable": executable,
                    "start_ticks": start_ticks,
                }
            )
    if readable == 0:
        return {
            "available": False,
            "reason": "/proc command lines are unreadable",
            "processes": [],
            "skipped": skipped,
        }
    return {
        "available": True,
        "processes": sorted(processes, key=lambda item: item["pid"]),
        "skipped": skipped,
    }


def _probe_lock_holders(paths: Sequence[Path]) -> dict[str, Any]:
    proc_locks = Path("/proc/locks")
    if not proc_locks.is_file():
        return {"available": False, "reason": "/proc/locks is unavailable"}
    requested: dict[tuple[int, int, int], str] = {}
    metadata: dict[str, os.stat_result] = {}
    for path in paths:
        item = _regular_file(path, label="runtime ownership lock")
        resolved = str(path.resolve(strict=True))
        key = (os.major(item.st_dev), os.minor(item.st_dev), item.st_ino)
        requested[key] = resolved
        metadata[resolved] = item
    holders = {resolved: [] for resolved in metadata}
    try:
        raw = proc_locks.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        return {"available": False, "reason": f"cannot read /proc/locks: {exc}"}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 6:
            return {"available": False, "reason": "malformed /proc/locks"}
        try:
            device_major, device_minor, inode = fields[5].split(":", 2)
            key = (int(device_major, 16), int(device_minor, 16), int(inode, 10))
        except (ValueError, IndexError):
            return {"available": False, "reason": "malformed /proc/locks identity"}
        resolved = requested.get(key)
        if resolved is None:
            continue
        try:
            pid = int(fields[4])
        except ValueError:
            return {"available": False, "reason": "malformed /proc/locks PID"}
        if fields[1] != "FLOCK" or fields[3] != "WRITE" or pid < 1:
            return {"available": False, "reason": "unsupported lock ownership evidence"}
        holders[resolved].append(pid)
    for path in paths:
        resolved = str(path.resolve(strict=True))
        if _file_identity(metadata[resolved]) != _file_identity(
            _regular_file(path, label="runtime ownership lock")
        ):
            return {"available": False, "reason": "runtime lock changed while probed"}
    return {
        "available": True,
        "holders": {key: sorted(value) for key, value in holders.items()},
    }


def _curation_lease(
    output_root: Path,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    path = output_root / CURATION_LEASE_FILE
    raw, metadata = _stable_file_bytes(
        path, label="curation ownership lease", maximum_bytes=64 * 1024
    )
    owner = _json_object(raw, label="curation ownership lease")
    expected_fields = {
        "lease_version",
        "hostname",
        "pid",
        "started_unix_ns",
        "output",
        "owner_token",
    }
    if set(owner) != expected_fields:
        raise DataProgressError("curation ownership lease schema mismatch")
    _require_exact_int(owner.get("lease_version"), 1, label="curation lease version")
    _positive_int(owner.get("pid"), label="curation lease PID")
    _positive_int(owner.get("started_unix_ns"), label="curation lease start time")
    if (
        not isinstance(owner.get("hostname"), str)
        or not owner["hostname"]
        or owner.get("output") != str(output_root.resolve(strict=True))
    ):
        raise DataProgressError("curation ownership lease identity mismatch")
    token = _sha256(owner.get("owner_token"), label="curation lease owner token")
    authority = {key: value for key, value in owner.items() if key != "owner_token"}
    if hashlib.sha256(_canonical_json_bytes(authority)).hexdigest() != token:
        raise DataProgressError("curation ownership lease token mismatch")
    return owner, raw, metadata


def _one_process_option(arguments: Sequence[str], option: str) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            if index + 1 >= len(arguments):
                return None
            values.append(arguments[index + 1])
            index += 2
            continue
        prefix = option + "="
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
        index += 1
    return values[0] if len(values) == 1 and values[0] else None


def _resolved_process_path(value: str, cwd: Path) -> Path | None:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _matches_scoped_process(
    item: Mapping[str, Any],
    *,
    script: Path,
    root: Path,
    output_option: str,
    output: Path,
    required_subcommand: str | None = None,
) -> bool:
    argv = item.get("argv")
    cwd_value = item.get("cwd")
    executable_value = item.get("executable")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) for value in argv)
        or not isinstance(cwd_value, str)
        or not isinstance(executable_value, str)
    ):
        return False
    _positive_int(item.get("start_ticks"), label="process start ticks")
    cwd = Path(cwd_value)
    expected_script = script.resolve(strict=True)
    try:
        executable = Path(executable_value).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    executable_name = executable.name.casefold()
    if not (
        executable_name.startswith("python") or executable_name.startswith("pypy")
    ):
        return False
    if len(argv) <= 1 or _resolved_process_path(argv[1], cwd) != expected_script:
        return False
    script_index = 1
    trailing = argv[script_index + 1 :]
    if required_subcommand is not None and (
        not trailing or trailing[0] != required_subcommand
    ):
        return False
    root_value = _one_process_option(trailing, "--root")
    output_value = _one_process_option(trailing, output_option)
    return (
        root_value is not None
        and output_value is not None
        and _resolved_process_path(root_value, cwd) == root.resolve(strict=True)
        and _resolved_process_path(output_value, cwd) == output.resolve(strict=True)
    )


def _probe_tmux() -> dict[str, Any]:
    executable = shutil.which("tmux")
    if executable is None:
        return {"available": False, "reason": "tmux is not installed", "sessions": []}
    result = subprocess.run(
        [
            executable,
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{pane_pid}\t#{pane_dead}\t#{pane_current_command}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        if "no server running" in detail or "no sessions" in detail:
            return {"available": True, "sessions": []}
        return {"available": False, "reason": detail or "tmux query failed", "sessions": []}
    sessions: dict[str, list[dict[str, Any]]] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            return {"available": False, "reason": "malformed tmux output", "sessions": []}
        name, raw_pid, raw_dead, command = fields
        try:
            pid = int(raw_pid)
        except ValueError:
            return {"available": False, "reason": "malformed tmux PID", "sessions": []}
        sessions.setdefault(name, []).append(
            {"pid": pid, "dead": raw_dead == "1", "command": command}
        )
    return {
        "available": True,
        "sessions": [
            {"name": name, "panes": panes}
            for name, panes in sorted(sessions.items())
        ],
    }


def _read_proc_mapping(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError(f"malformed counter line in {path}: {line!r}")
        result[fields[0]] = int(fields[1])
    return result


def _probe_memory() -> dict[str, Any]:
    cgroup = Path("/sys/fs/cgroup")
    try:
        current_path = cgroup / "memory.current"
        maximum_path = cgroup / "memory.max"
        events_path = cgroup / "memory.events"
        stat_path = cgroup / "memory.stat"
        if all(path.is_file() for path in (current_path, maximum_path, events_path, stat_path)):
            current = int(current_path.read_text(encoding="ascii").strip())
            raw_maximum = maximum_path.read_text(encoding="ascii").strip()
            maximum = None if raw_maximum == "max" else int(raw_maximum)
            events = _read_proc_mapping(events_path)
            memory_stat = _read_proc_mapping(stat_path)
            inactive_file = memory_stat.get("inactive_file", 0)
            return {
                "available": True,
                "scope": "cgroup-v2",
                "current_bytes": current,
                "limit_bytes": maximum,
                "usage_percent": (
                    None if maximum is None else round(100.0 * current / maximum, 6)
                ),
                "anonymous_bytes": memory_stat.get("anon"),
                "file_cache_bytes": memory_stat.get("file"),
                "inactive_file_bytes": inactive_file,
                "working_set_estimate_bytes": max(0, current - inactive_file),
                "events": events,
            }
    except (OSError, UnicodeError, ValueError) as exc:
        return {"available": False, "reason": f"cannot read cgroup memory: {exc}"}
    return {"available": False, "reason": "cgroup-v2 memory accounting is unavailable"}


def _disk_health(paths: Sequence[Path], *, minimum_free_bytes: int) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    seen_devices: set[int] = set()
    for candidate in paths:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
        if metadata.st_dev in seen_devices:
            continue
        seen_devices.add(metadata.st_dev)
        filesystem = os.statvfs(resolved)
        total = filesystem.f_blocks * filesystem.f_frsize
        free = filesystem.f_bavail * filesystem.f_frsize
        if total < 1 or free < minimum_free_bytes:
            raise DataProgressError(
                f"filesystem safety reserve failed at {resolved}: free={free}, "
                f"required={minimum_free_bytes}"
            )
        volumes.append(
            {
                "path": str(resolved),
                "device": metadata.st_dev,
                "total_bytes": total,
                "free_bytes": free,
                "used_percent": round(100.0 * (total - free) / total, 6),
                "available_inodes": filesystem.f_favail,
                "minimum_free_bytes": minimum_free_bytes,
                "reserve_pass": True,
            }
        )
    if not volumes:
        raise DataProgressError("no filesystem health evidence was collected")
    return volumes


def _runtime_health(
    *,
    curation_terminal: bool,
    cache_complete: bool,
    generation_root: Path,
    curation_output_root: Path,
    cache_root: Path,
    expected_tmux_sessions: Sequence[str],
    process_probe: Callable[[], dict[str, Any]],
    lock_probe: Callable[[Sequence[Path]], dict[str, Any]],
    tmux_probe: Callable[[], dict[str, Any]],
    memory_probe: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    process = process_probe()
    if not isinstance(process, dict) or not isinstance(process.get("available"), bool):
        raise DataProgressError("process probe returned malformed evidence")
    matches: dict[str, list[dict[str, Any]]] = {"curation": [], "raw_token_cache": []}
    active_locks: dict[str, Path] = {}
    lease_snapshot: tuple[dict[str, Any], bytes, os.stat_result] | None = None
    if not curation_terminal:
        lease_snapshot = _curation_lease(curation_output_root)
        active_locks["curation"] = curation_output_root / CURATION_LOCK_FILE
    if not cache_complete:
        active_locks["raw_token_cache"] = cache_root / CACHE_LOCK_FILE
    for path in active_locks.values():
        _regular_file(path, label="runtime ownership lock")

    lock_evidence: dict[str, Any] = {"available": True, "holders": {}}
    if active_locks:
        lock_evidence = lock_probe(tuple(active_locks.values()))
        if not isinstance(lock_evidence, dict) or not isinstance(
            lock_evidence.get("available"), bool
        ):
            raise DataProgressError("lock probe returned malformed evidence")
    ownership_available = bool(process["available"] and lock_evidence["available"])
    process_by_pid: dict[int, Mapping[str, Any]] = {}
    if process["available"]:
        raw_processes = process.get("processes")
        if not isinstance(raw_processes, list):
            raise DataProgressError("process probe lacks a process list")
        for item in raw_processes:
            if not isinstance(item, Mapping):
                raise DataProgressError("process probe contains a malformed process")
            pid = _positive_int(item.get("pid"), label="process PID")
            if pid in process_by_pid:
                raise DataProgressError("process probe contains a duplicate PID")
            process_by_pid[pid] = item
    else:
        warnings.append("process_health_unavailable")
    if active_locks and not lock_evidence["available"]:
        warnings.append("lock_ownership_unavailable")
    if active_locks and not ownership_available:
        warnings.append("liveness_authority_unavailable")

    authoritative_processes: dict[int, Mapping[str, Any]] = {}
    if ownership_available:
        holders = lock_evidence.get("holders")
        expected_lock_paths = {
            str(path.resolve(strict=True)) for path in active_locks.values()
        }
        if not isinstance(holders, Mapping) or set(holders) != expected_lock_paths:
            raise DataProgressError("lock probe lacks the exact ownership inventory")

        def owned_pid(role: str) -> int:
            path = str(active_locks[role].resolve(strict=True))
            raw_holders = holders.get(path)
            if (
                not isinstance(raw_holders, list)
                or len(raw_holders) != 1
                or isinstance(raw_holders[0], bool)
                or not isinstance(raw_holders[0], int)
                or raw_holders[0] < 1
            ):
                raise DataProgressError(f"{role} ownership lock has no unique live PID")
            return raw_holders[0]

        if not curation_terminal:
            assert lease_snapshot is not None
            owner = lease_snapshot[0]
            if owner["hostname"] != platform.node():
                warnings.append("curation_owner_is_remote")
                warnings.append("liveness_authority_unavailable")
            else:
                pid = owned_pid("curation")
                if pid != owner["pid"]:
                    raise DataProgressError(
                        "curation lease PID differs from advisory-lock owner"
                    )
                item = process_by_pid.get(pid)
                if item is None or not _matches_scoped_process(
                    item,
                    script=CURATION_SCRIPT,
                    root=generation_root,
                    output_option="--output",
                    output=curation_output_root,
                ):
                    raise DataProgressError(
                        "curation lease/lock owner is not the exact configured invocation"
                    )
                authoritative_processes[pid] = item
                matches["curation"].append(
                    {
                        "pid": pid,
                        "script": CURATION_SCRIPT.name,
                        "ownership": "lease+flock+proc",
                    }
                )
        if not cache_complete:
            pid = owned_pid("raw_token_cache")
            item = process_by_pid.get(pid)
            if item is None or not _matches_scoped_process(
                item,
                script=CACHE_SCRIPT,
                root=generation_root,
                output_option="--output-root",
                output=cache_root,
                required_subcommand="build",
            ):
                raise DataProgressError(
                    "cache lock owner is not the exact configured invocation"
                )
            authoritative_processes[pid] = item
            matches["raw_token_cache"].append(
                {
                    "pid": pid,
                    "script": CACHE_SCRIPT.name,
                    "ownership": "flock+proc",
                }
            )

        repeated_locks = lock_probe(tuple(active_locks.values()))
        if not _json_exact_equal(repeated_locks, lock_evidence):
            raise DataProgressError("runtime lock ownership changed while inspected")
        repeated_process = process_probe()
        if not isinstance(repeated_process, Mapping) or repeated_process.get(
            "available"
        ) is not True:
            raise DataProgressError("process ownership disappeared while inspected")
        repeated_items = repeated_process.get("processes")
        if not isinstance(repeated_items, list):
            raise DataProgressError("repeated process evidence is malformed")
        repeated_by_pid = {
            item.get("pid"): item
            for item in repeated_items
            if isinstance(item, Mapping)
            and type(item.get("pid")) is int
            and item["pid"] > 0
        }
        for pid, item in authoritative_processes.items():
            if not _json_exact_equal(repeated_by_pid.get(pid), item):
                raise DataProgressError("process ownership changed while inspected")
        if lease_snapshot is not None:
            repeated_lease = _curation_lease(curation_output_root)
            if (
                not _json_exact_equal(repeated_lease[0], lease_snapshot[0])
                or repeated_lease[1] != lease_snapshot[1]
                or _file_identity(repeated_lease[2])
                != _file_identity(lease_snapshot[2])
            ):
                raise DataProgressError("curation ownership lease changed while inspected")

    tmux = tmux_probe()
    if not isinstance(tmux, dict) or not isinstance(tmux.get("available"), bool):
        raise DataProgressError("tmux probe returned malformed evidence")
    if tmux["available"]:
        sessions = tmux.get("sessions")
        if not isinstance(sessions, list):
            raise DataProgressError("tmux probe lacks a session list")
        for session in sessions:
            if not isinstance(session, Mapping) or not isinstance(
                session.get("name"), str
            ):
                raise DataProgressError("tmux probe contains a malformed session")
            panes = session.get("panes")
            if not isinstance(panes, list) or any(
                not isinstance(pane, Mapping)
                or not isinstance(pane.get("dead"), bool)
                for pane in panes
            ):
                raise DataProgressError("tmux probe contains malformed panes")
        by_name = {
            item.get("name"): item
            for item in sessions
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if len(by_name) != len(sessions):
            raise DataProgressError("tmux probe contains malformed or duplicate sessions")
        for name in expected_tmux_sessions:
            session = by_name.get(name)
            if session is None:
                raise DataProgressError(f"required tmux session is missing: {name}")
            panes = session.get("panes")
            if (
                not isinstance(panes, list)
                or not panes
                or any(
                    not isinstance(pane, Mapping)
                    or not isinstance(pane.get("dead"), bool)
                    for pane in panes
                )
                or all(pane.get("dead") is True for pane in panes)
            ):
                raise DataProgressError(f"required tmux session has no live pane: {name}")
    elif expected_tmux_sessions:
        warnings.append("tmux_health_unavailable")

    memory = memory_probe()
    if not isinstance(memory, dict) or not isinstance(memory.get("available"), bool):
        raise DataProgressError("memory probe returned malformed evidence")
    if memory["available"]:
        events = memory.get("events")
        if not isinstance(events, Mapping):
            raise DataProgressError("memory probe lacks OOM event evidence")
        oom = _plain_int(events.get("oom", 0), label="memory.events oom")
        oom_kill = _plain_int(events.get("oom_kill", 0), label="memory.events oom_kill")
        oom_group_kill = _plain_int(
            events.get("oom_group_kill", 0), label="memory.events oom_group_kill"
        )
        if oom or oom_kill or oom_group_kill:
            raise DataProgressError(
                "cgroup OOM evidence is nonzero: "
                f"oom={oom}, oom_kill={oom_kill}, oom_group_kill={oom_group_kill}"
            )
        limit = memory.get("limit_bytes")
        working_set = memory.get("working_set_estimate_bytes")
        if limit is not None:
            limit = _positive_int(limit, label="memory limit_bytes")
        if working_set is not None:
            working_set = _plain_int(
                working_set, label="memory working_set_estimate_bytes"
            )
        if limit is not None and working_set is not None and working_set / limit >= 0.9:
            warnings.append("memory_working_set_above_90_percent")
    else:
        warnings.append("memory_health_unavailable")
    process_public = {
        "available": process["available"],
        "skipped": _plain_int(process.get("skipped", 0), label="processes skipped"),
        "matches": matches,
    }
    tmux_public: dict[str, Any] = {"available": tmux["available"], "sessions": []}
    if tmux["available"]:
        for session in tmux.get("sessions", []):
            panes_public = []
            for pane in session["panes"]:
                pid = pane.get("pid")
                if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
                    raise DataProgressError("tmux probe contains an invalid pane PID")
                panes_public.append({"pid": pid, "dead": pane["dead"]})
            tmux_public["sessions"].append(
                {"name": session["name"], "panes": panes_public}
            )
    memory_public = {
        key: memory[key]
        for key in (
            "available",
            "scope",
            "current_bytes",
            "limit_bytes",
            "usage_percent",
            "anonymous_bytes",
            "file_cache_bytes",
            "inactive_file_bytes",
            "working_set_estimate_bytes",
        )
        if key in memory
    }
    if memory["available"]:
        memory_public["events"] = {
            "oom": oom,
            "oom_kill": oom_kill,
            "oom_group_kill": oom_group_kill,
        }
    return {
        "processes": process_public,
        "tmux": tmux_public,
        "memory": memory_public,
    }, warnings


def build_report(
    *,
    generation_root: Path,
    curation_checkpoint: Path,
    curation_output_root: Path,
    cache_root: Path,
    preprocess_root: Path | None = None,
    tokenizer_root: Path | None = None,
    curation_journal: Path | None = None,
    checkpoint_stale_seconds: float = 3600.0,
    cache_stale_seconds: float = 7200.0,
    minimum_free_bytes: int = 10 * 1024**3,
    disk_paths: Sequence[Path] = (),
    expected_tmux_sessions: Sequence[str] = (),
    now: float | None = None,
    process_probe: Callable[[], dict[str, Any]] | None = None,
    lock_probe: Callable[[Sequence[Path]], dict[str, Any]] | None = None,
    tmux_probe: Callable[[], dict[str, Any]] | None = None,
    memory_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one fail-closed, JSON-serializable, read-only health projection."""

    if checkpoint_stale_seconds <= 0 or cache_stale_seconds <= 0:
        raise DataProgressError("freshness thresholds must be positive")
    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes < 1
    ):
        raise DataProgressError("minimum_free_bytes must be a positive integer")
    if len(set(expected_tmux_sessions)) != len(expected_tmux_sessions):
        raise DataProgressError("expected tmux sessions contain duplicates")
    observed_now = time.time() if now is None else now
    journal = curation_journal or curation_checkpoint.with_name("journal.jsonl")
    preprocess = preprocess_root or generation_root / "staging" / "preprocess"
    tokenizer = tokenizer_root or generation_root / "tokenizer" / "starcoder2"
    _safe_directory(curation_output_root, label="curation output root")
    resolved_output = curation_output_root.resolve(strict=True)
    try:
        curation_checkpoint.resolve(strict=True).relative_to(resolved_output)
        journal.resolve(strict=True).relative_to(resolved_output)
    except (OSError, ValueError) as exc:
        raise DataProgressError(
            "curation checkpoint/journal are outside the configured output root"
        ) from exc
    authority, curation, source_authority, generation = _curation_progress(
        curation_checkpoint,
        journal,
        generation_root,
        preprocess,
        tokenizer,
        stale_seconds=checkpoint_stale_seconds,
        now=observed_now,
    )
    cache = _cache_progress(
        cache_root,
        authority,
        source_authority,
        generation["tokenizer"],
        stale_seconds=cache_stale_seconds,
        now=observed_now,
    )
    volume_paths = [generation_root, curation_output_root, cache_root, *disk_paths]
    disks = _disk_health(volume_paths, minimum_free_bytes=minimum_free_bytes)
    runtime, warnings = _runtime_health(
        curation_terminal=curation["terminal"],
        cache_complete=cache["complete"],
        generation_root=generation_root,
        curation_output_root=curation_output_root,
        cache_root=cache_root,
        expected_tmux_sessions=expected_tmux_sessions,
        process_probe=_probe_processes if process_probe is None else process_probe,
        lock_probe=_probe_lock_holders if lock_probe is None else lock_probe,
        tmux_probe=_probe_tmux if tmux_probe is None else tmux_probe,
        memory_probe=_probe_memory if memory_probe is None else memory_probe,
    )
    complete = bool(curation["terminal"] and cache["complete"])
    return {
        "format": "pretraining-data-progress-health",
        "format_version": 1,
        "recorded_utc": _utc(observed_now),
        "status": "complete" if complete else ("warning" if warnings else "healthy"),
        "read_only": True,
        "warnings": warnings,
        "generation": generation,
        "authority": authority,
        "curation": curation,
        "raw_token_cache": cache,
        "runtime": {**runtime, "filesystems": disks},
    }


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_int_argument(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--curation-checkpoint", type=Path, required=True)
    parser.add_argument("--curation-output-root", type=Path, required=True)
    parser.add_argument(
        "--curation-journal",
        type=Path,
        help="default: journal.jsonl beside the curation checkpoint",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        help="default: GENERATION_ROOT/staging/preprocess",
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        help="default: GENERATION_ROOT/tokenizer/starcoder2",
    )
    parser.add_argument("--checkpoint-stale-seconds", type=_positive_float, default=3600.0)
    parser.add_argument("--cache-stale-seconds", type=_positive_float, default=7200.0)
    parser.add_argument(
        "--minimum-free-bytes",
        type=_positive_int_argument,
        default=10 * 1024**3,
    )
    parser.add_argument(
        "--disk-path",
        action="append",
        default=[],
        type=Path,
        help="additional filesystem whose free space is required (repeatable)",
    )
    parser.add_argument(
        "--expected-tmux-session",
        action="append",
        default=[],
        help="session required when tmux inspection is available (repeatable)",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(
            generation_root=args.generation_root,
            curation_checkpoint=args.curation_checkpoint,
            curation_output_root=args.curation_output_root,
            curation_journal=args.curation_journal,
            cache_root=args.cache_root,
            preprocess_root=args.preprocess_root,
            tokenizer_root=args.tokenizer_root,
            checkpoint_stale_seconds=args.checkpoint_stale_seconds,
            cache_stale_seconds=args.cache_stale_seconds,
            minimum_free_bytes=args.minimum_free_bytes,
            disk_paths=args.disk_path,
            expected_tmux_sessions=args.expected_tmux_session,
        )
    except (DataProgressError, OSError, ValueError, subprocess.SubprocessError) as exc:
        failure = {
            "format": "pretraining-data-progress-health",
            "format_version": 1,
            "recorded_utc": _now_utc(),
            "status": "failed",
            "read_only": True,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    if args.compact:
        print(json.dumps(report, sort_keys=True), flush=True)
    else:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
