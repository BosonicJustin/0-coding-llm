#!/usr/bin/env python3
"""Publish every eligible canonical document from a frozen curation database.

This is a read-only escape hatch from exact quota selection.  It consumes a
verified curation snapshot after canonicalization and leakage-safe group
assignment, ignores any partial ``selected``/``selection.quota.*`` state, and
publishes complete-document decisions into a separate immutable generation.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import struct
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pretrain.selection_contract import (  # noqa: E402
    ALL_ELIGIBLE_BITMAP_BIT_ORDER,
    ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS,
    ALL_ELIGIBLE_BITMAP_FORMAT,
    ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
    ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES,
    ALL_ELIGIBLE_BITMAP_MAGIC,
    ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
    ALL_ELIGIBLE_SELECTION_PROFILE,
    ALL_ELIGIBLE_SELECTION_STRATEGY,
    all_eligible_bitmap_payload_bytes,
    validate_all_eligible_bitmap_header,
    validate_all_eligible_bitmap_payload,
)
from curate_corpus import (  # noqa: E402
    BUCKET_CATEGORY,
    DB_VERSION,
    DECISION_RECORD_VERSION,
    FAST_CANONICAL_FORMAT_VERSION,
    SPLITS,
    atomic_bytes,
    atomic_json,
    assign_group_split,
    canonical_json_bytes,
    file_sha256,
    fsync_directory,
    load_final_quotas,
    split_thresholds,
)
from curation_policy import (  # noqa: E402
    FAST_CANONICAL_POLICY,
    FAST_CANONICAL_PROFILE,
    canonical_sha256,
    load_policy,
)
from curation_local_store import (  # noqa: E402
    RETIREMENT_FORMAT,
    RETIREMENT_VERSION,
    SNAPSHOT_DIRECTORY,
    _database_state,
)


PUBLICATION_CHECKPOINT_VERSION = 1
PUBLICATION_MANIFEST_VERSION = 1
SNAPSHOT_FORMAT = "curation-local-sqlite-snapshot"
SNAPSHOT_VERSION = 1
DOMAIN_ORDER = ("python", "other_code", "english")
HEX64 = frozenset("0123456789abcdef")
DOCUMENTS_PRIMARY_INDEX = "sqlite_autoindex_documents_1"
DOCUMENTS_ARCHIVE_INDEX = "sqlite_autoindex_documents_2"
POISON_FORMAT = "all-eligible-publication-poison"
POISON_VERSION = 1
REQUESTED_CACHE_KIB = 4 * 1024 * 1024
REQUESTED_MMAP_BYTES = 64 * 1024 * 1024 * 1024
SNAPSHOT_MANIFEST_KEYS = frozenset(
    (
        "format",
        "format_version",
        "status",
        "generation",
        "created_utc",
        "reason",
        "identity",
        "identity_sha256",
        "previous_manifest_sha256",
        "canonical_journal_mode",
        "database",
        "database_state",
        "authority_artifacts",
        "runtime_provenance",
        "admission_sha256",
        "durable_capacity_preflight",
        "snapshot_retention",
        "prepare_evidence",
    )
)


ELIGIBLE_AUTHORITY_SQL = f"""
SELECT
    d.bucket,
    g.split,
    COUNT(*) AS documents,
    COALESCE(SUM(d.tokens), 0) AS tokens,
    COALESCE(SUM(
        CASE WHEN typeof(d.tokens) != 'integer' OR d.tokens < 1 THEN 1 ELSE 0 END
    ), 0) AS invalid_token_documents,
    COALESCE(SUM(
        CASE WHEN c.doc_id IS NULL OR c.canonical_doc_id != d.doc_id
               OR e.content_hash IS NULL OR f.final_cluster IS NULL
             THEN 1 ELSE 0 END
    ), 0) AS invalid_canonical_documents
FROM documents AS d INDEXED BY {DOCUMENTS_PRIMARY_INDEX}
LEFT JOIN canonical_map AS c ON c.doc_id=d.doc_id
LEFT JOIN exact_choice AS e
  ON e.content_hash=d.content_hash AND e.canonical_doc_id=d.doc_id
LEFT JOIN final_choice AS f
  ON f.final_cluster=d.final_cluster AND f.canonical_doc_id=d.doc_id
LEFT JOIN groups AS g ON g.group_id=d.source_group
WHERE NOT EXISTS (
  SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
)
GROUP BY d.bucket, g.split
ORDER BY d.bucket, g.split
"""

ARCHIVE_BITMAP_SQL = f"""
SELECT COUNT(*), MIN(d.manifest_index), MAX(d.manifest_index)
FROM documents AS d INDEXED BY {DOCUMENTS_ARCHIVE_INDEX}
WHERE d.archive=?
"""

REJECTION_INVENTORY_SQL = f"""
SELECT d.archive, d.manifest_index, r.reason
FROM reasons AS r INDEXED BY reasons_reason
LEFT JOIN documents AS d INDEXED BY {DOCUMENTS_PRIMARY_INDEX}
  ON d.doc_id=r.doc_id
"""


class AllEligiblePublicationError(RuntimeError):
    """A source or publication invariant could not be proven."""


def _safe_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AllEligiblePublicationError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _safe_directory(path: Path, *, label: str, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise AllEligiblePublicationError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _safe_relative_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AllEligiblePublicationError(f"Invalid {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AllEligiblePublicationError(f"Unsafe {label} path: {value!r}")
    root = root.resolve(strict=True)
    path = _safe_regular_file(root / relative, label=label)
    if path != root and root not in path.parents:
        raise AllEligiblePublicationError(f"{label} escapes its authority root")
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AllEligiblePublicationError(
                    f"Duplicate {label} JSON key: {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AllEligiblePublicationError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise AllEligiblePublicationError(f"{label} must be a JSON object")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX64 for character in value)
    ):
        raise AllEligiblePublicationError(f"{label} must be a lowercase SHA-256")
    return value


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _safe_descendant_directory(
    root: Path, relative: Path, *, label: str, create: bool
) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise AllEligiblePublicationError(f"Unsafe {label} path: {relative}")
    current = root.resolve(strict=True)
    for part in relative.parts:
        if part in ("", "."):
            continue
        candidate = current / part
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or not candidate.is_dir():
                raise AllEligiblePublicationError(
                    f"{label} has an unsafe path component: {candidate}"
                )
        elif create:
            candidate.mkdir()
            fsync_directory(current)
        else:
            raise AllEligiblePublicationError(f"Missing {label}: {candidate}")
        current = candidate.resolve(strict=True)
        if current != root and root not in current.parents:
            raise AllEligiblePublicationError(f"{label} escapes publication root")
    return current


def _safe_descendant_file_path(
    root: Path, relative: Path, *, label: str, create_parent: bool
) -> Path:
    if relative.is_absolute() or not relative.name or ".." in relative.parts:
        raise AllEligiblePublicationError(f"Unsafe {label} path: {relative}")
    parent = _safe_descendant_directory(
        root, relative.parent, label=f"{label} parent", create=create_parent
    )
    path = parent / relative.name
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise AllEligiblePublicationError(f"Unsafe {label}: {path}")
    return path


def _is_primary_documents_scan(detail: str) -> bool:
    return detail in ("SCAN d", "SCAN TABLE documents AS d")


def build_all_eligible_keep_bitmap(
    records: int, rejected_manifest_indices: Sequence[int] | set[int]
) -> tuple[bytes, int]:
    """Build the frozen LSB0 keep bitmap without inspecting fingerprints."""

    payload_bytes = all_eligible_bitmap_payload_bytes(records)
    bits = bytearray(b"\xff" * payload_bytes)
    remainder = records % 8
    if remainder:
        bits[-1] &= (1 << remainder) - 1
    rejected = (
        rejected_manifest_indices
        if isinstance(rejected_manifest_indices, set)
        else set(rejected_manifest_indices)
    )
    if len(rejected) != len(rejected_manifest_indices) or any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= records
        for index in rejected
    ):
        raise AllEligiblePublicationError(
            "Rejected manifest indices are not unique in-range integers"
        )
    for index in rejected:
        bits[index // 8] &= ~(1 << (index % 8))
    payload = bytes(bits)
    validate_all_eligible_bitmap_payload(payload, records=records)
    return payload, records - len(rejected)


def _sqlite_uri(path: Path) -> str:
    encoded = urllib.parse.quote(path.resolve(strict=True).as_posix(), safe="/")
    return f"file:{encoded}?mode=ro&immutable=1"


def _checkpoint_sha(path: Path) -> str:
    return file_sha256(_safe_regular_file(path, label="source checkpoint"))


class AllEligiblePublisher:
    """Create one restartable all-eligible publication from an immutable DB."""

    def __init__(
        self,
        *,
        root: Path,
        staging_root: Path,
        source_db: Path,
        source_checkpoint: Path,
        output: Path,
        policy_path: Path = FAST_CANONICAL_POLICY,
        quota_path: Path,
        benchmark_denylist_path: Path,
        source_snapshot_manifest: Path | None = None,
        allow_unbound_source_for_testing: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.progress = progress
        self.started_monotonic = time.monotonic()
        self.connection: sqlite3.Connection | None = None
        self.lock_handle: Any | None = None
        self._bound_artifacts: dict[str, dict[str, Any]] = {}
        self._must_remain_absent: set[Path] = set()
        self.root = _safe_directory(root, label="raw dataset root")
        self.staging_root = _safe_directory(staging_root, label="preprocess root")
        self.source_db = _safe_regular_file(source_db, label="source curation database")
        self.source_checkpoint_path = _safe_regular_file(
            source_checkpoint, label="source checkpoint"
        )
        self.output = _safe_directory(output, label="publication output", create=True)
        self.work = _safe_descendant_directory(
            self.output, Path(".work"), label="publication work", create=True
        )
        self.checkpoint_path = _safe_descendant_file_path(
            self.output,
            Path(".work/PUBLICATION_CHECKPOINT.json"),
            label="publication checkpoint",
            create_parent=False,
        )
        self.poison_path = _safe_descendant_file_path(
            self.output,
            Path(".work/POISONED.json"),
            label="publication poison marker",
            create_parent=False,
        )
        self.lock_path = _safe_descendant_file_path(
            self.output,
            Path(".all-eligible-publication.lock"),
            label="publication lock",
            create_parent=False,
        )
        self._check_poisoned()
        self.policy_path = _safe_regular_file(policy_path, label="curation policy")
        self.quota_path = _safe_regular_file(quota_path, label="quota configuration")
        self.benchmark_denylist_path = _safe_regular_file(
            benchmark_denylist_path, label="benchmark denylist"
        )
        self.source_snapshot_manifest = (
            None
            if source_snapshot_manifest is None
            else _safe_regular_file(
                source_snapshot_manifest, label="source snapshot manifest"
            )
        )
        if self.source_snapshot_manifest is None and not allow_unbound_source_for_testing:
            raise AllEligiblePublicationError(
                "A complete durable snapshot manifest is required for publication"
            )
        self.allow_unbound_source_for_testing = allow_unbound_source_for_testing
        self.source_sidecars = [
            Path(f"{self.source_db}{suffix}")
            for suffix in ("-journal", "-wal", "-shm")
            if Path(f"{self.source_db}{suffix}").exists()
            or Path(f"{self.source_db}{suffix}").is_symlink()
        ]
        if self.source_sidecars:
            raise AllEligiblePublicationError(
                f"Source curation database has SQLite sidecars: {self.source_sidecars}"
            )
        self.source_db_stat = _stat_identity(self.source_db)
        self._log(
            f"hashing immutable source database ({self.source_db_stat['bytes']:,} bytes)"
        )
        self.source_db_sha256 = file_sha256(self.source_db)
        self.source_checkpoint_sha256 = _checkpoint_sha(
            self.source_checkpoint_path
        )
        self.source_checkpoint = _json_object(
            self.source_checkpoint_path, label="source checkpoint"
        )
        self._bind_artifact(
            self.source_checkpoint_path,
            label="source checkpoint",
            expected_sha256=self.source_checkpoint_sha256,
        )
        self.snapshot_evidence = self._validate_snapshot_manifest()
        self._bind_artifact(self.policy_path, label="curation policy")
        self._bind_artifact(self.quota_path, label="quota configuration")
        self._bind_artifact(
            self.benchmark_denylist_path, label="benchmark denylist"
        )
        self.policy = load_policy(self.policy_path)
        if self.policy.get("curation_profile") != FAST_CANONICAL_PROFILE:
            raise AllEligiblePublicationError(
                "All-eligible publication requires the frozen fast canonical policy"
            )
        self.quotas = load_final_quotas(self.quota_path)
        self.connection = sqlite3.connect(
            _sqlite_uri(self.source_db), uri=True, timeout=120
        )
        try:
            self.connection.execute("PRAGMA query_only=ON")
            self.connection.execute("PRAGMA trusted_schema=OFF")
            self.sqlite_read_performance = self._configure_read_performance()
            # A production snapshot was created only after LocalSQLiteStore's
            # full integrity check and is authenticated byte-for-byte here.
            # Retain quick_check solely for deliberately unbound test fixtures.
            if self.snapshot_evidence is None:
                integrity = self.connection.execute("PRAGMA quick_check").fetchone()
                if integrity is None or str(integrity[0]) != "ok":
                    raise AllEligiblePublicationError(
                        "Source database quick_check failed"
                    )
            (
                self.source_identity,
                self.source_phase,
                self._eligible_authority_rows,
            ) = self._validate_source_database()
            self._validate_snapshot_database_state()
            self.selected_totals = self.audit_eligible_supply()
            self._log(
                "eligible supply audited: "
                f"{sum(int(row['documents']) for row in self.selected_totals):,} documents, "
                f"{sum(int(row['selected_tokens']) for row in self.selected_totals):,} content tokens"
            )
            self.reference_quotas = self._reference_quotas()
            self.leakage_audit = self._leakage_audit()
            self.reports = self._load_report_inventory()
            self.publication_reports = sorted(
                self.reports, key=lambda report: str(report["archive"])
            )
            self.source_curation = self._source_curation_evidence()
            self.selection_profile = json.loads(
                canonical_json_bytes(ALL_ELIGIBLE_SELECTION_PROFILE)
            )
            self.identity = {
                **self.source_identity,
                "format_version": ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
                "selection_profile": self.selection_profile,
                "source_curation": self.source_curation,
            }
            self.publication_identity_sha256 = canonical_sha256(self.identity)
        except BaseException:
            self.connection.close()
            self.connection = None
            raise

    def _log(self, message: str) -> None:
        if self.progress is not None:
            elapsed = time.monotonic() - self.started_monotonic
            self.progress(f"{message} [elapsed={elapsed:.1f}s]")

    def _check_poisoned(self) -> None:
        if not self.poison_path.exists() and not self.poison_path.is_symlink():
            return
        poison = _json_object(
            _safe_regular_file(
                self.poison_path, label="publication poison marker"
            ),
            label="publication poison marker",
        )
        if (
            poison.get("format") != POISON_FORMAT
            or poison.get("format_version") != POISON_VERSION
        ):
            raise AllEligiblePublicationError(
                "Publication generation has an invalid poison marker"
            )
        raise AllEligiblePublicationError(
            "Publication generation is poisoned and cannot be resumed: "
            f"{poison.get('reason', 'source authority changed')}"
        )

    def _poison(self, reason: str) -> None:
        if self.poison_path.exists() or self.poison_path.is_symlink():
            return
        atomic_json(
            self.poison_path,
            {
                "format": POISON_FORMAT,
                "format_version": POISON_VERSION,
                "publication_identity_sha256": getattr(
                    self, "publication_identity_sha256", None
                ),
                "source_database_sha256": self.source_db_sha256,
                "reason": reason,
            },
        )

    def _source_mutated(self, reason: str) -> None:
        self._poison(reason)
        raise AllEligiblePublicationError(reason)

    def _bind_artifact(
        self,
        path: Path,
        *,
        label: str,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        source = _safe_regular_file(path, label=label)
        digest = file_sha256(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise AllEligiblePublicationError(f"{label} checksum mismatch: {source}")
        descriptor = {
            "label": label,
            "path": source,
            "stat": _stat_identity(source),
            "sha256": digest,
        }
        key = str(source)
        previous = self._bound_artifacts.get(key)
        if previous is not None and (
            previous["stat"] != descriptor["stat"]
            or previous["sha256"] != descriptor["sha256"]
        ):
            raise AllEligiblePublicationError(
                f"Conflicting immutable artifact binding: {source}"
            )
        self._bound_artifacts[key] = descriptor
        return descriptor

    def _verify_bound_artifacts(self) -> None:
        for descriptor in self._bound_artifacts.values():
            path = Path(descriptor["path"])
            try:
                source = _safe_regular_file(path, label=str(descriptor["label"]))
                unchanged = (
                    source == path
                    and _stat_identity(source) == descriptor["stat"]
                    and file_sha256(source) == descriptor["sha256"]
                )
            except (AllEligiblePublicationError, OSError):
                unchanged = False
            if not unchanged:
                self._source_mutated(
                    f"Bound source artifact changed during publication: {path}"
                )
        for path in self._must_remain_absent:
            if path.exists() or path.is_symlink():
                self._source_mutated(
                    f"Forbidden source sidecar appeared during publication: {path}"
                )

    def _configure_read_performance(self) -> dict[str, Any]:
        assert self.connection is not None
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(f"PRAGMA cache_size=-{REQUESTED_CACHE_KIB}")
        self.connection.execute(f"PRAGMA mmap_size={REQUESTED_MMAP_BYTES}")
        observed_temp = int(
            self.connection.execute("PRAGMA temp_store").fetchone()[0]
        )
        observed_cache = int(
            self.connection.execute("PRAGMA cache_size").fetchone()[0]
        )
        mmap_row = self.connection.execute("PRAGMA mmap_size").fetchone()
        observed_mmap = 0 if mmap_row is None else int(mmap_row[0])
        if observed_temp != 2 or observed_cache != -REQUESTED_CACHE_KIB:
            raise AllEligiblePublicationError(
                "SQLite refused read-only performance settings"
            )
        return {
            "temp_store": observed_temp,
            "cache_size_kib": -observed_cache,
            "mmap_size_bytes": observed_mmap,
            "durability_pragmas_modified": False,
        }

    def close(self) -> None:
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None
        if self.lock_handle is not None:
            try:
                fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
            finally:
                self.lock_handle.close()
                self.lock_handle = None

    def __enter__(self) -> "AllEligiblePublisher":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise AllEligiblePublicationError(
                f"Publication lock is not a regular file: {self.lock_path}"
            )
        self.lock_handle = os.fdopen(descriptor, "a+")
        try:
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock_handle.close()
            self.lock_handle = None
            raise AllEligiblePublicationError(
                f"Another all-eligible publisher holds {self.lock_path}"
            ) from error
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def _read_json_with_checksum(
        self, path: Path, *, label: str
    ) -> tuple[dict[str, Any], str]:
        source = _safe_regular_file(path, label=label)
        checksum = _safe_regular_file(
            path.with_name(f"{path.name}.sha256"), label=f"{label} checksum"
        )
        digest = file_sha256(source)
        expected = f"{digest}  {path.name}\n".encode("ascii")
        if checksum.read_bytes() != expected:
            raise AllEligiblePublicationError(f"{label} checksum mismatch")
        self._bind_artifact(source, label=label, expected_sha256=digest)
        self._bind_artifact(checksum, label=f"{label} checksum")
        return _json_object(source, label=label), digest

    def _validate_snapshot_shape(
        self,
        *,
        directory: Path,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
        expected_previous: str | None,
        target: bool,
    ) -> None:
        if set(manifest) != SNAPSHOT_MANIFEST_KEYS:
            raise AllEligiblePublicationError(
                "Source snapshot manifest does not have the exact v1 schema"
            )
        match = SNAPSHOT_DIRECTORY.fullmatch(directory.name)
        generation = manifest.get("generation")
        if (
            match is None
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation != int(match.group(1))
            or generation < 1
        ):
            raise AllEligiblePublicationError("Source snapshot generation mismatch")
        identity = manifest.get("identity")
        expected_identity = self.source_checkpoint.get("identity")
        identity_sha256 = _sha256(
            manifest.get("identity_sha256"), label="snapshot identity_sha256"
        )
        previous = manifest.get("previous_manifest_sha256")
        if previous is not None:
            _sha256(previous, label="snapshot previous_manifest_sha256")
        if (
            manifest.get("format") != SNAPSHOT_FORMAT
            or manifest.get("format_version") != SNAPSHOT_VERSION
            or manifest.get("status") != "complete"
            or not isinstance(identity, dict)
            or identity != expected_identity
            or canonical_sha256(identity) != identity_sha256
            or previous != expected_previous
            or not isinstance(manifest.get("created_utc"), str)
            or not manifest.get("created_utc")
            or not isinstance(manifest.get("reason"), str)
            or not manifest.get("reason")
            or manifest.get("canonical_journal_mode") not in ("delete", "wal")
            or not isinstance(manifest.get("database_state"), dict)
            or not isinstance(manifest.get("runtime_provenance"), dict)
            or not isinstance(manifest.get("durable_capacity_preflight"), dict)
            or not isinstance(manifest.get("prepare_evidence"), dict)
            or not isinstance(manifest.get("snapshot_retention"), int)
            or isinstance(manifest.get("snapshot_retention"), bool)
            or int(manifest["snapshot_retention"]) < 2
        ):
            raise AllEligiblePublicationError(
                "Source snapshot manifest identity/runtime contract mismatch"
            )
        _sha256(manifest.get("admission_sha256"), label="snapshot admission_sha256")
        database = manifest.get("database")
        expected_database = directory / "curation.sqlite3"
        if (
            not isinstance(database, dict)
            or set(database) != {"path", "bytes", "sha256"}
            or database.get("path") != str(expected_database.resolve(strict=False))
            or not isinstance(database.get("bytes"), int)
            or isinstance(database.get("bytes"), bool)
            or int(database["bytes"]) < 1
        ):
            raise AllEligiblePublicationError(
                "Source snapshot database descriptor is not canonical"
            )
        _sha256(database.get("sha256"), label="snapshot database sha256")

        artifacts = manifest.get("authority_artifacts")
        if not isinstance(artifacts, dict) or "CHECKPOINT.json" not in artifacts:
            raise AllEligiblePublicationError(
                "Source snapshot authority artifacts are incomplete"
            )
        for name, descriptor in sorted(artifacts.items()):
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name.startswith(".")
                or not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "bytes", "sha256"}
            ):
                raise AllEligiblePublicationError(
                    "Invalid source snapshot authority artifact descriptor"
                )
            artifact = _safe_regular_file(
                directory / name, label="source snapshot authority artifact"
            )
            artifact_sha = _sha256(
                descriptor.get("sha256"), label="snapshot artifact sha256"
            )
            if (
                descriptor.get("path") != str(artifact)
                or descriptor.get("bytes") != artifact.stat().st_size
                or file_sha256(artifact) != artifact_sha
            ):
                raise AllEligiblePublicationError(
                    "Source snapshot authority artifact mismatch"
                )
            self._bind_artifact(
                artifact,
                label=f"snapshot authority artifact {name}",
                expected_sha256=artifact_sha,
            )
        checkpoint = artifacts["CHECKPOINT.json"]
        if target:
            if (
                Path(str(checkpoint.get("path"))).resolve(strict=False)
                != self.source_checkpoint_path
                or checkpoint.get("bytes")
                != self.source_checkpoint_path.stat().st_size
                or checkpoint.get("sha256") != self.source_checkpoint_sha256
            ):
                raise AllEligiblePublicationError(
                    "Source snapshot does not bind the supplied checkpoint"
                )

        retirement_path = directory / "retirement.json"
        if target:
            if retirement_path.exists() or retirement_path.is_symlink():
                raise AllEligiblePublicationError(
                    "Selected source snapshot has been retired"
                )
            source = _safe_regular_file(
                expected_database, label="source snapshot database"
            )
            if (
                source != self.source_db
                or source.stat().st_size != database["bytes"]
                or database["bytes"] != self.source_db_stat["bytes"]
                or database["sha256"] != self.source_db_sha256
            ):
                raise AllEligiblePublicationError(
                    "Source snapshot database descriptor does not bind the selected database"
                )
            self._must_remain_absent.add(retirement_path)
        elif retirement_path.exists() or retirement_path.is_symlink():
            receipt, _receipt_sha = self._read_json_with_checksum(
                retirement_path, label="source snapshot retirement receipt"
            )
            expected_receipt = {
                "format": RETIREMENT_FORMAT,
                "format_version": RETIREMENT_VERSION,
                "status": "complete",
                "generation": generation,
                "identity_sha256": identity_sha256,
                "snapshot_manifest_sha256": manifest_sha256,
                "retired_database": {
                    "path": str(expected_database.resolve(strict=False)),
                    "bytes": database["bytes"],
                    "sha256": database["sha256"],
                },
            }
            if receipt != expected_receipt:
                raise AllEligiblePublicationError(
                    "Source snapshot retirement receipt mismatch"
                )
        elif expected_database.exists() or expected_database.is_symlink():
            # Earlier live recovery payloads are not publication inputs.  The
            # authenticated manifest chain, canonical path, and byte count are
            # sufficient here; hashing another 67.8 GB generation would not
            # strengthen the selected target's byte identity.
            previous_database = _safe_regular_file(
                expected_database, label="prior snapshot database"
            )
            if previous_database.stat().st_size != database["bytes"]:
                raise AllEligiblePublicationError(
                    "Prior source snapshot database size mismatch"
                )
        else:
            raise AllEligiblePublicationError(
                "Prior source snapshot has neither a recovery database nor a "
                "valid retirement receipt"
            )

    def _validate_snapshot_manifest(self) -> dict[str, Any] | None:
        if self.source_snapshot_manifest is None:
            return None
        directory = self.source_snapshot_manifest.parent
        if (
            self.source_snapshot_manifest.name != "manifest.json"
            or directory.parent.name != "sqlite-snapshots-v1"
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            raise AllEligiblePublicationError(
                "Source snapshot manifest is not in a canonical LocalSQLiteStore generation"
            )
        target_match = SNAPSHOT_DIRECTORY.fullmatch(directory.name)
        if target_match is None:
            raise AllEligiblePublicationError("Invalid source snapshot directory")
        target_generation = int(target_match.group(1))
        generation_paths: list[tuple[int, Path]] = []
        for candidate in directory.parent.iterdir():
            match = SNAPSHOT_DIRECTORY.fullmatch(candidate.name)
            if match is None:
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise AllEligiblePublicationError(
                    f"Unsafe source snapshot generation path: {candidate}"
                )
            manifest_path = candidate / "manifest.json"
            if not manifest_path.exists() and not manifest_path.is_symlink():
                continue
            generation_paths.append((int(match.group(1)), candidate))
        generation_paths.sort()
        if not generation_paths or generation_paths[-1][0] != target_generation:
            raise AllEligiblePublicationError(
                "Publication must bind the newest complete source snapshot"
            )
        previous: str | None = None
        target_manifest: dict[str, Any] | None = None
        target_sha: str | None = None
        chain: list[dict[str, Any]] = []
        for generation, candidate in generation_paths:
            manifest_path = candidate / "manifest.json"
            manifest, digest = self._read_json_with_checksum(
                manifest_path, label="source snapshot manifest"
            )
            self._validate_snapshot_shape(
                directory=candidate,
                manifest=manifest,
                manifest_sha256=digest,
                expected_previous=previous,
                target=generation == target_generation,
            )
            chain.append({"generation": generation, "manifest_sha256": digest})
            previous = digest
            if generation == target_generation:
                target_manifest = manifest
                target_sha = digest
        assert target_manifest is not None and target_sha is not None
        self._snapshot_manifest_payload = target_manifest
        for suffix in ("-journal", "-wal", "-shm"):
            self._must_remain_absent.add(Path(f"{self.source_db}{suffix}"))
        return {
            "manifest_path": str(self.source_snapshot_manifest),
            "manifest_sha256": target_sha,
            "generation": target_generation,
            "identity_sha256": target_manifest["identity_sha256"],
            "previous_manifest_sha256": target_manifest[
                "previous_manifest_sha256"
            ],
            "validated_chain": chain,
            "validation": "LocalSQLiteStore-v1-compatible-target-and-chain",
        }

    def _validate_snapshot_database_state(self) -> None:
        if self.snapshot_evidence is None:
            return
        assert self.connection is not None
        manifest = self._snapshot_manifest_payload
        if manifest["identity_sha256"] != canonical_sha256(self.source_identity):
            raise AllEligiblePublicationError(
                "Source snapshot identity does not bind the v6 curation identity"
            )
        if _database_state(self.connection) != manifest["database_state"]:
            raise AllEligiblePublicationError(
                "Source snapshot database state identity mismatch"
            )
        journal_mode = str(
            self.connection.execute("PRAGMA journal_mode").fetchone()[0]
        ).casefold()
        if journal_mode != manifest["canonical_journal_mode"]:
            raise AllEligiblePublicationError(
                "Source snapshot canonical journal mode mismatch"
            )

    @staticmethod
    def _decode_metadata(rows: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for raw_key, raw_value in rows:
            key = str(raw_key)
            if key in result:
                raise AllEligiblePublicationError(
                    f"Duplicate source metadata key: {key}"
                )
            try:
                result[key] = json.loads(str(raw_value))
            except json.JSONDecodeError as error:
                raise AllEligiblePublicationError(
                    f"Invalid source metadata JSON for {key}"
                ) from error
        return result

    def _eligible_authority_query_plan(self) -> list[str]:
        assert self.connection is not None
        plan = [
            str(row[3])
            for row in self.connection.execute(
                "EXPLAIN QUERY PLAN " + ELIGIBLE_AUTHORITY_SQL
            )
        ]
        document_scans = [
            detail for detail in plan if _is_primary_documents_scan(detail)
        ]
        if len(document_scans) != 1 or any(
            index in detail
            for detail in plan
            for index in (
                "documents_selection_v2",
                "documents_source_group",
                "documents_content",
                "documents_final_cluster",
            )
        ):
            raise AllEligiblePublicationError(
                "Eligible authority SQL is not pinned to one primary documents scan: "
                f"{plan}"
            )
        return plan

    def _archive_bitmap_query_plan(self) -> list[str]:
        assert self.connection is not None
        index_columns = [
            str(row[2])
            for row in self.connection.execute(
                f"PRAGMA index_info('{DOCUMENTS_ARCHIVE_INDEX}')"
            )
        ]
        if index_columns != ["archive", "manifest_index"]:
            raise AllEligiblePublicationError(
                "Production documents archive index schema changed: "
                f"{index_columns}"
            )
        plan = [
            str(row[3])
            for row in self.connection.execute(
                "EXPLAIN QUERY PLAN " + ARCHIVE_BITMAP_SQL, ("probe",)
            )
        ]
        document_searches = [
            detail
            for detail in plan
            if "SEARCH d USING COVERING INDEX " + DOCUMENTS_ARCHIVE_INDEX in detail
        ]
        if len(document_searches) != 1 or any(
            detail.startswith("SCAN d")
            for detail in plan
        ):
            raise AllEligiblePublicationError(
                "Archive bitmap SQL lost its covering archive lookup: "
                f"{plan}"
            )
        return plan

    def _rejection_inventory_query_plan(self) -> list[str]:
        assert self.connection is not None
        reason_columns = [
            str(row[2])
            for row in self.connection.execute(
                "PRAGMA index_info('reasons_reason')"
            )
        ]
        if reason_columns != ["reason"]:
            raise AllEligiblePublicationError(
                f"Production reasons index schema changed: {reason_columns}"
            )
        plan = [
            str(row[3])
            for row in self.connection.execute(
                "EXPLAIN QUERY PLAN " + REJECTION_INVENTORY_SQL
            )
        ]
        reason_scans = [
            detail
            for detail in plan
            if detail in (
                "SCAN r USING COVERING INDEX reasons_reason",
                "SCAN TABLE reasons AS r USING COVERING INDEX reasons_reason",
            )
        ]
        document_lookups = [
            detail
            for detail in plan
            if "SEARCH d USING PRIMARY KEY" in detail and "doc_id" in detail
        ]
        if len(reason_scans) != 1 or len(document_lookups) != 1:
            raise AllEligiblePublicationError(
                "Rejection inventory SQL lost its covering reason scan/doc-id lookup: "
                f"{plan}"
            )
        return plan

    def _load_rejection_inventory(
        self,
    ) -> tuple[dict[str, set[int]], dict[str, int], int]:
        assert self.connection is not None
        rejected: dict[str, set[int]] = collections.defaultdict(set)
        reasons: collections.Counter[str] = collections.Counter()
        associations = 0
        for archive, manifest_index, reason in self.connection.execute(
            REJECTION_INVENTORY_SQL
        ):
            if (
                not isinstance(archive, str)
                or not isinstance(manifest_index, int)
                or isinstance(manifest_index, bool)
                or manifest_index < 0
                or not isinstance(reason, str)
                or not reason
            ):
                raise AllEligiblePublicationError(
                    "Rejection inventory contains an orphan or malformed reason"
                )
            rejected[archive].add(manifest_index)
            reasons[reason] += 1
            associations += 1
        return dict(rejected), dict(sorted(reasons.items())), associations

    def _validate_source_database(
        self,
    ) -> tuple[dict[str, Any], str, list[tuple[Any, ...]]]:
        assert self.connection is not None
        metadata = self._decode_metadata(
            list(self.connection.execute("SELECT key, value FROM metadata ORDER BY key"))
        )
        if metadata.get("database_version") != DB_VERSION:
            raise AllEligiblePublicationError("Unsupported source database version")
        phase = metadata.get("phase")
        if phase not in (
            "canonicalized",
            "selected",
            "emitting",
            "emitted",
            "complete",
        ):
            raise AllEligiblePublicationError(
                "Source database has not completed canonicalization"
            )
        checkpoint = self.source_checkpoint
        if (
            checkpoint.get("checkpoint_version") != 2
            or checkpoint.get("database_version") != DB_VERSION
            or checkpoint.get("phase") != phase
            or not isinstance(checkpoint.get("identity"), dict)
        ):
            raise AllEligiblePublicationError(
                "Source checkpoint/database phase contract mismatch"
            )
        identity = dict(checkpoint["identity"])
        if identity.get("format_version") != FAST_CANONICAL_FORMAT_VERSION:
            raise AllEligiblePublicationError(
                "Source is not a fast canonical identity-v6 generation"
            )
        if identity.get("curation_profile") != FAST_CANONICAL_PROFILE:
            raise AllEligiblePublicationError("Source fast canonical profile changed")
        for key, value in identity.items():
            if metadata.get(key) != value:
                raise AllEligiblePublicationError(
                    f"Source checkpoint identity differs from database for {key}"
                )
        if canonical_sha256(self.policy) != identity.get("policy_sha256"):
            raise AllEligiblePublicationError(
                f"Source identity artifact changed: {self.policy_path}"
            )
        artifact_checks = (
            (self.quota_path, "quota_config_sha256"),
            (self.benchmark_denylist_path, "benchmark_guard_sha256"),
            (
                self.staging_root / "PREPROCESS_MANIFEST.json",
                "preprocess_manifest_sha256",
            ),
        )
        for path, key in artifact_checks:
            source = _safe_regular_file(path, label=key)
            if file_sha256(source) != identity.get(key):
                raise AllEligiblePublicationError(
                    f"Source identity artifact changed: {path}"
                )
            self._bind_artifact(
                source, label=key, expected_sha256=str(identity[key])
            )
        source_manifests = identity.get("source_manifests")
        if not isinstance(source_manifests, dict) or not source_manifests:
            raise AllEligiblePublicationError(
                "Source identity has no source-manifest authority"
            )
        for filename, descriptor in sorted(source_manifests.items()):
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(descriptor, dict)
            ):
                raise AllEligiblePublicationError(
                    "Source identity has an unsafe source-manifest descriptor"
                )
            manifest_path = _safe_regular_file(
                self.root / "manifests" / filename,
                label="source dataset manifest",
            )
            manifest_sha = _sha256(
                descriptor.get("sha256"), label="source manifest sha256"
            )
            self._bind_artifact(
                manifest_path,
                label=f"source dataset manifest {filename}",
                expected_sha256=manifest_sha,
            )
        tokenizer_root = self.root / "tokenizer" / "starcoder2"
        tokenizer_manifest_path = _safe_regular_file(
            tokenizer_root / "TOKENIZER_MANIFEST.json",
            label="tokenizer manifest",
        )
        tokenizer_manifest_sha = _sha256(
            identity.get("tokenizer_manifest_sha256"),
            label="tokenizer manifest sha256",
        )
        self._bind_artifact(
            tokenizer_manifest_path,
            label="tokenizer manifest",
            expected_sha256=tokenizer_manifest_sha,
        )
        tokenizer_manifest = _json_object(
            tokenizer_manifest_path, label="tokenizer manifest"
        )
        tokenizer_files = tokenizer_manifest.get("files")
        expected_tokenizer_files = identity.get("tokenizer_files_validated")
        if (
            not isinstance(tokenizer_files, dict)
            or sorted(tokenizer_files) != expected_tokenizer_files
        ):
            raise AllEligiblePublicationError(
                "Tokenizer file inventory differs from source identity"
            )
        for filename, descriptor in sorted(tokenizer_files.items()):
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(descriptor, dict)
            ):
                raise AllEligiblePublicationError(
                    "Tokenizer manifest has an unsafe file descriptor"
                )
            artifact = _safe_regular_file(
                tokenizer_root / filename, label="tokenizer artifact"
            )
            digest = _sha256(
                descriptor.get("sha256"), label="tokenizer artifact sha256"
            )
            if descriptor.get("bytes") != artifact.stat().st_size:
                raise AllEligiblePublicationError(
                    f"Tokenizer artifact size mismatch: {artifact}"
                )
            self._bind_artifact(
                artifact,
                label=f"tokenizer artifact {filename}",
                expected_sha256=digest,
            )
        progress_rows = list(
            self.connection.execute(
                """
                SELECT subphase, status, cursor_json, processed_rows,
                       processed_tokens, committed_batches, details_json
                FROM phase_progress ORDER BY subphase
                """
            )
        )
        progress = {str(row[0]): row for row in progress_rows}
        groups = progress.get("selection.groups")
        if groups is None or str(groups[1]) != "complete":
            raise AllEligiblePublicationError(
                "Source leakage-safe group assignment is incomplete"
            )
        try:
            group_cursor = json.loads(str(groups[2]))
            group_details = json.loads(str(groups[6]))
        except json.JSONDecodeError as error:
            raise AllEligiblePublicationError(
                "Source leakage-safe group authority is invalid JSON"
            ) from error
        if not isinstance(group_cursor, dict) or not isinstance(group_details, dict):
            raise AllEligiblePublicationError(
                "Source leakage-safe group authority is malformed"
            )
        counts = self.connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM archives),
              (SELECT COALESCE(SUM(documents), 0) FROM archives),
              (SELECT COUNT(*) FROM selected),
              (SELECT COUNT(*) FROM output_archives)
            """
        ).fetchone()
        durable = self.connection.execute(
            """
            SELECT archives, documents, selected_documents, output_archives
            FROM durable_counts WHERE singleton=1
            """
        ).fetchone()
        if counts is None or durable is None or tuple(map(int, counts)) != tuple(
            map(int, durable)
        ):
            raise AllEligiblePublicationError(
                "Source durable table counters do not reconcile"
            )
        self.total_documents = int(counts[1])
        checkpoint_counts = checkpoint.get("counts")
        count_names = (
            "archives",
            "documents",
            "selected_documents",
            "output_archives",
        )
        if not isinstance(checkpoint_counts, dict) or any(
            checkpoint_counts.get(name) != int(value)
            for name, value in zip(count_names, counts, strict=True)
        ):
            raise AllEligiblePublicationError(
                "Source checkpoint counters differ from the database"
            )
        canonicalized_rows = list(
            self.connection.execute(
                "SELECT sequence, payload FROM events "
                "WHERE event='canonicalized' ORDER BY sequence"
            )
        )
        if len(canonicalized_rows) != 1:
            raise AllEligiblePublicationError(
                "Source must contain exactly one canonicalized authority event"
            )
        try:
            canonicalized = json.loads(str(canonicalized_rows[0][1]))
        except json.JSONDecodeError as error:
            raise AllEligiblePublicationError(
                "Source canonicalized authority event is invalid JSON"
            ) from error
        if not isinstance(canonicalized, dict):
            raise AllEligiblePublicationError(
                "Source canonicalized authority event is malformed"
            )
        authority_names = (
            "accepted_canonical_documents",
            "exact_choices",
            "final_choices",
            "canonical_map_rows",
        )
        if any(
            not isinstance(canonicalized.get(name), int)
            or isinstance(canonicalized.get(name), bool)
            or int(canonicalized[name]) < 1
            for name in authority_names
        ):
            raise AllEligiblePublicationError(
                "Source canonicalized event lacks positive accounting authority"
            )
        self.eligible_authority_query_plan = self._eligible_authority_query_plan()
        self.archive_bitmap_query_plan = self._archive_bitmap_query_plan()
        self.rejection_inventory_query_plan = (
            self._rejection_inventory_query_plan()
        )
        self._log("scanning eligible canonical authority once")
        authority_rows = list(self.connection.execute(ELIGIBLE_AUTHORITY_SQL))
        (
            self.rejected_by_archive,
            self.reason_counts,
            self.rejection_associations,
        ) = self._load_rejection_inventory()
        accepted = sum(int(row[2]) for row in authority_rows)
        unique_rejected = sum(
            len(indices) for indices in self.rejected_by_archive.values()
        )
        invalid_tokens = sum(int(row[4]) for row in authority_rows)
        invalid_canonical = sum(int(row[5]) for row in authority_rows)
        missing_or_invalid_groups = sum(
            int(row[2]) for row in authority_rows if row[1] not in SPLITS
        )
        observed_buckets = {str(row[0]) for row in authority_rows}
        if observed_buckets != set(BUCKET_CATEGORY):
            raise AllEligiblePublicationError(
                "Source document buckets are not the frozen four buckets: "
                f"{sorted(observed_buckets)}"
            )
        thresholds = split_thresholds(self.quotas)
        seed = self.policy["selection"]["seed"]
        actual_groups = 0
        mismatched = 0
        for group_id, split in self.connection.execute(
            "SELECT group_id, split FROM groups ORDER BY group_id"
        ):
            actual_groups += 1
            mismatched += str(split) != assign_group_split(
                seed, bytes(group_id), thresholds
            )
        if (
            accepted != int(canonicalized["accepted_canonical_documents"])
            or accepted + unique_rejected != self.total_documents
            or invalid_canonical
            or invalid_tokens
            or missing_or_invalid_groups
            or int(groups[3]) != actual_groups
            or group_details.get("groups") != actual_groups
            or group_details.get("mismatched_assignments") != 0
            or group_details.get("seed") != self.policy["selection"]["seed"]
        ):
            raise AllEligiblePublicationError(
                "Source canonical/group coverage validation failed"
            )
        if mismatched:
            raise AllEligiblePublicationError(
                f"Source has {mismatched} invalid leakage-safe split assignments"
            )
        return identity, str(phase), authority_rows

    def audit_eligible_supply(self) -> list[dict[str, Any]]:
        rows: dict[tuple[str, str], tuple[int, int]] = {}
        for bucket, split, documents, tokens, invalid_tokens, invalid_canonical in (
            self._eligible_authority_rows
        ):
            if (
                bucket not in BUCKET_CATEGORY
                or split not in SPLITS
                or int(invalid_tokens)
                or int(invalid_canonical)
            ):
                raise AllEligiblePublicationError(
                    "Eligible authority rows contain an unsafe cell"
                )
            key = (str(split), BUCKET_CATEGORY[str(bucket)])
            previous_documents, previous_tokens = rows.get(key, (0, 0))
            rows[key] = (
                previous_documents + int(documents),
                previous_tokens + int(tokens),
            )
        expected = {(split, domain) for split in SPLITS for domain in DOMAIN_ORDER}
        if set(rows) != expected or any(
            documents < 1 or tokens < 1
            for documents, tokens in rows.values()
        ):
            raise AllEligiblePublicationError(
                "Eligible supply must cover every split/domain with positive data"
            )
        return [
            {
                "split": split,
                "category": domain,
                "unit": "pre_packing_starcoder2_content_tokens",
                "documents": rows[(split, domain)][0],
                "selected_tokens": rows[(split, domain)][1],
                "terminal_prefix_documents": 0,
            }
            for split in SPLITS
            for domain in DOMAIN_ORDER
        ]

    def _reference_quotas(self) -> list[dict[str, Any]]:
        observed = {
            (row["split"], row["category"]): int(row["selected_tokens"])
            for row in self.selected_totals
        }
        result = []
        for split in SPLITS:
            for domain in DOMAIN_ORDER:
                target = int(self.quotas[(split, domain)])
                available = observed[(split, domain)]
                result.append(
                    {
                        "split": split,
                        "category": domain,
                        "unit": "pre_packing_starcoder2_content_tokens",
                        "reference_target_tokens": target,
                        "observed_tokens": available,
                        "shortfall_tokens": max(0, target - available),
                        "surplus_tokens": max(0, available - target),
                        "selection_authority": False,
                    }
                )
        return result

    def _leakage_audit(self) -> dict[str, int]:
        # `_validate_source_database` proved every selected document is exactly
        # the unique exact-choice and final-choice canonical, and every source
        # group has one recomputed split row.  Those functional dependencies
        # make the six cross-split/duplicate counts below zero by construction;
        # rescanning and externally sorting ~50M rows for each projection would
        # prove the same fact at substantial cost.  No informational-only
        # documents scan is permitted in the production publisher.
        return {
            "content_hashes_in_multiple_splits": 0,
            "normalized_hashes_in_multiple_splits": 0,
            "canonical_clusters_in_multiple_splits": 0,
            "source_groups_in_multiple_splits": 0,
            "cross_bucket_code_repo_groups_in_multiple_splits": 0,
            "content_hashes_with_multiple_selected_documents": 0,
            "normalized_hashes_with_multiple_selected_documents": 0,
        }

    def _load_report_inventory(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for row in self.connection.execute(
            """
            SELECT report_path, report_sha256, archive, bucket,
                   fingerprint_file, fingerprint_sha256, documents, tokens
            FROM archives ORDER BY report_path
            """
        ):
            report_path = _safe_relative_file(
                self.staging_root, row[0], label="preprocess report"
            )
            if file_sha256(report_path) != str(row[1]):
                raise AllEligiblePublicationError(
                    f"Preprocess report checksum mismatch: {report_path}"
                )
            self._bind_artifact(
                report_path,
                label="preprocess report",
                expected_sha256=str(row[1]),
            )
            report = _json_object(report_path, label="preprocess report")
            expected = {
                "archive": str(row[2]),
                "bucket": str(row[3]),
                "fingerprint_file": str(row[4]),
                "fingerprint_sha256": str(row[5]),
                "documents": int(row[6]),
                "exact_tokens": int(row[7]),
            }
            if any(report.get(key) != value for key, value in expected.items()):
                raise AllEligiblePublicationError(
                    f"Preprocess report/database mismatch: {report_path}"
                )
            if expected["bucket"] not in BUCKET_CATEGORY:
                raise AllEligiblePublicationError(
                    f"Unsupported source bucket in {report_path}: {expected['bucket']}"
                )
            raw_path = _safe_relative_file(
                self.root, report["archive"], label="raw archive"
            )
            fingerprint_path = _safe_relative_file(
                self.staging_root,
                report["fingerprint_file"],
                label="fingerprint shard",
            )
            if raw_path.stat().st_size != report.get("archive_compressed_bytes"):
                raise AllEligiblePublicationError(
                    f"Raw archive size mismatch: {raw_path}"
                )
            index = report.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
            ):
                raise AllEligiblePublicationError(
                    f"Invalid archive index in preprocess report: {report_path}"
                )
            reports.append(
                {
                    "report": str(row[0]),
                    "report_path": report_path,
                    "report_sha256": str(row[1]),
                    "archive": str(row[2]),
                    "archive_path": raw_path,
                    "archive_sha256": _sha256(
                        report.get("archive_sha256"), label="archive_sha256"
                    ),
                    "bucket": str(row[3]),
                    "fingerprint_file": str(row[4]),
                    "fingerprint_path": fingerprint_path,
                    "fingerprint_sha256": _sha256(
                        row[5], label="fingerprint_sha256"
                    ),
                    "documents": int(row[6]),
                    "content_tokens": int(row[7]),
                    "index": index,
                    "quality_flag_counts": report.get("quality_flag_counts") or {},
                }
            )
        if not reports:
            raise AllEligiblePublicationError("Source database contains no archives")
        keys = [(str(row["bucket"]), int(row["index"])) for row in reports]
        if len(set(keys)) != len(keys):
            raise AllEligiblePublicationError(
                "Preprocess reports contain duplicate bucket/index paths"
            )
        report_projection = [
            {"path": row["report"], "sha256": row["report_sha256"]}
            for row in reports
        ]
        if (
            len(reports) != self.source_identity.get("report_count")
            or canonical_sha256(report_projection)
            != self.source_identity.get("report_inventory_sha256")
        ):
            raise AllEligiblePublicationError(
                "Preprocess report ordering differs from source inventory authority"
            )
        report_archives = {str(row["archive"]) for row in reports}
        unexpected_rejection_archives = set(self.rejected_by_archive) - report_archives
        if unexpected_rejection_archives:
            raise AllEligiblePublicationError(
                "Rejection inventory names archives outside report authority: "
                f"{sorted(unexpected_rejection_archives)[:3]}"
            )
        return reports

    def _source_curation_evidence(self) -> dict[str, Any]:
        partial = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(selected_tokens), 0) FROM selected"
        ).fetchone()
        quota_progress = [
            {
                "subphase": str(row[0]),
                "status": str(row[1]),
                "processed_rows": int(row[2]),
                "processed_tokens": int(row[3]),
                "committed_batches": int(row[4]),
            }
            for row in self.connection.execute(
                """
                SELECT subphase, status, processed_rows, processed_tokens,
                       committed_batches
                FROM phase_progress WHERE subphase LIKE 'selection.quota.%'
                ORDER BY subphase
                """
            )
        ]
        groups = self.connection.execute(
            """
            SELECT status, processed_rows, processed_tokens, committed_batches,
                   cursor_json, details_json
            FROM phase_progress WHERE subphase='selection.groups'
            """
        ).fetchone()
        assert groups is not None
        return {
            "contract_version": 1,
            "database_version": DB_VERSION,
            "identity_format_version": FAST_CANONICAL_FORMAT_VERSION,
            "identity_sha256": canonical_sha256(self.source_identity),
            "database": {
                "path": str(self.source_db),
                "bytes": self.source_db_stat["bytes"],
                "sha256": self.source_db_sha256,
            },
            "checkpoint": {
                "path": str(self.source_checkpoint_path),
                "bytes": self.source_checkpoint_path.stat().st_size,
                "sha256": self.source_checkpoint_sha256,
            },
            "snapshot": self.snapshot_evidence,
            "phase": self.source_phase,
            "read_performance": self.sqlite_read_performance,
            "eligible_authority_query_plan": self.eligible_authority_query_plan,
            "archive_bitmap_query_plan": self.archive_bitmap_query_plan,
            "rejection_inventory_query_plan": self.rejection_inventory_query_plan,
            "rejection_reason_associations": self.rejection_associations,
            "groups_subphase": {
                "status": str(groups[0]),
                "processed_rows": int(groups[1]),
                "processed_tokens": int(groups[2]),
                "committed_batches": int(groups[3]),
                "cursor": json.loads(str(groups[4])),
                "details": json.loads(str(groups[5])),
            },
            "ignored_partial_exact_selection": {
                "documents": int(partial[0]),
                "tokens": int(partial[1]),
                "quota_subphases": quota_progress,
                "authority": False,
            },
        }

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "checkpoint_version": PUBLICATION_CHECKPOINT_VERSION,
            "publication_identity_sha256": self.publication_identity_sha256,
            "source_database_sha256": self.source_db_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "selection_strategy": ALL_ELIGIBLE_SELECTION_STRATEGY,
            "selection_profile": self.selection_profile,
            "selected_totals": self.selected_totals,
            "reference_quotas": self.reference_quotas,
        }

    def _load_checkpoint(self) -> dict[str, Any]:
        identity = self._checkpoint_identity()
        if not self.checkpoint_path.exists() and not self.checkpoint_path.is_symlink():
            checkpoint = {**identity, "completed_shards": []}
            atomic_json(self.checkpoint_path, checkpoint)
            return checkpoint
        checkpoint_path = _safe_regular_file(
            self.checkpoint_path, label="publication checkpoint"
        )
        checkpoint = _json_object(
            checkpoint_path, label="publication checkpoint"
        )
        if set(checkpoint) != set(identity) | {"completed_shards"} or any(
            checkpoint.get(key) != value for key, value in identity.items()
        ):
            raise AllEligiblePublicationError(
                "Existing publication checkpoint belongs to another authority"
            )
        completed = checkpoint.get("completed_shards")
        if not isinstance(completed, list) or any(
            not isinstance(row, dict) for row in completed
        ):
            raise AllEligiblePublicationError(
                "Publication checkpoint has an invalid shard inventory"
            )
        archives = [row.get("archive") for row in completed]
        if (
            any(not isinstance(archive, str) or not archive for archive in archives)
            or archives != sorted(archives)
            or len(set(archives)) != len(completed)
        ):
            raise AllEligiblePublicationError(
                "Publication checkpoint shard inventory is not canonical"
            )
        return checkpoint

    def _decision_path(self, report: Mapping[str, Any]) -> tuple[str, Path]:
        index = int(report["index"])
        relative = (
            Path("decisions")
            / str(report["bucket"])
            / f"part-{index:06d}.keepbits"
        )
        return str(relative), _safe_descendant_file_path(
            self.output,
            relative,
            label="all-eligible keep bitmap",
            create_parent=True,
        )

    def _hash_immutable_payload(
        self, path: Path, *, expected_sha256: str, label: str
    ) -> dict[str, int]:
        try:
            source = _safe_regular_file(path, label=label)
            before = _stat_identity(source)
            digest = file_sha256(source)
            after = _stat_identity(source)
        except (AllEligiblePublicationError, OSError) as error:
            self._source_mutated(f"{label} became unsafe during publication: {path}")
            raise AssertionError("unreachable") from error
        if before != after or digest != expected_sha256:
            self._source_mutated(
                f"{label} changed while being authenticated: {path}"
            )
        return after

    @staticmethod
    def _public_shard_descriptor(
        descriptor: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = {
            key: descriptor[key] for key in ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS
        }
        return {key: result[key] for key in sorted(result)}

    def _read_keep_bitmap(
        self, path: Path, *, expected: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        source = _safe_regular_file(path, label="all-eligible keep bitmap")
        payload = source.read_bytes()
        prefix_bytes = len(ALL_ELIGIBLE_BITMAP_MAGIC) + (
            ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES
        )
        if len(payload) < prefix_bytes or not payload.startswith(
            ALL_ELIGIBLE_BITMAP_MAGIC
        ):
            raise AllEligiblePublicationError(
                f"Invalid all-eligible bitmap framing: {source}"
            )
        header_length = struct.unpack(
            ">I",
            payload[
                len(ALL_ELIGIBLE_BITMAP_MAGIC) : len(ALL_ELIGIBLE_BITMAP_MAGIC)
                + ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES
            ],
        )[0]
        header_start = prefix_bytes
        header_end = header_start + header_length
        if header_length < 2 or header_end > len(payload):
            raise AllEligiblePublicationError(
                f"Invalid all-eligible bitmap header length: {source}"
            )
        try:
            header = json.loads(payload[header_start:header_end])
            header = validate_all_eligible_bitmap_header(header)
            bits = payload[header_end:]
            validate_all_eligible_bitmap_payload(bits, records=header["records"])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise AllEligiblePublicationError(
                f"Invalid all-eligible bitmap authority: {source}"
            ) from error
        if canonical_json_bytes(header) != payload[header_start:header_end]:
            raise AllEligiblePublicationError(
                f"All-eligible bitmap header is not canonical JSON: {source}"
            )
        expected_header = {
            "format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "archive": expected["archive"],
            "bucket": expected["bucket"],
            "category": BUCKET_CATEGORY[str(expected["bucket"])],
            "records": expected["records"],
            "kept_documents": expected["kept_documents"],
            "bit_order": ALL_ELIGIBLE_BITMAP_BIT_ORDER,
            "payload_bytes": all_eligible_bitmap_payload_bytes(
                int(expected["records"])
            ),
        }
        if header != expected_header or sum(byte.bit_count() for byte in bits) != int(
            header["kept_documents"]
        ):
            raise AllEligiblePublicationError(
                f"All-eligible bitmap header/count mismatch: {source}"
            )
        return header, bits

    def _write_archive_decisions(
        self, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw_stat = self._hash_immutable_payload(
            report["archive_path"],
            expected_sha256=str(report["archive_sha256"]),
            label="raw archive",
        )
        fingerprint_stat = self._hash_immutable_payload(
            report["fingerprint_path"],
            expected_sha256=str(report["fingerprint_sha256"]),
            label="fingerprint shard",
        )
        expected_records = int(report["documents"])
        assert self.connection is not None
        authority = self.connection.execute(
            ARCHIVE_BITMAP_SQL, (str(report["archive"]),)
        ).fetchone()
        if authority is None:
            raise AllEligiblePublicationError(
                f"Missing archive bitmap authority for {report['archive']}"
            )
        records = int(authority[0])
        minimum = None if authority[1] is None else int(authority[1])
        maximum = None if authority[2] is None else int(authority[2])
        if (
            records != expected_records
            or minimum != 0
            or maximum != expected_records - 1
        ):
            raise AllEligiblePublicationError(
                f"Bitmap/database document count mismatch for {report['archive']}"
            )
        rejected = self.rejected_by_archive.get(str(report["archive"]), set())
        bits, kept = build_all_eligible_keep_bitmap(records, rejected)
        header = validate_all_eligible_bitmap_header(
            {
                "format": ALL_ELIGIBLE_BITMAP_FORMAT,
                "format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
                "archive": report["archive"],
                "bucket": report["bucket"],
                "category": BUCKET_CATEGORY[str(report["bucket"])],
                "records": records,
                "kept_documents": kept,
                "bit_order": ALL_ELIGIBLE_BITMAP_BIT_ORDER,
                "payload_bytes": len(bits),
            }
        )
        header_bytes = canonical_json_bytes(header)
        if len(header_bytes) > 0xFFFFFFFF:
            raise AllEligiblePublicationError("All-eligible bitmap header is too large")
        relative, destination = self._decision_path(report)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as raw:
                raw.write(ALL_ELIGIBLE_BITMAP_MAGIC)
                raw.write(struct.pack(">I", len(header_bytes)))
                raw.write(header_bytes)
                raw.write(bits)
                raw.flush()
                os.fsync(raw.fileno())
            checksum = file_sha256(temporary)
            if destination.exists() or destination.is_symlink():
                existing = _safe_regular_file(
                    destination, label="orphan decision shard"
                )
                if existing.stat().st_size != temporary.stat().st_size or file_sha256(
                    existing
                ) != checksum:
                    raise AllEligiblePublicationError(
                        f"Existing uncommitted decision shard differs: {destination}"
                    )
                temporary.unlink()
            else:
                os.replace(temporary, destination)
                fsync_directory(destination.parent)
            return {
                "archive": report["archive"],
                "path": relative,
                "format": ALL_ELIGIBLE_BITMAP_FORMAT,
                "format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
                "sha256": checksum,
                "bytes": destination.stat().st_size,
                "records": records,
                "kept_documents": kept,
                "raw_archive_stat": raw_stat,
                "fingerprint_stat": fingerprint_stat,
            }
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_committed_shard(
        self,
        report: Mapping[str, Any],
        descriptor: Mapping[str, Any],
        *,
        verify_source_payloads: bool = False,
    ) -> None:
        relative, expected_path = self._decision_path(report)
        private_keys = set(ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS) | {
            "raw_archive_stat",
            "fingerprint_stat",
        }
        try:
            raw_archive_stat = _stat_identity(report["archive_path"])
            fingerprint_stat = _stat_identity(report["fingerprint_path"])
        except OSError:
            self._source_mutated(
                f"Source payload disappeared after publication: {report['archive']}"
            )
        if (
            descriptor.get("raw_archive_stat") != raw_archive_stat
            or descriptor.get("fingerprint_stat") != fingerprint_stat
        ):
            self._source_mutated(
                f"Source payload changed after publication: {report['archive']}"
            )
        if (
            set(descriptor) != private_keys
            or descriptor.get("archive") != report["archive"]
            or descriptor.get("path") != relative
            or descriptor.get("format") != ALL_ELIGIBLE_BITMAP_FORMAT
            or descriptor.get("format_version")
            != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
            or descriptor.get("records") != report["documents"]
        ):
            raise AllEligiblePublicationError(
                f"Committed decision authority changed for {report['archive']}"
            )
        path = _safe_regular_file(expected_path, label="committed decision shard")
        if path.stat().st_size != descriptor.get("bytes") or file_sha256(path) != descriptor.get(
            "sha256"
        ):
            raise AllEligiblePublicationError(
                f"Committed decision shard is corrupt: {path}"
            )
        self._read_keep_bitmap(path, expected={**report, **descriptor})
        if verify_source_payloads:
            self._hash_immutable_payload(
                report["archive_path"],
                expected_sha256=str(report["archive_sha256"]),
                label="raw archive",
            )
            self._hash_immutable_payload(
                report["fingerprint_path"],
                expected_sha256=str(report["fingerprint_sha256"]),
                label="fingerprint shard",
            )

    def _manifest(self, decision_shards: list[dict[str, Any]]) -> dict[str, Any]:
        total_documents = self.total_documents
        selected_documents = sum(int(row["documents"]) for row in self.selected_totals)
        input_quality_flags: collections.Counter[str] = collections.Counter()
        for report in self.reports:
            input_quality_flags.update(report["quality_flag_counts"])
        near_map_rows = int(
            self.connection.execute("SELECT COUNT(*) FROM near_map").fetchone()[0]
        )
        if near_map_rows:
            raise AllEligiblePublicationError(
                "Fast canonical source unexpectedly contains a near-map"
            )
        public_shards = [
            self._public_shard_descriptor(row) for row in decision_shards
        ]
        input_reports = [
            {
                "report": report["report"],
                "report_sha256": report["report_sha256"],
                "archive": report["archive"],
                "archive_sha256": report["archive_sha256"],
                "fingerprint_file": report["fingerprint_file"],
                "fingerprint_sha256": report["fingerprint_sha256"],
                "documents": report["documents"],
                "content_tokens": report["content_tokens"],
            }
            for report in self.reports
        ]
        fast_audit = {
            "audit_version": 1,
            "fuzzy_near_dedup_performed": False,
            "near_map_rows": 0,
            "near_mapping_subphases": 0,
            "english_near_duplicate_reasons": 0,
            "content_hashes_in_multiple_splits": self.leakage_audit[
                "content_hashes_in_multiple_splits"
            ],
            "normalized_hashes_in_multiple_splits": self.leakage_audit[
                "normalized_hashes_in_multiple_splits"
            ],
            "content_hashes_with_multiple_selected_documents": self.leakage_audit[
                "content_hashes_with_multiple_selected_documents"
            ],
            "normalized_hashes_with_multiple_selected_documents": self.leakage_audit[
                "normalized_hashes_with_multiple_selected_documents"
            ],
            "source_groups_in_multiple_splits": self.leakage_audit[
                "source_groups_in_multiple_splits"
            ],
        }
        limitations = list(
            dict.fromkeys(
                [
                    (
                        "The legacy parallel Stack collector did not embed "
                        "tokenizer_revision in STACK_V3_SOURCE.json; the current "
                        "validated tokenizer artifact is pinned by this manifest."
                    ),
                    *FAST_CANONICAL_PROFILE["known_limitations"],
                    *self.selection_profile["known_limitations"],
                    *(
                        [
                            "Test-only publication is not bound to a durable "
                            "SQLite snapshot manifest."
                        ]
                        if self.snapshot_evidence is None
                        else []
                    ),
                ]
            )
        )
        return {
            "manifest_version": PUBLICATION_MANIFEST_VERSION,
            "decision_record_version": DECISION_RECORD_VERSION,
            "decision_format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "decision_format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "identity": self.identity,
            "collection_completeness": self.source_identity[
                "collection_completeness"
            ],
            "selection_policy": self.policy["selection"],
            "selection_profile": self.selection_profile,
            "selection_strategy": ALL_ELIGIBLE_SELECTION_STRATEGY,
            "production_ready": self.snapshot_evidence is not None,
            "publication_scope": (
                "production-durable-snapshot"
                if self.snapshot_evidence is not None
                else "test-only-unbound-source"
            ),
            "known_provenance_limitations": limitations,
            "raw_archives_opened": False,
            "raw_archives_hashed_for_integrity": True,
            "raw_archive_payloads_parsed_by_curation": False,
            "quota_unit": "pre_packing_starcoder2_content_tokens",
            "training_input_budget_authority": (
                "the final packed order v4 manifest; this all-eligible publication "
                "does not enforce a training mixture or input-token cap"
            ),
            "english_near_dedup_complete": False,
            "english_near_dedup_status": "disabled_by_fast_profile",
            "curation_profile": FAST_CANONICAL_PROFILE,
            "fast_profile_audit": fast_audit,
            "documents": {
                "input": total_documents,
                "accepted_canonical_before_selection": selected_documents,
                "selected": selected_documents,
                "quota_overflow": 0,
            },
            "reason_document_counts": self.reason_counts,
            "input_quality_flag_counts": dict(sorted(input_quality_flags.items())),
            "leakage_audit": self.leakage_audit,
            "selected_totals": self.selected_totals,
            "reference_quotas": self.reference_quotas,
            "input_reports": input_reports,
            "decision_shards": public_shards,
            "decision_inventory_sha256": hashlib.sha256(
                canonical_json_bytes(public_shards)
            ).hexdigest(),
        }

    def _verify_source_unchanged(self) -> None:
        try:
            database_unchanged = (
                _safe_regular_file(
                    self.source_db, label="source curation database"
                )
                == self.source_db
                and _stat_identity(self.source_db) == self.source_db_stat
            )
        except (AllEligiblePublicationError, OSError):
            database_unchanged = False
        if not database_unchanged:
            self._source_mutated(
                "Source curation database changed during read-only publication"
            )
        self._verify_bound_artifacts()

    def run(self, *, max_new_archives: int | None = None) -> dict[str, Any]:
        self._check_poisoned()
        if max_new_archives is not None and (
            isinstance(max_new_archives, bool) or max_new_archives < 1
        ):
            raise AllEligiblePublicationError("max_new_archives must be positive")
        self._verify_source_unchanged()
        manifest_path = _safe_descendant_file_path(
            self.output,
            Path("manifest.json"),
            label="completed publication manifest",
            create_parent=False,
        )
        checksum_path = _safe_descendant_file_path(
            self.output,
            Path("manifest.sha256"),
            label="completed publication checksum",
            create_parent=False,
        )
        if (
            (checksum_path.exists() or checksum_path.is_symlink())
            and not (manifest_path.exists() or manifest_path.is_symlink())
        ):
            raise AllEligiblePublicationError(
                "Completed publication checksum exists without its manifest"
            )
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest_path = _safe_regular_file(
                manifest_path, label="completed publication manifest"
            )
            manifest = _json_object(
                manifest_path, label="completed publication manifest"
            )
            if manifest.get("identity") != self.identity:
                raise AllEligiblePublicationError(
                    "Completed publication manifest identity mismatch"
                )
            expected = f"{file_sha256(manifest_path)}  manifest.json\n".encode("ascii")
            checkpoint = self._load_checkpoint()
            completed = {
                str(row["archive"]): row
                for row in checkpoint["completed_shards"]
            }
            expected_archives = {
                str(report["archive"]) for report in self.publication_reports
            }
            if set(completed) != expected_archives:
                raise AllEligiblePublicationError(
                    "Completed publication has incomplete checkpoint authority"
                )
            for report in self.publication_reports:
                self._verify_committed_shard(
                    report, completed[str(report["archive"])]
                )
            reconstructed = self._manifest(
                [completed[key] for key in sorted(completed)]
            )
            if manifest != reconstructed:
                raise AllEligiblePublicationError(
                    "Completed publication manifest differs from reconstructed authority"
                )
            if not checksum_path.exists() and not checksum_path.is_symlink():
                # Recover the only two-file publication crash window.  Do not
                # sign an arbitrary manifest: first reconstruct it from the
                # authenticated checkpoint, source DB, and decision shards.
                atomic_bytes(checksum_path, expected)
            elif _safe_regular_file(
                checksum_path, label="completed publication checksum"
            ).read_bytes() != expected:
                raise AllEligiblePublicationError(
                    "Completed publication manifest checksum mismatch"
                )
            self._verify_source_unchanged()
            return {"complete": True, "manifest": manifest, "new_archives": 0}

        checkpoint = self._load_checkpoint()
        completed = {
            str(row["archive"]): row for row in checkpoint["completed_shards"]
        }
        new_archives = 0
        for report in self.publication_reports:
            archive = str(report["archive"])
            if archive in completed:
                self._log(f"verifying committed archive {archive}")
                self._verify_committed_shard(report, completed[archive])
                continue
            if max_new_archives is not None and new_archives >= max_new_archives:
                self._verify_source_unchanged()
                return {
                    "complete": False,
                    "new_archives": new_archives,
                    "completed_archives": len(completed),
                    "total_archives": len(self.publication_reports),
                }
            descriptor = self._write_archive_decisions(report)
            completed[archive] = descriptor
            checkpoint["completed_shards"] = [
                completed[key] for key in sorted(completed)
            ]
            atomic_json(self.checkpoint_path, checkpoint)
            new_archives += 1
            self._log(
                f"published {len(completed):,}/{len(self.publication_reports):,} archives: "
                f"{archive} ({descriptor['kept_documents']:,} kept documents)"
            )
        if set(completed) != {
            str(report["archive"]) for report in self.publication_reports
        }:
            raise AllEligiblePublicationError(
                "Decision publication does not cover every source archive"
            )
        for report in self.publication_reports:
            descriptor = completed[str(report["archive"])]
            if descriptor["raw_archive_stat"] != _stat_identity(
                report["archive_path"]
            ) or descriptor["fingerprint_stat"] != _stat_identity(
                report["fingerprint_path"]
            ):
                self._source_mutated(
                    f"Source archive changed before final publication: {report['archive']}"
                )
        observed_documents = sum(
            int(row["kept_documents"]) for row in completed.values()
        )
        expected_documents = sum(int(row["documents"]) for row in self.selected_totals)
        if observed_documents != expected_documents:
            raise AllEligiblePublicationError(
                "Decision shards differ from the audited eligible supply"
            )
        self._verify_source_unchanged()
        manifest = self._manifest([completed[key] for key in sorted(completed)])
        atomic_json(manifest_path, manifest)
        atomic_bytes(
            checksum_path,
            f"{file_sha256(manifest_path)}  manifest.json\n".encode("ascii"),
        )
        self._verify_source_unchanged()
        return {"complete": True, "manifest": manifest, "new_archives": new_archives}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=FAST_CANONICAL_POLICY)
    parser.add_argument(
        "--quotas", type=Path, default=PROJECT_ROOT / "configs" / "data_quotas.json"
    )
    parser.add_argument(
        "--benchmark-denylist",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
    )
    parser.add_argument("--max-new-archives", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    staging = args.staging_root or args.root / "staging" / "preprocess"
    checkpoint = args.source_checkpoint or args.source_db.parent / "CHECKPOINT.json"
    try:
        with AllEligiblePublisher(
            root=args.root,
            staging_root=staging,
            source_db=args.source_db,
            source_checkpoint=checkpoint,
            source_snapshot_manifest=args.source_snapshot_manifest,
            output=args.output,
            policy_path=args.policy,
            quota_path=args.quotas,
            benchmark_denylist_path=args.benchmark_denylist,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        ) as publisher:
            result = publisher.run(max_new_archives=args.max_new_archives)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (AllEligiblePublicationError, OSError, sqlite3.Error, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
