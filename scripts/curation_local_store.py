#!/usr/bin/env python3
"""Crash-safe local SQLite working storage with durable snapshot generations.

This module is intentionally independent of corpus-selection logic.  A single
writer may run SQLite in WAL mode on pod-local NVMe or a sufficiently large
RAM filesystem while transactionally consistent backups are published to the
durable curation ``.work`` directory.  Complete snapshot generations are
immutable; an incomplete generation is never a recovery candidate.

The durable canonical database is replaced only from a fully authenticated
snapshot.  Its previous inode is hard-linked into the promotion generation
before the atomic replacement, so a crash leaves either the old canonical file
or the complete new one and always preserves the old bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_FORMAT = "curation-local-sqlite-snapshot"
SNAPSHOT_VERSION = 1
STORE_FORMAT = "curation-local-sqlite-store"
STORE_VERSION = 1
PROMOTION_FORMAT = "curation-local-sqlite-promotion"
PROMOTION_VERSION = 1
RETIREMENT_FORMAT = "curation-local-sqlite-retirement"
RETIREMENT_VERSION = 1
SNAPSHOT_DIRECTORY = re.compile(r"snapshot-(\d{12})\Z")
RAM_FILESYSTEMS = frozenset(("tmpfs", "ramfs"))
LOCAL_JOURNAL_MODE = "wal"
DEFAULT_SNAPSHOT_RETENTION = 2


class LocalStoreError(RuntimeError):
    """A local-work safety or recovery invariant was not proven."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalStoreError(f"Value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _real_directory(path: Path, *, create: bool, label: str) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise LocalStoreError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise LocalStoreError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _atomic_new(path: Path, payload: bytes) -> None:
    parent = _real_directory(path.parent, create=False, label="publication parent")
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise LocalStoreError(f"Refusing to overwrite immutable evidence: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        fsync_directory(parent)
    except FileExistsError as exc:
        raise LocalStoreError(f"Refusing to overwrite immutable evidence: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    parent = _real_directory(path.parent, create=False, label="publication parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json_with_sidecar(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    _atomic_new(path, encoded)
    _atomic_new(
        path.with_name(f"{path.name}.sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return {"path": str(path.resolve(strict=True)), "sha256": digest, "bytes": len(encoded)}


def _read_json_with_sidecar(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    source = _regular_file(path, label=label)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _regular_file(
        source.with_name(f"{source.name}.sha256"), label=f"{label} sidecar"
    )
    expected = f"{digest}  {source.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected:
        raise LocalStoreError(f"{label} sidecar mismatch: {sidecar}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalStoreError(f"Invalid {label} JSON: {source}") from exc
    if not isinstance(value, dict):
        raise LocalStoreError(f"{label} must be a JSON object")
    canonical_json_bytes(value)
    return value, digest


def read_cgroup_memory(
    *,
    v2_root: Path = Path("/sys/fs/cgroup"),
    v1_root: Path = Path("/sys/fs/cgroup/memory"),
) -> dict[str, Any]:
    """Return a conservative cgroup memory limit/current snapshot."""

    v2_max = v2_root / "memory.max"
    v2_current = v2_root / "memory.current"
    if v2_max.is_file() and v2_current.is_file():
        maximum_raw = v2_max.read_text(encoding="ascii").strip()
        current = int(v2_current.read_text(encoding="ascii").strip())
        maximum = None if maximum_raw == "max" else int(maximum_raw)
        return {
            "version": 2,
            "limit_bytes": maximum,
            "current_bytes": current,
            "available_bytes": None if maximum is None else max(0, maximum - current),
        }
    v1_limit = v1_root / "memory.limit_in_bytes"
    v1_usage = v1_root / "memory.usage_in_bytes"
    if v1_limit.is_file() and v1_usage.is_file():
        maximum = int(v1_limit.read_text(encoding="ascii").strip())
        current = int(v1_usage.read_text(encoding="ascii").strip())
        # Common v1 "unlimited" sentinel is near 2**63 and is not an admission bound.
        finite = maximum if maximum < (1 << 60) else None
        return {
            "version": 1,
            "limit_bytes": finite,
            "current_bytes": current,
            "available_bytes": None if finite is None else max(0, finite - current),
        }
    return {
        "version": None,
        "limit_bytes": None,
        "current_bytes": None,
        "available_bytes": None,
    }


def storage_admission(
    *,
    local_root: Path,
    filesystem_type: str,
    expected_documents: int,
    projected_bytes_per_document: int,
    safety_numerator: int,
    safety_denominator: int,
    transaction_sidecar_bytes: int,
    minimum_free_bytes: int,
    reclaimable_existing_bytes: int = 0,
    cgroup_memory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail unless the projected database can fit on the local filesystem.

    ``reclaimable_existing_bytes`` is storage already owned by this local
    SQLite authority. It is credited only against the projected final database
    size, never against transaction headroom or the minimum-free reserve. This
    avoids requiring space for a second complete database on every resume.
    """

    integer_values = (
        expected_documents,
        projected_bytes_per_document,
        safety_numerator,
        safety_denominator,
        transaction_sidecar_bytes,
        minimum_free_bytes,
        reclaimable_existing_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_values
    ) or any(
        value < 1
        for value in (
            expected_documents,
            projected_bytes_per_document,
            safety_numerator,
            safety_denominator,
            transaction_sidecar_bytes,
            minimum_free_bytes,
        )
    ):
        raise LocalStoreError(
            "Local-work storage projection inputs must be positive integers; "
            "reclaimable_existing_bytes may be zero"
        )
    probe = local_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    probe = _real_directory(probe, create=False, label="local-work storage parent")
    usage = shutil.disk_usage(probe)
    projected_database_bytes = (
        expected_documents
        * projected_bytes_per_document
        * safety_numerator
        + safety_denominator
        - 1
    ) // safety_denominator
    remaining_database_bytes = max(
        0, projected_database_bytes - reclaimable_existing_bytes
    )
    required = remaining_database_bytes + transaction_sidecar_bytes + minimum_free_bytes
    if usage.free < required:
        raise LocalStoreError(
            "Insufficient local SQLite capacity: "
            f"free={usage.free:,}, required={required:,}, "
            f"documents={expected_documents:,}"
        )
    normalized_fs = filesystem_type.casefold()
    memory = dict(cgroup_memory or read_cgroup_memory())
    if normalized_fs in RAM_FILESYSTEMS:
        available = memory.get("available_bytes")
        if available is not None and (
            isinstance(available, bool) or not isinstance(available, int) or available < required
        ):
            raise LocalStoreError(
                "RAM-backed SQLite exceeds cgroup headroom: "
                f"available={int(available):,}, required={required:,}"
            )
        # A finite cgroup reading is mandatory for RAM-backed mode. Without it,
        # tmpfs free space can exceed the memory actually granted to the pod.
        if available is None:
            raise LocalStoreError(
                "RAM-backed SQLite requires a finite readable cgroup memory limit"
            )
    return {
        "status": "pass",
        "filesystem_type": filesystem_type,
        "ram_backed": normalized_fs in RAM_FILESYSTEMS,
        "expected_documents": expected_documents,
        "projected_database_bytes_with_safety": projected_database_bytes,
        "reclaimable_existing_bytes": reclaimable_existing_bytes,
        "remaining_database_bytes_with_safety": remaining_database_bytes,
        "transaction_sidecar_bytes": transaction_sidecar_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "required_free_bytes": required,
        "observed_free_bytes": int(usage.free),
        "filesystem_total_bytes": int(usage.total),
        "cgroup_memory": memory,
    }


def _sqlite_uri(path: Path, *, immutable: bool = False) -> str:
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    return f"file:{path.resolve(strict=True).as_posix()}{suffix}"


def _database_state(connection: sqlite3.Connection) -> dict[str, Any]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    schema_rows = [
        tuple(row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type, name"
        )
    ]
    state: dict[str, Any] = {
        "schema_sha256": canonical_sha256(schema_rows),
        "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
        "page_size": int(connection.execute("PRAGMA page_size").fetchone()[0]),
    }
    if "metadata" in tables:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM metadata ORDER BY key")
        }
        state.update(
            metadata_sha256=canonical_sha256(metadata),
            database_version=json.loads(metadata.get("database_version", "null")),
            phase=json.loads(metadata.get("phase", "null")),
        )
    if "durable_counts" in tables:
        row = connection.execute(
            "SELECT archives, documents, selected_documents, output_archives "
            "FROM durable_counts WHERE singleton=1"
        ).fetchone()
        if row is not None:
            state["durable_counts"] = {
                "archives": int(row[0]),
                "documents": int(row[1]),
                "selected_documents": int(row[2]),
                "output_archives": int(row[3]),
            }
    if "phase_progress" in tables:
        progress = [
            tuple(row)
            for row in connection.execute(
                "SELECT subphase, status, cursor_json, processed_rows, "
                "processed_tokens, committed_batches, details_json "
                "FROM phase_progress ORDER BY subphase"
            )
        ]
        state["phase_progress_sha256"] = canonical_sha256(progress)
        state["phase_progress_rows"] = len(progress)
    if "events" in tables:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0), COUNT(*) FROM events"
        ).fetchone()
        state["last_event_sequence"] = int(row[0])
        state["event_rows"] = int(row[1])
    return state


def _check_database(path: Path, *, expected_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = _regular_file(path, label="SQLite database")
    connection = sqlite3.connect(_sqlite_uri(source, immutable=True), uri=True, timeout=120)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            raise LocalStoreError(f"SQLite integrity check failed: {source}")
        state = _database_state(connection)
    finally:
        connection.close()
    if expected_state is not None and state != dict(expected_state):
        raise LocalStoreError(f"SQLite state identity mismatch: {source}")
    return state


def _copy_file_verified(source: Path, destination: Path, *, expected_sha256: str) -> None:
    parent = _real_directory(destination.parent, create=True, label="copy destination")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while block := reader.read(8 * 1024 * 1024):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != expected_sha256:
            raise LocalStoreError("SQLite copy source checksum changed")
        os.replace(temporary, destination)
        fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


class LocalSQLiteStore:
    """Manage one dedicated local DB and immutable durable snapshot chain."""

    def __init__(
        self,
        *,
        local_root: Path,
        durable_work: Path,
        canonical_db: Path,
        identity: Mapping[str, Any],
        admission: Mapping[str, Any],
        canonical_journal_mode: str,
        snapshot_interval_seconds: int,
        snapshot_retention: int = DEFAULT_SNAPSHOT_RETENTION,
        runtime_provenance: Mapping[str, Any],
    ) -> None:
        if snapshot_interval_seconds < 1:
            raise LocalStoreError("Snapshot interval must be positive")
        if canonical_journal_mode not in ("delete", "wal"):
            raise LocalStoreError("Canonical journal mode must be delete or wal")
        if isinstance(snapshot_retention, bool) or snapshot_retention < 2:
            raise LocalStoreError("Snapshot retention must preserve at least two generations")
        self.local_root = local_root
        self.durable_work = durable_work
        self.canonical_db = canonical_db
        self.local_db = local_root / "curation.sqlite3"
        self.local_temp = local_root / "sqlite-tmp"
        self.snapshot_root = durable_work / "sqlite-snapshots-v1"
        self.control_path = local_root / "LOCAL_SQLITE_STORE.json"
        self.identity = dict(identity)
        self.identity_sha256 = canonical_sha256(self.identity)
        self.admission = dict(admission)
        self.canonical_journal_mode = canonical_journal_mode
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.snapshot_retention = snapshot_retention
        self.runtime_provenance = dict(runtime_provenance)
        self.last_snapshot_monotonic = time.monotonic()
        self.last_prepare_evidence: dict[str, Any] = {}

    def _control(self) -> dict[str, Any]:
        return {
            "format": STORE_FORMAT,
            "format_version": STORE_VERSION,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "canonical_db": str(self.canonical_db.resolve(strict=False)),
            "local_db": str(self.local_db.resolve(strict=False)),
            "snapshot_root": str(self.snapshot_root.resolve(strict=False)),
            "canonical_journal_mode": self.canonical_journal_mode,
            "snapshot_retention": self.snapshot_retention,
        }

    def prepare(self) -> dict[str, Any]:
        _real_directory(self.durable_work, create=True, label="durable curation work")
        _real_directory(self.snapshot_root, create=True, label="durable snapshot root")
        local = _real_directory(self.local_root, create=True, label="local SQLite root")
        _real_directory(self.local_temp, create=True, label="local SQLite temp")
        control = self._control()
        if self.control_path.exists() or self.control_path.is_symlink():
            if self.control_path.is_symlink() or not self.control_path.is_file():
                raise LocalStoreError("Unsafe local SQLite control path")
            try:
                found = json.loads(self.control_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise LocalStoreError("Invalid local SQLite control file") from exc
            if found != control:
                raise LocalStoreError(
                    "Local SQLite root belongs to another curation authority"
                )
        else:
            unexpected = [
                path
                for path in local.iterdir()
                if path.name not in ("sqlite-tmp",) and not path.name.endswith(".part")
            ]
            if unexpected:
                raise LocalStoreError(
                    f"Unowned local SQLite root is not empty: {unexpected[:3]}"
                )
            _atomic_new(
                self.control_path,
                json.dumps(control, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        for stale in local.glob(".*.part"):
            if stale.is_file() and not stale.is_symlink():
                stale.unlink()
        self._cleanup_incomplete_generations()
        snapshots, invalid = self.valid_snapshots(
            cleanup_retired_payloads=True
        )
        if self.local_db.exists() or self.local_db.is_symlink():
            _regular_file(self.local_db, label="local SQLite database")
            rollback = Path(f"{self.local_db}-journal")
            if rollback.exists() or rollback.is_symlink():
                raise LocalStoreError(
                    "Existing local WAL database has a rollback-journal sidecar"
                )
            sidecars: dict[str, dict[str, Any]] = {}
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self.local_db}{suffix}")
                if not sidecar.exists() and not sidecar.is_symlink():
                    continue
                source = _regular_file(sidecar, label=f"local SQLite {suffix} sidecar")
                sidecars[suffix] = {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": file_sha256(source),
                }
            if "-shm" in sidecars and "-wal" not in sidecars:
                raise LocalStoreError("Existing local SQLite root has an orphan SHM sidecar")
            if not sidecars:
                _check_database(self.local_db)
            self.last_prepare_evidence = {
                "source": "existing-local-crash-recovery",
                "database_sha256": file_sha256(self.local_db),
                "sqlite_sidecars_before_recovery": sidecars,
                "identity_validation": "mandatory-after-sqlite-recovery-by-builder-open",
                "invalid_durable_snapshots": invalid,
            }
            return dict(self.last_prepare_evidence)
        if snapshots:
            latest = snapshots[-1]
            source = Path(latest["database"]["path"])
            _copy_file_verified(
                source,
                self.local_db,
                expected_sha256=latest["database"]["sha256"],
            )
            _check_database(self.local_db, expected_state=latest["database_state"])
            evidence = {
                "source": "durable-snapshot",
                "snapshot_generation": latest["generation"],
                "snapshot_manifest_sha256": latest["_manifest_sha256"],
                "database_sha256": latest["database"]["sha256"],
                "invalid_durable_snapshots": invalid,
            }
        elif self.canonical_db.exists() or self.canonical_db.is_symlink():
            canonical = _regular_file(self.canonical_db, label="canonical curation database")
            unsafe = [
                Path(f"{canonical}{suffix}")
                for suffix in ("-journal", "-wal", "-shm")
                if Path(f"{canonical}{suffix}").exists()
            ]
            if unsafe:
                raise LocalStoreError(
                    "Canonical database has SQLite sidecars; stop/recover its writer first: "
                    + ", ".join(str(path) for path in unsafe)
                )
            state = _check_database(canonical)
            digest = file_sha256(canonical)
            _copy_file_verified(canonical, self.local_db, expected_sha256=digest)
            _check_database(self.local_db, expected_state=state)
            evidence = {
                "source": "canonical-durable-database",
                "database_sha256": digest,
                "invalid_durable_snapshots": invalid,
            }
        else:
            evidence = {
                "source": "new-empty-database",
                "database_sha256": None,
                "invalid_durable_snapshots": invalid,
            }
        self.last_prepare_evidence = evidence
        return dict(evidence)

    def _generation_paths(self) -> list[tuple[int, Path]]:
        if not self.snapshot_root.exists():
            return []
        result: list[tuple[int, Path]] = []
        for path in self.snapshot_root.iterdir():
            match = SNAPSHOT_DIRECTORY.fullmatch(path.name)
            if match is not None and path.is_dir() and not path.is_symlink():
                result.append((int(match.group(1)), path))
        return sorted(result)

    def _cleanup_incomplete_generations(self) -> None:
        """Reclaim generations that never published an authority manifest."""

        removed = False
        for _generation, directory in self._generation_paths():
            manifest = directory / "manifest.json"
            if manifest.exists() or manifest.is_symlink():
                continue
            shutil.rmtree(directory)
            removed = True
        if removed:
            fsync_directory(self.snapshot_root)

    def _validate_snapshot(
        self,
        directory: Path,
        *,
        previous_manifest_sha256: str | None,
    ) -> dict[str, Any]:
        manifest, manifest_sha = _read_json_with_sidecar(
            directory / "manifest.json", label="SQLite snapshot manifest"
        )
        if (
            manifest.get("format") != SNAPSHOT_FORMAT
            or manifest.get("format_version") != SNAPSHOT_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("identity_sha256") != self.identity_sha256
            or manifest.get("identity") != self.identity
            or manifest.get("previous_manifest_sha256") != previous_manifest_sha256
        ):
            raise LocalStoreError("SQLite snapshot manifest identity/chain mismatch")
        artifacts = manifest.get("authority_artifacts")
        if not isinstance(artifacts, dict):
            raise LocalStoreError("SQLite snapshot authority artifacts are missing")
        for name in sorted(artifacts):
            descriptor = artifacts[name]
            if (
                not isinstance(descriptor, dict)
                or Path(name).name != name
                or name.startswith(".")
            ):
                raise LocalStoreError("Invalid SQLite snapshot authority artifact")
            artifact = directory / name
            source_artifact = _regular_file(
                artifact, label="SQLite snapshot authority artifact"
            )
            if (
                descriptor.get("path") != str(artifact.resolve(strict=True))
                or descriptor.get("bytes") != source_artifact.stat().st_size
                or descriptor.get("sha256") != file_sha256(source_artifact)
            ):
                raise LocalStoreError("SQLite snapshot authority artifact mismatch")
        match = SNAPSHOT_DIRECTORY.fullmatch(directory.name)
        if match is None or manifest.get("generation") != int(match.group(1)):
            raise LocalStoreError("SQLite snapshot generation mismatch")
        database = manifest.get("database")
        if not isinstance(database, dict):
            raise LocalStoreError("SQLite snapshot database descriptor is missing")
        expected_path = directory / "curation.sqlite3"
        if database.get("path") != str(expected_path.resolve(strict=False)):
            raise LocalStoreError("SQLite snapshot database path is not canonical")
        retirement_path = directory / "retirement.json"
        if retirement_path.exists() or retirement_path.is_symlink():
            retirement, _retirement_sha = _read_json_with_sidecar(
                retirement_path, label="SQLite snapshot retirement receipt"
            )
            expected_retirement = {
                "format": RETIREMENT_FORMAT,
                "format_version": RETIREMENT_VERSION,
                "status": "complete",
                "generation": manifest.get("generation"),
                "identity_sha256": self.identity_sha256,
                "snapshot_manifest_sha256": manifest_sha,
                "retired_database": {
                    "path": str(expected_path.resolve(strict=False)),
                    "bytes": database.get("bytes"),
                    "sha256": database.get("sha256"),
                },
            }
            if retirement != expected_retirement:
                raise LocalStoreError("SQLite snapshot retirement receipt mismatch")
            if expected_path.exists() or expected_path.is_symlink():
                retired_source = _regular_file(
                    expected_path, label="retired SQLite snapshot database"
                )
                if (
                    retired_source.stat().st_size != database.get("bytes")
                    or file_sha256(retired_source) != database.get("sha256")
                ):
                    raise LocalStoreError(
                        "Retired SQLite snapshot payload does not match its receipt"
                    )
            result = dict(manifest)
            result["_manifest_path"] = str(
                (directory / "manifest.json").resolve(strict=True)
            )
            result["_manifest_sha256"] = manifest_sha
            result["_retired"] = True
            result["_retired_payload_present"] = expected_path.exists()
            return result
        source = _regular_file(expected_path, label="SQLite snapshot database")
        if source.stat().st_size != database.get("bytes") or file_sha256(source) != database.get(
            "sha256"
        ):
            raise LocalStoreError("SQLite snapshot database checksum mismatch")
        state = manifest.get("database_state")
        if not isinstance(state, dict):
            raise LocalStoreError("SQLite snapshot database state is missing")
        _check_database(source, expected_state=state)
        result = dict(manifest)
        result["_manifest_path"] = str((directory / "manifest.json").resolve(strict=True))
        result["_manifest_sha256"] = manifest_sha
        return result

    def valid_snapshots(
        self, *, cleanup_retired_payloads: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        previous: str | None = None
        for generation, directory in self._generation_paths():
            manifest_path = directory / "manifest.json"
            # A generation without a published manifest is an incomplete crash
            # artifact, not corruption and never a recovery candidate.
            if not manifest_path.exists() and not manifest_path.is_symlink():
                continue
            try:
                snapshot = self._validate_snapshot(
                    directory, previous_manifest_sha256=previous
                )
            except (LocalStoreError, OSError, sqlite3.Error) as exc:
                invalid.append({"generation": generation, "error": str(exc)})
                continue
            valid.append(snapshot)
            previous = snapshot["_manifest_sha256"]
            if snapshot.get("_retired"):
                valid.pop()
                if (
                    cleanup_retired_payloads
                    and snapshot.get("_retired_payload_present")
                ):
                    payload = directory / "curation.sqlite3"
                    payload.unlink()
                    fsync_directory(directory)
        return valid, invalid

    def _retire_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        directory = Path(str(snapshot["_manifest_path"])).parent
        database = snapshot["database"]
        receipt = {
            "format": RETIREMENT_FORMAT,
            "format_version": RETIREMENT_VERSION,
            "status": "complete",
            "generation": snapshot["generation"],
            "identity_sha256": self.identity_sha256,
            "snapshot_manifest_sha256": snapshot["_manifest_sha256"],
            "retired_database": {
                "path": str((directory / "curation.sqlite3").resolve(strict=False)),
                "bytes": database["bytes"],
                "sha256": database["sha256"],
            },
        }
        receipt_path = directory / "retirement.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            found, _digest = _read_json_with_sidecar(
                receipt_path, label="SQLite snapshot retirement receipt"
            )
            if found != receipt:
                raise LocalStoreError("SQLite snapshot retirement receipt collision")
        else:
            _publish_json_with_sidecar(receipt_path, receipt)
        database_path = directory / "curation.sqlite3"
        if database_path.exists() or database_path.is_symlink():
            _regular_file(database_path, label="retired SQLite snapshot database")
            database_path.unlink()
            fsync_directory(directory)

    def _make_snapshot_capacity(self, source: sqlite3.Connection) -> dict[str, int]:
        page_count = int(source.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(source.execute("PRAGMA page_size").fetchone()[0])
        estimated_bytes = page_count * page_size
        minimum_free = int(self.admission.get("minimum_free_bytes", 0))
        observed_free = int(shutil.disk_usage(self.snapshot_root).free)
        required_free = estimated_bytes + minimum_free
        if observed_free < required_free:
            raise LocalStoreError(
                "Insufficient durable capacity for the next SQLite snapshot: "
                f"free={observed_free:,}, required={required_free:,}, "
                f"estimated_database={estimated_bytes:,}"
            )
        return {
            "page_count": page_count,
            "page_size": page_size,
            "estimated_database_bytes": estimated_bytes,
            "observed_free_bytes": observed_free,
            "required_free_bytes": required_free,
        }

    def _next_generation(self) -> tuple[int, Path]:
        existing = self._generation_paths()
        generation = (existing[-1][0] + 1) if existing else 1
        directory = self.snapshot_root / f"snapshot-{generation:012d}"
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise LocalStoreError(f"Snapshot generation already exists: {directory}") from exc
        fsync_directory(self.snapshot_root)
        return generation, directory

    def snapshot(
        self,
        source: sqlite3.Connection,
        *,
        reason: str,
        authority_artifacts: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        if source.in_transaction:
            raise LocalStoreError("Cannot snapshot with an active SQLite transaction")
        snapshots, invalid = self.valid_snapshots(
            cleanup_retired_payloads=True
        )
        if invalid:
            # Continuing after a corrupt *complete* generation could fork the
            # provenance chain. Recovery may fall back, but new publication
            # requires explicit operator cleanup/investigation.
            raise LocalStoreError(f"Invalid complete durable snapshots exist: {invalid}")
        previous = snapshots[-1]["_manifest_sha256"] if snapshots else None
        # Retire oldest payloads *before* allocating the next full copy.  The
        # authenticated tiny manifests remain in the chain, and at least one
        # complete prior recovery database remains if the new backup fails.
        retire_count = max(0, len(snapshots) - self.snapshot_retention + 1)
        for old in snapshots[:retire_count]:
            self._retire_snapshot(old)
        capacity = self._make_snapshot_capacity(source)
        generation, directory = self._next_generation()
        temporary = directory / ".curation.sqlite3.part"
        destination = sqlite3.connect(temporary, timeout=120)
        try:
            destination.execute("PRAGMA synchronous=FULL")
            source.backup(destination)
            destination.commit()
            found = str(
                destination.execute(
                    f"PRAGMA journal_mode={self.canonical_journal_mode.upper()}"
                ).fetchone()[0]
            ).casefold()
            if found != self.canonical_journal_mode:
                raise LocalStoreError(
                    f"Snapshot refused canonical journal mode {self.canonical_journal_mode}"
                )
            if found == "wal":
                checkpoint = destination.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is None or int(checkpoint[0]) != 0:
                    raise LocalStoreError("Cannot checkpoint snapshot WAL")
            integrity = destination.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]) != "ok":
                raise LocalStoreError("Snapshot integrity check failed")
            state = _database_state(destination)
        finally:
            destination.close()
        # SQLite can leave an empty WAL and/or SHM bookkeeping file after the
        # final connection closes.  They contain no committed pages after the
        # successful mode transition/checkpoint and are not part of the
        # single-file durable snapshot.  Reject non-empty WAL/rollback state,
        # remove only empty bookkeeping files, then revalidate the standalone
        # database against the state observed before close.
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(f"{temporary}{suffix}")
            if not sidecar.exists() and not sidecar.is_symlink():
                continue
            source_sidecar = _regular_file(
                sidecar, label=f"closed snapshot {suffix} sidecar"
            )
            if suffix in ("-journal", "-wal") and source_sidecar.stat().st_size:
                raise LocalStoreError(
                    f"Closed snapshot retained non-empty SQLite sidecar: {sidecar}"
                )
            source_sidecar.unlink()
        fsync_directory(directory)
        _check_database(temporary, expected_state=state)
        database = directory / "curation.sqlite3"
        os.replace(temporary, database)
        fsync_directory(directory)
        database_sha = file_sha256(database)
        published_artifacts: dict[str, dict[str, Any]] = {}
        for name, payload in sorted((authority_artifacts or {}).items()):
            if Path(name).name != name or name.startswith(".") or not isinstance(payload, bytes):
                raise LocalStoreError("Snapshot artifacts must be named byte payloads")
            artifact_path = directory / name
            _atomic_new(artifact_path, payload)
            published_artifacts[name] = {
                "path": str(artifact_path.resolve(strict=True)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "format_version": SNAPSHOT_VERSION,
            "status": "complete",
            "generation": generation,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "previous_manifest_sha256": previous,
            "canonical_journal_mode": self.canonical_journal_mode,
            "database": {
                "path": str(database.resolve(strict=True)),
                "bytes": database.stat().st_size,
                "sha256": database_sha,
            },
            "database_state": state,
            "authority_artifacts": published_artifacts,
            "runtime_provenance": self.runtime_provenance,
            "admission_sha256": canonical_sha256(self.admission),
            "durable_capacity_preflight": capacity,
            "snapshot_retention": self.snapshot_retention,
            "prepare_evidence": self.last_prepare_evidence,
        }
        published = _publish_json_with_sidecar(directory / "manifest.json", manifest)
        manifest["_manifest_path"] = published["path"]
        manifest["_manifest_sha256"] = published["sha256"]
        self.last_snapshot_monotonic = time.monotonic()
        return manifest

    def maybe_snapshot(
        self,
        source: sqlite3.Connection,
        *,
        reason: str,
        force: bool = False,
        authority_artifacts: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any] | None:
        if not self.snapshot_due(force=force):
            return None
        return self.snapshot(
            source,
            reason=reason,
            authority_artifacts=authority_artifacts,
        )

    def snapshot_due(self, *, force: bool = False) -> bool:
        return force or (
            time.monotonic() - self.last_snapshot_monotonic
            >= self.snapshot_interval_seconds
        )

    def promote_latest(self) -> dict[str, Any]:
        snapshots, invalid = self.valid_snapshots()
        if invalid:
            raise LocalStoreError(f"Cannot promote with invalid snapshots: {invalid}")
        if not snapshots:
            raise LocalStoreError("No complete SQLite snapshot is available for promotion")
        latest = snapshots[-1]
        source = Path(latest["database"]["path"])
        source_sha = str(latest["database"]["sha256"])
        canonical_parent = _real_directory(
            self.canonical_db.parent, create=True, label="canonical database parent"
        )
        unsafe = [
            Path(f"{self.canonical_db}{suffix}")
            for suffix in ("-journal", "-wal", "-shm")
            if Path(f"{self.canonical_db}{suffix}").exists()
        ]
        if unsafe:
            raise LocalStoreError(f"Canonical database has live SQLite sidecars: {unsafe}")
        previous_canonical: dict[str, Any] | None = None
        backup = Path(latest["_manifest_path"]).parent / "canonical-before-promotion.sqlite3"
        already_current = False
        if self.canonical_db.exists() or self.canonical_db.is_symlink():
            canonical = _regular_file(self.canonical_db, label="canonical database")
            if file_sha256(canonical) == source_sha:
                already_current = True
                if backup.exists() or backup.is_symlink():
                    found_backup = _regular_file(
                        backup, label="canonical pre-promotion backup"
                    )
                    previous_canonical = {
                        "path": str(found_backup),
                        "bytes": found_backup.stat().st_size,
                        "sha256": file_sha256(found_backup),
                    }
            else:
                previous_sha = file_sha256(canonical)
                previous_canonical = {
                    "path": str(backup.resolve(strict=False)),
                    "bytes": canonical.stat().st_size,
                    "sha256": previous_sha,
                }
                if backup.exists() or backup.is_symlink():
                    found = _regular_file(
                        backup, label="canonical pre-promotion backup"
                    )
                    if (
                        found.stat().st_size != previous_canonical["bytes"]
                        or file_sha256(found) != previous_sha
                    ):
                        raise LocalStoreError("Canonical pre-promotion backup collision")
                else:
                    os.link(canonical, backup)
                    fsync_directory(backup.parent)
        elif backup.exists() or backup.is_symlink():
            found = _regular_file(backup, label="canonical pre-promotion backup")
            previous_canonical = {
                "path": str(found),
                "bytes": found.stat().st_size,
                "sha256": file_sha256(found),
            }
        if not already_current:
            source_bytes = source.stat().st_size
            minimum_free = int(self.admission.get("minimum_free_bytes", 0))
            observed_free = int(shutil.disk_usage(canonical_parent).free)
            required_free = source_bytes + minimum_free
            if observed_free < required_free:
                raise LocalStoreError(
                    "Insufficient durable capacity for canonical SQLite promotion: "
                    f"free={observed_free:,}, required={required_free:,}, "
                    f"database={source_bytes:,}"
                )
            candidate = canonical_parent / f".{self.canonical_db.name}.promotion-{latest['generation']:012d}.part"
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()
            _copy_file_verified(source, candidate, expected_sha256=source_sha)
            os.replace(candidate, self.canonical_db)
            fsync_directory(canonical_parent)
        if file_sha256(self.canonical_db) != source_sha:
            raise LocalStoreError("Canonical promotion checksum mismatch")
        _check_database(self.canonical_db, expected_state=latest["database_state"])
        receipt_path = Path(latest["_manifest_path"]).parent / "promotion.json"
        existing_receipt: dict[str, Any] | None = None
        if receipt_path.exists() or receipt_path.is_symlink():
            existing_receipt, _digest = _read_json_with_sidecar(
                receipt_path, label="SQLite canonical-promotion receipt"
            )
            if previous_canonical is None:
                previous_canonical = existing_receipt.get("previous_canonical")
        receipt = {
            "format": PROMOTION_FORMAT,
            "format_version": PROMOTION_VERSION,
            "status": "complete",
            "completed_utc": (
                existing_receipt.get("completed_utc")
                if existing_receipt is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            "generation": latest["generation"],
            "snapshot_manifest_sha256": latest["_manifest_sha256"],
            "database_sha256": source_sha,
            "canonical_database": str(self.canonical_db.resolve(strict=True)),
            "previous_canonical": previous_canonical,
        }
        if existing_receipt is not None:
            if existing_receipt != receipt:
                raise LocalStoreError("SQLite canonical-promotion receipt mismatch")
        else:
            _publish_json_with_sidecar(receipt_path, receipt)
        if backup.exists() or backup.is_symlink():
            _regular_file(backup, label="canonical pre-promotion backup").unlink()
            fsync_directory(backup.parent)
        return receipt
