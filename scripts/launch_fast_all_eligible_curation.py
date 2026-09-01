#!/usr/bin/env python3
"""Guard and launch the frozen generation-v2 all-eligible curator.

This is intentionally narrower than ``curate_corpus.py``.  It admits only the
production fast-canonical/all-eligible path, freezes one immutable launcher
identity per dataset generation, holds both a launcher singleton and the
preprocessor lock, and preserves the curator's exit status and durable logs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import curate_corpus as curate_corpus_module  # noqa: E402
import curation_local_store as local_store_module  # noqa: E402
from curate_corpus import (  # noqa: E402
    CURATION_DISK_SAFETY_DENOMINATOR,
    CURATION_DISK_SAFETY_NUMERATOR,
    CURATION_MINIMUM_FREE_BYTES,
    CURATION_MINIMUM_SIDECAR_LIMIT_BYTES,
    CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT,
    CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW,
    CURATION_STORAGE_PREFLIGHT_VERSION,
    CURATION_STORAGE_PROJECTION_BASIS,
    DEFAULT_DENYLIST,
    FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
    CurationBuilder,
    CurationError,
    detect_output_mount,
)
from curation_local_store import (  # noqa: E402
    LOCAL_JOURNAL_MODE,
    STORE_FORMAT,
    STORE_VERSION,
    LocalStoreError,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    storage_admission,
)


LAUNCHER_FORMAT = "fast-all-eligible-curation-launch-authority"
LAUNCHER_VERSION = 1
COMPLETION_FORMAT = "fast-all-eligible-curation-launch-completion"
COMPLETION_VERSION = 1
AUTHORITY_NAME = "FAST_ALL_ELIGIBLE_CURATION_AUTHORITY.json"
COMPLETION_NAME = "FAST_ALL_ELIGIBLE_CURATION_COMPLETE.json"
LOCK_NAME = ".fast-all-eligible-curation.launch.lock"
OUTPUT_RELATIVE = Path("curated/all-eligible-source-v2")
STAGING_RELATIVE = Path("staging/preprocess")
PREPROCESS_LOCK_RELATIVE = STAGING_RELATIVE / ".preprocess.lock"
BATCH_SIZE = 100_000
SNAPSHOT_INTERVAL_SECONDS = 86_400
SNAPSHOT_RETENTION = 2
MAXIMUM_CHILD_STDOUT_BYTES = 16 * 1024 * 1024


class FastCurationLaunchError(RuntimeError):
    """A production launch or immutable resume invariant was not proven."""


@dataclass(frozen=True)
class LaunchConfig:
    generation_root: Path
    staging_root: Path
    output: Path
    local_work_root: Path
    quotas: Path
    policy: Path
    benchmark_denylist: Path
    log_path: Path
    result_path: Path
    authority_path: Path
    completion_path: Path
    lock_path: Path
    preprocess_lock_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FastCurationLaunchError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FastCurationLaunchError(f"Invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FastCurationLaunchError(f"{label} must be a JSON object")
    return value


def _safe_json_file(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FastCurationLaunchError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    payload = path.read_bytes()
    return payload, _json_object_bytes(payload, label=label)


def _real_input_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise FastCurationLaunchError(f"{label} must be an absolute path")
    if path.is_symlink() or not path.is_file():
        raise FastCurationLaunchError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _real_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise FastCurationLaunchError(f"{label} must be an absolute path")
    if path.is_symlink() or not path.is_dir():
        raise FastCurationLaunchError(
            f"{label} must be a real existing directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_missing_or_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise FastCurationLaunchError(f"{label} must be an absolute path")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise FastCurationLaunchError(
                f"{label} must be a real directory when present: {path}"
            )
        return path.resolve(strict=True)
    parent = _real_directory(path.parent, label=f"{label} parent")
    if not path.name or path.name in (".", ".."):
        raise FastCurationLaunchError(f"{label} has an unsafe final component")
    return parent / path.name


def _canonical_output(root: Path) -> Path:
    output = root / OUTPUT_RELATIVE
    curated = output.parent
    if curated.exists() or curated.is_symlink():
        if curated.is_symlink() or not curated.is_dir():
            raise FastCurationLaunchError(
                f"Curation output parent must be a real directory: {curated}"
            )
        curated = curated.resolve(strict=True)
    else:
        curated = root / "curated"
    output = curated / output.name
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            raise FastCurationLaunchError(
                f"Curation output must be a real directory when present: {output}"
            )
        output = output.resolve(strict=True)
    return output


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_config(
    *,
    generation_root: Path,
    local_work_root: Path,
    quotas: Path,
    policy: Path,
    benchmark_denylist: Path,
    log_path: Path,
    result_path: Path | None,
) -> LaunchConfig:
    root = _real_directory(generation_root, label="generation root")
    staging = _real_directory(root / STAGING_RELATIVE, label="preprocess root")
    local = _canonical_missing_or_directory(
        local_work_root, label="local SQLite work root"
    )
    if _within(local, root):
        raise FastCurationLaunchError(
            "Local SQLite work root must be outside the durable generation"
        )
    quota_file = _real_input_file(quotas, label="quota configuration")
    policy_file = _real_input_file(policy, label="curation policy")
    denylist = _real_input_file(
        benchmark_denylist, label="benchmark denylist"
    )
    if not log_path.is_absolute():
        raise FastCurationLaunchError("Log path must be absolute")
    log_parent = _real_directory(log_path.parent, label="log parent")
    log = log_parent / log_path.name
    if not _within(log, root):
        raise FastCurationLaunchError(
            "Log path must be on the durable generation root"
        )
    if log.exists() or log.is_symlink():
        if log.is_symlink() or not log.is_file():
            raise FastCurationLaunchError(
                f"Log path must be a regular non-symlink file: {log}"
            )
    selected_result = (
        result_path
        if result_path is not None
        else log.with_name(f"{log.stem}.result.json")
    )
    if not selected_result.is_absolute():
        raise FastCurationLaunchError("Result path must be absolute")
    result_parent = _real_directory(selected_result.parent, label="result parent")
    result = result_parent / selected_result.name
    if not _within(result, root):
        raise FastCurationLaunchError(
            "Result path must be on the durable generation root"
        )
    if result == log:
        raise FastCurationLaunchError("Result and log paths must differ")
    if result.exists() or result.is_symlink():
        if result.is_symlink() or not result.is_file():
            raise FastCurationLaunchError(
                f"Result path must be a regular non-symlink file: {result}"
            )
    return LaunchConfig(
        generation_root=root,
        staging_root=staging,
        output=_canonical_output(root),
        local_work_root=local,
        quotas=quota_file,
        policy=policy_file,
        benchmark_denylist=denylist,
        log_path=log,
        result_path=result,
        authority_path=root / AUTHORITY_NAME,
        completion_path=root / COMPLETION_NAME,
        lock_path=root / LOCK_NAME,
        preprocess_lock_path=root / PREPROCESS_LOCK_RELATIVE,
    )


@contextmanager
def exclusive_lock(path: Path, *, label: str) -> Iterator[int]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise FastCurationLaunchError(f"Could not open {label}: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FastCurationLaunchError(f"{label} is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FastCurationLaunchError(f"Another process holds {label}: {path}") from exc
        yield descriptor
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_new(path: Path, payload: bytes) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise FastCurationLaunchError(
            f"Immutable evidence parent is unsafe: {path.parent}"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FastCurationLaunchError(
            f"Refusing to overwrite immutable evidence: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A failed O_EXCL publication is visibly incomplete and is never
        # silently reused. Leave it for explicit operator inspection.
        raise
    _fsync_directory(path.parent)


def _child_argv(config: LaunchConfig) -> list[str]:
    return [
        str(Path(sys.executable).resolve(strict=True)),
        "-u",
        str((SCRIPTS_ROOT / "curate_corpus.py").resolve(strict=True)),
        "--root",
        str(config.generation_root),
        "--staging-root",
        str(config.staging_root),
        "--policy",
        str(config.policy),
        "--quotas",
        str(config.quotas),
        "--benchmark-denylist",
        str(config.benchmark_denylist),
        "--output",
        str(config.output),
        "--sqlite-local-work-root",
        str(config.local_work_root),
        "--sqlite-snapshot-interval-seconds",
        str(SNAPSHOT_INTERVAL_SECONDS),
        "--sqlite-snapshot-retention",
        str(SNAPSHOT_RETENTION),
        "--defer-raw-archive-integrity-until-finalize",
        "--fast-all-eligible-handoff",
        "--batch-size",
        str(BATCH_SIZE),
    ]


def _expected_store_control(
    config: LaunchConfig, builder: CurationBuilder
) -> dict[str, Any]:
    canonical_db = config.output / ".work" / "curation.sqlite3"
    local_db = config.local_work_root / "curation.sqlite3"
    snapshot_root = config.output / ".work" / "sqlite-snapshots-v1"
    return {
        "format": STORE_FORMAT,
        "format_version": STORE_VERSION,
        "identity": builder.identity,
        "identity_sha256": canonical_sha256(builder.identity),
        "canonical_db": str(canonical_db.resolve(strict=False)),
        "local_db": str(local_db.resolve(strict=False)),
        "snapshot_root": str(snapshot_root.resolve(strict=False)),
        "canonical_journal_mode": builder.sqlite_runtime["journal_policy"][
            "selected_mode"
        ],
        "snapshot_retention": SNAPSHOT_RETENTION,
    }


def _local_resume_bytes(
    config: LaunchConfig,
    builder: CurationBuilder,
    *,
    authority_exists: bool,
) -> tuple[int, str]:
    local = config.local_work_root
    if not local.exists():
        return 0, "absent"
    if local.is_symlink() or not local.is_dir():
        raise FastCurationLaunchError("Local work root is unsafe")
    entries = sorted(local.iterdir(), key=lambda path: path.name)
    control = local / "LOCAL_SQLITE_STORE.json"
    if not control.exists() and not control.is_symlink():
        unexpected = [path for path in entries if path.name != "sqlite-tmp"]
        if unexpected:
            raise FastCurationLaunchError(
                "Local work root has state without launcher/store authority: "
                f"{unexpected[:3]}"
            )
        return 0, "empty"
    if not authority_exists:
        raise FastCurationLaunchError(
            "Local SQLite store exists without immutable launcher authority"
        )
    _raw, found = _safe_json_file(control, label="local SQLite control")
    if found != _expected_store_control(config, builder):
        raise FastCurationLaunchError(
            "Local SQLite store belongs to another curation identity"
        )
    reclaimable = 0
    database = local / "curation.sqlite3"
    for candidate in (
        database,
        Path(f"{database}-journal"),
        Path(f"{database}-wal"),
        Path(f"{database}-shm"),
    ):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise FastCurationLaunchError(
                f"Unsafe SQLite resume artifact: {candidate}"
            )
        reclaimable += candidate.stat().st_size
    return reclaimable, "authorized-local-store"


def _authority_identity(
    config: LaunchConfig,
    builder: CurationBuilder,
    *,
    expected_documents: int,
) -> dict[str, Any]:
    return {
        "contract_version": LAUNCHER_VERSION,
        "profile": FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
        "paths": {
            "generation_root": str(config.generation_root),
            "staging_root": str(config.staging_root),
            "output": str(config.output),
            "local_work_root": str(config.local_work_root),
            "log": str(config.log_path),
            "result": str(config.result_path),
        },
        "inputs": {
            "quotas": {
                "path": str(config.quotas),
                "sha256": file_sha256(config.quotas),
            },
            "policy": {
                "path": str(config.policy),
                "sha256": file_sha256(config.policy),
            },
            "benchmark_denylist": {
                "path": str(config.benchmark_denylist),
                "sha256": file_sha256(config.benchmark_denylist),
            },
        },
        "source": {
            "curation_identity": builder.identity,
            "curation_identity_sha256": canonical_sha256(builder.identity),
            "report_inventory_sha256": builder.inventory_sha,
            "report_count": len(builder.report_inventory),
            "expected_documents": expected_documents,
            "collection_completeness_sha256": canonical_sha256(
                builder.collection_completeness
            ),
        },
        "storage_contract": builder.storage_contract,
        "implementation": {
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
            "launcher_sha256": file_sha256(Path(__file__).resolve(strict=True)),
            "curator_sha256": file_sha256(
                Path(curate_corpus_module.__file__).resolve(strict=True)
            ),
            "local_store_sha256": file_sha256(
                Path(local_store_module.__file__).resolve(strict=True)
            ),
        },
        "child_argv": _child_argv(config),
    }


def _load_authority(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    _raw, payload = _safe_json_file(path, label="launcher authority")
    if set(payload) != {
        "format",
        "format_version",
        "status",
        "created_utc",
        "identity",
        "identity_sha256",
    }:
        raise FastCurationLaunchError("Launcher authority schema mismatch")
    identity = payload.get("identity")
    if (
        payload.get("format") != LAUNCHER_FORMAT
        or payload.get("format_version") != LAUNCHER_VERSION
        or payload.get("status") != "frozen"
        or not isinstance(payload.get("created_utc"), str)
        or not isinstance(identity, dict)
        or payload.get("identity_sha256") != canonical_sha256(identity)
    ):
        raise FastCurationLaunchError("Launcher authority is invalid")
    return payload


def _publish_authority(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "format": LAUNCHER_FORMAT,
        "format_version": LAUNCHER_VERSION,
        "status": "frozen",
        "created_utc": _utc_now(),
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
    }
    _atomic_new(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return payload


def build_preflight(config: LaunchConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform the complete read-only authority/capacity check.

    The caller holds the launcher and preprocess locks, closing the only
    cooperating preprocessing race between this check and child construction.
    """

    output_mount = detect_output_mount(config.output)
    local_mount = detect_output_mount(config.local_work_root)
    if output_mount.get("classification") != "network":
        raise FastCurationLaunchError(
            "Generation curation output is not on a positively identified "
            f"network filesystem: {output_mount}"
        )
    if local_mount.get("classification") != "local":
        raise FastCurationLaunchError(
            "SQLite work root is not on a positively identified local filesystem: "
            f"{local_mount}"
        )
    if (
        output_mount.get("mount_point") == local_mount.get("mount_point")
        or (
            output_mount.get("device") is not None
            and output_mount.get("device") == local_mount.get("device")
        )
    ):
        raise FastCurationLaunchError(
            "Durable generation and local SQLite work roots are not independent"
        )
    try:
        builder = CurationBuilder(
            root=config.generation_root,
            staging_root=config.staging_root,
            output=config.output,
            policy_path=config.policy,
            quota_path=config.quotas,
            denylist_path=config.benchmark_denylist,
            english_near_clusters=None,
            allow_missing_english_near_dedup=False,
            batch_size=BATCH_SIZE,
            sqlite_journal_mode="auto",
            sqlite_local_work_root=config.local_work_root,
            sqlite_snapshot_interval_seconds=SNAPSHOT_INTERVAL_SECONDS,
            sqlite_snapshot_retention=SNAPSHOT_RETENTION,
            defer_raw_archive_integrity_until_finalize=True,
            fast_all_eligible_handoff=True,
        )
    except (CurationError, LocalStoreError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise FastCurationLaunchError(
            f"Collection/preprocess authority did not close cleanly: {exc}"
        ) from exc
    if (
        builder.storage_contract.get("contract_version")
        != CURATION_STORAGE_PREFLIGHT_VERSION
        or builder.storage_contract.get("projection_basis")
        != CURATION_STORAGE_PROJECTION_BASIS
        or builder.storage_contract.get(
            "projected_additional_bytes_per_document"
        )
        != CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
        or builder.identity.get("fast_all_eligible_handoff")
        != FAST_ALL_ELIGIBLE_HANDOFF_PROFILE
    ):
        raise FastCurationLaunchError(
            "Curator no longer implements the frozen storage-v3 handoff profile"
        )
    expected_documents = sum(
        int(report[3]["documents"]) for report in builder.report_inventory
    )
    identity = _authority_identity(
        config, builder, expected_documents=expected_documents
    )
    existing_authority = _load_authority(config.authority_path)
    if existing_authority is None:
        if config.output.exists() or config.output.is_symlink():
            raise FastCurationLaunchError(
                "Curation output exists without immutable launcher authority"
            )
        if config.log_path.exists() or config.log_path.is_symlink():
            raise FastCurationLaunchError(
                "Durable log exists without immutable launcher authority"
            )
        if config.result_path.exists() or config.result_path.is_symlink():
            raise FastCurationLaunchError(
                "Result exists without immutable launcher authority"
            )
        if config.completion_path.exists() or config.completion_path.is_symlink():
            raise FastCurationLaunchError(
                "Completion receipt exists without immutable launcher authority"
            )
        resume_state = "fresh"
    else:
        if existing_authority["identity"] != identity:
            raise FastCurationLaunchError(
                "Existing launcher authority belongs to another input/path/code identity"
            )
        resume_state = "authorized-resume"
    reclaimable, local_state = _local_resume_bytes(
        config,
        builder,
        authority_exists=existing_authority is not None,
    )
    sidecar = max(
        CURATION_MINIMUM_SIDECAR_LIMIT_BYTES,
        BATCH_SIZE * CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW,
    )
    try:
        admission = storage_admission(
            local_root=config.local_work_root,
            filesystem_type=str(local_mount["filesystem_type"]),
            expected_documents=expected_documents,
            projected_bytes_per_document=(
                CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
            ),
            safety_numerator=CURATION_DISK_SAFETY_NUMERATOR,
            safety_denominator=CURATION_DISK_SAFETY_DENOMINATOR,
            transaction_sidecar_bytes=sidecar,
            minimum_free_bytes=CURATION_MINIMUM_FREE_BYTES,
            reclaimable_existing_bytes=reclaimable,
            projection_basis=CURATION_STORAGE_PROJECTION_BASIS,
        )
    except (LocalStoreError, OSError, ValueError) as exc:
        raise FastCurationLaunchError(
            f"Storage-v3 local admission failed: {exc}"
        ) from exc
    report = {
        "format": "fast-all-eligible-curation-preflight",
        "format_version": 1,
        "status": "pass",
        "profile": FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
        "resume_state": resume_state,
        "local_state": local_state,
        "launcher_identity_sha256": canonical_sha256(identity),
        "source_curation_identity_sha256": canonical_sha256(builder.identity),
        "expected_documents": expected_documents,
        "report_count": len(builder.report_inventory),
        "collection_completeness": builder.collection_completeness,
        "output_mount": output_mount,
        "local_mount": local_mount,
        "storage_admission": admission,
        "child_argv": identity["child_argv"],
        "paths": identity["paths"],
    }
    return report, identity


def _safe_open_log(config: LaunchConfig, *, authority_existed: bool) -> int:
    if not authority_existed and (config.log_path.exists() or config.log_path.is_symlink()):
        raise FastCurationLaunchError(
            "Refusing to append to a pre-existing unauthenticated log"
        )
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(config.log_path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise FastCurationLaunchError("Durable log is not a regular file")
    return descriptor


def _log_event(descriptor: int, event: str, **fields: Any) -> None:
    payload = {
        "launcher_event": event,
        "timestamp_utc": _utc_now(),
        **fields,
    }
    os.write(descriptor, b"[launcher] " + canonical_json_bytes(payload) + b"\n")
    os.fsync(descriptor)


def _validate_result(
    config: LaunchConfig,
    raw: bytes,
    *,
    expected_source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not raw or len(raw) > MAXIMUM_CHILD_STDOUT_BYTES:
        raise FastCurationLaunchError("Curator stdout is empty or exceeds its bound")
    result = _json_object_bytes(raw, label="curator result")
    authority = result.get("authority")
    snapshot = result.get("source_snapshot")
    if (
        result.get("complete") is not True
        or result.get("phase") != "canonicalized"
        or result.get("ready_for_all_eligible_publication") is not True
        or result.get("execution_profile") != FAST_ALL_ELIGIBLE_HANDOFF_PROFILE
        or not isinstance(authority, dict)
        or authority.get("selected_documents") != 0
        or authority.get("decision_archives") != 0
        or authority.get("quota_subphases") != 0
        or authority.get("fuzzy_near_map_rows") != 0
        or not isinstance(snapshot, dict)
    ):
        raise FastCurationLaunchError(
            "Curator did not return the frozen all-eligible completion contract"
        )
    try:
        generation = int(snapshot["generation"])
        manifest_path = Path(str(snapshot["manifest_path"]))
        manifest_sha = str(snapshot["manifest_sha256"])
        database = snapshot["database"]
        checkpoint = snapshot["checkpoint"]
    except (KeyError, TypeError, ValueError) as exc:
        raise FastCurationLaunchError("Curator snapshot descriptor is malformed") from exc
    expected_root = config.output / ".work" / "sqlite-snapshots-v1"
    if (
        generation < 1
        or not manifest_path.is_absolute()
        or manifest_path.name != "manifest.json"
        or manifest_path.parent.parent.resolve(strict=False)
        != expected_root.resolve(strict=False)
    ):
        raise FastCurationLaunchError("Curator snapshot path is outside its output authority")
    manifest_raw, manifest = _safe_json_file(
        manifest_path, label="curator snapshot manifest"
    )
    if _sha256_bytes(manifest_raw) != manifest_sha:
        raise FastCurationLaunchError("Curator snapshot manifest checksum mismatch")
    sidecar = manifest_path.with_name("manifest.json.sha256")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != (
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    ):
        raise FastCurationLaunchError("Curator snapshot manifest sidecar mismatch")
    if (
        manifest.get("generation") != generation
        or manifest.get("identity") != expected_source_identity
        or manifest.get("identity_sha256")
        != canonical_sha256(expected_source_identity)
        or manifest.get("database") != database
        or not isinstance(manifest.get("authority_artifacts"), dict)
        or manifest["authority_artifacts"].get("CHECKPOINT.json") != checkpoint
    ):
        raise FastCurationLaunchError("Curator result/snapshot authority mismatch")
    for label, descriptor, expected_name, hash_payload in (
        ("snapshot database", database, "curation.sqlite3", False),
        ("snapshot checkpoint", checkpoint, "CHECKPOINT.json", True),
    ):
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise FastCurationLaunchError(f"Invalid {label} descriptor")
        artifact = Path(str(descriptor["path"]))
        if (
            artifact.parent != manifest_path.parent
            or artifact.name != expected_name
            or artifact.is_symlink()
            or not artifact.is_file()
            or descriptor["bytes"] != artifact.stat().st_size
        ):
            raise FastCurationLaunchError(f"Invalid {label} path/size")
        digest = str(descriptor["sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise FastCurationLaunchError(f"Invalid {label} checksum")
        if hash_payload and file_sha256(artifact) != digest:
            raise FastCurationLaunchError(f"{label} checksum mismatch")
    evidence = {
        "generation": generation,
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": manifest_sha,
        "database_bytes": int(database["bytes"]),
        "database_sha256": str(database["sha256"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
    }
    return result, evidence


def _expected_completion(
    config: LaunchConfig,
    *,
    identity_sha256: str,
    result_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": COMPLETION_FORMAT,
        "format_version": COMPLETION_VERSION,
        "status": "complete",
        "launcher_identity_sha256": identity_sha256,
        "result": {
            "path": str(config.result_path),
            "sha256": result_sha256,
        },
        "snapshot": dict(snapshot),
    }


def _load_or_publish_completion(
    config: LaunchConfig,
    *,
    identity_sha256: str,
    result_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _expected_completion(
        config,
        identity_sha256=identity_sha256,
        result_sha256=result_sha256,
        snapshot=snapshot,
    )
    if config.completion_path.exists() or config.completion_path.is_symlink():
        _raw, found = _safe_json_file(
            config.completion_path, label="curation completion receipt"
        )
        if set(found) != set(expected) | {"created_utc"}:
            raise FastCurationLaunchError("Completion receipt schema mismatch")
        projected = {key: value for key, value in found.items() if key != "created_utc"}
        if projected != expected or not isinstance(found.get("created_utc"), str):
            raise FastCurationLaunchError("Completion receipt authority mismatch")
        return found
    published = {**expected, "created_utc": _utc_now()}
    _atomic_new(
        config.completion_path,
        json.dumps(published, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return published


def _completed_result_if_any(
    config: LaunchConfig,
    *,
    identity_sha256: str,
    expected_source_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_exists = config.result_path.exists() or config.result_path.is_symlink()
    completion_exists = (
        config.completion_path.exists() or config.completion_path.is_symlink()
    )
    if not result_exists:
        if completion_exists:
            raise FastCurationLaunchError(
                "Completion receipt exists but the immutable result is missing"
            )
        return None
    raw, _payload = _safe_json_file(config.result_path, label="curation result")
    result, snapshot = _validate_result(
        config,
        raw,
        expected_source_identity=expected_source_identity,
    )
    _load_or_publish_completion(
        config,
        identity_sha256=identity_sha256,
        result_sha256=_sha256_bytes(raw),
        snapshot=snapshot,
    )
    return result


def launch(
    config: LaunchConfig,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> tuple[int, dict[str, Any]]:
    with exclusive_lock(config.lock_path, label="curation launcher lock"):
        with exclusive_lock(
            config.preprocess_lock_path, label="preprocess generation lock"
        ):
            preflight, identity = build_preflight(config)
            authority = _load_authority(config.authority_path)
            authority_existed = authority is not None
            if authority is None:
                authority = _publish_authority(config.authority_path, identity)
            identity_sha = str(authority["identity_sha256"])
            complete = _completed_result_if_any(
                config,
                identity_sha256=identity_sha,
                expected_source_identity=identity["source"]["curation_identity"],
            )
            if complete is not None:
                return 0, {
                    "complete": True,
                    "already_complete": True,
                    "preflight": preflight,
                    "result": complete,
                }
            log_descriptor = _safe_open_log(
                config, authority_existed=authority_existed
            )
            try:
                _log_event(
                    log_descriptor,
                    "child_start",
                    launcher_identity_sha256=identity_sha,
                    argv=identity["child_argv"],
                    resume_state=preflight["resume_state"],
                )
                process = popen_factory(
                    identity["child_argv"],
                    cwd=str(PROJECT_ROOT),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=log_descriptor,
                    close_fds=True,
                )
                previous_handlers: dict[int, Any] = {}

                def forward(signum: int, _frame: Any) -> None:
                    try:
                        _log_event(log_descriptor, "signal_forwarded", signal=signum)
                    except OSError:
                        # Signal delivery is more important than telemetry when
                        # a full/unavailable log volume is itself the failure.
                        pass
                    process.send_signal(signum)

                handled_signals = (signal.SIGINT, signal.SIGTERM)
                try:
                    for signum in handled_signals:
                        previous_handlers[signum] = signal.getsignal(signum)
                        signal.signal(signum, forward)
                    stdout, _unused = process.communicate()
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                returncode = int(process.returncode)
                if stdout:
                    os.write(log_descriptor, b"[curator-stdout]\n" + stdout)
                    if not stdout.endswith(b"\n"):
                        os.write(log_descriptor, b"\n")
                _log_event(
                    log_descriptor,
                    "child_exit",
                    child_returncode=returncode,
                )
                if returncode != 0:
                    return returncode, {
                        "complete": False,
                        "child_returncode": returncode,
                        "preflight": preflight,
                        "log_path": str(config.log_path),
                    }
                result, snapshot = _validate_result(
                    config,
                    stdout,
                    expected_source_identity=identity["source"][
                        "curation_identity"
                    ],
                )
                _atomic_new(config.result_path, stdout)
                receipt = _load_or_publish_completion(
                    config,
                    identity_sha256=identity_sha,
                    result_sha256=_sha256_bytes(stdout),
                    snapshot=snapshot,
                )
                _log_event(
                    log_descriptor,
                    "completion_published",
                    result_path=str(config.result_path),
                    result_sha256=_sha256_bytes(stdout),
                    snapshot_manifest_sha256=snapshot["manifest_sha256"],
                )
                return 0, {
                    "complete": True,
                    "already_complete": False,
                    "preflight": preflight,
                    "completion": receipt,
                    "result": result,
                }
            finally:
                os.fsync(log_descriptor)
                os.close(log_descriptor)


def preflight_only(config: LaunchConfig) -> dict[str, Any]:
    with exclusive_lock(config.lock_path, label="curation launcher lock"):
        with exclusive_lock(
            config.preprocess_lock_path, label="preprocess generation lock"
        ):
            report, _identity = build_preflight(config)
            return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--local-work-root", type=Path, required=True)
    parser.add_argument("--quotas", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--benchmark-denylist", type=Path, default=DEFAULT_DENYLIST
    )
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate closure, identity, topology, locks, and capacity without launching",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(
            generation_root=args.generation_root,
            local_work_root=args.local_work_root,
            quotas=args.quotas,
            policy=args.policy,
            benchmark_denylist=args.benchmark_denylist,
            log_path=args.log_path,
            result_path=args.result_path,
        )
        if args.preflight_only:
            print(json.dumps(preflight_only(config), indent=2, sort_keys=True))
            return 0
        code, report = launch(config)
        stream = sys.stdout if code == 0 else sys.stderr
        print(json.dumps(report, indent=2, sort_keys=True), file=stream)
        if code < 0:
            return 128 + abs(code)
        return code
    except (
        FastCurationLaunchError,
        CurationError,
        LocalStoreError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Fast curation launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
