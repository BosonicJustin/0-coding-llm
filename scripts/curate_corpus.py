#!/usr/bin/env python3
"""Build an immutable, leakage-safe corpus selection manifest.

This is the offline boundary between the streaming fingerprint audit and raw
text tokenization/packing.  It reconciles collection quota records and
completion markers with finalized raw archive paths, integrity reports, and
compressed fingerprint shards. Raw archives are never modified. English
archives are hashed as opaque bytes to validate the near-dedup consumer
contract, but their tar payloads are not parsed here.

The implementation is intentionally restartable.  A SQLite state database is
the authoritative journal, archive rows and all later bulk writes commit in
bounded batches with durable cursors, decision shards are atomically published,
and the JSON checkpoint and journal are deterministic projections of committed
database state.
"""

from __future__ import annotations

import argparse
import array
import collections
import fcntl
import hashlib
import heapq
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import zstandard

from benchmark_guard import BenchmarkGuard
from curation_policy import (
    ALL_BUCKETS,
    CODE_BUCKETS,
    DEFAULT_POLICY,
    FAST_CANONICAL_PROFILE,
    canonical_sha256,
    curation_profile,
    load_policy,
    validate_trusted_stack_source,
)
from preprocess_raw_stream import (
    BUCKET_PATHS,
    FINGERPRINT_VERSION,
    POLICY_SHA256,
    quota_record_bucket,
)
from quota_tracker import RECORD_VERSION as QUOTA_RECORD_VERSION, record_path


FORMAT_VERSION = 5
FAST_CANONICAL_FORMAT_VERSION = 6
DB_VERSION = 4
DECISION_RECORD_VERSION = 1
COMPLETENESS_FORMAT_VERSION = 1
ENGLISH_NEAR_CONTRACT_VERSION = 2
DEFAULT_ROOT = Path("/workspace/dataset")
DEFAULT_QUOTAS = Path(__file__).resolve().parents[1] / "configs" / "data_quotas.json"
DEFAULT_DENYLIST = Path(__file__).resolve().parents[1] / "configs" / "mbpp_denylist.json"
DEFAULT_ENGLISH_NEAR_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "english_near_dedup.json"
)
ENGLISH_NEAR_BUILDER = Path(__file__).resolve().parent / "build_english_near_clusters.py"
ENGLISH_NEAR_CALIBRATION = (
    Path(__file__).resolve().parent / "calibrate_english_near_dedup.py"
)
DEFAULT_ENGLISH_NEAR_CALIBRATION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "english_near_dedup_calibration.json"
)
ENGLISH_NEAR_ALGORITHM = "raw-text-doph-lsh-plus-full-shingle-jaccard-v1"
ENGLISH_BUCKETS = ("fineweb_edu", "wikipedia")
BUCKET_CATEGORY = {
    "python": "python",
    "other_code": "other_code",
    "fineweb_edu": "english",
    "wikipedia": "english",
}
SPLITS = ("train", "validation", "test")
HEX64 = frozenset("0123456789abcdef")
ARCHIVE_NAME = re.compile(r"part-(\d{6})\.tar\.zst")
QUOTA_SHARD_SUFFIX = re.compile(r"-(\d{6})$")
COMPLETION_MARKERS = (
    ("COLLECTION_COMPLETE.json", ("python", "other_code")),
    ("ENGLISH_FINEWEB_EDU_COMPLETE.json", ("fineweb_edu",)),
    ("ENGLISH_WIKIPEDIA_COMPLETE.json", ("wikipedia",)),
)
SQLITE_JOURNAL_POLICY_VERSION = 1
SQLITE_TEMP_STORE = "FILE"
QUOTA_SELECTION_INDEX = "documents_selection_v2"
QUOTA_CANDIDATE_SQL = f"""
    SELECT d.doc_id, d.tokens, d.selection_rank
    FROM documents AS d INDEXED BY {QUOTA_SELECTION_INDEX}
    CROSS JOIN groups AS g
    WHERE d.bucket=?
      AND g.group_id=d.source_group
      AND g.split=?
      AND NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
    ORDER BY d.selection_rank, d.doc_id
"""
SQLITE_JOURNAL_MODES = frozenset(("auto", "delete", "wal"))
CURATION_LEASE_VERSION = 1
CURATION_LEASE_FILE = ".curation.cross-client-lease.json"
CURATION_PROGRESS_VERSION = 1
CURATION_STORAGE_PREFLIGHT_VERSION = 2
# A representative 200k-row schema probe reached ~1,438 bytes/document before
# decision shards, temp spill, sidecars, fragmentation, and real path lengths.
# Reserve 3 KiB/document and then apply the frozen 2x safety factor.  At the
# planned ~51M-document scale this makes the executable gate slightly stronger
# than the 300 GB local-NVMe runbook requirement instead of a false ~106 GB GO.
CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT = 3_072
CURATION_DISK_SAFETY_NUMERATOR = 2
CURATION_DISK_SAFETY_DENOMINATOR = 1
CURATION_MINIMUM_FREE_BYTES = 2 * 1_000_000_000
CURATION_MINIMUM_SIDECAR_LIMIT_BYTES = 256 * 1024 * 1024
CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW = 64 * 1024
LOCAL_FILESYSTEMS = frozenset(
    (
        "apfs",
        "bcachefs",
        "btrfs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "overlay",
        "ramfs",
        "tmpfs",
        "ufs",
        "xfs",
        "zfs",
    )
)
NETWORK_FILESYSTEMS = frozenset(
    (
        "9p",
        "afs",
        "beegfs",
        "ceph",
        "cifs",
        "davfs",
        "fuse.ceph",
        "fuse.gcsfuse",
        "fuse.glusterfs",
        "fuse.s3fs",
        "fuse.sshfs",
        "gcsfuse",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "s3fs",
        "smbfs",
        "sshfs",
        "virtiofs",
    )
)
MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


class CurationError(RuntimeError):
    """Raised when an input identity or corpus invariant is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_object_without_duplicate_keys(raw: bytes, label: str) -> dict[str, Any]:
    """Parse an evidence object while rejecting duplicate keys at every depth."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CurationError(f"Duplicate {label} JSON key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationError(f"Invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise CurationError(f"{label} must be a JSON object")
    return value


def _decode_mount_field(value: str) -> str:
    return MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _mount_classification(filesystem_type: str, options: Iterable[str]) -> str:
    normalized_type = filesystem_type.casefold()
    normalized_options = {option.casefold() for option in options}
    if (
        normalized_type in NETWORK_FILESYSTEMS
        or normalized_type.startswith("nfs")
        or "_netdev" in normalized_options
        or "remote" in normalized_options
    ):
        return "network"
    if normalized_type in LOCAL_FILESYSTEMS or "local" in normalized_options:
        return "local"
    return "unknown"


def _resolved_target(path: Path) -> Path:
    return Path(os.path.abspath(path)).resolve(strict=False)


def _nearest_existing(path: Path) -> Path:
    candidate = _resolved_target(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _select_mount_entry(
    target: Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for entry in entries:
        mount_point = Path(str(entry["mount_point"]))
        if target == mount_point or mount_point in target.parents:
            candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda entry: len(Path(str(entry["mount_point"])).parts))


def _linux_mount_entries(payload: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in payload.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) < separator + 4:
            continue
        options = set(fields[5].split(","))
        options.update(fields[separator + 3].split(","))
        filesystem_type = fields[separator + 1]
        entries.append(
            {
                "mount_point": _decode_mount_field(fields[4]),
                "filesystem_type": filesystem_type,
                "source": _decode_mount_field(fields[separator + 2]),
                "device": fields[2],
                "options": sorted(option for option in options if option),
                "classification": _mount_classification(filesystem_type, options),
            }
        )
    return entries


def _darwin_mount_entries(payload: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if " on " not in line or not line.endswith(")"):
            continue
        prefix, separator, raw_options = line.rpartition(" (")
        if not separator or " on " not in prefix:
            continue
        source, mount_point = prefix.split(" on ", 1)
        options = [value.strip() for value in raw_options[:-1].split(",") if value.strip()]
        if not options:
            continue
        filesystem_type = options[0]
        entries.append(
            {
                "mount_point": _decode_mount_field(mount_point),
                "filesystem_type": filesystem_type,
                "source": _decode_mount_field(source),
                "device": None,
                "options": sorted(set(options[1:])),
                "classification": _mount_classification(filesystem_type, options[1:]),
            }
        )
    return entries


def detect_output_mount(
    output: Path,
    *,
    system: str | None = None,
    linux_mountinfo: str | None = None,
    darwin_mounts: str | None = None,
    darwin_df: str | None = None,
) -> dict[str, Any]:
    """Return stable mount evidence used to choose safe SQLite journaling.

    The injectable payloads are intentionally keyword-only and exist for unit
    tests. Production calls read the kernel mount table for the current host.
    Unknown filesystems remain usable with rollback journaling but never WAL.
    """
    target = _resolved_target(output)
    operating_system = system or platform.system()
    detector: str
    entries: list[dict[str, Any]] = []
    if operating_system == "Linux":
        detector = "linux-proc-self-mountinfo"
        if linux_mountinfo is None:
            try:
                linux_mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
            except OSError:
                linux_mountinfo = ""
        entries = _linux_mount_entries(linux_mountinfo)
    elif operating_system == "Darwin":
        detector = "darwin-mount"
        if darwin_mounts is None:
            try:
                completed = subprocess.run(
                    ["mount"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                darwin_mounts = completed.stdout
            except (OSError, subprocess.SubprocessError):
                darwin_mounts = ""
        entries = _darwin_mount_entries(darwin_mounts)
    else:
        detector = f"unsupported-{operating_system.casefold() or 'unknown'}"

    selected: dict[str, Any] | None = None
    if operating_system == "Darwin":
        if darwin_df is None:
            try:
                completed = subprocess.run(
                    ["df", "-P", str(_nearest_existing(output))],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                darwin_df = completed.stdout
            except (OSError, subprocess.SubprocessError):
                darwin_df = ""
        data_lines = [line for line in darwin_df.splitlines()[1:] if line.strip()]
        if data_lines:
            fields = data_lines[-1].split()
            if len(fields) >= 2:
                df_source = _decode_mount_field(fields[0])
                df_mount = _decode_mount_field(fields[-1])
                exact = [
                    entry
                    for entry in entries
                    if entry["source"] == df_source
                    and entry["mount_point"] == df_mount
                ]
                if not exact:
                    exact = [entry for entry in entries if entry["source"] == df_source]
                if exact:
                    selected = max(
                        exact,
                        key=lambda entry: len(Path(str(entry["mount_point"])).parts),
                    )
    if selected is None:
        selected = _select_mount_entry(target, entries)
    if selected is None:
        probe = _nearest_existing(output)
        try:
            device: str | None = str(probe.stat().st_dev)
        except OSError:
            device = None
        return {
            "detector": detector,
            "mount_point": None,
            "filesystem_type": "unknown",
            "source": None,
            "device": device,
            "options": [],
            "classification": "unknown",
        }
    return {"detector": detector, **selected}


def select_sqlite_journal_policy(
    mount_evidence: dict[str, Any],
    requested_mode: str,
) -> dict[str, Any]:
    if requested_mode not in SQLITE_JOURNAL_MODES:
        raise CurationError(
            "sqlite_journal_mode must be exactly one of auto, delete, or wal"
        )
    classification = mount_evidence.get("classification")
    if classification not in ("local", "network", "unknown"):
        raise CurationError("Invalid output mount classification")
    if requested_mode == "wal" and classification != "local":
        raise CurationError(
            "SQLite WAL is permitted only on a positively identified local filesystem; "
            f"output mount classification is {classification!r}"
        )
    selected_mode = (
        "wal"
        if requested_mode == "wal"
        or (requested_mode == "auto" and classification == "local")
        else "delete"
    )
    return {
        "policy_version": SQLITE_JOURNAL_POLICY_VERSION,
        "requested_mode": requested_mode,
        "selected_mode": selected_mode,
        "mount": mount_evidence,
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def validate_cross_client_lease_owner(
    value: Any, *, output: Path
) -> dict[str, Any]:
    expected_keys = {
        "lease_version",
        "hostname",
        "pid",
        "started_unix_ns",
        "output",
        "owner_token",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CurationError("Cross-client curation lease owner is malformed")
    if (
        value.get("lease_version") != CURATION_LEASE_VERSION
        or not isinstance(value.get("hostname"), str)
        or not value["hostname"]
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("pid"), bool)
        or value["pid"] < 1
        or not isinstance(value.get("started_unix_ns"), int)
        or isinstance(value.get("started_unix_ns"), bool)
        or value["started_unix_ns"] < 1
        or value.get("output") != str(output.resolve())
        or not isinstance(value.get("owner_token"), str)
        or len(value["owner_token"]) != 64
        or any(character not in HEX64 for character in value["owner_token"])
    ):
        raise CurationError("Cross-client curation lease owner is malformed")
    authority = {key: item for key, item in value.items() if key != "owner_token"}
    if hashlib.sha256(canonical_json_bytes(authority)).hexdigest() != value["owner_token"]:
        raise CurationError("Cross-client curation lease owner token is invalid")
    return dict(value)


def publish_stale_recovery_claim(
    output: Path, previous: dict[str, Any]
) -> Path:
    """Permanently claim one stale owner token with no replacement."""
    claim_root = output / ".curation-recovery-claims"
    claim_root.mkdir(mode=0o700, exist_ok=True)
    if claim_root.is_symlink() or not claim_root.is_dir():
        raise CurationError("Curation recovery-claim path must be a real directory")
    token = previous["owner_token"]
    claim = claim_root / f"{token}.json"
    payload = {
        "claim_version": 1,
        "stale_owner_token": token,
        "recovery_hostname": platform.node(),
        "recovery_pid": os.getpid(),
        "started_unix_ns": time.time_ns(),
        "output": str(output.resolve()),
    }
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{token}.", suffix=".candidate", dir=claim_root
    )
    candidate = Path(candidate_name)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(candidate, claim, follow_symlinks=False)
        except FileExistsError as error:
            raise CurationError(
                "The requested stale owner token already has a permanent "
                f"recovery claim: {claim}"
            ) from error
        if claim.read_bytes() != candidate.read_bytes():
            raise CurationError("Published stale-recovery claim changed")
        fsync_directory(claim_root)
        candidate.unlink()
        fsync_directory(claim_root)
        return claim
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            candidate.unlink()
        except OSError:
            pass
        raise


def _acquire_durable_cross_client_lease(
    output: Path, *, recover_stale_owner_token: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Acquire an NFS-visible singleton using an atomic hard-link publish.

    The complete owner record is written and fsynced under a unique name before
    ``link(2)`` publishes it at the canonical path with no replacement.  Thus a
    crash can leave either no canonical lease or a complete one, never a
    partially written/ownerless lease.  A hard process/host failure
    intentionally leaves the canonical file behind.  Resuming then requires
    an explicit recovery flag, which atomically archives rather than deletes
    the prior owner record after refusing a live same-host PID.
    """
    lease = output / CURATION_LEASE_FILE
    if recover_stale_owner_token is not None:
        if (
            len(recover_stale_owner_token) != 64
            or any(character not in HEX64 for character in recover_stale_owner_token)
        ):
            raise CurationError(
                "--recover-stale-cross-client-lease requires the exact lowercase "
                "SHA-256 owner_token from the stale lease"
            )
        if not lease.exists() or lease.is_symlink() or not lease.is_file():
            raise CurationError("No safe stale cross-client curation lease to recover")
        previous = validate_cross_client_lease_owner(
            json.loads(lease.read_text(encoding="utf-8")), output=output
        )
        if previous["owner_token"] != recover_stale_owner_token:
            raise CurationError(
                "Stale cross-client curation lease owner_token does not match "
                "the explicitly requested token"
            )
        if previous["hostname"] == platform.node():
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                raise CurationError(
                    "Cannot prove the same-host stale lease owner has exited"
                ) from error
            else:
                raise CurationError(
                    "Refusing to recover a cross-client lease held by a live process"
                )
        publish_stale_recovery_claim(output, previous)
        # The recovery claim is the permanent no-replace fence for this stale
        # generation. Re-open the canonical record immediately before moving
        # it so no changed/live generation can be archived by this request.
        current = validate_cross_client_lease_owner(
            json.loads(lease.read_text(encoding="utf-8")), output=output
        )
        if current != previous:
            raise CurationError(
                "Cross-client curation lease changed after stale-recovery claim"
            )
        archive_root = output / ".stale-curation-leases"
        archive_root.mkdir(mode=0o700, exist_ok=True)
        archive = archive_root / (
            f"lease-{int(time.time())}-{previous['pid']}-{time.time_ns()}.json"
        )
        os.replace(lease, archive)
        fsync_directory(archive_root)
        fsync_directory(output)

    started_ns = time.time_ns()
    owner = {
        "lease_version": CURATION_LEASE_VERSION,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "started_unix_ns": started_ns,
        "output": str(output.resolve()),
    }
    owner["owner_token"] = hashlib.sha256(
        canonical_json_bytes(owner)
    ).hexdigest()
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=".curation.lease-candidate-", suffix=".json", dir=output
    )
    candidate = Path(candidate_name)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1  # fd ownership transferred to ``handle`` immediately.
        with handle:
            handle.write(
                json.dumps(owner, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(candidate, lease, follow_symlinks=False)
        except FileExistsError as error:
            raise CurationError(
                f"Another client may own the durable curation lease: {lease}"
            ) from error
        if lease.read_bytes() != candidate.read_bytes():
            raise CurationError("Published cross-client curation lease changed")
        fsync_directory(output)
        candidate.unlink()
        fsync_directory(output)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            candidate.unlink()
        except OSError:
            pass
        raise
    return lease, owner


def acquire_cross_client_lease(
    output: Path, *, recover_stale_owner_token: str | None = None
) -> tuple[Path, dict[str, Any], Any]:
    """Hold an NFS-visible advisory lock across recovery and the full run."""
    lock_path = output / ".curation.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CurationError("Cross-client curation lock must be a regular file")
    lock = os.fdopen(descriptor, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise CurationError(
            f"Another curation process holds {lock_path}"
        ) from error
    try:
        lease, owner = _acquire_durable_cross_client_lease(
            output,
            recover_stale_owner_token=recover_stale_owner_token,
        )
    except BaseException:
        lock.close()
        raise
    return lease, owner, lock


def release_cross_client_lease(
    output: Path, lease: Path, owner: dict[str, Any], lock: Any
) -> None:
    try:
        if lease.parent != output or lease.name != CURATION_LEASE_FILE:
            raise CurationError("Refusing to release a non-canonical curation lease")
        try:
            current = validate_cross_client_lease_owner(
                json.loads(lease.read_text(encoding="utf-8")), output=output
            )
        except (OSError, json.JSONDecodeError) as error:
            raise CurationError("Cross-client curation lease owner changed") from error
        if current != owner:
            raise CurationError("Cross-client curation lease ownership changed")

        # Moving the whole canonical lease away is the release linearization
        # point. The advisory lock remains held until publication is durable.
        archive_root = output / ".released-curation-leases"
        archive_root.mkdir(mode=0o700, exist_ok=True)
        archive = archive_root / (
            f"lease-{int(time.time())}-{owner['pid']}-{time.time_ns()}.json"
        )
        os.replace(lease, archive)
        fsync_directory(archive_root)
        fsync_directory(output)
        try:
            archive.unlink()
            fsync_directory(archive_root)
        except OSError:
            # The canonical lease is already released. Retaining the uniquely
            # named record is safe forensic evidence after an NFS error.
            pass
    finally:
        lock.close()


def parse_hex_digest(value: Any, field: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX64 for character in value)
    ):
        raise CurationError(f"{field} must be a lowercase SHA-256")
    return bytes.fromhex(value)


def stable_digest(namespace: str, value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(namespace.encode("utf-8") + b"\0" + raw).digest()


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        text = __import__("io").TextIOWrapper(reader, encoding="utf-8")
        try:
            for line_number, line in enumerate(text, 1):
                if not line.strip():
                    raise CurationError(f"Blank JSONL row in {path} at line {line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CurationError(f"Non-object JSONL row in {path} at line {line_number}")
                yield row
        finally:
            text.close()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".zst":
        yield from iter_jsonl_zst(path)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CurationError(f"Blank JSONL row in {path} at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CurationError(f"Non-object JSONL row in {path} at line {line_number}")
            yield row


def load_final_quotas(path: Path) -> dict[tuple[str, str], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("quotas"), list):
        raise CurationError(f"Unsupported quota config: {path}")
    result: dict[tuple[str, str], int] = {}
    for row in payload["quotas"]:
        if row.get("phase") != "final":
            continue
        split = row.get("split")
        category = row.get("category")
        target = row.get("target")
        if split not in SPLITS or category not in ("python", "other_code", "english"):
            raise CurationError(f"Invalid final quota row: {row!r}")
        if row.get("token_field") != "exact_tokens" or not isinstance(target, int) or target <= 0:
            raise CurationError(f"Invalid final exact-token target: {row!r}")
        key = (str(split), str(category))
        if key in result:
            raise CurationError(f"Duplicate final quota {key}")
        result[key] = target
    expected = {(split, category) for split in SPLITS for category in BUCKET_CATEGORY.values()}
    if set(result) != expected:
        raise CurationError(f"Final quotas must cover exactly {sorted(expected)}")
    return result


def load_collection_targets(path: Path) -> dict[str, int]:
    """Load the four source-bucket acquisition targets from the quota config."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("quotas"), list):
        raise CurationError(f"Unsupported quota config: {path}")
    targets: dict[str, int] = {}
    aggregate_english: int | None = None
    for row in payload["quotas"]:
        if row.get("phase") != "collection":
            continue
        target = row.get("target")
        if (
            row.get("token_field") != "exact_tokens"
            or not isinstance(target, int)
            or isinstance(target, bool)
            or target <= 0
        ):
            raise CurationError(f"Invalid collection exact-token target: {row!r}")
        category = row.get("category")
        language_group = row.get("language_group")
        if category in ("python", "other_code") and language_group is None:
            bucket = str(category)
        elif category == "english" and language_group in ("fineweb_edu", "wikipedia"):
            bucket = str(language_group)
        elif category == "english" and language_group is None:
            if aggregate_english is not None:
                raise CurationError("Duplicate aggregate English collection target")
            aggregate_english = target
            continue
        else:
            raise CurationError(f"Invalid collection quota row: {row!r}")
        if bucket in targets:
            raise CurationError(f"Duplicate collection target for {bucket}")
        targets[bucket] = target
    if set(targets) != set(ALL_BUCKETS):
        raise CurationError(
            f"Collection targets must cover exactly {sorted(ALL_BUCKETS)}"
        )
    if aggregate_english is not None and aggregate_english != (
        targets["fineweb_edu"] + targets["wikipedia"]
    ):
        raise CurationError(
            "Aggregate English collection target does not equal its source targets"
        )
    return {bucket: targets[bucket] for bucket in sorted(ALL_BUCKETS)}


def expected_source_strings(root: Path, policy: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate source/tokenizer artifacts and return the only accepted sources."""
    stack = validate_trusted_stack_source(root, policy)
    tokenizer_dir = root / "tokenizer" / "starcoder2"
    tokenizer_path = tokenizer_dir / "TOKENIZER_MANIFEST.json"
    if not tokenizer_path.is_file():
        raise CurationError(f"Missing tokenizer manifest: {tokenizer_path}")
    tokenizer_raw = tokenizer_path.read_bytes()
    tokenizer = json.loads(tokenizer_raw)
    if tokenizer.get("manifest_version") != 1:
        raise CurationError("Unsupported tokenizer manifest version")
    revision = tokenizer.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise CurationError("Tokenizer manifest has an invalid resolved revision")
    if tokenizer.get("validation", {}).get("vocab_size") != 49_152:
        raise CurationError("Tokenizer manifest does not validate the 49,152-token vocabulary")
    files = tokenizer.get("files")
    if not isinstance(files, dict) or not files:
        raise CurationError("Tokenizer manifest has no checksummed files")
    for name, descriptor in sorted(files.items()):
        if not isinstance(name, str) or Path(name).name != name or not isinstance(descriptor, dict):
            raise CurationError("Unsafe tokenizer file entry")
        artifact = tokenizer_dir / name
        if not artifact.is_file():
            raise CurationError(f"Missing tokenizer artifact: {artifact}")
        if artifact.stat().st_size != int(descriptor.get("bytes", -1)):
            raise CurationError(f"Tokenizer size mismatch: {artifact}")
        if file_sha256(artifact) != descriptor.get("sha256"):
            raise CurationError(f"Tokenizer checksum mismatch: {artifact}")

    sources = {
        "python": f"{stack['repo_id']}@{stack['resolved_revision']}",
        "other_code": f"{stack['repo_id']}@{stack['resolved_revision']}",
    }
    source_manifests: dict[str, Any] = {
        "STACK_V3_SOURCE.json": {
            "sha256": file_sha256(root / "manifests" / "STACK_V3_SOURCE.json"),
            "resolved_revision": stack["resolved_revision"],
        }
    }
    for bucket, filename in (
        ("fineweb_edu", "FINEWEB_EDU_SOURCE.json"),
        ("wikipedia", "WIKIPEDIA_SOURCE.json"),
    ):
        path = root / "manifests" / filename
        if not path.is_file():
            raise CurationError(f"Missing source manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for field in ("repo_id", "resolved_revision", "dataset_config"):
            if not isinstance(manifest.get(field), str) or not manifest[field]:
                raise CurationError(f"Invalid {filename} field {field}")
        if manifest.get("tokenizer_revision") != revision:
            raise CurationError(f"{filename} was counted with another tokenizer revision")
        sources[bucket] = (
            f"{manifest['repo_id']}@{manifest['resolved_revision']}#{manifest['dataset_config']}"
        )
        source_manifests[filename] = {
            "sha256": file_sha256(path),
            "resolved_revision": manifest["resolved_revision"],
            "tokenizer_revision": manifest["tokenizer_revision"],
        }
    return sources, {
        "tokenizer_manifest_sha256": hashlib.sha256(tokenizer_raw).hexdigest(),
        "tokenizer_revision": revision,
        "tokenizer_files_validated": sorted(files),
        "source_manifests": source_manifests,
        # The parallel Stack collector pins its source and runs with the same
        # tokenizer directory, but its v1 source manifest did not persist a
        # tokenizer field. Preserve this historical provenance limitation.
        "legacy_stack_tokenizer_binding": "collector_configuration_not_source_manifest_field",
    }


def english_source_identity(bucket: str, provenance: dict[str, Any]) -> str | None:
    if bucket == "fineweb_edu":
        for key in ("url", "id"):
            value = provenance.get(key)
            if value is not None and str(value).strip():
                return f"fineweb_edu:{key}:{str(value).strip()}"
    elif bucket == "wikipedia":
        for key in ("id", "url", "title"):
            value = provenance.get(key)
            if value is not None and str(value).strip():
                return f"wikipedia:{key}:{str(value).strip()}"
    return None


def split_thresholds(quotas: dict[tuple[str, str], int]) -> list[tuple[str, int]]:
    totals = {split: sum(quotas[(split, category)] for category in set(BUCKET_CATEGORY.values())) for split in SPLITS}
    grand = sum(totals.values())
    maximum = 1 << 64
    # Held-out splits receive the first ranges. Integer thresholds are exact
    # and the remainder goes to train.
    test_end = maximum * totals["test"] // grand
    validation_end = test_end + maximum * totals["validation"] // grand
    return [("test", test_end), ("validation", validation_end), ("train", maximum)]


def assign_group_split(seed: str, group_id: bytes, thresholds: Sequence[tuple[str, int]]) -> str:
    value = int.from_bytes(stable_digest(f"split:{seed}", group_id)[:8], "big")
    for split, end in thresholds:
        if value < end:
            return split
    raise AssertionError("final split threshold must cover uint64")


def iter_merged_quota_candidates(
    connection: sqlite3.Connection,
    *,
    split: str,
    buckets: Sequence[str],
) -> Iterator[tuple[bytes, int]]:
    """Merge independently indexed bucket scans in global deterministic rank order.

    Each SQLite cursor holds only its current B-tree position. The Python heap
    holds one row per bucket, so English selection uses constant memory instead
    of materializing a two-bucket ``IN (...) ORDER BY`` temporary B-tree.
    """
    for _rank, doc_id, tokens in iter_merged_quota_candidate_rows(
        connection, split=split, buckets=buckets, after=None
    ):
        yield doc_id, tokens


def iter_merged_quota_candidate_rows(
    connection: sqlite3.Connection,
    *,
    split: str,
    buckets: Sequence[str],
    after: tuple[bytes, bytes] | None,
) -> Iterator[tuple[bytes, bytes, int]]:
    """Yield quota candidates after an exclusive, durable total-order cursor."""
    ordered_buckets = sorted(buckets)
    if not ordered_buckets or len(set(ordered_buckets)) != len(ordered_buckets):
        raise CurationError("Quota candidate buckets must be unique and non-empty")
    cursors: list[sqlite3.Cursor] = []
    heap: list[tuple[bytes, bytes, int, int]] = []
    try:
        for ordinal, bucket in enumerate(ordered_buckets):
            if after is None:
                cursor = connection.execute(QUOTA_CANDIDATE_SQL, (bucket, split))
            else:
                after_rank, after_doc_id = after
                cursor = connection.execute(
                    f"""
                    SELECT d.doc_id, d.tokens, d.selection_rank
                    FROM documents AS d INDEXED BY {QUOTA_SELECTION_INDEX}
                    CROSS JOIN groups AS g
                    WHERE d.bucket=?
                      AND g.group_id=d.source_group
                      AND g.split=?
                      AND NOT EXISTS (
                          SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                      )
                      AND (
                          d.selection_rank > ? OR
                          (d.selection_rank = ? AND d.doc_id > ?)
                      )
                    ORDER BY d.selection_rank, d.doc_id
                    """,
                    (bucket, split, after_rank, after_rank, after_doc_id),
                )
            cursors.append(cursor)
            row = cursor.fetchone()
            if row is not None:
                doc_id = bytes(row[0])
                tokens = int(row[1])
                rank = bytes(row[2])
                heapq.heappush(heap, (rank, doc_id, ordinal, tokens))
        while heap:
            rank, doc_id, ordinal, tokens = heapq.heappop(heap)
            yield rank, doc_id, tokens
            row = cursors[ordinal].fetchone()
            if row is not None:
                next_doc_id = bytes(row[0])
                heapq.heappush(
                    heap,
                    (bytes(row[2]), next_doc_id, ordinal, int(row[1])),
                )
    finally:
        for cursor in cursors:
            cursor.close()


class CurationBuilder:
    def __init__(
        self,
        *,
        root: Path,
        staging_root: Path,
        output: Path,
        policy_path: Path,
        quota_path: Path,
        denylist_path: Path,
        english_near_clusters: Path | None,
        allow_missing_english_near_dedup: bool,
        batch_size: int = 5_000,
        sqlite_journal_mode: str = "auto",
    ) -> None:
        self.root = root
        self.staging_root = staging_root
        self.output = output
        self.policy_path = policy_path
        self.quota_path = quota_path
        self.denylist_path = denylist_path
        self.english_near_clusters = english_near_clusters
        self.allow_missing_english_near_dedup = allow_missing_english_near_dedup
        self.batch_size = batch_size
        self.sqlite_journal_mode = sqlite_journal_mode
        if batch_size < 1 or batch_size > 100_000:
            raise CurationError("batch_size must be between 1 and 100,000")
        self.sqlite_runtime = {
            "sqlite_version": sqlite3.sqlite_version,
            "journal_policy": select_sqlite_journal_policy(
                detect_output_mount(output),
                sqlite_journal_mode,
            ),
        }
        self.storage_contract = {
            "contract_version": CURATION_STORAGE_PREFLIGHT_VERSION,
            "progress_version": CURATION_PROGRESS_VERSION,
            "maximum_transaction_rows": batch_size,
            "transaction_sidecar_limit_bytes": max(
                CURATION_MINIMUM_SIDECAR_LIMIT_BYTES,
                batch_size * CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW,
            ),
            "projected_additional_bytes_per_document": (
                CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
            ),
            "disk_safety_numerator": CURATION_DISK_SAFETY_NUMERATOR,
            "disk_safety_denominator": CURATION_DISK_SAFETY_DENOMINATOR,
            "minimum_free_bytes_after_projection": CURATION_MINIMUM_FREE_BYTES,
            "sqlite_temp_store": SQLITE_TEMP_STORE,
            "sqlite_temp_relative_path": ".work/sqlite-tmp",
            "sqlite_temp_same_device_as_database": True,
        }
        # Database v2 predated the dedicated, same-filesystem SQLite sorter
        # directory.  This is the *only* legacy storage identity accepted for
        # a v2 resume; migration replaces it atomically with the current
        # contract before any phase work can continue.
        self.legacy_v3_storage_contract = {
            **self.storage_contract,
            "contract_version": 1,
            "projected_additional_bytes_per_document": 1_024,
        }
        self.legacy_v2_storage_contract = {
            key: value
            for key, value in self.legacy_v3_storage_contract.items()
            if key
            not in (
                "sqlite_temp_relative_path",
                "sqlite_temp_same_device_as_database",
            )
        }
        self.policy = load_policy(policy_path)
        self.policy_sha = canonical_sha256(self.policy)
        self.curation_profile = curation_profile(self.policy)
        self.fast_canonical_profile = self.curation_profile is not None
        if self.fast_canonical_profile:
            if english_near_clusters is not None:
                raise CurationError(
                    "Fast canonical profile must not consume an English near mapping"
                )
            if allow_missing_english_near_dedup:
                raise CurationError(
                    "--allow-missing-english-near-dedup is diagnostic-only and "
                    "cannot be combined with the production fast profile"
                )
        elif allow_missing_english_near_dedup and english_near_clusters is not None:
            raise CurationError(
                "--allow-missing-english-near-dedup is only valid when the "
                "required mapping is absent; remove the diagnostic override"
            )
        self.quotas = load_final_quotas(quota_path)
        self.collection_targets = load_collection_targets(quota_path)
        self.quota_sha = file_sha256(quota_path)
        self.guard = BenchmarkGuard(denylist_path)
        self.sources, self.artifact_identity = expected_source_strings(root, self.policy)
        self.preprocess_manifest = self._load_preprocess_manifest()
        self.report_inventory, self.collection_completeness = (
            self._load_complete_report_inventory()
        )
        self.inventory_sha = self.collection_completeness["reports"][
            "inventory_sha256"
        ]
        self.english_near_artifact = self._validate_english_near_artifact()
        self.near_sha = (
            self.english_near_artifact["mapping"]["sha256"]
            if self.english_near_artifact is not None
            else None
        )
        self.identity = {
            "format_version": (
                FAST_CANONICAL_FORMAT_VERSION
                if self.fast_canonical_profile
                else FORMAT_VERSION
            ),
            "policy_sha256": self.policy_sha,
            "quota_config_sha256": self.quota_sha,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "preprocess_manifest_sha256": file_sha256(
                staging_root / "PREPROCESS_MANIFEST.json"
            ),
            "report_inventory_sha256": self.inventory_sha,
            "report_count": len(self.report_inventory),
            "collection_completeness": self.collection_completeness,
            "english_near_clusters_sha256": self.near_sha,
            "english_near_artifact": self.english_near_artifact,
            "sqlite_runtime": self.sqlite_runtime,
            "curation_storage_contract": self.storage_contract,
            **self.artifact_identity,
        }
        if self.fast_canonical_profile:
            self.identity["curation_profile"] = self.curation_profile
        self.work = output / ".work"
        self.sqlite_temp = self.work / "sqlite-tmp"
        self.db_path = self.work / "curation.sqlite3"
        self.connection: sqlite3.Connection | None = None

    def _load_preprocess_manifest(self) -> dict[str, Any]:
        path = self.staging_root / "PREPROCESS_MANIFEST.json"
        if not path.is_file():
            raise CurationError(f"Missing preprocess manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "manifest_version": 1,
            "fingerprint_version": FINGERPRINT_VERSION,
            "policy_sha256": POLICY_SHA256,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "raw_data_mutated": False,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CurationError(f"Preprocess manifest {key} mismatch")
        return payload

    @staticmethod
    def _positive_count(value: Any, *, field: str, source: Path) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise CurationError(f"Invalid positive {field} in {source}")
        return value

    @staticmethod
    def _json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
        if not path.is_file() or path.is_symlink():
            raise CurationError(f"Missing or unsafe {label}: {path}")
        raw = path.read_bytes()
        payload = json_object_without_duplicate_keys(raw, label)
        return raw, payload

    def _load_collection_records(self) -> dict[tuple[str, int], dict[str, Any]]:
        directory = self.root / "state" / "quota_records" / "collection"
        if not directory.is_dir():
            raise CurationError(f"Missing finalized collection quota records: {directory}")
        paths = sorted(
            path for path in directory.iterdir() if path.is_file() or path.is_symlink()
        )
        if not paths:
            raise CurationError(f"No finalized collection quota records in {directory}")
        records: dict[tuple[str, int], dict[str, Any]] = {}
        shard_ids: set[str] = set()
        for path in paths:
            if path.name.startswith(".") or path.suffix != ".json":
                raise CurationError(f"Pending or unrecognized collection quota record: {path}")
            raw, record = self._json_object(path, label="collection quota record")
            if record.get("record_version") != QUOTA_RECORD_VERSION:
                raise CurationError(f"Unsupported collection quota record version: {path}")
            if record.get("phase") != "collection":
                raise CurationError(f"Non-collection record in collection ledger: {path}")
            bucket = quota_record_bucket(record)
            if bucket not in ALL_BUCKETS:
                raise CurationError(f"Unrecognized collection quota bucket: {path}")
            shard_id = record.get("shard_id")
            if not isinstance(shard_id, str) or not shard_id:
                raise CurationError(f"Invalid collection quota shard_id: {path}")
            match = QUOTA_SHARD_SUFFIX.search(shard_id)
            if match is None:
                raise CurationError(
                    f"Collection quota shard_id has no six-digit archive suffix: {path}"
                )
            index = int(match.group(1))
            key = (bucket, index)
            if key in records:
                raise CurationError(
                    f"Duplicate collection quota records for {bucket} part {index:06d}"
                )
            if shard_id in shard_ids:
                raise CurationError(f"Duplicate collection quota shard_id {shard_id!r}")
            shard_ids.add(shard_id)
            expected_path = record_path(self.root, record)
            if path != expected_path:
                raise CurationError(
                    f"Collection quota record is not at its canonical path: {path}"
                )
            if record.get("source") != self.sources[bucket]:
                raise CurationError(f"Collection quota source mismatch: {path}")
            counts = {
                field: self._positive_count(record.get(field), field=field, source=path)
                for field in ("documents", "clean_bytes", "exact_tokens")
            }
            archive = str(
                Path("raw")
                / BUCKET_PATHS[bucket]
                / f"part-{index:06d}.tar.zst"
            )
            records[key] = {
                "path": path,
                "relative_path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "record": record,
                "bucket": bucket,
                "index": index,
                "shard_id": shard_id,
                "archive": archive,
                **counts,
            }
        present_buckets = {bucket for bucket, _index in records}
        if present_buckets != set(ALL_BUCKETS):
            raise CurationError(
                "Finalized collection ledger must contain every source bucket; "
                f"found {sorted(present_buckets)}"
            )
        totals = {
            bucket: sum(
                int(item["exact_tokens"])
                for (item_bucket, _index), item in records.items()
                if item_bucket == bucket
            )
            for bucket in sorted(ALL_BUCKETS)
        }
        for bucket, target in self.collection_targets.items():
            if totals[bucket] < target:
                raise CurationError(
                    f"Collection target not reached for {bucket}: "
                    f"{totals[bucket]} < {target} exact tokens"
                )
        return records

    def _load_completion_markers(
        self,
        ledger_totals: dict[str, dict[str, int]],
    ) -> list[dict[str, Any]]:
        markers: list[dict[str, Any]] = []
        for filename, buckets in COMPLETION_MARKERS:
            path = self.root / "state" / filename
            raw, marker = self._json_object(path, label="collection completion marker")
            if marker.get("source") != self.sources[buckets[0]]:
                raise CurationError(f"Collection completion marker source mismatch: {path}")
            benchmark_identity = marker.get("benchmark_guard_sha256")
            if (
                benchmark_identity is not None
                and benchmark_identity != self.guard.manifest_sha256
            ):
                raise CurationError(
                    f"Collection completion marker benchmark guard mismatch: {path}"
                )
            tokenizer_revision = marker.get("tokenizer_revision")
            if (
                tokenizer_revision is not None
                and tokenizer_revision != self.artifact_identity["tokenizer_revision"]
            ):
                raise CurationError(
                    f"Collection completion marker tokenizer mismatch: {path}"
                )
            if len(buckets) == 2:
                for bucket in buckets:
                    field = f"{bucket}_tokens"
                    if marker.get(field) != ledger_totals[bucket]["exact_tokens"]:
                        raise CurationError(
                            f"Collection completion marker {field} does not match ledger: {path}"
                        )
                marker_targets = marker.get("targets")
                if marker_targets is not None and marker_targets != {
                    bucket: self.collection_targets[bucket] for bucket in buckets
                }:
                    raise CurationError(
                        f"Collection completion marker targets mismatch: {path}"
                    )
            else:
                bucket = buckets[0]
                if marker.get("english_tokens") != ledger_totals[bucket]["exact_tokens"]:
                    raise CurationError(
                        f"Collection completion marker tokens do not match ledger: {path}"
                    )
                if marker.get("target") != self.collection_targets[bucket]:
                    raise CurationError(
                        f"Collection completion marker target mismatch: {path}"
                    )
            markers.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "buckets": list(buckets),
                }
            )
        return markers

    def _validate_raw_archive_inventory(
        self,
        records: dict[tuple[str, int], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expected = set(records)
        found: set[tuple[str, int]] = set()
        pending: list[Path] = []
        for bucket in sorted(ALL_BUCKETS):
            directory = self.root / "raw" / BUCKET_PATHS[bucket]
            if not directory.is_dir():
                raise CurationError(f"Missing raw archive directory for {bucket}: {directory}")
            for path in sorted(directory.iterdir()):
                if path.name.startswith(".part-"):
                    pending.append(path)
                    continue
                if not path.name.startswith("part-"):
                    continue
                match = ARCHIVE_NAME.fullmatch(path.name)
                if match is None:
                    raise CurationError(f"Unfinalized or malformed raw archive input: {path}")
                if not path.is_file() or path.is_symlink():
                    raise CurationError(f"Missing or unsafe raw archive: {path}")
                found.add((bucket, int(match.group(1))))
        if pending:
            raise CurationError(
                "Pending raw archive inputs remain: "
                + ", ".join(str(path) for path in pending[:3])
            )
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing:
            raise CurationError(f"Finalized quota records are missing raw archives: {missing[:3]}")
        if extra:
            raise CurationError(f"Raw archives have no finalized quota record: {extra[:3]}")
        inventory: list[dict[str, Any]] = []
        for key in sorted(records):
            item = records[key]
            path = self.root / item["archive"]
            size = path.stat().st_size
            if size <= 0:
                raise CurationError(f"Finalized raw archive is empty: {path}")
            inventory.append(
                {
                    "archive": item["archive"],
                    "bytes": size,
                    "quota_shard_id": item["shard_id"],
                }
            )
        return inventory

    def _load_report_inventory(
        self,
        records: dict[tuple[str, int], dict[str, Any]],
    ) -> list[tuple[str, Path, str, dict[str, Any]]]:
        reports_root = self.staging_root / "reports"
        paths = (
            sorted(
                path
                for path in reports_root.rglob("*")
                if path.is_file() or path.is_symlink()
            )
            if reports_root.exists()
            else []
        )
        if not paths:
            raise CurationError(f"No fingerprint reports found under {reports_root}")
        result: list[tuple[str, Path, str, dict[str, Any]]] = []
        seen_archives: set[str] = set()
        seen_shards: set[str] = set()
        seen_keys: set[tuple[str, int]] = set()
        # The completeness snapshot is authoritative only after every raw
        # archive has been re-hashed. Keep the verified values so the English
        # near-artifact contract can reuse this byte evidence without reading
        # those archives a second time. A later final snapshot rebuilds the
        # mapping and therefore detects mutation during a long curation run.
        verified_raw_archive_sha256: dict[str, str] = {}
        for path in paths:
            if path.name.startswith(".") or path.suffix != ".json":
                raise CurationError(f"Pending or unrecognized fingerprint report: {path}")
            raw, report = self._json_object(path, label="fingerprint report")
            bucket = report.get("bucket")
            archive = report.get("archive")
            if report.get("report_version") != 1 or report.get("fingerprint_version") != FINGERPRINT_VERSION:
                raise CurationError(f"Unsupported report version: {path}")
            if report.get("policy_sha256") != POLICY_SHA256:
                raise CurationError(f"Fingerprint policy mismatch: {path}")
            if bucket not in ALL_BUCKETS or report.get("source") != self.sources[bucket]:
                raise CurationError(f"Source identity mismatch in {path}")
            index = report.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise CurationError(f"Invalid report archive index in {path}")
            key = (str(bucket), index)
            if key in seen_keys:
                raise CurationError(
                    f"Duplicate fingerprint reports for {bucket} part {index:06d}"
                )
            seen_keys.add(key)
            quota = records.get(key)
            if quota is None:
                raise CurationError(f"Fingerprint report has no finalized quota record: {path}")
            expected_path = reports_root / str(bucket) / f"part-{index:06d}.json"
            if path != expected_path:
                raise CurationError(f"Fingerprint report is not at its canonical path: {path}")
            if not isinstance(archive, str) or archive in seen_archives:
                raise CurationError(f"Duplicate/invalid report archive in {path}")
            seen_archives.add(archive)
            if archive != quota["archive"]:
                raise CurationError(f"Report/raw archive identity mismatch in {path}")
            quota_shard_id = report.get("quota_shard_id")
            if quota_shard_id != quota["shard_id"] or quota_shard_id in seen_shards:
                raise CurationError(f"Report quota shard identity mismatch/duplicate in {path}")
            seen_shards.add(str(quota_shard_id))
            parse_hex_digest(report.get("archive_sha256"), "archive_sha256")
            parse_hex_digest(report.get("fingerprint_sha256"), "fingerprint_sha256")
            for field in ("documents", "exact_tokens", "clean_bytes"):
                value = self._positive_count(report.get(field), field=field, source=path)
                if value != quota[field]:
                    raise CurationError(f"Report/quota {field} mismatch in {path}")
            archive_size = self._positive_count(
                report.get("archive_compressed_bytes"),
                field="archive_compressed_bytes",
                source=path,
            )
            raw_archive = self.root / quota["archive"]
            if raw_archive.stat().st_size != archive_size:
                raise CurationError(f"Report/raw archive size mismatch in {path}")
            actual_archive_sha256 = file_sha256(raw_archive)
            if actual_archive_sha256 != report["archive_sha256"]:
                raise CurationError(f"Report/raw archive checksum mismatch in {path}")
            verified_raw_archive_sha256[quota["archive"]] = actual_archive_sha256
            expected_fingerprint = (
                self.staging_root
                / "fingerprints"
                / str(bucket)
                / f"part-{index:06d}.jsonl.zst"
            )
            try:
                fingerprint = self.staging_root / str(report["fingerprint_file"])
            except KeyError as exc:
                raise CurationError(f"Missing fingerprint_file in {path}") from exc
            if fingerprint != expected_fingerprint:
                raise CurationError(f"Non-canonical fingerprint path in {path}")
            if not fingerprint.is_file() or fingerprint.is_symlink():
                raise CurationError(f"Missing or unsafe fingerprint shard: {fingerprint}")
            relative = str(path.relative_to(self.staging_root))
            result.append((relative, path, hashlib.sha256(raw).hexdigest(), report))
        expected_keys = set(records)
        if seen_keys != expected_keys:
            missing = sorted(expected_keys - seen_keys)
            raise CurationError(
                f"Finalized quota records are missing fingerprint reports: {missing[:3]}"
            )

        expected_fingerprints = {
            self.staging_root
            / "fingerprints"
            / bucket
            / f"part-{index:06d}.jsonl.zst"
            for bucket, index in expected_keys
        }
        fingerprints_root = self.staging_root / "fingerprints"
        actual_fingerprints = (
            {
                path
                for path in fingerprints_root.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            if fingerprints_root.exists()
            else set()
        )
        unexpected_fingerprints = sorted(actual_fingerprints - expected_fingerprints)
        missing_fingerprints = sorted(expected_fingerprints - actual_fingerprints)
        if missing_fingerprints:
            raise CurationError(f"Missing fingerprint shards: {missing_fingerprints[:3]}")
        if unexpected_fingerprints:
            raise CurationError(
                f"Pending, orphaned, or extra fingerprint shards: {unexpected_fingerprints[:3]}"
            )
        if set(verified_raw_archive_sha256) != seen_archives:
            raise CurationError("Raw archive checksum verification coverage is incomplete")
        self._verified_raw_archive_sha256 = verified_raw_archive_sha256
        return result

    def _load_complete_report_inventory(
        self,
    ) -> tuple[list[tuple[str, Path, str, dict[str, Any]]], dict[str, Any]]:
        errors_root = self.staging_root / "errors"
        error_records = (
            sorted(
                path
                for path in errors_root.rglob("*")
                if path.is_file() or path.is_symlink()
            )
            if errors_root.exists()
            else []
        )
        if error_records:
            raise CurationError(
                "Preprocessing error records remain: "
                + ", ".join(str(path) for path in error_records[:3])
            )
        records = self._load_collection_records()
        raw_inventory = self._validate_raw_archive_inventory(records)
        report_inventory = self._load_report_inventory(records)
        ledger_totals = {
            bucket: {
                "archives": sum(1 for record_bucket, _index in records if record_bucket == bucket),
                "documents": sum(
                    int(item["documents"])
                    for (record_bucket, _index), item in records.items()
                    if record_bucket == bucket
                ),
                "clean_bytes": sum(
                    int(item["clean_bytes"])
                    for (record_bucket, _index), item in records.items()
                    if record_bucket == bucket
                ),
                "exact_tokens": sum(
                    int(item["exact_tokens"])
                    for (record_bucket, _index), item in records.items()
                    if record_bucket == bucket
                ),
                "target_exact_tokens": self.collection_targets[bucket],
            }
            for bucket in sorted(ALL_BUCKETS)
        }
        report_totals = {
            bucket: {
                "archives": sum(
                    1
                    for *_prefix, report in report_inventory
                    if report["bucket"] == bucket
                ),
                "documents": sum(
                    int(report["documents"])
                    for *_prefix, report in report_inventory
                    if report["bucket"] == bucket
                ),
                "clean_bytes": sum(
                    int(report["clean_bytes"])
                    for *_prefix, report in report_inventory
                    if report["bucket"] == bucket
                ),
                "exact_tokens": sum(
                    int(report["exact_tokens"])
                    for *_prefix, report in report_inventory
                    if report["bucket"] == bucket
                ),
            }
            for bucket in sorted(ALL_BUCKETS)
        }
        for bucket in sorted(ALL_BUCKETS):
            expected = {
                field: ledger_totals[bucket][field]
                for field in ("archives", "documents", "clean_bytes", "exact_tokens")
            }
            if report_totals[bucket] != expected:
                raise CurationError(
                    f"Per-bucket report/collection-ledger totals mismatch for {bucket}"
                )
        markers = self._load_completion_markers(ledger_totals)
        quota_inventory = [
            {
                "path": item["relative_path"],
                "sha256": item["sha256"],
                "bucket": bucket,
                "index": index,
                "shard_id": item["shard_id"],
                "archive": item["archive"],
                "documents": item["documents"],
                "clean_bytes": item["clean_bytes"],
                "exact_tokens": item["exact_tokens"],
            }
            for (bucket, index), item in sorted(records.items())
        ]
        report_identity = [
            {"path": relative, "sha256": checksum}
            for relative, _path, checksum, _report in report_inventory
        ]
        archive_sha_by_path = {
            report["archive"]: report["archive_sha256"]
            for *_prefix, report in report_inventory
        }
        raw_inventory = [
            {**item, "sha256": archive_sha_by_path[item["archive"]]}
            for item in raw_inventory
        ]
        fingerprint_identity = [
            {
                "path": report["fingerprint_file"],
                "sha256": report["fingerprint_sha256"],
            }
            for *_prefix, report in report_inventory
        ]
        completeness = {
            "format_version": COMPLETENESS_FORMAT_VERSION,
            "complete": True,
            "legacy_dedup_index_required": False,
            "pending_inputs": 0,
            "preprocess_error_records": 0,
            "collection_targets_exact_tokens": {
                bucket: self.collection_targets[bucket] for bucket in sorted(ALL_BUCKETS)
            },
            "quota_records": {
                "count": len(quota_inventory),
                "inventory_sha256": hashlib.sha256(
                    canonical_json_bytes(quota_inventory)
                ).hexdigest(),
            },
            "completion_markers": {
                "count": len(markers),
                "inventory_sha256": hashlib.sha256(
                    canonical_json_bytes(markers)
                ).hexdigest(),
                "files": markers,
            },
            "raw_archives": {
                "count": len(raw_inventory),
                "inventory_sha256": hashlib.sha256(
                    canonical_json_bytes(raw_inventory)
                ).hexdigest(),
            },
            "reports": {
                "count": len(report_identity),
                "inventory_sha256": hashlib.sha256(
                    canonical_json_bytes(report_identity)
                ).hexdigest(),
            },
            "fingerprints": {
                "count": len(fingerprint_identity),
                "inventory_sha256": hashlib.sha256(
                    canonical_json_bytes(fingerprint_identity)
                ).hexdigest(),
            },
            "per_bucket": ledger_totals,
        }
        return report_inventory, completeness

    def _current_english_near_collection_evidence(self) -> dict[str, Any]:
        """Recreate the near-dedup builder's v1 English ledger evidence."""
        records = self._load_collection_records()
        english_records = {
            key: item
            for key, item in records.items()
            if key[0] in ENGLISH_BUCKETS
        }
        quota_records = [
            {
                "path": item["relative_path"],
                "sha256": item["sha256"],
                "shard_id": item["shard_id"],
                "bucket": bucket,
                "index": index,
                "documents": item["documents"],
                "clean_bytes": item["clean_bytes"],
                "exact_tokens": item["exact_tokens"],
                "source": item["record"]["source"],
            }
            for (bucket, index), item in sorted(english_records.items())
        ]
        reports = {
            (str(report["bucket"]), int(report["index"])): report
            for *_prefix, report in self.report_inventory
            if report["bucket"] in ENGLISH_BUCKETS
        }
        marker_names = {
            "fineweb_edu": "ENGLISH_FINEWEB_EDU_COMPLETE.json",
            "wikipedia": "ENGLISH_WIKIPEDIA_COMPLETE.json",
        }
        buckets: dict[str, Any] = {}
        for bucket in ENGLISH_BUCKETS:
            bucket_records = {
                index: item
                for (record_bucket, index), item in english_records.items()
                if record_bucket == bucket
            }
            bucket_reports = {
                index: report
                for (report_bucket, index), report in reports.items()
                if report_bucket == bucket
            }
            if set(bucket_records) != set(bucket_reports):
                raise CurationError(
                    f"English near-dedup evidence coverage mismatch for {bucket}"
                )
            marker_path = self.root / "state" / marker_names[bucket]
            marker_raw, marker = self._json_object(
                marker_path, label="English collection completion marker"
            )
            finalized_totals = {
                "archives": len(bucket_records),
                "documents": sum(item["documents"] for item in bucket_records.values()),
                "clean_bytes": sum(item["clean_bytes"] for item in bucket_records.values()),
                "exact_tokens": sum(item["exact_tokens"] for item in bucket_records.values()),
            }
            report_totals = {
                "archives": len(bucket_reports),
                "documents": sum(int(item["documents"]) for item in bucket_reports.values()),
                "clean_bytes": sum(int(item["clean_bytes"]) for item in bucket_reports.values()),
                "exact_tokens": sum(int(item["exact_tokens"]) for item in bucket_reports.values()),
            }
            if finalized_totals != report_totals:
                raise CurationError(
                    f"English report/ledger totals changed for {bucket}"
                )
            buckets[bucket] = {
                "target_exact_tokens": self.collection_targets[bucket],
                "completion_marker": {
                    "path": str(marker_path.relative_to(self.root)),
                    "sha256": hashlib.sha256(marker_raw).hexdigest(),
                    **marker,
                },
                "finalized_totals": finalized_totals,
                "archive_indices_sha256": hashlib.sha256(
                    canonical_json_bytes(sorted(bucket_records))
                ).hexdigest(),
                "report_totals": report_totals,
            }
        return {
            "evidence_version": 1,
            "quota_config_path": str(self.quota_path),
            "quota_config_sha256": self.quota_sha,
            "quota_record_inventory_sha256": hashlib.sha256(
                canonical_json_bytes(quota_records)
            ).hexdigest(),
            "quota_records": quota_records,
            "buckets": buckets,
        }

    def _current_english_near_source_manifests(self) -> dict[str, Any]:
        tokenizer = self.root / "tokenizer" / "starcoder2" / "TOKENIZER_MANIFEST.json"
        return {
            "TOKENIZER_MANIFEST.json": {
                "sha256": file_sha256(tokenizer),
                "resolved_revision": self.artifact_identity["tokenizer_revision"],
            },
            **{
                filename: self.artifact_identity["source_manifests"][filename]
                for filename in (
                    "FINEWEB_EDU_SOURCE.json",
                    "WIKIPEDIA_SOURCE.json",
                )
            },
        }

    def _validate_english_near_runtime(self, runtime: Any) -> None:
        if not isinstance(runtime, dict):
            raise CurationError("English near-dedup runtime evidence is missing")
        for field in ("python", "sqlite", "xxhash", "zstandard"):
            if not isinstance(runtime.get(field), str) or not runtime[field]:
                raise CurationError(
                    f"English near-dedup runtime {field} evidence is missing"
                )
        storage = runtime.get("storage")
        if not isinstance(storage, dict):
            raise CurationError("English near-dedup runtime storage evidence is missing")
        for field in (
            "filesystem_type",
            "mount_point",
            "mount_source",
            "mount_options",
            "detection",
            "classification",
            "sqlite_journal_mode_configured",
            "sqlite_journal_mode_requested",
            "sqlite_journal_mode_request_source",
            "sqlite_journal_mode_selected",
            "sqlite_journal_mode_actual",
        ):
            if not isinstance(storage.get(field), str) or not storage[field]:
                raise CurationError(
                    f"English near-dedup storage {field} evidence is missing"
                )
        configured = storage["sqlite_journal_mode_configured"]
        requested = storage["sqlite_journal_mode_requested"]
        selected = storage["sqlite_journal_mode_selected"]
        actual = storage["sqlite_journal_mode_actual"]
        request_source = storage["sqlite_journal_mode_request_source"]
        if configured not in ("auto", "delete", "wal"):
            raise CurationError("Invalid configured English near-dedup journal mode")
        if requested not in ("auto", "delete", "wal") or selected not in (
            "delete",
            "wal",
        ):
            raise CurationError("Invalid selected English near-dedup journal mode")
        if selected != actual:
            raise CurationError(
                "English near-dedup selected/actual SQLite journal modes differ"
            )
        if request_source not in ("cli", "config"):
            raise CurationError("Invalid English near-dedup journal request source")
        if request_source == "config" and requested != configured:
            raise CurationError("English near-dedup config journal request drifted")
        classification = storage["classification"]
        if classification not in ("proven-local", "non-local", "unknown"):
            raise CurationError("Invalid English near-dedup storage classification")
        if selected == "wal" and classification != "proven-local":
            raise CurationError("English near-dedup WAL lacks proven-local storage")
        if classification != "proven-local" and selected != "delete":
            raise CurationError("English near-dedup network/unknown storage used WAL")
        policy = storage.get("policy")
        if not isinstance(policy, dict):
            raise CurationError("English near-dedup storage policy evidence is missing")
        expected_config = json.loads(DEFAULT_ENGLISH_NEAR_CONFIG.read_text(encoding="utf-8"))
        expected_allowlist = expected_config["storage"]["wal_local_filesystem_allowlist"]
        expected_policy = {
            "network_or_unknown_action": "delete",
            "wal_on_non_allowlisted_action": "fail_closed",
            "wal_local_filesystem_allowlist": expected_allowlist,
        }
        if policy != expected_policy:
            raise CurationError("English near-dedup storage policy evidence mismatch")
        if configured != expected_config["storage"]["sqlite_journal_mode"]:
            raise CurationError("English near-dedup configured journal mode mismatch")

    def _validate_english_near_calibration_evidence(
        self,
        evidence: Any,
        *,
        near_identity: dict[str, Any],
        expected_reports: list[dict[str, Any]],
    ) -> None:
        evidence_fields = {
            "contract_version",
            "result_path",
            "result_sha256",
            "result_bytes",
            "sidecar_path",
            "sidecar_sha256",
            "result_version",
            "status",
            "production_gate_eligible",
            "acceptance_profile",
            "sampling_profile",
            "acceptance_failures",
            "identity_sha256",
            "identity",
        }
        if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
            raise CurationError("English near-dedup calibration evidence is incomplete")
        for field, expected in {
            "contract_version": 1,
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_failures": [],
        }.items():
            if evidence.get(field) != expected:
                raise CurationError(
                    f"English near-dedup calibration evidence {field} is unsafe"
                )
        result_sha = str(evidence.get("result_sha256"))
        sidecar_sha = str(evidence.get("sidecar_sha256"))
        parse_hex_digest(result_sha, "calibration result sha256")
        parse_hex_digest(sidecar_sha, "calibration sidecar sha256")
        result_bytes = evidence.get("result_bytes")
        if (
            not isinstance(result_bytes, int)
            or isinstance(result_bytes, bool)
            or result_bytes <= 0
        ):
            raise CurationError("English near-dedup calibration result size is invalid")

        def safe_calibration_path(value: Any, label: str) -> Path:
            if not isinstance(value, str) or not value:
                raise CurationError(f"English near-dedup {label} path is invalid")
            relative = Path(value)
            if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
                raise CurationError(f"English near-dedup {label} path is unsafe")
            root = self.root.resolve(strict=True)
            candidate = self.root / relative
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise CurationError(
                    f"Missing English near-dedup {label}: {candidate}"
                ) from exc
            if (
                not resolved.is_relative_to(root)
                or candidate.is_symlink()
                or not resolved.is_file()
            ):
                raise CurationError(f"English near-dedup {label} is unsafe")
            return resolved

        result_path = safe_calibration_path(evidence["result_path"], "calibration result")
        sidecar_path = safe_calibration_path(
            evidence["sidecar_path"], "calibration checksum sidecar"
        )
        if sidecar_path != result_path.with_name(result_path.name + ".sha256"):
            raise CurationError("English near-dedup calibration sidecar path is not canonical")
        result_raw = result_path.read_bytes()
        sidecar_raw = sidecar_path.read_bytes()
        if (
            len(result_raw) != result_bytes
            or hashlib.sha256(result_raw).hexdigest() != result_sha
        ):
            raise CurationError("English near-dedup calibration result identity mismatch")
        if (
            sidecar_raw != f"{result_sha}  {result_path.name}\n".encode("utf-8")
            or hashlib.sha256(sidecar_raw).hexdigest() != sidecar_sha
        ):
            raise CurationError("English near-dedup calibration sidecar mismatch")
        try:
            result = json.loads(result_raw)
        except json.JSONDecodeError as exc:
            raise CurationError("Invalid English near-dedup calibration result") from exc
        if not isinstance(result, dict):
            raise CurationError("English near-dedup calibration result must be an object")
        for field, expected in {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_overrides": {},
            "acceptance_failures": [],
        }.items():
            if result.get(field) != expected:
                raise CurationError(
                    f"English near-dedup calibration result {field} is unsafe"
                )
        for field in (
            "result_version",
            "status",
            "production_gate_eligible",
            "acceptance_profile",
            "sampling_profile",
            "acceptance_failures",
        ):
            if evidence[field] != result[field]:
                raise CurationError(
                    f"English near-dedup calibration evidence/result {field} mismatch"
                )

        calibration_identity = evidence.get("identity")
        identity_fields = {
            "harness_sha256",
            "production_builder_sha256",
            "calibration_algorithm",
            "calibration_seed",
            "production_config_file_sha256",
            "production_config_canonical_sha256",
            "calibration_config_file_sha256",
            "calibration_config_canonical_sha256",
            "input",
            "sample_manifest_sha256",
            "runtime",
        }
        if (
            not isinstance(calibration_identity, dict)
            or set(calibration_identity) != identity_fields
            or result.get("identity") != calibration_identity
        ):
            raise CurationError("English near-dedup calibration identity is incomplete")
        identity_sha = str(evidence.get("identity_sha256"))
        parse_hex_digest(identity_sha, "calibration identity sha256")
        if hashlib.sha256(canonical_json_bytes(calibration_identity)).hexdigest() != identity_sha:
            raise CurationError("English near-dedup calibration identity checksum mismatch")
        calibration_config_raw = DEFAULT_ENGLISH_NEAR_CALIBRATION_CONFIG.read_bytes()
        calibration_config = json.loads(calibration_config_raw)
        expected_identity = {
            "harness_sha256": file_sha256(ENGLISH_NEAR_CALIBRATION),
            "production_builder_sha256": file_sha256(ENGLISH_NEAR_BUILDER),
            "calibration_algorithm": calibration_config.get("calibration_algorithm"),
            "calibration_seed": calibration_config.get("seed"),
            "production_config_file_sha256": near_identity["config_file_sha256"],
            "production_config_canonical_sha256": near_identity["config_sha256"],
            "calibration_config_file_sha256": hashlib.sha256(
                calibration_config_raw
            ).hexdigest(),
            "calibration_config_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(calibration_config)
            ).hexdigest(),
        }
        for field, expected in expected_identity.items():
            if calibration_identity.get(field) != expected:
                raise CurationError(
                    f"English near-dedup calibration identity {field} mismatch"
                )
        parse_hex_digest(
            calibration_identity.get("sample_manifest_sha256"),
            "calibration sample manifest sha256",
        )
        runtime = calibration_identity.get("runtime")
        if not isinstance(runtime, dict) or set(runtime) != {
            "python",
            "xxhash",
            "zstandard",
        } or any(not isinstance(runtime[field], str) or not runtime[field] for field in runtime):
            raise CurationError("English near-dedup calibration runtime is invalid")

        input_identity = calibration_identity.get("input")
        input_fields = {
            "kind",
            "full_report_inventory_sha256",
            "preprocess_manifest_sha256",
            "curation_policy_sha256",
            "benchmark_guard_sha256",
            "source_manifests",
            "collection_completeness_sha256",
            "collection_completeness",
            "selection_seed",
            "maximum_archives_per_bucket",
            "maximum_documents_per_bucket",
            "minimum_source_words",
            "sampling_counts",
            "selected_reports",
            "documents_selected",
        }
        if not isinstance(input_identity, dict) or set(input_identity) != input_fields:
            raise CurationError("English near-dedup calibration input identity is incomplete")
        expected_input = {
            "kind": "immutable_real_english_sample",
            "full_report_inventory_sha256": near_identity[
                "report_inventory_sha256"
            ],
            "preprocess_manifest_sha256": near_identity[
                "preprocess_manifest_sha256"
            ],
            "curation_policy_sha256": near_identity["curation_policy_sha256"],
            "benchmark_guard_sha256": near_identity["benchmark_guard_sha256"],
            "source_manifests": near_identity["source_manifests"],
            "collection_completeness_sha256": hashlib.sha256(
                canonical_json_bytes(near_identity["collection_completeness"])
            ).hexdigest(),
            "collection_completeness": near_identity["collection_completeness"],
        }
        sampling_config = calibration_config.get("sampling")
        if not isinstance(sampling_config, dict):
            raise CurationError("Pinned English near-dedup calibration sampling is invalid")
        expected_input.update(
            selection_seed=calibration_config.get("seed"),
            maximum_archives_per_bucket=sampling_config.get(
                "maximum_archives_per_bucket"
            ),
            maximum_documents_per_bucket=sampling_config.get(
                "maximum_documents_per_bucket"
            ),
            minimum_source_words=sampling_config.get("minimum_source_words"),
        )
        for field, expected in expected_input.items():
            if input_identity.get(field) != expected:
                raise CurationError(
                    f"English near-dedup calibration input {field} mismatch"
                )
        selected_reports = input_identity.get("selected_reports")
        current_reports = {row["archive"]: row for row in expected_reports}
        if not isinstance(selected_reports, list) or not selected_reports:
            raise CurationError("English near-dedup calibration selected reports are empty")
        seen_reports: set[str] = set()
        for selected_report in selected_reports:
            if not isinstance(selected_report, dict):
                raise CurationError("Invalid English near-dedup calibration report")
            archive = selected_report.get("archive")
            if (
                not isinstance(archive, str)
                or archive in seen_reports
                or current_reports.get(archive) != selected_report
            ):
                raise CurationError(
                    "English near-dedup calibration selected-report identity mismatch"
                )
            seen_reports.add(archive)
        sampling_counts = input_identity.get("sampling_counts")
        if not isinstance(sampling_counts, dict) or set(sampling_counts) != set(ENGLISH_BUCKETS):
            raise CurationError("English near-dedup calibration sampling counts are invalid")
        selected_documents = 0
        for bucket in ENGLISH_BUCKETS:
            counts = sampling_counts[bucket]
            if not isinstance(counts, dict) or set(counts) != {
                "fingerprint_rows_scanned",
                "eligible_fingerprint_candidates",
                "documents_selected",
            }:
                raise CurationError("English near-dedup calibration bucket counts are invalid")
            for field, value in counts.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise CurationError(
                        f"English near-dedup calibration {bucket}.{field} is invalid"
                    )
            if counts["documents_selected"] < 1:
                raise CurationError(
                    f"English near-dedup calibration selected no {bucket} documents"
                )
            selected_documents += counts["documents_selected"]
        if input_identity.get("documents_selected") != selected_documents:
            raise CurationError("English near-dedup calibration document count mismatch")

        if result.get("acceptance") != calibration_config.get("acceptance"):
            raise CurationError("English near-dedup calibration acceptance policy mismatch")
        production_config = json.loads(DEFAULT_ENGLISH_NEAR_CONFIG.read_text(encoding="utf-8"))
        if result.get("production_threshold") != {
            "minimum_jaccard_numerator": production_config["refinement"][
                "minimum_jaccard_numerator"
            ],
            "minimum_jaccard_denominator": production_config["refinement"][
                "minimum_jaccard_denominator"
            ],
        }:
            raise CurationError("English near-dedup calibration threshold mismatch")
        sampling_result = result.get("sampling")
        if (
            not isinstance(sampling_result, dict)
            or sampling_result.get("documents_input") != selected_documents
        ):
            raise CurationError("English near-dedup calibration sampling result mismatch")
        sample_manifest = sampling_result.get("sample_manifest")
        if (
            not isinstance(sample_manifest, list)
            or not sample_manifest
            or hashlib.sha256(canonical_json_bytes(sample_manifest)).hexdigest()
            != calibration_identity["sample_manifest_sha256"]
        ):
            raise CurationError("English near-dedup calibration sample manifest mismatch")

    def _validate_english_near_operational_preflight(
        self,
        evidence: Any,
        *,
        publication_root: Path,
        near_identity: dict[str, Any],
        mapping_records: int,
    ) -> dict[str, Any]:
        """Reopen and independently verify the immutable refinement preflight."""

        evidence_fields = {
            "contract_version",
            "result_path",
            "result_sha256",
            "result_bytes",
            "sidecar_path",
            "sidecar_sha256",
            "status",
            "production_gate_eligible",
            "failures",
            "identity_sha256",
            "identity",
            "thresholds",
            "sample",
            "measurements",
        }
        if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
            raise CurationError(
                "English near-dedup operational preflight evidence is incomplete"
            )
        for field, expected in {
            "contract_version": 1,
            "result_path": "operational-preflight-v1/result.json",
            "sidecar_path": "operational-preflight-v1/result.json.sha256",
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
        }.items():
            if evidence.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight {field} is unsafe"
                )
        parse_hex_digest(
            evidence.get("result_sha256"), "operational preflight result sha256"
        )
        parse_hex_digest(
            evidence.get("sidecar_sha256"), "operational preflight sidecar sha256"
        )
        result_sha = str(evidence["result_sha256"])
        sidecar_sha = str(evidence["sidecar_sha256"])
        result_bytes = evidence.get("result_bytes")
        if (
            not isinstance(result_bytes, int)
            or isinstance(result_bytes, bool)
            or result_bytes <= 0
        ):
            raise CurationError(
                "English near-dedup operational preflight result size is invalid"
            )

        def publication_file(relative: str, label: str) -> Path:
            parts = Path(relative)
            if parts.is_absolute() or any(part in ("", ".", "..") for part in parts.parts):
                raise CurationError(f"English near-dedup {label} path is unsafe")
            candidate = publication_root / parts
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise CurationError(f"Missing English near-dedup {label}: {candidate}") from exc
            if (
                not resolved.is_relative_to(publication_root)
                or candidate.is_symlink()
                or not resolved.is_file()
            ):
                raise CurationError(f"English near-dedup {label} is unsafe")
            return resolved

        result_path = publication_file(evidence["result_path"], "operational preflight result")
        sidecar_path = publication_file(
            evidence["sidecar_path"], "operational preflight sidecar"
        )
        if sidecar_path != result_path.with_name("result.json.sha256"):
            raise CurationError(
                "English near-dedup operational preflight sidecar path is not canonical"
            )
        result_raw = result_path.read_bytes()
        sidecar_raw = sidecar_path.read_bytes()
        if (
            len(result_raw) != result_bytes
            or hashlib.sha256(result_raw).hexdigest() != result_sha
        ):
            raise CurationError(
                "English near-dedup operational preflight result identity mismatch"
            )
        if (
            sidecar_raw != f"{result_sha}  result.json\n".encode("ascii")
            or hashlib.sha256(sidecar_raw).hexdigest() != sidecar_sha
        ):
            raise CurationError(
                "English near-dedup operational preflight sidecar mismatch"
            )
        result = json_object_without_duplicate_keys(
            result_raw, "English near-dedup operational preflight result"
        )
        if set(result) != {
            "result_version",
            "status",
            "production_gate_eligible",
            "failures",
            "identity",
            "thresholds",
            "sample",
            "measurements",
            "statistical_scope",
        }:
            raise CurationError(
                "English near-dedup operational preflight result schema is incomplete"
            )
        for field, expected in {
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
        }.items():
            if result.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight result {field} is unsafe"
                )
        for field in ("status", "production_gate_eligible", "failures", "identity", "thresholds", "sample", "measurements"):
            if result.get(field) != evidence.get(field):
                raise CurationError(
                    f"English near-dedup operational preflight evidence/result {field} mismatch"
                )
        if not isinstance(result.get("statistical_scope"), str) or not result["statistical_scope"]:
            raise CurationError(
                "English near-dedup operational preflight statistical scope is missing"
            )

        preflight_identity = evidence.get("identity")
        identity_fields = {
            "contract_version",
            "builder_sha256",
            "builder_identity_sha256",
            "config_file_sha256",
            "config_sha256",
            "calibration_evidence_sha256",
            "report_inventory_sha256",
            "preprocess_manifest_sha256",
            "curation_policy_sha256",
            "benchmark_guard_sha256",
            "collection_completeness_sha256",
            "candidate_pairs_total",
            "documents_total",
            "candidate_blocks_total",
            "candidate_blocks_committed",
            "phase_at_measurement",
            "candidate_cursor_at_measurement",
            "refinement_cursor_at_measurement",
            "cache_archives",
            "cache_bytes",
            "cache_inventory_sha256",
            "runtime_storage",
        }
        if not isinstance(preflight_identity, dict) or set(preflight_identity) != identity_fields:
            raise CurationError(
                "English near-dedup operational preflight identity is incomplete"
            )
        parse_hex_digest(
            evidence.get("identity_sha256"), "operational preflight identity sha256"
        )
        identity_sha = str(evidence["identity_sha256"])
        if hashlib.sha256(canonical_json_bytes(preflight_identity)).hexdigest() != identity_sha:
            raise CurationError(
                "English near-dedup operational preflight identity checksum mismatch"
            )
        expected_identity = {
            "contract_version": 1,
            "builder_sha256": near_identity["builder_sha256"],
            "builder_identity_sha256": hashlib.sha256(
                canonical_json_bytes(near_identity)
            ).hexdigest(),
            "config_file_sha256": near_identity["config_file_sha256"],
            "config_sha256": near_identity["config_sha256"],
            "calibration_evidence_sha256": hashlib.sha256(
                canonical_json_bytes(near_identity["calibration_evidence"])
            ).hexdigest(),
            "report_inventory_sha256": near_identity["report_inventory_sha256"],
            "preprocess_manifest_sha256": near_identity["preprocess_manifest_sha256"],
            "curation_policy_sha256": near_identity["curation_policy_sha256"],
            "benchmark_guard_sha256": near_identity["benchmark_guard_sha256"],
            "collection_completeness_sha256": hashlib.sha256(
                canonical_json_bytes(near_identity["collection_completeness"])
            ).hexdigest(),
            "documents_total": mapping_records,
            "cache_archives": near_identity["report_count"],
            "runtime_storage": near_identity["runtime"]["storage"],
        }
        for field, expected in expected_identity.items():
            if preflight_identity.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight identity {field} mismatch"
                )
        for field in (
            "builder_sha256",
            "builder_identity_sha256",
            "config_file_sha256",
            "config_sha256",
            "calibration_evidence_sha256",
            "report_inventory_sha256",
            "preprocess_manifest_sha256",
            "curation_policy_sha256",
            "benchmark_guard_sha256",
            "collection_completeness_sha256",
            "cache_inventory_sha256",
        ):
            parse_hex_digest(preflight_identity.get(field), f"operational preflight {field}")
        for field in (
            "candidate_pairs_total",
            "documents_total",
            "candidate_blocks_total",
            "candidate_blocks_committed",
            "cache_archives",
            "cache_bytes",
        ):
            value = preflight_identity.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CurationError(
                    f"English near-dedup operational preflight identity {field} is invalid"
                )
        if (
            preflight_identity["candidate_pairs_total"]
            > json.loads(DEFAULT_ENGLISH_NEAR_CONFIG.read_text(encoding="utf-8"))[
                "candidate_signature"
            ]["maximum_unique_candidate_pairs"]
            or preflight_identity["candidate_blocks_total"]
            != preflight_identity["candidate_blocks_committed"]
            or preflight_identity["phase_at_measurement"] != "refine"
            or preflight_identity["refinement_cursor_at_measurement"] is not None
        ):
            raise CurationError(
                "English near-dedup operational preflight candidate authority is invalid"
            )
        cursor = preflight_identity["candidate_cursor_at_measurement"]
        if cursor is not None and (
            not isinstance(cursor, dict)
            or set(cursor) != {"band", "key"}
            or not isinstance(cursor["band"], int)
            or isinstance(cursor["band"], bool)
            or cursor["band"] < 0
            or not isinstance(cursor["key"], str)
            or not cursor["key"]
            or any(character not in HEX64 for character in cursor["key"])
        ):
            raise CurationError(
                "English near-dedup operational preflight candidate cursor is invalid"
            )

        production_config = json.loads(
            DEFAULT_ENGLISH_NEAR_CONFIG.read_text(encoding="utf-8")
        )
        thresholds = evidence.get("thresholds")
        if thresholds != production_config.get("operational_preflight"):
            raise CurationError(
                "English near-dedup operational preflight thresholds mismatch"
            )
        sample = evidence.get("sample")
        if not isinstance(sample, dict) or set(sample) != {
            "algorithm",
            "seed",
            "requested_pairs",
            "expected_pairs",
            "measured_pairs",
            "sample_pairs_sha256",
            "accepted_pairs",
            "sample_limited_by_total_candidates",
        }:
            raise CurationError(
                "English near-dedup operational preflight sample schema is incomplete"
            )
        total_candidates = preflight_identity["candidate_pairs_total"]
        expected_pairs = min(int(thresholds["requested_pairs"]), total_candidates)
        expected_sample = {
            "algorithm": thresholds["sampling_algorithm"],
            "seed": thresholds["sampling_seed"],
            "requested_pairs": thresholds["requested_pairs"],
            "expected_pairs": expected_pairs,
            "measured_pairs": expected_pairs,
            "sample_limited_by_total_candidates": (
                total_candidates < int(thresholds["requested_pairs"])
            ),
        }
        for field, expected in expected_sample.items():
            if sample.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight sample {field} mismatch"
                )
        parse_hex_digest(sample.get("sample_pairs_sha256"), "operational preflight sample sha256")
        accepted_pairs = sample.get("accepted_pairs")
        if (
            not isinstance(accepted_pairs, int)
            or isinstance(accepted_pairs, bool)
            or not 0 <= accepted_pairs <= expected_pairs
        ):
            raise CurationError(
                "English near-dedup operational preflight accepted-pair count is invalid"
            )

        measurements = evidence.get("measurements")
        measurement_fields = {
            "sample_elapsed_seconds",
            "measurement_batches",
            "measurement_batch_size",
            "refinement_pairs_per_second",
            "candidate_pairs_total",
            "projected_refinement_seconds",
            "measurement_sqlite_growth_bytes",
            "measurement_sqlite_bytes_per_pair",
            "projected_additional_refinement_sqlite_bytes",
            "projected_additional_with_safety_bytes",
            "required_filesystem_free_bytes",
            "union_parent_item_bytes",
            "union_parent_array_projected_bytes",
            "union_parent_array_with_safety_bytes",
            "union_projected_peak_process_rss_bytes",
            "resources_before",
            "resources_after",
        }
        if not isinstance(measurements, dict) or set(measurements) != measurement_fields:
            raise CurationError(
                "English near-dedup operational preflight measurement schema is incomplete"
            )
        batch_size = measurements.get("measurement_batch_size")
        elapsed = measurements.get("sample_elapsed_seconds")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
            or measurements.get("measurement_batches")
            != (math.ceil(expected_pairs / batch_size) if expected_pairs else 0)
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise CurationError(
                "English near-dedup operational preflight batch/time evidence is invalid"
            )
        expected_rate = (
            None if expected_pairs == 0 else round(expected_pairs / float(elapsed), 6)
        )
        if total_candidates > 0 and (expected_rate is None or expected_rate <= 0.0):
            raise CurationError(
                "English near-dedup operational preflight rate is invalid"
            )
        expected_projected_seconds = (
            0.0
            if total_candidates == 0
            else round(total_candidates / float(expected_rate), 3)
        )
        measured_growth = measurements.get("measurement_sqlite_growth_bytes")
        if (
            not isinstance(measured_growth, int)
            or isinstance(measured_growth, bool)
            or measured_growth < 0
        ):
            raise CurationError(
                "English near-dedup operational preflight SQLite growth is invalid"
            )
        expected_bytes_per_pair = (
            measured_growth / expected_pairs if expected_pairs else 0.0
        )
        expected_growth = math.ceil(expected_bytes_per_pair * total_candidates)
        expected_safety = math.ceil(
            expected_growth
            * int(thresholds["disk_projection_safety_numerator"])
            / int(thresholds["disk_projection_safety_denominator"])
        )
        expected_required_free = expected_safety + int(
            thresholds["minimum_post_refinement_free_bytes"]
        )
        formula_values = {
            "candidate_pairs_total": total_candidates,
            "refinement_pairs_per_second": expected_rate,
            "projected_refinement_seconds": expected_projected_seconds,
            "measurement_sqlite_bytes_per_pair": round(expected_bytes_per_pair, 6),
            "projected_additional_refinement_sqlite_bytes": expected_growth,
            "projected_additional_with_safety_bytes": expected_safety,
            "required_filesystem_free_bytes": expected_required_free,
        }
        for field, expected in formula_values.items():
            if measurements.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight formula {field} mismatch"
                )
        for resource_name in ("resources_before", "resources_after"):
            resources = measurements.get(resource_name)
            if not isinstance(resources, dict) or set(resources) != {
                "filesystem_total_bytes",
                "filesystem_free_bytes",
                "sqlite_state_bytes",
                "refinement_cache_bytes",
                "peak_process_rss_bytes",
            } or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in resources.values()
            ):
                raise CurationError(
                    f"English near-dedup operational preflight {resource_name} is invalid"
                )
            if (
                resources["filesystem_free_bytes"] > resources["filesystem_total_bytes"]
                or resources["refinement_cache_bytes"] != preflight_identity["cache_bytes"]
            ):
                raise CurationError(
                    f"English near-dedup operational preflight {resource_name} is inconsistent"
                )
        before = measurements["resources_before"]
        after = measurements["resources_after"]
        item_size = array.array("Q").itemsize
        projected_parent = (mapping_records + 1) * item_size
        projected_parent_with_safety = math.ceil(
            projected_parent
            * int(thresholds["union_parent_memory_safety_numerator"])
            / int(thresholds["union_parent_memory_safety_denominator"])
        )
        union_formulas = {
            "union_parent_item_bytes": item_size,
            "union_parent_array_projected_bytes": projected_parent,
            "union_parent_array_with_safety_bytes": projected_parent_with_safety,
            "union_projected_peak_process_rss_bytes": (
                before["peak_process_rss_bytes"] + projected_parent_with_safety
            ),
        }
        for field, expected in union_formulas.items():
            if measurements.get(field) != expected:
                raise CurationError(
                    f"English near-dedup operational preflight union formula {field} mismatch"
                )
        if (
            after["peak_process_rss_bytes"] < before["peak_process_rss_bytes"]
            or (
                total_candidates >= int(thresholds["requested_pairs"])
                and (
                    expected_rate is None
                    or expected_rate
                    < float(thresholds["minimum_production_pairs_per_second"])
                )
            )
            or expected_projected_seconds
            > int(thresholds["maximum_projected_refinement_seconds"])
            or after["peak_process_rss_bytes"]
            > int(thresholds["maximum_peak_process_rss_bytes"])
            or union_formulas["union_projected_peak_process_rss_bytes"]
            > int(thresholds["maximum_peak_process_rss_bytes"])
            or after["filesystem_free_bytes"] < expected_required_free
        ):
            raise CurationError(
                "English near-dedup operational preflight no longer passes resource gates"
            )
        return evidence

    def _validate_english_near_artifact(self) -> dict[str, Any] | None:
        """Validate the complete sibling mapping/manifest/checksum contract."""
        mapping_path = self.english_near_clusters
        if mapping_path is None:
            return None
        if (
            mapping_path.name != "clusters.jsonl.zst"
            or not mapping_path.is_file()
            or mapping_path.is_symlink()
        ):
            raise CurationError(
                "English near-dedup mapping must be a regular clusters.jsonl.zst file"
            )
        dataset_root = self.root.resolve(strict=True)
        publication_root = mapping_path.parent.resolve(strict=True)
        if not publication_root.is_relative_to(dataset_root):
            raise CurationError(
                "English near-dedup five-file publication must be under the dataset root"
            )
        publication_relative = publication_root.relative_to(dataset_root).as_posix()
        if not publication_relative:
            raise CurationError(
                "English near-dedup publication cannot be the dataset root"
            )
        manifest_path = publication_root / "manifest.json"
        checksum_path = publication_root / "manifest.sha256"
        manifest_raw, manifest = self._json_object(
            manifest_path, label="English near-dedup manifest"
        )
        if not checksum_path.is_file() or checksum_path.is_symlink():
            raise CurationError(
                f"Missing or unsafe English near-dedup manifest checksum: {checksum_path}"
            )
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        expected_checksum = f"{manifest_sha}  manifest.json\n".encode("ascii")
        checksum_raw = checksum_path.read_bytes()
        if checksum_raw != expected_checksum:
            raise CurationError("English near-dedup manifest.sha256 mismatch")
        if (
            manifest.get("manifest_version") != 1
            or manifest.get("mapping_record_version") != 1
            or manifest.get("production_ready") is not True
        ):
            raise CurationError("English near-dedup manifest is not production-ready v1")

        mapping = manifest.get("mapping")
        if not isinstance(mapping, dict):
            raise CurationError("English near-dedup mapping identity is missing")
        parse_hex_digest(mapping.get("sha256"), "English near mapping sha256")
        records = mapping.get("records")
        byte_count = mapping.get("bytes")
        if (
            mapping.get("path") != "clusters.jsonl.zst"
            or mapping.get("singleton_clusters_included") is not True
            or not isinstance(records, int)
            or isinstance(records, bool)
            or records <= 0
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
        ):
            raise CurationError("Invalid English near-dedup mapping identity")
        if mapping_path.stat().st_size != byte_count:
            raise CurationError("English near-dedup mapping byte-size mismatch")
        if file_sha256(mapping_path) != mapping["sha256"]:
            raise CurationError("English near-dedup mapping checksum mismatch")
        actual_records = 0
        for row in iter_jsonl(mapping_path):
            if set(row) != {"doc_id", "cluster_id"}:
                raise CurationError("Invalid English near-dedup mapping row schema")
            parse_hex_digest(row.get("doc_id"), "English near mapping doc_id")
            if not isinstance(row.get("cluster_id"), str) or not row["cluster_id"]:
                raise CurationError("Invalid English near-dedup cluster_id")
            actual_records += 1
        if actual_records != records:
            raise CurationError(
                "English near-dedup mapping row-count mismatch: "
                f"{actual_records} != {records}"
            )

        identity = manifest.get("identity")
        required_identity = {
            "format_version",
            "builder_sha256",
            "config_file_sha256",
            "config_sha256",
            "curation_policy_sha256",
            "preprocess_manifest_sha256",
            "benchmark_guard_sha256",
            "report_inventory_sha256",
            "report_count",
            "source_manifests",
            "collection_completeness",
            "calibration_evidence",
            "runtime",
        }
        if not isinstance(identity, dict) or set(identity) != required_identity:
            raise CurationError("English near-dedup manifest identity is incomplete")
        config_raw = DEFAULT_ENGLISH_NEAR_CONFIG.read_bytes()
        config = json.loads(config_raw)
        config_sha = hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in config.items() if not key.startswith("_")}
            )
        ).hexdigest()
        expected_identity_fields = {
            "format_version": 1,
            "builder_sha256": file_sha256(ENGLISH_NEAR_BUILDER),
            "config_file_sha256": hashlib.sha256(config_raw).hexdigest(),
            "config_sha256": config_sha,
            "curation_policy_sha256": self.policy_sha,
            "preprocess_manifest_sha256": file_sha256(
                self.staging_root / "PREPROCESS_MANIFEST.json"
            ),
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "source_manifests": self._current_english_near_source_manifests(),
        }
        for field, expected in expected_identity_fields.items():
            if identity.get(field) != expected:
                raise CurationError(
                    f"English near-dedup manifest identity {field} mismatch"
                )
        self._validate_english_near_runtime(identity.get("runtime"))

        collection = identity.get("collection_completeness")
        current_collection = self._current_english_near_collection_evidence()
        if not isinstance(collection, dict) or set(collection) != {
            "evidence_version",
            "quota_config_path",
            "quota_config_sha256",
            "quota_record_inventory_sha256",
            "quota_records",
            "buckets",
        }:
            raise CurationError("English near-dedup collection evidence is incomplete")
        if not isinstance(collection.get("quota_config_path"), str) or not collection[
            "quota_config_path"
        ]:
            raise CurationError("English near-dedup quota config path evidence is missing")
        for field, expected in current_collection.items():
            if collection.get(field) != expected:
                raise CurationError(
                    f"English near-dedup collection evidence {field} mismatch"
                )

        expected_reports = []
        for relative, report_path, report_sha, report in self.report_inventory:
            if report["bucket"] not in ENGLISH_BUCKETS:
                continue
            archive_path = self.root / report["archive"]
            fingerprint_path = self.staging_root / report["fingerprint_file"]
            if file_sha256(report_path) != report_sha:
                raise CurationError(f"English near-dedup report changed: {relative}")
            if (
                self._verified_raw_archive_sha256.get(report["archive"])
                != report["archive_sha256"]
            ):
                raise CurationError(
                    f"English near-dedup raw archive checksum mismatch: {archive_path}"
                )
            if file_sha256(fingerprint_path) != report["fingerprint_sha256"]:
                raise CurationError(
                    f"English near-dedup fingerprint checksum mismatch: {fingerprint_path}"
                )
            expected_reports.append(
                {
                    "report_path": relative,
                    "report_sha256": report_sha,
                    "archive": report["archive"],
                    "archive_sha256": report["archive_sha256"],
                    "fingerprint_file": report["fingerprint_file"],
                    "fingerprint_sha256": report["fingerprint_sha256"],
                    "documents": report["documents"],
                }
            )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {
            "report_inventory_sha256",
            "reports",
        }:
            raise CurationError("English near-dedup report evidence is incomplete")
        if inputs.get("reports") != expected_reports:
            raise CurationError("English near-dedup report inventory mismatch")
        projected_reports = [
            {
                "path": item["report_path"],
                "sha256": item["report_sha256"],
                "fingerprint_sha256": item["fingerprint_sha256"],
            }
            for item in expected_reports
        ]
        report_inventory_sha = hashlib.sha256(
            canonical_json_bytes(projected_reports)
        ).hexdigest()
        if (
            inputs.get("report_inventory_sha256") != report_inventory_sha
            or identity.get("report_inventory_sha256") != report_inventory_sha
            or identity.get("report_count") != len(expected_reports)
        ):
            raise CurationError("English near-dedup report identity mismatch")
        expected_documents = sum(int(item["documents"]) for item in expected_reports)
        if expected_documents != records:
            raise CurationError("English near-dedup mapping/report document mismatch")
        self._validate_english_near_calibration_evidence(
            identity["calibration_evidence"],
            near_identity=identity,
            expected_reports=expected_reports,
        )
        operational_preflight = self._validate_english_near_operational_preflight(
            manifest.get("refinement_operational_preflight"),
            publication_root=publication_root,
            near_identity=identity,
            mapping_records=records,
        )

        algorithm = manifest.get("algorithm")
        if (
            not isinstance(algorithm, dict)
            or algorithm.get("name") != ENGLISH_NEAR_ALGORITHM
            or algorithm.get("config_file_sha256") != identity["config_file_sha256"]
            or algorithm.get("config_sha256") != identity["config_sha256"]
            or algorithm.get("raw_text_candidate_pass") is not True
            or algorithm.get("full_shingle_refinement") is not True
            or algorithm.get("posting_overflow_action")
            != "fail_closed_without_truncation"
        ):
            raise CurationError("English near-dedup algorithm evidence mismatch")
        audit = manifest.get("completeness_and_leakage_audit")
        audit_fields = {
            "english_documents_inventory",
            "english_documents_mapped",
            "mapping_missing_documents",
            "mapping_unknown_documents",
            "mapping_duplicate_documents",
            "clusters",
            "singleton_clusters",
            "cross_source_clusters",
            "normalized_hashes_in_multiple_clusters",
            "invalid_cluster_roots",
        }
        if not isinstance(audit, dict) or set(audit) != audit_fields:
            raise CurationError("English near-dedup completeness audit is incomplete")
        if any(
            not isinstance(audit[field], int) or isinstance(audit[field], bool)
            for field in audit_fields
        ):
            raise CurationError("English near-dedup completeness audit has invalid counts")
        if (
            audit["english_documents_inventory"] != records
            or audit["english_documents_mapped"] != records
            or any(
                audit[field] != 0
                for field in (
                    "mapping_missing_documents",
                    "mapping_unknown_documents",
                    "mapping_duplicate_documents",
                    "normalized_hashes_in_multiple_clusters",
                    "invalid_cluster_roots",
                )
            )
        ):
            raise CurationError("English near-dedup completeness/leakage audit failed")
        clusters = audit["clusters"]
        singletons = audit["singleton_clusters"]
        cross_source = audit["cross_source_clusters"]
        if (
            clusters < 1
            or clusters > records
            or singletons < 0
            or singletons > clusters
            or 2 * clusters - singletons > records
            or cross_source < 0
            or cross_source > clusters
        ):
            raise CurationError("English near-dedup cluster audit is internally impossible")
        if manifest.get("database_integrity_check") != "ok":
            raise CurationError("English near-dedup database integrity audit failed")

        return {
            "contract_version": ENGLISH_NEAR_CONTRACT_VERSION,
            "publication_root": publication_relative,
            "manifest": {
                "path": f"{publication_relative}/manifest.json",
                "sha256": manifest_sha,
                "bytes": len(manifest_raw),
                "sidecar_path": f"{publication_relative}/manifest.sha256",
                "sidecar_sha256": hashlib.sha256(checksum_raw).hexdigest(),
            },
            "identity": identity,
            "mapping": mapping,
            "refinement_operational_preflight": operational_preflight,
            "database_integrity_check": "ok",
            "completeness_and_leakage_audit": audit,
        }

    def __enter__(self) -> "CurationBuilder":
        self.output.mkdir(parents=True, exist_ok=True)
        if self.output.is_symlink() or not self.output.is_dir():
            raise CurationError("Curation output must be a real directory")
        self.work.mkdir(parents=True, exist_ok=True)
        if self.work.is_symlink() or not self.work.is_dir():
            raise CurationError("Curation work path must be a real directory")
        self.sqlite_temp.mkdir(parents=True, exist_ok=True)
        if self.sqlite_temp.is_symlink() or not self.sqlite_temp.is_dir():
            raise CurationError("SQLite temp path must be a real directory")
        if len(
            {
                self.output.stat().st_dev,
                self.work.stat().st_dev,
                self.sqlite_temp.stat().st_dev,
            }
        ) != 1:
            raise CurationError(
                "Output, curation database, and SQLite temp paths must share one filesystem"
            )
        frozen_mount = self.sqlite_runtime["journal_policy"]["mount"]
        for label, path in (
            ("output", self.output),
            ("database work", self.work),
            ("SQLite temp", self.sqlite_temp),
        ):
            if detect_output_mount(path) != frozen_mount:
                raise CurationError(
                    f"{label} mount differs from the frozen curation mount identity"
                )
        current_runtime = {
            "sqlite_version": sqlite3.sqlite_version,
            "journal_policy": select_sqlite_journal_policy(
                detect_output_mount(self.output),
                self.sqlite_journal_mode,
            ),
        }
        if current_runtime != self.sqlite_runtime:
            raise CurationError(
                "Output mount or SQLite runtime changed before database open"
            )
        selected_mode = self.sqlite_runtime["journal_policy"]["selected_mode"]
        if selected_mode == "delete":
            unsafe_sidecars = [
                path
                for path in (
                    Path(f"{self.db_path}-wal"),
                    Path(f"{self.db_path}-shm"),
                )
                if path.exists()
            ]
            if unsafe_sidecars:
                raise CurationError(
                    "Rollback-journal curation output contains WAL sidecars; "
                    "refusing a possibly live or incompletely copied database: "
                    + ", ".join(str(path) for path in unsafe_sidecars)
                )
        nonempty_database = self.db_path.exists() and self.db_path.stat().st_size > 0
        self.connection = sqlite3.connect(self.db_path, timeout=120)
        try:
            self.connection.row_factory = sqlite3.Row
            escaped_temp = str(self.sqlite_temp.resolve()).replace("'", "''")
            self.connection.execute(
                f"PRAGMA temp_store_directory='{escaped_temp}'"
            )
            temp_row = self.connection.execute(
                "PRAGMA temp_store_directory"
            ).fetchone()
            if (
                temp_row is None
                or Path(str(temp_row[0])).resolve() != self.sqlite_temp.resolve()
            ):
                raise CurationError("SQLite refused the dedicated temp directory")
            self.connection.execute("PRAGMA synchronous=FULL")
            sidecar_limit = int(
                self.storage_contract["transaction_sidecar_limit_bytes"]
            )
            configured_limit = int(
                self.connection.execute(
                    f"PRAGMA journal_size_limit={sidecar_limit}"
                ).fetchone()[0]
            )
            if configured_limit != sidecar_limit:
                raise CurationError("SQLite refused the curation journal-size limit")
            self.connection.execute("PRAGMA wal_autocheckpoint=1024")
            if int(
                self.connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            ) != 1024:
                raise CurationError("SQLite refused the WAL autocheckpoint bound")
            self.connection.execute(f"PRAGMA temp_store={SQLITE_TEMP_STORE}")
            if int(self.connection.execute("PRAGMA temp_store").fetchone()[0]) != 1:
                raise CurationError("SQLite refused file-backed temporary storage")
            self.connection.execute("PRAGMA cache_size=-524288")
            self.connection.execute("PRAGMA mmap_size=8589934592")
            self.connection.execute("PRAGMA foreign_keys=ON")
            current_mode = str(
                self.connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            metadata_exists = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
            ).fetchone() is not None
            if metadata_exists:
                if current_mode != selected_mode:
                    raise CurationError(
                        "Existing curation database journal mode mismatch: "
                        f"found {current_mode}, expected {selected_mode}"
                    )
                # Validate the frozen mount/mode identity before any schema or
                # metadata write. In particular, never convert a copied WAL
                # database to rollback journaling on a network filesystem.
                self._initialize_or_validate_identity()
                self._create_schema()
                self._migrate_database_if_needed()
            else:
                if nonempty_database:
                    raise CurationError(
                        "Non-empty curation database has no identity metadata"
                    )
                configured_mode = str(
                    self.connection.execute(
                        f"PRAGMA journal_mode={selected_mode.upper()}"
                    ).fetchone()[0]
                ).casefold()
                if configured_mode != selected_mode:
                    raise CurationError(
                        "SQLite refused the selected journal mode: "
                        f"found {configured_mode}, expected {selected_mode}"
                    )
                self._create_schema()
                self._initialize_or_validate_identity()
            self._assert_no_storage_violation()
            self._sync_audit_files()
            return self
        except BaseException:
            self.connection.close()
            self.connection = None
            raise

    def __exit__(self, *_args: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @property
    def db(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("CurationBuilder must be used as a context manager")
        return self.connection

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS archives (
                report_path TEXT PRIMARY KEY,
                report_sha256 TEXT NOT NULL,
                archive TEXT NOT NULL UNIQUE,
                bucket TEXT NOT NULL,
                fingerprint_file TEXT NOT NULL,
                fingerprint_sha256 TEXT NOT NULL,
                documents INTEGER NOT NULL,
                tokens INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS documents (
                doc_id BLOB PRIMARY KEY,
                bucket TEXT NOT NULL,
                archive TEXT NOT NULL,
                manifest_index INTEGER NOT NULL,
                member_path TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                content_hash BLOB NOT NULL,
                normalized_hash BLOB NOT NULL,
                final_cluster BLOB NOT NULL,
                source_group BLOB NOT NULL,
                canonical_rank BLOB NOT NULL,
                selection_rank BLOB NOT NULL,
                UNIQUE(archive, manifest_index),
                UNIQUE(archive, member_path)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS reasons (
                doc_id BLOB NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY(doc_id, reason),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS near_map (
                doc_id BLOB PRIMARY KEY,
                cluster BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS benchmark_content_clusters (
                content_hash BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS benchmark_final_clusters (
                final_cluster BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS exact_choice (
                content_hash BLOB PRIMARY KEY,
                canonical_rank BLOB NOT NULL,
                canonical_doc_id BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS final_choice (
                final_cluster BLOB PRIMARY KEY,
                canonical_rank BLOB NOT NULL,
                canonical_doc_id BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS canonical_map (
                doc_id BLOB PRIMARY KEY,
                canonical_doc_id BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS groups (
                group_id BLOB PRIMARY KEY,
                split TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS selected (
                doc_id BLOB PRIMARY KEY,
                split TEXT NOT NULL,
                selected_tokens INTEGER NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS output_archives (
                archive TEXT PRIMARY KEY,
                decision_file TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL,
                records INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS phase_progress (
                subphase TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('running','complete')),
                cursor_json TEXT NOT NULL,
                processed_rows INTEGER NOT NULL CHECK(processed_rows >= 0),
                processed_tokens INTEGER NOT NULL CHECK(processed_tokens >= 0),
                committed_batches INTEGER NOT NULL CHECK(committed_batches >= 0),
                details_json TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS storage_metrics (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                preflight_json TEXT NOT NULL,
                maximum_database_bytes INTEGER NOT NULL CHECK(maximum_database_bytes >= 0),
                maximum_journal_bytes INTEGER NOT NULL CHECK(maximum_journal_bytes >= 0),
                maximum_wal_bytes INTEGER NOT NULL CHECK(maximum_wal_bytes >= 0),
                minimum_free_bytes INTEGER NOT NULL CHECK(minimum_free_bytes >= 0),
                committed_transactions INTEGER NOT NULL CHECK(committed_transactions >= 0),
                maximum_transaction_rows INTEGER NOT NULL CHECK(maximum_transaction_rows >= 0),
                violation_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS durable_counts (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                archives INTEGER NOT NULL CHECK(archives >= 0),
                documents INTEGER NOT NULL CHECK(documents >= 0),
                selected_documents INTEGER NOT NULL CHECK(selected_documents >= 0),
                output_archives INTEGER NOT NULL CHECK(output_archives >= 0)
            );
            INSERT OR IGNORE INTO durable_counts(
                singleton, archives, documents, selected_documents, output_archives
            ) VALUES (1, 0, 0, 0, 0);
            """
        )
        storage_columns = {
            str(row[1]) for row in self.db.execute("PRAGMA table_info(storage_metrics)")
        }
        if "violation_json" not in storage_columns:
            self.db.execute(
                "ALTER TABLE storage_metrics ADD COLUMN violation_json "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        self.db.commit()

    def _migrate_database_if_needed(self) -> None:
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key='database_version'"
        ).fetchone()
        version = json.loads(row[0]) if row is not None else None
        if version == DB_VERSION:
            return
        if version not in (1, 2, 3):
            raise CurationError(
                f"Unsupported curation database version {version!r}; expected {DB_VERSION}"
            )
        storage_row = self.db.execute(
            "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
        ).fetchone()
        migrated_preflight: dict[str, Any] | None = None
        if storage_row is not None:
            legacy_preflight_raw = str(storage_row[0])
            migrated_preflight = self._checked_json_object(
                legacy_preflight_raw, label="legacy curation storage preflight"
            )
            if version not in (2, 3):
                raise CurationError(
                    "Schema-v1 curation database unexpectedly has storage preflight state"
                )
            self._validate_legacy_storage_preflight(
                migrated_preflight, version=version
            )
            # Re-measure instead of relabeling the old 1 KiB/document pass as
            # satisfying the stronger 3 KiB/document contract.  The old
            # evidence remains cryptographically linked for auditability.
            migrated_preflight = self._build_storage_preflight(
                reason="database_contract_migration",
                from_database_version=version,
                previous_preflight_sha256=hashlib.sha256(
                    legacy_preflight_raw.encode("utf-8")
                ).hexdigest(),
            )
            self._validate_storage_preflight(migrated_preflight)
        # Database v1 made each public phase transition in the same atomic
        # transaction as its entire (unbounded) phase.  Therefore its main
        # phase is already a sufficient completion marker: there can be no
        # committed partial canonicalization/selection state to reconstruct.
        # New work after this migration uses the current bounded subphase
        # journal; the legacy public phase remains the only trusted cursor.
        with self.db:
            actual_counts = self.db.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM archives),
                    (SELECT COUNT(*) FROM documents),
                    (SELECT COUNT(*) FROM selected),
                    (SELECT COUNT(*) FROM output_archives)
                """
            ).fetchone()
            self.db.execute(
                """
                UPDATE durable_counts SET
                    archives=?, documents=?, selected_documents=?, output_archives=?
                WHERE singleton=1
                """,
                tuple(int(value) for value in actual_counts),
            )
            self.db.execute(
                "UPDATE metadata SET value=? WHERE key='database_version'",
                (json.dumps(DB_VERSION),),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (
                    "curation_storage_contract",
                    json.dumps(self.storage_contract, sort_keys=True),
                ),
            )
            if migrated_preflight is not None:
                self.db.execute(
                    "UPDATE storage_metrics SET preflight_json=? WHERE singleton=1",
                    (canonical_json_bytes(migrated_preflight).decode("utf-8"),),
                )
            self._event(
                "database_migrated",
                {
                    "from_version": version,
                    "to_version": DB_VERSION,
                    "legacy_phase": self._phase(),
                    "legacy_completed_phases_are_atomic": True,
                },
            )

    def _initialize_or_validate_identity(self) -> None:
        encoded = {key: json.dumps(value, sort_keys=True) for key, value in self.identity.items()}
        existing = dict(self.db.execute("SELECT key, value FROM metadata"))
        if not existing:
            with self.db:
                self.db.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    [("database_version", json.dumps(DB_VERSION)), ("phase", json.dumps("inventory")), *encoded.items()],
                )
                self._event("initialized", {"reports": len(self.report_inventory)})
            return
        if json.loads(existing.get("database_version", "null")) not in (
            1,
            2,
            3,
            DB_VERSION,
        ):
            raise CurationError("Curation database version mismatch")
        existing_version = json.loads(existing.get("database_version", "null"))
        for key, value in encoded.items():
            if (
                existing_version == 1
                and key == "curation_storage_contract"
                and key not in existing
            ):
                continue
            if (
                existing_version == 2
                and key == "curation_storage_contract"
                and existing.get(key)
                == json.dumps(self.legacy_v2_storage_contract, sort_keys=True)
            ):
                continue
            if (
                existing_version == 3
                and key == "curation_storage_contract"
                and existing.get(key)
                == json.dumps(self.legacy_v3_storage_contract, sort_keys=True)
            ):
                continue
            if existing.get(key) != value:
                raise CurationError(f"Resume identity mismatch for {key}")

    def _phase(self) -> str:
        row = self.db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()
        return str(json.loads(row[0]))

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO events(event, payload) VALUES (?, ?)",
            (event, canonical_json_bytes(payload).decode("utf-8")),
        )

    def _advance(self, expected: str, new: str, payload: dict[str, Any]) -> None:
        if self._phase() != expected:
            raise CurationError(f"Cannot advance {self._phase()} as if it were {expected}")
        with self.db:
            self.db.execute("UPDATE metadata SET value=? WHERE key='phase'", (json.dumps(new),))
            self._event(new, payload)
        self._sync_audit_files()

    @staticmethod
    def _checked_json_object(value: str, *, label: str) -> dict[str, Any]:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise CurationError(f"{label} must be a JSON object")
        return payload

    def _progress(self, subphase: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """
            SELECT status, cursor_json, processed_rows, processed_tokens,
                   committed_batches, details_json
            FROM phase_progress WHERE subphase=?
            """,
            (subphase,),
        ).fetchone()
        if row is None:
            return None
        cursor = self._checked_json_object(str(row[1]), label=f"{subphase} cursor")
        details = self._checked_json_object(str(row[5]), label=f"{subphase} details")
        result = {
            "subphase": subphase,
            "status": str(row[0]),
            "cursor": cursor,
            "processed_rows": int(row[2]),
            "processed_tokens": int(row[3]),
            "committed_batches": int(row[4]),
            "details": details,
        }
        if result["status"] not in ("running", "complete") or any(
            result[field] < 0
            for field in ("processed_rows", "processed_tokens", "committed_batches")
        ):
            raise CurationError(f"Invalid durable progress for {subphase}")
        return result

    def _start_subphase(
        self,
        subphase: str,
        *,
        cursor: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = self._progress(subphase)
        if progress is not None:
            return progress
        with self.db:
            self.db.execute(
                """
                INSERT INTO phase_progress(
                    subphase, status, cursor_json, processed_rows,
                    processed_tokens, committed_batches, details_json
                ) VALUES (?, 'running', ?, 0, 0, 0, ?)
                """,
                (
                    subphase,
                    canonical_json_bytes(cursor).decode("utf-8"),
                    canonical_json_bytes(details or {}).decode("utf-8"),
                ),
            )
        self._sync_checkpoint_file()
        progress = self._progress(subphase)
        assert progress is not None
        return progress

    @staticmethod
    def _size_or_zero(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def _storage_snapshot(self) -> dict[str, int]:
        usage = shutil.disk_usage(self.work)
        temp_usage = shutil.disk_usage(self.sqlite_temp)
        return {
            "database_bytes": self._size_or_zero(self.db_path),
            "journal_bytes": self._size_or_zero(Path(f"{self.db_path}-journal")),
            "wal_bytes": self._size_or_zero(Path(f"{self.db_path}-wal")),
            "free_bytes": int(usage.free),
            "filesystem_total_bytes": int(usage.total),
            "sqlite_temp_free_bytes": int(temp_usage.free),
            "sqlite_temp_total_bytes": int(temp_usage.total),
        }

    def _storage_projection(self, expected_documents: int) -> tuple[int, int]:
        projected = math.ceil(
            expected_documents
            * int(
                self.storage_contract[
                    "projected_additional_bytes_per_document"
                ]
            )
            * int(self.storage_contract["disk_safety_numerator"])
            / int(self.storage_contract["disk_safety_denominator"])
        )
        required = (
            projected
            + int(self.storage_contract["transaction_sidecar_limit_bytes"])
            + int(self.storage_contract["minimum_free_bytes_after_projection"])
        )
        return projected, required

    def _build_storage_preflight(
        self,
        *,
        reason: str,
        from_database_version: int | None = None,
        previous_preflight_sha256: str | None = None,
    ) -> dict[str, Any]:
        expected_documents = sum(
            int(report["documents"])
            for _relative, _path, _checksum, report in self.report_inventory
        )
        documents = int(
            self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        if documents > expected_documents:
            raise CurationError("Document count exceeds the frozen inventory")
        snapshot = self._storage_snapshot()
        projected, required = self._storage_projection(expected_documents)
        if min(snapshot["free_bytes"], snapshot["sqlite_temp_free_bytes"]) < required:
            raise CurationError(
                "Insufficient curation scratch space: "
                f"free={snapshot['free_bytes']:,}, "
                f"sqlite_temp_free={snapshot['sqlite_temp_free_bytes']:,}, "
                f"required={required:,}, documents={expected_documents:,}"
            )
        measurement_reason: dict[str, Any] = {"reason": reason}
        if reason == "database_contract_migration":
            if (
                from_database_version not in (2, 3)
                or not isinstance(previous_preflight_sha256, str)
                or len(previous_preflight_sha256) != 64
            ):
                raise CurationError("Invalid storage-preflight migration provenance")
            measurement_reason.update(
                from_database_version=from_database_version,
                previous_preflight_sha256=previous_preflight_sha256,
            )
        elif reason != "initial":
            raise CurationError("Invalid storage-preflight measurement reason")
        return {
            "preflight_version": CURATION_STORAGE_PREFLIGHT_VERSION,
            "status": "pass",
            "contract": self.storage_contract,
            "measurement_reason": measurement_reason,
            "documents_expected": expected_documents,
            "documents_committed_at_measurement": documents,
            "database_bytes_at_measurement": snapshot["database_bytes"],
            "free_bytes_at_measurement": snapshot["free_bytes"],
            "filesystem_total_bytes_at_measurement": snapshot[
                "filesystem_total_bytes"
            ],
            "sqlite_temp_relative_path": self.storage_contract[
                "sqlite_temp_relative_path"
            ],
            "sqlite_temp_free_bytes_at_measurement": snapshot[
                "sqlite_temp_free_bytes"
            ],
            "sqlite_temp_total_bytes_at_measurement": snapshot[
                "sqlite_temp_total_bytes"
            ],
            "sqlite_temp_same_device_as_database": (
                self.sqlite_temp.stat().st_dev == self.work.stat().st_dev
            ),
            "projected_additional_bytes_with_safety": projected,
            "required_free_bytes_at_measurement": required,
        }

    def _validate_storage_preflight(
        self, preflight: dict[str, Any]
    ) -> None:
        expected_keys = {
            "preflight_version",
            "status",
            "contract",
            "measurement_reason",
            "documents_expected",
            "documents_committed_at_measurement",
            "database_bytes_at_measurement",
            "free_bytes_at_measurement",
            "filesystem_total_bytes_at_measurement",
            "sqlite_temp_relative_path",
            "sqlite_temp_free_bytes_at_measurement",
            "sqlite_temp_total_bytes_at_measurement",
            "sqlite_temp_same_device_as_database",
            "projected_additional_bytes_with_safety",
            "required_free_bytes_at_measurement",
        }
        if set(preflight) != expected_keys:
            raise CurationError("Curation storage preflight schema mismatch")
        reason = preflight.get("measurement_reason")
        if not isinstance(reason, dict):
            raise CurationError("Curation storage preflight reason is missing")
        if reason.get("reason") == "initial":
            if reason != {"reason": "initial"}:
                raise CurationError("Initial storage preflight reason is malformed")
        elif reason.get("reason") == "database_contract_migration":
            if set(reason) != {
                "reason",
                "from_database_version",
                "previous_preflight_sha256",
            } or reason.get("from_database_version") not in (2, 3):
                raise CurationError("Migrated storage preflight reason is malformed")
            digest = reason.get("previous_preflight_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in HEX64 for character in digest)
            ):
                raise CurationError("Migrated storage preflight digest is invalid")
        else:
            raise CurationError("Unknown storage preflight measurement reason")
        integer_fields = (
            "documents_expected",
            "documents_committed_at_measurement",
            "database_bytes_at_measurement",
            "free_bytes_at_measurement",
            "filesystem_total_bytes_at_measurement",
            "sqlite_temp_free_bytes_at_measurement",
            "sqlite_temp_total_bytes_at_measurement",
            "projected_additional_bytes_with_safety",
            "required_free_bytes_at_measurement",
        )
        if any(
            not isinstance(preflight.get(field), int)
            or isinstance(preflight.get(field), bool)
            or int(preflight[field]) < 0
            for field in integer_fields
        ):
            raise CurationError("Curation storage preflight counters are invalid")
        expected_documents = sum(
            int(report["documents"])
            for _relative, _path, _checksum, report in self.report_inventory
        )
        projected, required = self._storage_projection(expected_documents)
        if (
            preflight.get("preflight_version")
            != CURATION_STORAGE_PREFLIGHT_VERSION
            or preflight.get("status") != "pass"
            or preflight.get("contract") != self.storage_contract
            or preflight["documents_expected"] != expected_documents
            or preflight["documents_committed_at_measurement"]
            > expected_documents
            or preflight["sqlite_temp_relative_path"]
            != self.storage_contract["sqlite_temp_relative_path"]
            or preflight["sqlite_temp_same_device_as_database"] is not True
            or preflight["projected_additional_bytes_with_safety"] != projected
            or preflight["required_free_bytes_at_measurement"] != required
            or preflight["free_bytes_at_measurement"] < required
            or preflight["sqlite_temp_free_bytes_at_measurement"] < required
        ):
            raise CurationError("Curation storage preflight evidence mismatch")

    def _validate_legacy_storage_preflight(
        self, preflight: dict[str, Any], *, version: int
    ) -> None:
        expected_keys = {
            "preflight_version",
            "status",
            "contract",
            "documents_expected",
            "documents_committed_at_measurement",
            "database_bytes_at_measurement",
            "free_bytes_at_measurement",
            "filesystem_total_bytes_at_measurement",
            "projected_additional_bytes_with_safety",
            "required_free_bytes_at_measurement",
        }
        if version == 3:
            expected_keys.update(
                {
                    "sqlite_temp_relative_path",
                    "sqlite_temp_free_bytes_at_measurement",
                    "sqlite_temp_total_bytes_at_measurement",
                    "sqlite_temp_same_device_as_database",
                }
            )
        expected_contract = (
            self.legacy_v2_storage_contract
            if version == 2
            else self.legacy_v3_storage_contract
        )
        if (
            set(preflight) != expected_keys
            or preflight.get("preflight_version") != 1
            or preflight.get("status") != "pass"
            or preflight.get("contract") != expected_contract
        ):
            raise CurationError(
                "Legacy curation storage preflight is not an exact supported schema"
            )

    def _ensure_storage_preflight(self) -> dict[str, Any]:
        self._assert_no_storage_violation()
        row = self.db.execute(
            "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
        ).fetchone()
        if row is not None:
            preflight = self._checked_json_object(
                str(row[0]), label="curation storage preflight"
            )
            self._validate_storage_preflight(preflight)
            current = self._storage_snapshot()
            if min(current["free_bytes"], current["sqlite_temp_free_bytes"]) < int(
                self.storage_contract["minimum_free_bytes_after_projection"]
            ):
                raise CurationError(
                    "Curation scratch space fell below its durable free-space floor"
                )
            return preflight

        preflight = self._build_storage_preflight(reason="initial")
        self._validate_storage_preflight(preflight)
        snapshot = self._storage_snapshot()
        with self.db:
            self.db.execute(
                """
                INSERT INTO storage_metrics(
                    singleton, preflight_json, maximum_database_bytes,
                    maximum_journal_bytes, maximum_wal_bytes,
                    minimum_free_bytes, committed_transactions,
                    maximum_transaction_rows
                ) VALUES (1, ?, ?, ?, ?, ?, 0, 0)
                """,
                (
                    canonical_json_bytes(preflight).decode("utf-8"),
                    snapshot["database_bytes"],
                    snapshot["journal_bytes"],
                    snapshot["wal_bytes"],
                    snapshot["free_bytes"],
                ),
            )
            self._event("curation_storage_preflight_passed", preflight)
        self._sync_audit_files()
        return preflight

    def _record_transaction_metrics(self, transaction_rows: int) -> dict[str, int]:
        self._assert_no_storage_violation()
        if transaction_rows < 0 or transaction_rows > self.batch_size:
            raise CurationError(
                f"Unbounded curation transaction: {transaction_rows:,} rows "
                f"exceeds {self.batch_size:,}"
            )
        snapshot = self._storage_snapshot()
        sidecar_limit = int(
            self.storage_contract["transaction_sidecar_limit_bytes"]
        )
        if max(snapshot["journal_bytes"], snapshot["wal_bytes"]) > sidecar_limit:
            raise CurationError(
                "Curation SQLite sidecar exceeded its transaction bound; "
                "this run's batch size is frozen, so recover free space if needed "
                "and use a new output generation with a smaller --batch-size"
            )
        if snapshot["free_bytes"] < int(
            self.storage_contract["minimum_free_bytes_after_projection"]
        ):
            raise CurationError(
                "Curation scratch space fell below its durable free-space floor"
            )
        self.db.execute(
            """
            UPDATE storage_metrics SET
                maximum_database_bytes=MAX(maximum_database_bytes, ?),
                maximum_journal_bytes=MAX(maximum_journal_bytes, ?),
                maximum_wal_bytes=MAX(maximum_wal_bytes, ?),
                minimum_free_bytes=MIN(minimum_free_bytes, ?),
                committed_transactions=committed_transactions + 1,
                maximum_transaction_rows=MAX(maximum_transaction_rows, ?)
            WHERE singleton=1
            """,
            (
                snapshot["database_bytes"],
                snapshot["journal_bytes"],
                snapshot["wal_bytes"],
                snapshot["free_bytes"],
                transaction_rows,
            ),
        )
        if self.db.execute("SELECT changes()").fetchone()[0] != 1:
            raise CurationError("Curation storage preflight was not initialized")
        return snapshot

    def _storage_violation(self) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT violation_json FROM storage_metrics WHERE singleton=1"
        ).fetchone()
        if row is None:
            return {}
        return self._checked_json_object(
            str(row[0]), label="curation storage violation"
        )

    def _assert_no_storage_violation(self) -> None:
        violation = self._storage_violation()
        if violation:
            raise CurationError(
                "Curation storage contract was permanently violated for this "
                f"output generation: {violation}"
            )

    def _persist_storage_violation(
        self, *, code: str, evidence: dict[str, Any]
    ) -> None:
        payload = {"code": code, "evidence": evidence}
        with self.db:
            self.db.execute(
                """
                UPDATE storage_metrics SET violation_json=?
                WHERE singleton=1 AND violation_json='{}'
                """,
                (canonical_json_bytes(payload).decode("utf-8"),),
            )

    def _commit_bounded_batch(
        self,
        subphase: str,
        *,
        cursor: dict[str, Any],
        rows: int,
        tokens: int = 0,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = self._progress(subphase)
        if progress is None or progress["status"] != "running":
            raise CurationError(f"Cannot commit inactive subphase {subphase}")
        if rows < 1 or tokens < 0:
            raise CurationError(f"Invalid bounded batch counters for {subphase}")
        self._record_transaction_metrics(rows)
        next_details = progress["details"] if details is None else details
        self.db.execute(
            """
            UPDATE phase_progress SET
                cursor_json=?, processed_rows=processed_rows + ?,
                processed_tokens=processed_tokens + ?,
                committed_batches=committed_batches + 1,
                details_json=?
            WHERE subphase=? AND status='running'
            """,
            (
                canonical_json_bytes(cursor).decode("utf-8"),
                rows,
                tokens,
                canonical_json_bytes(next_details).decode("utf-8"),
                subphase,
            ),
        )
        if self.db.execute("SELECT changes()").fetchone()[0] != 1:
            raise CurationError(f"Lost durable progress row for {subphase}")
        self.db.commit()
        self._bound_wal_after_commit()
        self._sync_checkpoint_file()
        committed = self._progress(subphase)
        assert committed is not None
        self._after_bounded_commit(subphase, committed)
        return committed

    def _complete_subphase(
        self, subphase: str, *, details: dict[str, Any]
    ) -> dict[str, Any]:
        progress = self._progress(subphase)
        if progress is None:
            raise CurationError(f"Unknown subphase {subphase}")
        if progress["status"] == "complete":
            if progress["details"] != details:
                raise CurationError(f"Completed subphase evidence changed: {subphase}")
            return progress
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self._record_transaction_metrics(0)
            self.db.execute(
                """
                UPDATE phase_progress
                SET status='complete', details_json=?
                WHERE subphase=? AND status='running'
                """,
                (canonical_json_bytes(details).decode("utf-8"), subphase),
            )
            if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                raise CurationError(f"Cannot complete subphase {subphase}")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._bound_wal_after_commit()
        self._sync_checkpoint_file()
        completed = self._progress(subphase)
        assert completed is not None
        return completed

    def _bound_wal_after_commit(self) -> None:
        if self.sqlite_runtime["journal_policy"]["selected_mode"] != "wal":
            return
        wal = Path(f"{self.db_path}-wal")
        limit = int(self.storage_contract["transaction_sidecar_limit_bytes"])
        snapshot = self._storage_snapshot()
        with self.db:
            self.db.execute(
                """
                UPDATE storage_metrics SET
                    maximum_database_bytes=MAX(maximum_database_bytes, ?),
                    maximum_journal_bytes=MAX(maximum_journal_bytes, ?),
                    maximum_wal_bytes=MAX(maximum_wal_bytes, ?),
                    minimum_free_bytes=MIN(minimum_free_bytes, ?)
                WHERE singleton=1
                """,
                (
                    snapshot["database_bytes"],
                    snapshot["journal_bytes"],
                    snapshot["wal_bytes"],
                    snapshot["free_bytes"],
                ),
            )
        post_metrics_wal = self._size_or_zero(wal)
        exceeded = max(snapshot["wal_bytes"], post_metrics_wal) > limit
        exceeded_evidence = {
            "observed_wal_bytes": max(snapshot["wal_bytes"], post_metrics_wal),
            "transaction_sidecar_limit_bytes": limit,
        }
        # A committed batch that crossed the frozen bound permanently taints
        # this output generation.  Record that fact *before* attempting a WAL
        # checkpoint: a busy/failed checkpoint must not erase the evidence on
        # the next reopen after the WAL happens to shrink.
        if exceeded:
            self._persist_storage_violation(
                code="bounded_transaction_wal_exceeded",
                evidence=exceeded_evidence,
            )
        if post_metrics_wal > limit // 2:
            result = self._truncate_wal()
            if result is None or int(result[0]) != 0 or self._size_or_zero(wal) > limit:
                if not exceeded:
                    self._persist_storage_violation(
                        code="wal_checkpoint_failed",
                        evidence={
                            "checkpoint_result": (
                                None if result is None else [int(value) for value in result]
                            ),
                            "observed_wal_bytes": post_metrics_wal,
                            "transaction_sidecar_limit_bytes": limit,
                        },
                    )
                raise CurationError("Could not checkpoint curation WAL below its bound")
        if exceeded:
            raise CurationError(
                "A committed curation batch exceeded the frozen WAL bound; its "
                "cursor is safely resumable, but production must use a new output "
                "generation with a smaller --batch-size"
            )

    def _truncate_wal(self) -> sqlite3.Row | tuple[Any, ...] | None:
        """Checkpoint seam used by fault-injection tests."""
        return self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    def _after_bounded_commit(
        self, _subphase: str, _progress: dict[str, Any]
    ) -> None:
        """Fault-injection seam: committed state is durable before this hook."""

    def _checkpoint_payload(self) -> dict[str, Any]:
        count_row = self.db.execute(
            """
            SELECT archives, documents, selected_documents, output_archives
            FROM durable_counts WHERE singleton=1
            """
        ).fetchone()
        if count_row is None:
            raise CurationError("Missing durable curation counters")
        counts = {
            "archives": int(count_row[0]),
            "documents": int(count_row[1]),
            "selected_documents": int(count_row[2]),
            "output_archives": int(count_row[3]),
        }
        progress = []
        for row in self.db.execute(
            """
            SELECT subphase, status, cursor_json, processed_rows,
                   processed_tokens, committed_batches, details_json
            FROM phase_progress ORDER BY subphase
            """
        ):
            progress.append(
                {
                    "subphase": str(row[0]),
                    "status": str(row[1]),
                    "cursor": self._checked_json_object(
                        str(row[2]), label="checkpoint cursor"
                    ),
                    "processed_rows": int(row[3]),
                    "processed_tokens": int(row[4]),
                    "committed_batches": int(row[5]),
                    "details": self._checked_json_object(
                        str(row[6]), label="checkpoint details"
                    ),
                }
            )
        storage_row = self.db.execute(
            """
            SELECT preflight_json, maximum_database_bytes,
                   maximum_journal_bytes, maximum_wal_bytes,
                   minimum_free_bytes, committed_transactions,
                   maximum_transaction_rows, violation_json
            FROM storage_metrics WHERE singleton=1
            """
        ).fetchone()
        storage = None
        if storage_row is not None:
            storage = {
                "preflight": self._checked_json_object(
                    str(storage_row[0]), label="checkpoint storage preflight"
                ),
                "maximum_database_bytes": int(storage_row[1]),
                "maximum_journal_bytes": int(storage_row[2]),
                "maximum_wal_bytes": int(storage_row[3]),
                "minimum_free_bytes": int(storage_row[4]),
                "committed_transactions": int(storage_row[5]),
                "maximum_transaction_rows": int(storage_row[6]),
                "violation": self._checked_json_object(
                    str(storage_row[7]), label="checkpoint storage violation"
                ),
            }
        last_event = self.db.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM events"
        ).fetchone()[0]
        return {
            "checkpoint_version": 2,
            "database_version": DB_VERSION,
            "phase": self._phase(),
            "identity": self.identity,
            "last_event_sequence": int(last_event),
            "counts": counts,
            "subphases": progress,
            "storage": storage,
        }

    def _sync_checkpoint_file(self) -> None:
        if self.connection is not None:
            atomic_json(self.work / "CHECKPOINT.json", self._checkpoint_payload())

    def _validate_durable_counts_full(self) -> dict[str, int]:
        actual_row = self.db.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM archives),
                (SELECT COUNT(*) FROM documents),
                (SELECT COUNT(*) FROM selected),
                (SELECT COUNT(*) FROM output_archives)
            """
        ).fetchone()
        durable_row = self.db.execute(
            """
            SELECT archives, documents, selected_documents, output_archives
            FROM durable_counts WHERE singleton=1
            """
        ).fetchone()
        if durable_row is None:
            raise CurationError("Missing durable curation counters")
        names = ("archives", "documents", "selected_documents", "output_archives")
        actual = {name: int(value) for name, value in zip(names, actual_row)}
        durable = {name: int(value) for name, value in zip(names, durable_row)}
        if actual != durable:
            raise CurationError(
                f"Durable curation counter reconciliation failed: "
                f"actual={actual}, durable={durable}"
            )
        return actual

    def _sync_audit_files(self) -> None:
        if self.connection is None:
            return
        events = [
            {"sequence": int(row[0]), "event": row[1], "payload": json.loads(row[2])}
            for row in self.db.execute("SELECT sequence, event, payload FROM events ORDER BY sequence")
        ]
        journal = b"".join(canonical_json_bytes(row) + b"\n" for row in events)
        atomic_bytes(self.work / "journal.jsonl", journal)
        self._sync_checkpoint_file()

    def ingest_inventory(self, max_new_archives: int | None = None) -> bool:
        if self._phase() != "inventory":
            return True
        self._ensure_storage_preflight()
        committed = {
            row[0]: row[1]
            for row in self.db.execute("SELECT report_path, report_sha256 FROM archives")
        }
        added = 0
        for relative, _report_path, report_sha, report in self.report_inventory:
            if relative in committed:
                if committed[relative] != report_sha:
                    raise CurationError(f"Committed report changed: {relative}")
                continue
            self._ingest_report(relative, report_sha, report)
            added += 1
            self._sync_audit_files()
            if max_new_archives is not None and added >= max_new_archives:
                return False
        actual_archives = int(
            self.db.execute("SELECT COUNT(*) FROM archives").fetchone()[0]
        )
        actual_documents = int(
            self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        durable = self.db.execute(
            "SELECT archives, documents FROM durable_counts WHERE singleton=1"
        ).fetchone()
        expected_documents = sum(
            int(report["documents"])
            for _relative, _path, _checksum, report in self.report_inventory
        )
        if actual_archives != len(self.report_inventory):
            raise CurationError("Inventory did not ingest every frozen report")
        if (
            int(durable[0]) != actual_archives
            or int(durable[1]) != actual_documents
            or actual_documents != expected_documents
        ):
            raise CurationError("Inventory durable row accounting failed")
        self._ensure_post_inventory_indexes()
        totals = self.db.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM documents").fetchone()
        self._advance("inventory", "inventory_complete", {"documents": totals[0], "tokens": totals[1]})
        return True

    def _ensure_post_inventory_indexes(self) -> None:
        """Bulk-build each global index as its own restartable DDL unit.

        SQLite cannot resume inside ``CREATE INDEX``.  The durable cursor is
        therefore one complete index, while the earlier storage projection
        reserves database plus transient WAL space for the largest unit.
        """
        specs = (
            (
                "documents_content",
                "CREATE INDEX IF NOT EXISTS documents_content ON documents(content_hash)",
            ),
            (
                "documents_final_cluster",
                "CREATE INDEX IF NOT EXISTS documents_final_cluster ON documents(final_cluster)",
            ),
            (
                "documents_source_group",
                "CREATE INDEX IF NOT EXISTS documents_source_group ON documents(source_group)",
            ),
            (
                QUOTA_SELECTION_INDEX,
                f"CREATE INDEX IF NOT EXISTS {QUOTA_SELECTION_INDEX} "
                "ON documents(bucket, selection_rank, doc_id)",
            ),
            (
                "reasons_reason",
                "CREATE INDEX IF NOT EXISTS reasons_reason ON reasons(reason)",
            ),
        )
        subphase = "inventory.bulk_indexes"
        progress = self._start_subphase(
            subphase,
            cursor={"completed_indexes": 0},
            details={
                "transaction_class": "sqlite_create_index",
                "restart_unit": "one_complete_index",
                "indexes": [name for name, _sql in specs],
            },
        )
        completed = progress["cursor"].get("completed_indexes")
        if not isinstance(completed, int) or completed < 0 or completed > len(specs):
            raise CurationError("Invalid bulk-index restart cursor")
        if progress["processed_rows"] != completed:
            raise CurationError("Bulk-index cursor/counter mismatch")
        if progress["status"] != "complete":
            for ordinal in range(completed, len(specs)):
                name, statement = specs[ordinal]
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    if name == QUOTA_SELECTION_INDEX:
                        self.db.execute("DROP INDEX IF EXISTS documents_selection")
                    self.db.execute(statement)
                    self._record_bulk_index_metrics()
                    self.db.execute(
                        """
                        UPDATE phase_progress SET
                            cursor_json=?, processed_rows=processed_rows + 1,
                            committed_batches=committed_batches + 1
                        WHERE subphase=? AND status='running'
                        """,
                        (
                            canonical_json_bytes(
                                {"completed_indexes": ordinal + 1}
                            ).decode("utf-8"),
                            subphase,
                        ),
                    )
                    if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                        raise CurationError("Lost bulk-index durable cursor")
                    self.db.commit()
                except BaseException:
                    self.db.rollback()
                    raise
                self._checkpoint_bulk_index_wal()
                self._sync_checkpoint_file()
                progress = self._progress(subphase)
                assert progress is not None
                self._after_bounded_commit(subphase, progress)
            self._complete_subphase(
                subphase,
                details={
                    "transaction_class": "sqlite_create_index",
                    "restart_unit": "one_complete_index",
                    "indexes": [name for name, _sql in specs],
                    "completed_indexes": len(specs),
                },
            )
        present = {
            str(row[0])
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        missing = [name for name, _sql in specs if name not in present]
        if missing or "documents_selection" in present:
            raise CurationError(
                f"Post-inventory index authority mismatch; missing={missing}"
            )

    def _record_bulk_index_metrics(self) -> None:
        snapshot = self._storage_snapshot()
        if snapshot["free_bytes"] < int(
            self.storage_contract["minimum_free_bytes_after_projection"]
        ):
            raise CurationError("Curation scratch space exhausted during index build")
        self.db.execute(
            """
            UPDATE storage_metrics SET
                maximum_database_bytes=MAX(maximum_database_bytes, ?),
                maximum_journal_bytes=MAX(maximum_journal_bytes, ?),
                maximum_wal_bytes=MAX(maximum_wal_bytes, ?),
                minimum_free_bytes=MIN(minimum_free_bytes, ?),
                committed_transactions=committed_transactions + 1
            WHERE singleton=1
            """,
            (
                snapshot["database_bytes"],
                snapshot["journal_bytes"],
                snapshot["wal_bytes"],
                snapshot["free_bytes"],
            ),
        )

    def _checkpoint_bulk_index_wal(self) -> None:
        snapshot = self._storage_snapshot()
        row = self.db.execute(
            "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise CurationError("Missing storage preflight during bulk index build")
        preflight = self._checked_json_object(
            str(row[0]), label="bulk-index storage preflight"
        )
        bulk_bound = max(
            int(preflight["projected_additional_bytes_with_safety"]),
            int(self.storage_contract["transaction_sidecar_limit_bytes"]),
        )
        if snapshot["wal_bytes"] > bulk_bound:
            self._persist_storage_violation(
                code="bulk_index_wal_projection_exceeded",
                evidence={
                    "observed_wal_bytes": snapshot["wal_bytes"],
                    "bulk_index_wal_bound_bytes": bulk_bound,
                },
            )
            raise CurationError("Bulk-index WAL exceeded its preflight projection")
        with self.db:
            self.db.execute(
                """
                UPDATE storage_metrics SET
                    maximum_database_bytes=MAX(maximum_database_bytes, ?),
                    maximum_wal_bytes=MAX(maximum_wal_bytes, ?),
                    minimum_free_bytes=MIN(minimum_free_bytes, ?)
                WHERE singleton=1
                """,
                (
                    snapshot["database_bytes"],
                    snapshot["wal_bytes"],
                    snapshot["free_bytes"],
                ),
            )
        if self.sqlite_runtime["journal_policy"]["selected_mode"] == "wal":
            result = self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise CurationError("Could not checkpoint WAL after bulk index build")

    def _ingest_report(self, relative: str, report_sha: str, report: dict[str, Any]) -> None:
        fingerprint = self.staging_root / str(report["fingerprint_file"])
        if not fingerprint.is_file():
            raise CurationError(f"Missing fingerprint shard: {fingerprint}")
        if file_sha256(fingerprint) != report["fingerprint_sha256"]:
            raise CurationError(f"Fingerprint checksum mismatch: {fingerprint}")
        bucket = str(report["bucket"])
        hard_flags = set(self.policy["selection"]["quality"]["hard_reject_flags"])
        hard_flags.update(
            self.policy["selection"]["quality"]["hard_reject_flags_by_bucket"][bucket]
        )
        weights = self.policy["selection"]["quality"]["soft_penalty_weights"]
        seed = self.policy["selection"]["seed"]
        subphase = f"inventory.archive.{bucket}.{int(report['index']):06d}"
        progress = self._start_subphase(
            subphase,
            cursor={"input_rows": 0},
            details={
                "archive": report["archive"],
                "report_sha256": report_sha,
                "expected_documents": int(report["documents"]),
                "expected_tokens": int(report["exact_tokens"]),
                "expected_clean_bytes": int(report["clean_bytes"]),
                "clean_bytes": 0,
            },
        )
        if progress["status"] == "complete":
            raise CurationError(
                f"Completed inventory subphase has no archive authority: {report['archive']}"
            )
        if set(progress["cursor"]) != {"input_rows"} or progress["cursor"][
            "input_rows"
        ] != progress["processed_rows"]:
            raise CurationError(f"Invalid inventory cursor for {report['archive']}")
        clean_value = progress["details"].get("clean_bytes")
        if not isinstance(clean_value, int) or clean_value < 0:
            raise CurationError(f"Invalid inventory byte counter for {report['archive']}")
        committed_rows = int(progress["processed_rows"])
        doc_batch: list[tuple[Any, ...]] = []
        reason_batch: list[tuple[bytes, str]] = []
        documents = committed_rows
        tokens = int(progress["processed_tokens"])
        clean_bytes = clean_value
        input_rows = 0

        def flush() -> None:
            nonlocal progress
            if not doc_batch:
                return
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(
                    """
                    INSERT INTO documents(
                        doc_id, bucket, archive, manifest_index, member_path, tokens,
                        content_hash, normalized_hash, final_cluster, source_group,
                        canonical_rank, selection_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    doc_batch,
                )
                self.db.executemany(
                    "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
                    reason_batch,
                )
                self.db.execute(
                    """
                    UPDATE durable_counts SET documents=documents + ?
                    WHERE singleton=1
                    """,
                    (len(doc_batch),),
                )
                progress = self._commit_bounded_batch(
                    subphase,
                    cursor={"input_rows": input_rows},
                    rows=len(doc_batch),
                    tokens=sum(int(row[5]) for row in doc_batch),
                    details={
                        **progress["details"],
                        "clean_bytes": clean_bytes,
                    },
                )
            except BaseException:
                self.db.rollback()
                raise
            doc_batch.clear()
            reason_batch.clear()

        for expected_index, row in enumerate(iter_jsonl_zst(fingerprint)):
            input_rows += 1
            if expected_index < committed_rows:
                continue
            if (
                row.get("record_version") != 1
                or row.get("fingerprint_version") != FINGERPRINT_VERSION
            ):
                raise CurationError(f"Unsupported fingerprint record in {fingerprint}")
            if row.get("bucket") != bucket or row.get("archive") != report["archive"]:
                raise CurationError(f"Fingerprint report identity mismatch in {fingerprint}")
            if row.get("manifest_index") != expected_index:
                raise CurationError(f"Non-contiguous manifest index in {fingerprint}")
            member_path = row.get("member_path")
            if not isinstance(member_path, str) or not member_path:
                raise CurationError(f"Invalid member path in {fingerprint}")
            expected_doc = hashlib.sha256(
                f"{report['archive']}\0{member_path}".encode("utf-8")
            ).hexdigest()
            if row.get("doc_id") != expected_doc:
                raise CurationError(f"Unstable document identity in {fingerprint}")
            doc_id = parse_hex_digest(row["doc_id"], "doc_id")
            content_hash = parse_hex_digest(
                row.get("content_sha256"), "content_sha256"
            )
            normalized_hash = parse_hex_digest(
                row.get("normalized_sha256"), "normalized_sha256"
            )
            token_count = row.get("starcoder2_tokens")
            size_bytes = row.get("size_bytes")
            if not isinstance(token_count, int) or token_count <= 0:
                raise CurationError(f"Invalid token count in {fingerprint}")
            if not isinstance(size_bytes, int) or size_bytes <= 0:
                raise CurationError(f"Invalid byte count in {fingerprint}")
            flags = row.get("quality_flags")
            if (
                not isinstance(flags, list)
                or flags != sorted(set(flags))
                or not all(isinstance(flag, str) for flag in flags)
            ):
                raise CurationError(f"Invalid quality flags in {fingerprint}")
            benchmark_reason = row.get("benchmark_reason")
            if bool(benchmark_reason) != ("benchmark_contamination" in flags):
                raise CurationError(f"Benchmark reason/flag mismatch in {fingerprint}")
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise CurationError(f"Missing provenance in {fingerprint}")
            if bucket in CODE_BUCKETS:
                repo_id = provenance.get("repo_id")
                source_identity = str(repo_id).strip() if repo_id is not None else ""
                source_group = stable_digest(
                    "code-repository", source_identity or row["doc_id"]
                )
                final_cluster = stable_digest("code-normalized", normalized_hash)
                if not source_identity:
                    reason_batch.append((doc_id, "missing_code_repo_id"))
            else:
                identity = english_source_identity(bucket, provenance)
                source_group = stable_digest(
                    "english-source", identity or f"missing:{row['doc_id']}"
                )
                final_cluster = stable_digest(
                    "english-normalized-provisional", normalized_hash
                )
                if identity is None:
                    reason_batch.append((doc_id, "missing_english_source_identity"))
            if self.fast_canonical_profile:
                final_cluster = stable_digest(
                    "fast-global-normalized", normalized_hash
                )
            for flag in flags:
                if flag in hard_flags:
                    reason_batch.append((doc_id, f"quality:{flag}"))
            if benchmark_reason:
                reason_batch.append((doc_id, f"benchmark:{benchmark_reason}"))
            penalty = sum(int(weights.get(flag, 0)) for flag in flags)
            canonical_rank = struct.pack(">Q", penalty) + doc_id
            selection_rank = stable_digest(f"selection:{seed}", doc_id)
            doc_batch.append(
                (
                    doc_id,
                    bucket,
                    report["archive"],
                    expected_index,
                    member_path,
                    token_count,
                    content_hash,
                    normalized_hash,
                    final_cluster,
                    source_group,
                    canonical_rank,
                    selection_rank,
                )
            )
            documents += 1
            tokens += token_count
            clean_bytes += size_bytes
            if len(doc_batch) >= self.batch_size:
                flush()
        flush()
        if input_rows != documents:
            raise CurationError(f"Fingerprint input cursor mismatch: {fingerprint}")
        if documents != report["documents"] or tokens != report["exact_tokens"] or clean_bytes != report["clean_bytes"]:
            raise CurationError(f"Fingerprint totals do not match report: {fingerprint}")
        progress = self._progress(subphase)
        assert progress is not None
        if (
            progress["processed_rows"] != documents
            or progress["processed_tokens"] != tokens
        ):
            raise CurationError(f"Fingerprint durable counters mismatch: {fingerprint}")
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                """
                INSERT INTO archives(
                    report_path, report_sha256, archive, bucket, fingerprint_file,
                    fingerprint_sha256, documents, tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relative,
                    report_sha,
                    report["archive"],
                    bucket,
                    report["fingerprint_file"],
                    report["fingerprint_sha256"],
                    documents,
                    tokens,
                ),
            )
            self.db.execute(
                """
                UPDATE durable_counts SET archives=archives + 1 WHERE singleton=1
                """
            )
            final_details = {
                **progress["details"],
                "clean_bytes": clean_bytes,
                "validated_documents": documents,
                "validated_tokens": tokens,
            }
            self.db.execute(
                """
                UPDATE phase_progress SET status='complete', details_json=?
                WHERE subphase=? AND status='running'
                """,
                (canonical_json_bytes(final_details).decode("utf-8"), subphase),
            )
            if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                raise CurationError(f"Cannot finalize inventory subphase {subphase}")
            self._record_transaction_metrics(0)
            self._event("archive_ingested", {"archive": report["archive"], "documents": documents, "tokens": tokens})
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._bound_wal_after_commit()
        self._sync_checkpoint_file()

    def _load_near_clusters(self) -> bool:
        english_documents = int(
            self.db.execute(
                "SELECT COUNT(*) FROM documents WHERE bucket IN ('fineweb_edu','wikipedia')"
            ).fetchone()[0]
        )
        if self.fast_canonical_profile:
            if self.english_near_clusters is not None or self.english_near_artifact is not None:
                raise CurationError(
                    "Fast canonical profile unexpectedly acquired a near artifact"
                )
            return False
        if english_documents == 0:
            return True
        if self.english_near_clusters is None:
            if not self.allow_missing_english_near_dedup:
                raise CurationError(
                    "English documents require a complete cross-source near-cluster mapping; "
                    "run the separate English near-dedup stage first"
                )
            return False
        if self.english_near_artifact is None:
            raise CurationError("English near-dedup artifact identity was not frozen")
        mapping_path = self.english_near_clusters
        manifest_path = mapping_path.parent / "manifest.json"
        checksum_path = mapping_path.parent / "manifest.sha256"
        publication = self.english_near_artifact["manifest"]
        preflight = self.english_near_artifact[
            "refinement_operational_preflight"
        ]
        preflight_result = mapping_path.parent / preflight["result_path"]
        preflight_sidecar = mapping_path.parent / preflight["sidecar_path"]
        if (
            not mapping_path.is_file()
            or mapping_path.is_symlink()
            or mapping_path.stat().st_size
            != self.english_near_artifact["mapping"]["bytes"]
            or file_sha256(mapping_path)
            != self.english_near_artifact["mapping"]["sha256"]
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.resolve(strict=True)
            != (self.root / publication["path"]).resolve(strict=True)
            or manifest_path.stat().st_size != publication["bytes"]
            or file_sha256(manifest_path)
            != publication["sha256"]
            or not checksum_path.is_file()
            or checksum_path.is_symlink()
            or checksum_path.resolve(strict=True)
            != (self.root / publication["sidecar_path"]).resolve(strict=True)
            or file_sha256(checksum_path)
            != publication["sidecar_sha256"]
            or not preflight_result.is_file()
            or preflight_result.is_symlink()
            or preflight_result.stat().st_size != preflight["result_bytes"]
            or file_sha256(preflight_result) != preflight["result_sha256"]
            or not preflight_sidecar.is_file()
            or preflight_sidecar.is_symlink()
            or file_sha256(preflight_sidecar) != preflight["sidecar_sha256"]
        ):
            raise CurationError(
                "English near-dedup publication changed before mapping ingestion"
            )
        expected_records = int(self.english_near_artifact["mapping"]["records"])
        if expected_records != english_documents:
            raise CurationError(
                "English near-map manifest/document inventory count mismatch"
            )

        load_name = "canonicalize.near_map_load"
        load_progress = self._start_subphase(
            load_name,
            cursor={"input_rows": 0},
            details={"mapping_sha256": self.near_sha, "expected_rows": expected_records},
        )
        if set(load_progress["cursor"]) != {"input_rows"} or not isinstance(
            load_progress["cursor"]["input_rows"], int
        ):
            raise CurationError("Invalid English near-map input cursor")
        committed_rows = int(load_progress["processed_rows"])
        if load_progress["cursor"]["input_rows"] != committed_rows:
            raise CurationError("English near-map cursor/counter mismatch")
        if load_progress["status"] != "complete":
            batch: list[tuple[bytes, bytes]] = []
            input_rows = 0

            def flush_mapping() -> None:
                nonlocal load_progress
                if not batch:
                    return
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.executemany(
                        "INSERT INTO near_map(doc_id, cluster) VALUES (?, ?)", batch
                    )
                    load_progress = self._commit_bounded_batch(
                        load_name,
                        cursor={"input_rows": input_rows},
                        rows=len(batch),
                    )
                except BaseException:
                    self.db.rollback()
                    raise
                batch.clear()

            for row in iter_jsonl(self.english_near_clusters):
                input_rows += 1
                if input_rows <= committed_rows:
                    continue
                doc_id = parse_hex_digest(row.get("doc_id"), "near-map doc_id")
                cluster = row.get("cluster_id")
                if not isinstance(cluster, str) or not cluster:
                    raise CurationError("near-map cluster_id must be a non-empty string")
                batch.append((doc_id, stable_digest("english-near-cluster", cluster)))
                if len(batch) >= self.batch_size:
                    flush_mapping()
            flush_mapping()
            if input_rows != expected_records:
                raise CurationError(
                    f"English near-map has {input_rows:,} rows; expected {expected_records:,}"
                )

            mapped = int(self.db.execute("SELECT COUNT(*) FROM near_map").fetchone()[0])
            unknown = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM near_map AS n
                    LEFT JOIN documents AS d ON d.doc_id=n.doc_id
                    WHERE d.doc_id IS NULL
                       OR d.bucket NOT IN ('fineweb_edu','wikipedia')
                    """
                ).fetchone()[0]
            )
            coverage = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM documents AS d
                    JOIN near_map AS n ON n.doc_id=d.doc_id
                    WHERE d.bucket IN ('fineweb_edu','wikipedia')
                    """
                ).fetchone()[0]
            )
            if unknown or mapped != expected_records or coverage != english_documents:
                raise CurationError(
                    "English near-map coverage/accounting failed "
                    f"(mapped={mapped}, covered={coverage}, expected={english_documents}, "
                    f"unknown={unknown})"
                )
            self._complete_subphase(
                load_name,
                details={
                    "mapping_sha256": self.near_sha,
                    "expected_rows": expected_records,
                    "mapped_rows": mapped,
                    "unknown_rows": unknown,
                },
            )
        elif int(self.db.execute("SELECT COUNT(*) FROM near_map").fetchone()[0]) != expected_records:
            raise CurationError("Completed English near-map table changed")

        apply_name = "canonicalize.near_map_apply"
        apply_progress = self._start_subphase(
            apply_name,
            cursor={"doc_id": ""},
            details={"expected_rows": english_documents},
        )
        if apply_progress["status"] != "complete":
            while True:
                cursor_hex = apply_progress["cursor"].get("doc_id")
                if not isinstance(cursor_hex, str) or (
                    cursor_hex and (len(cursor_hex) != 64 or any(c not in HEX64 for c in cursor_hex))
                ):
                    raise CurationError("Invalid English near-map apply cursor")
                cursor = bytes.fromhex(cursor_hex) if cursor_hex else b""
                rows = self.db.execute(
                    """
                    SELECT d.doc_id, n.cluster
                    FROM documents AS d JOIN near_map AS n ON n.doc_id=d.doc_id
                    WHERE d.bucket IN ('fineweb_edu','wikipedia') AND d.doc_id>?
                    ORDER BY d.doc_id LIMIT ?
                    """,
                    (cursor, self.batch_size),
                ).fetchall()
                if not rows:
                    break
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.executemany(
                        "UPDATE documents SET final_cluster=? WHERE doc_id=?",
                        [(bytes(row[1]), bytes(row[0])) for row in rows],
                    )
                    apply_progress = self._commit_bounded_batch(
                        apply_name,
                        cursor={"doc_id": bytes(rows[-1][0]).hex()},
                        rows=len(rows),
                    )
                except BaseException:
                    self.db.rollback()
                    raise
            mismatched = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM documents AS d
                    LEFT JOIN near_map AS n ON n.doc_id=d.doc_id
                    WHERE d.bucket IN ('fineweb_edu','wikipedia')
                      AND (n.doc_id IS NULL OR d.final_cluster != n.cluster)
                    """
                ).fetchone()[0]
            )
            apply_progress = self._progress(apply_name)
            assert apply_progress is not None
            if mismatched or apply_progress["processed_rows"] != english_documents:
                raise CurationError("English near-map application is incomplete")
            self._complete_subphase(
                apply_name,
                details={"expected_rows": english_documents, "mismatched_rows": 0},
            )
        else:
            mismatched = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM documents AS d
                    LEFT JOIN near_map AS n ON n.doc_id=d.doc_id
                    WHERE d.bucket IN ('fineweb_edu','wikipedia')
                      AND (n.doc_id IS NULL OR d.final_cluster != n.cluster)
                    """
                ).fetchone()[0]
            )
            if mismatched:
                raise CurationError("Completed English near-map application changed")
        return True

    def _run_keyset_write_subphase(
        self,
        *,
        subphase: str,
        select_sql: str,
        write_sql: str,
    ) -> dict[str, Any]:
        progress = self._start_subphase(
            subphase, cursor={"key": ""}, details={"batch_size": self.batch_size}
        )
        if progress["status"] == "complete":
            return progress
        while True:
            cursor_hex = progress["cursor"].get("key")
            if not isinstance(cursor_hex, str) or (
                cursor_hex and (len(cursor_hex) != 64 or any(c not in HEX64 for c in cursor_hex))
            ):
                raise CurationError(f"Invalid keyset cursor for {subphase}")
            cursor = bytes.fromhex(cursor_hex) if cursor_hex else b""
            rows = self.db.execute(select_sql, (cursor, self.batch_size)).fetchall()
            if not rows:
                break
            key = bytes(rows[-1][0])
            if key <= cursor:
                raise CurationError(f"Non-monotonic keyset cursor for {subphase}")
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(write_sql, [tuple(row) for row in rows])
                progress = self._commit_bounded_batch(
                    subphase,
                    cursor={"key": key.hex()},
                    rows=len(rows),
                )
            except BaseException:
                self.db.rollback()
                raise
        return progress

    def _validate_canonicalization(self) -> dict[str, int]:
        english_final_duplicate_reason = (
            "english_normalized_duplicate"
            if self.fast_canonical_profile
            else "english_near_duplicate"
        )
        residual_reasons = (
            "residual_exact_duplicate",
            "residual_normalized_duplicate",
            english_final_duplicate_reason,
        )
        marks = ",".join("?" for _ in residual_reasons)
        eligible_sql = f"""
            NOT EXISTS (
                SELECT 1 FROM reasons AS r
                WHERE r.doc_id=d.doc_id AND r.reason NOT IN ({marks})
            )
        """
        eligible = int(
            self.db.execute(
                f"SELECT COUNT(*) FROM documents AS d WHERE {eligible_sql}",
                residual_reasons,
            ).fetchone()[0]
        )
        exact_expected = int(
            self.db.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT d.content_hash FROM documents AS d
                    WHERE {eligible_sql} GROUP BY d.content_hash
                )
                """,
                residual_reasons,
            ).fetchone()[0]
        )
        exact_actual = int(self.db.execute("SELECT COUNT(*) FROM exact_choice").fetchone()[0])
        final_actual = int(self.db.execute("SELECT COUNT(*) FROM final_choice").fetchone()[0])
        canonical_map = int(self.db.execute("SELECT COUNT(*) FROM canonical_map").fetchone()[0])
        final_expected = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT d.final_cluster
                    FROM documents AS d JOIN exact_choice AS e
                      ON e.content_hash=d.content_hash
                     AND e.canonical_doc_id=d.doc_id
                    GROUP BY d.final_cluster
                )
                """
            ).fetchone()[0]
        )
        invalid_choice = int(
            self.db.execute(
                f"""
                SELECT COUNT(*) FROM exact_choice AS e
                LEFT JOIN (
                    SELECT d.content_hash, MIN(d.canonical_rank) AS wanted_rank,
                           substr(MIN(d.canonical_rank), 9, 32) AS wanted_doc
                    FROM documents AS d WHERE {eligible_sql}
                    GROUP BY d.content_hash
                ) AS wanted USING(content_hash)
                WHERE wanted.content_hash IS NULL
                   OR e.canonical_rank != wanted.wanted_rank
                   OR e.canonical_doc_id != wanted.wanted_doc
                """,
                residual_reasons,
            ).fetchone()[0]
        )
        invalid_final_choice = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM final_choice AS f
                LEFT JOIN (
                    SELECT d.final_cluster, MIN(d.canonical_rank) AS wanted_rank,
                           substr(MIN(d.canonical_rank), 9, 32) AS wanted_doc
                    FROM documents AS d JOIN exact_choice AS e
                      ON e.content_hash=d.content_hash
                     AND e.canonical_doc_id=d.doc_id
                    GROUP BY d.final_cluster
                ) AS wanted USING(final_cluster)
                WHERE wanted.final_cluster IS NULL
                   OR f.canonical_rank != wanted.wanted_rank
                   OR f.canonical_doc_id != wanted.wanted_doc
                """
            ).fetchone()[0]
        )
        invalid_map = int(
            self.db.execute(
                f"""
                SELECT COUNT(*) FROM documents AS d
                LEFT JOIN canonical_map AS c ON c.doc_id=d.doc_id
                LEFT JOIN exact_choice AS e ON e.content_hash=d.content_hash
                LEFT JOIN documents AS exact_doc ON exact_doc.doc_id=e.canonical_doc_id
                LEFT JOIN final_choice AS f
                  ON f.final_cluster=exact_doc.final_cluster
                WHERE {eligible_sql}
                  AND (c.doc_id IS NULL OR c.canonical_doc_id != f.canonical_doc_id)
                """,
                residual_reasons,
            ).fetchone()[0]
        )
        invalid_residual = int(
            self.db.execute(
                f"""
                SELECT COUNT(*) FROM documents AS d
                JOIN canonical_map AS c ON c.doc_id=d.doc_id
                JOIN exact_choice AS e ON e.content_hash=d.content_hash
                JOIN documents AS exact_doc ON exact_doc.doc_id=e.canonical_doc_id
                JOIN final_choice AS f ON f.final_cluster=exact_doc.final_cluster
                WHERE {eligible_sql} AND (
                    (SELECT COUNT(*) FROM reasons AS r WHERE r.doc_id=d.doc_id
                       AND r.reason IN ({marks})) !=
                        CASE WHEN d.doc_id != e.canonical_doc_id THEN 1
                             WHEN d.doc_id != f.canonical_doc_id THEN 1
                             ELSE 0 END
                    OR (d.doc_id != e.canonical_doc_id AND NOT EXISTS (
                        SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                          AND r.reason='residual_exact_duplicate'
                    ))
                    OR (d.doc_id = e.canonical_doc_id
                        AND d.doc_id != f.canonical_doc_id AND NOT EXISTS (
                        SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                          AND r.reason=CASE
                              WHEN d.bucket IN ('python','other_code')
                              THEN 'residual_normalized_duplicate'
                              ELSE '{english_final_duplicate_reason}' END
                    ))
                )
                """,
                (*residual_reasons, *residual_reasons),
            ).fetchone()[0]
        )
        benchmark_missing = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM documents AS d
                WHERE (
                    EXISTS (SELECT 1 FROM benchmark_content_clusters AS b
                            WHERE b.content_hash=d.content_hash)
                    OR EXISTS (SELECT 1 FROM benchmark_final_clusters AS b
                               WHERE b.final_cluster=d.final_cluster)
                ) AND NOT EXISTS (
                    SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                      AND r.reason='benchmark_cluster_contamination'
                )
                """
            ).fetchone()[0]
        )
        benchmark_extra = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM reasons AS r JOIN documents AS d
                  ON d.doc_id=r.doc_id
                WHERE r.reason='benchmark_cluster_contamination'
                  AND NOT EXISTS (SELECT 1 FROM benchmark_content_clusters AS b
                                  WHERE b.content_hash=d.content_hash)
                  AND NOT EXISTS (SELECT 1 FROM benchmark_final_clusters AS b
                                  WHERE b.final_cluster=d.final_cluster)
                """
            ).fetchone()[0]
        )
        accepted = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM documents AS d
                WHERE NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                """
            ).fetchone()[0]
        )
        if (
            exact_actual != exact_expected
            or final_actual != final_expected
            or canonical_map != eligible
            or accepted != final_actual
            or invalid_choice
            or invalid_final_choice
            or invalid_map
            or invalid_residual
            or benchmark_missing
            or benchmark_extra
        ):
            raise CurationError(
                "Canonicalization coverage/accounting validation failed: "
                f"eligible={eligible}, exact={exact_actual}/{exact_expected}, "
                f"final={final_actual}/{final_expected}, map={canonical_map}, "
                f"accepted={accepted}, invalid_choice={invalid_choice}, "
                f"invalid_final={invalid_final_choice}, invalid_map={invalid_map}, "
                f"invalid_residual={invalid_residual}, "
                f"benchmark_missing={benchmark_missing}, benchmark_extra={benchmark_extra}"
            )
        return {
            "eligible_documents": eligible,
            "exact_choices": exact_actual,
            "final_choices": final_actual,
            "canonical_map_rows": canonical_map,
            "accepted_canonical_documents": accepted,
        }

    def canonicalize(self) -> None:
        if self._phase() not in ("inventory_complete",):
            return
        self._ensure_storage_preflight()
        near_complete = self._load_near_clusters()
        english_final_duplicate_reason = (
            "english_normalized_duplicate"
            if self.fast_canonical_profile
            else "english_near_duplicate"
        )
        phases = (
            (
                "canonicalize.benchmark_content_clusters",
                """
                SELECT d.content_hash
                FROM reasons AS r JOIN documents AS d ON d.doc_id=r.doc_id
                WHERE d.content_hash>? AND r.reason LIKE 'benchmark:%'
                GROUP BY d.content_hash ORDER BY d.content_hash LIMIT ?
                """,
                "INSERT INTO benchmark_content_clusters(content_hash) VALUES (?)",
            ),
            (
                "canonicalize.benchmark_final_clusters",
                """
                SELECT d.final_cluster
                FROM reasons AS r JOIN documents AS d ON d.doc_id=r.doc_id
                WHERE d.final_cluster>? AND r.reason LIKE 'benchmark:%'
                GROUP BY d.final_cluster ORDER BY d.final_cluster LIMIT ?
                """,
                "INSERT INTO benchmark_final_clusters(final_cluster) VALUES (?)",
            ),
            (
                "canonicalize.benchmark_exact",
                """
                SELECT d.doc_id, 'benchmark_cluster_contamination'
                FROM documents AS d JOIN benchmark_content_clusters AS b
                  ON b.content_hash=d.content_hash
                WHERE d.doc_id>? ORDER BY d.doc_id LIMIT ?
                """,
                "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
            ),
            (
                "canonicalize.benchmark_final",
                """
                SELECT d.doc_id, 'benchmark_cluster_contamination'
                FROM documents AS d JOIN benchmark_final_clusters AS b
                  ON b.final_cluster=d.final_cluster
                WHERE d.doc_id>? ORDER BY d.doc_id LIMIT ?
                """,
                "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
            ),
            (
                "canonicalize.exact_choice",
                """
                SELECT d.content_hash, MIN(d.canonical_rank),
                       substr(MIN(d.canonical_rank), 9, 32)
                FROM documents AS d
                WHERE d.content_hash>?
                  AND NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                GROUP BY d.content_hash ORDER BY d.content_hash LIMIT ?
                """,
                "INSERT INTO exact_choice(content_hash, canonical_rank, canonical_doc_id) VALUES (?, ?, ?)",
            ),
            (
                "canonicalize.final_choice",
                """
                SELECT d.final_cluster, MIN(d.canonical_rank),
                       substr(MIN(d.canonical_rank), 9, 32)
                FROM documents AS d JOIN exact_choice AS e
                  ON e.content_hash=d.content_hash AND e.canonical_doc_id=d.doc_id
                WHERE d.final_cluster>?
                  AND NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                GROUP BY d.final_cluster ORDER BY d.final_cluster LIMIT ?
                """,
                "INSERT INTO final_choice(final_cluster, canonical_rank, canonical_doc_id) VALUES (?, ?, ?)",
            ),
            (
                "canonicalize.canonical_map",
                """
                SELECT d.doc_id, f.canonical_doc_id
                FROM documents AS d
                JOIN exact_choice AS e ON e.content_hash=d.content_hash
                JOIN documents AS exact_doc ON exact_doc.doc_id=e.canonical_doc_id
                JOIN final_choice AS f ON f.final_cluster=exact_doc.final_cluster
                WHERE d.doc_id>?
                  AND NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                ORDER BY d.doc_id LIMIT ?
                """,
                "INSERT INTO canonical_map(doc_id, canonical_doc_id) VALUES (?, ?)",
            ),
            (
                "canonicalize.residual_exact",
                """
                SELECT d.doc_id, 'residual_exact_duplicate'
                FROM documents AS d JOIN exact_choice AS e USING(content_hash)
                WHERE d.doc_id>? AND d.doc_id != e.canonical_doc_id
                  AND EXISTS (SELECT 1 FROM canonical_map AS c WHERE c.doc_id=d.doc_id)
                ORDER BY d.doc_id LIMIT ?
                """,
                "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
            ),
            (
                "canonicalize.residual_final",
                f"""
                SELECT d.doc_id,
                       CASE WHEN d.bucket IN ('python','other_code')
                            THEN 'residual_normalized_duplicate'
                            ELSE '{english_final_duplicate_reason}' END
                FROM documents AS d
                JOIN exact_choice AS e
                  ON e.content_hash=d.content_hash AND e.canonical_doc_id=d.doc_id
                JOIN final_choice AS f ON f.final_cluster=d.final_cluster
                WHERE d.doc_id>? AND d.doc_id != f.canonical_doc_id
                ORDER BY d.doc_id LIMIT ?
                """,
                "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
            ),
        )
        for subphase, select_sql, write_sql in phases:
            progress = self._run_keyset_write_subphase(
                subphase=subphase, select_sql=select_sql, write_sql=write_sql
            )
            if progress["status"] != "complete":
                self._complete_subphase(
                    subphase,
                    details={
                        "processed_rows": progress["processed_rows"],
                        "committed_batches": progress["committed_batches"],
                    },
                )

        accounting = self._validate_canonicalization()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self._record_transaction_metrics(0)
            if self._phase() != "inventory_complete":
                raise CurationError("Canonicalization phase authority changed")
            self.db.execute(
                "UPDATE metadata SET value=? WHERE key='phase'",
                (json.dumps("canonicalized"),),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('english_near_complete', ?)",
                (json.dumps(near_complete),),
            )
            self._event(
                "canonicalized",
                {**accounting, "english_near_complete": near_complete},
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._bound_wal_after_commit()
        self._sync_audit_files()

    def assign_splits_and_quotas(self) -> None:
        if self._phase() != "canonicalized":
            return
        self._ensure_storage_preflight()
        self._ensure_post_inventory_indexes()
        thresholds = split_thresholds(self.quotas)
        seed = self.policy["selection"]["seed"]
        group_name = "selection.groups"
        group_progress = self._start_subphase(
            group_name, cursor={"group_id": ""}, details={"seed": seed}
        )
        if group_progress["status"] != "complete":
            while True:
                cursor_hex = group_progress["cursor"].get("group_id")
                if not isinstance(cursor_hex, str) or (
                    cursor_hex and (len(cursor_hex) != 64 or any(c not in HEX64 for c in cursor_hex))
                ):
                    raise CurationError("Invalid split-group keyset cursor")
                cursor = bytes.fromhex(cursor_hex) if cursor_hex else b""
                rows = self.db.execute(
                    """
                    SELECT DISTINCT d.source_group
                    FROM documents AS d
                    WHERE d.source_group>?
                      AND NOT EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                    ORDER BY d.source_group LIMIT ?
                    """,
                    (cursor, self.batch_size),
                ).fetchall()
                if not rows:
                    break
                batch = [
                    (
                        bytes(row[0]),
                        assign_group_split(seed, bytes(row[0]), thresholds),
                    )
                    for row in rows
                ]
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.executemany(
                        "INSERT INTO groups(group_id, split) VALUES (?, ?)", batch
                    )
                    group_progress = self._commit_bounded_batch(
                        group_name,
                        cursor={"group_id": batch[-1][0].hex()},
                        rows=len(batch),
                    )
                except BaseException:
                    self.db.rollback()
                    raise

            expected_groups = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT d.source_group FROM documents AS d
                        WHERE NOT EXISTS (
                            SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                        ) GROUP BY d.source_group
                    )
                    """
                ).fetchone()[0]
            )
            actual_groups = int(self.db.execute("SELECT COUNT(*) FROM groups").fetchone()[0])
            missing_or_extra = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM groups AS g
                    LEFT JOIN documents AS d ON d.source_group=g.group_id
                      AND NOT EXISTS (
                          SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                      )
                    WHERE d.doc_id IS NULL
                    """
                ).fetchone()[0]
            )
            mismatched = 0
            for row in self.db.execute(
                "SELECT group_id, split FROM groups ORDER BY group_id"
            ):
                if str(row[1]) != assign_group_split(seed, bytes(row[0]), thresholds):
                    mismatched += 1
            if (
                expected_groups != actual_groups
                or missing_or_extra
                or mismatched
                or group_progress["processed_rows"] != actual_groups
            ):
                raise CurationError(
                    "Split-group coverage/accounting validation failed"
                )
            self._complete_subphase(
                group_name,
                details={
                    "seed": seed,
                    "groups": actual_groups,
                    "mismatched_assignments": 0,
                },
            )
        else:
            expected_groups = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT d.source_group FROM documents AS d
                        WHERE NOT EXISTS (
                            SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                        ) GROUP BY d.source_group
                    )
                    """
                ).fetchone()[0]
            )
            actual_groups = int(self.db.execute("SELECT COUNT(*) FROM groups").fetchone()[0])
            missing_or_extra = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM groups AS g
                    LEFT JOIN documents AS d ON d.source_group=g.group_id
                      AND NOT EXISTS (
                          SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                      )
                    WHERE d.doc_id IS NULL
                    """
                ).fetchone()[0]
            )
            mismatched = sum(
                str(row[1])
                != assign_group_split(seed, bytes(row[0]), thresholds)
                for row in self.db.execute(
                    "SELECT group_id, split FROM groups ORDER BY group_id"
                )
            )
            if (
                actual_groups != expected_groups
                or actual_groups != group_progress["processed_rows"]
                or missing_or_extra
                or mismatched
            ):
                raise CurationError("Completed split-group table changed")

        # Each quota has its own committed total-order cursor. This makes a
        # terminal prefix and its exact token counter atomic with selection.
        for split in self.policy["selection"]["split_order"]:
            for category in ("python", "other_code", "english"):
                self._select_quota_bounded(split=split, category=category)

        total_target = sum(self.quotas.values())
        selected_documents = int(
            self.db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        )
        durable_selected_documents = int(
            self.db.execute(
                "SELECT selected_documents FROM durable_counts WHERE singleton=1"
            ).fetchone()[0]
        )
        progress_selected_documents = int(
            self.db.execute(
                """
                SELECT COALESCE(SUM(processed_rows), 0) FROM phase_progress
                WHERE subphase LIKE 'selection.quota.%'
                """
            ).fetchone()[0]
        )
        selected_tokens = int(
            self.db.execute(
                "SELECT COALESCE(SUM(selected_tokens), 0) FROM selected"
            ).fetchone()[0]
        )
        invalid_selected = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM selected AS s
                JOIN documents AS d ON d.doc_id=s.doc_id
                LEFT JOIN groups AS g ON g.group_id=d.source_group
                WHERE s.selected_tokens < 1 OR s.selected_tokens > d.tokens
                   OR g.group_id IS NULL OR s.split != g.split
                   OR EXISTS (SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id)
                """
            ).fetchone()[0]
        )
        quota_mismatches = 0
        for split in SPLITS:
            for category in ("python", "other_code", "english"):
                progress = self._progress(f"selection.quota.{split}.{category}")
                if (
                    progress is None
                    or progress["status"] != "complete"
                    or progress["processed_tokens"] != self.quotas[(split, category)]
                ):
                    quota_mismatches += 1
        leakage = self._leakage_audit()
        leakage_failures = sum(
            leakage[key]
            for key in (
                "content_hashes_in_multiple_splits",
                "canonical_clusters_in_multiple_splits",
                "source_groups_in_multiple_splits",
                "cross_bucket_code_repo_groups_in_multiple_splits",
            )
        )
        if (
            selected_tokens != total_target
            or selected_documents != durable_selected_documents
            or selected_documents != progress_selected_documents
            or invalid_selected
            or quota_mismatches
            or leakage_failures
        ):
            raise CurationError(
                "Selection coverage/quota/accounting validation failed: "
                f"tokens={selected_tokens}/{total_target}, invalid={invalid_selected}, "
                f"documents={selected_documents}/{durable_selected_documents}/"
                f"{progress_selected_documents}, "
                f"quota_mismatches={quota_mismatches}, leakage={leakage_failures}"
            )
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self._record_transaction_metrics(0)
            if self._phase() != "canonicalized":
                raise CurationError("Selection phase authority changed")
            self.db.execute(
                "UPDATE metadata SET value=? WHERE key='phase'",
                (json.dumps("selected"),),
            )
            self._event(
                "selected",
                {
                    "tokens": selected_tokens,
                    "documents": selected_documents,
                    "quota_mismatches": 0,
                    "leakage_failures": 0,
                },
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._bound_wal_after_commit()
        self._sync_audit_files()

    def _select_quota_bounded(self, *, split: str, category: str) -> None:
        target = self.quotas[(split, category)]
        buckets = [
            bucket for bucket, mapped in BUCKET_CATEGORY.items() if mapped == category
        ]
        subphase = f"selection.quota.{split}.{category}"
        progress = self._start_subphase(
            subphase,
            cursor={"selection_rank": "", "doc_id": ""},
            details={"split": split, "category": category, "target_tokens": target},
        )

        durable_selected = int(
            self.db.execute(
                "SELECT selected_documents FROM durable_counts WHERE singleton=1"
            ).fetchone()[0]
        )
        progress_selected = int(
            self.db.execute(
                """
                SELECT COALESCE(SUM(processed_rows), 0) FROM phase_progress
                WHERE subphase LIKE 'selection.quota.%'
                """
            ).fetchone()[0]
        )
        if durable_selected != progress_selected:
            raise CurationError("Durable/global quota row counters disagree")
        if progress["status"] == "complete":
            if progress["processed_tokens"] != target:
                raise CurationError(f"Completed quota changed for {split}/{category}")
            self._validate_quota_prefix(
                split=split, category=category, progress=progress
            )
            return

        remaining = target - progress["processed_tokens"]
        if remaining < 0:
            raise CurationError(f"Quota progress exceeds target for {split}/{category}")
        while remaining:
            rank_hex = progress["cursor"].get("selection_rank")
            doc_hex = progress["cursor"].get("doc_id")
            if not isinstance(rank_hex, str) or not isinstance(doc_hex, str):
                raise CurationError(f"Invalid quota cursor for {split}/{category}")
            if bool(rank_hex) != bool(doc_hex) or any(
                value
                and (len(value) != 64 or any(character not in HEX64 for character in value))
                for value in (rank_hex, doc_hex)
            ):
                raise CurationError(f"Invalid quota cursor digest for {split}/{category}")
            after = (
                (bytes.fromhex(rank_hex), bytes.fromhex(doc_hex))
                if rank_hex
                else None
            )
            candidates = iter_merged_quota_candidate_rows(
                self.db, split=split, buckets=buckets, after=after
            )
            batch: list[tuple[bytes, str, int]] = []
            last_rank: bytes | None = None
            last_doc: bytes | None = None
            batch_tokens = 0
            try:
                for rank, doc_id, tokens in candidates:
                    count = min(tokens, remaining - batch_tokens)
                    if count <= 0:
                        break
                    batch.append((doc_id, split, count))
                    batch_tokens += count
                    last_rank, last_doc = rank, doc_id
                    if len(batch) >= self.batch_size or batch_tokens == remaining:
                        break
            finally:
                close = getattr(candidates, "close", None)
                if close is not None:
                    close()
            if not batch or last_rank is None or last_doc is None:
                raise CurationError(
                    f"Split {split}/{category} is short by {remaining:,} tokens "
                    "after filtering/grouping"
                )
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(
                    """
                    INSERT INTO selected(doc_id, split, selected_tokens)
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )
                self.db.execute(
                    """
                    UPDATE durable_counts
                    SET selected_documents=selected_documents + ?
                    WHERE singleton=1
                    """,
                    (len(batch),),
                )
                progress = self._commit_bounded_batch(
                    subphase,
                    cursor={
                        "selection_rank": last_rank.hex(),
                        "doc_id": last_doc.hex(),
                    },
                    rows=len(batch),
                    tokens=batch_tokens,
                )
            except BaseException:
                self.db.rollback()
                raise
            remaining -= batch_tokens

        if progress["processed_tokens"] != target:
            raise CurationError(f"Final quota accounting mismatch for {split}/{category}")
        completed = self._complete_subphase(
            subphase,
            details={
                "split": split,
                "category": category,
                "target_tokens": target,
                "selected_tokens": target,
                "selected_documents": int(progress["processed_rows"]),
            },
        )
        self._validate_quota_prefix(
            split=split, category=category, progress=completed
        )

    def _validate_quota_prefix(
        self, *, split: str, category: str, progress: dict[str, Any]
    ) -> None:
        """Prove selection is exactly the deterministic prefix ending at cursor."""
        target = self.quotas[(split, category)]
        rank_hex = progress["cursor"].get("selection_rank")
        doc_hex = progress["cursor"].get("doc_id")
        if (
            not isinstance(rank_hex, str)
            or not isinstance(doc_hex, str)
            or len(rank_hex) != 64
            or len(doc_hex) != 64
            or any(character not in HEX64 for character in rank_hex + doc_hex)
        ):
            raise CurationError(f"Completed quota cursor is invalid for {split}/{category}")
        rank, doc_id = bytes.fromhex(rank_hex), bytes.fromhex(doc_hex)
        buckets = [
            bucket for bucket, mapped in BUCKET_CATEGORY.items() if mapped == category
        ]
        prefix_rows = missing_rows = prefix_tokens_before_terminal = 0
        for bucket in buckets:
            row = self.db.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(
                    CASE WHEN d.selection_rank < ? OR d.doc_id < ?
                         THEN d.tokens ELSE 0 END
                ), 0),
                COALESCE(SUM(CASE WHEN s.doc_id IS NULL THEN 1 ELSE 0 END), 0)
                FROM documents AS d INDEXED BY {QUOTA_SELECTION_INDEX}
                CROSS JOIN groups AS g
                LEFT JOIN selected AS s ON s.doc_id=d.doc_id
                WHERE d.bucket=? AND g.group_id=d.source_group AND g.split=?
                  AND NOT EXISTS (
                      SELECT 1 FROM reasons AS r WHERE r.doc_id=d.doc_id
                  )
                  AND (
                      d.selection_rank < ? OR
                      (d.selection_rank = ? AND d.doc_id <= ?)
                  )
                """,
                (rank, doc_id, bucket, split, rank, rank, doc_id),
            ).fetchone()
            prefix_rows += int(row[0])
            prefix_tokens_before_terminal += int(row[1])
            missing_rows += int(row[2])
        terminal = self.db.execute(
            """
            SELECT s.selected_tokens, d.tokens
            FROM selected AS s JOIN documents AS d ON d.doc_id=s.doc_id
            WHERE s.doc_id=? AND s.split=?
            """,
            (doc_id, split),
        ).fetchone()
        expected_terminal = target - prefix_tokens_before_terminal
        if (
            prefix_rows != progress["processed_rows"]
            or missing_rows
            or terminal is None
            or int(terminal[0]) != expected_terminal
            or expected_terminal < 1
            or expected_terminal > int(terminal[1])
        ):
            raise CurationError(
                f"Deterministic quota-prefix validation failed for {split}/{category}: "
                f"prefix_rows={prefix_rows}, selected={progress['processed_rows']}, "
                f"missing={missing_rows}, "
                f"expected_terminal={expected_terminal}"
            )

    def _decision_rows(self, rows: list[dict[str, Any]]) -> dict[bytes, dict[str, Any]]:
        ids = [parse_hex_digest(row["doc_id"], "doc_id") for row in rows]
        if not ids:
            return {}
        result = {doc_id: {"reasons": []} for doc_id in ids}
        # SQLite defaults to 999 bind parameters.  Input batches are split
        # into conservative lookup chunks while output compression stays large.
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" for _ in chunk)
            for row in self.db.execute(
                f"SELECT doc_id, reason FROM reasons WHERE doc_id IN ({marks}) ORDER BY doc_id, reason",
                chunk,
            ):
                result[bytes(row[0])]["reasons"].append(str(row[1]))
            for row in self.db.execute(
                f"SELECT doc_id, canonical_doc_id FROM canonical_map WHERE doc_id IN ({marks})",
                chunk,
            ):
                result[bytes(row[0])]["canonical_doc_id"] = bytes(row[1]).hex()
            for row in self.db.execute(
                f"""
                SELECT d.doc_id, g.split, hex(d.source_group)
                FROM documents AS d JOIN groups AS g ON g.group_id=d.source_group
                WHERE d.doc_id IN ({marks})
                """,
                chunk,
            ):
                result[bytes(row[0])].update(
                    assigned_split=str(row[1]), group_id=str(row[2]).lower()
                )
            for row in self.db.execute(
                f"""
                SELECT s.doc_id, s.split, s.selected_tokens
                FROM selected AS s
                WHERE s.doc_id IN ({marks})
                """,
                chunk,
            ):
                result[bytes(row[0])].update(
                    split=str(row[1]), selected_tokens=int(row[2])
                )
        return result

    def emit_decisions(self) -> None:
        if self._phase() not in ("selected", "emitting"):
            return
        if self._phase() == "selected":
            self._advance("selected", "emitting", {})
        committed = {
            row[0]: (row[1], row[2], int(row[3]))
            for row in self.db.execute(
                "SELECT archive, decision_file, decision_sha256, records FROM output_archives"
            )
        }
        reports = {report["archive"]: report for *_prefix, report in self.report_inventory}
        for archive in sorted(reports):
            report = reports[archive]
            if archive in committed:
                relative, checksum, records = committed[archive]
                path = self.output / relative
                if not path.is_file() or file_sha256(path) != checksum or records != report["documents"]:
                    raise CurationError(f"Committed decision shard is corrupt: {path}")
                continue
            self._emit_archive(report)
            self._sync_audit_files()
        actual_outputs = int(
            self.db.execute("SELECT COUNT(*) FROM output_archives").fetchone()[0]
        )
        durable_outputs = int(
            self.db.execute(
                "SELECT output_archives FROM durable_counts WHERE singleton=1"
            ).fetchone()[0]
        )
        if actual_outputs != len(reports) or durable_outputs != actual_outputs:
            raise CurationError("Decision-output durable accounting mismatch")
        self._advance("emitting", "emitted", {"archives": len(reports)})

    def _emit_archive(self, report: dict[str, Any]) -> None:
        fingerprint = self.staging_root / report["fingerprint_file"]
        if file_sha256(fingerprint) != report["fingerprint_sha256"]:
            raise CurationError(f"Fingerprint changed before decision emission: {fingerprint}")
        relative = Path("decisions") / report["bucket"] / f"part-{int(report['index']):06d}.jsonl.zst"
        output_path = self.output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
        os.close(descriptor)
        temporary = Path(name)
        records = 0
        try:
            with temporary.open("wb") as raw:
                compressor = zstandard.ZstdCompressor(level=6, threads=1, write_checksum=True)
                with compressor.stream_writer(raw, closefd=False) as compressed:
                    pending: list[dict[str, Any]] = []

                    def flush() -> None:
                        nonlocal records
                        if not pending:
                            return
                        status = self._decision_rows(pending)
                        for source in pending:
                            doc_id = parse_hex_digest(source["doc_id"], "doc_id")
                            decision = status[doc_id]
                            reasons = list(decision["reasons"])
                            selected = decision.get("selected_tokens")
                            if selected is None and not reasons:
                                reasons.append("quota_overflow")
                            payload = {
                                "record_version": DECISION_RECORD_VERSION,
                                "doc_id": source["doc_id"],
                                "bucket": source["bucket"],
                                "category": BUCKET_CATEGORY[source["bucket"]],
                                "archive": source["archive"],
                                "archive_index": source["archive_index"],
                                "manifest_index": source["manifest_index"],
                                "member_path": source["member_path"],
                                "decision": "keep" if selected is not None else "reject",
                                "split": decision.get("split"),
                                "assigned_split": decision.get("assigned_split"),
                                "source_tokens": source["starcoder2_tokens"],
                                "selected_tokens": selected or 0,
                                "token_prefix": [0, selected] if selected is not None else None,
                                "terminal_quota_prefix": selected is not None and selected < source["starcoder2_tokens"],
                                "canonical_doc_id": decision.get("canonical_doc_id"),
                                "split_group_id": decision.get("group_id"),
                                "reasons": sorted(reasons),
                                "content_sha256": source["content_sha256"],
                                "normalized_sha256": source["normalized_sha256"],
                                "quality_flags": source["quality_flags"],
                                "benchmark_reason": source["benchmark_reason"],
                                "provenance": source["provenance"],
                            }
                            compressed.write(canonical_json_bytes(payload) + b"\n")
                            records += 1
                        pending.clear()

                    for row in iter_jsonl_zst(fingerprint):
                        pending.append(row)
                        if len(pending) >= self.batch_size:
                            flush()
                    flush()
                raw.flush()
                os.fsync(raw.fileno())
            if records != report["documents"]:
                raise CurationError(f"Decision record count mismatch for {report['archive']}")
            checksum = file_sha256(temporary)
            os.replace(temporary, output_path)
            fsync_directory(output_path.parent)
            with self.db:
                self.db.execute(
                    "INSERT INTO output_archives(archive, decision_file, decision_sha256, records) VALUES (?, ?, ?, ?)",
                    (report["archive"], str(relative), checksum, records),
                )
                self.db.execute(
                    """
                    UPDATE durable_counts
                    SET output_archives=output_archives + 1 WHERE singleton=1
                    """
                )
                self._event("decisions_emitted", {"archive": report["archive"], "records": records, "sha256": checksum})
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _verify_all_inputs_and_outputs(self) -> None:
        current_reports, current_completeness = self._load_complete_report_inventory()
        if current_completeness != self.collection_completeness:
            raise CurationError(
                "Collection completeness identity changed during curation"
            )
        current_report_identity = [
            (relative, checksum)
            for relative, _path, checksum, _report in current_reports
        ]
        frozen_report_identity = [
            (relative, checksum)
            for relative, _path, checksum, _report in self.report_inventory
        ]
        if current_report_identity != frozen_report_identity:
            raise CurationError("Frozen report inventory changed during curation")
        current_near_artifact = self._validate_english_near_artifact()
        if current_near_artifact != self.english_near_artifact:
            raise CurationError(
                "English near-dedup manifest identity changed during curation"
            )
        for relative, path, checksum, report in self.report_inventory:
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise CurationError(f"Report changed after inventory: {relative}")
            fingerprint = self.staging_root / report["fingerprint_file"]
            if file_sha256(fingerprint) != report["fingerprint_sha256"]:
                raise CurationError(f"Fingerprint changed after inventory: {fingerprint}")
        for row in self.db.execute(
            "SELECT decision_file, decision_sha256 FROM output_archives ORDER BY decision_file"
        ):
            path = self.output / row[0]
            if file_sha256(path) != row[1]:
                raise CurationError(f"Decision output checksum mismatch: {path}")

    def finalize(self) -> dict[str, Any]:
        if self._phase() == "complete":
            return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        if self._phase() != "emitted":
            raise CurationError(f"Cannot finalize from phase {self._phase()}")
        self._assert_no_storage_violation()
        self._validate_durable_counts_full()
        self._verify_all_inputs_and_outputs()
        quota_rows = []
        for split in SPLITS:
            for category in ("python", "other_code", "english"):
                buckets = [bucket for bucket, mapped in BUCKET_CATEGORY.items() if mapped == category]
                marks = ",".join("?" for _ in buckets)
                row = self.db.execute(
                    f"""
                    SELECT COUNT(*), COALESCE(SUM(s.selected_tokens), 0),
                           COALESCE(SUM(CASE WHEN s.selected_tokens < d.tokens THEN 1 ELSE 0 END), 0)
                    FROM selected AS s JOIN documents AS d ON d.doc_id=s.doc_id
                    WHERE s.split=? AND d.bucket IN ({marks})
                    """,
                    (split, *buckets),
                ).fetchone()
                target = self.quotas[(split, category)]
                if int(row[1]) != target:
                    raise CurationError(f"Final quota mismatch for {split}/{category}")
                quota_rows.append(
                    {
                        "split": split,
                        "category": category,
                        "unit": "pre_packing_starcoder2_content_tokens",
                        "target_tokens": target,
                        "selected_tokens": int(row[1]),
                        "documents": int(row[0]),
                        "terminal_prefix_documents": int(row[2]),
                    }
                )
        reason_counts = {
            str(row[0]): int(row[1])
            for row in self.db.execute("SELECT reason, COUNT(*) FROM reasons GROUP BY reason ORDER BY reason")
        }
        input_quality_flag_counts: collections.Counter[str] = collections.Counter()
        for *_prefix, report in self.report_inventory:
            input_quality_flag_counts.update(report.get("quality_flag_counts") or {})
        accepted_canonical = int(
            self.db.execute(
                "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS "
                "(SELECT 1 FROM reasons r WHERE r.doc_id=d.doc_id)"
            ).fetchone()[0]
        )
        selected_documents = int(self.db.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
        near_complete = bool(
            json.loads(
                self.db.execute("SELECT value FROM metadata WHERE key='english_near_complete'").fetchone()[0]
            )
        )
        production_ready = near_complete or self.fast_canonical_profile
        if self.fast_canonical_profile and near_complete:
            raise CurationError(
                "Fast canonical profile cannot claim completed fuzzy near-deduplication"
            )
        outputs = [
            {
                "archive": str(row[0]),
                "path": str(row[1]),
                "sha256": str(row[2]),
                "records": int(row[3]),
            }
            for row in self.db.execute(
                "SELECT archive, decision_file, decision_sha256, records FROM output_archives ORDER BY archive"
            )
        ]
        leakage_audit = self._leakage_audit()
        required_zero_audits = [
            "content_hashes_in_multiple_splits",
            "canonical_clusters_in_multiple_splits",
            "source_groups_in_multiple_splits",
            "cross_bucket_code_repo_groups_in_multiple_splits",
        ]
        if self.fast_canonical_profile:
            required_zero_audits.extend(
                (
                    "normalized_hashes_in_multiple_splits",
                    "content_hashes_with_multiple_selected_documents",
                    "normalized_hashes_with_multiple_selected_documents",
                )
            )
        if any(leakage_audit[key] for key in required_zero_audits):
            raise CurationError(f"Leakage audit failed: {leakage_audit}")
        fast_profile_audit: dict[str, Any] | None = None
        if self.fast_canonical_profile:
            near_map_rows = int(
                self.db.execute("SELECT COUNT(*) FROM near_map").fetchone()[0]
            )
            near_subphases = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM phase_progress "
                    "WHERE subphase IN "
                    "('canonicalize.near_map_load','canonicalize.near_map_apply')"
                ).fetchone()[0]
            )
            mislabeled_near_reasons = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM reasons "
                    "WHERE reason='english_near_duplicate'"
                ).fetchone()[0]
            )
            if near_map_rows or near_subphases or mislabeled_near_reasons:
                raise CurationError(
                    "Fast canonical profile unexpectedly consumed fuzzy-near state"
                )
            fast_profile_audit = {
                "audit_version": 1,
                "fuzzy_near_dedup_performed": False,
                "near_map_rows": 0,
                "near_mapping_subphases": 0,
                "english_near_duplicate_reasons": 0,
                "content_hashes_in_multiple_splits": leakage_audit[
                    "content_hashes_in_multiple_splits"
                ],
                "normalized_hashes_in_multiple_splits": leakage_audit[
                    "normalized_hashes_in_multiple_splits"
                ],
                "content_hashes_with_multiple_selected_documents": leakage_audit[
                    "content_hashes_with_multiple_selected_documents"
                ],
                "normalized_hashes_with_multiple_selected_documents": leakage_audit[
                    "normalized_hashes_with_multiple_selected_documents"
                ],
                "source_groups_in_multiple_splits": leakage_audit[
                    "source_groups_in_multiple_splits"
                ],
            }
        input_reports = [
            {
                "report": relative,
                "report_sha256": checksum,
                "archive": report["archive"],
                "archive_sha256": report["archive_sha256"],
                "fingerprint_file": report["fingerprint_file"],
                "fingerprint_sha256": report["fingerprint_sha256"],
                "documents": report["documents"],
                "content_tokens": report["exact_tokens"],
            }
            for relative, _path, checksum, report in self.report_inventory
        ]
        total_docs = int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        manifest = {
            "manifest_version": 1,
            "decision_record_version": DECISION_RECORD_VERSION,
            "identity": self.identity,
            "collection_completeness": self.collection_completeness,
            "selection_policy": self.policy["selection"],
            "production_ready": production_ready,
            "known_provenance_limitations": [
                "The legacy parallel Stack collector did not embed tokenizer_revision in STACK_V3_SOURCE.json; the current validated tokenizer artifact is pinned by this manifest.",
                *(
                    FAST_CANONICAL_PROFILE["known_limitations"]
                    if self.fast_canonical_profile
                    else []
                ),
            ],
            # Kept for the v1 selection/materializer contract: curation never
            # parses raw archive payloads as selection input. Every archive in
            # all four buckets is nevertheless opened as an opaque byte stream
            # for the independently reported integrity hash below.
            "raw_archives_opened": False,
            "raw_archives_hashed_for_integrity": True,
            "raw_archive_payloads_parsed_by_curation": False,
            "quota_unit": "pre_packing_starcoder2_content_tokens",
            "training_input_budget_authority": (
                "the final packed order v4 manifest full-optimizer-update-prefix "
                "consumed_input_tokens; EOS boundaries and packing tails mean these "
                "curation quotas are not model-input token counts"
            ),
            "english_near_dedup_complete": near_complete,
            "documents": {
                "input": total_docs,
                "accepted_canonical_before_quota": accepted_canonical,
                "selected": selected_documents,
                "quota_overflow": accepted_canonical - selected_documents,
            },
            "reason_document_counts": reason_counts,
            "input_quality_flag_counts": dict(sorted(input_quality_flag_counts.items())),
            "leakage_audit": leakage_audit,
            "quotas": quota_rows,
            "input_reports": input_reports,
            "decision_shards": outputs,
            "decision_inventory_sha256": hashlib.sha256(canonical_json_bytes(outputs)).hexdigest(),
        }
        if self.fast_canonical_profile:
            manifest.update(
                curation_profile=self.curation_profile,
                english_near_dedup_status="disabled_by_fast_profile",
                fast_profile_audit=fast_profile_audit,
            )
        atomic_json(self.output / "manifest.json", manifest)
        atomic_bytes(
            self.output / "manifest.sha256",
            (file_sha256(self.output / "manifest.json") + "  manifest.json\n").encode("ascii"),
        )
        self._advance("emitted", "complete", {"manifest_sha256": file_sha256(self.output / "manifest.json")})
        return manifest

    def _leakage_audit(self) -> dict[str, int]:
        """Prove selected content, canonical clusters, and groups are split-disjoint."""
        queries = {
            "content_hashes_in_multiple_splits": """
                SELECT COUNT(*) FROM (
                    SELECT d.content_hash FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                    GROUP BY d.content_hash HAVING COUNT(DISTINCT s.split) > 1
                )
            """,
            "canonical_clusters_in_multiple_splits": """
                SELECT COUNT(*) FROM (
                    SELECT d.final_cluster FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                    GROUP BY d.final_cluster HAVING COUNT(DISTINCT s.split) > 1
                )
            """,
            "source_groups_in_multiple_splits": """
                SELECT COUNT(*) FROM (
                    SELECT d.source_group FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                    GROUP BY d.source_group HAVING COUNT(DISTINCT s.split) > 1
                )
            """,
            "cross_bucket_code_repo_groups_in_multiple_splits": """
                SELECT COUNT(*) FROM (
                    SELECT d.source_group
                    FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                    WHERE d.bucket IN ('python','other_code')
                    GROUP BY d.source_group
                    HAVING COUNT(DISTINCT d.bucket) > 1 AND COUNT(DISTINCT s.split) > 1
                )
            """,
        }
        result = {key: int(self.db.execute(query).fetchone()[0]) for key, query in queries.items()}
        if self.fast_canonical_profile:
            result.update(
                normalized_hashes_in_multiple_splits=int(
                    self.db.execute(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT d.normalized_hash
                            FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                            GROUP BY d.normalized_hash
                            HAVING COUNT(DISTINCT s.split) > 1
                        )
                        """
                    ).fetchone()[0]
                ),
                content_hashes_with_multiple_selected_documents=int(
                    self.db.execute(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT d.content_hash
                            FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                            GROUP BY d.content_hash HAVING COUNT(*) > 1
                        )
                        """
                    ).fetchone()[0]
                ),
                normalized_hashes_with_multiple_selected_documents=int(
                    self.db.execute(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT d.normalized_hash
                            FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                            GROUP BY d.normalized_hash HAVING COUNT(*) > 1
                        )
                        """
                    ).fetchone()[0]
                ),
            )
        result["cross_bucket_code_repo_groups"] = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT d.source_group
                    FROM selected s JOIN documents d ON d.doc_id=s.doc_id
                    WHERE d.bucket IN ('python','other_code')
                    GROUP BY d.source_group HAVING COUNT(DISTINCT d.bucket) > 1
                )
                """
            ).fetchone()[0]
        )
        return result

    def run(self, max_new_archives: int | None = None, stop_after_phase: str | None = None) -> dict[str, Any]:
        completed_inventory = self.ingest_inventory(max_new_archives=max_new_archives)
        if not completed_inventory:
            return {"complete": False, "phase": self._phase(), "reason": "archive_limit"}
        if stop_after_phase == "inventory_complete":
            return {"complete": False, "phase": self._phase()}
        self.canonicalize()
        if stop_after_phase == "canonicalized":
            return {"complete": False, "phase": self._phase()}
        self.assign_splits_and_quotas()
        if stop_after_phase == "selected":
            return {"complete": False, "phase": self._phase()}
        self.emit_decisions()
        if stop_after_phase == "emitted":
            return {"complete": False, "phase": self._phase()}
        manifest = self.finalize()
        return {"complete": True, "phase": self._phase(), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--quotas", type=Path, default=DEFAULT_QUOTAS)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--english-near-clusters", type=Path)
    parser.add_argument(
        "--allow-missing-english-near-dedup",
        action="store_true",
        help="diagnostic only: use exact/normalized English clusters and mark output non-production",
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument(
        "--sqlite-journal-mode",
        choices=tuple(sorted(SQLITE_JOURNAL_MODES)),
        default="auto",
        help=(
            "SQLite journal policy (default: auto). Auto uses WAL only on a "
            "known-local output mount and DELETE rollback journaling otherwise; "
            "an explicit WAL override is rejected on network or unknown mounts."
        ),
    )
    parser.add_argument("--max-new-archives", type=int)
    parser.add_argument(
        "--recover-stale-cross-client-lease",
        metavar="OWNER_TOKEN",
        help=(
            "explicitly archive the stale durable lease with this exact "
            "owner_token after independently proving no other pod/process is "
            "running; permanently claims the token and must never be automated"
        ),
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=("inventory_complete", "canonicalized", "selected", "emitted"),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    staging = args.staging_root or args.root / "staging" / "preprocess"
    output = args.output or args.root / "curated" / "selection-v1"
    output.mkdir(parents=True, exist_ok=True)
    lease_info: tuple[Path, dict[str, Any], Any] | None = None
    exit_code = 0
    try:
        lease_info = acquire_cross_client_lease(
            output,
            recover_stale_owner_token=args.recover_stale_cross_client_lease,
        )
        with CurationBuilder(
            root=args.root,
            staging_root=staging,
            output=output,
            policy_path=args.policy,
            quota_path=args.quotas,
            denylist_path=args.benchmark_denylist,
            english_near_clusters=args.english_near_clusters,
            allow_missing_english_near_dedup=args.allow_missing_english_near_dedup,
            batch_size=args.batch_size,
            sqlite_journal_mode=args.sqlite_journal_mode,
        ) as builder:
            result = builder.run(
                max_new_archives=args.max_new_archives, stop_after_phase=args.stop_after_phase
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (CurationError, OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"Curation failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if lease_info is not None:
            try:
                release_cross_client_lease(output, *lease_info)
            except (CurationError, OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Curation lease release failed: {exc}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
