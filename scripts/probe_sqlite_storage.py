#!/usr/bin/env python3
"""Crash, locking, and throughput probe for a prospective curation filesystem.

The probe creates only a uniquely named temporary directory below ``--root``.
It exercises the same SQLite rollback-journal and FULL-synchronous policy used
for NFS curation, verifies that an uncommitted process crash is recovered, and
verifies that a competing writer cannot enter an active transaction.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PAYLOAD = b"x" * 256
CRASH_PAYLOAD = b"a" * 1024
CRASH_DIRTY_PAYLOAD = b"b" * 1024
CRASH_ROWS = 4_000
SQLITE_ROLLBACK_JOURNAL_MAGIC = bytes.fromhex("d9d505f920a163d7")


class ProbeError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def configure(connection: sqlite3.Connection, database: Path) -> Path:
    temp_directory = database.parent / "sqlite-tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    escaped_temp = str(temp_directory.resolve()).replace("'", "''")
    connection.execute(f"PRAGMA temp_store_directory='{escaped_temp}'")
    temp_row = connection.execute("PRAGMA temp_store_directory").fetchone()
    if (
        temp_row is None
        or Path(str(temp_row[0])).resolve() != temp_directory.resolve()
        or os.stat(temp_directory).st_dev != os.stat(database.parent).st_dev
    ):
        raise ProbeError("SQLite temp directory is not pinned beside the probe database")
    mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
    if mode != "delete":
        raise ProbeError(f"SQLite refused DELETE journal mode: {mode}")
    connection.execute("PRAGMA synchronous=FULL")
    if int(connection.execute("PRAGMA synchronous").fetchone()[0]) != 2:
        raise ProbeError("SQLite refused FULL synchronous mode")
    connection.execute("PRAGMA temp_store=FILE")
    if int(connection.execute("PRAGMA temp_store").fetchone()[0]) != 1:
        raise ProbeError("SQLite refused file-backed temp storage")
    return temp_directory


def lock_child(database: Path) -> int:
    connection = sqlite3.connect(database, timeout=0.25)
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower():
                return 0
            raise
        connection.rollback()
        return 3
    finally:
        connection.close()


def link_child(candidate: Path, canonical: Path) -> int:
    """Return success only when an existing canonical link excludes us."""
    try:
        os.link(candidate, canonical, follow_symlinks=False)
    except FileExistsError:
        return 0
    return 3


def flock_child(lock_path: Path) -> int:
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        fcntl.flock(lock, fcntl.LOCK_UN)
        return 3


def crash_child(database: Path) -> int:
    connection = sqlite3.connect(database, timeout=30)
    configure(connection, database)
    # The tiny cache and explicit spill make the transaction exceed cache by
    # orders of magnitude. SQLite must journal original pages and write dirty
    # pages to the main database before this uncommitted process exit.
    connection.execute("PRAGMA cache_size=16")
    connection.execute("PRAGMA cache_spill=ON")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "UPDATE crash_rows SET value=1, payload=?", (CRASH_DIRTY_PAYLOAD,)
    )
    # Make the rollback journal and dirty database pages observable before an
    # abrupt process exit. os._exit deliberately bypasses SQLite/Python cleanup.
    journal = Path(f"{database}-journal")
    deadline = time.monotonic() + 10
    journal_ready = False
    while time.monotonic() < deadline:
        if journal.exists() and journal.stat().st_size > 512:
            with journal.open("rb") as handle:
                journal_ready = handle.read(8) == SQLITE_ROLLBACK_JOURNAL_MAGIC
            if journal_ready:
                break
        time.sleep(0.01)
    if not journal_ready:
        os._exit(24)
    os._exit(23)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def mount_evidence(path: Path) -> dict[str, Any]:
    stat = os.statvfs(path)
    stat_device = int(path.stat().st_dev)
    try:
        output = subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        payload = json.loads(output)
        filesystems = payload.get("filesystems") or []
        if len(filesystems) == 1 and isinstance(filesystems[0], dict):
            return {
                **filesystems[0],
                "stat_device": stat_device,
                "statvfs_fsid": int(stat.f_fsid),
            }
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return {
        "target": str(path.resolve()),
        "source": None,
        "fstype": None,
        "options": None,
        "stat_device": stat_device,
        "statvfs_fsid": int(stat.f_fsid),
    }


def run_probe(
    *,
    root: Path,
    rows: int,
    batch_rows: int,
    minimum_rows_per_second: float,
    minimum_index_rows_per_second: float,
    required_fstype: str | None = None,
    required_source: str | None = None,
    required_mount_options: tuple[str, ...] = (),
) -> dict[str, Any]:
    if rows < batch_rows or batch_rows < 1:
        raise ProbeError("rows must be at least batch_rows, and both must be positive")
    root.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix=".sqlite-curation-probe-", dir=root))
    database = probe / "probe.sqlite3"
    started = time.time()
    try:
        connection = sqlite3.connect(database, timeout=30)
        temp_directory = configure(connection, database)

        lease_candidate = probe / "lease-candidate.json"
        lease_canonical = probe / "lease-canonical.json"
        with lease_candidate.open("wb") as handle:
            handle.write(b'{"complete":true}\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(lease_candidate, lease_canonical, follow_symlinks=False)
        directory_descriptor = os.open(
            probe, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        link_result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--link-child",
                str(lease_candidate),
                str(lease_canonical),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        atomic_hardlink_lease_exclusion = (
            link_result.returncode == 0
            and lease_candidate.read_bytes() == lease_canonical.read_bytes()
            and lease_candidate.stat().st_ino == lease_canonical.stat().st_ino
            and lease_candidate.stat().st_nlink >= 2
        )
        lease_canonical.unlink()
        lease_candidate.unlink()

        advisory_lock_path = probe / "cross-client.lock"
        with advisory_lock_path.open("a+") as advisory_lock:
            fcntl.flock(advisory_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            flock_result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--flock-child",
                    str(advisory_lock_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        advisory_flock_exclusion = flock_result.returncode == 0
        connection.execute(
            "CREATE TABLE rows("
            "id INTEGER PRIMARY KEY, digest BLOB NOT NULL, normalized BLOB NOT NULL, "
            "group_hash BLOB NOT NULL, bucket TEXT NOT NULL, rank BLOB NOT NULL, "
            "payload BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE reasons("
            "id INTEGER NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(id, reason))"
        )
        commit_seconds: list[float] = []
        inserted = 0
        reason_rows = 0
        insert_started = time.monotonic()
        while inserted < rows:
            count = min(batch_rows, rows - inserted)
            batch = [
                (
                    ordinal,
                    hashlib.sha256(str(ordinal).encode("ascii")).digest(),
                    hashlib.sha256(f"normalized:{ordinal}".encode("ascii")).digest(),
                    hashlib.sha256(f"group:{ordinal // 100}".encode("ascii")).digest(),
                    ("python", "other_code", "fineweb_edu", "wikipedia")[
                        ordinal % 4
                    ],
                    hashlib.sha256(f"rank:{ordinal}".encode("ascii")).digest(),
                    PAYLOAD,
                )
                for ordinal in range(inserted, inserted + count)
            ]
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO rows("
                "id, digest, normalized, group_hash, bucket, rank, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            reason_batch = [
                (ordinal, "quality:probe")
                for ordinal in range(inserted, inserted + count)
                if ordinal % 10 == 0
            ]
            connection.executemany(
                "INSERT INTO reasons(id, reason) VALUES (?, ?)", reason_batch
            )
            before_commit = time.monotonic()
            connection.commit()
            commit_seconds.append(time.monotonic() - before_commit)
            reason_rows += len(reason_batch)
            inserted += count
        insert_seconds = time.monotonic() - insert_started

        index_specs = (
            ("rows_digest", "CREATE INDEX rows_digest ON rows(digest)", rows),
            (
                "rows_normalized",
                "CREATE INDEX rows_normalized ON rows(normalized)",
                rows,
            ),
            (
                "rows_group_hash",
                "CREATE INDEX rows_group_hash ON rows(group_hash)",
                rows,
            ),
            (
                "rows_bucket_rank_id",
                "CREATE INDEX rows_bucket_rank_id ON rows(bucket, rank, id)",
                rows,
            ),
            (
                "reasons_reason",
                "CREATE INDEX reasons_reason ON reasons(reason)",
                reason_rows,
            ),
        )
        index_seconds_by_name: dict[str, float] = {}
        index_rows_total = 0
        for name, statement, indexed_rows in index_specs:
            index_started = time.monotonic()
            connection.execute(statement)
            connection.commit()
            index_seconds_by_name[name] = time.monotonic() - index_started
            index_rows_total += indexed_rows
        index_seconds = sum(index_seconds_by_name.values())
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ProbeError("Integrity check failed after throughput phase")

        connection.execute(
            "CREATE TABLE crash_rows("
            "id INTEGER PRIMARY KEY, value INTEGER NOT NULL, payload BLOB NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO crash_rows(id, value, payload) VALUES (?, 0, ?)",
            ((ordinal, CRASH_PAYLOAD) for ordinal in range(CRASH_ROWS)),
        )
        connection.commit()

        connection.execute("BEGIN IMMEDIATE")
        lock_result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--lock-child", str(database)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        connection.rollback()
        if lock_result.returncode != 0:
            raise ProbeError(
                "Competing writer entered or failed outside the expected lock path "
                f"(exit={lock_result.returncode}, stderr={lock_result.stderr!r})"
            )
        connection.close()

        committed_database_sha256 = file_sha256(database)

        crash_result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--crash-child", str(database)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if crash_result.returncode != 23:
            raise ProbeError(
                f"Crash child did not exit at the intended fault point: {crash_result.returncode}"
            )
        journal = Path(f"{database}-journal")
        hot_journal_observed = journal.exists() and journal.stat().st_size > 512
        hot_journal_bytes = journal.stat().st_size if journal.exists() else 0
        hot_journal_magic_valid = False
        if hot_journal_bytes > 512:
            with journal.open("rb") as handle:
                hot_journal_magic_valid = (
                    handle.read(8) == SQLITE_ROLLBACK_JOURNAL_MAGIC
                )
        post_crash_database_sha256 = file_sha256(database)
        main_database_changed_before_recovery = (
            post_crash_database_sha256 != committed_database_sha256
        )
        recovered = sqlite3.connect(database, timeout=30)
        configure(recovered, database)
        integrity = str(recovered.execute("PRAGMA integrity_check").fetchone()[0])
        changed = int(
            recovered.execute("SELECT COUNT(*) FROM crash_rows WHERE value != 0").fetchone()[0]
        )
        changed_payloads = int(
            recovered.execute(
                "SELECT COUNT(*) FROM crash_rows WHERE payload != ?", (CRASH_PAYLOAD,)
            ).fetchone()[0]
        )
        recovered.close()
        recovered_database_sha256 = file_sha256(database)
        main_database_restored_exactly = (
            recovered_database_sha256 == committed_database_sha256
        )

        rows_per_second = rows / insert_seconds
        index_rows_per_second = index_rows_total / index_seconds
        # Measure the actual probe/database subtree, not merely an ancestor
        # that could conceal an absent or over-mounted production work path.
        mount = mount_evidence(probe)
        mount_options = {
            option
            for option in str(mount.get("options") or "").split(",")
            if option
        }
        failures: list[str] = []
        if (
            required_fstype is None
            or required_source is None
            or not required_mount_options
        ):
            failures.append("durable_mount_requirements_not_fully_specified")
        if rows_per_second < minimum_rows_per_second:
            failures.append("measured_insert_rows_per_second_below_requested_minimum")
        if index_rows_per_second < minimum_index_rows_per_second:
            failures.append("measured_index_rows_per_second_below_requested_minimum")
        if not hot_journal_observed:
            failures.append("nonempty_hot_rollback_journal_not_observed")
        if not atomic_hardlink_lease_exclusion:
            failures.append("atomic_hardlink_lease_exclusion_failed")
        if not advisory_flock_exclusion:
            failures.append("advisory_flock_exclusion_failed")
        if not hot_journal_magic_valid:
            failures.append("hot_rollback_journal_header_is_invalid")
        if not main_database_changed_before_recovery:
            failures.append("crash_did_not_spill_dirty_pages_to_main_database")
        if integrity != "ok":
            failures.append("post_crash_integrity_check_failed")
        if changed != 0 or changed_payloads != 0:
            failures.append("uncommitted_crash_rows_visible_after_recovery")
        if not main_database_restored_exactly:
            failures.append("rollback_did_not_restore_exact_committed_database")
        if required_fstype is not None and mount.get("fstype") != required_fstype:
            failures.append("filesystem_type_does_not_match_required_mount")
        if required_source is not None and mount.get("source") != required_source:
            failures.append("filesystem_source_does_not_match_required_mount")
        missing_mount_options = sorted(set(required_mount_options) - mount_options)
        if missing_mount_options:
            failures.append("required_mount_options_are_missing")
        passed = not failures
        result = {
            "result_version": 1,
            "status": "pass" if passed else "fail",
            "production_gate_eligible": passed,
            "failures": failures,
            "root": str(root.resolve()),
            "mount": mount,
            "mount_requirements": {
                "fstype": required_fstype,
                "source": required_source,
                "options": list(required_mount_options),
                "missing_options": missing_mount_options,
            },
            "sqlite": {
                "version": sqlite3.sqlite_version,
                "journal_mode": "delete",
                "synchronous": "full",
                "temp_store": "file",
                "temp_relative_path": str(temp_directory.relative_to(probe)),
                "temp_same_device_as_database": (
                    os.stat(temp_directory).st_dev == os.stat(database.parent).st_dev
                ),
            },
            "workload": {
                "rows": rows,
                "reason_rows": reason_rows,
                "batch_rows": batch_rows,
                "payload_bytes_per_row": len(PAYLOAD),
                "database_bytes": database.stat().st_size,
            },
            "measurements": {
                "insert_seconds": round(insert_seconds, 6),
                "insert_rows_per_second": round(rows_per_second, 3),
                "index_seconds": round(index_seconds, 6),
                "index_seconds_by_name": {
                    name: round(seconds, 6)
                    for name, seconds in index_seconds_by_name.items()
                },
                "index_rows_total": index_rows_total,
                "index_rows_per_second": round(index_rows_per_second, 3),
                "commit_seconds_mean": round(statistics.mean(commit_seconds), 6),
                "commit_seconds_p50": round(percentile(commit_seconds, 0.50), 6),
                "commit_seconds_p95": round(percentile(commit_seconds, 0.95), 6),
                "commit_seconds_max": round(max(commit_seconds), 6),
            },
            "correctness": {
                "competing_writer_excluded": True,
                "atomic_hardlink_lease_exclusion": (
                    atomic_hardlink_lease_exclusion
                ),
                "advisory_flock_exclusion": advisory_flock_exclusion,
                "crash_child_exit_code": 23,
                "hot_journal_observed": hot_journal_observed,
                "hot_journal_bytes": hot_journal_bytes,
                "hot_journal_magic_valid": hot_journal_magic_valid,
                "main_database_changed_before_recovery": (
                    main_database_changed_before_recovery
                ),
                "main_database_restored_exactly": main_database_restored_exactly,
                "committed_database_sha256": committed_database_sha256,
                "post_crash_database_sha256": post_crash_database_sha256,
                "recovered_database_sha256": recovered_database_sha256,
                "post_crash_integrity_check": integrity,
                "post_crash_uncommitted_rows_visible": changed,
                "post_crash_uncommitted_payloads_visible": changed_payloads,
            },
            "minimum_insert_rows_per_second": minimum_rows_per_second,
            "minimum_index_rows_per_second": minimum_index_rows_per_second,
            "started_unix": int(started),
            "finished_unix": int(time.time()),
        }
        result["identity_sha256"] = hashlib.sha256(
            canonical_json_bytes(
                {
                    "mount": result["mount"],
                    "mount_requirements": result["mount_requirements"],
                    "sqlite": result["sqlite"],
                    "workload": result["workload"],
                    "correctness": result["correctness"],
                }
            )
        ).hexdigest()
        return result
    finally:
        shutil.rmtree(probe)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lock-child", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--crash-child", type=Path, help=argparse.SUPPRESS)
    mode.add_argument(
        "--link-child", type=Path, nargs=2, metavar=("CANDIDATE", "CANONICAL"),
        help=argparse.SUPPRESS,
    )
    mode.add_argument("--flock-child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--batch-rows", type=int, default=10_000)
    parser.add_argument("--minimum-rows-per-second", type=float, default=2_000.0)
    parser.add_argument(
        "--minimum-index-rows-per-second", type=float, default=10_000.0
    )
    parser.add_argument("--require-fstype")
    parser.add_argument("--require-source")
    parser.add_argument(
        "--require-mount-option", action="append", default=[]
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lock_child is not None:
        return lock_child(args.lock_child)
    if args.crash_child is not None:
        return crash_child(args.crash_child)
    if args.link_child is not None:
        return link_child(*args.link_child)
    if args.flock_child is not None:
        return flock_child(args.flock_child)
    if args.root is None or args.result is None:
        raise SystemExit("--root and --result are required")
    try:
        result = run_probe(
            root=args.root,
            rows=args.rows,
            batch_rows=args.batch_rows,
            minimum_rows_per_second=args.minimum_rows_per_second,
            minimum_index_rows_per_second=args.minimum_index_rows_per_second,
            required_fstype=args.require_fstype,
            required_source=args.require_source,
            required_mount_options=tuple(args.require_mount_option),
        )
        atomic_json(args.result, result)
        sidecar = args.result.with_name(f"{args.result.name}.sha256")
        digest = hashlib.sha256(args.result.read_bytes()).hexdigest()
        sidecar.write_text(f"{digest}  {args.result.name}\n", encoding="ascii")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["production_gate_eligible"] else 2
    except (OSError, ValueError, sqlite3.Error, ProbeError) as error:
        print(f"SQLite storage probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
