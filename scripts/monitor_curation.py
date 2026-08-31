#!/usr/bin/env python3
"""Continuously audit a live curation run without opening its SQLite database.

The monitor reads only the curator's atomically published checkpoint, event
journal, and lease. It never competes for a SQLite lock. Fatal invariant
violations are written to a durable alert artifact; a stale checkpoint is a
warning because later bulk SQL phases can legitimately be quiet for longer
than one inventory batch.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE_ORDER = {
    "inventory": 0,
    "inventory_complete": 1,
    "canonicalizing": 2,
    "canonicalized": 3,
    "selecting": 4,
    "selected": 5,
    "emitting": 6,
    "emitted": 7,
    "complete": 8,
}
STARTUP_PUBLICATION_GRACE_SECONDS = 60.0


class HealthError(RuntimeError):
    """The live curation projection violates a durable invariant."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HealthError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_checked_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HealthError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HealthError(f"JSON root must be an object: {path}")
    return payload


def _read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise HealthError(f"cannot read journal {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise HealthError(f"empty journal line {line_number}")
        try:
            event = json.loads(line, object_pairs_hook=_checked_object)
        except (json.JSONDecodeError, HealthError) as exc:
            raise HealthError(f"invalid journal line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise HealthError(f"journal line {line_number} is not an object")
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise HealthError(f"journal line {line_number} has invalid sequence")
        if not events and sequence != 1:
            raise HealthError("journal must begin at sequence 1")
        if events and sequence != int(events[-1]["sequence"]) + 1:
            raise HealthError(f"journal sequence discontinuity at line {line_number}")
        events.append(event)
    return events


def _plain_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise HealthError(f"{label} must be an integer >= {minimum}")
    return value


def _read_consistent_projection(
    checkpoint_path: Path,
    journal_path: Path,
    *,
    attempts: int = 5,
    retry_seconds: float = 0.02,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one coherent pair across two independently atomic publications."""

    mismatch: tuple[int, int] | None = None
    for attempt in range(attempts):
        checkpoint = _read_json(checkpoint_path)
        events = _read_journal(journal_path)
        raw_checkpoint_sequence = checkpoint.get("last_event_sequence")
        if not isinstance(raw_checkpoint_sequence, int) or isinstance(
            raw_checkpoint_sequence, bool
        ):
            raise HealthError("last event sequence must be an integer >= 0")
        checkpoint_sequence = int(raw_checkpoint_sequence)
        journal_sequence = int(events[-1]["sequence"]) if events else 0
        if checkpoint_sequence == journal_sequence:
            return checkpoint, events
        mismatch = (checkpoint_sequence, journal_sequence)
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    assert mismatch is not None
    raise HealthError(
        "checkpoint/journal sequence mismatch after publication-race retries: "
        f"{mismatch[0]} != {mismatch[1]}"
    )


def _process_matches(pid: int, output: Path) -> tuple[bool, str]:
    process = Path("/proc") / str(pid)
    if not process.is_dir():
        return False, "process_missing"
    try:
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError as exc:
        return False, f"process_unreadable:{exc}"
    if "curate_corpus.py" not in command:
        return False, "pid_does_not_run_curator"
    if "--output" not in command or str(output) not in command:
        return False, "curator_output_mismatch"
    return True, command.strip()


def _storage_checks(
    checkpoint: Mapping[str, Any],
    output: Path,
) -> tuple[list[str], dict[str, int]]:
    storage = checkpoint.get("storage")
    if not isinstance(storage, dict):
        raise HealthError("checkpoint storage evidence is missing")
    violation = storage.get("violation")
    if violation != {}:
        raise HealthError(f"curator recorded a storage violation: {violation!r}")
    committed_transactions = _plain_int(
        storage.get("committed_transactions"),
        label="committed_transactions",
    )
    maximum_transaction_rows = _plain_int(
        storage.get("maximum_transaction_rows"),
        label="maximum_transaction_rows",
    )
    preflight = storage.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "pass":
        raise HealthError("curation storage preflight is not passing")
    contract = preflight.get("contract")
    if not isinstance(contract, dict):
        raise HealthError("curation storage contract is missing")
    transaction_limit = _plain_int(
        contract.get("maximum_transaction_rows"),
        label="storage transaction limit",
        minimum=1,
    )
    if maximum_transaction_rows > transaction_limit:
        raise HealthError("observed transaction rows exceed the frozen limit")
    journal_limit = _plain_int(
        contract.get("transaction_sidecar_limit_bytes"),
        label="transaction sidecar limit",
        minimum=1,
    )
    maximum_journal_bytes = _plain_int(
        storage.get("maximum_journal_bytes"),
        label="maximum_journal_bytes",
    )
    maximum_wal_bytes = _plain_int(
        storage.get("maximum_wal_bytes"),
        label="maximum_wal_bytes",
    )
    if max(maximum_journal_bytes, maximum_wal_bytes) > journal_limit:
        raise HealthError("observed SQLite sidecar exceeds the frozen limit")
    minimum_free_required = _plain_int(
        contract.get("minimum_free_bytes_after_projection"),
        label="minimum free bytes",
        minimum=1,
    )
    filesystem = os.statvfs(output)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    warnings: list[str] = []
    if free_bytes < minimum_free_required:
        raise HealthError(
            f"filesystem free bytes {free_bytes} fell below {minimum_free_required}"
        )
    return warnings, {
        "committed_transactions": committed_transactions,
        "free_bytes": free_bytes,
        "maximum_journal_bytes": maximum_journal_bytes,
        "maximum_wal_bytes": maximum_wal_bytes,
    }


def inspect(
    output: Path,
    *,
    stall_seconds: float,
    live_work_root: Path | None = None,
    now: float | None = None,
    process_checker: Callable[[int, Path], tuple[bool, str]] = _process_matches,
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    work = (
        (output / ".work")
        if live_work_root is None
        else live_work_root.resolve(strict=True)
    )
    if work.is_symlink() or not work.is_dir():
        raise HealthError(f"live curation work root must be a real directory: {work}")
    checkpoint_path = work / "CHECKPOINT.json"
    journal_path = work / "journal.jsonl"
    checkpoint, events = _read_consistent_projection(
        checkpoint_path, journal_path
    )
    now = time.time() if now is None else now
    age_seconds = max(0.0, now - checkpoint_path.stat().st_mtime)

    phase = checkpoint.get("phase")
    if phase not in PHASE_ORDER:
        raise HealthError(f"unknown curation phase {phase!r}")
    identity = checkpoint.get("identity")
    if not isinstance(identity, dict):
        raise HealthError("checkpoint identity is missing")
    completeness = identity.get("collection_completeness")
    if not isinstance(completeness, dict) or completeness.get("complete") is not True:
        raise HealthError("collection completeness authority is not passing")
    if completeness.get("preprocess_error_records") != 0:
        raise HealthError("preprocessing authority contains error records")

    counts = checkpoint.get("counts")
    if not isinstance(counts, dict):
        raise HealthError("checkpoint counts are missing")
    archives = _plain_int(counts.get("archives"), label="archive count")
    documents = _plain_int(counts.get("documents"), label="document count")
    selected_documents = _plain_int(
        counts.get("selected_documents"), label="selected document count"
    )
    output_archives = _plain_int(
        counts.get("output_archives"), label="output archive count"
    )
    expected_documents = _plain_int(
        checkpoint["storage"]["preflight"].get("documents_expected"),
        label="expected document count",
        minimum=1,
    )
    if documents > expected_documents:
        raise HealthError("document count exceeds collection authority")

    last_event_sequence = _plain_int(
        checkpoint.get("last_event_sequence"),
        label="last event sequence",
    )
    journal_sequence = int(events[-1]["sequence"]) if events else 0
    if journal_sequence != last_event_sequence:
        raise HealthError("coherent projection sequence changed during validation")

    subphases = checkpoint.get("subphases")
    if not isinstance(subphases, list):
        raise HealthError("checkpoint subphases are missing")
    running = []
    for subphase in subphases:
        if not isinstance(subphase, dict):
            raise HealthError("subphase entry is not an object")
        status = subphase.get("status")
        if status not in ("complete", "running"):
            raise HealthError(
                f"subphase {subphase.get('subphase')!r} has unsafe status {status!r}"
            )
        if status == "running":
            running.append(subphase)
    if len(running) > 1:
        raise HealthError(f"active phase has multiple running subphases: {len(running)}")
    if phase == "complete" and running:
        raise HealthError("complete run still has a running subphase")

    # The storage contract describes the active SQLite filesystem. In the
    # accelerated mode that is pod-local storage, not the durable publication
    # volume. Durable snapshot capacity is enforced by the snapshot manager.
    warnings, storage = _storage_checks(checkpoint, work)
    if age_seconds > stall_seconds:
        warnings.append(f"checkpoint_stale_for_{int(age_seconds)}_seconds")

    process_alive = phase == "complete"
    process_detail = "completed"
    lease_path = output / ".curation.cross-client-lease.json"
    if phase != "complete":
        lease = _read_json(lease_path)
        lease_output = lease.get("output")
        if not isinstance(lease_output, str) or Path(lease_output).resolve(
            strict=False
        ) != output:
            raise HealthError("curation lease output identity mismatch")
        pid = _plain_int(lease.get("pid"), label="curation lease pid", minimum=1)
        process_alive, process_detail = process_checker(pid, output)
        if not process_alive:
            raise HealthError(f"curator process is not healthy: {process_detail}")
    else:
        pid = None

    startup_publication_grace = False
    if phase == "inventory" and not running:
        grace_seconds = min(stall_seconds, STARTUP_PUBLICATION_GRACE_SECONDS)
        if process_alive and age_seconds <= grace_seconds:
            startup_publication_grace = True
            warnings.append("inventory_active_archive_pending_startup_publication")
        else:
            raise HealthError(
                "inventory has no running archive outside startup publication grace: "
                f"checkpoint_age_seconds={age_seconds:.3f}, "
                f"grace_seconds={grace_seconds:.3f}"
            )

    if phase == "inventory":
        archive_events = [event for event in events if event.get("event") == "archive_ingested"]
        if len(archive_events) != archives:
            raise HealthError(
                f"archive journal/count mismatch: {len(archive_events)} != {archives}"
            )
        committed_documents = 0
        for event in archive_events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise HealthError("archive event payload is invalid")
            committed_documents += _plain_int(
                payload.get("documents"), label="archive event documents"
            )
        active_rows = 0
        if running:
            active_rows = _plain_int(
                running[0].get("processed_rows"), label="running processed rows"
            )
            details = running[0].get("details")
            if not isinstance(details, dict):
                raise HealthError("running inventory details are invalid")
            active_expected = _plain_int(
                details.get("expected_documents"),
                label="running expected documents",
                minimum=1,
            )
            if active_rows > active_expected:
                raise HealthError("running archive processed rows exceed its report")
        if committed_documents + active_rows != documents:
            raise HealthError(
                "inventory durable count differs from completed journal plus active cursor"
            )
        if selected_documents != 0 or output_archives != 0:
            raise HealthError("selection/output counters advanced during inventory")

    progress_percent = 100.0 * documents / expected_documents
    if not math.isfinite(progress_percent) or not 0.0 <= progress_percent <= 100.0:
        raise HealthError("invalid inventory progress percentage")
    active = running[0] if running else None
    return {
        "event": "curation_health",
        "recorded_utc": _utc_now(),
        "status": "warning" if warnings else ("complete" if phase == "complete" else "healthy"),
        "warnings": warnings,
        "phase": phase,
        "counts": {
            "archives": archives,
            "documents": documents,
            "expected_documents": expected_documents,
            "inventory_percent": progress_percent,
            "selected_documents": selected_documents,
            "output_archives": output_archives,
        },
        "active_subphase": None if active is None else active.get("subphase"),
        "active_processed_rows": 0 if active is None else active.get("processed_rows"),
        "startup_publication_grace": startup_publication_grace,
        "checkpoint_age_seconds": age_seconds,
        "live_work_root": str(work),
        "last_event_sequence": last_event_sequence,
        "storage": storage,
        "process": {
            "alive": process_alive,
            "pid": pid,
            "detail": process_detail,
        },
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live-work-root",
        type=Path,
        help=(
            "directory containing the live CHECKPOINT.json and journal.jsonl; "
            "defaults to OUTPUT/.work and should point to the pod-local work "
            "root for accelerated local-SQLite runs"
        ),
    )
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--stall-seconds", type=float, default=3600.0)
    parser.add_argument("--health-log", type=Path)
    parser.add_argument("--alert", type=Path)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_seconds <= 0 or args.stall_seconds <= 0:
        _parser().error("interval and stall thresholds must be positive")
    output = args.output.resolve(strict=True)
    health_log = args.health_log or output / ".work" / "health-monitor.jsonl"
    alert = args.alert or output / "CURATION_HEALTH_ALERT.json"
    lock_path = output / ".curation-health.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _parser().error(f"another health monitor holds {lock_path}")
        previous: dict[str, Any] | None = None
        while True:
            try:
                health = inspect(
                    output,
                    stall_seconds=args.stall_seconds,
                    live_work_root=args.live_work_root,
                )
                if previous is not None:
                    if PHASE_ORDER[str(health["phase"])] < PHASE_ORDER[str(previous["phase"])]:
                        raise HealthError("curation phase regressed")
                    for field in ("archives", "documents", "selected_documents", "output_archives"):
                        if health["counts"][field] < previous["counts"][field]:
                            raise HealthError(f"curation counter regressed: {field}")
                    if health["last_event_sequence"] < previous["last_event_sequence"]:
                        raise HealthError("event sequence regressed")
                    if (
                        health["storage"]["committed_transactions"]
                        < previous["storage"]["committed_transactions"]
                    ):
                        raise HealthError("transaction counter regressed")
            except BaseException as exc:
                failure = {
                    "event": "curation_health",
                    "recorded_utc": _utc_now(),
                    "status": "failed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
                _append_jsonl(health_log, failure)
                _atomic_json(alert, failure)
                print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
                return 1
            _append_jsonl(health_log, health)
            print(json.dumps(health, sort_keys=True), flush=True)
            previous = health
            if args.once or health["status"] == "complete":
                return 0
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
