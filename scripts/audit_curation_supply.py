#!/usr/bin/env python3
"""Audit exact post-filter curation supply from one live SQLite snapshot.

The database is opened with SQLite URI ``mode=ro`` and deliberately without
``immutable=1``.  That distinction is essential for a live WAL database:
immutable readers are allowed to ignore sidecars and can therefore report a
stale or internally inconsistent view.  All authority checks and the supply
aggregation run inside one read transaction.

This command never reads the quota-selection table.  It reports every
canonical, quality/benchmark-eligible document (a document with no ``reasons``
row), joined to its already assigned leakage-safe source group.  The expensive
part is one grouped scan of ``documents``; group coverage, bucket routing, and
token validity are checked from that same result.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


FORMAT = "curation-supply-audit"
FORMAT_VERSION = 1
SUPPORTED_DATABASE_VERSION = 4
SAFE_PHASES = frozenset(
    ("canonicalized", "selected", "emitting", "emitted", "complete")
)
SPLITS = ("train", "validation", "test")
BUCKET_CATEGORY = {
    "python": "python",
    "other_code": "other_code",
    "fineweb_edu": "english",
    "wikipedia": "english",
}
CATEGORIES = ("python", "other_code", "english")
DOCUMENTS_PRIMARY_INDEX = "sqlite_autoindex_documents_1"
# These are connection-local, read-only performance controls. The production
# host has 1 TiB of RAM; SQLite allocates the cache lazily, and mmap_size is
# capped by the SQLite library's compile-time maximum when it is lower.
REQUESTED_CACHE_KIB = 4 * 1024 * 1024
REQUESTED_MMAP_BYTES = 64 * 1024 * 1024 * 1024
SQLITE_PROGRESS_OPCODES = 100_000

REQUIRED_COLUMNS = {
    "metadata": frozenset(("key", "value")),
    "archives": frozenset(("bucket", "documents", "tokens")),
    "documents": frozenset(("doc_id", "bucket", "tokens", "source_group")),
    "events": frozenset(("sequence", "event", "payload")),
    "groups": frozenset(("group_id", "split")),
    "reasons": frozenset(("doc_id", "reason")),
    "phase_progress": frozenset(
        (
            "subphase",
            "status",
            "cursor_json",
            "processed_rows",
            "processed_tokens",
            "committed_batches",
            "details_json",
        )
    ),
}

# Keep this as the sole query whose outer input is documents.  The correlated
# NOT EXISTS lookup uses reasons' (doc_id, reason) primary key and the group
# lookup uses groups' primary key; neither requires another documents scan.
SUPPLY_SQL = """
SELECT
    d.bucket AS bucket,
    g.split AS split,
    COUNT(*) AS documents,
    COALESCE(SUM(d.tokens), 0) AS tokens,
    COALESCE(SUM(
        CASE
            WHEN typeof(d.tokens) != 'integer' OR d.tokens < 1 THEN 1
            ELSE 0
        END
    ), 0) AS invalid_token_documents
FROM documents AS d INDEXED BY sqlite_autoindex_documents_1
LEFT JOIN groups AS g ON g.group_id = d.source_group
WHERE NOT EXISTS (
    SELECT 1 FROM reasons AS r WHERE r.doc_id = d.doc_id
)
GROUP BY d.bucket, g.split
ORDER BY d.bucket, g.split
"""

RAW_TOTALS_SQL = """
SELECT
    bucket,
    COUNT(*) AS archives,
    COALESCE(SUM(documents), 0) AS documents,
    COALESCE(SUM(tokens), 0) AS tokens
FROM archives
GROUP BY bucket
ORDER BY bucket
"""

# The DISTINCT input is reasons, not documents. CROSS JOIN fixes that loop
# order and the explicit primary index makes each rejected-document lookup
# deterministic and bounded instead of inviting another 51M-document scan.
REJECTED_TOTALS_SQL = """
SELECT
    d.bucket AS bucket,
    COUNT(*) AS documents,
    COALESCE(SUM(d.tokens), 0) AS tokens
FROM (SELECT doc_id FROM reasons GROUP BY doc_id) AS rejected
CROSS JOIN documents AS d INDEXED BY sqlite_autoindex_documents_1
WHERE d.doc_id = rejected.doc_id
GROUP BY d.bucket
ORDER BY d.bucket
"""

# A reason row is an association, not a unique rejected document. A LEFT JOIN
# deliberately exposes an orphaned reason as bucket NULL so the audit fails.
REASON_TOTALS_SQL = """
SELECT
    d.bucket AS bucket,
    r.reason AS reason,
    COUNT(*) AS documents,
    COALESCE(SUM(d.tokens), 0) AS tokens,
    COALESCE(SUM(CASE WHEN d.doc_id IS NULL THEN 1 ELSE 0 END), 0)
        AS orphan_reason_rows
FROM reasons AS r INDEXED BY reasons_reason
LEFT JOIN documents AS d INDEXED BY sqlite_autoindex_documents_1
    ON d.doc_id = r.doc_id
GROUP BY d.bucket, r.reason
ORDER BY d.bucket, r.reason
"""


class SupplyAuditError(RuntimeError):
    """The database cannot safely authorize a supply report."""


class _Heartbeat:
    def __init__(self, interval_seconds: float, stream: TextIO) -> None:
        self.interval_seconds = interval_seconds
        self.stream = stream
        self.started = time.monotonic()
        self.next_emit = self.started + interval_seconds
        self.stage = "opening"

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.emit("stage")

    def emit(self, event: str) -> None:
        elapsed = time.monotonic() - self.started
        print(
            f"[{FORMAT}] event={event} stage={self.stage} "
            f"elapsed_seconds={elapsed:.1f}",
            file=self.stream,
            flush=True,
        )

    def sqlite_progress(self) -> int:
        now = time.monotonic()
        if now >= self.next_emit:
            self.emit("heartbeat")
            while self.next_emit <= now:
                self.next_emit += self.interval_seconds
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain_nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SupplyAuditError(f"{label} must be a non-negative integer")
    return value


def _json_scalar(value: Any) -> Any:
    """Return a lossless JSON-safe description of one malformed SQL scalar."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"sqlite_type": "blob", "hex": value.hex()}
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _checked_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SupplyAuditError(f"duplicate JSON key in SQLite authority: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise SupplyAuditError(f"non-finite JSON constant in SQLite authority: {value}")


def _parse_json_scalar(raw: Any, *, label: str) -> Any:
    if not isinstance(raw, str):
        raise SupplyAuditError(f"{label} metadata is not JSON text")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_checked_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SupplyAuditError(f"{label} metadata is invalid JSON") from exc


def _parse_json_object(raw: Any, *, label: str) -> dict[str, Any]:
    value = _parse_json_scalar(raw, label=label)
    if not isinstance(value, dict):
        raise SupplyAuditError(f"{label} must be a JSON object")
    return value


def _sidecar_state(path: Path) -> dict[str, Any]:
    try:
        status = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "bytes": 0, "mtime_ns": None}
    if not path.is_file():
        raise SupplyAuditError(f"SQLite sidecar is not a regular file: {path}")
    return {
        "path": str(path),
        "exists": True,
        "bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
    }


def _storage_observation(database: Path) -> dict[str, Any]:
    status = database.stat()
    return {
        "database": {
            "path": str(database),
            "bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns,
        },
        "wal": _sidecar_state(Path(f"{database}-wal")),
        "shm": _sidecar_state(Path(f"{database}-shm")),
        "rollback_journal": _sidecar_state(Path(f"{database}-journal")),
    }


def _required_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    objects = {
        str(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE name IN "
            "('metadata','archives','documents','events','groups','reasons',"
            "'phase_progress')"
        )
    }
    observed: dict[str, list[str]] = {}
    for table, required in REQUIRED_COLUMNS.items():
        if objects.get(table) != "table":
            raise SupplyAuditError(f"required curation table is missing: {table}")
        # Table names are constants above, never user input.
        columns = [
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        missing = sorted(required.difference(columns))
        if missing:
            raise SupplyAuditError(
                f"curation table {table} is missing required columns: {missing}"
            )
        observed[table] = columns
    return observed


def _metadata_authority(connection: sqlite3.Connection) -> tuple[int, str]:
    rows = list(
        connection.execute(
            "SELECT key, value FROM metadata "
            "WHERE key IN ('database_version','phase') ORDER BY key"
        )
    )
    values = {str(row[0]): row[1] for row in rows}
    if set(values) != {"database_version", "phase"} or len(rows) != 2:
        raise SupplyAuditError(
            "curation metadata lacks unique database_version/phase authority"
        )
    database_version = _parse_json_scalar(
        values["database_version"], label="database_version"
    )
    if (
        not isinstance(database_version, int)
        or isinstance(database_version, bool)
        or database_version != SUPPORTED_DATABASE_VERSION
    ):
        raise SupplyAuditError(
            "unsupported curation database version "
            f"{database_version!r}; expected {SUPPORTED_DATABASE_VERSION}"
        )
    phase = _parse_json_scalar(values["phase"], label="phase")
    if not isinstance(phase, str) or phase not in SAFE_PHASES:
        raise SupplyAuditError(
            f"curation phase {phase!r} is not safe for a final supply audit"
        )
    return database_version, phase


def _groups_authority(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT status, cursor_json, processed_rows, processed_tokens, "
        "committed_batches, details_json FROM phase_progress "
        "WHERE subphase='selection.groups'"
    ).fetchone()
    if row is None:
        raise SupplyAuditError("selection.groups has no completion authority")
    status = row[0]
    processed_rows = _plain_nonnegative_int(
        row[2], label="selection.groups processed_rows"
    )
    processed_tokens = _plain_nonnegative_int(
        row[3], label="selection.groups processed_tokens"
    )
    committed_batches = _plain_nonnegative_int(
        row[4], label="selection.groups committed_batches"
    )
    cursor = _parse_json_object(row[1], label="selection.groups cursor")
    details = _parse_json_object(row[5], label="selection.groups details")
    if status != "complete":
        raise SupplyAuditError(
            f"selection.groups is {status!r}, not complete; split totals are unsafe"
        )
    if processed_rows < 1:
        raise SupplyAuditError("selection.groups completed with no groups")
    if processed_tokens != 0:
        raise SupplyAuditError("selection.groups unexpectedly recorded tokens")
    if committed_batches < 1:
        raise SupplyAuditError("selection.groups completed with no committed batches")
    group_cursor = cursor.get("group_id")
    if (
        not isinstance(group_cursor, str)
        or len(group_cursor) != 64
        or any(character not in "0123456789abcdef" for character in group_cursor)
    ):
        raise SupplyAuditError("selection.groups has an invalid terminal cursor")
    details_groups = _plain_nonnegative_int(
        details.get("groups"), label="selection.groups details.groups"
    )
    mismatched = _plain_nonnegative_int(
        details.get("mismatched_assignments"),
        label="selection.groups details.mismatched_assignments",
    )
    actual_groups = _plain_nonnegative_int(
        connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
        label="groups table count",
    )
    if mismatched != 0:
        raise SupplyAuditError(
            f"selection.groups recorded {mismatched} mismatched assignments"
        )
    if details_groups != processed_rows or actual_groups != processed_rows:
        raise SupplyAuditError(
            "selection.groups row authority does not reconcile: "
            f"progress={processed_rows}, details={details_groups}, "
            f"table={actual_groups}"
        )
    return {
        "status": status,
        "processed_rows": processed_rows,
        "processed_tokens": processed_tokens,
        "committed_batches": committed_batches,
        "cursor": cursor,
        "details": details,
        "actual_table_rows": actual_groups,
    }


def _canonicalized_authority(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = list(
        connection.execute(
            "SELECT sequence, payload FROM events "
            "WHERE event='canonicalized' ORDER BY sequence"
        )
    )
    if len(rows) != 1:
        raise SupplyAuditError(
            f"expected exactly one canonicalized event, observed {len(rows)}"
        )
    sequence = _plain_nonnegative_int(rows[0][0], label="canonicalized sequence")
    payload = _parse_json_object(rows[0][1], label="canonicalized event payload")
    accepted = _plain_nonnegative_int(
        payload.get("accepted_canonical_documents"),
        label="canonicalized accepted_canonical_documents",
    )
    final_choices = _plain_nonnegative_int(
        payload.get("final_choices"), label="canonicalized final_choices"
    )
    if accepted < 1 or final_choices != accepted:
        raise SupplyAuditError(
            "canonicalized accepted/final-choice authority does not reconcile"
        )
    return {
        "sequence": sequence,
        "accepted_canonical_documents": accepted,
        "payload": payload,
    }


def _zero_counter() -> dict[str, int]:
    return {"documents": 0, "tokens": 0}


def _add(counter: dict[str, int], documents: int, tokens: int) -> None:
    counter["documents"] += documents
    counter["tokens"] += tokens


def _aggregate(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    by_split = {split: _zero_counter() for split in SPLITS}
    by_category = {category: _zero_counter() for category in CATEGORIES}
    by_bucket = {bucket: _zero_counter() for bucket in BUCKET_CATEGORY}
    by_split_category = {
        split: {category: _zero_counter() for category in CATEGORIES}
        for split in SPLITS
    }
    by_split_bucket = {
        split: {bucket: _zero_counter() for bucket in BUCKET_CATEGORY}
        for split in SPLITS
    }
    observed = _zero_counter()
    assigned_valid = _zero_counter()
    cells: list[dict[str, Any]] = []
    missing_group_cells: list[dict[str, Any]] = []
    unknown_bucket_cells: list[dict[str, Any]] = []
    unknown_split_cells: list[dict[str, Any]] = []
    invalid_token_cells: list[dict[str, Any]] = []

    for row in rows:
        bucket = row["bucket"]
        split = row["split"]
        documents = _plain_nonnegative_int(row["documents"], label="cell documents")
        tokens = _plain_nonnegative_int(row["tokens"], label="cell tokens")
        invalid_tokens = _plain_nonnegative_int(
            row["invalid_token_documents"], label="invalid token documents"
        )
        if documents < 1:
            raise SupplyAuditError("grouped documents scan returned an empty cell")
        if invalid_tokens > documents:
            raise SupplyAuditError("invalid-token counter exceeds its cell document count")

        category = BUCKET_CATEGORY.get(bucket) if isinstance(bucket, str) else None
        flags: list[str] = []
        if split is None:
            flags.append("missing_group")
        elif split not in SPLITS:
            flags.append("unknown_split")
        if category is None:
            flags.append("unknown_bucket")
        if invalid_tokens:
            flags.append("invalid_tokens")
        cell = {
            "bucket": _json_scalar(bucket),
            "category": category,
            "split": _json_scalar(split),
            "documents": documents,
            "tokens": tokens,
            "invalid_token_documents": invalid_tokens,
            "flags": flags,
        }
        cells.append(cell)
        _add(observed, documents, tokens)
        if "missing_group" in flags:
            missing_group_cells.append(cell)
        if "unknown_bucket" in flags:
            unknown_bucket_cells.append(cell)
        if "unknown_split" in flags:
            unknown_split_cells.append(cell)
        if "invalid_tokens" in flags:
            invalid_token_cells.append(cell)
        if flags:
            continue

        assert isinstance(bucket, str) and isinstance(split, str) and category is not None
        _add(assigned_valid, documents, tokens)
        _add(by_split[split], documents, tokens)
        _add(by_category[category], documents, tokens)
        _add(by_bucket[bucket], documents, tokens)
        _add(by_split_category[split][category], documents, tokens)
        _add(by_split_bucket[split][bucket], documents, tokens)

    anomaly_documents = {
        "missing_group": sum(cell["documents"] for cell in missing_group_cells),
        "unknown_bucket": sum(cell["documents"] for cell in unknown_bucket_cells),
        "unknown_split": sum(cell["documents"] for cell in unknown_split_cells),
        "invalid_tokens": sum(
            cell["invalid_token_documents"] for cell in invalid_token_cells
        ),
    }
    safe = all(value == 0 for value in anomaly_documents.values())
    if safe and observed != assigned_valid:
        raise SupplyAuditError("safe-cell accounting does not reconcile")
    return {
        "safe": safe,
        "totals": {
            "eligible_observed": observed,
            "assigned_valid": assigned_valid,
            "by_split": by_split,
            "by_category": by_category,
            "by_bucket": by_bucket,
            "by_split_category": by_split_category,
            "by_split_bucket": by_split_bucket,
        },
        "observed_cells": cells,
        "anomalies": {
            "document_counts": anomaly_documents,
            "missing_groups": missing_group_cells,
            "unknown_buckets": unknown_bucket_cells,
            "unknown_splits": unknown_split_cells,
            "invalid_tokens": invalid_token_cells,
        },
    }


def _aggregate_bucket_rows(
    rows: Sequence[sqlite3.Row], *, label: str, include_archives: bool = False
) -> dict[str, Any]:
    by_bucket = {bucket: _zero_counter() for bucket in BUCKET_CATEGORY}
    archives_by_bucket = {bucket: 0 for bucket in BUCKET_CATEGORY}
    unknown: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        bucket = row["bucket"]
        documents = _plain_nonnegative_int(
            row["documents"], label=f"{label} documents"
        )
        tokens = _plain_nonnegative_int(row["tokens"], label=f"{label} tokens")
        record: dict[str, Any] = {
            "bucket": _json_scalar(bucket),
            "documents": documents,
            "tokens": tokens,
        }
        if include_archives:
            record["archives"] = _plain_nonnegative_int(
                row["archives"], label=f"{label} archives"
            )
        if not isinstance(bucket, str) or bucket not in BUCKET_CATEGORY:
            unknown.append(record)
            continue
        if bucket in seen:
            raise SupplyAuditError(f"{label} returned duplicate bucket {bucket!r}")
        seen.add(bucket)
        by_bucket[bucket] = {"documents": documents, "tokens": tokens}
        if include_archives:
            archives_by_bucket[bucket] = int(record["archives"])

    by_category = {category: _zero_counter() for category in CATEGORIES}
    total = _zero_counter()
    for bucket, counter in by_bucket.items():
        _add(total, counter["documents"], counter["tokens"])
        _add(
            by_category[BUCKET_CATEGORY[bucket]],
            counter["documents"],
            counter["tokens"],
        )
    result: dict[str, Any] = {
        "total": total,
        "by_category": by_category,
        "by_bucket": by_bucket,
        "unknown_buckets": unknown,
    }
    if include_archives:
        result["archives_by_bucket"] = archives_by_bucket
        result["missing_buckets"] = [
            bucket for bucket in BUCKET_CATEGORY if archives_by_bucket[bucket] < 1
        ]
    return result


def _aggregate_reason_rows(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    associations: list[dict[str, Any]] = []
    by_reason: dict[str, dict[str, int]] = {}
    total = _zero_counter()
    orphan_rows = 0
    invalid_rows: list[dict[str, Any]] = []
    for row in rows:
        bucket = row["bucket"]
        reason = row["reason"]
        documents = _plain_nonnegative_int(
            row["documents"], label="reason-association documents"
        )
        tokens = _plain_nonnegative_int(
            row["tokens"], label="reason-association tokens"
        )
        orphans = _plain_nonnegative_int(
            row["orphan_reason_rows"], label="orphan reason rows"
        )
        record = {
            "bucket": _json_scalar(bucket),
            "category": (
                BUCKET_CATEGORY.get(bucket) if isinstance(bucket, str) else None
            ),
            "reason": _json_scalar(reason),
            "documents": documents,
            "tokens": tokens,
            "orphan_reason_rows": orphans,
        }
        associations.append(record)
        _add(total, documents, tokens)
        orphan_rows += orphans
        if (
            not isinstance(bucket, str)
            or bucket not in BUCKET_CATEGORY
            or not isinstance(reason, str)
            or not reason
            or orphans
        ):
            invalid_rows.append(record)
            continue
        counter = by_reason.setdefault(reason, _zero_counter())
        _add(counter, documents, tokens)
    return {
        "unit": "reason_associations_non_additive_across_reasons",
        "total_associations": total,
        "by_reason": dict(sorted(by_reason.items())),
        "rows": associations,
        "orphan_reason_rows": orphan_rows,
        "invalid_rows": invalid_rows,
    }


def _reconcile_raw_accounting(
    *, eligible: dict[str, Any], rejected: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    safe = not raw["unknown_buckets"] and not rejected["unknown_buckets"]
    safe = safe and not raw.get("missing_buckets")
    for bucket in BUCKET_CATEGORY:
        raw_counter = raw["by_bucket"][bucket]
        eligible_counter = eligible["by_bucket"][bucket]
        rejected_counter = rejected["by_bucket"][bucket]
        expected_documents = eligible_counter["documents"] + rejected_counter["documents"]
        expected_tokens = eligible_counter["tokens"] + rejected_counter["tokens"]
        cell_safe = (
            raw_counter["documents"] == expected_documents
            and raw_counter["tokens"] == expected_tokens
        )
        safe = safe and cell_safe
        cells.append(
            {
                "bucket": bucket,
                "raw": raw_counter,
                "eligible": eligible_counter,
                "unique_rejected": rejected_counter,
                "document_delta": raw_counter["documents"] - expected_documents,
                "token_delta": raw_counter["tokens"] - expected_tokens,
                "safe": cell_safe,
            }
        )
    return {
        "equation": "raw = eligible + unique_rejected",
        "safe": safe,
        "cells": cells,
    }


def _configure_read_performance(connection: sqlite3.Connection) -> dict[str, Any]:
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(f"PRAGMA cache_size=-{REQUESTED_CACHE_KIB}")
    connection.execute(f"PRAGMA mmap_size={REQUESTED_MMAP_BYTES}")
    temp_store = int(connection.execute("PRAGMA temp_store").fetchone()[0])
    cache_size = int(connection.execute("PRAGMA cache_size").fetchone()[0])
    mmap_row = connection.execute("PRAGMA mmap_size").fetchone()
    mmap_size = 0 if mmap_row is None else int(mmap_row[0])
    if temp_store != 2 or cache_size != -REQUESTED_CACHE_KIB or mmap_size < 0:
        raise SupplyAuditError("SQLite refused safe read-performance settings")
    return {
        "temp_store": {"requested": "MEMORY", "observed": temp_store},
        "cache_size_kib": {
            "requested": REQUESTED_CACHE_KIB,
            "observed_pragma": cache_size,
        },
        "mmap_size_bytes": {
            "requested": REQUESTED_MMAP_BYTES,
            "observed": mmap_size,
        },
        "durability_pragmas_modified": False,
    }


def _is_primary_documents_scan(detail: str) -> bool:
    """Accept equivalent modern and legacy SQLite EQP wording."""

    return detail in ("SCAN d", "SCAN TABLE documents AS d")


def _supply_query_plan(connection: sqlite3.Connection) -> list[str]:
    plan = [
        str(row[3])
        for row in connection.execute("EXPLAIN QUERY PLAN " + SUPPLY_SQL)
    ]
    document_scans = [
        detail
        for detail in plan
        if _is_primary_documents_scan(detail)
    ]
    if len(document_scans) != 1:
        raise SupplyAuditError(
            "eligible-supply query is not pinned to one primary-table documents scan: "
            f"{plan}"
        )
    if any(
        index in detail
        for detail in plan
        for index in ("documents_selection_v2", "documents_source_group")
    ):
        raise SupplyAuditError(
            f"eligible-supply query selected a non-covering documents index: {plan}"
        )
    return plan


def audit_database(
    database: Path,
    *,
    timeout_seconds: float = 120.0,
    heartbeat_seconds: float | None = None,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    if not isinstance(timeout_seconds, (int, float)) or isinstance(
        timeout_seconds, bool
    ):
        raise SupplyAuditError("timeout_seconds must be numeric")
    if timeout_seconds <= 0 or not math.isfinite(float(timeout_seconds)):
        raise SupplyAuditError("timeout_seconds must be positive")
    if heartbeat_seconds is not None and (
        not isinstance(heartbeat_seconds, (int, float))
        or isinstance(heartbeat_seconds, bool)
        or heartbeat_seconds <= 0
        or not math.isfinite(float(heartbeat_seconds))
    ):
        raise SupplyAuditError("heartbeat_seconds must be positive when supplied")
    heartbeat = (
        _Heartbeat(
            float(heartbeat_seconds),
            progress_stream if progress_stream is not None else sys.stderr,
        )
        if heartbeat_seconds is not None
        else None
    )
    if database.is_symlink():
        raise SupplyAuditError(f"SQLite database path must not be a symlink: {database}")
    try:
        database = database.resolve(strict=True)
    except OSError as exc:
        raise SupplyAuditError(f"cannot resolve SQLite database {database}: {exc}") from exc
    if database.is_symlink() or not database.is_file():
        raise SupplyAuditError(f"SQLite database must be a regular file: {database}")

    # Path.as_uri() percent-encodes URI metacharacters in filenames.  Appending
    # only mode=ro is intentional: never add immutable=1 for a live WAL reader.
    sqlite_uri = f"{database.as_uri()}?mode=ro"
    try:
        storage_before = _storage_observation(database)
    except OSError as exc:
        raise SupplyAuditError(
            f"cannot inspect SQLite database/sidecars before audit: {exc}"
        ) from exc
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            sqlite_uri,
            uri=True,
            timeout=float(timeout_seconds),
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise SupplyAuditError("SQLite refused query_only mode")
        performance = _configure_read_performance(connection)
        if heartbeat is not None:
            connection.set_progress_handler(
                heartbeat.sqlite_progress, SQLITE_PROGRESS_OPCODES
            )
            heartbeat.set_stage("authority")
        connection.execute("BEGIN")

        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
        schema_cookie = _plain_nonnegative_int(
            connection.execute("PRAGMA schema_version").fetchone()[0],
            label="SQLite schema_version",
        )
        data_version_before = _plain_nonnegative_int(
            connection.execute("PRAGMA data_version").fetchone()[0],
            label="SQLite data_version",
        )
        schema = _required_schema(connection)
        database_version, phase = _metadata_authority(connection)
        group_authority = _groups_authority(connection)
        canonicalized_authority = _canonicalized_authority(connection)
        supply_query_plan = _supply_query_plan(connection)

        if heartbeat is not None:
            heartbeat.set_stage("eligible_supply")

        # The only documents scan in this program.
        aggregate = _aggregate(list(connection.execute(SUPPLY_SQL)))
        if (
            aggregate["totals"]["eligible_observed"]["documents"]
            != canonicalized_authority["accepted_canonical_documents"]
        ):
            raise SupplyAuditError(
                "eligible document total does not match canonicalized authority: "
                f"scan={aggregate['totals']['eligible_observed']['documents']}, "
                "event="
                f"{canonicalized_authority['accepted_canonical_documents']}"
            )

        if heartbeat is not None:
            heartbeat.set_stage("raw_and_rejections")
        raw = _aggregate_bucket_rows(
            list(connection.execute(RAW_TOTALS_SQL)),
            label="raw archive",
            include_archives=True,
        )
        rejected = _aggregate_bucket_rows(
            list(connection.execute(REJECTED_TOTALS_SQL)),
            label="unique rejected",
        )
        reason_associations = _aggregate_reason_rows(
            list(connection.execute(REASON_TOTALS_SQL))
        )
        accounting = _reconcile_raw_accounting(
            eligible=aggregate["totals"], rejected=rejected, raw=raw
        )
        safe = (
            aggregate["safe"]
            and accounting["safe"]
            and not reason_associations["invalid_rows"]
        )
        data_version_after = _plain_nonnegative_int(
            connection.execute("PRAGMA data_version").fetchone()[0],
            label="SQLite data_version",
        )
        database_list = [
            {"sequence": int(row[0]), "name": str(row[1]), "path": str(row[2])}
            for row in connection.execute("PRAGMA database_list")
        ]
        connection.execute("ROLLBACK")
        if heartbeat is not None:
            heartbeat.set_stage("complete")
    except (OSError, sqlite3.Error) as exc:
        raise SupplyAuditError(f"SQLite read-only supply audit failed: {exc}") from exc
    finally:
        if connection is not None:
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            finally:
                connection.close()

    try:
        storage_after = _storage_observation(database)
    except OSError as exc:
        raise SupplyAuditError(
            f"cannot inspect SQLite database/sidecars after audit: {exc}"
        ) from exc
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "generated_at_utc": _utc_now(),
        "safe": safe,
        "scope": {
            "population": "documents_without_any_reason",
            "split_authority": "groups.split_joined_by_documents.source_group",
            "token_unit": "pre_packing_starcoder2_content_tokens",
            "documents_scans": 1,
            "rejection_semantics": (
                "unique_rejected counts each reason-bearing document once; "
                "reason_associations are non-additive"
            ),
        },
        "provenance": {
            "sqlite_open": {
                "uri": sqlite_uri,
                "mode": "ro",
                "immutable": False,
                "query_only": True,
                "transaction": "single_deferred_read_snapshot",
                "wal_visibility": "enabled_by_non_immutable_mode_ro",
            },
            "sqlite_library_version": sqlite3.sqlite_version,
            "journal_mode": journal_mode,
            "database_version": database_version,
            "schema_version": schema_cookie,
            "data_version_before": data_version_before,
            "data_version_after": data_version_after,
            "database_list": database_list,
            "read_performance": performance,
            "eligible_supply_query_plan": supply_query_plan,
            "phase": phase,
            "selection_groups": group_authority,
            "canonicalized": canonicalized_authority,
            "validated_schema": schema,
            "storage_before": storage_before,
            "storage_after": storage_after,
        },
        "totals": aggregate["totals"],
        "raw_input": raw,
        "unique_rejected": rejected,
        "reason_associations": reason_associations,
        "raw_accounting": accounting,
        "observed_cells": aggregate["observed_cells"],
        "anomalies": aggregate["anomalies"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, required=True, help="live curation.sqlite3"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="SQLite busy timeout for opening a coherent read snapshot (default: 120)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=None,
        help=(
            "emit stage/elapsed heartbeats to stderr at this interval; JSON stdout "
            "remains machine-readable"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = audit_database(
            args.database,
            timeout_seconds=args.timeout_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
            progress_stream=sys.stderr,
        )
    except SupplyAuditError as exc:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "safe": False,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["safe"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
