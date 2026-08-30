#!/usr/bin/env python3
"""Paired benchmark for the opt-in append-only curation inventory stage.

The baseline uses the current curator's document/reason DDL.  Both paths hash
the complete compressed fingerprint before inserting, validate the same scalar
records, use DELETE/FULL SQLite durability, and commit the requested bounded
batch size.  Results distinguish hot ingest from total query-ready time so
deferred index work cannot be hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import zstandard

from curation_inventory_stage import (
    InventoryStage,
    ReportSpec,
    StagePolicy,
    canonical_json_bytes,
    file_sha256,
    iter_jsonl_zst,
    validate_fingerprint_record,
)
from curation_policy import FAST_CANONICAL_POLICY, load_policy
from preprocess_raw_stream import FINGERPRINT_VERSION


CURRENT_DOCUMENT_DDL = """
CREATE TABLE documents (
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
CREATE TABLE reasons (
    doc_id BLOB NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY(doc_id, reason)
) WITHOUT ROWID;
"""


def configure(connection: sqlite3.Connection) -> None:
    mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])
    if mode.casefold() != "delete":
        raise RuntimeError("SQLite refused DELETE journal mode")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA mmap_size=8589934592")


def generate_fixture(root: Path, rows: int) -> ReportSpec:
    staging = root / "staging"
    fingerprint = staging / "fingerprints" / "python" / "part-000000.jsonl.zst"
    fingerprint.parent.mkdir(parents=True, exist_ok=True)
    archive = "raw/stack-v3/python/part-000000.tar.zst"
    token_total = 0
    byte_total = 0
    with fingerprint.open("wb") as raw:
        with zstandard.ZstdCompressor(level=1).stream_writer(
            raw, closefd=False
        ) as writer:
            for index in range(rows):
                member_path = f"files/{index:012d}.py"
                tokens = 16 + index % 113
                size_bytes = 80 + index % 2048
                flags = ["too_short"] if index % 10 == 0 else []
                row = {
                    "record_version": 1,
                    "fingerprint_version": FINGERPRINT_VERSION,
                    "doc_id": hashlib.sha256(
                        f"{archive}\0{member_path}".encode()
                    ).hexdigest(),
                    "bucket": "python",
                    "archive": archive,
                    "manifest_index": index,
                    "member_path": member_path,
                    "starcoder2_tokens": tokens,
                    "size_bytes": size_bytes,
                    "content_sha256": hashlib.sha256(
                        f"content:{index}".encode()
                    ).hexdigest(),
                    "normalized_sha256": hashlib.sha256(
                        f"normalized:{index}".encode()
                    ).hexdigest(),
                    "quality_flags": flags,
                    "benchmark_reason": None,
                    "provenance": {"repo_id": f"repo-{index // 16}"},
                }
                writer.write(canonical_json_bytes(row) + b"\n")
                token_total += tokens
                byte_total += size_bytes
    report_path = staging / "reports" / "python" / "part-000000.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "archive": archive,
        "index": 0,
        "bucket": "python",
        "documents": rows,
        "exact_tokens": token_total,
        "clean_bytes": byte_total,
        "fingerprint_file": str(fingerprint.relative_to(staging)),
        "fingerprint_sha256": file_sha256(fingerprint),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReportSpec.from_report(
        ordinal=0, report_path=report_path, staging_root=staging
    )


def baseline_inventory(
    database: Path,
    *,
    report: ReportSpec,
    policy: StagePolicy,
    batch_size: int,
) -> None:
    if file_sha256(report.report_path) != report.report_sha256:
        raise RuntimeError("Baseline report checksum mismatch")
    if file_sha256(report.fingerprint_path) != report.fingerprint_sha256:
        raise RuntimeError("Baseline fingerprint checksum mismatch")
    connection = sqlite3.connect(database, timeout=120)
    try:
        configure(connection)
        connection.executescript(CURRENT_DOCUMENT_DDL)
        document_batch: list[tuple[Any, ...]] = []
        reason_batch: list[tuple[bytes, str]] = []
        documents = tokens = clean_bytes = 0

        def flush() -> None:
            if not document_batch:
                return
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    """
                    INSERT INTO documents(
                        doc_id, bucket, archive, manifest_index, member_path,
                        tokens, content_hash, normalized_hash, final_cluster,
                        source_group, canonical_rank, selection_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    document_batch,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO reasons(doc_id, reason) VALUES (?, ?)",
                    reason_batch,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            document_batch.clear()
            reason_batch.clear()

        for index, row in enumerate(iter_jsonl_zst(report.fingerprint_path)):
            document = validate_fingerprint_record(
                row, expected_index=index, report=report, policy=policy
            )
            values = document.document_values(stage_id=0, archive_ordinal=0)
            document_batch.append(values[2:])
            reason_batch.extend(
                (document.doc_id, reason) for reason in document.reasons
            )
            documents += 1
            tokens += document.tokens
            clean_bytes += document.size_bytes
            if len(document_batch) >= batch_size:
                flush()
        flush()
        if (documents, tokens, clean_bytes) != (
            report.documents,
            report.tokens,
            report.clean_bytes,
        ):
            raise RuntimeError("Baseline totals mismatch")
        if str(connection.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise RuntimeError("Baseline SQLite integrity failure")
    finally:
        connection.close()


def target_rows(connection: sqlite3.Connection) -> Iterator[Sequence[Any]]:
    yield from connection.execute(
        """
        SELECT doc_id, bucket, archive, manifest_index, member_path, tokens,
               content_hash, normalized_hash, final_cluster, source_group,
               canonical_rank, selection_rank
        FROM documents ORDER BY doc_id
        """
    )
    yield from connection.execute(
        "SELECT doc_id, reason FROM reasons ORDER BY doc_id, reason"
    )


def target_sha256(database: Path) -> str:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        for row in target_rows(connection):
            payload = [
                value.hex() if isinstance(value, bytes) else value for value in row
            ]
            digest.update(canonical_json_bytes(payload) + b"\n")
        return digest.hexdigest()
    finally:
        connection.close()


def remove_database(path: Path) -> None:
    candidates = (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    )
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_benchmark(
    root: Path, *, rows: int, batch_size: int, repeats: int
) -> dict[str, Any]:
    report = generate_fixture(root, rows)
    policy = StagePolicy.from_curation_policy(
        load_policy(FAST_CANONICAL_POLICY), fast_canonical_profile=True
    )
    baseline_times: list[float] = []
    stage_ingest_times: list[float] = []
    promotion_times: list[float] = []
    total_times: list[float] = []
    database_sizes: dict[str, int] = {}
    for repetition in range(repeats):
        baseline = root / f"baseline-{repetition}.sqlite3"
        staged = root / f"staged-{repetition}.sqlite3"
        remove_database(baseline)
        remove_database(staged)
        # Alternate order to reduce simple warm-cache/order bias.
        order = ("baseline", "staged") if repetition % 2 == 0 else ("staged", "baseline")
        for variant in order:
            if variant == "baseline":
                started = time.perf_counter()
                baseline_inventory(
                    baseline,
                    report=report,
                    policy=policy,
                    batch_size=batch_size,
                )
                baseline_times.append(time.perf_counter() - started)
            else:
                with InventoryStage(
                    staged,
                    reports=[report],
                    policy=policy,
                    batch_size=batch_size,
                ) as stage:
                    started = time.perf_counter()
                    stage.ingest_all()
                    ingest_elapsed = time.perf_counter() - started
                    promoted = time.perf_counter()
                    stage.promote()
                    promotion_elapsed = time.perf_counter() - promoted
                stage_ingest_times.append(ingest_elapsed)
                promotion_times.append(promotion_elapsed)
                total_times.append(ingest_elapsed + promotion_elapsed)
        baseline_digest = target_sha256(baseline)
        staged_digest = target_sha256(staged)
        if baseline_digest != staged_digest:
            raise RuntimeError("Staged logical rows differ from current schema")
        database_sizes = {
            "baseline_bytes": baseline.stat().st_size,
            "staged_bytes": staged.stat().st_size,
        }

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "median_seconds": statistics.median(values),
            "p95_seconds": percentile(values, 0.95),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }

    baseline_median = statistics.median(baseline_times)
    ingest_median = statistics.median(stage_ingest_times)
    total_median = statistics.median(total_times)
    return {
        "benchmark_version": 1,
        "rows": rows,
        "batch_size": batch_size,
        "repeats": repeats,
        "fingerprint_sha256": report.fingerprint_sha256,
        "logical_rows_sha256": target_sha256(root / f"staged-{repeats - 1}.sqlite3"),
        "durability": {"journal_mode": "DELETE", "synchronous": "FULL"},
        "baseline": summary(baseline_times),
        "append_only_ingest": summary(stage_ingest_times),
        "bulk_promotion": summary(promotion_times),
        "append_only_query_ready_total": summary(total_times),
        "median_ingest_speedup": baseline_median / ingest_median,
        "median_query_ready_speedup": baseline_median / total_median,
        "rows_per_second": {
            "baseline": rows / baseline_median,
            "append_only_ingest": rows / ingest_median,
            "append_only_query_ready": rows / total_median,
        },
        "database_sizes": database_sizes,
        "runtime": {
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
            "zstandard": zstandard.__version__,
            "platform": platform.platform(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0 or args.repeats <= 0 or args.batch_size <= 0:
        raise SystemExit("rows, repeats, and batch-size must be positive")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="curation-stage-benchmark-")
        root = Path(temporary.name)
    else:
        root = args.work_root.resolve()
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"--work-root must be empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    try:
        result = run_benchmark(
            root,
            rows=args.rows,
            batch_size=args.batch_size,
            repeats=args.repeats,
        )
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.result is not None:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            temporary_result = args.result.with_name(f".{args.result.name}.tmp")
            temporary_result.write_text(encoded, encoding="utf-8")
            temporary_result.replace(args.result)
        sys.stdout.write(encoded)
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
