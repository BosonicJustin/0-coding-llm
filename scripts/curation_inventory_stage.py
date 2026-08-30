#!/usr/bin/env python3
"""Append-only, restart-safe staging for curation inventory construction.

This module is deliberately not wired into ``curate_corpus.py``.  It provides a
benchmarkable next-generation inventory engine whose hot ingest table has only
an ``INTEGER PRIMARY KEY``.  Document/reason uniqueness indexes are bulk-built
after the complete frozen report inventory has been committed.

The promoted tables preserve the logical columns and uniqueness semantics used
by the current curator.  Their physical layout is intentionally different, so
integration requires a new curation database/identity version.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import zstandard

from curation_policy import ALL_BUCKETS, CODE_BUCKETS
from preprocess_raw_stream import FINGERPRINT_VERSION


STAGE_FORMAT_VERSION = 1
DEFAULT_BATCH_SIZE = 100_000
MAXIMUM_BATCH_SIZE = 1_000_000
HEX64 = frozenset("0123456789abcdef")
ENGLISH_BUCKETS = frozenset(("fineweb_edu", "wikipedia"))
PROMOTION_INDEXES = (
    (
        "documents_doc_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS documents_doc_id "
        "ON stage_documents(doc_id)",
    ),
    (
        "documents_archive_manifest",
        "CREATE UNIQUE INDEX IF NOT EXISTS documents_archive_manifest "
        "ON stage_documents(archive, manifest_index)",
    ),
    (
        "documents_archive_member",
        "CREATE UNIQUE INDEX IF NOT EXISTS documents_archive_member "
        "ON stage_documents(archive, member_path)",
    ),
    (
        "reasons_doc_reason",
        "CREATE UNIQUE INDEX IF NOT EXISTS reasons_doc_reason "
        "ON stage_reasons(doc_id, reason)",
    ),
)


class InventoryStageError(RuntimeError):
    """Raised when staging input, state, or promotion is inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX64 for character in value)
    ):
        raise InventoryStageError(f"{field} must be a lowercase SHA-256")
    return value


def parse_hex_digest(value: Any, field: str) -> bytes:
    return bytes.fromhex(_require_sha256(value, field))


def stable_digest(namespace: str, value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(namespace.encode("utf-8") + b"\0" + raw).digest()


def english_source_identity(bucket: str, provenance: Mapping[str, Any]) -> str | None:
    keys = (
        ("url", "id")
        if bucket == "fineweb_edu"
        else (("id", "url", "title") if bucket == "wikipedia" else ())
    )
    for key in keys:
        value = provenance.get(key)
        if value is not None and str(value).strip():
            return f"{bucket}:{key}:{str(value).strip()}"
    return None


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(
            raw, read_across_frames=True, closefd=False
        )
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="strict")
        try:
            for line_number, line in enumerate(text, 1):
                if not line.strip():
                    raise InventoryStageError(
                        f"Blank fingerprint row in {path}:{line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise InventoryStageError(
                        f"Invalid fingerprint JSON in {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise InventoryStageError(
                        f"Non-object fingerprint row in {path}:{line_number}"
                    )
                yield row
        finally:
            text.close()


@dataclass(frozen=True)
class ReportSpec:
    ordinal: int
    report_path: Path
    report_sha256: str
    fingerprint_path: Path
    fingerprint_sha256: str
    archive: str
    archive_index: int
    bucket: str
    documents: int
    tokens: int
    clean_bytes: int

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.archive_index < 0:
            raise InventoryStageError("Report ordinals must be non-negative")
        if self.bucket not in ALL_BUCKETS:
            raise InventoryStageError(f"Unsupported report bucket: {self.bucket}")
        if not self.archive or min(self.documents, self.tokens, self.clean_bytes) <= 0:
            raise InventoryStageError("Report counts and archive identity must be positive")
        _require_sha256(self.report_sha256, "report_sha256")
        _require_sha256(self.fingerprint_sha256, "fingerprint_sha256")

    @classmethod
    def from_report(
        cls,
        *,
        ordinal: int,
        report_path: Path,
        staging_root: Path,
        expected_report_sha256: str | None = None,
    ) -> "ReportSpec":
        raw = report_path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if expected_report_sha256 is not None and checksum != expected_report_sha256:
            raise InventoryStageError(f"Report checksum mismatch: {report_path}")
        try:
            report = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InventoryStageError(f"Invalid report JSON: {report_path}") from error
        if not isinstance(report, dict):
            raise InventoryStageError(f"Report must be an object: {report_path}")
        try:
            for field in ("index", "documents", "exact_tokens", "clean_bytes"):
                value = report[field]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise InventoryStageError(
                        f"Report {field} must be an integer: {report_path}"
                    )
            fingerprint = staging_root / str(report["fingerprint_file"])
            return cls(
                ordinal=ordinal,
                report_path=report_path,
                report_sha256=checksum,
                fingerprint_path=fingerprint,
                fingerprint_sha256=str(report["fingerprint_sha256"]),
                archive=str(report["archive"]),
                archive_index=int(report["index"]),
                bucket=str(report["bucket"]),
                documents=int(report["documents"]),
                tokens=int(report["exact_tokens"]),
                clean_bytes=int(report["clean_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InventoryStageError(f"Incomplete report: {report_path}") from error

    def identity(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "report_path": str(self.report_path.resolve()),
            "report_sha256": self.report_sha256,
            "fingerprint_path": str(self.fingerprint_path.resolve()),
            "fingerprint_sha256": self.fingerprint_sha256,
            "archive": self.archive,
            "archive_index": self.archive_index,
            "bucket": self.bucket,
            "documents": self.documents,
            "tokens": self.tokens,
            "clean_bytes": self.clean_bytes,
        }


@dataclass(frozen=True)
class StagePolicy:
    seed: str
    hard_reject_flags: tuple[str, ...]
    hard_reject_flags_by_bucket: Mapping[str, tuple[str, ...]]
    soft_penalty_weights: Mapping[str, int]
    fast_canonical_profile: bool

    @classmethod
    def from_curation_policy(
        cls, policy: Mapping[str, Any], *, fast_canonical_profile: bool
    ) -> "StagePolicy":
        try:
            selection = policy["selection"]
            quality = selection["quality"]
            seed = selection["seed"]
            hard = tuple(sorted(set(quality["hard_reject_flags"])))
            by_bucket = {
                bucket: tuple(
                    sorted(set(quality["hard_reject_flags_by_bucket"][bucket]))
                )
                for bucket in sorted(ALL_BUCKETS)
            }
            weights = {
                str(flag): int(weight)
                for flag, weight in quality["soft_penalty_weights"].items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise InventoryStageError("Invalid curation policy projection") from error
        if not isinstance(seed, str) or not seed:
            raise InventoryStageError("Selection seed must be a non-empty string")
        if any(weight < 0 or weight >= 1 << 64 for weight in weights.values()):
            raise InventoryStageError("Soft penalty weights must fit unsigned 64-bit")
        return cls(seed, hard, by_bucket, weights, fast_canonical_profile)

    def identity(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "hard_reject_flags": list(self.hard_reject_flags),
            "hard_reject_flags_by_bucket": {
                key: list(value)
                for key, value in sorted(self.hard_reject_flags_by_bucket.items())
            },
            "soft_penalty_weights": dict(sorted(self.soft_penalty_weights.items())),
            "fast_canonical_profile": self.fast_canonical_profile,
        }


@dataclass(frozen=True)
class ValidatedDocument:
    doc_id: bytes
    bucket: str
    archive: str
    manifest_index: int
    member_path: str
    tokens: int
    size_bytes: int
    content_hash: bytes
    normalized_hash: bytes
    final_cluster: bytes
    source_group: bytes
    canonical_rank: bytes
    selection_rank: bytes
    reasons: tuple[str, ...]

    def document_values(self, *, stage_id: int, archive_ordinal: int) -> tuple[Any, ...]:
        return (
            stage_id,
            archive_ordinal,
            self.doc_id,
            self.bucket,
            self.archive,
            self.manifest_index,
            self.member_path,
            self.tokens,
            self.content_hash,
            self.normalized_hash,
            self.final_cluster,
            self.source_group,
            self.canonical_rank,
            self.selection_rank,
        )


def _digest_batch_payload(
    document_values: Iterable[Sequence[Any]],
    reason_values: Iterable[Sequence[Any]],
) -> str:
    """Digest the exact staged scalar payload, independent of SQLite encoding."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            digest.update(b"B" + struct.pack(">Q", len(raw)) + raw)
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            digest.update(b"S" + struct.pack(">Q", len(raw)) + raw)
        elif isinstance(value, int):
            # SQLite stores bool using its integer representation; normalize it
            # here so ingest-time and read-back digests remain identical.
            raw = str(int(value)).encode("ascii")
            digest.update(b"I" + struct.pack(">Q", len(raw)) + raw)
        else:
            raise InventoryStageError(
                f"Unsupported scalar in staged batch digest: {type(value).__name__}"
            )

    for values in document_values:
        digest.update(b"D" + struct.pack(">Q", len(values)))
        for value in values:
            update(value)
    for values in reason_values:
        digest.update(b"R" + struct.pack(">Q", len(values)))
        for value in values:
            update(value)
    return digest.hexdigest()


def validate_fingerprint_record(
    row: Mapping[str, Any],
    *,
    expected_index: int,
    report: ReportSpec,
    policy: StagePolicy,
) -> ValidatedDocument:
    if (
        row.get("record_version") != 1
        or row.get("fingerprint_version") != FINGERPRINT_VERSION
    ):
        raise InventoryStageError("Unsupported fingerprint record")
    if row.get("bucket") != report.bucket or row.get("archive") != report.archive:
        raise InventoryStageError("Fingerprint report identity mismatch")
    if row.get("manifest_index") != expected_index:
        raise InventoryStageError("Non-contiguous manifest index")
    member_path = row.get("member_path")
    if not isinstance(member_path, str) or not member_path:
        raise InventoryStageError("Invalid member path")
    expected_doc = hashlib.sha256(
        f"{report.archive}\0{member_path}".encode("utf-8")
    ).hexdigest()
    if row.get("doc_id") != expected_doc:
        raise InventoryStageError("Unstable document identity")
    doc_id = parse_hex_digest(row.get("doc_id"), "doc_id")
    content_hash = parse_hex_digest(row.get("content_sha256"), "content_sha256")
    normalized_hash = parse_hex_digest(
        row.get("normalized_sha256"), "normalized_sha256"
    )
    token_count = row.get("starcoder2_tokens")
    size_bytes = row.get("size_bytes")
    # Preserve the current curator's scalar semantics exactly.  In particular,
    # bool is an int subclass and is therefore accepted here just as it is by
    # ``curate_corpus.py``.  Tightening this requires a fingerprint version bump.
    if not isinstance(token_count, int) or token_count <= 0:
        raise InventoryStageError("Invalid token count")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise InventoryStageError("Invalid byte count")
    flags = row.get("quality_flags")
    if (
        not isinstance(flags, list)
        or flags != sorted(set(flags))
        or not all(isinstance(flag, str) for flag in flags)
    ):
        raise InventoryStageError("Invalid quality flags")
    benchmark_reason = row.get("benchmark_reason")
    if bool(benchmark_reason) != ("benchmark_contamination" in flags):
        raise InventoryStageError("Benchmark reason/flag mismatch")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise InventoryStageError("Missing provenance")

    reasons: list[str] = []
    if report.bucket in CODE_BUCKETS:
        repo_id = provenance.get("repo_id")
        source_identity = str(repo_id).strip() if repo_id is not None else ""
        source_group = stable_digest(
            "code-repository", source_identity or str(row["doc_id"])
        )
        final_cluster = stable_digest("code-normalized", normalized_hash)
        if not source_identity:
            reasons.append("missing_code_repo_id")
    elif report.bucket in ENGLISH_BUCKETS:
        identity = english_source_identity(report.bucket, provenance)
        source_group = stable_digest(
            "english-source", identity or f"missing:{row['doc_id']}"
        )
        final_cluster = stable_digest(
            "english-normalized-provisional", normalized_hash
        )
        if identity is None:
            reasons.append("missing_english_source_identity")
    else:
        raise InventoryStageError(f"Unsupported bucket: {report.bucket}")
    if policy.fast_canonical_profile:
        final_cluster = stable_digest("fast-global-normalized", normalized_hash)

    hard_flags = set(policy.hard_reject_flags)
    hard_flags.update(policy.hard_reject_flags_by_bucket[report.bucket])
    for flag in flags:
        if flag in hard_flags:
            reasons.append(f"quality:{flag}")
    if benchmark_reason:
        reasons.append(f"benchmark:{benchmark_reason}")
    penalty = sum(int(policy.soft_penalty_weights.get(flag, 0)) for flag in flags)
    if penalty >= 1 << 64:
        raise InventoryStageError("Combined quality penalty exceeds unsigned 64-bit")
    canonical_rank = struct.pack(">Q", penalty) + doc_id
    selection_rank = stable_digest(f"selection:{policy.seed}", doc_id)
    return ValidatedDocument(
        doc_id=doc_id,
        bucket=report.bucket,
        archive=report.archive,
        manifest_index=expected_index,
        member_path=member_path,
        tokens=token_count,
        size_bytes=size_bytes,
        content_hash=content_hash,
        normalized_hash=normalized_hash,
        final_cluster=final_cluster,
        source_group=source_group,
        canonical_rank=canonical_rank,
        selection_rank=selection_rank,
        reasons=tuple(sorted(reasons)),
    )


FaultHook = Callable[[str, Mapping[str, Any]], None]


class InventoryStage:
    """One-writer append-only inventory journal with deterministic promotion."""

    def __init__(
        self,
        path: Path,
        *,
        reports: Sequence[ReportSpec],
        policy: StagePolicy,
        batch_size: int = DEFAULT_BATCH_SIZE,
        fault_hook: FaultHook | None = None,
    ) -> None:
        if not 1 <= batch_size <= MAXIMUM_BATCH_SIZE:
            raise InventoryStageError(
                f"batch_size must be in [1, {MAXIMUM_BATCH_SIZE:,}]"
            )
        self.path = path
        self.reports = tuple(reports)
        self.policy = policy
        self.batch_size = batch_size
        self.fault_hook = fault_hook
        self.connection: sqlite3.Connection | None = None
        if [report.ordinal for report in self.reports] != list(range(len(self.reports))):
            raise InventoryStageError("Reports must have contiguous deterministic ordinals")
        archives = [report.archive for report in self.reports]
        if len(set(archives)) != len(archives):
            raise InventoryStageError("Report archive identities must be unique")
        self.identity = {
            "format_version": STAGE_FORMAT_VERSION,
            "batch_size": self.batch_size,
            "sqlite_version": sqlite3.sqlite_version,
            "policy": self.policy.identity(),
            "reports": [report.identity() for report in self.reports],
        }
        self.identity_sha256 = hashlib.sha256(
            canonical_json_bytes(self.identity)
        ).hexdigest()

    def __enter__(self) -> "InventoryStage":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise InventoryStageError("Inventory stage database cannot be a symlink")
        self.connection = sqlite3.connect(self.path, timeout=120)
        self.connection.row_factory = sqlite3.Row
        try:
            mode = str(
                self.connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            ).casefold()
            if mode != "delete":
                raise InventoryStageError("SQLite refused DELETE journaling")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("PRAGMA cache_size=-524288")
            self.connection.execute("PRAGMA mmap_size=8589934592")
            self._create_or_validate_schema()
            self._reconcile_state()
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
            raise RuntimeError("InventoryStage must be used as a context manager")
        return self.connection

    def _fault(self, event: str, **payload: Any) -> None:
        if self.fault_hook is not None:
            self.fault_hook(event, payload)

    def _create_or_validate_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS inventory_stage_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS inventory_stage_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                phase TEXT NOT NULL CHECK(
                    phase IN ('ingesting','ingest_complete','promoting','promoted')
                ),
                next_archive_ordinal INTEGER NOT NULL,
                documents INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                clean_bytes INTEGER NOT NULL,
                reasons INTEGER NOT NULL,
                committed_batches INTEGER NOT NULL,
                promotion_step INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inventory_stage_archives (
                archive_ordinal INTEGER PRIMARY KEY,
                report_sha256 TEXT NOT NULL,
                fingerprint_sha256 TEXT NOT NULL,
                archive TEXT NOT NULL,
                bucket TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('running','complete')),
                processed_rows INTEGER NOT NULL,
                processed_tokens INTEGER NOT NULL,
                processed_clean_bytes INTEGER NOT NULL,
                committed_batches INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inventory_stage_batches (
                batch_id INTEGER PRIMARY KEY,
                archive_ordinal INTEGER NOT NULL,
                first_stage_id INTEGER NOT NULL,
                last_stage_id INTEGER NOT NULL,
                first_reason_id INTEGER,
                last_reason_id INTEGER,
                documents INTEGER NOT NULL,
                reasons INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                clean_bytes INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL
            );
            """
        )
        existing = dict(self.db.execute("SELECT key, value FROM inventory_stage_metadata"))
        encoded = json.dumps(self.identity, sort_keys=True)
        if not existing:
            with self.db:
                self.db.execute(
                    "INSERT INTO inventory_stage_metadata(key, value) VALUES ('identity', ?)",
                    (encoded,),
                )
                self.db.execute(
                    "INSERT INTO inventory_stage_metadata(key, value) VALUES ('identity_sha256', ?)",
                    (json.dumps(self.identity_sha256),),
                )
                self.db.execute(
                    """
                    INSERT INTO inventory_stage_state(
                        singleton, phase, next_archive_ordinal, documents, tokens,
                        clean_bytes, reasons, committed_batches, promotion_step
                    ) VALUES (1, 'ingesting', 0, 0, 0, 0, 0, 0, 0)
                    """
                )
        elif (
            existing.get("identity") != encoded
            or json.loads(existing.get("identity_sha256", "null"))
            != self.identity_sha256
        ):
            raise InventoryStageError("Inventory stage resume identity mismatch")
        state_row = self.db.execute(
            "SELECT phase FROM inventory_stage_state WHERE singleton=1"
        ).fetchone()
        if state_row is None:
            raise InventoryStageError("Missing inventory stage state")
        phase = str(state_row[0])
        tables = {
            str(row[0])
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        staged_pair = {"stage_documents", "stage_reasons"}
        promoted_pair = {"documents", "reasons"}
        if phase == "promoted":
            if not promoted_pair.issubset(tables) or staged_pair & tables:
                raise InventoryStageError("Promoted inventory table authority is invalid")
        else:
            if promoted_pair & tables:
                raise InventoryStageError("Unpromoted inventory has promoted tables")
            if not staged_pair.issubset(tables):
                if staged_pair & tables:
                    raise InventoryStageError("Partial inventory staging schema")
                self.db.executescript(
                    """
                    CREATE TABLE stage_documents (
                        stage_id INTEGER PRIMARY KEY,
                        archive_ordinal INTEGER NOT NULL,
                        doc_id BLOB NOT NULL,
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
                        selection_rank BLOB NOT NULL
                    );
                    CREATE TABLE stage_reasons (
                        reason_id INTEGER PRIMARY KEY,
                        doc_id BLOB NOT NULL,
                        reason TEXT NOT NULL
                    );
                    """
                )
        self.db.commit()

    def _state(self) -> dict[str, Any]:
        row = self.db.execute(
            """
            SELECT phase, next_archive_ordinal, documents, tokens, clean_bytes,
                   reasons, committed_batches, promotion_step
            FROM inventory_stage_state WHERE singleton=1
            """
        ).fetchone()
        if row is None:
            raise InventoryStageError("Missing inventory stage state")
        return {
            "phase": str(row[0]),
            "next_archive_ordinal": int(row[1]),
            "documents": int(row[2]),
            "tokens": int(row[3]),
            "clean_bytes": int(row[4]),
            "reasons": int(row[5]),
            "committed_batches": int(row[6]),
            "promotion_step": int(row[7]),
        }

    def status(self) -> dict[str, Any]:
        return {**self._state(), "identity_sha256": self.identity_sha256}

    def _table_names(self) -> tuple[str, str]:
        return (
            ("documents", "reasons")
            if self._state()["phase"] == "promoted"
            else ("stage_documents", "stage_reasons")
        )

    def _reconcile_state(self) -> None:
        state = self._state()
        documents_table, reasons_table = self._table_names()
        row = self.db.execute(
            f"SELECT COUNT(*), COALESCE(MAX(stage_id), 0), "
            f"COALESCE(SUM(tokens), 0) FROM {documents_table}"
        ).fetchone()
        reasons = int(
            self.db.execute(f"SELECT COUNT(*) FROM {reasons_table}").fetchone()[0]
        )
        batches = self.db.execute(
            """
            SELECT COUNT(*), COALESCE(MAX(batch_id), 0),
                   COALESCE(SUM(documents), 0), COALESCE(SUM(reasons), 0),
                   COALESCE(SUM(tokens), 0), COALESCE(SUM(clean_bytes), 0)
            FROM inventory_stage_batches
            """
        ).fetchone()
        if (
            int(row[0]) != state["documents"]
            or int(row[1]) != state["documents"]
            or int(row[2]) != state["tokens"]
            or reasons != state["reasons"]
            or int(batches[0]) != state["committed_batches"]
            or int(batches[1]) != state["committed_batches"]
            or int(batches[2]) != state["documents"]
            or int(batches[3]) != state["reasons"]
            or int(batches[4]) != state["tokens"]
            or int(batches[5]) != state["clean_bytes"]
        ):
            raise InventoryStageError("Inventory stage durable counters disagree with rows")

    def _validate_report_authority(self, report: ReportSpec) -> None:
        if not report.report_path.is_file() or report.report_path.is_symlink():
            raise InventoryStageError(f"Missing or unsafe report: {report.report_path}")
        if file_sha256(report.report_path) != report.report_sha256:
            raise InventoryStageError(f"Report checksum mismatch: {report.report_path}")
        if not report.fingerprint_path.is_file() or report.fingerprint_path.is_symlink():
            raise InventoryStageError(
                f"Missing or unsafe fingerprint: {report.fingerprint_path}"
            )
        if file_sha256(report.fingerprint_path) != report.fingerprint_sha256:
            raise InventoryStageError(
                f"Fingerprint checksum mismatch: {report.fingerprint_path}"
            )

    def ingest_report(self, report: ReportSpec) -> dict[str, Any]:
        state = self._state()
        if report.ordinal >= len(self.reports) or self.reports[report.ordinal] != report:
            raise InventoryStageError("Report differs from frozen deterministic inventory")
        if state["phase"] != "ingesting":
            if state["phase"] in ("ingest_complete", "promoting", "promoted"):
                return state
            raise InventoryStageError("Inventory stage is not ingestible")
        if report.ordinal < state["next_archive_ordinal"]:
            row = self.db.execute(
                "SELECT status FROM inventory_stage_archives WHERE archive_ordinal=?",
                (report.ordinal,),
            ).fetchone()
            if row is None or str(row[0]) != "complete":
                raise InventoryStageError("Skipped report is not durably complete")
            return state
        if report.ordinal != state["next_archive_ordinal"]:
            raise InventoryStageError(
                f"Out-of-order report {report.ordinal}; expected "
                f"{state['next_archive_ordinal']}"
            )
        self._validate_report_authority(report)
        progress = self.db.execute(
            """
            SELECT status, processed_rows, processed_tokens,
                   processed_clean_bytes, committed_batches
            FROM inventory_stage_archives WHERE archive_ordinal=?
            """,
            (report.ordinal,),
        ).fetchone()
        if progress is None:
            with self.db:
                self.db.execute(
                    """
                    INSERT INTO inventory_stage_archives(
                        archive_ordinal, report_sha256, fingerprint_sha256,
                        archive, bucket, status, processed_rows, processed_tokens,
                        processed_clean_bytes, committed_batches
                    ) VALUES (?, ?, ?, ?, ?, 'running', 0, 0, 0, 0)
                    """,
                    (
                        report.ordinal,
                        report.report_sha256,
                        report.fingerprint_sha256,
                        report.archive,
                        report.bucket,
                    ),
                )
            committed_rows = committed_tokens = committed_bytes = 0
        else:
            if str(progress[0]) != "running":
                raise InventoryStageError("Current archive progress is not running")
            committed_rows = int(progress[1])
            committed_tokens = int(progress[2])
            committed_bytes = int(progress[3])

        batch: list[ValidatedDocument] = []
        input_rows = 0
        total_tokens = committed_tokens
        total_bytes = committed_bytes

        def flush() -> None:
            nonlocal state
            if not batch:
                return
            first_stage_id = state["documents"] + 1
            document_values = [
                document.document_values(
                    stage_id=first_stage_id + offset,
                    archive_ordinal=report.ordinal,
                )
                for offset, document in enumerate(batch)
            ]
            first_reason_id = state["reasons"] + 1
            reason_values = [
                (first_reason_id + offset, doc_id, reason)
                for offset, (doc_id, reason) in enumerate(
                    (document.doc_id, reason)
                    for document in batch
                    for reason in document.reasons
                )
            ]
            batch_tokens = sum(document.tokens for document in batch)
            batch_bytes = sum(document.size_bytes for document in batch)
            batch_id = state["committed_batches"] + 1
            payload_sha256 = _digest_batch_payload(document_values, reason_values)
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(
                    """
                    INSERT INTO stage_documents(
                        stage_id, archive_ordinal, doc_id, bucket, archive,
                        manifest_index, member_path, tokens, content_hash,
                        normalized_hash, final_cluster, source_group,
                        canonical_rank, selection_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    document_values,
                )
                self.db.executemany(
                    """
                    INSERT INTO stage_reasons(reason_id, doc_id, reason)
                    VALUES (?, ?, ?)
                    """,
                    reason_values,
                )
                self.db.execute(
                    """
                    INSERT INTO inventory_stage_batches(
                        batch_id, archive_ordinal, first_stage_id, last_stage_id,
                        first_reason_id, last_reason_id, documents, reasons,
                        tokens, clean_bytes, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        report.ordinal,
                        first_stage_id,
                        first_stage_id + len(document_values) - 1,
                        first_reason_id if reason_values else None,
                        first_reason_id + len(reason_values) - 1
                        if reason_values
                        else None,
                        len(document_values),
                        len(reason_values),
                        batch_tokens,
                        batch_bytes,
                        payload_sha256,
                    ),
                )
                self.db.execute(
                    """
                    UPDATE inventory_stage_archives SET
                        processed_rows=processed_rows + ?,
                        processed_tokens=processed_tokens + ?,
                        processed_clean_bytes=processed_clean_bytes + ?,
                        committed_batches=committed_batches + 1
                    WHERE archive_ordinal=? AND status='running'
                    """,
                    (len(batch), batch_tokens, batch_bytes, report.ordinal),
                )
                if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                    raise InventoryStageError("Lost archive progress during batch commit")
                self.db.execute(
                    """
                    UPDATE inventory_stage_state SET
                        documents=documents + ?, tokens=tokens + ?,
                        clean_bytes=clean_bytes + ?, reasons=reasons + ?,
                        committed_batches=committed_batches + 1
                    WHERE singleton=1 AND phase='ingesting'
                    """,
                    (len(batch), batch_tokens, batch_bytes, len(reason_values)),
                )
                if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                    raise InventoryStageError("Lost global progress during batch commit")
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            state = self._state()
            self._fault(
                "batch_committed",
                archive_ordinal=report.ordinal,
                processed_rows=input_rows,
                batch_rows=len(batch),
                documents=state["documents"],
            )
            batch.clear()

        for expected_index, row in enumerate(iter_jsonl_zst(report.fingerprint_path)):
            input_rows += 1
            if expected_index < committed_rows:
                continue
            document = validate_fingerprint_record(
                row,
                expected_index=expected_index,
                report=report,
                policy=self.policy,
            )
            batch.append(document)
            total_tokens += document.tokens
            total_bytes += document.size_bytes
            if len(batch) >= self.batch_size:
                flush()
        flush()
        archive_row = self.db.execute(
            """
            SELECT processed_rows, processed_tokens, processed_clean_bytes
            FROM inventory_stage_archives WHERE archive_ordinal=?
            """,
            (report.ordinal,),
        ).fetchone()
        if archive_row is None:
            raise InventoryStageError("Missing archive progress after ingest")
        processed_rows, processed_tokens, processed_bytes = map(int, archive_row)
        if (
            input_rows != report.documents
            or processed_rows != report.documents
            or processed_tokens != report.tokens
            or processed_bytes != report.clean_bytes
            or total_tokens != report.tokens
            or total_bytes != report.clean_bytes
        ):
            raise InventoryStageError("Fingerprint totals do not match frozen report")
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                """
                UPDATE inventory_stage_archives SET status='complete'
                WHERE archive_ordinal=? AND status='running'
                """,
                (report.ordinal,),
            )
            if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                raise InventoryStageError("Cannot finalize staged archive")
            self.db.execute(
                """
                UPDATE inventory_stage_state
                SET next_archive_ordinal=next_archive_ordinal + 1
                WHERE singleton=1 AND phase='ingesting' AND next_archive_ordinal=?
                """,
                (report.ordinal,),
            )
            if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                raise InventoryStageError("Cannot advance deterministic archive cursor")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._fault("archive_committed", archive_ordinal=report.ordinal)
        return self._state()

    def ingest_all(self) -> dict[str, Any]:
        for report in self.reports:
            self.ingest_report(report)
        state = self._state()
        if state["phase"] == "ingesting":
            if state["next_archive_ordinal"] != len(self.reports):
                raise InventoryStageError("Not every report was staged")
            with self.db:
                self.db.execute(
                    """
                    UPDATE inventory_stage_state SET phase='ingest_complete'
                    WHERE singleton=1 AND phase='ingesting'
                    """
                )
        self.validate_complete()
        return self._state()

    def validate_complete(
        self,
        *,
        verify_payload_digests: bool = False,
        include_logical_sha256: bool = False,
    ) -> dict[str, Any]:
        state = self._state()
        if state["phase"] not in ("ingest_complete", "promoting", "promoted"):
            raise InventoryStageError("Inventory staging is not complete")
        documents_table, reasons_table = self._table_names()
        expected_documents = sum(report.documents for report in self.reports)
        expected_tokens = sum(report.tokens for report in self.reports)
        expected_bytes = sum(report.clean_bytes for report in self.reports)
        if (
            state["next_archive_ordinal"] != len(self.reports)
            or state["documents"] != expected_documents
            or state["tokens"] != expected_tokens
            or state["clean_bytes"] != expected_bytes
        ):
            raise InventoryStageError("Global staged totals disagree with reports")
        invalid = int(
            self.db.execute(
                f"""
                SELECT COUNT(*) FROM {documents_table}
                WHERE stage_id < 1 OR length(doc_id) != 32 OR tokens < 1
                   OR length(content_hash) != 32 OR length(normalized_hash) != 32
                   OR length(final_cluster) != 32 OR length(source_group) != 32
                   OR length(canonical_rank) != 40 OR length(selection_rank) != 32
                   OR member_path = ''
                """
            ).fetchone()[0]
        )
        if invalid:
            raise InventoryStageError("Staged document field validation failed")
        archive_rows = {
            int(row[0]): tuple(int(value) for value in row[1:])
            for row in self.db.execute(
                f"""
                SELECT archive_ordinal, COUNT(*), MIN(manifest_index),
                       MAX(manifest_index), COALESCE(SUM(tokens), 0)
                FROM {documents_table} GROUP BY archive_ordinal
                """
            )
        }
        progress_rows = {
            int(row[0]): (str(row[1]), int(row[2]), int(row[3]), int(row[4]))
            for row in self.db.execute(
                """
                SELECT archive_ordinal, status, processed_rows,
                       processed_tokens, processed_clean_bytes
                FROM inventory_stage_archives
                """
            )
        }
        for report in self.reports:
            if archive_rows.get(report.ordinal) != (
                report.documents,
                0,
                report.documents - 1,
                report.tokens,
            ) or progress_rows.get(report.ordinal) != (
                "complete",
                report.documents,
                report.tokens,
                report.clean_bytes,
            ):
                raise InventoryStageError(
                    f"Archive staged coverage mismatch: {report.archive}"
                )
        batch_archives = {
            int(row[0]): tuple(int(value) for value in row[1:])
            for row in self.db.execute(
                """
                SELECT archive_ordinal, SUM(documents), SUM(tokens),
                       SUM(clean_bytes), SUM(reasons)
                FROM inventory_stage_batches GROUP BY archive_ordinal
                """
            )
        }
        for report in self.reports:
            values = batch_archives.get(report.ordinal)
            if values is None or values[:3] != (
                report.documents,
                report.tokens,
                report.clean_bytes,
            ):
                raise InventoryStageError(
                    f"Archive batch ledger mismatch: {report.archive}"
                )
        # Before the doc-id index exists, this join would turn an otherwise
        # linear append-only validation into quadratic work.  Reasons are
        # written from the same validated objects and transaction as documents;
        # validate their per-batch ranges/digests below.  Once promotion has
        # installed the doc-id index, additionally prove referential coverage.
        if state["phase"] == "promoted" or state["promotion_step"] >= 1:
            unknown_reasons = int(
                self.db.execute(
                    f"""
                    SELECT COUNT(*) FROM {reasons_table} AS r
                    LEFT JOIN {documents_table} AS d
                      INDEXED BY documents_doc_id ON d.doc_id=r.doc_id
                    WHERE d.stage_id IS NULL
                    """
                ).fetchone()[0]
            )
            if unknown_reasons:
                raise InventoryStageError("Staged reason refers to an unknown document")
        if verify_payload_digests:
            self.validate_batch_payloads()
        result = {
            "identity_sha256": self.identity_sha256,
            "documents": state["documents"],
            "tokens": state["tokens"],
            "clean_bytes": state["clean_bytes"],
            "reasons": state["reasons"],
            "archives": len(self.reports),
        }
        if include_logical_sha256:
            result["logical_sha256"] = self.logical_sha256()
        return result

    def validate_batch_payloads(self) -> None:
        """Recompute every durable batch digest and all sequential ranges."""

        documents_table, reasons_table = self._table_names()
        next_stage_id = 1
        next_reason_id = 1
        next_batch_id = 1
        for row in self.db.execute(
            """
            SELECT batch_id, archive_ordinal, first_stage_id, last_stage_id,
                   first_reason_id, last_reason_id, documents, reasons,
                   tokens, clean_bytes, payload_sha256
            FROM inventory_stage_batches ORDER BY batch_id
            """
        ).fetchall():
            batch_id = int(row[0])
            first_stage_id = int(row[2])
            last_stage_id = int(row[3])
            documents = int(row[6])
            reasons = int(row[7])
            if (
                batch_id != next_batch_id
                or first_stage_id != next_stage_id
                or last_stage_id != first_stage_id + documents - 1
                or documents <= 0
            ):
                raise InventoryStageError("Non-contiguous durable document batch")
            if reasons:
                if row[4] is None or row[5] is None:
                    raise InventoryStageError("Missing durable reason range")
                first_reason_id = int(row[4])
                last_reason_id = int(row[5])
                if (
                    first_reason_id != next_reason_id
                    or last_reason_id != first_reason_id + reasons - 1
                ):
                    raise InventoryStageError("Non-contiguous durable reason batch")
            elif row[4] is not None or row[5] is not None:
                raise InventoryStageError("Unexpected durable reason range")
            document_values = self.db.execute(
                f"""
                SELECT stage_id, archive_ordinal, doc_id, bucket, archive,
                       manifest_index, member_path, tokens, content_hash,
                       normalized_hash, final_cluster, source_group,
                       canonical_rank, selection_rank
                FROM {documents_table}
                WHERE stage_id BETWEEN ? AND ? ORDER BY stage_id
                """,
                (first_stage_id, last_stage_id),
            ).fetchall()
            if reasons:
                reason_values = self.db.execute(
                    f"""
                    SELECT reason_id, doc_id, reason FROM {reasons_table}
                    WHERE reason_id BETWEEN ? AND ? ORDER BY reason_id
                    """,
                    (first_reason_id, last_reason_id),
                ).fetchall()
            else:
                reason_values = []
            if (
                len(document_values) != documents
                or len(reason_values) != reasons
                or sum(int(value[7]) for value in document_values) != int(row[8])
                or _digest_batch_payload(document_values, reason_values) != str(row[10])
            ):
                raise InventoryStageError(
                    f"Durable batch payload mismatch at batch {batch_id}"
                )
            next_batch_id += 1
            next_stage_id = last_stage_id + 1
            if reasons:
                next_reason_id = last_reason_id + 1
        state = self._state()
        if (
            next_batch_id != state["committed_batches"] + 1
            or next_stage_id != state["documents"] + 1
            or next_reason_id != state["reasons"] + 1
        ):
            raise InventoryStageError("Durable batch coverage is incomplete")

    def logical_sha256(self) -> str:
        documents_table, reasons_table = self._table_names()
        digest = hashlib.sha256()
        for row in self.db.execute(
            f"""
            SELECT stage_id, archive_ordinal, doc_id, bucket, archive,
                   manifest_index, member_path, tokens, content_hash,
                   normalized_hash, final_cluster, source_group,
                   canonical_rank, selection_rank
            FROM {documents_table} ORDER BY stage_id
            """
        ):
            payload = list(row)
            for index in (2, 8, 9, 10, 11, 12, 13):
                payload[index] = bytes(payload[index]).hex()
            digest.update(canonical_json_bytes(payload) + b"\n")
        for row in self.db.execute(
            f"SELECT reason_id, doc_id, reason FROM {reasons_table} ORDER BY reason_id"
        ):
            digest.update(
                canonical_json_bytes([int(row[0]), bytes(row[1]).hex(), str(row[2])])
                + b"\n"
            )
        return digest.hexdigest()

    def promote(self) -> dict[str, Any]:
        evidence = self.validate_complete(
            verify_payload_digests=True, include_logical_sha256=True
        )
        state = self._state()
        if state["phase"] == "promoted":
            return evidence
        if state["phase"] == "ingest_complete":
            with self.db:
                self.db.execute(
                    """
                    UPDATE inventory_stage_state SET phase='promoting'
                    WHERE singleton=1 AND phase='ingest_complete'
                    """
                )
            state = self._state()
        for ordinal in range(state["promotion_step"], len(PROMOTION_INDEXES)):
            name, statement = PROMOTION_INDEXES[ordinal]
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute(statement)
                self.db.execute(
                    """
                    UPDATE inventory_stage_state SET promotion_step=?
                    WHERE singleton=1 AND phase='promoting' AND promotion_step=?
                    """,
                    (ordinal + 1, ordinal),
                )
                if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                    raise InventoryStageError("Lost deterministic promotion cursor")
                self.db.commit()
            except sqlite3.IntegrityError as error:
                self.db.rollback()
                raise InventoryStageError(
                    f"Staged uniqueness validation failed while building {name}"
                ) from error
            except BaseException:
                self.db.rollback()
                raise
            self._fault("promotion_index_committed", index=name, ordinal=ordinal)
        duplicates = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM stage_reasons AS r
                LEFT JOIN stage_documents AS d INDEXED BY documents_doc_id
                  ON d.doc_id=r.doc_id
                WHERE d.stage_id IS NULL
                """
            ).fetchone()[0]
        )
        if duplicates:
            raise InventoryStageError("Promoted reasons have missing documents")
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE stage_documents RENAME TO documents")
            self.db.execute("ALTER TABLE stage_reasons RENAME TO reasons")
            self.db.execute(
                """
                UPDATE inventory_stage_state SET phase='promoted'
                WHERE singleton=1 AND phase='promoting' AND promotion_step=?
                """,
                (len(PROMOTION_INDEXES),),
            )
            if self.db.execute("SELECT changes()").fetchone()[0] != 1:
                raise InventoryStageError("Cannot finalize inventory promotion")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._fault("promotion_committed")
        promoted = self.validate_complete()
        promoted["logical_sha256"] = self.logical_sha256()
        if promoted["logical_sha256"] != evidence["logical_sha256"]:
            raise InventoryStageError("Promotion changed logical inventory bytes")
        integrity = str(self.db.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise InventoryStageError(f"Promoted SQLite integrity failed: {integrity}")
        return {**promoted, "sqlite_integrity_check": integrity}

    def logical_rows(self) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        """Return target-compatible rows for tests, audits, and integration probes."""

        documents_table, reasons_table = self._table_names()
        documents = [
            tuple(row)
            for row in self.db.execute(
                f"""
                SELECT doc_id, bucket, archive, manifest_index, member_path, tokens,
                       content_hash, normalized_hash, final_cluster, source_group,
                       canonical_rank, selection_rank
                FROM {documents_table} ORDER BY doc_id
                """
            )
        ]
        reasons = [
            tuple(row)
            for row in self.db.execute(
                f"SELECT doc_id, reason FROM {reasons_table} ORDER BY doc_id, reason"
            )
        ]
        return documents, reasons


def assert_scalar_equivalent(
    stage: InventoryStage,
    *,
    expected_documents: Iterable[Sequence[Any]],
    expected_reasons: Iterable[Sequence[Any]],
) -> None:
    actual_documents, actual_reasons = stage.logical_rows()
    normalized_documents = sorted(
        (tuple(row) for row in expected_documents), key=lambda row: bytes(row[0])
    )
    normalized_reasons = sorted(
        (tuple(row) for row in expected_reasons),
        key=lambda row: (bytes(row[0]), str(row[1])),
    )
    if actual_documents != normalized_documents or actual_reasons != normalized_reasons:
        raise InventoryStageError("Promoted inventory differs from scalar reference")
