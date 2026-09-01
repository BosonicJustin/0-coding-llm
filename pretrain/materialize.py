"""Deterministically materialize curated raw text into packed training data.

This module is the cold-path bridge between the immutable raw/curation
artifacts and :mod:`pretrain.data`.  It deliberately does not implement any
quality filtering, deduplication, or benchmark filtering: completed curation
artifacts are authoritative for those choices.  Identity-v7 inputs carry a
keep bitmap, so the bridge independently recomputes each kept document's
frozen leakage-safe source group and split from its signed fingerprint.

Packing checkpoints are committed at raw-archive boundaries.  Every
split/domain ``PackedShardWriter`` stores the same next-archive cursor.  A
process may die between two writer checkpoints; on resume, writers that are
one archive ahead are left untouched while the lagging writers replay that
archive.  This makes the published payload byte-identical to an uninterrupted
run without requiring a multi-file transaction.
"""

from __future__ import annotations

import contextlib
import array
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import zstandard

from pretrain.data import (
    DOMAIN_ORDER,
    ORDER_FORMAT_VERSION,
    PackedShardWriter,
    build_training_order,
    validate_packed_manifest,
    validate_training_order,
)
from pretrain.selection_contract import (
    ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS,
    ALL_ELIGIBLE_BITMAP_BIT_ORDER,
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
    validate_all_eligible_selection_profile,
)
from pretrain.raw_token_cache_inventory import (
    LoadedRawTokenCacheInventory,
    RawTokenCacheInventoryError,
    load_raw_token_cache_inventory,
)
from pretrain.raw_token_cache import (
    MANIFEST_FILE as RAW_TOKEN_CACHE_MANIFEST_FILE,
    OFFSET_FILE as RAW_TOKEN_CACHE_OFFSET_FILE,
    SIDECAR_FILE as RAW_TOKEN_CACHE_SIDECAR_FILE,
    TOKEN_FILE as RAW_TOKEN_CACHE_TOKEN_FILE,
)
from pretrain.raw_token_cache_reader import (
    ArchiveAuthority as RawTokenArchiveAuthority,
    FileAuthority as RawTokenFileAuthority,
    RawTokenCacheAuthority,
    RawTokenCacheReadError,
    RawTokenCacheReader,
    TokenizerAuthority as RawTokenTokenizerAuthority,
)
from pretrain.tokenizer_identity import vocabulary_sha256


FORMAT = "curated-packed-corpus"
FORMAT_VERSION = 1
CURSOR_FORMAT = "curated-packed-writer-cursor"
CURSOR_VERSION = 1
JOURNAL_NAME = ".materialization-journal.json"
DOCUMENT_INDEX_FORMAT = "packed-document-position-index"
DOCUMENT_INDEX_VERSION = 1
FINGERPRINT_VERSION = 1
SPLITS = ("train", "validation", "test")
BUCKET_DOMAIN = {
    "python": "python",
    "other_code": "other_code",
    "fineweb_edu": "english",
    "wikipedia": "english",
}
PROVENANCE_KEYS = (
    "repo_path",
    "repo_id",
    "commit_id",
    "file_path",
    "content_id",
    "language",
    "license_type",
    "detected_licenses",
    "id",
    "title",
    "url",
    "dump",
    "language_score",
    "fineweb_token_count",
    "fineweb_edu_score",
    "fineweb_edu_int_score",
)
HEX = frozenset("0123456789abcdef")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGLISH_NEAR_BUILDER = PROJECT_ROOT / "scripts" / "build_english_near_clusters.py"
ENGLISH_NEAR_CONFIG = PROJECT_ROOT / "configs" / "english_near_dedup.json"
ENGLISH_NEAR_CALIBRATION = (
    PROJECT_ROOT / "scripts" / "calibrate_english_near_dedup.py"
)
ENGLISH_NEAR_CALIBRATION_CONFIG = (
    PROJECT_ROOT / "configs" / "english_near_dedup_calibration.json"
)
CURATION_STORAGE_CONTRACT_VERSION = 2
CURATION_PROGRESS_VERSION = 1
CURATION_MAXIMUM_TRANSACTION_ROWS = 100_000
CURATION_MINIMUM_SIDECAR_LIMIT_BYTES = 256 * 1024 * 1024
CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW = 64 * 1024
CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT = 3_072
CURATION_DISK_SAFETY_NUMERATOR = 2
CURATION_DISK_SAFETY_DENOMINATOR = 1
CURATION_MINIMUM_FREE_BYTES_AFTER_PROJECTION = 2 * 1_000_000_000
CURATION_SQLITE_TEMP_RELATIVE_PATH = ".work/sqlite-tmp"
ALL_ELIGIBLE_CURATION_STORAGE_CONTRACT_VERSION = 3
ALL_ELIGIBLE_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT = 1_322
ALL_ELIGIBLE_STORAGE_PROJECTION_METHOD = (
    "ceil(observed_v1_database_bytes/observed_v1_documents)"
    "*expected_documents*safety"
)
ALL_ELIGIBLE_STORAGE_PROJECTION_BASIS = {
    "format": "curation-observed-production-storage-v1",
    "source_generation": "selection-fast-local-v2",
    "measurement_scope": (
        "post-global-canonicalization-and-leakage-safe-group-assignment"
    ),
    "observed_documents": 51_328_930,
    "observed_database_bytes": 67_824_914_432,
    "observed_bytes_per_document_numerator": 67_824_914_432,
    "observed_bytes_per_document_denominator": 51_328_930,
    "projected_bytes_per_document_ceiling": 1_322,
    "observed_maximum_wal_bytes": 4_132_940_952,
    "observed_maximum_journal_bytes": 0,
    "observed_maximum_transaction_rows": 100_000,
    "observed_committed_transactions": 10_415,
    "observed_minimum_free_bytes": 275_562_160_128,
}
FAST_ALL_ELIGIBLE_HANDOFF_PROFILE = {
    "contract_version": 1,
    "name": "fast-all-eligible-publisher-handoff-v1",
    "exact_quota_selection": False,
    "decision_emission": False,
    "periodic_full_snapshots": False,
    "final_snapshot_required": True,
    "publisher": "all-eligible-identity-v7",
}
FAST_CURATION_IDENTITY_FORMAT_VERSION = 6
FAST_CURATION_PROFILE = {
    "contract_version": 1,
    "name": "fast-exact-normalized-canonical-v1",
    "production_tier": "baseline",
    "fuzzy_near_dedup": False,
    "canonicalization": "global_exact_then_global_normalized_hash",
    "benchmark_propagation": "global_exact_and_global_normalized_hash",
    "split_grouping": "stable_repository_or_english_source",
    "known_limitations": [
        "Semantic near-duplicate documents may remain and may cross source groups or data splits."
    ],
}
FAST_ENGLISH_NEAR_STATUS = "disabled_by_fast_profile"
ALL_ELIGIBLE_PUBLICATION_SCOPE = "production-durable-snapshot"
ALL_ELIGIBLE_TRAINING_INPUT_BUDGET_AUTHORITY = (
    "the final packed order v4 manifest; this all-eligible publication "
    "does not enforce a training mixture or input-token cap"
)
RAW_ARCHIVE_INTEGRITY_POLICIES = frozenset(
    (
        "deferred-full-sha256-mandatory-before-publication",
        "eager-full-sha256-at-open-and-before-publication",
    )
)
LOCAL_SQLITE_EXECUTION = {
    "mode": "local-wal-with-durable-snapshots",
    "protocol_version": 1,
    "active_journal_mode": "wal",
    "wal_autocheckpoint_pages": 32_768,
    "locking_mode": "exclusive",
}


class MaterializationError(RuntimeError):
    """Raised when an input identity or materialization invariant is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MaterializationError("Artifact contains a non-canonical JSON value") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        or any(character not in HEX for character in value)
    ):
        raise MaterializationError(f"{field} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MaterializationError(f"{field} must be a non-negative integer")
    return value


def _require_exact_object_keys(
    value: Any, expected: set[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise MaterializationError(
            f"{field} schema mismatch: expected {sorted(expected)}, found {found}"
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(
                    f"Duplicate JSON key {key!r} in artifact {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"Expected a JSON object in {path}")
    return value


def _load_json_bytes(raw: bytes, field: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(f"Duplicate {field} JSON key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"Invalid {field} JSON") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"{field} must be a JSON object")
    return value


def _safe_file(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MaterializationError(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise MaterializationError(f"Unsafe {field}: {relative!r}")
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise MaterializationError(f"Missing {field}: {candidate}") from error
    if not resolved.is_relative_to(root_resolved):
        raise MaterializationError(f"{field} escapes {root}: {relative!r}")
    if candidate.is_symlink() or not stat.S_ISREG(resolved.stat().st_mode):
        raise MaterializationError(f"{field} must be a regular non-symlink file: {candidate}")
    return resolved


def _regular_file_state(path: Path, field: str) -> tuple[int, int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise MaterializationError(f"Cannot inspect {field}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MaterializationError(f"{field} is not a regular non-symlink file: {path}")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _provenance_from_raw_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in PROVENANCE_KEYS if row.get(key) is not None}


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(
            raw, read_across_frames=True, closefd=False
        )
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="strict")
        try:
            for line_number, line in enumerate(text, 1):
                if not line.strip():
                    raise MaterializationError(f"Blank JSONL row in {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MaterializationError(
                        f"Invalid JSONL row in {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise MaterializationError(
                        f"Non-object JSONL row in {path}:{line_number}"
                    )
                yield row
        finally:
            text.close()


class _HashingReader(io.RawIOBase):
    def __init__(self, raw: BinaryIO, digest: Any) -> None:
        self.raw = raw
        self.digest = digest

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        if data:
            self.digest.update(data)
        return data

    def readinto(self, buffer: bytearray) -> int:
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


class _DocumentIndexShardWriter:
    """Deterministic, atomic provenance shard for one source archive."""

    def __init__(
        self,
        output_root: Path,
        *,
        archive_ordinal: int,
        split: str,
        domain: str,
    ) -> None:
        self.output_root = output_root
        relative = (
            Path("provenance")
            / "documents"
            / split
            / domain
            / f"archive-{archive_ordinal:06d}.jsonl.zst"
        )
        self.relative = relative.as_posix()
        self.final_path = output_root / relative
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.final_path.with_name(f".{self.final_path.name}.part")
        if self.temporary.exists():
            self.temporary.unlink()
        self.raw = self.temporary.open("xb")
        self.compressed = zstandard.ZstdCompressor(
            level=6,
            threads=0,
            write_checksum=True,
            write_content_size=False,
        ).stream_writer(self.raw, closefd=False)
        self.records = 0
        self._closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise MaterializationError("Document-index shard is already closed")
        self.compressed.write(canonical_json_bytes(dict(row)) + b"\n")
        self.records += 1

    def finish(self, *, archive_ordinal: int) -> dict[str, Any] | None:
        if self._closed:
            raise MaterializationError("Document-index shard was finalized twice")
        self.compressed.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        self._closed = True
        if self.records == 0:
            self.temporary.unlink()
            if self.final_path.exists():
                raise MaterializationError(
                    f"Unexpected non-empty document-index replay: {self.final_path}"
                )
            return None
        size = self.temporary.stat().st_size
        checksum = file_sha256(self.temporary)
        if self.final_path.exists():
            if (
                self.final_path.stat().st_size != size
                or file_sha256(self.final_path) != checksum
            ):
                raise MaterializationError(
                    f"Document-index replay differs: {self.final_path}"
                )
            self.temporary.unlink()
        else:
            os.replace(self.temporary, self.final_path)
            _fsync_directory(self.final_path.parent)
        return {
            "archive_ordinal": archive_ordinal,
            "path": self.relative,
            "bytes": size,
            "sha256": checksum,
            "records": self.records,
        }

    def abort(self) -> None:
        if not self._closed:
            with contextlib.suppress(Exception):
                self.compressed.close()
            with contextlib.suppress(Exception):
                self.raw.close()
            self._closed = True
        with contextlib.suppress(FileNotFoundError):
            self.temporary.unlink()


@dataclass(frozen=True)
class ArchiveSpec:
    ordinal: int
    archive: str
    archive_index: int
    bucket: str
    domain: str
    raw_path: Path
    raw_sha256: str
    report_path: Path
    report_relative: str
    report_sha256: str
    fingerprint_path: Path
    fingerprint_relative: str
    fingerprint_sha256: str
    decision_path: Path
    decision_relative: str
    decision_sha256: str
    documents: int
    clean_bytes: int
    content_tokens: int
    decision_format: str | None = None
    decision_format_version: int | None = None
    decision_bytes: int | None = None
    decision_kept_documents: int | None = None

    def identity(self) -> dict[str, Any]:
        result = {
            "ordinal": self.ordinal,
            "archive": self.archive,
            "archive_index": self.archive_index,
            "bucket": self.bucket,
            "domain": self.domain,
            "raw_sha256": self.raw_sha256,
            "report": self.report_relative,
            "report_sha256": self.report_sha256,
            "fingerprint": self.fingerprint_relative,
            "fingerprint_sha256": self.fingerprint_sha256,
            "decision": self.decision_relative,
            "decision_sha256": self.decision_sha256,
            "documents": self.documents,
            "clean_bytes": self.clean_bytes,
            "content_tokens": self.content_tokens,
        }
        if self.decision_format is not None:
            result.update(
                decision_format=self.decision_format,
                decision_format_version=self.decision_format_version,
                decision_bytes=self.decision_bytes,
                decision_kept_documents=self.decision_kept_documents,
            )
        return result


@dataclass(frozen=True)
class MaterializationConfig:
    sequence_length: int = 4_096
    rows_per_shard: int = 131_072
    construction_seed: int = 1_234
    order_seed: int = 1_234
    frozen_global_microbatch_rows: int | None = None
    frozen_gradient_accumulation_steps: int | None = None
    expected_train_input_tokens: int | None = 52_580_000_000
    expected_validation_input_tokens: int | None = 500_000_000
    expected_test_input_tokens: int | None = 500_000_000
    train_input_token_tolerance: int | None = None
    validation_input_token_tolerance: int | None = None
    test_input_token_tolerance: int | None = None
    enforce_input_weights: bool = True
    expected_vocab_size: int = 49_152
    expected_eos_token_id: int = 0

    def validate(self) -> None:
        for field, value in (
            ("construction_seed", self.construction_seed),
            ("order_seed", self.order_seed),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise MaterializationError(f"{field} must be a non-negative integer")
        if not isinstance(self.enforce_input_weights, bool):
            raise MaterializationError("enforce_input_weights must be boolean")
        for field, value, minimum in (
            ("sequence_length", self.sequence_length, 2),
            ("rows_per_shard", self.rows_per_shard, 1),
            ("expected_vocab_size", self.expected_vocab_size, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise MaterializationError(f"{field} must be an integer >= {minimum}")
        if (self.frozen_global_microbatch_rows is None) != (
            self.frozen_gradient_accumulation_steps is None
        ):
            raise MaterializationError(
                "Frozen global microbatch rows and accumulation must be supplied together"
            )
        for field, value in (
            ("frozen_global_microbatch_rows", self.frozen_global_microbatch_rows),
            (
                "frozen_gradient_accumulation_steps",
                self.frozen_gradient_accumulation_steps,
            ),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise MaterializationError(f"{field} must be a positive integer")
        if (
            not isinstance(self.expected_eos_token_id, int)
            or isinstance(self.expected_eos_token_id, bool)
            or not 0 <= self.expected_eos_token_id < self.expected_vocab_size
        ):
            raise MaterializationError("expected_eos_token_id is outside the vocabulary")
        for field, value in (
            ("expected_train_input_tokens", self.expected_train_input_tokens),
            (
                "expected_validation_input_tokens",
                self.expected_validation_input_tokens,
            ),
            ("expected_test_input_tokens", self.expected_test_input_tokens),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise MaterializationError(f"{field} must be positive")
        for field, value in (
            ("train_input_token_tolerance", self.train_input_token_tolerance),
            (
                "validation_input_token_tolerance",
                self.validation_input_token_tolerance,
            ),
            ("test_input_token_tolerance", self.test_input_token_tolerance),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise MaterializationError(f"{field} must be non-negative")
        for target, tolerance, split in (
            (
                self.expected_train_input_tokens,
                self.train_input_token_tolerance,
                "train",
            ),
            (
                self.expected_validation_input_tokens,
                self.validation_input_token_tolerance,
                "validation",
            ),
            (
                self.expected_test_input_tokens,
                self.test_input_token_tolerance,
                "test",
            ),
        ):
            if target is None and tolerance is not None:
                raise MaterializationError(
                    f"{split} input-token tolerance requires a token cap"
                )
        if not self.enforce_input_weights and any(
            value is not None
            for value in (
                self.expected_train_input_tokens,
                self.expected_validation_input_tokens,
                self.expected_test_input_tokens,
            )
        ):
            raise MaterializationError(
                "Diagnostic observed-mixture mode requires all three token caps to be 0/None"
            )

    def packing_identity(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "rows_per_shard": self.rows_per_shard,
            "construction_seed": self.construction_seed,
            "expected_vocab_size": self.expected_vocab_size,
            "expected_eos_token_id": self.expected_eos_token_id,
        }

    def order_identity(self) -> dict[str, Any]:
        return {
            "order_seeds": {
                split: self.order_seed + offset for offset, split in enumerate(SPLITS)
            },
            "frozen_global_microbatch_rows": self.frozen_global_microbatch_rows,
            "frozen_gradient_accumulation_steps": (
                self.frozen_gradient_accumulation_steps
            ),
            "expected_train_input_tokens": self.expected_train_input_tokens,
            "expected_validation_input_tokens": (
                self.expected_validation_input_tokens
            ),
            "expected_test_input_tokens": self.expected_test_input_tokens,
            "train_input_token_tolerance": self.train_input_token_tolerance,
            "validation_input_token_tolerance": (
                self.validation_input_token_tolerance
            ),
            "test_input_token_tolerance": self.test_input_token_tolerance,
            "enforce_input_weights": self.enforce_input_weights,
            "expected_input_weights": {
                "python": 0.4,
                "other_code": 0.4,
                "english": 0.2,
            },
        }


@dataclass
class _DestinationState:
    split: str
    domain: str
    writer: PackedShardWriter
    cursor: dict[str, Any]


FaultInjector = Callable[[str, Mapping[str, Any]], None]


class CorpusMaterializer:
    """Validate curation decisions and build all packed split/domain outputs."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        preprocess_root: str | Path,
        selection_root: str | Path,
        tokenizer_root: str | Path,
        policy_path: str | Path,
        quota_path: str | Path,
        benchmark_denylist_path: str | Path,
        output_root: str | Path,
        config: MaterializationConfig | None = None,
        tokenizer_batch_documents: int = 256,
        tokenizer_batch_bytes: int = 64 * 1024 * 1024,
        tokenizer_max_document_bytes: int | None = None,
        raw_token_cache_root: str | Path | None = None,
        raw_token_cache_inventory_root: str | Path | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        if ORDER_FORMAT_VERSION != 4:
            raise MaterializationError(
                f"This bridge requires training-order v4, found v{ORDER_FORMAT_VERSION}"
            )
        self.raw_root = Path(raw_root)
        self.preprocess_root = Path(preprocess_root)
        self.selection_root = Path(selection_root)
        self.tokenizer_root = Path(tokenizer_root)
        self.policy_path = Path(policy_path)
        self.quota_path = Path(quota_path)
        self.benchmark_denylist_path = Path(benchmark_denylist_path)
        self.output_root = Path(output_root)
        if (raw_token_cache_root is None) != (
            raw_token_cache_inventory_root is None
        ):
            raise MaterializationError(
                "raw-token-cache root and inventory root must be supplied together"
            )
        self.raw_token_cache_root = (
            Path(raw_token_cache_root) if raw_token_cache_root is not None else None
        )
        self.raw_token_cache_inventory_root = (
            Path(raw_token_cache_inventory_root)
            if raw_token_cache_inventory_root is not None
            else None
        )
        self.config = config or MaterializationConfig()
        self.config.validate()
        if tokenizer_max_document_bytes is None:
            tokenizer_max_document_bytes = tokenizer_batch_bytes
        if (
            not isinstance(tokenizer_batch_documents, int)
            or isinstance(tokenizer_batch_documents, bool)
            or tokenizer_batch_documents < 1
        ):
            raise MaterializationError("tokenizer_batch_documents must be positive")
        if (
            not isinstance(tokenizer_batch_bytes, int)
            or isinstance(tokenizer_batch_bytes, bool)
            or tokenizer_batch_bytes < 1
        ):
            raise MaterializationError("tokenizer_batch_bytes must be positive")
        if (
            not isinstance(tokenizer_max_document_bytes, int)
            or isinstance(tokenizer_max_document_bytes, bool)
            or tokenizer_max_document_bytes < 1
            or tokenizer_max_document_bytes > tokenizer_batch_bytes
        ):
            raise MaterializationError(
                "tokenizer_max_document_bytes must be positive and no larger "
                "than tokenizer_batch_bytes"
            )
        # Runtime-only throughput knobs: encode_batch is required to produce
        # exactly the same IDs as encode, so changing these cannot change or
        # invalidate the durable materialization identity.
        self.tokenizer_batch_documents = tokenizer_batch_documents
        self.tokenizer_batch_bytes = tokenizer_batch_bytes
        self.tokenizer_max_document_bytes = tokenizer_max_document_bytes
        self.fault_injector = fault_injector
        self.all_eligible_split_seed: str | None = None
        self.all_eligible_split_thresholds: tuple[tuple[str, int], ...] = ()
        self.selection_manifest_path = self.selection_root / "manifest.json"
        self.selection_manifest = self._validate_selection_manifest()
        self.selection_manifest_sha256 = file_sha256(self.selection_manifest_path)
        self.policy = self._validate_policy_and_supporting_artifacts()
        self.tokenizer, self.tokenizer_manifest = self._load_tokenizer()
        self.archives = self._build_archive_inventory()
        self._validate_archive_inventory_against_completeness()
        self.archive_inventory_sha256 = canonical_sha256(
            [archive.identity() for archive in self.archives]
        )
        self.raw_token_cache_inventory: LoadedRawTokenCacheInventory | None = None
        self.raw_token_cache_authorities: dict[int, RawTokenCacheAuthority] = {}
        if self.raw_token_cache_root is not None:
            self._load_raw_token_cache_inventory()
        self.identity = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "collection_completeness_sha256": canonical_sha256(
                self.selection_manifest["collection_completeness"]
            ),
            "decision_inventory_sha256": self.selection_manifest[
                "decision_inventory_sha256"
            ],
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "tokenizer_manifest_sha256": file_sha256(
                self.tokenizer_root / "TOKENIZER_MANIFEST.json"
            ),
            "curation_policy_sha256": canonical_sha256(self.policy),
            "quota_config_sha256": file_sha256(self.quota_path),
            "benchmark_guard_sha256": file_sha256(self.benchmark_denylist_path),
            "preprocess_manifest_sha256": file_sha256(
                self.preprocess_root / "PREPROCESS_MANIFEST.json"
            ),
            # Only construction-affecting fields belong to the durable
            # packing identity. Training geometry is deliberately chosen
            # later by a GPU memory/throughput smoke and is pinned by order v4.
            "packing_configuration": self.config.packing_identity(),
        }
        if self.raw_token_cache_inventory is not None:
            self.identity["raw_token_cache"] = (
                self.raw_token_cache_inventory.provenance_descriptor()
            )
        self.journal_path = self.output_root / JOURNAL_NAME
        self.destinations: dict[tuple[str, str], _DestinationState] = {}

    def _fault(self, event: str, **payload: Any) -> None:
        if self.fault_injector is not None:
            self.fault_injector(event, payload)

    def _validate_collection_completeness(
        self, manifest: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        completeness = manifest.get("collection_completeness")
        if not isinstance(completeness, dict):
            raise MaterializationError(
                "Curation manifest has no collection-completeness authority"
            )
        if identity.get("collection_completeness") != completeness:
            raise MaterializationError(
                "Curation identity does not pin collection completeness"
            )
        expected = {
            "format_version": 1,
            "complete": True,
            "legacy_dedup_index_required": False,
            "pending_inputs": 0,
            "preprocess_error_records": 0,
        }
        for key, value in expected.items():
            if completeness.get(key) != value:
                raise MaterializationError(
                    f"Collection completeness {key} mismatch: "
                    f"{completeness.get(key)!r} != {value!r}"
                )
        per_bucket = completeness.get("per_bucket")
        if not isinstance(per_bucket, dict) or set(per_bucket) != set(BUCKET_DOMAIN):
            raise MaterializationError(
                "Collection completeness must cover every raw source bucket"
            )
        targets = completeness.get("collection_targets_exact_tokens")
        if not isinstance(targets, dict) or set(targets) != set(BUCKET_DOMAIN):
            raise MaterializationError("Collection completeness targets are invalid")
        archive_count = 0
        for bucket in sorted(BUCKET_DOMAIN):
            totals = per_bucket[bucket]
            if not isinstance(totals, dict):
                raise MaterializationError(
                    f"Collection completeness totals are invalid for {bucket}"
                )
            for field in (
                "archives",
                "documents",
                "clean_bytes",
                "exact_tokens",
                "target_exact_tokens",
            ):
                value = totals.get(field)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                ):
                    raise MaterializationError(
                        f"Collection completeness {bucket}.{field} is invalid"
                    )
            if totals["exact_tokens"] < totals["target_exact_tokens"]:
                raise MaterializationError(
                    f"Collection target is not complete for {bucket}"
                )
            if targets[bucket] != totals["target_exact_tokens"]:
                raise MaterializationError(
                    f"Collection target identity differs for {bucket}"
                )
            archive_count += totals["archives"]
        inventories = {}
        for field in ("quota_records", "raw_archives", "reports", "fingerprints"):
            value = completeness.get(field)
            if not isinstance(value, dict):
                raise MaterializationError(
                    f"Collection completeness {field} inventory is invalid"
                )
            count = value.get("count")
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise MaterializationError(
                    f"Collection completeness {field} count is invalid"
                )
            _require_sha256(
                value.get("inventory_sha256"),
                f"collection_completeness.{field}.inventory_sha256",
            )
            inventories[field] = count
        if set(inventories.values()) != {archive_count}:
            raise MaterializationError(
                "Collection completeness archive/report inventories disagree"
            )
        if (
            completeness["reports"]["inventory_sha256"]
            != identity.get("report_inventory_sha256")
            or identity.get("report_count") != archive_count
        ):
            raise MaterializationError(
                "Collection completeness report authority differs from curation identity"
            )
        markers = completeness.get("completion_markers")
        if not isinstance(markers, dict):
            raise MaterializationError("Collection completion-marker inventory is invalid")
        marker_files = markers.get("files")
        marker_count = markers.get("count")
        if (
            not isinstance(marker_count, int)
            or isinstance(marker_count, bool)
            or marker_count < 1
            or not isinstance(marker_files, list)
            or len(marker_files) != marker_count
        ):
            raise MaterializationError("Collection completion markers are incomplete")
        _require_sha256(
            markers.get("inventory_sha256"),
            "collection_completeness.completion_markers.inventory_sha256",
        )
        marker_buckets: list[str] = []
        for marker in marker_files:
            if (
                not isinstance(marker, dict)
                or not isinstance(marker.get("path"), str)
                or not marker["path"]
                or not isinstance(marker.get("buckets"), list)
                or not marker["buckets"]
                or not set(marker["buckets"]).issubset(BUCKET_DOMAIN)
            ):
                raise MaterializationError("Invalid collection completion-marker identity")
            _require_sha256(
                marker.get("sha256"),
                "collection_completeness.completion_markers.files.sha256",
            )
            marker_path = _safe_file(
                self.raw_root,
                marker["path"],
                "collection completion marker",
            )
            if file_sha256(marker_path) != marker["sha256"]:
                raise MaterializationError(
                    f"Collection completion marker changed: {marker_path}"
                )
            marker_buckets.extend(marker["buckets"])
        if (
            set(marker_buckets) != set(BUCKET_DOMAIN)
            or len(marker_buckets) != len(set(marker_buckets))
        ):
            raise MaterializationError(
                "Collection completion markers must cover each bucket exactly once"
            )
        reports = manifest.get("input_reports")
        if not isinstance(reports, list) or len(reports) != archive_count:
            raise MaterializationError(
                "Curation report inventory does not match collection completeness"
            )
        return dict(completeness)

    @staticmethod
    def _validate_sqlite_runtime(identity: Mapping[str, Any]) -> None:
        runtime = _require_exact_object_keys(
            identity.get("sqlite_runtime"),
            {"sqlite_version", "journal_policy"},
            "selection.identity.sqlite_runtime",
        )
        sqlite_version = runtime["sqlite_version"]
        if not isinstance(sqlite_version, str) or not sqlite_version.strip():
            raise MaterializationError(
                "selection.identity.sqlite_runtime.sqlite_version is invalid"
            )
        policy = _require_exact_object_keys(
            runtime["journal_policy"],
            {"policy_version", "requested_mode", "selected_mode", "mount"},
            "selection.identity.sqlite_runtime.journal_policy",
        )
        requested = policy["requested_mode"]
        selected = policy["selected_mode"]
        if policy["policy_version"] != 1:
            raise MaterializationError("Unsupported curation SQLite journal policy")
        if requested not in ("auto", "delete", "wal"):
            raise MaterializationError("Invalid requested curation SQLite journal mode")
        if selected not in ("delete", "wal"):
            raise MaterializationError("Invalid selected curation SQLite journal mode")
        mount = _require_exact_object_keys(
            policy["mount"],
            {
                "detector",
                "mount_point",
                "filesystem_type",
                "source",
                "device",
                "options",
                "classification",
            },
            "selection.identity.sqlite_runtime.journal_policy.mount",
        )
        for field in ("detector", "filesystem_type"):
            if not isinstance(mount[field], str) or not mount[field]:
                raise MaterializationError(
                    f"Invalid curation SQLite mount field {field}"
                )
        for field in ("mount_point", "source", "device"):
            if mount[field] is not None and not isinstance(mount[field], str):
                raise MaterializationError(
                    f"Invalid curation SQLite mount field {field}"
                )
        if (
            not isinstance(mount["options"], list)
            or any(not isinstance(item, str) for item in mount["options"])
        ):
            raise MaterializationError("Invalid curation SQLite mount options")
        classification = mount["classification"]
        if classification not in ("local", "network", "unknown"):
            raise MaterializationError("Invalid curation SQLite mount classification")
        expected_selected = (
            "wal"
            if requested == "wal"
            or (requested == "auto" and classification == "local")
            else "delete"
        )
        if selected != expected_selected or (
            selected == "wal" and classification != "local"
        ):
            raise MaterializationError(
                "Curation SQLite journal policy is internally inconsistent"
            )

    @staticmethod
    def _validate_curation_storage_contract(
        identity: Mapping[str, Any],
        *,
        all_eligible: bool,
    ) -> dict[str, Any]:
        """Validate the curator's frozen production-scale storage contract."""
        keys = {
            "contract_version",
            "progress_version",
            "maximum_transaction_rows",
            "transaction_sidecar_limit_bytes",
            "projected_additional_bytes_per_document",
            "disk_safety_numerator",
            "disk_safety_denominator",
            "minimum_free_bytes_after_projection",
            "sqlite_temp_store",
            "sqlite_temp_relative_path",
            "sqlite_temp_same_device_as_database",
        }
        if all_eligible:
            keys.update(("projection_basis", "projection_method"))
        contract = _require_exact_object_keys(
            identity.get("curation_storage_contract"),
            keys,
            "selection.identity.curation_storage_contract",
        )
        maximum_rows = _require_nonnegative_int(
            contract["maximum_transaction_rows"],
            "selection.identity.curation_storage_contract.maximum_transaction_rows",
        )
        if not 1 <= maximum_rows <= CURATION_MAXIMUM_TRANSACTION_ROWS:
            raise MaterializationError(
                "Curation storage maximum_transaction_rows is outside the "
                "supported production range"
            )
        expected = {
            "contract_version": (
                ALL_ELIGIBLE_CURATION_STORAGE_CONTRACT_VERSION
                if all_eligible
                else CURATION_STORAGE_CONTRACT_VERSION
            ),
            "progress_version": CURATION_PROGRESS_VERSION,
            "transaction_sidecar_limit_bytes": max(
                CURATION_MINIMUM_SIDECAR_LIMIT_BYTES,
                maximum_rows * CURATION_SIDECAR_BYTES_PER_TRANSACTION_ROW,
            ),
            "projected_additional_bytes_per_document": (
                ALL_ELIGIBLE_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
                if all_eligible
                else CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
            ),
            "disk_safety_numerator": CURATION_DISK_SAFETY_NUMERATOR,
            "disk_safety_denominator": CURATION_DISK_SAFETY_DENOMINATOR,
            "minimum_free_bytes_after_projection": (
                CURATION_MINIMUM_FREE_BYTES_AFTER_PROJECTION
            ),
            "sqlite_temp_store": "FILE",
            "sqlite_temp_relative_path": CURATION_SQLITE_TEMP_RELATIVE_PATH,
            "sqlite_temp_same_device_as_database": True,
        }
        if all_eligible:
            basis = _require_exact_object_keys(
                contract["projection_basis"],
                set(ALL_ELIGIBLE_STORAGE_PROJECTION_BASIS),
                "selection.identity.curation_storage_contract.projection_basis",
            )
            for field in (
                "observed_documents",
                "observed_database_bytes",
                "observed_bytes_per_document_numerator",
                "observed_bytes_per_document_denominator",
                "projected_bytes_per_document_ceiling",
                "observed_maximum_wal_bytes",
                "observed_maximum_journal_bytes",
                "observed_maximum_transaction_rows",
                "observed_committed_transactions",
                "observed_minimum_free_bytes",
            ):
                _require_nonnegative_int(
                    basis[field],
                    "selection.identity.curation_storage_contract."
                    f"projection_basis.{field}",
                )
            if dict(basis) != ALL_ELIGIBLE_STORAGE_PROJECTION_BASIS:
                raise MaterializationError(
                    "All-eligible curation storage projection basis changed"
                )
            expected.update(
                {
                    "projection_basis": ALL_ELIGIBLE_STORAGE_PROJECTION_BASIS,
                    "projection_method": ALL_ELIGIBLE_STORAGE_PROJECTION_METHOD,
                }
            )
        for field, expected_value in expected.items():
            if contract[field] != expected_value:
                raise MaterializationError(
                    "Curation storage contract mismatch for "
                    f"{field}: {contract[field]!r} != {expected_value!r}"
                )
        return dict(contract)

    def _validate_calibration_evidence(
        self,
        evidence_value: Any,
        *,
        near_identity: Mapping[str, Any],
        curation_identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        evidence = _require_exact_object_keys(
            evidence_value,
            {
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
            },
            "selection.identity.english_near_artifact.identity.calibration_evidence",
        )
        required_gate = {
            "contract_version": 1,
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_failures": [],
        }
        for field, expected in required_gate.items():
            if evidence[field] != expected:
                raise MaterializationError(
                    f"Calibration evidence {field} is not production-passing"
                )
        result_sha = _require_sha256(
            evidence["result_sha256"], "calibration_evidence.result_sha256"
        )
        sidecar_sha = _require_sha256(
            evidence["sidecar_sha256"], "calibration_evidence.sidecar_sha256"
        )
        result_bytes = _require_nonnegative_int(
            evidence["result_bytes"], "calibration_evidence.result_bytes"
        )
        if result_bytes < 1:
            raise MaterializationError("Calibration result is empty")
        result_path = _safe_file(
            self.raw_root, evidence["result_path"], "calibration result"
        )
        sidecar_path = _safe_file(
            self.raw_root, evidence["sidecar_path"], "calibration checksum sidecar"
        )
        if sidecar_path != result_path.with_name(result_path.name + ".sha256"):
            raise MaterializationError("Calibration checksum sidecar path is not canonical")
        result_raw = result_path.read_bytes()
        sidecar_raw = sidecar_path.read_bytes()
        if len(result_raw) != result_bytes or hashlib.sha256(result_raw).hexdigest() != result_sha:
            raise MaterializationError("Calibration result identity mismatch")
        expected_sidecar = f"{result_sha}  {result_path.name}\n".encode("utf-8")
        if (
            sidecar_raw != expected_sidecar
            or hashlib.sha256(sidecar_raw).hexdigest() != sidecar_sha
        ):
            raise MaterializationError("Calibration result checksum mismatch")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload:
                    raise MaterializationError(
                        f"Duplicate calibration result key: {key!r}"
                    )
                payload[key] = value
            return payload

        try:
            result = json.loads(result_raw, object_pairs_hook=reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MaterializationError("Invalid calibration result JSON") from error
        if not isinstance(result, dict):
            raise MaterializationError("Calibration result must be a JSON object")
        result_gate = {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_overrides": {},
            "acceptance_failures": [],
        }
        for field, expected in result_gate.items():
            if result.get(field) != expected:
                raise MaterializationError(
                    f"Calibration result {field} is not production-passing"
                )
        for field in (
            "result_version",
            "status",
            "production_gate_eligible",
            "acceptance_profile",
            "sampling_profile",
            "acceptance_failures",
        ):
            if result[field] != evidence[field]:
                raise MaterializationError(
                    f"Calibration evidence differs from result for {field}"
                )

        calibration_identity = _require_exact_object_keys(
            evidence["identity"],
            {
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
            },
            "calibration_evidence.identity",
        )
        if result.get("identity") != calibration_identity:
            raise MaterializationError("Calibration result identity differs from evidence")
        if canonical_sha256(calibration_identity) != _require_sha256(
            evidence["identity_sha256"], "calibration_evidence.identity_sha256"
        ):
            raise MaterializationError("Calibration identity checksum mismatch")
        for field in (
            "harness_sha256",
            "production_builder_sha256",
            "production_config_file_sha256",
            "production_config_canonical_sha256",
            "calibration_config_file_sha256",
            "calibration_config_canonical_sha256",
            "sample_manifest_sha256",
        ):
            _require_sha256(calibration_identity[field], f"calibration.identity.{field}")
        for field in ("calibration_algorithm", "calibration_seed"):
            if not isinstance(calibration_identity[field], str) or not calibration_identity[field]:
                raise MaterializationError(f"Calibration identity {field} is invalid")
        calibration_config_raw = ENGLISH_NEAR_CALIBRATION_CONFIG.read_bytes()
        calibration_config = _load_json(ENGLISH_NEAR_CALIBRATION_CONFIG)
        current_calibration_authority = {
            "harness_sha256": file_sha256(ENGLISH_NEAR_CALIBRATION),
            "calibration_algorithm": calibration_config["calibration_algorithm"],
            "calibration_seed": calibration_config["seed"],
            "calibration_config_file_sha256": hashlib.sha256(
                calibration_config_raw
            ).hexdigest(),
            "calibration_config_canonical_sha256": canonical_sha256(
                calibration_config
            ),
        }
        for field, expected in current_calibration_authority.items():
            if calibration_identity[field] != expected:
                raise MaterializationError(
                    f"Calibration current authority differs for {field}"
                )
        if (
            calibration_identity["production_builder_sha256"]
            != near_identity["builder_sha256"]
            or calibration_identity["production_config_file_sha256"]
            != near_identity["config_file_sha256"]
            or calibration_identity["production_config_canonical_sha256"]
            != near_identity["config_sha256"]
        ):
            raise MaterializationError(
                "Calibration production builder/config differs from English near-dedup identity"
            )
        if (
            not isinstance(calibration_identity["runtime"], dict)
            or not calibration_identity["runtime"]
        ):
            raise MaterializationError("Calibration runtime identity is invalid")

        input_identity = _require_exact_object_keys(
            calibration_identity["input"],
            {
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
            },
            "calibration_evidence.identity.input",
        )
        if input_identity["kind"] != "immutable_real_english_sample":
            raise MaterializationError("Calibration did not use immutable real English data")
        cross_authority = {
            "full_report_inventory_sha256": near_identity[
                "report_inventory_sha256"
            ],
            "preprocess_manifest_sha256": curation_identity[
                "preprocess_manifest_sha256"
            ],
            "curation_policy_sha256": curation_identity["policy_sha256"],
            "benchmark_guard_sha256": curation_identity[
                "benchmark_guard_sha256"
            ],
            "source_manifests": near_identity["source_manifests"],
            "collection_completeness": near_identity[
                "collection_completeness"
            ],
            "collection_completeness_sha256": canonical_sha256(
                near_identity["collection_completeness"]
            ),
        }
        for field, expected in cross_authority.items():
            if input_identity[field] != expected:
                raise MaterializationError(
                    f"Calibration input authority differs for {field}"
                )
        for field in (
            "full_report_inventory_sha256",
            "preprocess_manifest_sha256",
            "curation_policy_sha256",
            "benchmark_guard_sha256",
            "collection_completeness_sha256",
        ):
            _require_sha256(input_identity[field], f"calibration.input.{field}")
        for field in (
            "maximum_archives_per_bucket",
            "maximum_documents_per_bucket",
            "minimum_source_words",
            "documents_selected",
        ):
            if _require_nonnegative_int(input_identity[field], f"calibration.input.{field}") < 1:
                raise MaterializationError(f"Calibration input {field} is empty")
        if (
            not isinstance(input_identity["selection_seed"], str)
            or not input_identity["selection_seed"]
            or not isinstance(input_identity["sampling_counts"], dict)
            or not input_identity["sampling_counts"]
        ):
            raise MaterializationError("Calibration sampling identity is invalid")
        expected_sampling_authority = {
            "selection_seed": calibration_config["seed"],
            "maximum_archives_per_bucket": calibration_config["sampling"][
                "maximum_archives_per_bucket"
            ],
            "maximum_documents_per_bucket": calibration_config["sampling"][
                "maximum_documents_per_bucket"
            ],
            "minimum_source_words": calibration_config["sampling"][
                "minimum_source_words"
            ],
        }
        for field, expected in expected_sampling_authority.items():
            if input_identity[field] != expected:
                raise MaterializationError(
                    f"Calibration sampling authority differs for {field}"
                )
        production_config = _load_json(ENGLISH_NEAR_CONFIG)
        if result.get("acceptance") != calibration_config["acceptance"]:
            raise MaterializationError("Calibration acceptance authority changed")
        if result.get("production_threshold") != {
            "minimum_jaccard_numerator": production_config["refinement"][
                "minimum_jaccard_numerator"
            ],
            "minimum_jaccard_denominator": production_config["refinement"][
                "minimum_jaccard_denominator"
            ],
        }:
            raise MaterializationError("Calibration production threshold changed")

        reports = manifest.get("input_reports")
        if not isinstance(reports, list):
            raise MaterializationError("Curation input reports are unavailable")
        current_reports = {
            row["archive"]: {
                "report_path": row["report"],
                "report_sha256": row["report_sha256"],
                "archive": row["archive"],
                "archive_sha256": row["archive_sha256"],
                "fingerprint_file": row["fingerprint_file"],
                "fingerprint_sha256": row["fingerprint_sha256"],
                "documents": row["documents"],
            }
            for row in reports
            if isinstance(row, dict)
            and all(
                field in row
                for field in (
                    "report",
                    "report_sha256",
                    "archive",
                    "archive_sha256",
                    "fingerprint_file",
                    "fingerprint_sha256",
                    "documents",
                )
            )
        }
        selected_reports = input_identity["selected_reports"]
        if not isinstance(selected_reports, list) or not selected_reports:
            raise MaterializationError("Calibration has no selected-report authority")
        seen: set[str] = set()
        for selected_report in selected_reports:
            if not isinstance(selected_report, dict):
                raise MaterializationError("Calibration selected report is invalid")
            archive = selected_report.get("archive")
            if (
                not isinstance(archive, str)
                or archive in seen
                or not archive.startswith(
                    (
                        "raw/english/fineweb_edu/",
                        "raw/english/wikipedia/",
                    )
                )
                or current_reports.get(archive) != selected_report
            ):
                raise MaterializationError(
                    "Calibration selected report differs from curation authority"
                )
            seen.add(archive)

        sampling = result.get("sampling")
        if not isinstance(sampling, dict) or sampling.get("documents_input") != input_identity[
            "documents_selected"
        ]:
            raise MaterializationError("Calibration selected-document count mismatch")
        sample_manifest = sampling.get("sample_manifest")
        if (
            not isinstance(sample_manifest, list)
            or not sample_manifest
            or canonical_sha256(sample_manifest)
            != calibration_identity["sample_manifest_sha256"]
        ):
            raise MaterializationError("Calibration sample-manifest identity mismatch")

    def _validate_operational_preflight(
        self,
        evidence_value: Any,
        *,
        publication_root: str,
        near_identity: Mapping[str, Any],
        mapping: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = _require_exact_object_keys(
            evidence_value,
            {
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
            },
            "selection.identity.english_near_artifact.refinement_operational_preflight",
        )
        for field, expected in {
            "contract_version": 1,
            "result_path": "operational-preflight-v1/result.json",
            "sidecar_path": "operational-preflight-v1/result.json.sha256",
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
        }.items():
            if evidence[field] != expected:
                raise MaterializationError(
                    f"Operational preflight {field} is not production-passing"
                )
        result_sha = _require_sha256(
            evidence["result_sha256"], "operational preflight result_sha256"
        )
        sidecar_sha = _require_sha256(
            evidence["sidecar_sha256"], "operational preflight sidecar_sha256"
        )
        result_bytes = _require_nonnegative_int(
            evidence["result_bytes"], "operational preflight result_bytes"
        )
        if result_bytes < 1:
            raise MaterializationError("Operational preflight result is empty")
        result_relative = f"{publication_root}/{evidence['result_path']}"
        sidecar_relative = f"{publication_root}/{evidence['sidecar_path']}"
        result_path = _safe_file(
            self.raw_root, result_relative, "operational preflight result"
        )
        sidecar_path = _safe_file(
            self.raw_root, sidecar_relative, "operational preflight sidecar"
        )
        if sidecar_path != result_path.with_name("result.json.sha256"):
            raise MaterializationError(
                "Operational preflight sidecar path is not canonical"
            )
        result_raw = result_path.read_bytes()
        sidecar_raw = sidecar_path.read_bytes()
        if (
            len(result_raw) != result_bytes
            or hashlib.sha256(result_raw).hexdigest() != result_sha
        ):
            raise MaterializationError("Operational preflight result identity mismatch")
        if (
            sidecar_raw != f"{result_sha}  result.json\n".encode("ascii")
            or hashlib.sha256(sidecar_raw).hexdigest() != sidecar_sha
        ):
            raise MaterializationError("Operational preflight sidecar mismatch")
        result = _load_json_bytes(result_raw, "operational preflight result")
        _require_exact_object_keys(
            result,
            {
                "result_version",
                "status",
                "production_gate_eligible",
                "failures",
                "identity",
                "thresholds",
                "sample",
                "measurements",
                "statistical_scope",
            },
            "operational preflight result",
        )
        for field, expected in {
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
        }.items():
            if result[field] != expected:
                raise MaterializationError(
                    f"Operational preflight result {field} is unsafe"
                )
        for field in (
            "status",
            "production_gate_eligible",
            "failures",
            "identity",
            "thresholds",
            "sample",
            "measurements",
        ):
            if result[field] != evidence[field]:
                raise MaterializationError(
                    f"Operational preflight result differs from evidence for {field}"
                )
        if not isinstance(result["statistical_scope"], str) or not result[
            "statistical_scope"
        ]:
            raise MaterializationError("Operational preflight statistical scope is empty")

        preflight_identity = _require_exact_object_keys(
            evidence["identity"],
            {
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
            },
            "operational preflight identity",
        )
        if canonical_sha256(preflight_identity) != _require_sha256(
            evidence["identity_sha256"], "operational preflight identity_sha256"
        ):
            raise MaterializationError("Operational preflight identity checksum mismatch")
        expected_identity = {
            "contract_version": 1,
            "builder_sha256": near_identity["builder_sha256"],
            "builder_identity_sha256": canonical_sha256(near_identity),
            "config_file_sha256": near_identity["config_file_sha256"],
            "config_sha256": near_identity["config_sha256"],
            "calibration_evidence_sha256": canonical_sha256(
                near_identity["calibration_evidence"]
            ),
            "report_inventory_sha256": near_identity["report_inventory_sha256"],
            "preprocess_manifest_sha256": near_identity["preprocess_manifest_sha256"],
            "curation_policy_sha256": near_identity["curation_policy_sha256"],
            "benchmark_guard_sha256": near_identity["benchmark_guard_sha256"],
            "collection_completeness_sha256": canonical_sha256(
                near_identity["collection_completeness"]
            ),
            "documents_total": mapping["records"],
            "cache_archives": near_identity["report_count"],
            "runtime_storage": near_identity["runtime"]["storage"],
        }
        for field, expected in expected_identity.items():
            if preflight_identity[field] != expected:
                raise MaterializationError(
                    f"Operational preflight identity differs for {field}"
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
            _require_sha256(preflight_identity[field], f"operational preflight {field}")
        for field in (
            "candidate_pairs_total",
            "documents_total",
            "candidate_blocks_total",
            "candidate_blocks_committed",
            "cache_archives",
            "cache_bytes",
        ):
            _require_nonnegative_int(
                preflight_identity[field], f"operational preflight identity.{field}"
            )
        if (
            preflight_identity["candidate_blocks_total"]
            != preflight_identity["candidate_blocks_committed"]
            or preflight_identity["phase_at_measurement"] != "refine"
            or preflight_identity["refinement_cursor_at_measurement"] is not None
        ):
            raise MaterializationError(
                "Operational preflight candidate authority is incomplete"
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
            or any(character not in HEX for character in cursor["key"])
        ):
            raise MaterializationError("Operational preflight candidate cursor is invalid")

        config_raw = ENGLISH_NEAR_CONFIG.read_bytes()
        config = _load_json(ENGLISH_NEAR_CONFIG)
        config_semantic_sha = canonical_sha256(
            {key: value for key, value in config.items() if not key.startswith("_")}
        )
        if (
            near_identity["builder_sha256"] != file_sha256(ENGLISH_NEAR_BUILDER)
            or near_identity["config_file_sha256"]
            != hashlib.sha256(config_raw).hexdigest()
            or near_identity["config_sha256"] != config_semantic_sha
        ):
            raise MaterializationError(
                "English near-dedup builder/config differs from current authority"
            )
        thresholds = evidence["thresholds"]
        if thresholds != config.get("operational_preflight"):
            raise MaterializationError("Operational preflight thresholds changed")
        total_candidates = preflight_identity["candidate_pairs_total"]
        if total_candidates > config["candidate_signature"][
            "maximum_unique_candidate_pairs"
        ]:
            raise MaterializationError("Operational preflight candidate total is unsafe")
        sample = _require_exact_object_keys(
            evidence["sample"],
            {
                "algorithm",
                "seed",
                "requested_pairs",
                "expected_pairs",
                "measured_pairs",
                "sample_pairs_sha256",
                "accepted_pairs",
                "sample_limited_by_total_candidates",
            },
            "operational preflight sample",
        )
        expected_pairs = min(int(thresholds["requested_pairs"]), total_candidates)
        expected_sample = {
            "algorithm": thresholds["sampling_algorithm"],
            "seed": thresholds["sampling_seed"],
            "requested_pairs": thresholds["requested_pairs"],
            "expected_pairs": expected_pairs,
            "measured_pairs": expected_pairs,
            "sample_limited_by_total_candidates": total_candidates
            < int(thresholds["requested_pairs"]),
        }
        for field, expected in expected_sample.items():
            if sample[field] != expected:
                raise MaterializationError(
                    f"Operational preflight sample differs for {field}"
                )
        _require_sha256(sample["sample_pairs_sha256"], "preflight sample_pairs_sha256")
        accepted_pairs = sample["accepted_pairs"]
        if (
            not isinstance(accepted_pairs, int)
            or isinstance(accepted_pairs, bool)
            or not 0 <= accepted_pairs <= expected_pairs
        ):
            raise MaterializationError("Operational preflight accepted pairs are invalid")

        measurements = _require_exact_object_keys(
            evidence["measurements"],
            {
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
            },
            "operational preflight measurements",
        )
        batch_size = measurements["measurement_batch_size"]
        elapsed = measurements["sample_elapsed_seconds"]
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or measurements["measurement_batches"]
            != (math.ceil(expected_pairs / batch_size) if expected_pairs else 0)
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise MaterializationError("Operational preflight timing is invalid")
        expected_rate = (
            None if expected_pairs == 0 else round(expected_pairs / float(elapsed), 6)
        )
        if total_candidates > 0 and (expected_rate is None or expected_rate <= 0.0):
            raise MaterializationError("Operational preflight rate is invalid")
        projected_seconds = (
            0.0
            if total_candidates == 0
            else round(total_candidates / float(expected_rate), 3)
        )
        measured_growth = _require_nonnegative_int(
            measurements["measurement_sqlite_growth_bytes"],
            "operational preflight measured SQLite growth",
        )
        bytes_per_pair = measured_growth / expected_pairs if expected_pairs else 0.0
        projected_growth = math.ceil(bytes_per_pair * total_candidates)
        projected_with_safety = math.ceil(
            projected_growth
            * int(thresholds["disk_projection_safety_numerator"])
            / int(thresholds["disk_projection_safety_denominator"])
        )
        required_free = projected_with_safety + int(
            thresholds["minimum_post_refinement_free_bytes"]
        )
        formulas = {
            "candidate_pairs_total": total_candidates,
            "refinement_pairs_per_second": expected_rate,
            "projected_refinement_seconds": projected_seconds,
            "measurement_sqlite_bytes_per_pair": round(bytes_per_pair, 6),
            "projected_additional_refinement_sqlite_bytes": projected_growth,
            "projected_additional_with_safety_bytes": projected_with_safety,
            "required_filesystem_free_bytes": required_free,
        }
        for field, expected in formulas.items():
            if measurements[field] != expected:
                raise MaterializationError(
                    f"Operational preflight formula differs for {field}"
                )
        for resource_name in ("resources_before", "resources_after"):
            resources = _require_exact_object_keys(
                measurements[resource_name],
                {
                    "filesystem_total_bytes",
                    "filesystem_free_bytes",
                    "sqlite_state_bytes",
                    "refinement_cache_bytes",
                    "peak_process_rss_bytes",
                },
                f"operational preflight {resource_name}",
            )
            for field, value in resources.items():
                _require_nonnegative_int(value, f"{resource_name}.{field}")
            if (
                resources["filesystem_free_bytes"] > resources["filesystem_total_bytes"]
                or resources["refinement_cache_bytes"]
                != preflight_identity["cache_bytes"]
            ):
                raise MaterializationError(
                    f"Operational preflight {resource_name} is inconsistent"
                )
        before = measurements["resources_before"]
        after = measurements["resources_after"]
        item_size = array.array("Q").itemsize
        parent_bytes = (mapping["records"] + 1) * item_size
        parent_with_safety = math.ceil(
            parent_bytes
            * int(thresholds["union_parent_memory_safety_numerator"])
            / int(thresholds["union_parent_memory_safety_denominator"])
        )
        union_formulas = {
            "union_parent_item_bytes": item_size,
            "union_parent_array_projected_bytes": parent_bytes,
            "union_parent_array_with_safety_bytes": parent_with_safety,
            "union_projected_peak_process_rss_bytes": before[
                "peak_process_rss_bytes"
            ]
            + parent_with_safety,
        }
        for field, expected in union_formulas.items():
            if measurements[field] != expected:
                raise MaterializationError(
                    f"Operational preflight union formula differs for {field}"
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
            or projected_seconds
            > int(thresholds["maximum_projected_refinement_seconds"])
            or after["peak_process_rss_bytes"]
            > int(thresholds["maximum_peak_process_rss_bytes"])
            or union_formulas["union_projected_peak_process_rss_bytes"]
            > int(thresholds["maximum_peak_process_rss_bytes"])
            or after["filesystem_free_bytes"] < required_free
        ):
            raise MaterializationError(
                "Operational preflight no longer passes its resource gates"
            )
        return dict(evidence)

    def _validate_english_near_artifact(
        self,
        identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        artifact = _require_exact_object_keys(
            identity.get("english_near_artifact"),
            {
                "contract_version",
                "publication_root",
                "manifest",
                "identity",
                "mapping",
                "refinement_operational_preflight",
                "database_integrity_check",
                "completeness_and_leakage_audit",
            },
            "selection.identity.english_near_artifact",
        )
        if artifact["contract_version"] != 2:
            raise MaterializationError("Unsupported English near-dedup artifact contract")
        publication_root = artifact["publication_root"]
        if not isinstance(publication_root, str) or not publication_root:
            raise MaterializationError("English near-dedup publication root is missing")
        publication_parts = PurePosixPath(publication_root)
        if publication_parts.is_absolute() or any(
            part in ("", ".", "..") for part in publication_parts.parts
        ):
            raise MaterializationError("English near-dedup publication root is unsafe")
        publication_root = publication_parts.as_posix()
        publication = _require_exact_object_keys(
            artifact["manifest"],
            {"path", "sha256", "bytes", "sidecar_path", "sidecar_sha256"},
            "selection.identity.english_near_artifact.manifest",
        )
        expected_manifest_paths = {
            "path": f"{publication_root}/manifest.json",
            "sidecar_path": f"{publication_root}/manifest.sha256",
        }
        for field, expected in expected_manifest_paths.items():
            if publication[field] != expected:
                raise MaterializationError(
                    f"English near-dedup manifest {field} is not canonical"
                )
        manifest_sha = _require_sha256(
            publication["sha256"], "English near-dedup manifest sha256"
        )
        manifest_sidecar_sha = _require_sha256(
            publication["sidecar_sha256"],
            "English near-dedup manifest sidecar sha256",
        )
        manifest_bytes = _require_nonnegative_int(
            publication["bytes"], "English near-dedup manifest bytes"
        )
        if manifest_bytes < 1:
            raise MaterializationError("English near-dedup manifest is empty")
        near_manifest_path = _safe_file(
            self.raw_root, publication["path"], "English near-dedup manifest"
        )
        near_sidecar_path = _safe_file(
            self.raw_root,
            publication["sidecar_path"],
            "English near-dedup manifest sidecar",
        )
        near_manifest_raw = near_manifest_path.read_bytes()
        near_sidecar_raw = near_sidecar_path.read_bytes()
        if (
            len(near_manifest_raw) != manifest_bytes
            or hashlib.sha256(near_manifest_raw).hexdigest() != manifest_sha
        ):
            raise MaterializationError("English near-dedup manifest identity mismatch")
        if (
            near_sidecar_raw != f"{manifest_sha}  manifest.json\n".encode("ascii")
            or hashlib.sha256(near_sidecar_raw).hexdigest() != manifest_sidecar_sha
        ):
            raise MaterializationError("English near-dedup manifest sidecar mismatch")
        near_manifest = _load_json_bytes(
            near_manifest_raw, "English near-dedup manifest"
        )
        _require_exact_object_keys(
            near_manifest,
            {
                "manifest_version",
                "mapping_record_version",
                "production_ready",
                "identity",
                "refinement_operational_preflight",
                "algorithm",
                "inputs",
                "candidate_stats",
                "completeness_and_leakage_audit",
                "database_integrity_check",
                "mapping",
            },
            "English near-dedup manifest",
        )
        if (
            near_manifest["manifest_version"] != 1
            or near_manifest["mapping_record_version"] != 1
            or near_manifest["production_ready"] is not True
        ):
            raise MaterializationError(
                "English near-dedup manifest is not production-ready v1"
            )
        if artifact["database_integrity_check"] != "ok":
            raise MaterializationError("English near-dedup database integrity failed")

        mapping = _require_exact_object_keys(
            artifact["mapping"],
            {
                "path",
                "sha256",
                "bytes",
                "records",
                "ordered_by",
                "singleton_clusters_included",
            },
            "selection.identity.english_near_artifact.mapping",
        )
        mapping_sha = _require_sha256(
            mapping["sha256"],
            "selection.identity.english_near_artifact.mapping.sha256",
        )
        if mapping_sha != identity.get("english_near_clusters_sha256"):
            raise MaterializationError(
                "english_near_clusters_sha256 differs from English near artifact mapping"
            )
        if (
            mapping["path"] != "clusters.jsonl.zst"
            or mapping["ordered_by"] != "frozen_inventory_ordinal"
            or mapping["singleton_clusters_included"] is not True
            or _require_nonnegative_int(
                mapping["bytes"],
                "selection.identity.english_near_artifact.mapping.bytes",
            )
            < 1
            or _require_nonnegative_int(
                mapping["records"],
                "selection.identity.english_near_artifact.mapping.records",
            )
            < 1
        ):
            raise MaterializationError("Invalid English near-dedup mapping identity")
        mapping_path = _safe_file(
            self.raw_root,
            f"{publication_root}/{mapping['path']}",
            "English near-dedup mapping",
        )
        if (
            mapping_path.stat().st_size != mapping["bytes"]
            or file_sha256(mapping_path) != mapping_sha
        ):
            raise MaterializationError("English near-dedup mapping identity mismatch")
        mapping_records = 0
        for row in iter_jsonl_zst(mapping_path):
            if set(row) != {"doc_id", "cluster_id"}:
                raise MaterializationError("English near-dedup mapping row is invalid")
            _require_sha256(row.get("doc_id"), "English near-dedup mapping doc_id")
            if not isinstance(row.get("cluster_id"), str) or not row["cluster_id"]:
                raise MaterializationError(
                    "English near-dedup mapping cluster_id is invalid"
                )
            mapping_records += 1
        if mapping_records != mapping["records"]:
            raise MaterializationError("English near-dedup mapping record count mismatch")

        near_identity = _require_exact_object_keys(
            artifact["identity"],
            {
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
            },
            "selection.identity.english_near_artifact.identity",
        )
        if near_identity["format_version"] != 1:
            raise MaterializationError("Unsupported English near-dedup identity")
        for field in (
            "builder_sha256",
            "config_file_sha256",
            "config_sha256",
            "curation_policy_sha256",
            "preprocess_manifest_sha256",
            "benchmark_guard_sha256",
            "report_inventory_sha256",
        ):
            _require_sha256(
                near_identity[field],
                f"selection.identity.english_near_artifact.identity.{field}",
            )
        for near_field, curation_field in (
            ("curation_policy_sha256", "policy_sha256"),
            ("preprocess_manifest_sha256", "preprocess_manifest_sha256"),
            ("benchmark_guard_sha256", "benchmark_guard_sha256"),
        ):
            if near_identity[near_field] != identity.get(curation_field):
                raise MaterializationError(
                    f"English near-dedup {near_field} differs from curation identity"
                )
        report_count = _require_nonnegative_int(
            near_identity["report_count"],
            "selection.identity.english_near_artifact.identity.report_count",
        )
        if report_count < 1:
            raise MaterializationError("English near-dedup report count is empty")
        for field in ("source_manifests", "collection_completeness", "runtime"):
            if not isinstance(near_identity[field], dict) or not near_identity[field]:
                raise MaterializationError(
                    f"English near-dedup embedded {field} identity is invalid"
                )
        runtime = _require_exact_object_keys(
            near_identity["runtime"],
            {"python", "sqlite", "xxhash", "zstandard", "storage"},
            "English near-dedup runtime",
        )
        for field in ("python", "sqlite", "xxhash", "zstandard"):
            if not isinstance(runtime[field], str) or not runtime[field]:
                raise MaterializationError(
                    f"English near-dedup runtime {field} is invalid"
                )
        storage = _require_exact_object_keys(
            runtime["storage"],
            {
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
                "policy",
            },
            "English near-dedup runtime storage",
        )
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
            if not isinstance(storage[field], str) or not storage[field]:
                raise MaterializationError(
                    f"English near-dedup storage {field} is invalid"
                )
        current_config = _load_json(ENGLISH_NEAR_CONFIG)
        expected_storage_policy = {
            "network_or_unknown_action": "delete",
            "wal_on_non_allowlisted_action": "fail_closed",
            "wal_local_filesystem_allowlist": current_config["storage"][
                "wal_local_filesystem_allowlist"
            ],
        }
        if (
            storage["policy"] != expected_storage_policy
            or storage["sqlite_journal_mode_configured"]
            != current_config["storage"]["sqlite_journal_mode"]
            or storage["sqlite_journal_mode_requested"]
            not in ("auto", "delete", "wal")
            or storage["sqlite_journal_mode_selected"] not in ("delete", "wal")
            or storage["sqlite_journal_mode_actual"]
            != storage["sqlite_journal_mode_selected"]
            or storage["sqlite_journal_mode_request_source"] not in ("cli", "config")
            or (
                storage["sqlite_journal_mode_request_source"] == "config"
                and storage["sqlite_journal_mode_requested"]
                != storage["sqlite_journal_mode_configured"]
            )
            or storage["classification"]
            not in ("proven-local", "non-local", "unknown")
            or (
                storage["sqlite_journal_mode_selected"] == "wal"
                and storage["classification"] != "proven-local"
            )
            or (
                storage["classification"] != "proven-local"
                and storage["sqlite_journal_mode_selected"] != "delete"
            )
        ):
            raise MaterializationError(
                "English near-dedup storage authority is inconsistent"
            )
        curation_sources = identity.get("source_manifests")
        if not isinstance(curation_sources, dict):
            raise MaterializationError("Curation source-manifest identity is invalid")
        expected_near_sources = {
            "TOKENIZER_MANIFEST.json": {
                "sha256": identity.get("tokenizer_manifest_sha256"),
                "resolved_revision": identity.get("tokenizer_revision"),
            },
            **{
                filename: curation_sources.get(filename)
                for filename in (
                    "FINEWEB_EDU_SOURCE.json",
                    "WIKIPEDIA_SOURCE.json",
                )
            },
        }
        if near_identity["source_manifests"] != expected_near_sources:
            raise MaterializationError(
                "English near-dedup source manifests differ from curation identity"
            )
        self._validate_calibration_evidence(
            near_identity["calibration_evidence"],
            near_identity=near_identity,
            curation_identity=identity,
            manifest=manifest,
        )
        verified_preflight = self._validate_operational_preflight(
            artifact["refinement_operational_preflight"],
            publication_root=publication_root,
            near_identity=near_identity,
            mapping=mapping,
        )
        if (
            near_manifest["identity"] != near_identity
            or near_manifest["mapping"] != mapping
            or near_manifest["refinement_operational_preflight"]
            != verified_preflight
            or near_manifest["database_integrity_check"]
            != artifact["database_integrity_check"]
        ):
            raise MaterializationError(
                "English near-dedup manifest differs from transitive curation evidence"
            )
        algorithm = _require_exact_object_keys(
            near_manifest["algorithm"],
            {
                "config",
                "config_file_sha256",
                "config_sha256",
                "name",
                "compact_preprocess_sketch_role",
                "raw_text_candidate_pass",
                "full_shingle_refinement",
                "jaccard_threshold",
                "ideal_independent_minhash_candidate_recall_at_threshold",
                "statistical_limitation",
                "hash_limitation",
                "posting_overflow_action",
            },
            "English near-dedup algorithm",
        )
        current_config = _load_json(ENGLISH_NEAR_CONFIG)
        expected_threshold = (
            current_config["refinement"]["minimum_jaccard_numerator"]
            / current_config["refinement"]["minimum_jaccard_denominator"]
        )
        if (
            algorithm["name"] != current_config["algorithm"]
            or algorithm["config_file_sha256"] != near_identity["config_file_sha256"]
            or algorithm["config_sha256"] != near_identity["config_sha256"]
            or algorithm["raw_text_candidate_pass"] is not True
            or algorithm["full_shingle_refinement"] is not True
            or algorithm["posting_overflow_action"]
            != "fail_closed_without_truncation"
            or algorithm["jaccard_threshold"] != expected_threshold
        ):
            raise MaterializationError("English near-dedup algorithm authority changed")
        candidate_stats = _require_exact_object_keys(
            near_manifest["candidate_stats"],
            {
                "blocks",
                "maximum_posting_documents",
                "raw_posting_pairs",
                "length_pruned_pairs",
                "unique_candidate_pairs",
                "accepted_near_pairs",
            },
            "English near-dedup candidate stats",
        )
        for field, value in candidate_stats.items():
            _require_nonnegative_int(value, f"English near candidate_stats.{field}")
        preflight_identity = verified_preflight["identity"]
        if (
            candidate_stats["blocks"]
            != preflight_identity["candidate_blocks_total"]
            or candidate_stats["unique_candidate_pairs"]
            != preflight_identity["candidate_pairs_total"]
            or candidate_stats["accepted_near_pairs"]
            > candidate_stats["unique_candidate_pairs"]
            or candidate_stats["length_pruned_pairs"]
            > candidate_stats["raw_posting_pairs"]
            or candidate_stats["maximum_posting_documents"]
            > current_config["candidate_signature"]["maximum_posting_documents"]
        ):
            raise MaterializationError(
                "English near-dedup candidate stats differ from preflight authority"
            )
        near_inputs = _require_exact_object_keys(
            near_manifest["inputs"],
            {"report_inventory_sha256", "reports"},
            "English near-dedup inputs",
        )
        curation_reports = manifest.get("input_reports")
        if not isinstance(curation_reports, list):
            raise MaterializationError("Curation input report inventory is invalid")
        expected_english_reports = [
            {
                "report_path": row["report"],
                "report_sha256": row["report_sha256"],
                "archive": row["archive"],
                "archive_sha256": row["archive_sha256"],
                "fingerprint_file": row["fingerprint_file"],
                "fingerprint_sha256": row["fingerprint_sha256"],
                "documents": row["documents"],
            }
            for row in curation_reports
            if isinstance(row, dict)
            and str(row.get("archive", "")).startswith(
                ("raw/english/fineweb_edu/", "raw/english/wikipedia/")
            )
        ]
        if (
            near_inputs["report_inventory_sha256"]
            != near_identity["report_inventory_sha256"]
            or near_inputs["reports"] != expected_english_reports
        ):
            raise MaterializationError(
                "English near-dedup report inventory differs from curation authority"
            )

        audit = _require_exact_object_keys(
            artifact["completeness_and_leakage_audit"],
            {
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
            },
            "selection.identity.english_near_artifact.completeness_and_leakage_audit",
        )
        for field, value in audit.items():
            _require_nonnegative_int(
                value,
                "selection.identity.english_near_artifact."
                f"completeness_and_leakage_audit.{field}",
            )
        records = mapping["records"]
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
            raise MaterializationError(
                "English near-dedup completeness/leakage audit is unsafe"
            )
        clusters = audit["clusters"]
        singletons = audit["singleton_clusters"]
        cross_source = audit["cross_source_clusters"]
        if (
            clusters < 1
            or clusters > records
            or singletons > clusters
            or 2 * clusters - singletons > records
            or cross_source > clusters
        ):
            raise MaterializationError(
                "English near-dedup cluster audit is internally impossible"
            )
        if near_manifest["completeness_and_leakage_audit"] != audit:
            raise MaterializationError(
                "English near-dedup manifest audit differs from curation evidence"
            )

    @staticmethod
    def _require_nonempty_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MaterializationError(f"{field} must be a non-empty string")
        return value

    @staticmethod
    def _require_positive_int(value: Any, field: str) -> int:
        result = _require_nonnegative_int(value, field)
        if result < 1:
            raise MaterializationError(f"{field} must be positive")
        return result

    @staticmethod
    def _stable_digest(namespace: str, value: str | bytes) -> bytes:
        raw = value if isinstance(value, bytes) else value.encode("utf-8")
        return hashlib.sha256(namespace.encode("utf-8") + b"\0" + raw).digest()

    @staticmethod
    def _english_source_identity(
        bucket: str, provenance: Mapping[str, Any]
    ) -> str | None:
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

    def _configure_all_eligible_split_authority(
        self, policy: Mapping[str, Any]
    ) -> None:
        if (
            self.selection_manifest["identity"]["format_version"]
            != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        ):
            return
        selection = policy.get("selection")
        seed = selection.get("seed") if isinstance(selection, dict) else None
        if not isinstance(seed, str) or not seed:
            raise MaterializationError(
                "All-eligible policy has no frozen split seed"
            )
        quota_config = _load_json(self.quota_path)
        rows = quota_config.get("quotas")
        if quota_config.get("version") != 1 or not isinstance(rows, list):
            raise MaterializationError("Unsupported all-eligible quota configuration")
        quotas: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise MaterializationError(
                    f"Invalid all-eligible quota row {index}"
                )
            if row.get("phase") != "final":
                continue
            split = row.get("split")
            category = row.get("category")
            target = row.get("target")
            key = (str(split), str(category))
            if (
                split not in SPLITS
                or category not in DOMAIN_ORDER
                or row.get("token_field") != "exact_tokens"
                or not isinstance(target, int)
                or isinstance(target, bool)
                or target < 1
                or key in quotas
            ):
                raise MaterializationError(
                    f"Invalid all-eligible final quota row {index}"
                )
            quotas[key] = target
        expected = {(split, domain) for split in SPLITS for domain in DOMAIN_ORDER}
        if set(quotas) != expected:
            raise MaterializationError(
                "All-eligible final quotas do not cover every split/domain"
            )
        references = {
            (row["split"], row["category"]): row["reference_target_tokens"]
            for row in self.selection_manifest["reference_quotas"]
        }
        if references != quotas:
            raise MaterializationError(
                "All-eligible reference quotas differ from the frozen quota file"
            )
        totals = {
            split: sum(quotas[(split, domain)] for domain in DOMAIN_ORDER)
            for split in SPLITS
        }
        grand = sum(totals.values())
        maximum = 1 << 64
        test_end = maximum * totals["test"] // grand
        validation_end = (
            test_end + maximum * totals["validation"] // grand
        )
        self.all_eligible_split_seed = seed
        self.all_eligible_split_thresholds = (
            ("test", test_end),
            ("validation", validation_end),
            ("train", maximum),
        )

    def _derive_all_eligible_group_and_split(
        self,
        *,
        bucket: str,
        provenance: Mapping[str, Any],
        doc_id: str,
    ) -> tuple[str, str]:
        if bucket in ("python", "other_code"):
            repo_id = provenance.get("repo_id")
            source_identity = str(repo_id).strip() if repo_id is not None else ""
            if not source_identity:
                raise MaterializationError(
                    "All-eligible kept code document has no repository identity"
                )
            group = self._stable_digest("code-repository", source_identity)
        else:
            source_identity = self._english_source_identity(bucket, provenance)
            if source_identity is None:
                raise MaterializationError(
                    "All-eligible kept English document has no source identity"
                )
            group = self._stable_digest("english-source", source_identity)
        if self.all_eligible_split_seed is None or not self.all_eligible_split_thresholds:
            raise MaterializationError("All-eligible split authority is not configured")
        value = int.from_bytes(
            self._stable_digest(
                f"split:{self.all_eligible_split_seed}", group
            )[:8],
            "big",
        )
        for split, end in self.all_eligible_split_thresholds:
            if value < end:
                return group.hex(), split
        raise MaterializationError("All-eligible split thresholds are incomplete")

    @classmethod
    def _validate_all_eligible_totals(
        cls, manifest: Mapping[str, Any]
    ) -> dict[tuple[str, str], dict[str, int]]:
        """Validate the non-quota v7 supply summary and reference-only quotas."""

        if "quotas" in manifest:
            raise MaterializationError(
                "All-eligible selection must not carry an authoritative quota summary"
            )
        rows = manifest.get("selected_totals")
        if not isinstance(rows, list):
            raise MaterializationError("All-eligible selection has no selected totals")
        expected_order = [
            (split, domain) for split in SPLITS for domain in DOMAIN_ORDER
        ]
        observed_order: list[tuple[str, str]] = []
        totals: dict[tuple[str, str], dict[str, int]] = {}
        for index, raw_row in enumerate(rows):
            row = _require_exact_object_keys(
                raw_row,
                {
                    "split",
                    "category",
                    "unit",
                    "documents",
                    "selected_tokens",
                    "terminal_prefix_documents",
                },
                f"selection.selected_totals[{index}]",
            )
            split = row["split"]
            category = row["category"]
            if split not in SPLITS or category not in DOMAIN_ORDER:
                raise MaterializationError(
                    f"Invalid all-eligible selected total key: {(split, category)!r}"
                )
            key = (str(split), str(category))
            if key in totals:
                raise MaterializationError(
                    f"Duplicate all-eligible selected total: {key}"
                )
            observed_order.append(key)
            if row["unit"] != "pre_packing_starcoder2_content_tokens":
                raise MaterializationError(
                    "All-eligible selected total uses an unsafe token unit"
                )
            terminal = _require_nonnegative_int(
                row["terminal_prefix_documents"],
                f"selection.selected_totals[{index}].terminal_prefix_documents",
            )
            if terminal != 0:
                raise MaterializationError(
                    "All-eligible selected totals must contain complete documents"
                )
            totals[key] = {
                "selected_documents": cls._require_positive_int(
                    row["documents"],
                    f"selection.selected_totals[{index}].documents",
                ),
                "selected_content_tokens": cls._require_positive_int(
                    row["selected_tokens"],
                    f"selection.selected_totals[{index}].selected_tokens",
                ),
                "terminal_prefix_documents": terminal,
            }
        if observed_order != expected_order:
            raise MaterializationError(
                "All-eligible selected totals do not cover every split/domain "
                "in canonical order"
            )

        reference_rows = manifest.get("reference_quotas")
        if not isinstance(reference_rows, list):
            raise MaterializationError("All-eligible selection has no reference quotas")
        reference_order: list[tuple[str, str]] = []
        for index, raw_row in enumerate(reference_rows):
            row = _require_exact_object_keys(
                raw_row,
                {
                    "split",
                    "category",
                    "unit",
                    "reference_target_tokens",
                    "observed_tokens",
                    "shortfall_tokens",
                    "surplus_tokens",
                    "selection_authority",
                },
                f"selection.reference_quotas[{index}]",
            )
            split = row["split"]
            category = row["category"]
            if split not in SPLITS or category not in DOMAIN_ORDER:
                raise MaterializationError(
                    f"Invalid all-eligible reference quota key: "
                    f"{(split, category)!r}"
                )
            key = (str(split), str(category))
            reference_order.append(key)
            if row["unit"] != "pre_packing_starcoder2_content_tokens":
                raise MaterializationError(
                    "All-eligible reference quota uses an unsafe token unit"
                )
            target = cls._require_positive_int(
                row["reference_target_tokens"],
                f"selection.reference_quotas[{index}].reference_target_tokens",
            )
            observed = cls._require_positive_int(
                row["observed_tokens"],
                f"selection.reference_quotas[{index}].observed_tokens",
            )
            shortfall = _require_nonnegative_int(
                row["shortfall_tokens"],
                f"selection.reference_quotas[{index}].shortfall_tokens",
            )
            surplus = _require_nonnegative_int(
                row["surplus_tokens"],
                f"selection.reference_quotas[{index}].surplus_tokens",
            )
            if (
                key not in totals
                or observed != totals[key]["selected_content_tokens"]
                or shortfall != max(0, target - observed)
                or surplus != max(0, observed - target)
                or row["selection_authority"] is not False
            ):
                raise MaterializationError(
                    f"All-eligible reference quota is inconsistent for {key}"
                )
        if reference_order != expected_order:
            raise MaterializationError(
                "All-eligible reference quotas do not cover every split/domain "
                "in canonical order"
            )

        documents = _require_exact_object_keys(
            manifest.get("documents"),
            {
                "input",
                "accepted_canonical_before_selection",
                "selected",
                "quota_overflow",
            },
            "selection.documents",
        )
        input_documents = cls._require_positive_int(
            documents["input"], "selection.documents.input"
        )
        accepted_documents = cls._require_positive_int(
            documents["accepted_canonical_before_selection"],
            "selection.documents.accepted_canonical_before_selection",
        )
        manifest_selected_documents = cls._require_positive_int(
            documents["selected"], "selection.documents.selected"
        )
        quota_overflow = _require_nonnegative_int(
            documents["quota_overflow"], "selection.documents.quota_overflow"
        )
        selected_documents = sum(
            row["selected_documents"] for row in totals.values()
        )
        if (
            accepted_documents != selected_documents
            or manifest_selected_documents != selected_documents
            or quota_overflow != 0
            or input_documents < selected_documents
        ):
            raise MaterializationError(
                "All-eligible document totals are internally inconsistent"
            )
        return totals

    @classmethod
    def _validate_all_eligible_source_curation(
        cls,
        identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        source = _require_exact_object_keys(
            identity.get("source_curation"),
            {
                "contract_version",
                "database_version",
                "identity_format_version",
                "identity_sha256",
                "database",
                "checkpoint",
                "snapshot",
                "phase",
                "read_performance",
                "eligible_authority_query_plan",
                "archive_bitmap_query_plan",
                "rejection_inventory_query_plan",
                "rejection_reason_associations",
                "groups_subphase",
                "ignored_partial_exact_selection",
            },
            "selection.identity.source_curation",
        )
        if (
            source["contract_version"] != 1
            or source["database_version"] != 4
            or source["identity_format_version"]
            != FAST_CURATION_IDENTITY_FORMAT_VERSION
            or source["phase"]
            not in ("canonicalized", "selected", "emitting", "emitted", "complete")
        ):
            raise MaterializationError(
                "All-eligible source curation contract is unsupported"
            )
        source_identity_sha = _require_sha256(
            source["identity_sha256"],
            "selection.identity.source_curation.identity_sha256",
        )
        source_identity = dict(identity)
        source_identity.pop("selection_profile", None)
        source_identity.pop("source_curation", None)
        source_identity["format_version"] = FAST_CURATION_IDENTITY_FORMAT_VERSION
        if canonical_sha256(source_identity) != source_identity_sha:
            raise MaterializationError(
                "All-eligible publication does not bind its source v6 identity"
            )

        for name in ("database", "checkpoint"):
            descriptor = _require_exact_object_keys(
                source[name],
                {"path", "bytes", "sha256"},
                f"selection.identity.source_curation.{name}",
            )
            cls._require_nonempty_string(
                descriptor["path"],
                f"selection.identity.source_curation.{name}.path",
            )
            cls._require_positive_int(
                descriptor["bytes"],
                f"selection.identity.source_curation.{name}.bytes",
            )
            _require_sha256(
                descriptor["sha256"],
                f"selection.identity.source_curation.{name}.sha256",
            )

        snapshot = _require_exact_object_keys(
            source["snapshot"],
            {
                "manifest_path",
                "manifest_sha256",
                "generation",
                "identity_sha256",
                "previous_manifest_sha256",
                "validated_chain",
                "validation",
            },
            "selection.identity.source_curation.snapshot",
        )
        cls._require_nonempty_string(
            snapshot["manifest_path"],
            "selection.identity.source_curation.snapshot.manifest_path",
        )
        _require_sha256(
            snapshot["manifest_sha256"],
            "selection.identity.source_curation.snapshot.manifest_sha256",
        )
        cls._require_positive_int(
            snapshot["generation"],
            "selection.identity.source_curation.snapshot.generation",
        )
        if (
            _require_sha256(
                snapshot["identity_sha256"],
                "selection.identity.source_curation.snapshot.identity_sha256",
            )
            != source_identity_sha
        ):
            raise MaterializationError(
                "All-eligible durable snapshot does not bind the source identity"
            )
        previous = snapshot["previous_manifest_sha256"]
        if previous is not None:
            _require_sha256(
                previous,
                "selection.identity.source_curation.snapshot.previous_manifest_sha256",
            )
        if snapshot["validation"] != (
            "LocalSQLiteStore-v1-compatible-target-and-chain"
        ) or not isinstance(snapshot["validated_chain"], list):
            raise MaterializationError(
                "All-eligible durable snapshot validation evidence is unsupported"
            )
        chain: list[tuple[int, str]] = []
        for index, raw_row in enumerate(snapshot["validated_chain"]):
            row = _require_exact_object_keys(
                raw_row,
                {"generation", "manifest_sha256"},
                "selection.identity.source_curation.snapshot."
                f"validated_chain[{index}]",
            )
            chain.append(
                (
                    cls._require_positive_int(
                        row["generation"],
                        f"source snapshot chain generation {index}",
                    ),
                    _require_sha256(
                        row["manifest_sha256"],
                        f"source snapshot chain manifest {index}",
                    ),
                )
            )
        if (
            not chain
            or any(
                left[0] >= right[0]
                for left, right in zip(chain, chain[1:])
            )
            or chain[-1]
            != (snapshot["generation"], snapshot["manifest_sha256"])
            or previous != (chain[-2][1] if len(chain) > 1 else None)
        ):
            raise MaterializationError(
                "All-eligible durable snapshot chain evidence is inconsistent"
            )

        read_performance = _require_exact_object_keys(
            source["read_performance"],
            {
                "temp_store",
                "cache_size_kib",
                "mmap_size_bytes",
                "durability_pragmas_modified",
            },
            "selection.identity.source_curation.read_performance",
        )
        if (
            read_performance["temp_store"] != 2
            or read_performance["cache_size_kib"] != 4_194_304
            or read_performance["durability_pragmas_modified"] is not False
        ):
            raise MaterializationError(
                "All-eligible source read-performance contract is unsupported"
            )
        _require_nonnegative_int(
            read_performance["mmap_size_bytes"],
            "selection.identity.source_curation.read_performance.mmap_size_bytes",
        )

        for name in (
            "eligible_authority_query_plan",
            "archive_bitmap_query_plan",
            "rejection_inventory_query_plan",
        ):
            query_plan = source[name]
            if (
                not isinstance(query_plan, list)
                or not query_plan
                or any(
                    not isinstance(row, str) or not row.strip()
                    for row in query_plan
                )
            ):
                raise MaterializationError(
                    f"All-eligible source {name} evidence is malformed"
                )

        reason_counts = manifest.get("reason_document_counts")
        if not isinstance(reason_counts, dict) or any(
            not isinstance(reason, str)
            or not reason
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for reason, count in reason_counts.items()
        ):
            raise MaterializationError(
                "All-eligible rejection reason accounting is malformed"
            )
        rejection_associations = _require_nonnegative_int(
            source["rejection_reason_associations"],
            "selection.identity.source_curation.rejection_reason_associations",
        )
        if rejection_associations != sum(reason_counts.values()):
            raise MaterializationError(
                "All-eligible rejection reason association accounting differs"
            )

        groups = _require_exact_object_keys(
            source["groups_subphase"],
            {
                "status",
                "processed_rows",
                "processed_tokens",
                "committed_batches",
                "cursor",
                "details",
            },
            "selection.identity.source_curation.groups_subphase",
        )
        if groups["status"] != "complete":
            raise MaterializationError(
                "All-eligible source leakage-safe group assignment is incomplete"
            )
        cls._require_positive_int(
            groups["processed_rows"],
            "selection.identity.source_curation.groups_subphase.processed_rows",
        )
        _require_nonnegative_int(
            groups["processed_tokens"],
            "selection.identity.source_curation.groups_subphase.processed_tokens",
        )
        cls._require_positive_int(
            groups["committed_batches"],
            "selection.identity.source_curation.groups_subphase.committed_batches",
        )
        if not isinstance(groups["cursor"], dict) or not isinstance(
            groups["details"], dict
        ):
            raise MaterializationError(
                "All-eligible source group progress evidence is malformed"
            )

        ignored = _require_exact_object_keys(
            source["ignored_partial_exact_selection"],
            {"documents", "tokens", "quota_subphases", "authority"},
            "selection.identity.source_curation.ignored_partial_exact_selection",
        )
        ignored_documents = _require_nonnegative_int(
            ignored["documents"],
            "selection.identity.source_curation.ignored_partial_exact_selection.documents",
        )
        ignored_tokens = _require_nonnegative_int(
            ignored["tokens"],
            "selection.identity.source_curation.ignored_partial_exact_selection.tokens",
        )
        if (
            ignored["authority"] is not False
            or (ignored_documents == 0) != (ignored_tokens == 0)
            or not isinstance(ignored["quota_subphases"], list)
        ):
            raise MaterializationError(
                "Ignored exact-quota selection evidence is inconsistent"
            )
        quota_names: list[str] = []
        for index, raw_row in enumerate(ignored["quota_subphases"]):
            row = _require_exact_object_keys(
                raw_row,
                {
                    "subphase",
                    "status",
                    "processed_rows",
                    "processed_tokens",
                    "committed_batches",
                },
                "selection.identity.source_curation.ignored_partial_exact_selection."
                f"quota_subphases[{index}]",
            )
            subphase = cls._require_nonempty_string(
                row["subphase"], f"source quota subphase {index}"
            )
            quota_names.append(subphase)
            if not subphase.startswith("selection.quota.") or row["status"] not in (
                "running",
                "complete",
            ):
                raise MaterializationError("Invalid ignored quota subphase evidence")
            for field in (
                "processed_rows",
                "processed_tokens",
                "committed_batches",
            ):
                _require_nonnegative_int(
                    row[field], f"source quota subphase {index}.{field}"
                )
        if quota_names != sorted(set(quota_names)):
            raise MaterializationError(
                "Ignored quota subphase evidence is not unique canonical order"
            )

    @classmethod
    def _validate_all_eligible_identity_fields(
        cls,
        identity: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        handoff = _require_exact_object_keys(
            identity.get("fast_all_eligible_handoff"),
            set(FAST_ALL_ELIGIBLE_HANDOFF_PROFILE),
            "selection.identity.fast_all_eligible_handoff",
        )
        if dict(handoff) != FAST_ALL_ELIGIBLE_HANDOFF_PROFILE:
            raise MaterializationError(
                "Unsupported fast all-eligible handoff profile"
            )
        policy = identity.get("raw_archive_integrity_policy")
        if policy not in RAW_ARCHIVE_INTEGRITY_POLICIES:
            raise MaterializationError(
                "Unsupported source raw-archive integrity policy"
            )
        execution = _require_exact_object_keys(
            identity.get("sqlite_execution"),
            {
                "mode",
                "protocol_version",
                "active_journal_mode",
                "canonical_journal_mode",
                "snapshot_retention",
                "wal_autocheckpoint_pages",
                "locking_mode",
            },
            "selection.identity.sqlite_execution",
        )
        for field, expected in LOCAL_SQLITE_EXECUTION.items():
            if execution[field] != expected:
                raise MaterializationError(
                    f"Source SQLite execution contract mismatch for {field}"
                )
        if execution["canonical_journal_mode"] not in ("delete", "wal"):
            raise MaterializationError(
                "Source SQLite canonical journal mode is invalid"
            )
        cls._require_positive_int(
            execution["snapshot_retention"],
            "selection.identity.sqlite_execution.snapshot_retention",
        )
        if (
            identity.get("sqlite_runtime", {})
            .get("journal_policy", {})
            .get("selected_mode")
            != execution["canonical_journal_mode"]
        ):
            raise MaterializationError(
                "Source SQLite execution/canonical runtime identities differ"
            )
        cls._validate_all_eligible_source_curation(identity, manifest)

    def _validate_selection_manifest(self) -> dict[str, Any]:
        if not self.selection_manifest_path.is_file():
            raise MaterializationError(
                f"Missing completed curation manifest: {self.selection_manifest_path}"
            )
        manifest = _load_json(self.selection_manifest_path)
        checksum_path = self.selection_root / "manifest.sha256"
        expected_checksum_line = (
            f"{file_sha256(self.selection_manifest_path)}  manifest.json\n"
        )
        try:
            checksum_line = checksum_path.read_text(encoding="ascii")
        except OSError as error:
            raise MaterializationError(f"Missing curation checksum: {checksum_path}") from error
        if checksum_line != expected_checksum_line:
            raise MaterializationError("Curation manifest.sha256 does not match manifest.json")
        expected = {
            "manifest_version": 1,
            "decision_record_version": 1,
            "production_ready": True,
            "raw_archives_opened": False,
            "raw_archives_hashed_for_integrity": True,
            "raw_archive_payloads_parsed_by_curation": False,
            "quota_unit": "pre_packing_starcoder2_content_tokens",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise MaterializationError(
                    f"Curation manifest {key} mismatch: {manifest.get(key)!r} != {value!r}"
                )
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise MaterializationError("Unsupported curation identity")
        identity_version = identity.get("format_version")
        if identity_version not in (
            5,
            FAST_CURATION_IDENTITY_FORMAT_VERSION,
            ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
        ):
            raise MaterializationError("Unsupported curation identity")
        fast_profile = identity_version in (
            FAST_CURATION_IDENTITY_FORMAT_VERSION,
            ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
        )
        all_eligible = identity_version == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        if all_eligible:
            try:
                top_selection_profile = validate_all_eligible_selection_profile(
                    manifest.get("selection_profile")
                )
                identity_selection_profile = validate_all_eligible_selection_profile(
                    identity.get("selection_profile")
                )
            except ValueError as error:
                raise MaterializationError(
                    "Unsupported all-eligible selection profile"
                ) from error
            if (
                top_selection_profile != identity_selection_profile
                or manifest.get("selection_strategy")
                != ALL_ELIGIBLE_SELECTION_STRATEGY
                or manifest.get("publication_scope")
                != ALL_ELIGIBLE_PUBLICATION_SCOPE
                or manifest.get("training_input_budget_authority")
                != ALL_ELIGIBLE_TRAINING_INPUT_BUDGET_AUTHORITY
                or manifest.get("decision_format")
                != ALL_ELIGIBLE_BITMAP_FORMAT
                or manifest.get("decision_format_version")
                != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
            ):
                raise MaterializationError(
                    "All-eligible publication authority contract mismatch"
                )
        if fast_profile:
            if manifest.get("english_near_dedup_complete") is not False:
                raise MaterializationError(
                    "Fast curation manifest must record incomplete English near-dedup"
                )
            if manifest.get("english_near_dedup_status") != FAST_ENGLISH_NEAR_STATUS:
                raise MaterializationError(
                    "Fast curation manifest English near-dedup status mismatch"
                )
        elif manifest.get("english_near_dedup_complete") is not True:
            raise MaterializationError(
                "Full curation manifest requires complete English near-dedup"
            )
        elif "english_near_dedup_status" in manifest:
            raise MaterializationError(
                "Full curation manifest must not carry a disabled near-dedup status"
            )

        leakage = manifest.get("leakage_audit")
        if not isinstance(leakage, dict):
            raise MaterializationError("Curation manifest has no leakage audit")
        leakage_gates = [
            "content_hashes_in_multiple_splits",
            "canonical_clusters_in_multiple_splits",
            "source_groups_in_multiple_splits",
            "cross_bucket_code_repo_groups_in_multiple_splits",
        ]
        if fast_profile:
            leakage_gates.append("normalized_hashes_in_multiple_splits")
        for key in leakage_gates:
            if leakage.get(key) != 0:
                raise MaterializationError(f"Curation leakage audit is unsafe: {key}")
        identity_keys = {
            "format_version",
            "policy_sha256",
            "quota_config_sha256",
            "benchmark_guard_sha256",
            "preprocess_manifest_sha256",
            "report_inventory_sha256",
            "report_count",
            "collection_completeness",
            "english_near_clusters_sha256",
            "english_near_artifact",
            "sqlite_runtime",
            "curation_storage_contract",
            "tokenizer_manifest_sha256",
            "tokenizer_revision",
            "tokenizer_files_validated",
            "source_manifests",
            "legacy_stack_tokenizer_binding",
        }
        if fast_profile:
            identity_keys.add("curation_profile")
        if all_eligible:
            identity_keys.update(
                {
                    "fast_all_eligible_handoff",
                    "raw_archive_integrity_policy",
                    "sqlite_execution",
                    "selection_profile",
                    "source_curation",
                }
            )
        _require_exact_object_keys(
            identity,
            identity_keys,
            "selection.identity",
        )
        self._validate_collection_completeness(manifest, identity)
        for key in (
            "policy_sha256",
            "quota_config_sha256",
            "benchmark_guard_sha256",
            "preprocess_manifest_sha256",
            "report_inventory_sha256",
            "tokenizer_manifest_sha256",
        ):
            _require_sha256(identity.get(key), f"selection.identity.{key}")
        if identity.get("legacy_stack_tokenizer_binding") != (
            "collector_configuration_not_source_manifest_field"
        ):
            raise MaterializationError(
                "Curation legacy Stack tokenizer binding identity mismatch"
            )
        if fast_profile:
            profile = _require_exact_object_keys(
                identity.get("curation_profile"),
                set(FAST_CURATION_PROFILE),
                "selection.identity.curation_profile",
            )
            if dict(profile) != FAST_CURATION_PROFILE:
                raise MaterializationError("Unsupported fast curation profile")
            if manifest.get("curation_profile") != profile:
                raise MaterializationError(
                    "Fast curation profile was not copied faithfully"
                )
            if (
                identity.get("english_near_clusters_sha256") is not None
                or identity.get("english_near_artifact") is not None
            ):
                raise MaterializationError(
                    "Fast curation profile must not carry English near artifacts"
                )
            limitations = manifest.get("known_provenance_limitations")
            if (
                not isinstance(limitations, list)
                or any(not isinstance(item, str) or not item for item in limitations)
                or len(limitations) != len(set(limitations))
                or any(
                    item not in limitations
                    for item in FAST_CURATION_PROFILE["known_limitations"]
                )
            ):
                raise MaterializationError(
                    "Fast curation profile limitations were not preserved verbatim"
                )
            if all_eligible and any(
                item not in limitations
                for item in ALL_ELIGIBLE_SELECTION_PROFILE["known_limitations"]
            ):
                raise MaterializationError(
                    "All-eligible selection limitations were not preserved verbatim"
                )
            fast_audit = _require_exact_object_keys(
                manifest.get("fast_profile_audit"),
                {
                    "audit_version",
                    "fuzzy_near_dedup_performed",
                    "near_map_rows",
                    "near_mapping_subphases",
                    "english_near_duplicate_reasons",
                    "content_hashes_in_multiple_splits",
                    "normalized_hashes_in_multiple_splits",
                    "content_hashes_with_multiple_selected_documents",
                    "normalized_hashes_with_multiple_selected_documents",
                    "source_groups_in_multiple_splits",
                },
                "selection.fast_profile_audit",
            )
            expected_fast_audit = {
                "audit_version": 1,
                "fuzzy_near_dedup_performed": False,
                "near_map_rows": 0,
                "near_mapping_subphases": 0,
                "english_near_duplicate_reasons": 0,
                "content_hashes_in_multiple_splits": leakage[
                    "content_hashes_in_multiple_splits"
                ],
                "normalized_hashes_in_multiple_splits": leakage[
                    "normalized_hashes_in_multiple_splits"
                ],
                "content_hashes_with_multiple_selected_documents": leakage.get(
                    "content_hashes_with_multiple_selected_documents"
                ),
                "normalized_hashes_with_multiple_selected_documents": leakage.get(
                    "normalized_hashes_with_multiple_selected_documents"
                ),
                "source_groups_in_multiple_splits": leakage[
                    "source_groups_in_multiple_splits"
                ],
            }
            if dict(fast_audit) != expected_fast_audit or any(
                value != 0
                for key, value in expected_fast_audit.items()
                if key != "audit_version"
                and key != "fuzzy_near_dedup_performed"
            ):
                raise MaterializationError("Fast curation audit is unsafe")
        else:
            if "curation_profile" in manifest or "fast_profile_audit" in manifest:
                raise MaterializationError(
                    "Full curation manifest must not carry fast-profile evidence"
                )
            _require_sha256(
                identity.get("english_near_clusters_sha256"),
                "selection.identity.english_near_clusters_sha256",
            )
            self._validate_english_near_artifact(identity, manifest)
        self._validate_sqlite_runtime(identity)
        self._validate_curation_storage_contract(
            identity,
            all_eligible=all_eligible,
        )
        if all_eligible:
            self._validate_all_eligible_identity_fields(identity, manifest)
            self._validate_all_eligible_totals(manifest)
        decisions = manifest.get("decision_shards")
        if not isinstance(decisions, list) or not decisions:
            raise MaterializationError("Curation manifest has no decision shards")
        if canonical_sha256(decisions) != manifest.get("decision_inventory_sha256"):
            raise MaterializationError("Decision inventory checksum is inconsistent")
        return manifest

    def _validate_policy_and_supporting_artifacts(self) -> dict[str, Any]:
        policy = _load_json(self.policy_path)
        identity = self.selection_manifest["identity"]
        if canonical_sha256(policy) != identity["policy_sha256"]:
            raise MaterializationError("Curation policy identity mismatch")
        if self.selection_manifest.get("selection_policy") != policy.get("selection"):
            raise MaterializationError("Selection policy was not copied faithfully")
        if identity["format_version"] in (
            FAST_CURATION_IDENTITY_FORMAT_VERSION,
            ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
        ):
            if (
                policy.get("policy_version") != 2
                or policy.get("curation_profile") != FAST_CURATION_PROFILE
                or policy.get("selection", {}).get("english_near_dedup")
                != {
                    "required_for_production": False,
                    "mapping_format": None,
                    "missing_mapping_action": FAST_ENGLISH_NEAR_STATUS,
                }
            ):
                raise MaterializationError(
                    "Fast curation policy/profile routing mismatch"
                )
            expected_english_route = {
                "local_near_dedup": False,
                "near_duplicate_authority": FAST_ENGLISH_NEAR_STATUS,
                "split_grouping": "stable_english_source_after_canonicalization",
            }
            buckets = policy.get("buckets")
            if not isinstance(buckets, dict) or any(
                buckets.get(bucket) != expected_english_route
                for bucket in ("fineweb_edu", "wikipedia")
            ):
                raise MaterializationError(
                    "Fast curation English bucket routing mismatch"
                )
        elif policy.get("policy_version") != 1 or "curation_profile" in policy:
            raise MaterializationError("Full curation policy/profile routing mismatch")
        for path, key, name in (
            (self.quota_path, "quota_config_sha256", "quota configuration"),
            (
                self.benchmark_denylist_path,
                "benchmark_guard_sha256",
                "benchmark denylist",
            ),
            (
                self.preprocess_root / "PREPROCESS_MANIFEST.json",
                "preprocess_manifest_sha256",
                "preprocess manifest",
            ),
        ):
            if not path.is_file() or file_sha256(path) != identity[key]:
                raise MaterializationError(f"{name} identity mismatch: {path}")
        self._configure_all_eligible_split_authority(policy)
        source_manifests = identity.get("source_manifests")
        if not isinstance(source_manifests, dict) or not source_manifests:
            raise MaterializationError("Curation identity has no source manifests")
        for filename, descriptor in sorted(source_manifests.items()):
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(descriptor, dict)
            ):
                raise MaterializationError("Unsafe source-manifest descriptor")
            path = _safe_file(self.raw_root / "manifests", filename, "source manifest")
            if file_sha256(path) != descriptor.get("sha256"):
                raise MaterializationError(f"Source manifest checksum mismatch: {path}")
        reports = self.selection_manifest.get("input_reports")
        if not isinstance(reports, list):
            raise MaterializationError("Curation manifest has no input report inventory")
        report_projection = [
            {"path": row.get("report"), "sha256": row.get("report_sha256")}
            for row in reports
            if isinstance(row, dict)
        ]
        if (
            len(report_projection) != len(reports)
            or canonical_sha256(report_projection) != identity["report_inventory_sha256"]
            or identity.get("report_count") != len(reports)
        ):
            raise MaterializationError("Input report inventory identity mismatch")
        return policy

    def _load_tokenizer(self) -> tuple[Any, dict[str, Any]]:
        manifest_path = self.tokenizer_root / "TOKENIZER_MANIFEST.json"
        manifest = _load_json(manifest_path)
        expected_sha = self.selection_manifest["identity"]["tokenizer_manifest_sha256"]
        if file_sha256(manifest_path) != expected_sha:
            raise MaterializationError("Tokenizer manifest identity mismatch")
        if manifest.get("manifest_version") != 1:
            raise MaterializationError("Unsupported tokenizer manifest version")
        revision = manifest.get("resolved_revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or revision != self.selection_manifest["identity"].get("tokenizer_revision")
        ):
            raise MaterializationError("Tokenizer revision mismatch")
        validation = manifest.get("validation")
        if not isinstance(validation, dict):
            raise MaterializationError("Tokenizer manifest has no validation payload")
        if validation.get("vocab_size") != self.config.expected_vocab_size:
            raise MaterializationError("Tokenizer vocabulary does not match the model")
        if validation.get("eos_token_id") != self.config.expected_eos_token_id:
            raise MaterializationError("Tokenizer EOS ID does not match the model")
        files = manifest.get("files")
        if not isinstance(files, dict) or "tokenizer.json" not in files:
            raise MaterializationError("Tokenizer manifest does not pin tokenizer.json")
        if self.selection_manifest["identity"].get("tokenizer_files_validated") != sorted(
            files
        ):
            raise MaterializationError(
                "Curation tokenizer-files identity differs from tokenizer manifest"
            )
        for filename, descriptor in sorted(files.items()):
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(descriptor, dict)
            ):
                raise MaterializationError("Unsafe tokenizer file descriptor")
            artifact = _safe_file(self.tokenizer_root, filename, "tokenizer artifact")
            expected_bytes = _require_nonnegative_int(
                descriptor.get("bytes"), f"tokenizer.files.{filename}.bytes"
            )
            expected_file_sha = _require_sha256(
                descriptor.get("sha256"), f"tokenizer.files.{filename}.sha256"
            )
            if artifact.stat().st_size != expected_bytes or file_sha256(artifact) != expected_file_sha:
                raise MaterializationError(f"Tokenizer artifact identity mismatch: {artifact}")
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise MaterializationError(
                "Install requirements-data.txt to materialize the corpus"
            ) from error
        tokenizer = Tokenizer.from_file(str(self.tokenizer_root / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        if tokenizer.get_vocab_size(with_added_tokens=True) != self.config.expected_vocab_size:
            raise MaterializationError("Loaded tokenizer vocabulary differs from its manifest")
        eos_token = validation.get("eos_token")
        if not isinstance(eos_token, str) or tokenizer.token_to_id(eos_token) != (
            self.config.expected_eos_token_id
        ):
            raise MaterializationError("Loaded tokenizer EOS differs from its manifest")
        return tokenizer, manifest

    def _load_raw_token_cache_inventory(self) -> None:
        """Bind one closed-world cache generation to the authenticated v7 inputs."""

        if (
            self.selection_manifest["identity"]["format_version"]
            != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        ):
            raise MaterializationError(
                "Raw-token-cache materialization is supported only for selection v7"
            )
        assert self.raw_token_cache_root is not None
        assert self.raw_token_cache_inventory_root is not None
        try:
            inventory = load_raw_token_cache_inventory(
                inventory_root=self.raw_token_cache_inventory_root,
                cache_root=self.raw_token_cache_root,
            )
            validation = self.tokenizer_manifest["validation"]
            expected_tokenizer = RawTokenTokenizerAuthority(
                repo_id=self.tokenizer_manifest.get("repo_id"),
                resolved_revision=self.tokenizer_manifest.get("resolved_revision"),
                manifest_sha256=file_sha256(
                    self.tokenizer_root / "TOKENIZER_MANIFEST.json"
                ),
                vocabulary_sha256=vocabulary_sha256(
                    self.tokenizer.get_vocab(with_added_tokens=True)
                ),
                vocab_size=self.config.expected_vocab_size,
                eos_token=validation.get("eos_token"),
                eos_token_id=self.config.expected_eos_token_id,
            )
        except (
            OSError,
            RawTokenCacheInventoryError,
            RawTokenCacheReadError,
            ValueError,
        ) as error:
            raise MaterializationError(
                f"Cannot authenticate raw-token-cache inventory: {error}"
            ) from error
        if inventory.selection_manifest_sha256 != self.selection_manifest_sha256:
            raise MaterializationError(
                "Raw-token-cache inventory selection-manifest identity mismatch"
            )
        if inventory.tokenizer != expected_tokenizer:
            raise MaterializationError(
                "Raw-token-cache inventory tokenizer identity mismatch"
            )
        if len(inventory.entries) != len(self.archives):
            raise MaterializationError(
                "Raw-token-cache inventory does not cover every source archive"
            )
        authorities: dict[int, RawTokenCacheAuthority] = {}
        for archive, entry in zip(self.archives, inventory.entries, strict=True):
            expected_archive = RawTokenArchiveAuthority(
                path=archive.archive,
                bucket=archive.bucket,
                index=archive.archive_index,
                bytes=archive.raw_path.stat().st_size,
                sha256=archive.raw_sha256,
            )
            expected_report = RawTokenFileAuthority(
                path=archive.report_relative,
                bytes=archive.report_path.stat().st_size,
                sha256=archive.report_sha256,
            )
            expected_fingerprint = RawTokenFileAuthority(
                path=archive.fingerprint_relative,
                bytes=archive.fingerprint_path.stat().st_size,
                sha256=archive.fingerprint_sha256,
            )
            authority = entry.authority
            if (
                entry.ordinal != archive.ordinal
                or authority.archive != expected_archive
                or authority.preprocess_report != expected_report
                or authority.fingerprint != expected_fingerprint
                or authority.tokenizer != expected_tokenizer
                or authority.records != archive.documents
                or authority.clean_bytes != archive.clean_bytes
                or authority.content_tokens != archive.content_tokens
            ):
                raise MaterializationError(
                    "Raw-token-cache inventory archive authority differs from "
                    f"selection/report authority at ordinal {archive.ordinal}"
                )
            authorities[archive.ordinal] = authority
        self.raw_token_cache_inventory = inventory
        self.raw_token_cache_authorities = authorities

    def _build_archive_inventory(self) -> list[ArchiveSpec]:
        input_reports = self.selection_manifest["input_reports"]
        decision_shards = self.selection_manifest["decision_shards"]
        all_eligible = (
            self.selection_manifest["identity"]["format_version"]
            == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        )
        report_by_archive: dict[str, dict[str, Any]] = {}
        for row in input_reports:
            if not isinstance(row, dict) or not isinstance(row.get("archive"), str):
                raise MaterializationError("Invalid input-report descriptor")
            if row["archive"] in report_by_archive:
                raise MaterializationError(f"Duplicate input report: {row['archive']}")
            report_by_archive[row["archive"]] = row
        if [row.get("archive") for row in decision_shards] != sorted(report_by_archive):
            raise MaterializationError(
                "Decision shards must cover every archive in canonical archive order"
            )
        archives: list[ArchiveSpec] = []
        for ordinal, decision in enumerate(decision_shards):
            if not isinstance(decision, dict):
                raise MaterializationError("Invalid decision-shard descriptor")
            archive_name = decision.get("archive")
            report_descriptor = report_by_archive.get(str(archive_name))
            if report_descriptor is None:
                raise MaterializationError(f"Decision archive has no report: {archive_name}")
            report_relative = report_descriptor.get("report")
            report_path = _safe_file(
                self.preprocess_root, report_relative, "preprocess report"
            )
            report = _load_json(report_path)
            bucket = report.get("bucket")
            if bucket not in BUCKET_DOMAIN:
                raise MaterializationError(f"Unsupported raw bucket in {report_path}: {bucket}")
            expected_report = {
                "archive": archive_name,
                "archive_sha256": report_descriptor.get("archive_sha256"),
                "fingerprint_file": report_descriptor.get("fingerprint_file"),
                "fingerprint_sha256": report_descriptor.get("fingerprint_sha256"),
                "documents": report_descriptor.get("documents"),
                "exact_tokens": report_descriptor.get("content_tokens"),
            }
            for key, expected in expected_report.items():
                if report.get(key) != expected:
                    raise MaterializationError(f"Report {key} mismatch in {report_path}")
            documents = _require_nonnegative_int(
                report_descriptor.get("documents"), "input report documents"
            )
            content_tokens = _require_nonnegative_int(
                report_descriptor.get("content_tokens"), "input report content_tokens"
            )
            clean_bytes = _require_nonnegative_int(
                report.get("clean_bytes"), "input report clean_bytes"
            )
            if documents < 1 or content_tokens < 1 or clean_bytes < 1:
                raise MaterializationError("Input archives must be non-empty")
            if decision.get("records") != documents:
                raise MaterializationError("Decision/report document count mismatch")
            decision_path = _safe_file(
                self.selection_root, decision.get("path"), "decision shard"
            )
            decision_format: str | None = None
            decision_format_version: int | None = None
            decision_bytes: int | None = None
            decision_kept_documents: int | None = None
            if all_eligible:
                _require_exact_object_keys(
                    decision,
                    set(ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS),
                    f"selection decision descriptor {archive_name}",
                )
                if (
                    decision.get("format") != ALL_ELIGIBLE_BITMAP_FORMAT
                    or decision.get("format_version")
                    != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
                ):
                    raise MaterializationError(
                        "Unsupported all-eligible decision bitmap format"
                    )
                decision_format = ALL_ELIGIBLE_BITMAP_FORMAT
                decision_format_version = ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
                decision_bytes = self._require_positive_int(
                    decision.get("bytes"), "decision bitmap bytes"
                )
                decision_kept_documents = _require_nonnegative_int(
                    decision.get("kept_documents"),
                    "decision bitmap kept_documents",
                )
                if (
                    decision_kept_documents > documents
                    or decision_path.stat().st_size != decision_bytes
                ):
                    raise MaterializationError(
                        "All-eligible decision bitmap descriptor is inconsistent"
                    )
            archives.append(
                ArchiveSpec(
                    ordinal=ordinal,
                    archive=str(archive_name),
                    archive_index=_require_nonnegative_int(
                        report.get("index"), "preprocess report index"
                    ),
                    bucket=str(bucket),
                    domain=BUCKET_DOMAIN[str(bucket)],
                    raw_path=_safe_file(self.raw_root, archive_name, "raw archive"),
                    raw_sha256=_require_sha256(
                        report_descriptor.get("archive_sha256"), "raw archive SHA-256"
                    ),
                    report_path=report_path,
                    report_relative=str(report_relative),
                    report_sha256=_require_sha256(
                        report_descriptor.get("report_sha256"), "report SHA-256"
                    ),
                    fingerprint_path=_safe_file(
                        self.preprocess_root,
                        report_descriptor.get("fingerprint_file"),
                        "fingerprint shard",
                    ),
                    fingerprint_relative=str(report_descriptor["fingerprint_file"]),
                    fingerprint_sha256=_require_sha256(
                        report_descriptor.get("fingerprint_sha256"),
                        "fingerprint SHA-256",
                    ),
                    decision_path=decision_path,
                    decision_relative=str(decision.get("path")),
                    decision_sha256=_require_sha256(
                        decision.get("sha256"), "decision SHA-256"
                    ),
                    documents=documents,
                    clean_bytes=clean_bytes,
                    content_tokens=content_tokens,
                    decision_format=decision_format,
                    decision_format_version=decision_format_version,
                    decision_bytes=decision_bytes,
                    decision_kept_documents=decision_kept_documents,
                )
            )
        return archives

    def _validate_archive_inventory_against_completeness(self) -> None:
        authority = self.selection_manifest["collection_completeness"]["per_bucket"]
        for bucket in sorted(BUCKET_DOMAIN):
            archives = [item for item in self.archives if item.bucket == bucket]
            observed = {
                "archives": len(archives),
                "documents": sum(item.documents for item in archives),
                "clean_bytes": sum(item.clean_bytes for item in archives),
                "exact_tokens": sum(item.content_tokens for item in archives),
            }
            expected = {
                field: authority[bucket][field] for field in observed
            }
            if observed != expected:
                raise MaterializationError(
                    f"Archive inventory differs from collection authority for {bucket}: "
                    f"{observed} != {expected}"
                )
        if (
            self.selection_manifest["identity"]["format_version"]
            == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
            and sum(item.documents for item in self.archives)
            != self.selection_manifest["documents"]["input"]
        ):
            raise MaterializationError(
                "All-eligible input document total differs from archive authority"
            )

    def _initial_cursor(self, split: str, domain: str) -> dict[str, Any]:
        return {
            "format": CURSOR_FORMAT,
            "version": CURSOR_VERSION,
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "split": split,
            "domain": domain,
            "next_archive": 0,
            "selected_documents": 0,
            "selected_content_tokens": 0,
            "terminal_prefix_documents": 0,
            "document_index_shards": [],
        }

    def _validate_cursor(
        self, cursor: Any, *, split: str, domain: str
    ) -> dict[str, Any]:
        if cursor is None:
            return self._initial_cursor(split, domain)
        if not isinstance(cursor, dict):
            raise MaterializationError(f"Invalid packed cursor for {split}/{domain}")
        expected = {
            "format": CURSOR_FORMAT,
            "version": CURSOR_VERSION,
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "split": split,
            "domain": domain,
        }
        for key, value in expected.items():
            if cursor.get(key) != value:
                raise MaterializationError(
                    f"Packed cursor {key} mismatch for {split}/{domain}"
                )
        for key in (
            "next_archive",
            "selected_documents",
            "selected_content_tokens",
            "terminal_prefix_documents",
        ):
            _require_nonnegative_int(cursor.get(key), f"packed cursor {key}")
        index_shards = cursor.get("document_index_shards")
        if not isinstance(index_shards, list):
            raise MaterializationError("Packed cursor has no document-index inventory")
        last_ordinal = -1
        index_records = 0
        for descriptor in index_shards:
            if not isinstance(descriptor, dict):
                raise MaterializationError("Invalid document-index shard descriptor")
            ordinal = _require_nonnegative_int(
                descriptor.get("archive_ordinal"), "document-index archive ordinal"
            )
            if ordinal <= last_ordinal or ordinal >= cursor["next_archive"]:
                raise MaterializationError(
                    "Document-index archive ordinals are not a committed prefix"
                )
            last_ordinal = ordinal
            records = _require_nonnegative_int(
                descriptor.get("records"), "document-index records"
            )
            size = _require_nonnegative_int(
                descriptor.get("bytes"), "document-index bytes"
            )
            if records < 1 or size < 1:
                raise MaterializationError("Document-index shards must be non-empty")
            _require_sha256(descriptor.get("sha256"), "document-index shard SHA-256")
            expected_path = (
                Path("provenance")
                / "documents"
                / split
                / domain
                / f"archive-{ordinal:06d}.jsonl.zst"
            ).as_posix()
            if descriptor.get("path") != expected_path:
                raise MaterializationError("Non-canonical document-index shard path")
            shard_path = _safe_file(
                self.output_root, expected_path, "document-index shard"
            )
            if (
                shard_path.stat().st_size != size
                or file_sha256(shard_path) != descriptor["sha256"]
            ):
                raise MaterializationError(
                    f"Document-index shard identity mismatch: {shard_path}"
                )
            index_records += records
        if index_records != cursor["selected_documents"]:
            raise MaterializationError(
                "Document-index record count differs from selected documents"
            )
        if cursor["next_archive"] > len(self.archives):
            raise MaterializationError("Packed cursor is past the archive inventory")
        return dict(cursor)

    def _journal_payload(self, phase: str) -> dict[str, Any]:
        cursors = {
            f"{split}/{domain}": state.cursor
            for (split, domain), state in sorted(self.destinations.items())
        }
        completed = (
            min(cursor["next_archive"] for cursor in cursors.values()) if cursors else 0
        )
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "identity": self.identity,
            "state": {
                "phase": phase,
                "completed_archives": completed,
                "archive_count": len(self.archives),
                "writer_cursors": cursors,
            },
        }

    def _write_journal(self, phase: str) -> None:
        atomic_json(self.journal_path, self._journal_payload(phase))

    def _prepare_output(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        final_manifest = self.output_root / "manifest.json"
        final_checksum = self.output_root / "manifest.sha256"
        if final_manifest.exists() and final_checksum.exists():
            return
        if self.journal_path.exists():
            journal = _load_json(self.journal_path)
            if (
                journal.get("format") != FORMAT
                or journal.get("format_version") != FORMAT_VERSION
                or journal.get("identity") != self.identity
            ):
                raise MaterializationError("Materialization journal identity mismatch")
        else:
            entries = list(self.output_root.iterdir())
            if entries:
                raise MaterializationError(
                    f"Refusing non-empty output without a valid journal: {self.output_root}"
                )
            atomic_json(
                self.journal_path,
                {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "identity": self.identity,
                    "state": {
                        "phase": "packing",
                        "completed_archives": 0,
                        "archive_count": len(self.archives),
                        "writer_cursors": {},
                    },
                },
            )
        allowed = {
            JOURNAL_NAME,
            "packed",
            "orders",
            "provenance",
            "manifest.json",
            "manifest.sha256",
        }
        unexpected = sorted(path.name for path in self.output_root.iterdir() if path.name not in allowed)
        if unexpected:
            raise MaterializationError(f"Unknown materialization outputs: {unexpected}")

    def _open_destinations(self) -> None:
        self.destinations = {}
        tokenizer_sha = self.identity["tokenizer_manifest_sha256"]
        policy_sha = self.identity["curation_policy_sha256"]
        selection_sha = self.identity["selection_manifest_sha256"]
        for split in SPLITS:
            for domain in DOMAIN_ORDER:
                writer = PackedShardWriter(
                    self.output_root / "packed" / split / domain,
                    domain=domain,
                    split=split,
                    sequence_length=self.config.sequence_length,
                    vocab_size=self.config.expected_vocab_size,
                    eos_token_id=self.config.expected_eos_token_id,
                    tokenizer_manifest_sha256=tokenizer_sha,
                    rows_per_shard=self.config.rows_per_shard,
                    construction_seed=self.config.construction_seed,
                    curation_policy_sha256=policy_sha,
                    selection_manifest_sha256=selection_sha,
                    resume=True,
                )
                cursor = self._validate_cursor(
                    writer.source_cursor, split=split, domain=domain
                )
                self.destinations[(split, domain)] = _DestinationState(
                    split=split, domain=domain, writer=writer, cursor=cursor
                )
        positions = [state.cursor["next_archive"] for state in self.destinations.values()]
        if max(positions) - min(positions) > 1:
            raise MaterializationError(
                "Writer cursors differ by more than one archive; checkpoint protocol was violated"
            )
        self._write_journal("packing")

    def _close_destinations_after_error(self, error: BaseException) -> None:
        for state in self.destinations.values():
            with contextlib.suppress(Exception):
                state.writer.__exit__(type(error), error, error.__traceback__)

    def _validate_decision(
        self, row: dict[str, Any], archive: ArchiveSpec, index: int
    ) -> None:
        expected_doc_id = hashlib.sha256(
            f"{archive.archive}\0{row.get('member_path')}".encode("utf-8")
        ).hexdigest()
        expected = {
            "record_version": 1,
            "bucket": archive.bucket,
            "category": archive.domain,
            "archive": archive.archive,
            "archive_index": archive.archive_index,
            "manifest_index": index,
            "doc_id": expected_doc_id,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise MaterializationError(
                    f"Decision {key} mismatch in {archive.decision_path} row {index}"
                )
        member_path = row.get("member_path")
        if not isinstance(member_path, str) or not member_path or member_path == "_manifest.jsonl":
            raise MaterializationError("Decision has an invalid raw member path")
        for key in ("content_sha256", "normalized_sha256"):
            _require_sha256(row.get(key), f"decision.{key}")
        source_tokens = _require_nonnegative_int(row.get("source_tokens"), "source_tokens")
        selected_tokens = _require_nonnegative_int(
            row.get("selected_tokens"), "selected_tokens"
        )
        if source_tokens < 1 or selected_tokens > source_tokens:
            raise MaterializationError("Decision token counts are inconsistent")
        flags = row.get("quality_flags")
        reasons = row.get("reasons")
        provenance = row.get("provenance")
        if (
            not isinstance(flags, list)
            or flags != sorted(set(flags))
            or not all(isinstance(value, str) for value in flags)
            or not isinstance(reasons, list)
            or reasons != sorted(reasons)
            or not all(isinstance(value, str) for value in reasons)
            or not isinstance(provenance, dict)
        ):
            raise MaterializationError("Decision flags/reasons/provenance are invalid")
        decision = row.get("decision")
        if (
            self.selection_manifest["identity"]["format_version"]
            == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
            and decision == "keep"
            and (
                selected_tokens != source_tokens
                or row.get("token_prefix") != [0, source_tokens]
                or row.get("terminal_quota_prefix") is not False
            )
        ):
            raise MaterializationError(
                "All-eligible kept decisions must preserve complete documents"
            )
        if decision == "keep":
            if (
                row.get("split") not in SPLITS
                or row.get("assigned_split") != row.get("split")
                or selected_tokens < 1
                or row.get("token_prefix") != [0, selected_tokens]
                or row.get("terminal_quota_prefix") != (selected_tokens < source_tokens)
                or reasons
            ):
                raise MaterializationError("Kept decision is internally inconsistent")
            _require_sha256(row.get("canonical_doc_id"), "canonical_doc_id")
            _require_sha256(row.get("split_group_id"), "split_group_id")
        elif decision == "reject":
            if (
                row.get("split") is not None
                or selected_tokens != 0
                or row.get("token_prefix") is not None
                or row.get("terminal_quota_prefix") is not False
                or not reasons
            ):
                raise MaterializationError("Rejected decision is internally inconsistent")
            assigned = row.get("assigned_split")
            if assigned is not None and assigned not in SPLITS:
                raise MaterializationError("Rejected decision has an invalid assigned split")
        else:
            raise MaterializationError(f"Unknown curation decision: {decision!r}")

    def _compare_raw_manifest_row(
        self,
        raw_row: dict[str, Any],
        decision: dict[str, Any],
        archive: ArchiveSpec,
        index: int,
    ) -> None:
        if raw_row.get("member_path") != decision["member_path"]:
            raise MaterializationError(
                f"Raw manifest member order mismatch in {archive.raw_path} row {index}"
            )
        if raw_row.get("starcoder2_tokens") != decision["source_tokens"]:
            raise MaterializationError(
                f"Raw manifest token count mismatch in {archive.raw_path} row {index}"
            )
        if _provenance_from_raw_manifest(raw_row) != decision["provenance"]:
            raise MaterializationError(
                f"Raw provenance mismatch in {archive.raw_path} row {index}"
            )
        size = raw_row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise MaterializationError("Raw manifest has an invalid byte count")

    def _load_all_eligible_bitmap(self, archive: ArchiveSpec) -> bytes:
        if (
            archive.decision_format != ALL_ELIGIBLE_BITMAP_FORMAT
            or archive.decision_format_version
            != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
            or archive.decision_bytes is None
            or archive.decision_kept_documents is None
        ):
            raise MaterializationError(
                "All-eligible archive has no authenticated bitmap descriptor"
            )
        try:
            raw = archive.decision_path.read_bytes()
        except OSError as error:
            raise MaterializationError(
                f"Could not read decision bitmap: {archive.decision_path}"
            ) from error
        prefix_bytes = len(ALL_ELIGIBLE_BITMAP_MAGIC) + (
            ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES
        )
        if len(raw) != archive.decision_bytes or len(raw) < prefix_bytes + 1:
            raise MaterializationError(
                "All-eligible decision bitmap file length is inconsistent"
            )
        if raw[: len(ALL_ELIGIBLE_BITMAP_MAGIC)] != ALL_ELIGIBLE_BITMAP_MAGIC:
            raise MaterializationError("All-eligible decision bitmap magic mismatch")
        length_start = len(ALL_ELIGIBLE_BITMAP_MAGIC)
        length_end = length_start + ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES
        header_bytes_count = int.from_bytes(raw[length_start:length_end], "big")
        expected_payload_bytes = all_eligible_bitmap_payload_bytes(
            archive.documents
        )
        if (
            header_bytes_count < 1
            or length_end + header_bytes_count + expected_payload_bytes != len(raw)
        ):
            raise MaterializationError(
                "All-eligible decision bitmap header length is inconsistent"
            )
        header_raw = raw[length_end : length_end + header_bytes_count]
        payload = raw[length_end + header_bytes_count :]
        try:
            header_value = json.loads(header_raw.decode("utf-8", errors="strict"))
            header = validate_all_eligible_bitmap_header(header_value)
            validate_all_eligible_bitmap_payload(
                payload, records=archive.documents
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MaterializationError(
                "Invalid all-eligible decision bitmap header/payload"
            ) from error
        if canonical_json_bytes(header) != header_raw:
            raise MaterializationError(
                "All-eligible decision bitmap header is not canonical JSON"
            )
        expected_header = {
            "format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "archive": archive.archive,
            "bucket": archive.bucket,
            "category": archive.domain,
            "records": archive.documents,
            "kept_documents": archive.decision_kept_documents,
            "bit_order": ALL_ELIGIBLE_BITMAP_BIT_ORDER,
            "payload_bytes": expected_payload_bytes,
        }
        if header != expected_header:
            raise MaterializationError(
                "All-eligible decision bitmap header differs from its descriptor"
            )
        if sum(byte.bit_count() for byte in payload) != (
            archive.decision_kept_documents
        ):
            raise MaterializationError(
                "All-eligible decision bitmap kept-document count mismatch"
            )
        return payload

    def _iter_all_eligible_decisions(
        self, archive: ArchiveSpec, bitmap: bytes
    ) -> Iterator[dict[str, Any]]:
        records = 0
        kept_documents = 0
        for index, raw_row in enumerate(iter_jsonl_zst(archive.fingerprint_path)):
            if index >= archive.documents:
                raise MaterializationError(
                    "All-eligible fingerprint has more rows than its bitmap"
                )
            if not isinstance(raw_row, dict):
                raise MaterializationError(
                    "All-eligible fingerprint row is not an object"
                )
            member_path = raw_row.get("member_path")
            expected_doc_id = (
                hashlib.sha256(
                    f"{archive.archive}\0{member_path}".encode("utf-8")
                ).hexdigest()
                if isinstance(member_path, str) and member_path
                else None
            )
            expected = {
                "record_version": 1,
                "fingerprint_version": FINGERPRINT_VERSION,
                "doc_id": expected_doc_id,
                "archive": archive.archive,
                "bucket": archive.bucket,
                "archive_index": archive.archive_index,
                "manifest_index": index,
            }
            if any(raw_row.get(key) != value for key, value in expected.items()):
                raise MaterializationError(
                    f"All-eligible fingerprint identity mismatch at row {index}"
                )
            if member_path == "_manifest.jsonl":
                raise MaterializationError(
                    "All-eligible fingerprint has an invalid member path"
                )
            source_tokens = self._require_positive_int(
                raw_row.get("starcoder2_tokens"),
                f"all-eligible fingerprint row {index} tokens",
            )
            size_bytes = self._require_positive_int(
                raw_row.get("size_bytes"),
                f"all-eligible fingerprint row {index} bytes",
            )
            content_sha = _require_sha256(
                raw_row.get("content_sha256"),
                f"all-eligible fingerprint row {index} content_sha256",
            )
            normalized_sha = _require_sha256(
                raw_row.get("normalized_sha256"),
                f"all-eligible fingerprint row {index} normalized_sha256",
            )
            flags = raw_row.get("quality_flags")
            benchmark_reason = raw_row.get("benchmark_reason")
            provenance = raw_row.get("provenance")
            if (
                not isinstance(flags, list)
                or flags != sorted(set(flags))
                or any(not isinstance(flag, str) for flag in flags)
                or (
                    benchmark_reason is not None
                    and (
                        not isinstance(benchmark_reason, str)
                        or not benchmark_reason
                    )
                )
                or bool(benchmark_reason)
                != ("benchmark_contamination" in flags)
                or not isinstance(provenance, dict)
            ):
                raise MaterializationError(
                    f"All-eligible fingerprint metadata is invalid at row {index}"
                )
            keep = bool((bitmap[index // 8] >> (index % 8)) & 1)
            split_group_id: str | None = None
            split: str | None = None
            if keep:
                assert expected_doc_id is not None
                split_group_id, split = self._derive_all_eligible_group_and_split(
                    bucket=archive.bucket,
                    provenance=provenance,
                    doc_id=expected_doc_id,
                )
                kept_documents += 1
            records += 1
            yield {
                "record_version": 1,
                "doc_id": expected_doc_id,
                "bucket": archive.bucket,
                "category": archive.domain,
                "archive": archive.archive,
                "archive_index": archive.archive_index,
                "manifest_index": index,
                "member_path": member_path,
                "decision": "keep" if keep else "reject",
                "split": split,
                "assigned_split": split,
                "source_tokens": source_tokens,
                "selected_tokens": source_tokens if keep else 0,
                "token_prefix": [0, source_tokens] if keep else None,
                "terminal_quota_prefix": False,
                "canonical_doc_id": expected_doc_id if keep else None,
                "split_group_id": split_group_id,
                "reasons": [],
                "content_sha256": content_sha,
                "normalized_sha256": normalized_sha,
                "quality_flags": flags,
                "benchmark_reason": benchmark_reason,
                "provenance": provenance,
                "_fingerprint_size_bytes": size_bytes,
            }
        if records != archive.documents or kept_documents != (
            archive.decision_kept_documents
        ):
            raise MaterializationError(
                "All-eligible fingerprint/bitmap record totals differ"
            )

    def _process_archive_from_raw_token_cache(
        self, archive: ArchiveSpec, bitmap: bytes
    ) -> dict[str, int]:
        """Join v7 bitmap/fingerprint ordinals to authenticated cached spans.

        This transaction is intentionally separate from the legacy raw-stream
        implementation below.  Opting into the accelerator therefore cannot
        silently alter v5/v6 validation, batching, prefix, or replay behavior;
        byte-identity tests keep their shared writer/provenance contract fixed.
        """

        if (
            self.raw_token_cache_inventory is None
            or self.raw_token_cache_root is None
            or archive.decision_format != ALL_ELIGIBLE_BITMAP_FORMAT
            or self.selection_manifest["identity"]["format_version"]
            != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        ):
            raise MaterializationError(
                "Raw-token-cache adapter requires an authenticated v7 keep bitmap"
            )
        authority = self.raw_token_cache_authorities.get(archive.ordinal)
        if authority is None:
            raise MaterializationError(
                f"Missing raw-token-cache authority for archive {archive.ordinal}"
            )
        entry = self.raw_token_cache_inventory.entries[archive.ordinal]
        cache_directory = self.raw_token_cache_root / entry.cache_directory
        guarded_paths = (
            archive.raw_path,
            archive.report_path,
            archive.fingerprint_path,
            archive.decision_path,
            cache_directory / RAW_TOKEN_CACHE_MANIFEST_FILE,
            cache_directory / RAW_TOKEN_CACHE_SIDECAR_FILE,
            cache_directory / RAW_TOKEN_CACHE_TOKEN_FILE,
            cache_directory / RAW_TOKEN_CACHE_OFFSET_FILE,
            self.tokenizer_root / "TOKENIZER_MANIFEST.json",
            *(
                self.tokenizer_root / filename
                for filename in sorted(self.tokenizer_manifest["files"])
            ),
        )
        guarded_states = {
            path: _regular_file_state(path, "raw-token-cache bound source")
            for path in guarded_paths
        }

        eligible = {
            key
            for key, state in self.destinations.items()
            if state.cursor["next_archive"] == archive.ordinal
        }
        if not eligible:
            raise MaterializationError("No lagging writer can consume the next archive")
        if any(
            state.cursor["next_archive"] not in (archive.ordinal, archive.ordinal + 1)
            for state in self.destinations.values()
        ):
            raise MaterializationError("Writer cursor vector is not recoverable")
        deltas = {
            key: {
                "selected_documents": 0,
                "selected_content_tokens": 0,
                "terminal_prefix_documents": 0,
            }
            for key in eligible
        }
        logical_positions = {
            key: (
                self.destinations[key].cursor["selected_content_tokens"]
                + self.destinations[key].cursor["selected_documents"]
            )
            for key in eligible
        }
        index_writers: dict[tuple[str, str], _DocumentIndexShardWriter] = {}
        index_descriptors: dict[tuple[str, str], dict[str, Any] | None] = {}
        documents = 0
        clean_bytes = 0
        content_tokens = 0

        try:
            # Reader.open completes every cache/source/tokenizer/payload and
            # manifest-index alignment check before a writer receives tokens.
            try:
                reader_context = RawTokenCacheReader.open(
                    cache_directory,
                    authority,
                    dataset_root=self.raw_root,
                    preprocess_root=self.preprocess_root,
                    tokenizer_root=self.tokenizer_root,
                )
            except RawTokenCacheReadError as error:
                raise MaterializationError(
                    f"Raw-token-cache authentication failed for {archive.archive}: {error}"
                ) from error
            with reader_context as reader:
                try:
                    for index, decision in enumerate(
                        self._iter_all_eligible_decisions(archive, bitmap)
                    ):
                        if decision["manifest_index"] != index:
                            raise MaterializationError(
                                "Raw-token-cache join left manifest-index order"
                            )
                        source_tokens = int(decision["source_tokens"])
                        documents += 1
                        clean_bytes += int(decision["_fingerprint_size_bytes"])
                        content_tokens += source_tokens
                        destination = (
                            (str(decision["split"]), archive.domain)
                            if decision["decision"] == "keep"
                            else None
                        )
                        if destination is None or destination not in eligible:
                            continue
                        span = reader.document(index)
                        selected = int(decision["selected_tokens"])
                        if (
                            len(span) != source_tokens
                            or selected != source_tokens
                            or decision["token_prefix"] != [0, source_tokens]
                            or decision["terminal_quota_prefix"] is not False
                        ):
                            raise MaterializationError(
                                "Cached v7 document is not an exact full-document join: "
                                f"{archive.archive}:{index}"
                            )
                        start = logical_positions[destination]
                        logical_positions[destination] += selected + 1
                        self.destinations[destination].writer.add_document(
                            span.to_list()
                        )
                        language = decision["provenance"].get("language")
                        if not isinstance(language, str) or not language.strip():
                            raise MaterializationError(
                                f"Selected document has no language: {archive.archive}:"
                                f"{decision['member_path']}"
                            )
                        index_writer = index_writers.get(destination)
                        if index_writer is None:
                            index_writer = _DocumentIndexShardWriter(
                                self.output_root,
                                archive_ordinal=archive.ordinal,
                                split=destination[0],
                                domain=destination[1],
                            )
                            index_writers[destination] = index_writer
                        index_writer.write(
                            {
                                "record_version": DOCUMENT_INDEX_VERSION,
                                "doc_id": decision["doc_id"],
                                "canonical_doc_id": decision["canonical_doc_id"],
                                "split_group_id": decision["split_group_id"],
                                "split": destination[0],
                                "domain": destination[1],
                                "bucket": archive.bucket,
                                "language": language,
                                "source_archive": archive.archive,
                                "source_archive_ordinal": archive.ordinal,
                                "source_manifest_index": index,
                                "source_member": decision["member_path"],
                                "source_tokens": source_tokens,
                                "selected_content_tokens": selected,
                                "terminal_quota_prefix": False,
                                "logical_stream_start": start,
                                "logical_content_end_exclusive": start + selected,
                                "logical_eos_position": start + selected,
                            }
                        )
                        delta = deltas[destination]
                        delta["selected_documents"] += 1
                        delta["selected_content_tokens"] += selected
                        self._fault(
                            "document_added",
                            archive=archive.ordinal,
                            split=destination[0],
                            domain=destination[1],
                            document=index,
                        )
                    if (
                        documents != archive.documents
                        or clean_bytes != archive.clean_bytes
                        or content_tokens != archive.content_tokens
                    ):
                        raise MaterializationError(
                            "Raw-token-cache/fingerprint aggregate totals mismatch: "
                            f"{archive.archive}"
                        )
                    reader.verify_unchanged()
                except RawTokenCacheReadError as error:
                    raise MaterializationError(
                        f"Raw-token-cache read failed for {archive.archive}: {error}"
                    ) from error

            # Recheck every bound path identity immediately before the archive
            # transaction is committed. The reader has already fully hashed
            # the raw archive and both mmap payloads for this generation; the
            # stat guard also detects rename/replace after those reads.
            if (
                any(
                    _regular_file_state(path, "raw-token-cache bound source")
                    != before
                    for path, before in guarded_states.items()
                )
                or file_sha256(archive.report_path) != archive.report_sha256
                or file_sha256(archive.fingerprint_path)
                != archive.fingerprint_sha256
                or file_sha256(archive.decision_path) != archive.decision_sha256
                or file_sha256(cache_directory / RAW_TOKEN_CACHE_MANIFEST_FILE)
                != entry.cache_manifest.sha256
                or file_sha256(cache_directory / RAW_TOKEN_CACHE_SIDECAR_FILE)
                != entry.cache_sidecar.sha256
            ):
                raise MaterializationError(
                    f"Raw-token-cache/source authority changed before checkpoint: {archive.archive}"
                )
            for key in sorted(eligible):
                index_writer = index_writers.get(key)
                if index_writer is None:
                    unexpected = (
                        self.output_root
                        / "provenance"
                        / "documents"
                        / key[0]
                        / key[1]
                        / f"archive-{archive.ordinal:06d}.jsonl.zst"
                    )
                    if unexpected.exists():
                        raise MaterializationError(
                            f"Unexpected document-index shard on replay: {unexpected}"
                        )
                    index_descriptors[key] = None
                else:
                    index_descriptors[key] = index_writer.finish(
                        archive_ordinal=archive.ordinal
                    )
        except BaseException:
            for writer in index_writers.values():
                writer.abort()
            raise

        for key in sorted(eligible):
            state = self.destinations[key]
            delta = deltas[key]
            cursor = dict(state.cursor)
            cursor["next_archive"] = archive.ordinal + 1
            for counter, value in delta.items():
                cursor[counter] += value
            cursor["document_index_shards"] = list(
                cursor["document_index_shards"]
            )
            descriptor = index_descriptors[key]
            if descriptor is not None:
                cursor["document_index_shards"].append(descriptor)
            state.writer.checkpoint(cursor)
            state.cursor = cursor
            self._fault(
                "writer_checkpoint",
                archive=archive.ordinal,
                split=state.split,
                domain=state.domain,
            )
        self._write_journal("packing")
        self._fault("archive_checkpoint", archive=archive.ordinal)
        return {
            "archives": 1,
            "documents": documents,
            "source_content_tokens": content_tokens,
            "selected_documents": sum(
                delta["selected_documents"] for delta in deltas.values()
            ),
            "selected_content_tokens": sum(
                delta["selected_content_tokens"] for delta in deltas.values()
            ),
            "raw_compressed_bytes": archive.raw_path.stat().st_size,
        }

    def _process_archive(self, archive: ArchiveSpec) -> dict[str, int]:
        if file_sha256(archive.report_path) != archive.report_sha256:
            raise MaterializationError(f"Preprocess report changed: {archive.report_path}")
        if file_sha256(archive.fingerprint_path) != archive.fingerprint_sha256:
            raise MaterializationError(f"Fingerprint changed: {archive.fingerprint_path}")
        if file_sha256(archive.decision_path) != archive.decision_sha256:
            raise MaterializationError(f"Decision shard changed: {archive.decision_path}")
        bitmap = (
            self._load_all_eligible_bitmap(archive)
            if archive.decision_format == ALL_ELIGIBLE_BITMAP_FORMAT
            else None
        )
        if self.raw_token_cache_inventory is not None:
            if bitmap is None:
                raise MaterializationError(
                    "Raw-token-cache adapter cannot consume legacy decision shards"
                )
            return self._process_archive_from_raw_token_cache(archive, bitmap)

        def decision_rows() -> Iterator[dict[str, Any]]:
            if bitmap is None:
                return iter_jsonl_zst(archive.decision_path)
            return self._iter_all_eligible_decisions(archive, bitmap)

        eligible = {
            key
            for key, state in self.destinations.items()
            if state.cursor["next_archive"] == archive.ordinal
        }
        if not eligible:
            raise MaterializationError("No lagging writer can consume the next archive")
        if any(
            state.cursor["next_archive"] not in (archive.ordinal, archive.ordinal + 1)
            for state in self.destinations.values()
        ):
            raise MaterializationError("Writer cursor vector is not recoverable")
        deltas = {
            key: {
                "selected_documents": 0,
                "selected_content_tokens": 0,
                "terminal_prefix_documents": 0,
            }
            for key in eligible
        }
        logical_positions = {
            key: (
                self.destinations[key].cursor["selected_content_tokens"]
                + self.destinations[key].cursor["selected_documents"]
            )
            for key in eligible
        }
        # Open provenance compressors lazily. An archive normally contributes
        # to only one domain, so this avoids six empty network-volume files and
        # zstd contexts per archive without changing canonical output bytes.
        index_writers: dict[tuple[str, str], _DocumentIndexShardWriter] = {}
        pending: list[dict[str, Any]] = []
        pending_bytes = 0

        def flush_pending() -> None:
            nonlocal pending_bytes
            if not pending:
                return
            encodings = self.tokenizer.encode_batch(
                [item["text"] for item in pending], add_special_tokens=False
            )
            if len(encodings) != len(pending):
                raise MaterializationError("Tokenizer batch returned the wrong row count")
            for item, encoding in zip(pending, encodings, strict=True):
                decision = item["decision"]
                token_ids = encoding.ids
                if len(token_ids) != decision["source_tokens"]:
                    raise MaterializationError(
                        f"Pinned tokenizer count mismatch: {archive.archive}:"
                        f"{decision['member_path']} ({len(token_ids)} != "
                        f"{decision['source_tokens']})"
                    )
                selected = int(decision["selected_tokens"])
                destination = item["destination"]
                self.destinations[destination].writer.add_document(
                    token_ids if selected == len(token_ids) else token_ids[:selected]
                )
                start = int(item["logical_stream_start"])
                language = decision["provenance"].get("language")
                if not isinstance(language, str) or not language.strip():
                    raise MaterializationError(
                        f"Selected document has no language: {archive.archive}:"
                        f"{decision['member_path']}"
                    )
                index_writer = index_writers.get(destination)
                if index_writer is None:
                    index_writer = _DocumentIndexShardWriter(
                        self.output_root,
                        archive_ordinal=archive.ordinal,
                        split=destination[0],
                        domain=destination[1],
                    )
                    index_writers[destination] = index_writer
                index_writer.write(
                    {
                        "record_version": DOCUMENT_INDEX_VERSION,
                        "doc_id": decision["doc_id"],
                        "canonical_doc_id": decision["canonical_doc_id"],
                        "split_group_id": decision["split_group_id"],
                        "split": destination[0],
                        "domain": destination[1],
                        "bucket": archive.bucket,
                        "language": language,
                        "source_archive": archive.archive,
                        "source_archive_ordinal": archive.ordinal,
                        "source_manifest_index": decision["manifest_index"],
                        "source_member": decision["member_path"],
                        "source_tokens": decision["source_tokens"],
                        "selected_content_tokens": selected,
                        "terminal_quota_prefix": decision[
                            "terminal_quota_prefix"
                        ],
                        "logical_stream_start": start,
                        "logical_content_end_exclusive": start + selected,
                        "logical_eos_position": start + selected,
                    }
                )
                delta = deltas[destination]
                delta["selected_documents"] += 1
                delta["selected_content_tokens"] += selected
                delta["terminal_prefix_documents"] += int(
                    selected < len(token_ids)
                )
                self._fault(
                    "document_added",
                    archive=archive.ordinal,
                    split=destination[0],
                    domain=destination[1],
                    document=item["document_index"],
                )
            pending.clear()
            pending_bytes = 0

        raw_digest = hashlib.sha256()
        documents = 0
        clean_bytes = 0
        content_tokens = 0
        manifest_seen = False
        member_sizes: list[int] = []
        decisions = iter(decision_rows())

        index_descriptors: dict[tuple[str, str], dict[str, Any] | None] = {}
        try:
            with archive.raw_path.open("rb") as raw_handle:
                hashing_reader = _HashingReader(raw_handle, raw_digest)
                decompressor = zstandard.ZstdDecompressor().stream_reader(
                    hashing_reader, read_across_frames=True, closefd=False
                )
                try:
                    with tarfile.open(fileobj=decompressor, mode="r|") as tar:
                        for member in tar:
                            if not member.isfile():
                                raise MaterializationError(
                                    f"Unexpected non-file tar member {member.name!r}"
                                )
                            extracted = tar.extractfile(member)
                            if extracted is None:
                                raise MaterializationError(
                                    f"Could not read tar member {member.name!r}"
                                )
                            if member.name == "_manifest.jsonl":
                                flush_pending()
                                if manifest_seen:
                                    raise MaterializationError(
                                        "Raw archive has multiple manifests"
                                    )
                                manifest_seen = True
                                try:
                                    next(decisions)
                                except StopIteration:
                                    pass
                                else:
                                    raise MaterializationError(
                                        "Raw archive has fewer documents than decisions"
                                    )
                                second_decisions = iter(decision_rows())
                                for index, raw_line in enumerate(extracted):
                                    try:
                                        line = raw_line.decode(
                                            "utf-8", errors="strict"
                                        )
                                    except UnicodeDecodeError as error:
                                        raise MaterializationError(
                                            "Raw internal manifest is not UTF-8"
                                        ) from error
                                    if not line.strip():
                                        raise MaterializationError(
                                            "Raw internal manifest contains a blank row"
                                        )
                                    raw_row = json.loads(line)
                                    if not isinstance(raw_row, dict):
                                        raise MaterializationError(
                                            "Raw internal manifest row is not an object"
                                        )
                                    try:
                                        decision = next(second_decisions)
                                    except StopIteration as error:
                                        raise MaterializationError(
                                            "Raw internal manifest has too many rows"
                                        ) from error
                                    self._compare_raw_manifest_row(
                                        raw_row, decision, archive, index
                                    )
                                    if (
                                        index >= len(member_sizes)
                                        or raw_row.get("size_bytes")
                                        != member_sizes[index]
                                    ):
                                        raise MaterializationError(
                                            f"Raw manifest byte count mismatch in "
                                            f"{archive.raw_path} row {index}"
                                        )
                                try:
                                    next(second_decisions)
                                except StopIteration:
                                    pass
                                else:
                                    raise MaterializationError(
                                        "Raw internal manifest has too few rows"
                                    )
                                continue
                            if manifest_seen:
                                raise MaterializationError(
                                    "Raw document appears after the internal manifest"
                                )
                            try:
                                decision = next(decisions)
                            except StopIteration as error:
                                raise MaterializationError(
                                    "Raw archive has more documents than decisions"
                                ) from error
                            if bitmap is None:
                                self._validate_decision(
                                    decision, archive, documents
                                )
                            if member.name != decision["member_path"]:
                                raise MaterializationError(
                                    f"Raw/decision member mismatch at document {documents}"
                                )
                            destination = (
                                (str(decision["split"]), archive.domain)
                                if decision["decision"] == "keep"
                                else None
                            )
                            needs_tokenization = (
                                destination is not None and destination in eligible
                            )
                            if needs_tokenization:
                                if member.size > self.tokenizer_max_document_bytes:
                                    raise MaterializationError(
                                        "Selected document exceeds tokenizer max-document "
                                        f"byte limit: {archive.archive}:{member.name} "
                                        f"({member.size} > "
                                        f"{self.tokenizer_max_document_bytes})"
                                    )
                                # Flush before reading/decoding the next selected
                                # member. Thus the configured byte cap is a hard
                                # per-batch bound rather than a post-add hint.
                                if pending and (
                                    len(pending) >= self.tokenizer_batch_documents
                                    or pending_bytes + member.size
                                    > self.tokenizer_batch_bytes
                                ):
                                    flush_pending()
                                content = extracted.read()
                                observed_size = len(content)
                                content_sha = hashlib.sha256(content).hexdigest()
                            else:
                                # Rejected documents and replayed archives still
                                # receive full integrity validation, but do not
                                # need their entire payload resident in Python.
                                content_digest = hashlib.sha256()
                                observed_size = 0
                                while chunk := extracted.read(8 * 1024 * 1024):
                                    observed_size += len(chunk)
                                    content_digest.update(chunk)
                                content_sha = content_digest.hexdigest()
                                content = None
                            if observed_size != member.size:
                                raise MaterializationError(
                                    f"Short raw member {member.name}: "
                                    f"{observed_size} != {member.size}"
                                )
                            if (
                                bitmap is not None
                                and decision.get("_fingerprint_size_bytes")
                                != member.size
                            ):
                                raise MaterializationError(
                                    "Fingerprint/raw member byte count mismatch: "
                                    f"{archive.archive}:{member.name}"
                                )
                            if content_sha != decision["content_sha256"]:
                                raise MaterializationError(
                                    f"Raw content hash mismatch: "
                                    f"{archive.archive}:{member.name}"
                                )
                            documents += 1
                            clean_bytes += observed_size
                            member_sizes.append(observed_size)
                            content_tokens += int(decision["source_tokens"])
                            if needs_tokenization:
                                assert content is not None and destination is not None
                                try:
                                    text_content = content.decode(
                                        "utf-8", errors="strict"
                                    )
                                except UnicodeDecodeError as error:
                                    raise MaterializationError(
                                        f"Selected document is not UTF-8: "
                                        f"{member.name}"
                                    ) from error
                                selected = int(decision["selected_tokens"])
                                start = logical_positions[destination]
                                logical_positions[destination] += selected + 1
                                pending.append(
                                    {
                                        "text": text_content,
                                        "decision": decision,
                                        "destination": destination,
                                        "logical_stream_start": start,
                                        "document_index": documents - 1,
                                    }
                                )
                                pending_bytes += observed_size
                                del content
                                if (
                                    len(pending) >= self.tokenizer_batch_documents
                                    or pending_bytes >= self.tokenizer_batch_bytes
                                ):
                                    flush_pending()
                    flush_pending()
                    while decompressor.read(8 * 1024 * 1024):
                        pass
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    zstandard.ZstdError,
                    tarfile.TarError,
                ) as error:
                    raise MaterializationError(
                        f"Invalid compressed/raw archive content in {archive.raw_path}"
                    ) from error
                finally:
                    decompressor.close()
                while hashing_reader.read(8 * 1024 * 1024):
                    pass
            if not manifest_seen:
                raise MaterializationError(
                    f"Raw archive has no internal manifest: {archive.raw_path}"
                )
            report = _load_json(archive.report_path)
            if (
                raw_digest.hexdigest() != archive.raw_sha256
                or documents != archive.documents
                or clean_bytes != report.get("clean_bytes")
                or content_tokens != archive.content_tokens
            ):
                raise MaterializationError(
                    f"Raw archive identity/totals mismatch: {archive.raw_path}"
                )
            for key in sorted(eligible):
                index_writer = index_writers.get(key)
                if index_writer is None:
                    unexpected = (
                        self.output_root
                        / "provenance"
                        / "documents"
                        / key[0]
                        / key[1]
                        / f"archive-{archive.ordinal:06d}.jsonl.zst"
                    )
                    if unexpected.exists():
                        raise MaterializationError(
                            f"Unexpected document-index shard on replay: {unexpected}"
                        )
                    index_descriptors[key] = None
                else:
                    index_descriptors[key] = index_writer.finish(
                        archive_ordinal=archive.ordinal
                    )
        except BaseException:
            for writer in index_writers.values():
                writer.abort()
            raise

        for key in sorted(eligible):
            state = self.destinations[key]
            delta = deltas[key]
            cursor = dict(state.cursor)
            cursor["next_archive"] = archive.ordinal + 1
            for counter, value in delta.items():
                cursor[counter] += value
            cursor["document_index_shards"] = list(
                cursor["document_index_shards"]
            )
            descriptor = index_descriptors[key]
            if descriptor is not None:
                cursor["document_index_shards"].append(descriptor)
            state.writer.checkpoint(cursor)
            state.cursor = cursor
            self._fault(
                "writer_checkpoint",
                archive=archive.ordinal,
                split=state.split,
                domain=state.domain,
            )
        self._write_journal("packing")
        self._fault("archive_checkpoint", archive=archive.ordinal)
        return {
            "archives": 1,
            "documents": documents,
            "source_content_tokens": content_tokens,
            "selected_documents": sum(
                delta["selected_documents"] for delta in deltas.values()
            ),
            "selected_content_tokens": sum(
                delta["selected_content_tokens"] for delta in deltas.values()
            ),
            "raw_compressed_bytes": archive.raw_path.stat().st_size,
        }

    def _validate_selected_totals(self) -> None:
        if (
            self.selection_manifest["identity"]["format_version"]
            == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        ):
            expected = self._validate_all_eligible_totals(self.selection_manifest)
            if set(expected) != set(self.destinations):
                raise MaterializationError(
                    "All-eligible totals do not cover every split/domain"
                )
            for key, state in self.destinations.items():
                observed = {
                    counter: state.cursor[counter]
                    for counter in (
                        "selected_documents",
                        "selected_content_tokens",
                        "terminal_prefix_documents",
                    )
                }
                if observed != expected[key]:
                    raise MaterializationError(
                        f"Materialized all-eligible total mismatch for {key}: "
                        f"{observed} != {expected[key]}"
                    )
            return
        expected: dict[tuple[str, str], dict[str, int]] = {}
        quotas = self.selection_manifest.get("quotas")
        if not isinstance(quotas, list):
            raise MaterializationError("Selection manifest has no quota summary")
        for row in quotas:
            if not isinstance(row, dict):
                raise MaterializationError("Invalid selection quota row")
            key = (str(row.get("split")), str(row.get("category")))
            if key in expected or key[0] not in SPLITS or key[1] not in DOMAIN_ORDER:
                raise MaterializationError(f"Invalid/duplicate quota summary: {key}")
            if row.get("unit") != "pre_packing_starcoder2_content_tokens":
                raise MaterializationError("Selection quota uses an unsafe token unit")
            expected[key] = {
                "selected_documents": _require_nonnegative_int(
                    row.get("documents"), "quota documents"
                ),
                "selected_content_tokens": _require_nonnegative_int(
                    row.get("selected_tokens"), "quota selected_tokens"
                ),
                "terminal_prefix_documents": _require_nonnegative_int(
                    row.get("terminal_prefix_documents"), "quota terminal prefixes"
                ),
            }
            if row.get("target_tokens") != row.get("selected_tokens"):
                raise MaterializationError("Curation quota was not filled exactly")
        if set(expected) != set(self.destinations):
            raise MaterializationError("Quota summary does not cover every split/domain")
        for key, state in self.destinations.items():
            observed = {
                counter: state.cursor[counter]
                for counter in (
                    "selected_documents",
                    "selected_content_tokens",
                    "terminal_prefix_documents",
                )
            }
            if observed != expected[key]:
                raise MaterializationError(
                    f"Materialized selection total mismatch for {key}: "
                    f"{observed} != {expected[key]}"
                )

    def _finish_packed_outputs(self) -> dict[tuple[str, str], dict[str, Any]]:
        end_archive = len(self.archives)
        if any(
            state.cursor["next_archive"] != end_archive
            for state in self.destinations.values()
        ):
            raise MaterializationError("Cannot finish before every archive is committed")
        self._validate_selected_totals()
        manifests: dict[tuple[str, str], dict[str, Any]] = {}
        for key in sorted(self.destinations):
            state = self.destinations[key]
            manifest = state.writer.finish(source_cursor=state.cursor)
            if (
                manifest.get("documents") != state.cursor["selected_documents"]
                or manifest.get("source_content_tokens")
                != state.cursor["selected_content_tokens"]
            ):
                raise MaterializationError(f"Packed manifest counters differ for {key}")
            manifests[key] = manifest
        return manifests

    def _validate_document_index_manifest(
        self,
        path: Path,
        *,
        split: str,
        domain: str,
        packed_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        manifest = _load_json(path)
        expected_identity = {
            "format": DOCUMENT_INDEX_FORMAT,
            "format_version": DOCUMENT_INDEX_VERSION,
            "split": split,
            "domain": domain,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "tokenizer_manifest_sha256": self.identity[
                "tokenizer_manifest_sha256"
            ],
            "sequence_length": self.config.sequence_length,
        }
        for key, value in expected_identity.items():
            if manifest.get(key) != value:
                raise MaterializationError(
                    f"Document-index {key} mismatch for {split}/{domain}"
                )
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise MaterializationError("Document-index manifest has no shard inventory")
        logical_position = 0
        records = 0
        selected_tokens = 0
        last_ordinal = -1
        for descriptor in shards:
            if not isinstance(descriptor, dict):
                raise MaterializationError("Invalid document-index shard descriptor")
            ordinal = _require_nonnegative_int(
                descriptor.get("archive_ordinal"), "document-index archive ordinal"
            )
            if ordinal <= last_ordinal or ordinal >= len(self.archives):
                raise MaterializationError("Document-index shard order is invalid")
            last_ordinal = ordinal
            archive = self.archives[ordinal]
            shard_path = _safe_file(
                self.output_root,
                descriptor.get("path"),
                "document-index shard",
            )
            expected_path = (
                Path("provenance")
                / "documents"
                / split
                / domain
                / f"archive-{ordinal:06d}.jsonl.zst"
            ).as_posix()
            if descriptor.get("path") != expected_path:
                raise MaterializationError("Document-index shard path is not canonical")
            if (
                descriptor.get("bytes") != shard_path.stat().st_size
                or file_sha256(shard_path) != descriptor.get("sha256")
            ):
                raise MaterializationError(
                    f"Document-index shard checksum mismatch: {shard_path}"
                )
            shard_records = 0
            for row in iter_jsonl_zst(shard_path):
                if (
                    row.get("record_version") != DOCUMENT_INDEX_VERSION
                    or row.get("split") != split
                    or row.get("domain") != domain
                    or row.get("bucket") != archive.bucket
                    or row.get("source_archive") != archive.archive
                    or row.get("source_archive_ordinal") != ordinal
                ):
                    raise MaterializationError(
                        f"Document-index source identity mismatch: {shard_path}"
                    )
                for field in ("doc_id", "canonical_doc_id", "split_group_id"):
                    _require_sha256(row.get(field), f"document_index.{field}")
                language = row.get("language")
                member = row.get("source_member")
                source_index = row.get("source_manifest_index")
                source_tokens = row.get("source_tokens")
                selected = row.get("selected_content_tokens")
                if (
                    not isinstance(language, str)
                    or not language.strip()
                    or not isinstance(member, str)
                    or not member
                    or not isinstance(source_index, int)
                    or isinstance(source_index, bool)
                    or source_index < 0
                    or not isinstance(source_tokens, int)
                    or isinstance(source_tokens, bool)
                    or not isinstance(selected, int)
                    or isinstance(selected, bool)
                    or not 1 <= selected <= source_tokens
                ):
                    raise MaterializationError(
                        f"Document-index record is invalid: {shard_path}"
                    )
                if (
                    row.get("logical_stream_start") != logical_position
                    or row.get("logical_content_end_exclusive")
                    != logical_position + selected
                    or row.get("logical_eos_position")
                    != logical_position + selected
                    or row.get("terminal_quota_prefix")
                    != (selected < source_tokens)
                ):
                    raise MaterializationError(
                        f"Document-index logical positions are discontinuous: {shard_path}"
                    )
                logical_position += selected + 1
                selected_tokens += selected
                records += 1
                shard_records += 1
            if descriptor.get("records") != shard_records or shard_records < 1:
                raise MaterializationError(
                    f"Document-index shard record count mismatch: {shard_path}"
                )
        expected_totals = {
            "documents": records,
            "selected_content_tokens": selected_tokens,
            "logical_stream_tokens": logical_position,
        }
        for field, observed in expected_totals.items():
            if manifest.get(field) != observed:
                raise MaterializationError(
                    f"Document-index {field} total is inconsistent"
                )
        if (
            records != packed_manifest.get("documents")
            or selected_tokens != packed_manifest.get("source_content_tokens")
            or logical_position != packed_manifest.get("stream_tokens")
        ):
            raise MaterializationError(
                f"Document-index totals differ from packed data for {split}/{domain}"
            )
        return manifest

    def _finish_document_indexes(
        self,
        packed: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for key in sorted(self.destinations):
            split, domain = key
            cursor = self.destinations[key].cursor
            payload = {
                "format": DOCUMENT_INDEX_FORMAT,
                "format_version": DOCUMENT_INDEX_VERSION,
                "split": split,
                "domain": domain,
                "selection_manifest_sha256": self.selection_manifest_sha256,
                "tokenizer_manifest_sha256": self.identity[
                    "tokenizer_manifest_sha256"
                ],
                "sequence_length": self.config.sequence_length,
                "documents": cursor["selected_documents"],
                "selected_content_tokens": cursor["selected_content_tokens"],
                "logical_stream_tokens": (
                    cursor["selected_content_tokens"]
                    + cursor["selected_documents"]
                ),
                "shards": cursor["document_index_shards"],
            }
            path = (
                self.output_root
                / "provenance"
                / "documents"
                / split
                / domain
                / "manifest.json"
            )
            if path.exists():
                if _load_json(path) != payload:
                    raise MaterializationError(
                        f"Completed document-index identity mismatch: {path}"
                    )
            else:
                atomic_json(path, payload)
            result[key] = self._validate_document_index_manifest(
                path,
                split=split,
                domain=domain,
                packed_manifest=packed[key],
            )
        return result

    def _validate_order_identity(self, split: str, manifest: dict[str, Any]) -> None:
        expected_seed = self.config.order_seed + SPLITS.index(split)
        if (
            manifest.get("format_version") != 4
            or manifest.get("split") != split
            or manifest.get("seed") != expected_seed
        ):
            raise MaterializationError(f"Order identity mismatch for {split}")
        consumption = manifest.get("training_consumption")
        if not isinstance(consumption, dict):
            raise MaterializationError(f"Order has no consumption contract for {split}")
        if split == "train":
            expected_geometry = (
                self.config.frozen_global_microbatch_rows,
                self.config.frozen_gradient_accumulation_steps,
            )
        else:
            expected_geometry = (None, None)
        found_geometry = (
            consumption.get("frozen_global_microbatch_rows"),
            consumption.get("frozen_gradient_accumulation_steps"),
        )
        if found_geometry != expected_geometry:
            raise MaterializationError(f"Order geometry mismatch for {split}")
        target_by_split = {
            "train": self.config.expected_train_input_tokens,
            "validation": self.config.expected_validation_input_tokens,
            "test": self.config.expected_test_input_tokens,
        }
        budget = manifest.get("input_token_budget")
        if (
            not isinstance(budget, dict)
            or budget.get("expected_total") != target_by_split[split]
        ):
            raise MaterializationError(f"Order input-token target mismatch for {split}")
        expected_weights = (
            {"python": 0.4, "other_code": 0.4, "english": 0.2}
            if self.config.enforce_input_weights
            else None
        )
        if manifest.get("expected_input_token_weights") != expected_weights:
            raise MaterializationError(f"Order domain weights mismatch for {split}")

    def _build_orders(self) -> dict[str, dict[str, Any]]:
        orders_root = self.output_root / "orders"
        orders_root.mkdir(parents=True, exist_ok=True)
        result: dict[str, dict[str, Any]] = {}
        expected_weights = (
            {"python": 0.4, "other_code": 0.4, "english": 0.2}
            if self.config.enforce_input_weights
            else None
        )
        for offset, split in enumerate(SPLITS):
            final = orders_root / split
            staging = orders_root / f".{split}.part"
            if final.exists():
                if staging.exists():
                    raise MaterializationError(
                        f"Both final and staging orders exist for {split}"
                    )
                manifest = validate_training_order(final / "manifest.json")
                self._validate_order_identity(split, manifest)
                result[split] = manifest
                continue
            if staging.exists():
                if staging.is_symlink() or not staging.is_dir():
                    raise MaterializationError(f"Unsafe order staging path: {staging}")
                shutil.rmtree(staging)
            manifests = {
                domain: self.output_root / "packed" / split / domain / "manifest.json"
                for domain in DOMAIN_ORDER
            }
            target_by_split = {
                "train": self.config.expected_train_input_tokens,
                "validation": self.config.expected_validation_input_tokens,
                "test": self.config.expected_test_input_tokens,
            }
            tolerance_by_split = {
                "train": self.config.train_input_token_tolerance,
                "validation": self.config.validation_input_token_tolerance,
                "test": self.config.test_input_token_tolerance,
            }
            kwargs: dict[str, Any] = {
                "expected_total_input_tokens": target_by_split[split],
                "input_token_tolerance": tolerance_by_split[split],
            }
            if split == "train":
                kwargs.update({
                    "frozen_global_microbatch_rows": (
                        self.config.frozen_global_microbatch_rows
                    ),
                    "frozen_gradient_accumulation_steps": (
                        self.config.frozen_gradient_accumulation_steps
                    ),
                })
            manifest = build_training_order(
                manifests,
                staging,
                seed=self.config.order_seed + offset,
                expected_weights=expected_weights,
                **kwargs,
            )
            self._validate_order_identity(split, manifest)
            validate_training_order(staging / "manifest.json")
            os.replace(staging, final)
            _fsync_directory(orders_root)
            result[split] = manifest
            self._write_journal("orders")
            self._fault("order_published", split=split)
        return result

    def _provenance_payloads(self) -> dict[str, dict[str, Any]]:
        identity = self.selection_manifest["identity"]
        payloads = {
            "source.json": {
                "format_version": 1,
                "selection_manifest_sha256": self.selection_manifest_sha256,
                "curation_identity_format_version": identity["format_version"],
                "curation_profile": identity.get("curation_profile"),
                "known_provenance_limitations": self.selection_manifest.get(
                    "known_provenance_limitations", []
                ),
                "raw_archives_hashed_for_integrity": self.selection_manifest[
                    "raw_archives_hashed_for_integrity"
                ],
                "raw_archive_payloads_parsed_by_curation": self.selection_manifest[
                    "raw_archive_payloads_parsed_by_curation"
                ],
                "curation_sqlite_runtime": identity["sqlite_runtime"],
                "curation_storage_contract": identity[
                    "curation_storage_contract"
                ],
                "collection_completeness_sha256": canonical_sha256(
                    self.selection_manifest["collection_completeness"]
                ),
                "collection_completeness": self.selection_manifest[
                    "collection_completeness"
                ],
                "source_manifests": identity["source_manifests"],
                "raw_archives": [
                    {
                        "archive": item.archive,
                        "sha256": item.raw_sha256,
                        "documents": item.documents,
                        "content_tokens": item.content_tokens,
                    }
                    for item in self.archives
                ],
            },
            "policy.json": {
                "format_version": 1,
                "curation_policy_sha256": identity["policy_sha256"],
                "quota_config_sha256": identity["quota_config_sha256"],
                "benchmark_guard_sha256": identity["benchmark_guard_sha256"],
                "selection_policy": self.selection_manifest["selection_policy"],
                "leakage_audit": self.selection_manifest["leakage_audit"],
                "curation_profile": identity.get("curation_profile"),
            },
            "tokenizer.json": {
                "format_version": 1,
                "tokenizer_manifest_sha256": identity["tokenizer_manifest_sha256"],
                "resolved_revision": self.tokenizer_manifest["resolved_revision"],
                "validation": self.tokenizer_manifest["validation"],
                "files": self.tokenizer_manifest["files"],
            },
            "fingerprints.json": {
                "format_version": 1,
                "preprocess_manifest_sha256": identity["preprocess_manifest_sha256"],
                "report_inventory_sha256": identity["report_inventory_sha256"],
                "english_near_clusters_sha256": identity[
                    "english_near_clusters_sha256"
                ],
                "english_near_artifact": identity["english_near_artifact"],
                "english_near_dedup_complete": self.selection_manifest[
                    "english_near_dedup_complete"
                ],
                "english_near_dedup_status": self.selection_manifest.get(
                    "english_near_dedup_status"
                ),
                "curation_profile": identity.get("curation_profile"),
                "decision_inventory_sha256": self.selection_manifest[
                    "decision_inventory_sha256"
                ],
                "archives": [
                    {
                        "archive": item.archive,
                        "report": item.report_relative,
                        "report_sha256": item.report_sha256,
                        "fingerprint": item.fingerprint_relative,
                        "fingerprint_sha256": item.fingerprint_sha256,
                        "decision": item.decision_relative,
                        "decision_sha256": item.decision_sha256,
                        **(
                            {
                                "decision_format": item.decision_format,
                                "decision_format_version": (
                                    item.decision_format_version
                                ),
                                "decision_bytes": item.decision_bytes,
                                "decision_kept_documents": (
                                    item.decision_kept_documents
                                ),
                            }
                            if item.decision_format is not None
                            else {}
                        ),
                    }
                    for item in self.archives
                ],
            },
        }
        if identity["format_version"] == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION:
            payloads["source.json"].update(
                {
                    "publication_scope": self.selection_manifest[
                        "publication_scope"
                    ],
                    "raw_archive_integrity_policy": identity[
                        "raw_archive_integrity_policy"
                    ],
                    "curation_sqlite_execution": identity["sqlite_execution"],
                    "fast_all_eligible_handoff": identity[
                        "fast_all_eligible_handoff"
                    ],
                    "source_curation": identity["source_curation"],
                }
            )
            payloads["policy.json"].update(
                {
                    "selection_strategy": self.selection_manifest[
                        "selection_strategy"
                    ],
                    "selection_profile": self.selection_manifest[
                        "selection_profile"
                    ],
                    "training_input_budget_authority": self.selection_manifest[
                        "training_input_budget_authority"
                    ],
                    "selected_totals": self.selection_manifest["selected_totals"],
                    "reference_quotas": self.selection_manifest[
                        "reference_quotas"
                    ],
                }
            )
        if self.raw_token_cache_inventory is not None:
            payloads["raw_token_cache.json"] = {
                "format_version": 1,
                "acceleration_only": True,
                "training_ready": False,
                "inventory": (
                    self.raw_token_cache_inventory.provenance_descriptor()
                ),
                "cache_contract": self.raw_token_cache_inventory.manifest["cache"],
                "tokenizer": self.raw_token_cache_inventory.manifest["tokenizer"],
                "archives": self.raw_token_cache_inventory.manifest["archives"],
                "non_authorities": self.raw_token_cache_inventory.manifest[
                    "non_authorities"
                ],
            }
        return payloads

    def _write_provenance(self) -> dict[str, dict[str, Any]]:
        root = self.output_root / "provenance"
        root.mkdir(parents=True, exist_ok=True)
        payloads = self._provenance_payloads()
        result: dict[str, dict[str, Any]] = {}
        for name, payload in sorted(payloads.items()):
            path = root / name
            atomic_json(path, payload)
            result[name.removesuffix(".json")] = {
                "path": f"provenance/{name}",
                "sha256": file_sha256(path),
            }
        self._write_journal("finalizing")
        return result

    def _final_manifest(
        self,
        packed: Mapping[tuple[str, str], Mapping[str, Any]],
        document_indexes: Mapping[tuple[str, str], Mapping[str, Any]],
        orders: Mapping[str, Mapping[str, Any]],
        provenance: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        splits: dict[str, Any] = {}
        for split in SPLITS:
            packed_descriptors = {}
            for domain in DOMAIN_ORDER:
                path = self.output_root / "packed" / split / domain / "manifest.json"
                index_path = (
                    self.output_root
                    / "provenance"
                    / "documents"
                    / split
                    / domain
                    / "manifest.json"
                )
                packed_descriptors[domain] = {
                    "path": str(path.relative_to(self.output_root)),
                    "sha256": file_sha256(path),
                    "rows": packed[(split, domain)]["rows"],
                    "documents": packed[(split, domain)]["documents"],
                    "source_content_tokens": packed[(split, domain)][
                        "source_content_tokens"
                    ],
                    "document_index": {
                        "path": str(index_path.relative_to(self.output_root)),
                        "sha256": file_sha256(index_path),
                        "documents": document_indexes[(split, domain)]["documents"],
                        "logical_stream_tokens": document_indexes[(split, domain)][
                            "logical_stream_tokens"
                        ],
                    },
                }
            order_path = self.output_root / "orders" / split / "manifest.json"
            splits[split] = {
                "packed": packed_descriptors,
                "order": {
                    "path": str(order_path.relative_to(self.output_root)),
                    "sha256": file_sha256(order_path),
                    "format_version": orders[split]["format_version"],
                    "rows": orders[split]["rows"],
                    "packed_available_rows": orders[split][
                        "packed_available_rows"
                    ],
                    "packed_surplus_rows": orders[split]["packed_surplus_rows"],
                    "authorized_input_tokens": orders[split][
                        "input_token_budget"
                    ]["actual_total"],
                },
            }
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "identity": self.identity,
            "order_configuration": self.config.order_identity(),
            "source_cursor": {
                "next_archive": len(self.archives),
                "archive_count": len(self.archives),
            },
            "split_isolation": {
                "authoritative_assignment": (
                    self.selection_manifest["selection_profile"][
                        "split_authority"
                    ]
                    if self.selection_manifest["identity"]["format_version"]
                    == ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
                    else "completed-curation-decision-shards"
                ),
                "physical_outputs_separate": True,
                "curation_leakage_audit": self.selection_manifest["leakage_audit"],
            },
            "splits": splits,
            "provenance": dict(provenance),
        }

    def _validate_completed(self) -> dict[str, Any]:
        manifest_path = self.output_root / "manifest.json"
        checksum_path = self.output_root / "manifest.sha256"
        manifest = _load_json(manifest_path)
        expected_line = f"{file_sha256(manifest_path)}  manifest.json\n"
        try:
            found_line = checksum_path.read_text(encoding="ascii")
        except OSError as error:
            raise MaterializationError("Completed output lacks manifest.sha256") from error
        if found_line != expected_line:
            raise MaterializationError("Materialized manifest checksum mismatch")
        if (
            manifest.get("format") != FORMAT
            or manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("identity") != self.identity
            or manifest.get("order_configuration") != self.config.order_identity()
        ):
            raise MaterializationError("Completed materialization identity mismatch")
        provenance = manifest.get("provenance")
        splits = manifest.get("splits")
        if not isinstance(provenance, dict) or not isinstance(splits, dict):
            raise MaterializationError("Completed materialization manifest is incomplete")
        for descriptor in provenance.values():
            if not isinstance(descriptor, dict):
                raise MaterializationError("Invalid provenance descriptor")
            path = _safe_file(self.output_root, descriptor.get("path"), "provenance sidecar")
            if file_sha256(path) != descriptor.get("sha256"):
                raise MaterializationError(f"Provenance checksum mismatch: {path}")
        for split in SPLITS:
            split_payload = splits.get(split)
            if not isinstance(split_payload, dict):
                raise MaterializationError(f"Missing completed split {split}")
            packed = split_payload.get("packed")
            if not isinstance(packed, dict) or set(packed) != set(DOMAIN_ORDER):
                raise MaterializationError(f"Invalid packed outputs for {split}")
            for domain in DOMAIN_ORDER:
                descriptor = packed[domain]
                path = _safe_file(
                    self.output_root, descriptor.get("path"), "packed manifest"
                )
                if file_sha256(path) != descriptor.get("sha256"):
                    raise MaterializationError(f"Packed manifest changed: {path}")
                packed_manifest = validate_packed_manifest(
                    path, verify_checksums=True
                )
                for field in ("rows", "documents", "source_content_tokens"):
                    if descriptor.get(field) != packed_manifest.get(field):
                        raise MaterializationError(
                            f"Packed descriptor {field} differs for {split}/{domain}"
                        )
                index_descriptor = descriptor.get("document_index")
                if not isinstance(index_descriptor, dict):
                    raise MaterializationError(
                        f"Missing document index for {split}/{domain}"
                    )
                index_path = _safe_file(
                    self.output_root,
                    index_descriptor.get("path"),
                    "document-index manifest",
                )
                if file_sha256(index_path) != index_descriptor.get("sha256"):
                    raise MaterializationError(
                        f"Document-index manifest changed: {index_path}"
                    )
                index_manifest = self._validate_document_index_manifest(
                    index_path,
                    split=split,
                    domain=domain,
                    packed_manifest=packed_manifest,
                )
                if (
                    index_descriptor.get("documents")
                    != index_manifest["documents"]
                    or index_descriptor.get("logical_stream_tokens")
                    != index_manifest["logical_stream_tokens"]
                ):
                    raise MaterializationError(
                        f"Document-index descriptor totals differ for {split}/{domain}"
                    )
            order_descriptor = split_payload.get("order")
            if not isinstance(order_descriptor, dict):
                raise MaterializationError(f"Missing order output for {split}")
            order_path = _safe_file(
                self.output_root, order_descriptor.get("path"), "order manifest"
            )
            if file_sha256(order_path) != order_descriptor.get("sha256"):
                raise MaterializationError(f"Order manifest changed: {order_path}")
            order = validate_training_order(order_path)
            self._validate_order_identity(split, order)
            expected_order_descriptor = {
                "format_version": order["format_version"],
                "rows": order["rows"],
                "packed_available_rows": order["packed_available_rows"],
                "packed_surplus_rows": order["packed_surplus_rows"],
                "authorized_input_tokens": order["input_token_budget"][
                    "actual_total"
                ],
            }
            for field, expected in expected_order_descriptor.items():
                if order_descriptor.get(field) != expected:
                    raise MaterializationError(
                        f"Order descriptor {field} differs for {split}"
                    )
        if self.journal_path.exists():
            self.journal_path.unlink()
            _fsync_directory(self.output_root)
        return manifest

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root.parent / f".{self.output_root.name}.materialize.lock"
        with lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise MaterializationError(
                    f"Another materializer holds the output lock: {lock_path}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run(
        self,
        *,
        max_archives: int | None = None,
        stop_after_packing: bool = False,
    ) -> dict[str, Any]:
        """Build/resume packing, optionally deferring geometry-bound orders."""

        if max_archives is not None and (
            not isinstance(max_archives, int)
            or isinstance(max_archives, bool)
            or max_archives < 0
        ):
            raise MaterializationError("max_archives must be non-negative")
        if not isinstance(stop_after_packing, bool):
            raise MaterializationError("stop_after_packing must be boolean")
        if not stop_after_packing and (
            self.config.frozen_global_microbatch_rows is None
            or self.config.frozen_gradient_accumulation_steps is None
        ):
            raise MaterializationError(
                "Final order construction requires --global-microbatch-rows and "
                "--gradient-accumulation-steps chosen by the GPU smoke"
            )
        started = time.monotonic()
        runtime_counts = {
            "archives": 0,
            "documents": 0,
            "source_content_tokens": 0,
            "selected_documents": 0,
            "selected_content_tokens": 0,
            "raw_compressed_bytes": 0,
        }

        def runtime_summary() -> dict[str, Any]:
            elapsed = max(time.monotonic() - started, 1e-9)
            selected_stream_tokens = (
                runtime_counts["selected_content_tokens"]
                + runtime_counts["selected_documents"]
            )
            return {
                **runtime_counts,
                "selected_stream_tokens_including_eos": selected_stream_tokens,
                "elapsed_seconds": elapsed,
                "source_content_tokens_per_second": (
                    runtime_counts["source_content_tokens"] / elapsed
                ),
                "selected_content_tokens_per_second": (
                    runtime_counts["selected_content_tokens"] / elapsed
                ),
                "selected_stream_tokens_per_second": (
                    selected_stream_tokens / elapsed
                ),
                "raw_mib_per_second": (
                    runtime_counts["raw_compressed_bytes"] / (1024 * 1024) / elapsed
                ),
                "tokenizer_batch_documents": self.tokenizer_batch_documents,
                "tokenizer_batch_bytes": self.tokenizer_batch_bytes,
                "tokenizer_max_document_bytes": self.tokenizer_max_document_bytes,
            }
        with self._exclusive_lock():
            self._prepare_output()
            if (self.output_root / "manifest.json").exists() and (
                self.output_root / "manifest.sha256"
            ).exists():
                return {"complete": True, "manifest": self._validate_completed()}
            self._open_destinations()
            processed = 0
            try:
                while True:
                    next_archive = min(
                        state.cursor["next_archive"]
                        for state in self.destinations.values()
                    )
                    if next_archive == len(self.archives):
                        break
                    if max_archives is not None and processed >= max_archives:
                        controlled_stop = RuntimeError("controlled materialization stop")
                        self._close_destinations_after_error(controlled_stop)
                        return {
                            "complete": False,
                            "phase": "packing",
                            "completed_archives": next_archive,
                            "archive_count": len(self.archives),
                            "runtime": runtime_summary(),
                        }
                    observed = self._process_archive(self.archives[next_archive])
                    for key, value in observed.items():
                        runtime_counts[key] += value
                    processed += 1
                packed = self._finish_packed_outputs()
                document_indexes = self._finish_document_indexes(packed)
                for split in SPLITS:
                    for domain in DOMAIN_ORDER:
                        validate_packed_manifest(
                            self.output_root
                            / "packed"
                            / split
                            / domain
                            / "manifest.json",
                            verify_checksums=True,
                        )
                self._write_journal("packed")
                if stop_after_packing:
                    return {
                        "complete": False,
                        "phase": "packed",
                        "completed_archives": len(self.archives),
                        "archive_count": len(self.archives),
                        "packed_outputs": len(packed),
                        "document_indexes": len(document_indexes),
                        "runtime": runtime_summary(),
                    }
                orders = self._build_orders()
                provenance = self._write_provenance()
                for split in SPLITS:
                    validate_training_order(
                        self.output_root / "orders" / split / "manifest.json"
                    )
                manifest = self._final_manifest(
                    packed, document_indexes, orders, provenance
                )
                atomic_json(self.output_root / "manifest.json", manifest)
                atomic_bytes(
                    self.output_root / "manifest.sha256",
                    (
                        file_sha256(self.output_root / "manifest.json")
                        + "  manifest.json\n"
                    ).encode("ascii"),
                )
                self.journal_path.unlink()
                _fsync_directory(self.output_root)
                return {
                    "complete": True,
                    "manifest": manifest,
                    "runtime": runtime_summary(),
                }
            except BaseException as error:
                self._close_destinations_after_error(error)
                raise
