"""Immutable, curation-independent token caches for finalized raw archives.

The cache is deliberately *not* a training-data format.  It contains every
document in one raw archive, in the archive's internal-manifest order, with no
EOS insertion, padding, truncation, split assignment, selection, packing, or
shuffle order.  Those policy decisions remain downstream responsibilities.

Each completed archive directory is an atomic publication containing:

* ``tokens.u16``: concatenated little-endian document token IDs;
* ``offsets.u64``: little-endian offsets with exactly ``documents + 1`` items;
* ``manifest.json`` and its exact ``manifest.sha256`` sidecar.

The manifest binds the raw archive, preprocess report and fingerprint shard,
the complete pinned StarCoder2 tokenizer identity, the two binary payloads,
and the builder/config identities.  A completed-cache fast path rehashes all
source and output payloads before returning it as reusable.
"""

from __future__ import annotations

import array
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import sys
import tarfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import zstandard

from pretrain.tokenizer_identity import (
    TokenizerIdentity,
    TokenizerIdentityError,
    require_sha256,
    sha256_file,
    verify_tokenizer_identity,
)


CACHE_FORMAT = "raw-document-token-cache"
CACHE_FORMAT_VERSION = 1
CACHE_PROFILE = "all-raw-documents-content-only-v1"
STARCODE_REPO_ID = "bigcode/starcoder2-tokenizer"
EXPECTED_VOCAB_SIZE = 49_152
TOKEN_FILE = "tokens.u16"
OFFSET_FILE = "offsets.u64"
MANIFEST_FILE = "manifest.json"
SIDECAR_FILE = "manifest.sha256"
OUTPUT_FILES = {TOKEN_FILE, OFFSET_FILE, MANIFEST_FILE, SIDECAR_FILE}
BUCKETS = {"python", "other_code", "fineweb_edu", "wikipedia"}
PART_REPORT = re.compile(r"part-(\d{6})\.json")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

TOKENIZATION_CONTRACT = {
    "added_special_tokens": False,
    "boundary_tokens": False,
    "document_order": "raw-internal-manifest-order",
    "document_selection": "all",
    "padding": False,
    "token_payload": "document-content-only",
    "truncation": False,
}
TOKENIZATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        TOKENIZATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class RawTokenCacheError(RuntimeError):
    """A fail-closed source, tokenizer, publication, or validation failure."""


@dataclass(frozen=True)
class CacheConfig:
    """Hard resource bounds and deterministic cache construction settings."""

    expected_vocab_size: int = EXPECTED_VOCAB_SIZE
    max_documents_per_archive: int = 2_000_000
    max_document_bytes: int = 16 * 1024 * 1024
    max_document_tokens: int = 8 * 1024 * 1024
    tokenizer_batch_documents: int = 64
    tokenizer_batch_bytes: int = 32 * 1024 * 1024
    tokenizer_batch_tokens: int = 2 * 1024 * 1024
    max_manifest_member_bytes: int = 8 * 1024 * 1024 * 1024
    max_json_line_bytes: int = 1024 * 1024
    minimum_free_bytes: int = 10 * 1024 * 1024 * 1024

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RawTokenCacheError(f"Cache config {name} must be a positive integer")
        if self.expected_vocab_size > 65_536:
            raise RawTokenCacheError("uint16 cache cannot represent the configured vocabulary")
        if self.tokenizer_batch_bytes > self.max_document_bytes * self.tokenizer_batch_documents:
            # This is not unsafe, but nearly always indicates a unit/config typo.
            raise RawTokenCacheError(
                "tokenizer_batch_bytes exceeds the maximum possible bounded batch"
            )
        if self.tokenizer_batch_tokens > self.max_document_tokens * self.tokenizer_batch_documents:
            raise RawTokenCacheError(
                "tokenizer_batch_tokens exceeds the maximum possible bounded batch"
            )

    def identity_payload(self) -> dict[str, int]:
        return dict(sorted(asdict(self).items()))


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class CacheJob:
    dataset_root: Path
    preprocess_root: Path
    report_path: Path
    report_relative: str
    report_sha256: str
    report_bytes: int
    archive_path: Path
    archive_relative: str
    fingerprint_path: Path
    fingerprint_relative: str
    bucket: str
    index: int
    documents: int
    clean_bytes: int
    exact_tokens: int
    archive_sha256: str
    archive_compressed_bytes: int
    fingerprint_sha256: str
    report: dict[str, Any]

    def target(self, output_root: Path) -> Path:
        return output_root / "archives" / self.bucket / f"part-{self.index:06d}"

    @property
    def estimated_output_bytes(self) -> int:
        return self.exact_tokens * 2 + (self.documents + 1) * 8 + 64 * 1024


@dataclass(frozen=True)
class CacheResult:
    archive: str
    target: str
    status: str
    documents: int
    content_tokens: int
    elapsed_seconds: float


@dataclass
class TokenizerRuntime:
    tokenizer: Any
    identity: TokenizerIdentity
    descriptor: dict[str, Any]


class _HashingReader(io.RawIOBase):
    def __init__(self, raw: BinaryIO, digest: Any):
        self.raw = raw
        self.digest = digest

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        payload = self.raw.read(size)
        if payload:
            self.digest.update(payload)
        return payload

    def readinto(self, buffer: bytearray) -> int:
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot_regular_file(path: Path, *, label: str) -> FileSnapshot:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RawTokenCacheError(f"Cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RawTokenCacheError(f"{label} is not a non-symlink regular file: {path}")
    return FileSnapshot(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _assert_unchanged(path: Path, before: FileSnapshot, *, label: str) -> None:
    after = _snapshot_regular_file(path, label=label)
    if after != before:
        raise RawTokenCacheError(f"{label} changed while it was being consumed: {path}")


def _hash_stable_file(path: Path, *, label: str) -> tuple[str, FileSnapshot]:
    before = _snapshot_regular_file(path, label=label)
    digest = sha256_file(path)
    _assert_unchanged(path, before, label=label)
    return digest, before


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RawTokenCacheError(f"Unsafe {label}: {value!r}")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise RawTokenCacheError(f"Unsafe {label}: {value!r}")
    return candidate.as_posix()


def _safe_member_name(value: Any) -> str:
    return _safe_relative(value, label="tar member path")


def _ensure_safe_directory(path: Path) -> None:
    """Create a directory while rejecting symlink/non-directory components."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise RawTokenCacheError(f"Unsafe output directory component: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir():
            raise RawTokenCacheError(f"Unsafe output directory component: {directory}")


def _file_under(root: Path, relative: str, *, label: str) -> Path:
    relative = _safe_relative(relative, label=f"{label} relative path")
    if root.is_symlink() or not root.is_dir():
        raise RawTokenCacheError(f"{label} root is unsafe: {root}")
    cursor = root
    for component in PurePosixPath(relative).parts:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise RawTokenCacheError(f"Missing {label}: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RawTokenCacheError(f"Symlink in {label} path: {cursor}")
    if not stat.S_ISREG(cursor.lstat().st_mode):
        raise RawTokenCacheError(f"{label} is not a regular file: {cursor}")
    return cursor


def _positive_int(value: Any, *, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RawTokenCacheError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise RawTokenCacheError(f"{field} exceeds the operational maximum {maximum}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except TokenizerIdentityError as exc:
        raise RawTokenCacheError(str(exc)) from exc


def _read_report(
    path: Path, *, maximum_bytes: int = 8 * 1024 * 1024
) -> tuple[bytes, dict[str, Any]]:
    before = _snapshot_regular_file(path, label="preprocess report")
    if before.size < 2 or before.size > maximum_bytes:
        raise RawTokenCacheError(f"Preprocess report has an unsafe size: {path}")
    raw = path.read_bytes()
    _assert_unchanged(path, before, label="preprocess report")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTokenCacheError(f"Invalid preprocess report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RawTokenCacheError(f"Preprocess report root is not an object: {path}")
    return raw, payload


def load_cache_job(
    dataset_root: Path,
    preprocess_root: Path,
    report_path: Path,
    *,
    config: CacheConfig,
) -> CacheJob:
    """Authenticate one canonical preprocess report enough to schedule a job."""

    config.validate()
    dataset_root = dataset_root.absolute()
    preprocess_root = preprocess_root.absolute()
    report_path = report_path.absolute()
    try:
        report_relative = report_path.relative_to(preprocess_root).as_posix()
    except ValueError as exc:
        raise RawTokenCacheError("Preprocess report is outside preprocess_root") from exc
    report_relative = _safe_relative(report_relative, label="preprocess report path")
    canonical_report_path = _file_under(
        preprocess_root, report_relative, label="preprocess report"
    )
    if canonical_report_path != report_path:
        raise RawTokenCacheError("Preprocess report path is not canonical")
    raw, report = _read_report(report_path)
    bucket = report.get("bucket")
    if bucket not in BUCKETS:
        raise RawTokenCacheError(f"Unsupported preprocess bucket: {bucket!r}")
    index = report.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise RawTokenCacheError("Preprocess report index must be non-negative")
    expected_report = f"reports/{bucket}/part-{index:06d}.json"
    if report_relative != expected_report:
        raise RawTokenCacheError(
            f"Non-canonical preprocess report path: {report_relative!r} != {expected_report!r}"
        )
    if report.get("report_version") != 1 or report.get("fingerprint_version") != 1:
        raise RawTokenCacheError("Unsupported preprocess report/fingerprint version")
    archive_relative = _safe_relative(report.get("archive"), label="raw archive path")
    fingerprint_relative = _safe_relative(
        report.get("fingerprint_file"), label="fingerprint path"
    )
    expected_fingerprint = f"fingerprints/{bucket}/part-{index:06d}.jsonl.zst"
    if fingerprint_relative != expected_fingerprint:
        raise RawTokenCacheError(
            f"Non-canonical fingerprint path: {fingerprint_relative!r}"
        )
    archive_path = _file_under(dataset_root, archive_relative, label="raw archive")
    fingerprint_path = _file_under(
        preprocess_root, fingerprint_relative, label="fingerprint shard"
    )
    documents = _positive_int(
        report.get("documents"),
        field="preprocess report documents",
        maximum=config.max_documents_per_archive,
    )
    clean_bytes = _positive_int(report.get("clean_bytes"), field="preprocess clean_bytes")
    exact_tokens = _positive_int(report.get("exact_tokens"), field="preprocess exact_tokens")
    archive_compressed_bytes = _positive_int(
        report.get("archive_compressed_bytes"), field="archive_compressed_bytes"
    )
    if archive_path.stat().st_size != archive_compressed_bytes:
        raise RawTokenCacheError("Raw archive size differs from preprocess report")
    return CacheJob(
        dataset_root=dataset_root,
        preprocess_root=preprocess_root,
        report_path=report_path,
        report_relative=report_relative,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        report_bytes=len(raw),
        archive_path=archive_path,
        archive_relative=archive_relative,
        fingerprint_path=fingerprint_path,
        fingerprint_relative=fingerprint_relative,
        bucket=str(bucket),
        index=index,
        documents=documents,
        clean_bytes=clean_bytes,
        exact_tokens=exact_tokens,
        archive_sha256=_sha256(report.get("archive_sha256"), field="archive SHA-256"),
        archive_compressed_bytes=archive_compressed_bytes,
        fingerprint_sha256=_sha256(
            report.get("fingerprint_sha256"), field="fingerprint SHA-256"
        ),
        report=report,
    )


def discover_cache_jobs(
    dataset_root: Path,
    preprocess_root: Path,
    *,
    config: CacheConfig,
) -> list[CacheJob]:
    reports_root = preprocess_root / "reports"
    if reports_root.is_symlink() or not reports_root.is_dir():
        raise RawTokenCacheError(f"Missing or unsafe preprocess reports root: {reports_root}")
    paths: list[Path] = []
    for bucket_path in sorted(reports_root.iterdir(), key=lambda item: item.name):
        if bucket_path.name not in BUCKETS:
            raise RawTokenCacheError(f"Unexpected preprocess report bucket: {bucket_path}")
        if bucket_path.is_symlink() or not bucket_path.is_dir():
            raise RawTokenCacheError(f"Unsafe preprocess report bucket: {bucket_path}")
        for report_path in sorted(bucket_path.iterdir(), key=lambda item: item.name):
            if report_path.name.startswith(".") or PART_REPORT.fullmatch(report_path.name) is None:
                raise RawTokenCacheError(f"Pending or unexpected preprocess report: {report_path}")
            paths.append(report_path)
    jobs = [
        load_cache_job(dataset_root, preprocess_root, path, config=config)
        for path in paths
    ]
    keys = [(job.bucket, job.index) for job in jobs]
    archives = [job.archive_relative for job in jobs]
    if len(set(keys)) != len(keys) or len(set(archives)) != len(archives):
        raise RawTokenCacheError("Duplicate cache job/report identity")
    return sorted(jobs, key=lambda item: item.archive_relative)


def _load_tokenizer_runtime(
    tokenizer_root: Path, *, expected_vocab_size: int
) -> TokenizerRuntime:
    try:
        identity = verify_tokenizer_identity(
            tokenizer_root,
            expected_vocab_size=expected_vocab_size,
        )
    except (TokenizerIdentityError, OSError, RuntimeError) as exc:
        raise RawTokenCacheError(f"Cannot authenticate tokenizer: {exc}") from exc
    manifest = identity.manifest
    if manifest.get("repo_id") != STARCODE_REPO_ID:
        raise RawTokenCacheError(
            f"Tokenizer repo must be {STARCODE_REPO_ID!r}, found {manifest.get('repo_id')!r}"
        )
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RawTokenCacheError("Tokenizer resolved revision is not a lowercase commit SHA")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise RawTokenCacheError("Tokenizer manifest lacks validation metadata")
    eos_token = validation.get("eos_token")
    eos_token_id = validation.get("eos_token_id")
    if not isinstance(eos_token, str) or not eos_token:
        raise RawTokenCacheError("Tokenizer manifest lacks EOS token identity")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < identity.vocab_size
    ):
        raise RawTokenCacheError("Tokenizer manifest has an invalid EOS token ID")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
    except Exception as exc:
        raise RawTokenCacheError(f"Cannot load pinned tokenizer: {exc}") from exc
    if tokenizer.token_to_id(eos_token) != eos_token_id:
        raise RawTokenCacheError("Loaded tokenizer EOS differs from tokenizer manifest")
    descriptor = {
        "repo_id": STARCODE_REPO_ID,
        "resolved_revision": revision,
        "manifest_sha256": identity.manifest_sha256,
        "vocabulary_sha256": identity.vocabulary_sha256,
        "vocab_size": identity.vocab_size,
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "eos_present_in_payload": False,
    }
    return TokenizerRuntime(tokenizer=tokenizer, identity=identity, descriptor=descriptor)


class _FingerprintReader:
    """Bounded compressed JSONL reader that hashes the exact source bytes."""

    def __init__(self, job: CacheJob, maximum_line_bytes: int):
        self.job = job
        self.maximum_line_bytes = maximum_line_bytes
        self.snapshot: FileSnapshot | None = None
        self.raw: BinaryIO | None = None
        self.hashing: _HashingReader | None = None
        self.decompressor: Any = None
        self.buffered: io.BufferedReader | None = None
        self.digest = hashlib.sha256()
        self.rows = 0
        self.finished = False

    def __enter__(self) -> "_FingerprintReader":
        self.snapshot = _snapshot_regular_file(
            self.job.fingerprint_path, label="fingerprint shard"
        )
        self.raw = self.job.fingerprint_path.open("rb")
        self.hashing = _HashingReader(self.raw, self.digest)
        self.decompressor = zstandard.ZstdDecompressor().stream_reader(
            self.hashing, read_across_frames=True, closefd=False
        )
        self.buffered = io.BufferedReader(self.decompressor, buffer_size=1024 * 1024)
        return self

    def next_row(self) -> dict[str, Any]:
        if self.buffered is None or self.finished:
            raise RawTokenCacheError("Fingerprint reader is not active")
        raw_line = self.buffered.readline(self.maximum_line_bytes + 1)
        if not raw_line:
            raise RawTokenCacheError("Fingerprint has fewer rows than raw archive")
        if len(raw_line) > self.maximum_line_bytes:
            raise RawTokenCacheError("Fingerprint row exceeds the configured byte bound")
        if not raw_line.endswith(b"\n") or not raw_line.strip():
            raise RawTokenCacheError("Fingerprint contains an unterminated or blank row")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawTokenCacheError(f"Invalid fingerprint JSON row {self.rows}: {exc}") from exc
        if not isinstance(row, dict):
            raise RawTokenCacheError("Fingerprint row is not an object")
        self.rows += 1
        return row

    def finish(self) -> str:
        if self.buffered is None or self.finished:
            raise RawTokenCacheError("Fingerprint reader cannot be finished twice")
        extra = self.buffered.readline(self.maximum_line_bytes + 1)
        if extra:
            raise RawTokenCacheError("Fingerprint has more rows than raw archive")
        while self.buffered.read(8 * 1024 * 1024):
            pass
        assert self.hashing is not None
        while self.hashing.read(8 * 1024 * 1024):
            pass
        self.finished = True
        if self.rows != self.job.documents:
            raise RawTokenCacheError("Fingerprint row count differs from preprocess report")
        digest = self.digest.hexdigest()
        if digest != self.job.fingerprint_sha256:
            raise RawTokenCacheError("Fingerprint checksum differs from preprocess report")
        assert self.snapshot is not None
        _assert_unchanged(
            self.job.fingerprint_path, self.snapshot, label="fingerprint shard"
        )
        return digest

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        with contextlib.suppress(Exception):
            if self.buffered is not None:
                self.buffered.close()
        with contextlib.suppress(Exception):
            if self.decompressor is not None:
                self.decompressor.close()
        with contextlib.suppress(Exception):
            if self.raw is not None:
                self.raw.close()


def _validate_fingerprint_row(
    job: CacheJob,
    row: Mapping[str, Any],
    index: int,
    member: tarfile.TarInfo,
) -> dict[str, Any]:
    expected = {
        "record_version": 1,
        "fingerprint_version": 1,
        "archive": job.archive_relative,
        "archive_index": job.index,
        "bucket": job.bucket,
        "manifest_index": index,
        "member_path": member.name,
        "size_bytes": member.size,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RawTokenCacheError(
                f"Fingerprint {field} mismatch at document {index}: {row.get(field)!r} != {value!r}"
            )
    expected_doc_id = hashlib.sha256(
        f"{job.archive_relative}\0{member.name}".encode("utf-8")
    ).hexdigest()
    if row.get("doc_id") != expected_doc_id:
        raise RawTokenCacheError(f"Fingerprint doc_id mismatch at document {index}")
    tokens = _positive_int(
        row.get("starcoder2_tokens"), field=f"fingerprint token count at document {index}"
    )
    content_sha = _sha256(
        row.get("content_sha256"), field=f"fingerprint content SHA-256 at document {index}"
    )
    return {
        "manifest_index": index,
        "member_path": member.name,
        "size_bytes": member.size,
        "starcoder2_tokens": tokens,
        "content_sha256": content_sha,
    }


def _write_u16(
    handle: BinaryIO,
    digest: Any,
    identifiers: Sequence[int],
    vocab_size: int,
) -> tuple[int, int]:
    minimum = vocab_size
    maximum = -1
    values = array.array("H")
    for identifier in identifiers:
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise RawTokenCacheError("Tokenizer emitted a non-integer token ID")
        if not 0 <= identifier < vocab_size or identifier >= 65_536:
            raise RawTokenCacheError(f"Tokenizer emitted out-of-range token ID {identifier}")
        minimum = min(minimum, identifier)
        maximum = max(maximum, identifier)
        values.append(identifier)
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    handle.write(payload)
    digest.update(payload)
    return minimum, maximum


def _write_u64(handle: BinaryIO, digest: Any, value: int) -> None:
    if not 0 <= value < 2**64:
        raise RawTokenCacheError("Token offset exceeds uint64 range")
    payload = struct.pack("<Q", value)
    handle.write(payload)
    digest.update(payload)


def _validate_internal_manifest(
    handle: BinaryIO,
    spool: BinaryIO,
    *,
    job: CacheJob,
    maximum_line_bytes: int,
) -> None:
    spool.flush()
    spool.seek(0)
    seen: set[str] = set()
    for index in range(job.documents):
        raw_line = handle.readline(maximum_line_bytes + 1)
        if not raw_line:
            raise RawTokenCacheError("Raw internal manifest has too few rows")
        if len(raw_line) > maximum_line_bytes:
            raise RawTokenCacheError("Raw internal manifest row exceeds the byte bound")
        if not raw_line.endswith(b"\n") or not raw_line.strip():
            raise RawTokenCacheError("Raw internal manifest contains a blank/unterminated row")
        spool_line = spool.readline(maximum_line_bytes + 1)
        if not spool_line or len(spool_line) > maximum_line_bytes:
            raise RawTokenCacheError("Internal observed-document spool is inconsistent")
        try:
            raw_row = json.loads(raw_line)
            observed = json.loads(spool_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawTokenCacheError(f"Invalid raw manifest JSON at row {index}: {exc}") from exc
        if not isinstance(raw_row, dict):
            raise RawTokenCacheError("Raw internal manifest row is not an object")
        member_path = _safe_member_name(raw_row.get("member_path"))
        if member_path in seen:
            raise RawTokenCacheError(f"Duplicate raw internal-manifest member: {member_path}")
        seen.add(member_path)
        expected = {
            "manifest_index": index,
            "member_path": member_path,
            "size_bytes": _positive_int(
                raw_row.get("size_bytes"), field=f"manifest size at row {index}"
            ),
            "starcoder2_tokens": _positive_int(
                raw_row.get("starcoder2_tokens"), field=f"manifest tokens at row {index}"
            ),
        }
        if any(observed.get(field) != value for field, value in expected.items()):
            raise RawTokenCacheError(f"Raw manifest/document order mismatch at row {index}")
    if handle.readline(maximum_line_bytes + 1):
        raise RawTokenCacheError("Raw internal manifest has too many rows")
    if spool.readline(1):
        raise RawTokenCacheError("Raw archive has more documents than its manifest")


def _verify_job_report(job: CacheJob) -> None:
    raw, report = _read_report(job.report_path)
    if (
        len(raw) != job.report_bytes
        or hashlib.sha256(raw).hexdigest() != job.report_sha256
        or report != job.report
    ):
        raise RawTokenCacheError("Preprocess report changed after cache job discovery")


def _builder_descriptor(config: CacheConfig) -> dict[str, Any]:
    implementation_sha256 = sha256_file(Path(__file__))
    config_payload = config.identity_payload()
    return {
        "implementation": "pretrain.raw_token_cache",
        "implementation_sha256": implementation_sha256,
        "config": config_payload,
        "config_sha256": _canonical_sha256(config_payload),
        "tokenization_contract_sha256": TOKENIZATION_CONTRACT_SHA256,
    }


def _manifest_payload(
    *,
    job: CacheJob,
    config: CacheConfig,
    runtime: TokenizerRuntime,
    tokens_sha256: str,
    offsets_sha256: str,
    minimum_token_id: int,
    maximum_token_id: int,
    alignment_sha256: str,
    fingerprint_bytes: int,
) -> dict[str, Any]:
    return {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "profile": CACHE_PROFILE,
        "cache_complete": True,
        "training_ready": False,
        "tokenization": dict(TOKENIZATION_CONTRACT),
        "non_authorities": [
            "curation",
            "document_selection",
            "eos_insertion",
            "packing",
            "shuffle_order",
            "split_assignment",
        ],
        "source": {
            "archive": {
                "path": job.archive_relative,
                "bucket": job.bucket,
                "index": job.index,
                "bytes": job.archive_compressed_bytes,
                "sha256": job.archive_sha256,
            },
            "preprocess_report": {
                "path": job.report_relative,
                "bytes": job.report_bytes,
                "sha256": job.report_sha256,
                "report_version": 1,
            },
            "fingerprint": {
                "path": job.fingerprint_relative,
                "bytes": fingerprint_bytes,
                "sha256": job.fingerprint_sha256,
                "fingerprint_version": 1,
            },
        },
        "tokenizer": dict(runtime.descriptor),
        "documents": {
            "records": job.documents,
            "clean_bytes": job.clean_bytes,
            "content_tokens": job.exact_tokens,
            "alignment": {
                "authority": "manifest-index+raw-manifest+preprocess-fingerprint",
                "record_sha256": alignment_sha256,
                "offset_items": job.documents + 1,
                "offset_zero": 0,
                "terminal_offset": job.exact_tokens,
            },
        },
        "payloads": {
            "tokens": {
                "path": TOKEN_FILE,
                "dtype": "uint16",
                "endianness": "little",
                "items": job.exact_tokens,
                "bytes": job.exact_tokens * 2,
                "sha256": tokens_sha256,
                "minimum_id": minimum_token_id,
                "maximum_id": maximum_token_id,
            },
            "offsets": {
                "path": OFFSET_FILE,
                "dtype": "uint64",
                "endianness": "little",
                "items": job.documents + 1,
                "bytes": (job.documents + 1) * 8,
                "sha256": offsets_sha256,
            },
        },
        "builder": _builder_descriptor(config),
    }


def _cleanup_stale_stages(target: Path) -> None:
    prefix = f".{target.name}.building-"
    if not target.parent.exists():
        return
    for candidate in target.parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise RawTokenCacheError(f"Unsafe stale cache staging entry: {candidate}")
        shutil.rmtree(candidate)
        _fsync_directory(target.parent)


def _build_new_archive(
    job: CacheJob,
    target: Path,
    *,
    config: CacheConfig,
    runtime: TokenizerRuntime,
) -> CacheResult:
    started = time.monotonic()
    _verify_job_report(job)
    archive_snapshot = _snapshot_regular_file(job.archive_path, label="raw archive")
    fingerprint_snapshot = _snapshot_regular_file(
        job.fingerprint_path, label="fingerprint shard"
    )
    if archive_snapshot.size != job.archive_compressed_bytes:
        raise RawTokenCacheError("Raw archive size differs from preprocess report")
    stage = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    tokens_path = stage / TOKEN_FILE
    offsets_path = stage / OFFSET_FILE
    spool_path = stage / ".observed-documents.jsonl"
    token_digest = hashlib.sha256()
    offset_digest = hashlib.sha256()
    alignment_digest = hashlib.sha256()
    archive_digest = hashlib.sha256()
    document_count = 0
    clean_bytes = 0
    content_tokens = 0
    min_token_id = runtime.identity.vocab_size
    max_token_id = -1
    manifest_seen = False
    member_names: set[str] = set()
    pending: list[dict[str, Any]] = []
    pending_bytes = 0
    pending_tokens = 0

    try:
        with (
            tokens_path.open("wb") as tokens_handle,
            offsets_path.open("wb") as offsets_handle,
            spool_path.open("w+b") as spool,
            _FingerprintReader(job, config.max_json_line_bytes) as fingerprints,
        ):
            _write_u64(offsets_handle, offset_digest, 0)

            def flush_pending() -> None:
                nonlocal content_tokens, min_token_id, max_token_id
                nonlocal pending_bytes, pending_tokens
                if not pending:
                    return
                try:
                    encodings = runtime.tokenizer.encode_batch(
                        [record["text"] for record in pending],
                        add_special_tokens=False,
                    )
                except Exception as exc:
                    raise RawTokenCacheError(f"Tokenizer batch failed: {exc}") from exc
                if len(encodings) != len(pending):
                    raise RawTokenCacheError("Tokenizer returned the wrong batch length")
                for record, encoding in zip(pending, encodings, strict=True):
                    identifiers = encoding.ids
                    expected_tokens = record["starcoder2_tokens"]
                    if len(identifiers) != expected_tokens:
                        raise RawTokenCacheError(
                            "Pinned tokenizer count mismatch for "
                            f"{job.archive_relative}:{record['member_path']} "
                            f"({len(identifiers)} != {expected_tokens})"
                        )
                    if len(identifiers) > config.max_document_tokens:
                        raise RawTokenCacheError(
                            f"Document exceeds max_document_tokens: {record['member_path']}"
                        )
                    document_start = content_tokens
                    observed_min, observed_max = _write_u16(
                        tokens_handle,
                        token_digest,
                        identifiers,
                        runtime.identity.vocab_size,
                    )
                    min_token_id = min(min_token_id, observed_min)
                    max_token_id = max(max_token_id, observed_max)
                    content_tokens += len(identifiers)
                    _write_u64(offsets_handle, offset_digest, content_tokens)
                    observed = {
                        "manifest_index": record["manifest_index"],
                        "member_path": record["member_path"],
                        "size_bytes": record["size_bytes"],
                        "starcoder2_tokens": expected_tokens,
                    }
                    spool.write(_canonical_json_bytes(observed))
                    alignment_digest.update(
                        _canonical_json_bytes(
                            {
                                "manifest_index": record["manifest_index"],
                                "member_path": record["member_path"],
                                "content_sha256": record["content_sha256"],
                                "token_start": document_start,
                                "token_end": content_tokens,
                            }
                        )
                    )
                pending.clear()
                pending_bytes = 0
                pending_tokens = 0

            with job.archive_path.open("rb") as raw_archive:
                hashing_reader = _HashingReader(raw_archive, archive_digest)
                decompressor = zstandard.ZstdDecompressor().stream_reader(
                    hashing_reader, read_across_frames=True, closefd=False
                )
                try:
                    with tarfile.open(fileobj=decompressor, mode="r|") as archive:
                        for member in archive:
                            if not member.isfile():
                                raise RawTokenCacheError(
                                    f"Unexpected non-regular tar member: {member.name!r}"
                                )
                            if member.name == "_manifest.jsonl":
                                if manifest_seen:
                                    raise RawTokenCacheError("Raw archive has multiple manifests")
                                if member.size > config.max_manifest_member_bytes:
                                    raise RawTokenCacheError(
                                        "Raw internal manifest exceeds its byte bound"
                                    )
                                manifest_seen = True
                                flush_pending()
                                extracted = archive.extractfile(member)
                                if extracted is None:
                                    raise RawTokenCacheError("Cannot read raw internal manifest")
                                _validate_internal_manifest(
                                    extracted,
                                    spool,
                                    job=job,
                                    maximum_line_bytes=config.max_json_line_bytes,
                                )
                                continue
                            member_name = _safe_member_name(member.name)
                            if manifest_seen:
                                raise RawTokenCacheError(
                                    "Raw document appears after the internal manifest"
                                )
                            if document_count >= job.documents:
                                raise RawTokenCacheError(
                                    "Raw archive has more documents than preprocess report"
                                )
                            if member_name in member_names:
                                raise RawTokenCacheError(
                                    f"Duplicate raw document member: {member_name}"
                                )
                            member_names.add(member_name)
                            if member.size < 1 or member.size > config.max_document_bytes:
                                raise RawTokenCacheError(
                                    f"Raw document size exceeds bounds: {member_name}"
                                )
                            fingerprint = _validate_fingerprint_row(
                                job, fingerprints.next_row(), document_count, member
                            )
                            expected_tokens = fingerprint["starcoder2_tokens"]
                            if expected_tokens > config.max_document_tokens:
                                raise RawTokenCacheError(
                                    f"Document token count exceeds bounds: {member_name}"
                                )
                            if pending and (
                                len(pending) >= config.tokenizer_batch_documents
                                or pending_bytes + member.size > config.tokenizer_batch_bytes
                                or pending_tokens + expected_tokens
                                > config.tokenizer_batch_tokens
                            ):
                                flush_pending()
                            extracted = archive.extractfile(member)
                            if extracted is None:
                                raise RawTokenCacheError(f"Cannot read tar member: {member_name}")
                            content = extracted.read(config.max_document_bytes + 1)
                            if len(content) != member.size:
                                raise RawTokenCacheError(
                                    f"Raw member size mismatch for {member_name}: "
                                    f"{len(content)} != {member.size}"
                                )
                            content_sha = hashlib.sha256(content).hexdigest()
                            if content_sha != fingerprint["content_sha256"]:
                                raise RawTokenCacheError(
                                    f"Raw/fingerprint content mismatch: {member_name}"
                                )
                            try:
                                text = content.decode("utf-8", errors="strict")
                            except UnicodeDecodeError as exc:
                                raise RawTokenCacheError(
                                    f"Raw document is not UTF-8: {member_name}"
                                ) from exc
                            pending.append({**fingerprint, "text": text})
                            pending_bytes += len(content)
                            pending_tokens += expected_tokens
                            document_count += 1
                            clean_bytes += len(content)
                            if (
                                len(pending) >= config.tokenizer_batch_documents
                                or pending_bytes >= config.tokenizer_batch_bytes
                                or pending_tokens >= config.tokenizer_batch_tokens
                            ):
                                flush_pending()
                    flush_pending()
                    while decompressor.read(8 * 1024 * 1024):
                        pass
                except (zstandard.ZstdError, tarfile.TarError, OSError) as exc:
                    raise RawTokenCacheError(
                        f"Corrupt raw archive {job.archive_path}: {exc}"
                    ) from exc
                finally:
                    decompressor.close()
                while hashing_reader.read(8 * 1024 * 1024):
                    pass
            fingerprint_digest = fingerprints.finish()
            for handle in (tokens_handle, offsets_handle, spool):
                handle.flush()
                os.fsync(handle.fileno())
        if not manifest_seen:
            raise RawTokenCacheError("Raw archive is missing _manifest.jsonl")
        _assert_unchanged(job.archive_path, archive_snapshot, label="raw archive")
        _assert_unchanged(
            job.fingerprint_path, fingerprint_snapshot, label="fingerprint shard"
        )
        if archive_digest.hexdigest() != job.archive_sha256:
            raise RawTokenCacheError("Raw archive checksum differs from preprocess report")
        if fingerprint_digest != job.fingerprint_sha256:
            raise RawTokenCacheError("Fingerprint checksum differs from preprocess report")
        if (
            document_count != job.documents
            or clean_bytes != job.clean_bytes
            or content_tokens != job.exact_tokens
        ):
            raise RawTokenCacheError(
                "Token-cache totals differ from authenticated preprocess report"
            )
        if min_token_id >= runtime.identity.vocab_size or max_token_id < 0:
            raise RawTokenCacheError("Token cache unexpectedly contains no token IDs")
        _verify_job_report(job)
        spool_path.unlink()
        manifest = _manifest_payload(
            job=job,
            config=config,
            runtime=runtime,
            tokens_sha256=token_digest.hexdigest(),
            offsets_sha256=offset_digest.hexdigest(),
            minimum_token_id=min_token_id,
            maximum_token_id=max_token_id,
            alignment_sha256=alignment_digest.hexdigest(),
            fingerprint_bytes=fingerprint_snapshot.size,
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        with (stage / MANIFEST_FILE).open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with (stage / SIDECAR_FILE).open("wb") as handle:
            handle.write(f"{manifest_sha}  {MANIFEST_FILE}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(stage)
        os.rename(stage, target)
        _fsync_directory(target.parent)
        return CacheResult(
            archive=job.archive_relative,
            target=str(target),
            status="built",
            documents=document_count,
            content_tokens=content_tokens,
            elapsed_seconds=round(time.monotonic() - started, 6),
        )
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
            _fsync_directory(target.parent)
        raise


def _read_manifest(target: Path) -> dict[str, Any]:
    if target.is_symlink() or not target.is_dir():
        raise RawTokenCacheError(f"Completed cache target is unsafe: {target}")
    names = {entry.name for entry in target.iterdir()}
    if names != OUTPUT_FILES:
        raise RawTokenCacheError(
            f"Completed cache file set mismatch: missing={sorted(OUTPUT_FILES - names)}, "
            f"extra={sorted(names - OUTPUT_FILES)}"
        )
    for name in OUTPUT_FILES:
        _snapshot_regular_file(target / name, label=f"cache output {name}")
    manifest_raw = (target / MANIFEST_FILE).read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    expected_sidecar = f"{manifest_sha}  {MANIFEST_FILE}\n".encode("ascii")
    if (target / SIDECAR_FILE).read_bytes() != expected_sidecar:
        raise RawTokenCacheError("Cache manifest sidecar mismatch")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTokenCacheError(f"Invalid cache manifest: {exc}") from exc
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != manifest_raw:
        raise RawTokenCacheError("Cache manifest is not canonical JSON")
    return manifest


def _hash_u16_payload(path: Path) -> tuple[str, int, int, int]:
    snapshot = _snapshot_regular_file(path, label="token payload")
    if snapshot.size % 2:
        raise RawTokenCacheError("uint16 token payload has an odd byte length")
    digest = hashlib.sha256()
    items = 0
    minimum = 65_536
    maximum = -1
    with path.open("rb") as handle:
        while payload := handle.read(8 * 1024 * 1024):
            digest.update(payload)
            values = array.array("H")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            if values:
                minimum = min(minimum, min(values))
                maximum = max(maximum, max(values))
                items += len(values)
    _assert_unchanged(path, snapshot, label="token payload")
    return digest.hexdigest(), items, minimum, maximum


def _hash_u64_offsets(path: Path) -> tuple[str, int, int]:
    snapshot = _snapshot_regular_file(path, label="offset payload")
    if snapshot.size % 8:
        raise RawTokenCacheError("uint64 offset payload has an invalid byte length")
    digest = hashlib.sha256()
    count = 0
    previous = -1
    with path.open("rb") as handle:
        while payload := handle.read(8 * 1024 * 1024):
            digest.update(payload)
            values = array.array("Q")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            for value in values:
                if count == 0 and value != 0:
                    raise RawTokenCacheError("First document offset is not zero")
                if value < previous:
                    raise RawTokenCacheError("Document offsets are not monotonic")
                previous = int(value)
                count += 1
    _assert_unchanged(path, snapshot, label="offset payload")
    return digest.hexdigest(), count, previous


def validate_completed_cache(
    job: CacheJob,
    target: Path,
    *,
    runtime: TokenizerRuntime,
) -> dict[str, Any]:
    """Rehash a completed cache and its source authorities before reuse."""

    _verify_job_report(job)
    manifest = _read_manifest(target)
    required_top = {
        "format",
        "format_version",
        "profile",
        "cache_complete",
        "training_ready",
        "tokenization",
        "non_authorities",
        "source",
        "tokenizer",
        "documents",
        "payloads",
        "builder",
    }
    if set(manifest) != required_top:
        raise RawTokenCacheError("Cache manifest top-level schema mismatch")
    if (
        manifest["format"] != CACHE_FORMAT
        or manifest["format_version"] != CACHE_FORMAT_VERSION
        or manifest["profile"] != CACHE_PROFILE
        or manifest["cache_complete"] is not True
        or manifest["training_ready"] is not False
        or manifest["tokenization"] != TOKENIZATION_CONTRACT
    ):
        raise RawTokenCacheError("Cache manifest contract mismatch")
    if manifest["non_authorities"] != [
        "curation",
        "document_selection",
        "eos_insertion",
        "packing",
        "shuffle_order",
        "split_assignment",
    ]:
        raise RawTokenCacheError("Cache non-authority declaration mismatch")
    if manifest["tokenizer"] != runtime.descriptor:
        raise RawTokenCacheError("Cache tokenizer identity mismatch")
    archive_sha, archive_snapshot = _hash_stable_file(job.archive_path, label="raw archive")
    fingerprint_sha, fingerprint_snapshot = _hash_stable_file(
        job.fingerprint_path, label="fingerprint shard"
    )
    expected_source = {
        "archive": {
            "path": job.archive_relative,
            "bucket": job.bucket,
            "index": job.index,
            "bytes": archive_snapshot.size,
            "sha256": archive_sha,
        },
        "preprocess_report": {
            "path": job.report_relative,
            "bytes": job.report_bytes,
            "sha256": job.report_sha256,
            "report_version": 1,
        },
        "fingerprint": {
            "path": job.fingerprint_relative,
            "bytes": fingerprint_snapshot.size,
            "sha256": fingerprint_sha,
            "fingerprint_version": 1,
        },
    }
    if manifest["source"] != expected_source:
        raise RawTokenCacheError("Cache source identity mismatch")
    if archive_sha != job.archive_sha256 or fingerprint_sha != job.fingerprint_sha256:
        raise RawTokenCacheError("Current source checksum differs from preprocess authority")
    documents = manifest.get("documents")
    if not isinstance(documents, dict):
        raise RawTokenCacheError("Cache document descriptor is invalid")
    if (
        documents.get("records") != job.documents
        or documents.get("clean_bytes") != job.clean_bytes
        or documents.get("content_tokens") != job.exact_tokens
    ):
        raise RawTokenCacheError("Cache document totals mismatch")
    alignment = documents.get("alignment")
    if not isinstance(alignment, dict) or (
        alignment.get("authority")
        != "manifest-index+raw-manifest+preprocess-fingerprint"
        or alignment.get("offset_items") != job.documents + 1
        or alignment.get("offset_zero") != 0
        or alignment.get("terminal_offset") != job.exact_tokens
        or SHA256_RE.fullmatch(str(alignment.get("record_sha256"))) is None
    ):
        raise RawTokenCacheError("Cache alignment authority mismatch")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != {"tokens", "offsets"}:
        raise RawTokenCacheError("Cache payload inventory mismatch")
    token_sha, token_items, minimum, maximum = _hash_u16_payload(target / TOKEN_FILE)
    offset_sha, offset_items, terminal = _hash_u64_offsets(target / OFFSET_FILE)
    expected_tokens = {
        "path": TOKEN_FILE,
        "dtype": "uint16",
        "endianness": "little",
        "items": token_items,
        "bytes": token_items * 2,
        "sha256": token_sha,
        "minimum_id": minimum,
        "maximum_id": maximum,
    }
    expected_offsets = {
        "path": OFFSET_FILE,
        "dtype": "uint64",
        "endianness": "little",
        "items": offset_items,
        "bytes": offset_items * 8,
        "sha256": offset_sha,
    }
    if payloads["tokens"] != expected_tokens or payloads["offsets"] != expected_offsets:
        raise RawTokenCacheError("Cache payload descriptor/checksum mismatch")
    if (
        token_items != job.exact_tokens
        or offset_items != job.documents + 1
        or terminal != job.exact_tokens
        or minimum < 0
        or maximum >= runtime.identity.vocab_size
    ):
        raise RawTokenCacheError("Cache payload count/range invariant mismatch")
    builder = manifest.get("builder")
    if not isinstance(builder, dict) or set(builder) != {
        "implementation",
        "implementation_sha256",
        "config",
        "config_sha256",
        "tokenization_contract_sha256",
    }:
        raise RawTokenCacheError("Cache builder identity is invalid")
    if (
        builder.get("implementation") != "pretrain.raw_token_cache"
        or SHA256_RE.fullmatch(str(builder.get("implementation_sha256"))) is None
        or builder.get("config_sha256") != _canonical_sha256(builder.get("config"))
        or builder.get("tokenization_contract_sha256")
        != TOKENIZATION_CONTRACT_SHA256
    ):
        raise RawTokenCacheError("Cache builder/config digest mismatch")
    return manifest


def build_archive_cache(
    job: CacheJob,
    output_root: Path,
    *,
    config: CacheConfig,
    runtime: TokenizerRuntime,
) -> CacheResult:
    config.validate()
    target = job.target(output_root)
    _ensure_safe_directory(target.parent)
    _cleanup_stale_stages(target)
    if target.exists() or target.is_symlink():
        started = time.monotonic()
        validate_completed_cache(job, target, runtime=runtime)
        return CacheResult(
            archive=job.archive_relative,
            target=str(target),
            status="verified",
            documents=job.documents,
            content_tokens=job.exact_tokens,
            elapsed_seconds=round(time.monotonic() - started, 6),
        )
    return _build_new_archive(job, target, config=config, runtime=runtime)


@contextlib.contextmanager
def output_lock(output_root: Path) -> Iterator[None]:
    _ensure_safe_directory(output_root)
    lock_path = output_root / ".raw-token-cache.lock"
    if lock_path.is_symlink():
        raise RawTokenCacheError(f"Unsafe output lock: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RawTokenCacheError(
                f"Another token-cache publisher holds {lock_path}"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


_WORKER_RUNTIME: TokenizerRuntime | None = None


def _initialize_worker(tokenizer_root: str, expected_vocab_size: int) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = _load_tokenizer_runtime(
        Path(tokenizer_root), expected_vocab_size=expected_vocab_size
    )


def _worker_build(job: CacheJob, output_root: Path, config: CacheConfig) -> CacheResult:
    if _WORKER_RUNTIME is None:
        raise RawTokenCacheError("Token-cache worker was not initialized")
    return build_archive_cache(
        job,
        output_root,
        config=config,
        runtime=_WORKER_RUNTIME,
    )


def run_cache_jobs(
    jobs: Sequence[CacheJob],
    output_root: Path,
    tokenizer_root: Path,
    *,
    config: CacheConfig,
    workers: int = 1,
) -> list[CacheResult]:
    """Build/verify archive caches under one exclusive output-root lock."""

    config.validate()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise RawTokenCacheError("workers must be a positive integer")
    targets = [(job.bucket, job.index) for job in jobs]
    archives = [job.archive_relative for job in jobs]
    if len(set(targets)) != len(targets) or len(set(archives)) != len(archives):
        raise RawTokenCacheError("Duplicate archive/target in token-cache job batch")
    output_root = output_root.absolute()
    tokenizer_root = tokenizer_root.absolute()
    with output_lock(output_root):
        implementation_sha256 = sha256_file(Path(__file__))
        initial = _load_tokenizer_runtime(
            tokenizer_root, expected_vocab_size=config.expected_vocab_size
        )
        pending_output_bytes = sum(
            job.estimated_output_bytes
            for job in jobs
            if not job.target(output_root).exists()
        )
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < pending_output_bytes + config.minimum_free_bytes:
            raise RawTokenCacheError(
                "Insufficient output space for pending caches plus safety reserve: "
                f"free={free_bytes}, pending={pending_output_bytes}, "
                f"reserve={config.minimum_free_bytes}"
            )
        results: list[CacheResult] = []
        if workers == 1:
            for job in jobs:
                results.append(
                    build_archive_cache(
                        job,
                        output_root,
                        config=config,
                        runtime=initial,
                    )
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(str(tokenizer_root), config.expected_vocab_size),
            ) as executor:
                futures = {
                    executor.submit(_worker_build, job, output_root, config): job
                    for job in jobs
                }
                for future in as_completed(futures):
                    results.append(future.result())
        try:
            verify_tokenizer_identity(
                tokenizer_root,
                expected_manifest_sha256=initial.identity.manifest_sha256,
                expected_vocabulary_sha256=initial.identity.vocabulary_sha256,
                expected_vocab_size=config.expected_vocab_size,
            )
        except (TokenizerIdentityError, OSError, RuntimeError) as exc:
            raise RawTokenCacheError(f"Tokenizer changed during cache run: {exc}") from exc
        if sha256_file(Path(__file__)) != implementation_sha256:
            raise RawTokenCacheError("Token-cache implementation changed during cache run")
        return sorted(results, key=lambda result: result.archive)


__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_VERSION",
    "CACHE_PROFILE",
    "CacheConfig",
    "CacheJob",
    "CacheResult",
    "RawTokenCacheError",
    "build_archive_cache",
    "discover_cache_jobs",
    "load_cache_job",
    "run_cache_jobs",
    "validate_completed_cache",
]
