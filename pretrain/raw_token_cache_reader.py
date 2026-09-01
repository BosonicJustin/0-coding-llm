"""Strict read-only consumer for authenticated raw-document token caches.

This module does not discover or self-authorize cache generations.  A caller
must provide an external :class:`RawTokenCacheAuthority` obtained from a
certified cache inventory.  Opening a reader then authenticates the exact cache
manifest and sidecar, all raw/preprocess/tokenizer source bindings, both binary
payloads, and per-document fingerprint/offset alignment before exposing token
IDs through read-only memory maps.

The reader carries no curation, split, mixture, EOS, packing, shuffle, or
attention-mask semantics.  It only maps one raw archive's ``manifest_index``
to the complete content-token sequence stored by ``raw-document-token-cache``.
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import io
import json
import mmap
import os
import re
import stat
import struct
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, overload

import zstandard

from pretrain.raw_token_cache import (
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    CACHE_PROFILE,
    MANIFEST_FILE,
    OFFSET_FILE,
    OUTPUT_FILES,
    SIDECAR_FILE,
    STARCODE_REPO_ID,
    TOKENIZATION_CONTRACT,
    TOKENIZATION_CONTRACT_SHA256,
    TOKEN_FILE,
)
from pretrain.tokenizer_identity import (
    TokenizerIdentityError,
    verify_tokenizer_identity,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_FINGERPRINT_LINE_BYTES = 1024 * 1024
ALIGNMENT_AUTHORITY = "manifest-index+raw-manifest+preprocess-fingerprint"
NON_AUTHORITIES = [
    "curation",
    "document_selection",
    "eos_insertion",
    "packing",
    "shuffle_order",
    "split_assignment",
]
BUCKET_ARCHIVE_DIRECTORIES = {
    "python": "raw/python",
    "other_code": "raw/other_code",
    "fineweb_edu": "raw/english/fineweb_edu",
    "wikipedia": "raw/english/wikipedia",
}


class RawTokenCacheReadError(RuntimeError):
    """Raised when a cache or one of its external authorities fails closed."""


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RawTokenCacheReadError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RawTokenCacheReadError(f"{field} must be a positive integer")
    return value


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RawTokenCacheReadError(f"Unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RawTokenCacheReadError(f"Unsafe {field}: {value!r}")
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class FileAuthority:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative(self.path, field="source path")
        _require_positive_int(self.bytes, field="source bytes")
        _require_sha256(self.sha256, field="source SHA-256")

    def descriptor(self, *, version_field: str, version: int) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            version_field: version,
        }


@dataclass(frozen=True)
class ArchiveAuthority:
    path: str
    bucket: str
    index: int
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative(self.path, field="archive path")
        if self.bucket not in {"python", "other_code", "fineweb_edu", "wikipedia"}:
            raise RawTokenCacheReadError(f"Unsupported archive bucket: {self.bucket!r}")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise RawTokenCacheReadError("Archive index must be a non-negative integer")
        _require_positive_int(self.bytes, field="archive bytes")
        _require_sha256(self.sha256, field="archive SHA-256")

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bucket": self.bucket,
            "index": self.index,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TokenizerAuthority:
    repo_id: str
    resolved_revision: str
    manifest_sha256: str
    vocabulary_sha256: str
    vocab_size: int
    eos_token: str
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.repo_id != STARCODE_REPO_ID:
            raise RawTokenCacheReadError(
                f"Tokenizer repo must be {STARCODE_REPO_ID!r}"
            )
        if not isinstance(self.resolved_revision, str) or REVISION_RE.fullmatch(
            self.resolved_revision
        ) is None:
            raise RawTokenCacheReadError("Tokenizer revision must be a lowercase commit SHA")
        _require_sha256(self.manifest_sha256, field="tokenizer manifest SHA-256")
        _require_sha256(self.vocabulary_sha256, field="tokenizer vocabulary SHA-256")
        vocab_size = _require_positive_int(self.vocab_size, field="tokenizer vocab_size")
        if vocab_size > 65_536:
            raise RawTokenCacheReadError("Tokenizer vocabulary does not fit uint16")
        if not isinstance(self.eos_token, str) or not self.eos_token:
            raise RawTokenCacheReadError("Tokenizer EOS token must be non-empty")
        if (
            isinstance(self.eos_token_id, bool)
            or not isinstance(self.eos_token_id, int)
            or not 0 <= self.eos_token_id < vocab_size
        ):
            raise RawTokenCacheReadError("Tokenizer EOS token ID is invalid")

    def descriptor(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "eos_present_in_payload": False,
        }


@dataclass(frozen=True)
class RawTokenCacheAuthority:
    """External trust input for opening exactly one archive cache.

    ``cache_manifest_sha256`` is the per-archive cache-generation identity and
    must come from an independently authenticated cache inventory, never from
    the cache directory being opened.
    """

    cache_manifest_bytes: int
    cache_manifest_sha256: str
    archive: ArchiveAuthority
    preprocess_report: FileAuthority
    fingerprint: FileAuthority
    tokenizer: TokenizerAuthority
    records: int
    clean_bytes: int
    content_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.archive, ArchiveAuthority):
            raise RawTokenCacheReadError("archive authority has the wrong type")
        if not isinstance(self.preprocess_report, FileAuthority):
            raise RawTokenCacheReadError("preprocess-report authority has the wrong type")
        if not isinstance(self.fingerprint, FileAuthority):
            raise RawTokenCacheReadError("fingerprint authority has the wrong type")
        if not isinstance(self.tokenizer, TokenizerAuthority):
            raise RawTokenCacheReadError("tokenizer authority has the wrong type")
        _require_positive_int(self.cache_manifest_bytes, field="cache manifest bytes")
        _require_sha256(
            self.cache_manifest_sha256, field="cache manifest SHA-256"
        )
        _require_positive_int(self.records, field="cache authority records")
        _require_positive_int(self.clean_bytes, field="cache authority clean_bytes")
        _require_positive_int(
            self.content_tokens, field="cache authority content_tokens"
        )
        part = f"part-{self.archive.index:06d}"
        expected_archive = f"{BUCKET_ARCHIVE_DIRECTORIES[self.archive.bucket]}/{part}.tar.zst"
        expected_report = f"reports/{self.archive.bucket}/{part}.json"
        expected_fingerprint = (
            f"fingerprints/{self.archive.bucket}/{part}.jsonl.zst"
        )
        if self.archive.path != expected_archive:
            raise RawTokenCacheReadError("Archive authority path is not canonical")
        if self.preprocess_report.path != expected_report:
            raise RawTokenCacheReadError("Preprocess-report authority path is not canonical")
        if self.fingerprint.path != expected_fingerprint:
            raise RawTokenCacheReadError("Fingerprint authority path is not canonical")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "cache_manifest_bytes": self.cache_manifest_bytes,
                "cache_manifest_sha256": self.cache_manifest_sha256,
                "archive": asdict(self.archive),
                "preprocess_report": asdict(self.preprocess_report),
                "fingerprint": asdict(self.fingerprint),
                "tokenizer": asdict(self.tokenizer),
                "records": self.records,
                "clean_bytes": self.clean_bytes,
                "content_tokens": self.content_tokens,
            }
        )


@dataclass(frozen=True)
class _FileState:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _state_from_stat(metadata: os.stat_result) -> _FileState:
    return _FileState(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _open_readonly(path: Path, *, label: str) -> tuple[int, _FileState]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RawTokenCacheReadError(f"Cannot open {label} read-only: {path}: {exc}") from exc
    state = _state_from_stat(os.fstat(descriptor))
    if not stat.S_ISREG(state.mode):
        os.close(descriptor)
        raise RawTokenCacheReadError(f"{label} is not a regular file: {path}")
    return descriptor, state


def _assert_fd_unchanged(descriptor: int, state: _FileState, *, label: str) -> None:
    if _state_from_stat(os.fstat(descriptor)) != state:
        raise RawTokenCacheReadError(f"{label} changed while being read")


def _assert_path_unchanged(path: Path, state: _FileState, *, label: str) -> None:
    try:
        current = _state_from_stat(path.lstat())
    except OSError as exc:
        raise RawTokenCacheReadError(f"Cannot recheck {label}: {path}: {exc}") from exc
    if current != state or stat.S_ISLNK(current.mode):
        raise RawTokenCacheReadError(f"{label} changed while being read: {path}")


def _safe_file_under(root: Path, relative: str, *, label: str) -> Path:
    relative = _safe_relative(relative, field=f"{label} path")
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise RawTokenCacheReadError(f"Cannot inspect {label} root: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RawTokenCacheReadError(f"Unsafe {label} root: {root}")
    cursor = root
    for component in PurePosixPath(relative).parts:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise RawTokenCacheReadError(f"Missing {label}: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RawTokenCacheReadError(f"Symlink in {label} path: {cursor}")
    if not stat.S_ISREG(cursor.lstat().st_mode):
        raise RawTokenCacheReadError(f"{label} is not a regular file: {cursor}")
    return cursor


def _hash_descriptor(
    descriptor: int,
    state: _FileState,
    *,
    label: str,
    block_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = state.size
    while remaining:
        payload = os.read(descriptor, min(block_size, remaining))
        if not payload:
            raise RawTokenCacheReadError(f"Short {label} during authentication")
        digest.update(payload)
        remaining -= len(payload)
    if os.read(descriptor, 1):
        raise RawTokenCacheReadError(f"Growing {label} during authentication")
    _assert_fd_unchanged(descriptor, state, label=label)
    return digest.hexdigest()


def _read_stable_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, _FileState]:
    descriptor, state = _open_readonly(path, label=label)
    try:
        if state.size < 1 or state.size > maximum_bytes:
            raise RawTokenCacheReadError(f"{label} size is outside its safety bound")
        payload = bytearray()
        while len(payload) < state.size:
            chunk = os.read(descriptor, min(1024 * 1024, state.size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != state.size or os.read(descriptor, 1):
            raise RawTokenCacheReadError(f"Short or growing {label}: {path}")
        _assert_fd_unchanged(descriptor, state, label=label)
        _assert_path_unchanged(path, state, label=label)
        return bytes(payload), state
    finally:
        os.close(descriptor)


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


class TokenSpan(Sequence[int]):
    """A bounds-checked, immutable view of one complete document's token IDs."""

    __slots__ = ("_reader", "manifest_index", "start", "end")

    def __init__(
        self,
        reader: "RawTokenCacheReader",
        manifest_index: int,
        start: int,
        end: int,
    ):
        self._reader = reader
        self.manifest_index = manifest_index
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        self._reader._require_open()
        length = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(length)
            if step == 1:
                return self._reader._copy_token_range(self.start + start, self.start + stop)
            return [self[position] for position in range(start, stop, step)]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("TokenSpan index must be an integer or slice")
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("TokenSpan index out of range")
        assert self._reader._tokens_map is not None
        return struct.unpack_from(
            "<H", self._reader._tokens_map, (self.start + index) * 2
        )[0]

    def __iter__(self) -> Iterator[int]:
        self._reader._require_open()
        assert self._reader._tokens_map is not None
        payload = memoryview(self._reader._tokens_map)[self.start * 2 : self.end * 2]
        try:
            if sys.byteorder == "little":
                values = payload.cast("H")
                try:
                    yield from values
                finally:
                    values.release()
            else:  # pragma: no cover - production hosts are little-endian
                for offset in range(0, len(payload), 2):
                    yield struct.unpack_from("<H", payload, offset)[0]
        finally:
            payload.release()

    def to_list(self) -> list[int]:
        self._reader._require_open()
        return self._reader._copy_token_range(self.start, self.end)


class RawTokenCacheReader:
    """Authenticated read-only mmap access to one complete archive cache."""

    def __init__(
        self,
        cache_directory: Path,
        authority: RawTokenCacheAuthority,
        *,
        dataset_root: Path,
        preprocess_root: Path,
        tokenizer_root: Path,
    ):
        self.cache_directory = cache_directory.absolute()
        self.authority = authority
        self.dataset_root = dataset_root.absolute()
        self.preprocess_root = preprocess_root.absolute()
        self.tokenizer_root = tokenizer_root.absolute()
        self.manifest: dict[str, Any] = {}
        self._tokens_fd: int | None = None
        self._offsets_fd: int | None = None
        self._tokens_state: _FileState | None = None
        self._offsets_state: _FileState | None = None
        self._tokens_map: mmap.mmap | None = None
        self._offsets_map: mmap.mmap | None = None
        self._closed = True
        try:
            self._open_and_authenticate()
        except BaseException:
            self._close_resources()
            raise

    @classmethod
    def open(
        cls,
        cache_directory: Path,
        authority: RawTokenCacheAuthority,
        *,
        dataset_root: Path,
        preprocess_root: Path,
        tokenizer_root: Path,
    ) -> "RawTokenCacheReader":
        return cls(
            cache_directory,
            authority,
            dataset_root=dataset_root,
            preprocess_root=preprocess_root,
            tokenizer_root=tokenizer_root,
        )

    @property
    def records(self) -> int:
        return self.authority.records

    @property
    def content_tokens(self) -> int:
        return self.authority.content_tokens

    @property
    def cache_manifest_sha256(self) -> str:
        return self.authority.cache_manifest_sha256

    def _open_and_authenticate(self) -> None:
        self._validate_cache_directory()
        manifest_raw, _manifest_state = _read_stable_file(
            self.cache_directory / MANIFEST_FILE,
            label="cache manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        if (
            len(manifest_raw) != self.authority.cache_manifest_bytes
            or hashlib.sha256(manifest_raw).hexdigest()
            != self.authority.cache_manifest_sha256
        ):
            raise RawTokenCacheReadError("Wrong cache generation/manifest identity")
        sidecar_raw, _sidecar_state = _read_stable_file(
            self.cache_directory / SIDECAR_FILE,
            label="cache manifest sidecar",
            maximum_bytes=1024,
        )
        expected_sidecar = (
            f"{self.authority.cache_manifest_sha256}  {MANIFEST_FILE}\n".encode("ascii")
        )
        if sidecar_raw != expected_sidecar:
            raise RawTokenCacheReadError("Cache manifest sidecar mismatch")
        try:
            manifest = json.loads(manifest_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawTokenCacheReadError(f"Invalid cache manifest JSON: {exc}") from exc
        if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != manifest_raw:
            raise RawTokenCacheReadError("Cache manifest is not canonical JSON")
        self._validate_manifest(manifest)
        self.manifest = manifest
        self._authenticate_source_files()
        self._authenticate_tokenizer()
        self._open_and_validate_payloads()
        # Internal alignment replay uses the same bounds-checked offset API as
        # callers. Resources are fully authenticated enough to enable that API
        # here; any later failure still closes everything in __init__.
        self._closed = False
        self._validate_fingerprint_alignment()
        self.verify_unchanged()

    def _validate_cache_directory(self) -> None:
        try:
            metadata = self.cache_directory.lstat()
        except OSError as exc:
            raise RawTokenCacheReadError(
                f"Cannot inspect cache directory: {self.cache_directory}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RawTokenCacheReadError(
                f"Cache path is not a non-symlink directory: {self.cache_directory}"
            )
        names = {entry.name for entry in self.cache_directory.iterdir()}
        if names != OUTPUT_FILES:
            raise RawTokenCacheReadError(
                "Cache directory closed-world inventory mismatch: "
                f"missing={sorted(OUTPUT_FILES - names)}, "
                f"extra={sorted(names - OUTPUT_FILES)}"
            )

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        expected_keys = {
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
        if set(manifest) != expected_keys:
            raise RawTokenCacheReadError("Cache manifest top-level schema mismatch")
        if (
            manifest.get("format") != CACHE_FORMAT
            or manifest.get("format_version") != CACHE_FORMAT_VERSION
            or manifest.get("profile") != CACHE_PROFILE
            or manifest.get("cache_complete") is not True
            or manifest.get("training_ready") is not False
            or manifest.get("tokenization") != TOKENIZATION_CONTRACT
            or manifest.get("non_authorities") != NON_AUTHORITIES
        ):
            raise RawTokenCacheReadError("Cache semantic format/profile mismatch")
        expected_source = {
            "archive": self.authority.archive.descriptor(),
            "preprocess_report": self.authority.preprocess_report.descriptor(
                version_field="report_version", version=1
            ),
            "fingerprint": self.authority.fingerprint.descriptor(
                version_field="fingerprint_version", version=1
            ),
        }
        if manifest.get("source") != expected_source:
            raise RawTokenCacheReadError("Cache source binding mismatch")
        if manifest.get("tokenizer") != self.authority.tokenizer.descriptor():
            raise RawTokenCacheReadError("Cache tokenizer binding mismatch")
        documents = manifest.get("documents")
        if not isinstance(documents, Mapping) or set(documents) != {
            "records",
            "clean_bytes",
            "content_tokens",
            "alignment",
        }:
            raise RawTokenCacheReadError("Cache document schema mismatch")
        if (
            documents.get("records") != self.authority.records
            or documents.get("clean_bytes") != self.authority.clean_bytes
            or documents.get("content_tokens") != self.authority.content_tokens
        ):
            raise RawTokenCacheReadError("Cache document totals mismatch")
        alignment = documents.get("alignment")
        if not isinstance(alignment, Mapping) or set(alignment) != {
            "authority",
            "record_sha256",
            "offset_items",
            "offset_zero",
            "terminal_offset",
        }:
            raise RawTokenCacheReadError("Cache alignment schema mismatch")
        if (
            alignment.get("authority") != ALIGNMENT_AUTHORITY
            or alignment.get("offset_items") != self.authority.records + 1
            or alignment.get("offset_zero") != 0
            or alignment.get("terminal_offset") != self.authority.content_tokens
        ):
            raise RawTokenCacheReadError("Cache alignment totals mismatch")
        _require_sha256(alignment.get("record_sha256"), field="alignment SHA-256")
        self._validate_payload_descriptors(manifest.get("payloads"))
        self._validate_builder(manifest.get("builder"))

    def _validate_payload_descriptors(self, payloads: Any) -> None:
        if not isinstance(payloads, Mapping) or set(payloads) != {"tokens", "offsets"}:
            raise RawTokenCacheReadError("Cache payload inventory mismatch")
        tokens = payloads["tokens"]
        offsets = payloads["offsets"]
        if not isinstance(tokens, Mapping) or set(tokens) != {
            "path",
            "dtype",
            "endianness",
            "items",
            "bytes",
            "sha256",
            "minimum_id",
            "maximum_id",
        }:
            raise RawTokenCacheReadError("Token payload schema mismatch")
        if (
            tokens.get("path") != TOKEN_FILE
            or tokens.get("dtype") != "uint16"
            or tokens.get("endianness") != "little"
            or tokens.get("items") != self.authority.content_tokens
            or tokens.get("bytes") != self.authority.content_tokens * 2
        ):
            raise RawTokenCacheReadError("Token payload dtype/count/order mismatch")
        _require_sha256(tokens.get("sha256"), field="token payload SHA-256")
        minimum = tokens.get("minimum_id")
        maximum = tokens.get("maximum_id")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 0 <= minimum <= maximum < self.authority.tokenizer.vocab_size
        ):
            raise RawTokenCacheReadError("Token payload ID range is invalid")
        if not isinstance(offsets, Mapping) or set(offsets) != {
            "path",
            "dtype",
            "endianness",
            "items",
            "bytes",
            "sha256",
        }:
            raise RawTokenCacheReadError("Offset payload schema mismatch")
        if (
            offsets.get("path") != OFFSET_FILE
            or offsets.get("dtype") != "uint64"
            or offsets.get("endianness") != "little"
            or offsets.get("items") != self.authority.records + 1
            or offsets.get("bytes") != (self.authority.records + 1) * 8
        ):
            raise RawTokenCacheReadError("Offset payload dtype/count/order mismatch")
        _require_sha256(offsets.get("sha256"), field="offset payload SHA-256")

    @staticmethod
    def _validate_builder(builder: Any) -> None:
        if not isinstance(builder, Mapping) or set(builder) != {
            "implementation",
            "implementation_sha256",
            "config",
            "config_sha256",
            "tokenization_contract_sha256",
        }:
            raise RawTokenCacheReadError("Cache builder schema mismatch")
        if builder.get("implementation") != "pretrain.raw_token_cache":
            raise RawTokenCacheReadError("Unknown cache builder implementation")
        _require_sha256(
            builder.get("implementation_sha256"), field="builder implementation SHA-256"
        )
        config = builder.get("config")
        if not isinstance(config, Mapping) or not config:
            raise RawTokenCacheReadError("Cache builder config is invalid")
        if builder.get("config_sha256") != _canonical_sha256(config):
            raise RawTokenCacheReadError("Cache builder config digest mismatch")
        if builder.get("tokenization_contract_sha256") != TOKENIZATION_CONTRACT_SHA256:
            raise RawTokenCacheReadError("Cache tokenization contract digest mismatch")

    def _authenticate_source_files(self) -> None:
        archive_path = _safe_file_under(
            self.dataset_root, self.authority.archive.path, label="raw archive"
        )
        report_path = _safe_file_under(
            self.preprocess_root,
            self.authority.preprocess_report.path,
            label="preprocess report",
        )
        fingerprint_path = _safe_file_under(
            self.preprocess_root,
            self.authority.fingerprint.path,
            label="fingerprint shard",
        )
        self._authenticate_file(
            archive_path,
            expected_bytes=self.authority.archive.bytes,
            expected_sha256=self.authority.archive.sha256,
            label="raw archive",
        )
        report_raw = self._authenticate_file(
            report_path,
            expected_bytes=self.authority.preprocess_report.bytes,
            expected_sha256=self.authority.preprocess_report.sha256,
            label="preprocess report",
            return_bytes=True,
        )
        assert report_raw is not None
        self._validate_preprocess_report(report_raw)
        # The fingerprint is authenticated while its manifest-index alignment
        # is replayed after payload mmap validation, avoiding a duplicate pass.
        fingerprint_state = fingerprint_path.lstat()
        if fingerprint_state.st_size != self.authority.fingerprint.bytes:
            raise RawTokenCacheReadError("Fingerprint byte count mismatch")
        self._fingerprint_path = fingerprint_path

    @staticmethod
    def _authenticate_file(
        path: Path,
        *,
        expected_bytes: int,
        expected_sha256: str,
        label: str,
        return_bytes: bool = False,
    ) -> bytes | None:
        descriptor, state = _open_readonly(path, label=label)
        try:
            if state.size != expected_bytes:
                raise RawTokenCacheReadError(f"{label} byte count mismatch")
            if return_bytes:
                if state.size > MAX_REPORT_BYTES:
                    raise RawTokenCacheReadError(f"{label} exceeds its safety bound")
                payload = bytearray()
                remaining = state.size
                while remaining:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        raise RawTokenCacheReadError(
                            f"Short {label} during authentication"
                        )
                    payload.extend(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise RawTokenCacheReadError(
                        f"Growing {label} during authentication"
                    )
                digest = hashlib.sha256(payload).hexdigest()
                result: bytes | None = bytes(payload)
            else:
                digest = _hash_descriptor(descriptor, state, label=label)
                result = None
            if digest != expected_sha256:
                raise RawTokenCacheReadError(f"{label} SHA-256 mismatch")
            _assert_fd_unchanged(descriptor, state, label=label)
            _assert_path_unchanged(path, state, label=label)
            return result
        finally:
            os.close(descriptor)

    def _validate_preprocess_report(self, raw: bytes) -> None:
        try:
            report = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawTokenCacheReadError(f"Invalid preprocess report: {exc}") from exc
        if not isinstance(report, Mapping):
            raise RawTokenCacheReadError("Preprocess report root is not an object")
        expected = {
            "report_version": 1,
            "fingerprint_version": 1,
            "archive": self.authority.archive.path,
            "archive_sha256": self.authority.archive.sha256,
            "archive_compressed_bytes": self.authority.archive.bytes,
            "bucket": self.authority.archive.bucket,
            "index": self.authority.archive.index,
            "fingerprint_file": self.authority.fingerprint.path,
            "fingerprint_sha256": self.authority.fingerprint.sha256,
            "documents": self.authority.records,
            "clean_bytes": self.authority.clean_bytes,
            "exact_tokens": self.authority.content_tokens,
        }
        for field, value in expected.items():
            if report.get(field) != value:
                raise RawTokenCacheReadError(
                    f"Preprocess report {field} binding mismatch"
                )

    def _authenticate_tokenizer(self) -> None:
        expected = self.authority.tokenizer
        try:
            identity = verify_tokenizer_identity(
                self.tokenizer_root,
                expected_manifest_sha256=expected.manifest_sha256,
                expected_vocabulary_sha256=expected.vocabulary_sha256,
                expected_vocab_size=expected.vocab_size,
            )
        except (TokenizerIdentityError, OSError, RuntimeError) as exc:
            raise RawTokenCacheReadError(f"Tokenizer authentication failed: {exc}") from exc
        manifest = identity.manifest
        validation = manifest.get("validation")
        if (
            manifest.get("repo_id") != expected.repo_id
            or manifest.get("resolved_revision") != expected.resolved_revision
            or not isinstance(validation, Mapping)
            or validation.get("eos_token") != expected.eos_token
            or validation.get("eos_token_id") != expected.eos_token_id
        ):
            raise RawTokenCacheReadError("Tokenizer semantic identity mismatch")

    def _open_and_validate_payloads(self) -> None:
        tokens_path = self.cache_directory / TOKEN_FILE
        offsets_path = self.cache_directory / OFFSET_FILE
        tokens_fd, tokens_state = _open_readonly(tokens_path, label="token payload")
        self._tokens_fd = tokens_fd
        self._tokens_state = tokens_state
        offsets_fd, offsets_state = _open_readonly(offsets_path, label="offset payload")
        self._offsets_fd = offsets_fd
        self._offsets_state = offsets_state
        expected_tokens_bytes = self.authority.content_tokens * 2
        expected_offsets_bytes = (self.authority.records + 1) * 8
        if tokens_state.size != expected_tokens_bytes:
            raise RawTokenCacheReadError("Token payload byte count mismatch")
        if offsets_state.size != expected_offsets_bytes:
            raise RawTokenCacheReadError("Offset payload byte count mismatch")
        self._tokens_map = mmap.mmap(tokens_fd, 0, access=mmap.ACCESS_READ)
        self._offsets_map = mmap.mmap(offsets_fd, 0, access=mmap.ACCESS_READ)
        token_sha, token_minimum, token_maximum = self._scan_tokens()
        offset_sha = self._scan_offsets()
        payloads = self.manifest["payloads"]
        if (
            token_sha != payloads["tokens"]["sha256"]
            or token_minimum != payloads["tokens"]["minimum_id"]
            or token_maximum != payloads["tokens"]["maximum_id"]
        ):
            raise RawTokenCacheReadError("Token payload hash/range corruption")
        if offset_sha != payloads["offsets"]["sha256"]:
            raise RawTokenCacheReadError("Offset payload hash corruption")

    def _scan_tokens(self) -> tuple[str, int, int]:
        assert self._tokens_map is not None
        digest = hashlib.sha256()
        minimum = 65_536
        maximum = -1
        chunk_bytes = 8 * 1024 * 1024
        for start in range(0, len(self._tokens_map), chunk_bytes):
            payload = self._tokens_map[start : start + chunk_bytes]
            digest.update(payload)
            values = array.array("H")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            if values:
                minimum = min(minimum, min(values))
                maximum = max(maximum, max(values))
        if minimum > maximum or maximum >= self.authority.tokenizer.vocab_size:
            raise RawTokenCacheReadError("Token payload contains invalid IDs")
        assert self._tokens_fd is not None and self._tokens_state is not None
        _assert_fd_unchanged(self._tokens_fd, self._tokens_state, label="token payload")
        return digest.hexdigest(), minimum, maximum

    def _scan_offsets(self) -> str:
        assert self._offsets_map is not None
        digest = hashlib.sha256(self._offsets_map).hexdigest()
        previous = -1
        for index in range(self.authority.records + 1):
            value = struct.unpack_from("<Q", self._offsets_map, index * 8)[0]
            if index == 0:
                if value != 0:
                    raise RawTokenCacheReadError("First document offset is not zero")
            elif value <= previous:
                raise RawTokenCacheReadError(
                    f"Document offsets are not strictly increasing at index {index}"
                )
            previous = value
        if previous != self.authority.content_tokens:
            raise RawTokenCacheReadError("Terminal document offset mismatch")
        assert self._offsets_fd is not None and self._offsets_state is not None
        _assert_fd_unchanged(
            self._offsets_fd, self._offsets_state, label="offset payload"
        )
        return digest

    def _validate_fingerprint_alignment(self) -> None:
        path = self._fingerprint_path
        descriptor, state = _open_readonly(path, label="fingerprint shard")
        if state.size != self.authority.fingerprint.bytes:
            os.close(descriptor)
            raise RawTokenCacheReadError("Fingerprint byte count changed")
        digest = hashlib.sha256()
        alignment = hashlib.sha256()
        records = 0
        clean_bytes = 0
        content_tokens = 0
        raw = os.fdopen(descriptor, "rb", closefd=False)
        hashing = _HashingReader(raw, digest)
        decompressor = zstandard.ZstdDecompressor().stream_reader(
            hashing, read_across_frames=True, closefd=False
        )
        buffered = io.BufferedReader(decompressor, buffer_size=1024 * 1024)
        try:
            for index in range(self.authority.records):
                raw_line = buffered.readline(MAX_FINGERPRINT_LINE_BYTES + 1)
                if not raw_line:
                    raise RawTokenCacheReadError("Fingerprint has too few rows")
                if len(raw_line) > MAX_FINGERPRINT_LINE_BYTES:
                    raise RawTokenCacheReadError("Fingerprint row exceeds safety bound")
                if not raw_line.endswith(b"\n") or not raw_line.strip():
                    raise RawTokenCacheReadError(
                        "Fingerprint contains a blank or unterminated row"
                    )
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RawTokenCacheReadError(
                        f"Invalid fingerprint row {index}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise RawTokenCacheReadError("Fingerprint row is not an object")
                member_path = _safe_relative(
                    row.get("member_path"), field="fingerprint member_path"
                )
                expected_doc_id = hashlib.sha256(
                    f"{self.authority.archive.path}\0{member_path}".encode("utf-8")
                ).hexdigest()
                expected = {
                    "record_version": 1,
                    "fingerprint_version": 1,
                    "archive": self.authority.archive.path,
                    "archive_index": self.authority.archive.index,
                    "bucket": self.authority.archive.bucket,
                    "manifest_index": index,
                    "doc_id": expected_doc_id,
                }
                if any(row.get(field) != value for field, value in expected.items()):
                    raise RawTokenCacheReadError(
                        f"Fingerprint manifest order/identity mismatch at row {index}"
                    )
                size_bytes = _require_positive_int(
                    row.get("size_bytes"), field=f"fingerprint size row {index}"
                )
                token_count = _require_positive_int(
                    row.get("starcoder2_tokens"),
                    field=f"fingerprint tokens row {index}",
                )
                content_sha = _require_sha256(
                    row.get("content_sha256"),
                    field=f"fingerprint content SHA-256 row {index}",
                )
                start, end = self.document_offsets(index)
                if end - start != token_count:
                    raise RawTokenCacheReadError(
                        f"Fingerprint/cache token-span mismatch at row {index}"
                    )
                alignment.update(
                    _canonical_json_bytes(
                        {
                            "manifest_index": index,
                            "member_path": member_path,
                            "content_sha256": content_sha,
                            "token_start": start,
                            "token_end": end,
                        }
                    )
                )
                records += 1
                clean_bytes += size_bytes
                content_tokens += token_count
            if buffered.readline(MAX_FINGERPRINT_LINE_BYTES + 1):
                raise RawTokenCacheReadError("Fingerprint has too many rows")
            while buffered.read(8 * 1024 * 1024):
                pass
            while hashing.read(8 * 1024 * 1024):
                pass
        except zstandard.ZstdError as exc:
            raise RawTokenCacheReadError(f"Corrupt fingerprint shard: {exc}") from exc
        finally:
            with contextlib.suppress(Exception):
                buffered.close()
            with contextlib.suppress(Exception):
                decompressor.close()
            raw.close()
            os.close(descriptor)
        if digest.hexdigest() != self.authority.fingerprint.sha256:
            raise RawTokenCacheReadError("Fingerprint SHA-256 mismatch")
        if (
            records != self.authority.records
            or clean_bytes != self.authority.clean_bytes
            or content_tokens != self.authority.content_tokens
        ):
            raise RawTokenCacheReadError("Fingerprint aggregate totals mismatch")
        expected_alignment = self.manifest["documents"]["alignment"]["record_sha256"]
        if alignment.hexdigest() != expected_alignment:
            raise RawTokenCacheReadError("Fingerprint/cache alignment digest mismatch")
        _assert_path_unchanged(path, state, label="fingerprint shard")

    def _require_open(self) -> None:
        if self._closed or self._tokens_map is None or self._offsets_map is None:
            raise RawTokenCacheReadError("Raw token cache reader is closed")

    def verify_unchanged(self) -> None:
        self._require_open()
        assert self._tokens_fd is not None and self._tokens_state is not None
        assert self._offsets_fd is not None and self._offsets_state is not None
        _assert_fd_unchanged(self._tokens_fd, self._tokens_state, label="token payload")
        _assert_fd_unchanged(
            self._offsets_fd, self._offsets_state, label="offset payload"
        )

    def document_offsets(self, manifest_index: int) -> tuple[int, int]:
        self._require_open()
        if (
            isinstance(manifest_index, bool)
            or not isinstance(manifest_index, int)
            or not 0 <= manifest_index < self.authority.records
        ):
            raise IndexError("manifest_index is outside this archive cache")
        assert self._offsets_map is not None
        start = struct.unpack_from("<Q", self._offsets_map, manifest_index * 8)[0]
        end = struct.unpack_from("<Q", self._offsets_map, (manifest_index + 1) * 8)[0]
        return start, end

    def document(self, manifest_index: int) -> TokenSpan:
        start, end = self.document_offsets(manifest_index)
        return TokenSpan(self, manifest_index, start, end)

    def document_ids(self, manifest_index: int) -> list[int]:
        return self.document(manifest_index).to_list()

    def _copy_token_range(self, start: int, end: int) -> list[int]:
        self._require_open()
        assert self._tokens_map is not None
        values = array.array("H")
        values.frombytes(self._tokens_map[start * 2 : end * 2])
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()

    def iter_documents(
        self, *, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, TokenSpan]]:
        self._require_open()
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("Sequential iteration start must be non-negative")
        if stop is None:
            stop = self.authority.records
        if (
            isinstance(stop, bool)
            or not isinstance(stop, int)
            or stop < start
            or stop > self.authority.records
        ):
            raise ValueError("Sequential iteration stop is invalid")
        for manifest_index in range(start, stop):
            yield manifest_index, self.document(manifest_index)

    def _close_resources(self) -> None:
        self._closed = True
        for mapping_name in ("_tokens_map", "_offsets_map"):
            mapping = getattr(self, mapping_name)
            if mapping is not None:
                with contextlib.suppress(Exception):
                    mapping.close()
                setattr(self, mapping_name, None)
        for descriptor_name in ("_tokens_fd", "_offsets_fd"):
            descriptor = getattr(self, descriptor_name)
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(self, descriptor_name, None)

    def close(self) -> None:
        self._close_resources()

    def __enter__(self) -> "RawTokenCacheReader":
        self._require_open()
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "ArchiveAuthority",
    "FileAuthority",
    "RawTokenCacheAuthority",
    "RawTokenCacheReadError",
    "RawTokenCacheReader",
    "TokenSpan",
    "TokenizerAuthority",
]
