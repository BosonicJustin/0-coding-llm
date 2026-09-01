"""Publish and authenticate a closed-world raw-token-cache inventory.

The per-archive cache manifests are deliberately not self-authorizing.  This
module turns a complete set of cache generations into one immutable authority
that a materializer can pin in its journal and final corpus provenance.

Inventory publication takes independently supplied raw/report/fingerprint and
tokenizer authorities, fully opens every cache through
``RawTokenCacheReader``, and only then atomically publishes canonical JSON plus
a SHA-256 sidecar.  Loading is read-only and fail-closed; it also rejects
missing, duplicate, differently ordered, symlinked, pending, or extra cache
archive directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from pretrain.raw_token_cache import (
    CACHE_FORMAT,
    CACHE_FORMAT_VERSION,
    CACHE_PROFILE,
    MANIFEST_FILE as CACHE_MANIFEST_FILE,
    RawTokenCacheError,
    SIDECAR_FILE as CACHE_SIDECAR_FILE,
    output_lock,
)
from pretrain.raw_token_cache_reader import (
    ArchiveAuthority,
    FileAuthority,
    RawTokenCacheAuthority,
    RawTokenCacheReadError,
    RawTokenCacheReader,
    TokenizerAuthority,
)


INVENTORY_FORMAT = "raw-token-cache-inventory"
INVENTORY_FORMAT_VERSION = 1
INVENTORY_MANIFEST_FILE = "manifest.json"
INVENTORY_SIDECAR_FILE = "manifest.sha256"
SELECTION_FORMAT_VERSION = 7
NON_AUTHORITIES = [
    "curation",
    "document_selection",
    "eos_insertion",
    "packing",
    "shuffle_order",
    "split_assignment",
]


class RawTokenCacheInventoryError(RuntimeError):
    """Raised when inventory publication or authentication fails closed."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RawTokenCacheInventoryError(f"{field} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RawTokenCacheInventoryError(f"{field} must be a positive integer")
    return value


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RawTokenCacheInventoryError(f"Unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RawTokenCacheInventoryError(f"Unsafe {field}: {value!r}")
    return value


def _regular_file(path: Path, *, field: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RawTokenCacheInventoryError(f"Missing {field}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RawTokenCacheInventoryError(
            f"{field} is not a non-symlink regular file: {path}"
        )
    return metadata


def _stable_file_bytes(
    path: Path, *, field: str, maximum_bytes: int
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RawTokenCacheInventoryError(f"Cannot open {field}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            raise RawTokenCacheInventoryError(f"{field} size/type is unsafe: {path}")
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, before.st_size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or identity(before) != identity(after)
            or identity(before) != identity(current)
            or stat.S_ISLNK(current.st_mode)
        ):
            raise RawTokenCacheInventoryError(f"{field} changed while read: {path}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _safe_directory(path: Path, *, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RawTokenCacheInventoryError(f"Missing {field}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RawTokenCacheInventoryError(
            f"{field} is not a non-symlink directory: {path}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(8 * 1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def _file_descriptor(path: Path, *, relative: str) -> dict[str, Any]:
    metadata = _regular_file(path, field="inventory-bound file")
    return {
        "path": _safe_relative(relative, field="inventory-bound path"),
        "bytes": metadata.st_size,
        "sha256": _sha256_file(path),
    }


def _expected_cache_directory(archive: ArchiveAuthority) -> str:
    return f"archives/{archive.bucket}/part-{archive.index:06d}"


@dataclass(frozen=True)
class InventorySource:
    """Independent source authority used to certify one cache generation."""

    ordinal: int
    archive: ArchiveAuthority
    preprocess_report: FileAuthority
    fingerprint: FileAuthority
    records: int
    clean_bytes: int
    content_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise RawTokenCacheInventoryError("Inventory ordinal must be non-negative")
        if not isinstance(self.archive, ArchiveAuthority):
            raise RawTokenCacheInventoryError("Inventory archive authority has wrong type")
        if not isinstance(self.preprocess_report, FileAuthority):
            raise RawTokenCacheInventoryError("Inventory report authority has wrong type")
        if not isinstance(self.fingerprint, FileAuthority):
            raise RawTokenCacheInventoryError("Inventory fingerprint authority has wrong type")
        _require_positive_int(self.records, field="inventory records")
        _require_positive_int(self.clean_bytes, field="inventory clean_bytes")
        _require_positive_int(self.content_tokens, field="inventory content_tokens")

    def authority(
        self,
        *,
        cache_manifest_bytes: int,
        cache_manifest_sha256: str,
        tokenizer: TokenizerAuthority,
    ) -> RawTokenCacheAuthority:
        return RawTokenCacheAuthority(
            cache_manifest_bytes=cache_manifest_bytes,
            cache_manifest_sha256=cache_manifest_sha256,
            archive=self.archive,
            preprocess_report=self.preprocess_report,
            fingerprint=self.fingerprint,
            tokenizer=tokenizer,
            records=self.records,
            clean_bytes=self.clean_bytes,
            content_tokens=self.content_tokens,
        )


@dataclass(frozen=True)
class CacheInventoryEntry:
    ordinal: int
    cache_directory: str
    cache_manifest: FileAuthority
    cache_sidecar: FileAuthority
    authority: RawTokenCacheAuthority

    def descriptor(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "cache_directory": self.cache_directory,
            "cache_manifest": asdict(self.cache_manifest),
            "cache_sidecar": asdict(self.cache_sidecar),
            "archive": asdict(self.authority.archive),
            "preprocess_report": asdict(self.authority.preprocess_report),
            "fingerprint": asdict(self.authority.fingerprint),
            "documents": {
                "records": self.authority.records,
                "clean_bytes": self.authority.clean_bytes,
                "content_tokens": self.authority.content_tokens,
            },
        }


@dataclass(frozen=True)
class LoadedRawTokenCacheInventory:
    root: Path
    cache_root: Path
    manifest: Mapping[str, Any]
    manifest_bytes: int
    manifest_sha256: str
    sidecar_bytes: int
    sidecar_sha256: str
    selection_manifest_sha256: str
    tokenizer: TokenizerAuthority
    entries: tuple[CacheInventoryEntry, ...]

    def provenance_descriptor(self) -> dict[str, Any]:
        return {
            "format": INVENTORY_FORMAT,
            "format_version": INVENTORY_FORMAT_VERSION,
            "manifest": {
                "path": INVENTORY_MANIFEST_FILE,
                "bytes": self.manifest_bytes,
                "sha256": self.manifest_sha256,
            },
            "sidecar": {
                "path": INVENTORY_SIDECAR_FILE,
                "bytes": self.sidecar_bytes,
                "sha256": self.sidecar_sha256,
            },
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "archive_count": len(self.entries),
            "archive_inventory_sha256": self.manifest[
                "archive_inventory_sha256"
            ],
        }


def _closed_world_cache_directories(
    cache_root: Path, expected: set[str]
) -> None:
    _safe_directory(cache_root, field="raw-token-cache root")
    archives_root = cache_root / "archives"
    _safe_directory(archives_root, field="raw-token-cache archives root")
    observed: set[str] = set()
    expected_buckets = {PurePosixPath(path).parts[1] for path in expected}
    observed_buckets: set[str] = set()
    for bucket_path in sorted(archives_root.iterdir(), key=lambda item: item.name):
        _safe_directory(bucket_path, field="raw-token-cache bucket")
        observed_buckets.add(bucket_path.name)
        for target in sorted(bucket_path.iterdir(), key=lambda item: item.name):
            _safe_directory(target, field="raw-token-cache archive directory")
            observed.add(target.relative_to(cache_root).as_posix())
    if observed != expected or observed_buckets != expected_buckets:
        raise RawTokenCacheInventoryError(
            "Raw-token-cache archive set is not closed-world: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}, "
            f"missing_buckets={sorted(expected_buckets - observed_buckets)}, "
            f"extra_buckets={sorted(observed_buckets - expected_buckets)}"
        )


def _publish_raw_token_cache_inventory_locked(
    *,
    cache_root: str | Path,
    inventory_root: str | Path,
    dataset_root: str | Path,
    preprocess_root: str | Path,
    tokenizer_root: str | Path,
    selection_manifest_sha256: str,
    selection_format_version: int,
    tokenizer: TokenizerAuthority,
    sources: Sequence[InventorySource],
) -> LoadedRawTokenCacheInventory:
    """Fully certify and atomically publish a complete cache inventory."""

    cache_root = Path(cache_root).absolute()
    inventory_root = Path(inventory_root).absolute()
    dataset_root = Path(dataset_root).absolute()
    preprocess_root = Path(preprocess_root).absolute()
    tokenizer_root = Path(tokenizer_root).absolute()
    _require_sha256(selection_manifest_sha256, field="selection manifest SHA-256")
    if selection_format_version != SELECTION_FORMAT_VERSION:
        raise RawTokenCacheInventoryError("Raw-token-cache inventory requires selection v7")
    if not isinstance(tokenizer, TokenizerAuthority):
        raise RawTokenCacheInventoryError("Inventory tokenizer authority has wrong type")
    if not sources:
        raise RawTokenCacheInventoryError("Cannot publish an empty cache inventory")
    if [source.ordinal for source in sources] != list(range(len(sources))):
        raise RawTokenCacheInventoryError(
            "Cache inventory sources are not in exact contiguous manifest order"
        )
    archives = [source.archive.path for source in sources]
    if len(set(archives)) != len(archives):
        raise RawTokenCacheInventoryError("Duplicate raw archive in cache inventory")
    expected_directories = {
        _expected_cache_directory(source.archive) for source in sources
    }
    if len(expected_directories) != len(sources):
        raise RawTokenCacheInventoryError("Duplicate cache target in inventory")
    _closed_world_cache_directories(cache_root, expected_directories)

    entries: list[CacheInventoryEntry] = []
    for source in sources:
        relative_directory = _expected_cache_directory(source.archive)
        target = cache_root / relative_directory
        manifest_path = target / CACHE_MANIFEST_FILE
        sidecar_path = target / CACHE_SIDECAR_FILE
        manifest_descriptor = _file_descriptor(
            manifest_path,
            relative=f"{relative_directory}/{CACHE_MANIFEST_FILE}",
        )
        sidecar_descriptor = _file_descriptor(
            sidecar_path,
            relative=f"{relative_directory}/{CACHE_SIDECAR_FILE}",
        )
        expected_sidecar = (
            f"{manifest_descriptor['sha256']}  {CACHE_MANIFEST_FILE}\n".encode("ascii")
        )
        if sidecar_path.read_bytes() != expected_sidecar:
            raise RawTokenCacheInventoryError(
                f"Cache manifest sidecar mismatch: {sidecar_path}"
            )
        authority = source.authority(
            cache_manifest_bytes=manifest_descriptor["bytes"],
            cache_manifest_sha256=manifest_descriptor["sha256"],
            tokenizer=tokenizer,
        )
        try:
            with RawTokenCacheReader.open(
                target,
                authority,
                dataset_root=dataset_root,
                preprocess_root=preprocess_root,
                tokenizer_root=tokenizer_root,
            ) as reader:
                reader.verify_unchanged()
        except RawTokenCacheReadError as exc:
            raise RawTokenCacheInventoryError(
                f"Cannot certify cache for {source.archive.path}: {exc}"
            ) from exc
        entries.append(
            CacheInventoryEntry(
                ordinal=source.ordinal,
                cache_directory=relative_directory,
                cache_manifest=FileAuthority(**manifest_descriptor),
                cache_sidecar=FileAuthority(**sidecar_descriptor),
                authority=authority,
            )
        )

    archive_payloads = [entry.descriptor() for entry in entries]
    implementation_path = Path(__file__)
    payload = {
        "format": INVENTORY_FORMAT,
        "format_version": INVENTORY_FORMAT_VERSION,
        "inventory_complete": True,
        "training_ready": False,
        "selection": {
            "identity_format_version": selection_format_version,
            "manifest_sha256": selection_manifest_sha256,
        },
        "cache": {
            "format": CACHE_FORMAT,
            "format_version": CACHE_FORMAT_VERSION,
            "profile": CACHE_PROFILE,
        },
        "tokenizer": tokenizer.descriptor(),
        "archive_count": len(entries),
        "archive_inventory_sha256": _canonical_sha256(archive_payloads),
        "archives": archive_payloads,
        "non_authorities": NON_AUTHORITIES,
        "builder": {
            "implementation": "pretrain.raw_token_cache_inventory",
            "implementation_sha256": _sha256_file(implementation_path),
        },
    }
    manifest_raw = _canonical_json_bytes(payload)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    sidecar_raw = f"{manifest_sha}  {INVENTORY_MANIFEST_FILE}\n".encode("ascii")

    if inventory_root.exists() or inventory_root.is_symlink():
        raise RawTokenCacheInventoryError(
            f"Refusing to replace existing cache inventory: {inventory_root}"
        )
    inventory_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{inventory_root.name}.stage-", dir=inventory_root.parent
        )
    )
    try:
        for name, raw in (
            (INVENTORY_MANIFEST_FILE, manifest_raw),
            (INVENTORY_SIDECAR_FILE, sidecar_raw),
        ):
            with (stage / name).open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(stage, inventory_root)
        parent_fd = os.open(
            inventory_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    return load_raw_token_cache_inventory(
        inventory_root=inventory_root,
        cache_root=cache_root,
    )


def publish_raw_token_cache_inventory(
    *,
    cache_root: str | Path,
    inventory_root: str | Path,
    dataset_root: str | Path,
    preprocess_root: str | Path,
    tokenizer_root: str | Path,
    selection_manifest_sha256: str,
    selection_format_version: int,
    tokenizer: TokenizerAuthority,
    sources: Sequence[InventorySource],
) -> LoadedRawTokenCacheInventory:
    """Lock the cache tree, certify every archive, and publish atomically."""

    cache_root_path = Path(cache_root).absolute()
    try:
        with output_lock(cache_root_path):
            return _publish_raw_token_cache_inventory_locked(
                cache_root=cache_root_path,
                inventory_root=inventory_root,
                dataset_root=dataset_root,
                preprocess_root=preprocess_root,
                tokenizer_root=tokenizer_root,
                selection_manifest_sha256=selection_manifest_sha256,
                selection_format_version=selection_format_version,
                tokenizer=tokenizer,
                sources=sources,
            )
    except RawTokenCacheError as exc:
        raise RawTokenCacheInventoryError(
            f"Cannot acquire stable raw-token-cache publication authority: {exc}"
        ) from exc


def _file_authority(payload: Any, *, field: str) -> FileAuthority:
    if not isinstance(payload, dict) or set(payload) != {"path", "bytes", "sha256"}:
        raise RawTokenCacheInventoryError(f"{field} descriptor schema mismatch")
    try:
        return FileAuthority(**payload)
    except (TypeError, RawTokenCacheReadError) as exc:
        raise RawTokenCacheInventoryError(f"Invalid {field} descriptor: {exc}") from exc


def _archive_authority(payload: Any) -> ArchiveAuthority:
    if not isinstance(payload, dict) or set(payload) != {
        "path", "bucket", "index", "bytes", "sha256"
    }:
        raise RawTokenCacheInventoryError("Archive descriptor schema mismatch")
    try:
        return ArchiveAuthority(**payload)
    except (TypeError, RawTokenCacheReadError) as exc:
        raise RawTokenCacheInventoryError(f"Invalid archive descriptor: {exc}") from exc


def _tokenizer_authority(payload: Any) -> TokenizerAuthority:
    if not isinstance(payload, dict) or payload.get("eos_present_in_payload") is not False:
        raise RawTokenCacheInventoryError("Tokenizer descriptor schema mismatch")
    stripped = dict(payload)
    stripped.pop("eos_present_in_payload", None)
    if set(stripped) != {
        "repo_id", "resolved_revision", "manifest_sha256", "vocabulary_sha256",
        "vocab_size", "eos_token", "eos_token_id",
    }:
        raise RawTokenCacheInventoryError("Tokenizer descriptor schema mismatch")
    try:
        return TokenizerAuthority(**stripped)
    except (TypeError, RawTokenCacheReadError) as exc:
        raise RawTokenCacheInventoryError(f"Invalid tokenizer descriptor: {exc}") from exc


def load_raw_token_cache_inventory(
    *, inventory_root: str | Path, cache_root: str | Path
) -> LoadedRawTokenCacheInventory:
    """Authenticate a published inventory and its closed-world cache tree."""

    inventory_root = Path(inventory_root).absolute()
    cache_root = Path(cache_root).absolute()
    _safe_directory(inventory_root, field="raw-token-cache inventory root")
    names = {entry.name for entry in inventory_root.iterdir()}
    expected_names = {INVENTORY_MANIFEST_FILE, INVENTORY_SIDECAR_FILE}
    if names != expected_names:
        raise RawTokenCacheInventoryError(
            "Cache inventory directory is not closed-world: "
            f"missing={sorted(expected_names - names)}, extra={sorted(names - expected_names)}"
        )
    manifest_path = inventory_root / INVENTORY_MANIFEST_FILE
    sidecar_path = inventory_root / INVENTORY_SIDECAR_FILE
    manifest_raw = _stable_file_bytes(
        manifest_path,
        field="cache inventory manifest",
        maximum_bytes=64 * 1024 * 1024,
    )
    sidecar_raw = _stable_file_bytes(
        sidecar_path,
        field="cache inventory sidecar",
        maximum_bytes=1024,
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    sidecar_sha = hashlib.sha256(sidecar_raw).hexdigest()
    if sidecar_raw != f"{manifest_sha}  {INVENTORY_MANIFEST_FILE}\n".encode("ascii"):
        raise RawTokenCacheInventoryError("Cache inventory sidecar mismatch")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTokenCacheInventoryError(f"Invalid cache inventory JSON: {exc}") from exc
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != manifest_raw:
        raise RawTokenCacheInventoryError("Cache inventory is not canonical JSON")
    expected_top = {
        "format", "format_version", "inventory_complete", "training_ready",
        "selection", "cache", "tokenizer", "archive_count",
        "archive_inventory_sha256", "archives", "non_authorities", "builder",
    }
    if set(manifest) != expected_top:
        raise RawTokenCacheInventoryError("Cache inventory top-level schema mismatch")
    if (
        manifest.get("format") != INVENTORY_FORMAT
        or manifest.get("format_version") != INVENTORY_FORMAT_VERSION
        or manifest.get("inventory_complete") is not True
        or manifest.get("training_ready") is not False
        or manifest.get("cache") != {
            "format": CACHE_FORMAT,
            "format_version": CACHE_FORMAT_VERSION,
            "profile": CACHE_PROFILE,
        }
        or manifest.get("non_authorities") != NON_AUTHORITIES
    ):
        raise RawTokenCacheInventoryError("Cache inventory contract mismatch")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "identity_format_version", "manifest_sha256"
    }:
        raise RawTokenCacheInventoryError("Cache inventory selection schema mismatch")
    if selection.get("identity_format_version") != SELECTION_FORMAT_VERSION:
        raise RawTokenCacheInventoryError("Cache inventory is not bound to selection v7")
    selection_sha = _require_sha256(
        selection.get("manifest_sha256"), field="inventory selection manifest SHA-256"
    )
    tokenizer = _tokenizer_authority(manifest.get("tokenizer"))
    builder = manifest.get("builder")
    if not isinstance(builder, dict) or set(builder) != {
        "implementation", "implementation_sha256"
    } or builder.get("implementation") != "pretrain.raw_token_cache_inventory":
        raise RawTokenCacheInventoryError("Cache inventory builder schema mismatch")
    _require_sha256(builder.get("implementation_sha256"), field="inventory builder SHA-256")

    raw_entries = manifest.get("archives")
    count = _require_positive_int(manifest.get("archive_count"), field="archive count")
    if not isinstance(raw_entries, list) or len(raw_entries) != count:
        raise RawTokenCacheInventoryError("Cache inventory archive count mismatch")
    if _canonical_sha256(raw_entries) != _require_sha256(
        manifest.get("archive_inventory_sha256"),
        field="archive inventory SHA-256",
    ):
        raise RawTokenCacheInventoryError("Cache archive-inventory digest mismatch")
    entries: list[CacheInventoryEntry] = []
    archives_seen: set[str] = set()
    directories_seen: set[str] = set()
    entry_keys = {
        "ordinal", "cache_directory", "cache_manifest", "cache_sidecar",
        "archive", "preprocess_report", "fingerprint", "documents",
    }
    for ordinal, payload in enumerate(raw_entries):
        if not isinstance(payload, dict) or set(payload) != entry_keys:
            raise RawTokenCacheInventoryError("Cache inventory archive schema mismatch")
        if payload.get("ordinal") != ordinal:
            raise RawTokenCacheInventoryError("Cache inventory order/ordinal mismatch")
        archive = _archive_authority(payload.get("archive"))
        if archive.path in archives_seen:
            raise RawTokenCacheInventoryError("Duplicate archive in cache inventory")
        archives_seen.add(archive.path)
        cache_directory = _safe_relative(
            payload.get("cache_directory"), field="cache directory"
        )
        if cache_directory != _expected_cache_directory(archive):
            raise RawTokenCacheInventoryError("Non-canonical cache directory")
        if cache_directory in directories_seen:
            raise RawTokenCacheInventoryError("Duplicate cache directory in inventory")
        directories_seen.add(cache_directory)
        cache_manifest = _file_authority(
            payload.get("cache_manifest"), field="cache manifest"
        )
        cache_sidecar = _file_authority(
            payload.get("cache_sidecar"), field="cache sidecar"
        )
        if cache_manifest.path != f"{cache_directory}/{CACHE_MANIFEST_FILE}" or (
            cache_sidecar.path != f"{cache_directory}/{CACHE_SIDECAR_FILE}"
        ):
            raise RawTokenCacheInventoryError("Non-canonical cache manifest/sidecar path")
        report = _file_authority(
            payload.get("preprocess_report"), field="preprocess report"
        )
        fingerprint = _file_authority(
            payload.get("fingerprint"), field="fingerprint"
        )
        documents = payload.get("documents")
        if not isinstance(documents, dict) or set(documents) != {
            "records", "clean_bytes", "content_tokens"
        }:
            raise RawTokenCacheInventoryError("Cache document totals schema mismatch")
        records = _require_positive_int(documents["records"], field="cache records")
        clean_bytes = _require_positive_int(
            documents["clean_bytes"], field="cache clean_bytes"
        )
        content_tokens = _require_positive_int(
            documents["content_tokens"], field="cache content tokens"
        )
        try:
            authority = RawTokenCacheAuthority(
                cache_manifest_bytes=cache_manifest.bytes,
                cache_manifest_sha256=cache_manifest.sha256,
                archive=archive,
                preprocess_report=report,
                fingerprint=fingerprint,
                tokenizer=tokenizer,
                records=records,
                clean_bytes=clean_bytes,
                content_tokens=content_tokens,
            )
        except RawTokenCacheReadError as exc:
            raise RawTokenCacheInventoryError(
                f"Invalid per-archive cache authority: {exc}"
            ) from exc
        entries.append(
            CacheInventoryEntry(
                ordinal=ordinal,
                cache_directory=cache_directory,
                cache_manifest=cache_manifest,
                cache_sidecar=cache_sidecar,
                authority=authority,
            )
        )
    _closed_world_cache_directories(cache_root, directories_seen)
    # Authenticate the inventory-bound cache manifest and sidecar bytes now;
    # payload/source authentication remains RawTokenCacheReader's job at use.
    for entry in entries:
        for descriptor, label in (
            (entry.cache_manifest, "cache manifest"),
            (entry.cache_sidecar, "cache sidecar"),
        ):
            path = cache_root / descriptor.path
            metadata = _regular_file(path, field=label)
            if metadata.st_size != descriptor.bytes or _sha256_file(path) != descriptor.sha256:
                raise RawTokenCacheInventoryError(
                    f"Inventory-bound {label} changed: {path}"
                )
        expected_sidecar = (
            f"{entry.cache_manifest.sha256}  {CACHE_MANIFEST_FILE}\n".encode("ascii")
        )
        if (cache_root / entry.cache_sidecar.path).read_bytes() != expected_sidecar:
            raise RawTokenCacheInventoryError("Inventory-bound cache sidecar mismatch")
    return LoadedRawTokenCacheInventory(
        root=inventory_root,
        cache_root=cache_root,
        manifest=manifest,
        manifest_bytes=len(manifest_raw),
        manifest_sha256=manifest_sha,
        sidecar_bytes=len(sidecar_raw),
        sidecar_sha256=sidecar_sha,
        selection_manifest_sha256=selection_sha,
        tokenizer=tokenizer,
        entries=tuple(entries),
    )


__all__ = [
    "CacheInventoryEntry",
    "INVENTORY_FORMAT",
    "INVENTORY_FORMAT_VERSION",
    "INVENTORY_MANIFEST_FILE",
    "INVENTORY_SIDECAR_FILE",
    "InventorySource",
    "LoadedRawTokenCacheInventory",
    "RawTokenCacheInventoryError",
    "load_raw_token_cache_inventory",
    "publish_raw_token_cache_inventory",
]
