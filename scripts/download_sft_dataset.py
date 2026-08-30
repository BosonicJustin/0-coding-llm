#!/usr/bin/env python3
"""Resumably download and certify the raw OpenCodeInstruct SFT snapshot.

The destination is deliberately a caller-supplied, dedicated post-training
root. Raw repository files remain under ``ROOT/raw``; deterministic SOURCE and
COMPLETION authorities are published only after the complete pinned inventory
has passed file-count, byte-count, Parquet-row-count, and SHA-256 checks.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_REPO_ID = "nvidia/OpenCodeInstruct"
DEFAULT_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
REPO_TYPE = "dataset"
EXPECTED_TRAIN_PARQUET_FILES = 50
EXPECTED_COMPRESSED_DOWNLOAD_BYTES = 6_861_113_102
EXPECTED_ROWS = 5_000_000
RAW_SUBDIRECTORY = "raw"
SOURCE_MANIFEST_NAME = "SOURCE.json"
COMPLETION_MANIFEST_NAME = "COMPLETION.json"
REPOSITORY_METADATA_FILES = ("README.md", ".gitattributes")
ALLOW_PATTERNS = ("data/train-*.parquet", *REPOSITORY_METADATA_FILES)
SOURCE_MANIFEST_VERSION = 1
COMPLETION_MANIFEST_VERSION = 1
INVENTORY_VERSION = 1
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_MAX_WORKERS = 128


class SFTDownloadError(RuntimeError):
    """Raised when download, inventory, or publication cannot be trusted."""


@dataclass(frozen=True)
class DatasetContract:
    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    repo_type: str = REPO_TYPE
    expected_files: int = EXPECTED_TRAIN_PARQUET_FILES
    expected_bytes: int = EXPECTED_COMPRESSED_DOWNLOAD_BYTES
    expected_rows: int = EXPECTED_ROWS


DEFAULT_CONTRACT = DatasetContract()
SnapshotDownload = Callable[..., str]
RowCounter = Callable[[Path], int]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def emit_event(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_if_missing_or_identical(path: Path, payload: bytes) -> bool:
    """Publish bytes atomically, refusing to replace an existing authority."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise SFTDownloadError(f"Manifest path is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise SFTDownloadError(
                f"Existing authority differs from the verified snapshot: {path}; "
                "use a new post-training root instead of overwriting it"
            )
        fsync_directory(path.parent)
        return False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
        return True
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _pretraining_markers(path: Path) -> tuple[Path, ...]:
    return (
        path / "manifests" / "STACK_V3_SOURCE.json",
        path / "state" / "COLLECTION_COMPLETE.json",
        path / "state" / "ENGLISH_FINEWEB_EDU_COMPLETE.json",
        path / "state" / "ENGLISH_WIKIPEDIA_COMPLETE.json",
        path / "raw" / "python",
        path / "raw" / "other_code",
        path / "raw" / "english",
        path / "staging" / "preprocess",
    )


def prepare_posttraining_root(root: Path) -> Path:
    """Create and validate a dedicated root that is not pre-training storage."""
    if root.exists() and root.is_symlink():
        raise SFTDownloadError(f"Post-training root cannot be a symlink: {root}")
    resolved = root.resolve(strict=False)
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise SFTDownloadError(f"Refusing broad post-training root: {resolved}")

    # A nested directory inside a pre-training root is still the wrong storage
    # boundary, so check the requested root and each existing ancestor.
    for candidate in (resolved, *resolved.parents):
        found = [marker for marker in _pretraining_markers(candidate) if marker.exists()]
        if found:
            raise SFTDownloadError(
                "Post-training data must not be placed in or below a pre-training "
                f"root; found marker {found[0]}"
            )

    resolved.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise SFTDownloadError(f"Post-training root is not a real directory: {resolved}")
    raw = resolved / RAW_SUBDIRECTORY
    if raw.exists() and (raw.is_symlink() or not raw.is_dir()):
        raise SFTDownloadError(f"Raw snapshot path is unsafe: {raw}")
    return resolved


def _sha256_regular_file(path: Path) -> tuple[str, os.stat_result]:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise SFTDownloadError(f"Expected a regular, non-symlink shard: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise SFTDownloadError(f"Shard changed while it was opened: {path}")
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ):
        raise SFTDownloadError(f"Shard changed during SHA-256 calculation: {path}")
    return digest.hexdigest(), after


def _read_parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SFTDownloadError(
            "Parquet inventory requires pyarrow (installed by requirements-data.txt)"
        ) from exc
    try:
        rows = int(parquet.read_metadata(path).num_rows)
    except Exception as exc:
        raise SFTDownloadError(f"Cannot read Parquet metadata from {path}: {exc}") from exc
    if rows <= 0:
        raise SFTDownloadError(f"Parquet shard has an invalid row count: {path}")
    return rows


def _train_shards(raw: Path, expected_files: int) -> list[Path]:
    data = raw / "data"
    if raw.is_symlink() or not raw.is_dir():
        raise SFTDownloadError(f"Missing or unsafe raw snapshot directory: {raw}")
    if data.is_symlink() or not data.is_dir():
        raise SFTDownloadError(f"Missing or unsafe dataset directory: {data}")
    expected = [
        data / f"train-{index:05d}-of-{expected_files:05d}.parquet"
        for index in range(expected_files)
    ]
    shards = sorted(data.glob("train-*.parquet"))
    all_parquet = sorted(raw.rglob("*.parquet"))
    if shards != expected:
        missing = sorted(path.name for path in set(expected) - set(shards))
        unexpected = sorted(path.name for path in set(shards) - set(expected))
        raise SFTDownloadError(
            "Exact train-shard filename inventory mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if set(shards) != set(all_parquet):
        unexpected = sorted(path.relative_to(raw).as_posix() for path in set(all_parquet) - set(shards))
        raise SFTDownloadError(f"Unexpected Parquet files in raw snapshot: {unexpected}")
    return shards


def _repository_metadata_inventory(raw: Path) -> list[dict[str, Any]]:
    records = []
    for relative in REPOSITORY_METADATA_FILES:
        path = raw / relative
        try:
            checksum, metadata = _sha256_regular_file(path)
        except OSError as exc:
            raise SFTDownloadError(
                f"Missing or unreadable repository metadata {path}: {exc}"
            ) from exc
        records.append(
            {
                "path": relative,
                "bytes": int(metadata.st_size),
                "sha256": checksum,
            }
        )
    return records


def inventory_snapshot(
    root: Path,
    *,
    contract: DatasetContract = DEFAULT_CONTRACT,
    row_counter: RowCounter = _read_parquet_rows,
) -> dict[str, Any]:
    raw = root / RAW_SUBDIRECTORY
    shards = _train_shards(raw, contract.expected_files)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    total_rows = 0
    started = time.monotonic()
    emit_event("sft_inventory_started", files_total=len(shards))
    for ordinal, path in enumerate(shards, start=1):
        relative = path.relative_to(raw).as_posix()
        checksum, hashed_stat = _sha256_regular_file(path)
        try:
            rows = row_counter(path)
        except SFTDownloadError:
            raise
        except Exception as exc:
            raise SFTDownloadError(
                f"Cannot read Parquet row count from {path}: {exc}"
            ) from exc
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise SFTDownloadError(f"Parquet shard has an invalid row count: {path}")
        after_rows = path.lstat()
        if path.is_symlink() or (
            after_rows.st_dev,
            after_rows.st_ino,
            after_rows.st_size,
            after_rows.st_mtime_ns,
        ) != (
            hashed_stat.st_dev,
            hashed_stat.st_ino,
            hashed_stat.st_size,
            hashed_stat.st_mtime_ns,
        ):
            raise SFTDownloadError(f"Shard changed during Parquet inventory: {path}")
        record = {
            "path": relative,
            "bytes": int(after_rows.st_size),
            "rows": rows,
            "sha256": checksum,
        }
        records.append(record)
        total_bytes += record["bytes"]
        total_rows += rows
        emit_event(
            "sft_inventory_progress",
            file_index=ordinal,
            files_total=len(shards),
            path=relative,
            bytes_scanned=total_bytes,
            rows_scanned=total_rows,
        )

    if total_bytes != contract.expected_bytes:
        raise SFTDownloadError(
            f"Compressed-byte inventory mismatch: expected {contract.expected_bytes:,}, "
            f"found {total_bytes:,}"
        )
    if total_rows != contract.expected_rows:
        raise SFTDownloadError(
            f"Parquet-row inventory mismatch: expected {contract.expected_rows:,}, "
            f"found {total_rows:,}"
        )
    metadata_files = _repository_metadata_inventory(raw)
    core = {
        "inventory_version": INVENTORY_VERSION,
        "files": records,
        "repository_metadata": metadata_files,
        "train_parquet_files": len(records),
        "compressed_download_bytes": total_bytes,
        "rows": total_rows,
    }
    inventory = {
        **core,
        "inventory_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }
    emit_event(
        "sft_inventory_complete",
        files=len(records),
        compressed_download_bytes=total_bytes,
        rows=total_rows,
        elapsed_seconds=round(time.monotonic() - started, 3),
        inventory_sha256=inventory["inventory_sha256"],
    )
    return inventory


def build_authorities(
    inventory: dict[str, Any], *, contract: DatasetContract = DEFAULT_CONTRACT
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "kind": "raw_sft_dataset_snapshot",
        "repo_id": contract.repo_id,
        "repo_type": contract.repo_type,
        "requested_revision": contract.revision,
        "resolved_revision": contract.revision,
        "raw_subdirectory": RAW_SUBDIRECTORY,
        "allow_patterns": list(ALLOW_PATTERNS),
        "raw_files_preserved": True,
        "expected": {
            "train_parquet_files": contract.expected_files,
            "compressed_download_bytes": contract.expected_bytes,
            "rows": contract.expected_rows,
        },
        "inventory": inventory,
    }
    source_sha = hashlib.sha256(pretty_json_bytes(source)).hexdigest()
    completion = {
        "completion_version": COMPLETION_MANIFEST_VERSION,
        "kind": "raw_sft_dataset_snapshot",
        "status": "complete",
        "source_manifest": SOURCE_MANIFEST_NAME,
        "source_manifest_sha256": source_sha,
        "repo_id": contract.repo_id,
        "repo_type": contract.repo_type,
        "resolved_revision": contract.revision,
        "raw_subdirectory": RAW_SUBDIRECTORY,
        "raw_files_preserved": True,
        "train_parquet_files": inventory["train_parquet_files"],
        "compressed_download_bytes": inventory["compressed_download_bytes"],
        "rows": inventory["rows"],
        "inventory_sha256": inventory["inventory_sha256"],
    }
    return source, completion


def publish_authorities(
    root: Path,
    source: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, bool]:
    source_path = root / SOURCE_MANIFEST_NAME
    completion_path = root / COMPLETION_MANIFEST_NAME
    if completion_path.exists() and not source_path.exists():
        raise SFTDownloadError(
            f"Completion authority exists without {SOURCE_MANIFEST_NAME}: {completion_path}"
        )
    source_written = atomic_write_if_missing_or_identical(
        source_path, pretty_json_bytes(source)
    )
    completion_written = atomic_write_if_missing_or_identical(
        completion_path, pretty_json_bytes(completion)
    )
    emit_event(
        "sft_authority_published",
        source=str(source_path),
        completion=str(completion_path),
        source_written=source_written,
        completion_written=completion_written,
    )
    return {"source_written": source_written, "completion_written": completion_written}


def _snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SFTDownloadError(
            "Install requirements-data.txt before downloading OpenCodeInstruct"
        ) from exc
    return str(snapshot_download(**kwargs))


def _validate_invocation(max_workers: int, contract: DatasetContract) -> None:
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise SFTDownloadError("max_workers must be an integer")
    if max_workers < 1 or max_workers > MAX_MAX_WORKERS:
        raise SFTDownloadError(
            f"max_workers must be between 1 and {MAX_MAX_WORKERS}"
        )
    if (
        contract.repo_id != DEFAULT_REPO_ID
        or contract.repo_type != REPO_TYPE
        or contract.expected_files < 1
        or contract.expected_bytes < 1
        or contract.expected_rows < 1
    ):
        raise SFTDownloadError("Dataset contract is invalid")
    if len(contract.revision) != 40 or any(
        character not in "0123456789abcdef" for character in contract.revision
    ):
        raise SFTDownloadError("Dataset revision must be a lowercase 40-character commit")


def _download_sft_dataset_unlocked(
    root: Path,
    *,
    max_workers: int = 8,
    verify_only: bool = False,
    contract: DatasetContract = DEFAULT_CONTRACT,
    snapshot_download: SnapshotDownload = _snapshot_download,
    row_counter: RowCounter = _read_parquet_rows,
) -> dict[str, Any]:
    _validate_invocation(max_workers, contract)
    raw = root / RAW_SUBDIRECTORY
    source_path = root / SOURCE_MANIFEST_NAME
    completion_path = root / COMPLETION_MANIFEST_NAME
    source_exists = source_path.exists() or source_path.is_symlink()
    completion_exists = completion_path.exists() or completion_path.is_symlink()
    if completion_exists and not source_exists:
        raise SFTDownloadError(
            f"{COMPLETION_MANIFEST_NAME} exists without {SOURCE_MANIFEST_NAME}"
        )

    # A complete local snapshot is authoritative enough to certify without a
    # network round trip. This handles verify-only, offline idempotent reruns,
    # and a crash after snapshot completion but before manifest publication.
    local_error: SFTDownloadError | None = None
    try:
        inventory = inventory_snapshot(root, contract=contract, row_counter=row_counter)
    except SFTDownloadError as exc:
        local_error = exc
    except OSError as exc:
        local_error = SFTDownloadError(f"Local snapshot inventory failed: {exc}")
    else:
        source, completion = build_authorities(inventory, contract=contract)
        publication = publish_authorities(root, source, completion)
        return {
            "complete": True,
            "downloaded": False,
            "verified_existing_snapshot": True,
            "root": str(root),
            "source": source,
            "completion": completion,
            "publication": publication,
        }

    if verify_only:
        assert local_error is not None
        raise SFTDownloadError(f"Verify-only inventory failed: {local_error}")
    if source_exists or completion_exists:
        assert local_error is not None
        raise SFTDownloadError(
            "Published SFT authority no longer matches its raw snapshot: "
            f"{local_error}"
        )

    raw.mkdir(parents=True, exist_ok=True)
    if raw.is_symlink() or not raw.is_dir():
        raise SFTDownloadError(f"Raw snapshot path is unsafe: {raw}")
    emit_event(
        "sft_download_started",
        repo_id=contract.repo_id,
        repo_type=contract.repo_type,
        revision=contract.revision,
        raw_directory=str(raw),
        max_workers=max_workers,
        resumable=True,
    )
    started = time.monotonic()
    try:
        returned_value = snapshot_download(
            repo_id=contract.repo_id,
            repo_type=contract.repo_type,
            revision=contract.revision,
            local_dir=raw,
            allow_patterns=list(ALLOW_PATTERNS),
            max_workers=max_workers,
            force_download=False,
        )
    except SFTDownloadError:
        raise
    except Exception as exc:
        raise SFTDownloadError(f"Hugging Face snapshot download failed: {exc}") from exc
    returned = Path(returned_value).resolve(strict=False)
    if returned != raw.resolve(strict=True):
        raise SFTDownloadError(
            f"snapshot_download returned an unexpected destination: {returned}"
        )
    emit_event(
        "sft_download_finished",
        elapsed_seconds=round(time.monotonic() - started, 3),
        raw_directory=str(raw),
    )
    inventory = inventory_snapshot(root, contract=contract, row_counter=row_counter)
    source, completion = build_authorities(inventory, contract=contract)
    publication = publish_authorities(root, source, completion)
    return {
        "complete": True,
        "downloaded": True,
        "verified_existing_snapshot": False,
        "root": str(root),
        "source": source,
        "completion": completion,
        "publication": publication,
    }


def download_sft_dataset(
    root: Path,
    *,
    max_workers: int = 8,
    verify_only: bool = False,
    contract: DatasetContract = DEFAULT_CONTRACT,
    snapshot_download: SnapshotDownload = _snapshot_download,
    row_counter: RowCounter = _read_parquet_rows,
) -> dict[str, Any]:
    """Run one mutually exclusive download/verification invocation."""
    _validate_invocation(max_workers, contract)
    root = prepare_posttraining_root(root)
    lock_path = root / ".download.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SFTDownloadError(f"Cannot open safe downloader lock {lock_path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SFTDownloadError(f"Downloader lock is not a regular file: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SFTDownloadError(
                f"Another SFT downloader or verifier holds {lock_path}"
            ) from exc
        return _download_sft_dataset_unlocked(
            root,
            max_workers=max_workers,
            verify_only=verify_only,
            contract=contract,
            snapshot_download=snapshot_download,
            row_counter=row_counter,
        )
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="dedicated post-training dataset root; raw files are stored in ROOT/raw",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="parallel Hugging Face download workers (default: 8; maximum: 128)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not access Hugging Face; verify ROOT/raw and publish authorities",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = download_sft_dataset(
            args.root,
            max_workers=args.max_workers,
            verify_only=args.verify_only,
        )
    except (OSError, SFTDownloadError) as exc:
        emit_event("sft_download_failed", error=str(exc))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
