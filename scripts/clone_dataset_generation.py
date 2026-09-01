#!/usr/bin/env python3
"""Publish a resumable dataset-generation clone without copying immutable data.

The clone is deliberately selective.  Immutable raw inputs and restart state
are hard-linked on one filesystem; closure/status control files are copied so
the next generation may replace them atomically without changing the frozen
source generation.  Live logs, locks, temporary files, SQLite telemetry, and
derived curation outputs never enter the clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_VERSION = 1
MANIFEST_NAME = "CLONE_MANIFEST.json"
MANIFEST_SIDECAR_NAME = "CLONE_MANIFEST.sha256"

# These trees are append-only for the next generation.  Existing regular files
# must never be edited in place: the source and destination names share inodes.
HARDLINK_TREES = (
    Path("raw"),
    Path("tokenizer"),
    Path("manifests"),
    Path("state/collector_checkpoints"),
    Path("state/collector_parallel"),
    Path("state/quota_records"),
    Path("staging/preprocess/reports"),
    Path("staging/preprocess/fingerprints"),
)

# These files are expected to change when the extended collection is closed or
# when the incremental preprocessing status is refreshed.  They therefore must
# have independent inodes in the clone.
COPIED_CONTROL_FILES = (
    Path("state/COLLECTION_COMPLETE.json"),
    Path("state/ENGLISH_FINEWEB_EDU_COMPLETE.json"),
    Path("state/ENGLISH_WIKIPEDIA_COMPLETE.json"),
    Path("staging/preprocess/PREPROCESS_MANIFEST.json"),
    Path("staging/preprocess/STATUS.json"),
)

# Parent directories that are not themselves hard-link-tree roots.
SCAFFOLD_DIRECTORIES = (
    Path("state"),
    Path("staging"),
    Path("staging/preprocess"),
)

EXCLUDED_TOP_LEVEL = frozenset(("audits", "curated", "final", "logs"))
FORBIDDEN_INCLUDED_NAMES = frozenset(
    (".preprocess.lock", ".tmp", "dedup.sqlite3")
)

OBSERVED_OTHER_RAW_TOKENS = 25_952_231_562
OBSERVED_OTHER_ELIGIBLE_TOKENS = 16_811_351_831
OBSERVED_OTHER_TRAIN_TOKENS = 16_527_423_703
TARGET_OTHER_TRAIN_TOKENS = 21_032_000_000
TARGET_OTHER_RAW_TOKENS = 35_000_000_000


class CloneError(RuntimeError):
    """The generation clone could not be proven safe."""


@dataclass(frozen=True)
class FileIdentity:
    relative_path: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, source: Path, path: Path) -> "FileIdentity":
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CloneError(f"Included source path is a symlink: {path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise CloneError(f"Included source path is not a regular file: {path}")
        return cls(
            relative_path=str(path.relative_to(source)),
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            mode=stat.S_IMODE(metadata.st_mode),
            size=int(metadata.st_size),
            mtime_ns=int(metadata.st_mtime_ns),
        )

    def source_record(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "bytes": self.size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class DirectoryIdentity:
    relative_path: str
    device: int
    inode: int
    mode: int
    mtime_ns: int

    @classmethod
    def from_path(cls, source: Path, path: Path) -> "DirectoryIdentity":
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CloneError(f"Included source directory is a symlink: {path}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise CloneError(f"Included source directory is not a directory: {path}")
        return cls(
            relative_path=str(path.relative_to(source)),
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            mode=stat.S_IMODE(metadata.st_mode),
            mtime_ns=int(metadata.st_mtime_ns),
        )


@dataclass(frozen=True)
class SourceSnapshot:
    directories: tuple[DirectoryIdentity, ...]
    linked_files: tuple[FileIdentity, ...]
    copied_files: tuple[FileIdentity, ...]

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "directories": [
                {
                    "path": row.relative_path,
                    "device": row.device,
                    "inode": row.inode,
                    "mode": row.mode,
                    "mtime_ns": row.mtime_ns,
                }
                for row in self.directories
            ],
            "linked_files": [row.source_record() for row in self.linked_files],
            "copied_files": [row.source_record() for row in self.copied_files],
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_roots(source: Path, destination: Path) -> tuple[Path, Path, Path]:
    if not source.is_absolute() or not destination.is_absolute():
        raise CloneError("--source and --destination must be absolute paths")
    if source.is_symlink() or not source.is_dir():
        raise CloneError(f"Source must be a real directory: {source}")
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise CloneError(f"Destination must not already exist: {destination}")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CloneError(f"Destination parent must be a real existing directory: {parent}")
    parent = parent.resolve(strict=True)
    destination = parent / destination.name
    if not destination.name or destination.name in (".", ".."):
        raise CloneError("Destination has an unsafe final component")
    if _is_within(parent, source):
        raise CloneError("Destination cannot be inside the immutable source generation")
    if source.stat().st_dev != parent.stat().st_dev:
        raise CloneError(
            "Source and destination parent are on different filesystems; "
            "hard-link cloning is impossible"
        )
    return source, destination, parent


def _walk_tree(source: Path, relative_root: Path) -> tuple[list[DirectoryIdentity], list[FileIdentity]]:
    root = source / relative_root
    if root.name in FORBIDDEN_INCLUDED_NAMES:
        raise CloneError(f"Forbidden tree requested for inclusion: {relative_root}")
    directories: list[DirectoryIdentity] = []
    files: list[FileIdentity] = []

    def visit(directory: Path) -> None:
        directories.append(DirectoryIdentity.from_path(source, directory))
        try:
            entries = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as error:
            raise CloneError(f"Could not enumerate included directory {directory}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            if entry.name in FORBIDDEN_INCLUDED_NAMES or entry.name.startswith(".part-"):
                raise CloneError(f"Forbidden or pending input in included tree: {path}")
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CloneError(f"Included source path is a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(FileIdentity.from_path(source, path))
            else:
                raise CloneError(f"Included source path has an unsafe file type: {path}")

    visit(root)
    return directories, files


def snapshot_source(source: Path) -> SourceSnapshot:
    directories: dict[str, DirectoryIdentity] = {}
    linked_files: dict[str, FileIdentity] = {}
    copied_files: dict[str, FileIdentity] = {}

    for relative in SCAFFOLD_DIRECTORIES:
        identity = DirectoryIdentity.from_path(source, source / relative)
        directories[identity.relative_path] = identity
    for relative in HARDLINK_TREES:
        tree_directories, tree_files = _walk_tree(source, relative)
        for identity in tree_directories:
            existing = directories.setdefault(identity.relative_path, identity)
            if existing != identity:
                raise CloneError(f"Conflicting source directory identity: {identity.relative_path}")
        for identity in tree_files:
            if identity.relative_path in linked_files or identity.relative_path in copied_files:
                raise CloneError(f"Duplicate included source file: {identity.relative_path}")
            linked_files[identity.relative_path] = identity
    for relative in COPIED_CONTROL_FILES:
        identity = FileIdentity.from_path(source, source / relative)
        if identity.relative_path in linked_files or identity.relative_path in copied_files:
            raise CloneError(f"Duplicate included control file: {identity.relative_path}")
        copied_files[identity.relative_path] = identity

    if not linked_files:
        raise CloneError("The source inventory contains no immutable files")
    return SourceSnapshot(
        directories=tuple(directories[key] for key in sorted(directories)),
        linked_files=tuple(linked_files[key] for key in sorted(linked_files)),
        copied_files=tuple(copied_files[key] for key in sorted(copied_files)),
    )


def _create_directories(incoming: Path, snapshot: SourceSnapshot) -> None:
    for identity in sorted(
        snapshot.directories,
        key=lambda row: (len(Path(row.relative_path).parts), row.relative_path),
    ):
        path = incoming / identity.relative_path
        path.mkdir(mode=identity.mode, parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise CloneError(f"Unsafe destination directory after creation: {path}")


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination, follow_symlinks=False)


def _copy_file(source: Path, destination: Path, mode: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        try:
            with os.fdopen(source_descriptor, "rb", closefd=False) as reader, os.fdopen(
                destination_descriptor, "wb", closefd=False
            ) as writer:
                while block := reader.read(1024 * 1024):
                    writer.write(block)
                    digest.update(block)
                writer.flush()
                os.fsync(writer.fileno())
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes, mode: int = 0o640) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _topup_rationale() -> dict[str, Any]:
    shortfall = TARGET_OTHER_TRAIN_TOKENS - OBSERVED_OTHER_TRAIN_TOKENS
    point_estimate = (
        shortfall * OBSERVED_OTHER_RAW_TOKENS + OBSERVED_OTHER_TRAIN_TOKENS - 1
    ) // OBSERVED_OTHER_TRAIN_TOKENS
    planned_increment = TARGET_OTHER_RAW_TOKENS - OBSERVED_OTHER_RAW_TOKENS
    expected_train_increment = (
        planned_increment * OBSERVED_OTHER_TRAIN_TOKENS
    ) // OBSERVED_OTHER_RAW_TOKENS
    margin_basis_points = (
        (planned_increment - point_estimate) * 10_000 // point_estimate
    )
    return {
        "bucket": "other_code",
        "target_cumulative_raw_tokens": TARGET_OTHER_RAW_TOKENS,
        "observed_raw_tokens": OBSERVED_OTHER_RAW_TOKENS,
        "observed_eligible_tokens": OBSERVED_OTHER_ELIGIBLE_TOKENS,
        "observed_train_tokens": OBSERVED_OTHER_TRAIN_TOKENS,
        "target_train_tokens": TARGET_OTHER_TRAIN_TOKENS,
        "train_shortfall_tokens": shortfall,
        "observed_train_yield": {
            "numerator": OBSERVED_OTHER_TRAIN_TOKENS,
            "denominator": OBSERVED_OTHER_RAW_TOKENS,
        },
        "point_estimate_raw_increment_tokens": point_estimate,
        "planned_raw_increment_tokens": planned_increment,
        "margin_over_point_estimate_basis_points": margin_basis_points,
        "expected_train_increment_at_observed_yield": expected_train_increment,
        "expected_train_surplus_at_observed_yield": expected_train_increment - shortfall,
        "reason": (
            "35.0B cumulative raw tokens provides about 27.9% headroom over "
            "the observed-yield point estimate while leaving all non-other-code "
            "collection and final quotas unchanged."
        ),
    }


def _verify_linked(
    source: Path, incoming: Path, files: Iterable[FileIdentity]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for identity in files:
        source_path = source / identity.relative_path
        destination_path = incoming / identity.relative_path
        source_stat = source_path.lstat()
        destination_stat = destination_path.lstat()
        if stat.S_ISLNK(destination_stat.st_mode) or not stat.S_ISREG(destination_stat.st_mode):
            raise CloneError(f"Linked destination is not a regular file: {destination_path}")
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            raise CloneError(f"Destination is not a hard link to source: {destination_path}")
        if destination_stat.st_size != identity.size:
            raise CloneError(f"Hard-linked file size changed: {destination_path}")
        records.append(
            {
                "path": identity.relative_path,
                "bytes": identity.size,
                "device": int(destination_stat.st_dev),
                "inode": int(destination_stat.st_ino),
            }
        )
    return records


def _verify_copied(
    source: Path,
    incoming: Path,
    files: Iterable[FileIdentity],
    copied_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for identity in files:
        source_path = source / identity.relative_path
        destination_path = incoming / identity.relative_path
        source_stat = source_path.lstat()
        destination_stat = destination_path.lstat()
        if stat.S_ISLNK(destination_stat.st_mode) or not stat.S_ISREG(destination_stat.st_mode):
            raise CloneError(f"Copied destination is not a regular file: {destination_path}")
        if (source_stat.st_dev, source_stat.st_ino) == (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            raise CloneError(f"Mutable control file was hard-linked: {destination_path}")
        source_sha = file_sha256(source_path)
        destination_sha = file_sha256(destination_path)
        if source_sha != destination_sha or source_sha != copied_hashes[identity.relative_path]:
            raise CloneError(f"Copied control checksum mismatch: {destination_path}")
        records.append(
            {
                "path": identity.relative_path,
                "bytes": identity.size,
                "source_sha256": source_sha,
                "destination_sha256": destination_sha,
                "source_device": int(source_stat.st_dev),
                "source_inode": int(source_stat.st_ino),
                "destination_device": int(destination_stat.st_dev),
                "destination_inode": int(destination_stat.st_ino),
            }
        )
    return records


def _fsync_tree_directories(incoming: Path, snapshot: SourceSnapshot) -> None:
    paths = {incoming}
    for identity in snapshot.directories:
        path = incoming / identity.relative_path
        paths.add(path)
        paths.update(path.parents)
    relevant = [path for path in paths if path == incoming or _is_within(path, incoming)]
    for path in sorted(relevant, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            fsync_directory(path)


def clone_generation(source: Path, destination: Path) -> dict[str, Any]:
    source, destination, destination_parent = _validate_roots(source, destination)
    before = snapshot_source(source)
    parent_device = destination_parent.stat().st_dev
    for identity in (*before.linked_files, *before.copied_files):
        if identity.device != parent_device:
            raise CloneError(
                f"Included source file is on a different filesystem: {identity.relative_path}"
            )

    incoming = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incoming-",
            dir=destination_parent,
        )
    )
    os.chmod(incoming, stat.S_IMODE(source.stat().st_mode))
    published = False
    try:
        _create_directories(incoming, before)
        for identity in before.linked_files:
            _link_file(
                source / identity.relative_path,
                incoming / identity.relative_path,
            )
        copied_hashes: dict[str, str] = {}
        for identity in before.copied_files:
            copied_hashes[identity.relative_path] = _copy_file(
                source / identity.relative_path,
                incoming / identity.relative_path,
                identity.mode,
            )

        linked_records = _verify_linked(source, incoming, before.linked_files)
        copied_records = _verify_copied(
            source, incoming, before.copied_files, copied_hashes
        )
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "selective-hardlink-dataset-generation-clone",
            "complete": True,
            "created_unix_ns": time.time_ns(),
            "source": {
                "path": str(source),
                "device": int(source.stat().st_dev),
                "inventory_sha256": canonical_sha256(before.canonical_identity()),
            },
            "destination": {
                "path": str(destination),
                "device": int(destination_parent.stat().st_dev),
            },
            "contract": {
                "hardlink_trees": [str(path) for path in HARDLINK_TREES],
                "copied_control_files": [str(path) for path in COPIED_CONTROL_FILES],
                "excluded_top_level": sorted(EXCLUDED_TOP_LEVEL),
                "excluded_runtime_artifacts": [
                    "locks",
                    "temporary files",
                    "staging/preprocess/dedup/dedup.sqlite3",
                ],
                "source_must_remain_immutable": True,
                "existing_hardlinked_files_must_never_be_modified_in_place": True,
            },
            "inventory": {
                "hardlinked_files": len(linked_records),
                "hardlinked_bytes": sum(row["bytes"] for row in linked_records),
                "copied_control_files": len(copied_records),
                "copied_control_bytes": sum(row["bytes"] for row in copied_records),
                "hardlinks_sha256": canonical_sha256(linked_records),
                "copied_controls_sha256": canonical_sha256(copied_records),
                "hardlinks": linked_records,
                "copied_controls": copied_records,
            },
            "other_code_topup_plan": _topup_rationale(),
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        manifest_path = incoming / MANIFEST_NAME
        _write_bytes(manifest_path, manifest_payload)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        _write_bytes(
            incoming / MANIFEST_SIDECAR_NAME,
            f"{manifest_sha}  {MANIFEST_NAME}\n".encode("ascii"),
        )
        _fsync_tree_directories(incoming, before)

        # Make the last source-stability check immediately before publication,
        # after even the potentially large manifest and directory fsync work.
        after = snapshot_source(source)
        if after != before:
            raise CloneError("Source generation changed while the clone was being assembled")

        # The destination was required absent at entry and is checked again
        # immediately before the single publication rename.
        if destination.exists() or destination.is_symlink():
            raise CloneError(f"Destination appeared during clone: {destination}")
        os.rename(incoming, destination)
        published = True
        fsync_directory(destination_parent)

        published_manifest = destination / MANIFEST_NAME
        if file_sha256(published_manifest) != manifest_sha:
            raise CloneError("Published clone manifest checksum mismatch")
        return {
            "complete": True,
            "destination": str(destination),
            "manifest": str(published_manifest),
            "manifest_sha256": manifest_sha,
            "hardlinked_files": len(linked_records),
            "copied_control_files": len(copied_records),
            "target_other_code_raw_tokens": TARGET_OTHER_RAW_TOKENS,
        }
    finally:
        if not published and incoming.exists():
            shutil.rmtree(incoming)
            fsync_directory(destination_parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = clone_generation(args.source, args.destination)
    except (CloneError, OSError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
