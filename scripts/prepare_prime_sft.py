#!/usr/bin/env python3
"""Curate raw OpenCodeInstruct into an immutable PrimeRL/HF messages dataset.

The command is restartable at source-shard boundaries and never edits raw
files.  Workers publish one checksummed sidecar last; a completed sidecar is
the only authority that permits a shard to be skipped after a restart.  The
derived directory itself is atomically renamed into place only after all
source shards, rejection ledgers, dataset metadata, and checksums validate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from posttrain.sft_data import (  # noqa: E402
    CHAT_FORMAT_ID,
    EXPECTED_SOURCE_BYTES,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHARDS,
    FilterPolicy,
    REQUIRED_COLUMNS,
    SOURCE_LICENSE,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    SplitPolicy,
    TOKENIZER_EOS_ID,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_VOCAB_SIZE,
    canonical_json_bytes,
    decide_row,
    file_sha256,
    prompt_group_id,
    sha256_bytes,
    validate_policy_payload,
)
from scripts.benchmark_guard import BenchmarkGuard  # noqa: E402


FORMAT_VERSION = 1
SHARD_RESULT_VERSION = 1
MANIFEST_VERSION = 1
COMPLETION_VERSION = 1
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "posttrain" / "opencodeinstruct_prime_sft_v1.json"
DEFAULT_TOKENIZER = Path("/workspace/dataset/tokenizer/starcoder2")
DEFAULT_ROOT = Path("/workspace/posttraining-data/sft/opencodeinstruct")
DEFAULT_OUTPUT_NAME = "prime-sft-v1"
HEX64 = frozenset("0123456789abcdef")
IMPLEMENTATION_FILES = {
    "curator": Path(__file__).resolve(),
    "benchmark_guard": PROJECT_ROOT / "scripts" / "benchmark_guard.py",
    "sft_data": PROJECT_ROOT / "posttrain" / "sft_data.py",
    "chat_format": PROJECT_ROOT / "posttrain" / "prime" / "chat_format.py",
    "prime_renderer": PROJECT_ROOT / "posttrain" / "prime" / "renderer.py",
}
IMPLEMENTATION_PACKAGES = ("pyarrow", "tokenizers")


class SFTPreparationError(RuntimeError):
    pass


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise SFTPreparationError(f"Cannot fsync non-directory path: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SFTPreparationError(f"Cannot fsync non-regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Durably flush an already-validated, symlink-free publication tree."""

    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        if directory.is_symlink() or not directory.is_dir():
            raise SFTPreparationError(f"Unsafe publication directory during fsync: {directory}")
        directories.append(directory)
        for name in directory_names:
            child = directory / name
            if child.is_symlink() or not child.is_dir():
                raise SFTPreparationError(f"Unsafe publication directory during fsync: {child}")
        for name in file_names:
            _fsync_regular_file(directory / name)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one authority through a no-follow descriptor and require stability."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SFTPreparationError(f"Cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SFTPreparationError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SFTPreparationError(f"{label} changed while reading: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise SFTPreparationError(f"{label} was truncated while reading: {path}")
        return payload
    finally:
        os.close(descriptor)


def _regular_file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_regular_bytes(path, label=label)).hexdigest()


def _regular_file_stat_sha256(path: Path, *, label: str) -> tuple[int, str]:
    """Stream a potentially large regular file through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SFTPreparationError(f"Cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SFTPreparationError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SFTPreparationError(f"{label} changed while hashing: {path}")
        if bytes_read != before.st_size:
            raise SFTPreparationError(f"{label} was truncated while hashing: {path}")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SFTPreparationError(f"Unsafe authority parent directory: {path.parent}")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SFTPreparationError(f"Refusing unsafe authority target: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path, label="JSON authority"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SFTPreparationError(f"Cannot read JSON authority {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SFTPreparationError(f"Expected a JSON object in {path}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX64 for char in value):
        raise SFTPreparationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _implementation_identity() -> dict[str, Any]:
    """Bind resumable artifacts to every implementation input that affects bytes."""

    files: dict[str, dict[str, Any]] = {}
    for name, path in IMPLEMENTATION_FILES.items():
        payload = _read_regular_bytes(path, label=f"implementation file {name}")
        files[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    packages: dict[str, str] = {}
    for name in IMPLEMENTATION_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SFTPreparationError(f"Required curation package is not installed: {name}") from exc
    return {
        "contracts": {
            "format_version": FORMAT_VERSION,
            "shard_result_version": SHARD_RESULT_VERSION,
            "manifest_version": MANIFEST_VERSION,
            "completion_version": COMPLETION_VERSION,
        },
        "files": files,
        "packages": packages,
    }


def _assert_implementation_identity(expected: Mapping[str, Any]) -> None:
    if _implementation_identity() != expected:
        raise SFTPreparationError(
            "Curation code or dependency versions changed after the run identity was frozen"
        )


def _safe_directory(path: Path, *, label: str, must_exist: bool) -> Path:
    if path.is_symlink():
        raise SFTPreparationError(f"{label} cannot be a symlink: {path}")
    if must_exist and not path.is_dir():
        raise SFTPreparationError(f"Missing {label}: {path}")
    if not must_exist:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_dir():
            raise SFTPreparationError(f"{label} must be a directory: {path}")
    return path.resolve(strict=must_exist)


def _validate_source(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_directory = root / "raw"
    data_directory = raw_directory / "data"
    for directory in (raw_directory, data_directory):
        if directory.is_symlink() or not directory.is_dir():
            raise SFTPreparationError(f"Missing or unsafe raw source directory: {directory}")
    source_path = root / "SOURCE.json"
    completion_path = root / "COMPLETION.json"
    source = _load_json(source_path)
    completion = _load_json(completion_path)
    source_sha = _regular_file_sha256(source_path, label="raw SOURCE.json")
    expected = {
        "repo_id": SOURCE_REPO_ID,
        "repo_type": "dataset",
        "resolved_revision": SOURCE_REVISION,
        "raw_subdirectory": "raw",
        "raw_files_preserved": True,
    }
    for field, wanted in expected.items():
        if source.get(field) != wanted or completion.get(field) != wanted:
            raise SFTPreparationError(f"Raw authority {field} does not equal {wanted!r}")
    if completion.get("status") != "complete":
        raise SFTPreparationError("Raw snapshot is not complete")
    if source.get("manifest_version") != 1 or source.get("kind") != "raw_sft_dataset_snapshot":
        raise SFTPreparationError("Unsupported raw SOURCE.json contract")
    if completion.get("completion_version") != 1 or completion.get("kind") != source.get("kind"):
        raise SFTPreparationError("Unsupported raw COMPLETION.json contract")
    if source.get("requested_revision") != SOURCE_REVISION:
        raise SFTPreparationError("Raw source requested revision is not frozen")
    if source.get("allow_patterns") != [
        "data/train-*.parquet",
        "README.md",
        ".gitattributes",
    ]:
        raise SFTPreparationError("Raw source allow-pattern contract changed")
    expected_inventory = {
        "train_parquet_files": EXPECTED_SOURCE_SHARDS,
        "compressed_download_bytes": EXPECTED_SOURCE_BYTES,
        "rows": EXPECTED_SOURCE_ROWS,
    }
    if source.get("expected") != expected_inventory:
        raise SFTPreparationError("Raw source expected inventory contract changed")
    if completion.get("source_manifest_sha256") != source_sha:
        raise SFTPreparationError("COMPLETION.json does not authenticate SOURCE.json")
    inventory = source.get("inventory")
    if not isinstance(inventory, dict):
        raise SFTPreparationError("SOURCE.json has no inventory")
    if (
        inventory.get("train_parquet_files") != EXPECTED_SOURCE_SHARDS
        or inventory.get("rows") != EXPECTED_SOURCE_ROWS
        or inventory.get("compressed_download_bytes") != EXPECTED_SOURCE_BYTES
    ):
        raise SFTPreparationError("Raw snapshot inventory does not match the frozen source contract")
    recorded_inventory_sha = _require_sha256(
        inventory.get("inventory_sha256"), field="inventory.inventory_sha256"
    )
    inventory_core = dict(inventory)
    inventory_core.pop("inventory_sha256", None)
    calculated_inventory_sha = hashlib.sha256(
        json.dumps(inventory_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recorded_inventory_sha != calculated_inventory_sha:
        raise SFTPreparationError("SOURCE.json inventory digest is invalid")
    if completion.get("inventory_sha256") != recorded_inventory_sha:
        raise SFTPreparationError("Completion and source inventory identities disagree")
    records = inventory.get("files")
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_SHARDS:
        raise SFTPreparationError("Raw source shard inventory is incomplete")
    expected_paths = [f"data/train-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.parquet" for index in range(EXPECTED_SOURCE_SHARDS)]
    if [record.get("path") for record in records] != expected_paths:
        raise SFTPreparationError("Raw source shard paths are not the exact frozen inventory")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SFTPreparationError(f"Invalid raw inventory record {index}")
        _require_sha256(record.get("sha256"), field=f"inventory.files[{index}].sha256")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise SFTPreparationError(f"Invalid compressed bytes for raw shard {index}")
        if not isinstance(record.get("rows"), int) or record["rows"] <= 0:
            raise SFTPreparationError(f"Invalid row count for raw shard {index}")
    if sum(record["bytes"] for record in records) != EXPECTED_SOURCE_BYTES:
        raise SFTPreparationError("Raw inventory shard bytes do not add up")
    if sum(record["rows"] for record in records) != EXPECTED_SOURCE_ROWS:
        raise SFTPreparationError("Raw inventory shard rows do not add up")
    actual_names = []
    for path in data_directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise SFTPreparationError(f"Unsafe entry in raw data directory: {path}")
        actual_names.append(path.name)
    if sorted(actual_names) != sorted(Path(path).name for path in expected_paths):
        raise SFTPreparationError("Raw data directory is not the exact frozen shard set")
    return {"source_manifest_sha256": source_sha, "inventory_sha256": recorded_inventory_sha}, records


def _validate_tokenizer(tokenizer_root: Path) -> dict[str, Any]:
    manifest_path = tokenizer_root / "TOKENIZER_MANIFEST.json"
    tokenizer_json = tokenizer_root / "tokenizer.json"
    manifest = _load_json(manifest_path)
    if manifest.get("repo_id") != TOKENIZER_REPO_ID:
        raise SFTPreparationError(f"Tokenizer repo must be {TOKENIZER_REPO_ID}")
    if manifest.get("resolved_revision") != TOKENIZER_REVISION:
        raise SFTPreparationError(f"Tokenizer revision must be {TOKENIZER_REVISION}")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise SFTPreparationError("Tokenizer manifest has no validation record")
    if validation.get("vocab_size") != TOKENIZER_VOCAB_SIZE:
        raise SFTPreparationError("Tokenizer vocabulary size mismatch")
    if validation.get("eos_token_id") != TOKENIZER_EOS_ID:
        raise SFTPreparationError("Tokenizer EOS ID mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("tokenizer.json"), dict):
        raise SFTPreparationError("Tokenizer manifest does not authenticate tokenizer.json")
    for name, record in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SFTPreparationError(f"Unsafe tokenizer manifest file name: {name!r}")
        if not isinstance(record, dict):
            raise SFTPreparationError(f"Invalid tokenizer manifest record: {name}")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise SFTPreparationError(f"Invalid tokenizer byte count: {name}")
        expected_file_sha = _require_sha256(
            record.get("sha256"), field=f"tokenizer.files[{name}].sha256"
        )
        path = tokenizer_root / name
        payload = _read_regular_bytes(path, label="tokenizer manifest file")
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_file_sha:
            raise SFTPreparationError(f"Tokenizer file no longer matches its manifest: {name}")
    expected_sha = files["tokenizer.json"]["sha256"]
    return {
        "tokenizer_manifest_sha256": _regular_file_sha256(
            manifest_path, label="tokenizer manifest"
        ),
        "tokenizer_json_sha256": expected_sha,
    }


def _resolve_policy(policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_bytes = _read_regular_bytes(policy_path, label="SFT policy")
    try:
        policy = json.loads(policy_bytes)
    except json.JSONDecodeError as exc:
        raise SFTPreparationError(f"Invalid SFT policy JSON {policy_path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise SFTPreparationError(f"Expected a JSON object in {policy_path}")
    try:
        validate_policy_payload(policy)
        split = SplitPolicy(**policy["split"])
        filters = FilterPolicy(**policy["filter"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SFTPreparationError(f"Invalid SFT policy {policy_path}: {exc}") from exc
    resolved_denylists = []
    for configured in policy["benchmark_denylists"]:
        if not isinstance(configured, str) or not configured:
            raise SFTPreparationError("benchmark denylist paths must be non-empty strings")
        candidate = Path(configured)
        path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        denylist_payload = _read_regular_bytes(path, label="benchmark denylist")
        denylist_sha = hashlib.sha256(denylist_payload).hexdigest()
        # Construction validates the manifest's complete benchmark contract.
        BenchmarkGuard(path)
        if _regular_file_sha256(path, label="benchmark denylist") != denylist_sha:
            raise SFTPreparationError(f"Benchmark denylist changed while reading: {path}")
        resolved_denylists.append(
            {"path": str(path.resolve()), "source_name": path.name, "sha256": denylist_sha}
        )
    runtime = {
        "split": asdict(split),
        "filter": asdict(filters),
        "denylists": resolved_denylists,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
    }
    return policy, runtime


def _runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": runtime["split"],
        "filter": runtime["filter"],
        "denylists": [
            {"name": record["source_name"], "sha256": record["sha256"]}
            for record in runtime["denylists"]
        ],
    }


def _snapshot_runtime_inputs(
    work: Path,
    *,
    tokenizer_root: Path,
    tokenizer_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Copy small mutable runtime inputs into the identity-bound work tree."""

    inputs = work / "inputs"
    if inputs.is_symlink():
        raise SFTPreparationError(f"Unsafe runtime input directory: {inputs}")
    inputs.mkdir(parents=True, exist_ok=True)

    tokenizer_source = tokenizer_root / "tokenizer.json"
    tokenizer_bytes = _read_regular_bytes(tokenizer_source, label="tokenizer source")
    if hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_identity["tokenizer_json_sha256"]:
        raise SFTPreparationError("Tokenizer changed before runtime snapshot")
    tokenizer_snapshot = inputs / "tokenizer.json"
    if tokenizer_snapshot.is_symlink():
        raise SFTPreparationError(f"Unsafe tokenizer snapshot: {tokenizer_snapshot}")
    if not tokenizer_snapshot.is_file() or file_sha256(tokenizer_snapshot) != tokenizer_identity["tokenizer_json_sha256"]:
        if tokenizer_snapshot.exists() or tokenizer_snapshot.is_symlink():
            if tokenizer_snapshot.is_symlink() or not tokenizer_snapshot.is_file():
                raise SFTPreparationError(f"Unsafe tokenizer snapshot: {tokenizer_snapshot}")
        _atomic_bytes(tokenizer_snapshot, tokenizer_bytes)

    snapshot_denylists: list[dict[str, Any]] = []
    expected_names = {"tokenizer.json"}
    for index, record in enumerate(runtime["denylists"]):
        source = Path(record["path"])
        payload = _read_regular_bytes(source, label="benchmark denylist source")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise SFTPreparationError(f"Benchmark denylist changed before snapshot: {source}")
        name = f"benchmark-denylist-{index:02d}.json"
        expected_names.add(name)
        destination = inputs / name
        if destination.is_symlink():
            raise SFTPreparationError(f"Unsafe benchmark snapshot: {destination}")
        if not destination.is_file() or file_sha256(destination) != record["sha256"]:
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_file():
                    raise SFTPreparationError(f"Unsafe benchmark snapshot: {destination}")
            _atomic_bytes(destination, payload)
        BenchmarkGuard(destination)
        snapshot_denylists.append(
            {
                "path": str(destination),
                "source_name": record["source_name"],
                "sha256": record["sha256"],
            }
        )
    entries = list(inputs.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SFTPreparationError("Runtime input snapshot contains an unsafe entry")
    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        raise SFTPreparationError("Runtime input snapshot contains unexpected files")
    snapshot_runtime = dict(runtime)
    snapshot_runtime["denylists"] = snapshot_denylists
    return inputs, snapshot_runtime


def _load_runtime_renderer(tokenizer: Any):
    try:
        from posttrain.prime.chat_format import StarCoder2CodingChatFormat
    except ImportError as exc:
        raise SFTPreparationError(
            "Prime chat-format adapter is missing; expected posttrain.prime.chat_format"
        ) from exc
    return StarCoder2CodingChatFormat(tokenizer)


def _load_tokenizer(tokenizer_root: Path, *, expected_sha256: str | None = None):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SFTPreparationError("Install requirements-data.txt (tokenizers is required)") from exc
    tokenizer_json = tokenizer_root / "tokenizer.json"
    tokenizer_sha = _regular_file_sha256(tokenizer_json, label="tokenizer snapshot")
    if expected_sha256 is not None and tokenizer_sha != expected_sha256:
        raise SFTPreparationError("Tokenizer snapshot changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    if tokenizer.get_vocab_size(with_added_tokens=True) != TOKENIZER_VOCAB_SIZE:
        raise SFTPreparationError("Loaded tokenizer vocabulary size mismatch")
    return tokenizer


def _schemas():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise SFTPreparationError("Install requirements-data.txt (pyarrow is required)") from exc
    message = pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
        ]
    )
    accepted = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("messages", pa.list_(message), nullable=False),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("generation_algorithm", pa.string(), nullable=False),
            pa.field("average_test_score", pa.float64(), nullable=False),
            pa.field("rendered_tokens", pa.int32(), nullable=False),
            pa.field("source_revision", pa.string(), nullable=False),
            pa.field("source_shard", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
        ]
    )
    rejected = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("group_id", pa.string()),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("rendered_tokens", pa.int32()),
            pa.field("source_shard", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
        ]
    )
    return accepted, rejected


def _partial_output_path(path: Path) -> Path:
    """Return the one deterministic unpublished path for a worker output.

    A deterministic name is important for crash recovery: after SIGKILL or a
    pod loss, the parent process can identify and remove precisely the partial
    file for a source-index/kind pair before resubmitting that shard.
    """

    return path.with_name(f".{path.name}.partial")


class _ParquetSink:
    def __init__(self, path: Path, schema: Any, *, flush_rows: int) -> None:
        import pyarrow.parquet as pq

        self.path = path
        self.partial = _partial_output_path(path)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise SFTPreparationError(f"Unsafe Parquet output directory: {path.parent}")
        if path.is_symlink() or path.exists():
            raise SFTPreparationError(f"Refusing existing Parquet output: {path}")
        if self.partial.is_symlink() or (
            self.partial.exists() and not self.partial.is_file()
        ):
            raise SFTPreparationError(f"Unsafe partial Parquet output: {self.partial}")
        if self.partial.exists():
            self.partial.unlink()
        self.schema = schema
        self.flush_rows = flush_rows
        self.rows: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(
            self.partial,
            schema,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        )
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa

        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.flush_rows)
        self.count += len(self.rows)
        self.rows.clear()

    def finish(self) -> dict[str, Any] | None:
        self.flush()
        self.writer.close()
        # Hugging Face Datasets 4.x can fail while discovering a split that
        # contains an empty Parquet shard before a non-empty shard.  Empty
        # worker outputs carry no information, so omit them and authenticate
        # their absence in the shard sidecar instead.
        if self.count == 0:
            self.partial.unlink()
            _fsync_directory(self.path.parent)
            return None
        _fsync_regular_file(self.partial)
        os.replace(self.partial, self.path)
        _fsync_directory(self.path.parent)
        output_bytes, output_sha256 = _regular_file_stat_sha256(
            self.path, label="Parquet output"
        )
        return {
            "path": self.path.name,
            "rows": self.count,
            "bytes": output_bytes,
            "sha256": output_sha256,
        }

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            try:
                self.partial.unlink()
                _fsync_directory(self.path.parent)
            except FileNotFoundError:
                pass


def _validate_arrow_schema(path: Path) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    if schema.names != list(REQUIRED_COLUMNS):
        raise SFTPreparationError(
            f"Unexpected OpenCodeInstruct schema in {path}: {schema.names}"
        )
    for name in REQUIRED_COLUMNS[:-1]:
        if not pa.types.is_string(schema.field(name).type):
            raise SFTPreparationError(f"Column {name} in {path} must be string")
    if not pa.types.is_floating(schema.field("average_test_score").type):
        raise SFTPreparationError(f"average_test_score in {path} must be floating point")
    return parquet_file


def _worker_contamination_guards(records: Iterable[Mapping[str, Any]]):
    guards = []
    for record in records:
        path = Path(record["path"])
        if _regular_file_sha256(path, label="benchmark denylist snapshot") != record["sha256"]:
            raise SFTPreparationError(f"Benchmark denylist snapshot changed: {path}")
        guards.append(BenchmarkGuard(path))

    def reason(content: str) -> str | None:
        for guard in guards:
            match = guard.contamination_reason("Python", content)
            if match is not None:
                return f"{guard.manifest_sha256[:12]}:{match}"
        return None

    return reason


def _output_record(work: Path, kind: str, index: int) -> Path:
    if kind in ("train", "validation"):
        return work / "data" / f"{kind}-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.parquet"
    if kind == "rejected":
        return work / "audit" / f"rejected-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.parquet"
    raise ValueError(kind)


def _sidecar_path(work: Path, index: int) -> Path:
    return work / "audit" / f"shard-{index:05d}.json"


def _preflight_path(work: Path, index: int) -> Path:
    return work / "audit" / f"contamination-{index:05d}.json"


def _content_for_benchmark_scan(prompt: Any, completion: Any, raw_tests: Any) -> str:
    values = [value if isinstance(value, str) else "" for value in (prompt, completion)]
    if isinstance(raw_tests, str):
        try:
            tests = json.loads(raw_tests)
        except json.JSONDecodeError:
            tests = None
        if isinstance(tests, list):
            values.append("\n".join(str(test) for test in tests))
        else:
            values.append(raw_tests)
    return "\n\n".join(values)


def _scan_contamination_shard(task: dict[str, Any]) -> dict[str, Any]:
    """First pass: find every prompt group touched by a frozen benchmark.

    This pass is what makes the later split genuinely leakage-safe.  A clean
    duplicate of a prompt cannot survive merely because benchmark material was
    present only in another completion or unit-test field.
    """

    _assert_implementation_identity(task["implementation"])
    index = int(task["index"])
    work = Path(task["work"])
    raw_path = Path(task["raw_path"])
    expected = task["source_record"]
    before = raw_path.lstat()
    if raw_path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise SFTPreparationError(f"Raw shard is not a regular file: {raw_path}")
    if before.st_size != expected["bytes"] or file_sha256(raw_path) != expected["sha256"]:
        raise SFTPreparationError(f"Raw shard changed before contamination scan: {raw_path}")
    parquet_file = _validate_arrow_schema(raw_path)
    if parquet_file.metadata.num_rows != expected["rows"]:
        raise SFTPreparationError(f"Raw shard row count changed: {raw_path}")
    contamination_reason = _worker_contamination_guards(task["runtime"]["denylists"])
    groups: set[str] = set()
    reasons: Counter[str] = Counter()
    rows = 0
    for batch in parquet_file.iter_batches(
        batch_size=task["read_batch_size"],
        columns=["input", "output", "unit_tests"],
        use_threads=True,
    ):
        for row in batch.to_pylist():
            prompt = row.get("input")
            reason = contamination_reason(
                _content_for_benchmark_scan(prompt, row.get("output"), row.get("unit_tests"))
            )
            if reason is not None:
                reasons[reason] += 1
                if isinstance(prompt, str) and prompt.strip():
                    groups.add(prompt_group_id(prompt))
            rows += 1
    after = raw_path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SFTPreparationError(f"Raw shard changed during contamination scan: {raw_path}")
    if rows != expected["rows"]:
        raise SFTPreparationError(f"Contamination scan row count mismatch for {raw_path}")
    result = {
        "result_version": SHARD_RESULT_VERSION,
        "kind": "benchmark_contamination_groups",
        "curation_identity_sha256": task["curation_identity_sha256"],
        "source_index": index,
        "source": {
            "path": expected["path"],
            "rows": expected["rows"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        },
        "rows_scanned": rows,
        "matched_rows": sum(reasons.values()),
        "match_reasons": dict(sorted(reasons.items())),
        "contaminated_group_ids": sorted(groups),
    }
    result["result_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _atomic_bytes(_preflight_path(work, index), pretty_json_bytes(result))
    return result


def _load_valid_preflight(
    work: Path,
    index: int,
    expected: Mapping[str, Any],
    *,
    curation_identity_sha256: str,
) -> dict[str, Any] | None:
    path = _preflight_path(work, index)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        result = _load_json(path)
        recorded_sha = _require_sha256(
            result.pop("result_sha256"), field="preflight.result_sha256"
        )
        calculated_sha = sha256_bytes(canonical_json_bytes(result))
        result["result_sha256"] = recorded_sha
        if recorded_sha != calculated_sha:
            return None
        if (
            result.get("result_version") != SHARD_RESULT_VERSION
            or result.get("kind") != "benchmark_contamination_groups"
            or result.get("curation_identity_sha256") != curation_identity_sha256
            or result.get("source_index") != index
            or result.get("rows_scanned") != expected["rows"]
        ):
            return None
        if result.get("source") != {
            "path": expected["path"],
            "rows": expected["rows"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            return None
        groups = result.get("contaminated_group_ids")
        if (
            not isinstance(groups, list)
            or groups != sorted(set(groups))
            or any(
                not isinstance(group_id, str)
                or not group_id.startswith("prompt-sha256:")
                or len(group_id) != len("prompt-sha256:") + 64
                or any(
                    character not in HEX64
                    for character in group_id[len("prompt-sha256:") :]
                )
                for group_id in groups
            )
        ):
            return None
        reasons = result.get("match_reasons")
        if not isinstance(reasons, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in reasons.items()
        ):
            return None
        if (
            not isinstance(result.get("matched_rows"), int)
            or isinstance(result["matched_rows"], bool)
            or result["matched_rows"] < 0
            or result["matched_rows"] > expected["rows"]
            or sum(reasons.values()) != result["matched_rows"]
        ):
            return None
        return result
    except (KeyError, OSError, SFTPreparationError, TypeError, ValueError):
        return None


def _process_shard(task: dict[str, Any]) -> dict[str, Any]:
    _assert_implementation_identity(task["implementation"])
    index = int(task["index"])
    work = Path(task["work"])
    raw_path = Path(task["raw_path"])
    expected = task["source_record"]
    before = raw_path.lstat()
    if raw_path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise SFTPreparationError(f"Raw shard is not a regular file: {raw_path}")
    if before.st_size != expected["bytes"]:
        raise SFTPreparationError(f"Raw shard byte size changed: {raw_path}")
    if file_sha256(raw_path) != expected["sha256"]:
        raise SFTPreparationError(f"Raw shard checksum changed: {raw_path}")

    tokenizer = _load_tokenizer(
        Path(task["tokenizer_root"]),
        expected_sha256=task["tokenizer_json_sha256"],
    )
    renderer = _load_runtime_renderer(tokenizer)
    if renderer.format_id != task["policy"]["chat_format_id"]:
        raise SFTPreparationError("Renderer format does not match the frozen policy")
    split_policy = SplitPolicy(**task["runtime"]["split"])
    filter_policy = FilterPolicy(**task["runtime"]["filter"])
    contaminated_groups = frozenset(task.get("contaminated_groups", ()))
    parquet_file = _validate_arrow_schema(raw_path)
    if parquet_file.metadata.num_rows != expected["rows"]:
        raise SFTPreparationError(f"Raw shard row count changed: {raw_path}")

    accepted_schema, rejected_schema = _schemas()
    sinks = {
        "train": _ParquetSink(_output_record(work, "train", index), accepted_schema, flush_rows=task["output_row_group_size"]),
        "validation": _ParquetSink(_output_record(work, "validation", index), accepted_schema, flush_rows=task["output_row_group_size"]),
        "rejected": _ParquetSink(_output_record(work, "rejected", index), rejected_schema, flush_rows=task["output_row_group_size"]),
    }
    counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_row = 0
    started = time.monotonic()
    try:
        for batch in parquet_file.iter_batches(
            batch_size=task["read_batch_size"],
            columns=[
                "id",
                "input",
                "output",
                "domain",
                "generation_algorithm",
                "average_test_score",
            ],
            use_threads=True,
        ):
            for row in batch.to_pylist():
                decision = decide_row(
                    row,
                    source_shard=expected["path"],
                    source_row=source_row,
                    tokenizer=tokenizer,
                    renderer=renderer,
                    split_policy=split_policy,
                    filter_policy=filter_policy,
                    max_sequence_length=int(task["policy"]["max_sequence_length"]),
                    # The authenticated first pass already scanned every row
                    # and supplied the global contaminated-group set.
                    contamination_reason=None,
                    contaminated_groups=contaminated_groups,
                )
                if decision.example is not None:
                    record = decision.example.as_record()
                    sinks[decision.example.split].append(record)
                    counts[decision.example.split] += 1
                    token_counts[decision.example.split] += decision.example.rendered_tokens - 1
                    domain_counts[f"{decision.example.split}:{decision.example.domain}"] += 1
                else:
                    assert decision.reason is not None
                    sinks["rejected"].append(
                        {
                            "id": decision.example_id,
                            "group_id": decision.group_id,
                            "reason": decision.reason,
                            "rendered_tokens": decision.rendered_tokens,
                            "source_shard": expected["path"],
                            "source_row": source_row,
                        }
                    )
                    counts[f"rejected:{decision.reason}"] += 1
                source_row += 1
        files = {name: sink.finish() for name, sink in sinks.items()}
    except BaseException:
        for sink in sinks.values():
            sink.abort()
        raise
    after = raw_path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SFTPreparationError(f"Raw shard changed while processing: {raw_path}")
    if source_row != expected["rows"]:
        raise SFTPreparationError(f"Processed row count mismatch for {raw_path}")
    if counts["train"] + counts["validation"] + sum(
        value for key, value in counts.items() if key.startswith("rejected:")
    ) != source_row:
        raise SFTPreparationError(f"Decision accounting mismatch for {raw_path}")

    result = {
        "result_version": SHARD_RESULT_VERSION,
        "curation_identity_sha256": task["curation_identity_sha256"],
        "contamination_authority_sha256": task["contamination_authority_sha256"],
        "source_index": index,
        "source": {
            "path": expected["path"],
            "rows": expected["rows"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        },
        "counts": dict(sorted(counts.items())),
        "tokens": dict(sorted(token_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
        "files": files,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    result["result_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _atomic_bytes(_sidecar_path(work, index), pretty_json_bytes(result))
    return result


def _load_valid_result(
    work: Path,
    index: int,
    expected: Mapping[str, Any],
    *,
    curation_identity_sha256: str,
    contamination_authority_sha256: str,
) -> dict[str, Any] | None:
    sidecar = _sidecar_path(work, index)
    if not sidecar.is_file() or sidecar.is_symlink():
        return None
    try:
        result = _load_json(sidecar)
        recorded_sha = _require_sha256(
            result.pop("result_sha256"), field="shard.result_sha256"
        )
        calculated_sha = sha256_bytes(canonical_json_bytes(result))
        result["result_sha256"] = recorded_sha
        if recorded_sha != calculated_sha:
            return None
        if (
            result.get("result_version") != SHARD_RESULT_VERSION
            or result.get("source_index") != index
            or result.get("curation_identity_sha256") != curation_identity_sha256
            or result.get("contamination_authority_sha256")
            != contamination_authority_sha256
        ):
            return None
        if result.get("source") != {
            "path": expected["path"],
            "rows": expected["rows"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            return None
        counts = result.get("counts")
        if not isinstance(counts, dict):
            return None
        for key, value in counts.items():
            if (
                not isinstance(key, str)
                or key not in ("train", "validation") and not key.startswith("rejected:")
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return None
        rejected_rows = sum(
            value for key, value in counts.items() if key.startswith("rejected:")
        )
        expected_file_rows = {
            "train": counts.get("train", 0),
            "validation": counts.get("validation", 0),
            "rejected": rejected_rows,
        }
        if sum(expected_file_rows.values()) != expected["rows"]:
            return None
        if set(result["files"]) != {"train", "validation", "rejected"}:
            return None
        accepted_schema, rejected_schema = _schemas()
        for kind, file_record in result["files"].items():
            path = _output_record(work, kind, index)
            if file_record is None:
                if expected_file_rows[kind] != 0 or path.exists() or path.is_symlink():
                    return None
                continue
            if not isinstance(file_record, dict):
                return None
            if set(file_record) != {"path", "rows", "bytes", "sha256"}:
                return None
            if (
                path.name != file_record["path"]
                or file_record["rows"] != expected_file_rows[kind]
                or not isinstance(file_record["bytes"], int)
                or isinstance(file_record["bytes"], bool)
                or file_record["bytes"] <= 0
            ):
                return None
            _require_sha256(file_record["sha256"], field=f"files.{kind}.sha256")
            output_bytes, output_sha = _regular_file_stat_sha256(
                path, label=f"{kind} Parquet output"
            )
            if output_bytes != file_record["bytes"] or output_sha != file_record["sha256"]:
                return None
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            expected_schema = rejected_schema if kind == "rejected" else accepted_schema
            if (
                parquet_file.metadata.num_rows != file_record["rows"]
                or not parquet_file.schema_arrow.equals(expected_schema, check_metadata=False)
            ):
                return None
        return result
    except (KeyError, OSError, SFTPreparationError, TypeError, ValueError):
        return None


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    files: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "rejected": []}
    for result in results:
        counts.update(result["counts"])
        tokens.update(result["tokens"])
        domains.update(result["domains"])
        for kind in files:
            file_record = result["files"][kind]
            if file_record is not None:
                files[kind].append(file_record)
    decided = counts["train"] + counts["validation"] + sum(
        value for key, value in counts.items() if key.startswith("rejected:")
    )
    if decided != EXPECTED_SOURCE_ROWS:
        raise SFTPreparationError(f"Global decision accounting is {decided:,}, expected {EXPECTED_SOURCE_ROWS:,}")
    return {
        "counts": dict(sorted(counts.items())),
        "tokens": dict(sorted(tokens.items())),
        "domains": dict(sorted(domains.items())),
        "files": files,
    }


def _remove_runtime_inputs(work: Path) -> None:
    """Remove private runtime snapshots before publishing the HF dataset."""

    inputs = work / "inputs"
    if not inputs.exists():
        return
    if inputs.is_symlink() or not inputs.is_dir():
        raise SFTPreparationError(f"Unsafe runtime input directory: {inputs}")
    for path in inputs.iterdir():
        if path.is_symlink() or not path.is_file():
            raise SFTPreparationError(f"Unsafe runtime snapshot entry: {path}")
        path.unlink()
    _fsync_directory(inputs)
    inputs.rmdir()
    _fsync_directory(work)


def _validate_published_output(
    output: Path,
    completion: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    records: list[dict[str, Any]],
    expected_identity: Mapping[str, Any],
) -> None:
    """Authenticate the complete published tree, including its exact file set."""

    if output.is_symlink() or not output.is_dir():
        raise SFTPreparationError(f"Unsafe published output: {output}")
    completion_path = output / "COMPLETION.json"
    if _load_json(completion_path) != completion:
        raise SFTPreparationError("Published COMPLETION.json changed while validating")
    if (
        completion.get("completion_version") != COMPLETION_VERSION
        or completion.get("kind") != "prime_rl_sft_messages_dataset"
        or completion.get("status") != "complete"
        or completion.get("dataset_manifest") != "DATASET_MANIFEST.json"
    ):
        raise SFTPreparationError("Published completion contract is invalid")
    manifest_path = output / "DATASET_MANIFEST.json"
    readme_path = output / "README.md"
    identity_path = output / "IDENTITY.json"
    if _load_json(manifest_path) != manifest:
        raise SFTPreparationError("Published DATASET_MANIFEST.json changed while validating")
    manifest_sha = _regular_file_sha256(manifest_path, label="published dataset manifest")
    if manifest_sha != _require_sha256(
        completion.get("dataset_manifest_sha256"),
        field="completion.dataset_manifest_sha256",
    ):
        raise SFTPreparationError("Published manifest is not authenticated by completion")
    readme_sha = _regular_file_sha256(readme_path, label="published README")
    if readme_sha != _require_sha256(
        completion.get("readme_sha256"), field="completion.readme_sha256"
    ):
        raise SFTPreparationError("Published README is not authenticated by completion")
    published_identity = _load_json(identity_path)
    if published_identity != expected_identity:
        raise SFTPreparationError("Published IDENTITY.json does not match the requested curation")
    identity_body = dict(published_identity)
    recorded_identity_sha = _require_sha256(
        identity_body.pop("identity_sha256", None), field="identity.identity_sha256"
    )
    if recorded_identity_sha != sha256_bytes(canonical_json_bytes(identity_body)):
        raise SFTPreparationError("Published IDENTITY.json has an invalid digest")
    if manifest.get("manifest_version") != MANIFEST_VERSION or manifest.get("kind") != completion.get("kind"):
        raise SFTPreparationError("Published dataset manifest contract is invalid")

    manifest_identity = manifest.get("identity")
    if not isinstance(manifest_identity, dict):
        raise SFTPreparationError("Published dataset manifest has no identity")
    identity_sha = expected_identity["identity_sha256"]
    expected_manifest_fields = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "implementation": expected_identity["implementation"],
        **expected_identity["source"],
        "curation_identity_sha256": identity_sha,
        "policy_sha256": expected_identity["policy_sha256"],
        "policy": expected_identity["policy"],
        "tokenizer_repo_id": TOKENIZER_REPO_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        **expected_identity["tokenizer"],
        "chat_format_id": expected_identity["policy"]["chat_format_id"],
        "max_sequence_length": expected_identity["policy"]["max_sequence_length"],
        "benchmark_denylists": [
            {"path": record["name"], "sha256": record["sha256"]}
            for record in expected_identity["runtime"]["denylists"]
        ],
    }
    for field, value in expected_manifest_fields.items():
        if manifest_identity.get(field) != value:
            raise SFTPreparationError(f"Published manifest identity field {field} changed")

    contamination_path = output / "audit" / "CONTAMINATION_GROUPS.json"
    if _regular_file_sha256(
        contamination_path, label="published contamination authority"
    ) != _require_sha256(
        manifest_identity.get("contamination_groups_sha256"),
        field="manifest.identity.contamination_groups_sha256",
    ):
        raise SFTPreparationError("Published contamination authority checksum mismatch")
    contamination = _load_json(contamination_path)
    contamination_body = dict(contamination)
    contamination_sha = _require_sha256(
        contamination_body.pop("authority_sha256", None),
        field="contamination.authority_sha256",
    )
    if contamination_sha != sha256_bytes(canonical_json_bytes(contamination_body)):
        raise SFTPreparationError("Published contamination authority digest is invalid")
    if contamination.get("curation_identity_sha256") != identity_sha:
        raise SFTPreparationError("Published contamination authority has the wrong identity")
    if (
        contamination.get("format_version") != FORMAT_VERSION
        or contamination.get("kind") != "global_benchmark_contamination_groups"
        or contamination.get("benchmark_denylists")
        != [
            {"path": record["name"], "sha256": record["sha256"]}
            for record in expected_identity["runtime"]["denylists"]
        ]
    ):
        raise SFTPreparationError("Published contamination authority contract is invalid")

    results: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    for index, source_record in enumerate(records):
        preflight = _load_valid_preflight(
            output,
            index,
            source_record,
            curation_identity_sha256=identity_sha,
        )
        if preflight is None:
            raise SFTPreparationError(f"Published contamination sidecar {index} is invalid")
        result = _load_valid_result(
            output,
            index,
            source_record,
            curation_identity_sha256=identity_sha,
            contamination_authority_sha256=str(contamination_sha),
        )
        if result is None:
            raise SFTPreparationError(f"Published curation sidecar {index} is invalid")
        preflights.append(preflight)
        results.append(result)
    if contamination.get("source_sidecars") != [item["result_sha256"] for item in preflights]:
        raise SFTPreparationError("Contamination authority does not bind every source sidecar")
    expected_contaminated_groups = sorted(
        {
            group_id
            for preflight in preflights
            for group_id in preflight["contaminated_group_ids"]
        }
    )
    if (
        contamination.get("matched_rows")
        != sum(preflight["matched_rows"] for preflight in preflights)
        or contamination.get("contaminated_group_ids") != expected_contaminated_groups
        or manifest_identity.get("contaminated_groups")
        != len(expected_contaminated_groups)
        or manifest_identity.get("benchmark_matched_rows")
        != contamination.get("matched_rows")
    ):
        raise SFTPreparationError("Contamination authority totals do not match source sidecars")
    if manifest.get("shards") != [
        {"source_index": index, "result_sha256": result["result_sha256"]}
        for index, result in enumerate(results)
    ]:
        raise SFTPreparationError("Dataset manifest does not bind every curation sidecar")
    aggregate = _aggregate(results)
    if manifest.get("statistics") != aggregate:
        raise SFTPreparationError("Published aggregate statistics do not match shard sidecars")
    if (
        completion.get("train_rows") != aggregate["counts"].get("train", 0)
        or completion.get("validation_rows") != aggregate["counts"].get("validation", 0)
        or completion.get("source_rows") != EXPECTED_SOURCE_ROWS
    ):
        raise SFTPreparationError("Published completion row totals are invalid")

    data_records = aggregate["files"]["train"] + aggregate["files"]["validation"]
    rejected_records = aggregate["files"]["rejected"]
    expected_data_names = {record["path"] for record in data_records}
    expected_audit_names = {
        "CONTAMINATION_GROUPS.json",
        *(f"contamination-{index:05d}.json" for index in range(len(records))),
        *(f"shard-{index:05d}.json" for index in range(len(records))),
        *(record["path"] for record in rejected_records),
    }
    data_directory = output / "data"
    audit_directory = output / "audit"
    if data_directory.is_symlink() or audit_directory.is_symlink():
        raise SFTPreparationError("Published data/audit directories cannot be symlinks")
    if not data_directory.is_dir() or not audit_directory.is_dir():
        raise SFTPreparationError("Published data/audit entries must be directories")
    data_entries = list(data_directory.iterdir())
    audit_entries = list(audit_directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in data_entries):
        raise SFTPreparationError("Published data directory contains a non-regular file")
    if any(path.is_symlink() or not path.is_file() for path in audit_entries):
        raise SFTPreparationError("Published audit directory contains a non-regular file")
    if {path.name for path in data_entries} != expected_data_names:
        raise SFTPreparationError("Published data directory contains a missing or rogue file")
    if {path.name for path in audit_entries} != expected_audit_names:
        raise SFTPreparationError("Published audit directory contains a missing or rogue file")
    top_level_entries = list(output.iterdir())
    if {path.name for path in top_level_entries} != {
        "COMPLETION.json",
        "DATASET_MANIFEST.json",
        "IDENTITY.json",
        "README.md",
        "audit",
        "data",
    }:
        raise SFTPreparationError("Published dataset contains an unexpected top-level entry")
    for path in top_level_entries:
        if path.name in {"data", "audit"}:
            if path.is_symlink() or not path.is_dir():
                raise SFTPreparationError(f"Published directory is unsafe: {path}")
        elif path.is_symlink() or not path.is_file():
            raise SFTPreparationError(f"Published authority is not a regular file: {path}")


def validate_published_dataset(output: Path) -> dict[str, Any]:
    """Authenticate an already-published SFT dataset without mutable inputs.

    Publication validation originally depended on the raw download inventory
    and the caller's in-memory curation identity.  A trainer must be able to
    repeat the same exact-tree check immediately before launch without reading
    the five-million-row raw snapshot.  The published ``IDENTITY.json`` and
    the 50 self-authenticating shard sidecars contain the same immutable
    authorities, so this entry point reconstructs those inputs and delegates
    to :func:`_validate_published_output`.

    This is an integrity and provenance gate, not an adversarial signature:
    the repository and published directory remain trusted authorities.
    """

    output = Path(output).expanduser()
    if output.is_symlink() or not output.is_dir():
        raise SFTPreparationError(f"Missing or unsafe published dataset: {output}")
    output = output.resolve(strict=True)

    authority_names = ("COMPLETION.json", "DATASET_MANIFEST.json", "IDENTITY.json")
    for name in authority_names:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise SFTPreparationError(f"Missing or unsafe published authority: {path}")
    completion = _load_json(output / "COMPLETION.json")
    if completion.get("dataset_manifest") != "DATASET_MANIFEST.json":
        raise SFTPreparationError("Published completion selects an unexpected manifest")
    manifest = _load_json(output / "DATASET_MANIFEST.json")
    identity = _load_json(output / "IDENTITY.json")

    expected_identity_keys = {
        "format_version",
        "implementation",
        "source",
        "tokenizer",
        "policy_sha256",
        "policy",
        "runtime",
        "identity_sha256",
    }
    if set(identity) != expected_identity_keys or identity.get("format_version") != FORMAT_VERSION:
        raise SFTPreparationError("Published curation identity contract is invalid")
    identity_body = dict(identity)
    recorded_identity_sha = _require_sha256(
        identity_body.pop("identity_sha256", None), field="identity.identity_sha256"
    )
    if recorded_identity_sha != sha256_bytes(canonical_json_bytes(identity_body)):
        raise SFTPreparationError("Published curation identity digest is invalid")

    implementation = identity.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {
        "contracts",
        "files",
        "packages",
    }:
        raise SFTPreparationError("Published implementation identity contract is invalid")
    contracts = implementation.get("contracts")
    if contracts != {
        "format_version": FORMAT_VERSION,
        "shard_result_version": SHARD_RESULT_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "completion_version": COMPLETION_VERSION,
    }:
        raise SFTPreparationError("Published implementation contract versions are unsupported")
    implementation_files = implementation.get("files")
    if not isinstance(implementation_files, dict) or set(implementation_files) != set(
        IMPLEMENTATION_FILES
    ):
        raise SFTPreparationError("Published implementation file identity is incomplete")
    for name, record in implementation_files.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
        ):
            raise SFTPreparationError(f"Published implementation file {name} is invalid")
        _require_sha256(record.get("sha256"), field=f"identity.implementation.files.{name}")
    packages = implementation.get("packages")
    if not isinstance(packages, dict) or set(packages) != set(IMPLEMENTATION_PACKAGES):
        raise SFTPreparationError("Published implementation package identity is incomplete")
    if any(not isinstance(version, str) or not version for version in packages.values()):
        raise SFTPreparationError("Published implementation package version is invalid")

    source_identity = identity.get("source")
    tokenizer_identity = identity.get("tokenizer")
    runtime_identity = identity.get("runtime")
    policy = identity.get("policy")
    if not isinstance(source_identity, dict) or set(source_identity) != {
        "source_manifest_sha256",
        "inventory_sha256",
    }:
        raise SFTPreparationError("Published source identity contract is invalid")
    if not isinstance(tokenizer_identity, dict) or set(tokenizer_identity) != {
        "tokenizer_manifest_sha256",
        "tokenizer_json_sha256",
    }:
        raise SFTPreparationError("Published tokenizer identity contract is invalid")
    for label, values in (("source", source_identity), ("tokenizer", tokenizer_identity)):
        for field, value in values.items():
            _require_sha256(value, field=f"identity.{label}.{field}")
    _require_sha256(identity.get("policy_sha256"), field="identity.policy_sha256")
    if not isinstance(policy, dict):
        raise SFTPreparationError("Published curation policy is not an object")
    try:
        validate_policy_payload(policy)
        expected_split = asdict(SplitPolicy(**policy["split"]))
        expected_filter = asdict(FilterPolicy(**policy["filter"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SFTPreparationError(f"Published curation policy is invalid: {exc}") from exc
    if not isinstance(runtime_identity, dict) or set(runtime_identity) != {
        "split",
        "filter",
        "denylists",
    }:
        raise SFTPreparationError("Published runtime identity contract is invalid")
    if runtime_identity.get("split") != expected_split or runtime_identity.get("filter") != expected_filter:
        raise SFTPreparationError("Published runtime policy resolution is inconsistent")
    denylists = runtime_identity.get("denylists")
    configured_denylists = policy.get("benchmark_denylists")
    if not isinstance(denylists, list) or not isinstance(configured_denylists, list):
        raise SFTPreparationError("Published benchmark denylist identity is invalid")
    if len(denylists) != len(configured_denylists):
        raise SFTPreparationError("Published benchmark denylist count changed")
    for index, (record, configured) in enumerate(zip(denylists, configured_denylists)):
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "sha256"}
            or not isinstance(configured, str)
            or record.get("name") != Path(configured).name
        ):
            raise SFTPreparationError(f"Published benchmark denylist {index} is inconsistent")
        _require_sha256(record.get("sha256"), field=f"identity.runtime.denylists[{index}].sha256")

    records: list[dict[str, Any]] = []
    for index in range(EXPECTED_SOURCE_SHARDS):
        sidecar = _sidecar_path(output, index)
        if sidecar.is_symlink() or not sidecar.is_file():
            raise SFTPreparationError(f"Published curation sidecar {index} is missing or unsafe")
        payload = _load_json(sidecar)
        source = payload.get("source")
        expected_path = f"data/train-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.parquet"
        if (
            payload.get("source_index") != index
            or not isinstance(source, dict)
            or set(source) != {"path", "rows", "bytes", "sha256"}
            or source.get("path") != expected_path
            or not isinstance(source.get("rows"), int)
            or source["rows"] <= 0
            or not isinstance(source.get("bytes"), int)
            or source["bytes"] <= 0
        ):
            raise SFTPreparationError(f"Published source record {index} is invalid")
        _require_sha256(source.get("sha256"), field=f"published.source[{index}].sha256")
        records.append(dict(source))
    if sum(record["rows"] for record in records) != EXPECTED_SOURCE_ROWS:
        raise SFTPreparationError("Published source row inventory changed")
    if sum(record["bytes"] for record in records) != EXPECTED_SOURCE_BYTES:
        raise SFTPreparationError("Published source byte inventory changed")

    _validate_published_output(
        output,
        completion,
        manifest,
        records=records,
        expected_identity=identity,
    )
    accepted_files = (
        manifest["statistics"]["files"]["train"]
        + manifest["statistics"]["files"]["validation"]
    )
    return {
        "output": str(output),
        "completion": completion,
        "manifest": manifest,
        "identity": identity,
        "dataset_manifest_sha256": _regular_file_sha256(
            output / "DATASET_MANIFEST.json", label="published dataset manifest"
        ),
        "curation_identity_sha256": recorded_identity_sha,
        "data_bytes": sum(int(record["bytes"]) for record in accepted_files),
    }


def _dataset_card(manifest: Mapping[str, Any]) -> str:
    counts = manifest["statistics"]["counts"]
    return f"""---
license: cc-by-4.0
pretty_name: OpenCodeInstruct Prime SFT v1
task_categories:
- text-generation
tags:
- code
- sft
---

# OpenCodeInstruct Prime SFT v1

Framework-neutral, assistant-supervised `messages` projection of
`{SOURCE_REPO_ID}@{SOURCE_REVISION}` for PrimeRL SFT.

- Train examples: {counts.get('train', 0):,}
- Validation examples: {counts.get('validation', 0):,}
- Rejected examples: {sum(v for k, v in counts.items() if k.startswith('rejected:')):,}
- License: {SOURCE_LICENSE}; attribution remains NVIDIA/OpenCodeInstruct.
- Chat format: `{manifest['identity']['chat_format_id']}`
- Tokenizer: `{TOKENIZER_REPO_ID}@{TOKENIZER_REVISION}`

Only `messages` and non-sensitive provenance/quality columns are published.
Unit tests, execution traces, LLM judgements, and benchmark content are not
copied into the derived corpus.  `DATASET_MANIFEST.json` and `COMPLETION.json`
are the machine-readable authorities.  MBPP remains final-evaluation-only.
"""


def _publish_final(
    work: Path,
    output: Path,
    *,
    root: Path,
    policy: Mapping[str, Any],
    runtime: Mapping[str, Any],
    curation_identity_sha256: str,
    source_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    source_records: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    statistics = _aggregate(results)
    contamination_path = work / "audit" / "CONTAMINATION_GROUPS.json"
    contamination = _load_json(contamination_path)
    contamination_body = dict(contamination)
    recorded_authority_sha = contamination_body.pop("authority_sha256", None)
    if recorded_authority_sha != sha256_bytes(canonical_json_bytes(contamination_body)):
        raise SFTPreparationError("Global contamination authority digest is invalid")
    if contamination.get("curation_identity_sha256") != curation_identity_sha256:
        raise SFTPreparationError("Global contamination authority belongs to another curation")
    _remove_runtime_inputs(work)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "prime_rl_sft_messages_dataset",
        "identity": {
            "source_repo_id": SOURCE_REPO_ID,
            "source_revision": SOURCE_REVISION,
            "source_license": SOURCE_LICENSE,
            "implementation": expected_identity["implementation"],
            **source_identity,
            "curation_identity_sha256": curation_identity_sha256,
            "policy_sha256": runtime["policy_sha256"],
            "policy": policy,
            "tokenizer_repo_id": TOKENIZER_REPO_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
            **tokenizer_identity,
            "chat_format_id": policy["chat_format_id"],
            "max_sequence_length": policy["max_sequence_length"],
            "benchmark_denylists": [
                {"path": record["source_name"], "sha256": record["sha256"]}
                for record in runtime["denylists"]
            ],
            "contamination_groups_sha256": _regular_file_sha256(
                contamination_path, label="global contamination authority"
            ),
            "contaminated_groups": len(contamination.get("contaminated_group_ids", [])),
            "benchmark_matched_rows": contamination.get("matched_rows", 0),
        },
        "statistics": statistics,
        "schema": {
            "training_columns": [
                "id",
                "group_id",
                "messages",
                "domain",
                "generation_algorithm",
                "average_test_score",
                "rendered_tokens",
                "source_revision",
                "source_shard",
                "source_row",
            ],
            "messages": ["user", "assistant"],
            "omitted_raw_columns": ["llm_judgement", "unit_tests", "tests_execution_status"],
        },
        "source_root_recorded_as": str(root),
        "shards": [
            {"source_index": result["source_index"], "result_sha256": result["result_sha256"]}
            for result in results
        ],
    }
    manifest_path = work / "DATASET_MANIFEST.json"
    _atomic_bytes(manifest_path, pretty_json_bytes(manifest))
    _atomic_bytes(work / "README.md", _dataset_card(manifest).encode("utf-8"))
    completion = {
        "completion_version": COMPLETION_VERSION,
        "kind": manifest["kind"],
        "status": "complete",
        "dataset_manifest": manifest_path.name,
        "dataset_manifest_sha256": _regular_file_sha256(
            manifest_path, label="dataset manifest"
        ),
        "readme_sha256": _regular_file_sha256(work / "README.md", label="dataset README"),
        "train_rows": statistics["counts"].get("train", 0),
        "validation_rows": statistics["counts"].get("validation", 0),
        "source_rows": EXPECTED_SOURCE_ROWS,
    }
    _atomic_bytes(work / "COMPLETION.json", pretty_json_bytes(completion))
    _validate_published_output(
        work,
        completion,
        manifest,
        records=source_records,
        expected_identity=expected_identity,
    )
    # Flush every regular file and every nested directory before the one
    # atomic visibility change, then make that parent-directory rename durable.
    _fsync_tree(work)
    os.replace(work, output)
    _fsync_directory(output.parent)
    return {"manifest": manifest, "completion": completion}


def prepare_dataset(
    *,
    root: Path,
    output: Path,
    tokenizer_root: Path,
    policy_path: Path,
    workers: int,
    read_batch_size: int,
    output_row_group_size: int,
) -> dict[str, Any]:
    if workers < 1 or workers > 64:
        raise SFTPreparationError("workers must be between 1 and 64")
    if read_batch_size < 1 or output_row_group_size < 1:
        raise SFTPreparationError("batch and row-group sizes must be positive")
    root = _safe_directory(root, label="raw SFT root", must_exist=True)
    tokenizer_root = _safe_directory(tokenizer_root, label="tokenizer root", must_exist=True)
    output = _safe_directory(output, label="derived output", must_exist=False)
    # Derived data may live beneath ROOT/derived, but it may never replace an
    # ancestor, the SFT root itself, or anything beneath raw/.
    if output == root or output in root.parents:
        raise SFTPreparationError("Derived output cannot replace the raw SFT root or an ancestor")
    if output == root / "raw" or root / "raw" in output.parents:
        raise SFTPreparationError("Derived output cannot live inside the raw snapshot")

    # Resolve every requested immutable identity before either resuming work or
    # accepting an already-published dataset.  A completed directory is not a
    # cache hit for a different source/tokenizer/policy/denylist contract.
    implementation_identity = _implementation_identity()
    source_identity, records = _validate_source(root)
    tokenizer_identity = _validate_tokenizer(tokenizer_root)
    policy, runtime = _resolve_policy(policy_path)
    identity_body = {
        "format_version": FORMAT_VERSION,
        "implementation": implementation_identity,
        "source": source_identity,
        "tokenizer": tokenizer_identity,
        "policy_sha256": runtime["policy_sha256"],
        "policy": policy,
        "runtime": _runtime_identity(runtime),
    }
    curation_identity_sha256 = sha256_bytes(canonical_json_bytes(identity_body))
    identity = {**identity_body, "identity_sha256": curation_identity_sha256}

    completion_path = output / "COMPLETION.json"
    if output.exists():
        if not completion_path.is_symlink() and completion_path.is_file():
            completion = _load_json(completion_path)
            manifest_path = output / str(completion.get("dataset_manifest", ""))
            if (
                completion.get("status") == "complete"
                and completion.get("dataset_manifest") == "DATASET_MANIFEST.json"
                and not manifest_path.is_symlink()
                and manifest_path.is_file()
                and _regular_file_sha256(
                    manifest_path, label="published dataset manifest"
                )
                == completion.get("dataset_manifest_sha256")
            ):
                manifest = _load_json(manifest_path)
                if manifest.get("identity", {}).get("curation_identity_sha256") != curation_identity_sha256:
                    raise SFTPreparationError(
                        "Published output belongs to a different curation identity"
                    )
                _validate_published_output(
                    output,
                    completion,
                    manifest,
                    records=records,
                    expected_identity=identity,
                )
                return {
                    "already_complete": True,
                    "completion": completion,
                    "manifest": manifest,
                    "output": str(output),
                }
        raise SFTPreparationError(f"Refusing existing non-authoritative output: {output}")

    work = output.with_name(f".{output.name}.work")
    lock_path = output.with_name(f".{output.name}.lock")
    if work.is_symlink() or lock_path.is_symlink():
        raise SFTPreparationError("Work and lock paths cannot be symlinks")
    work.mkdir(parents=True, exist_ok=True)
    for subdirectory in (work / "data", work / "audit"):
        subdirectory.mkdir(parents=True, exist_ok=True)
        if subdirectory.is_symlink():
            raise SFTPreparationError(f"Unsafe work subdirectory: {subdirectory}")

    identity_path = work / "IDENTITY.json"
    if identity_path.exists() or identity_path.is_symlink():
        if _read_regular_bytes(identity_path, label="work identity") != pretty_json_bytes(identity):
            raise SFTPreparationError("Existing work directory belongs to a different curation identity")
    else:
        _atomic_bytes(identity_path, pretty_json_bytes(identity))
    inputs, runtime = _snapshot_runtime_inputs(
        work,
        tokenizer_root=tokenizer_root,
        tokenizer_identity=tokenizer_identity,
        runtime=runtime,
    )

    preflights: dict[int, dict[str, Any]] = {}
    preflight_tasks = []
    for index, record in enumerate(records):
        existing = _load_valid_preflight(
            work,
            index,
            record,
            curation_identity_sha256=curation_identity_sha256,
        )
        if existing is not None:
            preflights[index] = existing
            continue
        try:
            _preflight_path(work, index).unlink()
        except FileNotFoundError:
            pass
        preflight_tasks.append(
            {
                "index": index,
                "work": str(work),
                "raw_path": str(root / "raw" / record["path"]),
                "source_record": record,
                "runtime": runtime,
                "implementation": implementation_identity,
                "curation_identity_sha256": curation_identity_sha256,
                "read_batch_size": read_batch_size,
            }
        )
    if preflight_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_scan_contamination_shard, task): task["index"]
                for task in preflight_tasks
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                preflights[index] = result
                print(
                    json.dumps(
                        {
                            "event": "sft_contamination_scan_complete",
                            "source_index": index,
                            "complete_shards": len(preflights),
                            "total_shards": len(records),
                            "matched_rows": result["matched_rows"],
                            "contaminated_groups": len(result["contaminated_group_ids"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    contaminated_groups = sorted(
        {
            group_id
            for index in range(len(records))
            for group_id in preflights[index]["contaminated_group_ids"]
        }
    )
    contamination_authority = {
        "format_version": FORMAT_VERSION,
        "kind": "global_benchmark_contamination_groups",
        "curation_identity_sha256": curation_identity_sha256,
        "benchmark_denylists": [
            {"path": record["source_name"], "sha256": record["sha256"]}
            for record in runtime["denylists"]
        ],
        "matched_rows": sum(preflights[index]["matched_rows"] for index in range(len(records))),
        "contaminated_group_ids": contaminated_groups,
        "source_sidecars": [preflights[index]["result_sha256"] for index in range(len(records))],
    }
    contamination_authority["authority_sha256"] = sha256_bytes(
        canonical_json_bytes(contamination_authority)
    )
    _atomic_bytes(work / "audit" / "CONTAMINATION_GROUPS.json", pretty_json_bytes(contamination_authority))
    contamination_authority_sha256 = contamination_authority["authority_sha256"]

    tasks = []
    results: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        existing = _load_valid_result(
            work,
            index,
            record,
            curation_identity_sha256=curation_identity_sha256,
            contamination_authority_sha256=contamination_authority_sha256,
        )
        if existing is not None:
            results[index] = existing
            continue
        # Remove only this deterministic worker's stale outputs and crash
        # partials; raw data and completed sidecars are never touched.  A
        # symlink or non-regular partial is refused instead of followed.
        for kind in ("train", "validation", "rejected"):
            output_record = _output_record(work, kind, index)
            partial = _partial_output_path(output_record)
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise SFTPreparationError(f"Unsafe stale partial output: {partial}")
            try:
                output_record.unlink()
            except FileNotFoundError:
                pass
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
        try:
            _sidecar_path(work, index).unlink()
        except FileNotFoundError:
            pass
        tasks.append(
            {
                "index": index,
                "work": str(work),
                "raw_path": str(root / "raw" / record["path"]),
                "source_record": record,
                "tokenizer_root": str(inputs),
                "tokenizer_json_sha256": tokenizer_identity["tokenizer_json_sha256"],
                "policy": policy,
                "runtime": runtime,
                "implementation": implementation_identity,
                "contaminated_groups": contaminated_groups,
                "curation_identity_sha256": curation_identity_sha256,
                "contamination_authority_sha256": contamination_authority_sha256,
                "read_batch_size": read_batch_size,
                "output_row_group_size": output_row_group_size,
            }
        )

    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(_process_shard, task): task["index"] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
                print(
                    json.dumps(
                        {
                            "event": "sft_shard_complete",
                            "source_index": index,
                            "complete_shards": len(results),
                            "total_shards": len(records),
                            "train": result["counts"].get("train", 0),
                            "validation": result["counts"].get("validation", 0),
                            "rejected": sum(v for k, v in result["counts"].items() if k.startswith("rejected:")),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    ordered = [results[index] for index in range(len(records))]
    publication = _publish_final(
        work,
        output,
        root=root,
        policy=policy,
        runtime=runtime,
        curation_identity_sha256=curation_identity_sha256,
        source_identity=source_identity,
        tokenizer_identity=tokenizer_identity,
        expected_identity=identity,
        source_records=records,
        results=ordered,
    )
    return {"already_complete": False, "output": str(output), **publication}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="certified raw OpenCodeInstruct root")
    parser.add_argument(
        "--output",
        type=Path,
        help="derived dataset path (default: ROOT/derived/prime-sft-v1)",
    )
    parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--read-batch-size", type=int, default=2_048)
    parser.add_argument("--output-row-group-size", type=int, default=4_096)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output = args.output or args.root / "derived" / DEFAULT_OUTPUT_NAME
    lock_path = output.with_name(f".{output.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SFTPreparationError(f"Lock path is not a regular file: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SFTPreparationError(f"Another SFT curator holds {lock_path}") from exc
        try:
            result = prepare_dataset(
                root=args.root,
                output=output,
                tokenizer_root=args.tokenizer_root,
                policy_path=args.policy,
                workers=args.workers,
                read_batch_size=args.read_batch_size,
                output_row_group_size=args.output_row_group_size,
            )
        finally:
            os.close(descriptor)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, SFTPreparationError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
