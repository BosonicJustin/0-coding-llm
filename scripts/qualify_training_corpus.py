#!/usr/bin/env python3
"""Fail closed unless a materialized corpus is safe for six-rank pre-training.

This is the final, cold-path acceptance gate for ``curated-packed-corpus``
outputs.  It composes the canonical packed/order/tokenizer validators, scans
the authenticated document-position indexes, and publishes one atomic JSON
receipt.  A passing receipt is evidence about the exact bytes inspected; it is
not a substitute for the separately bound launch/run authority.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain import data as training_data  # noqa: E402
from pretrain import hf_export as hf_export_module  # noqa: E402
from pretrain import materialize as materialize_module  # noqa: E402
from pretrain import model as model_module  # noqa: E402
from pretrain import tokenizer_identity as tokenizer_identity_module  # noqa: E402
from pretrain.hf_export import load_model_config_json  # noqa: E402
from pretrain.materialize import (  # noqa: E402
    DOCUMENT_INDEX_FORMAT,
    DOCUMENT_INDEX_VERSION,
    FORMAT as CORPUS_FORMAT,
    FORMAT_VERSION as CORPUS_FORMAT_VERSION,
    SPLITS,
    iter_jsonl_zst,
)
from pretrain.model import ModelConfig  # noqa: E402
from pretrain.tokenizer_identity import (  # noqa: E402
    TokenizerIdentityError,
    verify_tokenizer_identity,
)


QUALIFICATION_FORMAT = "materialized-pretraining-corpus-qualification"
QUALIFICATION_VERSION = 1
EXPECTED_WEIGHTS = {"python": 0.4, "other_code": 0.4, "english": 0.2}
DEFAULT_TARGETS = {
    "train": 52_580_000_000,
    "validation": 500_000_000,
    "test": 500_000_000,
}
IDENTITY_KINDS = {0: "source", 1: "split_group"}
SPLIT_MASKS = {split: 1 << index for index, split in enumerate(SPLITS)}
_HEX = frozenset("0123456789abcdef")


class QualificationError(RuntimeError):
    """Raised when the corpus cannot be accepted for training."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = None if details is None else dict(details)


@dataclass(frozen=True)
class QualificationConfig:
    corpus_root: Path
    tokenizer_root: Path
    output: Path
    model_config: Path | None = None
    expected_targets: Mapping[str, int] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_TARGETS)
    )
    expected_weights: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: dict(EXPECTED_WEIGHTS)
    )
    mixture_absolute_tolerance: float = 1e-6
    maximum_target_shortfall_fraction: float = 1e-3
    sample_rows_per_domain: int = 8
    sample_seed: int = 20_260_901
    world_size: int = 6
    scratch_directory: Path | None = None
    split_identity_batch_rows: int = 100_000
    collision_examples: int = 10

    def validate(self) -> None:
        if set(self.expected_targets) != set(SPLITS):
            raise QualificationError(
                f"Expected token targets must contain exactly {tuple(SPLITS)}"
            )
        for split, value in self.expected_targets.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise QualificationError(f"Expected {split} token target must be positive")
        if set(self.expected_weights) != set(training_data.DOMAIN_ORDER):
            raise QualificationError(
                "Expected mixture must contain exactly python, other_code, and english"
            )
        weights = [float(self.expected_weights[domain]) for domain in training_data.DOMAIN_ORDER]
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise QualificationError("Expected mixture weights must be finite and positive")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise QualificationError("Expected mixture weights must sum exactly to one")
        if (
            not math.isfinite(self.mixture_absolute_tolerance)
            or self.mixture_absolute_tolerance < 0
        ):
            raise QualificationError("Mixture tolerance must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_target_shortfall_fraction)
            or not 0 <= self.maximum_target_shortfall_fraction < 1
        ):
            raise QualificationError(
                "Maximum target shortfall fraction must be in [0, 1)"
            )
        for name, value in (
            ("sample_rows_per_domain", self.sample_rows_per_domain),
            ("world_size", self.world_size),
            ("split_identity_batch_rows", self.split_identity_batch_rows),
            ("collision_examples", self.collision_examples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise QualificationError(f"{name} must be a positive integer")
        if self.world_size != 6:
            raise QualificationError("This production gate requires exactly six DDP ranks")
        if (
            isinstance(self.sample_seed, bool)
            or not isinstance(self.sample_seed, int)
            or self.sample_seed < 0
        ):
            raise QualificationError("sample_seed must be a non-negative integer")
        if self.output.name in ("", ".", ".."):
            raise QualificationError("output must name a fresh generation directory")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_tree(root: Path) -> dict[str, dict[str, Any]]:
    """Capture mutation-sensitive metadata without rereading every payload."""

    result: dict[str, dict[str, Any]] = {}

    def descriptor(path: Path) -> dict[str, Any]:
        metadata = path.lstat()
        kind = (
            "symlink"
            if path.is_symlink()
            else "directory"
            if path.is_dir()
            else "file"
        )
        return {
            "kind": kind,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }

    try:
        result["."] = descriptor(root)
    except OSError as exc:
        raise QualificationError(f"Cannot stat corpus root {root}: {exc}") from exc
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError as exc:
        raise QualificationError(f"Cannot inventory corpus tree {root}: {exc}") from exc
    for path in paths:
        try:
            result[path.relative_to(root).as_posix()] = descriptor(path)
        except OSError as exc:
            raise QualificationError(f"Cannot stat corpus entry {path}: {exc}") from exc
    return result


def _inventory_digest(inventory: Mapping[str, Any]) -> str:
    payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validator_identity() -> dict[str, Any]:
    sources = {
        "qualification": Path(__file__),
        "pretrain_data": Path(training_data.__file__),
        "materialize_contract": Path(materialize_module.__file__),
        "tokenizer_identity": Path(tokenizer_identity_module.__file__),
        "model": Path(model_module.__file__),
        "model_config_loader": Path(hf_export_module.__file__),
    }
    artifacts: dict[str, Any] = {}
    for name, raw_path in sources.items():
        path = _regular_file(raw_path.resolve(strict=True), label=f"validator source {name}")
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "sources": artifacts,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": training_data.torch.__version__,
            "sqlite_version": sqlite3.sqlite_version,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "zstandard_version": importlib.metadata.version("zstandard"),
            "packed_format_version": training_data.FORMAT_VERSION,
            "order_format_version": training_data.ORDER_FORMAT_VERSION,
        },
    }


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path.resolve(strict=True), label=label)
    metadata = regular.stat()
    return {
        "path": str(regular),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": _sha256(regular),
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualificationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        # Reuse the data contract's duplicate-key and non-finite-number parser.
        return training_data._load_json(path)  # type: ignore[attr-defined]
    except (OSError, TypeError, ValueError) as exc:
        raise QualificationError(f"Invalid JSON object {path}: {exc}") from exc


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _safe_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise QualificationError(f"{label} has no relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise QualificationError(f"{label} path is unsafe: {relative!r}")
    path = root.joinpath(*pure.parts)
    _regular_file(path, label=label)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise QualificationError(f"{label} escapes the corpus root: {relative!r}") from exc
    if resolved != path.absolute():
        raise QualificationError(f"{label} traverses a symlink: {path}")
    return path


def _descriptor_file(
    root: Path,
    descriptor: Any,
    *,
    label: str,
    verify_bytes: bool = False,
) -> tuple[Path, str]:
    if not isinstance(descriptor, Mapping):
        raise QualificationError(f"{label} descriptor must be an object")
    path = _safe_file(root, descriptor.get("path"), label=label)
    expected_sha = _require_sha256(descriptor.get("sha256"), label=f"{label} SHA-256")
    if verify_bytes:
        expected_bytes = descriptor.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or path.stat().st_size != expected_bytes
        ):
            raise QualificationError(f"{label} byte count differs from its descriptor")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise QualificationError(
            f"{label} checksum mismatch: expected {expected_sha}, found {actual_sha}"
        )
    return path, actual_sha


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def publish_receipt(
    output_directory: Path, receipt: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Publish one immutable receipt+sidecar generation by directory rename."""

    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    parent = output_directory.parent.resolve(strict=True)
    destination = parent / output_directory.name
    lock_path = parent / ".training-corpus-qualification.publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace qualification generation: {destination}"
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.part-", dir=parent)
        )
        try:
            receipt_path = staging / "qualification.json"
            sidecar_path = staging / "qualification.json.sha256"
            _atomic_write(receipt_path, payload)
            _atomic_write(
                sidecar_path,
                f"{digest}  qualification.json\n".encode("ascii"),
            )
            directory_fd = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.rename(staging, destination)
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            if staging.exists() and staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return destination / "qualification.json", destination / "qualification.json.sha256"


def _verify_root_and_sidecar(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise QualificationError(
            f"Corpus root must be a regular non-symlink directory: {root}"
        )
    root = root.resolve(strict=True)
    manifest_path = _regular_file(root / "manifest.json", label="corpus manifest")
    sidecar_path = _regular_file(root / "manifest.sha256", label="corpus sidecar")
    digest = _sha256(manifest_path)
    expected_sidecar = f"{digest}  manifest.json\n"
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise QualificationError(f"Cannot read corpus sidecar: {exc}") from exc
    if sidecar != expected_sidecar:
        raise QualificationError("Corpus manifest sidecar does not authenticate manifest.json")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format") != CORPUS_FORMAT
        or manifest.get("format_version") != CORPUS_FORMAT_VERSION
    ):
        raise QualificationError("Unsupported materialized corpus format/version")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise QualificationError(f"Corpus must contain exactly the splits {tuple(SPLITS)}")
    source_cursor = manifest.get("source_cursor")
    if (
        not isinstance(source_cursor, Mapping)
        or isinstance(source_cursor.get("archive_count"), bool)
        or not isinstance(source_cursor.get("archive_count"), int)
        or source_cursor["archive_count"] < 1
        or source_cursor.get("next_archive") != source_cursor["archive_count"]
    ):
        raise QualificationError("Corpus source cursor does not prove complete materialization")
    journal = root / ".materialization-journal.json"
    if journal.exists() or journal.is_symlink():
        raise QualificationError("Completed corpus still contains a materialization journal")
    return manifest, {
        "manifest_path": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": digest,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
    }


def _validate_provenance(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    provenance = manifest.get("provenance")
    required = {"source", "policy", "tokenizer", "fingerprints"}
    if not isinstance(provenance, Mapping) or not required.issubset(provenance):
        raise QualificationError(
            f"Corpus provenance must contain at least {sorted(required)}"
        )
    result: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, descriptor in sorted(provenance.items()):
        path, digest = _descriptor_file(
            root, descriptor, label=f"{name} provenance"
        )
        payloads[name] = _load_json(path)
        result[name] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise QualificationError("Corpus manifest has no materialization identity")
    selection_sha = _require_sha256(
        identity.get("selection_manifest_sha256"),
        label="materialization selection manifest SHA-256",
    )
    tokenizer_sha = _require_sha256(
        identity.get("tokenizer_manifest_sha256"),
        label="materialization tokenizer manifest SHA-256",
    )
    if payloads["source"].get("selection_manifest_sha256") != selection_sha:
        raise QualificationError("Source provenance is bound to another selection manifest")
    if payloads["tokenizer"].get("tokenizer_manifest_sha256") != tokenizer_sha:
        raise QualificationError("Tokenizer provenance is bound to another tokenizer manifest")
    materialization_policy_sha = _require_sha256(
        identity.get("curation_policy_sha256"),
        label="materialization curation policy SHA-256",
    )
    # The materializer hashes the canonical parsed policy, while curation's
    # provenance retains the source-file digest. Both are authenticated by the
    # top-level manifest but they are intentionally different hash domains.
    source_policy_sha = _require_sha256(
        payloads["policy"].get("curation_policy_sha256"),
        label="source curation policy SHA-256",
    )

    isolation = manifest.get("split_isolation")
    if not isinstance(isolation, Mapping) or isolation.get("physical_outputs_separate") is not True:
        raise QualificationError("Corpus does not assert physically separate split outputs")
    if isolation.get("authoritative_assignment") not in (
        "frozen_leakage_safe_source_groups",
        "completed-curation-decision-shards",
    ):
        raise QualificationError("Corpus has no supported leakage-safe split authority")
    leakage = isolation.get("curation_leakage_audit")
    if not isinstance(leakage, Mapping):
        raise QualificationError("Corpus does not preserve the curation leakage audit")
    required_zero = (
        "content_hashes_in_multiple_splits",
        "canonical_clusters_in_multiple_splits",
        "source_groups_in_multiple_splits",
        "cross_bucket_code_repo_groups_in_multiple_splits",
    )
    for field in required_zero:
        if leakage.get(field) != 0:
            raise QualificationError(f"Authenticated curation leakage audit failed: {field}")
    if (
        "normalized_hashes_in_multiple_splits" in leakage
        and leakage["normalized_hashes_in_multiple_splits"] != 0
    ):
        raise QualificationError(
            "Authenticated curation leakage audit failed: normalized hashes"
        )
    raw_archives = payloads["source"].get("raw_archives")
    if not isinstance(raw_archives, list) or not raw_archives:
        raise QualificationError("Source provenance has no raw-archive inventory")
    archive_names: list[str] = []
    seen_archive_names: set[str] = set()
    for ordinal, descriptor in enumerate(raw_archives):
        if not isinstance(descriptor, Mapping):
            raise QualificationError(
                f"Source provenance raw archive {ordinal} is not an object"
            )
        archive = descriptor.get("archive")
        if not isinstance(archive, str) or not archive:
            raise QualificationError(
                f"Source provenance raw archive {ordinal} has no identity"
            )
        if archive in seen_archive_names:
            raise QualificationError(f"Source provenance repeats raw archive {archive!r}")
        _require_sha256(
            descriptor.get("sha256"),
            label=f"source raw archive {ordinal} SHA-256",
        )
        for field in ("documents", "content_tokens"):
            value = descriptor.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QualificationError(
                    f"Source raw archive {ordinal} has invalid {field}"
                )
        archive_names.append(archive)
        seen_archive_names.add(archive)
    source_cursor = manifest["source_cursor"]
    if source_cursor["archive_count"] != len(archive_names):
        raise QualificationError(
            "Corpus source cursor differs from authenticated raw-archive inventory"
        )
    result["bindings"] = {
        "selection_manifest_sha256": selection_sha,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "materialization_curation_policy_sha256": materialization_policy_sha,
        "source_curation_policy_sha256": source_policy_sha,
        "split_authority": isolation["authoritative_assignment"],
        "authenticated_leakage_audit": dict(leakage),
        "raw_archive_count": len(archive_names),
        "raw_archive_inventory_sha256": hashlib.sha256(
            json.dumps(
                raw_archives,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    return result, tuple(archive_names)


def _load_expected_model(config_path: Path | None) -> tuple[ModelConfig, dict[str, Any]]:
    if config_path is None:
        model = ModelConfig()
        return model, {
            "source": "pretrain.model.ModelConfig defaults",
            "sha256": None,
            "config": dataclasses.asdict(model),
        }
    path = _regular_file(config_path, label="model config")
    try:
        model = load_model_config_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise QualificationError(f"Invalid model config {path}: {exc}") from exc
    return model, {
        "source": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
        "config": dataclasses.asdict(model),
    }


def _validate_tokenizer(
    tokenizer_root: Path,
    *,
    tokenizer_manifest_sha256: str,
    vocab_size: int,
    eos_token_id: int,
) -> dict[str, Any]:
    try:
        identity = verify_tokenizer_identity(
            tokenizer_root,
            expected_manifest_sha256=tokenizer_manifest_sha256,
            expected_vocab_size=vocab_size,
        )
    except (OSError, RuntimeError, TokenizerIdentityError) as exc:
        raise QualificationError(f"Tokenizer authentication failed: {exc}") from exc
    validation = identity.manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise QualificationError("Tokenizer manifest has no validation object")
    if validation.get("eos_token_id") != eos_token_id:
        raise QualificationError(
            "Tokenizer EOS token ID differs from the packed corpus"
        )
    return {
        "root": str(Path(tokenizer_root).resolve(strict=True)),
        "manifest_path": str(identity.manifest_path),
        "manifest_sha256": identity.manifest_sha256,
        "vocabulary_sha256": identity.vocabulary_sha256,
        "vocab_size": identity.vocab_size,
        "eos_token_id": eos_token_id,
    }


def _sample_seed(seed: int, split: str, domain: str) -> int:
    payload = f"{seed}\0{split}\0{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _sample_packed_rows(
    manifest_path: Path,
    *,
    split: str,
    domain: str,
    domain_id: int,
    sample_rows: int,
    seed: int,
) -> dict[str, Any]:
    dataset = training_data.PackedShardDataset(manifest_path)
    try:
        if len(dataset) < 1:
            raise QualificationError(f"Packed dataset is empty: {split}/{domain}")
        count = min(sample_rows, len(dataset))
        row_ids = sorted(
            random.Random(_sample_seed(seed, split, domain)).sample(range(len(dataset)), count)
        )
        rows: list[dict[str, Any]] = []
        row_receipts: list[dict[str, Any]] = []
        for row_id in row_ids:
            first = dataset[row_id]
            second = dataset[row_id]
            if not np.array_equal(first["tokens"], second["tokens"]) or not np.array_equal(
                first["starts"], second["starts"]
            ):
                raise QualificationError(
                    f"Repeated packed read changed bytes for {split}/{domain} row {row_id}"
                )
            digest = hashlib.sha256(
                first["tokens"].tobytes(order="C")
                + first["starts"].tobytes(order="C")
            ).hexdigest()
            row_receipts.append({"row_id": row_id, "sha256": digest})
            rows.append(
                {
                    **first,
                    "domain_id": domain_id,
                    "sample_reference": int(training_data.encode_reference(domain_id, row_id)),
                }
            )
        collator = training_data.PackedBatchCollator(
            dataset.sequence_length,
            vocab_size=dataset.vocab_size,
            eos_token_id=dataset.eos_token_id,
        )
        batch = collator(rows)
        starts = np.stack([row["starts"] for row in rows], axis=0)
        unpacked = np.unpackbits(
            starts,
            axis=1,
            count=dataset.tokens_per_row,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        observed_mask = batch["labels"].numpy() == training_data.IGNORE_INDEX
        expected_mask = unpacked[:, 1:]
        if not np.array_equal(observed_mask, expected_mask):
            raise QualificationError(
                f"Loss mask differs from segment boundaries for {split}/{domain}"
            )
        if np.any(batch["input_ids"].numpy() >= dataset.vocab_size):
            raise QualificationError(f"Sample token is outside vocabulary for {split}/{domain}")
        if np.any(batch["position_ids"].numpy()[unpacked[:, :-1]] != 0):
            raise QualificationError(
                f"Position IDs do not reset at boundaries for {split}/{domain}"
            )
        loss_tokens = int(batch["num_loss_tokens"].item())
        if loss_tokens < 1:
            raise QualificationError(f"Sample has no supervised labels for {split}/{domain}")
        return {
            "algorithm": "sha256-derived-seed-python-random-sample-without-replacement",
            "seed": _sample_seed(seed, split, domain),
            "requested_rows": sample_rows,
            "sampled_rows": count,
            "valid_loss_tokens": loss_tokens,
            "masked_boundary_labels": int(observed_mask.sum()),
            "rows": row_receipts,
        }
    finally:
        dataset.close()


def _shard_row_offsets(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    offset = 0
    result: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        rows = int(shard["rows"])
        result.append(
            {
                "index": int(shard["index"]),
                "row_start": offset,
                "row_end_exclusive": offset + rows,
                "rows": rows,
                "token_bytes": int(shard["tokens"]["bytes"]),
                "starts_bytes": int(shard["starts"]["bytes"]),
                "tokens_sha256": shard["tokens"]["sha256"],
                "starts_sha256": shard["starts"]["sha256"],
            }
        )
        offset += rows
    if offset != manifest["rows"]:
        raise QualificationError("Packed shard row offsets do not cover the manifest")
    return result


def _validate_order_and_packed(
    root: Path,
    manifest: Mapping[str, Any],
    config: QualificationConfig,
    model: ModelConfig,
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[Path, dict[str, Any]]], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    packed_by_key: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    common: dict[str, Any] | None = None
    for split in SPLITS:
        split_descriptor = manifest["splits"].get(split)
        if not isinstance(split_descriptor, Mapping):
            raise QualificationError(f"Missing split descriptor: {split}")
        order_descriptor = split_descriptor.get("order")
        order_path, order_sha = _descriptor_file(
            root, order_descriptor, label=f"{split} order manifest"
        )
        try:
            order = training_data.validate_training_order(order_path)
        except (OSError, TypeError, ValueError) as exc:
            raise QualificationError(f"Invalid {split} order: {exc}") from exc
        if order.get("split") != split:
            raise QualificationError(f"Order split mismatch: expected {split}")
        for field in (
            "format_version",
            "rows",
            "packed_available_rows",
            "packed_surplus_rows",
        ):
            if order_descriptor.get(field) != order.get(field):
                raise QualificationError(f"Top-level {split} order {field} differs")
        actual_tokens = order["input_token_budget"]["actual_total"]
        if order_descriptor.get("authorized_input_tokens") != actual_tokens:
            raise QualificationError(
                f"Top-level {split} authorized token count differs from order"
            )
        expected_target = int(config.expected_targets[split])
        budget = order["input_token_budget"]
        if budget.get("expected_total") != expected_target:
            raise QualificationError(
                f"{split} order target is {budget.get('expected_total')}, "
                f"expected {expected_target}"
            )
        shortfall = expected_target - int(actual_tokens)
        maximum_shortfall = math.ceil(
            expected_target * config.maximum_target_shortfall_fraction
        )
        if not 0 <= shortfall <= maximum_shortfall:
            raise QualificationError(
                f"{split} token shortfall {shortfall} exceeds acceptance limit "
                f"{maximum_shortfall}"
            )
        weights = order.get("expected_input_token_weights")
        if not isinstance(weights, Mapping) or any(
            not math.isclose(
                float(weights.get(domain, -1)),
                float(config.expected_weights[domain]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for domain in training_data.DOMAIN_ORDER
        ):
            raise QualificationError(f"{split} order does not authorize the 40/40/20 mixture")
        realized = order.get("realized_input_token_weights")
        if not isinstance(realized, Mapping):
            raise QualificationError(f"{split} order has no realized mixture")
        mixture_delta = {
            domain: float(realized[domain]) - float(config.expected_weights[domain])
            for domain in training_data.DOMAIN_ORDER
        }
        if any(
            abs(delta) > config.mixture_absolute_tolerance
            for delta in mixture_delta.values()
        ):
            raise QualificationError(
                f"{split} realized mixture exceeds tolerance: {mixture_delta}"
            )
        if order["rows"] < config.world_size:
            raise QualificationError(
                f"{split} has {order['rows']} rows, fewer than {config.world_size} DDP ranks"
            )
        consumption = order["training_consumption"]
        if split == "train":
            global_rows = consumption.get("frozen_global_microbatch_rows")
            if (
                not isinstance(global_rows, int)
                or global_rows < config.world_size
                or global_rows % config.world_size
            ):
                raise QualificationError(
                    "Frozen train global microbatch must be a positive multiple of world size"
                )
            if consumption.get("dropped_tail_rows") != 0:
                raise QualificationError("Frozen train order drops rows or would require padding")
            if not isinstance(consumption.get("optimizer_updates"), int) or consumption[
                "optimizer_updates"
            ] < 1:
                raise QualificationError("Frozen train order has no complete optimizer update")
        elif consumption.get("frozen_global_microbatch_rows") is not None:
            raise QualificationError(f"Held-out {split} order unexpectedly freezes train geometry")

        order_common = {
            "sequence_length": order["sequence_length"],
            "vocab_size": order["vocab_size"],
            "eos_token_id": order["eos_token_id"],
            "tokenizer_manifest_sha256": order["tokenizer_manifest_sha256"],
        }
        if common is None:
            common = order_common
        elif order_common != common:
            raise QualificationError("Train/validation/test tokenizer or row geometry differs")
        packed_descriptors = split_descriptor.get("packed")
        if not isinstance(packed_descriptors, Mapping) or set(packed_descriptors) != set(
            training_data.DOMAIN_ORDER
        ):
            raise QualificationError(f"{split} does not contain all three packed domains")
        packed_summary: dict[str, Any] = {}
        sample_summary: dict[str, Any] = {}
        for domain_id, domain in enumerate(training_data.DOMAIN_ORDER):
            descriptor = packed_descriptors[domain]
            packed_path, packed_sha = _descriptor_file(
                root, descriptor, label=f"{split}/{domain} packed manifest"
            )
            try:
                packed = training_data.validate_packed_manifest(
                    packed_path, verify_checksums=True
                )
            except (OSError, TypeError, ValueError) as exc:
                raise QualificationError(f"Invalid packed {split}/{domain}: {exc}") from exc
            expected_fields = {
                "split": split,
                "domain": domain,
                **order_common,
            }
            for field, expected in expected_fields.items():
                if packed.get(field) != expected:
                    raise QualificationError(
                        f"Packed {split}/{domain} {field} differs: "
                        f"{packed.get(field)!r} != {expected!r}"
                    )
            if (
                packed.get("selection_manifest_sha256")
                != manifest["identity"]["selection_manifest_sha256"]
            ):
                raise QualificationError(
                    f"Packed {split}/{domain} is bound to another selection manifest"
                )
            if (
                packed.get("curation_policy_sha256")
                != manifest["identity"].get("curation_policy_sha256")
            ):
                raise QualificationError(
                    f"Packed {split}/{domain} is bound to another curation policy"
                )
            for field in ("rows", "documents", "source_content_tokens"):
                if descriptor.get(field) != packed.get(field):
                    raise QualificationError(
                        f"Top-level packed {split}/{domain} {field} differs"
                    )
            order_dataset_descriptor = order["dataset_manifests"][domain]
            if order_dataset_descriptor.get("sha256") != packed_sha:
                raise QualificationError(
                    f"{split} order is bound to another {domain} packed manifest"
                )
            packed_by_key[(split, domain)] = (packed_path, packed)
            packed_summary[domain] = {
                "manifest": str(packed_path.relative_to(root)),
                "manifest_sha256": packed_sha,
                "rows": packed["rows"],
                "documents": packed["documents"],
                "input_tokens": packed["input_tokens"],
                "valid_loss_tokens": packed["valid_loss_tokens"],
                "masked_boundary_labels": packed["masked_boundary_labels"],
                "source_content_tokens": packed["source_content_tokens"],
                "shards": _shard_row_offsets(packed),
            }
            sample_summary[domain] = _sample_packed_rows(
                packed_path,
                split=split,
                domain=domain,
                domain_id=domain_id,
                sample_rows=config.sample_rows_per_domain,
                seed=config.sample_seed,
            )
        summaries[split] = {
            "order": {
                "manifest": str(order_path.relative_to(root)),
                "manifest_sha256": order_sha,
                "rows": order["rows"],
                "rows_per_domain": order["rows_per_domain"],
                "input_tokens_per_domain": order["input_tokens_per_domain"],
                "valid_loss_tokens_per_domain": order["valid_loss_tokens_per_domain"],
                "expected_input_tokens": expected_target,
                "actual_input_tokens": actual_tokens,
                "shortfall_tokens": shortfall,
                "maximum_accepted_shortfall_tokens": maximum_shortfall,
                "declared_tolerance_tokens": budget["tolerance"],
                "realized_input_token_weights": dict(realized),
                "mixture_delta": mixture_delta,
                "training_consumption": dict(consumption),
            },
            "packed": packed_summary,
            "deterministic_samples": sample_summary,
        }
    assert common is not None
    identity = manifest["identity"]
    if common["tokenizer_manifest_sha256"] != identity["tokenizer_manifest_sha256"]:
        raise QualificationError("Orders are bound to another tokenizer manifest")
    packing_configuration = identity.get("packing_configuration")
    if not isinstance(packing_configuration, Mapping):
        raise QualificationError("Materialization packing configuration is invalid")
    expected_packing = {
        "sequence_length": common["sequence_length"],
        "expected_vocab_size": common["vocab_size"],
        "expected_eos_token_id": common["eos_token_id"],
    }
    for field, expected in expected_packing.items():
        if packing_configuration.get(field) != expected:
            raise QualificationError(
                f"Packing configuration {field} differs from packed data"
            )
    order_configuration = manifest.get("order_configuration")
    if not isinstance(order_configuration, Mapping):
        raise QualificationError("Materialization order configuration is invalid")
    expected_order_targets = {
        "expected_train_input_tokens": config.expected_targets["train"],
        "expected_validation_input_tokens": config.expected_targets["validation"],
        "expected_test_input_tokens": config.expected_targets["test"],
    }
    for field, expected in expected_order_targets.items():
        if order_configuration.get(field) != expected:
            raise QualificationError(
                f"Top-level order configuration {field} differs from acceptance policy"
            )
    if order_configuration.get("enforce_input_weights") is not True:
        raise QualificationError("Top-level order configuration does not enforce a mixture")
    configured_weights = order_configuration.get("expected_input_weights")
    if not isinstance(configured_weights, Mapping) or any(
        not math.isclose(
            float(configured_weights.get(domain, -1)),
            float(config.expected_weights[domain]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for domain in training_data.DOMAIN_ORDER
    ):
        raise QualificationError("Top-level order configuration mixture differs")
    train_consumption = summaries["train"]["order"]["training_consumption"]
    if (
        order_configuration.get("frozen_global_microbatch_rows")
        != train_consumption["frozen_global_microbatch_rows"]
        or order_configuration.get("frozen_gradient_accumulation_steps")
        != train_consumption["frozen_gradient_accumulation_steps"]
    ):
        raise QualificationError(
            "Top-level frozen training geometry differs from the train order"
        )
    if common["vocab_size"] != model.vocab_size:
        raise QualificationError(
            f"Corpus vocabulary {common['vocab_size']} differs from model {model.vocab_size}"
        )
    if common["sequence_length"] != model.max_seq_len:
        raise QualificationError(
            f"Corpus sequence length {common['sequence_length']} differs from model "
            f"{model.max_seq_len}"
        )
    return summaries, packed_by_key, common


class _SplitIdentityAudit:
    """Exact disk-backed union of source and group IDs across all splits."""

    def __init__(self, path: Path, *, batch_rows: int) -> None:
        self.path = path
        self.batch_rows = batch_rows
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.connection.execute("PRAGMA cache_size=-262144")
        self.connection.execute(
            "CREATE TABLE identities ("
            "kind INTEGER NOT NULL, identity BLOB NOT NULL, split_mask INTEGER NOT NULL, "
            "PRIMARY KEY(kind, identity)) WITHOUT ROWID"
        )
        # Coalesce repeated repository/source-group IDs inside each bounded
        # batch. Large repositories otherwise cause redundant SQLite updates
        # for every file while adding no split-disjointness information.
        self.pending: dict[tuple[int, bytes], int] = {}
        self.occurrences = {kind: 0 for kind in IDENTITY_KINDS}

    def add(self, kind: int, identity: bytes, split: str) -> None:
        self.occurrences[kind] += 1
        key = (kind, identity)
        self.pending[key] = self.pending.get(key, 0) | SPLIT_MASKS[split]
        if len(self.pending) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.connection.executemany(
            "INSERT INTO identities(kind, identity, split_mask) VALUES (?, ?, ?) "
            "ON CONFLICT(kind, identity) DO UPDATE SET "
            "split_mask=(identities.split_mask | excluded.split_mask)",
            (
                (kind, identity, split_mask)
                for (kind, identity), split_mask in self.pending.items()
            ),
        )
        self.connection.commit()
        self.pending.clear()

    def finish(self, *, example_limit: int) -> dict[str, Any]:
        self.flush()
        counts: dict[str, int] = {}
        collision_counts: dict[str, int] = {}
        examples: dict[str, list[dict[str, Any]]] = {}
        for kind, name in IDENTITY_KINDS.items():
            counts[name] = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM identities WHERE kind=?", (kind,)
                ).fetchone()[0]
            )
            where = "kind=? AND (split_mask & (split_mask - 1)) != 0"
            collision_counts[name] = int(
                self.connection.execute(
                    f"SELECT COUNT(*) FROM identities WHERE {where}", (kind,)
                ).fetchone()[0]
            )
            rows = self.connection.execute(
                f"SELECT hex(identity), split_mask FROM identities WHERE {where} "
                "ORDER BY identity LIMIT ?",
                (kind, example_limit),
            ).fetchall()
            examples[name] = [
                {
                    "identity_hex": str(identity_hex).lower(),
                    "splits": [
                        split
                        for split in SPLITS
                        if int(mask) & SPLIT_MASKS[split]
                    ],
                }
                for identity_hex, mask in rows
            ]
        self.connection.close()
        database_bytes = self.path.stat().st_size if self.path.exists() else 0
        return {
            "method": "exact-sqlite-without-rowid-split-bitmask-union",
            "identity_kinds": list(IDENTITY_KINDS.values()),
            "identity_occurrences": {
                IDENTITY_KINDS[kind]: count
                for kind, count in self.occurrences.items()
            },
            "unique_identities": counts,
            "cross_split_collisions": collision_counts,
            "collision_examples": examples,
            "scratch_database_bytes": database_bytes,
        }

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass


def _source_identity(row: Mapping[str, Any]) -> bytes:
    archive = row.get("source_archive")
    member = row.get("source_member")
    if not isinstance(archive, str) or not archive or not isinstance(member, str) or not member:
        raise QualificationError("Document-index row has an invalid source identity")
    digest = hashlib.sha256()
    digest.update(archive.encode("utf-8"))
    digest.update(b"\0")
    digest.update(member.encode("utf-8"))
    return digest.digest()


def _validate_document_indexes(
    root: Path,
    manifest: Mapping[str, Any],
    packed_by_key: Mapping[tuple[str, str], tuple[Path, Mapping[str, Any]]],
    common: Mapping[str, Any],
    config: QualificationConfig,
    archive_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scratch_parent: Path | None = None
    if config.scratch_directory is not None:
        scratch_parent = config.scratch_directory
        if scratch_parent.is_symlink() or not scratch_parent.is_dir():
            raise QualificationError(
                f"Scratch directory must be an existing non-symlink directory: {scratch_parent}"
            )
        scratch_parent = scratch_parent.resolve(strict=True)
    summaries: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="corpus-split-audit-", dir=scratch_parent) as temporary:
        audit = _SplitIdentityAudit(
            Path(temporary) / "split-identities.sqlite3",
            batch_rows=config.split_identity_batch_rows,
        )
        try:
            for split in SPLITS:
                split_summary: dict[str, Any] = {}
                for domain in training_data.DOMAIN_ORDER:
                    descriptor = manifest["splits"][split]["packed"][domain]
                    index_descriptor = descriptor.get("document_index")
                    index_path, index_sha = _descriptor_file(
                        root,
                        index_descriptor,
                        label=f"{split}/{domain} document-index manifest",
                    )
                    index = _load_json(index_path)
                    expected_identity = {
                        "format": DOCUMENT_INDEX_FORMAT,
                        "format_version": DOCUMENT_INDEX_VERSION,
                        "split": split,
                        "domain": domain,
                        "selection_manifest_sha256": manifest["identity"][
                            "selection_manifest_sha256"
                        ],
                        "tokenizer_manifest_sha256": common[
                            "tokenizer_manifest_sha256"
                        ],
                        "sequence_length": common["sequence_length"],
                    }
                    for field, expected in expected_identity.items():
                        if index.get(field) != expected:
                            raise QualificationError(
                                f"Document-index {field} differs for {split}/{domain}"
                            )
                    shards = index.get("shards")
                    if not isinstance(shards, list):
                        raise QualificationError(
                            f"Document-index has no shard inventory: {split}/{domain}"
                        )
                    logical_position = 0
                    documents = 0
                    selected_content_tokens = 0
                    last_ordinal = -1
                    shard_summaries: list[dict[str, Any]] = []
                    for shard_descriptor in shards:
                        if not isinstance(shard_descriptor, Mapping):
                            raise QualificationError("Document-index shard descriptor is invalid")
                        ordinal = shard_descriptor.get("archive_ordinal")
                        if (
                            isinstance(ordinal, bool)
                            or not isinstance(ordinal, int)
                            or ordinal < 0
                            or ordinal >= manifest["source_cursor"]["archive_count"]
                            or ordinal <= last_ordinal
                        ):
                            raise QualificationError(
                                f"Document-index shard order is invalid: {split}/{domain}"
                            )
                        last_ordinal = ordinal
                        expected_path = (
                            Path("provenance")
                            / "documents"
                            / split
                            / domain
                            / f"archive-{ordinal:06d}.jsonl.zst"
                        ).as_posix()
                        if shard_descriptor.get("path") != expected_path:
                            raise QualificationError(
                                f"Non-canonical document-index shard path: {split}/{domain}"
                            )
                        shard_path, shard_sha = _descriptor_file(
                            root,
                            shard_descriptor,
                            label=f"{split}/{domain} document-index shard {ordinal}",
                            verify_bytes=True,
                        )
                        shard_records = 0
                        try:
                            rows: Iterable[dict[str, Any]] = iter_jsonl_zst(shard_path)
                            for row in rows:
                                if (
                                    row.get("record_version") != DOCUMENT_INDEX_VERSION
                                    or row.get("split") != split
                                    or row.get("domain") != domain
                                    or row.get("source_archive_ordinal") != ordinal
                                    or row.get("source_archive") != archive_names[ordinal]
                                ):
                                    raise QualificationError(
                                        f"Document-index row identity differs: {shard_path}"
                                    )
                                for field in ("doc_id", "canonical_doc_id", "split_group_id"):
                                    _require_sha256(
                                        row.get(field),
                                        label=f"document-index {field}",
                                    )
                                for field in ("bucket", "language"):
                                    if (
                                        not isinstance(row.get(field), str)
                                        or not row[field].strip()
                                    ):
                                        raise QualificationError(
                                            f"Document-index row has invalid {field}: {shard_path}"
                                        )
                                source_index = row.get("source_manifest_index")
                                source_tokens = row.get("source_tokens")
                                selected = row.get("selected_content_tokens")
                                if (
                                    isinstance(source_index, bool)
                                    or not isinstance(source_index, int)
                                    or source_index < 0
                                    or isinstance(source_tokens, bool)
                                    or not isinstance(source_tokens, int)
                                    or isinstance(selected, bool)
                                    or not isinstance(selected, int)
                                    or not 1 <= selected <= source_tokens
                                ):
                                    raise QualificationError(
                                        "Document-index token/source counters are "
                                        f"invalid: {shard_path}"
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
                                    raise QualificationError(
                                        "Document-index logical offsets are "
                                        f"discontinuous: {shard_path}"
                                    )
                                group_id = bytes.fromhex(row["split_group_id"])
                                audit.add(0, _source_identity(row), split)
                                audit.add(1, group_id, split)
                                logical_position += selected + 1
                                selected_content_tokens += selected
                                documents += 1
                                shard_records += 1
                        except QualificationError:
                            raise
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            raise QualificationError(
                                f"Cannot scan document-index shard {shard_path}: {exc}"
                            ) from exc
                        if shard_records < 1 or shard_descriptor.get("records") != shard_records:
                            raise QualificationError(
                                f"Document-index shard record count differs: {shard_path}"
                            )
                        shard_summaries.append(
                            {
                                "archive_ordinal": ordinal,
                                "path": str(shard_path.relative_to(root)),
                                "bytes": shard_path.stat().st_size,
                                "sha256": shard_sha,
                                "records": shard_records,
                            }
                        )
                    expected_totals = {
                        "documents": documents,
                        "selected_content_tokens": selected_content_tokens,
                        "logical_stream_tokens": logical_position,
                    }
                    for field, expected in expected_totals.items():
                        if index.get(field) != expected:
                            raise QualificationError(
                                f"Document-index {field} total differs for {split}/{domain}"
                            )
                    packed = packed_by_key[(split, domain)][1]
                    if (
                        documents != packed["documents"]
                        or selected_content_tokens != packed["source_content_tokens"]
                        or logical_position != packed["stream_tokens"]
                    ):
                        raise QualificationError(
                            f"Document index and packed stream differ for {split}/{domain}"
                        )
                    if (
                        index_descriptor.get("documents") != documents
                        or index_descriptor.get("logical_stream_tokens") != logical_position
                    ):
                        raise QualificationError(
                            f"Top-level document-index totals differ for {split}/{domain}"
                        )
                    split_summary[domain] = {
                        "manifest": str(index_path.relative_to(root)),
                        "manifest_sha256": index_sha,
                        **expected_totals,
                        "shards": shard_summaries,
                    }
                summaries[split] = split_summary
            identity_summary = audit.finish(example_limit=config.collision_examples)
        except BaseException:
            audit.close()
            raise
        if any(identity_summary["cross_split_collisions"].values()):
            raise QualificationError(
                "Source or split-group identities occur in multiple splits: "
                f"{identity_summary['cross_split_collisions']}",
                details={"split_identity_audit": identity_summary},
            )
        if (
            identity_summary["identity_occurrences"]["source"]
            != identity_summary["unique_identities"]["source"]
        ):
            raise QualificationError(
                "A source archive/member identity occurs more than once in the corpus",
                details={"split_identity_audit": identity_summary},
            )
    return summaries, identity_summary


def qualify_corpus(config: QualificationConfig) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config.validate()
    if config.corpus_root.is_symlink():
        raise QualificationError("Corpus root must not be a symlink")
    root = config.corpus_root.resolve(strict=True)
    output_parent = config.output.parent.resolve(strict=True)
    if output_parent == root or output_parent.is_relative_to(root):
        raise QualificationError(
            "Qualification artifacts must be written outside the immutable corpus root"
        )
    if config.tokenizer_root.is_symlink():
        raise QualificationError("Tokenizer root must not be a symlink")
    tokenizer_root = config.tokenizer_root.resolve(strict=True)
    validator_before = _validator_identity()
    inventory_before = _snapshot_tree(root)
    tokenizer_inventory_before = _snapshot_tree(tokenizer_root)
    model_identity_before = (
        None
        if config.model_config is None
        else _file_identity(config.model_config, label="model config")
    )
    manifest, corpus_identity = _verify_root_and_sidecar(root)
    provenance, archive_names = _validate_provenance(root, manifest)
    model, model_identity = _load_expected_model(config.model_config)
    split_summaries, packed_by_key, common = _validate_order_and_packed(
        root, manifest, config, model
    )
    tokenizer = _validate_tokenizer(
        tokenizer_root,
        tokenizer_manifest_sha256=common["tokenizer_manifest_sha256"],
        vocab_size=common["vocab_size"],
        eos_token_id=common["eos_token_id"],
    )
    document_indexes, split_identity_audit = _validate_document_indexes(
        root, manifest, packed_by_key, common, config, archive_names
    )
    inventory_after = _snapshot_tree(root)
    if inventory_before != inventory_after:
        before_paths = set(inventory_before)
        after_paths = set(inventory_after)
        changed = sorted(
            path
            for path in before_paths & after_paths
            if inventory_before[path] != inventory_after[path]
        )
        raise QualificationError(
            "Corpus tree changed during qualification: "
            f"added={sorted(after_paths - before_paths)[:10]}, "
            f"removed={sorted(before_paths - after_paths)[:10]}, "
            f"changed={changed[:10]}"
        )
    tokenizer_inventory_after = _snapshot_tree(tokenizer_root)
    if tokenizer_inventory_before != tokenizer_inventory_after:
        raise QualificationError("Tokenizer tree changed during corpus qualification")
    model_identity_after = (
        None
        if config.model_config is None
        else _file_identity(config.model_config, label="model config")
    )
    if model_identity_before != model_identity_after:
        raise QualificationError("Model configuration changed during corpus qualification")
    validator_after = _validator_identity()
    if validator_before != validator_after:
        raise QualificationError("Validator implementation changed during qualification")
    corpus_identity["stable_tree_inventory"] = {
        "entries": len(inventory_after),
        "sha256": _inventory_digest(inventory_after),
    }
    tokenizer["stable_tree_inventory"] = {
        "entries": len(tokenizer_inventory_after),
        "sha256": _inventory_digest(tokenizer_inventory_after),
    }
    return {
        "format": QUALIFICATION_FORMAT,
        "format_version": QUALIFICATION_VERSION,
        "status": "pass",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "corpus": corpus_identity,
        "acceptance_policy": {
            "world_size": config.world_size,
            "expected_input_tokens": dict(config.expected_targets),
            "maximum_target_shortfall_fraction": (
                config.maximum_target_shortfall_fraction
            ),
            "expected_input_token_weights": dict(config.expected_weights),
            "mixture_absolute_tolerance": config.mixture_absolute_tolerance,
            "sample_rows_per_domain": config.sample_rows_per_domain,
            "sample_seed": config.sample_seed,
            "full_packed_payload_checksums": True,
            "full_packed_semantic_scan": True,
            "full_order_uniqueness_scan": True,
            "full_document_index_scan": True,
            "exact_disk_backed_split_identity_scan": True,
        },
        "checks": {
            "authenticated_corpus_manifest_and_sidecar": True,
            "authenticated_provenance": True,
            "tokenizer_model_and_corpus_compatible": True,
            "all_splits_and_domains_present": True,
            "orders_checksum_unique_and_in_range": True,
            "packed_shards_sizes_checksums_and_token_ranges": True,
            "boundary_bits_and_loss_masks_consistent": True,
            "no_padded_or_partial_training_rows": True,
            "exact_token_budgets_within_tolerance": True,
            "realized_mixture_within_tolerance": True,
            "document_offsets_and_totals_consistent": True,
            "source_and_group_identities_disjoint_across_splits": True,
            "deterministic_random_reads": True,
            "sufficient_rows_for_six_rank_ddp": True,
            "stable_corpus_identity_during_qualification": True,
        },
        "model": model_identity,
        "validator": validator_after,
        "tokenizer": tokenizer,
        "common_data_contract": dict(common),
        "provenance": provenance,
        "splits": split_summaries,
        "document_indexes": document_indexes,
        "split_identity_audit": split_identity_audit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="fresh immutable generation directory for qualification.json and its sidecar",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="exact ModelConfig JSON; omission binds the checked-in 1.284B defaults",
    )
    parser.add_argument("--expected-train-input-tokens", type=int, default=DEFAULT_TARGETS["train"])
    parser.add_argument(
        "--expected-validation-input-tokens",
        type=int,
        default=DEFAULT_TARGETS["validation"],
    )
    parser.add_argument("--expected-test-input-tokens", type=int, default=DEFAULT_TARGETS["test"])
    parser.add_argument("--maximum-target-shortfall-fraction", type=float, default=1e-3)
    parser.add_argument("--mixture-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--sample-rows-per-domain", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20_260_901)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument(
        "--scratch-directory",
        type=Path,
        help="existing fast local directory for the exact split-identity SQLite audit",
    )
    parser.add_argument("--split-identity-batch-rows", type=int, default=100_000)
    parser.add_argument("--collision-examples", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.corpus_root.exists():
            resolved_root = args.corpus_root.resolve(strict=True)
            intended_parent = args.output.parent.resolve(strict=False)
            if intended_parent == resolved_root or intended_parent.is_relative_to(
                resolved_root
            ):
                parser.error(
                    "Qualification artifacts must be outside the immutable corpus root"
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() or args.output.is_symlink():
            parser.error(
                f"Refusing to replace qualification generation: {args.output}"
            )
    except OSError as exc:
        parser.error(f"Cannot create qualification output directory: {exc}")
    config = QualificationConfig(
        corpus_root=args.corpus_root,
        tokenizer_root=args.tokenizer_root,
        output=args.output,
        model_config=args.model_config,
        expected_targets={
            "train": args.expected_train_input_tokens,
            "validation": args.expected_validation_input_tokens,
            "test": args.expected_test_input_tokens,
        },
        mixture_absolute_tolerance=args.mixture_absolute_tolerance,
        maximum_target_shortfall_fraction=args.maximum_target_shortfall_fraction,
        sample_rows_per_domain=args.sample_rows_per_domain,
        sample_seed=args.sample_seed,
        world_size=args.world_size,
        scratch_directory=args.scratch_directory,
        split_identity_batch_rows=args.split_identity_batch_rows,
        collision_examples=args.collision_examples,
    )
    started = datetime.now(timezone.utc).isoformat()
    try:
        receipt = qualify_corpus(config)
        code = 0
    except Exception as exc:
        receipt = {
            "format": QUALIFICATION_FORMAT,
            "format_version": QUALIFICATION_VERSION,
            "status": "fail",
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_root": str(args.corpus_root),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **(
                {"details": exc.details}
                if isinstance(exc, QualificationError) and exc.details is not None
                else {}
            ),
        }
        code = 1
    try:
        destination, sidecar = publish_receipt(args.output, receipt)
    except Exception as exc:
        parser.error(f"Cannot publish qualification artifact: {exc}")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "qualification": str(destination),
                "sidecar": str(sidecar),
                **({"error": receipt["error"]} if code else {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
