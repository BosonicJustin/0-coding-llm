#!/usr/bin/env python3
"""Fail-closed production pre-training preflight and ``torchrun`` launcher.

This command deliberately performs a metadata-only data check: it validates
the order and packed-manifest schemas, identities, paths, and recorded/actual
file sizes without reading ``order.bin`` or packed token/start payload bytes.
The expensive checksum/semantic scan is certified separately by
``scripts/certify_pretraining_data.py`` and its exact evidence is mandatory for
execution.

The accepted 1.3B execute path is exactly six-rank DDP and additionally
requires a self-bound immutable run authority whose canonical argv matches the
current invocation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import netrc
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.model import ModelConfig  # noqa: E402
from pretrain import data as training_data  # noqa: E402
from pretrain.run_authority import (  # noqa: E402
    RunAuthorityError,
    canonical_sha256,
    validate_run_authority,
)
from pretrain.tokenizer_identity import (  # noqa: E402
    TokenizerIdentityError,
    verify_tokenizer_identity,
)


ResumeGeneration = Literal["none", "latest", "previous"]
WandbMode = Literal["disabled", "offline", "online"]
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NETWORK_FILESYSTEMS = frozenset(
    {
        "afs",
        "ceph",
        "cifs",
        "davfs",
        "fuse.ceph",
        "fuse.glusterfs",
        "fuse.sshfs",
        "gcsfuse",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "s3fs",
        "smb3",
        "smbfs",
    }
)
_EPHEMERAL_FILESYSTEMS = frozenset(
    {"aufs", "devtmpfs", "overlay", "ramfs", "squashfs", "tmpfs"}
)
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_PYTHON_HASH_SEED = "0"
_FP32_BYTES = 4
_MINIMUM_1P3B_DDP_DEVICE_BYTES = 32 * 1024**3
_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 30 * 60
_STOP_REQUEST_ENVIRONMENT_VARIABLE = "PRETRAIN_STOP_REQUEST_FILE"
_INTERNAL_SUPERVISOR_FLAG = "--internal-supervise-torchrun"
_PROTECTED_TRAINER_OPTIONS = frozenset(
    {
        "--checkpoint",
        "--checkpoint-every",
        "--device",
        "--deterministic",
        "--eval-at-start",
        "--eval-batches",
        "--eval-every",
        "--global-microbatch-rows",
        "--graceful-shutdown-timeout-seconds",
        "--gradient-accumulation-steps",
        "--model-size",
        "--no-activation-checkpointing",
        "--no-eval-at-start",
        "--order-manifest",
        "--precision",
        "--resume",
        "--run-authority",
        "--steps",
        "--tokenizer",
        "--validation-order-manifest",
        "--verify-packed-payloads",
        "--wandb-entity",
        "--wandb-group",
        "--wandb-mode",
        "--wandb-project",
        "--wandb-run-id",
        "--wandb-run-name",
        "--wandb-tags",
        "--workers",
    }
)


class PreflightError(RuntimeError):
    """A launch invariant was not proven."""


@dataclass(frozen=True)
class MountEvidence:
    path: str
    device: int
    mount_point: str | None
    filesystem_type: str
    mount_source: str | None
    read_only: bool
    mount_read_only: bool
    classification: str


@dataclass(frozen=True)
class OrderInspection:
    path: str
    split: str
    format_version: int
    sequence_length: int
    vocab_size: int
    eos_token_id: int
    tokenizer_manifest_sha256: str
    geometry: dict[str, Any]
    order_payload_bytes: int
    packed_manifest_paths: dict[str, str]
    packed_payload_files: int
    packed_payload_bytes: int
    payload_bytes_read: int = 0
    metadata_inventory: dict[str, Any] | None = None
    metadata_inventory_sha256: str = ""


@dataclass(frozen=True)
class CheckpointInspection:
    checkpoint: str
    previous: str
    resume_generation: ResumeGeneration
    resume_path: str | None
    latest_exists: bool
    latest_bytes: int | None
    previous_exists: bool
    previous_bytes: int | None
    free_bytes: int
    required_free_bytes: int
    filesystem_capabilities: dict[str, bool]
    resume_read_storage: str
    local_resume_staging: bool
    guidance: str


@dataclass(frozen=True)
class RuntimeInspection:
    python_executable: str
    python_executable_sha256: str
    python_version: str
    python_implementation: str
    torch_version: str
    cuda_runtime: str
    cuda_devices: int
    world_size: int
    bf16_supported_devices: list[int]
    cuda_device_profiles: list[dict[str, Any]]
    launcher_module: str
    deterministic_algorithms: bool
    cublas_workspace_config: str
    python_hash_seed: str


@dataclass(frozen=True)
class ModelMemoryInspection:
    model_size: str
    parameter_count: int
    fp32_parameter_bytes: int
    fp32_gradient_bytes: int
    fp32_adam_moment_bytes: int
    persistent_training_state_bytes: int
    minimum_device_memory_bytes: int
    smallest_visible_device_memory_bytes: int
    smallest_available_device_memory_bytes: int
    admission_headroom_bytes: int
    measured_full_topology_smoke_required: bool
    basis: str


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _interpreter_identity() -> dict[str, str]:
    # Preserve the invocation path: resolving ``venv/bin/python`` to its base
    # interpreter bypasses the virtual environment on the subsequent exec and
    # can silently select a different PyTorch/CUDA installation. ``stat`` and
    # the content hash still follow the symlink to authenticate the real binary.
    executable = Path(os.path.abspath(sys.executable))
    try:
        resolved_executable = executable.resolve(strict=True)
        metadata = resolved_executable.stat()
    except OSError as exc:
        raise PreflightError(
            f"Current Python executable cannot be resolved: {executable}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PreflightError(
            "Current Python executable does not resolve to a regular file: "
            f"{executable} -> {resolved_executable}"
        )
    implementation = getattr(sys.implementation, "name", "unknown")
    return {
        "python_executable": str(executable),
        "python_executable_sha256": _sha256(executable),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "python_implementation": str(implementation),
    }


def _is_plain_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _resolve_existing_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PreflightError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a directory: {resolved}")
    return resolved


def _require_inside(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PreflightError(f"{label} does not exist: {path}") from exc
    if resolved != root and not resolved.is_relative_to(root):
        raise PreflightError(f"{label} escapes its authorized root {root}: {resolved}")
    return resolved


def _require_regular_file(path: Path, *, label: str, expected_device: int) -> Path:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise PreflightError(f"{label} is not a regular file: {resolved}")
    if metadata.st_dev != expected_device:
        raise PreflightError(
            f"{label} is on device {metadata.st_dev}, not authorized device "
            f"{expected_device}: {resolved}"
        )
    return resolved


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _linux_mount_evidence(path: Path, text: str) -> tuple[str, str, str, bool] | None:
    selected: tuple[int, str, str, str, bool] | None = None
    for line in text.splitlines():
        try:
            left, right = line.split(" - ", 1)
            left_fields = left.split()
            right_fields = right.split()
            mount_point = Path(_unescape_mount_field(left_fields[4])).resolve(
                strict=False
            )
            if path != mount_point and not path.is_relative_to(mount_point):
                continue
            options = set(left_fields[5].split(","))
            candidate = (
                len(mount_point.parts),
                str(mount_point),
                right_fields[0].casefold(),
                _unescape_mount_field(right_fields[1]),
                "ro" in options,
            )
            if selected is None or candidate[0] > selected[0]:
                selected = candidate
        except (IndexError, OSError, ValueError):
            continue
    if selected is None:
        return None
    _, mount_point, filesystem_type, source, read_only = selected
    return mount_point, filesystem_type, source, read_only


def inspect_mount(path: Path, *, mountinfo_text: str | None = None) -> MountEvidence:
    resolved = _resolve_existing_directory(path, label="storage root")
    filesystem_type = "unknown"
    mount_point: str | None = None
    mount_source: str | None = None
    permission_read_only = not os.access(resolved, os.W_OK)
    mount_read_only = False
    if sys.platform.startswith("linux"):
        if mountinfo_text is None:
            try:
                mountinfo_text = Path("/proc/self/mountinfo").read_text(
                    encoding="utf-8", errors="strict"
                )
            except OSError:
                mountinfo_text = ""
        detected = _linux_mount_evidence(resolved, mountinfo_text)
        if detected is not None:
            mount_point, filesystem_type, mount_source, mount_read_only = detected
            mount_read_only = bool(mount_read_only)
    if filesystem_type in _NETWORK_FILESYSTEMS or filesystem_type.startswith("fuse."):
        classification = "network"
    elif filesystem_type in _EPHEMERAL_FILESYSTEMS:
        classification = "ephemeral"
    elif filesystem_type == "unknown":
        classification = "unknown"
    else:
        classification = "local-or-block"
    return MountEvidence(
        path=str(resolved),
        device=resolved.stat().st_dev,
        mount_point=mount_point,
        filesystem_type=filesystem_type,
        mount_source=mount_source,
        read_only=permission_read_only or mount_read_only,
        mount_read_only=mount_read_only,
        classification=classification,
    )


def _inspect_packed_manifests(
    *,
    order_path: Path,
    order: Mapping[str, Any],
    local_root: Path,
    expected_device: int,
) -> tuple[dict[str, str], int, int, list[dict[str, Any]]]:
    packed_paths: dict[str, str] = {}
    payload_files = 0
    payload_bytes = 0
    inventory: list[dict[str, Any]] = []
    identities = order["dataset_manifests"]
    for domain in training_data.DOMAIN_ORDER:
        identity = identities[domain]
        raw_relative = identity["path"]
        if Path(raw_relative).is_absolute():
            raise PreflightError(
                f"Packed manifest path must be relative for {domain}: {raw_relative!r}"
            )
        candidate = order_path.parent / raw_relative
        packed_path = _require_inside(
            candidate, local_root, label=f"{domain} packed manifest"
        )
        _require_regular_file(
            packed_path,
            label=f"{domain} packed manifest",
            expected_device=expected_device,
        )
        expected_manifest_hash = identity.get("sha256")
        if not isinstance(expected_manifest_hash, str) or not _LOWERCASE_SHA256.fullmatch(
            expected_manifest_hash
        ):
            raise PreflightError(f"Invalid packed-manifest SHA-256 for {domain}")
        actual_manifest_hash = _sha256(packed_path)
        if actual_manifest_hash != expected_manifest_hash:
            raise PreflightError(
                f"Packed manifest changed after order creation: {packed_path}"
            )
        inventory.append(
            _inventory_record(
                packed_path,
                kind="packed-manifest",
                domain=domain,
                declared_sha256=expected_manifest_hash,
                actual_sha256=actual_manifest_hash,
            )
        )
        try:
            packed, shards = training_data._parse_packed_manifest(packed_path)
        except (FileNotFoundError, IOError, TypeError, ValueError) as exc:
            raise PreflightError(f"Invalid packed manifest {packed_path}: {exc}") from exc
        expected_fields = {
            "domain": domain,
            "rows": order["packed_available_rows_per_domain"][domain],
            "input_tokens": order["packed_available_input_tokens_per_domain"][domain],
            "valid_loss_tokens": order[
                "packed_available_supervised_tokens_per_domain"
            ][domain],
            "split": order["split"],
            "sequence_length": order["sequence_length"],
            "vocab_size": order["vocab_size"],
            "eos_token_id": order["eos_token_id"],
            "tokenizer_manifest_sha256": order["tokenizer_manifest_sha256"],
        }
        for field, expected in expected_fields.items():
            if packed.get(field) != expected:
                raise PreflightError(
                    f"{domain} packed manifest {field}={packed.get(field)!r} differs "
                    f"from order value {expected!r}"
                )
        for shard, recorded in zip(shards, packed["shards"], strict=True):
            token_path = _require_inside(
                shard.tokens_path,
                local_root,
                label=f"{domain} token shard {shard.index}",
            )
            starts_path = _require_inside(
                shard.starts_path,
                local_root,
                label=f"{domain} starts shard {shard.index}",
            )
            token_path = _require_regular_file(
                token_path,
                label=f"{domain} token shard {shard.index}",
                expected_device=expected_device,
            )
            starts_path = _require_regular_file(
                starts_path,
                label=f"{domain} starts shard {shard.index}",
                expected_device=expected_device,
            )
            expected_token_bytes = shard.rows * packed["tokens_per_row"] * 2
            expected_starts_bytes = shard.rows * packed["starts_bytes_per_row"]
            found_token_bytes = token_path.stat().st_size
            found_starts_bytes = starts_path.stat().st_size
            if (
                recorded["tokens"].get("bytes") != expected_token_bytes
                or found_token_bytes != expected_token_bytes
            ):
                raise PreflightError(
                    f"Token shard size mismatch for {token_path}: expected "
                    f"{expected_token_bytes}, recorded {recorded['tokens'].get('bytes')!r}, "
                    f"found {found_token_bytes}"
                )
            if (
                recorded["starts"].get("bytes") != expected_starts_bytes
                or found_starts_bytes != expected_starts_bytes
            ):
                raise PreflightError(
                    f"Starts shard size mismatch for {starts_path}: expected "
                    f"{expected_starts_bytes}, recorded "
                    f"{recorded['starts'].get('bytes')!r}, found {found_starts_bytes}"
                )
            payload_files += 2
            payload_bytes += found_token_bytes + found_starts_bytes
            inventory.extend(
                (
                    _inventory_record(
                        token_path,
                        kind="packed-tokens",
                        domain=domain,
                        shard_index=shard.index,
                        declared_sha256=shard.tokens_sha256,
                    ),
                    _inventory_record(
                        starts_path,
                        kind="packed-starts",
                        domain=domain,
                        shard_index=shard.index,
                        declared_sha256=shard.starts_sha256,
                    ),
                )
            )
        packed_paths[domain] = str(packed_path)
    return packed_paths, payload_files, payload_bytes, inventory


def _inventory_record(
    path: Path,
    *,
    kind: str,
    declared_sha256: str,
    actual_sha256: str | None = None,
    domain: str | None = None,
    shard_index: int | None = None,
) -> dict[str, Any]:
    metadata = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve(strict=True)),
        "kind": kind,
        "bytes": metadata.st_size,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "declared_sha256": declared_sha256,
    }
    if actual_sha256 is not None:
        record["actual_sha256"] = actual_sha256
    if domain is not None:
        record["domain"] = domain
    if shard_index is not None:
        record["shard_index"] = shard_index
    return record


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_order(
    path: Path,
    *,
    expected_split: str,
    local_data_root: Path,
    global_microbatch_rows: int | None = None,
) -> OrderInspection:
    local_root = _resolve_existing_directory(local_data_root, label="local data root")
    expected_device = local_root.stat().st_dev
    order_manifest_path = _require_inside(
        path, local_root, label=f"{expected_split} order manifest"
    )
    _require_regular_file(
        order_manifest_path,
        label=f"{expected_split} order manifest",
        expected_device=expected_device,
    )
    try:
        order, order_payload_path = training_data._load_order_manifest(
            order_manifest_path, verify_checksum=False
        )
        if expected_split == "train":
            if global_microbatch_rows is not None:
                raise PreflightError(
                    "Training order inspection must use its own frozen geometry"
                )
            geometry = training_data.frozen_training_geometry(
                order_manifest_path, verify_checksum=False
            )
        else:
            if global_microbatch_rows is None:
                raise PreflightError(
                    "Held-out order inspection requires the training global microbatch rows"
                )
            geometry = training_data.evaluation_order_geometry(
                order_manifest_path,
                global_microbatch_rows=global_microbatch_rows,
                verify_checksum=False,
            )
    except (FileNotFoundError, IOError, KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"Invalid {expected_split} order: {exc}") from exc
    if order.get("split") != expected_split:
        raise PreflightError(
            f"Expected {expected_split!r} order, found split {order.get('split')!r}: "
            f"{order_manifest_path}"
        )
    order_payload_path = _require_inside(
        order_payload_path, local_root, label=f"{expected_split} order payload"
    )
    _require_regular_file(
        order_payload_path,
        label=f"{expected_split} order payload",
        expected_device=expected_device,
    )
    packed_paths, payload_files, payload_bytes, packed_inventory = (
        _inspect_packed_manifests(
            order_path=order_manifest_path,
            order=order,
            local_root=local_root,
            expected_device=expected_device,
        )
    )
    order_manifest_sha256 = _sha256(order_manifest_path)
    inventory = {
        "order_manifest": _inventory_record(
            order_manifest_path,
            kind="order-manifest",
            declared_sha256=order_manifest_sha256,
            actual_sha256=order_manifest_sha256,
        ),
        "order_payload": _inventory_record(
            order_payload_path,
            kind="order-payload",
            declared_sha256=order["order"]["sha256"],
        ),
        "packed_files": packed_inventory,
    }
    return OrderInspection(
        path=str(order_manifest_path),
        split=expected_split,
        format_version=int(order["format_version"]),
        sequence_length=int(order["sequence_length"]),
        vocab_size=int(order["vocab_size"]),
        eos_token_id=int(order["eos_token_id"]),
        tokenizer_manifest_sha256=str(order["tokenizer_manifest_sha256"]),
        geometry=geometry,
        order_payload_bytes=order_payload_path.stat().st_size,
        packed_manifest_paths=packed_paths,
        packed_payload_files=payload_files,
        packed_payload_bytes=payload_bytes,
        metadata_inventory=inventory,
        metadata_inventory_sha256=_canonical_sha256(inventory),
    )


def validate_order_pair(
    train: OrderInspection,
    validation: OrderInspection,
    *,
    world_size: int,
    eval_batches: int,
) -> None:
    if Path(train.path) == Path(validation.path):
        raise PreflightError("Training and validation orders must be distinct")
    for field in (
        "sequence_length",
        "vocab_size",
        "eos_token_id",
        "tokenizer_manifest_sha256",
    ):
        if getattr(train, field) != getattr(validation, field):
            raise PreflightError(
                f"Validation {field}={getattr(validation, field)!r} differs from "
                f"training value {getattr(train, field)!r}"
            )
    if validation.geometry["global_microbatch_rows"] != train.geometry[
        "global_microbatch_rows"
    ]:
        raise PreflightError(
            "Validation evaluation geometry does not use the training global "
            "microbatch rows"
        )
    if not _is_plain_int(world_size, minimum=1):
        raise PreflightError("World size must be a positive integer")
    global_rows = train.geometry["global_microbatch_rows"]
    if global_rows < world_size or global_rows % world_size:
        raise PreflightError(
            f"Frozen global_microbatch_rows={global_rows} is not divisible by "
            f"world_size={world_size}"
        )
    if not _is_plain_int(eval_batches, minimum=1):
        raise PreflightError("Evaluation batches must be a positive integer")
    available = validation.geometry["available_global_microbatches"]
    if eval_batches > available:
        raise PreflightError(
            f"eval_batches={eval_batches} exceeds validation order capacity {available}"
        )


def _previous_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.previous{path.suffix}")


def _existing_checkpoint_size(path: Path, *, label: str, device: int) -> int | None:
    if path.is_symlink():
        raise PreflightError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise PreflightError(f"{label} must be a regular non-symlink file: {path}")
    metadata = path.stat()
    if metadata.st_dev != device:
        raise PreflightError(f"{label} is not on the durable checkpoint device: {path}")
    if metadata.st_size < 1:
        raise PreflightError(f"{label} is empty: {path}")
    return metadata.st_size


def _assert_checkpoint_lease_available(checkpoint: Path) -> None:
    lease = checkpoint.with_name(f".{checkpoint.name}.lock")
    if not lease.exists():
        return
    if lease.is_symlink() or not lease.is_file():
        raise PreflightError(f"Unsafe checkpoint lease path: {lease}")
    with lease.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PreflightError(f"Another trainer holds checkpoint lease {lease}") from exc
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _probe_checkpoint_filesystem(path: Path) -> dict[str, bool]:
    """Exercise every filesystem primitive used by checkpoint publication."""

    primary_fd = -1
    contender_fd = -1
    replacement_fd = -1
    primary_name = ""
    link_name = ""
    replacement_name = ""
    try:
        primary_fd, primary_name = tempfile.mkstemp(
            prefix=".pretrain-preflight-primary.", dir=path
        )
        os.write(primary_fd, b"preflight-primary\n")
        os.fsync(primary_fd)

        fcntl.flock(primary_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contender_fd = os.open(primary_name, os.O_RDWR)
        try:
            fcntl.flock(contender_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise PreflightError(
                f"Filesystem does not enforce checkpoint lease contention: {path}"
            )
        finally:
            fcntl.flock(primary_fd, fcntl.LOCK_UN)

        link_name = f"{primary_name}.link"
        os.link(primary_name, link_name)
        if os.stat(primary_name).st_ino != os.stat(link_name).st_ino:
            raise PreflightError(f"Filesystem hard-link identity check failed: {path}")

        replacement_fd, replacement_name = tempfile.mkstemp(
            prefix=".pretrain-preflight-replacement.", dir=path
        )
        os.write(replacement_fd, b"preflight-replacement\n")
        os.fsync(replacement_fd)
        os.close(replacement_fd)
        replacement_fd = -1
        os.replace(replacement_name, link_name)
        replacement_name = ""
        if Path(link_name).read_bytes() != b"preflight-replacement\n":
            raise PreflightError(f"Filesystem atomic-replace probe failed: {path}")

        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise PreflightError(
            f"Checkpoint filesystem lacks required flock/hard-link/atomic-replace/fsync "
            f"semantics at {path}: {exc}"
        ) from exc
    finally:
        for descriptor in (contender_fd, replacement_fd, primary_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        for temporary in (replacement_name, link_name, primary_name):
            if not temporary:
                continue
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return {
        "file_fsync": True,
        "nonblocking_flock_contention": True,
        "hard_link": True,
        "atomic_replace": True,
        "directory_fsync": True,
    }


def inspect_storage_and_checkpoint(
    *,
    local_data_root: Path,
    durable_checkpoint_root: Path,
    checkpoint: Path,
    resume_generation: ResumeGeneration,
    checkpoint_generation_bytes: int,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    mount_inspector: Callable[[Path], MountEvidence] = inspect_mount,
) -> tuple[MountEvidence, MountEvidence, CheckpointInspection]:
    local_root = _resolve_existing_directory(local_data_root, label="local data root")
    durable_root = _resolve_existing_directory(
        durable_checkpoint_root, label="durable checkpoint root"
    )
    local_mount = mount_inspector(local_root)
    durable_mount = mount_inspector(durable_root)
    if local_mount.classification != "local-or-block":
        raise PreflightError(
            "Local packed data root must be on a recognized local/block filesystem: "
            f"{local_root} ({local_mount.filesystem_type}, "
            f"classification={local_mount.classification})"
        )
    if not local_mount.mount_read_only:
        raise PreflightError(
            "Local packed data root must be on an explicitly read-only mount; "
            "use a read-only bind mount or immutable read-only snapshot so payloads "
            "cannot change between certification, preflight, and DataLoader mmap: "
            f"{local_root}"
        )
    if durable_mount.read_only:
        raise PreflightError(f"Durable checkpoint root is read-only: {durable_root}")
    if durable_mount.classification == "ephemeral":
        raise PreflightError(
            f"Durable checkpoint root is on an ephemeral filesystem: {durable_root} "
            f"({durable_mount.filesystem_type})"
        )
    if local_mount.device == durable_mount.device:
        raise PreflightError(
            "Local packed data and durable checkpoints are on the same filesystem "
            f"device ({local_mount.device}); use separate local-NVMe and durable roots"
        )
    if not _is_plain_int(checkpoint_generation_bytes, minimum=1):
        raise PreflightError("checkpoint_generation_bytes must be a positive integer")
    if checkpoint.suffix != ".pt":
        raise PreflightError("Checkpoint path must end in .pt")
    checkpoint_parent = _resolve_existing_directory(
        checkpoint.parent, label="checkpoint directory"
    )
    checkpoint_path = checkpoint_parent / checkpoint.name
    resolved_parent = checkpoint_parent.resolve(strict=True)
    if resolved_parent != durable_root and not resolved_parent.is_relative_to(durable_root):
        raise PreflightError(
            f"Checkpoint path is outside durable root {durable_root}: {checkpoint_path}"
        )
    if checkpoint_parent.stat().st_dev != durable_mount.device:
        raise PreflightError(
            f"Checkpoint directory is on device {checkpoint_parent.stat().st_dev}, not "
            f"durable-root device {durable_mount.device}"
        )
    filesystem_capabilities = _probe_checkpoint_filesystem(checkpoint_parent)
    _assert_checkpoint_lease_available(checkpoint_path)
    previous = _previous_checkpoint(checkpoint_path)
    latest_bytes = _existing_checkpoint_size(
        checkpoint_path, label="latest checkpoint", device=durable_mount.device
    )
    previous_bytes = _existing_checkpoint_size(
        previous, label="previous checkpoint", device=durable_mount.device
    )
    if resume_generation == "none":
        if latest_bytes is not None or previous_bytes is not None:
            raise PreflightError(
                "Checkpoint lineage already exists; use --resume-generation latest, "
                "use --resume-generation previous after a failed latest load, or choose "
                "a new checkpoint path"
            )
        resume_path: Path | None = None
        guidance = "new lineage; neither latest nor previous exists"
    elif resume_generation == "latest":
        if latest_bytes is None:
            suffix = (
                "; previous exists, so select --resume-generation previous explicitly"
                if previous_bytes is not None
                else ""
            )
            raise PreflightError(f"Latest checkpoint does not exist: {checkpoint_path}{suffix}")
        resume_path = checkpoint_path
        guidance = (
            "resume latest first; if trainer checkpoint loading fails, rerun this "
            "preflight with --resume-generation previous"
        )
    elif resume_generation == "previous":
        if previous_bytes is None:
            raise PreflightError(f"Previous checkpoint does not exist: {previous}")
        resume_path = previous
        guidance = (
            "explicit rollback: load previous and publish the next successful save as "
            "canonical latest"
        )
    else:
        raise PreflightError(f"Unsupported resume generation: {resume_generation!r}")
    existing_sizes = [size for size in (latest_bytes, previous_bytes) if size is not None]
    if existing_sizes and max(existing_sizes) > checkpoint_generation_bytes:
        raise PreflightError(
            f"Checkpoint generation estimate {checkpoint_generation_bytes} bytes is lower "
            f"than existing generation size {max(existing_sizes)} bytes"
        )
    required_free = 2 * checkpoint_generation_bytes
    free_bytes = int(disk_usage(checkpoint_parent).free)
    if free_bytes < required_free:
        raise PreflightError(
            f"Insufficient durable checkpoint free space: found {free_bytes} bytes, "
            f"require at least {required_free} bytes for two generations"
        )
    inspection = CheckpointInspection(
        checkpoint=str(checkpoint_path),
        previous=str(previous),
        resume_generation=resume_generation,
        resume_path=None if resume_path is None else str(resume_path),
        latest_exists=latest_bytes is not None,
        latest_bytes=latest_bytes,
        previous_exists=previous_bytes is not None,
        previous_bytes=previous_bytes,
        free_bytes=free_bytes,
        required_free_bytes=required_free,
        filesystem_capabilities=filesystem_capabilities,
        resume_read_storage="durable-checkpoint-filesystem",
        local_resume_staging=False,
        guidance=guidance,
    )
    return local_mount, durable_mount, inspection


def inspect_runtime(
    *,
    nproc_per_node: int,
    torch_module: ModuleType | Any,
    environment: Mapping[str, str] = os.environ,
) -> RuntimeInspection:
    if not _is_plain_int(nproc_per_node, minimum=1):
        raise PreflightError("nproc_per_node must be a positive integer")
    configured_workspace = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in (None, _CUBLAS_WORKSPACE_CONFIG):
        raise PreflightError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with deterministic production value "
            f"{_CUBLAS_WORKSPACE_CONFIG!r}: found {configured_workspace!r}"
        )
    configured_hash_seed = environment.get("PYTHONHASHSEED")
    if configured_hash_seed not in (None, _PYTHON_HASH_SEED):
        raise PreflightError(
            "PYTHONHASHSEED conflicts with deterministic production value "
            f"{_PYTHON_HASH_SEED!r}: found {configured_hash_seed!r}"
        )
    nested = sorted(name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if name in environment)
    if nested:
        raise PreflightError(
            "Refusing to start a nested torchrun; unset launcher variables: "
            + ", ".join(nested)
        )
    interpreter = _interpreter_identity()
    torch_version = str(torch_module.__version__)
    version_match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if version_match is None or (
        int(version_match.group(1)), int(version_match.group(2))
    ) < (2, 6):
        raise PreflightError(
            f"PyTorch {torch_version!r} is unsupported; requirements-train.txt "
            "requires torch>=2.6"
        )
    if getattr(torch_module.version, "cuda", None) is None:
        raise PreflightError("Installed PyTorch is not a CUDA build")
    if not torch_module.cuda.is_available():
        raise PreflightError("CUDA is not available")
    devices = int(torch_module.cuda.device_count())
    if devices != nproc_per_node:
        raise PreflightError(
            f"CUDA_VISIBLE_DEVICES exposes {devices} GPUs but nproc_per_node="
            f"{nproc_per_node}; production launch requires an exact match"
        )
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or not distributed.is_available():
        raise PreflightError("torch.distributed is unavailable")
    if not distributed.is_nccl_available():
        raise PreflightError("NCCL is unavailable in this PyTorch build")
    supported: list[int] = []
    device_profiles: list[dict[str, Any]] = []
    for index in range(devices):
        with torch_module.cuda.device(index):
            if not torch_module.cuda.is_bf16_supported(
                including_emulation=False
            ):
                name = torch_module.cuda.get_device_name(index)
                raise PreflightError(f"CUDA device {index} ({name}) does not support BF16")
        try:
            name = str(torch_module.cuda.get_device_name(index))
            capability = [
                int(value)
                for value in torch_module.cuda.get_device_capability(index)
            ]
            properties = torch_module.cuda.get_device_properties(index)
            total_memory = int(properties.total_memory)
            multiprocessors = int(properties.multi_processor_count)
            with torch_module.cuda.device(index):
                free_memory, allocator_total = torch_module.cuda.mem_get_info()
            free_memory = int(free_memory)
            allocator_total = int(allocator_total)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise PreflightError(
                f"Cannot inspect CUDA device {index} topology and memory: {exc}"
            ) from exc
        if len(capability) != 2 or any(value < 0 for value in capability):
            raise PreflightError(
                f"CUDA device {index} reported an invalid compute capability: "
                f"{capability!r}"
            )
        if (
            total_memory < 1
            or free_memory < 1
            or allocator_total < 1
            or free_memory > allocator_total
            or multiprocessors < 1
            or not name
        ):
            raise PreflightError(
                f"CUDA device {index} reported invalid hardware properties"
            )
        device_profiles.append(
            {
                "index": index,
                "name": name,
                "compute_capability": capability,
                "total_memory_bytes": total_memory,
                "available_memory_bytes": free_memory,
                "allocator_total_memory_bytes": allocator_total,
                "multiprocessor_count": multiprocessors,
            }
        )
        supported.append(index)
    hardware_identities = {
        (
            profile["name"],
            tuple(profile["compute_capability"]),
            profile["total_memory_bytes"],
            profile["allocator_total_memory_bytes"],
            profile["multiprocessor_count"],
        )
        for profile in device_profiles
    }
    if len(hardware_identities) != 1:
        raise PreflightError(
            "Visible CUDA devices are heterogeneous; production DDP requires the "
            "same GPU name, compute capability, physical/allocator-visible VRAM, "
            f"and SM count: {device_profiles}"
        )
    return RuntimeInspection(
        python_executable=interpreter["python_executable"],
        python_executable_sha256=interpreter["python_executable_sha256"],
        python_version=interpreter["python_version"],
        python_implementation=interpreter["python_implementation"],
        torch_version=torch_version,
        cuda_runtime=str(torch_module.version.cuda),
        cuda_devices=devices,
        world_size=nproc_per_node,
        bf16_supported_devices=supported,
        cuda_device_profiles=device_profiles,
        launcher_module="torch.distributed.run",
        deterministic_algorithms=True,
        cublas_workspace_config=_CUBLAS_WORKSPACE_CONFIG,
        python_hash_seed=_PYTHON_HASH_SEED,
    )


def inspect_model_memory(
    runtime: RuntimeInspection,
    *,
    model_size: str,
    vocab_size: int,
    max_seq_len: int,
) -> ModelMemoryInspection:
    """Reject a topology that cannot hold the replicated FP32 AdamW state.

    This is deliberately only an admission floor.  Activations, FlexAttention,
    CUDA graphs/compiler workspaces, the first DDP bucket construction, and the
    fused optimizer's peak temporaries depend on the frozen batch geometry and
    selected PyTorch/CUDA build.  A measured full-topology smoke remains a
    separate launch gate.
    """

    if model_size not in ("tiny", "1.3b"):
        raise PreflightError(f"Unsupported model size for memory inspection: {model_size}")
    if not _is_plain_int(vocab_size, minimum=1) or not _is_plain_int(
        max_seq_len, minimum=1
    ):
        raise PreflightError("Model vocabulary and sequence length must be positive")
    if not runtime.cuda_device_profiles:
        raise PreflightError("Runtime inspection contains no CUDA device profiles")
    if model_size == "1.3b":
        model_config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            activation_checkpointing=True,
        )
        admission_floor = _MINIMUM_1P3B_DDP_DEVICE_BYTES
    else:
        # Mirrors ``pretrain.train.tiny_model_config`` without importing the
        # training entry point into the fail-fast launcher.
        model_config = ModelConfig(
            vocab_size=vocab_size,
            dim=32,
            hidden_dim=88,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=max_seq_len,
            loss_chunk_size=min(32, max_seq_len),
        )
        admission_floor = 0
    parameter_count = model_config.expected_parameter_count
    parameter_bytes = parameter_count * _FP32_BYTES
    gradient_bytes = parameter_bytes
    adam_moment_bytes = 2 * parameter_bytes
    persistent_bytes = parameter_bytes + gradient_bytes + adam_moment_bytes
    minimum_device_bytes = max(admission_floor, persistent_bytes)
    smallest_device = min(
        int(profile["total_memory_bytes"])
        for profile in runtime.cuda_device_profiles
    )
    smallest_available = min(
        int(profile["available_memory_bytes"])
        for profile in runtime.cuda_device_profiles
    )
    if (
        smallest_device < minimum_device_bytes
        or smallest_available < minimum_device_bytes
    ):
        raise PreflightError(
            f"{model_size} replicated DDP is not admitted on the visible GPUs: "
            f"smallest device has {smallest_device} total and {smallest_available} "
            f"currently available bytes, but the FP32 model, "
            f"gradients, and Adam moments require {persistent_bytes} persistent "
            f"bytes and the production admission floor is {minimum_device_bytes} "
            "bytes before geometry-dependent activation/workspace peaks"
        )
    return ModelMemoryInspection(
        model_size=model_size,
        parameter_count=parameter_count,
        fp32_parameter_bytes=parameter_bytes,
        fp32_gradient_bytes=gradient_bytes,
        fp32_adam_moment_bytes=adam_moment_bytes,
        persistent_training_state_bytes=persistent_bytes,
        minimum_device_memory_bytes=minimum_device_bytes,
        smallest_visible_device_memory_bytes=smallest_device,
        smallest_available_device_memory_bytes=smallest_available,
        admission_headroom_bytes=smallest_available - persistent_bytes,
        measured_full_topology_smoke_required=True,
        basis=(
            "FP32 parameters + FP32 gradients + two FP32 Adam moments; the "
            "1.3B path additionally requires at least 32 GiB physical VRAM per "
            "replica. This is not a substitute for the frozen-geometry smoke."
        ),
    )


def _wandb_credentials_present(environment: Mapping[str, str]) -> bool:
    if environment.get("WANDB_API_KEY", "").strip():
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return credentials is not None and bool(credentials[2])


def inspect_wandb(
    *,
    mode: WandbMode,
    checkpoint_parent: Path,
    durable_root: Path,
    environment: Mapping[str, str] = os.environ,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    package_version: Callable[[str], str] = importlib.metadata.version,
    credentials_present: Callable[[Mapping[str, str]], bool] = _wandb_credentials_present,
) -> dict[str, Any]:
    if mode == "disabled":
        return {
            "mode": mode,
            "dependency_checked": False,
            "dependency_version": None,
            "directory": None,
            "credentials_checked": False,
        }
    if find_spec("wandb") is None:
        raise PreflightError(
            "W&B mode is enabled but wandb is not installed; install "
            "requirements-wandb.txt or use --wandb-mode disabled"
        )
    try:
        version = package_version("wandb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreflightError("W&B package metadata is unavailable") from exc
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None or not (
        int(match.group(1)) == 0 and int(match.group(2)) >= 18
    ):
        raise PreflightError(
            f"Unsupported wandb version {version!r}; requirements-wandb.txt requires "
            "wandb>=0.18,<1"
        )
    directory = checkpoint_parent.resolve(strict=True)
    durable = durable_root.resolve(strict=True)
    if directory != durable and not directory.is_relative_to(durable):
        raise PreflightError(f"W&B directory is outside durable root: {directory}")
    credentials_checked = mode == "online"
    if mode == "online" and not credentials_present(environment):
        raise PreflightError(
            "Online W&B requires WANDB_API_KEY or an api.wandb.ai entry in ~/.netrc"
        )
    return {
        "mode": mode,
        "dependency_checked": True,
        "dependency_version": version,
        "directory": str(directory),
        "credentials_checked": credentials_checked,
    }


def _verified_sha256_sidecar(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise PreflightError(f"Missing regular full-validation evidence sidecar: {sidecar}")
    expected = f"{_sha256(path)}  {path.name}\n"
    try:
        found = sidecar.read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise PreflightError(f"Cannot read evidence sidecar {sidecar}: {exc}") from exc
    if found != expected:
        raise PreflightError(
            f"Full-validation evidence sidecar is not exact or checksum mismatches: {sidecar}"
        )
    return expected.split(" ", 1)[0]


def _current_validator_identity() -> dict[str, Any]:
    import numpy as np

    certifier = PROJECT_ROOT / "scripts" / "certify_pretraining_data.py"
    data_source = Path(training_data.__file__).resolve(strict=True)
    launcher_source = Path(__file__).resolve(strict=True)
    return {
        "evidence_contract_version": 1,
        **_interpreter_identity(),
        "numpy_version": np.__version__,
        "order_format_version": training_data.ORDER_FORMAT_VERSION,
        "packed_format_version": training_data.FORMAT_VERSION,
        "pretrain_data_path": str(data_source),
        "pretrain_data_sha256": _sha256(data_source),
        "launcher_path": str(launcher_source),
        "launcher_sha256": _sha256(launcher_source),
        "certifier_path": str(certifier.resolve(strict=True)),
        "certifier_sha256": _sha256(certifier),
    }


def verify_full_validation_evidence(
    evidence_path: Path,
    *,
    expected_order: OrderInspection,
    durable_root: Path,
) -> dict[str, Any]:
    durable = _resolve_existing_directory(durable_root, label="durable checkpoint root")
    evidence = _require_inside(
        evidence_path, durable, label=f"{expected_order.split} full-validation evidence"
    )
    if evidence.is_symlink() or not evidence.is_file():
        raise PreflightError(f"Full-validation evidence must be a regular file: {evidence}")
    receipt_sha256 = _verified_sha256_sidecar(evidence)
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Invalid full-validation evidence {evidence}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Full-validation evidence root must be an object: {evidence}")
    if (
        payload.get("format") != "production-pretraining-full-data-validation"
        or payload.get("format_version") != 1
        or payload.get("status") != "pass"
    ):
        raise PreflightError(f"Unsupported or non-passing validation evidence: {evidence}")
    if payload.get("split") != expected_order.split:
        raise PreflightError(
            f"Validation evidence split {payload.get('split')!r} does not match "
            f"{expected_order.split!r}"
        )
    if payload.get("order_manifest") != expected_order.path:
        raise PreflightError(
            f"Validation evidence is bound to another order path: "
            f"{payload.get('order_manifest')!r}"
        )
    expected_inventory = expected_order.metadata_inventory
    if expected_inventory is None:
        raise PreflightError("Order inspection did not produce a metadata inventory")
    if payload.get("metadata_inventory") != expected_inventory:
        raise PreflightError(
            f"Local {expected_order.split} order or packed payload identity changed "
            "after full validation"
        )
    if (
        payload.get("metadata_inventory_sha256")
        != expected_order.metadata_inventory_sha256
        or _canonical_sha256(payload["metadata_inventory"])
        != expected_order.metadata_inventory_sha256
    ):
        raise PreflightError("Full-validation metadata inventory digest mismatches")
    expected_checks = {
        "order_payload_checksum": True,
        "order_reference_semantics": True,
        "packed_manifest_identities": True,
        "packed_payload_checksums": True,
        "packed_payload_semantics": True,
        "stable_file_identity_during_validation": True,
    }
    if payload.get("checks") != expected_checks:
        raise PreflightError("Full-validation evidence does not prove every required check")
    if payload.get("validator") != _current_validator_identity():
        raise PreflightError(
            "Full-validation evidence was produced by a different validator implementation "
            "or runtime; recertify the final local copy"
        )
    completed_raw = payload.get("completed_utc")
    if not isinstance(completed_raw, str):
        raise PreflightError("Full-validation evidence lacks a completion timestamp")
    try:
        completed = datetime.fromisoformat(completed_raw)
    except ValueError as exc:
        raise PreflightError("Full-validation completion timestamp is invalid") from exc
    if completed.tzinfo is None:
        raise PreflightError("Full-validation completion timestamp must include a timezone")
    if completed > datetime.now(timezone.utc).astimezone(completed.tzinfo):
        raise PreflightError("Full-validation completion timestamp is in the future")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise PreflightError("Full-validation evidence summary is missing")
    if (
        summary.get("packed_payload_files") != expected_order.packed_payload_files
        or summary.get("packed_payload_bytes") != expected_order.packed_payload_bytes
    ):
        raise PreflightError("Full-validation evidence payload totals mismatch")
    return {
        "status": "pass",
        "path": str(evidence),
        "receipt_sha256": receipt_sha256,
        "completed_utc": completed_raw,
        "metadata_inventory_sha256": expected_order.metadata_inventory_sha256,
        "validator": payload["validator"],
    }


def require_full_validation_for_mode(
    *,
    execute: bool,
    train_evidence: Mapping[str, Any],
    validation_evidence: Mapping[str, Any],
) -> bool:
    ready = (
        train_evidence.get("status") == "pass"
        and validation_evidence.get("status") == "pass"
    )
    if execute and not ready:
        raise PreflightError(
            "--execute requires passing --train-data-evidence and "
            "--validation-data-evidence bound to the exact local files"
        )
    return ready


def validate_extra_trainer_args(values: Sequence[str]) -> list[str]:
    result = list(values)
    if result and result[0] == "--":
        result = result[1:]
    for token in result:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        protected_matches = sorted(
            candidate
            for candidate in _PROTECTED_TRAINER_OPTIONS
            if candidate == option or candidate.startswith(option)
        )
        if protected_matches:
            raise PreflightError(
                f"Trainer option or abbreviation {option} is owned by the launcher "
                f"({', '.join(protected_matches)}) and cannot be supplied after --"
            )
    return result


def validate_production_launch_selection(
    *,
    model_size: str,
    nproc_per_node: int,
    activation_checkpointing: bool | None,
) -> bool:
    """Resolve and enforce the production strategy before CUDA inspection.

    The accepted 1.3B experiment authority is specifically a six-replica DDP
    trajectory.  Allowing a different world size would change optimizer and
    checkpoint identity; allowing checkpointing to be implicit would leave a
    memory-critical recipe decision outside the canonical launch argv.
    """

    if model_size not in ("tiny", "1.3b"):
        raise PreflightError(f"Unsupported model size: {model_size!r}")
    if not _is_plain_int(nproc_per_node, minimum=1):
        raise PreflightError("nproc_per_node must be a positive integer")
    if activation_checkpointing is not None and not isinstance(
        activation_checkpointing, bool
    ):
        raise PreflightError("activation_checkpointing must be boolean or omitted")
    if model_size == "1.3b":
        if nproc_per_node != 6:
            raise PreflightError(
                "The production 1.3B trajectory requires exactly six DDP ranks"
            )
        if activation_checkpointing is not True:
            raise PreflightError(
                "The production 1.3B trajectory requires the explicit "
                "--activation-checkpointing flag"
            )
        return True
    return bool(activation_checkpointing)


def validate_launch_authority(
    path: Path,
    *,
    invocation_argv: Sequence[str],
) -> dict[str, Any]:
    """Revalidate an authority and bind it to this exact launcher invocation."""

    if (
        not invocation_argv
        or any(
            not isinstance(value, str) or not value or "\x00" in value
            for value in invocation_argv
        )
    ):
        raise PreflightError("Cannot authenticate an invalid launcher argv")
    try:
        result = validate_run_authority(path)
    except (RunAuthorityError, OSError, ValueError) as exc:
        raise PreflightError(f"Run-authority validation failed: {exc}") from exc
    expected = result.get("launcher_argv_sha256")
    actual = canonical_sha256(list(invocation_argv))
    if not isinstance(expected, str) or _LOWERCASE_SHA256.fullmatch(expected) is None:
        raise PreflightError("Run authority lacks a valid canonical launcher digest")
    if actual != expected:
        raise PreflightError(
            "Current launcher argv differs from the exact command authorized for this run"
        )
    return {**result, "current_launcher_argv_sha256": actual}


def render_torchrun_command(
    *,
    runtime: RuntimeInspection,
    train_order: OrderInspection,
    validation_order: OrderInspection,
    checkpoint: CheckpointInspection,
    tokenizer_path: str,
    model_size: str,
    workers: int,
    checkpoint_every: int,
    eval_every: int,
    eval_batches: int,
    eval_at_start: bool,
    wandb_mode: WandbMode,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_group: str | None,
    wandb_tags: Sequence[str],
    extra_trainer_args: Sequence[str],
    activation_checkpointing: bool,
) -> list[str]:
    command = [
        runtime.python_executable,
        "-m",
        runtime.launcher_module,
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={runtime.world_size}",
        "--max-restarts=0",
        "-m",
        "pretrain.train",
        "--order-manifest",
        train_order.path,
        "--tokenizer",
        tokenizer_path,
        "--validation-order-manifest",
        validation_order.path,
        "--model-size",
        model_size,
        "--device",
        "cuda",
        "--precision",
        "bfloat16",
        "--deterministic",
        "--workers",
        str(workers),
        "--checkpoint",
        checkpoint.checkpoint,
        "--checkpoint-every",
        str(checkpoint_every),
        "--eval-every",
        str(eval_every),
        "--eval-batches",
        str(eval_batches),
        "--wandb-mode",
        wandb_mode,
        "--wandb-project",
        wandb_project,
    ]
    if activation_checkpointing:
        command.append("--activation-checkpointing")
    else:
        command.append("--no-activation-checkpointing")
    if eval_at_start:
        command.append("--eval-at-start")
    if checkpoint.resume_path is not None:
        command.extend(("--resume", checkpoint.resume_path))
    for option, value in (
        ("--wandb-entity", wandb_entity),
        ("--wandb-run-name", wandb_run_name),
        ("--wandb-group", wandb_group),
    ):
        if value is not None:
            command.extend((option, value))
    if wandb_tags:
        command.append("--wandb-tags")
        command.extend(wandb_tags)
    command.extend(validate_extra_trainer_args(extra_trainer_args))
    return command


def _publish_stop_request(path: Path, signum: int) -> None:
    """Atomically publish one supervisor-to-worker stop request."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{int(signum)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _terminate_torchrun(process: subprocess.Popen[Any]) -> int:
    """Give TorchElastic its normal cleanup window, then kill its whole session.

    The child is always a session/process-group leader because it was started
    with ``start_new_session=True``. If TorchElastic itself is wedged, killing
    only that parent would orphan its worker ranks and their checkpoint lease.
    """

    try:
        process.terminate()
    except ProcessLookupError:
        return int(process.wait())
    try:
        return int(process.wait(timeout=35))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def supervise_torchrun(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    graceful_shutdown_timeout_seconds: int,
) -> int:
    """Keep torchrun alive while workers finish a potentially large checkpoint.

    TorchElastic's own signal shutdown waits only about 30 seconds before it
    SIGKILLs workers. A mature replicated 1.3B checkpoint written to network
    storage can legitimately take longer. This supervisor consumes the pod's
    signal, publishes it through a rank-shared local request file, and lets the
    trainer ranks stop collectively at a clean optimizer boundary. Only after
    the explicit grace period does it terminate torchrun itself.
    """

    if not command:
        raise PreflightError("Cannot execute an empty launch command")
    if not _is_plain_int(graceful_shutdown_timeout_seconds, minimum=1):
        raise PreflightError("graceful shutdown timeout must be a positive integer")
    requested_signal = 0
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal requested_signal
        del frame
        if requested_signal == 0:
            requested_signal = int(signum)

    handled = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    if hasattr(signal, "SIGUSR1"):
        handled.append(signal.SIGUSR1)
    with tempfile.TemporaryDirectory(prefix="pretrain-torchrun-supervisor-") as root:
        stop_request = Path(root) / "stop-request"
        child_environment = dict(environment)
        child_environment[_STOP_REQUEST_ENVIRONMENT_VARIABLE] = str(stop_request)
        for signum in handled:
            value = int(signum)
            previous_handlers[value] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        try:
            try:
                # Isolate torchrun from terminal-generated process-group
                # signals. The supervisor alone receives them and grants the
                # workers enough time to checkpoint through the request file.
                process = subprocess.Popen(
                    list(command),
                    env=child_environment,
                    start_new_session=True,
                )
            except OSError as exc:
                raise PreflightError(
                    f"Cannot start validated launcher {command[0]}: {exc}"
                ) from exc
            request_published = False
            shutdown_deadline: float | None = None
            forced_shutdown = False
            while True:
                try:
                    return_code = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if requested_signal and not request_published:
                    try:
                        _publish_stop_request(stop_request, requested_signal)
                    except OSError as exc:
                        _terminate_torchrun(process)
                        raise PreflightError(
                            f"Cannot publish graceful stop request: {exc}"
                        ) from exc
                    request_published = True
                    shutdown_deadline = (
                        time.monotonic() + graceful_shutdown_timeout_seconds
                    )
                if (
                    shutdown_deadline is not None
                    and time.monotonic() >= shutdown_deadline
                ):
                    forced_shutdown = True
                    print(
                        "warning: graceful preemption deadline expired; terminating "
                        "torchrun and allowing its final worker-kill fallback",
                        file=sys.stderr,
                        flush=True,
                    )
                    return_code = _terminate_torchrun(process)
                    break
            if requested_signal:
                if forced_shutdown:
                    print(
                        "warning: graceful-stop checkpoint may not have completed",
                        file=sys.stderr,
                        flush=True,
                    )
                return 128 + requested_signal
            return int(return_code)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def exec_clean_supervisor(
    *,
    python_executable: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    graceful_shutdown_timeout_seconds: int,
) -> None:
    """Exec a fresh, CUDA-context-free copy that supervises torchrun.

    Production preflight necessarily queries every GPU. Keeping that process
    alive as the supervisor would also keep its CUDA contexts resident on all
    training devices. ``exec`` releases those contexts while preserving the
    launcher PID and the exact validated command/environment.
    """

    argv = [
        python_executable,
        str(Path(__file__).resolve()),
        _INTERNAL_SUPERVISOR_FLAG,
        str(graceful_shutdown_timeout_seconds),
        "--",
        *command,
    ]
    try:
        os.execvpe(python_executable, argv, dict(environment))
    except OSError as exc:
        raise PreflightError(
            f"Cannot exec clean pre-training supervisor {python_executable}: {exc}"
        ) from exc
    raise AssertionError("os.execvpe unexpectedly returned")


def _internal_supervisor_main(argv: Sequence[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        raise PreflightError("Invalid internal supervisor invocation")
    try:
        timeout = int(argv[0])
    except ValueError as exc:
        raise PreflightError("Invalid internal supervisor timeout") from exc
    return supervise_torchrun(
        argv[2:],
        os.environ,
        graceful_shutdown_timeout_seconds=timeout,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PreflightError(f"Refusing to overwrite preflight report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-order-manifest", type=Path, required=True)
    parser.add_argument("--validation-order-manifest", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="immutable tokenizer directory bound to both training orders",
    )
    parser.add_argument("--local-data-root", type=Path, required=True)
    parser.add_argument("--durable-checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-generation-bytes",
        type=int,
        required=True,
        help="measured upper bound for one mature checkpoint; free-space gate uses 2x",
    )
    parser.add_argument(
        "--resume-generation",
        choices=("none", "latest", "previous"),
        default="none",
    )
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--model-size", choices=("tiny", "1.3b"), default="1.3b")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "explicit whole-block activation-checkpointing decision; production "
            "1.3B execution requires --activation-checkpointing"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument(
        "--graceful-shutdown-timeout-seconds",
        type=int,
        default=_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
        help=(
            "time allowed for all ranks to finish a clean-boundary preemption "
            "checkpoint before torchrun is terminated"
        ),
    )
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, required=True)
    parser.add_argument(
        "--eval-at-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="coding-model-from-scratch")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument(
        "--run-authority",
        type=Path,
        help=(
            "write-once immutable six-GPU run authority; mandatory for --execute "
            "and forbidden for --dry-run"
        ),
    )
    parser.add_argument("--train-data-evidence", type=Path)
    parser.add_argument("--validation-data-evidence", type=Path)
    launch = parser.add_mutually_exclusive_group(required=True)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--execute", action="store_true")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="additional non-owned pretrain.train arguments after --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        activation_checkpointing = validate_production_launch_selection(
            model_size=args.model_size,
            nproc_per_node=args.nproc_per_node,
            activation_checkpointing=args.activation_checkpointing,
        )
        if args.execute and args.run_authority is None:
            raise PreflightError("--execute requires --run-authority")
        if args.dry_run and args.run_authority is not None:
            raise PreflightError(
                "--run-authority authorizes an exact --execute argv and cannot be "
                "used with --dry-run"
            )
        if args.workers < 0:
            raise PreflightError("workers must be non-negative")
        if args.checkpoint_every < 1:
            raise PreflightError("checkpoint_every must be positive for production")
        if args.graceful_shutdown_timeout_seconds < 1:
            raise PreflightError(
                "graceful_shutdown_timeout_seconds must be positive"
            )
        if args.eval_every < 1:
            raise PreflightError("eval_every must be positive for production")
        if not args.wandb_project.strip():
            raise PreflightError("wandb_project must not be empty")
        extra_args = validate_extra_trainer_args(args.trainer_args)
        runtime = inspect_runtime(
            nproc_per_node=args.nproc_per_node,
            torch_module=__import__("torch"),
        )
        local_mount, durable_mount, checkpoint = inspect_storage_and_checkpoint(
            local_data_root=args.local_data_root,
            durable_checkpoint_root=args.durable_checkpoint_root,
            checkpoint=args.checkpoint,
            resume_generation=args.resume_generation,
            checkpoint_generation_bytes=args.checkpoint_generation_bytes,
        )
        authority_path: Path | None = None
        if args.run_authority is not None:
            authority_path = _require_inside(
                args.run_authority,
                Path(durable_mount.path),
                label="run authority",
            )
            authority_path = _require_regular_file(
                authority_path,
                label="run authority",
                expected_device=durable_mount.device,
            )
        train_order = inspect_order(
            args.train_order_manifest,
            expected_split="train",
            local_data_root=args.local_data_root,
        )
        validation_order = inspect_order(
            args.validation_order_manifest,
            expected_split="validation",
            local_data_root=args.local_data_root,
            global_microbatch_rows=train_order.geometry["global_microbatch_rows"],
        )
        model_memory = inspect_model_memory(
            runtime,
            model_size=args.model_size,
            vocab_size=train_order.vocab_size,
            max_seq_len=train_order.sequence_length,
        )
        validate_order_pair(
            train_order,
            validation_order,
            world_size=runtime.world_size,
            eval_batches=args.eval_batches,
        )
        try:
            tokenizer_identity = verify_tokenizer_identity(
                args.tokenizer,
                expected_manifest_sha256=train_order.tokenizer_manifest_sha256,
                expected_vocab_size=train_order.vocab_size,
            )
        except (OSError, RuntimeError, TokenizerIdentityError) as exc:
            raise PreflightError(f"Tokenizer identity verification failed: {exc}") from exc
        train_evidence = (
            {"status": "missing", "required_for_execute": True}
            if args.train_data_evidence is None
            else verify_full_validation_evidence(
                args.train_data_evidence,
                expected_order=train_order,
                durable_root=Path(durable_mount.path),
            )
        )
        validation_evidence = (
            {"status": "missing", "required_for_execute": True}
            if args.validation_data_evidence is None
            else verify_full_validation_evidence(
                args.validation_data_evidence,
                expected_order=validation_order,
                durable_root=Path(durable_mount.path),
            )
        )
        evidence_ready = require_full_validation_for_mode(
            execute=args.execute,
            train_evidence=train_evidence,
            validation_evidence=validation_evidence,
        )
        if args.checkpoint_every > train_order.geometry["optimizer_updates"]:
            raise PreflightError(
                "checkpoint_every exceeds the frozen training optimizer-update count"
            )
        if args.eval_every > train_order.geometry["optimizer_updates"]:
            raise PreflightError(
                "eval_every exceeds the frozen training optimizer-update count"
            )
        wandb = inspect_wandb(
            mode=args.wandb_mode,
            checkpoint_parent=Path(checkpoint.checkpoint).parent,
            durable_root=Path(durable_mount.path),
        )
        command = render_torchrun_command(
            runtime=runtime,
            train_order=train_order,
            validation_order=validation_order,
            checkpoint=checkpoint,
            tokenizer_path=str(args.tokenizer.resolve()),
            model_size=args.model_size,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            eval_at_start=args.eval_at_start,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tag,
            extra_trainer_args=extra_args,
            activation_checkpointing=activation_checkpointing,
        )
        run_authority: dict[str, Any]
        if authority_path is None:
            run_authority = {
                "status": "not-validated",
                "required_for_execute": True,
            }
        else:
            invocation_argv = [
                sys.executable,
                (
                    sys.argv[0]
                    if argv is None
                    else str(Path(__file__).resolve())
                ),
                *raw_argv,
            ]
            run_authority = validate_launch_authority(
                authority_path,
                invocation_argv=invocation_argv,
            )
        report: dict[str, Any] = {
            "format": "production-pretraining-preflight",
            "format_version": 3,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "mode": "execute" if args.execute else "dry-run",
            "execute_ready": evidence_ready,
            "metadata_only_data_check": True,
            "payload_bytes_read": 0,
            "warning": (
                None
                if evidence_ready
                else "Dry-run only: full train and validation evidence is missing."
            ),
            "full_validation_evidence": {
                "train": train_evidence,
                "validation": validation_evidence,
            },
            "run_authority": run_authority,
            "distributed_strategy": "ddp",
            "runtime": asdict(runtime),
            "model_memory": asdict(model_memory),
            "process_handoff": {
                "strategy": "exec-clean-supervisor-then-isolated-torchrun",
                "preserves_launcher_pid": True,
                "signal_forwarding_wrapper": True,
                "preflight_cuda_contexts_survive": False,
                "graceful_shutdown_timeout_seconds": (
                    args.graceful_shutdown_timeout_seconds
                ),
                "reason": (
                    "torchrun's approximately 30-second worker shutdown timeout "
                    "can be shorter than a mature durable checkpoint"
                ),
            },
            "storage": {
                "local_data": asdict(local_mount),
                "durable_checkpoint": asdict(durable_mount),
                "durability_authority": (
                    "explicit --durable-checkpoint-root on a separate device"
                ),
            },
            "checkpoint": asdict(checkpoint),
            "tokenizer": {
                "path": str(args.tokenizer.resolve()),
                "manifest_sha256": tokenizer_identity.manifest_sha256,
                "vocabulary_sha256": tokenizer_identity.vocabulary_sha256,
                "vocab_size": tokenizer_identity.vocab_size,
            },
            "wandb": wandb,
            "train_order": asdict(train_order),
            "validation_order": asdict(validation_order),
            "command_argv": command,
            "command_shell": shlex.join(command),
        }
        if args.preflight_report is not None:
            report_path = args.preflight_report.resolve(strict=False)
            durable_root = Path(durable_mount.path)
            parent = _resolve_existing_directory(
                report_path.parent, label="preflight report directory"
            )
            if parent != durable_root and not parent.is_relative_to(durable_root):
                raise PreflightError(
                    f"Preflight report must be inside durable root {durable_root}: {report_path}"
                )
            _atomic_json(parent / report_path.name, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.dry_run:
            return 0
        launch_environment = dict(os.environ)
        launch_environment["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        launch_environment["PYTHONHASHSEED"] = _PYTHON_HASH_SEED
        launch_environment["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode != "disabled":
            launch_environment["WANDB_DIR"] = str(Path(checkpoint.checkpoint).parent)
        os.chdir(PROJECT_ROOT)
        exec_clean_supervisor(
            python_executable=runtime.python_executable,
            command=command,
            environment=launch_environment,
            graceful_shutdown_timeout_seconds=(
                args.graceful_shutdown_timeout_seconds
            ),
        )
        raise AssertionError("exec_clean_supervisor unexpectedly returned")
    except PreflightError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _INTERNAL_SUPERVISOR_FLAG:
        try:
            result = _internal_supervisor_main(sys.argv[2:])
        except PreflightError as exc:
            print(f"internal supervisor error: {exc}", file=sys.stderr, flush=True)
            result = 2
        raise SystemExit(result)
    raise SystemExit(main())
