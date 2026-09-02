#!/usr/bin/env python3
"""Fail-closed qualification of a six-GPU RunPod training pod.

This command is intentionally read-only apart from its explicit immutable
receipt and a bounded NCCL smoke-test scratch directory.  It never installs a
package, downloads data, launches training, or talks to a cloud API.

``bootstrap`` publishes explicitly provisional evidence before final training
orders exist; geometry qualification may consume it, but run authority cannot.
``verify`` publishes the final ``pretraining-six-gpu-hardware-runtime`` v1
contract consumed by :mod:`pretrain.run_authority`.  Both publications are
write-once and include an exact ``.sha256`` sidecar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import resource
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import launch_pretraining as launch  # noqa: E402
from pretrain import data as training_data  # noqa: E402
from pretrain.materialize import (  # noqa: E402
    FORMAT as MATERIALIZE_FORMAT,
    FORMAT_VERSION as MATERIALIZE_FORMAT_VERSION,
    JOURNAL_NAME as MATERIALIZE_JOURNAL_NAME,
)
from pretrain.run_authority import (  # noqa: E402
    HARDWARE_FORMAT,
    POD_QUALIFICATION_FORMAT,
    POD_QUALIFICATION_VERSION,
    PROVISIONAL_HARDWARE_FORMAT,
    RunAuthorityError,
    inspect_hardware_contract,
    inspect_provisional_hardware_contract,
    inspect_clean_git,
    inspect_package_lock,
)
from pretrain.tokenizer_identity import (  # noqa: E402
    TokenizerIdentityError,
    verify_tokenizer_identity,
)


FORMAT_VERSION = 1
WORLD_SIZE = 6
MINIMUM_PROVISIONAL_MEMLOCK_BYTES = 8 * 1024**2
PORTABLE_ORDER_SEED = 1_234
PORTABLE_HELDOUT_MAXIMUM_INPUT_TOKENS = 500_000_000
PORTABLE_INPUT_WEIGHTS = {"python": 0.4, "other_code": 0.4, "english": 0.2}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")
_NVIDIA_CUDA_HEADER = re.compile(r"CUDA Version:\s*(\d+\.\d+)")
_NVLINK = re.compile(r"NV\d+\Z")
_GPU_UUID = re.compile(
    r"(?:GPU-)?(?P<body>(?:[0-9a-f]{32}|"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}))\Z",
    re.IGNORECASE,
)
_TOPOLOGY_SGR = re.compile(r"\x1b\[[0-9]{1,3}(?:;[0-9]{1,3})*m")
_SAFE_TOPOLOGY_CODES = frozenset(
    {"X", "PIX", "PXB", "PHB", "NODE", "SYS", "N/A"}
)
_EXPECTED_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_ENABLE_MONITORING": "1",
    "NCCL_DEBUG": "WARN",
}
_DANGEROUS_NCCL_ENVIRONMENT = (
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
)
_BYTES_PATTERN = re.compile(
    r"^([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)?$", re.IGNORECASE
)
_BYTE_MULTIPLIERS = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

NvlinkPolicy = Literal["observe", "require-any", "require-all"]
WandbMode = Literal["disabled", "offline", "online"]


class PodQualificationError(RuntimeError):
    """A final-pod invariant could not be proven."""


def _plain_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def parse_bytes(value: str) -> int:
    match = _BYTES_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected a positive byte count, optionally suffixed B/KiB/MiB/GiB/TiB"
        )
    suffix = (match.group(2) or "B").casefold()
    return int(match.group(1)) * _BYTE_MULTIPLIERS[suffix]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise PodQualificationError(f"Receipt is not canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PodQualificationError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise PodQualificationError(f"{label} changed while it was hashed: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def _require_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise PodQualificationError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def _require_inside(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PodQualificationError(f"{label} does not exist: {path}") from exc
    if resolved != root and not resolved.is_relative_to(root):
        raise PodQualificationError(f"{label} escapes {root}: {resolved}")
    return resolved


def _run(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=None if environment is None else dict(environment),
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise PodQualificationError(f"Command failed: {list(command)!r}: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        raise PodQualificationError(
            f"Command timed out after {timeout_seconds}s: {list(command)!r}"
        ) from exc
    except UnicodeError as exc:
        _terminate_process_tree(process)
        raise PodQualificationError(f"Command output was not UTF-8: {list(command)!r}") from exc
    completed = subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if completed.returncode:
        stderr = completed.stderr.strip()[-4000:]
        raise PodQualificationError(
            f"Command exited {completed.returncode}: {list(command)!r}: {stderr}"
        )
    return completed


def _terminate_process_tree(
    process: subprocess.Popen[str], *, grace_seconds: float = 2.0
) -> None:
    """Terminate the complete subprocess session, including torchrun workers."""

    if os.name != "posix":
        process.kill()
        process.wait()
        return
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _resolve_executable(name: str) -> tuple[Path, dict[str, Any]]:
    located = shutil.which(name)
    if located is None:
        raise PodQualificationError(f"Required executable is unavailable on PATH: {name}")
    candidate = Path(located)
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PodQualificationError(f"Cannot resolve executable {name}: {candidate}") from exc
    if not stat.S_ISREG(resolved.stat().st_mode) or not os.access(resolved, os.X_OK):
        raise PodQualificationError(f"{name} is not a regular executable: {resolved}")
    return resolved, _artifact(resolved, label=f"{name} executable")


def _version_pair(value: str, *, label: str) -> tuple[int, int]:
    match = _CUDA_VERSION.fullmatch(value)
    if match is None:
        raise PodQualificationError(f"Cannot parse {label} version {value!r}")
    return int(match.group(1)), int(match.group(2))


def _format_library_version(value: Any, *, label: str) -> str:
    if isinstance(value, tuple):
        if not value or any(not _plain_int(part) for part in value):
            raise PodQualificationError(f"Invalid {label} version tuple: {value!r}")
        return ".".join(str(part) for part in value)
    if _plain_int(value, minimum=1):
        major = value // 10000
        minor = (value // 100) % 100
        patch = value % 100
        if major < 1:
            raise PodQualificationError(f"Invalid {label} version code: {value!r}")
        return f"{major}.{minor}.{patch}"
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise PodQualificationError(f"Cannot determine {label} version")


def _normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{1,8}):([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        raise PodQualificationError(f"Cannot parse PCI bus ID {value!r}")
    device = int(match.group(3), 16)
    if device > 0x1F:
        raise PodQualificationError(f"PCI device is outside [0, 31]: {value!r}")
    return (
        f"{int(match.group(1), 16):08x}:{int(match.group(2), 16):02x}:"
        f"{device:02x}.{int(match.group(4), 16)}"
    )


def _canonical_gpu_uuid(
    value: Any, *, label: str, require_gpu_prefix: bool = False
) -> str:
    """Normalize valid PyTorch/nvidia-smi UUID spellings to ``GPU-...``."""

    if not isinstance(value, str):
        value = str(value)
    raw = value.strip()
    if require_gpu_prefix and not raw.upper().startswith("GPU-"):
        raise PodQualificationError(f"{label} lacks the required GPU- prefix")
    match = _GPU_UUID.fullmatch(raw)
    if match is None:
        raise PodQualificationError(f"{label} is malformed or empty: {raw!r}")
    return f"GPU-{match.group('body').lower()}"


def _canonical_torch_pci_bus_id(properties: Any, *, label: str) -> str:
    """Read either legacy string or PyTorch 2.9 numeric PCI properties."""

    bus = getattr(properties, "pci_bus_id", None)
    if isinstance(bus, str):
        if not bus.strip():
            raise PodQualificationError(f"{label} PCI bus ID is empty")
        return _normalize_pci_bus_id(bus)
    if not isinstance(bus, int) or isinstance(bus, bool):
        raise PodQualificationError(f"{label} PCI bus ID has invalid type")
    domain = getattr(properties, "pci_domain_id", None)
    device = getattr(properties, "pci_device_id", None)
    function = getattr(properties, "pci_function_id", 0)
    fields = {
        "domain": (domain, 0xFFFFFFFF),
        "bus": (bus, 0xFF),
        "device": (device, 0x1F),
        "function": (function, 0x7),
    }
    for field, (found, maximum) in fields.items():
        if (
            not isinstance(found, int)
            or isinstance(found, bool)
            or not 0 <= found <= maximum
        ):
            raise PodQualificationError(
                f"{label} PCI {field} must be an integer in [0, {maximum}]"
            )
    assert isinstance(domain, int)
    assert isinstance(device, int)
    assert isinstance(function, int)
    return f"{domain:08x}:{bus:02x}:{device:02x}.{function}"


def _parse_visible_devices(environment: Mapping[str, str]) -> list[str]:
    raw = environment.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        raise PodQualificationError(
            "CUDA_VISIBLE_DEVICES must explicitly select exactly six GPUs"
        )
    devices = [item.strip() for item in raw.split(",")]
    if len(devices) != WORLD_SIZE or any(not item for item in devices):
        raise PodQualificationError(
            "CUDA_VISIBLE_DEVICES must contain exactly six non-empty entries"
        )
    if len(set(devices)) != WORLD_SIZE:
        raise PodQualificationError("CUDA_VISIBLE_DEVICES contains duplicate entries")
    return devices


def inspect_deterministic_environment(
    environment: Mapping[str, str], *, wandb_mode: WandbMode, omp_threads: int
) -> dict[str, Any]:
    if not _plain_int(omp_threads, minimum=1):
        raise PodQualificationError("expected OMP thread count must be positive")
    expected = {
        **_EXPECTED_ENVIRONMENT,
        "OMP_NUM_THREADS": str(omp_threads),
        "WANDB_MODE": wandb_mode,
    }
    mismatched = {
        name: {"expected": value, "found": environment.get(name)}
        for name, value in expected.items()
        if environment.get(name) != value
    }
    if mismatched:
        raise PodQualificationError(
            "Deterministic/robustness environment is not frozen: "
            + json.dumps(mismatched, sort_keys=True)
        )
    dangerous = {
        name: environment[name]
        for name in _DANGEROUS_NCCL_ENVIRONMENT
        if environment.get(name) not in (None, "", "0")
    }
    if dangerous:
        raise PodQualificationError(
            "NCCL peer/shared-memory transport is disabled: "
            + json.dumps(dangerous, sort_keys=True)
        )
    nested = [name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if name in environment]
    if nested:
        raise PodQualificationError(
            "Run pod qualification outside torchrun; nested rank variables found: "
            + ", ".join(nested)
        )
    visible = _parse_visible_devices(environment)
    return {
        "required": expected,
        "cuda_visible_devices": visible,
        "transport_disable_overrides": {
            name: environment.get(name) for name in _DANGEROUS_NCCL_ENVIRONMENT
        },
        "secrets_recorded": False,
    }


def _parse_nvidia_smi_inventory(text: str) -> list[dict[str, Any]]:
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    inventory: list[dict[str, Any]] = []
    for fields in rows:
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != 8:
            raise PodQualificationError(
                f"Unexpected nvidia-smi inventory row with {len(fields)} fields: {fields!r}"
            )
        index, uuid, name, memory_mib, capability, driver, pci_bus, mig_mode = (
            field.strip() for field in fields
        )
        try:
            index_value = int(index)
            memory_bytes = int(memory_mib) * 1024**2
            capability_pair = [int(part) for part in capability.split(".")]
        except ValueError as exc:
            raise PodQualificationError(
                f"Invalid nvidia-smi inventory values: {fields!r}"
            ) from exc
        if (
            index_value < 0
            or memory_bytes < 1
            or len(capability_pair) != 2
            or any(part < 0 for part in capability_pair)
            or not name
            or not driver
            or not pci_bus
        ):
            raise PodQualificationError(f"Invalid nvidia-smi GPU identity: {fields!r}")
        canonical_uuid = _canonical_gpu_uuid(
            uuid, label="nvidia-smi GPU UUID", require_gpu_prefix=True
        )
        inventory.append(
            {
                "index": index_value,
                "uuid": canonical_uuid,
                "name": name,
                "memory_bytes_reported": memory_bytes,
                "compute_capability": capability_pair,
                "driver_version": driver,
                "pci_bus_id": _normalize_pci_bus_id(pci_bus),
                "mig_mode": mig_mode.casefold(),
            }
        )
    if not inventory:
        raise PodQualificationError("nvidia-smi returned no GPU inventory")
    if len({row["index"] for row in inventory}) != len(inventory):
        raise PodQualificationError("nvidia-smi returned duplicate GPU indices")
    if len({row["uuid"] for row in inventory}) != len(inventory):
        raise PodQualificationError("nvidia-smi returned duplicate GPU UUIDs")
    return inventory


def _resolve_visible_inventory(
    selections: Sequence[str], inventory: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_index = {str(row["index"]): row for row in inventory}
    selected: list[dict[str, Any]] = []
    for token in selections:
        if token in by_index:
            match = by_index[token]
        else:
            uuid_matches = [
                row
                for row in inventory
                if str(row["uuid"]) == token or str(row["uuid"]).startswith(token)
            ]
            if len(uuid_matches) != 1:
                raise PodQualificationError(
                    f"CUDA_VISIBLE_DEVICES entry {token!r} does not uniquely identify a GPU"
                )
            match = uuid_matches[0]
        selected.append(dict(match))
    if len({row["uuid"] for row in selected}) != WORLD_SIZE:
        raise PodQualificationError("Visible GPU selection does not resolve to six unique GPUs")
    return selected


def _parse_topology(
    text: str, *, physical_indices: Sequence[int]
) -> dict[str, Any]:
    # Some nvidia-smi releases underline the header even when stdout is piped.
    # Normalize only numeric SGR sequences in the parsing copy.  The receipt
    # retains and hashes ``text`` below, and all other escape/control sequences
    # fail closed rather than being silently discarded.
    parse_text = _TOPOLOGY_SGR.sub("", text)
    forbidden_controls = [
        character
        for character in parse_text
        if (ord(character) < 32 and character not in "\t\n\r")
        or ord(character) in {0x7F, 0x9B}
    ]
    if "\x1b" in parse_text or forbidden_controls:
        raise PodQualificationError(
            "nvidia-smi topology contains a non-SGR escape/control sequence"
        )
    labels = [f"GPU{index}" for index in physical_indices]
    header: list[str] | None = None
    rows: dict[str, list[str]] = {}
    for line in parse_text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        gpu_tokens = [token for token in tokens if re.fullmatch(r"GPU\d+", token)]
        if header is None and all(label in gpu_tokens for label in labels):
            header = gpu_tokens
            continue
        if tokens[0] in labels:
            rows[tokens[0]] = tokens[1:]
    if header is None or any(label not in header for label in labels):
        raise PodQualificationError("Cannot resolve selected GPUs in nvidia-smi topo -m header")
    if any(label not in rows for label in labels):
        raise PodQualificationError("Cannot resolve selected GPU rows in nvidia-smi topo -m")
    matrix: list[list[str]] = []
    nvlink_pairs = 0
    for row_label in labels:
        values = rows[row_label]
        if len(values) < len(header):
            raise PodQualificationError(f"Truncated topology row for {row_label}")
        selected_values: list[str] = []
        for column_label in labels:
            value = values[header.index(column_label)].upper()
            if value not in _SAFE_TOPOLOGY_CODES and _NVLINK.fullmatch(value) is None:
                raise PodQualificationError(
                    f"Unknown GPU topology code {value!r} for {row_label}/{column_label}"
                )
            selected_values.append(value)
        matrix.append(selected_values)
    for left in range(WORLD_SIZE):
        if matrix[left][left] != "X":
            raise PodQualificationError("GPU topology diagonal must be X")
        for right in range(left + 1, WORLD_SIZE):
            if matrix[left][right] != matrix[right][left]:
                raise PodQualificationError("GPU topology matrix is not symmetric")
            if _NVLINK.fullmatch(matrix[left][right]) is not None:
                nvlink_pairs += 1
    return {
        "labels_in_visible_order": labels,
        "matrix": matrix,
        "nvlink_pairs": nvlink_pairs,
        "possible_pairs": WORLD_SIZE * (WORLD_SIZE - 1) // 2,
        "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "raw": text,
    }


def validate_gpu_observation(
    observation: Mapping[str, Any], *, nvlink_policy: NvlinkPolicy
) -> dict[str, Any]:
    if nvlink_policy not in ("observe", "require-any", "require-all"):
        raise PodQualificationError(f"Invalid NVLink policy: {nvlink_policy!r}")
    devices = observation.get("devices")
    if not isinstance(devices, list) or len(devices) != WORLD_SIZE:
        raise PodQualificationError("Exactly six CUDA devices must be observed")
    required_device_fields = {
        "visible_index",
        "physical_index",
        "uuid",
        "name",
        "pci_bus_id",
        "compute_capability",
        "total_memory_bytes",
        "available_memory_bytes",
        "multiprocessor_count",
        "bf16_supported",
    }
    for index, device in enumerate(devices):
        if not isinstance(device, dict) or required_device_fields - set(device):
            raise PodQualificationError(f"CUDA device {index} has incomplete evidence")
        if device["visible_index"] != index:
            raise PodQualificationError("CUDA visible indices are not contiguous and ordered")
        if not device["uuid"] or not device["name"] or not device["pci_bus_id"]:
            raise PodQualificationError(f"CUDA device {index} has an empty identity field")
        if device["uuid"] != _canonical_gpu_uuid(
            device["uuid"],
            label=f"CUDA device {index} UUID",
            require_gpu_prefix=True,
        ):
            raise PodQualificationError(
                f"CUDA device {index} UUID is not in canonical GPU- form"
            )
        if device["pci_bus_id"] != _normalize_pci_bus_id(device["pci_bus_id"]):
            raise PodQualificationError(
                f"CUDA device {index} PCI bus ID is not canonical"
            )
        if device["bf16_supported"] is not True:
            raise PodQualificationError(f"CUDA device {index} lacks native BF16")
        if not _plain_int(device["total_memory_bytes"], minimum=1):
            raise PodQualificationError(f"CUDA device {index} has invalid total memory")
        if not _plain_int(device["available_memory_bytes"], minimum=1):
            raise PodQualificationError(f"CUDA device {index} has invalid free memory")
        if device["available_memory_bytes"] > device["total_memory_bytes"]:
            raise PodQualificationError(f"CUDA device {index} free memory exceeds total")
        if not _plain_int(device["multiprocessor_count"], minimum=1):
            raise PodQualificationError(f"CUDA device {index} has invalid SM count")
        capability = device["compute_capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(not _plain_int(part) for part in capability)
        ):
            raise PodQualificationError(f"CUDA device {index} has invalid capability")
    identities = {
        (
            device["name"],
            device["total_memory_bytes"],
            tuple(device["compute_capability"]),
            device["multiprocessor_count"],
        )
        for device in devices
    }
    if len(identities) != 1:
        raise PodQualificationError("The six visible CUDA devices are heterogeneous")
    for field in ("uuid", "pci_bus_id", "physical_index"):
        if len({device[field] for device in devices}) != WORLD_SIZE:
            raise PodQualificationError(f"CUDA device {field} values are not unique")

    peer = observation.get("peer_access_matrix")
    if (
        not isinstance(peer, list)
        or len(peer) != WORLD_SIZE
        or any(not isinstance(row, list) or len(row) != WORLD_SIZE for row in peer)
    ):
        raise PodQualificationError("Peer-access matrix must be exactly 6x6")
    for left in range(WORLD_SIZE):
        for right in range(WORLD_SIZE):
            if not isinstance(peer[left][right], bool):
                raise PodQualificationError("Peer-access matrix entries must be boolean")
            if not peer[left][right]:
                raise PodQualificationError(
                    f"CUDA peer access is unavailable for GPU {left} -> GPU {right}"
                )
            if peer[left][right] != peer[right][left]:
                raise PodQualificationError("CUDA peer-access matrix is not symmetric")

    topology = observation.get("nvidia_topology")
    if not isinstance(topology, dict):
        raise PodQualificationError("nvidia-smi topology evidence is missing")
    topology_matrix = topology.get("matrix")
    if (
        not isinstance(topology_matrix, list)
        or len(topology_matrix) != WORLD_SIZE
        or any(
            not isinstance(row, list) or len(row) != WORLD_SIZE
            for row in topology_matrix
        )
    ):
        raise PodQualificationError("nvidia-smi topology matrix must be exactly 6x6")
    counted_nvlink_pairs = 0
    for left in range(WORLD_SIZE):
        if topology_matrix[left][left] != "X":
            raise PodQualificationError("nvidia-smi topology diagonal must be X")
        for right in range(left + 1, WORLD_SIZE):
            code = topology_matrix[left][right]
            if not isinstance(code, str) or (
                code not in _SAFE_TOPOLOGY_CODES and _NVLINK.fullmatch(code) is None
            ):
                raise PodQualificationError("nvidia-smi topology contains an invalid code")
            if code != topology_matrix[right][left]:
                raise PodQualificationError("nvidia-smi topology matrix is not symmetric")
            if _NVLINK.fullmatch(code) is not None:
                counted_nvlink_pairs += 1
    nvlink_pairs = topology.get("nvlink_pairs")
    possible_pairs = WORLD_SIZE * (WORLD_SIZE - 1) // 2
    if (
        not _plain_int(nvlink_pairs)
        or topology.get("possible_pairs") != possible_pairs
        or nvlink_pairs != counted_nvlink_pairs
    ):
        raise PodQualificationError("NVLink pair accounting is invalid")
    raw_topology = topology.get("raw")
    raw_sha256 = topology.get("raw_sha256")
    if (
        not isinstance(raw_topology, str)
        or not isinstance(raw_sha256, str)
        or hashlib.sha256(raw_topology.encode("utf-8")).hexdigest() != raw_sha256
    ):
        raise PodQualificationError("Raw nvidia-smi topology digest is invalid")
    if nvlink_policy == "require-any" and nvlink_pairs < 1:
        raise PodQualificationError("NVLink policy requires at least one visible link")
    if nvlink_policy == "require-all" and nvlink_pairs != possible_pairs:
        raise PodQualificationError(
            f"NVLink policy requires all {possible_pairs} GPU pairs; found {nvlink_pairs}"
        )

    smoke = observation.get("nccl_smoke")
    if (
        not isinstance(smoke, dict)
        or smoke.get("status") != "pass"
        or smoke.get("backend") != "nccl"
    ):
        raise PodQualificationError("Six-rank NCCL collective smoke did not pass")
    if smoke.get("world_size") != WORLD_SIZE or smoke.get("completed_ranks") != WORLD_SIZE:
        raise PodQualificationError("NCCL smoke did not complete exactly six ranks")
    if smoke.get("all_reduce_sum") != sum(range(1, WORLD_SIZE + 1)):
        raise PodQualificationError("NCCL all-reduce result is incorrect")
    if smoke.get("all_reduce_dtype") != "bfloat16":
        raise PodQualificationError("NCCL smoke did not exercise BF16 collectives")
    if not _SHA256.fullmatch(str(smoke.get("rank_results_sha256", ""))):
        raise PodQualificationError("NCCL rank-results identity is invalid")
    expected_local_ranks = list(range(WORLD_SIZE))
    if smoke.get("local_ranks") != expected_local_ranks:
        raise PodQualificationError("NCCL smoke did not prove one unique local rank per GPU")
    expected_device_uuids = [str(device["uuid"]) for device in devices]
    if smoke.get("device_uuids_in_rank_order") != expected_device_uuids:
        raise PodQualificationError(
            "NCCL smoke rank-to-GPU UUID mapping differs from visible CUDA order"
        )

    runtime_cuda = observation.get("cuda_runtime_version")
    driver_cuda = observation.get("driver_supported_cuda_version")
    if not isinstance(runtime_cuda, str) or not isinstance(driver_cuda, str):
        raise PodQualificationError("CUDA runtime/driver compatibility evidence is missing")
    if _version_pair(driver_cuda, label="driver-supported CUDA") < _version_pair(
        runtime_cuda, label="PyTorch CUDA runtime"
    ):
        raise PodQualificationError(
            f"Driver supports CUDA {driver_cuda}, older than PyTorch CUDA {runtime_cuda}"
        )
    for field in (
        "torch_version",
        "driver_version",
        "cudnn_version",
        "nccl_version",
    ):
        if not isinstance(observation.get(field), str) or not observation[field].strip():
            raise PodQualificationError(f"GPU observation lacks exact {field}")
    if observation.get("distributed_nccl_available") is not True:
        raise PodQualificationError("torch.distributed NCCL backend is unavailable")
    if observation.get("compiled_arch_supported") is not True:
        raise PodQualificationError("PyTorch CUDA build lacks this GPU architecture")

    exemplar = devices[0]
    return {
        "gpu_model": exemplar["name"],
        "gpu_memory_bytes": exemplar["total_memory_bytes"],
        "compute_capability": exemplar["compute_capability"],
        "multiprocessor_count": exemplar["multiprocessor_count"],
        "driver_version": observation["driver_version"],
        "cuda_runtime_version": runtime_cuda,
        "cudnn_version": observation["cudnn_version"],
        "nccl_version": observation["nccl_version"],
        "torch_version": observation["torch_version"],
        "bf16_supported": True,
    }


def validate_host_observation(
    observation: Mapping[str, Any], *, provisional: bool = False
) -> None:
    storage = observation.get("storage")
    if not isinstance(storage, dict):
        raise PodQualificationError("Storage evidence is missing")
    network = storage.get("network")
    work = storage.get("local_work")
    data = storage.get("local_data")
    shm = storage.get("shared_memory")
    if any(not isinstance(item, dict) for item in (network, work, data, shm)):
        raise PodQualificationError("Storage evidence has incomplete roots")
    assert isinstance(network, dict)
    assert isinstance(work, dict)
    assert isinstance(data, dict)
    assert isinstance(shm, dict)
    if network.get("classification") != "network" or network.get("read_only") is not False:
        raise PodQualificationError("Network root must be a writable network filesystem")
    policy = observation.get("admission_policy")
    if not isinstance(policy, dict):
        policy = {
            "scope": "final-launch-strict",
            "allow_overlay_local_storage": False,
            "allow_bounded_memlock": False,
            "minimum_memlock_bytes": None,
        }
    overlay_exception = (
        provisional
        and policy.get("scope") == "geometry-only-provisional"
        and policy.get("allow_overlay_local_storage") is True
    )
    if overlay_exception:
        if (
            work.get("classification") not in {"local-or-block", "ephemeral"}
            or work.get("filesystem_type") not in {"overlay", "ext4", "xfs"}
            or work.get("read_only") is not False
        ):
            raise PodQualificationError(
                "Provisional local work must be a writable overlay/local filesystem"
            )
        if (
            data.get("classification") not in {"local-or-block", "ephemeral"}
            or data.get("filesystem_type") not in {"overlay", "ext4", "xfs"}
        ):
            raise PodQualificationError(
                "Provisional local data must be on the qualified overlay/local filesystem"
            )
    else:
        if (
            work.get("classification") != "local-or-block"
            or work.get("read_only") is not False
        ):
            raise PodQualificationError(
                "Local work root must be writable local/block storage"
            )
        if (
            data.get("classification") != "local-or-block"
            or data.get("mount_read_only") is not True
        ):
            raise PodQualificationError(
                "Local training data must be on a read-only local/block mount"
            )
    if network.get("device") == work.get("device"):
        raise PodQualificationError("Network and local work roots resolve to the same device")
    if data.get("device") != work.get("device"):
        raise PodQualificationError("Local data bind and local work root are not on one device")
    for label, item in (("network", network), ("local work", work), ("shared memory", shm)):
        free = item.get("free_bytes")
        required = item.get("minimum_free_bytes")
        if not _plain_int(free) or not _plain_int(required, minimum=1) or free < required:
            raise PodQualificationError(
                f"Insufficient {label} capacity: found {free!r}, require {required!r}"
            )
    if shm.get("filesystem_type") != "tmpfs":
        raise PodQualificationError("/dev/shm must be a tmpfs mount")

    limits = observation.get("resource_limits")
    if not isinstance(limits, dict):
        raise PodQualificationError("Resource-limit evidence is missing")
    if limits.get("nofile_sufficient") is not True:
        raise PodQualificationError("RLIMIT_NOFILE is below the production floor")
    if limits.get("stack_sufficient") is not True:
        raise PodQualificationError("RLIMIT_STACK is below the production floor")
    bounded_memlock_exception = (
        provisional
        and policy.get("scope") == "geometry-only-provisional"
        and policy.get("allow_bounded_memlock") is True
    )
    if bounded_memlock_exception:
        minimum_memlock = policy.get("minimum_memlock_bytes")
        memlock = limits.get("memlock")
        if (
            not _plain_int(minimum_memlock, minimum=MINIMUM_PROVISIONAL_MEMLOCK_BYTES)
            or not isinstance(memlock, dict)
            or not _plain_int(memlock.get("soft"), minimum=minimum_memlock)
            or not _plain_int(memlock.get("hard"), minimum=minimum_memlock)
        ):
            raise PodQualificationError(
                "Bounded provisional RLIMIT_MEMLOCK is below the explicit floor"
            )
    elif limits.get("memlock_unlimited") is not True:
        raise PodQualificationError("RLIMIT_MEMLOCK soft limit must be unlimited")
    if not provisional and (
        policy.get("scope") != "final-launch-strict"
        or policy.get("allow_overlay_local_storage") is not False
        or policy.get("allow_bounded_memlock") is not False
    ):
        raise PodQualificationError(
            "Provisional storage/memlock exceptions can never authorize final launch"
        )

    environment = observation.get("environment")
    if not isinstance(environment, dict) or not environment.get("required"):
        raise PodQualificationError("Frozen environment evidence is missing")
    wandb = observation.get("wandb")
    if not isinstance(wandb, dict) or wandb.get("available") is not True:
        raise PodQualificationError("Requested W&B mode is unavailable")
    data_evidence = observation.get("data")
    if not isinstance(data_evidence, dict) or data_evidence.get("status") != "pass":
        raise PodQualificationError("Tokenizer/order path validation did not pass")
    if overlay_exception:
        authentication = data_evidence.get("content_authentication")
        limitation = data_evidence.get("writable_overlay_limitation")
        if not isinstance(authentication, dict) or authentication.get("status") not in {
            "pass",
            "ready",
        }:
            raise PodQualificationError(
                "Overlay-local provisional data lacks authenticated content evidence"
            )
        evidence_kind = authentication.get("kind")
        if evidence_kind == "authenticated-s3-restore-receipt":
            valid_authentication = (
                set(authentication.get("artifact", {}))
                == {"path", "bytes", "sha256"}
                and authentication.get("payload_sha256_verified") is True
            )
        elif evidence_kind == "portable-heldout-publication-completion":
            orders = authentication.get("orders")
            valid_authentication = (
                authentication.get("producer_contract")
                == (
                    "all-nine-packed-payloads-and-provenance-deep-authenticated-"
                    "before-atomic-heldout-publication"
                )
                and authentication.get("restore_receipt_present_at_seal") is False
                and authentication.get("payloads_rehashed_by_bootstrap") is False
                and authentication.get("kernel_write_protection") is False
                and authentication.get("final_launch_authorized") is False
                and isinstance(orders, dict)
                and set(orders) == {"validation", "test"}
                and all(
                    isinstance(orders[split], dict)
                    and set(orders[split].get("manifest", {}))
                    == {"path", "bytes", "sha256"}
                    and set(orders[split].get("payload", {}))
                    == {"path", "bytes", "sha256"}
                    for split in ("validation", "test")
                )
            )
        else:
            valid_authentication = False
        if not valid_authentication:
            raise PodQualificationError(
                "Overlay-local provisional content-authentication evidence is invalid"
            )
        if (
            not isinstance(limitation, dict)
            or limitation.get("observed_writable") is not True
            or limitation.get("kernel_write_protection") is not False
            or limitation.get("content_can_change_after_receipt") is not True
            or limitation.get("scope") != "geometry-only-provisional"
            or limitation.get("final_launch_authorized") is not False
        ):
            raise PodQualificationError(
                "Writable-overlay exception is not explicitly recorded as provisional"
            )
    package = observation.get("package_lock")
    if not isinstance(package, dict) or not _SHA256.fullmatch(
        str(package.get("lock", {}).get("sha256", ""))
    ):
        raise PodQualificationError("Exact package-lock evidence is missing")


def build_hardware_receipt(
    observation: Mapping[str, Any],
    *,
    nvlink_policy: NvlinkPolicy,
    provisional: bool = False,
    created_utc: str | None = None,
) -> dict[str, Any]:
    gpu_contract = validate_gpu_observation(observation["gpu"], nvlink_policy=nvlink_policy)
    validate_host_observation(observation["host"], provisional=provisional)
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()
    try:
        timestamp = datetime.fromisoformat(created_utc)
    except (TypeError, ValueError) as exc:
        raise PodQualificationError("created_utc must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise PodQualificationError("created_utc must be timezone-aware")
    return {
        "format": PROVISIONAL_HARDWARE_FORMAT if provisional else HARDWARE_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "provisional" if provisional else "accepted",
        "topology": "single-node",
        "world_size": WORLD_SIZE,
        "gpu_count": WORLD_SIZE,
        **gpu_contract,
        "distributed_strategy": "ddp",
        "created_utc": created_utc,
        "qualification": {
            "format": POD_QUALIFICATION_FORMAT,
            "format_version": POD_QUALIFICATION_VERSION,
            "status": "pass",
            "nvlink_policy": nvlink_policy,
            "gpu": observation["gpu"],
            "host": observation["host"],
            "source": observation["source"],
        },
    }


def publish_receipt(
    path: Path, payload: Mapping[str, Any], *, provisional: bool = False
) -> dict[str, Any]:
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    sidecar = parent / f"{path.name}.sha256"
    if destination.exists() or destination.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise PodQualificationError(
            f"Refusing to overwrite or repair immutable receipt pair: {destination}"
        )
    encoded = _canonical_json_bytes(payload)
    digest = _sha256_bytes(encoded)
    sidecar_payload = f"{digest}  {destination.name}\n".encode("ascii")
    temporary_paths: list[Path] = []
    try:
        for final_path, content in ((destination, encoded), (sidecar, sidecar_payload)):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", suffix=".part", dir=parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, final_path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise PodQualificationError(
            f"Racing writer published immutable receipt pair: {destination}"
        ) from exc
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    # Exercise the exact reader after durable publication.  The provisional
    # reader is intentionally distinct from the launch-authorizing reader.
    try:
        validated = (
            inspect_provisional_hardware_contract(destination)
            if provisional
            else inspect_hardware_contract(destination)
        )
    except RunAuthorityError as exc:
        raise PodQualificationError(
            f"Published receipt is not run-authority compatible: {exc}"
        ) from exc
    if validated["contract"]["sha256"] != digest:
        raise PodQualificationError("Published receipt changed during compatibility validation")
    return {
        "path": str(destination.resolve(strict=True)),
        "bytes": len(encoded),
        "sha256": digest,
        "sidecar": str(sidecar.resolve(strict=True)),
    }


def _inspect_resource_limits(
    *, minimum_nofile: int, minimum_stack_bytes: int
) -> dict[str, Any]:
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
    memlock_soft, memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    infinity = resource.RLIM_INFINITY
    return {
        "nofile": {"soft": nofile_soft, "hard": nofile_hard},
        "stack": {"soft": stack_soft, "hard": stack_hard},
        "memlock": {"soft": memlock_soft, "hard": memlock_hard},
        "minimum_nofile": minimum_nofile,
        "minimum_stack_bytes": minimum_stack_bytes,
        "nofile_sufficient": nofile_soft == infinity or nofile_soft >= minimum_nofile,
        "stack_sufficient": stack_soft == infinity or stack_soft >= minimum_stack_bytes,
        "memlock_unlimited": memlock_soft == infinity,
    }


def _storage_record(
    mount: launch.MountEvidence, *, minimum_free_bytes: int
) -> dict[str, Any]:
    usage = shutil.disk_usage(mount.path)
    return {
        **asdict(mount),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": minimum_free_bytes,
    }


def _inspect_restore_readiness(
    path: Path, *, local_data_root: Path
) -> dict[str, Any]:
    data_root = local_data_root.resolve(strict=True)
    receipt = _require_inside(path, data_root, label="immutable data receipt")
    if receipt.parent != data_root:
        raise PodQualificationError(
            "Immutable data receipt must be a direct child of local data root"
        )
    descriptor = _artifact(receipt, label="immutable data receipt")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PodQualificationError(f"Cannot read immutable data receipt: {exc}") from exc
    if not isinstance(payload, dict) or (
        payload.get("format") != "transcendent-logic-pretraining-data-restore"
        or payload.get("format_version") != 1
        or payload.get("status") != "ready"
        or payload.get("payload_sha256_verified") is not True
        or not _plain_int(payload.get("packed_file_count"), minimum=1)
        or not _plain_int(payload.get("packed_total_bytes"), minimum=1)
        or not isinstance(payload.get("remote_inventory_sha256"), str)
        or _SHA256.fullmatch(payload["remote_inventory_sha256"]) is None
    ):
        raise PodQualificationError(
            "Immutable data receipt does not prove a completed deep-hashed restore"
        )
    return {
        "artifact": descriptor,
        "format": payload["format"],
        "remote_inventory_sha256": payload["remote_inventory_sha256"],
        "packed_file_count": payload["packed_file_count"],
        "packed_total_bytes": payload["packed_total_bytes"],
        "payload_sha256_verified": True,
    }


def _portable_corpus_root(
    train_manifests: Mapping[str, Path], *, local_data_root: Path
) -> Path:
    if set(train_manifests) != set(training_data.DOMAIN_ORDER):
        raise PodQualificationError(
            "Portable completion evidence requires exactly three train manifests"
        )
    roots: set[Path] = set()
    for domain in training_data.DOMAIN_ORDER:
        path = train_manifests[domain].resolve(strict=True)
        try:
            root = path.parents[3]
        except IndexError as exc:
            raise PodQualificationError(
                f"Train packed-manifest path is not canonical: {path}"
            ) from exc
        expected = root / "packed" / "train" / domain / "manifest.json"
        if path != expected:
            raise PodQualificationError(
                f"Train packed-manifest path is not canonical: {path}; expected {expected}"
            )
        roots.add(root)
    if len(roots) != 1:
        raise PodQualificationError("Train packed manifests do not share one corpus root")
    corpus_root = roots.pop()
    _require_inside(corpus_root, local_data_root, label="portable corpus root")
    return corpus_root


def _inspect_portable_heldout_completion(
    *,
    validation_order_manifest: Path,
    test_order_manifest: Path,
    train_manifests: Mapping[str, Path],
    common: Mapping[str, Any],
    local_data_root: Path,
) -> dict[str, Any]:
    """Seal the finalizer's two atomic held-out publications without rescanning data.

    ``PortablePackedFinalizer`` authenticates every packed payload (full local
    SHA-256 when no restore receipt is supplied), its provenance, and the
    document indexes *before* publishing either held-out order.  Both canonical
    order directories therefore form a trusted completion marker.  This check
    re-hashes their compact order payloads and all referenced packed manifests;
    it deliberately does not re-read the roughly 130 GB packed payload.

    The resulting evidence is admissible only in the explicitly provisional
    geometry receipt.  It is not a claim of kernel immutability and can never
    authorize the production launch.
    """

    local_data_root = local_data_root.resolve(strict=True)
    corpus_root = _portable_corpus_root(
        train_manifests, local_data_root=local_data_root
    )
    journal_path = corpus_root / MATERIALIZE_JOURNAL_NAME
    journal_descriptor = _artifact(
        journal_path, label="portable packed-corpus journal"
    )
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PodQualificationError(f"Cannot read portable packed journal: {exc}") from exc
    identity = journal.get("identity") if isinstance(journal, dict) else None
    state = journal.get("state") if isinstance(journal, dict) else None
    if (
        not isinstance(journal, dict)
        or set(journal) != {"format", "format_version", "identity", "state"}
        or journal.get("format") != MATERIALIZE_FORMAT
        or journal.get("format_version") != MATERIALIZE_FORMAT_VERSION
        or not isinstance(identity, dict)
        or not isinstance(state, dict)
        or state.get("phase") != "packed"
        or not _plain_int(state.get("archive_count"), minimum=1)
        or state.get("completed_archives") != state.get("archive_count")
    ):
        raise PodQualificationError(
            "Portable packed journal does not prove a completed packed phase"
        )
    packing = identity.get("packing_configuration")
    expected_identity = {
        "tokenizer_manifest_sha256": common.get("tokenizer_manifest_sha256"),
        "curation_policy_sha256": common.get("curation_policy_sha256"),
        "selection_manifest_sha256": common.get("selection_manifest_sha256"),
    }
    if any(identity.get(field) != value for field, value in expected_identity.items()):
        raise PodQualificationError(
            "Portable journal identity differs from the train packed manifests"
        )
    if (
        not isinstance(packing, dict)
        or packing.get("sequence_length") != common.get("sequence_length")
        or packing.get("expected_vocab_size") != common.get("vocab_size")
        or packing.get("expected_eos_token_id") != common.get("eos_token_id")
    ):
        raise PodQualificationError(
            "Portable journal packing configuration differs from packed manifests"
        )

    supplied_orders = {
        "validation": validation_order_manifest,
        "test": test_order_manifest,
    }
    orders: dict[str, Any] = {}
    for split, supplied in supplied_orders.items():
        expected_path = corpus_root / "orders" / split / "manifest.json"
        path = _require_inside(
            supplied, local_data_root, label=f"portable {split} order manifest"
        )
        if path != expected_path or path.parent.name != split:
            raise PodQualificationError(
                f"Portable {split} order path is not canonical: {path}; "
                f"expected {expected_path}"
            )
        staging = corpus_root / "orders" / f".{split}.portable-part"
        if staging.exists() or staging.is_symlink():
            raise PodQualificationError(
                f"Portable {split} order still has a staging publication: {staging}"
            )
        try:
            order, order_payload_path = training_data._load_order_manifest(  # type: ignore[attr-defined]
                path, verify_checksum=True
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise PodQualificationError(
                f"Invalid portable {split} order publication: {exc}"
            ) from exc
        expected_seed = PORTABLE_ORDER_SEED + (
            1 if split == "validation" else 2
        )
        consumption = order.get("training_consumption")
        budget = order.get("input_token_budget")
        if (
            order.get("split") != split
            or order.get("seed") != expected_seed
            or order.get("expected_input_token_weights") != PORTABLE_INPUT_WEIGHTS
            or order.get("tokenizer_manifest_sha256")
            != common.get("tokenizer_manifest_sha256")
            or order.get("sequence_length") != common.get("sequence_length")
            or order.get("vocab_size") != common.get("vocab_size")
            or order.get("eos_token_id") != common.get("eos_token_id")
            or not isinstance(consumption, dict)
            or consumption.get("policy")
            != "runtime-training-geometry-not-frozen"
            or consumption.get("frozen_global_microbatch_rows") is not None
            or consumption.get("frozen_gradient_accumulation_steps") is not None
            or not isinstance(budget, dict)
            or budget.get("expected_total") != budget.get("actual_total")
            or budget.get("tolerance") != 0
            or not _plain_int(budget.get("actual_total"), minimum=1)
            or budget["actual_total"] > PORTABLE_HELDOUT_MAXIMUM_INPUT_TOKENS
        ):
            raise PodQualificationError(
                f"Portable {split} order is not the expected held-out publication"
            )

        referenced_manifests: dict[str, Any] = {}
        for domain in training_data.DOMAIN_ORDER:
            descriptor = order["dataset_manifests"][domain]
            relative = descriptor["path"]
            if Path(relative).is_absolute():
                raise PodQualificationError(
                    f"Portable {split}/{domain} packed path is absolute"
                )
            packed_path = (path.parent / relative).resolve(strict=True)
            expected_packed = (
                corpus_root / "packed" / split / domain / "manifest.json"
            )
            if packed_path != expected_packed:
                raise PodQualificationError(
                    f"Portable {split}/{domain} packed path is not canonical"
                )
            packed_descriptor = _artifact(
                packed_path, label=f"portable {split}/{domain} packed manifest"
            )
            if packed_descriptor["sha256"] != descriptor["sha256"]:
                raise PodQualificationError(
                    f"Portable {split}/{domain} packed manifest changed after order publication"
                )
            try:
                packed, _ = training_data._parse_packed_manifest(packed_path)  # type: ignore[attr-defined]
            except (OSError, TypeError, ValueError) as exc:
                raise PodQualificationError(
                    f"Invalid portable {split}/{domain} packed manifest: {exc}"
                ) from exc
            expected_packed_identity = {
                "split": split,
                "domain": domain,
                "tokenizer_manifest_sha256": common.get(
                    "tokenizer_manifest_sha256"
                ),
                "curation_policy_sha256": common.get("curation_policy_sha256"),
                "selection_manifest_sha256": common.get(
                    "selection_manifest_sha256"
                ),
                "sequence_length": common.get("sequence_length"),
                "vocab_size": common.get("vocab_size"),
                "eos_token_id": common.get("eos_token_id"),
            }
            if any(
                packed.get(field) != value
                for field, value in expected_packed_identity.items()
            ):
                raise PodQualificationError(
                    f"Portable {split}/{domain} identity differs from train corpus"
                )
            referenced_manifests[domain] = packed_descriptor

        order_payload_descriptor = _artifact(
            order_payload_path, label=f"portable {split} order payload"
        )
        if (
            order_payload_descriptor["sha256"] != order["order"]["sha256"]
            or order_payload_descriptor["bytes"] != order["order"]["bytes"]
        ):
            raise PodQualificationError(
                f"Portable {split} order payload changed during inspection"
            )
        orders[split] = {
            "manifest": _artifact(path, label=f"portable {split} order manifest"),
            "payload": order_payload_descriptor,
            "rows": order["rows"],
            "input_tokens": budget["actual_total"],
            "packed_manifests": referenced_manifests,
        }

    restore_marker = corpus_root.parent / ".RESTORE_READY.json"
    if restore_marker.exists() or restore_marker.is_symlink():
        raise PodQualificationError(
            "Portable held-out deep-verification evidence is ambiguous because a "
            f"restore receipt is present: {restore_marker}"
        )
    return {
        "kind": "portable-heldout-publication-completion",
        "format_version": 1,
        "status": "pass",
        "corpus_root": str(corpus_root),
        "packed_journal": journal_descriptor,
        "trusted_producer": _artifact(
            PROJECT_ROOT / "pretrain" / "portable_finalize.py",
            label="portable finalizer source",
        ),
        "orders": orders,
        "producer_contract": (
            "all-nine-packed-payloads-and-provenance-deep-authenticated-before-"
            "atomic-heldout-publication"
        ),
        "restore_receipt_present_at_seal": False,
        "payloads_rehashed_by_bootstrap": False,
        "kernel_write_protection": False,
        "final_launch_authorized": False,
    }


def _inspect_storage_roots(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path]:
    network_root = _require_directory(args.network_root, label="network root")
    local_work_root = _require_directory(args.local_work_root, label="local work root")
    local_data_root = _require_directory(args.local_data_root, label="local data root")
    wandb_dir = _require_directory(args.wandb_dir, label="W&B directory")
    _require_inside(wandb_dir, network_root, label="W&B directory")
    receipt_parent = _require_inside(args.receipt.parent, network_root, label="receipt parent")
    network_mount = launch.inspect_mount(network_root)
    work_mount = launch.inspect_mount(local_work_root)
    data_mount = launch.inspect_mount(local_data_root)
    if receipt_parent.stat().st_dev != network_mount.device:
        raise PodQualificationError(
            "Receipt parent is a nested mount on another device, not the network root"
        )
    if wandb_dir.stat().st_dev != network_mount.device or not os.access(wandb_dir, os.W_OK):
        raise PodQualificationError("W&B directory is not writable on the network device")
    shm_root = _require_directory(Path("/dev/shm"), label="shared-memory root")
    shm_mount = launch.inspect_mount(shm_root)
    storage = {
        "network": _storage_record(
            network_mount, minimum_free_bytes=args.minimum_network_free_bytes
        ),
        "local_work": _storage_record(
            work_mount, minimum_free_bytes=args.minimum_local_free_bytes
        ),
        "local_data": _storage_record(data_mount, minimum_free_bytes=1),
        "shared_memory": _storage_record(
            shm_mount, minimum_free_bytes=args.minimum_shm_bytes
        ),
        "receipt_parent": str(receipt_parent),
    }
    return storage, local_data_root, network_root


def _inspect_storage_and_data(args: argparse.Namespace) -> dict[str, Any]:
    storage, local_data_root, _ = _inspect_storage_roots(args)
    tokenizer_root = _require_inside(args.tokenizer, local_data_root, label="tokenizer")
    train_path = _require_inside(
        args.train_order_manifest, local_data_root, label="training order manifest"
    )
    validation_path = _require_inside(
        args.validation_order_manifest,
        local_data_root,
        label="validation order manifest",
    )

    try:
        train = launch.inspect_order(
            train_path, expected_split="train", local_data_root=local_data_root
        )
        validation = launch.inspect_order(
            validation_path,
            expected_split="validation",
            local_data_root=local_data_root,
            global_microbatch_rows=train.geometry["global_microbatch_rows"],
        )
        launch.validate_order_pair(
            train,
            validation,
            world_size=WORLD_SIZE,
            eval_batches=args.eval_batches,
        )
        tokenizer = verify_tokenizer_identity(
            tokenizer_root,
            expected_manifest_sha256=train.tokenizer_manifest_sha256,
            expected_vocab_size=train.vocab_size,
        )
    except (launch.PreflightError, TokenizerIdentityError, RuntimeError, OSError) as exc:
        raise PodQualificationError(f"Tokenizer/order validation failed: {exc}") from exc
    if validation.tokenizer_manifest_sha256 != tokenizer.manifest_sha256:
        raise PodQualificationError("Validation order uses another tokenizer")
    if validation.vocab_size != tokenizer.vocab_size:
        raise PodQualificationError("Validation order vocabulary differs from tokenizer")
    data = {
        "status": "pass",
        "local_data_root": str(local_data_root),
        "train_order": asdict(train),
        "validation_order": asdict(validation),
        "tokenizer": {
            "path": str(tokenizer_root),
            "manifest_path": str(tokenizer.manifest_path.resolve(strict=True)),
            "manifest_sha256": tokenizer.manifest_sha256,
            "vocabulary_sha256": tokenizer.vocabulary_sha256,
            "vocab_size": tokenizer.vocab_size,
        },
        "payload_checksum_scope": (
            "metadata and recorded sizes only; immutable full-data certification "
            "receipts remain mandatory for launch"
        ),
    }
    return {"storage": storage, "data": data}


def _inspect_provisional_storage_and_data(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate packed inputs without pretending final orders already exist."""

    storage, local_data_root, _ = _inspect_storage_roots(args)
    tokenizer_root = _require_inside(args.tokenizer, local_data_root, label="tokenizer")
    manifest_arguments = {
        "python": args.python_packed_manifest,
        "other_code": args.other_code_packed_manifest,
        "english": args.english_packed_manifest,
    }
    records: dict[str, Any] = {}
    train_manifest_paths: dict[str, Path] = {}
    common: dict[str, Any] | None = None
    for domain, raw_path in manifest_arguments.items():
        path = _require_inside(
            raw_path, local_data_root, label=f"{domain} packed manifest"
        )
        if path.is_symlink() or not path.is_file():
            raise PodQualificationError(
                f"{domain} packed manifest must be a regular file: {path}"
            )
        descriptor = _artifact(path, label=f"{domain} packed manifest")
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PodQualificationError(
                f"Cannot read {domain} packed manifest: {exc}"
            ) from exc
        if not isinstance(payload, dict) or (
            payload.get("format") != "packed-document-causal"
            or payload.get("domain") != domain
            or payload.get("split") != "train"
            or not _plain_int(payload.get("rows"), minimum=1)
        ):
            raise PodQualificationError(
                f"Packed manifest does not identify non-empty train/{domain}"
            )
        if common is None:
            common = payload
        else:
            for field in (
                "split",
                "sequence_length",
                "vocab_size",
                "eos_token_id",
                "tokenizer_manifest_sha256",
                "curation_policy_sha256",
                "selection_manifest_sha256",
            ):
                if payload.get(field) != common.get(field):
                    raise PodQualificationError(
                        f"Packed manifests disagree on {field}"
                    )
        records[domain] = {
            "manifest": descriptor,
            "rows": payload["rows"],
        }
        train_manifest_paths[domain] = path
    assert common is not None
    try:
        tokenizer = verify_tokenizer_identity(
            tokenizer_root,
            expected_manifest_sha256=common.get("tokenizer_manifest_sha256"),
            expected_vocab_size=common.get("vocab_size"),
        )
    except (TokenizerIdentityError, RuntimeError, OSError, ValueError) as exc:
        raise PodQualificationError(
            f"Tokenizer/packed-manifest validation failed: {exc}"
        ) from exc
    content_authentication: dict[str, Any] | None = None
    writable_overlay_limitation: dict[str, Any] | None = None
    if args.allow_provisional_overlay_local_storage:
        heldout_arguments = (
            args.heldout_validation_order_manifest,
            args.heldout_test_order_manifest,
        )
        has_heldout = any(path is not None for path in heldout_arguments)
        has_restore = args.immutable_data_receipt is not None
        if has_restore == has_heldout:
            raise PodQualificationError(
                "Overlay-local provisional qualification requires exactly one "
                "content-evidence route: --immutable-data-receipt, or both "
                "--heldout-validation-order-manifest and "
                "--heldout-test-order-manifest"
            )
        if has_restore:
            assert args.immutable_data_receipt is not None
            content_authentication = {
                "kind": "authenticated-s3-restore-receipt",
                "status": "ready",
                **_inspect_restore_readiness(
                    args.immutable_data_receipt,
                    local_data_root=local_data_root,
                ),
            }
        else:
            if not all(path is not None for path in heldout_arguments):
                raise PodQualificationError(
                    "Both held-out order manifests are required for portable "
                    "finalizer completion evidence"
                )
            assert args.heldout_validation_order_manifest is not None
            assert args.heldout_test_order_manifest is not None
            content_authentication = _inspect_portable_heldout_completion(
                validation_order_manifest=args.heldout_validation_order_manifest,
                test_order_manifest=args.heldout_test_order_manifest,
                train_manifests=train_manifest_paths,
                common=common,
                local_data_root=local_data_root,
            )
        writable_overlay_limitation = {
            "observed_writable": storage["local_data"]["mount_read_only"] is not True,
            "kernel_write_protection": storage["local_data"]["mount_read_only"] is True,
            "content_can_change_after_receipt": storage["local_data"][
                "mount_read_only"
            ]
            is not True,
            "scope": "geometry-only-provisional",
            "final_launch_authorized": False,
        }
        if storage["local_data"]["mount_read_only"] is True:
            # The exception flag is only for the actual writable-overlay case.
            raise PodQualificationError(
                "Do not request the overlay exception for a read-only data mount"
            )
    data = {
        "status": "pass",
        "qualification_scope": (
            "provisional-packed-inputs-without-final-train-or-validation-orders"
        ),
        "local_data_root": str(local_data_root),
        "packed_manifests": records,
        "common": {
            "split": common["split"],
            "sequence_length": common["sequence_length"],
            "vocab_size": common["vocab_size"],
            "eos_token_id": common["eos_token_id"],
            "tokenizer_manifest_sha256": common["tokenizer_manifest_sha256"],
            "curation_policy_sha256": common.get("curation_policy_sha256"),
            "selection_manifest_sha256": common.get("selection_manifest_sha256"),
        },
        "tokenizer": {
            "path": str(tokenizer_root),
            "manifest_path": str(tokenizer.manifest_path.resolve(strict=True)),
            "manifest_sha256": tokenizer.manifest_sha256,
            "vocabulary_sha256": tokenizer.vocabulary_sha256,
            "vocab_size": tokenizer.vocab_size,
        },
        "payload_checksum_scope": (
            "manifest bytes and tokenizer identity only; final order inspection and "
            "full-data certification remain mandatory for launch"
        ),
        "content_authentication": content_authentication,
        "writable_overlay_limitation": writable_overlay_limitation,
    }
    return {"storage": storage, "data": data}


def _collect_smi(
    environment: Mapping[str, str],
) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    executable, executable_artifact = _resolve_executable("nvidia-smi")
    executable_text = str(executable)
    query = (
        "index,uuid,name,memory.total,compute_cap,driver_version,"
        "pci.bus_id,mig.mode.current"
    )
    inventory_result = _run(
        [
            executable_text,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=30,
        environment=environment,
    )
    inventory = _parse_nvidia_smi_inventory(inventory_result.stdout)
    header = _run([executable_text], timeout_seconds=30, environment=environment).stdout
    match = _NVIDIA_CUDA_HEADER.search(header)
    if match is None:
        raise PodQualificationError("Cannot parse driver-supported CUDA version")
    topology = _run(
        [executable_text, "topo", "-m"], timeout_seconds=30, environment=environment
    ).stdout
    return inventory, match.group(1), topology, executable_artifact


def _run_nccl_smoke(
    *,
    local_work_root: Path,
    environment: Mapping[str, str],
    expected_device_uuids: Sequence[str],
) -> dict[str, Any]:
    if len(expected_device_uuids) != WORLD_SIZE or len(set(expected_device_uuids)) != WORLD_SIZE:
        raise PodQualificationError("NCCL smoke requires six unique expected GPU UUIDs")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="pretrain-nccl-smoke-", dir=local_work_root
    ) as temporary:
        output = Path(temporary)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={WORLD_SIZE}",
            str(Path(__file__).resolve(strict=True)),
            "_nccl-worker",
            "--output-dir",
            str(output),
        ]
        completed = _run(command, timeout_seconds=180, environment=environment)
        results: list[dict[str, Any]] = []
        for rank in range(WORLD_SIZE):
            path = output / f"rank-{rank}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PodQualificationError(
                    f"NCCL worker {rank} did not publish valid evidence: {exc}"
                ) from exc
            if payload.get("rank") != rank or payload.get("world_size") != WORLD_SIZE:
                raise PodQualificationError(f"NCCL worker {rank} identity mismatch")
            if payload.get("local_rank") != rank:
                raise PodQualificationError(
                    f"NCCL worker {rank} did not use its matching single-node local rank"
                )
            if payload.get("device_uuid") != expected_device_uuids[rank]:
                raise PodQualificationError(
                    f"NCCL worker {rank} ran on an unexpected CUDA device UUID"
                )
            if payload.get("all_reduce_sum") != sum(range(1, WORLD_SIZE + 1)):
                raise PodQualificationError(f"NCCL worker {rank} all-reduce mismatch")
            if payload.get("all_reduce_dtype") != "bfloat16":
                raise PodQualificationError(f"NCCL worker {rank} did not use BF16")
            if payload.get("all_gather_ranks") != list(range(WORLD_SIZE)):
                raise PodQualificationError(f"NCCL worker {rank} all-gather mismatch")
            if payload.get("broadcast_marker") != 8675309:
                raise PodQualificationError(f"NCCL worker {rank} broadcast mismatch")
            results.append(payload)
        return {
            "status": "pass",
            "backend": "nccl",
            "world_size": WORLD_SIZE,
            "completed_ranks": len(results),
            "all_reduce_sum": sum(range(1, WORLD_SIZE + 1)),
            "all_reduce_dtype": "bfloat16",
            "local_ranks": [int(payload["local_rank"]) for payload in results],
            "device_uuids_in_rank_order": [
                str(payload["device_uuid"]) for payload in results
            ],
            "rank_results_sha256": _sha256_bytes(_canonical_json_bytes(results)),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "torchrun_stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "torchrun_stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
        }


def _collect_gpu_observation(
    *, environment: Mapping[str, str], local_work_root: Path
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise PodQualificationError("PyTorch is not installed") from exc
    try:
        runtime = launch.inspect_runtime(
            nproc_per_node=WORLD_SIZE,
            torch_module=torch,
            environment=environment,
        )
    except launch.PreflightError as exc:
        raise PodQualificationError(f"Launcher runtime check failed: {exc}") from exc

    selections = _parse_visible_devices(environment)
    inventory, driver_cuda, topology_raw, nvidia_smi = _collect_smi(environment)
    selected = _resolve_visible_inventory(selections, inventory)
    if any(row["mig_mode"] not in ("disabled", "n/a", "[n/a]") for row in selected):
        raise PodQualificationError("MIG must be disabled for all six full GPUs")

    devices: list[dict[str, Any]] = []
    peer: list[list[bool]] = []
    for visible_index, (profile, smi) in enumerate(
        zip(runtime.cuda_device_profiles, selected, strict=True)
    ):
        properties = torch.cuda.get_device_properties(visible_index)
        uuid = _canonical_gpu_uuid(
            getattr(properties, "uuid", ""),
            label=f"PyTorch CUDA-visible GPU {visible_index} UUID",
        )
        pci_bus = _canonical_torch_pci_bus_id(
            properties, label=f"PyTorch CUDA-visible GPU {visible_index}"
        )
        if uuid != smi["uuid"]:
            raise PodQualificationError(
                f"CUDA-visible GPU {visible_index} UUID differs from nvidia-smi selection"
            )
        if pci_bus != smi["pci_bus_id"]:
            raise PodQualificationError(
                f"CUDA-visible GPU {visible_index} PCI bus differs from nvidia-smi"
            )
        if profile["name"] != smi["name"]:
            raise PodQualificationError(
                f"CUDA-visible GPU {visible_index} name differs from nvidia-smi"
            )
        if profile["compute_capability"] != smi["compute_capability"]:
            raise PodQualificationError(
                f"CUDA-visible GPU {visible_index} capability differs from nvidia-smi"
            )
        devices.append(
            {
                "visible_index": visible_index,
                "physical_index": smi["index"],
                "uuid": uuid,
                "name": profile["name"],
                "pci_bus_id": pci_bus,
                "compute_capability": profile["compute_capability"],
                "total_memory_bytes": profile["total_memory_bytes"],
                "available_memory_bytes": profile["available_memory_bytes"],
                "multiprocessor_count": profile["multiprocessor_count"],
                "bf16_supported": visible_index in runtime.bf16_supported_devices,
                "nvidia_smi_memory_bytes": smi["memory_bytes_reported"],
                "mig_mode": smi["mig_mode"],
            }
        )
        peer.append(
            [
                True
                if visible_index == other
                else bool(torch.cuda.can_device_access_peer(visible_index, other))
                for other in range(WORLD_SIZE)
            ]
        )
    topology = _parse_topology(
        topology_raw,
        physical_indices=[device["physical_index"] for device in devices],
    )

    capability = devices[0]["compute_capability"]
    architecture = f"sm_{capability[0]}{capability[1]}"
    arch_list = list(torch.cuda.get_arch_list())
    compiled_arch_supported = architecture in arch_list
    if not compiled_arch_supported:
        # A matching PTX target is acceptable because the driver can JIT it.
        compiled_arch_supported = f"compute_{capability[0]}{capability[1]}" in arch_list
    cudnn = _format_library_version(torch.backends.cudnn.version(), label="cuDNN")
    try:
        nccl_raw = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError) as exc:
        raise PodQualificationError(f"Cannot inspect NCCL version: {exc}") from exc
    nccl = _format_library_version(nccl_raw, label="NCCL")
    driver_versions = {device["driver_version"] for device in selected}
    if len(driver_versions) != 1:
        raise PodQualificationError("nvidia-smi reports inconsistent driver versions")
    return {
        "devices": devices,
        "peer_access_matrix": peer,
        "nvidia_topology": topology,
        "nvidia_smi_executable": nvidia_smi,
        "driver_version": next(iter(driver_versions)),
        "driver_supported_cuda_version": driver_cuda,
        "cuda_runtime_version": runtime.cuda_runtime,
        "torch_version": runtime.torch_version,
        "cudnn_version": cudnn,
        "nccl_version": nccl,
        "distributed_nccl_available": True,
        "torch_cuda_arch_list": arch_list,
        "compiled_arch_supported": compiled_arch_supported,
        "nccl_smoke": _run_nccl_smoke(
            local_work_root=local_work_root,
            environment=environment,
            expected_device_uuids=[str(device["uuid"]) for device in devices],
        ),
    }


def collect_observation(args: argparse.Namespace) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise PodQualificationError("Final RunPod qualification requires Linux")
    if args.command == "verify" and not _plain_int(args.eval_batches, minimum=1):
        raise PodQualificationError("eval_batches must be positive")
    if not _plain_int(args.minimum_nofile, minimum=1):
        raise PodQualificationError("minimum_nofile must be positive")
    if not _plain_int(args.minimum_stack_bytes, minimum=1):
        raise PodQualificationError("minimum_stack_bytes must be positive")

    environment_evidence = inspect_deterministic_environment(
        os.environ,
        wandb_mode=args.wandb_mode,
        omp_threads=args.omp_threads,
    )
    storage_data = (
        _inspect_provisional_storage_and_data(args)
        if args.command == "bootstrap"
        else _inspect_storage_and_data(args)
    )
    network_root = Path(storage_data["storage"]["network"]["path"])
    wandb_dir = args.wandb_dir.resolve(strict=True)
    try:
        wandb = launch.inspect_wandb(
            mode=args.wandb_mode,
            checkpoint_parent=wandb_dir,
            durable_root=network_root,
            environment=os.environ,
        )
    except launch.PreflightError as exc:
        raise PodQualificationError(f"W&B check failed: {exc}") from exc
    wandb_evidence = {
        **wandb,
        "available": True,
        "network_probe_performed": False,
        "credential_value_recorded": False,
    }
    try:
        package_lock = inspect_package_lock(args.package_lock)
    except RunAuthorityError as exc:
        raise PodQualificationError(f"Package lock does not match this pod: {exc}") from exc
    try:
        git_identity = inspect_clean_git(PROJECT_ROOT)
    except RunAuthorityError as exc:
        raise PodQualificationError(f"Qualification source tree is not immutable: {exc}") from exc

    requirements_train = _artifact(
        PROJECT_ROOT / "requirements-train.txt", label="training requirements"
    )
    requirements_wandb = _artifact(
        PROJECT_ROOT / "requirements-wandb.txt", label="W&B requirements"
    )
    script = _artifact(Path(__file__).resolve(strict=True), label="qualification script")
    provisional = args.command == "bootstrap"
    admission_policy = {
        "scope": (
            "geometry-only-provisional" if provisional else "final-launch-strict"
        ),
        "allow_overlay_local_storage": bool(
            provisional and args.allow_provisional_overlay_local_storage
        ),
        "allow_bounded_memlock": bool(
            provisional and args.allow_provisional_bounded_memlock
        ),
        "minimum_memlock_bytes": (
            args.minimum_provisional_memlock_bytes
            if provisional and args.allow_provisional_bounded_memlock
            else None
        ),
        "final_launch_authorized": not provisional,
    }
    host = {
        **storage_data,
        "admission_policy": admission_policy,
        "environment": environment_evidence,
        "resource_limits": _inspect_resource_limits(
            minimum_nofile=args.minimum_nofile,
            minimum_stack_bytes=args.minimum_stack_bytes,
        ),
        "wandb": wandb_evidence,
        "package_lock": package_lock,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
    }
    # Reject bad mounts, headroom, ulimits, data, environment, or W&B before
    # initializing CUDA contexts or spending time on the six-process smoke.
    validate_host_observation(host, provisional=provisional)
    local_work_root = Path(storage_data["storage"]["local_work"]["path"])
    gpu = _collect_gpu_observation(
        environment=os.environ,
        local_work_root=local_work_root,
    )
    if gpu["torch_version"] != package_lock["torch_version"]:
        raise PodQualificationError(
            "PyTorch runtime version differs from exact package-lock version"
        )
    return {
        "gpu": gpu,
        "host": host,
        "source": {
            "qualification_script": script,
            "requirements_train": requirements_train,
            "requirements_wandb": requirements_wandb,
            "git": git_identity,
            "argv": list(sys.argv),
        },
    }


def _write_worker_result(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _nccl_worker(output_dir: Path) -> int:
    try:
        import torch
        import torch.distributed as distributed

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if (
            world_size != WORLD_SIZE
            or not 0 <= rank < WORLD_SIZE
            or not 0 <= local_rank < WORLD_SIZE
            or local_rank != rank
        ):
            raise PodQualificationError("NCCL worker received invalid rank topology")
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise PodQualificationError("NCCL output directory is invalid")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        try:
            reduced = torch.tensor(
                [rank + 1], device=f"cuda:{local_rank}", dtype=torch.bfloat16
            )
            distributed.all_reduce(reduced)
            rank_tensor = torch.tensor([rank], device=f"cuda:{local_rank}", dtype=torch.int64)
            gathered = [torch.zeros_like(rank_tensor) for _ in range(WORLD_SIZE)]
            distributed.all_gather(gathered, rank_tensor)
            marker = torch.tensor(
                [8675309 if rank == 0 else -1],
                device=f"cuda:{local_rank}",
                dtype=torch.int64,
            )
            distributed.broadcast(marker, src=0)
            distributed.barrier()
            torch.cuda.synchronize(local_rank)
            payload = {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "device_uuid": _canonical_gpu_uuid(
                    getattr(
                        torch.cuda.get_device_properties(local_rank), "uuid", ""
                    ),
                    label=f"NCCL worker rank {rank} GPU UUID",
                ),
                "all_reduce_sum": int(reduced.item()),
                "all_reduce_dtype": "bfloat16",
                "all_gather_ranks": [int(item.item()) for item in gathered],
                "broadcast_marker": int(marker.item()),
            }
            _write_worker_result(output_dir / f"rank-{rank}.json", payload)
        finally:
            distributed.destroy_process_group()
    except Exception as exc:
        print(f"nccl-worker: ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(child: argparse.ArgumentParser) -> None:
        child.add_argument("--network-root", type=Path, required=True)
        child.add_argument("--local-work-root", type=Path, required=True)
        child.add_argument("--local-data-root", type=Path, required=True)
        child.add_argument("--tokenizer", type=Path, required=True)
        child.add_argument("--package-lock", type=Path, required=True)
        child.add_argument("--wandb-dir", type=Path, required=True)
        child.add_argument("--receipt", type=Path, required=True)
        child.add_argument(
            "--wandb-mode", choices=("disabled", "offline", "online"), required=True
        )
        child.add_argument(
            "--nvlink-policy",
            choices=("observe", "require-any", "require-all"),
            required=True,
        )
        child.add_argument("--omp-threads", type=int, required=True)
        child.add_argument(
            "--minimum-network-free-bytes", type=parse_bytes, required=True
        )
        child.add_argument(
            "--minimum-local-free-bytes", type=parse_bytes, required=True
        )
        child.add_argument(
            "--minimum-shm-bytes", type=parse_bytes, default=16 * 1024**3
        )
        child.add_argument("--minimum-nofile", type=int, default=65_536)
        child.add_argument(
            "--minimum-stack-bytes", type=parse_bytes, default=8 * 1024**2
        )

    bootstrap = commands.add_parser(
        "bootstrap",
        help=(
            "qualify the real pod and packed inputs before final orders exist; "
            "the provisional receipt cannot authorize training"
        ),
    )
    add_common(bootstrap)
    bootstrap.add_argument("--python-packed-manifest", type=Path, required=True)
    bootstrap.add_argument("--other-code-packed-manifest", type=Path, required=True)
    bootstrap.add_argument("--english-packed-manifest", type=Path, required=True)
    bootstrap.add_argument(
        "--allow-provisional-overlay-local-storage",
        action="store_true",
        help=(
            "geometry-only exception for a writable overlay/local data copy; "
            "requires authenticated restore evidence or both atomic held-out "
            "publications, and never authorizes full training"
        ),
    )
    bootstrap.add_argument(
        "--immutable-data-receipt",
        type=Path,
        help="deep-hashed S3 restore readiness receipt for the local data root",
    )
    bootstrap.add_argument(
        "--heldout-validation-order-manifest",
        type=Path,
        help=(
            "canonical validation order published after portable finalizer deep "
            "authentication; use together with --heldout-test-order-manifest"
        ),
    )
    bootstrap.add_argument(
        "--heldout-test-order-manifest",
        type=Path,
        help=(
            "canonical test order published after portable finalizer deep "
            "authentication; use together with --heldout-validation-order-manifest"
        ),
    )
    bootstrap.add_argument(
        "--allow-provisional-bounded-memlock",
        action="store_true",
        help=(
            "geometry-only exception for finite RLIMIT_MEMLOCK; the observed limit "
            "and explicit floor remain in the provisional receipt"
        ),
    )
    bootstrap.add_argument(
        "--minimum-provisional-memlock-bytes",
        type=parse_bytes,
        default=MINIMUM_PROVISIONAL_MEMLOCK_BYTES,
    )

    verify = commands.add_parser(
        "verify", help="run final read-only pod checks and publish a hardware receipt"
    )
    add_common(verify)
    verify.add_argument("--train-order-manifest", type=Path, required=True)
    verify.add_argument("--validation-order-manifest", type=Path, required=True)
    verify.add_argument("--eval-batches", type=int, required=True)
    verify.set_defaults(
        allow_provisional_overlay_local_storage=False,
        immutable_data_receipt=None,
        heldout_validation_order_manifest=None,
        heldout_test_order_manifest=None,
        allow_provisional_bounded_memlock=False,
        minimum_provisional_memlock_bytes=None,
    )

    worker = commands.add_parser("_nccl-worker", help=argparse.SUPPRESS)
    worker.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "_nccl-worker":
        return _nccl_worker(args.output_dir)
    try:
        provisional = args.command == "bootstrap"
        observation = collect_observation(args)
        receipt = build_hardware_receipt(
            observation,
            nvlink_policy=args.nvlink_policy,
            provisional=provisional,
        )
        result = publish_receipt(args.receipt, receipt, provisional=provisional)
    except (PodQualificationError, OSError, RuntimeError, ValueError) as exc:
        print(f"pod-qualification: ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
