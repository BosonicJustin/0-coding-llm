"""Immutable, fail-closed authorization artifact for a production pretraining run.

The authority records every
operator-controlled input that can change the scientific or economic meaning
of a run and can be revalidated without network access.  Publication is
write-once and accompanied by a SHA-256 sidecar.

This module does *not* claim that an expected hardware contract describes the
current host.  The CUDA preflight must prove that separately immediately before
launch.  Here, the contract and the measured six-GPU qualification receipt are
cryptographically bound to the exact command that the operator approved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any

from .model import ModelConfig
from pretrain import data as training_data
from pretrain.tokenizer_identity import verify_tokenizer_identity


AUTHORITY_FORMAT = "immutable-pretraining-run-authority"
AUTHORITY_VERSION = 1
PACKAGE_LOCK_FORMAT = "pretraining-python-package-lock"
HARDWARE_FORMAT = "pretraining-six-gpu-hardware-runtime"
PROVISIONAL_HARDWARE_FORMAT = "pretraining-six-gpu-provisional-hardware-runtime"
POD_QUALIFICATION_FORMAT = "runpod-six-gpu-pod-qualification"
POD_QUALIFICATION_VERSION = 1
CORPUS_QUALIFICATION_FORMAT = "materialized-pretraining-corpus-qualification"
CORPUS_QUALIFICATION_VERSION = 1
GEOMETRY_FORMAT = "pretraining-accepted-geometry"
RECIPE_FORMAT = "pretraining-training-recipe"
CERTIFICATION_FORMAT = "production-pretraining-full-data-validation"
WORLD_SIZE = 6
MODEL_SIZE = "1.3b"
EXPECTED_PARAMETER_COUNT = 1_283_557_376
EXPECTED_INPUT_TOKEN_TARGETS = {
    "train": 52_580_000_000,
    "validation": 500_000_000,
    "test": 500_000_000,
}
EXPECTED_INPUT_TOKEN_WEIGHTS = {"python": 0.4, "other_code": 0.4, "english": 0.2}
MAXIMUM_TARGET_SHORTFALL_FRACTION = Decimal("0.001")
MAXIMUM_MIXTURE_ABSOLUTE_TOLERANCE = Decimal("0.000001")
MINIMUM_SAMPLE_ROWS_PER_DOMAIN = 8
CORPUS_QUALIFICATION_CHECKS = frozenset(
    {
        "authenticated_corpus_manifest_and_sidecar",
        "authenticated_provenance",
        "tokenizer_model_and_corpus_compatible",
        "all_splits_and_domains_present",
        "orders_checksum_unique_and_in_range",
        "packed_shards_sizes_checksums_and_token_ranges",
        "boundary_bits_and_loss_masks_consistent",
        "no_padded_or_partial_training_rows",
        "exact_token_budgets_within_tolerance",
        "realized_mixture_within_tolerance",
        "document_offsets_and_totals_consistent",
        "source_and_group_identities_disjoint_across_splits",
        "deterministic_random_reads",
        "sufficient_rows_for_six_rank_ddp",
        "stable_corpus_identity_during_qualification",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_PACKAGE_NORMALIZER = re.compile(r"[-_.]+")


class RunAuthorityError(RuntimeError):
    """Raised when a production run invariant cannot be proven."""


def _plain_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RunAuthorityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunAuthorityError(f"Value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise RunAuthorityError(f"{label} must be a regular non-symlink file: {candidate}")
    return candidate.resolve(strict=True)


def _directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise RunAuthorityError(f"{label} must be a non-symlink directory: {candidate}")
    return candidate.resolve(strict=True)


def _artifact(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise RunAuthorityError(f"{label} changed while it was being hashed: {resolved}")
    return {
        "path": str(resolved),
        "bytes": after.st_size,
        "sha256": digest,
    }


def _read_bound_bytes(descriptor: Mapping[str, Any], *, label: str) -> bytes:
    path = Path(str(descriptor["path"]))
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RunAuthorityError(f"Cannot read {label}: {exc}") from exc
    if (
        len(payload) != descriptor.get("bytes")
        or hashlib.sha256(payload).hexdigest() != descriptor.get("sha256")
    ):
        raise RunAuthorityError(f"{label} changed after it was hashed: {path}")
    return payload


def _json_object(path: str | Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _artifact(path, label=label)
    try:
        encoded = _read_bound_bytes(descriptor, label=label)
        payload = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunAuthorityError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunAuthorityError(f"{label} JSON root must be an object")
    # Reject values that Python's permissive decoder accepts but canonical JSON cannot encode.
    _canonical_json_bytes(payload)
    return payload, descriptor


def _verified_sidecar(path: str | Path, *, label: str) -> dict[str, Any]:
    subject = _artifact(path, label=label)
    subject_path = Path(subject["path"])
    sidecar_path = subject_path.with_name(f"{subject_path.name}.sha256")
    sidecar = _artifact(sidecar_path, label=f"{label} SHA-256 sidecar")
    try:
        text = _read_bound_bytes(
            sidecar, label=f"{label} SHA-256 sidecar"
        ).decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RunAuthorityError(f"Cannot read {label} SHA-256 sidecar: {exc}") from exc
    expected = f"{subject['sha256']}  {subject_path.name}\n"
    if text != expected:
        raise RunAuthorityError(
            f"{label} SHA-256 sidecar must contain exactly {expected!r}"
        )
    return {"artifact": subject, "sidecar": sidecar}


def _git(project_root: str | Path, *args: str, allow_failure: bool = False) -> str | None:
    command = ["git", "-C", str(project_root), *args]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode:
        if allow_failure:
            return None
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RunAuthorityError(f"Git command failed ({' '.join(command)}): {detail}")
    return completed.stdout.rstrip("\n")


def _git_archive_sha256(project_root: Path) -> tuple[str, int]:
    """Hash canonical ``git archive HEAD`` bytes with SHA-256, not Git SHA-1 alone."""

    process = subprocess.Popen(
        ["git", "-C", str(project_root), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    digest = hashlib.sha256()
    byte_count = 0
    stderr = ""
    try:
        while chunk := process.stdout.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    finally:
        process.stdout.close()
        process.stderr.close()
    returncode = process.wait()
    if returncode:
        raise RunAuthorityError(f"Cannot archive clean Git HEAD: {stderr}")
    return digest.hexdigest(), byte_count


def inspect_clean_git(project_root: str | Path) -> dict[str, Any]:
    """Return an immutable Git identity, rejecting every tracked/untracked change."""

    root = _directory(project_root, label="project root")
    top = _git(root, "rev-parse", "--show-toplevel")
    assert top is not None
    if Path(top).resolve(strict=True) != root:
        raise RunAuthorityError(f"Project root is not the Git top-level directory: {root}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise RunAuthorityError(f"Git worktree is not clean:\n{preview}")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not isinstance(commit, str) or _GIT_OBJECT.fullmatch(commit) is None:
        raise RunAuthorityError("Git HEAD is not a full object identity")
    if not isinstance(tree, str) or _GIT_OBJECT.fullmatch(tree) is None:
        raise RunAuthorityError("Git tree is not a full object identity")
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    remote = _git(root, "remote", "get-url", "origin", allow_failure=True)
    archive_sha256, archive_bytes = _git_archive_sha256(root)
    return {
        "project_root": str(root),
        "commit": commit,
        "tree": tree,
        "head_archive_sha256": archive_sha256,
        "head_archive_bytes": archive_bytes,
        "branch": branch,
        # Never copy a possibly credential-bearing remote URL into run evidence.
        "origin_sha256": hashlib.sha256(remote.encode("utf-8")).hexdigest()
        if remote
        else None,
        "clean": True,
    }


def _normalize_package(name: str) -> str:
    return _PACKAGE_NORMALIZER.sub("-", name).lower()


def _major_minor(version: str, *, package: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RunAuthorityError(f"Cannot validate {package} version {version!r}")
    return int(match.group(1)), int(match.group(2))


def _validate_training_package_versions(packages: Mapping[str, str]) -> None:
    required = {"torch", "numpy", "tokenizers"}
    missing = sorted(required - set(packages))
    if missing:
        raise RunAuthorityError(f"Environment is missing training dependencies: {missing}")
    if _major_minor(packages["torch"], package="torch") < (2, 6):
        raise RunAuthorityError("Training environment requires torch>=2.6 for FlexAttention")
    numpy_version = _major_minor(packages["numpy"], package="numpy")
    if not (numpy_version >= (2, 0) and numpy_version < (3, 0)):
        raise RunAuthorityError("Training environment requires numpy>=2.0,<3")
    tokenizers_version = _major_minor(packages["tokenizers"], package="tokenizers")
    if not ((0, 21) <= tokenizers_version < (0, 24)):
        raise RunAuthorityError("Training environment requires tokenizers>=0.21,<0.24")


def current_environment_identity() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            raise RunAuthorityError("Installed distribution lacks a name or version")
        normalized = _normalize_package(str(name))
        previous = packages.get(normalized)
        if previous is not None and previous != str(version):
            raise RunAuthorityError(
                f"Installed environment contains conflicting {normalized!r} versions"
            )
        packages[normalized] = str(version)
    executable = _regular_file(Path(sys.executable).resolve(), label="Python executable")
    return {
        "python": {
            "implementation": platform.python_implementation().lower(),
            "version": platform.python_version(),
            "executable": str(executable),
            "executable_bytes": executable.stat().st_size,
            "executable_sha256": sha256_file(executable),
        },
        "packages": dict(sorted(packages.items())),
    }


def inspect_package_lock(path: str | Path) -> dict[str, Any]:
    lock, descriptor = _json_object(path, label="Python package lock")
    if lock.get("format") != PACKAGE_LOCK_FORMAT or lock.get("format_version") != 1:
        raise RunAuthorityError("Unsupported Python package-lock format")
    expected_python = lock.get("python")
    expected_packages = lock.get("packages")
    if not isinstance(expected_python, dict) or set(expected_python) != {
        "implementation",
        "version",
    }:
        raise RunAuthorityError(
            "Package lock python must contain exactly implementation and version"
        )
    if not isinstance(expected_packages, dict) or not expected_packages:
        raise RunAuthorityError("Package lock packages must be a non-empty object")
    normalized: dict[str, str] = {}
    for name, version in expected_packages.items():
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RunAuthorityError("Package lock contains an invalid name/version")
        canonical_name = _normalize_package(name)
        if canonical_name != name or canonical_name in normalized:
            raise RunAuthorityError(
                "Package-lock names must be unique normalized PEP 503 names"
            )
        normalized[canonical_name] = version
    _validate_training_package_versions(normalized)
    actual = current_environment_identity()
    if expected_python != {
        "implementation": actual["python"]["implementation"],
        "version": actual["python"]["version"],
    }:
        raise RunAuthorityError("Package-lock Python identity differs from this interpreter")
    if dict(sorted(normalized.items())) != actual["packages"]:
        expected_names = set(normalized)
        actual_names = set(actual["packages"])
        missing_installed = sorted(expected_names - actual_names)[:10]
        unexpected = sorted(actual_names - expected_names)[:10]
        mismatched = sorted(
            name
            for name in expected_names & actual_names
            if normalized[name] != actual["packages"][name]
        )[:10]
        raise RunAuthorityError(
            "Installed packages do not exactly match lock "
            f"(missing={missing_installed}, unexpected={unexpected}, "
            f"version_mismatch={mismatched})"
        )
    return {
        "lock": descriptor,
        "python": actual["python"],
        "packages_sha256": canonical_sha256(actual["packages"]),
        "package_count": len(actual["packages"]),
        "torch_version": actual["packages"]["torch"],
        "numpy_version": actual["packages"]["numpy"],
    }


def _recorded_artifact(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise RunAuthorityError(
            f"{label} must contain exactly path, bytes, and sha256"
        )
    path = value["path"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise RunAuthorityError(f"{label} path must be absolute")
    if not _plain_int(value["bytes"], minimum=1):
        raise RunAuthorityError(f"{label} bytes must be positive")
    _require_sha256(value["sha256"], field=f"{label} sha256")
    return dict(value)


def _validate_qualification_git(value: Any) -> dict[str, Any]:
    required = {
        "project_root",
        "commit",
        "tree",
        "head_archive_sha256",
        "head_archive_bytes",
        "branch",
        "origin_sha256",
        "clean",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RunAuthorityError("Pod qualification lacks an exact clean-Git identity")
    if value["clean"] is not True:
        raise RunAuthorityError("Pod qualification source tree was not clean")
    if not isinstance(value["project_root"], str) or not Path(
        value["project_root"]
    ).is_absolute():
        raise RunAuthorityError("Pod qualification Git root must be absolute")
    for field in ("commit", "tree"):
        if not isinstance(value[field], str) or _GIT_OBJECT.fullmatch(value[field]) is None:
            raise RunAuthorityError(f"Pod qualification Git {field} is invalid")
    _require_sha256(
        value["head_archive_sha256"], field="pod qualification Git archive sha256"
    )
    if not _plain_int(value["head_archive_bytes"], minimum=1):
        raise RunAuthorityError("Pod qualification Git archive bytes must be positive")
    if value["branch"] is not None and not isinstance(value["branch"], str):
        raise RunAuthorityError("Pod qualification Git branch is invalid")
    if value["origin_sha256"] is not None:
        _require_sha256(
            value["origin_sha256"], field="pod qualification Git origin sha256"
        )
    return dict(value)


def _validate_pod_qualification(contract: Mapping[str, Any]) -> dict[str, Any]:
    qualification = contract.get("qualification")
    required_qualification = {
        "format",
        "format_version",
        "status",
        "nvlink_policy",
        "gpu",
        "host",
        "source",
    }
    if not isinstance(qualification, dict) or set(qualification) != required_qualification:
        raise RunAuthorityError(
            "Hardware contract must contain the exact RunPod qualification evidence"
        )
    if (
        qualification["format"] != POD_QUALIFICATION_FORMAT
        or qualification["format_version"] != POD_QUALIFICATION_VERSION
        or qualification["status"] != "pass"
    ):
        raise RunAuthorityError("RunPod six-GPU qualification did not pass")
    if qualification["nvlink_policy"] not in (
        "observe",
        "require-any",
        "require-all",
    ):
        raise RunAuthorityError("RunPod qualification has an invalid NVLink policy")

    gpu = qualification["gpu"]
    host = qualification["host"]
    source = qualification["source"]
    if not isinstance(gpu, dict) or not isinstance(host, dict) or not isinstance(source, dict):
        raise RunAuthorityError("RunPod qualification evidence objects are incomplete")
    devices = gpu.get("devices")
    if not isinstance(devices, list) or len(devices) != WORLD_SIZE:
        raise RunAuthorityError("RunPod qualification must contain exactly six GPU records")
    expected_uuids: list[str] = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict) or device.get("visible_index") != index:
            raise RunAuthorityError("RunPod qualification GPU order is invalid")
        uuid = device.get("uuid")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise RunAuthorityError("RunPod qualification GPU UUID is invalid")
        expected_uuids.append(uuid)
        if device.get("bf16_supported") is not True:
            raise RunAuthorityError("RunPod qualification includes a non-BF16 GPU")
        expected_device_fields = {
            "name": contract["gpu_model"],
            "total_memory_bytes": contract["gpu_memory_bytes"],
            "compute_capability": contract["compute_capability"],
            "multiprocessor_count": contract["multiprocessor_count"],
        }
        if any(device.get(field) != expected for field, expected in expected_device_fields.items()):
            raise RunAuthorityError(
                "RunPod qualification GPU evidence disagrees with the hardware contract"
            )
    if len(set(expected_uuids)) != WORLD_SIZE:
        raise RunAuthorityError("RunPod qualification GPU UUIDs are not unique")
    for nested, top_level in (
        ("driver_version", "driver_version"),
        ("cuda_runtime_version", "cuda_runtime_version"),
        ("cudnn_version", "cudnn_version"),
        ("nccl_version", "nccl_version"),
        ("torch_version", "torch_version"),
    ):
        if gpu.get(nested) != contract[top_level]:
            raise RunAuthorityError(
                f"RunPod qualification {nested} disagrees with the hardware contract"
            )
    smoke = gpu.get("nccl_smoke")
    if not isinstance(smoke, dict) or (
        smoke.get("status") != "pass"
        or smoke.get("backend") != "nccl"
        or smoke.get("world_size") != WORLD_SIZE
        or smoke.get("completed_ranks") != WORLD_SIZE
        or smoke.get("all_reduce_sum") != sum(range(1, WORLD_SIZE + 1))
        or smoke.get("all_reduce_dtype") != "bfloat16"
        or smoke.get("local_ranks") != list(range(WORLD_SIZE))
        or smoke.get("device_uuids_in_rank_order") != expected_uuids
    ):
        raise RunAuthorityError("RunPod qualification lacks a valid six-rank NCCL smoke")
    _require_sha256(
        smoke.get("rank_results_sha256"), field="NCCL rank-results sha256"
    )

    package = host.get("package_lock")
    required_package = {
        "lock",
        "python",
        "packages_sha256",
        "package_count",
        "torch_version",
        "numpy_version",
    }
    if not isinstance(package, dict) or set(package) != required_package:
        raise RunAuthorityError("RunPod qualification lacks exact package-lock evidence")
    _recorded_artifact(package["lock"], label="qualified package lock")
    python_identity = package["python"]
    if not isinstance(python_identity, dict) or set(python_identity) != {
        "implementation",
        "version",
        "executable",
        "executable_bytes",
        "executable_sha256",
    }:
        raise RunAuthorityError("RunPod qualification Python identity is incomplete")
    if not isinstance(python_identity["implementation"], str) or not isinstance(
        python_identity["version"], str
    ):
        raise RunAuthorityError("RunPod qualification Python identity is invalid")
    if not isinstance(python_identity["executable"], str) or not Path(
        python_identity["executable"]
    ).is_absolute():
        raise RunAuthorityError("RunPod qualification Python executable is invalid")
    if not _plain_int(python_identity["executable_bytes"], minimum=1):
        raise RunAuthorityError("RunPod qualification Python executable size is invalid")
    _require_sha256(
        python_identity["executable_sha256"],
        field="RunPod qualification Python executable sha256",
    )
    _require_sha256(package["packages_sha256"], field="qualified packages sha256")
    if not _plain_int(package["package_count"], minimum=1):
        raise RunAuthorityError("RunPod qualification package count is invalid")
    if package["torch_version"] != contract["torch_version"]:
        raise RunAuthorityError("Qualified package-lock Torch differs from hardware runtime")
    if not isinstance(package["numpy_version"], str) or not package["numpy_version"]:
        raise RunAuthorityError("RunPod qualification NumPy version is invalid")
    data_evidence = host.get("data")
    environment_evidence = host.get("environment")
    wandb_evidence = host.get("wandb")
    if not isinstance(data_evidence, dict) or data_evidence.get("status") != "pass":
        raise RunAuthorityError("RunPod qualification data/path inspection did not pass")
    if (
        not isinstance(environment_evidence, dict)
        or environment_evidence.get("secrets_recorded") is not False
    ):
        raise RunAuthorityError("RunPod qualification environment evidence is not secret-safe")
    if (
        not isinstance(wandb_evidence, dict)
        or wandb_evidence.get("credential_value_recorded") is not False
    ):
        raise RunAuthorityError("RunPod qualification recorded W&B credential material")

    required_source = {
        "qualification_script",
        "requirements_train",
        "requirements_wandb",
        "git",
        "argv",
    }
    if set(source) != required_source:
        raise RunAuthorityError("RunPod qualification source provenance is incomplete")
    for field in ("qualification_script", "requirements_train", "requirements_wandb"):
        _recorded_artifact(source[field], label=f"pod qualification {field}")
    _validate_qualification_git(source["git"])
    argv = source["argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item or "\x00" in item for item in argv
    ):
        raise RunAuthorityError("RunPod qualification argv is invalid")
    return qualification


def _validate_qualification_provenance(
    *,
    hardware: Mapping[str, Any],
    environment: Mapping[str, Any],
    git: Mapping[str, Any],
    project_root: Path,
) -> None:
    qualification = hardware["expected"]["qualification"]
    package = qualification["host"]["package_lock"]
    for field in (
        "lock",
        "python",
        "packages_sha256",
        "package_count",
        "torch_version",
        "numpy_version",
    ):
        if package[field] != environment[field]:
            raise RunAuthorityError(
                f"Qualified pod {field} differs from the run-authority package lock"
            )

    source = qualification["source"]
    if source["git"] != git:
        raise RunAuthorityError(
            "Qualified pod source identity differs from the current clean Git tree"
        )
    current_sources = {
        "qualification_script": project_root / "scripts" / "qualify_runpod_pod.py",
        "requirements_train": project_root / "requirements-train.txt",
        "requirements_wandb": project_root / "requirements-wandb.txt",
    }
    for field, path in current_sources.items():
        current = _artifact(path, label=f"current {field}")
        if source[field] != current:
            raise RunAuthorityError(
                f"Qualified pod {field} differs from the current clean source"
            )

    argv = source["argv"]
    invoked_script = Path(argv[0])
    if not invoked_script.is_absolute():
        invoked_script = project_root / invoked_script
    try:
        same_script = invoked_script.resolve(strict=True) == current_sources[
            "qualification_script"
        ].resolve(strict=True)
    except OSError:
        same_script = False
    if not same_script:
        raise RunAuthorityError(
            "Pod qualification argv did not invoke the bound qualification script"
        )

    package_arguments = [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--package-lock"
    ]
    package_arguments.extend(
        token.split("=", 1)[1]
        for token in argv
        if token.startswith("--package-lock=")
    )
    if len(package_arguments) != 1:
        raise RunAuthorityError(
            "Pod qualification argv must bind exactly one --package-lock"
        )
    try:
        argv_lock = Path(package_arguments[0]).resolve(strict=True)
        environment_lock = Path(environment["lock"]["path"]).resolve(strict=True)
    except (KeyError, OSError, TypeError) as exc:
        raise RunAuthorityError("Cannot resolve the qualified package-lock path") from exc
    if argv_lock != environment_lock:
        raise RunAuthorityError(
            "Pod qualification argv used another package lock"
        )


def _inspect_hardware_contract(
    path: str | Path,
    *,
    expected_format: str,
    expected_status: str,
    label: str,
) -> dict[str, Any]:
    contract, descriptor = _json_object(path, label=label)
    bound = _verified_sidecar(path, label=label)
    if bound["artifact"] != descriptor:
        raise RunAuthorityError("Hardware contract changed during inspection")
    if contract.get("format") != expected_format or contract.get("format_version") != 1:
        raise RunAuthorityError("Unsupported six-GPU hardware contract format")
    required = {
        "status",
        "topology",
        "world_size",
        "gpu_count",
        "gpu_model",
        "gpu_memory_bytes",
        "compute_capability",
        "multiprocessor_count",
        "driver_version",
        "cuda_runtime_version",
        "cudnn_version",
        "nccl_version",
        "torch_version",
        "bf16_supported",
        "distributed_strategy",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RunAuthorityError(f"Hardware contract is missing fields: {missing}")
    if contract["status"] != expected_status or contract["topology"] != "single-node":
        raise RunAuthorityError(
            f"Hardware contract must be a {expected_status} single-node contract"
        )
    if contract["world_size"] != WORLD_SIZE or contract["gpu_count"] != WORLD_SIZE:
        raise RunAuthorityError("Hardware contract must declare exactly six visible GPUs/ranks")
    if contract["bf16_supported"] is not True:
        raise RunAuthorityError("All expected GPUs must support BF16")
    if contract["distributed_strategy"] != "ddp":
        raise RunAuthorityError("Current production launcher requires distributed_strategy=ddp")
    if not _plain_int(contract["gpu_memory_bytes"], minimum=1):
        raise RunAuthorityError("Hardware gpu_memory_bytes must be positive")
    if not _plain_int(contract["multiprocessor_count"], minimum=1):
        raise RunAuthorityError("Hardware multiprocessor_count must be positive")
    capability = contract["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(not _plain_int(item) for item in capability)
    ):
        raise RunAuthorityError("Hardware compute_capability must be [major, minor]")
    for field in (
        "gpu_model",
        "driver_version",
        "cuda_runtime_version",
        "cudnn_version",
        "nccl_version",
        "torch_version",
    ):
        if not isinstance(contract[field], str) or not contract[field].strip():
            raise RunAuthorityError(f"Hardware {field} must be an exact non-empty string")
    created_utc = contract.get("created_utc")
    try:
        created = datetime.fromisoformat(created_utc)
    except (TypeError, ValueError) as exc:
        raise RunAuthorityError("Qualified hardware created_utc must be ISO-8601") from exc
    if created.tzinfo is None:
        raise RunAuthorityError("Qualified hardware created_utc must be timezone-aware")
    _validate_pod_qualification(contract)
    admission_policy = contract["qualification"]["host"].get("admission_policy")
    if expected_format == HARDWARE_FORMAT and admission_policy is not None:
        if (
            not isinstance(admission_policy, dict)
            or admission_policy.get("scope") != "final-launch-strict"
            or admission_policy.get("allow_overlay_local_storage") is not False
            or admission_policy.get("allow_bounded_memlock") is not False
            or admission_policy.get("final_launch_authorized") is not True
        ):
            raise RunAuthorityError(
                "Geometry-only storage/memlock exceptions cannot authorize final launch"
            )
    if expected_format == PROVISIONAL_HARDWARE_FORMAT and admission_policy is not None:
        if (
            not isinstance(admission_policy, dict)
            or admission_policy.get("scope") != "geometry-only-provisional"
            or admission_policy.get("final_launch_authorized") is not False
        ):
            raise RunAuthorityError(
                "Provisional hardware admission policy is not geometry-only"
            )
    return {
        "contract": bound["artifact"],
        "sidecar": bound["sidecar"],
        "expected": contract,
    }


def inspect_hardware_contract(path: str | Path) -> dict[str, Any]:
    """Inspect only the final launch-authorizing six-GPU contract."""

    return _inspect_hardware_contract(
        path,
        expected_format=HARDWARE_FORMAT,
        expected_status="accepted",
        label="six-GPU hardware contract",
    )


def inspect_provisional_hardware_contract(path: str | Path) -> dict[str, Any]:
    """Inspect pod evidence usable for geometry work, never full training.

    The distinct format and ``provisional`` status are deliberate.  The
    production run-authority path calls :func:`inspect_hardware_contract`, so
    this artifact cannot authorize a launch even though it proves the same
    six-GPU/NCCL/package/runtime invariants.
    """

    return _inspect_hardware_contract(
        path,
        expected_format=PROVISIONAL_HARDWARE_FORMAT,
        expected_status="provisional",
        label="provisional six-GPU hardware contract",
    )


def _stable_tree_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"entries", "sha256"}:
        raise RunAuthorityError(f"{label} stable-tree identity is incomplete")
    if not _plain_int(value["entries"], minimum=1):
        raise RunAuthorityError(f"{label} stable-tree entry count must be positive")
    _require_sha256(value["sha256"], field=f"{label} stable-tree sha256")
    return dict(value)


def _current_tree_identity(root: Path, *, label: str) -> dict[str, Any]:
    """Recompute the corpus qualifier's mutation-sensitive metadata digest."""

    root = _directory(root, label=f"{label} root")
    inventory: dict[str, dict[str, Any]] = {}

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
        inventory["."] = descriptor(root)
        paths = sorted(
            root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
        )
        for path in paths:
            inventory[path.relative_to(root).as_posix()] = descriptor(path)
    except OSError as exc:
        raise RunAuthorityError(f"Cannot inventory qualified {label} tree: {exc}") from exc
    return {
        "entries": len(inventory),
        "sha256": canonical_sha256(inventory),
    }


def _validate_corpus_validator(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"sources", "runtime"}:
        raise RunAuthorityError("Corpus qualification validator identity is incomplete")
    sources = value["sources"]
    runtime = value["runtime"]
    expected_source_names = {
        "qualification",
        "pretrain_data",
        "materialize_contract",
        "tokenizer_identity",
        "model",
        "model_config_loader",
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_names:
        raise RunAuthorityError("Corpus qualification validator sources are incomplete")
    recorded_sources = {
        name: _recorded_artifact(descriptor, label=f"corpus validator {name}")
        for name, descriptor in sources.items()
    }
    project_root = Path(__file__).resolve().parents[1]
    current_paths = {
        "qualification": project_root / "scripts" / "qualify_training_corpus.py",
        "pretrain_data": project_root / "pretrain" / "data.py",
        "materialize_contract": project_root / "pretrain" / "materialize.py",
        "tokenizer_identity": project_root / "pretrain" / "tokenizer_identity.py",
        "model": project_root / "pretrain" / "model.py",
        "model_config_loader": project_root / "pretrain" / "hf_export.py",
    }
    for name, current_path in current_paths.items():
        current = _artifact(current_path, label=f"current corpus validator {name}")
        recorded = recorded_sources[name]
        if (
            recorded["bytes"] != current["bytes"]
            or recorded["sha256"] != current["sha256"]
        ):
            raise RunAuthorityError(
                f"Corpus qualification validator source changed: {name}"
            )
    expected_runtime_fields = {
        "python_executable",
        "python_version",
        "numpy_version",
        "torch_version",
        "sqlite_version",
        "tokenizers_version",
        "zstandard_version",
        "packed_format_version",
        "order_format_version",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime_fields:
        raise RunAuthorityError("Corpus qualification validator runtime is incomplete")
    for field in expected_runtime_fields - {
        "packed_format_version",
        "order_format_version",
    }:
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise RunAuthorityError(
                f"Corpus qualification validator runtime {field} is invalid"
            )
    if runtime["packed_format_version"] != training_data.FORMAT_VERSION:
        raise RunAuthorityError("Corpus qualification used another packed-data format")
    if runtime["order_format_version"] != training_data.ORDER_FORMAT_VERSION:
        raise RunAuthorityError("Corpus qualification used another order format")
    return value


def inspect_corpus_qualification(path: str | Path) -> dict[str, Any]:
    """Authenticate and validate the final materialized-corpus qualification."""

    receipt, descriptor = _json_object(path, label="training-corpus qualification")
    bound = _verified_sidecar(path, label="training-corpus qualification")
    if bound["artifact"] != descriptor:
        raise RunAuthorityError("Training-corpus qualification changed during inspection")
    if (
        receipt.get("format") != CORPUS_QUALIFICATION_FORMAT
        or receipt.get("format_version") != CORPUS_QUALIFICATION_VERSION
        or receipt.get("status") != "pass"
    ):
        raise RunAuthorityError("Training-corpus qualification is unsupported or did not pass")
    receipt_path = Path(descriptor["path"])
    if receipt_path.name != "qualification.json":
        raise RunAuthorityError("Training-corpus qualification must be qualification.json")
    try:
        generation_entries = sorted(entry.name for entry in receipt_path.parent.iterdir())
    except OSError as exc:
        raise RunAuthorityError(
            f"Cannot inventory training-corpus qualification generation: {exc}"
        ) from exc
    if generation_entries != ["qualification.json", "qualification.json.sha256"]:
        raise RunAuthorityError(
            "Training-corpus qualification generation must contain exactly its receipt pair"
        )
    expected_receipt_fields = {
        "format",
        "format_version",
        "status",
        "started_utc",
        "completed_utc",
        "elapsed_seconds",
        "corpus",
        "acceptance_policy",
        "checks",
        "model",
        "validator",
        "tokenizer",
        "common_data_contract",
        "provenance",
        "splits",
        "document_indexes",
        "split_identity_audit",
    }
    if set(receipt) != expected_receipt_fields:
        raise RunAuthorityError("Training-corpus qualification receipt schema is not exact")
    timestamps: dict[str, datetime] = {}
    for field in ("started_utc", "completed_utc"):
        try:
            timestamp = datetime.fromisoformat(receipt[field])
        except (TypeError, ValueError) as exc:
            raise RunAuthorityError(
                f"Training-corpus qualification {field} must be ISO-8601"
            ) from exc
        if timestamp.tzinfo is None:
            raise RunAuthorityError(
                f"Training-corpus qualification {field} must be timezone-aware"
            )
        timestamps[field] = timestamp
    if timestamps["completed_utc"] < timestamps["started_utc"]:
        raise RunAuthorityError("Training-corpus qualification timestamps are reversed")
    elapsed = receipt["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise RunAuthorityError("Training-corpus qualification elapsed time is invalid")
    try:
        elapsed_decimal = Decimal(str(elapsed))
    except InvalidOperation as exc:
        raise RunAuthorityError(
            "Training-corpus qualification elapsed time is invalid"
        ) from exc
    if not elapsed_decimal.is_finite() or elapsed_decimal < 0:
        raise RunAuthorityError("Training-corpus qualification elapsed time is invalid")

    checks = receipt.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != CORPUS_QUALIFICATION_CHECKS
        or any(value is not True for value in checks.values())
    ):
        raise RunAuthorityError("Training-corpus qualification checks are incomplete")
    policy = receipt.get("acceptance_policy")
    expected_policy_fields = {
        "world_size",
        "expected_input_tokens",
        "maximum_target_shortfall_fraction",
        "expected_input_token_weights",
        "mixture_absolute_tolerance",
        "sample_rows_per_domain",
        "sample_seed",
        "full_packed_payload_checksums",
        "full_packed_semantic_scan",
        "full_order_uniqueness_scan",
        "full_document_index_scan",
        "exact_disk_backed_split_identity_scan",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != expected_policy_fields
        or policy.get("world_size") != WORLD_SIZE
    ):
        raise RunAuthorityError("Corpus qualification must target exactly six ranks")
    if policy.get("expected_input_tokens") != EXPECTED_INPUT_TOKEN_TARGETS:
        raise RunAuthorityError("Corpus qualification token targets differ from production")
    if policy.get("expected_input_token_weights") != EXPECTED_INPUT_TOKEN_WEIGHTS:
        raise RunAuthorityError("Corpus qualification mixture differs from 40/40/20")
    try:
        shortfall = Decimal(str(policy["maximum_target_shortfall_fraction"]))
        mixture_tolerance = Decimal(str(policy["mixture_absolute_tolerance"]))
    except (InvalidOperation, ValueError) as exc:
        raise RunAuthorityError("Corpus qualification tolerances are invalid") from exc
    if (
        not shortfall.is_finite()
        or shortfall < 0
        or shortfall > MAXIMUM_TARGET_SHORTFALL_FRACTION
    ):
        raise RunAuthorityError("Corpus qualification target-shortfall tolerance is too loose")
    if (
        not mixture_tolerance.is_finite()
        or mixture_tolerance < 0
        or mixture_tolerance > MAXIMUM_MIXTURE_ABSOLUTE_TOLERANCE
    ):
        raise RunAuthorityError("Corpus qualification mixture tolerance is too loose")
    if not _plain_int(
        policy["sample_rows_per_domain"], minimum=MINIMUM_SAMPLE_ROWS_PER_DOMAIN
    ) or not _plain_int(policy["sample_seed"]):
        raise RunAuthorityError("Corpus qualification sampling policy is insufficient")
    for field in (
        "full_packed_payload_checksums",
        "full_packed_semantic_scan",
        "full_order_uniqueness_scan",
        "full_document_index_scan",
        "exact_disk_backed_split_identity_scan",
    ):
        if policy[field] is not True:
            raise RunAuthorityError(f"Corpus qualification disabled required check: {field}")

    corpus = receipt.get("corpus")
    if not isinstance(corpus, dict):
        raise RunAuthorityError("Corpus qualification lacks corpus identity")
    manifest_path_raw = corpus.get("manifest_path")
    sidecar_path_raw = corpus.get("sidecar_path")
    if not isinstance(manifest_path_raw, str) or not Path(manifest_path_raw).is_absolute():
        raise RunAuthorityError("Qualified corpus manifest path must be absolute")
    if not isinstance(sidecar_path_raw, str) or not Path(sidecar_path_raw).is_absolute():
        raise RunAuthorityError("Qualified corpus sidecar path must be absolute")
    manifest = _artifact(manifest_path_raw, label="qualified corpus manifest")
    if Path(manifest["path"]).name != "manifest.json":
        raise RunAuthorityError("Qualified corpus manifest must be named manifest.json")
    if (
        manifest["bytes"] != corpus.get("manifest_bytes")
        or manifest["sha256"] != corpus.get("manifest_sha256")
    ):
        raise RunAuthorityError("Qualified corpus manifest identity changed")
    sidecar = _artifact(sidecar_path_raw, label="qualified corpus manifest sidecar")
    if sidecar["sha256"] != corpus.get("sidecar_sha256"):
        raise RunAuthorityError("Qualified corpus manifest sidecar identity changed")
    manifest_path = Path(manifest["path"])
    expected_corpus_sidecar = manifest_path.with_name("manifest.sha256")
    if Path(sidecar["path"]) != expected_corpus_sidecar:
        raise RunAuthorityError("Qualified corpus sidecar is not adjacent to manifest.json")
    expected_sidecar_bytes = f"{manifest['sha256']}  manifest.json\n".encode("ascii")
    if _read_bound_bytes(sidecar, label="qualified corpus manifest sidecar") != expected_sidecar_bytes:
        raise RunAuthorityError("Qualified corpus manifest sidecar is not exact")
    recorded_corpus_tree = _stable_tree_identity(
        corpus.get("stable_tree_inventory"), label="corpus"
    )
    current_corpus_tree = _current_tree_identity(manifest_path.parent, label="corpus")
    if current_corpus_tree != recorded_corpus_tree:
        raise RunAuthorityError("Qualified corpus tree identity changed")

    tokenizer = receipt.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise RunAuthorityError("Corpus qualification lacks tokenizer identity")
    tokenizer_root_raw = tokenizer.get("root")
    tokenizer_manifest_raw = tokenizer.get("manifest_path")
    if not isinstance(tokenizer_root_raw, str) or not Path(tokenizer_root_raw).is_absolute():
        raise RunAuthorityError("Qualified tokenizer root must be absolute")
    if not isinstance(tokenizer_manifest_raw, str) or not Path(
        tokenizer_manifest_raw
    ).is_absolute():
        raise RunAuthorityError("Qualified tokenizer manifest path must be absolute")
    tokenizer_root = _directory(tokenizer_root_raw, label="qualified tokenizer root")
    tokenizer_manifest = _artifact(
        tokenizer_manifest_raw, label="qualified tokenizer manifest"
    )
    if Path(tokenizer_manifest["path"]) != tokenizer_root / Path(
        tokenizer_manifest["path"]
    ).name:
        raise RunAuthorityError("Qualified tokenizer manifest is outside its root")
    if Path(tokenizer_manifest["path"]).name != "TOKENIZER_MANIFEST.json":
        raise RunAuthorityError(
            "Qualified tokenizer manifest must be TOKENIZER_MANIFEST.json"
        )
    if tokenizer_manifest["sha256"] != tokenizer.get("manifest_sha256"):
        raise RunAuthorityError("Qualified tokenizer manifest identity changed")
    _require_sha256(
        tokenizer.get("vocabulary_sha256"), field="qualified tokenizer vocabulary sha256"
    )
    if not _plain_int(tokenizer.get("vocab_size"), minimum=1):
        raise RunAuthorityError("Qualified tokenizer vocabulary size is invalid")
    if not _plain_int(tokenizer.get("eos_token_id")) or tokenizer["eos_token_id"] >= tokenizer[
        "vocab_size"
    ]:
        raise RunAuthorityError("Qualified tokenizer EOS token ID is invalid")
    recorded_tokenizer_tree = _stable_tree_identity(
        tokenizer.get("stable_tree_inventory"), label="tokenizer"
    )
    current_tokenizer_tree = _current_tree_identity(tokenizer_root, label="tokenizer")
    if current_tokenizer_tree != recorded_tokenizer_tree:
        raise RunAuthorityError("Qualified tokenizer tree identity changed")

    common = receipt.get("common_data_contract")
    required_common = {
        "sequence_length",
        "vocab_size",
        "eos_token_id",
        "tokenizer_manifest_sha256",
    }
    if not isinstance(common, dict) or set(common) != required_common:
        raise RunAuthorityError("Corpus qualification common data contract is incomplete")
    if (
        common["vocab_size"] != tokenizer["vocab_size"]
        or common["eos_token_id"] != tokenizer["eos_token_id"]
        or common["tokenizer_manifest_sha256"] != tokenizer["manifest_sha256"]
        or not _plain_int(common["sequence_length"], minimum=1)
    ):
        raise RunAuthorityError("Corpus and tokenizer qualification identities disagree")

    model = receipt.get("model")
    if not isinstance(model, dict) or set(model) != {"source", "sha256", "config"}:
        raise RunAuthorityError("Corpus qualification model identity is incomplete")
    if not isinstance(model["config"], dict) or not model["config"]:
        raise RunAuthorityError("Corpus qualification model config is invalid")
    if model["sha256"] is None:
        if model["source"] != "pretrain.model.ModelConfig defaults":
            raise RunAuthorityError("Unhashed qualified model source is unsupported")
    else:
        if not isinstance(model["source"], str) or not Path(model["source"]).is_absolute():
            raise RunAuthorityError("Qualified model config path must be absolute")
        _require_sha256(model["sha256"], field="qualified model config sha256")
        model_artifact = _artifact(model["source"], label="qualified model config")
        if model_artifact["sha256"] != model["sha256"]:
            raise RunAuthorityError("Qualified model config identity changed")
    _validate_corpus_validator(receipt.get("validator"))

    provenance = receipt.get("provenance")
    required_provenance = {"source", "policy", "tokenizer", "fingerprints", "bindings"}
    if not isinstance(provenance, dict) or not required_provenance.issubset(provenance):
        raise RunAuthorityError("Corpus qualification provenance evidence is incomplete")
    provenance_paths: set[Path] = set()
    for name, evidence in provenance.items():
        if name == "bindings":
            continue
        if not isinstance(evidence, dict) or set(evidence) != {"path", "bytes", "sha256"}:
            raise RunAuthorityError(f"Corpus qualification {name} provenance is invalid")
        relative = evidence["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _plain_int(evidence["bytes"], minimum=1)
        ):
            raise RunAuthorityError(f"Corpus qualification {name} provenance is invalid")
        _require_sha256(evidence["sha256"], field=f"qualified {name} provenance sha256")
        current = _artifact(
            manifest_path.parent / relative,
            label=f"qualified {name} provenance",
        )
        current_path = Path(current["path"])
        if not current_path.is_relative_to(manifest_path.parent):
            raise RunAuthorityError(
                f"Qualified {name} provenance escapes the corpus root"
            )
        if current_path in provenance_paths:
            raise RunAuthorityError("Corpus qualification repeats a provenance artifact")
        provenance_paths.add(current_path)
        if current["bytes"] != evidence["bytes"] or current["sha256"] != evidence["sha256"]:
            raise RunAuthorityError(f"Qualified {name} provenance identity changed")
    bindings = provenance["bindings"]
    required_bindings = {
        "selection_manifest_sha256",
        "tokenizer_manifest_sha256",
        "materialization_curation_policy_sha256",
        "source_curation_policy_sha256",
        "split_authority",
        "authenticated_leakage_audit",
        "raw_archive_count",
        "raw_archive_inventory_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != required_bindings:
        raise RunAuthorityError("Corpus qualification provenance bindings are incomplete")
    for field in (
        "selection_manifest_sha256",
        "tokenizer_manifest_sha256",
        "materialization_curation_policy_sha256",
        "source_curation_policy_sha256",
        "raw_archive_inventory_sha256",
    ):
        _require_sha256(bindings[field], field=f"qualified provenance {field}")
    if bindings["tokenizer_manifest_sha256"] != tokenizer["manifest_sha256"]:
        raise RunAuthorityError("Corpus provenance is bound to another tokenizer")
    if bindings["split_authority"] not in (
        "frozen_leakage_safe_source_groups",
        "completed-curation-decision-shards",
    ) or not _plain_int(bindings["raw_archive_count"], minimum=1):
        raise RunAuthorityError("Corpus qualification provenance authority is invalid")
    leakage = bindings["authenticated_leakage_audit"]
    required_zero = {
        "content_hashes_in_multiple_splits",
        "canonical_clusters_in_multiple_splits",
        "source_groups_in_multiple_splits",
        "cross_bucket_code_repo_groups_in_multiple_splits",
    }
    if not isinstance(leakage, dict) or not required_zero.issubset(leakage) or any(
        leakage[field] != 0 for field in required_zero
    ):
        raise RunAuthorityError("Corpus qualification provenance leakage audit failed")
    if leakage.get("normalized_hashes_in_multiple_splits", 0) != 0:
        raise RunAuthorityError("Corpus qualification normalized-hash leakage audit failed")

    splits = receipt.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise RunAuthorityError("Corpus qualification must cover train/validation/test")
    corpus_root = manifest_path.parent
    order_paths: set[Path] = set()
    for split, summary in splits.items():
        if not isinstance(summary, dict) or set(summary) != {
            "order",
            "packed",
            "deterministic_samples",
        }:
            raise RunAuthorityError(f"Corpus qualification {split} evidence is incomplete")
        for field in ("packed", "deterministic_samples"):
            if not isinstance(summary[field], dict) or set(summary[field]) != set(
                training_data.DOMAIN_ORDER
            ):
                raise RunAuthorityError(
                    f"Corpus qualification {split} {field} evidence is incomplete"
                )
        order = summary.get("order") if isinstance(summary, dict) else None
        if not isinstance(order, dict):
            raise RunAuthorityError(f"Corpus qualification lacks {split} order identity")
        order_relative = order.get("manifest")
        order_sha = order.get("manifest_sha256")
        if (
            not isinstance(order_relative, str)
            or Path(order_relative).is_absolute()
            or ".." in Path(order_relative).parts
        ):
            raise RunAuthorityError(f"Qualified {split} order path is invalid")
        _require_sha256(order_sha, field=f"qualified {split} order sha256")
        order_artifact = _artifact(
            corpus_root / order_relative, label=f"qualified {split} order manifest"
        )
        order_path = Path(order_artifact["path"])
        if not order_path.is_relative_to(corpus_root):
            raise RunAuthorityError(f"Qualified {split} order escapes the corpus root")
        if order_path in order_paths:
            raise RunAuthorityError("Corpus qualification repeats a split order")
        order_paths.add(order_path)
        if order_artifact["sha256"] != order_sha:
            raise RunAuthorityError(f"Qualified {split} order identity changed")
    document_indexes = receipt.get("document_indexes")
    if not isinstance(document_indexes, dict) or set(document_indexes) != set(splits):
        raise RunAuthorityError("Corpus qualification document-index evidence is incomplete")
    for split, indexes in document_indexes.items():
        if not isinstance(indexes, dict) or set(indexes) != set(training_data.DOMAIN_ORDER):
            raise RunAuthorityError(
                f"Corpus qualification {split} document-index evidence is incomplete"
            )
        if any(not isinstance(index, dict) or not index for index in indexes.values()):
            raise RunAuthorityError(
                f"Corpus qualification {split} document-index evidence is invalid"
            )
    split_audit = receipt.get("split_identity_audit")
    expected_split_audit_fields = {
        "method",
        "identity_kinds",
        "identity_occurrences",
        "unique_identities",
        "cross_split_collisions",
        "collision_examples",
        "scratch_database_bytes",
    }
    if not isinstance(split_audit, dict) or set(split_audit) != expected_split_audit_fields:
        raise RunAuthorityError("Corpus qualification split-identity audit is incomplete")
    if (
        split_audit["method"]
        != "exact-sqlite-without-rowid-split-bitmask-union"
        or split_audit["identity_kinds"] != ["source", "split_group"]
        or not _plain_int(split_audit["scratch_database_bytes"])
    ):
        raise RunAuthorityError("Corpus qualification split-identity audit is invalid")
    collisions = (
        split_audit.get("cross_split_collisions")
        if isinstance(split_audit, dict)
        else None
    )
    identity_kinds = {"source", "split_group"}
    if not isinstance(collisions, dict) or set(collisions) != identity_kinds or any(
        value != 0 for value in collisions.values()
    ):
        raise RunAuthorityError("Corpus qualification split-identity audit did not pass")
    examples = split_audit["collision_examples"]
    if (
        not isinstance(examples, dict)
        or set(examples) != identity_kinds
        or any(examples[kind] != [] for kind in identity_kinds)
    ):
        raise RunAuthorityError("Corpus qualification collision examples are inconsistent")
    occurrences = split_audit.get("identity_occurrences")
    unique = split_audit.get("unique_identities")
    if (
        not isinstance(occurrences, dict)
        or not isinstance(unique, dict)
        or set(occurrences) != identity_kinds
        or set(unique) != identity_kinds
        or any(not _plain_int(occurrences[kind], minimum=1) for kind in identity_kinds)
        or any(occurrences[kind] != unique[kind] for kind in identity_kinds)
    ):
        raise RunAuthorityError("Corpus qualification contains duplicate source/group identities")
    return {
        **bound,
        "receipt": receipt,
        "corpus_manifest": manifest,
        "corpus_sidecar": sidecar,
    }


def _validate_corpus_qualification_bindings(
    *,
    qualification: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    receipt = qualification["receipt"]
    corpus_root = Path(qualification["corpus_manifest"]["path"]).parent
    for split, order in (("train", train), ("validation", validation)):
        recorded = receipt["splits"][split]["order"]
        qualified_path = (corpus_root / recorded["manifest"]).resolve(strict=True)
        if qualified_path != Path(order["manifest"]["path"]):
            raise RunAuthorityError(
                f"Run authority {split} order is outside the qualified corpus"
            )
        if recorded["manifest_sha256"] != order["manifest"]["sha256"]:
            raise RunAuthorityError(
                f"Run authority {split} order differs from corpus qualification"
            )
    recorded_tokenizer = receipt["tokenizer"]
    tokenizer_fields = {
        "root": tokenizer["root"],
        "manifest_path": tokenizer["manifest"]["path"],
        "manifest_sha256": tokenizer["manifest_sha256"],
        "vocabulary_sha256": tokenizer["vocabulary_sha256"],
        "vocab_size": tokenizer["vocab_size"],
        "eos_token_id": tokenizer["eos_token_id"],
    }
    if any(recorded_tokenizer[field] != expected for field, expected in tokenizer_fields.items()):
        raise RunAuthorityError("Run-authority tokenizer differs from corpus qualification")
    common = receipt["common_data_contract"]
    expected_common = {
        "sequence_length": train["sequence_length"],
        "vocab_size": train["vocab_size"],
        "eos_token_id": train["eos_token_id"],
        "tokenizer_manifest_sha256": train["tokenizer_manifest_sha256"],
    }
    if common != expected_common:
        raise RunAuthorityError("Run-authority data contract differs from corpus qualification")
    qualified_model = dict(receipt["model"]["config"])
    authority_model = dict(model["config"])
    # Activation checkpointing is a runtime memory decision bound by recipe and
    # geometry; it does not change corpus/model shape compatibility.
    qualified_model.pop("activation_checkpointing", None)
    authority_model.pop("activation_checkpointing", None)
    if qualified_model != authority_model:
        raise RunAuthorityError("Run-authority model differs from corpus qualification")


def _decimal(value: Any, *, field: str, positive: bool = True) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise RunAuthorityError(f"{field} must be a decimal value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise RunAuthorityError(f"{field} is not a valid decimal") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise RunAuthorityError(f"{field} must be finite and positive")
    return result


def _decimal_string(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return "0" if rendered == "-0" else rendered


def inspect_geometry_receipt(
    path: str | Path,
    *,
    hardware_contract_sha256: str,
    measured_input_tokens_per_second: str,
) -> dict[str, Any]:
    receipt, descriptor = _json_object(path, label="accepted-geometry receipt")
    bound = _verified_sidecar(path, label="accepted-geometry receipt")
    if bound["artifact"] != descriptor:
        raise RunAuthorityError("Geometry receipt changed during inspection")
    if receipt.get("format") != GEOMETRY_FORMAT or receipt.get("format_version") != 1:
        raise RunAuthorityError("Unsupported accepted-geometry receipt format")
    if receipt.get("status") != "pass":
        raise RunAuthorityError("Accepted-geometry receipt did not pass")
    if receipt.get("hardware_contract_sha256") != hardware_contract_sha256:
        raise RunAuthorityError("Geometry receipt is bound to another hardware contract")
    accepted = receipt.get("accepted")
    measurements = receipt.get("measurements")
    if not isinstance(accepted, dict) or not isinstance(measurements, dict):
        raise RunAuthorityError("Geometry receipt lacks accepted/measurements objects")
    for field in (
        "global_microbatch_rows",
        "gradient_accumulation_steps",
        "workers",
        "overfit_batch_rows",
    ):
        minimum = 0 if field == "workers" else 1
        if not _plain_int(accepted.get(field), minimum=minimum):
            raise RunAuthorityError(f"Geometry accepted.{field} is invalid")
    if accepted["global_microbatch_rows"] % WORLD_SIZE:
        raise RunAuthorityError("Global microbatch rows must be divisible across six ranks")
    for field in ("compile_model", "activation_checkpointing"):
        if not isinstance(accepted.get(field), bool):
            raise RunAuthorityError(f"Geometry accepted.{field} must be boolean")
    if accepted.get("precision") != "bfloat16" or accepted.get("parameter_dtype") != "float32":
        raise RunAuthorityError("Geometry must qualify BF16 compute with FP32 parameters")
    measured = _decimal(
        measurements.get("aggregate_input_tokens_per_second"),
        field="geometry aggregate_input_tokens_per_second",
    )
    requested = _decimal(
        measured_input_tokens_per_second,
        field="measured input tokens per second",
    )
    if measured != requested:
        raise RunAuthorityError(
            "Explicit measured throughput differs from the qualification receipt"
        )
    for field in (
        "peak_memory_allocated_bytes_per_gpu",
        "peak_memory_reserved_bytes_per_gpu",
        "minimum_free_memory_bytes_per_gpu",
        "soak_steps",
    ):
        if not _plain_int(measurements.get(field), minimum=1):
            raise RunAuthorityError(f"Geometry measurements.{field} is invalid")
    for field in ("checkpoint_seconds", "data_wait_fraction", "scaling_efficiency"):
        value = _decimal(
            measurements.get(field),
            field=f"geometry measurements.{field}",
            positive=field != "data_wait_fraction",
        )
        if value < 0 or (field in ("data_wait_fraction", "scaling_efficiency") and value > 1):
            raise RunAuthorityError(f"Geometry measurements.{field} cannot exceed 1")
    return {**bound, "receipt": receipt}


def inspect_order(path: str | Path, *, expected_split: str) -> dict[str, Any]:
    manifest_path = _regular_file(path, label=f"{expected_split} order manifest")
    try:
        manifest, order_path = training_data._load_order_manifest(  # type: ignore[attr-defined]
            manifest_path, verify_checksum=True
        )
    except (OSError, ValueError) as exc:
        raise RunAuthorityError(f"Invalid {expected_split} order: {exc}") from exc
    if manifest["split"] != expected_split:
        raise RunAuthorityError(
            f"Expected {expected_split!r} order, found {manifest['split']!r}"
        )
    manifest_descriptor = _artifact(manifest_path, label=f"{expected_split} order manifest")
    payload_descriptor = _artifact(order_path, label=f"{expected_split} order payload")
    if manifest["order"]["sha256"] != payload_descriptor["sha256"]:
        raise RunAuthorityError(f"{expected_split} order checksum changed during inspection")
    if expected_split == "train":
        try:
            geometry = training_data._frozen_training_geometry_from_manifest(manifest)  # type: ignore[attr-defined]
        except ValueError as exc:
            raise RunAuthorityError(f"Training order geometry is not frozen: {exc}") from exc
    else:
        geometry = None
    return {
        "manifest": manifest_descriptor,
        "payload": payload_descriptor,
        "split": manifest["split"],
        "sequence_length": manifest["sequence_length"],
        "vocab_size": manifest["vocab_size"],
        "eos_token_id": manifest["eos_token_id"],
        "tokenizer_manifest_sha256": manifest["tokenizer_manifest_sha256"],
        "shuffle_seed": manifest["seed"],
        "shuffle_rng": manifest["rng"],
        "rows": manifest["rows"],
        "geometry": geometry,
    }


def _validate_certified_inventory(
    inventory: Mapping[str, Any], *, order: Mapping[str, Any]
) -> None:
    order_manifest = inventory.get("order_manifest")
    order_payload = inventory.get("order_payload")
    packed_files = inventory.get("packed_files")
    if (
        not isinstance(order_manifest, dict)
        or not isinstance(order_payload, dict)
        or not isinstance(packed_files, list)
        or not packed_files
    ):
        raise RunAuthorityError("Certification metadata inventory has an invalid shape")
    records = [order_manifest, order_payload, *packed_files]
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RunAuthorityError("Certification metadata inventory record is invalid")
        path_raw = record.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            raise RunAuthorityError("Certification inventory record lacks a path")
        path = _regular_file(path_raw, label="certified data artifact")
        if str(path) in seen:
            raise RunAuthorityError("Certification metadata inventory repeats a path")
        seen.add(str(path))
        metadata = path.stat()
        expected_identity = {
            "bytes": metadata.st_size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }
        for field, actual in expected_identity.items():
            if record.get(field) != actual:
                raise RunAuthorityError(
                    f"Certified {record.get('kind', 'data')} identity changed: {path}"
                )
        declared = _require_sha256(
            record.get("declared_sha256"),
            field=f"certified digest for {path}",
        )
        actual_digest = record.get("actual_sha256")
        if actual_digest is not None:
            _require_sha256(actual_digest, field=f"certified actual digest for {path}")
            if actual_digest != declared or sha256_file(path) != actual_digest:
                raise RunAuthorityError(f"Certified checksum changed: {path}")
        final_metadata = path.stat()
        if any(
            record.get(field) != actual
            for field, actual in {
                "bytes": final_metadata.st_size,
                "device": final_metadata.st_dev,
                "inode": final_metadata.st_ino,
                "mtime_ns": final_metadata.st_mtime_ns,
                "ctime_ns": final_metadata.st_ctime_ns,
            }.items()
        ):
            raise RunAuthorityError(f"Certified data changed during inspection: {path}")
    if Path(order_manifest["path"]) != Path(order["manifest"]["path"]):
        raise RunAuthorityError("Certification inventory names another order manifest")
    if Path(order_payload["path"]) != Path(order["payload"]["path"]):
        raise RunAuthorityError("Certification inventory names another order payload")
    if order_manifest.get("actual_sha256") != order["manifest"]["sha256"]:
        raise RunAuthorityError("Certification inventory order-manifest digest mismatch")
    if order_payload.get("declared_sha256") != order["payload"]["sha256"]:
        raise RunAuthorityError("Certification inventory order-payload digest mismatch")


def inspect_certification(
    path: str | Path,
    *,
    expected_split: str,
    order: Mapping[str, Any],
    project_root: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    receipt, descriptor = _json_object(path, label=f"{expected_split} certification")
    bound = _verified_sidecar(path, label=f"{expected_split} certification")
    if descriptor != bound["artifact"]:
        raise RunAuthorityError("Certification changed during inspection")
    if (
        receipt.get("format") != CERTIFICATION_FORMAT
        or receipt.get("format_version") != 1
        or receipt.get("status") != "pass"
        or receipt.get("split") != expected_split
    ):
        raise RunAuthorityError(f"Unsupported or non-passing {expected_split} certification")
    try:
        receipt_order = Path(receipt["order_manifest"]).resolve(strict=True)
    except (KeyError, TypeError, OSError) as exc:
        raise RunAuthorityError("Certification has an invalid order_manifest path") from exc
    if receipt_order != Path(order["manifest"]["path"]):
        raise RunAuthorityError("Certification is bound to another order manifest")
    expected_checks = {
        "order_payload_checksum": True,
        "order_reference_semantics": True,
        "packed_manifest_identities": True,
        "packed_payload_checksums": True,
        "packed_payload_semantics": True,
        "stable_file_identity_during_validation": True,
    }
    if receipt.get("checks") != expected_checks:
        raise RunAuthorityError("Certification does not prove every full-data check")
    inventory = receipt.get("metadata_inventory")
    inventory_sha256 = receipt.get("metadata_inventory_sha256")
    if not isinstance(inventory, dict) or inventory_sha256 != canonical_sha256(inventory):
        raise RunAuthorityError("Certification metadata inventory digest is inconsistent")
    _validate_certified_inventory(inventory, order=order)
    summary = receipt.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("order_rows") != order["rows"]
        or summary.get("order_payload_sha256") != order["payload"]["sha256"]
    ):
        raise RunAuthorityError("Certification summary differs from the current order")
    validator = receipt.get("validator")
    if not isinstance(validator, dict):
        raise RunAuthorityError("Certification lacks validator identity")
    source_bindings = {
        "pretrain_data_sha256": project_root / "pretrain" / "data.py",
        "launcher_sha256": project_root / "scripts" / "launch_pretraining.py",
        "certifier_sha256": project_root / "scripts" / "certify_pretraining_data.py",
    }
    for field, source in source_bindings.items():
        if validator.get(field) != sha256_file(_regular_file(source, label=field)):
            raise RunAuthorityError(
                f"Certification validator source {field} differs from clean Git tree"
            )
    if validator.get("order_format_version") != training_data.ORDER_FORMAT_VERSION:
        raise RunAuthorityError("Certification used another order format implementation")
    if validator.get("packed_format_version") != training_data.FORMAT_VERSION:
        raise RunAuthorityError("Certification used another packed-data format implementation")
    if (
        validator.get("python_executable_sha256")
        != environment["python"]["executable_sha256"]
        or validator.get("python_version") != environment["python"]["version"]
        or str(validator.get("python_implementation", "")).lower()
        != environment["python"]["implementation"]
        or validator.get("numpy_version") != environment["numpy_version"]
    ):
        raise RunAuthorityError(
            "Certification runtime differs from the locked training environment"
        )
    completed_raw = receipt.get("completed_utc")
    if not isinstance(completed_raw, str):
        raise RunAuthorityError("Certification lacks a completion timestamp")
    try:
        completed = datetime.fromisoformat(completed_raw)
    except ValueError as exc:
        raise RunAuthorityError("Certification completion timestamp is invalid") from exc
    if completed.tzinfo is None or completed > datetime.now(timezone.utc).astimezone(completed.tzinfo):
        raise RunAuthorityError("Certification completion timestamp is naive or in the future")
    return {**bound, "receipt": receipt}


def inspect_recipe(path: str | Path) -> dict[str, Any]:
    recipe, descriptor = _json_object(path, label="training recipe")
    if recipe.get("format") != RECIPE_FORMAT or recipe.get("format_version") != 1:
        raise RunAuthorityError("Unsupported training-recipe format")
    if recipe.get("model_size") != MODEL_SIZE:
        raise RunAuthorityError("Training recipe must select the 1.3b model")
    if recipe.get("precision") != "bfloat16" or recipe.get("parameter_dtype") != "float32":
        raise RunAuthorityError("Training recipe must use BF16 compute and FP32 parameters")
    if recipe.get("deterministic") is not True:
        raise RunAuthorityError("Production recipe must request deterministic execution")
    seed = recipe.get("seed")
    if not _plain_int(seed, minimum=0):
        raise RunAuthorityError("Training seed must be a non-negative integer")
    optimizer = recipe.get("optimizer")
    schedule = recipe.get("schedule")
    cadence = recipe.get("cadence")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "AdamW":
        raise RunAuthorityError("Training optimizer must be AdamW")
    if not isinstance(optimizer.get("fused"), bool):
        raise RunAuthorityError("optimizer.fused must be an explicit boolean")
    if not isinstance(schedule, dict) or schedule.get("name") != "warmup-cosine":
        raise RunAuthorityError("Training schedule must be warmup-cosine")
    if not isinstance(cadence, dict):
        raise RunAuthorityError("Training recipe lacks cadence")
    learning_rate = _decimal(optimizer.get("learning_rate"), field="optimizer.learning_rate")
    weight_decay = _decimal(
        optimizer.get("weight_decay"), field="optimizer.weight_decay", positive=False
    )
    if weight_decay < 0:
        raise RunAuthorityError("optimizer.weight_decay cannot be negative")
    for field in ("beta1", "beta2"):
        beta = _decimal(optimizer.get(field), field=f"optimizer.{field}")
        if beta >= 1:
            raise RunAuthorityError(f"optimizer.{field} must be below 1")
    for field in ("epsilon", "max_grad_norm"):
        _decimal(optimizer.get(field), field=f"optimizer.{field}")
    minimum_learning_rate = _decimal(
        schedule.get("minimum_learning_rate"),
        field="schedule.minimum_learning_rate",
        positive=False,
    )
    if minimum_learning_rate < 0 or minimum_learning_rate > learning_rate:
        raise RunAuthorityError(
            "schedule.minimum_learning_rate must be between zero and learning rate"
        )
    if not _plain_int(schedule.get("warmup_steps"), minimum=0):
        raise RunAuthorityError("schedule.warmup_steps must be non-negative")
    for field in ("checkpoint_every", "eval_every", "eval_batches", "log_every"):
        if not _plain_int(cadence.get(field), minimum=1):
            raise RunAuthorityError(f"cadence.{field} must be positive")
    if cadence.get("eval_at_start") is not True:
        raise RunAuthorityError("Production recipe must evaluate at start")
    if recipe.get("wandb_mode") not in ("offline", "online"):
        raise RunAuthorityError("Production recipe must enable offline or online W&B")
    if recipe.get("activation_checkpointing") is not True:
        raise RunAuthorityError(
            "The production 1.3B DDP recipe requires activation_checkpointing=true"
        )
    if not isinstance(recipe.get("compile_model"), bool):
        raise RunAuthorityError("compile_model must be boolean")
    return {"artifact": descriptor, "recipe": recipe}


def _option(argv: Sequence[str], name: str, *, required: bool = True) -> str | None:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1] == "--":
                raise RunAuthorityError(f"Canonical launcher argv lacks a value for {name}")
            values.append(argv[index + 1])
        elif token.startswith(f"{name}="):
            values.append(token[len(name) + 1 :])
    if len(values) > 1:
        raise RunAuthorityError(f"Canonical launcher argv repeats {name}")
    if required and not values:
        raise RunAuthorityError(f"Canonical launcher argv is missing {name}")
    return values[0] if values else None


def _flag(argv: Sequence[str], name: str) -> bool:
    count = sum(token == name for token in argv)
    if count > 1:
        raise RunAuthorityError(f"Canonical launcher argv repeats {name}")
    return count == 1


def _same_path(value: str | None, expected: str | Path, *, project_root: Path) -> bool:
    if value is None:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    try:
        return path.resolve(strict=True) == Path(expected).resolve(strict=True)
    except OSError:
        return False


def inspect_launcher_argv(
    path: str | Path,
    *,
    project_root: Path,
    train_order: Mapping[str, Any],
    validation_order: Mapping[str, Any],
    tokenizer_root: Path,
    train_certification: Mapping[str, Any],
    validation_certification: Mapping[str, Any],
    recipe: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = _artifact(path, label="canonical launcher argv")
    try:
        argv = json.loads(
            _read_bound_bytes(descriptor, label="canonical launcher argv").decode(
                "utf-8", errors="strict"
            )
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunAuthorityError(f"Invalid canonical launcher argv JSON: {exc}") from exc
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
    ):
        raise RunAuthorityError("Canonical launcher argv must be a non-empty JSON string array")
    executable = Path(argv[0]).resolve()
    if executable != Path(sys.executable).resolve():
        raise RunAuthorityError("Canonical launcher argv must use the locked Python executable")
    script_tokens = [item for item in argv[1:4] if item.endswith("launch_pretraining.py")]
    if len(script_tokens) != 1:
        raise RunAuthorityError("Canonical argv must execute scripts/launch_pretraining.py")
    script = Path(script_tokens[0])
    if not script.is_absolute():
        script = project_root / script
    expected_script = project_root / "scripts" / "launch_pretraining.py"
    if script.resolve(strict=True) != expected_script.resolve(strict=True):
        raise RunAuthorityError("Canonical argv references a launcher outside the clean Git tree")
    if not _flag(argv, "--execute") or _flag(argv, "--dry-run"):
        raise RunAuthorityError("Canonical argv must select --execute and must not select --dry-run")
    if argv.count("--") != 1:
        raise RunAuthorityError(
            "Canonical argv must contain one -- delimiter before trainer arguments"
        )
    delimiter = argv.index("--")
    launcher_argv = argv[:delimiter]
    trainer_argv = argv[delimiter + 1 :]
    if not _flag(launcher_argv, "--execute") or _flag(launcher_argv, "--dry-run"):
        raise RunAuthorityError("Launch selection must appear before the -- delimiter")
    accepted = geometry["receipt"]["accepted"]
    cadence = recipe["cadence"]
    path_options = {
        "--train-order-manifest": train_order["manifest"]["path"],
        "--validation-order-manifest": validation_order["manifest"]["path"],
        "--tokenizer": tokenizer_root,
        "--train-data-evidence": train_certification["artifact"]["path"],
        "--validation-data-evidence": validation_certification["artifact"]["path"],
    }
    for name, expected in path_options.items():
        if not _same_path(_option(launcher_argv, name), expected, project_root=project_root):
            raise RunAuthorityError(f"Canonical argv {name} is not bound to the authorized input")
    authority_destination = Path(_option(launcher_argv, "--run-authority"))
    if not authority_destination.is_absolute():
        raise RunAuthorityError("Canonical argv --run-authority must be absolute")
    authority_destination = authority_destination.resolve(strict=False)
    if authority_destination == project_root or authority_destination.is_relative_to(
        project_root
    ):
        raise RunAuthorityError(
            "Canonical argv --run-authority must be outside the Git worktree"
        )
    scalar_options = {
        "--nproc-per-node": str(WORLD_SIZE),
        "--model-size": MODEL_SIZE,
        "--workers": str(accepted["workers"]),
        "--checkpoint-every": str(cadence["checkpoint_every"]),
        "--eval-every": str(cadence["eval_every"]),
        "--eval-batches": str(cadence["eval_batches"]),
        "--wandb-mode": str(recipe["wandb_mode"]),
    }
    for name, expected in scalar_options.items():
        if _option(launcher_argv, name) != expected:
            raise RunAuthorityError(f"Canonical argv {name} must equal {expected}")
    if not _flag(launcher_argv, "--eval-at-start") or _flag(
        launcher_argv, "--no-eval-at-start"
    ):
        raise RunAuthorityError("Canonical argv must explicitly select --eval-at-start")
    optimizer = recipe["optimizer"]
    schedule = recipe["schedule"]
    trainer_options = {
        "--learning-rate": optimizer["learning_rate"],
        "--weight-decay": optimizer["weight_decay"],
        "--beta1": optimizer["beta1"],
        "--beta2": optimizer["beta2"],
        "--adam-eps": optimizer["epsilon"],
        "--max-grad-norm": optimizer["max_grad_norm"],
        "--min-learning-rate": schedule["minimum_learning_rate"],
        "--warmup-steps": schedule["warmup_steps"],
        "--seed": recipe["seed"],
    }
    for name, expected in trainer_options.items():
        value = _option(trainer_argv, name)
        if name == "--seed" or name == "--warmup-steps":
            matches = value == str(expected)
        else:
            matches = value is not None and _decimal(value, field=name, positive=False) == _decimal(
                expected, field=f"recipe value for {name}", positive=False
            )
        if not matches:
            raise RunAuthorityError(f"Canonical argv {name} differs from the recipe")
    if _option(trainer_argv, "--log-every") != str(cadence["log_every"]):
        raise RunAuthorityError("Canonical argv --log-every differs from the recipe")
    # Activation checkpointing is owned by the production launcher because it
    # is part of the launcher's 1.3B memory-admission contract.  Requiring the
    # flag in trainer passthrough made the authority impossible to execute: the
    # launcher correctly rejects owned options after ``--``.  Freeze the
    # explicit launcher-level decision and reject a duplicate passthrough.
    if _flag(trainer_argv, "--activation-checkpointing") or _flag(
        trainer_argv, "--no-activation-checkpointing"
    ):
        raise RunAuthorityError(
            "Canonical argv must place activation checkpointing before the -- delimiter"
        )
    activation_on = _flag(launcher_argv, "--activation-checkpointing")
    activation_off = _flag(launcher_argv, "--no-activation-checkpointing")
    if activation_on == activation_off or activation_on != recipe["activation_checkpointing"]:
        raise RunAuthorityError("Canonical argv activation checkpointing differs from recipe")
    fused_on = _flag(trainer_argv, "--fused-adamw")
    fused_off = _flag(trainer_argv, "--no-fused-adamw")
    if fused_on == fused_off or fused_on != optimizer["fused"]:
        raise RunAuthorityError("Canonical argv fused AdamW decision differs from recipe")
    compile_flag = _flag(trainer_argv, "--compile")
    if compile_flag != recipe["compile_model"]:
        raise RunAuthorityError("Canonical argv compile decision differs from recipe")
    return {
        "argv_file": descriptor,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "launcher_source": _artifact(expected_script, label="launcher source"),
    }


def _validate_model(*, vocab_size: int, sequence_length: int, activation_checkpointing: bool) -> dict[str, Any]:
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=sequence_length,
        activation_checkpointing=activation_checkpointing,
    )
    parameters = config.expected_parameter_count
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise RunAuthorityError(
            f"1.3B architecture parameter count changed: {parameters:,} != "
            f"{EXPECTED_PARAMETER_COUNT:,}"
        )
    return {
        "model_size": MODEL_SIZE,
        "config": dataclasses.asdict(config),
        "expected_parameter_count": parameters,
    }


def _economics(
    *,
    consumed_input_tokens: int,
    measured_input_tokens_per_second: str,
    hourly_cost_usd: str,
    total_cost_cap_usd: str,
) -> dict[str, Any]:
    if not _plain_int(consumed_input_tokens, minimum=1):
        raise RunAuthorityError("Training order lacks a positive consumed-token count")
    throughput = _decimal(measured_input_tokens_per_second, field="measured throughput")
    hourly = _decimal(hourly_cost_usd, field="hourly cost")
    cap = _decimal(total_cost_cap_usd, field="total cost cap")
    duration_seconds = int(
        (Decimal(consumed_input_tokens) / throughput).to_integral_value(rounding=ROUND_CEILING)
    )
    duration_hours = Decimal(duration_seconds) / Decimal(3600)
    projected_cost = (duration_hours * hourly).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if projected_cost > cap:
        raise RunAuthorityError(
            f"Projected run cost ${projected_cost} exceeds authorized cap ${cap}"
        )
    return {
        "currency": "USD",
        "measured_aggregate_input_tokens_per_second": _decimal_string(throughput),
        "consumed_input_tokens": consumed_input_tokens,
        "projected_duration_seconds": duration_seconds,
        "projected_duration_hours": _decimal_string(duration_hours),
        "hourly_six_gpu_pod_cost_usd": _decimal_string(hourly),
        "projected_total_cost_usd": _decimal_string(projected_cost),
        "total_cost_cap_usd": _decimal_string(cap),
    }


def collect_run_authority(
    *,
    project_root: str | Path,
    package_lock: str | Path,
    container_image_digest: str,
    hardware_contract: str | Path,
    geometry_receipt: str | Path,
    corpus_qualification: str | Path,
    train_order_manifest: str | Path,
    validation_order_manifest: str | Path,
    train_certification: str | Path,
    validation_certification: str | Path,
    tokenizer_root: str | Path,
    training_recipe: str | Path,
    launcher_argv_json: str | Path,
    measured_input_tokens_per_second: str,
    hourly_cost_usd: str,
    total_cost_cap_usd: str,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Inspect every launch input and return a canonical authority payload."""

    if not isinstance(container_image_digest, str) or not container_image_digest.startswith("sha256:"):
        raise RunAuthorityError("Container image must be supplied as sha256:<digest>")
    _require_sha256(container_image_digest[7:], field="container image digest")
    git = inspect_clean_git(project_root)
    root = Path(git["project_root"])
    environment = inspect_package_lock(package_lock)
    environment["container_image_digest"] = container_image_digest
    hardware = inspect_hardware_contract(hardware_contract)
    _validate_qualification_provenance(
        hardware=hardware,
        environment=environment,
        git=git,
        project_root=root,
    )
    if hardware["expected"]["torch_version"] != environment["torch_version"]:
        raise RunAuthorityError(
            "Hardware contract torch_version differs from the locked installed package"
        )
    geometry = inspect_geometry_receipt(
        geometry_receipt,
        hardware_contract_sha256=hardware["contract"]["sha256"],
        measured_input_tokens_per_second=measured_input_tokens_per_second,
    )
    train = inspect_order(train_order_manifest, expected_split="train")
    validation = inspect_order(validation_order_manifest, expected_split="validation")
    corpus_gate = inspect_corpus_qualification(corpus_qualification)
    if Path(train["manifest"]["path"]) == Path(validation["manifest"]["path"]):
        raise RunAuthorityError("Training and validation orders must be distinct")
    for field in ("sequence_length", "vocab_size", "eos_token_id", "tokenizer_manifest_sha256"):
        if train[field] != validation[field]:
            raise RunAuthorityError(f"Train/validation orders disagree on {field}")
    geometry_receipt_payload = geometry["receipt"]
    if geometry_receipt_payload.get("train_order_manifest_sha256") != train["manifest"]["sha256"]:
        raise RunAuthorityError("Accepted geometry was measured against another training order")
    if (
        geometry_receipt_payload.get("validation_order_manifest_sha256")
        != validation["manifest"]["sha256"]
    ):
        raise RunAuthorityError("Accepted geometry was measured against another validation order")
    accepted = geometry["receipt"]["accepted"]
    train_geometry = train["geometry"]
    assert isinstance(train_geometry, dict)
    if train_geometry["global_microbatch_rows"] != accepted["global_microbatch_rows"]:
        raise RunAuthorityError("Accepted global microbatch differs from frozen training order")
    if train_geometry["gradient_accumulation_steps"] != accepted["gradient_accumulation_steps"]:
        raise RunAuthorityError("Accepted accumulation differs from frozen training order")
    if train_geometry["global_microbatch_rows"] % WORLD_SIZE:
        raise RunAuthorityError("Frozen global microbatch cannot be divided across six ranks")
    train_cert = inspect_certification(
        train_certification,
        expected_split="train",
        order=train,
        project_root=root,
        environment=environment,
    )
    validation_cert = inspect_certification(
        validation_certification,
        expected_split="validation",
        order=validation,
        project_root=root,
        environment=environment,
    )
    tokenizer_path = _directory(tokenizer_root, label="tokenizer root")
    tokenizer = verify_tokenizer_identity(
        tokenizer_path,
        expected_manifest_sha256=train["tokenizer_manifest_sha256"],
        expected_vocab_size=train["vocab_size"],
    )
    tokenizer_payload = {
        "root": str(tokenizer_path),
        "manifest": _artifact(tokenizer.manifest_path, label="tokenizer manifest"),
        "manifest_sha256": tokenizer.manifest_sha256,
        "vocabulary_sha256": tokenizer.vocabulary_sha256,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": train["eos_token_id"],
    }
    if tokenizer_payload["manifest"]["sha256"] != tokenizer.manifest_sha256:
        raise RunAuthorityError("Tokenizer manifest changed during identity verification")
    recipe = inspect_recipe(training_recipe)
    recipe_payload = recipe["recipe"]
    if recipe_payload["activation_checkpointing"] != accepted["activation_checkpointing"]:
        raise RunAuthorityError("Recipe activation checkpointing differs from accepted geometry")
    if recipe_payload["compile_model"] != accepted["compile_model"]:
        raise RunAuthorityError("Recipe compile decision differs from accepted geometry")
    optimizer_updates = train_geometry["optimizer_updates"]
    if not _plain_int(optimizer_updates, minimum=1):
        raise RunAuthorityError("Training order lacks optimizer updates")
    cadence = recipe_payload["cadence"]
    if cadence["checkpoint_every"] > optimizer_updates or cadence["eval_every"] > optimizer_updates:
        raise RunAuthorityError("Checkpoint/eval cadence exceeds the complete training run")
    if recipe_payload["schedule"]["warmup_steps"] >= optimizer_updates:
        raise RunAuthorityError("Warmup must end before the final optimizer update")
    validation_microbatches = validation["rows"] // train_geometry["global_microbatch_rows"]
    if cadence["eval_batches"] > validation_microbatches:
        raise RunAuthorityError("Requested eval batches exceed the held-out complete microbatches")
    model = _validate_model(
        vocab_size=train["vocab_size"],
        sequence_length=train["sequence_length"],
        activation_checkpointing=recipe_payload["activation_checkpointing"],
    )
    _validate_corpus_qualification_bindings(
        qualification=corpus_gate,
        train=train,
        validation=validation,
        tokenizer=tokenizer_payload,
        model=model,
    )
    economics = _economics(
        consumed_input_tokens=train_geometry["consumed_input_tokens"],
        measured_input_tokens_per_second=measured_input_tokens_per_second,
        hourly_cost_usd=hourly_cost_usd,
        total_cost_cap_usd=total_cost_cap_usd,
    )
    launcher = inspect_launcher_argv(
        launcher_argv_json,
        project_root=root,
        train_order=train,
        validation_order=validation,
        tokenizer_root=tokenizer_path,
        train_certification=train_cert,
        validation_certification=validation_cert,
        recipe=recipe_payload,
        geometry=geometry,
    )
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()
    try:
        timestamp = datetime.fromisoformat(created_utc)
    except (TypeError, ValueError) as exc:
        raise RunAuthorityError("created_utc is not an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise RunAuthorityError("created_utc must be timezone-aware")
    payload: dict[str, Any] = {
        "format": AUTHORITY_FORMAT,
        "format_version": AUTHORITY_VERSION,
        "status": "authorized",
        "created_utc": created_utc,
        "git": git,
        "environment": environment,
        "hardware": hardware,
        "geometry": geometry,
        "data": {
            "corpus_qualification": corpus_gate,
            "train_order": train,
            "validation_order": validation,
            "train_certification": train_cert,
            "validation_certification": validation_cert,
        },
        "tokenizer": tokenizer_payload,
        "model": model,
        "training": {
            "recipe": recipe,
            "frozen_geometry": train_geometry,
            "optimizer_updates": optimizer_updates,
        },
        "economics": economics,
        "launcher": launcher,
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def _atomic_write_new(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise RunAuthorityError(f"Refusing to overwrite immutable artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication gives true create-if-absent semantics even with a racing writer.
        os.link(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RunAuthorityError(f"Refusing to overwrite immutable artifact: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_package_lock(output: str | Path) -> dict[str, Any]:
    """Write a complete, exact lock for the interpreter executing this command."""

    environment = current_environment_identity()
    _validate_training_package_versions(environment["packages"])
    payload = {
        "format": PACKAGE_LOCK_FORMAT,
        "format_version": 1,
        "python": {
            "implementation": environment["python"]["implementation"],
            "version": environment["python"]["version"],
        },
        "packages": environment["packages"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    destination = Path(output)
    _atomic_write_new(destination, encoded)
    return {
        "path": str(destination.resolve(strict=True)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "package_count": len(environment["packages"]),
    }


def publish_run_authority(output: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish a write-once authority JSON followed by its sidecar."""

    destination = Path(output)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if payload.get("format") == AUTHORITY_FORMAT:
        try:
            authorized_destination = Path(
                _option(payload["launcher"]["argv"], "--run-authority")
            ).resolve(strict=False)
        except (KeyError, TypeError) as exc:
            raise RunAuthorityError(
                "Run authority lacks its canonical --run-authority destination"
            ) from exc
        if authorized_destination != destination:
            raise RunAuthorityError(
                "Publication path differs from canonical --run-authority destination"
            )
    sidecar = destination.with_name(f"{destination.name}.sha256")
    if destination.exists() or destination.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise RunAuthorityError(
            f"Refusing to overwrite run authority or sidecar: {destination}"
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar_bytes = f"{digest}  {destination.name}\n".encode("ascii")
    _atomic_write_new(destination, encoded)
    try:
        _atomic_write_new(sidecar, sidecar_bytes)
    except Exception:
        # Never delete the already-published authority: preserving evidence is safer.
        # The absent sidecar makes the partial publication unusable and obvious.
        raise
    return {
        "path": str(destination),
        "sha256": digest,
        "sidecar": str(sidecar),
    }


def validate_run_authority(path: str | Path) -> dict[str, Any]:
    """Recollect every input and reject any byte, environment, or Git mutation."""

    bound = _verified_sidecar(path, label="run authority")
    authority_path = Path(bound["artifact"]["path"])
    try:
        payload = json.loads(
            _read_bound_bytes(bound["artifact"], label="run authority").decode(
                "utf-8", errors="strict"
            )
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunAuthorityError(f"Invalid run-authority JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunAuthorityError("Run-authority root must be an object")
    if (
        payload.get("format") != AUTHORITY_FORMAT
        or payload.get("format_version") != AUTHORITY_VERSION
        or payload.get("status") != "authorized"
    ):
        raise RunAuthorityError("Unsupported or non-authorized run authority")
    expected_authorization = payload.get("authorization_sha256")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256", None)
    if expected_authorization != canonical_sha256(unsigned):
        raise RunAuthorityError("Run-authority internal authorization digest mismatch")
    try:
        rebuilt = collect_run_authority(
            project_root=payload["git"]["project_root"],
            package_lock=payload["environment"]["lock"]["path"],
            container_image_digest=payload["environment"]["container_image_digest"],
            hardware_contract=payload["hardware"]["contract"]["path"],
            geometry_receipt=payload["geometry"]["artifact"]["path"],
            corpus_qualification=payload["data"]["corpus_qualification"]["artifact"][
                "path"
            ],
            train_order_manifest=payload["data"]["train_order"]["manifest"]["path"],
            validation_order_manifest=payload["data"]["validation_order"]["manifest"]["path"],
            train_certification=payload["data"]["train_certification"]["artifact"]["path"],
            validation_certification=payload["data"]["validation_certification"]["artifact"]["path"],
            tokenizer_root=payload["tokenizer"]["root"],
            training_recipe=payload["training"]["recipe"]["artifact"]["path"],
            launcher_argv_json=payload["launcher"]["argv_file"]["path"],
            measured_input_tokens_per_second=payload["economics"][
                "measured_aggregate_input_tokens_per_second"
            ],
            hourly_cost_usd=payload["economics"]["hourly_six_gpu_pod_cost_usd"],
            total_cost_cap_usd=payload["economics"]["total_cost_cap_usd"],
            created_utc=payload["created_utc"],
        )
    except (KeyError, TypeError) as exc:
        raise RunAuthorityError("Run-authority schema is incomplete") from exc
    if rebuilt != payload:
        raise RunAuthorityError(
            "Run authority no longer matches the current immutable inputs/environment"
        )
    return {
        "status": "valid",
        "path": str(authority_path),
        "sha256": bound["artifact"]["sha256"],
        "authorization_sha256": payload["authorization_sha256"],
        "git_commit": payload["git"]["commit"],
        "launcher_argv_sha256": payload["launcher"]["argv_sha256"],
        "projected_total_cost_usd": payload["economics"]["projected_total_cost_usd"],
    }
