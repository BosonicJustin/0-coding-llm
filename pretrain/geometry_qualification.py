"""Fail-closed six-GPU geometry-grid and end-to-end soak qualification.

This module deliberately does not derive economic throughput from the
trainer's per-step timer.  It supervises exact ``torchrun`` commands, starts an
external monotonic interval after warm-up, forces a checkpointed stop and a
six-rank resume inside that interval, and ends only after a second durable stop.

The three grid geometries all consume 192 packed rows per optimizer update:
``(global rows, accumulation) = (6, 32), (12, 16), (24, 8)``.  Both compile
decisions are measured.  A separately authenticated single-GPU baseline is
mandatory because six-GPU scaling efficiency cannot be inferred from the grid
itself without inventing evidence.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import selectors
import shlex
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from pretrain import data as training_data
from pretrain.geometry_evidence import (
    MAXIMUM_DATA_WAIT_FRACTION,
    MINIMUM_FREE_MEMORY_BYTES,
    MINIMUM_FREE_MEMORY_FRACTION,
    MINIMUM_SCALING_EFFICIENCY,
    MINIMUM_SOAK_STEPS,
    THROUGHPUT_COUNTER,
    THROUGHPUT_SCOPE,
    THROUGHPUT_TIMER,
)
from pretrain.tokenizer_identity import verify_tokenizer_identity


WORLD_SIZE = 6
UPDATE_ROWS = 192
BASELINE_WORLD_SIZE = 1
BASELINE_UPDATE_ROWS = UPDATE_ROWS // WORLD_SIZE
FINAL_TRAIN_TARGET_INPUT_TOKENS = 52_580_000_000
FINAL_TRAIN_EXPECTED_OPTIMIZER_UPDATES = 66_858
FINAL_TRAIN_EXPECTED_ROWS = FINAL_TRAIN_EXPECTED_OPTIMIZER_UPDATES * UPDATE_ROWS
FINAL_TRAIN_EXPECTED_CONSUMED_INPUT_TOKENS = FINAL_TRAIN_EXPECTED_ROWS * 4_096
GRID: tuple[tuple[int, int], ...] = ((6, 32), (12, 16), (24, 8))
PLAN_FORMAT = "six-gpu-geometry-qualification-plan"
BASELINE_PLAN_FORMAT = "single-gpu-geometry-baseline-plan"
GRID_RESULT_FORMAT = "six-gpu-geometry-grid-result"
CANDIDATE_RESULT_FORMAT = "six-gpu-geometry-candidate-result"
BASELINE_CANDIDATE_RESULT_FORMAT = "single-gpu-geometry-baseline-candidate-result"
BASELINE_FAILURE_FORMAT = "single-gpu-geometry-baseline-failure"
BASELINE_FORMAT = "single-gpu-geometry-baselines"
GEOMETRY_RECEIPT_FORMAT = "pretraining-accepted-geometry"
FORMAT_VERSION = 1
# Each trainer worker deliberately returns ``128 + SIGUSR1`` after publishing
# its clean-stop checkpoint.  ``torch.distributed.run`` reports that expected
# child termination through ``ChildFailedError`` and therefore exits 1.
_EXPECTED_TORCHRUN_GRACEFUL_STOP_RETURN_CODE = 1
_EXPECTED_TRAINER_GRACEFUL_STOP_EXIT_CODE = 128 + int(signal.SIGUSR1)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SECRET_NAME = re.compile(
    r"(?:api[-_]?key|authorization|credential|password|secret|"
    r"access[-_]?token|auth[-_]?token)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[-_]?key|authorization|credential|password|secret|"
    r"access[-_]?token|auth[-_]?token)\s*[=:]\s*[^\s,;]+"
)
_URI_USERINFO = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")
_WANDB_FAILURE_MARKERS = (
    "wandb initialization failed",
    "warning: disabling failed metric logger:",
    "disabling failed metric logger WandbLogger",
    "metric logger WandbLogger failed",
    "runtime cleanup failures:",
)


class GeometryQualificationError(RuntimeError):
    """A qualification invariant was not established."""


@dataclass(frozen=True)
class CandidateSpec:
    global_microbatch_rows: int
    gradient_accumulation_steps: int
    compile_model: bool

    @property
    def candidate_id(self) -> str:
        compile_value = 1 if self.compile_model else 0
        return (
            f"g{self.global_microbatch_rows}-"
            f"a{self.gradient_accumulation_steps}-compile{compile_value}"
        )

    @property
    def optimizer_update_rows(self) -> int:
        return self.global_microbatch_rows * self.gradient_accumulation_steps

    @property
    def local_microbatch_rows(self) -> int:
        if self.global_microbatch_rows % WORLD_SIZE:
            raise GeometryQualificationError(
                "Candidate global microbatch rows are not divisible by six"
            )
        return self.global_microbatch_rows // WORLD_SIZE

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "global_microbatch_rows": self.global_microbatch_rows,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "local_microbatch_rows_per_rank": self.local_microbatch_rows,
            "optimizer_update_rows": self.optimizer_update_rows,
            "compile_model": self.compile_model,
        }


CANDIDATES: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(global_rows, accumulation, compile_model)
    for global_rows, accumulation in GRID
    for compile_model in (False, True)
)


@dataclass(frozen=True)
class SoakSettings:
    measurement_warmup_steps: int = 5
    minimum_soak_steps: int = MINIMUM_SOAK_STEPS
    stop_after_soak_steps: int = 50
    order_buffer_steps: int = 8
    workers: int = 2
    eval_every: int = 25
    eval_batches: int = 8
    seed: int = 1234
    learning_rate: str = "0.0003"
    minimum_learning_rate: str = "0.00003"
    weight_decay: str = "0.1"
    beta1: str = "0.9"
    beta2: str = "0.95"
    adam_epsilon: str = "0.00000001"
    max_grad_norm: str = "1"
    trainer_warmup_steps: int = 0
    phase_timeout_seconds: int = 7200
    graceful_shutdown_seconds: int = 1800
    gpu_poll_interval_seconds: str = "0.5"
    wandb_mode: str = "offline"
    wandb_project: str = "coding-model-from-scratch"
    wandb_run_name_prefix: str = "geometry"
    checkpoint_generation_bytes: int = 16 * 1024**3

    def validate(self) -> None:
        integer_minima = {
            "measurement_warmup_steps": 1,
            "minimum_soak_steps": MINIMUM_SOAK_STEPS,
            "stop_after_soak_steps": 1,
            "order_buffer_steps": 2,
            "workers": 0,
            "eval_every": 1,
            "eval_batches": 1,
            "seed": 0,
            "trainer_warmup_steps": 0,
            "phase_timeout_seconds": 1,
            "graceful_shutdown_seconds": 1,
            "checkpoint_generation_bytes": 1024**3,
        }
        for field, minimum in integer_minima.items():
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise GeometryQualificationError(
                    f"{field} must be an integer >= {minimum}"
                )
        if self.stop_after_soak_steps >= self.minimum_soak_steps:
            raise GeometryQualificationError(
                "stop_after_soak_steps must leave work for the resumed phase"
            )
        if self.eval_every > self.minimum_soak_steps:
            raise GeometryQualificationError(
                "eval_every must guarantee validation inside the timed soak"
            )
        poll = _decimal(
            self.gpu_poll_interval_seconds,
            field="gpu_poll_interval_seconds",
        )
        if poll > Decimal("10"):
            raise GeometryQualificationError(
                "gpu_poll_interval_seconds must not exceed 10 seconds"
            )
        for field in (
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "adam_epsilon",
            "max_grad_norm",
        ):
            _decimal(getattr(self, field), field=field, allow_zero=field == "weight_decay")
        if self.wandb_mode not in {"offline", "online"}:
            raise GeometryQualificationError(
                "Authoritative geometry requires --wandb-mode offline or online; "
                "disabled cannot prove metric visibility"
            )
        if not self.wandb_project or _SECRET_NAME.search(self.wandb_project):
            raise GeometryQualificationError("wandb_project is empty or secret-like")
        if (
            not self.wandb_run_name_prefix
            or _SECRET_NAME.search(self.wandb_run_name_prefix)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.wandb_run_name_prefix)
        ):
            raise GeometryQualificationError(
                "wandb_run_name_prefix is empty, secret-like, or unsafe"
            )

    @property
    def diagnostic_optimizer_updates(self) -> int:
        return (
            self.measurement_warmup_steps
            + self.minimum_soak_steps
            + self.order_buffer_steps
        )


@dataclass
class PhaseObservation:
    start_monotonic_ns: int | None = None
    start_step: int | None = None
    start_consumed_input_tokens: int | None = None
    last_step: int = 0
    last_consumed_input_tokens: int = 0
    validation_events_after_start: int = 0
    wandb_log_events_after_start: int = 0
    data_wait_samples: int = 0
    maximum_data_wait_fraction: Decimal = Decimal("0")
    peak_memory_allocated_bytes: int = 0
    peak_memory_reserved_bytes: int = 0
    stop_request_monotonic_ns: int | None = None
    stop_requested_after_step: int | None = None
    json_metric_records: int = 0
    graceful_stop_events: int = 0
    wandb_failure: str | None = None

    def feed(
        self,
        line: str,
        *,
        clock_ns: Callable[[], int],
        start_after_step: int | None,
        request_stop_after_step: int | None,
        publish_stop: Callable[[], None] | None,
    ) -> None:
        lowered = line.lower()
        for marker in _WANDB_FAILURE_MARKERS:
            if marker.lower() in lowered:
                self.wandb_failure = marker
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        step = payload.get("train/step")
        consumed = payload.get("train/input_tokens")
        if isinstance(step, int) and not isinstance(step, bool):
            self.last_step = max(self.last_step, step)
        if isinstance(consumed, int) and not isinstance(consumed, bool):
            self.last_consumed_input_tokens = max(
                self.last_consumed_input_tokens, consumed
            )
        is_train_metric = "perf/input_tokens_per_second" in payload
        is_validation_metric = "validation/loss" in payload
        if "system/graceful_stop_signal" in payload:
            self.graceful_stop_events += 1
        if is_train_metric or is_validation_metric:
            self.json_metric_records += 1
        if (
            self.start_monotonic_ns is None
            and start_after_step is not None
            and step == start_after_step
            and isinstance(consumed, int)
            and not isinstance(consumed, bool)
        ):
            self.start_monotonic_ns = clock_ns()
            self.start_step = step
            self.start_consumed_input_tokens = consumed
        after_start = self.start_monotonic_ns is not None
        if after_start and is_validation_metric:
            self.validation_events_after_start += 1
        if after_start and (is_train_metric or is_validation_metric):
            self.wandb_log_events_after_start += 1
        if after_start and is_train_metric:
            wait = payload.get("perf/data_wait_fraction")
            if not isinstance(wait, (int, float)) or isinstance(wait, bool):
                raise GeometryQualificationError(
                    "Timed trainer metric lacks numeric perf/data_wait_fraction"
                )
            parsed = _decimal(
                str(wait), field="logged perf/data_wait_fraction", allow_zero=True
            )
            self.data_wait_samples += 1
            self.maximum_data_wait_fraction = max(
                self.maximum_data_wait_fraction, parsed
            )
            for field, attribute in (
                (
                    "system/cuda_peak_memory_allocated_bytes",
                    "peak_memory_allocated_bytes",
                ),
                (
                    "system/cuda_peak_memory_reserved_bytes",
                    "peak_memory_reserved_bytes",
                ),
            ):
                value = payload.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    setattr(self, attribute, max(getattr(self, attribute), value))
        if (
            publish_stop is not None
            and request_stop_after_step is not None
            and self.stop_request_monotonic_ns is None
            and isinstance(step, int)
            and step >= request_stop_after_step
        ):
            publish_stop()
            self.stop_request_monotonic_ns = clock_ns()
            self.stop_requested_after_step = step

    def merge(self, later: "PhaseObservation") -> "PhaseObservation":
        result = dataclasses.replace(self)
        result.last_step = max(self.last_step, later.last_step)
        result.last_consumed_input_tokens = max(
            self.last_consumed_input_tokens,
            later.last_consumed_input_tokens,
        )
        result.validation_events_after_start += later.validation_events_after_start
        result.wandb_log_events_after_start += later.wandb_log_events_after_start
        result.data_wait_samples += later.data_wait_samples
        result.maximum_data_wait_fraction = max(
            self.maximum_data_wait_fraction,
            later.maximum_data_wait_fraction,
        )
        result.peak_memory_allocated_bytes = max(
            self.peak_memory_allocated_bytes,
            later.peak_memory_allocated_bytes,
        )
        result.peak_memory_reserved_bytes = max(
            self.peak_memory_reserved_bytes,
            later.peak_memory_reserved_bytes,
        )
        result.json_metric_records += later.json_metric_records
        result.graceful_stop_events += later.graceful_stop_events
        result.wandb_failure = self.wandb_failure or later.wandb_failure
        result.stop_request_monotonic_ns = later.stop_request_monotonic_ns
        result.stop_requested_after_step = later.stop_requested_after_step
        return result


@dataclass
class GPUMemoryObservation:
    gpu_count: int = 0
    total_bytes_per_gpu: tuple[int, ...] = ()
    minimum_free_bytes_per_gpu: int | None = None
    samples: int = 0

    def add(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        expected_gpu_count: int = WORLD_SIZE,
    ) -> None:
        if len(rows) != expected_gpu_count:
            raise GeometryQualificationError(
                f"GPU sampler returned {len(rows)} devices, expected "
                f"{expected_gpu_count}"
            )
        totals = tuple(total for total, _ in rows)
        if any(total < 1 or free < 0 or free > total for total, free in rows):
            raise GeometryQualificationError("GPU memory sampler returned invalid bytes")
        if self.samples and totals != self.total_bytes_per_gpu:
            raise GeometryQualificationError("Visible GPU memory topology changed during soak")
        self.gpu_count = expected_gpu_count
        self.total_bytes_per_gpu = totals
        sample_minimum = min(free for _, free in rows)
        self.minimum_free_bytes_per_gpu = (
            sample_minimum
            if self.minimum_free_bytes_per_gpu is None
            else min(self.minimum_free_bytes_per_gpu, sample_minimum)
        )
        self.samples += 1

    def merge(self, later: "GPUMemoryObservation") -> "GPUMemoryObservation":
        if not self.samples:
            return dataclasses.replace(later)
        if not later.samples:
            return dataclasses.replace(self)
        if self.total_bytes_per_gpu != later.total_bytes_per_gpu:
            raise GeometryQualificationError("GPU topology changed across resume")
        assert self.minimum_free_bytes_per_gpu is not None
        assert later.minimum_free_bytes_per_gpu is not None
        return GPUMemoryObservation(
            gpu_count=self.gpu_count,
            total_bytes_per_gpu=self.total_bytes_per_gpu,
            minimum_free_bytes_per_gpu=min(
                self.minimum_free_bytes_per_gpu,
                later.minimum_free_bytes_per_gpu,
            ),
            samples=self.samples + later.samples,
        )


def _decimal(value: Any, *, field: str, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise GeometryQualificationError(f"{field} must be a decimal string or integer")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise GeometryQualificationError(f"{field} is not a decimal") from exc
    if not result.is_finite() or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise GeometryQualificationError(f"{field} must be finite and {qualifier}")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GeometryQualificationError(f"Value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise GeometryQualificationError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    resolved = candidate.resolve(strict=True)
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise GeometryQualificationError(f"{label} changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def read_bound_json(
    descriptor: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if set(descriptor) != {"path", "bytes", "sha256"}:
        raise GeometryQualificationError(f"{label} descriptor has the wrong schema")
    path = Path(str(descriptor["path"]))
    if path.is_symlink() or not path.is_file():
        raise GeometryQualificationError(f"{label} is not a regular file: {path}")
    payload = path.read_bytes()
    if (
        len(payload) != descriptor["bytes"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise GeometryQualificationError(f"{label} changed after binding")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GeometryQualificationError(f"{label} is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GeometryQualificationError(f"{label} root must be an object")
    canonical_json_bytes(value)
    return value


def verified_json_with_sidecar(
    path: str | Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = artifact(path, label=label)
    subject = Path(descriptor["path"])
    sidecar_path = subject.with_name(f"{subject.name}.sha256")
    sidecar = artifact(sidecar_path, label=f"{label} sidecar")
    expected = f"{descriptor['sha256']}  {subject.name}\n"
    try:
        found = Path(sidecar["path"]).read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise GeometryQualificationError(f"Cannot read {label} sidecar") from exc
    if found != expected:
        raise GeometryQualificationError(
            f"{label} sidecar must contain exactly {expected!r}"
        )
    return read_bound_json(descriptor, label=label), {
        "artifact": descriptor,
        "sidecar": sidecar,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json_new(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    sidecar = destination.with_name(f"{destination.name}.sha256")
    if destination.exists() or destination.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise GeometryQualificationError(
            f"Refusing to overwrite immutable artifact or sidecar: {destination}"
        )
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    _publish_bytes_new(destination, encoded)
    try:
        _publish_bytes_new(
            sidecar,
            f"{digest}  {destination.name}\n".encode("ascii"),
        )
    except BaseException:
        # Preserve the poisoned partial publication for explicit diagnosis.
        raise
    return {
        "artifact": {
            "path": str(destination),
            "bytes": len(encoded),
            "sha256": digest,
        },
        "sidecar": artifact(sidecar, label=f"{destination.name} sidecar"),
    }


def _publish_bytes_new(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise GeometryQualificationError(
            f"Refusing to overwrite immutable artifact: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def replace_journal(path: Path, payload: Mapping[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _regular_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise GeometryQualificationError(
            f"{label} must be a non-symlink directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _ensure_confined_child_directory(
    root: Path, child: Path, *, label: str, create: bool = True
) -> Path:
    """Create/verify a real directory whose resolved path remains below root."""

    root_resolved = _regular_directory(root, label=f"{label} root")
    if child.is_symlink():
        raise GeometryQualificationError(f"{label} must not be a symlink: {child}")
    if child.exists():
        if not child.is_dir():
            raise GeometryQualificationError(f"{label} is not a directory: {child}")
    elif create:
        child.mkdir(parents=False)
        _fsync_directory(child.parent.resolve(strict=True))
    else:
        return child
    resolved = child.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise GeometryQualificationError(
            f"{label} escapes its qualification root: {resolved}"
        ) from exc
    return resolved


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    return read_bound_json(artifact(path, label=label), label=label)


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise GeometryQualificationError(
            "Linux boot_id is required for resumable monotonic timing"
        )
    value = path.read_text(encoding="ascii", errors="strict").strip().lower()
    if not re.fullmatch(r"[0-9a-f-]{36}", value):
        raise GeometryQualificationError("Linux boot_id is invalid")
    return value


def _geometry_hardware_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable hardware/runtime identity shared by both pod receipts.

    The bootstrap and final RunPod receipts intentionally contain different
    data-path evidence, argv, timestamps, and free-space observations.  None of
    those fields describes the hardware/runtime on which geometry was measured.
    This projection retains the device UUID/order, CUDA/NCCL stack, package
    lock, deterministic environment, and clean source identity while excluding
    only phase-specific or volatile observations.
    """

    try:
        qualification = contract["qualification"]
        gpu = qualification["gpu"]
        host = qualification["host"]
        source = qualification["source"]
        devices = gpu["devices"]
        package = host["package_lock"]
        environment = host["environment"]
        git = source["git"]
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Hardware contract lacks stable geometry identity fields"
        ) from exc
    if not isinstance(devices, list) or len(devices) != WORLD_SIZE:
        raise GeometryQualificationError(
            "Hardware contract lacks the six-device geometry identity"
        )
    stable_device_fields = (
        "visible_index",
        "physical_index",
        "uuid",
        "name",
        "pci_bus_id",
        "compute_capability",
        "total_memory_bytes",
        "multiprocessor_count",
        "bf16_supported",
        "nvidia_smi_memory_bytes",
        "mig_mode",
    )
    stable_devices: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise GeometryQualificationError(
                "Hardware contract contains an invalid GPU identity"
            )
        stable = {
            field: device[field]
            for field in stable_device_fields
            if field in device
        }
        if (
            stable.get("visible_index") != index
            or not isinstance(stable.get("uuid"), str)
            or not stable["uuid"].startswith("GPU-")
        ):
            raise GeometryQualificationError(
                "Hardware contract GPU identity/order is invalid"
            )
        stable_devices.append(stable)
    try:
        identity = {
            "topology": contract["topology"],
            "world_size": contract["world_size"],
            "gpu_count": contract["gpu_count"],
            "gpu_model": contract["gpu_model"],
            "gpu_memory_bytes": contract["gpu_memory_bytes"],
            "compute_capability": contract["compute_capability"],
            "multiprocessor_count": contract["multiprocessor_count"],
            "driver_version": contract["driver_version"],
            "cuda_runtime_version": contract["cuda_runtime_version"],
            "cudnn_version": contract["cudnn_version"],
            "nccl_version": contract["nccl_version"],
            "torch_version": contract["torch_version"],
            "bf16_supported": contract["bf16_supported"],
            "distributed_strategy": contract["distributed_strategy"],
            "nvlink_policy": qualification["nvlink_policy"],
            "devices": stable_devices,
            "peer_access_matrix": gpu["peer_access_matrix"],
            "nvidia_topology": {
                "labels_in_visible_order": gpu["nvidia_topology"][
                    "labels_in_visible_order"
                ],
                "matrix": gpu["nvidia_topology"]["matrix"],
                "nvlink_pairs": gpu["nvidia_topology"]["nvlink_pairs"],
                "possible_pairs": gpu["nvidia_topology"]["possible_pairs"],
            },
            "torch_cuda_arch_list": gpu["torch_cuda_arch_list"],
            "compiled_arch_supported": gpu["compiled_arch_supported"],
            "package_lock": package,
            "deterministic_environment": {
                "required": environment["required"],
                "cuda_visible_devices": environment["cuda_visible_devices"],
            },
            "source": {
                "qualification_script_sha256": source[
                    "qualification_script"
                ]["sha256"],
                "requirements_train_sha256": source["requirements_train"][
                    "sha256"
                ],
                "requirements_wandb_sha256": source["requirements_wandb"][
                    "sha256"
                ],
                "git_commit": git["commit"],
                "git_tree": git["tree"],
                "git_head_archive_sha256": git["head_archive_sha256"],
                "git_clean": git["clean"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Hardware contract stable geometry identity is incomplete"
        ) from exc
    canonical_json_bytes(identity)
    return identity


def _validate_hardware_contract(
    path: Path,
    *,
    allow_provisional: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Reuse the launch authority's complete receipt validators.  A superficial
    # six-field check here would allow a hand-written object to impersonate the
    # RunPod qualifier's NCCL/topology/package evidence.
    try:
        from pretrain.run_authority import (
            HARDWARE_FORMAT,
            PROVISIONAL_HARDWARE_FORMAT,
            RunAuthorityError,
            inspect_hardware_contract,
            inspect_provisional_hardware_contract,
        )

        raw = _load_json_file(path, label="six-GPU hardware contract")
        contract_format = raw.get("format")
        if contract_format == HARDWARE_FORMAT:
            inspected = inspect_hardware_contract(path)
            scope = "final-launch-authorizing"
        elif contract_format == PROVISIONAL_HARDWARE_FORMAT and allow_provisional:
            inspected = inspect_provisional_hardware_contract(path)
            scope = "geometry-only-provisional"
        elif contract_format == PROVISIONAL_HARDWARE_FORMAT:
            raise GeometryQualificationError(
                "Provisional hardware evidence cannot authorize the final soak"
            )
        else:
            raise GeometryQualificationError(
                "Unsupported hardware receipt format for geometry qualification"
            )
    except (OSError, ValueError, RunAuthorityError) as exc:
        raise GeometryQualificationError(
            f"Hardware contract is not production-authoritative: {exc}"
        ) from exc
    payload = inspected["expected"]
    bound = {
        "artifact": inspected["contract"],
        "sidecar": inspected["sidecar"],
    }
    allowed_statuses = {"accepted", "provisional"} if allow_provisional else {"accepted"}
    if (
        payload.get("status") not in allowed_statuses
        or payload.get("topology") != "single-node"
        or payload.get("world_size") != WORLD_SIZE
        or payload.get("gpu_count") != WORLD_SIZE
        or payload.get("bf16_supported") is not True
        or payload.get("distributed_strategy") != "ddp"
    ):
        raise GeometryQualificationError(
            "Hardware contract is not a valid six-GPU BF16 DDP contract"
        )
    identity = _geometry_hardware_identity(payload)
    return payload, bound, {
        "scope": scope,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
    }


def verify_plan_artifacts(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash every artifact descriptor frozen into a published plan."""

    verified = 0

    def walk(value: Any, *, label: str) -> None:
        nonlocal verified
        if isinstance(value, Mapping):
            if set(value) == {"path", "bytes", "sha256"}:
                current = artifact(Path(str(value["path"])), label=label)
                if current != dict(value):
                    raise GeometryQualificationError(
                        f"Published plan artifact changed: {label}"
                    )
                verified += 1
                return
            for key, child in value.items():
                walk(child, label=f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, label=f"{label}[{index}]")

    walk(plan, label="plan")
    tokenizer = plan.get("inputs", {}).get("common", {}).get("tokenizer", {})
    if not isinstance(tokenizer, Mapping):
        raise GeometryQualificationError("Plan tokenizer binding is missing")
    try:
        identity = verify_tokenizer_identity(
            Path(str(tokenizer["root"])),
            expected_manifest_sha256=str(tokenizer["manifest_sha256"]),
            expected_vocab_size=int(plan["inputs"]["common"]["vocab_size"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise GeometryQualificationError(
            f"Plan tokenizer identity no longer verifies: {exc}"
        ) from exc
    return {
        "artifact_descriptors_verified": verified,
        "tokenizer_manifest_sha256": identity.manifest_sha256,
        "tokenizer_vocabulary_sha256": identity.vocabulary_sha256,
    }


def verify_live_runtime(
    *,
    plan: Mapping[str, Any],
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Fail closed before starting a candidate on the launch-day runtime."""

    bindings = verify_plan_artifacts(plan)
    try:
        import torch
        from scripts import launch_pretraining

        runtime = launch_pretraining.inspect_runtime(
            nproc_per_node=WORLD_SIZE,
            torch_module=torch,
            environment=environment,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise GeometryQualificationError(
            f"Six-GPU runtime preflight failed: {exc}"
        ) from exc
    hardware = plan.get("inputs", {}).get("hardware", {}).get("expected")
    if not isinstance(hardware, Mapping):
        raise GeometryQualificationError("Plan hardware expectation is missing")
    try:
        environment_evidence = hardware["qualification"]["host"]["environment"]
        required_environment = environment_evidence["required"]
        recorded_visible = environment_evidence["cuda_visible_devices"]
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Hardware contract lacks deterministic environment evidence"
        ) from exc
    if (
        not isinstance(required_environment, Mapping)
        or required_environment.get("WANDB_MODE")
        != plan.get("settings", {}).get("wandb_mode", "offline")
        or any(
            environment.get(str(name)) != value
            for name, value in required_environment.items()
        )
    ):
        raise GeometryQualificationError(
            "Launch-day deterministic environment differs from the hardware contract"
        )
    current_visible = [
        item.strip() for item in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if current_visible != recorded_visible:
        raise GeometryQualificationError(
            "CUDA_VISIBLE_DEVICES differs from the qualified visible-device order"
        )
    dangerous_transport = {
        name: environment.get(name)
        for name in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
        if environment.get(name) not in (None, "", "0")
    }
    if dangerous_transport:
        raise GeometryQualificationError(
            "NCCL peer/shared-memory transport is disabled in the live environment"
        )
    expected_profile = (
        hardware.get("gpu_model"),
        hardware.get("gpu_memory_bytes"),
        hardware.get("compute_capability"),
        hardware.get("multiprocessor_count"),
    )
    for profile in runtime.cuda_device_profiles:
        found_profile = (
            profile["name"],
            profile["total_memory_bytes"],
            profile["compute_capability"],
            profile["multiprocessor_count"],
        )
        if found_profile != expected_profile:
            raise GeometryQualificationError(
                "Live CUDA device profile differs from the hardware contract"
            )
    if (
        runtime.cuda_devices != WORLD_SIZE
        or runtime.world_size != WORLD_SIZE
        or runtime.bf16_supported_devices != list(range(WORLD_SIZE))
        or runtime.torch_version != hardware.get("torch_version")
        or runtime.cuda_runtime != hardware.get("cuda_runtime_version")
    ):
        raise GeometryQualificationError(
            "Live PyTorch/CUDA runtime differs from the hardware contract"
        )
    planned_python = plan.get("runtime", {}).get("python", {})
    if (
        not isinstance(planned_python, Mapping)
        or runtime.python_executable != planned_python.get("invocation_path")
        or runtime.python_executable_sha256
        != planned_python.get("resolved", {}).get("sha256")
    ):
        raise GeometryQualificationError(
            "Live qualification interpreter differs from the planned torchrun interpreter"
        )
    try:
        nvidia_smi = Path(
            str(
                hardware["qualification"]["gpu"]["nvidia_smi_executable"][
                    "path"
                ]
            )
        )
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Hardware contract lacks its authenticated nvidia-smi executable"
        ) from exc
    memory = sample_nvidia_smi_memory(nvidia_smi)
    identity = sample_nvidia_smi_identity(nvidia_smi)
    expected_devices = hardware["qualification"]["gpu"]["devices"]
    if len(expected_devices) != WORLD_SIZE:
        raise GeometryQualificationError("Hardware contract GPU inventory is incomplete")
    expected_identity = sorted([
        {
            "index": int(device["physical_index"]),
            "uuid": str(device["uuid"]),
            "name": str(device["name"]),
            "memory_bytes": device.get("nvidia_smi_memory_bytes"),
        }
        for device in expected_devices
    ], key=lambda row: row["index"])
    if any(
        found["index"] != expected["index"]
        or found["uuid"] != expected["uuid"]
        or found["name"] != expected["name"]
        or (
            expected["memory_bytes"] is not None
            and found["memory_bytes"] != expected["memory_bytes"]
        )
        for found, expected in zip(identity, expected_identity, strict=True)
    ):
        raise GeometryQualificationError(
            "Live nvidia-smi GPU identity differs from the hardware contract"
        )
    expected_smi = _expected_nvidia_smi_memory(hardware)
    if expected_smi is not None and tuple(total for total, _ in memory) != expected_smi:
        raise GeometryQualificationError(
            "Live nvidia-smi memory differs from the hardware contract"
        )
    return {
        "status": "pass",
        "world_size": WORLD_SIZE,
        "torch_version": runtime.torch_version,
        "cuda_runtime_version": runtime.cuda_runtime,
        "gpu_model": hardware["gpu_model"],
        "gpu_memory_bytes": hardware["gpu_memory_bytes"],
        "minimum_free_memory_bytes": _normalized_nvidia_free_memory(
            memory,
            physical_memory_bytes=int(hardware["gpu_memory_bytes"]),
        ),
        "bindings": bindings,
        "output_storage": verify_output_storage(
            Path(str(plan["output_root"])), plan=plan
        ),
    }


def verify_output_storage(
    root: Path, *, plan: Mapping[str, Any], allow_missing_root: bool = False
) -> dict[str, Any]:
    """Require durable network capacity for every retained checkpoint lineage."""

    settings = SoakSettings(**plan["settings"])
    settings.validate()
    try:
        network = plan["inputs"]["hardware"]["expected"]["qualification"]["host"]
        network = network["storage"]["network"]
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Hardware contract lacks qualified network-storage evidence"
        ) from exc
    network_root = Path(str(network.get("path", "")))
    if network_root.is_symlink() or not network_root.is_dir():
        raise GeometryQualificationError("Qualified network root is unavailable")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise GeometryQualificationError("Qualification output root is unsafe")
        probe = root.resolve(strict=True)
    else:
        if not allow_missing_root:
            raise GeometryQualificationError("Qualification output root is missing")
        if root.parent.is_symlink() or not root.parent.is_dir():
            raise GeometryQualificationError(
                "Qualification output parent is unavailable"
            )
        probe = root.parent.resolve(strict=True)
    network_resolved = network_root.resolve(strict=True)
    try:
        probe.relative_to(network_resolved)
    except ValueError as exc:
        raise GeometryQualificationError(
            "Qualification output root is outside the qualified network volume"
        ) from exc
    if (
        network.get("classification") != "network"
        or network.get("read_only") is not False
        or probe.stat().st_dev != network.get("device")
        or not os.access(probe, os.W_OK)
    ):
        raise GeometryQualificationError(
            "Qualification output is not the writable qualified network filesystem"
        )
    mode = plan.get("mode")
    if mode == "grid":
        generations = 2 * len(CANDIDATES)
    elif mode == "single-gpu-baselines":
        # Baseline throughput uses the same end-to-end stop/checkpoint/resume
        # scope as the six-GPU candidates, so each lineage retains both the
        # pre-resume and post-resume generations.
        generations = 2 * len(CANDIDATES)
    elif mode == "final-soak":
        generations = 2
    else:
        raise GeometryQualificationError("Unknown qualification storage mode")
    estimate = settings.checkpoint_generation_bytes
    existing_bytes = 0
    if root.is_dir() and not root.is_symlink():
        if mode in {"grid", "single-gpu-baselines"}:
            structural_directories = [root / "orders", root / "candidates"]
            structural_directories.extend(
                root / "candidates" / candidate.candidate_id / "checkpoints"
                for candidate in CANDIDATES
            )
            expected_paths = {
                root / "candidates" / candidate.candidate_id / "checkpoints" / name
                for candidate in CANDIDATES
                for name in (
                    ("last.pt", "last.previous.pt")
                )
            }
        else:
            structural_directories = [
                root / "final-candidate",
                root / "final-candidate" / "checkpoints",
            ]
            expected_paths = {
                root / "final-candidate" / "checkpoints" / name
                for name in ("last.pt", "last.previous.pt")
            }
        for directory in structural_directories:
            if directory.exists() or directory.is_symlink():
                _ensure_confined_child_directory(
                    root,
                    directory,
                    label="qualification storage directory",
                    create=False,
                )
        found_paths = set(root.glob("**/checkpoints/last*.pt"))
        unexpected = found_paths - expected_paths
        if unexpected:
            raise GeometryQualificationError(
                "Unexpected checkpoint-like paths prevent capacity credit: "
                + ", ".join(str(path) for path in sorted(unexpected))
            )
        for path in sorted(found_paths):
            if path.is_symlink() or not path.is_file():
                raise GeometryQualificationError(
                    f"Qualification checkpoint path is unsafe: {path}"
                )
            size = path.stat().st_size
            if size > estimate:
                raise GeometryQualificationError(
                    "Observed checkpoint exceeds checkpoint_generation_bytes; "
                    "use a new root with a larger frozen estimate"
                )
            existing_bytes += size
    projected_checkpoint_bytes = generations * estimate
    remaining_checkpoint_bytes = max(0, projected_checkpoint_bytes - existing_bytes)
    headroom = max(1024**3, (estimate + 9) // 10) + 1024**3
    required_free = remaining_checkpoint_bytes + headroom
    usage = shutil.disk_usage(probe)
    if usage.free < required_free:
        raise GeometryQualificationError(
            f"Insufficient durable geometry-output capacity: found {usage.free} "
            f"bytes free, require {required_free} bytes"
        )
    return {
        "status": "pass",
        "network_root": str(network_resolved),
        "probe": str(probe),
        "checkpoint_generation_bytes": estimate,
        "projected_checkpoint_generations": generations,
        "projected_checkpoint_bytes": projected_checkpoint_bytes,
        "existing_checkpoint_bytes": existing_bytes,
        "explicit_headroom_bytes": headroom,
        "required_free_bytes": required_free,
        "observed_free_bytes": int(usage.free),
    }


def _expected_nvidia_smi_memory(
    hardware: Mapping[str, Any],
) -> tuple[int, ...] | None:
    try:
        devices = hardware["qualification"]["gpu"]["devices"]
    except (KeyError, TypeError):
        return None
    if not isinstance(devices, list) or len(devices) != WORLD_SIZE:
        return None
    values = [device.get("nvidia_smi_memory_bytes") for device in devices]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in values
    ):
        return None
    return tuple(values)


def _normalized_nvidia_free_memory(
    rows: Sequence[tuple[int, int]], *, physical_memory_bytes: int
) -> int:
    """Conservatively normalize nvidia-smi free bytes to CUDA's VRAM basis."""

    if len(rows) != WORLD_SIZE:
        raise GeometryQualificationError("Expected six nvidia-smi memory rows")
    normalized: list[int] = []
    for total, free in rows:
        used = total - free
        candidate = min(free, physical_memory_bytes - used)
        if candidate < 1:
            raise GeometryQualificationError(
                "nvidia-smi reports no positive CUDA-basis free-memory margin"
            )
        normalized.append(candidate)
    return min(normalized)


def load_single_gpu_baselines(
    path: Path,
    *,
    hardware_identity_sha256: str,
    sequence_length: int,
) -> tuple[dict[str, Decimal], dict[str, Any]]:
    payload, bound = verified_json_with_sidecar(
        path, label="single-GPU baseline receipt"
    )
    if set(payload) != {
        "format",
        "format_version",
        "status",
        "hardware_identity_sha256",
        "producer_plan",
        "exact_shared_order_payload_sha256",
        "results",
        "candidates",
    } or (
        payload.get("format") != BASELINE_FORMAT
        or payload.get("format_version") != FORMAT_VERSION
        or payload.get("status") != "pass"
        or payload.get("hardware_identity_sha256") != hardware_identity_sha256
    ):
        raise GeometryQualificationError(
            "Single-GPU baseline receipt format/hardware binding is invalid"
        )
    entries = payload.get("candidates")
    if not isinstance(entries, dict) or set(entries) != {
        candidate.candidate_id for candidate in CANDIDATES
    }:
        raise GeometryQualificationError(
            "Single-GPU baseline receipt must contain all six grid candidates"
        )
    rates: dict[str, Decimal] = {}
    producer_bound = payload.get("producer_plan")
    try:
        producer_path = Path(str(producer_bound["artifact"]["path"]))
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Single-GPU baseline receipt lacks its producer plan"
        ) from exc
    producer_plan, verified_producer_bound = verified_json_with_sidecar(
        producer_path, label="single-GPU baseline producer plan"
    )
    _validate_plan_self_hash(producer_plan)
    if (
        verified_producer_bound != producer_bound
        or producer_plan.get("format") != BASELINE_PLAN_FORMAT
        or producer_plan.get("mode") != "single-gpu-baselines"
        or producer_plan.get("inputs", {})
        .get("hardware", {})
        .get("geometry_identity", {})
        .get("identity_sha256")
        != hardware_identity_sha256
        or producer_plan.get("inputs", {}).get("common", {}).get("sequence_length")
        != sequence_length
    ):
        raise GeometryQualificationError(
            "Single-GPU baseline producer plan binding is invalid"
        )
    shared_order_sha256 = payload.get("exact_shared_order_payload_sha256")
    if (
        not isinstance(shared_order_sha256, str)
        or _SHA256.fullmatch(shared_order_sha256) is None
    ):
        raise GeometryQualificationError(
            "Single-GPU baseline shared-order identity is invalid"
        )
    result_bindings = payload.get("results")
    if not isinstance(result_bindings, dict) or set(result_bindings) != set(entries):
        raise GeometryQualificationError(
            "Single-GPU baseline receipt lacks all candidate result bindings"
        )
    required = {
        "global_microbatch_rows",
        "gradient_accumulation_steps",
        "compile_model",
        "gpu_count",
        "scope",
        "timer",
        "counter",
        "start_consumed_input_tokens",
        "end_consumed_input_tokens",
        "elapsed_wall_time_ns",
        "aggregate_input_tokens_per_second",
    }
    for candidate in CANDIDATES:
        entry = entries[candidate.candidate_id]
        if not isinstance(entry, dict) or set(entry) != required:
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} has the wrong schema"
            )
        if (
            entry["global_microbatch_rows"] != candidate.local_microbatch_rows
            or entry["gradient_accumulation_steps"]
            != candidate.gradient_accumulation_steps
            or entry["compile_model"] is not candidate.compile_model
            or entry["gpu_count"] != 1
            or entry["scope"] != THROUGHPUT_SCOPE
            or entry["timer"] != THROUGHPUT_TIMER
            or entry["counter"] != THROUGHPUT_COUNTER
        ):
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} does not match the local-rank recipe"
            )
        start = entry["start_consumed_input_tokens"]
        end = entry["end_consumed_input_tokens"]
        elapsed = entry["elapsed_wall_time_ns"]
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, end, elapsed)
        ) or start < 0 or end <= start or elapsed < 1:
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} counters/timer are invalid"
            )
        local_update_tokens = (
            candidate.local_microbatch_rows
            * candidate.gradient_accumulation_steps
            * sequence_length
        )
        if start % local_update_tokens or end % local_update_tokens:
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} counters are not update-aligned"
            )
        measured_updates = (end - start) // local_update_tokens
        if measured_updates < MINIMUM_SOAK_STEPS:
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} covers only {measured_updates} "
                f"updates; at least {MINIMUM_SOAK_STEPS} are required"
            )
        calculated = Decimal(end - start) * Decimal(1_000_000_000) / Decimal(elapsed)
        reported = _decimal(
            entry["aggregate_input_tokens_per_second"],
            field=f"baseline {candidate.candidate_id} throughput",
        )
        if abs(calculated - reported) / calculated > Decimal("0.000001"):
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} throughput does not reconcile"
            )
        rates[candidate.candidate_id] = reported
        result_bound = result_bindings[candidate.candidate_id]
        try:
            result_path = Path(str(result_bound["artifact"]["path"]))
        except (KeyError, TypeError) as exc:
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} result binding is invalid"
            ) from exc
        result, verified_result_bound = verified_json_with_sidecar(
            result_path,
            label=f"single-GPU baseline {candidate.candidate_id} result",
        )
        if (
            verified_result_bound != result_bound
            or result.get("format") != BASELINE_CANDIDATE_RESULT_FORMAT
            or result.get("format_version") != FORMAT_VERSION
            or result.get("status") != "pass"
            or result.get("plan_sha256") != producer_plan["plan_sha256"]
            or result.get("candidate") != candidate.as_dict()
            or result.get("measurement") != entry
            or result.get("order", {}).get("order", {}).get("sha256")
            != shared_order_sha256
        ):
            raise GeometryQualificationError(
                f"Baseline {candidate.candidate_id} result/provenance is invalid"
            )
    return rates, bound


def seal_single_gpu_baselines(
    *,
    draft: Path,
    output: Path,
    hardware_contract: Path,
    sequence_length: int,
) -> dict[str, Any]:
    """Validate a manual measurement draft before write-once publication."""

    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
    ):
        raise GeometryQualificationError("Baseline sequence_length must be positive")
    _, _, hardware_identity = _validate_hardware_contract(
        hardware_contract, allow_provisional=True
    )
    draft_payload = _load_json_file(draft, label="single-GPU baseline draft")
    parent = output.parent.resolve(strict=True)
    if output.parent.is_symlink() or not parent.is_dir():
        raise GeometryQualificationError("Baseline output parent is unsafe")
    # Exercise the exact sidecar-aware production reader in a private temporary
    # generation before making the requested destination immutable.
    with tempfile.TemporaryDirectory(
        prefix=".baseline-validation-", dir=parent
    ) as temporary:
        candidate = Path(temporary) / "baseline.json"
        publish_json_new(candidate, draft_payload)
        load_single_gpu_baselines(
            candidate,
            hardware_identity_sha256=hardware_identity["identity_sha256"],
            sequence_length=sequence_length,
        )
    bound = publish_json_new(output, draft_payload)
    _, verified_bound = load_single_gpu_baselines(
        output,
        hardware_identity_sha256=hardware_identity["identity_sha256"],
        sequence_length=sequence_length,
    )
    if verified_bound != bound:
        raise GeometryQualificationError(
            "Published baseline receipt changed during final verification"
        )
    return bound


def render_torchrun_command(
    *,
    python_executable: Path,
    order_manifest: Path,
    validation_order_manifest: Path,
    tokenizer_root: Path,
    checkpoint: Path,
    candidate: CandidateSpec,
    settings: SoakSettings,
    run_name: str,
    run_group: str,
    resume: bool,
    world_size: int = WORLD_SIZE,
    global_microbatch_rows: int | None = None,
    wandb_tag: str = "geometry-qualification",
) -> list[str]:
    if global_microbatch_rows is None:
        global_microbatch_rows = candidate.global_microbatch_rows
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size < 1
        or not isinstance(global_microbatch_rows, int)
        or isinstance(global_microbatch_rows, bool)
        or global_microbatch_rows < 1
        or global_microbatch_rows % world_size
    ):
        raise GeometryQualificationError(
            "Torchrun world-size/global-microbatch geometry is invalid"
        )
    command = [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        "pretrain.train",
        "--order-manifest",
        str(order_manifest),
        "--tokenizer",
        str(tokenizer_root),
        "--validation-order-manifest",
        str(validation_order_manifest),
        "--model-size",
        "1.3b",
        "--device",
        "cuda",
        "--precision",
        "bfloat16",
        "--deterministic",
        "--activation-checkpointing",
        "--fused-adamw",
        "--global-microbatch-rows",
        str(global_microbatch_rows),
        "--gradient-accumulation-steps",
        str(candidate.gradient_accumulation_steps),
        "--workers",
        str(settings.workers),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-every",
        str(settings.diagnostic_optimizer_updates),
        "--eval-every",
        str(settings.eval_every),
        "--eval-batches",
        str(settings.eval_batches),
        "--eval-at-start",
        "--log-every",
        "1",
        "--wandb-mode",
        settings.wandb_mode,
        "--wandb-project",
        settings.wandb_project,
        "--wandb-run-name",
        run_name,
        "--wandb-group",
        run_group,
        "--wandb-tags",
        wandb_tag,
        candidate.candidate_id,
        "--learning-rate",
        settings.learning_rate,
        "--min-learning-rate",
        settings.minimum_learning_rate,
        "--warmup-steps",
        str(settings.trainer_warmup_steps),
        "--weight-decay",
        settings.weight_decay,
        "--beta1",
        settings.beta1,
        "--beta2",
        settings.beta2,
        "--adam-eps",
        settings.adam_epsilon,
        "--max-grad-norm",
        settings.max_grad_norm,
        "--seed",
        str(settings.seed),
    ]
    if candidate.compile_model:
        command.append("--compile")
    if resume:
        command.extend(("--resume", str(checkpoint)))
    _assert_secret_free_command(command)
    return command


def _assert_secret_free_command(
    command: Sequence[str], environment: Mapping[str, str] | None = None
) -> None:
    inherited = os.environ if environment is None else environment
    secret_values = {
        value
        for name, value in inherited.items()
        if _SECRET_NAME.search(name) and len(value) >= 4
    }
    for index, token in enumerate(command):
        if "\x00" in token or not token:
            raise GeometryQualificationError("Command contains an empty/NUL argument")
        if token.startswith("--") and _SECRET_NAME.search(token):
            raise GeometryQualificationError(
                f"Qualification command contains a secret-like option: {token}"
            )
        if index and _SECRET_NAME.search(command[index - 1]):
            raise GeometryQualificationError("Qualification command would record a secret")
        if _SECRET_ASSIGNMENT.search(token) or _URI_USERINFO.search(token):
            raise GeometryQualificationError(
                "Qualification command contains a credential-bearing argument"
            )
        if any(secret in token for secret in secret_values):
            raise GeometryQualificationError(
                "Qualification command contains an inherited secret value"
            )


def command_sha256(command: Sequence[str]) -> str:
    return canonical_sha256(list(command))


def command_display(command: Sequence[str]) -> str:
    _assert_secret_free_command(command)
    return shlex.join(command)


def _verify_torchrun_graceful_exit(
    *, return_code: int, log: Mapping[str, Any], label: str
) -> None:
    """Authenticate torchrun's wrapper failure for a clean trainer stop.

    The checkpoint proves all six ranks reached the synchronized save.  The
    torchrun root-cause line proves at least one worker then returned the
    trainer's intentional ``128 + SIGUSR1`` status rather than an unrelated
    worker exception; torchrun normalizes that ChildFailedError to process 1.
    """

    if return_code != _EXPECTED_TORCHRUN_GRACEFUL_STOP_RETURN_CODE:
        raise GeometryQualificationError(
            f"{label} torchrun did not use the frozen intentional-stop wrapper "
            f"status (found {return_code}, expected "
            f"{_EXPECTED_TORCHRUN_GRACEFUL_STOP_RETURN_CODE})"
        )
    descriptor = artifact(Path(str(log.get("path", ""))), label=f"{label} log")
    if descriptor != dict(log):
        raise GeometryQualificationError(f"{label} log changed before exit validation")
    try:
        contents = Path(descriptor["path"]).read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeError) as exc:
        raise GeometryQualificationError(f"Cannot read {label} log: {exc}") from exc
    expected = re.compile(
        rf"(?m)^\s*exitcode\s*:\s*{_EXPECTED_TRAINER_GRACEFUL_STOP_EXIT_CODE}"
        r"(?:\s+\(pid:\s*\d+\))?\s*$"
    )
    if expected.search(contents) is None:
        raise GeometryQualificationError(
            f"{label} log lacks torchrun's trainer exitcode "
            f"{_EXPECTED_TRAINER_GRACEFUL_STOP_EXIT_CODE} root-cause evidence"
        )


@dataclass(frozen=True)
class PhaseRun:
    return_code: int
    completed_monotonic_ns: int
    observation: PhaseObservation
    gpu_memory: GPUMemoryObservation
    log: dict[str, Any]


class QualificationLock:
    """Nonblocking singleton lock retained for the complete qualification."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".geometry-qualification.lock"
        self._handle: TextIO | None = None

    def __enter__(self) -> "QualificationLock":
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise GeometryQualificationError(
                f"Cannot safely open qualification lock {self.path}: {exc}"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise GeometryQualificationError(
                f"Qualification lock is not a regular file: {self.path}"
            )
        self._handle = os.fdopen(descriptor, "a+", encoding="ascii")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise GeometryQualificationError(
                f"Another geometry qualification owns {self.path}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def redact_text(text: str, environment: Mapping[str, str]) -> str:
    """Remove inherited secret values and common inline credential forms."""

    redacted = text
    for name, value in environment.items():
        if _SECRET_NAME.search(name) and len(value) >= 4:
            redacted = redacted.replace(value, "<redacted>")
    redacted = re.sub(
        r"(?i)\b(api[-_]?key|authorization|password|secret|token)"
        r"([=:]\s*)([^\s,;]+)",
        r"\1\2<redacted>",
        redacted,
    )
    return redacted


def sample_nvidia_smi_memory(
    executable: str | Path = "nvidia-smi",
) -> list[tuple[int, int]]:
    """Return ``(total_bytes, free_bytes)`` for all six visible GPUs."""

    command = [
        str(executable),
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise GeometryQualificationError(f"Cannot sample nvidia-smi memory: {exc}") from exc
    if completed.returncode != 0:
        raise GeometryQualificationError(
            f"nvidia-smi memory query failed with exit {completed.returncode}"
        )
    parsed: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise GeometryQualificationError("nvidia-smi memory output is malformed")
        try:
            index, total_mib, free_mib = (int(field) for field in fields)
        except ValueError as exc:
            raise GeometryQualificationError(
                "nvidia-smi memory output contains a non-integer"
            ) from exc
        parsed.append((index, total_mib * 1024**2, free_mib * 1024**2))
    parsed.sort()
    if [row[0] for row in parsed] != list(range(WORLD_SIZE)):
        raise GeometryQualificationError(
            "nvidia-smi must expose exactly indexed GPUs 0..5"
        )
    return [(total, free) for _, total, free in parsed]


def sample_nvidia_smi_identity(
    executable: str | Path = "nvidia-smi",
) -> list[dict[str, Any]]:
    """Read the physical GPU UUID/name identity without recording topology noise."""

    try:
        completed = subprocess.run(
            [
                str(executable),
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise GeometryQualificationError(
            f"Cannot inspect nvidia-smi GPU identity: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise GeometryQualificationError(
            f"nvidia-smi identity query failed with exit {completed.returncode}"
        )
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise GeometryQualificationError("nvidia-smi identity output is malformed")
        try:
            index = int(fields[0])
            memory_bytes = int(fields[3]) * 1024**2
        except ValueError as exc:
            raise GeometryQualificationError(
                "nvidia-smi identity output contains an invalid integer"
            ) from exc
        if not fields[1].startswith("GPU-") or not fields[2] or memory_bytes < 1:
            raise GeometryQualificationError("nvidia-smi GPU identity is invalid")
        records.append(
            {
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "memory_bytes": memory_bytes,
            }
        )
    records.sort(key=lambda row: row["index"])
    if [record["index"] for record in records] != list(range(WORLD_SIZE)):
        raise GeometryQualificationError(
            "nvidia-smi must expose exactly physical GPUs 0..5"
        )
    if len({record["uuid"] for record in records}) != WORLD_SIZE:
        raise GeometryQualificationError("nvidia-smi GPU UUIDs are not unique")
    return records


def _publish_stop_request(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise GeometryQualificationError(f"Stop request already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{int(signal.SIGUSR1)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=35)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=35)


def supervise_phase(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    stop_request_path: Path,
    working_directory: Path,
    settings: SoakSettings,
    start_after_step: int | None,
    request_stop_after_step: int,
    prior_observation: PhaseObservation | None = None,
    expected_gpu_count: int = WORLD_SIZE,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    gpu_sampler: Callable[[], Sequence[tuple[int, int]]] = sample_nvidia_smi_memory,
    popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
) -> PhaseRun:
    """Run one torchrun phase and force a clean checkpointed stop."""

    _assert_secret_free_command(command)
    if log_path.exists() or log_path.is_symlink():
        raise GeometryQualificationError(
            f"Refusing to append to an existing phase log: {log_path}"
        )
    if stop_request_path.exists() or stop_request_path.is_symlink():
        raise GeometryQualificationError(
            f"Stale stop request makes the phase ambiguous: {stop_request_path}"
        )
    child_environment = dict(environment)
    child_environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "PRETRAIN_STOP_REQUEST_FILE": str(stop_request_path),
            "WANDB_MODE": settings.wandb_mode,
            "WANDB_SILENT": "true",
        }
    )
    observation = (
        PhaseObservation()
        if prior_observation is None
        else dataclasses.replace(
            prior_observation,
            validation_events_after_start=0,
            wandb_log_events_after_start=0,
            data_wait_samples=0,
            maximum_data_wait_fraction=Decimal("0"),
            peak_memory_allocated_bytes=0,
            peak_memory_reserved_bytes=0,
            stop_request_monotonic_ns=None,
            stop_requested_after_step=None,
            json_metric_records=0,
            graceful_stop_events=0,
            wandb_failure=None,
        )
    )
    if (
        not isinstance(expected_gpu_count, int)
        or isinstance(expected_gpu_count, bool)
        or expected_gpu_count < 1
    ):
        raise GeometryQualificationError("expected_gpu_count must be positive")
    memory = GPUMemoryObservation()
    phase_started_ns = clock_ns()
    phase_timeout_ns = settings.phase_timeout_seconds * 1_000_000_000
    graceful_timeout_ns = settings.graceful_shutdown_seconds * 1_000_000_000
    poll_interval_ns = int(
        _decimal(
            settings.gpu_poll_interval_seconds,
            field="gpu_poll_interval_seconds",
        )
        * Decimal(1_000_000_000)
    )
    next_gpu_poll_ns = phase_started_ns
    stop_deadline_ns: int | None = None
    process: subprocess.Popen[str] | None = None
    selector: selectors.BaseSelector | None = None
    with log_path.open("x", encoding="utf-8", errors="strict") as log:
        try:
            process = popen_factory(
                list(command),
                cwd=str(working_directory),
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            if process.stdout is None:
                raise GeometryQualificationError("torchrun stdout pipe is unavailable")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            def publish_stop() -> None:
                nonlocal stop_deadline_ns
                _publish_stop_request(stop_request_path)
                stop_deadline_ns = clock_ns() + graceful_timeout_ns

            while True:
                now = clock_ns()
                if now >= next_gpu_poll_ns:
                    memory.add(
                        gpu_sampler(), expected_gpu_count=expected_gpu_count
                    )
                    next_gpu_poll_ns = now + poll_interval_ns
                if now - phase_started_ns > phase_timeout_ns:
                    if observation.stop_request_monotonic_ns is None:
                        publish_stop()
                    raise GeometryQualificationError("Qualification phase timed out")
                if stop_deadline_ns is not None and now > stop_deadline_ns:
                    raise GeometryQualificationError(
                        "Graceful-stop checkpoint exceeded its deadline"
                    )
                events = selector.select(timeout=0.25)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    safe_line = redact_text(line, child_environment)
                    log.write(safe_line)
                    log.flush()
                    observation.feed(
                        line,
                        clock_ns=clock_ns,
                        start_after_step=start_after_step,
                        request_stop_after_step=request_stop_after_step,
                        publish_stop=publish_stop,
                    )
                if process.poll() is not None:
                    for line in process.stdout:
                        safe_line = redact_text(line, child_environment)
                        log.write(safe_line)
                        observation.feed(
                            line,
                            clock_ns=clock_ns,
                            start_after_step=start_after_step,
                            request_stop_after_step=request_stop_after_step,
                            publish_stop=None,
                        )
                    break
            completed_ns = clock_ns()
            memory.add(gpu_sampler(), expected_gpu_count=expected_gpu_count)
            return_code = int(process.wait())
            log.flush()
            os.fsync(log.fileno())
        except BaseException:
            if process is not None and process.poll() is None:
                if not stop_request_path.exists():
                    try:
                        _publish_stop_request(stop_request_path)
                    except BaseException:
                        pass
                deadline = time.monotonic() + settings.graceful_shutdown_seconds
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.25)
                _terminate_process_group(process)
            raise
        finally:
            if selector is not None:
                selector.close()
            if process is not None and process.stdout is not None:
                process.stdout.close()
    if observation.stop_request_monotonic_ns is None:
        raise GeometryQualificationError(
            "Phase ended without the supervisor publishing its stop request"
        )
    if observation.graceful_stop_events < 1:
        raise GeometryQualificationError(
            "Phase log lacks the trainer's graceful-stop acknowledgement"
        )
    if observation.wandb_failure is not None:
        raise GeometryQualificationError(
            f"W&B failed during qualification: {observation.wandb_failure}"
        )
    return PhaseRun(
        return_code=return_code,
        completed_monotonic_ns=completed_ns,
        observation=observation,
        gpu_memory=memory,
        log=artifact(log_path, label="qualification phase log"),
    )


def _checkpoint_summary(
    path: Path,
    *,
    expected_order_sha256: str,
    expected_order_payload_sha256: str,
    expected_world_size: int = WORLD_SIZE,
) -> dict[str, Any]:
    """Inspect a trusted qualification-generated checkpoint with mmap loading."""

    descriptor = artifact(path, label="qualification checkpoint")
    before_load = path.stat()
    try:
        import torch

        payload = torch.load(
            descriptor["path"],
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except Exception as exc:
        raise GeometryQualificationError(
            f"Cannot inspect qualification-generated checkpoint: {exc}"
        ) from exc
    after_load = path.stat()
    if any(
        getattr(before_load, field) != getattr(after_load, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise GeometryQualificationError(
            "Qualification checkpoint changed while it was being inspected"
        )
    if not isinstance(payload, dict):
        raise GeometryQualificationError("Qualification checkpoint root is not an object")
    train_state = payload.get("train_state")
    data_identity = payload.get("data_identity")
    metadata = payload.get("metadata")
    if (
        payload.get("world_size") != expected_world_size
        or not isinstance(train_state, dict)
        or not isinstance(data_identity, str)
        or not isinstance(metadata, dict)
    ):
        raise GeometryQualificationError(
            "Qualification checkpoint lacks expected-rank state/identity metadata"
        )
    if (
        f"order-manifest-sha256:{expected_order_sha256}" not in data_identity
        or f"order-payload-sha256:{expected_order_payload_sha256}"
        not in data_identity
    ):
        raise GeometryQualificationError("Checkpoint is bound to another training order")
    required_counters = (
        "completed_steps",
        "completed_microbatches",
        "consumed_rows",
        "consumed_input_tokens",
        "last_validated_step",
    )
    counters: dict[str, int] = {}
    for field in required_counters:
        value = train_state.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GeometryQualificationError(f"Checkpoint train_state.{field} is invalid")
        counters[field] = value
    wandb_run_id = metadata.get("wandb_run_id")
    if not isinstance(wandb_run_id, str) or not wandb_run_id:
        raise GeometryQualificationError(
            "Checkpoint lacks the successful W&B run identity"
        )
    summary = {
        **descriptor,
        **counters,
        "world_size": expected_world_size,
        "wandb_run_id_sha256": hashlib.sha256(
            wandb_run_id.encode("utf-8")
        ).hexdigest(),
        "data_identity_sha256": hashlib.sha256(
            data_identity.encode("utf-8")
        ).hexdigest(),
        "training_geometry_sha256": canonical_sha256(payload.get("training_geometry")),
        "trajectory_sha256": canonical_sha256(payload.get("train_trajectory_config")),
        "runtime_signature_sha256": canonical_sha256(payload.get("runtime_signature")),
        "implementation_signature_sha256": canonical_sha256(
            payload.get("implementation_signature")
        ),
    }
    del payload
    return summary


def _wandb_inventory(root: Path) -> dict[str, Any]:
    wandb_root = root / "wandb"
    if wandb_root.is_symlink() or not wandb_root.is_dir():
        raise GeometryQualificationError(
            f"Local W&B evidence directory is missing: {wandb_root}"
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(wandb_root.rglob("*")):
        if path.is_symlink():
            raise GeometryQualificationError(
                f"Local W&B evidence tree contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise GeometryQualificationError(
                f"Local W&B evidence tree contains a special filesystem node: {path}"
            )
        relative = path.relative_to(wandb_root).as_posix()
        descriptor = artifact(path, label=f"local W&B evidence file {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
        )
    if not entries or not any(entry["bytes"] > 0 for entry in entries):
        raise GeometryQualificationError(
            "Local W&B evidence tree contains no durable data"
        )
    return {
        "root": str(wandb_root.resolve(strict=True)),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "inventory_sha256": canonical_sha256(entries),
    }


def _executable_identity(path: Path) -> dict[str, Any]:
    invocation = Path(os.path.abspath(path))
    current_invocation = Path(os.path.abspath(sys.executable))
    if invocation != current_invocation:
        raise GeometryQualificationError(
            "The torchrun interpreter must be the interpreter executing the qualifier"
        )
    try:
        resolved = invocation.resolve(strict=True)
    except OSError as exc:
        raise GeometryQualificationError(
            f"Cannot resolve Python executable {invocation}: {exc}"
        ) from exc
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise GeometryQualificationError(
            f"Python executable does not resolve to a regular file: {invocation}"
        )
    return {
        "invocation_path": str(invocation),
        "resolved": artifact(resolved, label="Python executable"),
        "version": ".".join(str(value) for value in sys.version_info[:3]),
        "implementation": sys.implementation.name,
    }


def _source_identity(cli_script: Path) -> dict[str, Any]:
    import pretrain.data
    import pretrain.geometry_evidence
    import pretrain.model
    import pretrain.run_authority
    import pretrain.tokenizer_identity
    import pretrain.train
    from scripts import launch_pretraining

    return {
        "qualification_module": artifact(
            Path(__file__), label="geometry qualification module"
        ),
        "qualification_cli": artifact(cli_script, label="geometry qualification CLI"),
        "trainer": artifact(Path(pretrain.train.__file__), label="trainer source"),
        "model": artifact(Path(pretrain.model.__file__), label="model source"),
        "data": artifact(Path(pretrain.data.__file__), label="data source"),
        "geometry_evidence": artifact(
            Path(pretrain.geometry_evidence.__file__), label="geometry evidence source"
        ),
        "run_authority": artifact(
            Path(pretrain.run_authority.__file__), label="run authority source"
        ),
        "tokenizer_identity": artifact(
            Path(pretrain.tokenizer_identity.__file__), label="tokenizer identity source"
        ),
        "production_launcher": artifact(
            Path(launch_pretraining.__file__), label="production launcher source"
        ),
    }


def _packed_inputs(
    manifests: Mapping[str, Path],
    *,
    tokenizer_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(manifests) != set(training_data.DOMAIN_ORDER):
        raise GeometryQualificationError(
            f"Packed manifests must be exactly {training_data.DOMAIN_ORDER}"
        )
    records: dict[str, Any] = {}
    common: dict[str, Any] | None = None
    for domain in training_data.DOMAIN_ORDER:
        descriptor = artifact(manifests[domain], label=f"{domain} packed manifest")
        payload = read_bound_json(descriptor, label=f"{domain} packed manifest")
        if payload.get("format") != "packed-document-causal" or payload.get(
            "domain"
        ) != domain:
            raise GeometryQualificationError(
                f"Packed manifest does not identify domain {domain}: {manifests[domain]}"
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
            ):
                if payload.get(field) != common.get(field):
                    raise GeometryQualificationError(
                        f"Packed manifests disagree on {field}"
                    )
        rows = payload.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise GeometryQualificationError(f"{domain} packed rows are invalid")
        records[domain] = {"manifest": descriptor, "rows": rows}
    assert common is not None
    sequence_length = common.get("sequence_length")
    vocab_size = common.get("vocab_size")
    tokenizer_sha = common.get("tokenizer_manifest_sha256")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
        or not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 1
        or not isinstance(tokenizer_sha, str)
        or _SHA256.fullmatch(tokenizer_sha) is None
    ):
        raise GeometryQualificationError("Packed manifest common geometry is invalid")
    tokenizer = verify_tokenizer_identity(
        tokenizer_root,
        expected_manifest_sha256=tokenizer_sha,
        expected_vocab_size=vocab_size,
    )
    return records, {
        "split": common["split"],
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "eos_token_id": common["eos_token_id"],
        "tokenizer": {
            "root": str(Path(tokenizer_root).resolve(strict=True)),
            "manifest": artifact(
                tokenizer.manifest_path, label="tokenizer manifest"
            ),
            "manifest_sha256": tokenizer.manifest_sha256,
            "vocabulary_sha256": tokenizer.vocabulary_sha256,
        },
    }


def _validation_input(
    path: Path,
    *,
    common: Mapping[str, Any],
    eval_batches: int,
) -> dict[str, Any]:
    descriptor = artifact(path, label="validation order manifest")
    payload = read_bound_json(descriptor, label="validation order manifest")
    if payload.get("split") != "validation":
        raise GeometryQualificationError("Validation order does not identify validation")
    for field in (
        "sequence_length",
        "vocab_size",
        "eos_token_id",
        "tokenizer_manifest_sha256",
    ):
        expected = (
            common["tokenizer"]["manifest_sha256"]
            if field == "tokenizer_manifest_sha256"
            else common[field]
        )
        if payload.get(field) != expected:
            raise GeometryQualificationError(
                f"Validation order differs from packed data on {field}"
            )
    geometries: dict[str, Any] = {}
    evaluation_rows = {
        candidate.global_microbatch_rows for candidate in CANDIDATES
    } | {candidate.local_microbatch_rows for candidate in CANDIDATES}
    for global_rows in sorted(evaluation_rows):
        key = str(global_rows)
        if key in geometries:
            continue
        geometry = training_data.evaluation_order_geometry(
            path,
            global_microbatch_rows=global_rows,
            verify_checksum=True,
        )
        if geometry["available_global_microbatches"] < eval_batches:
            raise GeometryQualificationError(
                f"Validation order has fewer than {eval_batches} microbatches for "
                f"global rows {global_rows}"
            )
        geometries[key] = geometry
    return {"manifest": descriptor, "geometry_by_global_rows": geometries}


def build_baseline_plan(
    *,
    output_root: Path,
    packed_manifests: Mapping[str, Path],
    validation_order_manifest: Path,
    tokenizer_root: Path,
    hardware_contract: Path,
    settings: SoakSettings,
    cli_script: Path,
    python_executable: Path = Path(sys.executable),
    baseline_gpu_visible_index: int = 0,
) -> dict[str, Any]:
    """Build the immutable plan that automatically measures one-GPU baselines."""

    settings.validate()
    if (
        not isinstance(baseline_gpu_visible_index, int)
        or isinstance(baseline_gpu_visible_index, bool)
        or not 0 <= baseline_gpu_visible_index < WORLD_SIZE
    ):
        raise GeometryQualificationError(
            "baseline_gpu_visible_index must select one of the six qualified GPUs"
        )
    hardware, hardware_bound, hardware_identity = _validate_hardware_contract(
        hardware_contract, allow_provisional=True
    )
    packed, common = _packed_inputs(
        packed_manifests,
        tokenizer_root=tokenizer_root,
    )
    if common["split"] != "train":
        raise GeometryQualificationError("Baseline packed manifests must be from train")
    validation = _validation_input(
        validation_order_manifest,
        common=common,
        eval_batches=settings.eval_batches,
    )
    devices = hardware["qualification"]["gpu"]["devices"]
    selected_device = devices[baseline_gpu_visible_index]
    plan: dict[str, Any] = {
        "format": BASELINE_PLAN_FORMAT,
        "format_version": FORMAT_VERSION,
        "mode": "single-gpu-baselines",
        "output_root": str(Path(output_root).resolve(strict=False)),
        "world_size": BASELINE_WORLD_SIZE,
        "effective_optimizer_update_rows": BASELINE_UPDATE_ROWS,
        "baseline_gpu": {
            "visible_index": baseline_gpu_visible_index,
            "physical_index": selected_device["physical_index"],
            "uuid": selected_device["uuid"],
        },
        "candidates": [candidate.as_dict() for candidate in CANDIDATES],
        "settings": dataclasses.asdict(settings),
        "inputs": {
            "packed": packed,
            "common": common,
            "validation_order": validation,
            "hardware": {
                "bound": hardware_bound,
                "expected": hardware,
                "geometry_identity": hardware_identity,
            },
        },
        "runtime": {
            "python": _executable_identity(python_executable),
            "sources": _source_identity(cli_script),
        },
        "secrets_recorded": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_baseline_plan_authority(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild every semantic input to an automatic one-GPU baseline plan."""

    _validate_plan_self_hash(plan)
    if (
        plan.get("format") != BASELINE_PLAN_FORMAT
        or plan.get("format_version") != FORMAT_VERSION
        or plan.get("mode") != "single-gpu-baselines"
        or plan.get("world_size") != BASELINE_WORLD_SIZE
        or plan.get("effective_optimizer_update_rows") != BASELINE_UPDATE_ROWS
        or plan.get("candidates")
        != [candidate.as_dict() for candidate in CANDIDATES]
    ):
        raise GeometryQualificationError("Single-GPU baseline plan is not canonical")
    artifacts = verify_plan_artifacts(plan)
    hardware_path = Path(plan["inputs"]["hardware"]["bound"]["artifact"]["path"])
    hardware, bound, identity = _validate_hardware_contract(
        hardware_path, allow_provisional=True
    )
    if (
        hardware != plan["inputs"]["hardware"]["expected"]
        or bound != plan["inputs"]["hardware"]["bound"]
        or identity != plan["inputs"]["hardware"]["geometry_identity"]
    ):
        raise GeometryQualificationError("Baseline hardware authority changed")
    packed_paths = {
        domain: Path(plan["inputs"]["packed"][domain]["manifest"]["path"])
        for domain in training_data.DOMAIN_ORDER
    }
    packed, common = _packed_inputs(
        packed_paths,
        tokenizer_root=Path(plan["inputs"]["common"]["tokenizer"]["root"]),
    )
    if packed != plan["inputs"]["packed"] or common != plan["inputs"]["common"]:
        raise GeometryQualificationError("Baseline packed-data authority changed")
    settings = SoakSettings(**plan["settings"])
    validation = _validation_input(
        Path(plan["inputs"]["validation_order"]["manifest"]["path"]),
        common=common,
        eval_batches=settings.eval_batches,
    )
    if validation != plan["inputs"]["validation_order"]:
        raise GeometryQualificationError("Baseline validation authority changed")
    selected = plan.get("baseline_gpu")
    devices = hardware["qualification"]["gpu"]["devices"]
    try:
        expected = devices[int(selected["visible_index"])]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise GeometryQualificationError("Baseline GPU selection is invalid") from exc
    if selected != {
        "visible_index": expected["visible_index"],
        "physical_index": expected["physical_index"],
        "uuid": expected["uuid"],
    }:
        raise GeometryQualificationError("Baseline GPU identity changed")
    return {"artifacts": artifacts, "status": "pass"}


def build_grid_plan(
    *,
    output_root: Path,
    packed_manifests: Mapping[str, Path],
    validation_order_manifest: Path,
    tokenizer_root: Path,
    hardware_contract: Path,
    single_gpu_baselines: Path,
    settings: SoakSettings,
    cli_script: Path,
    python_executable: Path = Path(sys.executable),
) -> dict[str, Any]:
    settings.validate()
    hardware, hardware_bound, hardware_identity = _validate_hardware_contract(
        hardware_contract, allow_provisional=True
    )
    packed, common = _packed_inputs(
        packed_manifests,
        tokenizer_root=tokenizer_root,
    )
    if common["split"] != "train":
        raise GeometryQualificationError("Grid packed manifests must be from train")
    validation = _validation_input(
        validation_order_manifest,
        common=common,
        eval_batches=settings.eval_batches,
    )
    _, baseline_bound = load_single_gpu_baselines(
        single_gpu_baselines,
        hardware_identity_sha256=hardware_identity["identity_sha256"],
        sequence_length=common["sequence_length"],
    )
    baseline_receipt = read_bound_json(
        baseline_bound["artifact"], label="single-GPU baseline receipt"
    )
    baseline_plan = read_bound_json(
        baseline_receipt["producer_plan"]["artifact"],
        label="single-GPU baseline producer plan",
    )
    verify_baseline_plan_authority(baseline_plan)
    runtime = {
        "python": _executable_identity(python_executable),
        "sources": _source_identity(cli_script),
    }
    if (
        baseline_plan.get("settings") != dataclasses.asdict(settings)
        or baseline_plan.get("inputs", {}).get("packed") != packed
        or baseline_plan.get("inputs", {}).get("common") != common
        or baseline_plan.get("inputs", {}).get("validation_order") != validation
        or baseline_plan.get("inputs", {})
        .get("hardware", {})
        .get("geometry_identity", {})
        .get("identity_sha256")
        != hardware_identity["identity_sha256"]
        or baseline_plan.get("runtime") != runtime
    ):
        raise GeometryQualificationError(
            "Grid settings/data/hardware/runtime differ from automatic baselines"
        )
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "format_version": FORMAT_VERSION,
        "mode": "grid",
        "output_root": str(Path(output_root).resolve(strict=False)),
        "world_size": WORLD_SIZE,
        "effective_optimizer_update_rows": UPDATE_ROWS,
        "candidates": [candidate.as_dict() for candidate in CANDIDATES],
        "settings": dataclasses.asdict(settings),
        "inputs": {
            "packed": packed,
            "common": common,
            "validation_order": validation,
            "hardware": {
                "bound": hardware_bound,
                "expected": hardware,
                "geometry_identity": hardware_identity,
            },
            "single_gpu_baselines": baseline_bound,
        },
        "runtime": runtime,
        "secrets_recorded": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _validate_plan_self_hash(plan: Mapping[str, Any]) -> None:
    recorded = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(unsigned):
        raise GeometryQualificationError("Qualification plan self-hash is invalid")


def verify_grid_plan_authority(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild every semantic grid input from its authenticated path."""

    artifacts = verify_plan_artifacts(plan)
    hardware_path = Path(plan["inputs"]["hardware"]["bound"]["artifact"]["path"])
    hardware, hardware_bound, hardware_identity = _validate_hardware_contract(
        hardware_path, allow_provisional=True
    )
    if (
        hardware != plan["inputs"]["hardware"]["expected"]
        or hardware_bound != plan["inputs"]["hardware"]["bound"]
        or hardware_identity
        != plan["inputs"]["hardware"]["geometry_identity"]
    ):
        raise GeometryQualificationError("Grid hardware authority changed")
    settings = SoakSettings(**plan["settings"])
    packed_paths = {
        domain: Path(plan["inputs"]["packed"][domain]["manifest"]["path"])
        for domain in training_data.DOMAIN_ORDER
    }
    packed, common = _packed_inputs(
        packed_paths,
        tokenizer_root=Path(plan["inputs"]["common"]["tokenizer"]["root"]),
    )
    if packed != plan["inputs"]["packed"] or common != plan["inputs"]["common"]:
        raise GeometryQualificationError("Grid packed-data authority changed")
    validation = _validation_input(
        Path(plan["inputs"]["validation_order"]["manifest"]["path"]),
        common=common,
        eval_batches=settings.eval_batches,
    )
    if validation != plan["inputs"]["validation_order"]:
        raise GeometryQualificationError("Grid validation authority changed")
    _, baseline_bound = load_single_gpu_baselines(
        Path(plan["inputs"]["single_gpu_baselines"]["artifact"]["path"]),
        hardware_identity_sha256=hardware_identity["identity_sha256"],
        sequence_length=common["sequence_length"],
    )
    if baseline_bound != plan["inputs"]["single_gpu_baselines"]:
        raise GeometryQualificationError("Grid baseline authority changed")
    return {"artifacts": artifacts, "status": "pass"}


def _load_validated_grid_result(
    path: Path, *, require_pass: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, bound = verified_json_with_sidecar(path, label="geometry grid result")
    required = {
        "format",
        "format_version",
        "status",
        "plan_sha256",
        "world_size",
        "effective_optimizer_update_rows",
        "exact_shared_order_payload_sha256",
        "candidates_attempted",
        "candidates",
        "selection_policy",
        "accepted_candidate",
        "accepted_result",
        "live_runtime",
        "secrets_recorded",
    }
    if set(payload) != required or (
        payload.get("format") != GRID_RESULT_FORMAT
        or payload.get("format_version") != FORMAT_VERSION
        or payload.get("status") not in {"pass", "fail"}
        or payload.get("world_size") != WORLD_SIZE
        or payload.get("effective_optimizer_update_rows") != UPDATE_ROWS
        or payload.get("candidates_attempted") != len(CANDIDATES)
        or payload.get("selection_policy")
        != (
            "highest-external-end-to-end-throughput-among-threshold-passing-"
            "candidates-stable-grid-order-tie-break"
        )
        or payload.get("secrets_recorded") is not False
        or not isinstance(payload.get("exact_shared_order_payload_sha256"), str)
        or _SHA256.fullmatch(payload["exact_shared_order_payload_sha256"]) is None
    ):
        raise GeometryQualificationError("Grid result top-level authority is invalid")
    plan_path = Path(bound["artifact"]["path"]).parent / "PLAN.json"
    grid_plan, grid_plan_bound = verified_json_with_sidecar(
        plan_path, label="geometry grid plan"
    )
    _validate_plan_self_hash(grid_plan)
    if (
        grid_plan.get("mode") != "grid"
        or grid_plan.get("plan_sha256") != payload.get("plan_sha256")
        or grid_plan.get("candidates")
        != [candidate.as_dict() for candidate in CANDIDATES]
    ):
        raise GeometryQualificationError("Grid result does not match its grid plan")
    verify_grid_plan_authority(grid_plan)
    records = payload.get("candidates")
    if not isinstance(records, list) or len(records) != len(CANDIDATES):
        raise GeometryQualificationError("Grid result lacks all six candidates")
    passing: list[tuple[Decimal, int, CandidateSpec, Mapping[str, Any]]] = []
    for index, (candidate, record) in enumerate(zip(CANDIDATES, records, strict=True)):
        if (
            not isinstance(record, Mapping)
            or record.get("candidate") != candidate.as_dict()
            or record.get("status") not in {"pass", "fail"}
            or not isinstance(record.get("result"), Mapping)
            or not isinstance(record.get("failures"), list)
        ):
            raise GeometryQualificationError("Grid candidate record is invalid")
        result_bound = record["result"]
        try:
            result_path = Path(str(result_bound["artifact"]["path"]))
        except (KeyError, TypeError) as exc:
            raise GeometryQualificationError(
                "Grid candidate result binding is invalid"
            ) from exc
        candidate_payload, verified_bound = verified_json_with_sidecar(
            result_path, label=f"grid candidate {candidate.candidate_id} result"
        )
        if verified_bound != result_bound or (
            candidate_payload.get("plan_sha256") != payload["plan_sha256"]
            or candidate_payload.get("candidate") != candidate.as_dict()
            or candidate_payload.get("status") != record["status"]
            or candidate_payload.get("failures", []) != record["failures"]
        ):
            raise GeometryQualificationError(
                "Grid candidate result differs from the grid authority"
            )
        candidate_measurements = candidate_payload.get("measurements")
        if isinstance(candidate_measurements, Mapping):
            recomputed_failures = _candidate_failures(
                measurements=candidate_measurements,
                physical_memory_bytes=int(
                    grid_plan["inputs"]["hardware"]["expected"][
                        "gpu_memory_bytes"
                    ]
                ),
            )
            expected_status = "pass" if not recomputed_failures else "fail"
            if (
                recomputed_failures != candidate_payload.get("failures")
                or candidate_payload.get("status") != expected_status
            ):
                raise GeometryQualificationError(
                    "Grid candidate thresholds do not reconcile"
                )
        elif (
            candidate_payload.get("status") != "fail"
            or candidate_payload.get("failures")
            != ["candidate_execution_or_evidence_failure"]
        ):
            raise GeometryQualificationError(
                "Grid candidate lacks measurements without a terminal execution failure"
            )
        if record["status"] == "pass":
            try:
                rate = _decimal(
                    candidate_payload["measurements"][
                        "aggregate_input_tokens_per_second"
                    ],
                    field=f"candidate {candidate.candidate_id} throughput",
                )
            except (KeyError, TypeError) as exc:
                raise GeometryQualificationError(
                    "Passing grid candidate lacks throughput evidence"
                ) from exc
            if record.get("aggregate_input_tokens_per_second") != _decimal_string(rate):
                raise GeometryQualificationError(
                    "Grid candidate throughput differs from its result"
                )
            passing.append((rate, -index, candidate, candidate_payload))
        elif "aggregate_input_tokens_per_second" in record:
            raise GeometryQualificationError(
                "Failed grid candidate must not claim accepted throughput"
            )
    if not passing:
        if (
            require_pass
            or payload.get("status") != "fail"
            or payload.get("accepted_candidate") is not None
            or payload.get("accepted_result") is not None
        ):
            raise GeometryQualificationError(
                "Grid failure/pass state does not reconcile to candidate results"
            )
        return payload, bound, grid_plan_bound
    if payload.get("status") != "pass":
        raise GeometryQualificationError(
            "Grid has a passing candidate but is marked failed"
        )
    _, _, selected, selected_payload = max(passing)
    if payload.get("accepted_candidate") != selected.as_dict():
        raise GeometryQualificationError("Grid accepted candidate is not the fastest pass")
    selected_record = records[CANDIDATES.index(selected)]
    accepted_result = payload.get("accepted_result")
    if not isinstance(accepted_result, Mapping) or (
        accepted_result.get("artifact") != selected_record["result"]
        or accepted_result.get("measurements_sha256")
        != canonical_sha256(selected_payload["measurements"])
    ):
        raise GeometryQualificationError("Grid accepted result binding is invalid")
    return payload, bound, grid_plan_bound


def build_final_train_order_from_grid(
    *,
    grid_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish the selected 52.58B, 40/40/20 order from grid authority.

    The target is quantized down to 66,858 complete 192-row optimizer updates:
    12,836,736 rows and 52,579,270,656 consumed 4,096-token positions.  No row
    is repeated and no partial optimizer update is authorized.
    """

    grid, _, grid_plan_bound = _load_validated_grid_result(grid_result)
    grid_plan = read_bound_json(
        grid_plan_bound["artifact"], label="authenticated geometry grid plan"
    )
    if grid_plan.get("inputs", {}).get("common", {}).get("sequence_length") != 4_096:
        raise GeometryQualificationError(
            "Final selected-run constants require the frozen 4,096-token corpus"
        )
    accepted = grid.get("accepted_candidate")
    matches = [candidate for candidate in CANDIDATES if candidate.as_dict() == accepted]
    if len(matches) != 1:
        raise GeometryQualificationError("Grid accepted candidate is not canonical")
    candidate = matches[0]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = read_bound_json(
            artifact(manifest_path, label="existing final train order manifest"),
            label="existing final train order manifest",
        )
    else:
        if output_dir.exists() and (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise GeometryQualificationError(
                "Final train-order output must be absent or an empty real directory"
            )
        manifests = {
            domain: grid_plan["inputs"]["packed"][domain]["manifest"]["path"]
            for domain in training_data.DOMAIN_ORDER
        }
        manifest = training_data.build_training_order(
            manifests,
            output_dir,
            seed=int(grid_plan["settings"]["seed"]),
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            expected_total_input_tokens=FINAL_TRAIN_TARGET_INPUT_TOKENS,
            input_token_tolerance=UPDATE_ROWS * 4_096 - 1,
            frozen_global_microbatch_rows=candidate.global_microbatch_rows,
            frozen_gradient_accumulation_steps=(
                candidate.gradient_accumulation_steps
            ),
        )
    geometry = training_data.frozen_training_geometry(
        manifest_path, verify_checksum=True
    )
    if (
        manifest.get("split") != "train"
        or manifest.get("input_token_budget", {}).get("expected_total")
        != FINAL_TRAIN_TARGET_INPUT_TOKENS
        or manifest.get("expected_input_token_weights")
        != {"python": 0.4, "other_code": 0.4, "english": 0.2}
        or geometry["global_microbatch_rows"] != candidate.global_microbatch_rows
        or geometry["gradient_accumulation_steps"]
        != candidate.gradient_accumulation_steps
        or geometry["optimizer_update_rows"] != UPDATE_ROWS
        or geometry["optimizer_updates"]
        != FINAL_TRAIN_EXPECTED_OPTIMIZER_UPDATES
        or geometry["consumed_rows"] != FINAL_TRAIN_EXPECTED_ROWS
        or geometry["consumed_input_tokens"]
        != FINAL_TRAIN_EXPECTED_CONSUMED_INPUT_TOKENS
        or geometry.get("dropped_tail_rows") != 0
    ):
        raise GeometryQualificationError(
            "Final train order does not match the authenticated 52.58B selection"
        )
    for domain in training_data.DOMAIN_ORDER:
        record = manifest.get("dataset_manifests", {}).get(domain)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise GeometryQualificationError(
                f"Final train order lacks packed-manifest binding for {domain}"
            )
        current = artifact(
            manifest_path.parent / record["path"],
            label=f"final train {domain} packed manifest",
        )
        if current != grid_plan["inputs"]["packed"][domain]["manifest"]:
            raise GeometryQualificationError(
                f"Final train {domain} packed manifest differs from the grid"
            )
    return manifest


def build_final_plan(
    *,
    output_root: Path,
    train_order_manifest: Path,
    validation_order_manifest: Path,
    tokenizer_root: Path,
    hardware_contract: Path,
    single_gpu_baselines: Path,
    grid_result: Path,
    settings: SoakSettings,
    cli_script: Path,
    python_executable: Path = Path(sys.executable),
) -> dict[str, Any]:
    settings.validate()
    hardware, hardware_bound, hardware_identity = _validate_hardware_contract(
        hardware_contract
    )
    grid_payload, grid_bound, grid_plan_bound = _load_validated_grid_result(
        grid_result
    )
    grid_plan = read_bound_json(
        grid_plan_bound["artifact"], label="authenticated geometry grid plan"
    )
    accepted = grid_payload.get("accepted_candidate")
    if not isinstance(accepted, dict):
        raise GeometryQualificationError("Grid result lacks accepted_candidate")
    matches = [
        candidate
        for candidate in CANDIDATES
        if candidate.as_dict() == accepted
    ]
    if len(matches) != 1:
        raise GeometryQualificationError("Grid accepted candidate is not canonical")
    candidate = matches[0]
    train_descriptor = artifact(train_order_manifest, label="final train order manifest")
    train_payload = read_bound_json(train_descriptor, label="final train order manifest")
    geometry = training_data.frozen_training_geometry(
        train_order_manifest, verify_checksum=True
    )
    if (
        train_payload.get("split") != "train"
        or geometry["global_microbatch_rows"] != candidate.global_microbatch_rows
        or geometry["gradient_accumulation_steps"]
        != candidate.gradient_accumulation_steps
        or geometry["optimizer_updates"]
        <= settings.measurement_warmup_steps + settings.minimum_soak_steps
    ):
        raise GeometryQualificationError(
            "Final train order does not match the selected geometry or soak length"
        )
    tokenizer_sha = train_payload.get("tokenizer_manifest_sha256")
    tokenizer = verify_tokenizer_identity(
        tokenizer_root,
        expected_manifest_sha256=tokenizer_sha,
        expected_vocab_size=train_payload.get("vocab_size"),
    )
    common = {
        "sequence_length": train_payload["sequence_length"],
        "vocab_size": train_payload["vocab_size"],
        "eos_token_id": train_payload["eos_token_id"],
        "tokenizer": {
            "root": str(Path(tokenizer_root).resolve(strict=True)),
            "manifest": artifact(
                tokenizer.manifest_path, label="tokenizer manifest"
            ),
            "manifest_sha256": tokenizer.manifest_sha256,
            "vocabulary_sha256": tokenizer.vocabulary_sha256,
        },
    }
    validation = _validation_input(
        validation_order_manifest,
        common=common,
        eval_batches=settings.eval_batches,
    )
    _, baseline_bound = load_single_gpu_baselines(
        single_gpu_baselines,
        hardware_identity_sha256=hardware_identity["identity_sha256"],
        sequence_length=train_payload["sequence_length"],
    )
    runtime = {
        "python": _executable_identity(python_executable),
        "sources": _source_identity(cli_script),
    }
    grid_common = grid_plan["inputs"]["common"]
    if (
        dataclasses.asdict(settings) != grid_plan.get("settings")
        or hardware_identity["identity_sha256"]
        != grid_plan["inputs"]["hardware"]["geometry_identity"][
            "identity_sha256"
        ]
        or hardware_identity["identity"]
        != grid_plan["inputs"]["hardware"]["geometry_identity"]["identity"]
        or baseline_bound != grid_plan["inputs"]["single_gpu_baselines"]
        or runtime != grid_plan.get("runtime")
        or any(
            common[field] != grid_common.get(field)
            for field in ("sequence_length", "vocab_size", "eos_token_id", "tokenizer")
        )
    ):
        raise GeometryQualificationError(
            "Final soak stable hardware identity/settings/tokenizer/baseline/runtime "
            "differs from the grid"
        )
    dataset_manifests = train_payload.get("dataset_manifests")
    if not isinstance(dataset_manifests, Mapping) or set(dataset_manifests) != set(
        training_data.DOMAIN_ORDER
    ):
        raise GeometryQualificationError(
            "Final train order lacks the exact grid packed-manifest inventory"
        )
    for domain in training_data.DOMAIN_ORDER:
        record = dataset_manifests[domain]
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise GeometryQualificationError(
                f"Final train order packed manifest for {domain} is invalid"
            )
        final_dataset = artifact(
            Path(train_descriptor["path"]).parent / record["path"],
            label=f"final train {domain} packed manifest",
        )
        if final_dataset != grid_plan["inputs"]["packed"][domain]["manifest"]:
            raise GeometryQualificationError(
                f"Final train {domain} packed manifest differs from the grid"
            )
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "format_version": FORMAT_VERSION,
        "mode": "final-soak",
        "output_root": str(Path(output_root).resolve(strict=False)),
        "world_size": WORLD_SIZE,
        "effective_optimizer_update_rows": UPDATE_ROWS,
        "candidate": candidate.as_dict(),
        "settings": dataclasses.asdict(settings),
        "inputs": {
            "train_order": {
                "manifest": train_descriptor,
                "geometry": geometry,
            },
            "common": common,
            "validation_order": validation,
            "hardware": {
                "bound": hardware_bound,
                "expected": hardware,
                "geometry_identity": hardware_identity,
                "grid_hardware": grid_plan["inputs"]["hardware"],
            },
            "single_gpu_baselines": baseline_bound,
            "grid_result": grid_bound,
            "grid_plan": grid_plan_bound,
        },
        "runtime": runtime,
        "secrets_recorded": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def prepare_plan_root(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Create or authenticate the exact plan; never reuse another identity."""

    _validate_plan_self_hash(plan)
    expected_root = Path(str(plan.get("output_root", "")))
    if expected_root != root.resolve(strict=False):
        raise GeometryQualificationError("Plan output_root differs from requested root")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise GeometryQualificationError(f"Output root is unsafe: {root}")
    else:
        root.mkdir(parents=False)
        _fsync_directory(root.parent.resolve(strict=True))
    plan_path = root / "PLAN.json"
    if plan_path.exists() or plan_path.is_symlink():
        found, bound = verified_json_with_sidecar(plan_path, label="qualification plan")
        if found != dict(plan):
            raise GeometryQualificationError(
                "Output root belongs to another qualification identity"
            )
        return bound
    unexpected = [
        path.name
        for path in root.iterdir()
        if path.name != ".geometry-qualification.lock"
    ]
    if unexpected:
        raise GeometryQualificationError(
            f"Refusing to initialize non-empty qualification root: {sorted(unexpected)}"
        )
    return publish_json_new(plan_path, plan)


def ensure_grid_order(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
) -> dict[str, Any]:
    orders_root = root / "orders"
    _ensure_confined_child_directory(root, orders_root, label="diagnostic orders")
    order_dir = orders_root / candidate.candidate_id
    if order_dir.is_symlink() or (order_dir.exists() and not order_dir.is_dir()):
        raise GeometryQualificationError(
            f"Diagnostic order directory is unsafe: {order_dir}"
        )
    if order_dir.exists():
        _ensure_confined_child_directory(
            orders_root, order_dir, label="diagnostic candidate order", create=False
        )
    manifest_path = order_dir / "manifest.json"
    settings = SoakSettings(**plan["settings"])
    sequence_length = int(plan["inputs"]["common"]["sequence_length"])
    expected_tokens = (
        settings.diagnostic_optimizer_updates * UPDATE_ROWS * sequence_length
    )
    if manifest_path.is_symlink():
        raise GeometryQualificationError(
            f"Diagnostic order manifest must not be a symlink: {manifest_path}"
        )
    if manifest_path.exists():
        geometry = training_data.frozen_training_geometry(
            manifest_path, verify_checksum=True
        )
    else:
        if order_dir.exists():
            raise GeometryQualificationError(
                f"Partial diagnostic order must not be reused: {order_dir}"
            )
        manifests = {
            domain: plan["inputs"]["packed"][domain]["manifest"]["path"]
            for domain in training_data.DOMAIN_ORDER
        }
        training_data.build_training_order(
            manifests,
            order_dir,
            seed=settings.seed,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            expected_total_input_tokens=expected_tokens,
            input_token_tolerance=0,
            frozen_global_microbatch_rows=candidate.global_microbatch_rows,
            frozen_gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        )
        geometry = training_data.frozen_training_geometry(
            manifest_path, verify_checksum=True
        )
    if (
        geometry["global_microbatch_rows"] != candidate.global_microbatch_rows
        or geometry["gradient_accumulation_steps"]
        != candidate.gradient_accumulation_steps
        or geometry["optimizer_update_rows"] != UPDATE_ROWS
        or geometry["optimizer_updates"] != settings.diagnostic_optimizer_updates
        or geometry["consumed_input_tokens"] != expected_tokens
    ):
        raise GeometryQualificationError(
            f"Diagnostic order geometry differs for {candidate.candidate_id}"
        )
    descriptor = artifact(manifest_path, label="diagnostic order manifest")
    payload = read_bound_json(descriptor, label="diagnostic order manifest")
    order_path = manifest_path.parent / payload["order"]["path"]
    order_descriptor = artifact(order_path, label="diagnostic order payload")
    if order_descriptor["sha256"] != payload["order"]["sha256"]:
        raise GeometryQualificationError("Diagnostic order payload changed")
    return {
        "manifest": descriptor,
        "order": order_descriptor,
        "geometry": geometry,
    }


def ensure_baseline_order(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
) -> dict[str, Any]:
    """Build/reopen the exact one-rank counterpart of a six-rank candidate."""

    orders_root = root / "orders"
    _ensure_confined_child_directory(root, orders_root, label="baseline orders")
    order_dir = orders_root / candidate.candidate_id
    if order_dir.is_symlink() or (order_dir.exists() and not order_dir.is_dir()):
        raise GeometryQualificationError(
            f"Baseline order directory is unsafe: {order_dir}"
        )
    manifest_path = order_dir / "manifest.json"
    settings = SoakSettings(**plan["settings"])
    sequence_length = int(plan["inputs"]["common"]["sequence_length"])
    expected_tokens = (
        settings.diagnostic_optimizer_updates
        * BASELINE_UPDATE_ROWS
        * sequence_length
    )
    if manifest_path.is_symlink():
        raise GeometryQualificationError(
            f"Baseline order manifest must not be a symlink: {manifest_path}"
        )
    if manifest_path.exists():
        geometry = training_data.frozen_training_geometry(
            manifest_path, verify_checksum=True
        )
    else:
        if order_dir.exists():
            raise GeometryQualificationError(
                f"Partial baseline order must not be reused: {order_dir}"
            )
        manifests = {
            domain: plan["inputs"]["packed"][domain]["manifest"]["path"]
            for domain in training_data.DOMAIN_ORDER
        }
        training_data.build_training_order(
            manifests,
            order_dir,
            seed=settings.seed,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            expected_total_input_tokens=expected_tokens,
            input_token_tolerance=0,
            frozen_global_microbatch_rows=candidate.local_microbatch_rows,
            frozen_gradient_accumulation_steps=(
                candidate.gradient_accumulation_steps
            ),
        )
        geometry = training_data.frozen_training_geometry(
            manifest_path, verify_checksum=True
        )
    if (
        geometry["global_microbatch_rows"] != candidate.local_microbatch_rows
        or geometry["gradient_accumulation_steps"]
        != candidate.gradient_accumulation_steps
        or geometry["optimizer_update_rows"] != BASELINE_UPDATE_ROWS
        or geometry["optimizer_updates"] != settings.diagnostic_optimizer_updates
        or geometry["consumed_input_tokens"] != expected_tokens
    ):
        raise GeometryQualificationError(
            f"Baseline order geometry differs for {candidate.candidate_id}"
        )
    descriptor = artifact(manifest_path, label="baseline order manifest")
    payload = read_bound_json(descriptor, label="baseline order manifest")
    order_path = manifest_path.parent / payload["order"]["path"]
    order_descriptor = artifact(order_path, label="baseline order payload")
    if order_descriptor["sha256"] != payload["order"]["sha256"]:
        raise GeometryQualificationError("Baseline order payload changed")
    return {
        "manifest": descriptor,
        "order": order_descriptor,
        "geometry": geometry,
    }


def verify_order_binding(order: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile a manifest descriptor, payload bytes, and frozen geometry."""

    manifest_descriptor = order.get("manifest")
    expected_geometry = order.get("geometry")
    if not isinstance(manifest_descriptor, Mapping) or not isinstance(
        expected_geometry, Mapping
    ):
        raise GeometryQualificationError("Qualification order binding is incomplete")
    current_manifest = artifact(
        Path(str(manifest_descriptor.get("path", ""))), label="qualification order manifest"
    )
    if current_manifest != dict(manifest_descriptor):
        raise GeometryQualificationError("Qualification order manifest changed")
    payload = read_bound_json(current_manifest, label="qualification order manifest")
    order_record = payload.get("order")
    if not isinstance(order_record, Mapping) or not isinstance(
        order_record.get("path"), str
    ):
        raise GeometryQualificationError("Qualification order payload identity is missing")
    payload_path = Path(current_manifest["path"]).parent / order_record["path"]
    current_payload = artifact(payload_path, label="qualification order payload")
    if current_payload["sha256"] != order_record.get("sha256"):
        raise GeometryQualificationError("Qualification order payload changed")
    recorded_payload = order.get("order")
    if recorded_payload is not None and current_payload != recorded_payload:
        raise GeometryQualificationError("Qualification order descriptor changed")
    geometry = training_data.frozen_training_geometry(
        Path(current_manifest["path"]), verify_checksum=True
    )
    if geometry != dict(expected_geometry):
        raise GeometryQualificationError("Qualification frozen order geometry changed")
    return {
        "manifest": current_manifest,
        "order": current_payload,
        "geometry_sha256": canonical_sha256(geometry),
    }


def _decimal_string(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return "0" if rendered == "-0" else rendered


def _observation_payload(observation: PhaseObservation) -> dict[str, Any]:
    return {
        "start_monotonic_ns": observation.start_monotonic_ns,
        "start_step": observation.start_step,
        "start_consumed_input_tokens": observation.start_consumed_input_tokens,
        "last_step": observation.last_step,
        "last_consumed_input_tokens": observation.last_consumed_input_tokens,
        "validation_events_after_start": observation.validation_events_after_start,
        "wandb_log_events_after_start": observation.wandb_log_events_after_start,
        "data_wait_samples": observation.data_wait_samples,
        "maximum_data_wait_fraction": _decimal_string(
            observation.maximum_data_wait_fraction
        ),
        "peak_memory_allocated_bytes": observation.peak_memory_allocated_bytes,
        "peak_memory_reserved_bytes": observation.peak_memory_reserved_bytes,
        "stop_request_monotonic_ns": observation.stop_request_monotonic_ns,
        "stop_requested_after_step": observation.stop_requested_after_step,
        "json_metric_records": observation.json_metric_records,
        "graceful_stop_events": observation.graceful_stop_events,
        "wandb_failure": observation.wandb_failure,
    }


def _observation_from_payload(payload: Mapping[str, Any]) -> PhaseObservation:
    required = {
        "start_monotonic_ns",
        "start_step",
        "start_consumed_input_tokens",
        "last_step",
        "last_consumed_input_tokens",
        "validation_events_after_start",
        "wandb_log_events_after_start",
        "data_wait_samples",
        "maximum_data_wait_fraction",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "stop_request_monotonic_ns",
        "stop_requested_after_step",
        "json_metric_records",
        "graceful_stop_events",
        "wandb_failure",
    }
    if set(payload) != required:
        raise GeometryQualificationError("Phase observation journal schema is invalid")
    integer_fields = required - {"maximum_data_wait_fraction", "wandb_failure"}
    for field in integer_fields:
        value = payload[field]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise GeometryQualificationError(
                f"Phase observation journal {field} is invalid"
            )
    if payload["wandb_failure"] is not None and not isinstance(
        payload["wandb_failure"], str
    ):
        raise GeometryQualificationError("Phase observation W&B failure is invalid")
    return PhaseObservation(
        start_monotonic_ns=payload["start_monotonic_ns"],
        start_step=payload["start_step"],
        start_consumed_input_tokens=payload["start_consumed_input_tokens"],
        last_step=payload["last_step"],
        last_consumed_input_tokens=payload["last_consumed_input_tokens"],
        validation_events_after_start=payload["validation_events_after_start"],
        wandb_log_events_after_start=payload["wandb_log_events_after_start"],
        data_wait_samples=payload["data_wait_samples"],
        maximum_data_wait_fraction=_decimal(
            payload["maximum_data_wait_fraction"],
            field="journal maximum_data_wait_fraction",
            allow_zero=True,
        ),
        peak_memory_allocated_bytes=payload["peak_memory_allocated_bytes"],
        peak_memory_reserved_bytes=payload["peak_memory_reserved_bytes"],
        stop_request_monotonic_ns=payload["stop_request_monotonic_ns"],
        stop_requested_after_step=payload["stop_requested_after_step"],
        json_metric_records=payload["json_metric_records"],
        graceful_stop_events=payload["graceful_stop_events"],
        wandb_failure=payload["wandb_failure"],
    )


def _memory_payload(memory: GPUMemoryObservation) -> dict[str, Any]:
    return {
        "gpu_count": memory.gpu_count,
        "total_bytes_per_gpu": list(memory.total_bytes_per_gpu),
        "minimum_free_bytes_per_gpu": memory.minimum_free_bytes_per_gpu,
        "samples": memory.samples,
    }


def _memory_from_payload(
    payload: Mapping[str, Any], *, expected_gpu_count: int = WORLD_SIZE
) -> GPUMemoryObservation:
    if set(payload) != {
        "gpu_count",
        "total_bytes_per_gpu",
        "minimum_free_bytes_per_gpu",
        "samples",
    }:
        raise GeometryQualificationError("GPU memory journal schema is invalid")
    totals = payload["total_bytes_per_gpu"]
    if not isinstance(totals, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in totals
    ):
        raise GeometryQualificationError("GPU memory journal totals are invalid")
    memory = GPUMemoryObservation(
        gpu_count=payload["gpu_count"],
        total_bytes_per_gpu=tuple(totals),
        minimum_free_bytes_per_gpu=payload["minimum_free_bytes_per_gpu"],
        samples=payload["samples"],
    )
    if (
        memory.gpu_count != expected_gpu_count
        or len(memory.total_bytes_per_gpu) != expected_gpu_count
        or not isinstance(memory.minimum_free_bytes_per_gpu, int)
        or memory.minimum_free_bytes_per_gpu < 0
        or not isinstance(memory.samples, int)
        or memory.samples < 1
    ):
        raise GeometryQualificationError("GPU memory journal values are invalid")
    return memory


def _stat_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GeometryQualificationError(f"Checkpoint is not a regular file: {path}")
    metadata = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def _verify_stat_identity(identity: Mapping[str, Any], *, label: str) -> None:
    required = {"path", "device", "inode", "bytes", "mtime_ns", "ctime_ns"}
    if set(identity) != required or not isinstance(identity["path"], str):
        raise GeometryQualificationError(f"{label} identity schema is invalid")
    found = _stat_identity(Path(identity["path"]))
    if found != dict(identity):
        raise GeometryQualificationError(f"{label} changed between qualification phases")


def _remove_exact_stop_request(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise GeometryQualificationError(f"Stop request is unsafe: {path}")
    expected = f"{int(signal.SIGUSR1)}\n"
    if path.read_text(encoding="ascii", errors="strict") != expected:
        raise GeometryQualificationError(f"Stop request contents are invalid: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def _project_root_from_plan(plan: Mapping[str, Any]) -> Path:
    trainer = Path(plan["runtime"]["sources"]["trainer"]["path"])
    root = trainer.parents[1]
    if root.is_symlink() or not root.is_dir():
        raise GeometryQualificationError("Bound project root is unavailable")
    return root


def _candidate_failures(
    *,
    measurements: Mapping[str, Any],
    physical_memory_bytes: int,
) -> list[str]:
    failures: list[str] = []
    scaling = _decimal(measurements["scaling_efficiency"], field="scaling efficiency")
    wait = _decimal(
        measurements["data_wait_fraction"],
        field="data wait fraction",
        allow_zero=True,
    )
    required_free = max(
        MINIMUM_FREE_MEMORY_BYTES,
        int(
            (
                Decimal(physical_memory_bytes) * MINIMUM_FREE_MEMORY_FRACTION
            ).to_integral_value(rounding=ROUND_CEILING)
        ),
    )
    if scaling < MINIMUM_SCALING_EFFICIENCY or scaling > 1:
        failures.append("scaling_efficiency_outside_authorized_range")
    if wait > MAXIMUM_DATA_WAIT_FRACTION:
        failures.append("data_wait_fraction_above_authorized_maximum")
    if measurements["minimum_free_memory_bytes_per_gpu"] < required_free:
        failures.append("minimum_free_gpu_memory_below_authorized_margin")
    if not (
        0
        < measurements["peak_memory_allocated_bytes_per_gpu"]
        <= measurements["peak_memory_reserved_bytes_per_gpu"]
        <= physical_memory_bytes
    ):
        failures.append("allocator_memory_counters_invalid")
    if (
        measurements["peak_memory_reserved_bytes_per_gpu"]
        + measurements["minimum_free_memory_bytes_per_gpu"]
        > physical_memory_bytes
    ):
        failures.append("allocator_and_system_memory_counters_do_not_reconcile")
    external = measurements["throughput_measurement"]
    if external["validation_events"] < 1:
        failures.append("no_timed_validation_event")
    if external["checkpoint_events"] < 2:
        failures.append("timed_interval_did_not_cross_both_checkpoints")
    if external["wandb_log_events"] < 1:
        failures.append("no_timed_wandb_event")
    provenance = measurements.get("measurement_provenance")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("data_wait_samples"), int)
        or isinstance(provenance.get("data_wait_samples"), bool)
        or provenance["data_wait_samples"] < measurements["soak_steps"]
    ):
        failures.append("missing_timed_data_wait_measurements")
    if external["resume_verified"] is not True:
        failures.append("six_rank_resume_not_verified")
    if measurements["soak_steps"] < MINIMUM_SOAK_STEPS:
        failures.append("soak_too_short")
    return failures


def run_candidate(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
    order: Mapping[str, Any],
    baseline_rate: Decimal,
    environment: Mapping[str, str] = os.environ,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    gpu_sampler: Callable[[], Sequence[tuple[int, int]]] = sample_nvidia_smi_memory,
    phase_runner: Callable[..., PhaseRun] = supervise_phase,
    checkpoint_inspector: Callable[..., dict[str, Any]] = _checkpoint_summary,
    binding_verifier: Callable[..., dict[str, Any]] = verify_plan_artifacts,
    order_verifier: Callable[[Mapping[str, Any]], dict[str, Any]] = verify_order_binding,
    live_runtime_verifier: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or resume one exact six-rank candidate and publish its result."""

    settings = SoakSettings(**plan["settings"])
    settings.validate()
    starting_bindings = binding_verifier(plan=plan)
    starting_order = order_verifier(order)
    effective_gpu_sampler = gpu_sampler
    if gpu_sampler is sample_nvidia_smi_memory:
        try:
            nvidia_smi_path = Path(
                str(
                    plan["inputs"]["hardware"]["expected"]["qualification"][
                        "gpu"
                    ]["nvidia_smi_executable"]["path"]
                )
            )
        except (KeyError, TypeError) as exc:
            raise GeometryQualificationError(
                "Plan lacks its authenticated nvidia-smi executable"
            ) from exc

        def effective_gpu_sampler() -> Sequence[tuple[int, int]]:
            return sample_nvidia_smi_memory(nvidia_smi_path)

    result_path = root / "RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        payload, bound = verified_json_with_sidecar(
            result_path, label="candidate result"
        )
        if (
            payload.get("format") != CANDIDATE_RESULT_FORMAT
            or payload.get("candidate") != candidate.as_dict()
            or payload.get("plan_sha256") != plan["plan_sha256"]
        ):
            raise GeometryQualificationError("Existing candidate result has another identity")
        if binding_verifier(plan=plan) != starting_bindings or order_verifier(
            order
        ) != starting_order:
            raise GeometryQualificationError(
                "Qualification inputs changed after candidate publication"
            )
        return payload, bound
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise GeometryQualificationError(f"Candidate root is unsafe: {root}")
    else:
        root.mkdir(parents=True)
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    _ensure_confined_child_directory(root, logs, label="candidate logs")
    _ensure_confined_child_directory(root, checkpoints, label="candidate checkpoints")
    checkpoint = checkpoints / "last.pt"
    previous = checkpoints / "last.previous.pt"
    stop_request = root / "stop-request"
    journal_path = root / "JOURNAL.json"
    journal_sidecar = root / "JOURNAL.json.sha256"
    python_path = Path(plan["runtime"]["python"]["invocation_path"])
    validation_path = Path(
        plan["inputs"]["validation_order"]["manifest"]["path"]
    )
    tokenizer_root = Path(plan["inputs"]["common"]["tokenizer"]["root"])
    order_path = Path(order["manifest"]["path"])
    order_sha256 = order["manifest"]["sha256"]
    order_payload_sha256 = starting_order["order"]["sha256"]
    run_group = plan["plan_sha256"][:16]
    run_name = (
        f"{settings.wandb_run_name_prefix}-{candidate.candidate_id}-{run_group}"
    )
    command_one = render_torchrun_command(
        python_executable=python_path,
        order_manifest=order_path,
        validation_order_manifest=validation_path,
        tokenizer_root=tokenizer_root,
        checkpoint=checkpoint,
        candidate=candidate,
        settings=settings,
        run_name=run_name,
        run_group=run_group,
        resume=False,
    )
    command_two = render_torchrun_command(
        python_executable=python_path,
        order_manifest=order_path,
        validation_order_manifest=validation_path,
        tokenizer_root=tokenizer_root,
        checkpoint=checkpoint,
        candidate=candidate,
        settings=settings,
        run_name=run_name,
        run_group=run_group,
        resume=True,
    )
    project_root = _project_root_from_plan(plan)
    boot = _boot_id()
    phase_one_payload: dict[str, Any]
    if (
        journal_path.exists()
        or journal_path.is_symlink()
        or journal_sidecar.exists()
        or journal_sidecar.is_symlink()
    ):
        journal, _ = verified_json_with_sidecar(
            journal_path, label="candidate journal"
        )
        if (
            journal.get("format") != "six-gpu-geometry-candidate-journal"
            or journal.get("format_version") != FORMAT_VERSION
            or journal.get("plan_sha256") != plan["plan_sha256"]
            or journal.get("candidate") != candidate.as_dict()
            or journal.get("boot_id") != boot
            or journal.get("stage") != "phase-one-complete"
        ):
            raise GeometryQualificationError(
                "Candidate journal is incomplete, stale-boot, or belongs to another plan"
            )
        phase_one_payload = journal["phase_one"]
        required_phase_one = {
            "command",
            "command_sha256",
            "return_code",
            "completed_monotonic_ns",
            "observation",
            "gpu_memory",
            "log",
            "checkpoint_identity",
        }
        if (
            not isinstance(phase_one_payload, dict)
            or set(phase_one_payload) != required_phase_one
            or phase_one_payload.get("command") != command_one
            or phase_one_payload.get("command_sha256") != command_sha256(command_one)
            or not isinstance(phase_one_payload.get("return_code"), int)
            or isinstance(phase_one_payload.get("return_code"), bool)
            or phase_one_payload.get("return_code")
            != _EXPECTED_TORCHRUN_GRACEFUL_STOP_RETURN_CODE
            or not isinstance(phase_one_payload.get("completed_monotonic_ns"), int)
            or isinstance(phase_one_payload.get("completed_monotonic_ns"), bool)
        ):
            raise GeometryQualificationError("Candidate phase-one journal is invalid")
        journal_observation = _observation_from_payload(
            phase_one_payload["observation"]
        )
        _memory_from_payload(phase_one_payload["gpu_memory"])
        if (
            journal_observation.start_step != settings.measurement_warmup_steps
            or journal_observation.start_monotonic_ns is None
            or journal_observation.start_consumed_input_tokens is None
            or journal_observation.last_step <= journal_observation.start_step
            or journal_observation.stop_request_monotonic_ns is None
            or phase_one_payload["completed_monotonic_ns"]
            < journal_observation.stop_request_monotonic_ns
        ):
            raise GeometryQualificationError(
                "Candidate phase-one journal lacks a valid timed stop"
            )
        _verify_stat_identity(
            phase_one_payload["checkpoint_identity"], label="phase-one checkpoint"
        )
        if (
            phase_one_payload["checkpoint_identity"]["bytes"]
            > settings.checkpoint_generation_bytes
        ):
            raise GeometryQualificationError(
                "Phase-one checkpoint exceeds the frozen generation estimate"
            )
        verified_log = artifact(
            phase_one_payload["log"]["path"], label="phase-one log"
        )
        if verified_log != phase_one_payload["log"]:
            raise GeometryQualificationError("Phase-one log changed after journal commit")
        _verify_torchrun_graceful_exit(
            return_code=phase_one_payload["return_code"],
            log=phase_one_payload["log"],
            label="phase one",
        )
        if clock_ns() < journal_observation.start_monotonic_ns:
            raise GeometryQualificationError("Monotonic clock moved backwards across resume")
    else:
        if checkpoint.exists() or previous.exists() or (logs / "phase-one.log").exists():
            raise GeometryQualificationError(
                "Unjournaled candidate artifacts are poisoned; use a new output root"
            )
        phase_one = phase_runner(
            command=command_one,
            environment=environment,
            log_path=logs / "phase-one.log",
            stop_request_path=stop_request,
            working_directory=project_root,
            settings=settings,
            start_after_step=settings.measurement_warmup_steps,
            request_stop_after_step=(
                settings.measurement_warmup_steps + settings.stop_after_soak_steps
            ),
            clock_ns=clock_ns,
            gpu_sampler=effective_gpu_sampler,
        )
        observation = phase_one.observation
        _verify_torchrun_graceful_exit(
            return_code=phase_one.return_code,
            log=phase_one.log,
            label="phase one",
        )
        if (
            observation.start_monotonic_ns is None
            or observation.start_step != settings.measurement_warmup_steps
            or observation.start_consumed_input_tokens is None
            or observation.last_step <= observation.start_step
        ):
            raise GeometryQualificationError(
                "Phase one did not establish an aligned external timing start"
            )
        checkpoint_identity = _stat_identity(checkpoint)
        if checkpoint_identity["bytes"] > settings.checkpoint_generation_bytes:
            raise GeometryQualificationError(
                "Phase-one checkpoint exceeds the frozen generation estimate"
            )
        phase_one_payload = {
            "command": command_one,
            "command_sha256": command_sha256(command_one),
            "return_code": phase_one.return_code,
            "completed_monotonic_ns": phase_one.completed_monotonic_ns,
            "observation": _observation_payload(observation),
            "gpu_memory": _memory_payload(phase_one.gpu_memory),
            "log": phase_one.log,
            "checkpoint_identity": checkpoint_identity,
        }
        publish_json_new(
            journal_path,
            {
                "format": "six-gpu-geometry-candidate-journal",
                "format_version": FORMAT_VERSION,
                "stage": "phase-one-complete",
                "plan_sha256": plan["plan_sha256"],
                "candidate": candidate.as_dict(),
                "boot_id": boot,
                "phase_one": phase_one_payload,
            },
        )
    mid_bindings = binding_verifier(plan=plan)
    mid_order = order_verifier(order)
    if mid_bindings != starting_bindings or mid_order != starting_order:
        raise GeometryQualificationError(
            "Qualification inputs changed before the six-rank resume"
        )
    if live_runtime_verifier is not None:
        live_runtime_verifier(plan=plan, environment=environment)
    _remove_exact_stop_request(stop_request)
    if (logs / "phase-two.log").exists() or previous.exists():
        raise GeometryQualificationError(
            "Unjournaled phase-two artifacts are poisoned; use a new output root"
        )
    phase_one_observation = _observation_from_payload(
        phase_one_payload["observation"]
    )
    phase_two = phase_runner(
        command=command_two,
        environment=environment,
        log_path=logs / "phase-two.log",
        stop_request_path=stop_request,
        working_directory=project_root,
        settings=settings,
        start_after_step=None,
        request_stop_after_step=(
            phase_one_observation.start_step + settings.minimum_soak_steps
        ),
        prior_observation=phase_one_observation,
        clock_ns=clock_ns,
        gpu_sampler=effective_gpu_sampler,
    )
    _verify_torchrun_graceful_exit(
        return_code=phase_two.return_code,
        log=phase_two.log,
        label="phase two",
    )
    _remove_exact_stop_request(stop_request)
    phase_two_observation = phase_two.observation
    if phase_two_observation.start_monotonic_ns is None:
        raise GeometryQualificationError("Phase-two lost the external timing origin")
    mid_checkpoint = checkpoint_inspector(
        previous,
        expected_order_sha256=order_sha256,
        expected_order_payload_sha256=order_payload_sha256,
    )
    final_checkpoint = checkpoint_inspector(
        checkpoint,
        expected_order_sha256=order_sha256,
        expected_order_payload_sha256=order_payload_sha256,
    )
    if max(mid_checkpoint["bytes"], final_checkpoint["bytes"]) > (
        settings.checkpoint_generation_bytes
    ):
        raise GeometryQualificationError(
            "Observed checkpoint exceeds the frozen generation estimate"
        )
    if (
        mid_checkpoint["completed_steps"] != phase_one_observation.last_step
        or mid_checkpoint["consumed_input_tokens"]
        != phase_one_observation.last_consumed_input_tokens
        or final_checkpoint["completed_steps"] != phase_two_observation.last_step
        or final_checkpoint["consumed_input_tokens"]
        != phase_two_observation.last_consumed_input_tokens
        or final_checkpoint["completed_steps"] <= mid_checkpoint["completed_steps"]
        or final_checkpoint["wandb_run_id_sha256"]
        != mid_checkpoint["wandb_run_id_sha256"]
        or any(
            final_checkpoint[field] != mid_checkpoint[field]
            for field in (
                "training_geometry_sha256",
                "trajectory_sha256",
                "runtime_signature_sha256",
                "implementation_signature_sha256",
                "data_identity_sha256",
            )
        )
    ):
        raise GeometryQualificationError(
            "Checkpoint counters/W&B/trajectory identity do not prove exact stop and resume"
        )
    expected_implementation = {
        "trainer_class": "pretrain.train.Trainer",
        "model_class": "pretrain.model.CausalLM",
        "trainer_source_sha256": plan["runtime"]["sources"]["trainer"]["sha256"],
        "model_source_sha256": plan["runtime"]["sources"]["model"]["sha256"],
        "data_source_sha256": plan["runtime"]["sources"]["data"]["sha256"],
    }
    if (
        final_checkpoint["training_geometry_sha256"]
        != canonical_sha256(order["geometry"])
        or final_checkpoint["implementation_signature_sha256"]
        != canonical_sha256(expected_implementation)
    ):
        raise GeometryQualificationError(
            "Checkpoint geometry/implementation differs from the frozen plan"
        )
    start_step = phase_one_observation.start_step
    start_tokens = phase_one_observation.start_consumed_input_tokens
    assert start_step is not None and start_tokens is not None
    end_step = final_checkpoint["completed_steps"]
    end_tokens = final_checkpoint["consumed_input_tokens"]
    soak_steps = end_step - start_step
    token_delta = end_tokens - start_tokens
    sequence_length = int(plan["inputs"]["common"]["sequence_length"])
    tokens_per_update = UPDATE_ROWS * sequence_length
    if soak_steps < settings.minimum_soak_steps or token_delta != soak_steps * tokens_per_update:
        raise GeometryQualificationError(
            "External token delta does not match the completed optimizer updates"
        )
    elapsed_ns = phase_two.completed_monotonic_ns - phase_one_observation.start_monotonic_ns
    if elapsed_ns < 1:
        raise GeometryQualificationError("External monotonic interval is invalid")
    throughput = Decimal(token_delta) * Decimal(1_000_000_000) / Decimal(elapsed_ns)
    scaling = throughput / (Decimal(WORLD_SIZE) * baseline_rate)
    phase_one_memory = _memory_from_payload(phase_one_payload["gpu_memory"])
    combined_memory = phase_one_memory.merge(phase_two.gpu_memory)
    physical_memory = int(plan["inputs"]["hardware"]["expected"]["gpu_memory_bytes"])
    expected_smi_memory = _expected_nvidia_smi_memory(
        plan["inputs"]["hardware"]["expected"]
    )
    if (
        expected_smi_memory is not None
        and combined_memory.total_bytes_per_gpu != expected_smi_memory
    ):
        raise GeometryQualificationError(
            "Live nvidia-smi memory totals differ from the hardware contract"
        )
    raw_minimum_free = combined_memory.minimum_free_bytes_per_gpu
    assert raw_minimum_free is not None
    minimum_free = _normalized_nvidia_free_memory(
        list(zip(combined_memory.total_bytes_per_gpu, [raw_minimum_free] * WORLD_SIZE)),
        physical_memory_bytes=physical_memory,
    )
    combined_observation = phase_one_observation.merge(phase_two_observation)
    if combined_observation.wandb_failure is not None:
        raise GeometryQualificationError("W&B failed during the timed soak")
    phase_one_latency_ns = (
        phase_one_payload["completed_monotonic_ns"]
        - phase_one_observation.stop_request_monotonic_ns
    )
    phase_two_latency_ns = (
        phase_two.completed_monotonic_ns
        - phase_two_observation.stop_request_monotonic_ns
    )
    if min(phase_one_latency_ns, phase_two_latency_ns) < 0:
        raise GeometryQualificationError("Checkpoint stop latency is negative")
    checkpoint_seconds = Decimal(
        max(phase_one_latency_ns, phase_two_latency_ns)
    ) / Decimal(1_000_000_000)
    wandb_inventory = _wandb_inventory(checkpoints)
    ending_bindings = binding_verifier(plan=plan)
    ending_order = order_verifier(order)
    if ending_bindings != starting_bindings or ending_order != starting_order:
        raise GeometryQualificationError(
            "Qualification inputs changed during the timed candidate"
        )
    measurements: dict[str, Any] = {
        "aggregate_input_tokens_per_second": _decimal_string(throughput),
        "peak_memory_allocated_bytes_per_gpu": max(
            phase_one_observation.peak_memory_allocated_bytes,
            phase_two_observation.peak_memory_allocated_bytes,
        ),
        "peak_memory_reserved_bytes_per_gpu": max(
            phase_one_observation.peak_memory_reserved_bytes,
            phase_two_observation.peak_memory_reserved_bytes,
        ),
        "minimum_free_memory_bytes_per_gpu": minimum_free,
        "checkpoint_seconds": _decimal_string(checkpoint_seconds),
        "data_wait_fraction": _decimal_string(
            max(
                phase_one_observation.maximum_data_wait_fraction,
                phase_two_observation.maximum_data_wait_fraction,
            )
        ),
        "scaling_efficiency": _decimal_string(scaling),
        "soak_steps": soak_steps,
        "throughput_measurement": {
            "scope": THROUGHPUT_SCOPE,
            "timer": THROUGHPUT_TIMER,
            "counter": THROUGHPUT_COUNTER,
            "start_consumed_input_tokens": start_tokens,
            "end_consumed_input_tokens": end_tokens,
            "elapsed_wall_time_ns": elapsed_ns,
            "validation_events": (
                phase_one_observation.validation_events_after_start
                + phase_two_observation.validation_events_after_start
            ),
            "checkpoint_events": 2,
            "wandb_log_events": (
                phase_one_observation.wandb_log_events_after_start
                + phase_two_observation.wandb_log_events_after_start
            ),
            "resume_verified": True,
        },
        "measurement_provenance": {
            "throughput_source": "external-supervisor-not-trainer-step-rate",
            "allocator_peak_scope": "rank-zero-trainer-telemetry",
            "minimum_free_scope": "minimum-across-six-gpus-nvidia-smi",
            "nvidia_smi_total_bytes_per_gpu": list(
                combined_memory.total_bytes_per_gpu
            ),
            "nvidia_smi_raw_minimum_free_bytes_per_gpu": raw_minimum_free,
            "minimum_free_normalization": (
                "min(raw_free, cuda_physical_bytes - nvidia_smi_used_bytes)"
            ),
            "gpu_memory_samples": combined_memory.samples,
            "data_wait_samples": combined_observation.data_wait_samples,
            "single_gpu_baseline_input_tokens_per_second": _decimal_string(
                baseline_rate
            ),
            "wandb_artifacts": wandb_inventory,
            "mid_checkpoint": mid_checkpoint,
            "final_checkpoint": final_checkpoint,
            "phase_one_log": phase_one_payload["log"],
            "phase_two_log": phase_two.log,
            "phase_one_command_sha256": phase_one_payload["command_sha256"],
            "phase_two_command_sha256": command_sha256(command_two),
            "phase_one_torchrun_return_code": phase_one_payload["return_code"],
            "phase_two_torchrun_return_code": phase_two.return_code,
            "plan_artifact_verification": ending_bindings,
            "order_verification": ending_order,
        },
    }
    failures = _candidate_failures(
        measurements=measurements,
        physical_memory_bytes=physical_memory,
    )
    result: dict[str, Any] = {
        "format": CANDIDATE_RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "pass" if not failures else "fail",
        "plan_sha256": plan["plan_sha256"],
        "candidate": candidate.as_dict(),
        "order": dict(order),
        "measurements": measurements,
        "failures": failures,
        "commands": {
            "phase_one": command_one,
            "phase_two": command_two,
        },
        "secrets_recorded": False,
    }
    bound = publish_json_new(result_path, result)
    return result, bound


def _load_baseline_rates_from_plan(
    plan: Mapping[str, Any],
) -> dict[str, Decimal]:
    baseline_path = Path(
        plan["inputs"]["single_gpu_baselines"]["artifact"]["path"]
    )
    rates, bound = load_single_gpu_baselines(
        baseline_path,
        hardware_identity_sha256=plan["inputs"]["hardware"][
            "geometry_identity"
        ]["identity_sha256"],
        sequence_length=int(plan["inputs"]["common"]["sequence_length"]),
    )
    if bound != plan["inputs"]["single_gpu_baselines"]:
        raise GeometryQualificationError(
            "Single-GPU baselines changed after plan publication"
        )
    return rates


def _candidate_failure_result(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
    error: BaseException,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "FAILURE.json"
    if path.exists() or path.is_symlink():
        payload, bound = verified_json_with_sidecar(path, label="candidate failure")
        if (
            payload.get("plan_sha256") != plan["plan_sha256"]
            or payload.get("candidate") != candidate.as_dict()
        ):
            raise GeometryQualificationError(
                "Existing candidate failure belongs to another identity"
            )
        return payload, bound
    root.mkdir(parents=True, exist_ok=True)
    message = redact_text(f"{type(error).__name__}: {error}", environment)
    payload = {
        "format": CANDIDATE_RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "fail",
        "plan_sha256": plan["plan_sha256"],
        "candidate": candidate.as_dict(),
        "failures": ["candidate_execution_or_evidence_failure"],
        "failure": {
            "type": type(error).__name__,
            "message": message,
        },
        "secrets_recorded": False,
    }
    return payload, publish_json_new(path, payload)


def _baseline_failure_result(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
    error: BaseException,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "FAILURE.json"
    if path.exists() or path.is_symlink():
        payload, bound = verified_json_with_sidecar(
            path, label="single-GPU baseline failure"
        )
        if (
            payload.get("plan_sha256") != plan["plan_sha256"]
            or payload.get("candidate") != candidate.as_dict()
        ):
            raise GeometryQualificationError(
                "Existing baseline failure belongs to another identity"
            )
        return payload, bound
    payload = {
        "format": BASELINE_FAILURE_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "fail",
        "plan_sha256": plan["plan_sha256"],
        "candidate": candidate.as_dict(),
        "failure": {
            "type": type(error).__name__,
            "message": redact_text(f"{error}", environment),
        },
        "secrets_recorded": False,
    }
    return payload, publish_json_new(path, payload)


def run_baseline_candidate(
    *,
    root: Path,
    plan: Mapping[str, Any],
    candidate: CandidateSpec,
    order: Mapping[str, Any],
    environment: Mapping[str, str] = os.environ,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    phase_runner: Callable[..., PhaseRun] = supervise_phase,
    checkpoint_inspector: Callable[..., dict[str, Any]] = _checkpoint_summary,
    binding_verifier: Callable[..., dict[str, Any]] = verify_plan_artifacts,
    order_verifier: Callable[[Mapping[str, Any]], dict[str, Any]] = verify_order_binding,
    gpu_sampler: Callable[[], Sequence[tuple[int, int]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure one one-GPU counterpart without any hand-edited evidence.

    The baseline deliberately crosses a durable checkpoint and resumes from it.
    That gives the denominator the same external end-to-end scope as the
    six-GPU candidate: validation, W&B, checkpoint I/O, process restart, and
    resume are all inside the measured monotonic interval.
    """

    settings = SoakSettings(**plan["settings"])
    settings.validate()
    starting_bindings = binding_verifier(plan=plan)
    starting_order = order_verifier(order)
    result_path = root / "RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        payload, bound = verified_json_with_sidecar(
            result_path, label="single-GPU baseline result"
        )
        if (
            payload.get("format") != BASELINE_CANDIDATE_RESULT_FORMAT
            or payload.get("plan_sha256") != plan["plan_sha256"]
            or payload.get("candidate") != candidate.as_dict()
            or payload.get("status") != "pass"
        ):
            raise GeometryQualificationError(
                "Existing single-GPU result has another identity"
            )
        if binding_verifier(plan=plan) != starting_bindings or order_verifier(
            order
        ) != starting_order:
            raise GeometryQualificationError(
                "Baseline inputs changed after candidate publication"
            )
        return payload, bound
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise GeometryQualificationError(f"Baseline root is unsafe: {root}")
    else:
        root.mkdir(parents=True)
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    _ensure_confined_child_directory(root, logs, label="baseline logs")
    _ensure_confined_child_directory(root, checkpoints, label="baseline checkpoints")
    checkpoint = checkpoints / "last.pt"
    previous = checkpoints / "last.previous.pt"
    stop_request = root / "stop-request"
    journal_path = root / "JOURNAL.json"
    journal_sidecar = root / "JOURNAL.json.sha256"
    baseline_gpu = plan["baseline_gpu"]
    try:
        visible_devices = plan["inputs"]["hardware"]["expected"]["qualification"][
            "host"
        ]["environment"]["cuda_visible_devices"]
        selected_visible = visible_devices[baseline_gpu["visible_index"]]
        physical_index = int(baseline_gpu["physical_index"])
        nvidia_smi_path = Path(
            plan["inputs"]["hardware"]["expected"]["qualification"]["gpu"][
                "nvidia_smi_executable"
            ]["path"]
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise GeometryQualificationError(
            "Baseline plan lacks its selected GPU/runtime evidence"
        ) from exc
    child_environment = dict(environment)
    child_environment["CUDA_VISIBLE_DEVICES"] = str(selected_visible)
    if gpu_sampler is None:

        def selected_gpu_sampler() -> Sequence[tuple[int, int]]:
            rows = sample_nvidia_smi_memory(nvidia_smi_path)
            try:
                return [rows[physical_index]]
            except IndexError as exc:
                raise GeometryQualificationError(
                    "Selected physical GPU disappeared during baseline"
                ) from exc

        gpu_sampler = selected_gpu_sampler
    python_path = Path(plan["runtime"]["python"]["invocation_path"])
    validation_path = Path(
        plan["inputs"]["validation_order"]["manifest"]["path"]
    )
    tokenizer_root = Path(plan["inputs"]["common"]["tokenizer"]["root"])
    order_path = Path(order["manifest"]["path"])
    run_group = plan["plan_sha256"][:16]
    run_name = (
        f"{settings.wandb_run_name_prefix}-baseline-"
        f"{candidate.candidate_id}-{run_group}"
    )
    command_one = render_torchrun_command(
        python_executable=python_path,
        order_manifest=order_path,
        validation_order_manifest=validation_path,
        tokenizer_root=tokenizer_root,
        checkpoint=checkpoint,
        candidate=candidate,
        settings=settings,
        run_name=run_name,
        run_group=run_group,
        resume=False,
        world_size=BASELINE_WORLD_SIZE,
        global_microbatch_rows=candidate.local_microbatch_rows,
        wandb_tag="single-gpu-geometry-baseline",
    )
    command_two = render_torchrun_command(
        python_executable=python_path,
        order_manifest=order_path,
        validation_order_manifest=validation_path,
        tokenizer_root=tokenizer_root,
        checkpoint=checkpoint,
        candidate=candidate,
        settings=settings,
        run_name=run_name,
        run_group=run_group,
        resume=True,
        world_size=BASELINE_WORLD_SIZE,
        global_microbatch_rows=candidate.local_microbatch_rows,
        wandb_tag="single-gpu-geometry-baseline",
    )
    project_root = _project_root_from_plan(plan)
    boot = _boot_id()
    phase_one_payload: dict[str, Any]
    if (
        journal_path.exists()
        or journal_path.is_symlink()
        or journal_sidecar.exists()
        or journal_sidecar.is_symlink()
    ):
        journal, _ = verified_json_with_sidecar(
            journal_path, label="single-GPU baseline journal"
        )
        if (
            journal.get("format") != "single-gpu-geometry-baseline-journal"
            or journal.get("format_version") != FORMAT_VERSION
            or journal.get("plan_sha256") != plan["plan_sha256"]
            or journal.get("candidate") != candidate.as_dict()
            or journal.get("boot_id") != boot
            or journal.get("stage") != "phase-one-complete"
        ):
            raise GeometryQualificationError(
                "Baseline journal is incomplete, stale-boot, or belongs to another plan"
            )
        phase_one_payload = journal.get("phase_one")
        required_phase_one = {
            "command",
            "command_sha256",
            "return_code",
            "completed_monotonic_ns",
            "observation",
            "gpu_memory",
            "log",
            "checkpoint_identity",
        }
        if (
            not isinstance(phase_one_payload, dict)
            or set(phase_one_payload) != required_phase_one
            or phase_one_payload.get("command") != command_one
            or phase_one_payload.get("command_sha256")
            != command_sha256(command_one)
            or phase_one_payload.get("return_code")
            != _EXPECTED_TORCHRUN_GRACEFUL_STOP_RETURN_CODE
            or not isinstance(phase_one_payload.get("completed_monotonic_ns"), int)
            or isinstance(phase_one_payload.get("completed_monotonic_ns"), bool)
        ):
            raise GeometryQualificationError("Baseline phase-one journal is invalid")
        journal_observation = _observation_from_payload(
            phase_one_payload["observation"]
        )
        _memory_from_payload(
            phase_one_payload["gpu_memory"],
            expected_gpu_count=BASELINE_WORLD_SIZE,
        )
        if (
            journal_observation.start_step != settings.measurement_warmup_steps
            or journal_observation.start_monotonic_ns is None
            or journal_observation.start_consumed_input_tokens is None
            or journal_observation.last_step <= journal_observation.start_step
            or journal_observation.stop_request_monotonic_ns is None
            or phase_one_payload["completed_monotonic_ns"]
            < journal_observation.stop_request_monotonic_ns
        ):
            raise GeometryQualificationError(
                "Baseline phase-one journal lacks a valid timed stop"
            )
        _verify_stat_identity(
            phase_one_payload["checkpoint_identity"],
            label="baseline phase-one checkpoint",
        )
        if (
            phase_one_payload["checkpoint_identity"]["bytes"]
            > settings.checkpoint_generation_bytes
        ):
            raise GeometryQualificationError(
                "Baseline phase-one checkpoint exceeds the frozen generation estimate"
            )
        verified_log = artifact(
            phase_one_payload["log"]["path"], label="baseline phase-one log"
        )
        if verified_log != phase_one_payload["log"]:
            raise GeometryQualificationError(
                "Baseline phase-one log changed after journal commit"
            )
        _verify_torchrun_graceful_exit(
            return_code=phase_one_payload["return_code"],
            log=phase_one_payload["log"],
            label="single-GPU baseline phase one",
        )
        if clock_ns() < journal_observation.start_monotonic_ns:
            raise GeometryQualificationError(
                "Monotonic clock moved backwards across baseline resume"
            )
    else:
        if (
            checkpoint.exists()
            or previous.exists()
            or (logs / "phase-one.log").exists()
        ):
            raise GeometryQualificationError(
                "Unjournaled baseline artifacts are poisoned; use a new output root"
            )
        phase_one = phase_runner(
            command=command_one,
            environment=child_environment,
            log_path=logs / "phase-one.log",
            stop_request_path=stop_request,
            working_directory=project_root,
            settings=settings,
            start_after_step=settings.measurement_warmup_steps,
            request_stop_after_step=(
                settings.measurement_warmup_steps + settings.stop_after_soak_steps
            ),
            expected_gpu_count=BASELINE_WORLD_SIZE,
            clock_ns=clock_ns,
            gpu_sampler=gpu_sampler,
        )
        observation = phase_one.observation
        _verify_torchrun_graceful_exit(
            return_code=phase_one.return_code,
            log=phase_one.log,
            label="single-GPU baseline phase one",
        )
        if (
            observation.start_monotonic_ns is None
            or observation.start_step != settings.measurement_warmup_steps
            or observation.start_consumed_input_tokens is None
            or observation.last_step <= observation.start_step
            or observation.stop_request_monotonic_ns is None
        ):
            raise GeometryQualificationError(
                "Baseline phase one did not establish an aligned timed stop"
            )
        checkpoint_identity = _stat_identity(checkpoint)
        if checkpoint_identity["bytes"] > settings.checkpoint_generation_bytes:
            raise GeometryQualificationError(
                "Baseline phase-one checkpoint exceeds the frozen generation estimate"
            )
        phase_one_payload = {
            "command": command_one,
            "command_sha256": command_sha256(command_one),
            "return_code": phase_one.return_code,
            "completed_monotonic_ns": phase_one.completed_monotonic_ns,
            "observation": _observation_payload(observation),
            "gpu_memory": _memory_payload(phase_one.gpu_memory),
            "log": phase_one.log,
            "checkpoint_identity": checkpoint_identity,
        }
        publish_json_new(
            journal_path,
            {
                "format": "single-gpu-geometry-baseline-journal",
                "format_version": FORMAT_VERSION,
                "stage": "phase-one-complete",
                "plan_sha256": plan["plan_sha256"],
                "candidate": candidate.as_dict(),
                "boot_id": boot,
                "phase_one": phase_one_payload,
            },
        )
    mid_bindings = binding_verifier(plan=plan)
    mid_order = order_verifier(order)
    if mid_bindings != starting_bindings or mid_order != starting_order:
        raise GeometryQualificationError(
            "Baseline inputs changed before the one-rank resume"
        )
    _remove_exact_stop_request(stop_request)
    if (logs / "phase-two.log").exists() or previous.exists():
        raise GeometryQualificationError(
            "Unjournaled baseline phase-two artifacts are poisoned; use a new root"
        )
    phase_one_observation = _observation_from_payload(
        phase_one_payload["observation"]
    )
    phase_two = phase_runner(
        command=command_two,
        environment=child_environment,
        log_path=logs / "phase-two.log",
        stop_request_path=stop_request,
        working_directory=project_root,
        settings=settings,
        start_after_step=None,
        request_stop_after_step=(
            phase_one_observation.start_step + settings.minimum_soak_steps
        ),
        prior_observation=phase_one_observation,
        expected_gpu_count=BASELINE_WORLD_SIZE,
        clock_ns=clock_ns,
        gpu_sampler=gpu_sampler,
    )
    _verify_torchrun_graceful_exit(
        return_code=phase_two.return_code,
        log=phase_two.log,
        label="single-GPU baseline phase two",
    )
    _remove_exact_stop_request(stop_request)
    mid_checkpoint = checkpoint_inspector(
        previous,
        expected_order_sha256=order["manifest"]["sha256"],
        expected_order_payload_sha256=starting_order["order"]["sha256"],
        expected_world_size=BASELINE_WORLD_SIZE,
    )
    final_checkpoint = checkpoint_inspector(
        checkpoint,
        expected_order_sha256=order["manifest"]["sha256"],
        expected_order_payload_sha256=starting_order["order"]["sha256"],
        expected_world_size=BASELINE_WORLD_SIZE,
    )
    phase_two_observation = phase_two.observation
    if (
        phase_two_observation.start_monotonic_ns is None
        or mid_checkpoint["completed_steps"] != phase_one_observation.last_step
        or mid_checkpoint["consumed_input_tokens"]
        != phase_one_observation.last_consumed_input_tokens
        or final_checkpoint["completed_steps"] != phase_two_observation.last_step
        or final_checkpoint["consumed_input_tokens"]
        != phase_two_observation.last_consumed_input_tokens
        or final_checkpoint["completed_steps"] <= mid_checkpoint["completed_steps"]
        or final_checkpoint["world_size"] != BASELINE_WORLD_SIZE
        or mid_checkpoint["world_size"] != BASELINE_WORLD_SIZE
        or final_checkpoint["wandb_run_id_sha256"]
        != mid_checkpoint["wandb_run_id_sha256"]
        or any(
            final_checkpoint[field] != mid_checkpoint[field]
            for field in (
                "training_geometry_sha256",
                "trajectory_sha256",
                "runtime_signature_sha256",
                "implementation_signature_sha256",
                "data_identity_sha256",
            )
        )
    ):
        raise GeometryQualificationError(
            "Baseline checkpoint counters/W&B/trajectory do not prove exact resume"
        )
    expected_implementation = {
        "trainer_class": "pretrain.train.Trainer",
        "model_class": "pretrain.model.CausalLM",
        "trainer_source_sha256": plan["runtime"]["sources"]["trainer"]["sha256"],
        "model_source_sha256": plan["runtime"]["sources"]["model"]["sha256"],
        "data_source_sha256": plan["runtime"]["sources"]["data"]["sha256"],
    }
    if (
        final_checkpoint["training_geometry_sha256"]
        != canonical_sha256(order["geometry"])
        or final_checkpoint["implementation_signature_sha256"]
        != canonical_sha256(expected_implementation)
    ):
        raise GeometryQualificationError(
            "Baseline checkpoint geometry/implementation differs from the plan"
        )
    start_step = phase_one_observation.start_step
    start_tokens = phase_one_observation.start_consumed_input_tokens
    assert start_step is not None and start_tokens is not None
    soak_steps = final_checkpoint["completed_steps"] - start_step
    end_tokens = final_checkpoint["consumed_input_tokens"]
    token_delta = end_tokens - start_tokens
    sequence_length = int(plan["inputs"]["common"]["sequence_length"])
    expected_delta = soak_steps * BASELINE_UPDATE_ROWS * sequence_length
    elapsed_ns = (
        phase_two.completed_monotonic_ns
        - phase_one_observation.start_monotonic_ns
    )
    combined_observation = phase_one_observation.merge(phase_two_observation)
    phase_one_memory = _memory_from_payload(
        phase_one_payload["gpu_memory"],
        expected_gpu_count=BASELINE_WORLD_SIZE,
    )
    combined_memory = phase_one_memory.merge(phase_two.gpu_memory)
    expected_memory = plan["inputs"]["hardware"]["expected"]["qualification"][
        "gpu"
    ]["devices"][baseline_gpu["visible_index"]].get("nvidia_smi_memory_bytes")
    if (
        expected_memory is not None
        and combined_memory.total_bytes_per_gpu != (expected_memory,)
    ):
        raise GeometryQualificationError(
            "Baseline nvidia-smi memory differs from the selected GPU contract"
        )
    if (
        soak_steps < settings.minimum_soak_steps
        or token_delta != expected_delta
        or elapsed_ns < 1
        or combined_observation.validation_events_after_start < 1
        or combined_observation.wandb_log_events_after_start < 1
        or combined_observation.data_wait_samples < soak_steps
        or max(mid_checkpoint["bytes"], final_checkpoint["bytes"])
        > settings.checkpoint_generation_bytes
        or combined_observation.wandb_failure is not None
    ):
        raise GeometryQualificationError(
            "Single-GPU baseline lacks timed validation/resume/checkpoint/W&B evidence"
        )
    throughput = Decimal(token_delta) * Decimal(1_000_000_000) / Decimal(elapsed_ns)
    ending_bindings = binding_verifier(plan=plan)
    ending_order = order_verifier(order)
    if ending_bindings != starting_bindings or ending_order != starting_order:
        raise GeometryQualificationError(
            "Baseline inputs changed during the timed candidate"
        )
    measurement = {
        "global_microbatch_rows": candidate.local_microbatch_rows,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "compile_model": candidate.compile_model,
        "gpu_count": BASELINE_WORLD_SIZE,
        "scope": THROUGHPUT_SCOPE,
        "timer": THROUGHPUT_TIMER,
        "counter": THROUGHPUT_COUNTER,
        "start_consumed_input_tokens": start_tokens,
        "end_consumed_input_tokens": end_tokens,
        "elapsed_wall_time_ns": elapsed_ns,
        "aggregate_input_tokens_per_second": _decimal_string(throughput),
    }
    payload = {
        "format": BASELINE_CANDIDATE_RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "pass",
        "plan_sha256": plan["plan_sha256"],
        "candidate": candidate.as_dict(),
        "baseline_gpu": dict(baseline_gpu),
        "order": dict(order),
        "measurement": measurement,
        "evidence": {
            "soak_steps": soak_steps,
            "validation_events": combined_observation.validation_events_after_start,
            "wandb_log_events": combined_observation.wandb_log_events_after_start,
            "data_wait_samples": combined_observation.data_wait_samples,
            "resume_verified": True,
            "gpu_memory": _memory_payload(combined_memory),
            "wandb_artifacts": _wandb_inventory(checkpoints),
            "mid_checkpoint": mid_checkpoint,
            "final_checkpoint": final_checkpoint,
            "phase_one_log": phase_one_payload["log"],
            "phase_two_log": phase_two.log,
            "phase_one_command_sha256": phase_one_payload["command_sha256"],
            "phase_two_command_sha256": command_sha256(command_two),
            "plan_artifact_verification": ending_bindings,
            "order_verification": ending_order,
        },
        "commands": {"phase_one": command_one, "phase_two": command_two},
        "secrets_recorded": False,
    }
    return payload, publish_json_new(result_path, payload)


def run_single_gpu_baselines(
    *,
    root: Path,
    plan: Mapping[str, Any],
    environment: Mapping[str, str] = os.environ,
    candidate_runner: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        run_baseline_candidate
    ),
    runtime_verifier: Callable[..., dict[str, Any]] = verify_live_runtime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all six automatic baselines and publish the aggregate receipt."""

    if plan.get("mode") != "single-gpu-baselines":
        raise GeometryQualificationError(
            "run_single_gpu_baselines requires a baseline plan"
        )
    verify_baseline_plan_authority(plan)
    runtime_verifier(plan=plan, environment=environment)
    receipt_path = root / "single-gpu-baselines.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        payload, bound = verified_json_with_sidecar(
            receipt_path, label="single-GPU baseline receipt"
        )
        load_single_gpu_baselines(
            receipt_path,
            hardware_identity_sha256=plan["inputs"]["hardware"][
                "geometry_identity"
            ]["identity_sha256"],
            sequence_length=int(plan["inputs"]["common"]["sequence_length"]),
        )
        return payload, bound
    orders: dict[str, dict[str, Any]] = {}
    order_digest: str | None = None
    for candidate in CANDIDATES:
        order = ensure_baseline_order(root=root, plan=plan, candidate=candidate)
        digest = order["order"]["sha256"]
        if order_digest is None:
            order_digest = digest
        elif digest != order_digest:
            raise GeometryQualificationError(
                "Baseline candidates do not consume the exact same shuffled row order"
            )
        orders[candidate.candidate_id] = order
    candidates_root = root / "candidates"
    _ensure_confined_child_directory(root, candidates_root, label="baseline candidates")
    entries: dict[str, Any] = {}
    results: dict[str, Any] = {}
    failures: list[str] = []
    for candidate in CANDIDATES:
        candidate_root = candidates_root / candidate.candidate_id
        try:
            runtime_verifier(plan=plan, environment=environment)
            payload, bound = candidate_runner(
                root=candidate_root,
                plan=plan,
                candidate=candidate,
                order=orders[candidate.candidate_id],
                environment=environment,
            )
        except Exception as exc:
            payload, bound = _baseline_failure_result(
                root=candidate_root,
                plan=plan,
                candidate=candidate,
                error=exc,
                environment=environment,
            )
        results[candidate.candidate_id] = {
            "status": payload["status"],
            "result": bound,
        }
        if payload["status"] == "pass":
            entries[candidate.candidate_id] = payload["measurement"]
        else:
            failures.append(candidate.candidate_id)
    if failures:
        summary = {
            "format": BASELINE_FAILURE_FORMAT,
            "format_version": FORMAT_VERSION,
            "status": "fail",
            "plan_sha256": plan["plan_sha256"],
            "failed_candidates": failures,
            "results": results,
            "secrets_recorded": False,
        }
        bound = publish_json_new(root / "BASELINE-FAILURE.json", summary)
        return summary, bound
    producer_plan, producer_plan_bound = verified_json_with_sidecar(
        root / "PLAN.json", label="single-GPU baseline producer plan"
    )
    if producer_plan != dict(plan):
        raise GeometryQualificationError(
            "Published baseline producer plan differs from the running plan"
        )
    receipt = {
        "format": BASELINE_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "pass",
        "hardware_identity_sha256": plan["inputs"]["hardware"][
            "geometry_identity"
        ]["identity_sha256"],
        "producer_plan": producer_plan_bound,
        "exact_shared_order_payload_sha256": order_digest,
        "results": {
            candidate_id: record["result"]
            for candidate_id, record in results.items()
        },
        "candidates": entries,
    }
    bound = publish_json_new(receipt_path, receipt)
    load_single_gpu_baselines(
        receipt_path,
        hardware_identity_sha256=receipt["hardware_identity_sha256"],
        sequence_length=int(plan["inputs"]["common"]["sequence_length"]),
    )
    return receipt, bound


def run_grid(
    *,
    root: Path,
    plan: Mapping[str, Any],
    environment: Mapping[str, str] = os.environ,
    candidate_runner: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = run_candidate,
    runtime_verifier: Callable[..., dict[str, Any]] = verify_live_runtime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("mode") != "grid":
        raise GeometryQualificationError("run_grid requires a grid plan")
    _validate_plan_self_hash(plan)
    live_runtime = runtime_verifier(plan=plan, environment=environment)
    final_path = root / "GRID-RESULT.json"
    if final_path.exists() or final_path.is_symlink():
        payload, bound, _ = _load_validated_grid_result(
            final_path, require_pass=False
        )
        if payload.get("plan_sha256") != plan["plan_sha256"]:
            raise GeometryQualificationError("Grid result belongs to another plan")
        return payload, bound
    baseline_rates = _load_baseline_rates_from_plan(plan)
    orders: dict[str, dict[str, Any]] = {}
    order_payload_digest: str | None = None
    for candidate in CANDIDATES:
        order = ensure_grid_order(root=root, plan=plan, candidate=candidate)
        digest = order["order"]["sha256"]
        if order_payload_digest is None:
            order_payload_digest = digest
        elif digest != order_payload_digest:
            raise GeometryQualificationError(
                "Grid candidates do not consume the exact same shuffled row order"
            )
        orders[candidate.candidate_id] = order
    records: list[dict[str, Any]] = []
    passing: list[tuple[Decimal, int, CandidateSpec, dict[str, Any]]] = []
    run_root = root / "candidates"
    _ensure_confined_child_directory(root, run_root, label="grid candidates")
    for index, candidate in enumerate(CANDIDATES):
        candidate_root = run_root / candidate.candidate_id
        if candidate_root.is_symlink():
            raise GeometryQualificationError(
                f"Grid candidate root must not be a symlink: {candidate_root}"
            )
        failure_path = candidate_root / "FAILURE.json"
        result_path = candidate_root / "RESULT.json"
        if failure_path.exists() or failure_path.is_symlink():
            if result_path.exists() or result_path.is_symlink():
                raise GeometryQualificationError(
                    f"Candidate has conflicting terminal artifacts: {candidate_root}"
                )
            payload, bound = verified_json_with_sidecar(
                failure_path, label="candidate failure"
            )
            if (
                payload.get("plan_sha256") != plan["plan_sha256"]
                or payload.get("candidate") != candidate.as_dict()
                or payload.get("status") != "fail"
            ):
                raise GeometryQualificationError(
                    "Existing candidate failure belongs to another identity"
                )
        else:
            try:
                runtime_verifier(plan=plan, environment=environment)
                payload, bound = candidate_runner(
                    root=candidate_root,
                    plan=plan,
                    candidate=candidate,
                    order=orders[candidate.candidate_id],
                    baseline_rate=baseline_rates[candidate.candidate_id],
                    environment=environment,
                    live_runtime_verifier=runtime_verifier,
                )
            except Exception as exc:
                payload, bound = _candidate_failure_result(
                    root=candidate_root,
                    plan=plan,
                    candidate=candidate,
                    error=exc,
                    environment=environment,
                )
        record = {
            "candidate": candidate.as_dict(),
            "status": payload["status"],
            "result": bound,
            "failures": list(payload.get("failures", [])),
        }
        if payload["status"] == "pass":
            rate = _decimal(
                payload["measurements"]["aggregate_input_tokens_per_second"],
                field=f"candidate {candidate.candidate_id} throughput",
            )
            record["aggregate_input_tokens_per_second"] = _decimal_string(rate)
            passing.append((rate, -index, candidate, payload))
        records.append(record)
    if not passing:
        status = "fail"
        accepted_candidate: dict[str, Any] | None = None
        accepted_result: dict[str, Any] | None = None
    else:
        status = "pass"
        _, _, selected, selected_payload = max(passing)
        accepted_candidate = selected.as_dict()
        selected_record = next(
            record
            for record in records
            if record["candidate"] == accepted_candidate
        )
        accepted_result = {
            "artifact": selected_record["result"],
            "measurements_sha256": canonical_sha256(
                selected_payload["measurements"]
            ),
        }
    payload = {
        "format": GRID_RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": status,
        "plan_sha256": plan["plan_sha256"],
        "world_size": WORLD_SIZE,
        "effective_optimizer_update_rows": UPDATE_ROWS,
        "exact_shared_order_payload_sha256": order_payload_digest,
        "candidates_attempted": len(records),
        "candidates": records,
        "selection_policy": (
            "highest-external-end-to-end-throughput-among-threshold-passing-"
            "candidates-stable-grid-order-tie-break"
        ),
        "accepted_candidate": accepted_candidate,
        "accepted_result": accepted_result,
        "live_runtime": live_runtime,
        "secrets_recorded": False,
    }
    return payload, publish_json_new(final_path, payload)


def _candidate_from_plan(plan: Mapping[str, Any]) -> CandidateSpec:
    candidate_payload = plan.get("candidate")
    matches = [
        candidate
        for candidate in CANDIDATES
        if candidate.as_dict() == candidate_payload
    ]
    if len(matches) != 1:
        raise GeometryQualificationError("Final plan candidate is not canonical")
    return matches[0]


def _prevalidate_geometry_receipt(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    from pretrain.geometry_evidence import validate_authority_geometry_soak

    encoded = json.dumps(
        dict(receipt),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".accepted-geometry-validation-", suffix=".json", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        geometry_descriptor = artifact(
            temporary, label="temporary accepted-geometry receipt"
        )
        authority = {
            "geometry": {
                "artifact": geometry_descriptor,
                "receipt": dict(receipt),
            },
            "hardware": {
                "expected": plan["inputs"]["hardware"]["expected"]
            },
            "training": {
                "frozen_geometry": plan["inputs"]["train_order"]["geometry"]
            },
            "data": {
                "train_order": {
                    "sequence_length": plan["inputs"]["common"]["sequence_length"]
                }
            },
        }
        return validate_authority_geometry_soak(authority)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_final_receipt_binding(
    *, root: Path, receipt: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if set(receipt) != {
        "format",
        "format_version",
        "status",
        "hardware_contract_sha256",
        "train_order_manifest_sha256",
        "validation_order_manifest_sha256",
        "accepted",
        "measurements",
        "qualification",
    } or (
        receipt.get("format") != GEOMETRY_RECEIPT_FORMAT
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("status") != "pass"
        or receipt.get("hardware_contract_sha256")
        != plan["inputs"]["hardware"]["bound"]["artifact"]["sha256"]
        or receipt.get("train_order_manifest_sha256")
        != plan["inputs"]["train_order"]["manifest"]["sha256"]
        or receipt.get("validation_order_manifest_sha256")
        != plan["inputs"]["validation_order"]["manifest"]["sha256"]
    ):
        raise GeometryQualificationError(
            "Accepted geometry receipt does not match the final plan"
        )
    candidate = _candidate_from_plan(plan)
    expected_accepted = {
        "global_microbatch_rows": candidate.global_microbatch_rows,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "workers": plan["settings"]["workers"],
        "overfit_batch_rows": candidate.global_microbatch_rows,
        "compile_model": candidate.compile_model,
        "activation_checkpointing": True,
        "precision": "bfloat16",
        "parameter_dtype": "float32",
    }
    qualification_record = receipt.get("qualification")
    if receipt.get("accepted") != expected_accepted or not isinstance(
        qualification_record, Mapping
    ):
        raise GeometryQualificationError("Accepted geometry recipe is not canonical")
    if set(qualification_record) != {
        "producer_plan_sha256",
        "producer_sources",
        "grid_result",
        "final_candidate_result",
        "live_runtime",
        "strict_geometry_validation",
        "secrets_recorded",
    } or (
        qualification_record.get("producer_plan_sha256") != plan["plan_sha256"]
        or qualification_record.get("producer_sources") != plan["runtime"]["sources"]
        or qualification_record.get("grid_result") != plan["inputs"]["grid_result"]
        or qualification_record.get("secrets_recorded") is not False
    ):
        raise GeometryQualificationError(
            "Accepted geometry qualification provenance is invalid"
        )
    candidate_bound = qualification_record.get("final_candidate_result")
    try:
        candidate_path = Path(str(candidate_bound["artifact"]["path"]))
    except (KeyError, TypeError) as exc:
        raise GeometryQualificationError(
            "Final candidate result binding is invalid"
        ) from exc
    candidate_payload, verified_bound = verified_json_with_sidecar(
        candidate_path, label="final-order candidate result"
    )
    if verified_bound != candidate_bound or (
        candidate_payload.get("status") != "pass"
        or candidate_payload.get("plan_sha256") != plan["plan_sha256"]
        or candidate_payload.get("candidate") != candidate.as_dict()
        or candidate_payload.get("measurements") != receipt.get("measurements")
    ):
        raise GeometryQualificationError(
            "Final candidate result differs from the accepted receipt"
        )
    strict = _prevalidate_geometry_receipt(root=root, receipt=receipt, plan=plan)
    if qualification_record.get("strict_geometry_validation") != strict:
        raise GeometryQualificationError(
            "Accepted receipt strict-validation summary is inconsistent"
        )
    return strict


def run_final_soak(
    *,
    root: Path,
    plan: Mapping[str, Any],
    receipt_path: Path,
    environment: Mapping[str, str] = os.environ,
    candidate_runner: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = run_candidate,
    runtime_verifier: Callable[..., dict[str, Any]] = verify_live_runtime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("mode") != "final-soak":
        raise GeometryQualificationError("run_final_soak requires a final-soak plan")
    _validate_plan_self_hash(plan)
    live_runtime = runtime_verifier(plan=plan, environment=environment)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt, bound = verified_json_with_sidecar(
            receipt_path, label="accepted geometry receipt"
        )
        _validate_final_receipt_binding(root=root, receipt=receipt, plan=plan)
        return receipt, bound
    candidate = _candidate_from_plan(plan)
    rates = _load_baseline_rates_from_plan(plan)
    order = {
        "manifest": plan["inputs"]["train_order"]["manifest"],
        "geometry": plan["inputs"]["train_order"]["geometry"],
    }
    candidate_payload, candidate_bound = candidate_runner(
        root=root / "final-candidate",
        plan=plan,
        candidate=candidate,
        order=order,
        baseline_rate=rates[candidate.candidate_id],
        environment=environment,
        live_runtime_verifier=runtime_verifier,
    )
    if candidate_payload.get("status") != "pass":
        raise GeometryQualificationError(
            "Final-order soak failed; refusing to publish accepted geometry"
        )
    measurements = candidate_payload["measurements"]
    receipt: dict[str, Any] = {
        "format": GEOMETRY_RECEIPT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "pass",
        "hardware_contract_sha256": plan["inputs"]["hardware"]["bound"][
            "artifact"
        ]["sha256"],
        "train_order_manifest_sha256": plan["inputs"]["train_order"][
            "manifest"
        ]["sha256"],
        "validation_order_manifest_sha256": plan["inputs"]["validation_order"][
            "manifest"
        ]["sha256"],
        "accepted": {
            "global_microbatch_rows": candidate.global_microbatch_rows,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "workers": plan["settings"]["workers"],
            "overfit_batch_rows": candidate.global_microbatch_rows,
            "compile_model": candidate.compile_model,
            "activation_checkpointing": True,
            "precision": "bfloat16",
            "parameter_dtype": "float32",
        },
        "measurements": measurements,
        "qualification": {
            "producer_plan_sha256": plan["plan_sha256"],
            "producer_sources": plan["runtime"]["sources"],
            "grid_result": plan["inputs"]["grid_result"],
            "final_candidate_result": candidate_bound,
            "live_runtime": live_runtime,
            "secrets_recorded": False,
        },
    }
    validation = _prevalidate_geometry_receipt(root=root, receipt=receipt, plan=plan)
    receipt["qualification"]["strict_geometry_validation"] = validation
    # Revalidate the final byte content including the embedded validation summary.
    _validate_final_receipt_binding(root=root, receipt=receipt, plan=plan)
    return receipt, publish_json_new(receipt_path, receipt)


def qualification_status(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise GeometryQualificationError(f"Qualification root does not exist: {root}")
    plan_path = root / "PLAN.json"
    plan, bound = verified_json_with_sidecar(plan_path, label="qualification plan")
    _validate_plan_self_hash(plan)
    result_files = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.json")
        if path.name in {"RESULT.json", "FAILURE.json", "GRID-RESULT.json"}
        and not path.is_symlink()
    )
    accepted = root / "accepted-geometry.json"
    return {
        "format": "six-gpu-geometry-qualification-status",
        "format_version": FORMAT_VERSION,
        "mode": plan["mode"],
        "plan": bound,
        "plan_sha256": plan["plan_sha256"],
        "results": result_files,
        "accepted_geometry_exists": accepted.is_file() and not accepted.is_symlink(),
        "journal_files": sorted(
            str(path.relative_to(root))
            for path in root.rglob("JOURNAL.json")
            if not path.is_symlink()
        ),
    }
