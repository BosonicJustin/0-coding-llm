"""Deterministic native-PyTorch pre-training harness.

The trainer deliberately consumes the batch contract from :mod:`pretrain.data`
without knowing anything about raw text or tokenization.  It supports exact
step-boundary resume, token-correct gradient accumulation, DDP, optional AMP,
atomic checkpoints, and optional Weights & Biases logging.

The command-line entry point trains either the exact 1.284B architecture or a
small architecture-compatible debug model against an immutable packed order.
W&B is disabled by default and is imported only when explicitly enabled.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import CausalLM, ModelConfig
from pretrain.data import (
    DOMAIN_ORDER,
    DistributedBatchSampler,
    create_training_dataloader,
    evaluation_order_geometry,
    frozen_training_geometry,
)
from pretrain.tokenizer_identity import (
    TokenizerIdentityError,
    require_sha256,
    verify_tokenizer_identity,
)


CHECKPOINT_FORMAT = "native-pytorch-pretrain"
CHECKPOINT_VERSION = 5
Precision = Literal["float32", "bfloat16"]
WandbMode = Literal["disabled", "offline", "online"]
_REQUIRED_BATCH_KEYS = ("input_ids", "position_ids", "document_ids", "labels")
_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = frozenset((":4096:8", ":16:8"))


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a stable identity for an immutable manifest or artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_order_payload_checksum(order_manifest_path: str | Path) -> str:
    """Verify the immutable row-order payload and return its SHA-256."""

    manifest_path = Path(order_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = payload.get("order", {})
    relative_path = order.get("path")
    expected_digest = order.get("sha256")
    expected_bytes = order.get("bytes")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Order manifest does not declare an order payload path")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("Order manifest does not declare a valid order SHA-256")
    order_path = manifest_path.parent / relative_path
    if not order_path.is_file():
        raise FileNotFoundError(order_path)
    if not isinstance(expected_bytes, int) or order_path.stat().st_size != expected_bytes:
        raise IOError(f"Order payload size mismatch: {order_path}")
    actual_digest = sha256_file(order_path)
    if actual_digest != expected_digest:
        raise IOError(f"Order payload checksum mismatch: {order_path}")
    return actual_digest


def batch_fingerprint(batch: Mapping[str, torch.Tensor]) -> str:
    """Fingerprint all tensors in a fixed diagnostic batch.

    Shape, dtype, key, and bytes are included so a resumed overfit experiment
    cannot silently continue against a different packed row.
    """

    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def seed_everything(seed: int, *, deterministic: bool) -> None:
    """Seed process RNGs without touching CUDA devices owned by other ranks."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # ``torch.manual_seed`` also calls ``torch.cuda.manual_seed_all``. In
    # one-process-per-GPU DDP that makes every rank initialize and seed every
    # visible device. Seed the host generator directly, then only the CUDA
    # device selected for this process.
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic


def validate_deterministic_cuda_environment(
    device: torch.device | str,
    *,
    deterministic: bool,
) -> None:
    """Fail before CUDA initialization if exact-resume prerequisites are absent."""

    resolved = torch.device(device)
    if resolved.type != "cuda" or not deterministic:
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
        allowed = ", ".join(sorted(_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS))
        raise RuntimeError(
            "deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG to be "
            f"set before Python starts; found {workspace!r}, expected one of {allowed}"
        )


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG, including only this rank's CUDA device."""

    cuda_state: dict[str, Any] | None = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        cuda_state = {
            "device_index": device_index,
            "state": torch.cuda.get_rng_state(device_index),
        }

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNGs, rejecting a changed rank-local CUDA assignment."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError(f"Checkpoint RNG state must contain exactly {sorted(required)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, Mapping) or set(cuda_state) != {
            "device_index",
            "state",
        }:
            raise ValueError("Checkpoint CUDA RNG state has an invalid schema")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        saved_index = cuda_state["device_index"]
        if type(saved_index) is not int:
            raise ValueError("Checkpoint CUDA RNG device index must be an integer")
        current_index = torch.cuda.current_device()
        if saved_index != current_index:
            raise RuntimeError(
                "Rank-local CUDA device changed across exact resume: checkpoint has "
                f"cuda:{saved_index}, runtime has cuda:{current_index}"
            )
        cuda_rng = cuda_state["state"]
        if not isinstance(cuda_rng, torch.Tensor):
            raise ValueError("Checkpoint CUDA RNG state must be a tensor")
        torch.cuda.set_rng_state(cuda_rng, current_index)


@contextlib.contextmanager
def preserve_host_rng_state() -> Iterator[None]:
    """Keep infrastructure callbacks from consuming model-side host RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@contextlib.contextmanager
def preserve_all_rng_state() -> Iterator[None]:
    """Prevent validation and infrastructure work from changing training RNGs."""

    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


@dataclass(frozen=True)
class TrainConfig:
    """Settings that define one optimizer trajectory."""

    max_steps: int
    global_microbatch_rows: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    precision: Precision = "bfloat16"
    seed: int = 1234
    deterministic: bool = False
    compile_model: bool = False
    log_every: int = 1
    checkpoint_every: int = 0
    eval_every: int = 0
    eval_at_start: bool = False
    fused_adamw: bool | None = None

    def __post_init__(self) -> None:
        finite_fields = (
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "adam_eps",
            "max_grad_norm",
        )
        nonfinite = [
            field
            for field in finite_fields
            if not math.isfinite(float(getattr(self, field)))
        ]
        if nonfinite:
            raise ValueError(
                "Optimizer hyperparameters must be finite: "
                + ", ".join(nonfinite)
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.global_microbatch_rows < 1:
            raise ValueError("global_microbatch_rows must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_eps <= 0 or self.max_grad_norm <= 0:
            raise ValueError("adam_eps and max_grad_norm must be positive")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError(f"Unsupported precision {self.precision!r}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.eval_every < 0:
            raise ValueError("eval_every must be non-negative")
        if not isinstance(self.eval_at_start, bool):
            raise TypeError("eval_at_start must be boolean")

    def trajectory_dict(self) -> dict[str, Any]:
        """Return only fields whose change would alter future parameters."""

        payload = dataclasses.asdict(self)
        payload.pop("log_every")
        payload.pop("checkpoint_every")
        payload.pop("eval_every")
        payload.pop("eval_at_start")
        return payload


@dataclass
class TrainState:
    """Progress committed at completed optimizer-step boundaries."""

    completed_steps: int = 0
    completed_microbatches: int = 0
    consumed_rows: int = 0
    consumed_input_tokens: int = 0
    consumed_supervised_tokens: int = 0
    last_validated_step: int = 0
    consumed_rows_per_domain: dict[str, int] = dataclasses.field(
        default_factory=lambda: {domain: 0 for domain in DOMAIN_ORDER}
    )
    consumed_input_tokens_per_domain: dict[str, int] = dataclasses.field(
        default_factory=lambda: {domain: 0 for domain in DOMAIN_ORDER}
    )
    consumed_supervised_tokens_per_domain: dict[str, int] = dataclasses.field(
        default_factory=lambda: {domain: 0 for domain in DOMAIN_ORDER}
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainState":
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != expected:
            raise ValueError(f"Invalid train-state keys: expected {sorted(expected)}")
        scalar_fields = {
            "completed_steps",
            "completed_microbatches",
            "consumed_rows",
            "consumed_input_tokens",
            "consumed_supervised_tokens",
            "last_validated_step",
        }
        if any(type(payload[key]) is not int for key in scalar_fields):
            raise ValueError("Train-state scalar counters must be plain integers")
        values: dict[str, Any] = {key: payload[key] for key in scalar_fields}
        if any(value < 0 for value in values.values()):
            raise ValueError("Train-state counters must be non-negative")
        if values["last_validated_step"] > values["completed_steps"]:
            raise ValueError("Train-state validation step exceeds training progress")
        for field in expected.difference(scalar_fields):
            raw = payload[field]
            if (
                not isinstance(raw, dict)
                or set(raw) != set(DOMAIN_ORDER)
                or any(
                    type(raw[domain]) is not int or raw[domain] < 0
                    for domain in DOMAIN_ORDER
                )
            ):
                raise ValueError(f"Invalid train-state per-domain counter: {field}")
            values[field] = dict(raw)
        return cls(**values)


class MetricLogger(Protocol):
    def log(self, metrics: Mapping[str, float | int]) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    def log(self, metrics: Mapping[str, float | int]) -> None:
        del metrics

    def finish(self) -> None:
        pass


class ConsoleLogger:
    """Emit one machine-readable JSON object per logged optimizer step."""

    def log(self, metrics: Mapping[str, float | int]) -> None:
        print(json.dumps(dict(metrics), sort_keys=True), flush=True)

    def finish(self) -> None:
        pass


class CompositeLogger:
    def __init__(self, *loggers: MetricLogger) -> None:
        self.loggers = list(loggers)

    def log(self, metrics: Mapping[str, float | int]) -> None:
        healthy: list[MetricLogger] = []
        for logger in self.loggers:
            try:
                logger.log(metrics)
                healthy.append(logger)
            except Exception as exc:
                print(
                    f"warning: disabling failed metric logger "
                    f"{type(logger).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        self.loggers = healthy

    def finish(self) -> None:
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as exc:
                print(
                    f"warning: metric logger {type(logger).__name__} failed to finish: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


class WandbLogger:
    """Lazy optional W&B adapter.

    ``mode='disabled'`` does not import ``wandb`` at all.  ``offline`` never
    sends metrics to the network; a later explicit ``wandb sync`` is required.
    """

    def __init__(
        self,
        *,
        mode: WandbMode = "disabled",
        project: str = "coding-model-from-scratch",
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: Iterable[str] | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        directory: str | Path | None = None,
    ) -> None:
        if mode not in ("disabled", "offline", "online"):
            raise ValueError(f"Unsupported W&B mode {mode!r}")
        self.mode = mode
        self._run: Any | None = None
        self.run_id: str | None = None
        if mode == "disabled":
            return
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was enabled but `wandb` is not installed; install "
                "requirements-wandb.txt or use --wandb-mode disabled"
            ) from exc
        init_kwargs: dict[str, Any] = dict(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=list(tags or ()),
            config=dict(config or {}),
            dir=str(directory) if directory is not None else None,
            mode=mode,
        )
        if run_id is not None:
            init_kwargs.update(id=run_id, resume="allow")
        self._run = wandb.init(**init_kwargs)
        self.run_id = str(self._run.id)
        # Training and validation are separate records at the same optimizer
        # step. An explicit W&B internal ``step=`` would cause the later record
        # to be treated as stale; use train/step as the shared chart axis while
        # allowing both records to be committed.
        self._run.define_metric("train/step")
        self._run.define_metric("*", step_metric="train/step")

    def log(self, metrics: Mapping[str, float | int]) -> None:
        if self._run is not None:
            self._run.log(dict(metrics))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def initialize_optional_metric_logger(
    factory: Callable[[], MetricLogger],
) -> tuple[MetricLogger, str | None]:
    """Initialize non-authoritative tracking without making training depend on it.

    Data, validation, and checkpoint failures remain fatal. Experiment-tracker
    startup is intentionally different: credentials may expire or a local W&B
    service may fail after the production preflight. The caller broadcasts the
    returned diagnostic so every rank crosses setup in the same order, while
    rank zero falls back to console-only metrics.
    """

    try:
        return factory(), None
    except Exception as exc:
        return NullLogger(), f"{type(exc).__name__}: {exc}"


class GracefulStopController:
    """Convert process signals into a checkpoint request at the next safe step.

    Signal handlers only mutate one Python integer.  Distributed collectives
    and checkpoint I/O remain in the normal training thread, where all ranks
    can participate without invoking non-async-signal-safe operations.
    """

    def __init__(self) -> None:
        self._requested_signal = 0
        self._previous_handlers: dict[int, Any] = {}

    @property
    def requested_signal(self) -> int:
        return self._requested_signal

    def request(self, signum: int = int(signal.SIGTERM)) -> None:
        value = int(signum)
        if value < 1:
            raise ValueError("Stop signal must be positive")
        if self._requested_signal == 0:
            self._requested_signal = value

    def poll_external_request(self) -> None:
        """Observe a supervisor request file without doing I/O in a handler."""

        if self._requested_signal:
            return
        raw_path = os.environ.get("PRETRAIN_STOP_REQUEST_FILE")
        if not raw_path:
            return
        path = Path(raw_path)
        try:
            raw_signal = path.read_text(encoding="ascii", errors="strict").strip()
        except FileNotFoundError:
            return
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Cannot read external stop request {path}: {exc}") from exc
        try:
            requested = int(raw_signal)
        except ValueError as exc:
            raise RuntimeError(
                f"External stop request {path} is not an integer signal"
            ) from exc
        allowed = {int(signal.SIGINT), int(signal.SIGTERM)}
        if hasattr(signal, "SIGHUP"):
            allowed.add(int(signal.SIGHUP))
        if hasattr(signal, "SIGUSR1"):
            allowed.add(int(signal.SIGUSR1))
        if requested not in allowed:
            raise RuntimeError(
                f"External stop request {path} has unsupported signal {requested}"
            )
        self.request(requested)

    def _handle(self, signum: int, frame: Any) -> None:
        del frame
        self.request(signum)

    def install(self) -> None:
        if self._previous_handlers:
            raise RuntimeError("Graceful stop handlers are already installed")
        handled = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            handled.append(signal.SIGHUP)
        if hasattr(signal, "SIGUSR1"):
            handled.append(signal.SIGUSR1)
        for signum in handled:
            value = int(signum)
            self._previous_handlers[value] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def __enter__(self) -> "GracefulStopController":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.restore()


class FixedBatchStream(Iterable[Mapping[str, torch.Tensor]]):
    """Yield one immutable diagnostic chunk forever."""

    def __init__(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        identity: str | None = None,
    ) -> None:
        missing = set(_REQUIRED_BATCH_KEYS).difference(batch)
        if missing:
            raise ValueError(f"Fixed batch is missing keys: {sorted(missing)}")
        if identity is not None and (
            not isinstance(identity, str) or not identity.strip()
        ):
            raise ValueError("Fixed-batch identity must be a non-empty string")
        self.batch = {
            key: value.detach().cpu().clone().contiguous()
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        self.identity = identity or (
            f"fixed-batch-sha256:{batch_fingerprint(self.batch)}"
        )

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        while True:
            yield self.batch


def tiny_model_config(*, vocab_size: int, max_seq_len: int) -> ModelConfig:
    """Architecture-compatible CPU/debug configuration."""

    return ModelConfig(
        vocab_size=vocab_size,
        dim=32,
        hidden_dim=88,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=max_seq_len,
        attention_backend="auto",
        loss_chunk_size=min(32, max_seq_len),
    )


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    if isinstance(current, DistributedDataParallel):
        current = current.module
    # torch.compile wraps the original module without copying parameters.
    if hasattr(current, "_orig_mod"):
        current = current._orig_mod
    return current


def _is_compiled_model(model: torch.nn.Module) -> bool:
    current = model.module if isinstance(model, DistributedDataParallel) else model
    return hasattr(current, "_orig_mod")


def _distributed_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def wrap_distributed_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> DistributedDataParallel:
    """Wrap ``model`` with the exact DDP policy used by production launches.

    Keep this policy in a directly testable helper. In particular,
    ``static_graph=True`` is not safe with DDP ``no_sync`` gradient
    accumulation on supported PyTorch builds and can fail inside the reducer
    on the first synchronized backward pass.
    """

    ddp_kwargs: dict[str, Any] = {
        "device_ids": [device.index] if device.type == "cuda" else None,
        "gradient_as_bucket_view": True,
        "static_graph": False,
    }
    # Newer PyTorch releases split forward-time buffer synchronization from
    # initialization and deprecate ``broadcast_buffers``. Our nonpersistent
    # RoPE caches are identical on every rank, so one initialization sync is
    # harmless; only repeated forward-time broadcasts need to be disabled.
    if "forward_sync_buffers" in inspect.signature(
        DistributedDataParallel
    ).parameters:
        ddp_kwargs["forward_sync_buffers"] = False
    else:  # PyTorch releases before ``forward_sync_buffers`` was introduced.
        ddp_kwargs["broadcast_buffers"] = False
    return DistributedDataParallel(model, **ddp_kwargs)


def _dtype_for_precision(precision: Precision) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[precision]


def _optimizer_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(
    model: torch.nn.Module,
    config: TrainConfig,
    *,
    device: torch.device,
) -> torch.optim.AdamW:
    """Build deterministic decay/no-decay AdamW parameter groups."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("Model has no trainable parameters")
    fused = config.fused_adamw
    if fused is None:
        fused = device.type == "cuda"
    if fused and device.type != "cuda":
        raise ValueError("Fused AdamW requires CUDA")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        fused=bool(fused),
    )


def learning_rate_for_step(config: TrainConfig, zero_based_step: int) -> float:
    """Linear warmup followed by cosine decay, indexed before each update."""

    if not 0 <= zero_based_step < config.max_steps:
        raise ValueError("step is outside the configured schedule")
    if config.warmup_steps and zero_based_step < config.warmup_steps:
        return config.learning_rate * (zero_based_step + 1) / config.warmup_steps
    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps <= 1:
        return config.min_learning_rate
    progress = (zero_based_step - config.warmup_steps) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


class CheckpointLease:
    """Hold a single-writer advisory lock for one checkpoint lineage."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.path = self.checkpoint_path.with_name(
            f".{self.checkpoint_path.name}.lock"
        )
        self._handle: Any | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("Checkpoint lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"Another trainer holds checkpoint lease {self.path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": os.uname().nodename,
                    "checkpoint": str(self.checkpoint_path.resolve()),
                    "acquired_unix": time.time(),
                },
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "CheckpointLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.release()


def reconcile_checkpoint_temporaries(path: str | Path) -> list[Path]:
    """Remove only this writer's unpublished temporary checkpoint files.

    Call while holding :class:`CheckpointLease`.  Published ``last`` and
    ``previous`` generations are never candidates.
    """

    checkpoint = Path(path)
    parent = checkpoint.parent
    if not parent.exists():
        return []
    previous = checkpoint.with_name(
        f"{checkpoint.stem}.previous{checkpoint.suffix}"
    )
    lease_path = checkpoint.with_name(f".{checkpoint.name}.lock")
    patterns = (
        f".{checkpoint.name}.checkpoint-part-*",
        f".{previous.name}.previous-link-*",
    )
    removed: list[Path] = []
    for pattern in patterns:
        for candidate in parent.glob(pattern):
            if candidate in (checkpoint, previous, lease_path):
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise IOError(
                    f"Unsafe unknown checkpoint temporary candidate: {candidate}"
                )
            candidate.unlink()
            removed.append(candidate)
    if removed:
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return removed


def _atomic_torch_save(
    payload: Mapping[str, Any],
    path: Path,
    *,
    preserve_previous: bool = False,
) -> None:
    """Publish one checkpoint generation with a durable rollback generation.

    ``preserve_previous`` is used exactly once after explicitly recovering from
    ``last.previous``. In that case the current ``last`` is the rejected
    generation, so it must not replace the known-good rollback checkpoint.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.checkpoint-part-",
        dir=path.parent,
    )
    previous_temporary: Path | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not preserve_previous:
            if not path.is_file():
                raise IOError(f"Checkpoint destination is not a file: {path}")
            previous = path.with_name(f"{path.stem}.previous{path.suffix}")
            previous_temporary = path.parent / (
                f".{previous.name}.previous-link-{os.getpid()}-{time.time_ns()}"
            )
            # A hard link retains the known-readable inode without copying a
            # ~15 GB optimizer checkpoint. Both names live in one directory,
            # so replacement remains atomic on a POSIX network volume.
            os.link(path, previous_temporary)
        # Publish the new complete generation before rotating the old latest
        # into ``previous``.  A crash between these replacements leaves the new
        # latest plus the older previous and an orphan hard link; it never
        # destroys the only known-good previous before latest is durable.
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if previous_temporary is not None:
            os.replace(previous_temporary, previous)
        # Make the directory entry durable as well as the file contents.
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
        if previous_temporary is not None:
            try:
                previous_temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _canonical_training_geometry(
    geometry: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and freeze the order's optimizer-update accounting contract."""

    if geometry is None:
        return None
    expected_keys = {
        "global_microbatch_rows",
        "gradient_accumulation_steps",
        "optimizer_update_rows",
        "optimizer_updates",
        "consumed_global_microbatches",
        "sequence_length",
        "consumed_rows",
        "dropped_rows",
        "consumed_input_tokens",
        "dropped_input_tokens",
        "consumed_supervised_tokens",
        "dropped_supervised_tokens",
        "consumed_input_tokens_per_domain",
        "consumed_supervised_tokens_per_domain",
    }
    if set(geometry) != expected_keys:
        raise ValueError(
            "Training geometry schema mismatch: "
            f"missing={sorted(expected_keys.difference(geometry))}, "
            f"extra={sorted(set(geometry).difference(expected_keys))}"
        )
    canonical = json.loads(
        json.dumps(dict(geometry), sort_keys=True, allow_nan=False)
    )
    positive_fields = (
        "global_microbatch_rows",
        "gradient_accumulation_steps",
        "optimizer_update_rows",
        "optimizer_updates",
        "consumed_global_microbatches",
        "sequence_length",
        "consumed_rows",
        "consumed_input_tokens",
    )
    nonnegative_fields = (
        "dropped_rows",
        "dropped_input_tokens",
        "consumed_supervised_tokens",
        "dropped_supervised_tokens",
    )
    if any(type(canonical[field]) is not int or canonical[field] < 1 for field in positive_fields):
        raise ValueError("Training geometry positive counters must be positive integers")
    if any(
        type(canonical[field]) is not int or canonical[field] < 0
        for field in nonnegative_fields
    ):
        raise ValueError("Training geometry token/tail counters must be non-negative integers")
    microbatch_rows = canonical["global_microbatch_rows"]
    accumulation = canonical["gradient_accumulation_steps"]
    update_rows = canonical["optimizer_update_rows"]
    updates = canonical["optimizer_updates"]
    sequence_length = canonical["sequence_length"]
    if update_rows != microbatch_rows * accumulation:
        raise ValueError("Training geometry optimizer_update_rows is inconsistent")
    if canonical["consumed_global_microbatches"] != updates * accumulation:
        raise ValueError("Training geometry consumed microbatches are inconsistent")
    if canonical["consumed_rows"] != updates * update_rows:
        raise ValueError("Training geometry consumed rows are inconsistent")
    if canonical["consumed_input_tokens"] != canonical["consumed_rows"] * sequence_length:
        raise ValueError("Training geometry consumed input tokens are inconsistent")
    if canonical["dropped_input_tokens"] != canonical["dropped_rows"] * sequence_length:
        raise ValueError("Training geometry dropped input tokens are inconsistent")
    if canonical["consumed_supervised_tokens"] > canonical["consumed_input_tokens"]:
        raise ValueError("Training geometry consumes more supervised than input tokens")
    if canonical["dropped_supervised_tokens"] > canonical["dropped_input_tokens"]:
        raise ValueError("Training geometry drops more supervised than input tokens")
    for field, expected_total in (
        ("consumed_input_tokens_per_domain", canonical["consumed_input_tokens"]),
        (
            "consumed_supervised_tokens_per_domain",
            canonical["consumed_supervised_tokens"],
        ),
    ):
        values = canonical[field]
        if (
            type(values) is not dict
            or set(values) != set(DOMAIN_ORDER)
            or any(type(values[domain]) is not int or values[domain] < 0 for domain in DOMAIN_ORDER)
            or sum(values.values()) != expected_total
        ):
            raise ValueError(f"Training geometry {field} is inconsistent")
    input_by_domain = canonical["consumed_input_tokens_per_domain"]
    supervised_by_domain = canonical["consumed_supervised_tokens_per_domain"]
    for domain in DOMAIN_ORDER:
        if input_by_domain[domain] % sequence_length:
            raise ValueError(
                "Training geometry per-domain input tokens are not whole packed rows"
            )
        if supervised_by_domain[domain] > input_by_domain[domain]:
            raise ValueError(
                "Training geometry per-domain supervised tokens exceed inputs"
            )
    return canonical


def _move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    local_batch_size: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Validate one packed CPU batch and move model inputs to a device."""

    missing = set(_REQUIRED_BATCH_KEYS).difference(batch)
    if missing:
        raise ValueError(f"Packed batch is missing keys: {sorted(missing)}")
    labels = batch["labels"]
    if not isinstance(labels, torch.Tensor) or labels.device.type != "cpu":
        raise ValueError("Packed batches must originate on CPU before device transfer")
    counted_loss_tokens = int(labels.ne(-100).sum())
    declared_loss_tokens = batch.get("num_loss_tokens")
    if declared_loss_tokens is not None:
        if not isinstance(declared_loss_tokens, torch.Tensor):
            raise TypeError("num_loss_tokens must be a tensor when supplied")
        if declared_loss_tokens.device.type != "cpu" or declared_loss_tokens.numel() != 1:
            raise ValueError("num_loss_tokens must be one CPU scalar")
        if int(declared_loss_tokens) != counted_loss_tokens:
            raise ValueError("Batch num_loss_tokens disagrees with labels")
    moved = {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in _REQUIRED_BATCH_KEYS
    }
    if any(value.dtype != torch.int64 for value in moved.values()):
        raise TypeError("Model input, position, document, and label tensors must be int64")
    if any(value.shape != moved["input_ids"].shape for value in moved.values()):
        raise ValueError("All four packed tensors must have identical [B, T] shape")
    if moved["input_ids"].ndim != 2:
        raise ValueError("All four packed tensors must have shape [B, T]")
    if moved["input_ids"].shape[0] != local_batch_size:
        raise ValueError(
            "Local packed-row batch differs from configured microbatch/world size: "
            f"found {moved['input_ids'].shape[0]}, expected {local_batch_size}"
        )
    domain_ids = batch.get("domain_ids")
    if domain_ids is not None:
        if (
            not isinstance(domain_ids, torch.Tensor)
            or domain_ids.device.type != "cpu"
            or domain_ids.dtype != torch.int64
            or domain_ids.shape != (local_batch_size,)
        ):
            raise ValueError("domain_ids must be one int64 CPU value per packed row")
        if domain_ids.numel() and (
            int(domain_ids.min()) < 0 or int(domain_ids.max()) >= len(DOMAIN_ORDER)
        ):
            raise ValueError("domain_ids contains an unknown domain")
        moved["domain_ids"] = domain_ids.to(
            device, non_blocking=device.type == "cuda"
        )
    return moved, counted_loss_tokens


class ValidationRunner:
    """Repeatable, distributed held-out evaluation over an immutable order."""

    def __init__(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        device: torch.device | str,
        precision: Precision,
        max_batches: int,
    ) -> None:
        if max_batches < 1:
            raise ValueError("Validation max_batches must be positive")
        sampler = getattr(batches, "batch_sampler", None)
        if not isinstance(sampler, DistributedBatchSampler):
            raise ValueError(
                "Validation requires a DistributedBatchSampler-backed DataLoader"
            )
        if sampler.manifest.get("split") != "validation":
            raise ValueError("Validation runner requires the reserved validation split")
        if sampler.start_global_microbatch != 0:
            raise ValueError("Validation must always begin at the immutable order start")
        if max_batches > sampler.total_global_microbatches:
            raise ValueError(
                "Validation max_batches exceeds the complete frozen validation order"
            )
        self.batches = batches
        self.sampler = sampler
        self.device = torch.device(device)
        self.precision = precision
        self.max_batches = int(max_batches)
        self.rank, self.world_size = _distributed_rank_world()
        if sampler.rank != self.rank or sampler.world_size != self.world_size:
            raise ValueError("Validation sampler rank/world size differs from runtime")
        self.local_batch_size = sampler.local_batch_size
        self.data_identity = sampler.data_identity
        self.configuration = {
            "data_identity": self.data_identity,
            "max_batches": self.max_batches,
            "global_microbatch_rows": sampler.global_microbatch_rows,
            "world_size": self.world_size,
            "sequence_length": sampler.manifest["sequence_length"],
            "vocab_size": sampler.manifest["vocab_size"],
            "tokenizer_manifest_sha256": sampler.manifest[
                "tokenizer_manifest_sha256"
            ],
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        *,
        train_step: int,
    ) -> dict[str, float | int]:
        """Return globally token-normalized validation metrics."""

        started = time.perf_counter()
        was_training = model.training
        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        local_model_loss_tokens = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        local_loss_tokens = 0
        local_input_tokens = 0
        local_rows = 0
        local_domain_loss_sums = torch.zeros(
            len(DOMAIN_ORDER), dtype=torch.float64, device=self.device
        )
        local_domain_loss_tokens = torch.zeros(
            len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
        )
        local_domain_input_tokens = torch.zeros(
            len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
        )
        local_domain_rows = torch.zeros(
            len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
        )
        completed_batches = 0
        with preserve_all_rng_state():
            iterator_state = capture_rng_state()
            iterator = iter(self.batches)
            restore_rng_state(iterator_state)
            model.eval()
            try:
                for _ in range(self.max_batches):
                    try:
                        raw_batch = next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Validation order ended before the configured batch count"
                        ) from exc
                    batch, token_count = _move_batch_to_device(
                        raw_batch,
                        device=self.device,
                        local_batch_size=self.local_batch_size,
                    )
                    with torch.autocast(
                        self.device.type,
                        dtype=_dtype_for_precision(self.precision),
                        enabled=self.precision != "float32",
                    ):
                        output = model(
                            batch["input_ids"],
                            batch["position_ids"],
                            batch["document_ids"],
                            batch["labels"],
                        )
                    if output.loss_sum is None or output.num_loss_tokens is None:
                        raise RuntimeError("Model did not return validation loss statistics")
                    if token_count < 1:
                        raise ValueError("Validation microbatch has no supervised tokens")
                    if (
                        output.num_loss_tokens.dtype != torch.int64
                        or output.num_loss_tokens.numel() != 1
                    ):
                        raise ValueError(
                            "Model validation-token counter must be one int64 scalar"
                        )
                    domain_ids = batch.get("domain_ids")
                    if domain_ids is None or output.loss_sums_per_row is None:
                        raise ValueError(
                            "Validation batches require domain IDs and per-row losses"
                        )
                    row_loss_tokens = batch["labels"].ne(-100).sum(dim=1)
                    row_input_tokens = torch.full_like(
                        row_loss_tokens,
                        batch["input_ids"].shape[1],
                    )
                    local_domain_loss_sums += torch.bincount(
                        domain_ids,
                        weights=output.loss_sums_per_row.detach().double(),
                        minlength=len(DOMAIN_ORDER),
                    )
                    local_domain_loss_tokens += torch.bincount(
                        domain_ids,
                        weights=row_loss_tokens,
                        minlength=len(DOMAIN_ORDER),
                    ).to(torch.int64)
                    local_domain_input_tokens += torch.bincount(
                        domain_ids,
                        weights=row_input_tokens,
                        minlength=len(DOMAIN_ORDER),
                    ).to(torch.int64)
                    local_domain_rows += torch.bincount(
                        domain_ids,
                        minlength=len(DOMAIN_ORDER),
                    ).to(torch.int64)
                    local_loss_sum += output.loss_sum.detach().double()
                    local_model_loss_tokens += output.num_loss_tokens.detach()
                    local_loss_tokens += token_count
                    local_input_tokens += batch["input_ids"].numel()
                    local_rows += batch["input_ids"].shape[0]
                    completed_batches += 1
            finally:
                model.train(was_training)

        aggregate_stats = torch.tensor(
            [
                float(local_loss_sum),
                local_loss_tokens,
                local_input_tokens,
                local_rows,
                float(local_model_loss_tokens),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        stats = torch.cat(
            (
                aggregate_stats,
                local_domain_loss_sums,
                local_domain_loss_tokens.to(torch.float64),
                local_domain_input_tokens.to(torch.float64),
                local_domain_rows.to(torch.float64),
            )
        )
        if self.world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        values = stats.cpu().tolist()
        (
            loss_sum,
            loss_tokens_value,
            input_tokens_value,
            rows_value,
            model_loss_tokens_value,
        ) = values[:5]
        loss_tokens = int(loss_tokens_value)
        input_tokens = int(input_tokens_value)
        rows = int(rows_value)
        if completed_batches != self.max_batches or loss_tokens < 1:
            raise RuntimeError("Validation did not produce the configured complete sample")
        if int(model_loss_tokens_value) != loss_tokens:
            raise ValueError(
                "Model validation-token counter disagrees with packed batches"
            )
        if not math.isfinite(loss_sum):
            raise FloatingPointError("Non-finite validation loss")
        elapsed = time.perf_counter() - started
        if self.world_size > 1:
            elapsed_tensor = torch.tensor(
                elapsed, dtype=torch.float64, device=self.device
            )
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor)
        token_loss = loss_sum / loss_tokens
        metrics: dict[str, float | int] = {
            "train/step": int(train_step),
            "validation/loss": token_loss,
            "validation/token_loss": token_loss,
            "validation/loss_sum": loss_sum,
            "validation/perplexity": math.exp(min(token_loss, 80.0)),
            "validation/batches": self.max_batches,
            "validation/rows": rows,
            "validation/input_tokens": input_tokens,
            "validation/supervised_tokens": loss_tokens,
            "validation/seconds": elapsed,
            "validation/input_tokens_per_second": input_tokens / elapsed,
            "validation/supervised_tokens_per_second": loss_tokens / elapsed,
        }
        width = len(DOMAIN_ORDER)
        cursor = 5
        domain_loss_sums = values[cursor : cursor + width]
        cursor += width
        domain_loss_tokens = values[cursor : cursor + width]
        cursor += width
        domain_input_tokens = values[cursor : cursor + width]
        cursor += width
        domain_rows = values[cursor : cursor + width]
        for index, domain in enumerate(DOMAIN_ORDER):
            tokens = int(domain_loss_tokens[index])
            metrics.update(
                {
                    f"validation/{domain}/rows": int(domain_rows[index]),
                    f"validation/{domain}/input_tokens": int(
                        domain_input_tokens[index]
                    ),
                    f"validation/{domain}/supervised_tokens": tokens,
                }
            )
            if tokens:
                value = float(domain_loss_sums[index]) / tokens
                metrics[f"validation/{domain}/loss"] = value
                metrics[f"validation/{domain}/perplexity"] = math.exp(
                    min(value, 80.0)
                )
        return metrics


class Trainer:
    """Token-normalized optimizer loop with exact step-boundary resume."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
        *,
        device: torch.device | str,
        data_identity: str,
        tokenizer_manifest_sha256: str | None = None,
        tokenizer_vocabulary_sha256: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        logger: MetricLogger | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        training_geometry: Mapping[str, Any] | None = None,
        validation_runner: ValidationRunner | None = None,
        stop_controller: GracefulStopController | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if self.device.type == "cuda" and self.device.index is not None:
            current_cuda_device = torch.cuda.current_device()
            if current_cuda_device != self.device.index:
                raise ValueError(
                    "Trainer CUDA device must be the process's current device "
                    "before construction so rank-local RNG state is unambiguous: "
                    f"trainer={self.device}, current=cuda:{current_cuda_device}"
                )
        if not data_identity:
            raise ValueError("data_identity must not be empty")
        self.data_identity = data_identity
        if (tokenizer_manifest_sha256 is None) != (
            tokenizer_vocabulary_sha256 is None
        ):
            raise ValueError(
                "tokenizer manifest and vocabulary SHA-256 values must be supplied together"
            )
        try:
            self.tokenizer_manifest_sha256 = (
                None
                if tokenizer_manifest_sha256 is None
                else require_sha256(
                    tokenizer_manifest_sha256,
                    field="tokenizer_manifest_sha256",
                )
            )
            self.tokenizer_vocabulary_sha256 = (
                None
                if tokenizer_vocabulary_sha256 is None
                else require_sha256(
                    tokenizer_vocabulary_sha256,
                    field="tokenizer_vocabulary_sha256",
                )
            )
        except TokenizerIdentityError as exc:
            raise ValueError(str(exc)) from exc
        if self.tokenizer_manifest_sha256 is None and (
            checkpoint_path is not None or config.checkpoint_every > 0
        ):
            raise ValueError(
                "Checkpointing requires tokenizer manifest and vocabulary SHA-256 identities"
            )
        self.training_geometry = _canonical_training_geometry(training_geometry)
        self.rank, self.world_size = _distributed_rank_world()
        if _is_compiled_model(model) != config.compile_model:
            raise ValueError("TrainConfig.compile_model disagrees with the model wrapper")
        if config.global_microbatch_rows % self.world_size:
            raise ValueError("global_microbatch_rows must be divisible by world_size")
        self.local_batch_size = config.global_microbatch_rows // self.world_size
        if (config.eval_every > 0 or config.eval_at_start) != (
            validation_runner is not None
        ):
            raise ValueError(
                "Validation runner presence must match eval_every/eval_at_start"
            )
        if validation_runner is not None:
            validation_checks = {
                "rank": (validation_runner.rank, self.rank),
                "world_size": (validation_runner.world_size, self.world_size),
                "device": (validation_runner.device, self.device),
                "precision": (validation_runner.precision, config.precision),
                "global_microbatch_rows": (
                    validation_runner.sampler.global_microbatch_rows,
                    config.global_microbatch_rows,
                ),
            }
            for field, (found, expected) in validation_checks.items():
                if found != expected:
                    raise ValueError(
                        f"Validation {field} mismatch: found {found!r}, "
                        f"expected {expected!r}"
                    )
        if self.training_geometry is not None:
            expected = {
                "global_microbatch_rows": config.global_microbatch_rows,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "optimizer_updates": config.max_steps,
            }
            for field, runtime_value in expected.items():
                if self.training_geometry[field] != runtime_value:
                    raise ValueError(
                        f"TrainConfig {field} differs from frozen order geometry: "
                        f"found {runtime_value}, expected {self.training_geometry[field]}"
                    )
        seed_everything(config.seed + self.rank, deterministic=config.deterministic)
        for parameter in model.parameters():
            if parameter.device.type != self.device.type or (
                self.device.index is not None and parameter.device.index != self.device.index
            ):
                raise ValueError(
                    f"Model parameter is on {parameter.device}, trainer device is {self.device}"
                )
        self.optimizer = optimizer or build_optimizer(model, config, device=self.device)
        self.logger = logger or NullLogger()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.validation_runner = validation_runner
        self.validation_configuration = (
            None
            if validation_runner is None
            else {
                **validation_runner.configuration,
                "eval_every": config.eval_every,
                "eval_at_start": config.eval_at_start,
            }
        )
        self.stop_controller = stop_controller
        self.stop_signal = 0
        self._preserve_previous_on_next_save = False
        self.implementation_signature = self._compute_implementation_signature()
        self._last_checkpoint: tuple[Path, int] | None = None
        self.state = TrainState()
        if (
            config.precision == "bfloat16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise ValueError("This CUDA device does not support bfloat16 training")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("The training harness currently supports CPU and CUDA devices")
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def raw_model(self) -> torch.nn.Module:
        return _raw_model(self.model)

    def _model_config(self) -> dict[str, Any]:
        config = getattr(self.raw_model, "config", None)
        if not dataclasses.is_dataclass(config):
            raise TypeError("The raw model must expose a dataclass `config`")
        return dataclasses.asdict(config)

    def _parameter_dtypes(self) -> dict[str, str]:
        return {
            name: str(parameter.dtype)
            for name, parameter in self.raw_model.named_parameters()
        }

    def _runtime_signature(self) -> dict[str, Any]:
        signature: dict[str, Any] = {
            "device_type": self.device.type,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy_version": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        }
        if self.device.type == "cuda":
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            signature.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(index),
                    "cuda_capability": list(torch.cuda.get_device_capability(index)),
                    "cuda_total_memory_bytes": int(properties.total_memory),
                    "cuda_multiprocessor_count": int(
                        properties.multi_processor_count
                    ),
                    "cuda_runtime": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cuda_matmul_allow_fp16_reduced_precision_reduction": getattr(
                        torch.backends.cuda.matmul,
                        "allow_fp16_reduced_precision_reduction",
                        None,
                    ),
                    "cuda_matmul_allow_bf16_reduced_precision_reduction": getattr(
                        torch.backends.cuda.matmul,
                        "allow_bf16_reduced_precision_reduction",
                        None,
                    ),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                }
            )
        return signature

    def _compute_implementation_signature(self) -> dict[str, str]:
        result = {
            "trainer_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_class": (
                f"{type(self.raw_model).__module__}.{type(self.raw_model).__qualname__}"
            ),
            "trainer_source_sha256": sha256_file(__file__),
        }
        model_source = inspect.getsourcefile(type(self.raw_model))
        if model_source is not None and Path(model_source).is_file():
            result["model_source_sha256"] = sha256_file(model_source)
        data_source = inspect.getsourcefile(create_training_dataloader)
        if data_source is not None and Path(data_source).is_file():
            result["data_source_sha256"] = sha256_file(data_source)
        return result

    def _collect_rng_states(self) -> list[dict[str, Any]] | None:
        local = capture_rng_state()
        if self.world_size == 1:
            return [local]
        gathered: list[dict[str, Any] | None] | None = (
            [None] * self.world_size if self.rank == 0 else None
        )
        dist.gather_object(local, gathered, dst=0)
        if self.rank != 0:
            return None
        assert gathered is not None and all(item is not None for item in gathered)
        return [item for item in gathered if item is not None]

    def save_checkpoint(self, path: str | Path | None = None) -> None:
        """Synchronously commit a complete, atomic optimizer-step checkpoint.

        All ranks must call this method when distributed; rank zero writes one
        replicated-model checkpoint containing every rank's RNG state.
        """

        destination = Path(path) if path is not None else self.checkpoint_path
        if destination is None:
            raise ValueError("No checkpoint path was configured")
        if (
            self.tokenizer_manifest_sha256 is None
            or self.tokenizer_vocabulary_sha256 is None
        ):
            raise RuntimeError(
                "Refusing to create an unbound checkpoint: tokenizer manifest and "
                "vocabulary SHA-256 identities are required"
            )
        checkpoint_key = (destination.resolve(), self.state.completed_steps)
        skip_checkpoint = self._last_checkpoint == checkpoint_key
        dirty_boundary = any(
            parameter.grad is not None for parameter in self.model.parameters()
        )
        if self.world_size > 1:
            flags = torch.tensor(
                [int(skip_checkpoint), int(dirty_boundary)],
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            skip_count, dirty_count = (int(value) for value in flags)
            if skip_count not in (0, self.world_size):
                raise RuntimeError(
                    "Checkpoint generation cache diverged across distributed ranks"
                )
            if skip_count == self.world_size:
                return
            dirty_boundary = dirty_count > 0
        elif skip_checkpoint:
            return
        if dirty_boundary:
            raise RuntimeError("Checkpoints are only valid at clean optimizer-step boundaries")
        rng_states = self._collect_rng_states()
        local_save_error: BaseException | None = None
        preserve_previous = bool(
            self._preserve_previous_on_next_save
            and self.checkpoint_path is not None
            and destination.resolve() == self.checkpoint_path.resolve()
        )
        if self.rank == 0:
            try:
                assert rng_states is not None
                payload = {
                    "format": CHECKPOINT_FORMAT,
                    "format_version": CHECKPOINT_VERSION,
                    "torch_version": torch.__version__,
                    "world_size": self.world_size,
                    "runtime_signature": self._runtime_signature(),
                    # This is cached when Trainer is constructed. If a deployment
                    # overwrites source files mid-run, the checkpoint still records
                    # the code objects this process loaded rather than the new bytes.
                    "implementation_signature": self.implementation_signature,
                    "data_identity": self.data_identity,
                    "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
                    "tokenizer_vocabulary_sha256": self.tokenizer_vocabulary_sha256,
                    "training_geometry": self.training_geometry,
                    "validation_configuration": self.validation_configuration,
                    "model_config": self._model_config(),
                    "parameter_dtypes": self._parameter_dtypes(),
                    "optimizer_class": (
                        f"{type(self.optimizer).__module__}."
                        f"{type(self.optimizer).__qualname__}"
                    ),
                    "train_trajectory_config": self.config.trajectory_dict(),
                    "train_state": dataclasses.asdict(self.state),
                    "metadata": self.checkpoint_metadata,
                    "model": self.raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "rng_states": rng_states,
                }
                _atomic_torch_save(
                    payload,
                    destination,
                    preserve_previous=preserve_previous,
                )
            except BaseException as exc:
                local_save_error = exc
        if self.world_size > 1:
            status = [
                None
                if local_save_error is None
                else f"{type(local_save_error).__name__}: {local_save_error}"
            ]
            dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(
                    f"Rank-zero checkpoint commit failed: {status[0]}"
                ) from local_save_error
        elif local_save_error is not None:
            raise local_save_error
        if preserve_previous:
            self._preserve_previous_on_next_save = False
        self._last_checkpoint = checkpoint_key

    def load_checkpoint(self, path: str | Path) -> TrainState:
        """Restore full state and RNGs, validating all trajectory identities."""

        # Checkpoints are executable pickle data. Only load experiment-owned
        # files; weights_only=False is required for Python/NumPy RNG tuples.
        # mmap avoids materializing another complete ~15 GB CPU copy per rank.
        # The production launcher currently reads the durable generation in
        # place; benchmark that startup path on the chosen network volume.
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint root must be a dictionary")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported checkpoint format")
        if payload.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        checks = (
            ("torch_version", payload.get("torch_version"), torch.__version__),
            ("world_size", payload.get("world_size"), self.world_size),
            (
                "runtime_signature",
                payload.get("runtime_signature"),
                self._runtime_signature(),
            ),
            (
                "implementation_signature",
                payload.get("implementation_signature"),
                self.implementation_signature,
            ),
            ("data_identity", payload.get("data_identity"), self.data_identity),
            (
                "tokenizer_manifest_sha256",
                payload.get("tokenizer_manifest_sha256"),
                self.tokenizer_manifest_sha256,
            ),
            (
                "tokenizer_vocabulary_sha256",
                payload.get("tokenizer_vocabulary_sha256"),
                self.tokenizer_vocabulary_sha256,
            ),
            (
                "training_geometry",
                payload.get("training_geometry"),
                self.training_geometry,
            ),
            (
                "validation_configuration",
                payload.get("validation_configuration"),
                self.validation_configuration,
            ),
            ("model_config", payload.get("model_config"), self._model_config()),
            (
                "parameter_dtypes",
                payload.get("parameter_dtypes"),
                self._parameter_dtypes(),
            ),
            (
                "optimizer_class",
                payload.get("optimizer_class"),
                f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            ),
            (
                "train_trajectory_config",
                payload.get("train_trajectory_config"),
                self.config.trajectory_dict(),
            ),
        )
        for name, found, expected in checks:
            if found != expected:
                raise ValueError(
                    f"Checkpoint {name} mismatch: found {found!r}, expected {expected!r}"
                )
        candidate_state = TrainState.from_dict(payload["train_state"])
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) for key in metadata
        ):
            raise ValueError("Checkpoint metadata must be a string-keyed dictionary")
        if candidate_state.completed_steps > self.config.max_steps:
            raise ValueError("Checkpoint is beyond max_steps")
        if self.validation_runner is None:
            if candidate_state.last_validated_step != 0:
                raise ValueError(
                    "Checkpoint declares validation progress without validation authority"
                )
        elif candidate_state.last_validated_step not in (0, self.config.max_steps) and (
            self.config.eval_every == 0
            or candidate_state.last_validated_step % self.config.eval_every != 0
        ):
            raise ValueError(
                "Checkpoint last_validated_step is not on the configured schedule"
            )
        expected_microbatches = (
            candidate_state.completed_steps * self.config.gradient_accumulation_steps
        )
        if candidate_state.completed_microbatches != expected_microbatches:
            raise ValueError("Checkpoint was not saved at an accumulation boundary")
        expected_rows = (
            candidate_state.completed_microbatches
            * self.config.global_microbatch_rows
        )
        if candidate_state.consumed_rows != expected_rows:
            raise ValueError("Checkpoint consumed_rows does not match completed microbatches")
        if (
            candidate_state.consumed_supervised_tokens
            > candidate_state.consumed_input_tokens
        ):
            raise ValueError("Checkpoint supervised tokens exceed input tokens")
        if candidate_state.completed_steps == 0 and (
            candidate_state.consumed_input_tokens != 0
            or candidate_state.consumed_supervised_tokens != 0
        ):
            raise ValueError("Zero-step checkpoint has nonzero token counters")
        if self.training_geometry is not None:
            expected_input_tokens = (
                expected_rows * self.training_geometry["sequence_length"]
            )
            if candidate_state.consumed_input_tokens != expected_input_tokens:
                raise ValueError(
                    "Checkpoint consumed_input_tokens does not match frozen geometry"
                )
            if (
                sum(candidate_state.consumed_rows_per_domain.values())
                != candidate_state.consumed_rows
                or sum(candidate_state.consumed_input_tokens_per_domain.values())
                != candidate_state.consumed_input_tokens
                or sum(candidate_state.consumed_supervised_tokens_per_domain.values())
                != candidate_state.consumed_supervised_tokens
            ):
                raise ValueError(
                    "Checkpoint per-domain counters do not sum to global progress"
                )
            for domain in DOMAIN_ORDER:
                if (
                    candidate_state.consumed_input_tokens_per_domain[domain]
                    != candidate_state.consumed_rows_per_domain[domain]
                    * self.training_geometry["sequence_length"]
                    or candidate_state.consumed_supervised_tokens_per_domain[domain]
                    > candidate_state.consumed_input_tokens_per_domain[domain]
                ):
                    raise ValueError(
                        f"Checkpoint per-domain counters are inconsistent for {domain}"
                    )
            if (
                candidate_state.consumed_supervised_tokens
                > self.training_geometry["consumed_supervised_tokens"]
            ):
                raise ValueError(
                    "Checkpoint consumed supervised tokens exceed frozen authority"
                )
            if (
                candidate_state.completed_steps == self.config.max_steps
                and candidate_state.consumed_supervised_tokens
                != self.training_geometry["consumed_supervised_tokens"]
            ):
                raise ValueError(
                    "Final checkpoint supervised tokens differ from frozen authority"
                )
            if candidate_state.completed_steps == self.config.max_steps:
                for state_field, geometry_field in (
                    (
                        "consumed_input_tokens_per_domain",
                        "consumed_input_tokens_per_domain",
                    ),
                    (
                        "consumed_supervised_tokens_per_domain",
                        "consumed_supervised_tokens_per_domain",
                    ),
                ):
                    if getattr(candidate_state, state_field) != self.training_geometry[
                        geometry_field
                    ]:
                        raise ValueError(
                            f"Final checkpoint {state_field} differs from frozen authority"
                        )
        rng_states = payload.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != self.world_size:
            raise ValueError("Checkpoint does not contain one RNG state per rank")
        self.raw_model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        self.state = candidate_state
        self.checkpoint_metadata = dict(metadata)
        restore_rng_state(rng_states[self.rank])
        loaded_path = Path(path).resolve()
        if self.checkpoint_path is not None:
            canonical_previous = self.checkpoint_path.with_name(
                f"{self.checkpoint_path.stem}.previous{self.checkpoint_path.suffix}"
            ).resolve()
            self._preserve_previous_on_next_save = loaded_path == canonical_previous
        self._last_checkpoint = (loaded_path, self.state.completed_steps)
        return self.state

    def _move_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        return _move_batch_to_device(
            batch,
            device=self.device,
            local_batch_size=self.local_batch_size,
        )

    def _reduce_window_stats(
        self,
        local_loss_sum: torch.Tensor,
        local_loss_tokens: int,
        local_model_loss_tokens: torch.Tensor,
        local_input_tokens: int,
        local_domain_loss_sums: torch.Tensor | None = None,
        local_domain_loss_tokens: torch.Tensor | None = None,
        local_domain_input_tokens: torch.Tensor | None = None,
        local_domain_rows: torch.Tensor | None = None,
    ) -> tuple[float, int, int, dict[str, dict[str, float | int]] | None]:
        counts = torch.tensor(
            [local_loss_tokens, local_input_tokens],
            dtype=torch.float64,
            device=self.device,
        )
        domain_values = (
            local_domain_loss_sums,
            local_domain_loss_tokens,
            local_domain_input_tokens,
            local_domain_rows,
        )
        if any(value is None for value in domain_values) and not all(
            value is None for value in domain_values
        ):
            raise ValueError("Per-domain window statistics are only partially supplied")
        if (
            local_model_loss_tokens.device != self.device
            or local_model_loss_tokens.dtype != torch.int64
            or local_model_loss_tokens.numel() != 1
        ):
            raise ValueError("Model loss-token accumulator must be one int64 device scalar")
        parts = [
            local_loss_sum.reshape(1),
            counts,
            local_model_loss_tokens.reshape(1).to(torch.float64),
        ]
        if local_domain_loss_sums is not None:
            parts.extend(value.to(torch.float64) for value in domain_values if value is not None)
        stats = torch.cat(parts)
        if self.world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        values = stats.cpu().tolist()
        if int(values[3]) != int(values[1]):
            raise ValueError(
                "Model supervised-token counter disagrees with packed batches"
            )
        domain_stats: dict[str, dict[str, float | int]] | None = None
        if local_domain_loss_sums is not None:
            width = len(DOMAIN_ORDER)
            cursor = 4
            loss_sums = values[cursor : cursor + width]
            cursor += width
            loss_tokens = values[cursor : cursor + width]
            cursor += width
            input_tokens = values[cursor : cursor + width]
            cursor += width
            rows = values[cursor : cursor + width]
            domain_stats = {
                domain: {
                    "loss_sum": float(loss_sums[index]),
                    "loss_tokens": int(loss_tokens[index]),
                    "input_tokens": int(input_tokens[index]),
                    "rows": int(rows[index]),
                }
                for index, domain in enumerate(DOMAIN_ORDER)
            }
        return float(values[0]), int(values[1]), int(values[2]), domain_stats

    def _max_step_timings(
        self,
        elapsed: float,
        data_wait: float,
    ) -> tuple[float, float, float]:
        wait_fraction = data_wait / elapsed
        if self.world_size == 1:
            return elapsed, data_wait, wait_fraction
        values = torch.tensor(
            [elapsed, data_wait, wait_fraction],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        result = values.cpu().tolist()
        return float(result[0]), float(result[1]), float(result[2])

    def _validate_batch_source(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
    ) -> None:
        """Bind the runtime loader cursor to the frozen trainer trajectory."""

        if self.training_geometry is None:
            return
        sampler = getattr(batches, "batch_sampler", None)
        if not isinstance(sampler, DistributedBatchSampler):
            raise ValueError(
                "Frozen-order training requires a DistributedBatchSampler-backed "
                "DataLoader"
            )
        checks = {
            "data_identity": (sampler.data_identity, self.data_identity),
            "training_geometry": (
                sampler.training_geometry,
                self.training_geometry,
            ),
            "global_microbatch_rows": (
                sampler.global_microbatch_rows,
                self.config.global_microbatch_rows,
            ),
            "gradient_accumulation_steps": (
                sampler.gradient_accumulation_steps,
                self.config.gradient_accumulation_steps,
            ),
            "optimizer_update_rows": (
                sampler.optimizer_update_rows,
                self.training_geometry["optimizer_update_rows"],
            ),
            "optimizer_updates": (
                sampler.total_optimizer_updates,
                self.config.max_steps,
            ),
            "resume_global_microbatch": (
                sampler.start_global_microbatch,
                self.state.completed_microbatches,
            ),
        }
        for field, (found, expected) in checks.items():
            if found != expected:
                raise ValueError(
                    f"Training loader {field} mismatch: found {found}, expected {expected}"
                )

    def _log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        if self.rank != 0:
            return
        try:
            with preserve_host_rng_state():
                self.logger.log(metrics)
        except Exception as exc:
            # Tracking must never strand other ranks between collectives or
            # prevent a scheduled/preemption checkpoint.
            print(
                f"warning: disabling failed metric logger: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self.logger = NullLogger()

    def _distributed_stop_signal(self) -> int:
        if self.stop_controller is not None:
            self.stop_controller.poll_external_request()
        requested = (
            0 if self.stop_controller is None else self.stop_controller.requested_signal
        )
        if self.world_size == 1:
            return requested
        signal_tensor = torch.tensor(
            requested,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(signal_tensor, op=dist.ReduceOp.MAX)
        return int(signal_tensor)

    def evaluate_validation(self) -> dict[str, float | int]:
        if self.validation_runner is None:
            raise RuntimeError("No validation runner is configured")
        metrics = self.validation_runner.evaluate(
            self.model,
            train_step=self.state.completed_steps,
        )
        self._log_metrics(metrics)
        if self.state.completed_steps > self.state.last_validated_step:
            self.state.last_validated_step = self.state.completed_steps
            # Validation progress is checkpoint authority even though it does
            # not alter weights/RNG. Force the next save at this same step to
            # publish the newly completed evaluation cursor.
            self._last_checkpoint = None
        return metrics

    def _validation_due(self) -> bool:
        if self.validation_runner is None:
            return False
        step = self.state.completed_steps
        if step <= self.state.last_validated_step or step == 0:
            return False
        return step == self.config.max_steps or (
            self.config.eval_every > 0 and step % self.config.eval_every == 0
        )

    def train(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        until_step: int | None = None,
    ) -> dict[str, float | int]:
        """Train to an absolute optimizer step and return its final metrics.

        For immutable-order training, ``batches`` must start at
        ``state.completed_microbatches``.  The supplied data loader's sampler
        provides exactly that contract. A fixed-batch stream may simply repeat
        forever. Checkpoints are committed only after a full accumulation
        window and optimizer update.
        """

        target_step = self.config.max_steps if until_step is None else int(until_step)
        if not self.state.completed_steps <= target_step <= self.config.max_steps:
            raise ValueError("until_step must be between current progress and max_steps")
        initial_stop = self._distributed_stop_signal()
        if initial_stop:
            if self.checkpoint_path is None:
                raise RuntimeError(
                    "A graceful stop was requested but no checkpoint path is configured"
                )
            self.stop_signal = initial_stop
            self.save_checkpoint()
            return {}
        # A preemption checkpoint deliberately wins over a potentially long
        # held-out pass. Catch up that evaluation on resume before another
        # optimizer update (or before returning from an already-final state).
        if self._validation_due():
            self.evaluate_validation()
            if self.checkpoint_path is not None:
                self.save_checkpoint()
        if target_step == self.state.completed_steps:
            return {}
        self._validate_batch_source(batches)

        # DataLoader iterator construction samples a worker base seed from the
        # global torch RNG even when the dataset itself is deterministic. Keep
        # infrastructure RNG use from perturbing the model RNG across resume.
        iterator_rng_state = capture_rng_state()
        iterator = iter(batches)
        restore_rng_state(iterator_rng_state)
        self.model.train()
        last_metrics: dict[str, float | int] = {}
        parameters = _optimizer_parameters(self.model)
        for _ in range(self.state.completed_steps, target_step):
            started = time.perf_counter()
            step_index = self.state.completed_steps
            learning_rate = learning_rate_for_step(self.config, step_index)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

            local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            local_loss_tokens = 0
            local_model_loss_tokens = torch.zeros(
                (), dtype=torch.int64, device=self.device
            )
            local_input_tokens = 0
            local_data_wait_seconds = 0.0
            local_domain_loss_sums = torch.zeros(
                len(DOMAIN_ORDER), dtype=torch.float64, device=self.device
            )
            local_domain_loss_tokens = torch.zeros(
                len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
            )
            local_domain_input_tokens = torch.zeros(
                len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
            )
            local_domain_rows = torch.zeros(
                len(DOMAIN_ORDER), dtype=torch.int64, device=self.device
            )
            domain_metrics_available = True
            consumed_microbatches = 0
            try:
                for accumulation_index in range(self.config.gradient_accumulation_steps):
                    data_wait_started = time.perf_counter()
                    raw_batch = next(iterator)
                    local_data_wait_seconds += (
                        time.perf_counter() - data_wait_started
                    )
                    batch, token_count = self._move_batch(raw_batch)
                    consumed_microbatches += 1
                    should_sync = (
                        accumulation_index + 1 == self.config.gradient_accumulation_steps
                    )
                    sync_context = (
                        contextlib.nullcontext()
                        if should_sync or not hasattr(self.model, "no_sync")
                        else self.model.no_sync()
                    )
                    autocast_enabled = self.config.precision != "float32"
                    # DDP's no_sync context must cover both forward and
                    # backward; entering it only for backward is insufficient.
                    with sync_context:
                        with torch.autocast(
                            self.device.type,
                            dtype=_dtype_for_precision(self.config.precision),
                            enabled=autocast_enabled,
                        ):
                            output = self.model(
                                batch["input_ids"],
                                batch["position_ids"],
                                batch["document_ids"],
                                batch["labels"],
                            )
                        if output.loss_sum is None or output.num_loss_tokens is None:
                            raise RuntimeError("Model did not return training loss statistics")
                        if token_count < 1:
                            raise ValueError("Microbatch contains no supervised tokens")
                        if (
                            output.num_loss_tokens.dtype != torch.int64
                            or output.num_loss_tokens.numel() != 1
                        ):
                            raise ValueError(
                                "Model supervised-token counter must be one int64 scalar"
                            )
                        output.loss_sum.backward()
                    local_loss_sum += output.loss_sum.detach().double()
                    local_loss_tokens += token_count
                    local_model_loss_tokens += output.num_loss_tokens.detach()
                    local_input_tokens += batch["input_ids"].numel()
                    domain_ids = batch.get("domain_ids")
                    if domain_ids is None:
                        domain_metrics_available = False
                        if self.training_geometry is not None:
                            raise ValueError(
                                "Production packed batches must include domain_ids"
                            )
                    else:
                        if output.loss_sums_per_row is None:
                            raise RuntimeError(
                                "Model did not return per-row loss statistics"
                            )
                        row_loss_tokens = batch["labels"].ne(-100).sum(dim=1)
                        row_input_tokens = torch.full_like(
                            row_loss_tokens,
                            batch["input_ids"].shape[1],
                        )
                        local_domain_loss_sums += torch.bincount(
                            domain_ids,
                            weights=output.loss_sums_per_row.detach().double(),
                            minlength=len(DOMAIN_ORDER),
                        )
                        local_domain_loss_tokens += torch.bincount(
                            domain_ids,
                            weights=row_loss_tokens,
                            minlength=len(DOMAIN_ORDER),
                        ).to(torch.int64)
                        local_domain_input_tokens += torch.bincount(
                            domain_ids,
                            weights=row_input_tokens,
                            minlength=len(DOMAIN_ORDER),
                        ).to(torch.int64)
                        local_domain_rows += torch.bincount(
                            domain_ids,
                            minlength=len(DOMAIN_ORDER),
                        ).to(torch.int64)
            except StopIteration as exc:
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "Data iterator ended before max_steps at an accumulation boundary"
                    if consumed_microbatches == 0
                    else "Data iterator ended in the middle of a gradient-accumulation window"
                ) from exc

            global_loss_sum, global_loss_tokens, global_input_tokens, domain_stats = (
                self._reduce_window_stats(
                    local_loss_sum,
                    local_loss_tokens,
                    local_model_loss_tokens,
                    local_input_tokens,
                    local_domain_loss_sums if domain_metrics_available else None,
                    local_domain_loss_tokens if domain_metrics_available else None,
                    local_domain_input_tokens if domain_metrics_available else None,
                    local_domain_rows if domain_metrics_available else None,
                )
            )
            if global_loss_tokens < 1:
                raise ValueError("Accumulation window contains no supervised tokens")
            if not math.isfinite(global_loss_sum):
                raise FloatingPointError("Non-finite training loss")
            update_rows = (
                self.config.global_microbatch_rows
                * self.config.gradient_accumulation_steps
            )
            if domain_stats is not None and (
                sum(int(values["rows"]) for values in domain_stats.values())
                != update_rows
                or sum(
                    int(values["input_tokens"]) for values in domain_stats.values()
                )
                != global_input_tokens
                or sum(
                    int(values["loss_tokens"]) for values in domain_stats.values()
                )
                != global_loss_tokens
            ):
                raise ValueError(
                    "Per-domain optimizer-window counters do not match global totals"
                )
            if self.training_geometry is not None:
                expected_window_inputs = (
                    update_rows * self.training_geometry["sequence_length"]
                )
                if global_input_tokens != expected_window_inputs:
                    raise ValueError(
                        "Optimizer update input-token count differs from frozen geometry: "
                        f"found {global_input_tokens}, expected {expected_window_inputs}"
                    )
                projected_input_tokens = (
                    self.state.consumed_input_tokens + global_input_tokens
                )
                projected_supervised_tokens = (
                    self.state.consumed_supervised_tokens + global_loss_tokens
                )
                if (
                    projected_input_tokens
                    > self.training_geometry["consumed_input_tokens"]
                    or projected_supervised_tokens
                    > self.training_geometry["consumed_supervised_tokens"]
                ):
                    raise RuntimeError(
                        "Optimizer update would exceed the frozen order authority"
                    )
                if step_index + 1 == self.config.max_steps and (
                    projected_input_tokens
                    != self.training_geometry["consumed_input_tokens"]
                    or projected_supervised_tokens
                    != self.training_geometry["consumed_supervised_tokens"]
                ):
                    raise RuntimeError(
                        "Final optimizer update counters differ from frozen order authority"
                    )
            # DDP averages the accumulated rank gradients. Convert them from a
            # sum of local token losses to the global mean token loss.
            gradient_scale = self.world_size / global_loss_tokens
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.completed_steps += 1
            self.state.completed_microbatches += self.config.gradient_accumulation_steps
            self.state.consumed_rows += update_rows
            self.state.consumed_input_tokens += global_input_tokens
            self.state.consumed_supervised_tokens += global_loss_tokens
            if domain_stats is not None:
                for domain in DOMAIN_ORDER:
                    self.state.consumed_rows_per_domain[domain] += int(
                        domain_stats[domain]["rows"]
                    )
                    self.state.consumed_input_tokens_per_domain[domain] += int(
                        domain_stats[domain]["input_tokens"]
                    )
                    self.state.consumed_supervised_tokens_per_domain[domain] += int(
                        domain_stats[domain]["loss_tokens"]
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            grad_norm = float(grad_norm_tensor)
            elapsed, data_wait_seconds, data_wait_fraction = self._max_step_timings(
                time.perf_counter() - started,
                local_data_wait_seconds,
            )
            token_loss = global_loss_sum / global_loss_tokens
            last_metrics = {
                "train/step": self.state.completed_steps,
                "train/loss": token_loss,
                "train/token_loss": token_loss,
                "train/loss_sum": global_loss_sum,
                "train/perplexity": math.exp(min(token_loss, 80.0)),
                "train/learning_rate": learning_rate,
                "train/grad_norm": grad_norm,
                "train/rows": self.state.consumed_rows,
                "train/input_tokens": self.state.consumed_input_tokens,
                "train/supervised_tokens": self.state.consumed_supervised_tokens,
                "train/loss_tokens": self.state.consumed_supervised_tokens,
                "train/microbatches": self.state.completed_microbatches,
                "perf/input_tokens_per_second": global_input_tokens / elapsed,
                "perf/loss_tokens_per_second": global_loss_tokens / elapsed,
                "perf/step_seconds": elapsed,
                "perf/data_wait_seconds": data_wait_seconds,
                "perf/data_wait_fraction": data_wait_fraction,
            }
            if self.device.type == "cuda":
                last_metrics.update(
                    {
                        "system/cuda_memory_allocated_bytes": torch.cuda.memory_allocated(
                            self.device
                        ),
                        "system/cuda_memory_reserved_bytes": torch.cuda.memory_reserved(
                            self.device
                        ),
                        "system/cuda_peak_memory_allocated_bytes": (
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                        "system/cuda_peak_memory_reserved_bytes": (
                            torch.cuda.max_memory_reserved(self.device)
                        ),
                    }
                )
            if domain_stats is not None:
                for domain in DOMAIN_ORDER:
                    window_tokens = int(domain_stats[domain]["loss_tokens"])
                    last_metrics.update(
                        {
                            f"train/{domain}/rows": self.state.consumed_rows_per_domain[
                                domain
                            ],
                            f"train/{domain}/input_tokens": (
                                self.state.consumed_input_tokens_per_domain[domain]
                            ),
                            f"train/{domain}/supervised_tokens": (
                                self.state.consumed_supervised_tokens_per_domain[domain]
                            ),
                        }
                    )
                    if window_tokens:
                        domain_loss = (
                            float(domain_stats[domain]["loss_sum"]) / window_tokens
                        )
                        last_metrics.update(
                            {
                                f"train/{domain}/loss": domain_loss,
                                f"train/{domain}/perplexity": math.exp(
                                    min(domain_loss, 80.0)
                                ),
                            }
                        )
            if self.state.completed_steps % self.config.log_every == 0:
                self._log_metrics(last_metrics)
            requested_stop = self._distributed_stop_signal()
            if requested_stop:
                if self.checkpoint_path is None:
                    raise RuntimeError(
                        "A graceful stop was requested but no checkpoint path is configured"
                    )
                self.stop_signal = requested_stop
                self.save_checkpoint()
                self._log_metrics(
                    {
                        "train/step": self.state.completed_steps,
                        "system/graceful_stop_signal": requested_stop,
                    }
                )
                break
            if self._validation_due():
                self.evaluate_validation()
            if (
                self.checkpoint_path is not None
                and self.config.checkpoint_every
                and self.state.completed_steps % self.config.checkpoint_every == 0
            ):
                self.save_checkpoint()
            # A signal received during validation or checkpoint I/O is handled
            # without waiting for another optimizer update.
            requested_stop = self._distributed_stop_signal()
            if requested_stop:
                if self.checkpoint_path is None:
                    raise RuntimeError(
                        "A graceful stop was requested but no checkpoint path is configured"
                    )
                self.stop_signal = requested_stop
                self.save_checkpoint()
                self._log_metrics(
                    {
                        "train/step": self.state.completed_steps,
                        "system/graceful_stop_signal": requested_stop,
                    }
                )
                break
        if (
            self.training_geometry is not None
            and self.state.completed_steps == self.config.max_steps
        ):
            final_checks = {
                "consumed_rows": (
                    self.state.consumed_rows,
                    self.training_geometry["consumed_rows"],
                ),
                "consumed_input_tokens": (
                    self.state.consumed_input_tokens,
                    self.training_geometry["consumed_input_tokens"],
                ),
                "consumed_supervised_tokens": (
                    self.state.consumed_supervised_tokens,
                    self.training_geometry["consumed_supervised_tokens"],
                ),
            }
            for field, (found, expected) in final_checks.items():
                if found != expected:
                    raise RuntimeError(
                        f"Final trainer {field} differs from frozen order authority: "
                        f"found {found}, expected {expected}"
                    )
            final_domain_checks = {
                "consumed_input_tokens_per_domain": (
                    self.state.consumed_input_tokens_per_domain,
                    self.training_geometry["consumed_input_tokens_per_domain"],
                ),
                "consumed_supervised_tokens_per_domain": (
                    self.state.consumed_supervised_tokens_per_domain,
                    self.training_geometry[
                        "consumed_supervised_tokens_per_domain"
                    ],
                ),
            }
            for field, (found, expected) in final_domain_checks.items():
                if found != expected:
                    raise RuntimeError(
                        f"Final trainer {field} differs from frozen order authority: "
                        f"found {found}, expected {expected}"
                    )
        return last_metrics


@torch.no_grad()
def evaluate_fixed_batch(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str,
    precision: Precision = "float32",
) -> float:
    """Return exact token-normalized loss on one fixed packed batch."""

    target = torch.device(device)
    was_training = model.training
    model.eval()
    moved = {
        key: batch[key].to(target)
        for key in _REQUIRED_BATCH_KEYS
    }
    with torch.autocast(
        target.type,
        dtype=_dtype_for_precision(precision),
        enabled=precision != "float32",
    ):
        output = model(
            moved["input_ids"],
            moved["position_ids"],
            moved["document_ids"],
            moved["labels"],
        )
    if was_training:
        model.train()
    if output.loss is None:
        raise RuntimeError("Model did not return an evaluation loss")
    return float(output.loss)


def _resolve_device(requested: str, *, local_rank: int) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = f"cuda:{local_rank}"
    return torch.device(requested)


def _init_distributed_if_requested(device: torch.device) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def _distributed_verify_order(
    path: Path,
    *,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], str, str]:
    """Verify an immutable order once and broadcast rank-zero failure."""

    if world_size == 1:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_digest = sha256_file(path)
        order_digest = verify_order_payload_checksum(path)
    else:
        verification: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                verification[0] = {
                    "payload": json.loads(path.read_text(encoding="utf-8")),
                    "manifest_digest": sha256_file(path),
                    "order_digest": verify_order_payload_checksum(path),
                }
            except Exception as exc:
                verification[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(verification, src=0)
        assert verification[0] is not None
        if "error" in verification[0]:
            raise RuntimeError(
                f"Rank-zero order validation failed for {path}: "
                f"{verification[0]['error']}"
            )
        payload = verification[0]["payload"]
        manifest_digest = verification[0]["manifest_digest"]
        order_digest = verification[0]["order_digest"]
    return payload, manifest_digest, order_digest


def _distributed_verify_tokenizer(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_vocab_size: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Authenticate the tokenizer once on rank zero and broadcast its identity."""

    result: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            identity = verify_tokenizer_identity(
                path,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_vocab_size=expected_vocab_size,
            )
            result[0] = {
                "path": str(path.resolve()),
                "manifest_sha256": identity.manifest_sha256,
                "vocabulary_sha256": identity.vocabulary_sha256,
                "vocab_size": identity.vocab_size,
            }
        except BaseException as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if world_size > 1:
        dist.broadcast_object_list(result, src=0)
    assert result[0] is not None
    if "error" in result[0]:
        raise RuntimeError(f"Tokenizer identity verification failed: {result[0]['error']}")
    return result[0]


def _raise_if_distributed_stage_failed(
    local_error: BaseException | None,
    *,
    stage: str,
    rank: int,
    world_size: int,
) -> None:
    """Make recoverable rank-local setup failures terminate every rank coherently."""

    local_status = (
        None
        if local_error is None
        else f"rank {rank}: {type(local_error).__name__}: {local_error}"
    )
    if world_size == 1:
        if local_error is not None:
            raise local_error
        return
    statuses: list[str | None] = [None] * world_size
    dist.all_gather_object(statuses, local_status)
    failures = [status for status in statuses if status is not None]
    if failures:
        raise RuntimeError(
            f"Distributed setup stage {stage!r} failed: {'; '.join(failures)}"
        ) from local_error


def _close_loader(loader: Any, sampler: DistributedBatchSampler) -> None:
    sampler.close()
    close_dataset = getattr(loader.dataset, "close", None)
    if close_dataset is not None:
        close_dataset()


def _finalize_training_run(trainer: Trainer) -> int:
    """Publish final state, then catch a stop received during that publication.

    Every rank calls both collectives in the same order.  This closes the
    otherwise vulnerable window between ``Trainer.train`` returning and the
    caller restoring the process signal handlers.
    """

    stop_already_reported = bool(trainer.stop_signal)
    trainer.save_checkpoint()
    requested_stop = trainer._distributed_stop_signal()
    if requested_stop:
        trainer.stop_signal = requested_stop
        if not stop_already_reported:
            trainer._log_metrics(
                {
                    "train/step": trainer.state.completed_steps,
                    "system/graceful_stop_signal": requested_stop,
                }
            )
        return 128 + requested_stop
    return 0


def _bind_wandb_run_id(trainer: Trainer, run_id: str | None) -> None:
    """Make a tracker identity checkpoint-authoritative on every rank.

    A resume can be preempted before its first new optimizer update. If W&B had
    to create or deliberately replace the run ID, invalidate the same-step
    checkpoint cache so that clean stop still publishes that identity.
    """

    if run_id is None:
        return
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("W&B run ID must be a non-empty string")
    if trainer.checkpoint_metadata.get("wandb_run_id") == run_id:
        return
    trainer.checkpoint_metadata["wandb_run_id"] = run_id
    trainer._last_checkpoint = None


def _run_initialized_training(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    device: torch.device,
    rank: int,
    world_size: int,
    training_geometry: Mapping[str, Any],
    validation_geometry: Mapping[str, Any] | None,
) -> int:
    """Run all post-process-group setup and training with coherent cleanup."""

    checkpoint_lease: CheckpointLease | None = None
    validation_loader: Any | None = None
    validation_sampler: DistributedBatchSampler | None = None
    loader: Any | None = None
    sampler: DistributedBatchSampler | None = None
    logger: MetricLogger = NullLogger()
    try:
        lease_error: BaseException | None = None
        removed_temporaries: list[Path] = []
        if rank == 0:
            checkpoint_lease = CheckpointLease(args.checkpoint)
            try:
                checkpoint_lease.acquire()
                atexit.register(checkpoint_lease.release)
                removed_temporaries = reconcile_checkpoint_temporaries(
                    args.checkpoint
                )
            except BaseException as exc:
                lease_error = exc
        _raise_if_distributed_stage_failed(
            lease_error,
            stage="checkpoint lease and recovery",
            rank=rank,
            world_size=world_size,
        )
        if removed_temporaries:
            print(
                json.dumps(
                    {
                        "checkpoint/reconciled_temporaries": [
                            str(path) for path in removed_temporaries
                        ]
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        precision: Precision = args.precision or (
            "bfloat16" if device.type == "cuda" else "float32"
        )
        seed_everything(args.seed + rank, deterministic=args.deterministic)

        order_payload, order_manifest_digest, order_payload_digest = (
            _distributed_verify_order(
                args.order_manifest,
                rank=rank,
                world_size=world_size,
            )
        )
        if order_payload.get("split") != "train":
            parser.error("--order-manifest must identify the train split")
        tokenizer_identity = _distributed_verify_tokenizer(
            args.tokenizer,
            expected_manifest_sha256=str(
                order_payload.get("tokenizer_manifest_sha256", "")
            ),
            expected_vocab_size=int(order_payload["vocab_size"]),
            rank=rank,
            world_size=world_size,
        )
        validation_payload: dict[str, Any] | None = None
        validation_manifest_digest: str | None = None
        validation_payload_digest: str | None = None
        if args.validation_order_manifest is not None:
            (
                validation_payload,
                validation_manifest_digest,
                validation_payload_digest,
            ) = _distributed_verify_order(
                args.validation_order_manifest,
                rank=rank,
                world_size=world_size,
            )
            if validation_payload.get("split") != "validation":
                parser.error("--validation-order-manifest must identify validation")
            for field in (
                "vocab_size",
                "sequence_length",
                "tokenizer_manifest_sha256",
            ):
                if validation_payload.get(field) != order_payload.get(field):
                    parser.error(
                        f"validation order {field} differs from the training order"
                    )

        vocab_size = int(order_payload["vocab_size"])
        sequence_length = int(order_payload["sequence_length"])
        activation_checkpointing = (
            args.model_size == "1.3b"
            if args.activation_checkpointing is None
            else args.activation_checkpointing
        )
        if args.model_size == "tiny":
            model_config = dataclasses.replace(
                tiny_model_config(
                    vocab_size=vocab_size,
                    max_seq_len=sequence_length,
                ),
                activation_checkpointing=activation_checkpointing,
            )
        else:
            model_config = ModelConfig(
                vocab_size=vocab_size,
                max_seq_len=sequence_length,
                activation_checkpointing=activation_checkpointing,
            )

        # Validate the complete optimizer trajectory before allocating the
        # multi-gigabyte production model on every rank.
        train_config = TrainConfig(
            max_steps=args.steps,
            global_microbatch_rows=args.global_microbatch_rows,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=min(args.warmup_steps, args.steps),
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            adam_eps=args.adam_eps,
            max_grad_norm=args.max_grad_norm,
            precision=precision,
            seed=args.seed,
            deterministic=args.deterministic,
            compile_model=args.compile,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            eval_at_start=args.eval_at_start,
            fused_adamw=args.fused_adamw,
        )

        model: torch.nn.Module | None = None
        model_error: BaseException | None = None
        try:
            model = CausalLM(model_config, device=device, dtype=torch.float32)
            if args.compile:
                model = torch.compile(model)
        except BaseException as exc:
            model_error = exc
        _raise_if_distributed_stage_failed(
            model_error,
            stage="model allocation and compilation",
            rank=rank,
            world_size=world_size,
        )
        assert model is not None
        if world_size > 1:
            ddp_error: BaseException | None = None
            try:
                model = wrap_distributed_model(model, device=device)
            except BaseException as exc:
                ddp_error = exc
            _raise_if_distributed_stage_failed(
                ddp_error,
                stage="DDP construction",
                rank=rank,
                world_size=world_size,
            )

        validation_runner: ValidationRunner | None = None
        validation_error: BaseException | None = None
        if args.validation_order_manifest is not None:
            try:
                validation_loader, validation_sampler = create_training_dataloader(
                    args.validation_order_manifest,
                    global_microbatch_rows=args.global_microbatch_rows,
                    # Evaluation has no optimizer-update boundary. Using train
                    # accumulation would discard complete held-out microbatches.
                    gradient_accumulation_steps=1,
                    rank=rank,
                    world_size=world_size,
                    start_global_microbatch=0,
                    num_workers=args.workers,
                    pin_memory=device.type == "cuda",
                    verify_order_checksum=False,
                    verify_payload_checksums=args.verify_packed_payloads,
                )
                validation_runner = ValidationRunner(
                    validation_loader,
                    device=device,
                    precision=precision,
                    max_batches=args.eval_batches,
                )
            except BaseException as exc:
                validation_error = exc
        _raise_if_distributed_stage_failed(
            validation_error,
            stage="validation loader construction",
            rank=rank,
            world_size=world_size,
        )

        run_config = {
            "model": dataclasses.asdict(model_config),
            "training": dataclasses.asdict(train_config),
            "order_manifest": str(args.order_manifest.resolve()),
            "order_manifest_sha256": order_manifest_digest,
            "order_payload_sha256": order_payload_digest,
            "tokenizer": tokenizer_identity,
            "training_geometry": dict(training_geometry),
            "validation": (
                None
                if args.validation_order_manifest is None
                else {
                    "order_manifest": str(
                        args.validation_order_manifest.resolve()
                    ),
                    "order_manifest_sha256": validation_manifest_digest,
                    "order_payload_sha256": validation_payload_digest,
                    "geometry": validation_geometry,
                    "configuration": validation_runner.configuration,
                    "eval_every": args.eval_every,
                    "eval_at_start": args.eval_at_start,
                }
            ),
            "world_size": world_size,
        }
        stop_controller = GracefulStopController()
        trainer: Trainer | None = None
        trainer_error: BaseException | None = None
        try:
            trainer = Trainer(
                model,
                train_config,
                device=device,
                data_identity=(
                    f"order-manifest-sha256:{order_manifest_digest};"
                    f"order-payload-sha256:{order_payload_digest};"
                    "tokenizer-manifest-sha256:"
                    f"{order_payload['tokenizer_manifest_sha256']};"
                    "tokenizer-vocabulary-sha256:"
                    f"{tokenizer_identity['vocabulary_sha256']}"
                ),
                tokenizer_manifest_sha256=str(
                    order_payload["tokenizer_manifest_sha256"]
                ),
                tokenizer_vocabulary_sha256=str(
                    tokenizer_identity["vocabulary_sha256"]
                ),
                logger=NullLogger(),
                checkpoint_path=args.checkpoint,
                training_geometry=training_geometry,
                validation_runner=validation_runner,
                stop_controller=stop_controller,
            )
        except BaseException as exc:
            trainer_error = exc
        _raise_if_distributed_stage_failed(
            trainer_error,
            stage="trainer and optimizer construction",
            rank=rank,
            world_size=world_size,
        )
        assert trainer is not None

        resume_error: BaseException | None = None
        if args.resume is not None:
            try:
                trainer.load_checkpoint(args.resume)
            except BaseException as exc:
                resume_error = exc
        _raise_if_distributed_stage_failed(
            resume_error,
            stage="checkpoint resume",
            rank=rank,
            world_size=world_size,
        )

        wandb_logger: MetricLogger = NullLogger()
        wandb_initialization_error: str | None = None
        initialized_wandb_run_id: str | None = None
        if rank == 0:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

            def create_wandb_logger() -> MetricLogger:
                wandb_run_id = args.wandb_run_id or trainer.checkpoint_metadata.get(
                    "wandb_run_id"
                )
                if wandb_run_id is not None and not isinstance(wandb_run_id, str):
                    raise ValueError(
                        "Checkpoint wandb_run_id metadata must be a string"
                    )
                return WandbLogger(
                    mode=args.wandb_mode,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    group=args.wandb_group,
                    tags=args.wandb_tags,
                    run_id=wandb_run_id,
                    config=run_config,
                    directory=args.checkpoint.parent,
                )

            with preserve_host_rng_state():
                wandb_logger, wandb_initialization_error = (
                    initialize_optional_metric_logger(create_wandb_logger)
                )
            if wandb_initialization_error is None:
                if (
                    isinstance(wandb_logger, WandbLogger)
                    and wandb_logger.run_id is not None
                ):
                    initialized_wandb_run_id = wandb_logger.run_id
        if world_size > 1:
            initialization_status = [
                wandb_initialization_error,
                initialized_wandb_run_id,
            ]
            dist.broadcast_object_list(initialization_status, src=0)
            wandb_initialization_error, initialized_wandb_run_id = (
                initialization_status
            )
        _bind_wandb_run_id(trainer, initialized_wandb_run_id)
        if rank == 0 and wandb_initialization_error is not None:
            print(
                "warning: W&B initialization failed; continuing with JSON "
                f"console metrics only: {wandb_initialization_error}",
                file=sys.stderr,
                flush=True,
            )
        logger = (
            CompositeLogger(ConsoleLogger(), wandb_logger)
            if rank == 0
            else NullLogger()
        )
        trainer.logger = logger

        train_loader_error: BaseException | None = None
        try:
            loader, sampler = create_training_dataloader(
                args.order_manifest,
                global_microbatch_rows=args.global_microbatch_rows,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                rank=rank,
                world_size=world_size,
                start_global_microbatch=trainer.state.completed_microbatches,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                verify_order_checksum=False,
                verify_payload_checksums=args.verify_packed_payloads,
            )
        except BaseException as exc:
            train_loader_error = exc
        _raise_if_distributed_stage_failed(
            train_loader_error,
            stage="training loader construction",
            rank=rank,
            world_size=world_size,
        )
        assert loader is not None and sampler is not None

        with stop_controller:
            if (
                trainer.validation_runner is not None
                and train_config.eval_at_start
                and trainer.state.completed_steps == 0
            ):
                trainer.evaluate_validation()
            trainer.train(loader)
            exit_code = _finalize_training_run(trainer)
        return exit_code
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors: list[str] = []
        cleanup_operations: list[tuple[str, Any]] = [("metric logger", logger.finish)]
        if loader is not None and sampler is not None:
            cleanup_operations.append(
                ("training loader", lambda: _close_loader(loader, sampler))
            )
        if validation_loader is not None and validation_sampler is not None:
            cleanup_operations.append(
                (
                    "validation loader",
                    lambda: _close_loader(validation_loader, validation_sampler),
                )
            )
        if checkpoint_lease is not None:
            cleanup_operations.append(("checkpoint lease", checkpoint_lease.release))
        for label, operation in cleanup_operations:
            try:
                operation()
            except BaseException as exc:
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            message = "runtime cleanup failures: " + "; ".join(cleanup_errors)
            if active_exception:
                print(f"warning: {message}", file=sys.stderr, flush=True)
            else:
                raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-manifest", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help=(
            "immutable tokenizer directory; TOKENIZER_MANIFEST.json and the "
            "canonical token-to-ID mapping are authenticated before training"
        ),
    )
    parser.add_argument("--model-size", choices=("tiny", "1.3b"), default="1.3b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--steps",
        type=int,
        help="optimizer updates; omitted means the immutable order's full prefix",
    )
    parser.add_argument(
        "--global-microbatch-rows",
        type=int,
        help="global packed rows per forward/backward pass; frozen by the order",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="global microbatches per optimizer update; frozen by the order",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verify-packed-payloads",
        action="store_true",
        help=(
            "re-hash and semantically scan every packed payload while opening "
            "the loader; use the standalone validator once before torchrun to "
            "avoid repeating this work on every rank"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validation-order-manifest", type=Path)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="checkpoint complete transformer blocks; defaults on for the 1.3B model",
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="coding-model-from-scratch")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.checkpoint_every < 0 or args.eval_every < 0:
        parser.error("--checkpoint-every and --eval-every must be non-negative")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be positive")
    validation_requested = args.eval_every > 0 or args.eval_at_start
    if validation_requested and args.validation_order_manifest is None:
        parser.error(
            "--validation-order-manifest is required when evaluation is enabled"
        )
    if args.validation_order_manifest is not None and not validation_requested:
        parser.error(
            "a validation order requires --eval-every or --eval-at-start"
        )
    try:
        # The rank-zero verification/broadcast below hashes order.bin. Avoid
        # an otherwise redundant full read merely to parse its frozen schema.
        training_geometry = frozen_training_geometry(
            args.order_manifest,
            verify_checksum=False,
        )
    except (FileNotFoundError, IOError, ValueError) as exc:
        parser.error(str(exc))
    frozen_cli_fields = (
        ("steps", "optimizer_updates"),
        ("global_microbatch_rows", "global_microbatch_rows"),
        (
            "gradient_accumulation_steps",
            "gradient_accumulation_steps",
        ),
    )
    for argument, geometry_field in frozen_cli_fields:
        supplied = getattr(args, argument)
        expected = training_geometry[geometry_field]
        if supplied is None:
            setattr(args, argument, expected)
        elif supplied != expected:
            parser.error(
                f"--{argument.replace('_', '-')}={supplied} differs from immutable "
                f"order value {expected}"
            )
    try:
        validation_geometry = (
            None
            if args.validation_order_manifest is None
            else evaluation_order_geometry(
                args.validation_order_manifest,
                global_microbatch_rows=args.global_microbatch_rows,
                verify_checksum=False,
            )
        )
    except (FileNotFoundError, IOError, ValueError) as exc:
        parser.error(str(exc))
    if validation_geometry is not None:
        if validation_geometry["sequence_length"] != training_geometry["sequence_length"]:
            parser.error(
                "validation sequence_length differs from the training order"
            )
        if args.eval_batches > validation_geometry["available_global_microbatches"]:
            parser.error(
                "--eval-batches exceeds the validation order's complete "
                "global-microbatch prefix"
            )
    if args.checkpoint is None:
        if args.resume is None:
            parser.error("--checkpoint is required for a new pre-training run")
        args.checkpoint = args.resume
    checkpoint_previous = args.checkpoint.with_name(
        f"{args.checkpoint.stem}.previous{args.checkpoint.suffix}"
    )
    if args.resume is None:
        if args.checkpoint.exists() or checkpoint_previous.exists():
            parser.error(
                "refusing to overwrite an existing checkpoint generation; "
                "use --resume or choose a new --checkpoint path"
            )
    else:
        if not args.resume.is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume}")
        recovering_previous_generation = (
            args.resume.resolve() == checkpoint_previous.resolve()
        )
        if (
            args.checkpoint.resolve() != args.resume.resolve()
            and (args.checkpoint.exists() or checkpoint_previous.exists())
            and not recovering_previous_generation
        ):
            parser.error("redirected --checkpoint generation exists; choose a new path")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(args.device, local_rank=local_rank)
    try:
        validate_deterministic_cuda_environment(
            device,
            deterministic=args.deterministic,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rank, world_size = _init_distributed_if_requested(device)
    try:
        return _run_initialized_training(
            args=args,
            parser=parser,
            device=device,
            rank=rank,
            world_size=world_size,
            training_geometry=training_geometry,
            validation_geometry=validation_geometry,
        )
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
