#!/usr/bin/env python3
"""Qualify CUDA FlexAttention correctness before an expensive training run.

This is a bounded model-backend gate, not a throughput or 1.3B memory test. It
compares FlexAttention with the dense SDPA reference, proves packed-document
and physical-row isolation in both outputs and gradients, and executes the
actual chunked-loss backward plus one AdamW update under BF16 autocast.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.model import CausalLM, ModelConfig  # noqa: E402


class QualificationError(RuntimeError):
    """The CUDA backend failed a required qualification invariant."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_source_path() -> Path:
    source = inspect.getsourcefile(CausalLM)
    if source is None:
        raise QualificationError("Cannot resolve the imported CausalLM source file")
    path = Path(source).resolve(strict=True)
    if not path.is_file():
        raise QualificationError(f"CausalLM source is not a regular file: {path}")
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _labels_for_documents(
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2 or document_ids.shape != input_ids.shape:
        raise ValueError("input_ids and document_ids must have identical [B, T] shape")
    labels = torch.full_like(input_ids, -100)
    same_next_document = document_ids[:, :-1] == document_ids[:, 1:]
    labels[:, :-1] = torch.where(
        same_next_document,
        input_ids[:, 1:],
        torch.full_like(input_ids[:, 1:], -100),
    )
    return labels


def _fixture(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size < 2:
        raise ValueError("qualification fixture requires at least two physical rows")
    if sequence_length < 12 or sequence_length % 4:
        raise ValueError("sequence_length must be divisible by four and at least 12")
    required_vocab = 3 * batch_size * sequence_length + 1
    if vocab_size < required_vocab:
        raise ValueError(f"vocab_size must be at least {required_vocab}")

    input_ids = torch.arange(
        1,
        batch_size * sequence_length + 1,
        dtype=torch.int64,
        device=device,
    ).reshape(batch_size, sequence_length)
    quarter = sequence_length // 4
    one_row_documents = torch.cat(
        (
            torch.zeros(quarter, dtype=torch.int64, device=device),
            torch.ones(2 * quarter, dtype=torch.int64, device=device),
            torch.full((quarter,), 2, dtype=torch.int64, device=device),
        )
    )
    document_ids = one_row_documents.repeat(batch_size, 1)
    one_row_positions = torch.cat(
        (
            torch.arange(quarter, dtype=torch.int64, device=device),
            torch.arange(2 * quarter, dtype=torch.int64, device=device),
            torch.arange(quarter, dtype=torch.int64, device=device),
        )
    )
    position_ids = one_row_positions.repeat(batch_size, 1)
    labels = _labels_for_documents(input_ids, document_ids)
    return input_ids, position_ids, document_ids, labels


def _driver_version() -> int | str | None:
    public = getattr(torch.cuda, "driver_version", None)
    if callable(public):
        try:
            return public()
        except RuntimeError:
            pass
    private = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(private):
        try:
            return private()
        except RuntimeError:
            pass
    return None


def _runtime_identity(device_index: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    nccl_version: Any = None
    try:
        nccl_version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        pass
    if isinstance(nccl_version, tuple):
        nccl_version = ".".join(str(part) for part in nccl_version)
    return {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _driver_version(),
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
        "device_uuid": str(getattr(properties, "uuid", "")) or None,
    }


def _maximum_absolute_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def qualify(
    *,
    device_index: int,
    sequence_length: int,
    seed: int,
    parity_rtol: float,
    parity_atol: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise QualificationError("CUDA is unavailable")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise QualificationError(
            f"device_index {device_index} is outside 0..{torch.cuda.device_count() - 1}"
        )
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise QualificationError("Selected CUDA runtime lacks native BF16 support")

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    vocab_size = max(1024, 3 * 2 * sequence_length + 1)
    base_config = ModelConfig(
        vocab_size=vocab_size,
        dim=64,
        hidden_dim=176,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=sequence_length,
        attention_backend="flex",
        loss_chunk_size=min(32, sequence_length),
    )
    flex_model = CausalLM(base_config, device=device, dtype=torch.float32).eval()
    reference_config = dataclasses.replace(base_config, attention_backend="sdpa")
    reference_model = CausalLM(
        reference_config,
        device=device,
        dtype=torch.float32,
    ).eval()
    reference_model.load_state_dict(flex_model.state_dict())

    input_ids, position_ids, document_ids, labels = _fixture(
        batch_size=2,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        device=device,
    )
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), autocast():
        flex_output = flex_model(
            input_ids,
            position_ids,
            document_ids,
            labels,
            return_logits=True,
        )
        reference_output = reference_model(
            input_ids,
            position_ids,
            document_ids,
            labels,
            return_logits=True,
        )
    assert flex_output.logits is not None and reference_output.logits is not None
    assert flex_output.loss is not None and reference_output.loss is not None
    torch.testing.assert_close(
        flex_output.logits,
        reference_output.logits,
        rtol=parity_rtol,
        atol=parity_atol,
    )
    torch.testing.assert_close(
        flex_output.loss,
        reference_output.loss,
        rtol=parity_rtol,
        atol=parity_atol,
    )

    changed = input_ids.clone()
    first_document = document_ids[0] == 0
    changed[0, first_document] += 2 * sequence_length
    with torch.no_grad(), autocast():
        changed_output = flex_model(
            changed,
            position_ids,
            document_ids,
            return_logits=True,
        )
    assert changed_output.logits is not None
    later_documents = document_ids[0] != 0
    if not torch.equal(
        flex_output.logits[0, later_documents],
        changed_output.logits[0, later_documents],
    ):
        raise QualificationError("FlexAttention leaked across packed documents")
    if not torch.equal(flex_output.logits[1], changed_output.logits[1]):
        raise QualificationError("FlexAttention leaked across physical batch rows")

    flex_model.zero_grad(set_to_none=True)
    with autocast():
        gradient_output = flex_model(
            input_ids,
            position_ids,
            document_ids,
            return_logits=True,
        )
        assert gradient_output.logits is not None
        probe = gradient_output.logits[0, later_documents, 7].float().sum()
    probe.backward()
    embedding_gradient = flex_model.tok_embeddings.weight.grad
    if embedding_gradient is None:
        raise QualificationError("Embedding gradient was not produced")
    first_document_tokens = input_ids[0, first_document]
    cross_document_gradient = embedding_gradient[first_document_tokens]
    if int(torch.count_nonzero(cross_document_gradient).item()) != 0:
        raise QualificationError("Gradient leaked across packed documents")

    flex_model.train()
    flex_model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(
        flex_model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    with autocast():
        training_output = flex_model(
            input_ids,
            position_ids,
            document_ids,
            labels,
            return_logits=False,
        )
        assert training_output.loss is not None
    if not bool(torch.isfinite(training_output.loss).item()):
        raise QualificationError("Chunked training loss is non-finite")
    training_output.loss.backward()
    for name, parameter in flex_model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise QualificationError(f"Non-finite gradient in {name}")
    optimizer.step()
    torch.cuda.synchronize(device)

    return {
        "status": "passed",
        "runtime": _runtime_identity(device_index),
        "model_config": dataclasses.asdict(base_config),
        "seed": seed,
        "fixture": {
            "batch_size": 2,
            "sequence_length": sequence_length,
            "supervised_tokens": int(labels.ne(-100).sum().item()),
            "documents_per_row": 3,
        },
        "checks": {
            "flex_vs_sdpa_logits": "passed",
            "flex_vs_sdpa_loss": "passed",
            "cross_document_output_isolation": "bit_exact",
            "cross_row_output_isolation": "bit_exact",
            "cross_document_gradient_isolation": "exact_zero",
            "bf16_chunked_loss_backward": "passed",
            "adamw_first_update": "passed",
        },
        "measurements": {
            "flex_sdpa_logits_max_abs_difference": _maximum_absolute_difference(
                flex_output.logits, reference_output.logits
            ),
            "flex_loss": float(flex_output.loss.item()),
            "sdpa_loss": float(reference_output.loss.item()),
            "training_loss": float(training_output.loss.item()),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--parity-rtol", type=float, default=0.02)
    parser.add_argument("--parity-atol", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() and not args.overwrite:
        _parser().error(f"refusing to overwrite existing evidence: {output}")
    source_identity = {
        "model_py_sha256": _sha256(_model_source_path()),
        "qualification_script_sha256": _sha256(Path(__file__).resolve()),
    }
    running = {
        "format": "cuda-model-qualification",
        "format_version": 1,
        "status": "running",
        "created_utc": _utc_now(),
        "source_identity": source_identity,
        "requested": {
            "device_index": args.device_index,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "parity_rtol": args.parity_rtol,
            "parity_atol": args.parity_atol,
        },
    }
    _atomic_write_json(output, running)
    try:
        result = qualify(
            device_index=args.device_index,
            sequence_length=args.sequence_length,
            seed=args.seed,
            parity_rtol=args.parity_rtol,
            parity_atol=args.parity_atol,
        )
    except BaseException as exc:
        failed = {
            **running,
            "status": "failed",
            "completed_utc": _utc_now(),
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        _atomic_write_json(output, failed)
        print(json.dumps(failed, indent=2, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    passed = {
        **running,
        **result,
        "completed_utc": _utc_now(),
        "source_identity": source_identity,
    }
    _atomic_write_json(output, passed)
    print(json.dumps(passed, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
