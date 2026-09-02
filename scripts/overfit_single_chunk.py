#!/usr/bin/env python3
"""Overfit one immutable packed batch as an end-to-end correctness gate.

With no arguments this creates a tiny temporary packed corpus, loads one global
batch through the real mmap/order/collation path, and trains a tiny CPU model.
Pass a real order manifest and ``--model-size 1.3b --device cuda`` under
six-rank torchrun for the GPU gate. Separate uninterrupted, partial, and resume
processes can prove exact final trajectory equality. W&B can be disabled,
recorded offline, or streamed online when live preflight visibility is desired.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.model import CausalLM, ModelConfig
from pretrain.data import (
    DOMAIN_ORDER,
    PackedShardWriter,
    build_training_order,
    load_diagnostic_order_prefix,
)
from pretrain.train import (
    CheckpointLease,
    CompositeLogger,
    ConsoleLogger,
    FixedBatchStream,
    NullLogger,
    TrainConfig,
    Trainer,
    WandbLogger,
    _bind_wandb_run_id,
    _distributed_verify_order,
    _distributed_verify_tokenizer,
    _init_distributed_if_requested,
    _raise_if_distributed_stage_failed,
    _resolve_device as resolve_training_device,
    batch_fingerprint,
    evaluate_fixed_batch,
    preserve_host_rng_state,
    reconcile_checkpoint_temporaries,
    seed_everything,
    tiny_model_config,
    validate_deterministic_cuda_environment,
    wrap_distributed_model,
)
from pretrain.tokenizer_identity import vocabulary_sha256


class OverfitGateError(AssertionError):
    """A completed overfit run that failed one or more acceptance criteria."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = dict(result)
        failures = self.result.get("failures", [])
        super().__init__(f"Fixed-chunk overfit gate failed: {failures}")


def synthetic_tokenizer_identities(vocab_size: int) -> tuple[str, str]:
    """Return reproducible identities for the diagnostic numeric vocabulary."""

    if vocab_size < 1:
        raise ValueError("Synthetic vocabulary size must be positive")
    manifest_sha256 = hashlib.sha256(
        f"single-chunk-overfit-synthetic-tokenizer-v1:{vocab_size}".encode("ascii")
    ).hexdigest()
    vocabulary = {
        f"<synthetic-token-{token_id}>": token_id for token_id in range(vocab_size)
    }
    return manifest_sha256, vocabulary_sha256(vocabulary)


def build_synthetic_order(
    root: Path,
    *,
    sequence_length: int,
    vocab_size: int,
    global_microbatch_rows: int | None = None,
    gradient_accumulation_steps: int | None = None,
    split: str = "train",
) -> Path:
    """Build a deterministic 40/40/20 corpus through the production writer."""

    if sequence_length < 4:
        raise ValueError("Synthetic packed fixture requires sequence_length >= 4")
    tokenizer_manifest_sha256, _ = synthetic_tokenizer_identities(vocab_size)
    manifests: dict[str, Path] = {}
    for domain_id, (domain, rows) in enumerate(
        zip(DOMAIN_ORDER, (4, 4, 2), strict=True)
    ):
        output = root / "packed" / domain
        writer = PackedShardWriter(
            output,
            domain=domain,
            split=split,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            eos_token_id=0,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            rows_per_shard=2,
            construction_seed=17,
        )
        span = max(vocab_size - 1, 1)
        # Make two source documents per eventual row and choose their content
        # lengths so content+EOS stream length is exactly rows*T+1. This makes
        # every loaded row exercise in-row document boundaries while still
        # emitting the requested deterministic row count.
        document_count = rows * 2
        content_tokens = rows * sequence_length + 1 - document_count
        base_length, longer_documents = divmod(content_tokens, document_count)
        token_offset = 0
        for document_index in range(document_count):
            document_length = base_length + (document_index < longer_documents)
            tokens = [
                1
                + (
                    (
                        domain_id * 19
                        + (token_offset + index) * 7
                        + (token_offset + index) // 3
                    )
                    % span
                )
                for index in range(document_length)
            ]
            writer.add_document(tokens)
            token_offset += document_length
        writer.finish()
        manifests[domain] = output / "manifest.json"
    order_dir = root / "order"
    build_training_order(
        manifests,
        order_dir,
        seed=123,
        expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
        frozen_global_microbatch_rows=global_microbatch_rows,
        frozen_gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return order_dir / "manifest.json"


def load_one_packed_batch(
    order_manifest: Path,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Load and detach exactly one batch, then close all mmap handles."""

    batch = load_diagnostic_order_prefix(order_manifest, rows=batch_size)
    return {key: value.detach().cpu().clone() for key, value in batch.items()}


def packed_boundary_diagnostics(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, int | str]:
    """Prove the diagnostic chunk exercises the production boundary contract."""

    labels = batch["labels"]
    position_ids = batch["position_ids"]
    document_ids = batch["document_ids"]
    if (
        labels.ndim != 2
        or position_ids.shape != labels.shape
        or document_ids.shape != labels.shape
    ):
        raise ValueError("Packed diagnostic tensors must share shape [B, T]")
    transitions = document_ids[:, 1:] != document_ids[:, :-1]
    transition_targets = labels[:, :-1].eq(-100)
    if not torch.equal(transitions, transition_targets):
        raise ValueError(
            "Packed labels do not mask exactly the in-row document transitions"
        )
    reset_positions = position_ids[:, 1:][transitions]
    if reset_positions.numel() and not torch.equal(
        reset_positions, torch.zeros_like(reset_positions)
    ):
        raise ValueError("Packed position IDs do not reset at document boundaries")
    if not torch.equal(
        position_ids[:, 0], torch.zeros_like(position_ids[:, 0])
    ):
        raise ValueError("Every packed physical row must start at position zero")
    boundaries = int(transitions.sum())
    if boundaries < 1:
        raise ValueError(
            "The fixed chunk contains no in-row document boundary; choose a larger "
            "diagnostic batch so the block-diagonal mask is exercised"
        )
    supervised_tokens = int(labels.ne(-100).sum())
    if supervised_tokens < 1:
        raise ValueError("The fixed chunk has no supervised tokens")
    return {
        "contract": "packed-document-causal-boundaries-v1",
        "rows": int(labels.shape[0]),
        "sequence_length": int(labels.shape[1]),
        "in_row_document_boundaries": boundaries,
        "masked_boundary_targets": int(labels.eq(-100).sum()),
        "supervised_tokens": supervised_tokens,
        "distinct_document_segments": int(
            sum(torch.unique(row).numel() for row in document_ids)
        ),
    }


def shard_fixed_batch(
    global_batch: Mapping[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
) -> tuple[FixedBatchStream, str]:
    """Return the production-contiguous rank slice with one shared identity."""

    rows = int(global_batch["input_ids"].shape[0])
    if rows % world_size:
        raise ValueError(
            f"Fixed global batch rows ({rows}) must divide world_size ({world_size})"
        )
    local_rows = rows // world_size
    begin = rank * local_rows
    end = begin + local_rows
    global_digest = batch_fingerprint(global_batch)
    identity = (
        f"fixed-global-batch-sha256:{global_digest};"
        f"global-rows:{rows};world-size:{world_size}"
    )
    local_batch = {
        key: value[begin:end].detach().cpu().clone().contiguous()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == rows
        else value.detach().cpu().clone().contiguous()
        for key, value in global_batch.items()
        if isinstance(value, torch.Tensor)
    }
    local_batch["num_loss_tokens"] = local_batch["labels"].ne(-100).sum()
    return FixedBatchStream(local_batch, identity=identity), global_digest


def evaluate_distributed_fixed_batch(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    precision: str,
) -> float:
    """Return one globally token-weighted fixed-chunk loss on every rank."""

    local_tokens = int(batch["labels"].ne(-100).sum())
    if local_tokens < 1:
        raise ValueError("Every rank-local fixed chunk must have supervised tokens")
    local_loss = evaluate_fixed_batch(
        model,
        batch,
        device=device,
        precision=precision,  # type: ignore[arg-type]
    )
    totals = torch.tensor(
        [local_loss * local_tokens, float(local_tokens)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if not math.isfinite(float(totals[0])) or int(totals[1]) < 1:
        raise FloatingPointError("Fixed-chunk evaluation produced invalid loss totals")
    return float(totals[0] / totals[1])


def _hash_semantic_value(digest: Any, value: Any) -> None:
    """Hash nested checkpoint state without depending on torch.save bytes."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _hash_semantic_value(digest, key)
            _hash_semantic_value(digest, value[key])
        digest.update(b"mapping-end\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _hash_semantic_value(digest, item)
        digest.update(b"sequence-end\0")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0" + len(value).to_bytes(8, "big") + value)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(repr(value).encode("utf-8") + b"\0")
        return
    raise TypeError(f"Unsupported checkpoint comparison value: {type(value)!r}")


def checkpoint_trajectory_digests(path: Path) -> dict[str, str]:
    """Digest the fields that must match across uninterrupted and resumed runs."""

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    components = (
        "format",
        "format_version",
        "torch_version",
        "runtime_signature",
        "implementation_signature",
        "model",
        "model_config",
        "parameter_dtypes",
        "optimizer",
        "optimizer_class",
        "train_state",
        "rng_states",
        "train_trajectory_config",
        "training_geometry",
        "validation_configuration",
        "data_identity",
        "tokenizer_manifest_sha256",
        "tokenizer_vocabulary_sha256",
        "world_size",
    )
    missing = [component for component in components if component not in payload]
    if missing:
        raise ValueError(f"Checkpoint lacks exact-resume fields: {missing}")
    result: dict[str, str] = {}
    for component in components:
        digest = hashlib.sha256()
        _hash_semantic_value(digest, payload[component])
        result[component] = digest.hexdigest()
    return result


def compare_checkpoint_trajectories(
    actual: Path,
    reference: Path,
) -> dict[str, Any]:
    """Return exact semantic equality evidence for two final checkpoints."""

    actual_digests = checkpoint_trajectory_digests(actual)
    reference_digests = checkpoint_trajectory_digests(reference)
    mismatches = [
        component
        for component in actual_digests
        if actual_digests[component] != reference_digests[component]
    ]
    return {
        "requested": True,
        "reference_checkpoint": str(reference.resolve()),
        "actual_checkpoint": str(actual.resolve()),
        "components": actual_digests,
        "reference_components": reference_digests,
        "mismatches": mismatches,
        "exact_match": not mismatches,
    }


def checkpoint_wandb_run_id(path: Path | None) -> str | None:
    """Read the experiment-owned tracker identity before logger initialization."""

    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Resume checkpoint metadata is malformed")
    value = payload["metadata"].get("wandb_run_id")
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("Resume checkpoint W&B run ID is malformed")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_overfit(
    *,
    order_manifest: Path,
    model_config: ModelConfig,
    device: torch.device,
    parameter_dtype: torch.dtype,
    precision: str,
    steps: int,
    checkpoint_every: int,
    batch_size: int,
    learning_rate: float,
    required_loss_ratio: float,
    required_final_loss: float,
    seed: int,
    checkpoint_path: Path,
    resume_path: Path | None,
    logger: Any,
    compile_model: bool,
    tokenizer_manifest_sha256: str,
    tokenizer_vocabulary_sha256: str,
    wandb_run_id: str | None = None,
    until_step: int | None = None,
    exact_reference_checkpoint: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Run one production Trainer trajectory over an immutable global chunk."""

    if not 0 < required_loss_ratio < 1:
        raise ValueError("required_loss_ratio must be in (0, 1)")
    if not math.isfinite(required_final_loss) or required_final_loss <= 0:
        raise ValueError("required_final_loss must be finite and positive")
    if until_step is not None and not 1 <= until_step < steps:
        raise ValueError("until_step must be in [1, steps) for a partial run")
    if exact_reference_checkpoint is not None and until_step is not None:
        raise ValueError("A partial run cannot compare a final reference checkpoint")

    global_batch = load_one_packed_batch(order_manifest, batch_size=batch_size)
    boundaries = packed_boundary_diagnostics(global_batch)
    fixed_stream, global_batch_digest = shard_fixed_batch(
        global_batch,
        rank=rank,
        world_size=world_size,
    )
    if world_size > 1:
        observed: list[str] = [""] * world_size
        dist.all_gather_object(observed, global_batch_digest)
        if len(set(observed)) != 1:
            raise RuntimeError(
                "Ranks loaded different immutable global fixed-chunk bytes"
            )

    seed_everything(seed + rank, deterministic=True)
    model: torch.nn.Module = CausalLM(
        model_config,
        device=device,
        dtype=parameter_dtype,
    )
    if compile_model:
        model = torch.compile(model)
    if world_size > 1:
        model = wrap_distributed_model(model, device=device)
    baseline_loss = evaluate_distributed_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,
    )
    config = TrainConfig(
        max_steps=steps,
        global_microbatch_rows=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        min_learning_rate=learning_rate * 0.1,
        warmup_steps=min(3, steps),
        weight_decay=0.0,
        max_grad_norm=10.0,
        precision=precision,  # type: ignore[arg-type]
        seed=seed,
        deterministic=True,
        compile_model=compile_model,
        log_every=1,
        checkpoint_every=checkpoint_every,
        fused_adamw=False if device.type == "cpu" else None,
    )
    trainer = Trainer(
        model,
        config,
        device=device,
        data_identity=fixed_stream.identity,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        tokenizer_vocabulary_sha256=tokenizer_vocabulary_sha256,
        logger=logger,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={
            "qualification": "single-packed-chunk-overfit-v2",
            "global_batch_sha256": global_batch_digest,
            "packed_boundaries": boundaries,
        },
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    _bind_wandb_run_id(trainer, wandb_run_id)
    resume_start_loss = evaluate_distributed_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,
    )
    trainer.train(fixed_stream, until_step=until_step)
    final_loss = evaluate_distributed_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,
    )
    trainer.save_checkpoint()
    ratio = final_loss / baseline_loss
    result: dict[str, Any] = {
        "format": "single-packed-chunk-overfit-qualification",
        "format_version": 2,
        "status": "pending",
        "order_manifest": str(order_manifest.resolve()),
        "fixed_batch_identity": fixed_stream.identity,
        "global_batch_sha256": global_batch_digest,
        "global_input_shape": list(global_batch["input_ids"].shape),
        "local_input_shape": list(fixed_stream.batch["input_ids"].shape),
        "packed_boundaries": boundaries,
        "rank": rank,
        "world_size": world_size,
        "model_config": dataclasses.asdict(model_config),
        "parameter_count": sum(
            parameter.numel() for parameter in trainer.raw_model.parameters()
        ),
        "steps": trainer.state.completed_steps,
        "target_steps": steps,
        "partial_until_step": until_step,
        "initial_loss": baseline_loss,
        "resume_start_loss": resume_start_loss,
        "final_loss": final_loss,
        "final_perplexity": math.exp(min(final_loss, 80.0)),
        "loss_ratio": ratio,
        "required_loss_ratio": required_loss_ratio,
        "required_final_loss": required_final_loss,
        "checkpoint": str(checkpoint_path.resolve()),
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256,
    }
    if until_step is not None:
        result.update(
            status="checkpointed",
            failures=[],
            exact_resume={"requested": False},
        )
        return result

    exact_resume: dict[str, Any] = {"requested": False}
    comparison_error: BaseException | None = None
    if exact_reference_checkpoint is not None and rank == 0:
        try:
            exact_resume = compare_checkpoint_trajectories(
                checkpoint_path,
                exact_reference_checkpoint,
            )
        except BaseException as exc:
            comparison_error = exc
    _raise_if_distributed_stage_failed(
        comparison_error,
        stage="exact final-checkpoint comparison",
        rank=rank,
        world_size=world_size,
    )
    if world_size > 1:
        shared_comparison: list[dict[str, Any] | None] = [
            exact_resume if rank == 0 else None
        ]
        dist.broadcast_object_list(shared_comparison, src=0)
        assert shared_comparison[0] is not None
        exact_resume = shared_comparison[0]
    result["exact_resume"] = exact_resume

    failures: list[str] = []
    if not all(math.isfinite(value) for value in (baseline_loss, resume_start_loss, final_loss)):
        failures.append("non_finite_loss")
    if not final_loss < baseline_loss:
        failures.append("final_loss_not_below_initial_loss")
    if ratio > required_loss_ratio:
        failures.append("loss_ratio_above_required_threshold")
    if final_loss > required_final_loss:
        failures.append("final_loss_above_memorization_threshold")
    if exact_resume.get("requested") and not exact_resume.get("exact_match"):
        failures.append("resumed_trajectory_differs_from_uninterrupted_reference")
    result["failures"] = failures
    result["status"] = "failed" if failures else "passed"
    if failures:
        raise OverfitGateError(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-manifest", type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="required with --order-manifest; authenticates the real token-to-ID mapping",
    )
    parser.add_argument("--model-size", choices=("tiny", "1.3b"), default="tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parameter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help=(
            "publish a clean partial checkpoint at this absolute step and exit "
            "with status=checkpointed; rerun with --resume to finish"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="defaults to 3e-3 for tiny and 3e-4 for 1.3b",
    )
    parser.add_argument("--required-loss-ratio", type=float, default=0.1)
    parser.add_argument("--required-final-loss", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/overfit-single-chunk"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--exact-reference-checkpoint",
        type=Path,
        help=(
            "uninterrupted final checkpoint whose model, optimizer, counters, "
            "RNG states, trajectory, data, tokenizer, and world size must match"
        ),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="defaults on for 1.3b and off for tiny",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="coding-model-from-scratch")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.checkpoint_every < 0:
        parser.error("--steps/--batch-size must be positive and --checkpoint-every non-negative")
    if args.stop_after_step is not None and not 1 <= args.stop_after_step < args.steps:
        parser.error("--stop-after-step must be in [1, steps)")
    if args.stop_after_step is not None and args.exact_reference_checkpoint is not None:
        parser.error("a partial checkpoint cannot be compared with a final reference")
    if not 0 < args.required_loss_ratio < 1:
        parser.error("--required-loss-ratio must be in (0, 1)")
    if not math.isfinite(args.required_final_loss) or args.required_final_loss <= 0:
        parser.error("--required-final-loss must be finite and positive")
    if args.learning_rate is None:
        args.learning_rate = 3e-3 if args.model_size == "tiny" else 3e-4
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    activation_checkpointing = (
        args.model_size == "1.3b"
        if args.activation_checkpointing is None
        else args.activation_checkpointing
    )
    if args.model_size == "1.3b" and not activation_checkpointing:
        parser.error("the 1.3b qualification requires activation checkpointing")

    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_previous = args.output_dir / "checkpoint.previous.pt"
    if args.resume is None and (
        checkpoint_path.exists() or checkpoint_previous.exists()
    ):
        parser.error(
            "refusing to overwrite an existing checkpoint lineage; use --resume "
            "or choose a new --output-dir"
        )
    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume}")
        if args.resume.resolve() != checkpoint_path.resolve():
            parser.error("--resume must be this output directory's checkpoint.pt")
    if (
        args.exact_reference_checkpoint is not None
        and not args.exact_reference_checkpoint.is_file()
    ):
        parser.error(
            "--exact-reference-checkpoint must name an existing trusted checkpoint"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_training_device(args.device, local_rank=local_rank)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    try:
        validate_deterministic_cuda_environment(device, deterministic=True)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and device.type == "cpu":
        os.environ.setdefault(
            "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
        )
    rank, world_size = _init_distributed_if_requested(device)
    if world_size > 1 and device.type == "cuda" and device.index != local_rank:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise RuntimeError("Every torchrun rank must use its LOCAL_RANK CUDA device")
    parameter_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.parameter_dtype]
    precision = args.precision or ("bfloat16" if device.type == "cuda" else "float32")
    synthetic_context: tempfile.TemporaryDirectory[str] | None = None
    logger: Any = NullLogger()
    checkpoint_lease: CheckpointLease | None = None
    try:
        if args.order_manifest is None:
            if args.tokenizer is not None:
                parser.error("--tokenizer is only valid with --order-manifest")
            synthetic_context = tempfile.TemporaryDirectory()
            synthetic_root = Path(synthetic_context.name)
            synthetic_vocab = 64 if args.model_size == "tiny" else 49_152
            order_manifest = build_synthetic_order(
                synthetic_root,
                sequence_length=16,
                vocab_size=synthetic_vocab,
            )
            order_payload, _, _ = _distributed_verify_order(
                order_manifest,
                rank=rank,
                world_size=world_size,
            )
            (
                tokenizer_manifest_sha256,
                tokenizer_vocabulary_sha256,
            ) = synthetic_tokenizer_identities(synthetic_vocab)
        else:
            if args.tokenizer is None:
                parser.error("--tokenizer is required with --order-manifest")
            order_manifest = args.order_manifest
            order_payload, _, _ = _distributed_verify_order(
                order_manifest,
                rank=rank,
                world_size=world_size,
            )
            tokenizer_identity = _distributed_verify_tokenizer(
                args.tokenizer,
                expected_manifest_sha256=str(
                    order_payload["tokenizer_manifest_sha256"]
                ),
                expected_vocab_size=int(order_payload["vocab_size"]),
                rank=rank,
                world_size=world_size,
            )
            tokenizer_manifest_sha256 = str(
                tokenizer_identity["manifest_sha256"]
            )
            tokenizer_vocabulary_sha256 = str(
                tokenizer_identity["vocabulary_sha256"]
            )
        if order_payload.get("split") != "train":
            parser.error("--order-manifest must identify the train split")
        vocab_size = int(order_payload["vocab_size"])
        sequence_length = int(order_payload["sequence_length"])
        if args.batch_size % world_size:
            parser.error("--batch-size is global and must divide torchrun world size")
        base_model_config = (
            tiny_model_config(vocab_size=vocab_size, max_seq_len=sequence_length)
            if args.model_size == "tiny"
            else ModelConfig(vocab_size=vocab_size, max_seq_len=sequence_length)
        )
        model_config = dataclasses.replace(
            base_model_config,
            activation_checkpointing=activation_checkpointing,
        )

        lease_error: BaseException | None = None
        if rank == 0:
            checkpoint_lease = CheckpointLease(checkpoint_path)
            try:
                checkpoint_lease.acquire()
                reconcile_checkpoint_temporaries(checkpoint_path)
            except BaseException as exc:
                lease_error = exc
        _raise_if_distributed_stage_failed(
            lease_error,
            stage="overfit checkpoint lease",
            rank=rank,
            world_size=world_size,
        )

        wandb_error: BaseException | None = None
        wandb_run_id: str | None = None
        if rank == 0:
            try:
                resume_wandb_run_id = checkpoint_wandb_run_id(args.resume)
                with preserve_host_rng_state():
                    wandb_logger = WandbLogger(
                        mode=args.wandb_mode,
                        project=args.wandb_project,
                        name=args.wandb_run_name,
                        run_id=resume_wandb_run_id,
                        config={
                            "experiment": "single-packed-chunk-overfit-v2",
                            "model": dataclasses.asdict(model_config),
                            "steps": args.steps,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "seed": args.seed,
                            "world_size": world_size,
                        },
                        directory=args.output_dir,
                    )
                wandb_run_id = wandb_logger.run_id
                logger = CompositeLogger(ConsoleLogger(), wandb_logger)
            except BaseException as exc:
                wandb_error = exc
        _raise_if_distributed_stage_failed(
            wandb_error,
            stage="overfit W&B initialization",
            rank=rank,
            world_size=world_size,
        )
        if world_size > 1:
            shared_run_id = [wandb_run_id]
            dist.broadcast_object_list(shared_run_id, src=0)
            wandb_run_id = shared_run_id[0]

        result_path = args.output_dir / "result.json"
        if rank == 0:
            _atomic_write_json(
                result_path,
                {
                    "format": "single-packed-chunk-overfit-qualification",
                    "format_version": 2,
                    "status": "running",
                    "order_manifest": str(order_manifest.resolve()),
                    "model_size": args.model_size,
                    "device": str(device),
                    "world_size": world_size,
                    "steps": args.steps,
                    "stop_after_step": args.stop_after_step,
                    "seed": args.seed,
                },
            )

        exit_code = 0
        try:
            result = run_overfit(
                order_manifest=order_manifest,
                model_config=model_config,
                device=device,
                parameter_dtype=parameter_dtype,
                precision=precision,
                steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                required_loss_ratio=args.required_loss_ratio,
                required_final_loss=args.required_final_loss,
                seed=args.seed,
                checkpoint_path=checkpoint_path,
                resume_path=args.resume,
                logger=logger,
                compile_model=args.compile,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
                tokenizer_vocabulary_sha256=tokenizer_vocabulary_sha256,
                wandb_run_id=wandb_run_id,
                until_step=args.stop_after_step,
                exact_reference_checkpoint=args.exact_reference_checkpoint,
                rank=rank,
                world_size=world_size,
            )
        except OverfitGateError as exc:
            result = exc.result
            exit_code = 1
        if rank == 0:
            _atomic_write_json(result_path, result)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return exit_code
    finally:
        logger.finish()
        if checkpoint_lease is not None:
            checkpoint_lease.release()
        if synthetic_context is not None:
            synthetic_context.cleanup()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
