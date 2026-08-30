#!/usr/bin/env python3
"""Overfit one immutable packed batch as an end-to-end correctness gate.

With no arguments this creates a tiny temporary packed corpus, loads one batch
through the real mmap/order/collation path, and trains a tiny CPU model. Pass a
real order manifest and ``--model-size 1.3b --device cuda`` for the GPU gate.
No network service is used unless W&B online mode is explicitly requested.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import CausalLM, ModelConfig
from pretrain.data import (
    DOMAIN_ORDER,
    PackedShardWriter,
    build_training_order,
    load_diagnostic_order_prefix,
)
from pretrain.train import (
    CompositeLogger,
    ConsoleLogger,
    FixedBatchStream,
    NullLogger,
    TrainConfig,
    Trainer,
    WandbLogger,
    evaluate_fixed_batch,
    seed_everything,
    tiny_model_config,
)
from pretrain.tokenizer_identity import verify_tokenizer_identity, vocabulary_sha256


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
    seed: int,
    checkpoint_path: Path,
    resume_path: Path | None,
    logger: Any,
    compile_model: bool,
    tokenizer_manifest_sha256: str,
    tokenizer_vocabulary_sha256: str,
) -> dict[str, Any]:
    """Run the fixed-batch experiment and assert meaningful memorization."""

    if not 0 < required_loss_ratio < 1:
        raise ValueError("required_loss_ratio must be in (0, 1)")
    fixed_stream = FixedBatchStream(
        load_one_packed_batch(order_manifest, batch_size=batch_size)
    )
    seed_everything(seed, deterministic=True)
    model: torch.nn.Module = CausalLM(
        model_config,
        device=device,
        dtype=parameter_dtype,
    )
    if compile_model:
        model = torch.compile(model)
    baseline_loss = evaluate_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,  # type: ignore[arg-type]
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
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    resume_start_loss = evaluate_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,  # type: ignore[arg-type]
    )
    trainer.train(fixed_stream)
    final_loss = evaluate_fixed_batch(
        model,
        fixed_stream.batch,
        device=device,
        precision=precision,  # type: ignore[arg-type]
    )
    trainer.save_checkpoint()
    ratio = final_loss / baseline_loss
    result: dict[str, Any] = {
        "status": "pending",
        "order_manifest": str(order_manifest.resolve()),
        "fixed_batch_identity": fixed_stream.identity,
        "input_shape": list(fixed_stream.batch["input_ids"].shape),
        "model_config": dataclasses.asdict(model_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": trainer.state.completed_steps,
        "initial_loss": baseline_loss,
        "resume_start_loss": resume_start_loss,
        "final_loss": final_loss,
        "loss_ratio": ratio,
        "required_loss_ratio": required_loss_ratio,
        "checkpoint": str(checkpoint_path.resolve()),
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256,
    }
    failures: list[str] = []
    if not all(math.isfinite(value) for value in (baseline_loss, resume_start_loss, final_loss)):
        failures.append("non_finite_loss")
    if not final_loss < baseline_loss:
        failures.append("final_loss_not_below_initial_loss")
    if ratio > required_loss_ratio:
        failures.append("loss_ratio_above_required_threshold")
    result["failures"] = failures
    result["status"] = "failed" if failures else "passed"
    if failures:
        raise OverfitGateError(result)
    return result


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


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
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--required-loss-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/overfit-single-chunk"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="coding-model-from-scratch")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.checkpoint_every < 0:
        parser.error("--steps/--batch-size must be positive and --checkpoint-every non-negative")
    device = _resolve_device(args.device)
    parameter_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.parameter_dtype]
    precision = args.precision or ("bfloat16" if device.type == "cuda" else "float32")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_context: tempfile.TemporaryDirectory[str] | None = None
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
        (
            tokenizer_manifest_sha256,
            tokenizer_vocabulary_sha256,
        ) = synthetic_tokenizer_identities(synthetic_vocab)
    else:
        if args.tokenizer is None:
            parser.error("--tokenizer is required with --order-manifest")
        order_manifest = args.order_manifest
    order_payload = json.loads(order_manifest.read_text(encoding="utf-8"))
    vocab_size = int(order_payload["vocab_size"])
    sequence_length = int(order_payload["sequence_length"])
    if args.order_manifest is not None:
        assert args.tokenizer is not None
        tokenizer_identity = verify_tokenizer_identity(
            args.tokenizer,
            expected_manifest_sha256=str(
                order_payload["tokenizer_manifest_sha256"]
            ),
            expected_vocab_size=vocab_size,
        )
        tokenizer_manifest_sha256 = tokenizer_identity.manifest_sha256
        tokenizer_vocabulary_sha256 = tokenizer_identity.vocabulary_sha256
    model_config = (
        tiny_model_config(vocab_size=vocab_size, max_seq_len=sequence_length)
        if args.model_size == "tiny"
        else ModelConfig(vocab_size=vocab_size, max_seq_len=sequence_length)
    )
    wandb_logger = WandbLogger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "experiment": "single-packed-chunk-overfit",
            "model": dataclasses.asdict(model_config),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        directory=args.output_dir,
    )
    logger = CompositeLogger(ConsoleLogger(), wandb_logger)
    result_path = args.output_dir / "result.json"
    _atomic_write_json(
        result_path,
        {
            "status": "running",
            "order_manifest": str(order_manifest.resolve()),
            "model_size": args.model_size,
            "device": str(device),
            "steps": args.steps,
            "seed": args.seed,
        },
    )
    exit_code = 0
    try:
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
                seed=args.seed,
                checkpoint_path=args.output_dir / "checkpoint.pt",
                resume_path=args.resume,
                logger=logger,
                compile_model=args.compile,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
                tokenizer_vocabulary_sha256=tokenizer_vocabulary_sha256,
            )
        except OverfitGateError as exc:
            result = exc.result
            exit_code = 1
        _atomic_write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        logger.finish()
        if synthetic_context is not None:
            synthetic_context.cleanup()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
