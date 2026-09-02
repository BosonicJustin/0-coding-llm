#!/usr/bin/env python3
"""Produce authenticated six-GPU geometry-grid and final-soak evidence.

Preflight commands are read-only.  Run commands supervise real six-rank
``torchrun`` processes and publish only immutable, checksummed evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain import geometry_qualification as qualification  # noqa: E402


def _add_settings(parser: argparse.ArgumentParser) -> None:
    defaults = qualification.SoakSettings()
    parser.add_argument(
        "--measurement-warmup-steps",
        type=int,
        default=defaults.measurement_warmup_steps,
    )
    parser.add_argument(
        "--minimum-soak-steps", type=int, default=defaults.minimum_soak_steps
    )
    parser.add_argument(
        "--stop-after-soak-steps",
        type=int,
        default=defaults.stop_after_soak_steps,
    )
    parser.add_argument(
        "--order-buffer-steps", type=int, default=defaults.order_buffer_steps
    )
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--learning-rate", default=defaults.learning_rate)
    parser.add_argument(
        "--minimum-learning-rate", default=defaults.minimum_learning_rate
    )
    parser.add_argument("--weight-decay", default=defaults.weight_decay)
    parser.add_argument("--beta1", default=defaults.beta1)
    parser.add_argument("--beta2", default=defaults.beta2)
    parser.add_argument("--adam-epsilon", default=defaults.adam_epsilon)
    parser.add_argument("--max-grad-norm", default=defaults.max_grad_norm)
    parser.add_argument(
        "--trainer-warmup-steps", type=int, default=defaults.trainer_warmup_steps
    )
    parser.add_argument(
        "--phase-timeout-seconds", type=int, default=defaults.phase_timeout_seconds
    )
    parser.add_argument(
        "--graceful-shutdown-seconds",
        type=int,
        default=defaults.graceful_shutdown_seconds,
    )
    parser.add_argument(
        "--gpu-poll-interval-seconds",
        default=defaults.gpu_poll_interval_seconds,
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=defaults.wandb_mode,
        help=(
            "authoritative runs require offline or online; disabled is parsed "
            "only to fail closed with an explicit evidence error"
        ),
    )
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument(
        "--wandb-run-name-prefix", default=defaults.wandb_run_name_prefix
    )
    parser.add_argument(
        "--checkpoint-generation-bytes",
        type=int,
        default=defaults.checkpoint_generation_bytes,
        help=(
            "conservative bytes per mature 1.3B checkpoint; frozen into the plan "
            "and used to reserve every retained grid/final generation"
        ),
    )


def _settings(args: argparse.Namespace) -> qualification.SoakSettings:
    names = {field.name for field in dataclasses.fields(qualification.SoakSettings)}
    value = qualification.SoakSettings(
        **{name: getattr(args, name) for name in names}
    )
    value.validate()
    return value


def _add_common_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-order-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--hardware-contract", type=Path, required=True)
    parser.add_argument("--single-gpu-baselines", type=Path, required=True)
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    _add_settings(parser)


def _add_baseline_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-order-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--hardware-contract", type=Path, required=True)
    parser.add_argument("--python-packed-manifest", type=Path, required=True)
    parser.add_argument("--other-code-packed-manifest", type=Path, required=True)
    parser.add_argument("--english-packed-manifest", type=Path, required=True)
    parser.add_argument("--baseline-gpu-visible-index", type=int, default=0)
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    _add_settings(parser)


def _add_grid_inputs(parser: argparse.ArgumentParser) -> None:
    _add_common_plan_inputs(parser)
    parser.add_argument("--python-packed-manifest", type=Path, required=True)
    parser.add_argument("--other-code-packed-manifest", type=Path, required=True)
    parser.add_argument("--english-packed-manifest", type=Path, required=True)


def _add_final_inputs(parser: argparse.ArgumentParser) -> None:
    _add_common_plan_inputs(parser)
    parser.add_argument("--train-order-manifest", type=Path, required=True)
    parser.add_argument("--grid-result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)


def _grid_plan(args: argparse.Namespace) -> dict[str, Any]:
    return qualification.build_grid_plan(
        output_root=args.output_root,
        packed_manifests={
            "python": args.python_packed_manifest,
            "other_code": args.other_code_packed_manifest,
            "english": args.english_packed_manifest,
        },
        validation_order_manifest=args.validation_order_manifest,
        tokenizer_root=args.tokenizer,
        hardware_contract=args.hardware_contract,
        single_gpu_baselines=args.single_gpu_baselines,
        settings=_settings(args),
        cli_script=Path(__file__),
        python_executable=args.python_executable,
    )


def _baseline_plan(args: argparse.Namespace) -> dict[str, Any]:
    return qualification.build_baseline_plan(
        output_root=args.output_root,
        packed_manifests={
            "python": args.python_packed_manifest,
            "other_code": args.other_code_packed_manifest,
            "english": args.english_packed_manifest,
        },
        validation_order_manifest=args.validation_order_manifest,
        tokenizer_root=args.tokenizer,
        hardware_contract=args.hardware_contract,
        settings=_settings(args),
        cli_script=Path(__file__),
        python_executable=args.python_executable,
        baseline_gpu_visible_index=args.baseline_gpu_visible_index,
    )


def _final_plan(args: argparse.Namespace) -> dict[str, Any]:
    return qualification.build_final_plan(
        output_root=args.output_root,
        train_order_manifest=args.train_order_manifest,
        validation_order_manifest=args.validation_order_manifest,
        tokenizer_root=args.tokenizer,
        hardware_contract=args.hardware_contract,
        single_gpu_baselines=args.single_gpu_baselines,
        grid_result=args.grid_result,
        settings=_settings(args),
        cli_script=Path(__file__),
        python_executable=args.python_executable,
    )


def _preview_grid_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    settings = qualification.SoakSettings(**plan["settings"])
    python_path = Path(plan["runtime"]["python"]["invocation_path"])
    validation = Path(plan["inputs"]["validation_order"]["manifest"]["path"])
    tokenizer = Path(plan["inputs"]["common"]["tokenizer"]["root"])
    commands: list[dict[str, Any]] = []
    for candidate in qualification.CANDIDATES:
        candidate_root = Path(plan["output_root"]) / "candidates" / candidate.candidate_id
        order = Path(plan["output_root"]) / "orders" / candidate.candidate_id / "manifest.json"
        checkpoint = candidate_root / "checkpoints" / "last.pt"
        common = {
            "python_executable": python_path,
            "order_manifest": order,
            "validation_order_manifest": validation,
            "tokenizer_root": tokenizer,
            "checkpoint": checkpoint,
            "candidate": candidate,
            "settings": settings,
            "run_name": (
                f"{settings.wandb_run_name_prefix}-{candidate.candidate_id}-"
                f"{plan['plan_sha256'][:16]}"
            ),
            "run_group": plan["plan_sha256"][:16],
        }
        commands.append(
            {
                "candidate": candidate.as_dict(),
                "phase_one": qualification.render_torchrun_command(
                    **common, resume=False
                ),
                "phase_two": qualification.render_torchrun_command(
                    **common, resume=True
                ),
            }
        )
    return commands


def _preview_baseline_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    settings = qualification.SoakSettings(**plan["settings"])
    python_path = Path(plan["runtime"]["python"]["invocation_path"])
    validation = Path(plan["inputs"]["validation_order"]["manifest"]["path"])
    tokenizer = Path(plan["inputs"]["common"]["tokenizer"]["root"])
    commands: list[dict[str, Any]] = []
    for candidate in qualification.CANDIDATES:
        candidate_root = (
            Path(plan["output_root"]) / "candidates" / candidate.candidate_id
        )
        order = (
            Path(plan["output_root"])
            / "orders"
            / candidate.candidate_id
            / "manifest.json"
        )
        commands.append(
            {
                "candidate": candidate.as_dict(),
                "command": qualification.render_torchrun_command(
                    python_executable=python_path,
                    order_manifest=order,
                    validation_order_manifest=validation,
                    tokenizer_root=tokenizer,
                    checkpoint=candidate_root / "checkpoints" / "last.pt",
                    candidate=candidate,
                    settings=settings,
                    run_name=(
                        f"{settings.wandb_run_name_prefix}-baseline-"
                        f"{candidate.candidate_id}-"
                        f"{plan['plan_sha256'][:16]}"
                    ),
                    run_group=plan["plan_sha256"][:16],
                    resume=False,
                    world_size=qualification.BASELINE_WORLD_SIZE,
                    global_microbatch_rows=candidate.local_microbatch_rows,
                    wandb_tag="single-gpu-geometry-baseline",
                ),
            }
        )
    return commands


def _preview_final_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = next(
        item
        for item in qualification.CANDIDATES
        if item.as_dict() == plan["candidate"]
    )
    settings = qualification.SoakSettings(**plan["settings"])
    root = Path(plan["output_root"]) / "final-candidate"
    common = {
        "python_executable": Path(plan["runtime"]["python"]["invocation_path"]),
        "order_manifest": Path(plan["inputs"]["train_order"]["manifest"]["path"]),
        "validation_order_manifest": Path(
            plan["inputs"]["validation_order"]["manifest"]["path"]
        ),
        "tokenizer_root": Path(plan["inputs"]["common"]["tokenizer"]["root"]),
        "checkpoint": root / "checkpoints" / "last.pt",
        "candidate": candidate,
        "settings": settings,
        "run_name": (
            f"{settings.wandb_run_name_prefix}-{candidate.candidate_id}-"
            f"{plan['plan_sha256'][:16]}"
        ),
        "run_group": plan["plan_sha256"][:16],
    }
    return [
        {
            "candidate": candidate.as_dict(),
            "phase_one": qualification.render_torchrun_command(**common, resume=False),
            "phase_two": qualification.render_torchrun_command(**common, resume=True),
        }
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight-grid", "run-grid"):
        child = subparsers.add_parser(action)
        _add_grid_inputs(child)
    for action in ("preflight-baselines", "run-baselines"):
        child = subparsers.add_parser(action)
        _add_baseline_inputs(child)
    for action in ("preflight-final", "run-final"):
        child = subparsers.add_parser(action)
        _add_final_inputs(child)
    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    final_order = subparsers.add_parser("build-final-train-order")
    final_order.add_argument("--grid-result", type=Path, required=True)
    final_order.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            output = qualification.qualification_status(args.output_root)
        elif args.action == "build-final-train-order":
            manifest = qualification.build_final_train_order_from_grid(
                grid_result=args.grid_result,
                output_dir=args.output,
            )
            output = {"status": "pass", "manifest": manifest}
        elif args.action in {"preflight-baselines", "run-baselines"}:
            plan = _baseline_plan(args)
            if args.action == "preflight-baselines":
                output = {
                    "status": "preflight-pass-no-gpu-action",
                    "plan": plan,
                    "commands": _preview_baseline_commands(plan),
                    "output_storage": qualification.verify_output_storage(
                        args.output_root, plan=plan, allow_missing_root=True
                    ),
                }
            else:
                qualification.prepare_plan_root(args.output_root, plan)
                with qualification.QualificationLock(args.output_root):
                    payload, bound = qualification.run_single_gpu_baselines(
                        root=args.output_root,
                        plan=plan,
                        environment=os.environ,
                    )
                output = {"status": payload["status"], "result": bound}
        elif args.action in {"preflight-grid", "run-grid"}:
            plan = _grid_plan(args)
            if args.action == "preflight-grid":
                output = {
                    "status": "preflight-pass-no-gpu-action",
                    "plan": plan,
                    "commands": _preview_grid_commands(plan),
                    "output_storage": qualification.verify_output_storage(
                        args.output_root, plan=plan, allow_missing_root=True
                    ),
                }
            else:
                qualification.prepare_plan_root(args.output_root, plan)
                with qualification.QualificationLock(args.output_root):
                    payload, bound = qualification.run_grid(
                        root=args.output_root,
                        plan=plan,
                        environment=os.environ,
                    )
                output = {"status": payload["status"], "result": bound}
        else:
            plan = _final_plan(args)
            if args.action == "preflight-final":
                output = {
                    "status": "preflight-pass-no-gpu-action",
                    "plan": plan,
                    "commands": _preview_final_commands(plan),
                    "output_storage": qualification.verify_output_storage(
                        args.output_root, plan=plan, allow_missing_root=True
                    ),
                }
            else:
                receipt = args.receipt or (args.output_root / "accepted-geometry.json")
                qualification.prepare_plan_root(args.output_root, plan)
                if receipt.parent.resolve(strict=True) != args.output_root.resolve(
                    strict=True
                ):
                    raise qualification.GeometryQualificationError(
                        "Accepted receipt must be a direct child of --output-root"
                    )
                with qualification.QualificationLock(args.output_root):
                    payload, bound = qualification.run_final_soak(
                        root=args.output_root,
                        plan=plan,
                        receipt_path=receipt,
                        environment=os.environ,
                    )
                output = {"status": payload["status"], "receipt": bound}
    except (OSError, ValueError, qualification.GeometryQualificationError) as exc:
        safe_error = qualification.redact_text(str(exc), os.environ)
        print(f"error: {safe_error}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if args.action in {"run-baselines", "run-grid"} and output.get("status") != "pass":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
