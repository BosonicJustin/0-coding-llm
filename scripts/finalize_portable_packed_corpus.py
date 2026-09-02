#!/usr/bin/env python3
"""Finalize an authenticated packed-only corpus on the training pod."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.portable_finalize import (  # noqa: E402
    PortableFinalizationConfig,
    PortableFinalizationError,
    PortablePackedFinalizer,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("heldout", "final"),
        required=True,
        help=(
            "heldout publishes only validation/test; final additionally requires "
            "the measured geometry and publishes the capped train order"
        ),
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--cache-inventory-root", type=Path, required=True)
    parser.add_argument(
        "--restore-ready",
        type=Path,
        help=(
            "S3 restore .RESTORE_READY.json; required unless "
            "--verify-packed-payloads is used"
        ),
    )
    parser.add_argument(
        "--verify-packed-payloads",
        action="store_true",
        help=(
            "re-hash and semantically scan every token/start payload locally; "
            "use when no authenticated S3 restore receipt is available"
        ),
    )
    parser.add_argument(
        "--curation-policy",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "curation_policy_fast_exact_normalized.json",
    )
    parser.add_argument(
        "--quota-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_quotas_other_code_topup_v2.json",
    )
    parser.add_argument(
        "--benchmark-denylist",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
    )
    parser.add_argument("--order-seed", type=_nonnegative_int, default=1_234)
    parser.add_argument(
        "--maximum-train-input-tokens", type=_positive_int, default=52_580_000_000
    )
    parser.add_argument(
        "--maximum-validation-input-tokens", type=_positive_int, default=500_000_000
    )
    parser.add_argument(
        "--maximum-test-input-tokens", type=_positive_int, default=500_000_000
    )
    parser.add_argument(
        "--expected-optimizer-batch-rows",
        type=_positive_int,
        default=192,
        help="frozen effective optimizer batch (default: 192 packed rows)",
    )
    parser.add_argument("--world-size", type=_positive_int, default=6)
    parser.add_argument(
        "--global-microbatch-rows",
        type=_positive_int,
        help="required only in final mode; measured global rows per forward/backward",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        help="required only in final mode; measured accumulation count",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        finalizer = PortablePackedFinalizer(
            PortableFinalizationConfig(
                corpus_root=args.corpus_root,
                selection_root=args.selection_root,
                tokenizer_root=args.tokenizer_root,
                cache_inventory_root=args.cache_inventory_root,
                policy_path=args.curation_policy,
                quota_path=args.quota_config,
                benchmark_denylist_path=args.benchmark_denylist,
                restore_ready_path=args.restore_ready,
                order_seed=args.order_seed,
                maximum_train_input_tokens=args.maximum_train_input_tokens,
                maximum_validation_input_tokens=(
                    args.maximum_validation_input_tokens
                ),
                maximum_test_input_tokens=args.maximum_test_input_tokens,
                expected_optimizer_batch_rows=args.expected_optimizer_batch_rows,
                world_size=args.world_size,
                verify_packed_payloads=args.verify_packed_payloads,
            )
        )
        result = finalizer.run(
            mode=args.mode,
            global_microbatch_rows=args.global_microbatch_rows,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, PortableFinalizationError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
