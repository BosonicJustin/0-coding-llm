#!/usr/bin/env python3
"""Inspect or safely advance CPU post-curation preparation stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.postcuration_orchestrator import (  # noqa: E402
    FORMAT,
    FORMAT_VERSION,
    PostCurationConfig,
    PostCurationOrchestrationError,
    PostCurationOrchestrator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-inventory-root", type=Path)
    parser.add_argument("--materialization-output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--rows-per-shard", type=int, default=131_072)
    parser.add_argument("--construction-seed", type=int, default=1_234)
    parser.add_argument("--order-seed", type=int, default=1_234)
    parser.add_argument("--expected-train-input-tokens", type=int, default=52_580_000_000)
    parser.add_argument("--expected-validation-input-tokens", type=int, default=500_000_000)
    parser.add_argument("--expected-test-input-tokens", type=int, default=500_000_000)
    parser.add_argument("--expected-vocab-size", type=int, default=49_152)
    parser.add_argument("--expected-eos-token-id", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform authorized resumable CPU writes (default is read-only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = PostCurationConfig.for_generation(
            args.generation_root,
            selection_root=args.selection_root,
            tokenizer_root=args.tokenizer_root,
            cache_root=args.cache_root,
            cache_inventory_root=args.cache_inventory_root,
            materialization_output=args.materialization_output,
            sequence_length=args.sequence_length,
            rows_per_shard=args.rows_per_shard,
            construction_seed=args.construction_seed,
            order_seed=args.order_seed,
            expected_train_input_tokens=args.expected_train_input_tokens,
            expected_validation_input_tokens=args.expected_validation_input_tokens,
            expected_test_input_tokens=args.expected_test_input_tokens,
            expected_vocab_size=args.expected_vocab_size,
            expected_eos_token_id=args.expected_eos_token_id,
        )
        report = PostCurationOrchestrator(config, execute=args.execute).run()
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, PostCurationOrchestrationError)
            else PostCurationOrchestrationError(
                "CLI boundary failure: "
                f"{type(exc).__name__}: {exc}"
            )
        )
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "status": "error",
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
