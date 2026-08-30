#!/usr/bin/env python3
"""Materialize completed curation decisions into immutable packed corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.materialize import (
    CorpusMaterializer,
    MaterializationConfig,
    MaterializationError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        help="default: ROOT/staging/preprocess",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        help="completed curate_corpus.py output (default: ROOT/curated/selection-v1)",
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        help="default: ROOT/tokenizer/starcoder2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: ROOT/final/packed-v1",
    )
    parser.add_argument(
        "--curation-policy",
        type=Path,
        default=PROJECT_ROOT / "configs" / "curation_policy.json",
    )
    parser.add_argument(
        "--quota-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_quotas.json",
    )
    parser.add_argument(
        "--benchmark-denylist",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
    )
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--rows-per-shard", type=int, default=131_072)
    parser.add_argument("--construction-seed", type=int, default=1_234)
    parser.add_argument("--order-seed", type=int, default=1_234)
    parser.add_argument(
        "--global-microbatch-rows",
        type=int,
        help="required for final orders; choose after the real GPU memory/throughput smoke",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="required for final orders; choose after the real GPU memory/throughput smoke",
    )
    parser.add_argument(
        "--expected-train-input-tokens",
        type=int,
        default=52_580_000_000,
        help="authoritative packed-input target; use 0 only for a diagnostic untargeted build",
    )
    parser.add_argument("--train-input-token-tolerance", type=int)
    parser.add_argument(
        "--expected-validation-input-tokens", type=int, default=500_000_000
    )
    parser.add_argument(
        "--expected-test-input-tokens", type=int, default=500_000_000
    )
    parser.add_argument("--validation-input-token-tolerance", type=int)
    parser.add_argument("--test-input-token-tolerance", type=int)
    parser.add_argument(
        "--diagnostic-allow-observed-mixture",
        action="store_true",
        help=(
            "untargeted diagnostic only: also set train/validation/test caps to 0; "
            "never use for the final run"
        ),
    )
    parser.add_argument("--expected-vocab-size", type=int, default=49_152)
    parser.add_argument("--expected-eos-token-id", type=int, default=0)
    parser.add_argument("--tokenizer-batch-documents", type=int, default=256)
    parser.add_argument(
        "--tokenizer-batch-bytes", type=int, default=64 * 1024 * 1024
    )
    parser.add_argument(
        "--tokenizer-max-document-bytes",
        type=int,
        help=(
            "hard selected-document UTF-8 byte limit; defaults to the batch-byte "
            "limit and fails closed before reading/tokenizing an oversized member"
        ),
    )
    parser.add_argument(
        "--stop-after-packing",
        action="store_true",
        help=(
            "finish and verify packed data/indexes without orders; rerun the same "
            "output later with GPU-smoke-derived training geometry"
        ),
    )
    parser.add_argument(
        "--max-archives",
        type=int,
        help=(
            "bounded throughput/integrity smoke after this many archive "
            "reconciliations; reports rates and resumes with the same output"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    preprocess_root = args.preprocess_root or args.root / "staging" / "preprocess"
    selection_root = args.selection_root or args.root / "curated" / "selection-v1"
    tokenizer_root = args.tokenizer_root or args.root / "tokenizer" / "starcoder2"
    output = args.output or args.root / "final" / "packed-v1"
    expected_tokens = (
        None if args.expected_train_input_tokens == 0 else args.expected_train_input_tokens
    )
    expected_validation_tokens = (
        None
        if args.expected_validation_input_tokens == 0
        else args.expected_validation_input_tokens
    )
    expected_test_tokens = (
        None
        if args.expected_test_input_tokens == 0
        else args.expected_test_input_tokens
    )
    try:
        materializer = CorpusMaterializer(
            raw_root=args.root,
            preprocess_root=preprocess_root,
            selection_root=selection_root,
            tokenizer_root=tokenizer_root,
            policy_path=args.curation_policy,
            quota_path=args.quota_config,
            benchmark_denylist_path=args.benchmark_denylist,
            output_root=output,
            config=MaterializationConfig(
                sequence_length=args.sequence_length,
                rows_per_shard=args.rows_per_shard,
                construction_seed=args.construction_seed,
                order_seed=args.order_seed,
                frozen_global_microbatch_rows=args.global_microbatch_rows,
                frozen_gradient_accumulation_steps=(
                    args.gradient_accumulation_steps
                ),
                expected_train_input_tokens=expected_tokens,
                expected_validation_input_tokens=expected_validation_tokens,
                expected_test_input_tokens=expected_test_tokens,
                train_input_token_tolerance=args.train_input_token_tolerance,
                validation_input_token_tolerance=(
                    args.validation_input_token_tolerance
                ),
                test_input_token_tolerance=args.test_input_token_tolerance,
                enforce_input_weights=not args.diagnostic_allow_observed_mixture,
                expected_vocab_size=args.expected_vocab_size,
                expected_eos_token_id=args.expected_eos_token_id,
            ),
            tokenizer_batch_documents=args.tokenizer_batch_documents,
            tokenizer_batch_bytes=args.tokenizer_batch_bytes,
            tokenizer_max_document_bytes=args.tokenizer_max_document_bytes,
        )
        result = materializer.run(
            max_archives=args.max_archives,
            stop_after_packing=args.stop_after_packing,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (MaterializationError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
