#!/usr/bin/env python3
"""Build an immutable capped 40/40/20 packed-row training order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import build_training_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--other-code-manifest", type=Path, required=True)
    parser.add_argument("--english-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--global-microbatch-rows",
        type=int,
        required=True,
        help=(
            "immutable number of packed rows in one global microbatch; the manifest "
            "records the exact complete-update token prefix and training rejects a "
            "different value"
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        required=True,
        help=(
            "immutable number of global microbatches in one optimizer update; this "
            "is part of the token-budget authority and cannot change at runtime"
        ),
    )
    parser.add_argument(
        "--expected-total-input-tokens",
        type=int,
        default=52_580_000_000,
        help=(
            "target packed model-input tokens, including inserted EOS but excluding "
            "the duplicated T+1 lookahead storage token (default: 52.58B)"
        ),
    )
    parser.add_argument(
        "--input-token-tolerance",
        type=int,
        help=(
            "maximum target shortfall (default: fewer than one frozen optimizer "
            "update of input tokens)"
        ),
    )
    args = parser.parse_args()
    try:
        manifest = build_training_order(
            {
                "python": args.python_manifest,
                "other_code": args.other_code_manifest,
                "english": args.english_manifest,
            },
            args.output,
            seed=args.seed,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            expected_total_input_tokens=args.expected_total_input_tokens,
            input_token_tolerance=args.input_token_tolerance,
            frozen_global_microbatch_rows=args.global_microbatch_rows,
            frozen_gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (FileExistsError, FileNotFoundError, IOError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
