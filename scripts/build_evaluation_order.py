#!/usr/bin/env python3
"""Build a geometry-independent validation or test packed-row order.

Held-out orders deliberately omit frozen training microbatch/accumulation
geometry.  They can therefore be published immediately after packing and used
by one-GPU baselines and the six-GPU grid before the final train order exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import build_training_order, validate_packed_manifest  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--other-code-manifest", type=Path, required=True)
    parser.add_argument("--english-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-total-input-tokens", type=int, default=500_000_000)
    parser.add_argument("--input-token-tolerance", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sources = {
            "python": args.python_manifest,
            "other_code": args.other_code_manifest,
            "english": args.english_manifest,
        }
        for domain, path in sources.items():
            packed = validate_packed_manifest(path, verify_checksums=False)
            if packed.get("split") != args.split or packed.get("domain") != domain:
                raise ValueError(
                    f"{path} does not identify packed {args.split}/{domain}"
                )
        manifest = build_training_order(
            sources,
            args.output,
            seed=args.seed,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            expected_total_input_tokens=args.expected_total_input_tokens,
            input_token_tolerance=args.input_token_tolerance,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
