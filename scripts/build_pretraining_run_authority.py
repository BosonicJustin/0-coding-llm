#!/usr/bin/env python3
"""Build or revalidate an immutable six-GPU pretraining run authority.

All inputs are local and explicit.  This command never downloads, installs, or
queries a cloud API.  A successful ``build`` writes a new JSON file and an
adjacent ``.sha256`` sidecar; neither is ever overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.run_authority import (  # noqa: E402
    RunAuthorityError,
    collect_run_authority,
    publish_run_authority,
    snapshot_package_lock,
    validate_run_authority,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser(
        "snapshot-package-lock",
        help="write the exact local Python distribution set as an immutable lock",
    )
    snapshot.add_argument("--output", type=Path, required=True)

    build = commands.add_parser(
        "build",
        help="prove all inputs and publish a new immutable authority",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--package-lock", type=Path, required=True)
    build.add_argument(
        "--container-image-digest",
        required=True,
        help="operator-supplied immutable digest in sha256:<64 lowercase hex> form",
    )
    build.add_argument(
        "--hardware-contract",
        type=Path,
        required=True,
        help="qualified six-GPU receipt with its exact adjacent .sha256 sidecar",
    )
    build.add_argument("--geometry-receipt", type=Path, required=True)
    build.add_argument(
        "--corpus-qualification",
        type=Path,
        required=True,
        help="passing materialized-corpus qualification JSON with exact adjacent sidecar",
    )
    build.add_argument("--train-order-manifest", type=Path, required=True)
    build.add_argument("--validation-order-manifest", type=Path, required=True)
    build.add_argument("--train-certification", type=Path, required=True)
    build.add_argument("--validation-certification", type=Path, required=True)
    build.add_argument("--tokenizer-root", type=Path, required=True)
    build.add_argument("--training-recipe", type=Path, required=True)
    build.add_argument(
        "--launcher-argv-json",
        type=Path,
        required=True,
        help="JSON array containing the exact argv, with no shell interpolation",
    )
    build.add_argument(
        "--measured-input-tokens-per-second",
        required=True,
        help="aggregate six-GPU throughput measured by the accepted geometry run",
    )
    build.add_argument(
        "--hourly-cost-usd",
        required=True,
        help="total hourly price of the complete six-GPU pod",
    )
    build.add_argument("--total-cost-cap-usd", required=True)

    validate = commands.add_parser(
        "validate",
        help="reinspect every bound input and reject any mutation",
    )
    validate.add_argument("authority", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot-package-lock":
            result = snapshot_package_lock(args.output)
        elif args.command == "build":
            payload = collect_run_authority(
                project_root=args.project_root,
                package_lock=args.package_lock,
                container_image_digest=args.container_image_digest,
                hardware_contract=args.hardware_contract,
                geometry_receipt=args.geometry_receipt,
                corpus_qualification=args.corpus_qualification,
                train_order_manifest=args.train_order_manifest,
                validation_order_manifest=args.validation_order_manifest,
                train_certification=args.train_certification,
                validation_certification=args.validation_certification,
                tokenizer_root=args.tokenizer_root,
                training_recipe=args.training_recipe,
                launcher_argv_json=args.launcher_argv_json,
                measured_input_tokens_per_second=args.measured_input_tokens_per_second,
                hourly_cost_usd=args.hourly_cost_usd,
                total_cost_cap_usd=args.total_cost_cap_usd,
            )
            result = publish_run_authority(args.output, payload)
            result["authorization_sha256"] = payload["authorization_sha256"]
        else:
            result = validate_run_authority(args.authority)
    except (RunAuthorityError, OSError, ValueError) as exc:
        print(f"run-authority: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
