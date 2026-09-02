#!/usr/bin/env python3
"""Publish the explicitly approved, separate-final-soak-only geometry waiver."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pretrain.geometry_qualification import (
    GeometryQualificationError,
    publish_json_new,
    verified_json_with_sidecar,
)
from pretrain.geometry_waiver import (
    GeometryWaiverError,
    collect_geometry_waiver,
    validate_geometry_waiver_receipt,
)


OUTPUT_NAME = "accepted-geometry-with-waiver.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-result", type=Path, required=True)
    parser.add_argument("--hardware-contract", type=Path, required=True)
    parser.add_argument("--train-order-manifest", type=Path, required=True)
    parser.add_argument("--validation-order-manifest", type=Path, required=True)
    parser.add_argument("--uninterrupted-result", type=Path, required=True)
    parser.add_argument("--resumed-result", type=Path, required=True)
    parser.add_argument("--overfit-order-manifest", type=Path, required=True)
    parser.add_argument("--phase-one-metadata", type=Path, required=True)
    parser.add_argument("--phase-one-summary", type=Path, required=True)
    parser.add_argument("--phase-one-event-log", type=Path, required=True)
    parser.add_argument("--phase-two-metadata", type=Path, required=True)
    parser.add_argument("--phase-two-summary", type=Path, required=True)
    parser.add_argument("--phase-two-event-log", type=Path, required=True)
    parser.add_argument("--decision-by", required=True)
    parser.add_argument(
        "--decision-utc",
        default=None,
        help="Timezone-aware ISO-8601 decision timestamp (default: current UTC)",
    )
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.name != OUTPUT_NAME:
        print(
            f"ERROR: --output basename must be exactly {OUTPUT_NAME!r}",
            file=sys.stderr,
        )
        return 2
    decision_utc = args.decision_utc or datetime.now(timezone.utc).isoformat()
    try:
        receipt = collect_geometry_waiver(
            grid_result=args.grid_result,
            hardware_contract=args.hardware_contract,
            train_order_manifest=args.train_order_manifest,
            validation_order_manifest=args.validation_order_manifest,
            uninterrupted_result=args.uninterrupted_result,
            resumed_result=args.resumed_result,
            overfit_order_manifest=args.overfit_order_manifest,
            phase_one_metadata=args.phase_one_metadata,
            phase_one_summary=args.phase_one_summary,
            phase_one_event_log=args.phase_one_event_log,
            phase_two_metadata=args.phase_two_metadata,
            phase_two_summary=args.phase_two_summary,
            phase_two_event_log=args.phase_two_event_log,
            waiver_cli=Path(__file__),
            decision_by=args.decision_by,
            decision_utc=decision_utc,
            rationale=args.rationale,
        )
        if args.output.exists() or args.output.is_symlink():
            existing, bound = verified_json_with_sidecar(
                args.output, label="existing geometry waiver receipt"
            )
            validate_geometry_waiver_receipt(existing)
            if existing != receipt:
                raise GeometryWaiverError(
                    "Existing immutable waiver receipt has another identity"
                )
        else:
            args.output.parent.resolve(strict=True)
            bound = publish_json_new(args.output, receipt)
            published, verified = verified_json_with_sidecar(
                args.output, label="published geometry waiver receipt"
            )
            if published != receipt or verified != bound:
                raise GeometryWaiverError("Published waiver receipt did not revalidate")
            validate_geometry_waiver_receipt(published)
    except (GeometryQualificationError, GeometryWaiverError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "output": bound["artifact"],
                "sidecar": bound["sidecar"],
                "final_soak_waived": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
