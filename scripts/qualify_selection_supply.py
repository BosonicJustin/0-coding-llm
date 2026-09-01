#!/usr/bin/env python3
"""Fail closed when a production selection cannot fill the packed-data caps.

This is deliberately a manifest-only preflight.  It authenticates the
published selection-v7 manifest and uses the same strict selected-total and
largest-remainder validators as materialization/order construction.  It does
not read raw archives, token caches, or decision bitmaps, so a shortfall is
reported before expensive packing I/O begins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import (  # noqa: E402
    DOMAIN_ORDER,
    _strict_weighted_row_counts,
)
from pretrain.materialize import (  # noqa: E402
    SPLITS,
    CorpusMaterializer,
    MaterializationError,
)
from pretrain.raw_token_cache_inventory import (  # noqa: E402
    RawTokenCacheInventoryError,
)
from scripts.publish_raw_token_cache_inventory import (  # noqa: E402
    _load_selection,
)


FORMAT = "selection-v7-packed-supply-qualification"
FORMAT_VERSION = 1
EXPECTED_WEIGHTS = {
    "python": 0.4,
    "other_code": 0.4,
    "english": 0.2,
}
DEFAULT_TARGETS = {
    "train": 52_580_000_000,
    "validation": 500_000_000,
    "test": 500_000_000,
}


def _positive_integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def geometry_independent_available_rows(
    *,
    selected_content_tokens: int,
    documents: int,
    sequence_length: int,
) -> int:
    """Return rows emitted by the no-padding packer before order selection.

    ``PackedShardWriter`` adds one EOS token per complete document.  A T-token
    causal row also needs one following lookahead token, and adjacent rows
    overlap at that token.  A stream of S tokens therefore produces
    ``floor((S - 1) / T)`` rows, not ``floor(S / T)``.
    """

    content = _positive_integer(
        selected_content_tokens, field="selected_content_tokens"
    )
    document_count = _positive_integer(documents, field="documents")
    length = _positive_integer(sequence_length, field="sequence_length")
    return (content + document_count - 1) // length


def qualify_selection_supply(
    selection_root: str | Path,
    *,
    sequence_length: int = 4_096,
    targets: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Authenticate selection v7 and return an exact packed-row supply report."""

    length = _positive_integer(sequence_length, field="sequence_length")
    requested_targets = dict(DEFAULT_TARGETS if targets is None else targets)
    if set(requested_targets) != set(SPLITS):
        raise ValueError(f"targets must contain exactly {SPLITS}")
    requested_targets = {
        split: _positive_integer(
            requested_targets[split], field=f"targets.{split}"
        )
        for split in SPLITS
    }

    root = Path(selection_root)
    selection, selection_manifest_sha256 = _load_selection(root)
    # This is the materializer's canonical strict parser.  In addition to the
    # selected-total schema, it reconciles reference quotas and document totals.
    selected_totals = CorpusMaterializer._validate_all_eligible_totals(selection)

    split_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for split in SPLITS:
        target_input_tokens = requested_targets[split]
        selectable_rows_cap = target_input_tokens // length
        if selectable_rows_cap < 1:
            raise ValueError(
                f"targets.{split} is smaller than one sequence_length row"
            )
        required_rows = _strict_weighted_row_counts(
            selectable_rows_cap, EXPECTED_WEIGHTS
        )
        domain_reports: list[dict[str, Any]] = []
        for domain in DOMAIN_ORDER:
            total = selected_totals[(split, domain)]
            selected_content_tokens = int(total["selected_content_tokens"])
            documents = int(total["selected_documents"])
            stream_tokens = selected_content_tokens + documents
            available_rows = geometry_independent_available_rows(
                selected_content_tokens=selected_content_tokens,
                documents=documents,
                sequence_length=length,
            )
            required = required_rows[domain]
            shortfall_rows = max(0, required - available_rows)
            surplus_rows = max(0, available_rows - required)
            domain_report = {
                "domain": domain,
                "selected_content_tokens": selected_content_tokens,
                "documents": documents,
                "stream_tokens_with_eos": stream_tokens,
                "available_rows": available_rows,
                "available_input_tokens": available_rows * length,
                "required_rows": required,
                "required_input_tokens": required * length,
                "shortfall_rows": shortfall_rows,
                "shortfall_input_tokens": shortfall_rows * length,
                "surplus_rows": surplus_rows,
                "status": "fail" if shortfall_rows else "pass",
            }
            domain_reports.append(domain_report)
            if shortfall_rows:
                failures.append(
                    {
                        "split": split,
                        "domain": domain,
                        "available_rows": available_rows,
                        "required_rows": required,
                        "shortfall_rows": shortfall_rows,
                        "shortfall_input_tokens": shortfall_rows * length,
                    }
                )
        split_reports.append(
            {
                "split": split,
                "target_input_tokens": target_input_tokens,
                "selectable_rows_cap": selectable_rows_cap,
                "selectable_input_tokens": selectable_rows_cap * length,
                "target_rounding_shortfall_tokens": (
                    target_input_tokens - selectable_rows_cap * length
                ),
                "required_rows_per_domain": required_rows,
                "domains": domain_reports,
                "status": (
                    "fail"
                    if any(row["status"] == "fail" for row in domain_reports)
                    else "pass"
                ),
            }
        )

    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "fail" if failures else "pass",
        "selection": {
            "root": str(root.resolve(strict=True)),
            "manifest_sha256": selection_manifest_sha256,
            "identity_format_version": selection["identity"]["format_version"],
        },
        "packing": {
            "sequence_length": length,
            "eos_boundary_tokens_per_document": 1,
            "lookahead_tokens_required": 1,
            "padding": False,
            "available_rows_formula": (
                "floor((selected_content_tokens + documents - 1) / "
                "sequence_length)"
            ),
        },
        "mixture": {
            "allocation": "largest_remainder_stable_domain_order",
            "domain_order": list(DOMAIN_ORDER),
            "weights": dict(EXPECTED_WEIGHTS),
        },
        "splits": split_reports,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument(
        "--expected-train-input-tokens", type=int, default=DEFAULT_TARGETS["train"]
    )
    parser.add_argument(
        "--expected-validation-input-tokens",
        type=int,
        default=DEFAULT_TARGETS["validation"],
    )
    parser.add_argument(
        "--expected-test-input-tokens", type=int, default=DEFAULT_TARGETS["test"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = qualify_selection_supply(
            args.selection_root,
            sequence_length=args.sequence_length,
            targets={
                "train": args.expected_train_input_tokens,
                "validation": args.expected_validation_input_tokens,
                "test": args.expected_test_input_tokens,
            },
        )
    except (
        OSError,
        ValueError,
        MaterializationError,
        RawTokenCacheInventoryError,
    ) as error:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "status": "error",
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
