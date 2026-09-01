"""Frozen contracts shared by curation publishers and packed-data consumers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION = 7
ALL_ELIGIBLE_SELECTION_STRATEGY = "all_eligible_canonical_documents"
ALL_ELIGIBLE_BITMAP_FORMAT = "all-eligible-keep-bitmap"
ALL_ELIGIBLE_BITMAP_FORMAT_VERSION = 1
ALL_ELIGIBLE_BITMAP_MAGIC = b"AEKEEP01"
ALL_ELIGIBLE_BITMAP_BIT_ORDER = "manifest-index-lsb0"
ALL_ELIGIBLE_BITMAP_HEADER_LENGTH_BYTES = 4
ALL_ELIGIBLE_BITMAP_HEADER_KEYS = frozenset(
    (
        "format",
        "format_version",
        "archive",
        "bucket",
        "category",
        "records",
        "kept_documents",
        "bit_order",
        "payload_bytes",
    )
)
ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS = frozenset(
    (
        "archive",
        "path",
        "format",
        "format_version",
        "sha256",
        "bytes",
        "records",
        "kept_documents",
    )
)
ALL_ELIGIBLE_SELECTION_PROFILE: dict[str, Any] = {
    "contract_version": 1,
    "name": "all-eligible-canonical-v1",
    "production_tier": "baseline",
    "document_selection": "all_reason_free_canonical_documents",
    "document_action": "keep_complete_document",
    "split_authority": "frozen_leakage_safe_source_groups",
    "mixture_authority": "packed_training_order_v4",
    "quota_enforcement": "none",
    "known_limitations": [
        (
            "Curation does not enforce exact per-split/domain token quotas; "
            "the packed training order is the mixture and input-token authority."
        )
    ],
}


def validate_all_eligible_selection_profile(value: Any) -> dict[str, Any]:
    """Return the frozen profile or reject any semantic drift."""

    if not isinstance(value, Mapping) or dict(value) != ALL_ELIGIBLE_SELECTION_PROFILE:
        raise ValueError("Unsupported all-eligible selection profile")
    return copy.deepcopy(ALL_ELIGIBLE_SELECTION_PROFILE)


def all_eligible_bitmap_payload_bytes(records: int) -> int:
    """Return the exact bitmap payload size for ``records`` manifest rows."""

    if not isinstance(records, int) or isinstance(records, bool) or records < 1:
        raise ValueError("all-eligible bitmap records must be a positive integer")
    return (records + 7) // 8


def validate_all_eligible_bitmap_header(value: Any) -> dict[str, Any]:
    """Validate and return one frozen v1 bitmap header."""

    if not isinstance(value, Mapping) or set(value) != ALL_ELIGIBLE_BITMAP_HEADER_KEYS:
        raise ValueError("all-eligible bitmap header does not have the exact v1 schema")
    header = dict(value)
    records = header.get("records")
    kept = header.get("kept_documents")
    if (
        header.get("format") != ALL_ELIGIBLE_BITMAP_FORMAT
        or header.get("format_version") != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
        or header.get("bit_order") != ALL_ELIGIBLE_BITMAP_BIT_ORDER
        or not isinstance(header.get("archive"), str)
        or not header["archive"]
        or not isinstance(header.get("bucket"), str)
        or not header["bucket"]
        or not isinstance(header.get("category"), str)
        or not header["category"]
        or not isinstance(records, int)
        or isinstance(records, bool)
        or records < 1
        or not isinstance(kept, int)
        or isinstance(kept, bool)
        or kept < 0
        or kept > records
        or header.get("payload_bytes") != all_eligible_bitmap_payload_bytes(records)
    ):
        raise ValueError("invalid all-eligible bitmap v1 header")
    return header


def validate_all_eligible_bitmap_payload(payload: bytes, *, records: int) -> None:
    """Reject the wrong length or non-zero unused high bits."""

    expected = all_eligible_bitmap_payload_bytes(records)
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise ValueError("all-eligible bitmap payload length mismatch")
    remainder = records % 8
    if remainder and payload[-1] & ~((1 << remainder) - 1):
        raise ValueError("all-eligible bitmap has non-zero padding bits")
