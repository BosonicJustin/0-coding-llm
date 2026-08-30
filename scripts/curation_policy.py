#!/usr/bin/env python3
"""Validate the source-aware final-corpus curation policy.

This module is the routing guard for the later final selector. In particular,
Stack v3 Train code must never enter a second local near-duplicate LSH pass,
while the two independently acquired English sources must enter one shared
near-duplicate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "curation_policy.json"
FAST_CANONICAL_POLICY = (
    PROJECT_ROOT / "configs" / "curation_policy_fast_exact_normalized.json"
)
CODE_BUCKETS = frozenset(("python", "other_code"))
ENGLISH_BUCKETS = frozenset(("fineweb_edu", "wikipedia"))
ALL_BUCKETS = CODE_BUCKETS | ENGLISH_BUCKETS
FAST_CANONICAL_PROFILE = {
    "contract_version": 1,
    "name": "fast-exact-normalized-canonical-v1",
    "production_tier": "baseline",
    "fuzzy_near_dedup": False,
    "canonicalization": "global_exact_then_global_normalized_hash",
    "benchmark_propagation": "global_exact_and_global_normalized_hash",
    "split_grouping": "stable_repository_or_english_source",
    "known_limitations": [
        "Semantic near-duplicate documents may remain and may cross source groups or data splits."
    ],
}


class CurationPolicyError(ValueError):
    """Raised when a curation policy or trusted source manifest is unsafe."""


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CurationPolicyError(f"Missing curation policy: {path}") from exc
    if not isinstance(policy, dict):
        raise CurationPolicyError("Curation policy must be a JSON object")
    validate_policy(policy)
    return policy


def validate_policy(policy: dict[str, Any]) -> None:
    policy_version = policy.get("policy_version")
    if policy_version not in (1, 2):
        raise CurationPolicyError("Unsupported curation policy version")

    source = policy.get("trusted_code_source")
    if not isinstance(source, dict):
        raise CurationPolicyError("trusted_code_source must be an object")
    required_source_fields = {
        "repo_id": str,
        "resolved_revision": str,
        "source_shard_count": int,
        "source_shards_sha256": str,
        "upstream_release": str,
        "upstream_deduplication": str,
    }
    for field, expected_type in required_source_fields.items():
        if not isinstance(source.get(field), expected_type):
            raise CurationPolicyError(f"Invalid trusted_code_source.{field}")
    if len(source["resolved_revision"]) != 40:
        raise CurationPolicyError("Trusted Stack revision must be a 40-character commit")
    if len(source["source_shards_sha256"]) != 64:
        raise CurationPolicyError("Trusted source-shard digest must be a SHA-256")

    common = policy.get("common")
    if not isinstance(common, dict):
        raise CurationPolicyError("common must be an object")
    for flag in ("raw_archives_immutable", "exact_hash_audit", "normalized_hash_audit"):
        if common.get(flag) is not True:
            raise CurationPolicyError(f"Curation safety flag {flag} must remain enabled")
    if common.get("residual_exact_normalized_action") != "remove_deterministically_before_split":
        raise CurationPolicyError("Residual exact/normalized duplicates must be removed before splits")
    if common.get("benchmark_contamination_action") != "reject_before_split":
        raise CurationPolicyError("Benchmark contamination must be rejected before splits")

    selection = policy.get("selection")
    if not isinstance(selection, dict) or selection.get("selection_version") != 1:
        raise CurationPolicyError("selection must be a version-1 object")
    seed = selection.get("seed")
    if not isinstance(seed, str) or not seed:
        raise CurationPolicyError("selection.seed must be a non-empty string")
    if selection.get("terminal_document_action") != "keep_token_prefix_to_exact_quota":
        raise CurationPolicyError("Final quotas require deterministic terminal-document prefixes")
    if selection.get("split_order") != ["test", "validation", "train"]:
        raise CurationPolicyError("selection.split_order must be test, validation, train")
    quality = selection.get("quality")
    if not isinstance(quality, dict):
        raise CurationPolicyError("selection.quality must be an object")
    hard_flags = quality.get("hard_reject_flags")
    by_bucket = quality.get("hard_reject_flags_by_bucket")
    weights = quality.get("soft_penalty_weights")
    if not isinstance(hard_flags, list) or not all(isinstance(flag, str) for flag in hard_flags):
        raise CurationPolicyError("selection quality hard flags must be strings")
    if not isinstance(by_bucket, dict) or set(by_bucket) != ALL_BUCKETS:
        raise CurationPolicyError("Per-bucket quality policy must cover every bucket")
    if not all(
        isinstance(flags, list) and all(isinstance(flag, str) for flag in flags)
        for flags in by_bucket.values()
    ):
        raise CurationPolicyError("Per-bucket hard flags must be string lists")
    if not isinstance(weights, dict) or not all(
        isinstance(flag, str) and isinstance(weight, int) and weight >= 0
        for flag, weight in weights.items()
    ):
        raise CurationPolicyError("Soft quality weights must be non-negative integers")
    if quality.get("missing_code_repo_id_action") != "reject":
        raise CurationPolicyError("Code without a repository identity must be rejected")
    if quality.get("missing_english_source_identity_action") != "reject":
        raise CurationPolicyError("English without a stable source identity must be rejected")
    english_near = selection.get("english_near_dedup")
    if not isinstance(english_near, dict):
        raise CurationPolicyError("selection.english_near_dedup must be an object")
    if policy_version == 1:
        if "curation_profile" in policy:
            raise CurationPolicyError(
                "Version-1 full policy cannot declare a fast curation profile"
            )
        expected_english_near = {
            "required_for_production": True,
            "mapping_format": "jsonl_or_jsonl_zst_doc_id_cluster_id",
            "missing_mapping_action": "fail",
        }
    else:
        if policy.get("curation_profile") != FAST_CANONICAL_PROFILE:
            raise CurationPolicyError(
                "Version-2 policy must use the exact fast canonical profile"
            )
        expected_english_near = {
            "required_for_production": False,
            "mapping_format": None,
            "missing_mapping_action": "disabled_by_fast_profile",
        }
    if english_near != expected_english_near:
        raise CurationPolicyError(
            "English near-dedup policy does not match the curation profile"
        )

    buckets = policy.get("buckets")
    if not isinstance(buckets, dict) or set(buckets) != ALL_BUCKETS:
        raise CurationPolicyError(f"Policy buckets must be exactly {sorted(ALL_BUCKETS)}")
    for bucket in CODE_BUCKETS:
        route = buckets[bucket]
        if route.get("local_near_dedup") is not False:
            raise CurationPolicyError(f"{bucket} must not enter local near-deduplication")
        if route.get("near_duplicate_authority") != "trusted_upstream_stack_v3_train":
            raise CurationPolicyError(f"{bucket} must trust the pinned Stack v3 Train deduplication")
        if route.get("split_grouping") != "repository_and_residual_duplicate_union":
            raise CurationPolicyError(f"{bucket} must use repository-safe split grouping")
    for bucket in ENGLISH_BUCKETS:
        route = buckets[bucket]
        expected_route = (
            {
                "local_near_dedup": True,
                "near_duplicate_authority": "local_cross_source_english",
                "split_grouping": "english_duplicate_cluster",
            }
            if policy_version == 1
            else {
                "local_near_dedup": False,
                "near_duplicate_authority": "disabled_by_fast_profile",
                "split_grouping": "stable_english_source_after_canonicalization",
            }
        )
        if route != expected_route:
            raise CurationPolicyError(
                f"{bucket} route does not match curation policy version {policy_version}"
            )


def curation_profile(policy: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact optional profile after validating the whole policy."""
    validate_policy(policy)
    if policy["policy_version"] == 1:
        return None
    return dict(FAST_CANONICAL_PROFILE)


def bucket_policy(policy: dict[str, Any], bucket: str) -> dict[str, Any]:
    if bucket not in ALL_BUCKETS:
        raise CurationPolicyError(f"Unknown source bucket: {bucket}")
    return dict(policy["buckets"][bucket])


def should_run_local_near_dedup(policy: dict[str, Any], bucket: str) -> bool:
    return bool(bucket_policy(policy, bucket)["local_near_dedup"])


def validate_trusted_stack_source(root: Path, policy: dict[str, Any]) -> dict[str, Any]:
    path = root / "manifests" / "STACK_V3_SOURCE.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CurationPolicyError(f"Missing pinned Stack source manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise CurationPolicyError("Stack source manifest must be a JSON object")

    trusted = policy["trusted_code_source"]
    for field in ("repo_id", "resolved_revision", "source_shard_count", "source_shards_sha256"):
        if manifest.get(field) != trusted[field]:
            raise CurationPolicyError(
                f"Untrusted Stack source {field}: {manifest.get(field)!r} != {trusted[field]!r}"
            )
    return manifest


def build_summary(root: Path, policy_path: Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    manifest = validate_trusted_stack_source(root, policy)
    summary = {
        "curation_policy_sha256": canonical_sha256(policy),
        "stack_source": {
            "repo_id": manifest["repo_id"],
            "resolved_revision": manifest["resolved_revision"],
            "source_shard_count": manifest["source_shard_count"],
            "source_shards_sha256": manifest["source_shards_sha256"],
            "upstream_release": policy["trusted_code_source"]["upstream_release"],
        },
        "local_near_dedup_buckets": sorted(
            bucket for bucket in ALL_BUCKETS if should_run_local_near_dedup(policy, bucket)
        ),
        "upstream_near_dedup_buckets": sorted(
            bucket
            for bucket in ALL_BUCKETS
            if bucket_policy(policy, bucket)["near_duplicate_authority"]
            == "trusted_upstream_stack_v3_train"
        ),
        "exact_normalized_audit_buckets": sorted(ALL_BUCKETS),
        "benchmark_rejection_buckets": sorted(ALL_BUCKETS),
    }
    profile = curation_profile(policy)
    if profile is not None:
        summary["curation_profile"] = profile
        summary["disabled_fuzzy_near_dedup_buckets"] = sorted(ENGLISH_BUCKETS)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = build_summary(args.root, args.policy)
    except (CurationPolicyError, json.JSONDecodeError) as exc:
        print(f"Curation policy validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
