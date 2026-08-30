from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from curation_policy import (
    CurationPolicyError,
    DEFAULT_POLICY,
    FAST_CANONICAL_POLICY,
    FAST_CANONICAL_PROFILE,
    build_summary,
    curation_profile,
    load_policy,
    should_run_local_near_dedup,
    validate_policy,
    validate_trusted_stack_source,
)


class CurationPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(DEFAULT_POLICY)

    def test_near_dedup_routes_only_english(self) -> None:
        self.assertFalse(should_run_local_near_dedup(self.policy, "python"))
        self.assertFalse(should_run_local_near_dedup(self.policy, "other_code"))
        self.assertTrue(should_run_local_near_dedup(self.policy, "fineweb_edu"))
        self.assertTrue(should_run_local_near_dedup(self.policy, "wikipedia"))

    def test_policy_rejects_local_code_near_dedup(self) -> None:
        unsafe = copy.deepcopy(self.policy)
        unsafe["buckets"]["python"]["local_near_dedup"] = True
        with self.assertRaisesRegex(CurationPolicyError, "must not enter"):
            validate_policy(unsafe)

    def test_policy_rejects_unsafe_missing_english_identity_fallback(self) -> None:
        unsafe = copy.deepcopy(self.policy)
        unsafe["selection"]["quality"]["missing_english_source_identity_action"] = "singleton"
        with self.assertRaisesRegex(CurationPolicyError, "stable source identity"):
            validate_policy(unsafe)

    def test_fast_canonical_profile_is_exact_and_fail_closed(self) -> None:
        fast = load_policy(FAST_CANONICAL_POLICY)
        self.assertEqual(curation_profile(fast), FAST_CANONICAL_PROFILE)
        self.assertFalse(should_run_local_near_dedup(fast, "fineweb_edu"))
        self.assertFalse(should_run_local_near_dedup(fast, "wikipedia"))

        for mutation in ("name", "canonicalization", "limitation", "route"):
            with self.subTest(mutation=mutation):
                unsafe = copy.deepcopy(fast)
                if mutation == "name":
                    unsafe["curation_profile"]["name"] = "other"
                elif mutation == "canonicalization":
                    unsafe["curation_profile"]["canonicalization"] = "exact_only"
                elif mutation == "limitation":
                    unsafe["curation_profile"]["known_limitations"] = []
                else:
                    unsafe["buckets"]["wikipedia"]["local_near_dedup"] = True
                with self.assertRaises(CurationPolicyError):
                    validate_policy(unsafe)

    def test_source_manifest_must_match_the_pinned_cleaned_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifests" / "STACK_V3_SOURCE.json"
            manifest_path.parent.mkdir(parents=True)
            trusted = self.policy["trusted_code_source"]
            manifest = {
                "manifest_version": 3,
                "repo_id": trusted["repo_id"],
                "resolved_revision": trusted["resolved_revision"],
                "source_shard_count": trusted["source_shard_count"],
                "source_shards_sha256": trusted["source_shards_sha256"],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                validate_trusted_stack_source(root, self.policy)["resolved_revision"],
                trusted["resolved_revision"],
            )
            summary = build_summary(root, DEFAULT_POLICY)
            self.assertEqual(summary["local_near_dedup_buckets"], ["fineweb_edu", "wikipedia"])
            self.assertEqual(summary["upstream_near_dedup_buckets"], ["other_code", "python"])
            fast_summary = build_summary(root, FAST_CANONICAL_POLICY)
            self.assertEqual(fast_summary["local_near_dedup_buckets"], [])
            self.assertEqual(
                fast_summary["disabled_fuzzy_near_dedup_buckets"],
                ["fineweb_edu", "wikipedia"],
            )
            self.assertEqual(
                fast_summary["curation_profile"], FAST_CANONICAL_PROFILE
            )

            manifest["resolved_revision"] = "0" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CurationPolicyError, "resolved_revision"):
                validate_trusted_stack_source(root, self.policy)


if __name__ == "__main__":
    unittest.main()
