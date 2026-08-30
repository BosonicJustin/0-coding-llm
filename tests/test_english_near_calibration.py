from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import calibrate_english_near_dedup as calibration_module
from calibrate_english_near_dedup import (
    DEFAULT_CALIBRATION_CONFIG,
    calibrate,
    load_calibration_config,
    load_fixture_documents,
    main,
    sample_real_documents,
)
from build_english_near_clusters import DEFAULT_CONFIG, EnglishNearDedupBuilder
from curation_policy import DEFAULT_POLICY
from tests.test_english_near_dedup import NearFixture


def write_fixture(path: Path, documents: int = 4) -> None:
    rows = []
    for document in range(documents):
        text = " ".join(
            f"document{document}word{index:04d}" for index in range(320)
        )
        rows.append(
            json.dumps(
                {"doc_id": f"fixture-{document}", "bucket": "fixture", "text": text},
                sort_keys=True,
            )
            + "\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnglishNearCalibrationTest(unittest.TestCase):
    def test_fixture_calibration_is_deterministic_and_covers_threshold_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.jsonl"
            write_fixture(fixture)
            calibration_config = load_calibration_config(DEFAULT_CALIBRATION_CONFIG)
            documents, identity = load_fixture_documents(
                fixture,
                seed=calibration_config["seed"],
                maximum_documents=4,
            )
            production_before = sha256(DEFAULT_CONFIG)
            calibration_before = sha256(DEFAULT_CALIBRATION_CONFIG)
            first_path = root / "first.json"
            first = calibrate(
                documents=documents,
                input_identity=identity,
                production_config_path=DEFAULT_CONFIG,
                calibration_config_path=DEFAULT_CALIBRATION_CONFIG,
                output_path=first_path,
                minimum_candidate_recall=0.5,
                minimum_pairs=4,
                minimum_documents=2,
            )
            self.assertEqual(first["status"], "pass")
            self.assertEqual(first["summary"]["exact_refinement_decision_errors"], 0)
            self.assertGreaterEqual(
                first["summary"]["pairs_at_or_above_production_threshold"], 4
            )
            self.assertTrue(all(row["generated_pairs"] for row in first["bins"]))
            self.assertEqual(
                {row["operation"] for row in first["pairs"]},
                {
                    "append_donor_fragment",
                    "truncate_tail",
                    "replace_contiguous_with_donor",
                },
            )
            self.assertTrue(
                all(
                    not row["exact_refinement_accepted"]
                    for row in first["unrelated_controls"]
                )
            )
            self.assertTrue(first["production_configuration_unchanged"])
            self.assertEqual(
                first["identity"]["production_builder_sha256"],
                sha256(PROJECT_ROOT / "scripts" / "build_english_near_clusters.py"),
            )
            self.assertFalse(first["production_gate_eligible"])
            self.assertEqual(first["acceptance_profile"], "cli-override-non-production")
            self.assertGreater(
                first["summary"]["candidate_recall_one_sided_wilson_interval"]["lower"],
                0.5,
            )
            self.assertEqual(sha256(DEFAULT_CONFIG), production_before)
            self.assertEqual(sha256(DEFAULT_CALIBRATION_CONFIG), calibration_before)
            sidecar = first_path.with_name(first_path.name + ".sha256")
            self.assertEqual(sidecar.read_text().split()[0], sha256(first_path))

            second_path = root / "second.json"
            second = calibrate(
                documents=documents,
                input_identity=identity,
                production_config_path=DEFAULT_CONFIG,
                calibration_config_path=DEFAULT_CALIBRATION_CONFIG,
                output_path=second_path,
                minimum_candidate_recall=0.5,
                minimum_pairs=4,
                minimum_documents=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

            bins = {row["name"]: row["bounds"] for row in first["bins"]}
            for pair in first["pairs"]:
                bounds = bins[pair["bin"]]
                intersection = int(pair["intersection_shingles"])
                union = int(pair["union_shingles"])
                self.assertGreaterEqual(
                    intersection * bounds["denominator"],
                    union * bounds["minimum_numerator"],
                )
                self.assertLessEqual(
                    intersection * bounds["denominator"],
                    union * bounds["maximum_numerator"],
                )

    def test_failed_acceptance_is_published_without_config_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.jsonl"
            write_fixture(fixture, documents=2)
            config = load_calibration_config(DEFAULT_CALIBRATION_CONFIG)
            documents, identity = load_fixture_documents(
                fixture, seed=config["seed"], maximum_documents=2
            )
            before = sha256(DEFAULT_CONFIG)
            output = root / "failed.json"
            result = calibrate(
                documents=documents,
                input_identity=identity,
                production_config_path=DEFAULT_CONFIG,
                calibration_config_path=DEFAULT_CALIBRATION_CONFIG,
                output_path=output,
                minimum_candidate_recall=1.0,
                minimum_pairs=1_000_000,
                minimum_documents=1,
            )
            self.assertEqual(result["status"], "fail")
            self.assertTrue(result["acceptance_failures"])
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".sha256").is_file())
            self.assertEqual(before, sha256(DEFAULT_CONFIG))

    def test_builder_source_drift_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.jsonl"
            write_fixture(fixture, documents=2)
            config = load_calibration_config(DEFAULT_CALIBRATION_CONFIG)
            documents, identity = load_fixture_documents(
                fixture, seed=config["seed"], maximum_documents=2
            )
            output = root / "source-drift.json"
            actual_file_sha256 = calibration_module.file_sha256
            builder_calls = 0

            def drifting_sha(path: Path) -> str:
                nonlocal builder_calls
                if (
                    Path(path).resolve()
                    == calibration_module.PRODUCTION_BUILDER.resolve()
                ):
                    builder_calls += 1
                    if builder_calls > 1:
                        return "0" * 64
                return actual_file_sha256(path)

            with patch.object(
                calibration_module, "file_sha256", side_effect=drifting_sha
            ):
                with self.assertRaisesRegex(
                    calibration_module.CalibrationError,
                    "Production builder changed during calibration",
                ):
                    calibrate(
                        documents=documents,
                        input_identity=identity,
                        production_config_path=DEFAULT_CONFIG,
                        calibration_config_path=DEFAULT_CALIBRATION_CONFIG,
                        output_path=output,
                        minimum_candidate_recall=0.0,
                        minimum_pairs=0,
                        minimum_documents=1,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".sha256").exists())

    def test_real_sampler_is_bounded_and_pins_finalized_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            # Regression: sampling is the prerequisite for calibration, so the
            # identity probe must never attempt to load calibration evidence.
            with patch.object(
                EnglishNearDedupBuilder,
                "_load_calibration_evidence",
                side_effect=AssertionError("calibration bootstrap cycle"),
            ):
                documents, identity = sample_real_documents(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    production_config_path=DEFAULT_CONFIG,
                    policy_path=DEFAULT_POLICY,
                    denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                    quota_config_path=fixture.quota_path,
                    seed="real-sample-test",
                    maximum_archives_per_bucket=1,
                    maximum_documents_per_bucket=2,
                    minimum_source_words=96,
                    identity_output_hint=fixture.root / "never-created",
                )
            self.assertEqual(len(documents), 4)
            self.assertEqual(identity["kind"], "immutable_real_english_sample")
            self.assertEqual(len(identity["selected_reports"]), 2)
            self.assertTrue(identity["full_report_inventory_sha256"])
            self.assertTrue(identity["collection_completeness_sha256"])
            self.assertEqual(
                set(identity["collection_completeness"]["buckets"]),
                {"fineweb_edu", "wikipedia"},
            )
            self.assertFalse((fixture.root / "never-created").exists())

    def test_cli_acceptance_failure_returns_two_after_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.jsonl"
            output = root / "result.json"
            write_fixture(fixture, documents=2)
            arguments = [
                "calibrate_english_near_dedup.py",
                "--input-jsonl",
                str(fixture),
                "--output",
                str(output),
                "--minimum-documents",
                "1",
                "--minimum-pairs",
                "1000000",
                "--minimum-candidate-recall",
                "0.0",
            ]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            published = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(published["status"], "fail")
            self.assertFalse(published["production_gate_eligible"])
            self.assertTrue(output.with_name(output.name + ".sha256").is_file())


if __name__ == "__main__":
    unittest.main()
