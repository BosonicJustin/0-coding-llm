from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pretrain import geometry_evidence as evidence
from scripts import launch_pretraining as launcher


GIB = 1024**3


class GeometryEvidenceTest(unittest.TestCase):
    def _authority(self, root: Path) -> dict[str, object]:
        tokens_per_update = 6 * 2 * 4096
        token_delta = 100 * tokens_per_update
        receipt: dict[str, object] = {
            "format": "pretraining-accepted-geometry",
            "format_version": 1,
            "status": "pass",
            "accepted": {
                "global_microbatch_rows": 6,
                "gradient_accumulation_steps": 2,
                "workers": 2,
                "overfit_batch_rows": 6,
                "compile_model": False,
                "activation_checkpointing": True,
                "precision": "bfloat16",
                "parameter_dtype": "float32",
            },
            "measurements": {
                "aggregate_input_tokens_per_second": "1000",
                "peak_memory_allocated_bytes_per_gpu": 60 * GIB,
                "peak_memory_reserved_bytes_per_gpu": 64 * GIB,
                "minimum_free_memory_bytes_per_gpu": 12 * GIB,
                "checkpoint_seconds": "12.5",
                "data_wait_fraction": "0.02",
                "scaling_efficiency": "0.80",
                "soak_steps": 100,
                "throughput_measurement": {
                    "scope": evidence.THROUGHPUT_SCOPE,
                    "timer": evidence.THROUGHPUT_TIMER,
                    "counter": evidence.THROUGHPUT_COUNTER,
                    "start_consumed_input_tokens": 0,
                    "end_consumed_input_tokens": token_delta,
                    "elapsed_wall_time_ns": token_delta * 1_000_000,
                    "validation_events": 1,
                    "checkpoint_events": 1,
                    "wandb_log_events": 1,
                    "resume_verified": True,
                },
            },
        }
        receipt_path = root / "accepted-geometry.json"
        payload = json.dumps(receipt, sort_keys=True).encode("utf-8")
        receipt_path.write_bytes(payload)
        return {
            "geometry": {
                "artifact": {
                    "path": str(receipt_path),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "receipt": receipt,
            },
            "hardware": {"expected": {"gpu_memory_bytes": 80 * GIB}},
            "training": {
                "frozen_geometry": {"consumed_input_tokens": 52_580_000_000}
            },
            "data": {"train_order": {"sequence_length": 4096}},
        }

    @staticmethod
    def _rewrite_receipt(root: Path, authority: dict[str, object]) -> None:
        geometry = authority["geometry"]
        assert isinstance(geometry, dict)
        receipt = geometry["receipt"]
        artifact = geometry["artifact"]
        assert isinstance(receipt, dict) and isinstance(artifact, dict)
        payload = json.dumps(receipt, sort_keys=True).encode("utf-8")
        path = root / "accepted-geometry.json"
        path.write_bytes(payload)
        artifact.update(
            path=str(path),
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def test_accepts_complete_external_wall_clock_soak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            authority = self._authority(Path(temporary))
            result = evidence.validate_authority_geometry_soak(authority)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["soak_steps"], 100)
        self.assertEqual(result["tokens_per_update"], 49_152)
        self.assertEqual(result["input_token_delta"], 4_915_200)
        self.assertEqual(result["calculated_input_tokens_per_second"], "1000")
        self.assertEqual(result["required_free_memory_bytes_per_gpu"], 8 * GIB)

    def test_rejects_compute_only_or_incomplete_throughput_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._authority(root)
            mutations = (
                ("trainer scope", "scope", "trainer-step-timer"),
                ("non-monotonic timer", "timer", "time.time_ns"),
                ("derived counter", "counter", "batch_size-times-steps"),
                ("no validation", "validation_events", 0),
                ("no checkpoint", "checkpoint_events", 0),
                ("no W&B", "wandb_log_events", 0),
                ("no resume", "resume_verified", False),
            )
            for label, field, value in mutations:
                with self.subTest(label=label):
                    authority = copy.deepcopy(baseline)
                    external = authority["geometry"]["receipt"]["measurements"][
                        "throughput_measurement"
                    ]
                    external[field] = value
                    self._rewrite_receipt(root, authority)
                    with self.assertRaises(evidence.GeometryEvidenceError):
                        evidence.validate_authority_geometry_soak(authority)

    def test_rejects_short_or_misreconciled_soak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._authority(root)
            for field, value, pattern in (
                ("soak_steps", 99, "soak_steps"),
                (
                    "aggregate_input_tokens_per_second",
                    "1001",
                    "does not match external",
                ),
            ):
                with self.subTest(field=field):
                    authority = copy.deepcopy(baseline)
                    authority["geometry"]["receipt"]["measurements"][field] = value
                    self._rewrite_receipt(root, authority)
                    with self.assertRaisesRegex(evidence.GeometryEvidenceError, pattern):
                        evidence.validate_authority_geometry_soak(authority)

            authority = copy.deepcopy(baseline)
            external = authority["geometry"]["receipt"]["measurements"][
                "throughput_measurement"
            ]
            external["end_consumed_input_tokens"] += 49_152
            self._rewrite_receipt(root, authority)
            with self.assertRaisesRegex(
                evidence.GeometryEvidenceError, "does not match soak_steps"
            ):
                evidence.validate_authority_geometry_soak(authority)

    def test_enforces_scale_data_wait_and_memory_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._authority(root)
            mutations = (
                ("scaling_efficiency", "0.69", "Scaling efficiency"),
                ("data_wait_fraction", "0.051", "Data-wait fraction"),
                (
                    "minimum_free_memory_bytes_per_gpu",
                    8 * GIB - 1,
                    "below required",
                ),
            )
            for field, value, pattern in mutations:
                with self.subTest(field=field):
                    authority = copy.deepcopy(baseline)
                    authority["geometry"]["receipt"]["measurements"][field] = value
                    self._rewrite_receipt(root, authority)
                    with self.assertRaisesRegex(evidence.GeometryEvidenceError, pattern):
                        evidence.validate_authority_geometry_soak(authority)

    def test_rejects_receipt_mutation_after_authority_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = self._authority(root)
            (root / "accepted-geometry.json").write_bytes(b"{}")
            with self.assertRaisesRegex(
                evidence.GeometryEvidenceError, "changed after authority publication"
            ):
                evidence.validate_authority_geometry_soak(authority)

    def test_launcher_rebinds_strict_check_to_authenticated_authority_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = self._authority(root)
            authority_path = root / "run-authority.json"
            payload = json.dumps(authority, sort_keys=True).encode("utf-8")
            authority_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()

            result = launcher._load_strict_geometry_evidence(
                authority_path,
                expected_authority_sha256=digest,
            )
            self.assertEqual(result["status"], "pass")

            authority_path.write_bytes(payload + b"\n")
            with self.assertRaisesRegex(
                launcher.PreflightError, "changed after core validation"
            ):
                launcher._load_strict_geometry_evidence(
                    authority_path,
                    expected_authority_sha256=digest,
                )


if __name__ == "__main__":
    unittest.main()
