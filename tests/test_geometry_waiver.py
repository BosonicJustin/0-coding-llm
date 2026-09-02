from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pretrain import geometry_evidence
from pretrain import geometry_qualification as qualification
from pretrain import geometry_waiver
from pretrain import run_authority
from scripts import accept_geometry_waiver


GIB = 1024**3


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


class GeometryWaiverTest(unittest.TestCase):
    def test_hardware_compatibility_allows_only_git_object_identity_change(self) -> None:
        identity = {
            "scope": "geometry-only-provisional",
            "identity": {
                "gpu_model": "H100",
                "package_lock": {
                    "lock": {"path": "/frozen/lock.json", "bytes": 10, "sha256": "a" * 64},
                    "python": {
                        "executable": "/frozen/python",
                        "executable_bytes": 10,
                        "executable_sha256": "9" * 64,
                    },
                    "packages_sha256": "a" * 64,
                },
                "source": {
                    "qualification_script_sha256": "b" * 64,
                    "requirements_train_sha256": "c" * 64,
                    "requirements_wandb_sha256": "d" * 64,
                    "git_commit": "1" * 40,
                    "git_tree": "2" * 40,
                    "git_head_archive_sha256": "3" * 64,
                    "git_clean": True,
                },
            },
            "identity_sha256": "4" * 64,
        }
        final = copy.deepcopy(identity)
        final["scope"] = "final-launch-authorizing"
        final["identity_sha256"] = "5" * 64
        final["identity"]["source"]["git_commit"] = "6" * 40
        final["identity"]["source"]["git_tree"] = "7" * 40
        final["identity"]["source"]["git_head_archive_sha256"] = "8" * 64
        final["identity"]["package_lock"]["lock"]["path"] = "/current/lock.json"
        final["identity"]["package_lock"]["python"]["executable"] = "/current/python"
        result = geometry_waiver._hardware_compatibility(  # noqa: SLF001
            final_identity=final, grid_identity=identity
        )
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(
            result["comparison"], "exact-except-git-commit-tree-and-head-archive"
        )

        changed = copy.deepcopy(final)
        changed["identity"]["gpu_model"] = "not-the-grid-GPU"
        with self.assertRaisesRegex(
            geometry_waiver.GeometryWaiverError, "beyond permitted Git object IDs"
        ):
            geometry_waiver._hardware_compatibility(  # noqa: SLF001
                final_identity=changed, grid_identity=identity
            )

    def test_wandb_restart_phase_is_exact_and_tamper_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = root / "manifest.json"
            checkpoint = root / "checkpoint.pt"
            reference = root / "reference.pt"
            order.write_text("{}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            reference.write_bytes(b"reference")
            run = root / "run-20260902_141502-runid123"
            files = run / "files"
            files.mkdir(parents=True)
            metadata = files / "wandb-metadata.json"
            summary = files / "wandb-summary.json"
            event = run / "run-runid123.wandb"
            args = [
                "--order-manifest",
                str(order),
                "--model-size",
                "tiny",
                "--device",
                "cuda",
                "--parameter-dtype",
                "bfloat16",
                "--precision",
                "bfloat16",
                "--steps",
                "1000",
                "--batch-size",
                "6",
                "--wandb-mode",
                "online",
                "--compile",
                "--resume",
                str(checkpoint),
                "--exact-reference-checkpoint",
                str(reference),
            ]
            _write_json(
                metadata,
                {
                    "args": args,
                    "gpu_count": 6,
                    "program": "/repo/scripts/overfit_single_chunk.py",
                },
            )
            telemetry = {
                "train/step": 1000,
                "train/world_size": 6,
                "train/global_microbatch_rows": 6,
                "train/local_microbatch_rows": 1,
                "train/gradient_accumulation_steps": 1,
                "train/input_tokens": 24_576_000,
                "checkpoint/completed": 1,
                "checkpoint/has_previous_generation": 1,
                "checkpoint/train_input_tokens": 24_576_000,
            }
            for prefix, minimum in geometry_waiver._WANDB_METRIC_MINIMUMS.items():  # noqa: SLF001
                for index in range(minimum):
                    telemetry.setdefault(f"{prefix}fixture_{index}", index)
            _write_json(summary, telemetry)
            event.write_bytes(b"durable-wandb-events")
            descriptors = {
                "metadata_descriptor": qualification.artifact(metadata, label="metadata"),
                "summary_descriptor": qualification.artifact(summary, label="summary"),
                "event_descriptor": qualification.artifact(event, label="event"),
            }
            result = geometry_waiver._validate_wandb_phase(  # noqa: SLF001
                **descriptors,
                expected_step=1000,
                order_manifest=order,
                checkpoint=checkpoint,
                reference_checkpoint=reference,
            )
            self.assertEqual(result["run_id"], "runid123")
            self.assertEqual(result["input_tokens"], 24_576_000)

            telemetry["train/step"] = 999
            _write_json(summary, telemetry)
            descriptors["summary_descriptor"] = qualification.artifact(
                summary, label="tampered summary"
            )
            with self.assertRaisesRegex(
                geometry_waiver.GeometryWaiverError, "train/step"
            ):
                geometry_waiver._validate_wandb_phase(  # noqa: SLF001
                    **descriptors,
                    expected_step=1000,
                    order_manifest=order,
                    checkpoint=checkpoint,
                    reference_checkpoint=reference,
                )

    @staticmethod
    def _waiver_receipt() -> dict[str, object]:
        tokens_per_update = 6 * 2 * 4096
        token_delta = 100 * tokens_per_update
        return {
            "format": geometry_waiver.WAIVER_FORMAT,
            "format_version": 1,
            "status": "pass",
            "hardware_contract_sha256": "a" * 64,
            "train_order_manifest_sha256": "b" * 64,
            "validation_order_manifest_sha256": "c" * 64,
            "final_soak_waived": True,
            "waiver": {},
            "qualification": {},
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
                    "scope": geometry_evidence.THROUGHPUT_SCOPE,
                    "timer": geometry_evidence.THROUGHPUT_TIMER,
                    "counter": geometry_evidence.THROUGHPUT_COUNTER,
                    "start_consumed_input_tokens": 0,
                    "end_consumed_input_tokens": token_delta,
                    "elapsed_wall_time_ns": token_delta * 1_000_000,
                    "validation_events": 1,
                    "checkpoint_events": 2,
                    "wandb_log_events": 100,
                    "resume_verified": True,
                },
            },
        }

    def test_run_authority_dispatches_to_fail_closed_waiver_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / accept_geometry_waiver.OUTPUT_NAME
            receipt = self._waiver_receipt()
            _write_json(path, receipt)
            _write_sidecar(path)
            with mock.patch(
                "pretrain.geometry_waiver.validate_geometry_waiver_receipt",
                return_value={"status": "pass"},
            ) as validator:
                result = run_authority.inspect_geometry_receipt(
                    path,
                    hardware_contract_sha256="a" * 64,
                    measured_input_tokens_per_second="1000",
                )
            self.assertEqual(result["receipt"], receipt)
            validator.assert_called_once_with(
                receipt,
                expected_hardware_contract_sha256="a" * 64,
            )

    def test_launch_evidence_preserves_all_normal_gates_under_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / accept_geometry_waiver.OUTPUT_NAME
            receipt = self._waiver_receipt()
            _write_json(path, receipt)
            descriptor = qualification.artifact(path, label="waiver receipt")
            authority = {
                "geometry": {"artifact": descriptor, "receipt": receipt},
                "hardware": {
                    "contract": {
                        "path": str(root / "hardware.json"),
                        "bytes": 1,
                        "sha256": "a" * 64,
                    },
                    "expected": {"gpu_memory_bytes": 80 * GIB},
                },
                "training": {
                    "frozen_geometry": {"consumed_input_tokens": 52_580_000_000}
                },
                "data": {
                    "train_order": {"sequence_length": 4096},
                    "validation_order": {"sequence_length": 4096},
                },
            }
            with mock.patch(
                "pretrain.geometry_waiver.validate_geometry_waiver_receipt",
                return_value={"status": "pass", "final_soak_waived": True},
            ) as validator:
                result = geometry_evidence.validate_authority_geometry_soak(authority)
            self.assertTrue(result["final_soak_waived"])
            self.assertEqual(result["soak_steps"], 100)
            self.assertEqual(result["input_token_delta"], 4_915_200)
            validator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
