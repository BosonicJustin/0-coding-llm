from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERFIT_SCRIPT = PROJECT_ROOT / "scripts" / "overfit_single_chunk.py"
OVERFIT_SPEC = importlib.util.spec_from_file_location(
    "overfit_single_chunk_gate_test", OVERFIT_SCRIPT
)
assert OVERFIT_SPEC is not None and OVERFIT_SPEC.loader is not None
OVERFIT_MODULE = importlib.util.module_from_spec(OVERFIT_SPEC)
OVERFIT_SPEC.loader.exec_module(OVERFIT_MODULE)


def _checkpoint_payload(
    *,
    model_dtype: torch.dtype = torch.float32,
    model_delta: float = 0.0,
    optimizer_delta: float = 0.0,
    optimizer_seen: int = 7,
    train_step: int = 11,
) -> dict[str, object]:
    return {
        "format": "pretrain-checkpoint",
        "format_version": 1,
        "torch_version": str(torch.__version__),
        "runtime_signature": {"device": "test"},
        "implementation_signature": "implementation-sha256",
        "model": {
            "weight": torch.tensor(
                [1.0 + model_delta, -2.0, 0.0],
                dtype=model_dtype,
            )
        },
        "model_config": {"dim": 3},
        "parameter_dtypes": {"weight": str(model_dtype)},
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(11.0),
                    "exp_avg": torch.tensor([0.25 + optimizer_delta, -0.5]),
                    "seen": torch.tensor(optimizer_seen, dtype=torch.int64),
                }
            },
            "param_groups": [{"lr": 1e-3, "params": [0]}],
        },
        "optimizer_class": "AdamW",
        "train_state": {"completed_steps": train_step},
        "rng_states": {"cpu": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        "train_trajectory_config": {"seed": 1234},
        "training_geometry": {"world_size": 6},
        "validation_configuration": {"every": 10},
        "data_identity": "data-sha256",
        "tokenizer_manifest_sha256": "a" * 64,
        "tokenizer_vocabulary_sha256": "b" * 64,
        "world_size": 6,
    }


def _save_checkpoint(path: Path, **changes: object) -> None:
    torch.save(_checkpoint_payload(**changes), path)


class OverfitGateArtifactTest(unittest.TestCase):
    def test_explicit_tensor_tolerance_is_narrow_and_json_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pt"
            actual = root / "actual.pt"
            _save_checkpoint(reference)
            _save_checkpoint(
                actual,
                model_delta=1e-7,
                optimizer_delta=5e-8,
            )

            exact = OVERFIT_MODULE.compare_checkpoint_trajectories(
                actual,
                reference,
            )
            self.assertFalse(exact["exact_match"])
            self.assertFalse(exact["accepted_match"])
            self.assertEqual(exact["comparison_mode"], "exact")

            with mock.patch.object(
                OVERFIT_MODULE,
                "CHECKPOINT_COMPARE_CHUNK_ELEMENTS",
                2,
            ):
                tolerant = OVERFIT_MODULE.compare_checkpoint_trajectories(
                    actual,
                    reference,
                    tensor_atol=2e-7,
                    tensor_rtol=1e-5,
                )
            self.assertFalse(tolerant["exact_match"])
            self.assertTrue(tolerant["tolerant_match"])
            self.assertTrue(tolerant["accepted_match"])
            self.assertEqual(
                tolerant["mismatches"],
                ["model", "optimizer"],
            )
            diagnostics = tolerant["numeric_diagnostics"]
            self.assertEqual(diagnostics["chunk_elements"], 2)
            self.assertEqual(diagnostics["out_of_tolerance_element_count"], 0)
            self.assertEqual(diagnostics["non_finite_element_count"], 0)
            self.assertGreaterEqual(
                diagnostics["numerically_different_element_count"],
                2,
            )
            self.assertTrue(
                math.isfinite(diagnostics["max_absolute_difference"])
            )
            self.assertTrue(
                math.isfinite(diagnostics["max_relative_difference"])
            )
            json.dumps(tolerant, allow_nan=False)

    def test_tolerance_never_relaxes_structure_or_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pt"
            _save_checkpoint(reference)

            dtype_mismatch = root / "dtype.pt"
            _save_checkpoint(dtype_mismatch, model_dtype=torch.float64)
            dtype_result = OVERFIT_MODULE.compare_checkpoint_trajectories(
                dtype_mismatch,
                reference,
                tensor_atol=1.0,
                tensor_rtol=1.0,
            )
            self.assertFalse(dtype_result["accepted_match"])
            self.assertIn("model", dtype_result["tolerant_mismatches"])
            self.assertEqual(
                dtype_result["numeric_diagnostics"]["structural_mismatch_count"],
                1,
            )

            integer_mismatch = root / "integer.pt"
            _save_checkpoint(integer_mismatch, optimizer_seen=8)
            integer_result = OVERFIT_MODULE.compare_checkpoint_trajectories(
                integer_mismatch,
                reference,
                tensor_atol=1.0,
                tensor_rtol=1.0,
            )
            self.assertFalse(integer_result["accepted_match"])
            self.assertIn("optimizer", integer_result["tolerant_mismatches"])
            self.assertEqual(
                integer_result["numeric_diagnostics"]["exact_value_mismatch_count"],
                1,
            )

            state_mismatch = root / "state.pt"
            _save_checkpoint(state_mismatch, train_step=12)
            state_result = OVERFIT_MODULE.compare_checkpoint_trajectories(
                state_mismatch,
                reference,
                tensor_atol=1.0,
                tensor_rtol=1.0,
            )
            self.assertFalse(state_result["accepted_match"])
            self.assertIn("train_state", state_result["tolerant_mismatches"])

    def test_non_finite_float_tensor_is_never_tolerance_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pt"
            actual = root / "actual.pt"
            _save_checkpoint(reference)
            payload = _checkpoint_payload()
            payload["model"]["weight"][0] = torch.nan
            torch.save(payload, actual)

            result = OVERFIT_MODULE.compare_checkpoint_trajectories(
                actual,
                reference,
                tensor_atol=2e-7,
                tensor_rtol=1e-5,
            )
            self.assertFalse(result["tolerant_match"])
            self.assertFalse(result["accepted_match"])
            diagnostics = result["numeric_diagnostics"]
            self.assertEqual(diagnostics["non_finite_element_count"], 1)
            self.assertEqual(diagnostics["out_of_tolerance_element_count"], 1)
            json.dumps(result, allow_nan=False)

    def test_resume_tensor_tolerances_must_be_finite_and_nonnegative(self) -> None:
        for value in (float("nan"), float("inf"), -1e-9):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    OVERFIT_MODULE.compare_checkpoint_trajectories(
                        Path("unused-actual.pt"),
                        Path("unused-reference.pt"),
                        tensor_atol=value,
                    )

    def test_fixture_exercises_real_packed_boundary_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            order = OVERFIT_MODULE.build_synthetic_order(
                Path(temporary),
                sequence_length=16,
                vocab_size=64,
            )
            batch = OVERFIT_MODULE.load_one_packed_batch(order, batch_size=2)
            diagnostics = OVERFIT_MODULE.packed_boundary_diagnostics(batch)
            self.assertGreater(diagnostics["in_row_document_boundaries"], 0)
            self.assertEqual(
                diagnostics["in_row_document_boundaries"],
                diagnostics["masked_boundary_targets"],
            )

            transitions = (
                batch["document_ids"][:, 1:]
                != batch["document_ids"][:, :-1]
            )
            row, column = torch.nonzero(transitions, as_tuple=False)[0].tolist()
            batch["labels"][row, column] = 1
            with self.assertRaisesRegex(ValueError, "mask exactly"):
                OVERFIT_MODULE.packed_boundary_diagnostics(batch)

    def test_production_trainer_partial_resume_matches_reference_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = OVERFIT_MODULE.build_synthetic_order(
                root / "fixture",
                sequence_length=16,
                vocab_size=64,
            )
            tokenizer_manifest, tokenizer_vocabulary = (
                OVERFIT_MODULE.synthetic_tokenizer_identities(64)
            )
            common = dict(
                order_manifest=order,
                model_config=OVERFIT_MODULE.tiny_model_config(
                    vocab_size=64,
                    max_seq_len=16,
                ),
                device=torch.device("cpu"),
                parameter_dtype=torch.float32,
                precision="float32",
                steps=30,
                checkpoint_every=10,
                batch_size=2,
                learning_rate=3e-3,
                required_loss_ratio=0.9,
                required_final_loss=10.0,
                seed=1234,
                logger=OVERFIT_MODULE.NullLogger(),
                compile_model=False,
                tokenizer_manifest_sha256=tokenizer_manifest,
                tokenizer_vocabulary_sha256=tokenizer_vocabulary,
            )
            control_checkpoint = root / "control" / "checkpoint.pt"
            control = OVERFIT_MODULE.run_overfit(
                **common,
                checkpoint_path=control_checkpoint,
                resume_path=None,
            )
            self.assertEqual(control["status"], "passed")
            self.assertTrue(
                control_checkpoint.with_name("checkpoint.previous.pt").is_file()
            )

            resumed_checkpoint = root / "resumed" / "checkpoint.pt"
            partial = OVERFIT_MODULE.run_overfit(
                **common,
                checkpoint_path=resumed_checkpoint,
                resume_path=None,
                until_step=15,
            )
            self.assertEqual(partial["status"], "checkpointed")
            resumed = OVERFIT_MODULE.run_overfit(
                **common,
                checkpoint_path=resumed_checkpoint,
                resume_path=resumed_checkpoint,
                exact_reference_checkpoint=control_checkpoint,
            )
            self.assertEqual(resumed["status"], "passed")
            self.assertTrue(resumed["exact_resume"]["exact_match"])
            self.assertTrue(resumed["exact_resume"]["accepted_match"])
            self.assertEqual(resumed["exact_resume"]["mismatches"], [])

            accepted_checkpoint = root / "accepted-tolerance" / "checkpoint.pt"
            OVERFIT_MODULE.run_overfit(
                **common,
                checkpoint_path=accepted_checkpoint,
                resume_path=None,
                until_step=15,
            )
            accepted_comparison = {
                "requested": True,
                "exact_match": False,
                "tolerant_match": True,
                "accepted_match": True,
                "mismatches": ["model"],
                "tolerant_mismatches": [],
            }
            with mock.patch.object(
                OVERFIT_MODULE,
                "compare_checkpoint_trajectories",
                return_value=accepted_comparison,
            ):
                accepted = OVERFIT_MODULE.run_overfit(
                    **common,
                    checkpoint_path=accepted_checkpoint,
                    resume_path=accepted_checkpoint,
                    exact_reference_checkpoint=control_checkpoint,
                    resume_tensor_atol=2e-7,
                    resume_tensor_rtol=1e-5,
                )
            self.assertEqual(accepted["status"], "passed")
            self.assertFalse(accepted["exact_resume"]["exact_match"])
            self.assertTrue(accepted["exact_resume"]["accepted_match"])

            tampered_checkpoint = root / "tampered.pt"
            tampered = torch.load(
                control_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            parameter_name = next(iter(tampered["model"]))
            tampered["model"][parameter_name].reshape(-1)[0].add_(1)
            torch.save(tampered, tampered_checkpoint)
            comparison = OVERFIT_MODULE.compare_checkpoint_trajectories(
                resumed_checkpoint,
                tampered_checkpoint,
            )
            self.assertFalse(comparison["exact_match"])
            self.assertIn("model", comparison["mismatches"])

    def test_acceptance_failure_is_persisted_and_returns_nonzero(self) -> None:
        failure = {
            "status": "failed",
            "failures": ["loss_ratio_above_required_threshold"],
            "initial_loss": 4.0,
            "final_loss": 2.0,
            "loss_ratio": 0.5,
            "required_loss_ratio": 0.25,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "overfit"
            argv = [
                str(OVERFIT_SCRIPT),
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--wandb-mode",
                "disabled",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    OVERFIT_MODULE,
                    "run_overfit",
                    side_effect=OVERFIT_MODULE.OverfitGateError(failure),
                ),
            ):
                self.assertEqual(OVERFIT_MODULE.main(), 1)

            persisted = json.loads(
                (output / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, failure)

    def test_unexpected_interruption_cannot_leave_stale_pass_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "overfit"
            output.mkdir()
            result_path = output / "result.json"
            result_path.write_text('{"status":"passed"}\n', encoding="utf-8")
            argv = [
                str(OVERFIT_SCRIPT),
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--wandb-mode",
                "disabled",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    OVERFIT_MODULE,
                    "run_overfit",
                    side_effect=RuntimeError("simulated interruption"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                OVERFIT_MODULE.main()

            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "running")
            self.assertEqual(persisted["model_size"], "tiny")


if __name__ == "__main__":
    unittest.main()
