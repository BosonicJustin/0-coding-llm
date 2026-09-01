from __future__ import annotations

import importlib.util
import json
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


class OverfitGateArtifactTest(unittest.TestCase):
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
            self.assertEqual(resumed["exact_resume"]["mismatches"], [])

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
