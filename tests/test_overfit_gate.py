from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERFIT_SCRIPT = PROJECT_ROOT / "scripts" / "overfit_single_chunk.py"
OVERFIT_SPEC = importlib.util.spec_from_file_location(
    "overfit_single_chunk_gate_test", OVERFIT_SCRIPT
)
assert OVERFIT_SPEC is not None and OVERFIT_SPEC.loader is not None
OVERFIT_MODULE = importlib.util.module_from_spec(OVERFIT_SPEC)
OVERFIT_SPEC.loader.exec_module(OVERFIT_MODULE)


class OverfitGateArtifactTest(unittest.TestCase):
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
