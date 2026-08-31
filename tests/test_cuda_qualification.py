from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "qualify_cuda_model.py"
SPEC = importlib.util.spec_from_file_location("qualify_cuda_model_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CudaQualificationTest(unittest.TestCase):
    def test_labels_mask_every_document_boundary(self) -> None:
        input_ids = torch.tensor([[11, 12, 21, 22, 23, 31]])
        document_ids = torch.tensor([[0, 0, 1, 1, 1, 2]])
        labels = MODULE._labels_for_documents(input_ids, document_ids)
        self.assertEqual(labels.tolist(), [[12, -100, 22, 23, -100, -100]])

    def test_no_cuda_persists_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "qualification.json"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                exit_code = MODULE.main(["--output", str(output)])
            self.assertEqual(exit_code, 1)
            evidence = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["failure"]["type"], "QualificationError")
            self.assertIn("CUDA is unavailable", evidence["failure"]["message"])
            self.assertEqual(
                evidence["source_identity"]["model_py_sha256"],
                MODULE._sha256(PROJECT_ROOT / "pretrain" / "model.py"),
            )

    def test_existing_evidence_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "qualification.json"
            output.write_text('{"status":"passed"}\n', encoding="utf-8")
            with self.assertRaises(SystemExit):
                MODULE.main(["--output", str(output)])
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "passed",
            )


if __name__ == "__main__":
    unittest.main()
