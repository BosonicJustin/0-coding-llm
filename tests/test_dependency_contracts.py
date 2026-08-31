from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match is None:
            raise AssertionError(f"Unparseable requirement line in {path}: {line!r}")
        requirements[match.group(1).casefold().replace("_", "-")] = line
    return requirements


class DependencyContractTest(unittest.TestCase):
    def test_data_runtime_declares_every_direct_external_import(self) -> None:
        requirements = requirement_lines(PROJECT_ROOT / "requirements-data.txt")
        self.assertTrue(
            {
                "datasets",
                "huggingface-hub",
                "pyarrow",
                "tokenizers",
                "transformers",
                "xxhash",
                "zstandard",
            }
            <= requirements.keys()
        )

    def test_training_runtime_declares_every_direct_external_import(self) -> None:
        requirements = requirement_lines(PROJECT_ROOT / "requirements-train.txt")
        self.assertTrue({"numpy", "torch", "tokenizers"} <= requirements.keys())

    def test_shared_tokenizer_constraint_is_identical_across_environments(self) -> None:
        training = requirement_lines(PROJECT_ROOT / "requirements-train.txt")
        data = requirement_lines(PROJECT_ROOT / "requirements-data.txt")
        self.assertEqual(training["tokenizers"], data["tokenizers"])

    def test_ci_verifies_training_environment_before_data_install(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(
            encoding="utf-8"
        )
        training_install = workflow.index("pip install -r requirements-train.txt")
        standalone_import = workflow.index("Verify standalone training runtime")
        data_install = workflow.index("pip install -r requirements-data.txt")
        self.assertLess(training_install, standalone_import)
        self.assertLess(standalone_import, data_install)


if __name__ == "__main__":
    unittest.main()
