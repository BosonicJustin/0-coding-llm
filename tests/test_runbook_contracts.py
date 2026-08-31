from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_POLICY = "configs/curation_policy_fast_exact_normalized.json"


class RunbookContractTest(unittest.TestCase):
    def test_network_curation_probe_is_bound_to_the_exact_work_mount(self) -> None:
        text = (
            PROJECT_ROOT / "docs" / "operations" / "production-runbook.md"
        ).read_text(encoding="utf-8")
        blocks = [
            block
            for block in re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)
            if "probe_sqlite_storage.py" in block
        ]
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertIn('--root "$CURATION_WORK/.work"', block)
        self.assertIn("--require-fstype nfs4", block)
        self.assertIn('RUNPOD_NFS_SOURCE="${RUNPOD_NFS_SOURCE:?', block)
        self.assertIn('--require-source "$RUNPOD_NFS_SOURCE"', block)
        self.assertIn('jq --arg expected_source "$RUNPOD_NFS_SOURCE"', block)
        self.assertIn(".mount.source == $expected_source", block)
        self.assertIn("--require-mount-option hard", block)
        self.assertIn("--require-mount-option local_lock=none", block)

    def test_fast_materialization_commands_pin_the_matching_policy(self) -> None:
        checked = 0
        for relative in (
            "docs/data/materialization.md",
            "docs/operations/production-runbook.md",
        ):
            text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for block in re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL):
                if (
                    "materialize_training_corpus.py" not in block
                    or "selection-fast-v1" not in block
                ):
                    continue
                checked += 1
                self.assertIn("--curation-policy", block, relative)
                self.assertIn(FAST_POLICY, block, relative)
        self.assertGreaterEqual(checked, 5)


if __name__ == "__main__":
    unittest.main()
