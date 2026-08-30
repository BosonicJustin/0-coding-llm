from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import probe_sqlite_storage as probe_module
from probe_sqlite_storage import run_probe


class SQLiteStorageProbeTest(unittest.TestCase):
    MOUNT = {
        "target": "/network/work",
        "source": "server:/durable-volume",
        "fstype": "nfs4",
        "options": "rw,hard,local_lock=none",
        "stat_device": 42,
        "statvfs_fsid": 43,
    }

    def run_qualified_probe(self, **kwargs: object) -> dict[str, object]:
        with mock.patch.object(
            probe_module, "mount_evidence", return_value=self.MOUNT
        ):
            return run_probe(
                required_fstype="nfs4",
                required_source="server:/durable-volume",
                required_mount_options=("hard", "local_lock=none"),
                **kwargs,
            )

    def test_process_crash_locking_and_throughput_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_qualified_probe(
                root=root,
                rows=2_000,
                batch_rows=500,
                minimum_rows_per_second=1,
                minimum_index_rows_per_second=1,
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["production_gate_eligible"])
            self.assertTrue(result["correctness"]["competing_writer_excluded"])
            self.assertTrue(
                result["correctness"]["atomic_hardlink_lease_exclusion"]
            )
            self.assertTrue(result["correctness"]["advisory_flock_exclusion"])
            self.assertTrue(result["correctness"]["hot_journal_observed"])
            self.assertTrue(result["correctness"]["hot_journal_magic_valid"])
            self.assertGreater(result["correctness"]["hot_journal_bytes"], 512)
            self.assertTrue(
                result["correctness"]["main_database_changed_before_recovery"]
            )
            self.assertTrue(
                result["correctness"]["main_database_restored_exactly"]
            )
            self.assertEqual(
                result["correctness"]["post_crash_integrity_check"], "ok"
            )
            self.assertEqual(
                result["correctness"]["post_crash_uncommitted_rows_visible"], 0
            )
            self.assertFalse(
                any(path.name.startswith(".sqlite-curation-probe-") for path in root.iterdir())
            )

    def test_throughput_threshold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_qualified_probe(
                root=Path(temporary),
                rows=1_000,
                batch_rows=500,
                minimum_rows_per_second=10**15,
                minimum_index_rows_per_second=1,
            )
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["production_gate_eligible"])
            self.assertEqual(
                result["failures"],
                ["measured_insert_rows_per_second_below_requested_minimum"],
            )

    def test_index_throughput_threshold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_qualified_probe(
                root=Path(temporary),
                rows=1_000,
                batch_rows=500,
                minimum_rows_per_second=1,
                minimum_index_rows_per_second=10**15,
            )
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["production_gate_eligible"])
            self.assertIn(
                "measured_index_rows_per_second_below_requested_minimum",
                result["failures"],
            )

    def test_mount_requirements_are_mandatory_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            probe_module, "mount_evidence", return_value=self.MOUNT
        ):
            result = run_probe(
                root=Path(temporary),
                rows=1_000,
                batch_rows=500,
                minimum_rows_per_second=1,
                minimum_index_rows_per_second=1,
            )
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["production_gate_eligible"])
            self.assertIn(
                "durable_mount_requirements_not_fully_specified",
                result["failures"],
            )

    def test_mount_identity_and_options_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            probe_module, "mount_evidence", return_value=self.MOUNT
        ):
            result = run_probe(
                root=Path(temporary),
                rows=1_000,
                batch_rows=500,
                minimum_rows_per_second=1,
                minimum_index_rows_per_second=1,
                required_fstype="ext4",
                required_source="wrong:/volume",
                required_mount_options=("hard", "nconnect=99"),
            )
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["production_gate_eligible"])
            self.assertIn(
                "filesystem_type_does_not_match_required_mount",
                result["failures"],
            )
            self.assertIn(
                "filesystem_source_does_not_match_required_mount",
                result["failures"],
            )
            self.assertIn("required_mount_options_are_missing", result["failures"])
            self.assertEqual(
                result["mount_requirements"]["missing_options"], ["nconnect=99"]
            )


if __name__ == "__main__":
    unittest.main()
