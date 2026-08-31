from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "monitor_curation.py"
SPEC = importlib.util.spec_from_file_location("monitor_curation_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class CurationMonitorTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        output = root / "selection"
        work = output / ".work"
        work.mkdir(parents=True)
        checkpoint = {
            "phase": "inventory",
            "identity": {
                "collection_completeness": {
                    "complete": True,
                    "preprocess_error_records": 0,
                }
            },
            "last_event_sequence": 2,
            "counts": {
                "archives": 1,
                "documents": 13,
                "selected_documents": 0,
                "output_archives": 0,
            },
            "subphases": [
                {
                    "subphase": "inventory.archive.python.000000",
                    "status": "complete",
                    "processed_rows": 10,
                    "details": {"expected_documents": 10},
                },
                {
                    "subphase": "inventory.archive.python.000001",
                    "status": "running",
                    "processed_rows": 3,
                    "details": {"expected_documents": 8},
                },
            ],
            "storage": {
                "committed_transactions": 3,
                "maximum_transaction_rows": 5,
                "maximum_journal_bytes": 100,
                "maximum_wal_bytes": 0,
                "violation": {},
                "preflight": {
                    "status": "pass",
                    "documents_expected": 18,
                    "contract": {
                        "maximum_transaction_rows": 5,
                        "transaction_sidecar_limit_bytes": 1000,
                        "minimum_free_bytes_after_projection": 1,
                    },
                },
            },
        }
        write_json(work / "CHECKPOINT.json", checkpoint)
        events = [
            {"event": "initialized", "payload": {}, "sequence": 1},
            {
                "event": "archive_ingested",
                "payload": {"archive": "a", "documents": 10, "tokens": 20},
                "sequence": 2,
            },
        ]
        (work / "journal.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        write_json(
            output / ".curation.cross-client-lease.json",
            {"output": str(output), "pid": os.getpid()},
        )
        return output

    def inspect(self, output: Path):
        return MODULE.inspect(
            output,
            stall_seconds=60,
            now=(output / ".work" / "CHECKPOINT.json").stat().st_mtime,
            process_checker=lambda pid, path: (True, f"fake:{pid}:{path}"),
        )

    def remove_active_archive(self, output: Path, *, documents: int = 10) -> Path:
        checkpoint_path = output / ".work" / "CHECKPOINT.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["counts"]["documents"] = documents
        checkpoint["subphases"] = [
            subphase
            for subphase in checkpoint["subphases"]
            if subphase["status"] == "complete"
        ]
        write_json(checkpoint_path, checkpoint)
        return checkpoint_path

    def test_healthy_inventory_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            health = self.inspect(output)
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["counts"]["archives"], 1)
        self.assertEqual(health["counts"]["documents"], 13)
        self.assertAlmostEqual(health["counts"]["inventory_percent"], 100 * 13 / 18)

    def test_local_sqlite_mode_reads_live_work_without_changing_output_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            durable_work = output / ".work"
            local_work = root / "local-sqlite"
            durable_work.rename(local_work)
            health = MODULE.inspect(
                output,
                stall_seconds=60,
                live_work_root=local_work,
                now=(local_work / "CHECKPOINT.json").stat().st_mtime,
                process_checker=lambda pid, path: (True, f"fake:{pid}:{path}"),
            )
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["counts"]["documents"], 13)
        self.assertEqual(health["live_work_root"], str(local_work.resolve()))

    def test_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint_path = output / ".work" / "CHECKPOINT.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["counts"]["documents"] = 14
            write_json(checkpoint_path, checkpoint)
            with self.assertRaisesRegex(MODULE.HealthError, "durable count"):
                self.inspect(output)

    def test_storage_violation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint_path = output / ".work" / "CHECKPOINT.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["storage"]["violation"] = {"reason": "sidecar"}
            write_json(checkpoint_path, checkpoint)
            with self.assertRaisesRegex(MODULE.HealthError, "storage violation"):
                self.inspect(output)

    def test_journal_sequence_gap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            journal = output / ".work" / "journal.jsonl"
            lines = journal.read_text(encoding="utf-8").splitlines()
            second = json.loads(lines[1])
            second["sequence"] = 3
            journal.write_text(lines[0] + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.HealthError, "discontinuity"):
                self.inspect(output)

    def test_journal_must_start_at_sequence_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            journal = output / ".work" / "journal.jsonl"
            event = json.loads(journal.read_text(encoding="utf-8").splitlines()[1])
            journal.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.HealthError, "begin at sequence 1"):
                self.inspect(output)

    def test_publication_race_retries_until_checkpoint_and_journal_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            journal = output / ".work" / "journal.jsonl"
            complete = MODULE._read_journal(journal)
            stale = complete[:1]
            with mock.patch.object(
                MODULE,
                "_read_journal",
                side_effect=[stale, complete],
            ), mock.patch.object(MODULE.time, "sleep") as sleep:
                health = self.inspect(output)
        self.assertEqual(health["status"], "healthy")
        sleep.assert_called_once()

    def test_persistent_projection_mismatch_fails_after_bounded_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            journal = output / ".work" / "journal.jsonl"
            stale = MODULE._read_journal(journal)[:1]
            with mock.patch.object(
                MODULE,
                "_read_journal",
                return_value=stale,
            ), mock.patch.object(MODULE.time, "sleep") as sleep:
                with self.assertRaisesRegex(
                    MODULE.HealthError, "mismatch after publication-race retries"
                ):
                    self.inspect(output)
        self.assertEqual(sleep.call_count, 4)

    def test_fresh_live_resume_without_active_archive_gets_bounded_grace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint = self.remove_active_archive(output)
            health = MODULE.inspect(
                output,
                stall_seconds=3_600,
                now=checkpoint.stat().st_mtime,
                process_checker=lambda pid, path: (True, f"fake:{pid}:{path}"),
            )
        self.assertEqual(health["status"], "warning")
        self.assertTrue(health["startup_publication_grace"])
        self.assertIsNone(health["active_subphase"])
        self.assertIn(
            "inventory_active_archive_pending_startup_publication",
            health["warnings"],
        )

    def test_missing_active_archive_fails_after_narrow_grace_even_before_stall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint = self.remove_active_archive(output)
            with self.assertRaisesRegex(
                MODULE.HealthError, "outside startup publication grace"
            ):
                MODULE.inspect(
                    output,
                    stall_seconds=3_600,
                    now=(
                        checkpoint.stat().st_mtime
                        + MODULE.STARTUP_PUBLICATION_GRACE_SECONDS
                        + 1
                    ),
                    process_checker=lambda pid, path: (
                        True,
                        f"fake:{pid}:{path}",
                    ),
                )

    def test_missing_active_archive_never_gets_grace_for_dead_curator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint = self.remove_active_archive(output)
            with self.assertRaisesRegex(MODULE.HealthError, "process is not healthy"):
                MODULE.inspect(
                    output,
                    stall_seconds=3_600,
                    now=checkpoint.stat().st_mtime,
                    process_checker=lambda pid, path: (False, "process_missing"),
                )

    def test_startup_grace_does_not_mask_inventory_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self.fixture(Path(temporary))
            checkpoint = self.remove_active_archive(output, documents=11)
            with self.assertRaisesRegex(MODULE.HealthError, "durable count"):
                MODULE.inspect(
                    output,
                    stall_seconds=3_600,
                    now=checkpoint.stat().st_mtime,
                    process_checker=lambda pid, path: (
                        True,
                        f"fake:{pid}:{path}",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
