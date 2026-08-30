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


if __name__ == "__main__":
    unittest.main()
