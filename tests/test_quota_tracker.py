from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quota_tracker import estimate_tokens, quota_status, write_record


class QuotaTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.config = {
            "version": 1,
            "estimation": {"bytes_per_token": {"python": 4.0}},
            "quotas": [
                {
                    "name": "collection/python",
                    "phase": "collection",
                    "category": "python",
                    "token_field": "estimated_tokens",
                    "target": 100,
                }
            ],
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_estimate_tokens(self) -> None:
        self.assertEqual(estimate_tokens(400, "python", self.config), 100)

    def test_idempotent_record_and_status(self) -> None:
        record = {
            "phase": "collection",
            "category": "python",
            "shard_id": "shard-1",
            "estimated_tokens": 100,
        }
        path, created = write_record(self.root, record)
        self.assertTrue(created)
        same_path, created_again = write_record(self.root, record)
        self.assertEqual(path, same_path)
        self.assertFalse(created_again)
        status = quota_status(self.root, self.config, phase="collection")
        self.assertEqual(status[0]["current"], 100)
        self.assertTrue(status[0]["reached"])

    def test_conflicting_retry_is_rejected(self) -> None:
        record = {
            "phase": "collection",
            "category": "python",
            "shard_id": "shard-1",
            "estimated_tokens": 10,
        }
        write_record(self.root, record)
        changed = {**record, "estimated_tokens": 11}
        with self.assertRaises(FileExistsError):
            write_record(self.root, changed)


if __name__ == "__main__":
    unittest.main()
