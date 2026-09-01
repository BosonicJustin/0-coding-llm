from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_curation_supply.py"
SPEC = importlib.util.spec_from_file_location("audit_curation_supply_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE documents (
    doc_id BLOB PRIMARY KEY,
    bucket TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    source_group BLOB NOT NULL,
    selection_rank BLOB NOT NULL
) WITHOUT ROWID;
CREATE TABLE reasons (
    doc_id BLOB NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY(doc_id, reason)
) WITHOUT ROWID;
CREATE TABLE groups (
    group_id BLOB PRIMARY KEY,
    split TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE phase_progress (
    subphase TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    cursor_json TEXT NOT NULL,
    processed_rows INTEGER NOT NULL,
    processed_tokens INTEGER NOT NULL,
    committed_batches INTEGER NOT NULL,
    details_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE archives (
    report_path TEXT PRIMARY KEY,
    bucket TEXT NOT NULL,
    documents INTEGER NOT NULL,
    tokens INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX documents_source_group ON documents(source_group);
CREATE INDEX documents_selection_v2
    ON documents(bucket, selection_rank, doc_id);
CREATE INDEX reasons_reason ON reasons(reason);
"""


class SupplyAuditTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, sqlite3.Connection]:
        database = root / "curation.sqlite3"
        connection = sqlite3.connect(database)
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        self.assertEqual(str(mode).casefold(), "wal")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (("database_version", "4"), ("phase", json.dumps("canonicalized"))),
        )
        groups = (
            (b"p", "train"),
            (b"o", "validation"),
            (b"f", "train"),
            (b"w", "test"),
            (b"r", "train"),
        )
        connection.executemany(
            "INSERT INTO groups(group_id, split) VALUES (?, ?)", groups
        )
        documents = (
            (b"p1", "python", 10, b"p", b"1"),
            (b"p2", "python", 11, b"p", b"2"),
            (b"o1", "other_code", 20, b"o", b"3"),
            (b"f1", "fineweb_edu", 30, b"f", b"4"),
            (b"w1", "wikipedia", 40, b"w", b"5"),
            (b"reject", "python", 999, b"r", b"6"),
        )
        connection.executemany(
            "INSERT INTO documents(doc_id, bucket, tokens, source_group, selection_rank) "
            "VALUES (?, ?, ?, ?, ?)",
            documents,
        )
        connection.execute(
            "INSERT INTO reasons(doc_id, reason) VALUES (?, ?)",
            (b"reject", "quality:fixture"),
        )
        connection.execute(
            "INSERT INTO phase_progress VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "selection.groups",
                "complete",
                json.dumps({"group_id": "ff" * 32}),
                5,
                0,
                1,
                json.dumps({"groups": 5, "mismatched_assignments": 0}),
            ),
        )
        connection.executemany(
            "INSERT INTO archives VALUES (?, ?, ?, ?)",
            (
                ("python", "python", 3, 1020),
                ("other", "other_code", 1, 20),
                ("fineweb", "fineweb_edu", 1, 30),
                ("wikipedia", "wikipedia", 1, 40),
            ),
        )
        connection.execute(
            "INSERT INTO events(event, payload) VALUES ('canonicalized', ?)",
            (
                json.dumps(
                    {
                        "accepted_canonical_documents": 5,
                        "final_choices": 5,
                    }
                ),
            ),
        )
        connection.commit()
        # Keep the writer and its uncheckpointed WAL open while the audit runs.
        self.assertTrue(Path(f"{database}-wal").is_file())
        return database, connection

    @staticmethod
    def set_accepted_authority(connection: sqlite3.Connection, documents: int) -> None:
        connection.execute(
            "UPDATE events SET payload=? WHERE event='canonicalized'",
            (
                json.dumps(
                    {
                        "accepted_canonical_documents": documents,
                        "final_choices": documents,
                    }
                ),
            ),
        )

    def test_live_wal_snapshot_reports_exact_multidimensional_supply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            try:
                report = MODULE.audit_database(database)
            finally:
                writer.close()

        self.assertTrue(report["safe"])
        self.assertEqual(
            report["totals"]["eligible_observed"],
            {"documents": 5, "tokens": 111},
        )
        self.assertEqual(
            report["totals"]["by_split_category"]["train"]["python"],
            {"documents": 2, "tokens": 21},
        )
        self.assertEqual(
            report["totals"]["by_split_category"]["train"]["english"],
            {"documents": 1, "tokens": 30},
        )
        self.assertEqual(
            report["totals"]["by_bucket"]["wikipedia"],
            {"documents": 1, "tokens": 40},
        )
        opened = report["provenance"]["sqlite_open"]
        self.assertEqual(opened["mode"], "ro")
        self.assertIs(opened["immutable"], False)
        self.assertEqual(report["provenance"]["journal_mode"], "wal")
        self.assertTrue(report["provenance"]["storage_before"]["wal"]["exists"])
        self.assertEqual(report["scope"]["documents_scans"], 1)
        self.assertEqual(
            report["unique_rejected"]["by_bucket"]["python"],
            {"documents": 1, "tokens": 999},
        )
        self.assertEqual(
            report["reason_associations"]["by_reason"]["quality:fixture"],
            {"documents": 1, "tokens": 999},
        )
        self.assertTrue(report["raw_accounting"]["safe"])
        performance = report["provenance"]["read_performance"]
        self.assertEqual(performance["temp_store"]["observed"], 2)
        self.assertEqual(
            performance["cache_size_kib"]["observed_pragma"],
            -MODULE.REQUESTED_CACHE_KIB,
        )
        self.assertFalse(performance["durability_pragmas_modified"])

    def test_missing_group_is_reported_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (b"missing", "python", 17, b"absent", b"7"),
            )
            self.set_accepted_authority(writer, 6)
            writer.commit()
            try:
                report = MODULE.audit_database(database)
            finally:
                writer.close()

        self.assertFalse(report["safe"])
        self.assertEqual(
            report["anomalies"]["document_counts"]["missing_group"], 1
        )
        self.assertEqual(report["anomalies"]["missing_groups"][0]["tokens"], 17)
        self.assertEqual(report["totals"]["eligible_observed"]["documents"], 6)
        self.assertEqual(report["totals"]["assigned_valid"]["documents"], 5)

    def test_unknown_bucket_is_reported_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (b"x1", "rustish", 23, b"p", b"7"),
            )
            self.set_accepted_authority(writer, 6)
            writer.commit()
            try:
                report = MODULE.audit_database(database)
            finally:
                writer.close()

        self.assertFalse(report["safe"])
        self.assertEqual(
            report["anomalies"]["document_counts"]["unknown_bucket"], 1
        )
        self.assertEqual(report["anomalies"]["unknown_buckets"][0]["bucket"], "rustish")

    def test_incomplete_group_authority_is_rejected_before_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "UPDATE phase_progress SET status='running' "
                "WHERE subphase='selection.groups'"
            )
            writer.commit()
            try:
                with self.assertRaisesRegex(MODULE.SupplyAuditError, "not complete"):
                    MODULE.audit_database(database)
            finally:
                writer.close()

    def test_group_and_canonical_authorities_reconcile_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute("DELETE FROM groups WHERE group_id=?", (b"r",))
            writer.commit()
            try:
                with self.assertRaisesRegex(MODULE.SupplyAuditError, "does not reconcile"):
                    MODULE.audit_database(database)
            finally:
                writer.close()

        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            self.set_accepted_authority(writer, 4)
            writer.commit()
            try:
                with self.assertRaisesRegex(
                    MODULE.SupplyAuditError, "eligible document total"
                ):
                    MODULE.audit_database(database)
            finally:
                writer.close()

    def test_raw_accounting_mismatch_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "UPDATE archives SET tokens=tokens+1 WHERE bucket='python'"
            )
            writer.commit()
            try:
                report = MODULE.audit_database(database)
            finally:
                writer.close()
        self.assertFalse(report["safe"])
        self.assertFalse(report["raw_accounting"]["safe"])

    def test_unsupported_phase_and_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "UPDATE metadata SET value=? WHERE key='phase'", (json.dumps("inventory"),)
            )
            writer.commit()
            try:
                with self.assertRaisesRegex(MODULE.SupplyAuditError, "not safe"):
                    MODULE.audit_database(database)
            finally:
                writer.close()

        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "bad.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(MODULE.SupplyAuditError, "table is missing"):
                MODULE.audit_database(database)

    def test_cli_returns_distinct_unsafe_status_with_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            writer.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (b"missing", "python", 17, b"absent", b"7"),
            )
            self.set_accepted_authority(writer, 6)
            writer.commit()
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    status = MODULE.main(["--database", str(database)])
            finally:
                writer.close()
        self.assertEqual(status, 3)
        self.assertFalse(json.loads(stdout.getvalue())["safe"])

    def test_cli_heartbeat_uses_stderr_without_corrupting_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, writer = self.fixture(Path(temporary))
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    status = MODULE.main(
                        [
                            "--database",
                            str(database),
                            "--heartbeat-seconds",
                            "0.001",
                        ]
                    )
            finally:
                writer.close()
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(stdout.getvalue())["safe"])
        self.assertIn("elapsed_seconds=", stderr.getvalue())
        self.assertIn("stage=eligible_supply", stderr.getvalue())

    def test_supply_sql_has_one_outer_documents_scan(self) -> None:
        normalized = " ".join(MODULE.SUPPLY_SQL.casefold().split())
        self.assertEqual(normalized.count("from documents as d"), 1)

        with tempfile.TemporaryDirectory() as temporary:
            _database, connection = self.fixture(Path(temporary))
            try:
                plan = MODULE._supply_query_plan(connection)
            finally:
                connection.close()
        self.assertEqual([detail for detail in plan if detail.startswith("SCAN d")], ["SCAN d"])
        self.assertFalse(any("documents_selection_v2" in detail for detail in plan))
        self.assertFalse(any("documents_source_group" in detail for detail in plan))


if __name__ == "__main__":
    unittest.main()
