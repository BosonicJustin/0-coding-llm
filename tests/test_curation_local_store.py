from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from curation_local_store import (  # noqa: E402
    LocalSQLiteStore,
    LocalStoreError,
    storage_admission,
)


def create_database(path: Path, *, value: int = 0) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE durable_counts(
                singleton INTEGER PRIMARY KEY,
                archives INTEGER NOT NULL,
                documents INTEGER NOT NULL,
                selected_documents INTEGER NOT NULL,
                output_archives INTEGER NOT NULL
            );
            CREATE TABLE phase_progress(
                subphase TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                cursor_json TEXT NOT NULL,
                processed_rows INTEGER NOT NULL,
                processed_tokens INTEGER NOT NULL,
                committed_batches INTEGER NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE payload(value INTEGER NOT NULL);
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (("database_version", "4"), ("phase", json.dumps("inventory"))),
        )
        connection.execute(
            "INSERT INTO durable_counts VALUES (1, ?, ?, 0, 0)",
            (value, value),
        )
        connection.execute(
            "INSERT INTO phase_progress VALUES (?, ?, ?, ?, 0, ?, ?)",
            (
                "inventory.decode",
                "running",
                json.dumps({"cursor": value}, sort_keys=True),
                value,
                value,
                "{}",
            ),
        )
        connection.execute(
            "INSERT INTO events(event, payload) VALUES ('progress', ?)",
            (json.dumps({"value": value}, sort_keys=True),),
        )
        connection.execute("INSERT INTO payload VALUES (?)", (value,))
        connection.commit()
    finally:
        connection.close()


def advance(connection: sqlite3.Connection, value: int) -> None:
    connection.execute("UPDATE payload SET value=?", (value,))
    connection.execute(
        "UPDATE durable_counts SET archives=?, documents=? WHERE singleton=1",
        (value, value),
    )
    connection.execute(
        "UPDATE phase_progress SET cursor_json=?, processed_rows=?, "
        "committed_batches=? WHERE subphase='inventory.decode'",
        (json.dumps({"cursor": value}, sort_keys=True), value, value),
    )
    connection.execute(
        "INSERT INTO events(event, payload) VALUES ('progress', ?)",
        (json.dumps({"value": value}, sort_keys=True),),
    )
    connection.commit()


class LocalStoreFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.durable = root / "durable" / ".work"
        self.durable.mkdir(parents=True)
        self.canonical = self.durable / "curation.sqlite3"
        create_database(self.canonical)
        self.identity = {"corpus": "fixture", "reports_sha256": "a" * 64}
        self.admission = {
            "status": "pass",
            "minimum_free_bytes": 1,
            "required_free_bytes": 1,
        }

    def store(self, local: Path) -> LocalSQLiteStore:
        return LocalSQLiteStore(
            local_root=local,
            durable_work=self.durable,
            canonical_db=self.canonical,
            identity=self.identity,
            admission=self.admission,
            canonical_journal_mode="delete",
            snapshot_interval_seconds=3_600,
            snapshot_retention=2,
            runtime_provenance={"test": True},
        )


class CurationLocalStoreTests(unittest.TestCase):
    def test_storage_admission_credits_owned_bytes_only_against_projection(self) -> None:
        usage = shutil._ntuple_diskusage(1_000, 800, 200)
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "curation_local_store.shutil.disk_usage", return_value=usage
        ):
            admission = storage_admission(
                local_root=Path(temporary) / "nested" / "not-created-yet",
                filesystem_type="ext4",
                expected_documents=10,
                projected_bytes_per_document=10,
                safety_numerator=2,
                safety_denominator=1,
                transaction_sidecar_bytes=25,
                minimum_free_bytes=50,
                reclaimable_existing_bytes=100,
                projection_basis={"format": "measured-fixture-v1"},
            )
        self.assertEqual(admission["projected_database_bytes_with_safety"], 200)
        self.assertEqual(admission["remaining_database_bytes_with_safety"], 100)
        self.assertEqual(admission["required_free_bytes"], 175)
        self.assertEqual(admission["projected_bytes_per_document"], 10)
        self.assertEqual(admission["safety_numerator"], 2)
        self.assertEqual(admission["safety_denominator"], 1)
        self.assertEqual(
            admission["projection_basis"], {"format": "measured-fixture-v1"}
        )

        # Existing database occupancy cannot pay for the transaction reserve
        # or the minimum-free invariant.
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "curation_local_store.shutil.disk_usage", return_value=usage
        ):
            with self.assertRaisesRegex(LocalStoreError, "Insufficient"):
                storage_admission(
                    local_root=Path(temporary),
                    filesystem_type="ext4",
                    expected_documents=10,
                    projected_bytes_per_document=10,
                    safety_numerator=2,
                    safety_denominator=1,
                    transaction_sidecar_bytes=100,
                    minimum_free_bytes=101,
                    reclaimable_existing_bytes=1_000,
                )

    def test_storage_admission_rejects_tmpfs_without_cgroup_headroom(self) -> None:
        usage = shutil._ntuple_diskusage(10_000, 0, 10_000)
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "curation_local_store.shutil.disk_usage", return_value=usage
        ):
            with self.assertRaisesRegex(LocalStoreError, "cgroup headroom"):
                storage_admission(
                    local_root=Path(temporary),
                    filesystem_type="tmpfs",
                    expected_documents=10,
                    projected_bytes_per_document=10,
                    safety_numerator=2,
                    safety_denominator=1,
                    transaction_sidecar_bytes=10,
                    minimum_free_bytes=10,
                    cgroup_memory={"available_bytes": 100},
                )

    def test_snapshot_retention_recovery_falls_back_from_corrupt_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            store = fixture.store(Path(temporary) / "local-a")
            self.assertEqual(store.prepare()["source"], "canonical-durable-database")
            connection = sqlite3.connect(store.local_db)
            try:
                self.assertEqual(
                    connection.execute("PRAGMA journal_mode=WAL").fetchone()[0],
                    "wal",
                )
                for value in (1, 2, 3):
                    advance(connection, value)
                    store.snapshot(
                        connection,
                        reason=f"value:{value}",
                        authority_artifacts={
                            "CHECKPOINT.json": json.dumps(
                                {"cursor": value}, sort_keys=True
                            ).encode("utf-8"),
                            "journal.jsonl": f'{{"value":{value}}}\n'.encode("ascii"),
                        },
                    )
            finally:
                connection.close()

            generations = sorted(store.snapshot_root.glob("snapshot-*"))
            self.assertEqual(len(generations), 3)
            self.assertFalse((generations[0] / "curation.sqlite3").exists())
            self.assertTrue((generations[0] / "retirement.json").is_file())
            valid, invalid = store.valid_snapshots()
            self.assertEqual([row["generation"] for row in valid], [2, 3])
            self.assertEqual(invalid, [])

            latest_db = generations[-1] / "curation.sqlite3"
            latest_db.write_bytes(b"corrupt")
            recovered = fixture.store(Path(temporary) / "local-b")
            evidence = recovered.prepare()
            self.assertEqual(evidence["source"], "durable-snapshot")
            self.assertEqual(evidence["snapshot_generation"], 2)
            self.assertEqual(evidence["invalid_durable_snapshots"][0]["generation"], 3)
            restored = sqlite3.connect(recovered.local_db)
            try:
                self.assertEqual(
                    restored.execute("SELECT value FROM payload").fetchone()[0], 2
                )
            finally:
                restored.close()

    def test_prepare_reclaims_incomplete_and_retired_crash_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            local = Path(temporary) / "local"
            store = fixture.store(local)
            store.prepare()
            connection = sqlite3.connect(store.local_db)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                snapshots = []
                first_payload_bytes: bytes | None = None
                for value in (1, 2, 3):
                    advance(connection, value)
                    snapshots.append(store.snapshot(connection, reason=str(value)))
                    if value == 1:
                        first_payload_bytes = Path(
                            snapshots[-1]["database"]["path"]
                        ).read_bytes()
            finally:
                connection.close()

            first_directory = Path(snapshots[0]["_manifest_path"]).parent
            # Recreate the retired generation's exact bytes to model a crash
            # after its receipt was durable but before unlink completed.
            assert first_payload_bytes is not None
            (first_directory / "curation.sqlite3").write_bytes(first_payload_bytes)
            incomplete = store.snapshot_root / "snapshot-000000000004"
            incomplete.mkdir()
            (incomplete / ".curation.sqlite3.part").write_bytes(b"partial")

            resumed = fixture.store(local)
            resumed.prepare()
            self.assertFalse((first_directory / "curation.sqlite3").exists())
            self.assertFalse(incomplete.exists())

    def test_ahead_local_audit_is_not_durable_or_recovery_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            local = Path(temporary) / "local-a"
            store = fixture.store(local)
            store.prepare()
            connection = sqlite3.connect(store.local_db)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                advance(connection, 1)
                snapshot = store.snapshot(
                    connection,
                    reason="durable-cursor:1",
                    authority_artifacts={
                        "CHECKPOINT.json": b'{"cursor":1}\n',
                        "journal.jsonl": b'{"cursor":1}\n',
                    },
                )
                advance(connection, 2)
                (local / "CHECKPOINT.json").write_bytes(b'{"cursor":2}\n')
                (local / "journal.jsonl").write_bytes(b'{"cursor":2}\n')
            finally:
                connection.close()

            self.assertFalse((fixture.durable / "CHECKPOINT.json").exists())
            self.assertEqual(
                json.loads(
                    (Path(snapshot["_manifest_path"]).parent / "CHECKPOINT.json")
                    .read_text(encoding="utf-8")
                )["cursor"],
                1,
            )
            shutil.rmtree(local)
            recovered = fixture.store(Path(temporary) / "local-b")
            recovered.prepare()
            restored = sqlite3.connect(recovered.local_db)
            try:
                self.assertEqual(
                    restored.execute("SELECT value FROM payload").fetchone()[0], 1
                )
            finally:
                restored.close()

    def test_promotion_is_verified_and_does_not_retain_full_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            store = fixture.store(Path(temporary) / "local")
            store.prepare()
            connection = sqlite3.connect(store.local_db)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                advance(connection, 4)
                snapshot = store.snapshot(
                    connection,
                    reason="promotion",
                    authority_artifacts={
                        "CHECKPOINT.json": b'{"cursor":4}\n',
                        "journal.jsonl": b'{"cursor":4}\n',
                    },
                )
            finally:
                connection.close()
            receipt = store.promote_latest()
            self.assertEqual(receipt["status"], "complete")
            generation = Path(snapshot["_manifest_path"]).parent
            self.assertFalse(
                (generation / "canonical-before-promotion.sqlite3").exists()
            )
            canonical = sqlite3.connect(fixture.canonical)
            try:
                self.assertEqual(
                    canonical.execute("SELECT value FROM payload").fetchone()[0], 4
                )
            finally:
                canonical.close()
            self.assertEqual(store.promote_latest(), receipt)

    def test_promotion_capacity_failure_preserves_previous_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            before = fixture.canonical.read_bytes()
            store = fixture.store(Path(temporary) / "local")
            store.prepare()
            connection = sqlite3.connect(store.local_db)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                advance(connection, 7)
                snapshot = store.snapshot(connection, reason="capacity")
            finally:
                connection.close()
            database_bytes = int(snapshot["database"]["bytes"])
            constrained = shutil._ntuple_diskusage(
                database_bytes * 4,
                database_bytes * 3,
                database_bytes,
            )
            with mock.patch(
                "curation_local_store.shutil.disk_usage", return_value=constrained
            ):
                with self.assertRaisesRegex(
                    LocalStoreError, "capacity for canonical SQLite promotion"
                ):
                    store.promote_latest()
            self.assertEqual(fixture.canonical.read_bytes(), before)
            self.assertFalse(
                (Path(snapshot["_manifest_path"]).parent / "promotion.json").exists()
            )

    def test_snapshot_rejects_an_active_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalStoreFixture(Path(temporary))
            store = fixture.store(Path(temporary) / "local")
            store.prepare()
            connection = sqlite3.connect(store.local_db)
            try:
                connection.execute("BEGIN IMMEDIATE")
                with self.assertRaisesRegex(LocalStoreError, "active SQLite transaction"):
                    store.snapshot(connection, reason="unsafe")
            finally:
                connection.rollback()
                connection.close()


if __name__ == "__main__":
    unittest.main()
