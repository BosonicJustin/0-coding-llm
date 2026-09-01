from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from curate_corpus import CurationBuilder
from curation_local_store import LocalSQLiteStore
from curation_policy import FAST_CANONICAL_POLICY
from pretrain.selection_contract import (
    ALL_ELIGIBLE_BITMAP_FORMAT,
    ALL_ELIGIBLE_BITMAP_MAGIC,
    validate_all_eligible_bitmap_header,
    validate_all_eligible_bitmap_payload,
)
from publish_all_eligible_selection import (
    AllEligiblePublicationError,
    AllEligiblePublisher,
)
from test_curate_corpus import CorpusFixture


class AllEligiblePublisherTest(unittest.TestCase):
    def _partial_source(
        self, fixture: CorpusFixture, source: Path
    ) -> tuple[Path, Path]:
        with CurationBuilder(
            root=fixture.root,
            staging_root=fixture.staging,
            output=source,
            policy_path=FAST_CANONICAL_POLICY,
            quota_path=fixture.quota_path,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            english_near_clusters=None,
            allow_missing_english_near_dedup=False,
            batch_size=3,
        ) as builder:
            builder.run(stop_after_phase="canonicalized")
            original = builder._select_quota_bounded
            calls = 0

            def stop_after_one_quota(*, split: str, category: str) -> None:
                nonlocal calls
                if calls:
                    raise RuntimeError("intentional quota stop")
                calls += 1
                original(split=split, category=category)

            with mock.patch.object(
                builder,
                "_select_quota_bounded",
                side_effect=stop_after_one_quota,
            ), self.assertRaisesRegex(RuntimeError, "intentional quota stop"):
                builder.assign_splits_and_quotas()

        live_database = source / ".work" / "curation.sqlite3"
        live_checkpoint = source / ".work" / "CHECKPOINT.json"
        # Model the curator's immutable single-file snapshot without relying on
        # the production snapshot writer in this focused publisher fixture.
        snapshot = source / "test-snapshot"
        snapshot.mkdir()
        database = snapshot / "curation.sqlite3"
        source_connection = sqlite3.connect(live_database)
        destination_connection = sqlite3.connect(database)
        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
            destination_connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            destination_connection.close()
            source_connection.close()
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(f"{database}{suffix}")
            if sidecar.exists():
                if suffix in ("-journal", "-wal"):
                    self.assertEqual(sidecar.stat().st_size, 0)
                sidecar.unlink()
        checkpoint = snapshot / "CHECKPOINT.json"
        shutil.copyfile(live_checkpoint, checkpoint)
        connection = sqlite3.connect(database)
        try:
            self.assertEqual(
                json.loads(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='phase'"
                    ).fetchone()[0]
                ),
                "canonicalized",
            )
            self.assertEqual(
                connection.execute(
                    "SELECT status FROM phase_progress "
                    "WHERE subphase='selection.groups'"
                ).fetchone()[0],
                "complete",
            )
            self.assertGreater(
                connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0],
                0,
            )
        finally:
            connection.close()
        return database, checkpoint

    def _publisher(
        self,
        fixture: CorpusFixture,
        database: Path,
        checkpoint: Path,
        output: Path,
    ) -> AllEligiblePublisher:
        return AllEligiblePublisher(
            root=fixture.root,
            staging_root=fixture.staging,
            source_db=database,
            source_checkpoint=checkpoint,
            source_snapshot_manifest=None,
            allow_unbound_source_for_testing=True,
            output=output,
            policy_path=FAST_CANONICAL_POLICY,
            quota_path=fixture.quota_path,
            benchmark_denylist_path=(
                PROJECT_ROOT / "configs" / "mbpp_denylist.json"
            ),
        )

    def _snapshot_manifest(
        self, database: Path, checkpoint: Path
    ) -> tuple[Path, Path, Path]:
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        durable = database.parent.parent / "durable-snapshot-work"
        snapshot_root = durable / "sqlite-snapshots-v1"
        snapshot_root.mkdir(parents=True)
        store = LocalSQLiteStore(
            local_root=database.parent.parent / "unused-local-store",
            durable_work=durable,
            canonical_db=durable / "unused-canonical.sqlite3",
            identity=checkpoint_payload["identity"],
            admission={"minimum_free_bytes": 1},
            canonical_journal_mode="delete",
            snapshot_interval_seconds=3_600,
            snapshot_retention=2,
            runtime_provenance={"test": True},
        )
        source = sqlite3.connect(database)
        try:
            manifest = store.snapshot(
                source,
                reason="publisher-test",
                authority_artifacts={"CHECKPOINT.json": checkpoint.read_bytes()},
            )
        finally:
            source.close()
        return (
            Path(manifest["database"]["path"]),
            Path(manifest["authority_artifacts"]["CHECKPOINT.json"]["path"]),
            Path(manifest["_manifest_path"]),
        )

    @staticmethod
    def _bitmap(path: Path) -> tuple[dict[str, object], bytes]:
        payload = path.read_bytes()
        assert payload.startswith(ALL_ELIGIBLE_BITMAP_MAGIC)
        start = len(ALL_ELIGIBLE_BITMAP_MAGIC)
        header_length = struct.unpack(">I", payload[start : start + 4])[0]
        header_start = start + 4
        header_end = header_start + header_length
        header = validate_all_eligible_bitmap_header(
            json.loads(payload[header_start:header_end])
        )
        bits = payload[header_end:]
        validate_all_eligible_bitmap_payload(bits, records=int(header["records"]))
        return header, bits

    def test_partial_exact_selection_is_ignored_and_source_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            source_before = database.read_bytes()
            output = root / "all-eligible"

            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                audit = publisher.selected_totals
                result = publisher.run()

            self.assertTrue(result["complete"])
            self.assertEqual(database.read_bytes(), source_before)
            manifest = result["manifest"]
            self.assertFalse(manifest["production_ready"])
            self.assertEqual(
                manifest["publication_scope"], "test-only-unbound-source"
            )
            ignored = manifest["identity"]["source_curation"][
                "ignored_partial_exact_selection"
            ]
            self.assertGreater(ignored["documents"], 0)
            self.assertGreater(ignored["tokens"], 0)
            self.assertFalse(ignored["authority"])
            self.assertNotIn("quotas", manifest)
            self.assertEqual(manifest["selected_totals"], audit)
            self.assertTrue(
                all(
                    row["terminal_prefix_documents"] == 0
                    for row in manifest["selected_totals"]
                )
            )

            observed_records = observed_kept = 0
            for descriptor in manifest["decision_shards"]:
                self.assertEqual(
                    descriptor["format"], ALL_ELIGIBLE_BITMAP_FORMAT
                )
                header, bits = self._bitmap(output / descriptor["path"])
                self.assertEqual(header["archive"], descriptor["archive"])
                self.assertEqual(header["records"], descriptor["records"])
                self.assertEqual(
                    header["kept_documents"], descriptor["kept_documents"]
                )
                self.assertEqual(
                    sum(byte.bit_count() for byte in bits),
                    descriptor["kept_documents"],
                )
                observed_records += int(header["records"])
                observed_kept += int(header["kept_documents"])
            self.assertEqual(observed_records, manifest["documents"]["input"])
            self.assertEqual(
                observed_kept, sum(row["documents"] for row in audit)
            )
            self.assertGreater(observed_records - observed_kept, 0)
            self.assertTrue(
                all(value == 0 for key, value in manifest["leakage_audit"].items()
                    if key != "cross_bucket_code_repo_groups")
            )

    def test_bounded_restart_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            resumed = root / "resumed"
            with self._publisher(
                fixture, database, checkpoint, resumed
            ) as publisher:
                partial = publisher.run(max_new_archives=1)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["new_archives"], 1)

            with self._publisher(
                fixture, database, checkpoint, resumed
            ) as publisher:
                completed = publisher.run()
            self.assertTrue(completed["complete"])

            fresh = root / "fresh"
            with self._publisher(
                fixture, database, checkpoint, fresh
            ) as publisher:
                fresh_result = publisher.run()
            self.assertEqual(completed["manifest"], fresh_result["manifest"])
            for descriptor in completed["manifest"]["decision_shards"]:
                self.assertEqual(
                    (resumed / descriptor["path"]).read_bytes(),
                    (fresh / descriptor["path"]).read_bytes(),
                )

    def test_manifest_sidecar_crash_window_recovers_only_from_full_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "publication"
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                first = publisher.run()
            checksum = output / "manifest.sha256"
            checksum.unlink()
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                recovered = publisher.run()
            self.assertEqual(recovered["manifest"], first["manifest"])
            expected = (
                f"{hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()}  "
                "manifest.json\n"
            )
            self.assertEqual(checksum.read_text(encoding="ascii"), expected)

            (output / "manifest.json").unlink()
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher, self.assertRaisesRegex(
                AllEligiblePublicationError, "checksum exists without its manifest"
            ):
                publisher.run()

    def test_incomplete_groups_and_source_sidecars_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )

            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE phase_progress SET status='running' "
                    "WHERE subphase='selection.groups'"
                )
                connection.commit()
            finally:
                connection.close()
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            for row in checkpoint_payload["subphases"]:
                if row["subphase"] == "selection.groups":
                    row["status"] = "running"
            checkpoint.write_text(
                json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "group assignment is incomplete"
            ):
                self._publisher(
                    fixture, database, checkpoint, root / "incomplete-output"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            Path(f"{database}-wal").write_bytes(b"live")
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "SQLite sidecars"
            ):
                self._publisher(
                    fixture, database, checkpoint, root / "sidecar-output"
                )

    def test_group_assignment_and_bucket_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            connection = sqlite3.connect(database)
            try:
                group_id, split = connection.execute(
                    "SELECT group_id, split FROM groups ORDER BY group_id LIMIT 1"
                ).fetchone()
                replacement = "train" if split != "train" else "test"
                connection.execute(
                    "UPDATE groups SET split=? WHERE group_id=?",
                    (replacement, group_id),
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "invalid leakage-safe split assignments"
            ):
                self._publisher(
                    fixture, database, checkpoint, root / "bad-split-output"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            connection = sqlite3.connect(database)
            try:
                doc_id = connection.execute(
                    "SELECT doc_id FROM documents ORDER BY doc_id LIMIT 1"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE documents SET bucket='unexpected' WHERE doc_id=?",
                    (doc_id,),
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "not the frozen four buckets"
            ):
                self._publisher(
                    fixture, database, checkpoint, root / "bad-bucket-output"
                )

    def test_checkpoint_and_committed_shard_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "publication"
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                publisher.run(max_new_archives=1)

            publication_checkpoint = (
                output / ".work" / "PUBLICATION_CHECKPOINT.json"
            )
            value = json.loads(publication_checkpoint.read_text(encoding="utf-8"))
            value["selected_totals"][0]["selected_tokens"] += 1
            publication_checkpoint.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher, self.assertRaisesRegex(
                AllEligiblePublicationError, "another authority"
            ):
                publisher.run()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "completed-publication"
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                result = publisher.run()
            shard = output / result["manifest"]["decision_shards"][0]["path"]
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher, self.assertRaisesRegex(
                AllEligiblePublicationError, "decision shard is corrupt"
            ):
                publisher.run()

    def test_bitmap_padding_and_authenticated_header_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "publication"
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                publisher.run()
            checkpoint_path = output / ".work" / "PUBLICATION_CHECKPOINT.json"
            authority = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            descriptor = next(
                row
                for row in authority["completed_shards"]
                if int(row["records"]) % 8
            )
            shard = output / descriptor["path"]
            payload = bytearray(shard.read_bytes())
            payload[-1] |= 0x80
            shard.write_bytes(payload)
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            checkpoint_path.write_text(
                json.dumps(authority, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher, self.assertRaisesRegex(
                AllEligiblePublicationError, "bitmap authority"
            ):
                publisher.run()

    def test_bound_artifact_mutation_poisons_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "publication"
            publisher = self._publisher(fixture, database, checkpoint, output)
            report_path = Path(publisher.reports[0]["report_path"])
            original = report_path.read_bytes()
            report_path.write_bytes(original + b"\n")
            try:
                with publisher, self.assertRaisesRegex(
                    AllEligiblePublicationError, "Bound source artifact changed"
                ):
                    publisher.run()
            finally:
                report_path.write_bytes(original)
                publisher.close()
            poison = json.loads(
                (output / ".work" / "POISONED.json").read_text(encoding="utf-8")
            )
            self.assertEqual(poison["format"], "all-eligible-publication-poison")
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "generation is poisoned"
            ):
                self._publisher(fixture, database, checkpoint, output)

    def test_committed_source_payload_mutation_poisons_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            output = root / "publication"
            with self._publisher(
                fixture, database, checkpoint, output
            ) as publisher:
                raw_archive = Path(publisher.publication_reports[0]["archive_path"])
                publisher.run(max_new_archives=1)
            original = raw_archive.read_bytes()
            self.assertGreater(len(original), 1)
            raw_archive.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            try:
                with self._publisher(
                    fixture, database, checkpoint, output
                ) as publisher, self.assertRaisesRegex(
                    AllEligiblePublicationError, "Source payload changed"
                ):
                    publisher.run()
            finally:
                raw_archive.write_bytes(original)
            self.assertTrue((output / ".work" / "POISONED.json").is_file())
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "generation is poisoned"
            ):
                self._publisher(fixture, database, checkpoint, output)

    def test_production_query_plans_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            with self._publisher(
                fixture, database, checkpoint, root / "publication"
            ) as publisher:
                self.assertEqual(
                    sum(
                        detail in ("SCAN d", "SCAN TABLE documents AS d")
                        for detail in publisher.eligible_authority_query_plan
                    ),
                    1,
                )
                self.assertTrue(
                    any(
                        "COVERING INDEX sqlite_autoindex_documents_2" in detail
                        for detail in publisher.archive_bitmap_query_plan
                    )
                )
                self.assertTrue(
                    any(
                        "COVERING INDEX reasons_reason" in detail
                        for detail in publisher.rejection_inventory_query_plan
                    )
                )

    def test_production_constructor_requires_snapshot_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "snapshot manifest is required"
            ):
                AllEligiblePublisher(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    source_db=database,
                    source_checkpoint=checkpoint,
                    source_snapshot_manifest=None,
                    output=root / "output",
                    policy_path=FAST_CANONICAL_POLICY,
                    quota_path=fixture.quota_path,
                    benchmark_denylist_path=(
                        PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                    ),
                )

    def test_minimal_or_retired_snapshot_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            database, checkpoint, snapshot_manifest = self._snapshot_manifest(
                database, checkpoint
            )
            manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
            manifest.pop("database_state")
            snapshot_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest()
            snapshot_manifest.with_name("manifest.json.sha256").write_text(
                f"{digest}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "exact v1 schema"
            ):
                AllEligiblePublisher(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    source_db=database,
                    source_checkpoint=checkpoint,
                    source_snapshot_manifest=snapshot_manifest,
                    output=root / "invalid-output",
                    policy_path=FAST_CANONICAL_POLICY,
                    quota_path=fixture.quota_path,
                    benchmark_denylist_path=(
                        PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                    ),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            database, checkpoint, snapshot_manifest = self._snapshot_manifest(
                database, checkpoint
            )
            (snapshot_manifest.parent / "retirement.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "has been retired"
            ):
                AllEligiblePublisher(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    source_db=database,
                    source_checkpoint=checkpoint,
                    source_snapshot_manifest=snapshot_manifest,
                    output=root / "retired-output",
                    policy_path=FAST_CANONICAL_POLICY,
                    quota_path=fixture.quota_path,
                    benchmark_denylist_path=(
                        PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                    ),
                )

    def test_snapshot_bound_publication_is_production_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CorpusFixture(root / "dataset")
            database, checkpoint = self._partial_source(
                fixture, root / "exact-generation"
            )
            database, checkpoint, snapshot_manifest = self._snapshot_manifest(
                database, checkpoint
            )
            with AllEligiblePublisher(
                root=fixture.root,
                staging_root=fixture.staging,
                source_db=database,
                source_checkpoint=checkpoint,
                source_snapshot_manifest=snapshot_manifest,
                output=root / "production-output",
                policy_path=FAST_CANONICAL_POLICY,
                quota_path=fixture.quota_path,
                benchmark_denylist_path=(
                    PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                ),
            ) as publisher:
                result = publisher.run()
            self.assertTrue(result["manifest"]["production_ready"])
            self.assertEqual(
                result["manifest"]["publication_scope"],
                "production-durable-snapshot",
            )
            self.assertIsNotNone(
                result["manifest"]["identity"]["source_curation"]["snapshot"]
            )
            forged = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
            forged["identity_sha256"] = "0" * 64
            snapshot_manifest.write_text(
                json.dumps(forged, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            forged_sha = hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest()
            (snapshot_manifest.parent / "manifest.json.sha256").write_text(
                f"{forged_sha}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                AllEligiblePublicationError, "identity/runtime contract mismatch"
            ):
                AllEligiblePublisher(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    source_db=database,
                    source_checkpoint=checkpoint,
                    source_snapshot_manifest=snapshot_manifest,
                    output=root / "forged-output",
                    policy_path=FAST_CANONICAL_POLICY,
                    quota_path=fixture.quota_path,
                    benchmark_denylist_path=(
                        PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
