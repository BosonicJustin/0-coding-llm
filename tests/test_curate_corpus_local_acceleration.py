from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import curate_corpus as curate_corpus_module  # noqa: E402
from curate_corpus import (  # noqa: E402
    CURATION_OBSERVED_V1_DATABASE_BYTES,
    CURATION_OBSERVED_V1_DOCUMENTS,
    CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT,
    CURATION_STORAGE_PROJECTION_BASIS,
    FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
    CurationBuilder,
    CurationError,
    LOCAL_WAL_AUTOCHECKPOINT_PAGES,
)
from curation_policy import FAST_CANONICAL_POLICY  # noqa: E402
from publish_all_eligible_selection import AllEligiblePublisher  # noqa: E402
from tests.test_curate_corpus import CorpusFixture  # noqa: E402


def decision_bytes(output: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(output)): path.read_bytes()
        for path in sorted((output / "decisions").rglob("*.jsonl.zst"))
    }


def _crash_after_first_local_commit(
    root: str,
    staging: str,
    output: str,
    local: str,
    quota_path: str,
) -> None:
    builder = CurationBuilder(
        root=Path(root),
        staging_root=Path(staging),
        output=Path(output),
        policy_path=FAST_CANONICAL_POLICY,
        quota_path=Path(quota_path),
        denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
        english_near_clusters=None,
        allow_missing_english_near_dedup=False,
        batch_size=3,
        sqlite_journal_mode="delete",
        sqlite_local_work_root=Path(local),
        sqlite_snapshot_interval_seconds=3_600,
        sqlite_snapshot_retention=2,
        defer_raw_archive_integrity_until_finalize=True,
        allow_same_device_local_store_for_testing=True,
    )
    with builder as active:
        def terminate(_subphase: str, _progress: dict[str, object]) -> None:
            os._exit(91)

        active._after_bounded_commit = terminate  # type: ignore[method-assign]
        active.run()


class LocalAccelerationIntegrationTests(unittest.TestCase):
    def builder(
        self,
        fixture: CorpusFixture,
        output: Path,
        local: Path,
        *,
        defer_raw: bool = True,
        all_eligible_handoff: bool = False,
    ) -> CurationBuilder:
        return CurationBuilder(
            root=fixture.root,
            staging_root=fixture.staging,
            output=output,
            policy_path=FAST_CANONICAL_POLICY,
            quota_path=fixture.quota_path,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            english_near_clusters=None,
            allow_missing_english_near_dedup=False,
            batch_size=3,
            sqlite_journal_mode="delete",
            sqlite_local_work_root=local,
            sqlite_snapshot_interval_seconds=3_600,
            sqlite_snapshot_retention=2,
            defer_raw_archive_integrity_until_finalize=defer_raw,
            allow_same_device_local_store_for_testing=True,
            fast_all_eligible_handoff=all_eligible_handoff,
        )

    def test_measured_storage_contract_binds_v1_production_peak(self) -> None:
        self.assertEqual(CURATION_OBSERVED_V1_DOCUMENTS, 51_328_930)
        self.assertEqual(CURATION_OBSERVED_V1_DATABASE_BYTES, 67_824_914_432)
        self.assertEqual(CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT, 1_322)
        self.assertEqual(
            CURATION_STORAGE_PROJECTION_BASIS["observed_maximum_wal_bytes"],
            4_132_940_952,
        )
        projected_database = (
            CURATION_OBSERVED_V1_DOCUMENTS
            * CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT
            * 2
        )
        self.assertEqual(projected_database, 135_713_690_920)

    def test_all_eligible_handoff_skips_quotas_and_periodic_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            output = Path(temporary) / "all-eligible-source"
            local = Path(temporary) / "pod-local"
            with self.builder(
                fixture,
                output,
                local,
                all_eligible_handoff=True,
            ) as builder:
                self.assertEqual(
                    builder.identity["fast_all_eligible_handoff"],
                    FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
                )
                self.assertIsNone(
                    builder._snapshot_local_state(
                        reason="synthetic-periodic-check", force=False
                    )
                )
                self.assertEqual(
                    list(
                        (output / ".work" / "sqlite-snapshots-v1").glob(
                            "snapshot-*"
                        )
                    ),
                    [],
                )
                result = builder.run()
                self.assertTrue(result["complete"])
                self.assertTrue(result["ready_for_all_eligible_publication"])
                self.assertEqual(result["phase"], "canonicalized")
                self.assertEqual(
                    result["execution_profile"],
                    FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
                )
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM selected").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM output_archives"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM phase_progress "
                        "WHERE subphase LIKE 'selection.quota.%'"
                    ).fetchone()[0],
                    0,
                )
                self.assertIsNone(
                    builder.db.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type='index' AND name='documents_selection_v2'"
                    ).fetchone()
                )
                self.assertEqual(
                    builder.db.execute(
                        "SELECT status FROM phase_progress "
                        "WHERE subphase='selection.groups'"
                    ).fetchone()[0],
                    "complete",
                )

            generations = sorted(
                (output / ".work" / "sqlite-snapshots-v1").glob("snapshot-*")
            )
            self.assertEqual(len(generations), 1)
            self.assertFalse((output / ".work" / "curation.sqlite3").exists())
            snapshot = result["source_snapshot"]
            self.assertEqual(
                Path(snapshot["manifest_path"]).resolve(),
                (generations[0] / "manifest.json").resolve(),
            )
            self.assertEqual(
                Path(snapshot["database"]["path"]).resolve(),
                (generations[0] / "curation.sqlite3").resolve(),
            )
            checkpoint = json.loads(
                Path(snapshot["checkpoint"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["phase"], "canonicalized")
            self.assertEqual(checkpoint["counts"]["selected_documents"], 0)
            self.assertEqual(checkpoint["counts"]["output_archives"], 0)
            self.assertTrue(
                any(
                    row["subphase"] == "selection.groups"
                    and row["status"] == "complete"
                    for row in checkpoint["subphases"]
                )
            )
            publication = Path(temporary) / "all-eligible-publication"
            with AllEligiblePublisher(
                root=fixture.root,
                staging_root=fixture.staging,
                source_db=Path(snapshot["database"]["path"]),
                source_checkpoint=Path(snapshot["checkpoint"]["path"]),
                source_snapshot_manifest=Path(snapshot["manifest_path"]),
                output=publication,
                policy_path=FAST_CANONICAL_POLICY,
                quota_path=fixture.quota_path,
                benchmark_denylist_path=(
                    PROJECT_ROOT / "configs" / "mbpp_denylist.json"
                ),
            ) as publisher:
                published = publisher.run()
            self.assertTrue(published["complete"])
            self.assertEqual(
                published["manifest"]["identity"]["fast_all_eligible_handoff"],
                FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
            )

    def test_all_eligible_handoff_resumes_recovery_snapshot_without_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            output = Path(temporary) / "all-eligible-source"
            local = Path(temporary) / "pod-local"
            with self.builder(
                fixture,
                output,
                local,
                all_eligible_handoff=True,
            ) as builder:
                partial = builder.run(max_new_archives=1)
                self.assertFalse(partial["complete"])
            self.assertFalse((output / ".work" / "curation.sqlite3").exists())
            first_generations = sorted(
                (output / ".work" / "sqlite-snapshots-v1").glob("snapshot-*")
            )
            self.assertEqual(len(first_generations), 1)

            # Simulate a pod replacement: the local WAL store is ephemeral, while
            # the authenticated recovery generation on the durable volume remains.
            shutil.rmtree(local)

            with self.builder(
                fixture,
                output,
                local,
                all_eligible_handoff=True,
            ) as builder:
                self.assertEqual(
                    builder.local_store.last_prepare_evidence["source"],
                    "durable-snapshot",
                )
                completed = builder.run()
                self.assertTrue(completed["ready_for_all_eligible_publication"])
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM selected").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM phase_progress "
                        "WHERE subphase LIKE 'selection.quota.%'"
                    ).fetchone()[0],
                    0,
                )
            generations = sorted(
                (output / ".work" / "sqlite-snapshots-v1").glob("snapshot-*")
            )
            self.assertEqual(len(generations), 2)

    def test_local_wal_run_is_selection_equivalent_and_publishes_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            baseline = Path(temporary) / "baseline"
            accelerated = Path(temporary) / "accelerated"
            local = Path(temporary) / "pod-local"
            fixture.build_fast(baseline)

            with self.builder(fixture, accelerated, local) as builder:
                self.assertEqual(builder.db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                self.assertEqual(
                    builder.db.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
                    LOCAL_WAL_AUTOCHECKPOINT_PAGES,
                )
                self.assertEqual(
                    builder.db.execute("PRAGMA locking_mode").fetchone()[0],
                    "exclusive",
                )
                result = builder.run()
                self.assertTrue(result["complete"])

            self.assertEqual(decision_bytes(accelerated), decision_bytes(baseline))
            self.assertTrue((accelerated / ".work" / "curation.sqlite3").is_file())
            self.assertFalse((accelerated / ".work" / "CHECKPOINT.json").exists())
            generations = sorted(
                (accelerated / ".work" / "sqlite-snapshots-v1").glob("snapshot-*")
            )
            self.assertEqual(len(generations), 1)
            snapshot_manifest = json.loads(
                (generations[0] / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(snapshot_manifest["authority_artifacts"]),
                ["CHECKPOINT.json", "journal.jsonl"],
            )
            checkpoint = json.loads(
                (generations[0] / "CHECKPOINT.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["phase"], "complete")
            self.assertEqual(
                checkpoint["identity"]["raw_archive_integrity_policy"],
                "deferred-full-sha256-mandatory-before-publication",
            )

    def test_deferred_raw_hash_is_skipped_at_start_but_mandatory_at_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            output = Path(temporary) / "accelerated"
            local = Path(temporary) / "pod-local"
            original_sha = curate_corpus_module.file_sha256
            raw_hashes: list[Path] = []

            def observe(path: Path, *args: object, **kwargs: object) -> str:
                if str(path).endswith(".tar.zst"):
                    raw_hashes.append(Path(path))
                return original_sha(path, *args, **kwargs)

            with mock.patch.object(
                curate_corpus_module, "file_sha256", side_effect=observe
            ):
                with self.builder(fixture, output, local) as builder:
                    partial = builder.run(max_new_archives=1)
                    self.assertFalse(partial["complete"])
            self.assertEqual(raw_hashes, [])

            raw = next((fixture.root / "raw").rglob("*.tar.zst"))
            payload = bytearray(raw.read_bytes())
            payload[len(payload) // 2] ^= 1
            raw.write_bytes(payload)
            with self.assertRaisesRegex(CurationError, "raw archive checksum mismatch"):
                with self.builder(fixture, output, local) as builder:
                    builder.run()

    def test_real_process_death_recovers_local_wal_and_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            baseline = Path(temporary) / "baseline"
            accelerated = Path(temporary) / "accelerated"
            local = Path(temporary) / "pod-local"
            fixture.build_fast(baseline)
            process = multiprocessing.get_context("spawn").Process(
                target=_crash_after_first_local_commit,
                args=(
                    str(fixture.root),
                    str(fixture.staging),
                    str(accelerated),
                    str(local),
                    str(fixture.quota_path),
                ),
            )
            process.start()
            process.join(20)
            self.assertEqual(process.exitcode, 91)
            self.assertTrue((local / "curation.sqlite3").is_file())

            with self.builder(fixture, accelerated, local) as builder:
                result = builder.run()
                self.assertTrue(result["complete"])

            self.assertEqual(decision_bytes(accelerated), decision_bytes(baseline))
            snapshots = sorted(
                (accelerated / ".work" / "sqlite-snapshots-v1").glob(
                    "snapshot-*"
                )
            )
            self.assertEqual(len(snapshots), 1)

    def test_exit_snapshot_failure_does_not_mask_original_error(self) -> None:
        class OriginalFailure(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            builder = self.builder(
                fixture,
                Path(temporary) / "accelerated",
                Path(temporary) / "pod-local",
            )
            with mock.patch.object(
                builder,
                "_snapshot_local_state",
                side_effect=OSError("synthetic snapshot failure"),
            ):
                with self.assertRaisesRegex(
                    OriginalFailure, "original curation failure"
                ) as caught:
                    with builder:
                        raise OriginalFailure("original curation failure")
            self.assertTrue(
                any(
                    "snapshot/promotion also failed" in note
                    for note in caught.exception.__notes__
                )
            )

    def test_local_mode_refuses_to_convert_or_overwrite_baseline_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            baseline = Path(temporary) / "baseline"
            local = Path(temporary) / "pod-local"
            fixture.build_fast(baseline)
            canonical = baseline / ".work" / "curation.sqlite3"
            before = canonical.read_bytes()

            with self.assertRaisesRegex(
                CurationError, "conversion or overwrite|baseline curation"
            ):
                with self.builder(fixture, baseline, local):
                    pass

            self.assertEqual(canonical.read_bytes(), before)
            self.assertFalse(local.exists())

    def test_production_local_mode_requires_distinct_durable_network_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary) / "fixture")
            builder = self.builder(
                fixture,
                Path(temporary) / "accelerated",
                Path(temporary) / "pod-local",
            )
            builder.allow_same_device_local_store_for_testing = False
            with self.assertRaisesRegex(
                CurationError, "durable network filesystem"
            ):
                with builder:
                    pass


if __name__ == "__main__":
    unittest.main()
