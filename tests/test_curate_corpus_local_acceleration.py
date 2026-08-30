from __future__ import annotations

import json
import multiprocessing
import os
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
    CurationBuilder,
    CurationError,
    LOCAL_WAL_AUTOCHECKPOINT_PAGES,
)
from curation_policy import FAST_CANONICAL_POLICY  # noqa: E402
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
        )

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
