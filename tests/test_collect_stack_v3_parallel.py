from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_stack_v3 as legacy
from collect_stack_v3_parallel import (
    assigned_source_shard,
    commit_worker_checkpoint,
    create_or_load_plan,
    recover_worker_checkpoints,
    worker_archive_index,
)
from quota_tracker import iter_records


def add_test_document(writer: legacy.RawArchiveWriter, marker: str) -> None:
    writer.add(
        f"def {marker}():\n    return 1\n".encode(),
        8,
        {"repo_path": "owner/repo", "repo_id": 123, "commit_id": "abc"},
        {
            "file_path": f"src/{marker}.py",
            "content_id": marker,
            "language": "Python",
            "license_type": "permissive",
            "detected_licenses": ["MIT"],
        },
    )


class ParallelCollectorTest(unittest.TestCase):
    def test_other_code_topup_profile_changes_only_collection_target(self) -> None:
        base = json.loads(
            (PROJECT_ROOT / "configs" / "data_quotas.json").read_text(
                encoding="utf-8"
            )
        )
        topup = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "data_quotas_other_code_topup_v2.json"
            ).read_text(encoding="utf-8")
        )

        def targets(config: dict) -> dict[str, int]:
            return {row["name"]: int(row["target"]) for row in config["quotas"]}

        base_targets = targets(base)
        topup_targets = targets(topup)
        self.assertEqual(base_targets.keys(), topup_targets.keys())
        changed = {
            name
            for name in base_targets
            if base_targets[name] != topup_targets[name]
        }
        self.assertEqual(changed, {"collection/other_code"})
        self.assertEqual(topup_targets["collection/other_code"], 35_000_000_000)

    def test_download_launcher_forwards_versioned_quota_config(self) -> None:
        launcher = (PROJECT_ROOT / "scripts" / "run_download.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('QUOTA_CONFIG="${QUOTA_CONFIG:-', launcher)
        self.assertIn('--quota-config "$QUOTA_CONFIG"', launcher)

        preprocessor = (PROJECT_ROOT / "scripts" / "run_preprocess.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('QUOTA_CONFIG="${QUOTA_CONFIG:-', preprocessor)
        self.assertEqual(preprocessor.count('--quotas "$QUOTA_CONFIG"'), 3)

    def test_assignments_and_archive_indices_do_not_overlap(self) -> None:
        plan = {
            "frontier_source_shard_index": 41,
            "worker_count": 8,
            "base_archive_indices": {"python": 24, "other_code": 27},
        }
        assignments = {
            worker: [assigned_source_shard(plan, worker, position) for position in range(20)]
            for worker in range(8)
        }
        flattened = [value for rows in assignments.values() for value in rows]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(range(41, 41 + len(flattened))))

        for category in ("python", "other_code"):
            indexes = {
                worker_archive_index(plan, category, worker, sequence)
                for worker in range(8)
                for sequence in range(20)
            }
            self.assertEqual(len(indexes), 160)

    def test_legacy_cursor_migrates_and_plan_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            source = "dataset@" + "a" * 40
            writer = legacy.RawArchiveWriter(root, "python", 0, 1, 1)
            add_test_document(writer, "legacy")
            legacy.commit_checkpoint(
                root,
                {"python": writer},
                repos_consumed=1234,
                source_shard_index=17,
                row_offset=89,
                sequence=1,
                dataset_revision="a" * 40,
                source=source,
                benchmark_guard_sha256="guard",
            )
            args = argparse.Namespace(root=root, workers=4)
            shards = [f"data/{index:05d}.parquet" for index in range(100)]
            plan = create_or_load_plan(args, source, "a" * 40, shards, "guard")
            self.assertEqual(plan["frontier_source_shard_index"], 17)
            self.assertEqual(plan["frontier_row_offset"], 89)
            self.assertEqual(plan["base_archive_indices"]["python"], 1)
            self.assertEqual(create_or_load_plan(args, source, "a" * 40, shards, "guard"), plan)

    def test_worker_checkpoint_finalizes_once_and_resumes_exact_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            plan = {
                "plan_version": 1,
                "source": "dataset@" + "b" * 40,
                "dataset_revision": "b" * 40,
                "source_shards_sha256": "digest",
                "source_shard_count": 100,
                "benchmark_guard_sha256": "guard",
                "worker_count": 4,
                "frontier_source_shard_index": 12,
                "frontier_row_offset": 37,
                "base_archive_indices": {"python": 5, "other_code": 7},
                "legacy_cursor": {
                    "sequence": 3,
                    "repos_consumed": 500,
                    "source_shard_index": 12,
                    "row_offset": 37,
                },
            }
            worker_id = 2
            index = worker_archive_index(plan, "python", worker_id, 0)
            writer = legacy.RawArchiveWriter(root, "python", index, 1, 1)
            add_test_document(writer, "parallel")
            commit_worker_checkpoint(
                root,
                plan,
                worker_id,
                {"python": writer},
                sequence=1,
                assignment_position=3,
                row_offset=44,
                repos_consumed=900,
                archive_sequences={"python": 1, "other_code": 0},
            )
            cursor = recover_worker_checkpoints(root, plan, worker_id)
            self.assertEqual(cursor["assignment_position"], 3)
            self.assertEqual(cursor["row_offset"], 44)
            self.assertEqual(cursor["repos_consumed"], 900)
            self.assertTrue((root / "raw" / "python" / f"part-{index:06d}.tar.zst").is_file())
            self.assertEqual(len(list(iter_records(root))), 1)
            self.assertEqual(recover_worker_checkpoints(root, plan, worker_id), cursor)
            self.assertEqual(len(list(iter_records(root))), 1)

    def test_cloned_root_resume_does_not_depend_on_absolute_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source_root = parent / "dataset-v1"
            clone_root = parent / "dataset-v2"
            plan = {
                "plan_version": 1,
                "source": "dataset@" + "c" * 40,
                "dataset_revision": "c" * 40,
                "source_shards_sha256": "digest",
                "source_shard_count": 100,
                "benchmark_guard_sha256": "guard",
                "worker_count": 4,
                "frontier_source_shard_index": 12,
                "frontier_row_offset": 0,
                "base_archive_indices": {"python": 5, "other_code": 7},
                "legacy_cursor": {
                    "sequence": 0,
                    "repos_consumed": 0,
                    "source_shard_index": 12,
                    "row_offset": 0,
                },
            }
            worker_id = 1
            index = worker_archive_index(plan, "python", worker_id, 0)
            writer = legacy.RawArchiveWriter(source_root, "python", index, 1, 1)
            add_test_document(writer, "clone_resume")
            commit_worker_checkpoint(
                source_root,
                plan,
                worker_id,
                {"python": writer},
                sequence=1,
                assignment_position=2,
                row_offset=17,
                repos_consumed=44,
                archive_sequences={"python": 1, "other_code": 0},
            )

            source_archive = (
                source_root / "raw" / "python" / f"part-{index:06d}.tar.zst"
            )
            clone_archive = (
                clone_root / "raw" / "python" / f"part-{index:06d}.tar.zst"
            )
            clone_archive.parent.mkdir(parents=True)
            os.link(source_archive, clone_archive)
            shutil.copytree(
                source_root / "state" / "collector_parallel",
                clone_root / "state" / "collector_parallel",
                copy_function=os.link,
            )

            # Make every absolute descriptor in the copied checkpoint dead.
            retired_source = parent / "dataset-v1-retired"
            source_root.rename(retired_source)
            cursor = recover_worker_checkpoints(clone_root, plan, worker_id)
            self.assertEqual(cursor["assignment_position"], 2)
            self.assertEqual(cursor["row_offset"], 17)
            self.assertTrue(clone_archive.is_file())
            records = list(iter_records(clone_root))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["category"], "python")

    def test_missing_cloned_final_never_recovers_pending_from_old_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            old_root = parent / "dataset-v1"
            clone_root = parent / "dataset-v2"
            (old_root / "raw/python").mkdir(parents=True)
            (clone_root / "raw/python").mkdir(parents=True)
            pending = old_root / "raw/python" / f".part-000000-{'0' * 32}.tar.zst"
            pending.write_bytes(b"pending")
            archive = {
                "category": "python",
                "index": 0,
                "pending_path": str(pending),
                "final_path": str(old_root / "raw/python/part-000000.tar.zst"),
            }
            with self.assertRaisesRegex(RuntimeError, "different root"):
                legacy.recover_checkpoint_archive(
                    clone_root,
                    archive,
                    checkpoint_path=clone_root / "checkpoint.json",
                )
            self.assertTrue(pending.is_file())
            self.assertFalse((clone_root / "raw/python/part-000000.tar.zst").exists())

    def test_checkpoint_archive_descriptor_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            final = root / "raw/python/part-000000.tar.zst"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"final")
            archive = {
                "category": "python",
                "index": 0,
                "pending_path": str(
                    root / "raw/python" / f".part-000000-{'0' * 32}.tar.zst"
                ),
                "final_path": f"{root}/raw/python/../python/part-000000.tar.zst",
            }
            with self.assertRaisesRegex(RuntimeError, "unsafe final_path"):
                legacy.recover_checkpoint_archive(
                    root,
                    archive,
                    checkpoint_path=root / "checkpoint.json",
                )


if __name__ == "__main__":
    unittest.main()
