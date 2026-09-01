from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import clone_dataset_generation as clone_module
from clone_dataset_generation import (
    COPIED_CONTROL_FILES,
    HARDLINK_TREES,
    CloneError,
    clone_generation,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GenerationFixture:
    def __init__(self, parent: Path) -> None:
        self.source = parent / "dataset-v1"
        self.destination = parent / "dataset-v2"
        self.source.mkdir()

        linked_payloads = {
            "raw/python/part-000000.tar.zst": b"python raw\n",
            "raw/other_code/part-000000.tar.zst": b"other raw\n",
            "raw/english/fineweb_edu/part-000000.tar.zst": b"fineweb raw\n",
            "raw/english/wikipedia/part-000000.tar.zst": b"wikipedia raw\n",
            "tokenizer/starcoder2/tokenizer.json": b'{"vocab":{}}\n',
            "tokenizer/starcoder2/TOKENIZER_MANIFEST.json": b'{"manifest_version":1}\n',
            "manifests/STACK_V3_SOURCE.json": b'{"source":"stack"}\n',
            "manifests/FINEWEB_EDU_SOURCE.json": b'{"source":"fineweb"}\n',
            "manifests/WIKIPEDIA_SOURCE.json": b'{"source":"wikipedia"}\n',
            "state/collector_checkpoints/checkpoint-00000001.json": b'{"sequence":1}\n',
            "state/collector_parallel/PLAN.json": b'{"plan_version":1}\n',
            "state/collector_parallel/worker-00/checkpoints/checkpoint-00000001.json": b'{"worker_id":0}\n',
            "state/quota_records/collection/record.json": b'{"record_version":1}\n',
            "staging/preprocess/reports/python/part-000000.json": b'{"report_version":1}\n',
            "staging/preprocess/fingerprints/python/part-000000.jsonl.zst": b"fingerprint\n",
        }
        for relative, payload in linked_payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        for relative in HARDLINK_TREES:
            (self.source / relative).mkdir(parents=True, exist_ok=True)

        copied_payloads = {
            "state/COLLECTION_COMPLETE.json": b'{"other_code_tokens":25952231562}\n',
            "state/ENGLISH_FINEWEB_EDU_COMPLETE.json": b'{"complete":true}\n',
            "state/ENGLISH_WIKIPEDIA_COMPLETE.json": b'{"complete":true}\n',
            "staging/preprocess/PREPROCESS_MANIFEST.json": b'{"manifest_version":1}\n',
            "staging/preprocess/STATUS.json": b'{"raw_audit_complete":true}\n',
        }
        for relative, payload in copied_payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        excluded_payloads = {
            "logs/collector.log": b"do not clone\n",
            "audits/readiness.json": b"do not clone\n",
            "curated/selection-fast-v1/manifest.json": b"do not clone\n",
            "final/packed-v1/manifest.json": b"do not clone\n",
            "staging/preprocess/.preprocess.lock": b"",
            "staging/preprocess/.tmp/pending": b"do not clone\n",
            "staging/preprocess/dedup/dedup.sqlite3": b"sqlite\n",
        }
        for relative, payload in excluded_payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    def source_hashes(self) -> dict[str, str]:
        return {
            str(path.relative_to(self.source)): sha256(path)
            for path in sorted(self.source.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }


class CloneDatasetGenerationTest(unittest.TestCase):
    def test_selective_clone_links_immutable_and_copies_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            before = fixture.source_hashes()

            result = clone_generation(fixture.source, fixture.destination)

            self.assertTrue(result["complete"])
            self.assertEqual(result["target_other_code_raw_tokens"], 35_000_000_000)
            self.assertEqual(before, fixture.source_hashes())
            self.assertTrue(fixture.destination.is_dir())

            for relative in HARDLINK_TREES:
                source_tree = fixture.source / relative
                for source_file in source_tree.rglob("*"):
                    if not source_file.is_file():
                        continue
                    destination_file = fixture.destination / source_file.relative_to(
                        fixture.source
                    )
                    self.assertTrue(destination_file.is_file())
                    self.assertEqual(
                        (source_file.stat().st_dev, source_file.stat().st_ino),
                        (destination_file.stat().st_dev, destination_file.stat().st_ino),
                    )

            for relative in COPIED_CONTROL_FILES:
                source_file = fixture.source / relative
                destination_file = fixture.destination / relative
                self.assertNotEqual(
                    (source_file.stat().st_dev, source_file.stat().st_ino),
                    (destination_file.stat().st_dev, destination_file.stat().st_ino),
                )
                self.assertEqual(sha256(source_file), sha256(destination_file))

            for relative in (
                "logs",
                "audits",
                "curated",
                "final",
                "staging/preprocess/.preprocess.lock",
                "staging/preprocess/.tmp",
                "staging/preprocess/dedup",
            ):
                self.assertFalse((fixture.destination / relative).exists(), relative)

            manifest_path = fixture.destination / "CLONE_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(
                manifest["other_code_topup_plan"]["target_cumulative_raw_tokens"],
                35_000_000_000,
            )
            self.assertEqual(
                manifest["other_code_topup_plan"]["train_shortfall_tokens"],
                4_504_576_297,
            )
            self.assertEqual(
                manifest["inventory"]["copied_control_files"],
                len(COPIED_CONTROL_FILES),
            )
            sidecar = (fixture.destination / "CLONE_MANIFEST.sha256").read_text(
                encoding="ascii"
            )
            self.assertEqual(sidecar, f"{sha256(manifest_path)}  CLONE_MANIFEST.json\n")

    def test_existing_destination_is_rejected_without_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            fixture.destination.mkdir()
            sentinel = fixture.destination / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(CloneError, "must not already exist"):
                clone_generation(fixture.source, fixture.destination)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_symlink_in_included_tree_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            target = fixture.source / "outside"
            target.write_bytes(b"outside")
            (fixture.source / "raw/python/unsafe-link").symlink_to(target)

            with self.assertRaisesRegex(CloneError, "symlink"):
                clone_generation(fixture.source, fixture.destination)

            self.assertFalse(fixture.destination.exists())

    def test_symlink_control_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            control = fixture.source / "staging/preprocess/STATUS.json"
            control.unlink()
            control.symlink_to(fixture.source / "state/COLLECTION_COMPLETE.json")

            with self.assertRaisesRegex(CloneError, "symlink"):
                clone_generation(fixture.source, fixture.destination)

            self.assertFalse(fixture.destination.exists())

    def test_assembly_failure_removes_incoming_and_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            with mock.patch.object(
                clone_module,
                "_copy_file",
                side_effect=OSError("injected copy failure"),
            ), self.assertRaisesRegex(OSError, "injected copy failure"):
                clone_generation(fixture.source, fixture.destination)

            self.assertFalse(fixture.destination.exists())
            self.assertEqual(
                list(Path(temporary).glob(".dataset-v2.incoming-*")),
                [],
            )

    def test_source_change_during_clone_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GenerationFixture(Path(temporary))
            original_verify = clone_module._verify_copied

            def mutate_then_verify(*args: object, **kwargs: object) -> list[dict[str, object]]:
                result = original_verify(*args, **kwargs)
                source = fixture.source / "raw/python/part-000000.tar.zst"
                source.write_bytes(source.read_bytes() + b"changed")
                return result

            with mock.patch.object(
                clone_module,
                "_verify_copied",
                side_effect=mutate_then_verify,
            ), self.assertRaisesRegex(CloneError, "changed while"):
                clone_generation(fixture.source, fixture.destination)

            self.assertFalse(fixture.destination.exists())

    def test_topup_quota_config_changes_only_other_collection_target(self) -> None:
        original = json.loads(
            (PROJECT_ROOT / "configs/data_quotas.json").read_text(encoding="utf-8")
        )
        topup = json.loads(
            (PROJECT_ROOT / "configs/data_quotas_other_code_topup_v2.json").read_text(
                encoding="utf-8"
            )
        )
        original_rows = {row["name"]: row for row in original["quotas"]}
        topup_rows = {row["name"]: row for row in topup["quotas"]}
        self.assertEqual(set(original_rows), set(topup_rows))
        for name in sorted(original_rows):
            expected = dict(original_rows[name])
            if name == "collection/other_code":
                expected["target"] = 35_000_000_000
            self.assertEqual(topup_rows[name], expected)


if __name__ == "__main__":
    unittest.main()
