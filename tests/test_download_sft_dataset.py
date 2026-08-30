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

import download_sft_dataset as downloader


TEST_REVISION = "a" * 40


def tiny_contract(*, expected_bytes: int, expected_rows: int) -> downloader.DatasetContract:
    return downloader.DatasetContract(
        repo_id=downloader.DEFAULT_REPO_ID,
        revision=TEST_REVISION,
        repo_type="dataset",
        expected_files=2,
        expected_bytes=expected_bytes,
        expected_rows=expected_rows,
    )


def write_tiny_snapshot(raw: Path, payloads: tuple[bytes, bytes]) -> list[Path]:
    data = raw / "data"
    data.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, payload in enumerate(payloads):
        path = data / f"train-{index:05d}-of-00002.parquet"
        path.write_bytes(payload)
        paths.append(path)
    (raw / "README.md").write_text("# fixture\n", encoding="utf-8")
    (raw / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")
    return paths


class DownloadSFTDatasetTest(unittest.TestCase):
    def test_download_is_pinned_inventoried_and_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "posttraining" / "opencodeinstruct"
            payloads = (b"parquet-one", b"parquet-two-longer")
            rows = {0: 3, 1: 5}
            contract = tiny_contract(
                expected_bytes=sum(map(len, payloads)), expected_rows=sum(rows.values())
            )
            calls: list[dict[str, object]] = []

            def snapshot(**kwargs: object) -> str:
                calls.append(dict(kwargs))
                write_tiny_snapshot(Path(kwargs["local_dir"]), payloads)
                return str(kwargs["local_dir"])

            def row_counter(path: Path) -> int:
                return rows[int(path.name.split("-")[1])]

            result = downloader.download_sft_dataset(
                root,
                max_workers=7,
                contract=contract,
                snapshot_download=snapshot,
                row_counter=row_counter,
            )
            self.assertTrue(result["complete"])
            self.assertTrue(result["downloaded"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0],
                {
                    "repo_id": downloader.DEFAULT_REPO_ID,
                    "repo_type": "dataset",
                    "revision": TEST_REVISION,
                    "local_dir": root.resolve() / "raw",
                    "allow_patterns": [
                        "data/train-*.parquet",
                        "README.md",
                        ".gitattributes",
                    ],
                    "max_workers": 7,
                    "force_download": False,
                },
            )
            source_path = root / "SOURCE.json"
            completion_path = root / "COMPLETION.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            self.assertEqual(source["resolved_revision"], TEST_REVISION)
            self.assertEqual(source["repo_type"], "dataset")
            self.assertTrue(source["raw_files_preserved"])
            self.assertEqual(source["inventory"]["train_parquet_files"], 2)
            self.assertEqual(source["inventory"]["rows"], 8)
            self.assertEqual(
                [row["sha256"] for row in source["inventory"]["files"]],
                [hashlib.sha256(payload).hexdigest() for payload in payloads],
            )
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(
                completion["source_manifest_sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                completion["inventory_sha256"],
                source["inventory"]["inventory_sha256"],
            )
            self.assertEqual(
                [path.read_bytes() for path in sorted((root / "raw/data").glob("*.parquet"))],
                list(payloads),
            )

    def test_failed_download_publishes_nothing_and_rerun_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sft"
            payloads = (b"first", b"second")
            contract = tiny_contract(expected_bytes=11, expected_rows=4)

            def interrupted(**kwargs: object) -> str:
                raw = Path(kwargs["local_dir"])
                write_tiny_snapshot(raw, payloads)
                (raw / "data/train-00001-of-00002.parquet").unlink()
                raise RuntimeError("injected transport interruption")

            with self.assertRaisesRegex(
                downloader.SFTDownloadError, "injected transport"
            ):
                downloader.download_sft_dataset(
                    root,
                    contract=contract,
                    snapshot_download=interrupted,
                    row_counter=lambda _path: 2,
                )
            self.assertFalse((root / "SOURCE.json").exists())
            self.assertFalse((root / "COMPLETION.json").exists())
            self.assertTrue((root / "raw/data/train-00000-of-00002.parquet").is_file())

            def resumed(**kwargs: object) -> str:
                write_tiny_snapshot(Path(kwargs["local_dir"]), payloads)
                return str(kwargs["local_dir"])

            result = downloader.download_sft_dataset(
                root,
                contract=contract,
                snapshot_download=resumed,
                row_counter=lambda _path: 2,
            )
            self.assertTrue(result["complete"])
            self.assertTrue(result["downloaded"])

    def test_verify_only_certifies_existing_snapshot_without_hub_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sft"
            payloads = (b"one", b"two-two")
            paths = write_tiny_snapshot(root / "raw", payloads)
            row_counts = {
                paths[0].name: 10,
                paths[1].name: 20,
            }
            contract = tiny_contract(expected_bytes=10, expected_rows=30)
            snapshot = mock.Mock(side_effect=AssertionError("network call"))
            result = downloader.download_sft_dataset(
                root,
                verify_only=True,
                contract=contract,
                snapshot_download=snapshot,
                row_counter=lambda path: row_counts[path.name],
            )
            snapshot.assert_not_called()
            self.assertFalse(result["downloaded"])
            self.assertTrue(result["verified_existing_snapshot"])
            self.assertTrue((root / "SOURCE.json").is_file())
            self.assertTrue((root / "COMPLETION.json").is_file())

    def test_complete_rerun_is_offline_and_does_not_rewrite_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sft"
            payloads = (b"first", b"second")
            paths = write_tiny_snapshot(root / "raw", payloads)
            contract = tiny_contract(expected_bytes=11, expected_rows=4)
            row_counts = {path.name: 2 for path in paths}
            first = downloader.download_sft_dataset(
                root,
                verify_only=True,
                contract=contract,
                row_counter=lambda path: row_counts[path.name],
            )
            source_stat = (root / "SOURCE.json").stat()
            completion_stat = (root / "COMPLETION.json").stat()
            source_bytes = (root / "SOURCE.json").read_bytes()
            completion_bytes = (root / "COMPLETION.json").read_bytes()
            snapshot = mock.Mock(side_effect=AssertionError("network call"))
            second = downloader.download_sft_dataset(
                root,
                contract=contract,
                snapshot_download=snapshot,
                row_counter=lambda path: row_counts[path.name],
            )
            snapshot.assert_not_called()
            self.assertFalse(second["downloaded"])
            self.assertFalse(second["publication"]["source_written"])
            self.assertFalse(second["publication"]["completion_written"])
            self.assertEqual((root / "SOURCE.json").read_bytes(), source_bytes)
            self.assertEqual((root / "COMPLETION.json").read_bytes(), completion_bytes)
            self.assertEqual((root / "SOURCE.json").stat().st_ino, source_stat.st_ino)
            self.assertEqual(
                (root / "COMPLETION.json").stat().st_ino,
                completion_stat.st_ino,
            )
            self.assertEqual(first["completion"], second["completion"])

    def test_source_only_crash_window_is_completed_without_rewriting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sft"
            paths = write_tiny_snapshot(root / "raw", (b"a", b"bb"))
            contract = tiny_contract(expected_bytes=3, expected_rows=2)
            inventory = downloader.inventory_snapshot(
                root, contract=contract, row_counter=lambda _path: 1
            )
            source, _completion = downloader.build_authorities(
                inventory, contract=contract
            )
            downloader.atomic_write_if_missing_or_identical(
                root / "SOURCE.json", downloader.pretty_json_bytes(source)
            )
            source_inode = (root / "SOURCE.json").stat().st_ino
            result = downloader.download_sft_dataset(
                root,
                verify_only=True,
                contract=contract,
                row_counter=lambda _path: 1,
            )
            self.assertTrue(result["complete"])
            self.assertEqual((root / "SOURCE.json").stat().st_ino, source_inode)
            self.assertTrue((root / "COMPLETION.json").is_file())

    def test_bad_inventory_never_publishes_authority(self) -> None:
        cases = ("files", "bytes", "rows", "unexpected")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "sft"
                paths = write_tiny_snapshot(root / "raw", (b"a", b"bb"))
                contract = tiny_contract(expected_bytes=3, expected_rows=2)
                if case == "files":
                    paths[-1].unlink()
                elif case == "unexpected":
                    (root / "raw/data/eval.parquet").write_bytes(b"x")
                rows = 3 if case == "rows" else 1
                if case == "bytes":
                    contract = tiny_contract(expected_bytes=4, expected_rows=2)
                with self.assertRaises(downloader.SFTDownloadError):
                    downloader.download_sft_dataset(
                        root,
                        verify_only=True,
                        contract=contract,
                        row_counter=lambda _path, value=rows: value,
                    )
                self.assertFalse((root / "SOURCE.json").exists())
                self.assertFalse((root / "COMPLETION.json").exists())

    def test_existing_authority_drift_and_pretraining_root_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "sft"
            paths = write_tiny_snapshot(root / "raw", (b"a", b"bb"))
            contract = tiny_contract(expected_bytes=3, expected_rows=2)
            downloader.download_sft_dataset(
                root,
                verify_only=True,
                contract=contract,
                row_counter=lambda _path: 1,
            )
            source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
            source["resolved_revision"] = "b" * 40
            (root / "SOURCE.json").write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(
                downloader.SFTDownloadError, "Existing authority differs"
            ):
                downloader.download_sft_dataset(
                    root,
                    verify_only=True,
                    contract=contract,
                    row_counter=lambda _path: 1,
                )

            pretraining = base / "pretraining"
            marker = pretraining / "state/COLLECTION_COMPLETE.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}\n", encoding="utf-8")
            snapshot = mock.Mock()
            with self.assertRaisesRegex(
                downloader.SFTDownloadError, "pre-training root"
            ):
                downloader.download_sft_dataset(
                    pretraining / "posttraining",
                    contract=contract,
                    snapshot_download=snapshot,
                    row_counter=lambda _path: 1,
                )
            snapshot.assert_not_called()

    def test_real_parquet_row_metadata_reader(self) -> None:
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.parquet"
            parquet.write_table(arrow.table({"value": [1, 2, 3, 4]}), path)
            self.assertEqual(downloader._read_parquet_rows(path), 4)

    def test_worker_bounds_fail_before_any_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = mock.Mock()
            for workers in (0, downloader.MAX_MAX_WORKERS + 1):
                with self.subTest(workers=workers), self.assertRaisesRegex(
                    downloader.SFTDownloadError, "max_workers"
                ):
                    downloader.download_sft_dataset(
                        Path(temporary) / f"sft-{workers}",
                        max_workers=workers,
                        snapshot_download=snapshot,
                    )
            snapshot.assert_not_called()

    def test_exact_shard_names_and_process_lock_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sft"
            paths = write_tiny_snapshot(root / "raw", (b"a", b"bb"))
            paths[1].rename(root / "raw/data/train-wrong.parquet")
            contract = tiny_contract(expected_bytes=3, expected_rows=2)
            with self.assertRaisesRegex(
                downloader.SFTDownloadError, "filename inventory mismatch"
            ):
                downloader.download_sft_dataset(
                    root,
                    verify_only=True,
                    contract=contract,
                    row_counter=lambda _path: 1,
                )

            paths = write_tiny_snapshot(root / "raw", (b"a", b"bb"))
            (root / "raw/data/train-wrong.parquet").unlink()
            lock_path = root / ".download.lock"
            descriptor = lock_path.open("a+")
            try:
                import fcntl

                fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    downloader.SFTDownloadError, "Another SFT downloader"
                ):
                    downloader.download_sft_dataset(
                        root,
                        verify_only=True,
                        contract=contract,
                        row_counter=lambda _path: 1,
                    )
            finally:
                descriptor.close()


if __name__ == "__main__":
    unittest.main()
