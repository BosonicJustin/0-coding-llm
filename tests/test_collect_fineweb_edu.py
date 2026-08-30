from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_fineweb_edu import (
    RawTextArchiveWriter,
    commit_checkpoint,
    english_collection_target,
    is_url_quarantined,
    load_rejection_ids,
    ordered_source_shards,
    record_rejection,
    recover_checkpoints,
)


class FineWebEduCollectorTest(unittest.TestCase):
    def test_english_target_and_url_quarantine(self) -> None:
        config = {
            "quotas": [
                {
                    "phase": "collection",
                    "category": "english",
                    "language_group": "fineweb_edu",
                    "token_field": "exact_tokens",
                    "target": 123,
                }
            ]
        }
        self.assertEqual(english_collection_target(config), 123)
        self.assertTrue(is_url_quarantined("https://example.org/evals/mbpp/task-1"))
        self.assertTrue(is_url_quarantined("https://example.org/HumanEval/solution"))
        self.assertFalse(is_url_quarantined("https://en.wikipedia.org/wiki/Python"))

    def test_source_shard_order_is_deterministic(self) -> None:
        names = ["c.parquet", "a.parquet", "b.parquet"]
        self.assertEqual(ordered_source_shards(names, 7), ordered_source_shards(names, 7))
        self.assertEqual(sorted(ordered_source_shards(names, 7)), sorted(names))

    def test_raw_archive_round_trip_and_checkpoint(self) -> None:
        import zstandard

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawTextArchiveWriter(root, 0, 1, 1)
            text = "A compact English document for the archive.\n"
            row = {
                "id": "document-1",
                "dump": "CC-MAIN-TEST",
                "url": "https://example.org/article",
                "file_path": "s3://example/warc.gz",
                "language": "en",
                "language_score": 0.99,
                "token_count": 10,
                "score": 3.0,
                "int_score": 3,
            }
            writer.add(text.encode("utf-8"), 8, row)
            source = "HuggingFaceFW/fineweb-edu@" + "a" * 40 + "#sample-100BT"
            commit_checkpoint(
                root,
                writer,
                1,
                0,
                1,
                1,
                "a" * 40,
                source,
                "b" * 40,
                "c" * 64,
            )
            cursor = recover_checkpoints(root, source, "b" * 40, "c" * 64)
            self.assertEqual(cursor["row_offset"], 1)
            archive = root / "raw" / "english" / "fineweb_edu" / "part-000000.tar.zst"
            self.assertTrue(archive.is_file())
            with archive.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as tar:
                        members = {}
                        for member in tar:
                            extracted = tar.extractfile(member)
                            members[member.name] = extracted.read() if extracted else b""
            document_name = next(name for name in members if name.startswith("documents/"))
            self.assertEqual(members[document_name].decode("utf-8"), text)
            manifest = json.loads(members["_manifest.jsonl"])
            self.assertEqual(manifest["id"], "document-1")
            self.assertEqual(manifest["starcoder2_tokens"], 8)

    def test_rejection_ledger_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rejections.jsonl"
            seen = load_rejection_ids(path)
            row = {
                "id": "document-1",
                "url": "https://example.org/mbpp",
                "dump": "CC-MAIN-TEST",
                "file_path": "s3://example/warc.gz",
            }
            for _ in range(2):
                record_rejection(path, seen, "mbpp-marker", "source@revision", 0, 1, row)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(load_rejection_ids(path)), 1)


if __name__ == "__main__":
    unittest.main()
