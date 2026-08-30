from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_wikipedia import (
    RawTextArchiveWriter,
    commit_checkpoint,
    english_collection_target,
    is_url_quarantined,
    ordered_source_shards,
    recover_checkpoints,
)


class WikipediaCollectorTest(unittest.TestCase):
    def test_wikipedia_target_and_url_quarantine(self) -> None:
        config = {
            "quotas": [
                {
                    "phase": "collection",
                    "category": "english",
                    "language_group": "wikipedia",
                    "token_field": "exact_tokens",
                    "target": 123,
                }
            ]
        }
        self.assertEqual(english_collection_target(config), 123)
        self.assertTrue(is_url_quarantined("https://en.wikipedia.org/wiki/MBPP"))
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
            text = "An English encyclopedia article.\n"
            row = {
                "id": "42",
                "url": "https://en.wikipedia.org/wiki/Example",
                "title": "Example",
            }
            writer.add(text.encode("utf-8"), 7, row)
            source = "wikimedia/wikipedia@" + "a" * 40 + "#20231101.en"
            commit_checkpoint(root, writer, 1, 0, 1, 1, "a" * 40, source, "b" * 40, "c" * 64)
            cursor = recover_checkpoints(root, source, "b" * 40, "c" * 64)
            self.assertEqual(cursor["row_offset"], 1)
            archive = root / "raw" / "english" / "wikipedia" / "part-000000.tar.zst"
            with archive.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as tar:
                        members = {
                            member.name: tar.extractfile(member).read()
                            for member in tar
                            if member.isfile()
                        }
            document_name = next(name for name in members if name.startswith("documents/"))
            self.assertEqual(members[document_name].decode("utf-8"), text)
            manifest = json.loads(members["_manifest.jsonl"])
            self.assertEqual(manifest["title"], "Example")
            self.assertEqual(manifest["starcoder2_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
