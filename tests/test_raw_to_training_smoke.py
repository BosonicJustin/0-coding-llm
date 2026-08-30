from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from smoke_raw_to_training_data import iter_archive_documents


def write_archive(path: Path, *, include_manifest: bool = True) -> None:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for name, content in (("a.py", b"print(1)\n"), ("large.py", b"x" * 32)):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if include_manifest:
            content = b"{}\n{}\n"
            member = tarfile.TarInfo("_manifest.jsonl")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    path.write_bytes(zstandard.ZstdCompressor().compress(raw_tar.getvalue()))


class RawToTrainingSmokeTest(unittest.TestCase):
    def test_streams_utf8_documents_and_skips_oversized_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "part-000000.tar.zst"
            write_archive(archive)
            documents = list(iter_archive_documents(archive, max_document_bytes=16))
            self.assertEqual(documents, [("a.py", "print(1)\n")])

    def test_requires_trailing_internal_manifest_when_fully_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "part-000000.tar.zst"
            write_archive(archive, include_manifest=False)
            with self.assertRaisesRegex(ValueError, "missing its trailing manifest"):
                list(iter_archive_documents(archive, max_document_bytes=100))


if __name__ == "__main__":
    unittest.main()
