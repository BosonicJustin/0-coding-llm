from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_stack_v3 import (
    RawArchiveWriter,
    classify_language,
    is_quarantined,
    load_rejection_ids,
    record_benchmark_rejection,
    safe_suffix,
)


class CollectorPolicyTest(unittest.TestCase):
    def test_strict_language_classification(self) -> None:
        python = {"Python"}
        other = {"C", "Rust", "TypeScript"}
        self.assertEqual(classify_language("Python", python, other), "python")
        self.assertEqual(classify_language("Rust", python, other), "other_code")
        for rejected in ("Markdown", "HTML", "CSS", "JSON", "YAML", "Jupyter Notebook", None):
            self.assertIsNone(classify_language(rejected, python, other))

    def test_benchmark_paths_are_quarantined(self) -> None:
        self.assertTrue(is_quarantined("evalplus/evalplus", "data/mbpp.py"))
        self.assertTrue(is_quarantined("nuprl/MultiPL-E", "prompts/foo.py"))
        self.assertTrue(is_quarantined("owner/project", "tests/humaneval_test.py"))
        self.assertFalse(is_quarantined("python/cpython", "Lib/functools.py"))

    def test_suffix_sanitization(self) -> None:
        self.assertEqual(safe_suffix("src/main.cpp"), ".cpp")
        self.assertEqual(safe_suffix("../odd.$bad"), "")

    def test_benchmark_rejection_ledger_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark_rejections.jsonl"
            seen = load_rejection_ids(path)
            repo = {"repo_path": "owner/repo", "repo_id": 7, "commit_id": "abc"}
            file = {
                "file_path": "solution.py",
                "content_id": "content-hash",
                "language": "Python",
            }
            for _ in range(2):
                record_benchmark_rejection(
                    path,
                    seen,
                    "mbpp-exact-code",
                    "dataset@revision",
                    2,
                    3,
                    repo,
                    file,
                )
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(load_rejection_ids(path)), 1)

    def test_raw_archive_round_trip(self) -> None:
        import zstandard

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = RawArchiveWriter(root, "python", 0, 1, 1)
            content = b"def answer():\n    return 42\n"
            writer.add(
                content,
                9,
                {"repo_path": "owner/repo", "repo_id": 123, "commit_id": "abc"},
                {
                    "file_path": "src/main.py",
                    "content_id": "deadbeef",
                    "language": "Python",
                    "license_type": "permissive",
                    "detected_licenses": ["MIT"],
                },
            )
            archive = writer.close()
            self.assertIsNotNone(archive)
            assert archive is not None
            with Path(archive["pending_path"]).open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as tar:
                        members = {}
                        for member in tar:
                            extracted = tar.extractfile(member)
                            members[member.name] = extracted.read() if extracted else b""
            code_member = next(name for name in members if name.startswith("files/"))
            self.assertEqual(members[code_member], content)
            manifest = json.loads(members["_manifest.jsonl"])
            self.assertEqual(manifest["file_path"], "src/main.py")
            self.assertEqual(manifest["starcoder2_tokens"], 9)


if __name__ == "__main__":
    unittest.main()
