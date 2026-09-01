from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

import zstandard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from pretrain.raw_token_cache import (
    CacheConfig,
    RawTokenCacheError,
    load_cache_job,
    run_cache_jobs,
)
from pretrain.tokenizer_identity import sha256_file


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class RawTokenCacheFixture:
    def __init__(self, root: Path):
        self.root = root / "dataset"
        self.preprocess = self.root / "staging" / "preprocess"
        self.tokenizer_root = self.root / "tokenizer" / "starcoder2"
        self.archive = self.root / "raw" / "python" / "part-000000.tar.zst"
        self.fingerprint = (
            self.preprocess / "fingerprints" / "python" / "part-000000.jsonl.zst"
        )
        self.report = self.preprocess / "reports" / "python" / "part-000000.json"
        self.config = CacheConfig(
            max_documents_per_archive=100,
            max_document_bytes=1024 * 1024,
            max_document_tokens=1024 * 1024,
            tokenizer_batch_documents=2,
            tokenizer_batch_bytes=1024 * 1024,
            tokenizer_batch_tokens=1024 * 1024,
            max_manifest_member_bytes=1024 * 1024,
            max_json_line_bytes=64 * 1024,
            minimum_free_bytes=1,
        )
        self._write_tokenizer()
        self.documents = [
            ("files/repo/000000000-a.py", b"alpha beta\n"),
            ("files/repo/000000001-b.py", b"gamma alpha\n"),
        ]
        self._write_sources(self.documents)

    def _write_tokenizer(self) -> None:
        self.tokenizer_root.mkdir(parents=True)
        vocabulary = {
            "<unk>": 0,
            "<|endoftext|>": 1,
            "alpha": 2,
            "beta": 3,
            "gamma": 4,
        }
        for identifier in range(5, 49_152):
            vocabulary[f"token-{identifier:05d}"] = identifier
        tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer_path = self.tokenizer_root / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        manifest = {
            "manifest_version": 1,
            "repo_id": "bigcode/starcoder2-tokenizer",
            "requested_revision": "a" * 40,
            "resolved_revision": "a" * 40,
            "files": {
                "tokenizer.json": {
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": sha256_file(tokenizer_path),
                }
            },
            "validation": {
                "vocab_size": 49_152,
                "bos_token": None,
                "bos_token_id": None,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 1,
                "special_token_ids": {"<|endoftext|>": 1},
            },
        }
        (self.tokenizer_root / "TOKENIZER_MANIFEST.json").write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

    def token_ids(self, content: bytes) -> list[int]:
        tokenizer = Tokenizer.from_file(str(self.tokenizer_root / "tokenizer.json"))
        return tokenizer.encode(
            content.decode("utf-8"), add_special_tokens=False
        ).ids

    def _write_sources(
        self,
        documents: list[tuple[str, bytes]],
        *,
        token_count_delta: int = 0,
        non_regular_first_member: bool = False,
    ) -> None:
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        raw = self.archive.open("wb")
        compressor = zstandard.ZstdCompressor(
            level=1, write_checksum=True
        ).stream_writer(raw, closefd=False)
        archive = tarfile.open(fileobj=compressor, mode="w|", format=tarfile.PAX_FORMAT)
        manifest_rows: list[dict[str, object]] = []
        fingerprint_rows: list[dict[str, object]] = []
        total_tokens = 0
        try:
            for index, (member_path, content) in enumerate(documents):
                identifiers = self.token_ids(content)
                tokens = len(identifiers) + (token_count_delta if index == 0 else 0)
                if non_regular_first_member and index == 0:
                    info = tarfile.TarInfo(member_path)
                    info.type = tarfile.SYMTYPE
                    info.linkname = "elsewhere"
                    info.size = 0
                    archive.addfile(info)
                else:
                    info = tarfile.TarInfo(member_path)
                    info.size = len(content)
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(content))
                manifest_rows.append(
                    {
                        "member_path": member_path,
                        "repo_id": "fixture/repo",
                        "file_path": f"file-{index}.py",
                        "content_id": f"content-{index}",
                        "language": "Python",
                        "license_type": "permissive",
                        "size_bytes": len(content),
                        "starcoder2_tokens": tokens,
                    }
                )
                fingerprint_rows.append(
                    {
                        "record_version": 1,
                        "fingerprint_version": 1,
                        "doc_id": hashlib.sha256(
                            (
                                "raw/python/part-000000.tar.zst\0" + member_path
                            ).encode("utf-8")
                        ).hexdigest(),
                        "bucket": "python",
                        "archive": "raw/python/part-000000.tar.zst",
                        "archive_index": 0,
                        "manifest_index": index,
                        "member_path": member_path,
                        "size_bytes": len(content),
                        "starcoder2_tokens": tokens,
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "normalized_sha256": hashlib.sha256(content).hexdigest(),
                        "near_sketch": [],
                        "metrics": {},
                        "quality_flags": [],
                        "benchmark_reason": None,
                        "provenance": {"language": "Python"},
                    }
                )
                total_tokens += tokens
            manifest = b"".join(canonical_json(row) for row in manifest_rows)
            info = tarfile.TarInfo("_manifest.jsonl")
            info.size = len(manifest)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest))
        finally:
            archive.close()
            compressor.close()
            raw.close()

        self.fingerprint.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_raw = self.fingerprint.open("wb")
        fingerprint_compressor = zstandard.ZstdCompressor(
            level=1, write_checksum=True
        ).stream_writer(fingerprint_raw, closefd=False)
        try:
            for row in fingerprint_rows:
                fingerprint_compressor.write(canonical_json(row))
            fingerprint_compressor.flush(zstandard.FLUSH_FRAME)
        finally:
            fingerprint_compressor.close()
            fingerprint_raw.close()

        report = {
            "report_version": 1,
            "fingerprint_version": 1,
            "policy_sha256": "b" * 64,
            "archive": "raw/python/part-000000.tar.zst",
            "archive_sha256": sha256_file(self.archive),
            "archive_compressed_bytes": self.archive.stat().st_size,
            "bucket": "python",
            "index": 0,
            "quota_shard_id": "fixture-python-000000",
            "source": "fixture-source",
            "fingerprint_file": "fingerprints/python/part-000000.jsonl.zst",
            "fingerprint_sha256": sha256_file(self.fingerprint),
            "documents": len(documents),
            "clean_bytes": sum(len(content) for _, content in documents),
            "exact_tokens": total_tokens,
            "benchmark_hits": 0,
            "quality_flag_counts": {},
            "language_documents": {"Python": len(documents)},
            "language_tokens": {"Python": total_tokens},
        }
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")

    def job(self):
        return load_cache_job(
            self.root,
            self.preprocess,
            self.report,
            config=self.config,
        )


class RawTokenCacheTest(unittest.TestCase):
    def test_rebuild_is_byte_identical_and_preserves_full_document_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            first_root = Path(temporary) / "cache-one"
            second_root = Path(temporary) / "cache-two"
            first = run_cache_jobs(
                [fixture.job()],
                first_root,
                fixture.tokenizer_root,
                config=fixture.config,
            )
            second = run_cache_jobs(
                [fixture.job()],
                second_root,
                fixture.tokenizer_root,
                config=fixture.config,
            )
            self.assertEqual(first[0].status, "built")
            self.assertEqual(second[0].status, "built")
            first_target = fixture.job().target(first_root)
            second_target = fixture.job().target(second_root)
            for name in ("tokens.u16", "offsets.u64", "manifest.json", "manifest.sha256"):
                self.assertEqual(
                    (first_target / name).read_bytes(),
                    (second_target / name).read_bytes(),
                )
            expected_ids = [fixture.token_ids(content) for _, content in fixture.documents]
            raw_tokens = (first_target / "tokens.u16").read_bytes()
            observed_tokens = list(struct.unpack(f"<{len(raw_tokens) // 2}H", raw_tokens))
            self.assertEqual(observed_tokens, [item for row in expected_ids for item in row])
            raw_offsets = (first_target / "offsets.u64").read_bytes()
            observed_offsets = list(struct.unpack("<3Q", raw_offsets))
            self.assertEqual(
                observed_offsets,
                [0, len(expected_ids[0]), len(expected_ids[0]) + len(expected_ids[1])],
            )
            manifest = json.loads((first_target / "manifest.json").read_text())
            self.assertFalse(manifest["training_ready"])
            self.assertFalse(manifest["tokenization"]["boundary_tokens"])
            self.assertEqual(manifest["documents"]["records"], 2)

    def test_stale_stage_is_recovered_and_completed_fast_path_rehashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            output = Path(temporary) / "cache"
            target = fixture.job().target(output)
            target.parent.mkdir(parents=True)
            stale = target.parent / f".{target.name}.building-crashed"
            stale.mkdir()
            (stale / "junk").write_bytes(b"incomplete")
            built = run_cache_jobs(
                [fixture.job()],
                output,
                fixture.tokenizer_root,
                config=fixture.config,
            )
            self.assertEqual(built[0].status, "built")
            self.assertFalse(stale.exists())
            verified = run_cache_jobs(
                [fixture.job()],
                output,
                fixture.tokenizer_root,
                config=fixture.config,
            )
            self.assertEqual(verified[0].status, "verified")
            token_path = target / "tokens.u16"
            payload = bytearray(token_path.read_bytes())
            payload[0] ^= 1
            token_path.write_bytes(payload)
            with self.assertRaisesRegex(RawTokenCacheError, "payload descriptor/checksum"):
                run_cache_jobs(
                    [fixture.job()],
                    output,
                    fixture.tokenizer_root,
                    config=fixture.config,
                )

    def test_report_mutation_after_discovery_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            job = fixture.job()
            with fixture.report.open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(RawTokenCacheError, "report changed"):
                run_cache_jobs(
                    [job],
                    Path(temporary) / "cache",
                    fixture.tokenizer_root,
                    config=fixture.config,
                )

    def test_corrupt_archive_and_non_regular_member_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            job = fixture.job()
            payload = bytearray(fixture.archive.read_bytes())
            payload[len(payload) // 2] ^= 0xFF
            fixture.archive.write_bytes(payload)
            with self.assertRaises(RawTokenCacheError):
                run_cache_jobs(
                    [job],
                    Path(temporary) / "cache-corrupt",
                    fixture.tokenizer_root,
                    config=fixture.config,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            fixture._write_sources(fixture.documents, non_regular_first_member=True)
            with self.assertRaisesRegex(RawTokenCacheError, "non-regular tar member"):
                run_cache_jobs(
                    [fixture.job()],
                    Path(temporary) / "cache-link",
                    fixture.tokenizer_root,
                    config=fixture.config,
                )

    def test_token_count_mismatch_is_never_truncated_or_padded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheFixture(Path(temporary))
            fixture._write_sources(fixture.documents, token_count_delta=1)
            with self.assertRaisesRegex(RawTokenCacheError, "tokenizer count mismatch"):
                run_cache_jobs(
                    [fixture.job()],
                    Path(temporary) / "cache",
                    fixture.tokenizer_root,
                    config=fixture.config,
                )


if __name__ == "__main__":
    unittest.main()
