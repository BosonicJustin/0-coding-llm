from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pretrain.raw_token_cache import run_cache_jobs
from pretrain.raw_token_cache_reader import (
    ArchiveAuthority,
    FileAuthority,
    RawTokenCacheAuthority,
    RawTokenCacheReadError,
    RawTokenCacheReader,
    TokenizerAuthority,
)
from tests.test_raw_token_cache import RawTokenCacheFixture, canonical_json


def authority_from_cache(target: Path) -> RawTokenCacheAuthority:
    manifest_raw = (target / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    archive = manifest["source"]["archive"]
    report = manifest["source"]["preprocess_report"]
    fingerprint = manifest["source"]["fingerprint"]
    tokenizer = dict(manifest["tokenizer"])
    tokenizer.pop("eos_present_in_payload")
    documents = manifest["documents"]
    return RawTokenCacheAuthority(
        cache_manifest_bytes=len(manifest_raw),
        cache_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        archive=ArchiveAuthority(**archive),
        preprocess_report=FileAuthority(
            path=report["path"], bytes=report["bytes"], sha256=report["sha256"]
        ),
        fingerprint=FileAuthority(
            path=fingerprint["path"],
            bytes=fingerprint["bytes"],
            sha256=fingerprint["sha256"],
        ),
        tokenizer=TokenizerAuthority(**tokenizer),
        records=documents["records"],
        clean_bytes=documents["clean_bytes"],
        content_tokens=documents["content_tokens"],
    )


def publish_manifest(target: Path, manifest: dict[str, object]) -> None:
    raw = canonical_json(manifest)
    sha256 = hashlib.sha256(raw).hexdigest()
    (target / "manifest.json").write_bytes(raw)
    (target / "manifest.sha256").write_bytes(
        f"{sha256}  manifest.json\n".encode("ascii")
    )


class RawTokenCacheReaderFixture:
    def __init__(self, root: Path):
        self.source = RawTokenCacheFixture(root)
        self.output = root / "cache"
        run_cache_jobs(
            [self.source.job()],
            self.output,
            self.source.tokenizer_root,
            config=self.source.config,
        )
        self.target = self.source.job().target(self.output)

    def authority(self) -> RawTokenCacheAuthority:
        return authority_from_cache(self.target)

    def open(self, authority: RawTokenCacheAuthority | None = None) -> RawTokenCacheReader:
        return RawTokenCacheReader.open(
            self.target,
            authority or self.authority(),
            dataset_root=self.source.root,
            preprocess_root=self.source.preprocess,
            tokenizer_root=self.source.tokenizer_root,
        )


class RawTokenCacheReaderTest(unittest.TestCase):
    def test_random_lookup_and_sequential_iteration_preserve_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            expected = [
                fixture.source.token_ids(content)
                for _member, content in fixture.source.documents
            ]
            with fixture.open() as reader:
                self.assertEqual(reader.records, 2)
                self.assertEqual(reader.content_tokens, sum(map(len, expected)))
                self.assertEqual(reader.document_ids(0), expected[0])
                self.assertEqual(reader.document_ids(1), expected[1])
                span = reader.document(0)
                self.assertEqual(len(span), len(expected[0]))
                self.assertEqual(span[0], expected[0][0])
                self.assertEqual(span[-1], expected[0][-1])
                self.assertEqual(span[:], expected[0])
                self.assertEqual(
                    [(index, tokens.to_list()) for index, tokens in reader.iter_documents()],
                    list(enumerate(expected)),
                )
                reader.verify_unchanged()
                with self.assertRaises(IndexError):
                    reader.document_ids(2)
            with self.assertRaisesRegex(RawTokenCacheReadError, "closed"):
                span.to_list()

    def test_wrong_generation_tokenizer_and_source_authorities_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            authority = fixture.authority()
            wrong_generation = replace(
                authority, cache_manifest_sha256="0" * 64
            )
            with self.assertRaisesRegex(RawTokenCacheReadError, "Wrong cache generation"):
                fixture.open(wrong_generation)
            wrong_tokenizer = replace(
                authority,
                tokenizer=replace(
                    authority.tokenizer, resolved_revision="b" * 40
                ),
            )
            with self.assertRaisesRegex(RawTokenCacheReadError, "tokenizer binding"):
                fixture.open(wrong_tokenizer)
            wrong_source = replace(
                authority,
                archive=replace(authority.archive, sha256="c" * 64),
            )
            with self.assertRaisesRegex(RawTokenCacheReadError, "source binding"):
                fixture.open(wrong_source)
            raw_archive = fixture.source.archive
            raw_payload = bytearray(raw_archive.read_bytes())
            raw_payload[len(raw_payload) // 2] ^= 1
            raw_archive.write_bytes(raw_payload)
            with self.assertRaisesRegex(RawTokenCacheReadError, "raw archive SHA-256"):
                fixture.open(authority)

    def test_payload_hash_corruption_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            authority = fixture.authority()
            tokens = fixture.target / "tokens.u16"
            payload = bytearray(tokens.read_bytes())
            payload[0] ^= 1
            tokens.write_bytes(payload)
            with self.assertRaisesRegex(RawTokenCacheReadError, "hash/range corruption"):
                fixture.open(authority)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            authority = fixture.authority()
            tokens = fixture.target / "tokens.u16"
            outside = Path(temporary) / "outside.u16"
            tokens.rename(outside)
            tokens.symlink_to(outside)
            with self.assertRaisesRegex(RawTokenCacheReadError, "read-only"):
                fixture.open(authority)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            authority = fixture.authority()
            (fixture.target / "manifest.sha256").write_text(
                f"{'0' * 64}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(RawTokenCacheReadError, "sidecar mismatch"):
                fixture.open(authority)

    def test_reauthorized_wrong_dtype_and_offset_order_still_fail_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            manifest = json.loads((fixture.target / "manifest.json").read_text())
            manifest["payloads"]["tokens"]["dtype"] = "int16"
            publish_manifest(fixture.target, manifest)
            with self.assertRaisesRegex(RawTokenCacheReadError, "dtype/count/order"):
                fixture.open(fixture.authority())

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            offsets_path = fixture.target / "offsets.u64"
            offsets = list(struct.unpack("<3Q", offsets_path.read_bytes()))
            offsets[1] = 0
            payload = struct.pack("<3Q", *offsets)
            offsets_path.write_bytes(payload)
            manifest = json.loads((fixture.target / "manifest.json").read_text())
            manifest["payloads"]["offsets"]["sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
            publish_manifest(fixture.target, manifest)
            with self.assertRaisesRegex(RawTokenCacheReadError, "strictly increasing"):
                fixture.open(fixture.authority())

    def test_fingerprint_order_is_replayed_not_assumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RawTokenCacheReaderFixture(Path(temporary))
            fingerprint = fixture.source.fingerprint
            import zstandard

            with zstandard.open(fingerprint, "rb") as handle:
                rows = [json.loads(line) for line in handle.read().splitlines()]
            rows.reverse()
            raw = b"".join(canonical_json(row) for row in rows)
            compressed = zstandard.ZstdCompressor(
                level=1, write_checksum=True
            ).compress(raw)
            fingerprint.write_bytes(compressed)
            fingerprint_sha = hashlib.sha256(compressed).hexdigest()

            report = json.loads(fixture.source.report.read_text())
            report["fingerprint_sha256"] = fingerprint_sha
            fixture.source.report.write_bytes(
                json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            report_raw = fixture.source.report.read_bytes()

            manifest = json.loads((fixture.target / "manifest.json").read_text())
            manifest["source"]["fingerprint"]["bytes"] = len(compressed)
            manifest["source"]["fingerprint"]["sha256"] = fingerprint_sha
            manifest["source"]["preprocess_report"]["bytes"] = len(report_raw)
            manifest["source"]["preprocess_report"]["sha256"] = hashlib.sha256(
                report_raw
            ).hexdigest()
            publish_manifest(fixture.target, manifest)
            with self.assertRaisesRegex(
                RawTokenCacheReadError, "manifest order/identity mismatch"
            ):
                fixture.open(fixture.authority())


if __name__ == "__main__":
    unittest.main()
