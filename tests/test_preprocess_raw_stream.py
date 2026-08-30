from __future__ import annotations

import io
import hashlib
import json
import concurrent.futures
import multiprocessing
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import preprocess_raw_stream as preprocess

from preprocess_raw_stream import (
    ArchiveCandidate,
    DedupIndex,
    FreeSpaceSafetyError,
    ManifestSafetyError,
    OversizeDocumentError,
    POLICY_SHA256,
    analyze_document,
    collection_closure_payload,
    discover_archives,
    error_path,
    fingerprint_path,
    initialize_worker,
    normalize_content,
    process_archive,
    quality_metrics,
    report_coverage_payload,
    reconcile_published_states,
    record_error,
    report_path,
    require_free_space,
    status_payload,
    terminal_gate_failures,
)
from quota_tracker import write_record


def write_archive(path: Path, rows: list[tuple[str, bytes, int]]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    raw = path.open("wb")
    compressed = zstandard.ZstdCompressor(level=1, write_checksum=True).stream_writer(
        raw, closefd=False
    )
    archive = tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT)
    try:
        for index, (member_path, content, tokens) in enumerate(rows):
            info = tarfile.TarInfo(member_path)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
            manifest_lines.append(
                json.dumps(
                    {
                        "member_path": member_path,
                        "repo_id": "repo-1",
                        "file_path": f"src/file-{index}.py",
                        "content_id": f"content-{index}",
                        "language": "Python",
                        "license_type": "permissive",
                        "size_bytes": len(content),
                        "starcoder2_tokens": tokens,
                    },
                    separators=(",", ":"),
                )
            )
        manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
        info = tarfile.TarInfo("_manifest.jsonl")
        info.size = len(manifest)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest))
    finally:
        archive.close()
        compressed.close()
        raw.close()
    return {
        "documents": len(rows),
        "clean_bytes": sum(len(content) for _, content, _ in rows),
        "exact_tokens": sum(tokens for _, _, tokens in rows),
    }


def write_test_quota_config(path: Path, targets: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "quotas": [
            {
                "name": "collection/python",
                "phase": "collection",
                "category": "python",
                "token_field": "exact_tokens",
                "target": targets["python"],
            },
            {
                "name": "collection/other_code",
                "phase": "collection",
                "category": "other_code",
                "token_field": "exact_tokens",
                "target": targets["other_code"],
            },
            {
                "name": "collection/english",
                "phase": "collection",
                "category": "english",
                "token_field": "exact_tokens",
                "target": targets["fineweb_edu"] + targets["wikipedia"],
            },
            *[
                {
                    "name": f"collection/english/{bucket}",
                    "phase": "collection",
                    "category": "english",
                    "language_group": bucket,
                    "token_field": "exact_tokens",
                    "target": targets[bucket],
                }
                for bucket in ("fineweb_edu", "wikipedia")
            ],
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_closed_collection(root: Path, quota_path: Path) -> dict[str, int]:
    targets = {"python": 11, "other_code": 13, "fineweb_edu": 17, "wikipedia": 19}
    write_test_quota_config(quota_path, targets)
    sources = {
        "python": "code-source",
        "other_code": "code-source",
        "fineweb_edu": "fineweb-source",
        "wikipedia": "wikipedia-source",
    }
    for index, bucket in enumerate(("python", "other_code", "fineweb_edu", "wikipedia")):
        relative = preprocess.BUCKET_PATHS[bucket]
        archive = root / "raw" / relative / "part-000000.tar.zst"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(f"finalized-{bucket}".encode("utf-8"))
        record = {
            "phase": "collection",
            "category": "english" if bucket in ("fineweb_edu", "wikipedia") else bucket,
            "shard_id": f"fixture-{index:02d}-000000",
            "source": sources[bucket],
            "documents": 1,
            "clean_bytes": archive.stat().st_size,
            "exact_tokens": targets[bucket],
        }
        if bucket in ("fineweb_edu", "wikipedia"):
            record["language_group"] = bucket
        write_record(root, record)
    tokenizer_revision = "a" * 40
    benchmark_guard_sha256 = "b" * 64
    preprocess.atomic_json(
        root / "tokenizer" / "starcoder2" / "TOKENIZER_MANIFEST.json",
        {"manifest_version": 1, "resolved_revision": tokenizer_revision},
    )
    preprocess.atomic_json(
        root / "staging" / "preprocess" / "PREPROCESS_MANIFEST.json",
        {"manifest_version": 1, "benchmark_guard_sha256": benchmark_guard_sha256},
    )
    preprocess.atomic_json(
        root / "state" / "COLLECTION_COMPLETE.json",
        {
            "source": sources["python"],
            "tokenizer_revision": tokenizer_revision,
            "benchmark_guard_sha256": benchmark_guard_sha256,
            "python_tokens": targets["python"],
            "other_code_tokens": targets["other_code"],
            "targets": {
                "python": targets["python"],
                "other_code": targets["other_code"],
            },
        },
    )
    for bucket, marker_name in (
        ("fineweb_edu", "ENGLISH_FINEWEB_EDU_COMPLETE.json"),
        ("wikipedia", "ENGLISH_WIKIPEDIA_COMPLETE.json"),
    ):
        preprocess.atomic_json(
            root / "state" / marker_name,
            {
                "source": sources[bucket],
                "tokenizer_revision": tokenizer_revision,
                "benchmark_guard_sha256": benchmark_guard_sha256,
                "english_tokens": targets[bucket],
                "target": targets[bucket],
            },
        )
    return targets


class RawStreamPreprocessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_worker(str(PROJECT_ROOT / "configs" / "mbpp_denylist.json"))

    def test_normalization_metrics_and_fingerprint(self) -> None:
        # The live volume contains v1 reports bound to this identity. Operational
        # memory hardening must not silently turn it into a new data policy.
        self.assertEqual(
            POLICY_SHA256,
            "a340dc1f71a915bfec0a76509099a769eec886b2eb94c7ed31bb453f56d3d032",
        )
        left = "def value():\r\n    return 1   \r\n"
        right = "def value():\n    return 1\n"
        self.assertEqual(normalize_content(left, "python"), normalize_content(right, "python"))
        analysis = analyze_document(left.encode(), "python")
        self.assertEqual(len(analysis["content_sha256"]), 64)
        self.assertEqual(len(analysis["normalized_sha256"]), 64)
        self.assertTrue(analysis["near_sketch"])
        metrics, flags = quality_metrics("x", b"x", "python")
        self.assertEqual(metrics["characters"], 1)
        self.assertIn("too_short", flags)

    def test_archive_audit_and_dedup_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            content = b"def combine(left, right):\n    return left + right\n"
            totals = write_archive(
                archive,
                [
                    ("files/repo/000000000-a.py", content, 12),
                    ("files/repo/000000001-b.py", content, 12),
                ],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record=quota,
            )
            report = process_archive(candidate, staging, None, 1)
            self.assertEqual(report["documents"], 2)
            self.assertEqual(report["exact_tokens"], 24)
            fingerprints = fingerprint_path(staging, "python", 0)
            with fingerprints.open("rb") as raw:
                with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                    payload = stream.read().decode("utf-8")
            records = [json.loads(line) for line in payload.splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["content_sha256"], records[1]["content_sha256"])
            self.assertNotIn("return left + right", payload)

            index = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            mismatched_report = {**report, "exact_tokens": 25}
            with self.assertRaisesRegex(RuntimeError, "Fingerprint token count mismatch"):
                index.ingest(staging, mismatched_report, batch_size=1)
            self.assertEqual(index.status()["documents"], 0)
            result = index.ingest(staging, report)
            self.assertEqual(result["documents"], 2)
            self.assertEqual(result["exact_duplicate_documents"], 1)
            self.assertEqual(result["normalized_duplicate_documents"], 1)
            status = index.status()
            self.assertEqual(status["documents"], 2)
            self.assertEqual(status["exact_clusters"], 1)
            self.assertEqual(index.ingest(staging, report)["documents"], 0)
            with self.assertRaisesRegex(RuntimeError, "Indexed archive summary mismatch"):
                index.ingest(staging, {**report, "fingerprint_sha256": "0" * 64})
            index.close()

    def test_batched_dedup_index_is_correct_across_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            exact = b"def answer():\n    return 42\n"
            normalized_variant = b"def answer():\r\n    return 42   \r\n"
            reports = []
            for index, rows in enumerate(
                (
                    [
                        ("files/repo-a/000000000-a.py", exact, 7),
                        ("files/repo-a/000000001-b.py", exact, 11),
                    ],
                    [
                        ("files/repo-b/000000000-c.py", exact, 13),
                        ("files/repo-b/000000001-d.py", normalized_variant, 17),
                    ],
                )
            ):
                archive = root / "raw" / "python" / f"part-{index:06d}.tar.zst"
                totals = write_archive(archive, rows)
                reports.append(
                    process_archive(
                        ArchiveCandidate(
                            bucket="python",
                            index=index,
                            path=archive,
                            relative_path=f"raw/python/part-{index:06d}.tar.zst",
                            quota_record={
                                "phase": "collection",
                                "category": "python",
                                "shard_id": f"stack-v3-test-python-{index:06d}",
                                "source": "test-source",
                                **totals,
                            },
                        ),
                        staging,
                        None,
                        1,
                    )
                )

            dedup = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            first = dedup.ingest(staging, reports[0], batch_size=1)
            second = dedup.ingest(staging, reports[1], batch_size=1)
            self.assertEqual(first["exact_duplicate_documents"], 1)
            self.assertEqual(first["normalized_duplicate_documents"], 1)
            self.assertEqual(second["exact_duplicate_documents"], 1)
            self.assertEqual(second["normalized_duplicate_documents"], 2)
            status = dedup.status()
            self.assertEqual(status["documents"], 4)
            self.assertEqual(status["tokens"], 48)
            self.assertEqual(status["exact_clusters"], 2)
            self.assertEqual(status["exact_duplicate_documents"], 2)
            self.assertEqual(status["normalized_clusters"], 1)
            self.assertEqual(status["normalized_duplicate_documents"], 3)
            dedup.close()

    def test_status_exposes_deferred_index_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/000000000-a.py", b"print('ready')\n", 4)],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            write_record(root, quota)
            report = process_archive(
                ArchiveCandidate(
                    bucket="python",
                    index=0,
                    path=archive,
                    relative_path="raw/python/part-000000.tar.zst",
                    quota_record=quota,
                ),
                staging,
                None,
                1000,
            )
            index = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            before = status_payload(root, staging, index)["dedup_index"]
            self.assertEqual(before["fingerprint_archives_pending_index"], 1)
            self.assertEqual(before["fingerprint_tokens_pending_index"], 4)
            index.ingest(staging, report)
            after = status_payload(root, staging, index)["dedup_index"]
            self.assertEqual(after["fingerprint_archives_pending_index"], 0)
            self.assertEqual(after["fingerprint_tokens_pending_index"], 0)
            index.close()

    def test_deferred_then_index_only_is_restart_safe_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "scratch"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [
                    ("files/repo/000000000-a.py", b"print('a')\n", 4),
                    ("files/repo/000000001-b.py", b"print('b')\n", 5),
                ],
            )
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            base_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "preprocess_raw_stream.py"),
                "--root",
                str(root),
                "--staging-root",
                str(staging),
                "--scratch-root",
                str(scratch),
                "--workers",
                "1",
                "--analysis-batch-size",
                "2",
                "--status-interval-seconds",
                "3600",
                "--min-free-gb",
                "0",
                "--once",
            ]
            deferred = subprocess.run(
                [*base_command, "--index-mode", "deferred"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"event": "dedup_deferred"', deferred.stdout)
            index = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            self.assertEqual(index.status()["documents"], 0)
            index.close()

            first_index = subprocess.run(
                [*base_command, "--index-mode", "only", "--max-archives", "1"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first_index.stdout.count('"event": "dedup_ingested"'), 1)
            second_index = subprocess.run(
                [*base_command, "--index-mode", "only"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotIn('"event": "dedup_ingested"', second_index.stdout)
            index = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            self.assertEqual(index.status()["ingested_archives"], 1)
            self.assertEqual(index.status()["documents"], 2)
            self.assertEqual(index.status()["tokens"], 9)
            index.close()

    def test_discovery_requires_quota_record_and_ignores_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/000000000-a.py", b"print('ready')\n", 4)],
            )
            pending = archive.parent / ".part-000001-random.tar.zst"
            pending.write_bytes(b"incomplete")
            self.assertEqual(discover_archives(root, staging), [])
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            candidates = discover_archives(root, staging)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].path, archive)

    def test_quota_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/000000000-a.py", b"print('ready')\n", 4)],
            )
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record={
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **{**totals, "exact_tokens": 999},
                },
            )
            with self.assertRaisesRegex(ValueError, "Quota mismatch"):
                process_archive(candidate, staging, None, 1)

    def test_batched_process_pool_is_fingerprint_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            rows = [
                (
                    f"files/repo/{index:09d}-file.py",
                    f"def value_{index}():\n    return {index}\n".encode(),
                    8 + index,
                )
                for index in range(17)
            ]
            totals = write_archive(archive, rows)
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record=quota,
            )
            sequential_staging = root / "staging" / "sequential"
            parallel_staging = root / "staging" / "parallel"
            scratch_root = root / "scratch"
            sequential = process_archive(
                candidate, sequential_staging, None, 1000, analysis_batch_size=1
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=2,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(str(PROJECT_ROOT / "configs" / "mbpp_denylist.json"),),
            ) as executor:
                parallel = process_archive(
                    candidate,
                    parallel_staging,
                    executor,
                    1000,
                    analysis_batch_size=4,
                    scratch_root=scratch_root,
                )
            self.assertEqual(
                sequential["fingerprint_sha256"], parallel["fingerprint_sha256"]
            )
            for field in (
                "documents",
                "clean_bytes",
                "exact_tokens",
                "benchmark_hits",
                "quality_flag_counts",
                "language_documents",
                "language_tokens",
            ):
                self.assertEqual(sequential[field], parallel[field])
            self.assertEqual(list(scratch_root.iterdir()), [])
            self.assertEqual(
                parallel["fingerprint_sha256"],
                hashlib.sha256(
                    fingerprint_path(parallel_staging, "python", 0).read_bytes()
                ).hexdigest(),
            )

    def test_byte_backpressure_preserves_exact_fingerprint_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            rows = [
                (
                    f"files/repo/{index:09d}-file.py",
                    (f"def value_{index}():\n    return {index}\n# ".encode() + b"x" * 55),
                    10 + index,
                )
                for index in range(7)
            ]
            totals = write_archive(archive, rows)
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record={
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            baseline = process_archive(
                candidate,
                root / "staging" / "baseline",
                None,
                1000,
                analysis_batch_size=7,
            )
            document_bytes = max(len(content) for _, content, _ in rows)
            batch_cap = document_bytes * 2
            inflight_cap = document_bytes * 3
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                bounded = process_archive(
                    candidate,
                    root / "staging" / "bounded",
                    executor,
                    1000,
                    analysis_batch_size=7,
                    max_document_bytes=document_bytes,
                    max_analysis_batch_bytes=batch_cap,
                    max_inflight_bytes=inflight_cap,
                )
            self.assertEqual(baseline["fingerprint_sha256"], bounded["fingerprint_sha256"])
            memory = bounded["operational_memory"]
            self.assertLessEqual(memory["peak_analysis_batch_bytes"], batch_cap)
            self.assertLessEqual(memory["peak_inflight_payload_bytes"], inflight_cap)
            self.assertGreater(memory["submitted_batches"], 1)
            self.assertEqual(memory["oversize_documents"], 0)

    def test_internal_manifest_has_independent_member_line_and_row_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [
                    ("files/repo/000000000-a.py", b"def a():\n    return 1\n", 7),
                    ("files/repo/000000001-b.py", b"def b():\n    return 2\n", 8),
                ],
            )

            def candidate(documents: int = 2) -> ArchiveCandidate:
                return ArchiveCandidate(
                    bucket="python",
                    index=0,
                    path=archive,
                    relative_path="raw/python/part-000000.tar.zst",
                    quota_record={
                        "phase": "collection",
                        "category": "python",
                        "shard_id": "stack-v3-test-python-000000",
                        "source": "test-source",
                        **{**totals, "documents": documents},
                    },
                )

            with self.assertRaises(ManifestSafetyError) as member_error:
                process_archive(
                    candidate(),
                    root / "staging" / "member",
                    None,
                    1000,
                    max_manifest_member_bytes=32,
                )
            self.assertEqual(member_error.exception.limit, "member_bytes")

            with self.assertRaises(ManifestSafetyError) as line_error:
                process_archive(
                    candidate(),
                    root / "staging" / "line",
                    None,
                    1000,
                    max_manifest_line_bytes=64,
                )
            self.assertEqual(line_error.exception.limit, "line_bytes")
            self.assertEqual(line_error.exception.row_index, 0)

            with self.assertRaises(ManifestSafetyError) as row_error:
                process_archive(
                    candidate(documents=1),
                    root / "staging" / "rows",
                    None,
                    1000,
                )
            self.assertEqual(row_error.exception.limit, "document_rows")
            self.assertEqual(row_error.exception.maximum, 1)

    def test_manifest_bound_failure_is_durable_and_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "scratch"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "preprocess_raw_stream.py"),
                    "--root",
                    str(root),
                    "--staging-root",
                    str(staging),
                    "--scratch-root",
                    str(scratch),
                    "--workers",
                    "1",
                    "--max-manifest-line-bytes",
                    "64",
                    "--min-free-gb",
                    "0",
                    "--scratch-min-free-gb",
                    "0",
                    "--index-mode",
                    "deferred",
                    "--once",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            error = json.loads(error_path(staging, "python", 0).read_text(encoding="utf-8"))
            self.assertEqual(
                error["error_code"], "manifest_exceeds_operational_safety_bound"
            )
            self.assertEqual(error["limit"], "line_bytes")
            self.assertEqual(error["manifest_safety_version"], 1)
            self.assertFalse(report_path(staging, "python", 0).exists())

    def test_production_closure_gate_rejects_empty_open_and_pending_collections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            quotas = Path(temporary) / "quotas.json"
            targets = {"python": 1, "other_code": 1, "fineweb_edu": 1, "wikipedia": 1}
            write_test_quota_config(quotas, targets)

            # Keep the historical diagnostic meaning: the empty current ledger
            # has complete report coverage, but it is not a closed collection.
            empty = status_payload(root, staging, None, quota_path=quotas)
            self.assertTrue(empty["raw_audit_complete"])
            self.assertFalse(empty["collection_closure"]["complete"])
            self.assertEqual(terminal_gate_failures(empty), [])
            self.assertTrue(
                terminal_gate_failures(empty, require_closed_collection=True)
            )

            build_closed_collection(root, quotas)
            closed = collection_closure_payload(root, quotas)
            self.assertTrue(closed["complete"], closed)
            synthetic = {
                "raw_audit_complete": True,
                "collection_closure": closed,
            }
            self.assertEqual(
                terminal_gate_failures(synthetic, require_closed_collection=True), []
            )

            pending = root / "raw" / "python" / ".part-000001-live.tar.zst"
            pending.write_bytes(b"still being collected")
            open_collection = collection_closure_payload(root, quotas)
            self.assertFalse(open_collection["complete"])
            self.assertEqual(open_collection["raw_inventory"]["pending_inputs"], 1)
            pending.unlink()

            marker = root / "state" / "COLLECTION_COMPLETE.json"
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["python_tokens"] += 1
            preprocess.atomic_json(marker, payload)
            stale = collection_closure_payload(root, quotas)
            self.assertFalse(stale["complete"])
            self.assertTrue(
                any("python_tokens mismatch" in item for item in stale["failures"]),
                stale,
            )

    def test_cli_production_gate_and_deferred_audit_do_not_require_dedup_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "scratch"
            quotas = Path(temporary) / "quotas.json"
            write_test_quota_config(
                quotas,
                {"python": 1, "other_code": 1, "fineweb_edu": 1, "wikipedia": 1},
            )
            database = staging / "dedup" / "dedup.sqlite3"
            database.parent.mkdir(parents=True, exist_ok=True)
            corrupt_bytes = b"not a sqlite database"
            database.write_bytes(corrupt_bytes)
            base = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "preprocess_raw_stream.py"),
                "--root",
                str(root),
                "--staging-root",
                str(staging),
                "--scratch-root",
                str(scratch),
                "--quotas",
                str(quotas),
                "--workers",
                "1",
                "--min-free-gb",
                "0",
                "--scratch-min-free-gb",
                "0",
            ]
            deferred = subprocess.run(
                [*base, "--index-mode", "deferred", "--once"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(deferred.returncode, 0, deferred.stdout + deferred.stderr)
            self.assertEqual(database.read_bytes(), corrupt_bytes)
            self.assertIn('"available": false', deferred.stdout)

            gated = subprocess.run(
                [
                    *base,
                    "--index-mode",
                    "deferred",
                    "--once",
                    "--require-closed-collection",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(gated.returncode, 2, gated.stdout + gated.stderr)
            self.assertIn("collection is not closed", gated.stdout)
            self.assertEqual(database.read_bytes(), corrupt_bytes)

            fresh_status = subprocess.run(
                [
                    *base,
                    "--status",
                    "--require-complete",
                    "--skip-dedup-status",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                fresh_status.returncode, 0, fresh_status.stdout + fresh_status.stderr
            )
            self.assertFalse(json.loads(fresh_status.stdout)["dedup_index"]["available"])

    def test_production_wrapper_enables_closed_collection_gate(self) -> None:
        wrapper = (PROJECT_ROOT / "scripts" / "run_preprocess.sh").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(wrapper.count("--require-closed-collection"), 3)
        self.assertIn("--skip-dedup-status", wrapper)

    def test_oversize_document_is_explicitly_quarantined_and_terminal_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "scratch"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/huge.py", b"def huge():\n    return 1\n" + b"#" * 128, 20)],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            write_record(root, quota)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "preprocess_raw_stream.py"),
                "--root",
                str(root),
                "--staging-root",
                str(staging),
                "--scratch-root",
                str(scratch),
                "--workers",
                "1",
                "--max-document-bytes",
                "64",
                "--max-analysis-batch-bytes",
                "128",
                "--max-inflight-bytes",
                "256",
                "--min-free-gb",
                "0",
                "--scratch-min-free-gb",
                "0",
                "--index-mode",
                "deferred",
                "--once",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn('"event": "archive_error"', completed.stdout)
            self.assertIn('"event": "terminal_incomplete"', completed.stdout)
            error = json.loads(error_path(staging, "python", 0).read_text(encoding="utf-8"))
            self.assertEqual(error["error_code"], "document_exceeds_operational_byte_cap")
            self.assertEqual(error["quarantine_scope"], "archive")
            self.assertEqual(error["quarantined_documents"], 1)
            self.assertEqual(error["member_path"], "files/repo/huge.py")
            self.assertEqual(error["maximum_bytes"], 64)
            self.assertFalse(report_path(staging, "python", 0).exists())
            self.assertFalse(fingerprint_path(staging, "python", 0).exists())
            self.assertEqual(discover_archives(root, staging), [])
            self.assertEqual(len(discover_archives(root, staging, retry_errors=True)), 1)

    def test_injected_report_publication_crash_is_rediscoverable_and_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            write_record(root, quota)
            candidate = discover_archives(root, staging)[0]
            preprocess.atomic_json(
                error_path(staging, "python", 0),
                {"error_version": 1, "archive": candidate.relative_path, "error": "old"},
            )
            with mock.patch.object(
                preprocess, "atomic_json", side_effect=RuntimeError("injected report crash")
            ):
                with self.assertRaisesRegex(RuntimeError, "injected report crash"):
                    process_archive(candidate, staging, None, 1000)
            self.assertFalse(report_path(staging, "python", 0).exists())
            self.assertFalse(error_path(staging, "python", 0).exists())
            orphan_sha = hashlib.sha256(
                fingerprint_path(staging, "python", 0).read_bytes()
            ).hexdigest()
            self.assertEqual(len(discover_archives(root, staging)), 1)
            resumed = process_archive(candidate, staging, None, 1000)
            self.assertEqual(resumed["fingerprint_sha256"], orphan_sha)
            self.assertEqual(discover_archives(root, staging), [])

    def test_legacy_report_and_error_state_is_reconciled_durably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record={
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            process_archive(candidate, staging, None, 1000)
            preprocess.atomic_json(
                error_path(staging, "python", 0),
                {"error_version": 1, "archive": candidate.relative_path, "error": "stale"},
            )
            with mock.patch.object(preprocess, "fsync_directory") as directory_fsync:
                self.assertEqual(reconcile_published_states(staging), 1)
            self.assertFalse(error_path(staging, "python", 0).exists())
            directory_fsync.assert_called_once_with(error_path(staging, "python", 0).parent)
            self.assertFalse(record_error(staging, candidate, RuntimeError("late")))
            self.assertFalse(error_path(staging, "python", 0).exists())

    def test_scratch_free_space_gate_uses_actual_spool_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            with mock.patch.object(
                preprocess.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=99),
            ) as disk_usage:
                with self.assertRaises(FreeSpaceSafetyError) as raised:
                    require_free_space(scratch, 100, "scratch/spool")
            disk_usage.assert_called_once_with(scratch)
            self.assertEqual(raised.exception.error_details()["path"], str(scratch))

    def test_main_stops_nonzero_on_scratch_filesystem_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "separate-scratch"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )

            def disk_usage(path: Path) -> SimpleNamespace:
                return SimpleNamespace(free=0 if Path(path) == scratch else 10**12)

            argv = [
                "preprocess_raw_stream.py",
                "--root",
                str(root),
                "--staging-root",
                str(staging),
                "--scratch-root",
                str(scratch),
                "--workers",
                "1",
                "--min-free-gb",
                "0",
                "--scratch-min-free-gb",
                "1",
                "--index-mode",
                "deferred",
                "--once",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                preprocess.shutil, "disk_usage", side_effect=disk_usage
            ) as usage:
                self.assertEqual(preprocess.main(), 2)
            called_paths = [Path(call.args[0]) for call in usage.call_args_list]
            self.assertIn(root, called_paths)
            self.assertIn(scratch, called_paths)
            self.assertFalse(report_path(staging, "python", 0).exists())

    def test_scratch_space_is_rechecked_while_archive_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            scratch = Path(temporary) / "scratch"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            candidate = ArchiveCandidate(
                bucket="python",
                index=0,
                path=archive,
                relative_path="raw/python/part-000000.tar.zst",
                quota_record={
                    "phase": "collection",
                    "category": "python",
                    "shard_id": "stack-v3-test-python-000000",
                    "source": "test-source",
                    **totals,
                },
            )
            free_values = iter((1_000, 99))
            with mock.patch.object(
                preprocess.shutil,
                "disk_usage",
                side_effect=lambda _path: SimpleNamespace(free=next(free_values)),
            ), mock.patch.object(preprocess, "SCRATCH_RECHECK_SECONDS", 0.0):
                with self.assertRaises(FreeSpaceSafetyError):
                    process_archive(
                        candidate,
                        staging,
                        None,
                        1000,
                        scratch_root=scratch,
                        scratch_min_free_bytes=100,
                    )
            self.assertFalse(report_path(staging, "python", 0).exists())
            self.assertFalse(fingerprint_path(staging, "python", 0).exists())
            self.assertEqual(list(scratch.iterdir()), [])

    def test_missing_fingerprint_fails_exact_terminal_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            write_record(root, quota)
            process_archive(discover_archives(root, staging)[0], staging, None, 1000)
            fingerprint_path(staging, "python", 0).unlink()
            index = DedupIndex(staging / "dedup" / "dedup.sqlite3")
            try:
                snapshot = status_payload(root, staging, index)
            finally:
                index.close()
            self.assertFalse(snapshot["raw_audit_complete"])
            self.assertEqual(snapshot["report_coverage"]["invalid_reports"], 1)
            self.assertTrue(terminal_gate_failures(snapshot))

    def test_existing_report_without_operational_metrics_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            staging = root / "staging" / "preprocess"
            archive = root / "raw" / "python" / "part-000000.tar.zst"
            totals = write_archive(
                archive,
                [("files/repo/a.py", b"def answer():\n    return 42\n", 8)],
            )
            quota = {
                "phase": "collection",
                "category": "python",
                "shard_id": "stack-v3-test-python-000000",
                "source": "test-source",
                **totals,
            }
            write_record(root, quota)
            process_archive(discover_archives(root, staging)[0], staging, None, 1000)
            published = report_path(staging, "python", 0)
            legacy = json.loads(published.read_text(encoding="utf-8"))
            legacy.pop("operational_memory")
            preprocess.atomic_json(published, legacy)
            coverage = report_coverage_payload(root, staging)
            self.assertTrue(coverage["complete"], coverage)
            self.assertEqual(coverage["valid_reports"], 1)

    def test_existing_dedup_database_gets_constant_time_token_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "dedup.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute(
                """
                CREATE TABLE ingested_archives (
                    archive TEXT PRIMARY KEY,
                    fingerprint_sha256 TEXT NOT NULL,
                    documents INTEGER NOT NULL,
                    exact_duplicate_documents INTEGER NOT NULL,
                    normalized_duplicate_documents INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                "INSERT INTO ingested_archives VALUES (?, ?, ?, ?, ?, datetime('now'))",
                ("raw/python/part-000000.tar.zst", "abc", 10, 2, 3),
            )
            connection.commit()
            connection.close()

            index = DedupIndex(database)
            index.backfill_archive_tokens(
                [
                    {
                        "archive": "raw/python/part-000000.tar.zst",
                        "exact_tokens": 123,
                    }
                ]
            )
            status = index.status()
            self.assertEqual(status["ingested_archives"], 1)
            self.assertEqual(status["documents"], 10)
            self.assertEqual(status["tokens"], 123)
            self.assertEqual(status["exact_clusters"], 8)
            self.assertEqual(status["normalized_clusters"], 7)
            index.close()


if __name__ == "__main__":
    unittest.main()
