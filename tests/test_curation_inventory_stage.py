from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from curation_inventory_stage import (
    MAXIMUM_BATCH_SIZE,
    InventoryStage,
    InventoryStageError,
    ReportSpec,
    StagePolicy,
    assert_scalar_equivalent,
    validate_fingerprint_record,
)
from curation_policy import DEFAULT_POLICY, load_policy
from preprocess_raw_stream import FINGERPRINT_VERSION


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _row(
    *,
    archive: str,
    bucket: str,
    index: int,
    member_path: str | None = None,
    tokens: int = 3,
    size_bytes: int = 100,
    flags: list[str] | None = None,
    benchmark_reason: Any = None,
) -> dict[str, Any]:
    path = member_path or f"files/{index:08d}.txt"
    quality_flags = sorted(set(flags or []))
    if benchmark_reason:
        quality_flags = sorted({*quality_flags, "benchmark_contamination"})
    provenance: dict[str, str]
    if bucket in ("python", "other_code"):
        provenance = {"repo_id": f"repository-{index // 2}"}
    else:
        provenance = {"url": f"https://example.invalid/{index}"}
    return {
        "record_version": 1,
        "fingerprint_version": FINGERPRINT_VERSION,
        "doc_id": hashlib.sha256(f"{archive}\0{path}".encode()).hexdigest(),
        "bucket": bucket,
        "archive": archive,
        "manifest_index": index,
        "member_path": path,
        "starcoder2_tokens": tokens,
        "size_bytes": size_bytes,
        "content_sha256": hashlib.sha256(f"content:{archive}:{index}".encode()).hexdigest(),
        "normalized_sha256": hashlib.sha256(
            f"normalized:{archive}:{index}".encode()
        ).hexdigest(),
        "quality_flags": quality_flags,
        "benchmark_reason": benchmark_reason,
        "provenance": provenance,
    }


def _make_report(
    staging: Path,
    *,
    ordinal: int,
    bucket: str,
    rows: list[dict[str, Any]],
) -> ReportSpec:
    archive = str(rows[0]["archive"])
    fingerprint = staging / "fingerprints" / bucket / f"part-{ordinal:06d}.jsonl.zst"
    fingerprint.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    fingerprint.write_bytes(zstandard.ZstdCompressor(level=1).compress(payload))
    report_path = staging / "reports" / bucket / f"part-{ordinal:06d}.json"
    report = {
        "archive": archive,
        "index": ordinal,
        "bucket": bucket,
        "documents": len(rows),
        "exact_tokens": sum(row["starcoder2_tokens"] for row in rows),
        "clean_bytes": sum(row["size_bytes"] for row in rows),
        "fingerprint_file": str(fingerprint.relative_to(staging)),
        "fingerprint_sha256": hashlib.sha256(fingerprint.read_bytes()).hexdigest(),
    }
    _write_json(report_path, report)
    return ReportSpec.from_report(
        ordinal=ordinal, report_path=report_path, staging_root=staging
    )


def _policy() -> StagePolicy:
    return StagePolicy.from_curation_policy(
        load_policy(DEFAULT_POLICY), fast_canonical_profile=True
    )


def _expected_rows(
    reports: list[ReportSpec], policy: StagePolicy
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    documents: list[tuple[Any, ...]] = []
    reasons: list[tuple[Any, ...]] = []
    for report in reports:
        from curation_inventory_stage import iter_jsonl_zst

        for index, row in enumerate(iter_jsonl_zst(report.fingerprint_path)):
            document = validate_fingerprint_record(
                row, expected_index=index, report=report, policy=policy
            )
            values = document.document_values(stage_id=0, archive_ordinal=0)
            documents.append(tuple(values[2:]))
            reasons.extend((document.doc_id, reason) for reason in document.reasons)
    return documents, reasons


def _kill_after_first_batch(
    db_path: str, staging_path: str, report_path: str, batch_size: int
) -> None:
    staging = Path(staging_path)
    report = ReportSpec.from_report(
        ordinal=0, report_path=Path(report_path), staging_root=staging
    )

    def fault(event: str, _payload: object) -> None:
        if event == "batch_committed":
            os._exit(91)

    with InventoryStage(
        Path(db_path),
        reports=[report],
        policy=_policy(),
        batch_size=batch_size,
        fault_hook=fault,
    ) as stage:
        stage.ingest_all()


class InventoryStageTest(unittest.TestCase):
    def test_append_only_ingest_then_scalar_equivalent_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            rows_a = [
                _row(
                    archive="raw/python/part-000000.tar.zst",
                    bucket="python",
                    index=index,
                    flags=["few_lines"] if index == 1 else [],
                )
                for index in range(5)
            ]
            rows_b = [
                _row(
                    archive="raw/english/part-000001.tar.zst",
                    bucket="fineweb_edu",
                    index=index,
                    benchmark_reason="synthetic" if index == 2 else None,
                )
                for index in range(4)
            ]
            reports = [
                _make_report(staging, ordinal=0, bucket="python", rows=rows_a),
                _make_report(
                    staging, ordinal=1, bucket="fineweb_edu", rows=rows_b
                ),
            ]
            policy = _policy()
            expected_documents, expected_reasons = _expected_rows(reports, policy)
            database = root / "inventory.sqlite3"
            with InventoryStage(
                database, reports=reports, policy=policy, batch_size=3
            ) as stage:
                status = stage.ingest_all()
                self.assertEqual(status["phase"], "ingest_complete")
                for table in ("stage_documents", "stage_reasons"):
                    self.assertEqual(
                        stage.db.execute(f"PRAGMA index_list('{table}')").fetchall(),
                        [],
                    )
                stage.validate_complete(verify_payload_digests=True)
                assert_scalar_equivalent(
                    stage,
                    expected_documents=expected_documents,
                    expected_reasons=expected_reasons,
                )
                before = stage.logical_sha256()
                evidence = stage.promote()
                self.assertEqual(evidence["logical_sha256"], before)
                self.assertEqual(evidence["sqlite_integrity_check"], "ok")
                assert_scalar_equivalent(
                    stage,
                    expected_documents=expected_documents,
                    expected_reasons=expected_reasons,
                )
                document_indexes = {
                    row[1]
                    for row in stage.db.execute("PRAGMA index_list('documents')")
                }
                self.assertEqual(
                    document_indexes,
                    {
                        "documents_doc_id",
                        "documents_archive_manifest",
                        "documents_archive_member",
                    },
                )
            with InventoryStage(
                database, reports=reports, policy=policy, batch_size=3
            ) as reopened:
                tables = {
                    row[0]
                    for row in reopened.db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("documents", tables)
                self.assertNotIn("stage_documents", tables)

    def test_real_process_death_after_batch_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            rows = [
                _row(
                    archive="raw/python/part-000000.tar.zst",
                    bucket="python",
                    index=index,
                    flags=["few_lines"] if index % 2 else [],
                )
                for index in range(8)
            ]
            report = _make_report(staging, ordinal=0, bucket="python", rows=rows)
            database = root / "resumed.sqlite3"
            # The full suite initializes Torch and native compression threads
            # before reaching this test. Forking that multithreaded process can
            # segfault inside the child on macOS before the intended crash seam
            # executes, which tests fork-unsafety rather than SQLite recovery.
            # A fresh spawned interpreter still provides a real process death
            # while keeping the failure deterministic across suite orderings.
            process = multiprocessing.get_context("spawn").Process(
                target=_kill_after_first_batch,
                args=(str(database), str(staging), str(report.report_path), 3),
            )
            process.start()
            process.join(20)
            self.assertEqual(process.exitcode, 91)
            with InventoryStage(
                database, reports=[report], policy=_policy(), batch_size=3
            ) as stage:
                self.assertEqual(stage.status()["documents"], 3)
                stage.ingest_all()
                resumed_digest = stage.promote()["logical_sha256"]
            fresh_database = root / "fresh.sqlite3"
            with InventoryStage(
                fresh_database, reports=[report], policy=_policy(), batch_size=3
            ) as fresh:
                fresh.ingest_all()
                fresh_digest = fresh.promote()["logical_sha256"]
            self.assertEqual(resumed_digest, fresh_digest)

    def test_interrupted_promotion_resumes_at_next_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            rows = [
                _row(
                    archive="raw/python/part-000000.tar.zst",
                    bucket="python",
                    index=index,
                )
                for index in range(7)
            ]
            report = _make_report(staging, ordinal=0, bucket="python", rows=rows)
            database = root / "inventory.sqlite3"

            def interrupt(event: str, payload: dict[str, Any]) -> None:
                if event == "promotion_index_committed" and payload["ordinal"] == 0:
                    raise RuntimeError("simulated interruption")

            with InventoryStage(
                database,
                reports=[report],
                policy=_policy(),
                batch_size=3,
                fault_hook=interrupt,
            ) as stage:
                stage.ingest_all()
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    stage.promote()
                self.assertEqual(stage.status()["promotion_step"], 1)
            with InventoryStage(
                database, reports=[report], policy=_policy(), batch_size=3
            ) as resumed:
                self.assertEqual(resumed.promote()["sqlite_integrity_check"], "ok")
                self.assertEqual(resumed.status()["phase"], "promoted")

    def test_payload_corruption_is_detected_before_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            rows = [
                _row(
                    archive="raw/python/part-000000.tar.zst",
                    bucket="python",
                    index=index,
                )
                for index in range(4)
            ]
            report = _make_report(staging, ordinal=0, bucket="python", rows=rows)
            database = root / "inventory.sqlite3"
            with InventoryStage(
                database, reports=[report], policy=_policy(), batch_size=2
            ) as stage:
                stage.ingest_all()
                stage.db.execute(
                    "UPDATE stage_documents SET content_hash=? WHERE stage_id=2",
                    (b"x" * 32,),
                )
                stage.db.commit()
                with self.assertRaisesRegex(
                    InventoryStageError, "Durable batch payload mismatch"
                ):
                    stage.promote()

    def test_duplicate_identity_is_deferred_and_rejected_at_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            archive = "raw/python/part-000000.tar.zst"
            rows = [
                _row(
                    archive=archive,
                    bucket="python",
                    index=index,
                    member_path="same/path.py",
                )
                for index in range(2)
            ]
            report = _make_report(staging, ordinal=0, bucket="python", rows=rows)
            with InventoryStage(
                root / "inventory.sqlite3",
                reports=[report],
                policy=_policy(),
                batch_size=2,
            ) as stage:
                stage.ingest_all()
                with self.assertRaisesRegex(
                    InventoryStageError, "documents_doc_id"
                ):
                    stage.promote()

    def test_report_order_batch_bounds_and_scalar_parity_traps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            report_zero = _make_report(
                staging,
                ordinal=0,
                bucket="python",
                rows=[
                    _row(
                        archive="raw/python/part-000000.tar.zst",
                        bucket="python",
                        index=0,
                        tokens=True,
                        size_bytes=True,
                        benchmark_reason={"truthy": True},
                    )
                ],
            )
            report_one = _make_report(
                staging,
                ordinal=1,
                bucket="other_code",
                rows=[
                    _row(
                        archive="raw/other/part-000001.tar.zst",
                        bucket="other_code",
                        index=0,
                    )
                ],
            )
            with InventoryStage(
                root / "inventory.sqlite3",
                reports=[report_zero, report_one],
                policy=_policy(),
                batch_size=250_000,
            ) as stage:
                with self.assertRaisesRegex(InventoryStageError, "Out-of-order"):
                    stage.ingest_report(report_one)
                self.assertEqual(stage.status()["documents"], 0)
                stage.ingest_all()
            with self.assertRaisesRegex(InventoryStageError, "batch_size"):
                InventoryStage(
                    root / "too-large.sqlite3",
                    reports=[report_zero, report_one],
                    policy=_policy(),
                    batch_size=MAXIMUM_BATCH_SIZE + 1,
                )

    def test_report_boolean_count_is_not_silently_coerced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            fingerprint = staging / "fingerprints" / "python" / "x.zst"
            fingerprint.parent.mkdir(parents=True)
            fingerprint.write_bytes(b"x")
            report_path = staging / "report.json"
            _write_json(
                report_path,
                {
                    "archive": "raw/python/x.tar.zst",
                    "index": 0,
                    "bucket": "python",
                    "documents": True,
                    "exact_tokens": 1,
                    "clean_bytes": 1,
                    "fingerprint_file": str(fingerprint.relative_to(staging)),
                    "fingerprint_sha256": hashlib.sha256(b"x").hexdigest(),
                },
            )
            with self.assertRaisesRegex(InventoryStageError, "documents"):
                ReportSpec.from_report(
                    ordinal=0, report_path=report_path, staging_root=staging
                )


if __name__ == "__main__":
    unittest.main()
