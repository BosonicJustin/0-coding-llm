from __future__ import annotations

import array
import collections
import hashlib
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_guard import BenchmarkGuard
import build_english_near_clusters as near_module
from build_english_near_clusters import (
    CALIBRATION_HARNESS,
    CANDIDATE_DOCUMENTS_FOR_ARCHIVE_SQL,
    DEFAULT_CALIBRATION_CONFIG,
    DEFAULT_CONFIG,
    EnglishNearDedupBuilder,
    EnglishNearDedupError,
    NEXT_CANDIDATE_BLOCK_SQL,
    REFINEMENT_BATCH_SQL,
    UNION_DOCUMENT_BATCH_SQL,
    UNION_PARENT_BATCH_SQL,
    candidate_bands,
    exact_shingle_hashes,
    iter_jsonl_zst,
    jaccard_counts,
    load_config,
    parse_bsd_mount_output,
    parse_linux_mountinfo,
    select_sqlite_journal_mode,
)
from curation_policy import DEFAULT_POLICY, canonical_sha256
from preprocess_raw_stream import (
    FINGERPRINT_VERSION,
    POLICY_DESCRIPTOR,
    POLICY_SHA256,
    analyze_document,
)


TOKENIZER_REVISION = "1" * 40
FINEWEB_REVISION = "2" * 40
WIKIPEDIA_REVISION = "3" * 40


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_sha256_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="utf-8"
    )


class NearFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging = root / "staging" / "preprocess"
        self.guard = BenchmarkGuard(PROJECT_ROOT / "configs" / "mbpp_denylist.json")
        self.docs: dict[str, str] = {}
        self._identities()
        base = [f"sharedword{index:03d}" for index in range(160)]
        near = list(base)
        near[40] = "smallchangealpha"
        near[100] = "smallchangebeta"
        self._archive(
            "fineweb_edu",
            0,
            [
                ("exact_fine", "  This is the same normalized article " + " ".join(base) + "\n"),
                ("near_fine", " ".join(base)),
                ("unrelated_fine", " ".join(f"fineonly{index:03d}" for index in range(160))),
            ],
        )
        self._archive(
            "wikipedia",
            0,
            [
                ("exact_wiki", "this is the same normalized article " + " ".join(base)),
                ("near_wiki", " ".join(near)),
                ("unrelated_wiki", " ".join(f"wikionly{index:03d}" for index in range(160))),
            ],
        )
        self._finalize_collection_evidence()

    def _identities(self) -> None:
        write_json(
            self.root / "tokenizer" / "starcoder2" / "TOKENIZER_MANIFEST.json",
            {"manifest_version": 1, "resolved_revision": TOKENIZER_REVISION},
        )
        for filename, repo, revision, config in (
            ("FINEWEB_EDU_SOURCE.json", "HuggingFaceFW/fineweb-edu", FINEWEB_REVISION, "sample-10BT"),
            ("WIKIPEDIA_SOURCE.json", "wikimedia/wikipedia", WIKIPEDIA_REVISION, "20231101.en"),
        ):
            write_json(
                self.root / "manifests" / filename,
                {
                    "manifest_version": 1,
                    "repo_id": repo,
                    "resolved_revision": revision,
                    "dataset_config": config,
                    "tokenizer_revision": TOKENIZER_REVISION,
                },
            )
        write_json(
            self.staging / "PREPROCESS_MANIFEST.json",
            {
                "manifest_version": 1,
                "fingerprint_version": FINGERPRINT_VERSION,
                "policy": POLICY_DESCRIPTOR,
                "policy_sha256": POLICY_SHA256,
                "benchmark_guard_sha256": self.guard.manifest_sha256,
                "raw_data_mutated": False,
            },
        )

    def _source(self, bucket: str) -> str:
        if bucket == "fineweb_edu":
            return f"HuggingFaceFW/fineweb-edu@{FINEWEB_REVISION}#sample-10BT"
        return f"wikimedia/wikipedia@{WIKIPEDIA_REVISION}#20231101.en"

    def _archive(self, bucket: str, index: int, documents: list[tuple[str, str]]) -> None:
        relative_archive = {
            "fineweb_edu": f"raw/english/fineweb_edu/part-{index:06d}.tar.zst",
            "wikipedia": f"raw/english/wikipedia/part-{index:06d}.tar.zst",
        }[bucket]
        archive_path = self.root / relative_archive
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tar_buffer = io.BytesIO()
        fingerprint_rows: list[dict[str, object]] = []
        manifest_rows: list[dict[str, object]] = []
        with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
            for member_index, (name, text) in enumerate(documents):
                content = text.encode("utf-8")
                member_path = f"files/{member_index:08d}.txt"
                info = tarfile.TarInfo(member_path)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
                analysis = analyze_document(content, bucket)
                doc_id = hashlib.sha256(
                    f"{relative_archive}\0{member_path}".encode("utf-8")
                ).hexdigest()
                self.docs[name] = doc_id
                tokens = max(1, len(text.split()))
                provenance = (
                    {"url": f"https://example.test/{name}"}
                    if bucket == "fineweb_edu"
                    else {"id": name, "title": name}
                )
                fingerprint_rows.append(
                    {
                        "record_version": 1,
                        "fingerprint_version": FINGERPRINT_VERSION,
                        "doc_id": doc_id,
                        "bucket": bucket,
                        "archive": relative_archive,
                        "archive_index": index,
                        "manifest_index": member_index,
                        "member_path": member_path,
                        "size_bytes": len(content),
                        "starcoder2_tokens": tokens,
                        "content_sha256": analysis["content_sha256"],
                        "normalized_sha256": analysis["normalized_sha256"],
                        "near_sketch": analysis["near_sketch"],
                        "metrics": analysis["metrics"],
                        "quality_flags": analysis["quality_flags"],
                        "benchmark_reason": analysis["benchmark_reason"],
                        "provenance": provenance,
                    }
                )
                manifest_rows.append(
                    {
                        "member_path": member_path,
                        "size_bytes": len(content),
                        "starcoder2_tokens": tokens,
                        **provenance,
                    }
                )
            manifest_payload = b"".join(
                json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
                for row in manifest_rows
            )
            info = tarfile.TarInfo("_manifest.jsonl")
            info.size = len(manifest_payload)
            archive.addfile(info, io.BytesIO(manifest_payload))
        archive_path.write_bytes(
            zstandard.ZstdCompressor(level=1, write_checksum=True).compress(
                tar_buffer.getvalue()
            )
        )
        fingerprint = (
            self.staging / "fingerprints" / bucket / f"part-{index:06d}.jsonl.zst"
        )
        fingerprint.parent.mkdir(parents=True, exist_ok=True)
        fingerprint.write_bytes(
            zstandard.ZstdCompressor(level=1, write_checksum=True).compress(
                b"".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                    for row in fingerprint_rows
                )
            )
        )
        report = {
            "report_version": 1,
            "fingerprint_version": FINGERPRINT_VERSION,
            "policy_sha256": POLICY_SHA256,
            "archive": relative_archive,
            "archive_sha256": sha256(archive_path),
            "archive_compressed_bytes": archive_path.stat().st_size,
            "bucket": bucket,
            "index": index,
            "quota_shard_id": f"fixture-{bucket}-{index:06d}",
            "source": self._source(bucket),
            "fingerprint_file": str(fingerprint.relative_to(self.staging)),
            "fingerprint_sha256": sha256(fingerprint),
            "documents": len(fingerprint_rows),
            "clean_bytes": sum(int(row["size_bytes"]) for row in fingerprint_rows),
            "exact_tokens": sum(int(row["starcoder2_tokens"]) for row in fingerprint_rows),
        }
        write_json(self.staging / "reports" / bucket / f"part-{index:06d}.json", report)
        write_json(
            self.root
            / "state"
            / "quota_records"
            / "collection"
            / f"fixture-{bucket}-{index:06d}.json",
            {
                "record_version": 1,
                "phase": "collection",
                "category": "english",
                "language_group": bucket,
                "shard_id": report["quota_shard_id"],
                "source": report["source"],
                "documents": report["documents"],
                "clean_bytes": report["clean_bytes"],
                "exact_tokens": report["exact_tokens"],
            },
        )

    def _finalize_collection_evidence(self) -> None:
        quotas: list[dict[str, object]] = []
        markers = {
            "fineweb_edu": "ENGLISH_FINEWEB_EDU_COMPLETE.json",
            "wikipedia": "ENGLISH_WIKIPEDIA_COMPLETE.json",
        }
        for bucket in ("fineweb_edu", "wikipedia"):
            reports = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((self.staging / "reports" / bucket).glob("*.json"))
            ]
            tokens = sum(int(row["exact_tokens"]) for row in reports)
            quotas.append(
                {
                    "name": f"collection/english/{bucket}",
                    "phase": "collection",
                    "category": "english",
                    "language_group": bucket,
                    "token_field": "exact_tokens",
                    "target": tokens,
                }
            )
            write_json(
                self.root / "state" / markers[bucket],
                {
                    "source": self._source(bucket),
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "benchmark_guard_sha256": self.guard.manifest_sha256,
                    "english_tokens": tokens,
                    "target": tokens,
                },
            )
        self.quota_path = self.root / "tiny-collection-quotas.json"
        write_json(self.quota_path, {"version": 1, "quotas": quotas})

    def builder(
        self,
        output: Path,
        config: Path = DEFAULT_CONFIG,
        policy: Path = DEFAULT_POLICY,
        *,
        calibration_result: Path | None = None,
        sqlite_journal_mode: str | None = None,
        batch_size: int = 2,
    ) -> EnglishNearDedupBuilder:
        calibration = calibration_result or self._calibration_result(config, policy)
        return EnglishNearDedupBuilder(
            root=self.root,
            staging_root=self.staging,
            output=output,
            config_path=config,
            policy_path=policy,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            quota_config_path=self.quota_path,
            calibration_result_path=calibration,
            batch_size=batch_size,
            sqlite_journal_mode=sqlite_journal_mode,
        )

    def _calibration_result(self, config: Path, policy: Path) -> Path:
        probe = EnglishNearDedupBuilder(
            root=self.root,
            staging_root=self.staging,
            output=self.root / ".identity-probe-not-created",
            config_path=config,
            policy_path=policy,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            quota_config_path=self.quota_path,
            batch_size=2,
            identity_probe_only=True,
        )
        calibration_config = json.loads(
            DEFAULT_CALIBRATION_CONFIG.read_text(encoding="utf-8")
        )
        result_path = (
            self.root
            / "audits"
            / f"calibration-{probe.config_file_sha[:12]}-{probe.policy_sha[:12]}.json"
        )
        if result_path.is_file():
            return result_path
        selected_reports = [
            {
                "report_path": item.relative_report,
                "report_sha256": item.report_sha256,
                "archive": item.report["archive"],
                "archive_sha256": item.report["archive_sha256"],
                "fingerprint_file": item.report["fingerprint_file"],
                "fingerprint_sha256": item.report["fingerprint_sha256"],
                "documents": item.report["documents"],
            }
            for item in probe.reports
        ]
        sample_manifest = [
            {"doc_id": doc_id, "fixture_calibration_evidence": True}
            for doc_id in sorted(self.docs.values())
        ]
        documents_selected = len(sample_manifest)
        completeness_sha = hashlib.sha256(
            json.dumps(
                probe.collection_completeness,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        input_identity = {
            "kind": "immutable_real_english_sample",
            "full_report_inventory_sha256": probe.report_inventory_sha,
            "preprocess_manifest_sha256": probe.preprocess_manifest_sha,
            "curation_policy_sha256": probe.policy_sha,
            "benchmark_guard_sha256": probe.guard.manifest_sha256,
            "source_manifests": probe.source_manifests,
            "collection_completeness_sha256": completeness_sha,
            "collection_completeness": probe.collection_completeness,
            "selection_seed": calibration_config["seed"],
            "maximum_archives_per_bucket": calibration_config["sampling"][
                "maximum_archives_per_bucket"
            ],
            "maximum_documents_per_bucket": calibration_config["sampling"][
                "maximum_documents_per_bucket"
            ],
            "minimum_source_words": calibration_config["sampling"][
                "minimum_source_words"
            ],
            "selected_reports": selected_reports,
            "documents_selected": documents_selected,
        }
        result = {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "identity": {
                "harness_sha256": sha256(CALIBRATION_HARNESS),
                "production_builder_sha256": sha256(
                    PROJECT_ROOT / "scripts" / "build_english_near_clusters.py"
                ),
                "calibration_algorithm": calibration_config[
                    "calibration_algorithm"
                ],
                "calibration_seed": calibration_config["seed"],
                "production_config_file_sha256": probe.config_file_sha,
                "production_config_canonical_sha256": probe.config_sha,
                "calibration_config_file_sha256": sha256(
                    DEFAULT_CALIBRATION_CONFIG
                ),
                "calibration_config_canonical_sha256": canonical_sha256(
                    calibration_config
                ),
                "input": input_identity,
                "sample_manifest_sha256": hashlib.sha256(
                    json.dumps(
                        sample_manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "production_threshold": {
                "minimum_jaccard_numerator": probe.config["refinement"][
                    "minimum_jaccard_numerator"
                ],
                "minimum_jaccard_denominator": probe.config["refinement"][
                    "minimum_jaccard_denominator"
                ],
            },
            "acceptance": calibration_config["acceptance"],
            "acceptance_profile": "pinned-production",
            "acceptance_overrides": {},
            "sampling_profile": "pinned-production",
            "acceptance_failures": [],
            "sampling": {
                "documents_input": documents_selected,
                "eligible_documents": documents_selected,
                "sample_manifest": sample_manifest,
            },
        }
        write_json(result_path, result)
        write_sha256_sidecar(result_path)
        return result_path


class EnglishNearDedupTest(unittest.TestCase):
    def test_missing_failed_stale_and_tampered_calibration_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            common = {
                "root": fixture.root,
                "staging_root": fixture.staging,
                "output": fixture.root / "missing-calibration",
                "config_path": DEFAULT_CONFIG,
                "policy_path": DEFAULT_POLICY,
                "denylist_path": PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                "quota_config_path": fixture.quota_path,
                "batch_size": 2,
            }
            with self.assertRaisesRegex(
                EnglishNearDedupError, "Missing required production calibration"
            ):
                EnglishNearDedupBuilder(**common)

            valid = fixture._calibration_result(DEFAULT_CONFIG, DEFAULT_POLICY)
            original = json.loads(valid.read_text(encoding="utf-8"))

            failed = fixture.root / "audits" / "failed.json"
            failed_payload = json.loads(json.dumps(original))
            failed_payload["status"] = "fail"
            failed_payload["production_gate_eligible"] = False
            failed_payload["acceptance_failures"] = ["forced-test-failure"]
            write_json(failed, failed_payload)
            write_sha256_sidecar(failed)
            with self.assertRaisesRegex(
                EnglishNearDedupError, "status is not production-passing"
            ):
                fixture.builder(
                    fixture.root / "failed-calibration",
                    calibration_result=failed,
                )

            stale = fixture.root / "audits" / "stale.json"
            stale_payload = json.loads(json.dumps(original))
            stale_payload["identity"]["input"][
                "full_report_inventory_sha256"
            ] = "0" * 64
            write_json(stale, stale_payload)
            write_sha256_sidecar(stale)
            with self.assertRaisesRegex(
                EnglishNearDedupError,
                "Calibration input identity mismatch for full_report_inventory_sha256",
            ):
                fixture.builder(
                    fixture.root / "stale-calibration",
                    calibration_result=stale,
                )

            tampered = fixture.root / "audits" / "tampered.json"
            tampered.write_bytes(valid.read_bytes())
            write_sha256_sidecar(tampered)
            tampered.write_bytes(tampered.read_bytes() + b" ")
            with self.assertRaisesRegex(
                EnglishNearDedupError, "Calibration result checksum mismatch"
            ):
                fixture.builder(
                    fixture.root / "tampered-calibration",
                    calibration_result=tampered,
                )

    def test_calibration_path_drift_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            first = fixture._calibration_result(DEFAULT_CONFIG, DEFAULT_POLICY)
            second = fixture.root / "audits" / "same-result-new-path.json"
            second.write_bytes(first.read_bytes())
            write_sha256_sidecar(second)
            output = fixture.root / "calibration-resume-identity"
            with fixture.builder(output, calibration_result=first) as builder:
                partial = builder.run(max_new_inventory_archives=1)
                self.assertFalse(partial["complete"])
            with self.assertRaisesRegex(
                EnglishNearDedupError, "Resume identity mismatch for calibration_evidence"
            ):
                with fixture.builder(output, calibration_result=second):
                    pass

    def test_filesystem_journal_policy_never_uses_wal_on_nfs_or_unknown(self) -> None:
        mountinfo = "\n".join(
            (
                "24 1 0:1 / / rw,relatime - ext4 /dev/root rw",
                "25 24 0:2 / /workspace rw,relatime - nfs4 server:/volume rw",
            )
        )
        evidence = parse_linux_mountinfo(
            mountinfo, Path("/workspace/dataset/english-near")
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["filesystem_type"], "nfs4")
        darwin = parse_bsd_mount_output(
            "/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled)\n",
            mount_source="/dev/disk3s5",
            mount_point="/System/Volumes/Data",
        )
        self.assertIsNotNone(darwin)
        assert darwin is not None
        self.assertEqual(darwin["filesystem_type"], "apfs")
        allowlist = ["apfs", "ext4", "xfs"]
        self.assertEqual(
            select_sqlite_journal_mode("nfs4", "auto", allowlist),
            ("delete", "non-local"),
        )
        self.assertEqual(
            select_sqlite_journal_mode("unknown", "auto", allowlist),
            ("delete", "unknown"),
        )
        self.assertEqual(
            select_sqlite_journal_mode("ext4", "auto", allowlist),
            ("wal", "proven-local"),
        )
        with self.assertRaisesRegex(EnglishNearDedupError, "Refusing SQLite WAL"):
            select_sqlite_journal_mode("nfs", "wal", allowlist)

    def test_delete_resume_rejects_wal_sidecars_on_network_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "network-delete-sidecar"
            evidence = {
                "filesystem_type": "nfs4",
                "mount_point": str(fixture.root),
                "mount_source": "fixture-server:/network-volume",
                "mount_options": "rw",
                "detection": "fixture-mount-evidence",
            }
            with patch.object(near_module, "detect_filesystem", return_value=evidence):
                with fixture.builder(output) as builder:
                    result = builder.run(max_new_inventory_archives=1)
                    self.assertFalse(result["complete"])
                sidecar = Path(
                    f"{output / '.work' / 'english_near.sqlite3'}-wal"
                )
                sidecar.write_bytes(b"unsafe-or-incompletely-copied-wal")
                with self.assertRaisesRegex(
                    EnglishNearDedupError, "contains WAL sidecars"
                ):
                    with fixture.builder(output):
                        pass

    def test_resume_rejects_journal_mode_drift_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "journal-mode-drift"
            database = output / ".work" / "english_near.sqlite3"
            evidence = {
                "filesystem_type": "ext4",
                "mount_point": str(fixture.root),
                "mount_source": "/dev/fixture-local",
                "mount_options": "rw",
                "detection": "fixture-mount-evidence",
            }
            with patch.object(near_module, "detect_filesystem", return_value=evidence):
                with fixture.builder(
                    output, sqlite_journal_mode="delete"
                ) as builder:
                    result = builder.run(max_new_inventory_archives=1)
                    self.assertFalse(result["complete"])
                before = sha256(database)
                with self.assertRaisesRegex(
                    EnglishNearDedupError, "journal mode mismatch"
                ):
                    with fixture.builder(
                        output, sqlite_journal_mode="wal"
                    ):
                        pass
                self.assertEqual(sha256(database), before)
                with sqlite3.connect(database) as connection:
                    self.assertEqual(
                        connection.execute("PRAGMA journal_mode").fetchone()[0],
                        "delete",
                    )

    def test_resume_rejects_frozen_mount_evidence_drift_before_schema_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "mount-evidence-drift"
            database = output / ".work" / "english_near.sqlite3"

            def evidence(source: str) -> dict[str, str]:
                return {
                    "filesystem_type": "ext4",
                    "mount_point": str(fixture.root),
                    "mount_source": source,
                    "mount_options": "rw",
                    "detection": "fixture-mount-evidence",
                }

            with patch.object(
                near_module, "detect_filesystem", return_value=evidence("/dev/local-a")
            ):
                with fixture.builder(
                    output, sqlite_journal_mode="delete"
                ) as builder:
                    result = builder.run(max_new_inventory_archives=1)
                    self.assertFalse(result["complete"])
            before = sha256(database)
            with patch.object(
                near_module, "detect_filesystem", return_value=evidence("/dev/local-b")
            ):
                with self.assertRaisesRegex(
                    EnglishNearDedupError, "Resume identity mismatch for runtime"
                ):
                    with fixture.builder(
                        output, sqlite_journal_mode="delete"
                    ):
                        pass
            self.assertEqual(sha256(database), before)

    def test_finalized_archive_without_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            (fixture.staging / "reports" / "wikipedia" / "part-000000.json").unlink()
            with self.assertRaisesRegex(
                EnglishNearDedupError, "finalized report coverage incomplete"
            ):
                fixture.builder(fixture.root / "incomplete")

    def test_report_totals_must_match_finalized_quota_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            report_path = (
                fixture.staging / "reports" / "fineweb_edu" / "part-000000.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["documents"] += 1
            write_json(report_path, report)
            with self.assertRaisesRegex(
                EnglishNearDedupError, "report/ledger documents mismatch"
            ):
                fixture.builder(fixture.root / "mismatched-totals")

    def test_full_raw_refinement_mapping_and_interrupted_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "english-near-resumed"

            with fixture.builder(output) as builder:
                result = builder.run(max_new_inventory_archives=1)
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "inventory")
            with fixture.builder(output) as builder:
                result = builder.run(max_new_signature_archives=1)
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "signatures")
            with fixture.builder(output) as builder:
                result = builder.run(max_new_candidate_blocks=1)
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "candidates")

            # Continue through any number of deterministic controlled stops.
            while True:
                with fixture.builder(output) as builder:
                    result = builder.run(
                        max_new_cache_archives=1,
                        max_new_refinement_pairs=1,
                        max_new_union_edges=1,
                    )
                if result["complete"]:
                    break
            manifest = result["manifest"]
            self.assertTrue(manifest["production_ready"])
            self.assertEqual(manifest["mapping"]["records"], 6)
            storage = manifest["identity"]["runtime"]["storage"]
            self.assertEqual(
                storage["sqlite_journal_mode_actual"],
                storage["sqlite_journal_mode_selected"],
            )
            if storage["classification"] != "proven-local":
                self.assertEqual(storage["sqlite_journal_mode_actual"], "delete")
            checkpoint = json.loads(
                (output / ".work" / "CHECKPOINT.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["identity"]["runtime"]["storage"], storage)
            calibration = manifest["identity"]["calibration_evidence"]
            self.assertEqual(calibration["contract_version"], 1)
            self.assertEqual(calibration["status"], "pass")
            self.assertTrue(calibration["production_gate_eligible"])
            self.assertEqual(calibration["acceptance_profile"], "pinned-production")
            self.assertEqual(calibration["sampling_profile"], "pinned-production")
            self.assertEqual(
                checkpoint["identity"]["calibration_evidence"], calibration
            )
            preflight = manifest["refinement_operational_preflight"]
            self.assertEqual(preflight["contract_version"], 1)
            self.assertEqual(preflight["status"], "pass")
            self.assertTrue(preflight["production_gate_eligible"])
            self.assertEqual(preflight["failures"], [])
            self.assertEqual(
                preflight["sample"]["measured_pairs"],
                preflight["sample"]["expected_pairs"],
            )
            self.assertEqual(
                checkpoint["refinement_operational_preflight"], preflight
            )
            preflight_path = output / preflight["result_path"]
            self.assertEqual(preflight["result_sha256"], sha256(preflight_path))
            self.assertEqual(
                (output / preflight["sidecar_path"]).read_text(encoding="utf-8"),
                f"{preflight['result_sha256']}  result.json\n",
            )
            self.assertEqual(
                set(manifest["identity"]["collection_completeness"]["buckets"]),
                {"fineweb_edu", "wikipedia"},
            )
            self.assertEqual(
                manifest["identity"]["collection_completeness"]["buckets"]
                ["fineweb_edu"]["report_totals"],
                manifest["identity"]["collection_completeness"]["buckets"]
                ["fineweb_edu"]["finalized_totals"],
            )
            self.assertEqual(
                (output / "manifest.sha256").read_text(encoding="utf-8").split()[0],
                sha256(output / "manifest.json"),
            )
            self.assertEqual(
                manifest["algorithm"]["compact_preprocess_sketch_role"],
                "validated-integrity-only-not-candidate-authority",
            )
            self.assertEqual(
                manifest["completeness_and_leakage_audit"]["mapping_missing_documents"], 0
            )
            self.assertGreaterEqual(
                manifest["completeness_and_leakage_audit"]["cross_source_clusters"], 1
            )

            mapping = {
                row["doc_id"]: row["cluster_id"]
                for row in iter_jsonl_zst(output / "clusters.jsonl.zst")
            }
            self.assertEqual(mapping[fixture.docs["exact_fine"]], mapping[fixture.docs["exact_wiki"]])
            self.assertEqual(mapping[fixture.docs["near_fine"]], mapping[fixture.docs["near_wiki"]])
            self.assertNotEqual(
                mapping[fixture.docs["unrelated_fine"]],
                mapping[fixture.docs["unrelated_wiki"]],
            )

            # A clean rebuild is mapping-byte deterministic.
            fresh = fixture.root / "english-near-fresh"
            with fixture.builder(fresh) as builder:
                fresh_result = builder.run()
            self.assertTrue(fresh_result["complete"])
            self.assertEqual(
                sha256(output / "clusters.jsonl.zst"), sha256(fresh / "clusters.jsonl.zst")
            )

            # A completed-run reopen revalidates the closed manifest, its
            # sidecar, mapping, identity, and operational-preflight authority.
            with fixture.builder(output) as builder:
                reopened = builder.run()
            self.assertTrue(reopened["complete"])
            manifest_path = output / "manifest.json"
            manifest_raw = manifest_path.read_bytes()
            manifest_path.write_bytes(manifest_raw + b" ")
            with self.assertRaisesRegex(
                EnglishNearDedupError, "manifest checksum changed"
            ):
                with fixture.builder(output) as builder:
                    builder.run()
            manifest_path.write_bytes(manifest_raw)

    def test_candidate_archive_lookup_uses_both_leading_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "candidate-query-plan"
            with fixture.builder(output) as builder:
                result = builder.run(stop_after_phase="candidates")
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "cache")
                archive = builder.reports[0].report["archive"]
                plan = "\n".join(
                    str(row[3])
                    for row in builder.db.execute(
                        "EXPLAIN QUERY PLAN "
                        + CANDIDATE_DOCUMENTS_FOR_ARCHIVE_SQL,
                        (archive, archive),
                    )
                )
                self.assertIn("documents_archive_ordinal", plan)
                self.assertIn("candidates_by_right", plan)
                self.assertNotIn("SCAN c", plan)
                archive_ordinals = {
                    int(row[0])
                    for row in builder.db.execute(
                        "SELECT ordinal FROM documents WHERE archive=?", (archive,)
                    )
                }
                expected: set[int] = set()
                for left, right in builder.db.execute(
                    "SELECT left_document,right_document FROM candidates"
                ):
                    if int(left) in archive_ordinals:
                        expected.add(int(left))
                    if int(right) in archive_ordinals:
                        expected.add(int(right))
                self.assertEqual(
                    builder._candidate_documents_for_archive(str(archive)),
                    expected,
                )

    def test_union_finalization_is_keyset_bounded_and_restart_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "union-finalization-resume"
            with fixture.builder(output, batch_size=1) as builder:
                result = builder.run(stop_after_phase="refine")
                self.assertEqual(result["phase"], "union")
                self.assertFalse(
                    builder.union_edges(max_new_finalization_documents=1)
                )
                flatten = json.loads(
                    builder.db.execute(
                        "SELECT value FROM metadata WHERE key='union_flatten_cursor'"
                    ).fetchone()[0]
                )
                self.assertEqual(flatten["processed_documents"], 1)
                self.assertGreater(flatten["parent_documents_total"], 1)
                self.assertIsNone(
                    builder.db.execute(
                        "SELECT value FROM metadata WHERE key='union_document_cursor'"
                    ).fetchone()
                )
                parent_plan = "\n".join(
                    str(row[3])
                    for row in builder.db.execute(
                        "EXPLAIN QUERY PLAN " + UNION_PARENT_BATCH_SQL, (0, 1)
                    )
                )
                document_plan = "\n".join(
                    str(row[3])
                    for row in builder.db.execute(
                        "EXPLAIN QUERY PLAN " + UNION_DOCUMENT_BATCH_SQL, (0, 1)
                    )
                )
                self.assertIn("SEARCH parents USING PRIMARY KEY", parent_plan)
                self.assertNotIn("SCAN parents", parent_plan)
                self.assertIn("SEARCH documents USING INTEGER PRIMARY KEY", document_plan)
                self.assertNotIn("SCAN documents", document_plan)

            # Every clean stop commits at most the requested number of rows
            # across the parent-flatten and document-root keyset passes.
            previous_total = 1
            union_complete = False
            for _attempt in range(32):
                with fixture.builder(output, batch_size=1) as builder:
                    union_complete = builder.union_edges(
                        max_new_finalization_documents=1
                    )
                    flatten_row = builder.db.execute(
                        "SELECT value FROM metadata WHERE key='union_flatten_cursor'"
                    ).fetchone()
                    document_row = builder.db.execute(
                        "SELECT value FROM metadata WHERE key='union_document_cursor'"
                    ).fetchone()
                    flattened = (
                        json.loads(flatten_row[0])["processed_documents"]
                        if flatten_row is not None
                        else 0
                    )
                    rooted = (
                        json.loads(document_row[0])["processed_documents"]
                        if document_row is not None
                        else 0
                    )
                    self.assertLessEqual(flattened + rooted - previous_total, 1)
                    previous_total = flattened + rooted
                    if union_complete:
                        self.assertEqual(builder._phase(), "emit")
                        event = json.loads(
                            builder.db.execute(
                                "SELECT payload FROM events WHERE event='emit'"
                            ).fetchone()[0]
                        )
                        gate = event["operational_rss_gate"]
                        self.assertEqual(gate["status"], "pass")
                        self.assertEqual(gate["failures"], [])
                        self.assertLessEqual(
                            gate["observed_peak_process_rss_bytes"],
                            gate["maximum_peak_process_rss_bytes"],
                        )
                        break
            self.assertTrue(union_complete)
            with fixture.builder(output, batch_size=1) as builder:
                result = builder.run()
            self.assertTrue(result["complete"])
            preflight_measurements = result["manifest"][
                "refinement_operational_preflight"
            ]["measurements"]
            self.assertEqual(preflight_measurements["union_parent_item_bytes"], 8)
            self.assertLessEqual(
                preflight_measurements[
                    "union_projected_peak_process_rss_bytes"
                ],
                result["manifest"]["refinement_operational_preflight"][
                    "thresholds"
                ]["maximum_peak_process_rss_bytes"],
            )

    def test_union_live_rss_gate_fails_before_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "union-live-rss-gate"
            with fixture.builder(output) as builder:
                result = builder.run(stop_after_phase="refine")
                self.assertEqual(result["phase"], "union")
                resources = builder._resource_snapshot()
                resources["peak_process_rss_bytes"] = (
                    int(
                        builder.config["operational_preflight"][
                            "maximum_peak_process_rss_bytes"
                        ]
                    )
                    + 1
                )
                with patch.object(
                    builder, "_resource_snapshot", return_value=resources
                ):
                    with self.assertRaisesRegex(
                        EnglishNearDedupError, "Union operational RSS gate failed"
                    ):
                        builder.union_edges()
                operational = json.loads(
                    (output / ".work" / "OPERATIONAL.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(operational["phases"]["union"]["status"], "fail")
                self.assertFalse((output / "clusters.jsonl.zst").exists())

    def test_many_candidate_blocks_and_singletons_use_tuple_cursor_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "candidate-tuple-cursor"
            duplicate_blocks = 0
            synthetic_bands: list[tuple[int, bytes, int]] = []
            for band in range(4):
                for key_number in range(1_000):
                    key = key_number.to_bytes(8, "big")
                    synthetic_bands.append((band, key, 1))
                    if key_number % 50 == 0:
                        synthetic_bands.append((band, key, 2))
                        duplicate_blocks += 1

            with fixture.builder(output, batch_size=31) as builder:
                result = builder.run(stop_after_phase="signatures")
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "candidates")
                with builder.db:
                    builder.db.execute("DELETE FROM bands")
                    builder.db.execute("DELETE FROM candidates")
                    builder.db.execute("DELETE FROM candidate_blocks")
                    builder.db.execute(
                        "DELETE FROM metadata WHERE key='candidate_cursor'"
                    )
                    builder.db.executemany(
                        "INSERT INTO bands VALUES (?,?,?)", synthetic_bands
                    )
                plan = "\n".join(
                    str(row[3])
                    for row in builder.db.execute(
                        "EXPLAIN QUERY PLAN " + NEXT_CANDIDATE_BLOCK_SQL,
                        (-1, b""),
                    )
                )
                self.assertIn("SEARCH bands USING PRIMARY KEY", plan)
                self.assertNotIn("SCAN bands", plan)
                self.assertFalse(builder.generate_candidates(max_new_blocks=17))
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM candidate_blocks"
                    ).fetchone()[0],
                    17,
                )

            with fixture.builder(output, batch_size=31) as builder:
                self.assertFalse(builder.generate_candidates(max_new_blocks=23))
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM candidate_blocks"
                    ).fetchone()[0],
                    40,
                )

            with fixture.builder(output, batch_size=31) as builder:
                self.assertTrue(builder.generate_candidates())
                self.assertEqual(builder._phase(), "cache")
                self.assertEqual(
                    builder.db.execute(
                        "SELECT COUNT(*) FROM candidate_blocks"
                    ).fetchone()[0],
                    duplicate_blocks,
                )
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                    1,
                )
                cursor = json.loads(
                    builder.db.execute(
                        "SELECT value FROM metadata WHERE key='candidate_cursor'"
                    ).fetchone()[0]
                )
                self.assertEqual(cursor["band"], 3)
                self.assertEqual(cursor["key"], (950).to_bytes(8, "big").hex())

    def test_large_refinement_cursor_range_scan_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "large-refinement-resume"
            pairs = [(index, 100_000 + index) for index in range(1, 20_001)]

            def fake_shingles(
                _document: int,
                _cache: collections.OrderedDict[int, array.array[int]],
                _maximum_cached: int = 256,
            ) -> array.array[int]:
                return array.array("Q", (1, 2, 3, 4))

            with fixture.builder(output, batch_size=257) as builder:
                result = builder.run(stop_after_phase="cache")
                self.assertFalse(result["complete"])
                self.assertEqual(result["phase"], "refine")
                with builder.db:
                    builder.db.execute("DELETE FROM candidates")
                    builder.db.execute("DELETE FROM refined")
                    builder.db.execute("DELETE FROM edges")
                    builder.db.execute(
                        "DELETE FROM metadata WHERE key='refinement_cursor'"
                    )
                    builder.db.executemany(
                        "INSERT INTO candidates VALUES (?,?)", pairs
                    )
                    cache_event = builder.db.execute(
                        "SELECT sequence,payload FROM events WHERE event='cache'"
                    ).fetchone()
                    cache_payload = json.loads(cache_event[1])
                    cache_payload["candidate_pairs"] = len(pairs)
                    builder.db.execute(
                        "UPDATE events SET payload=? WHERE sequence=?",
                        (
                            json.dumps(
                                cache_payload, sort_keys=True, separators=(",", ":")
                            ),
                            int(cache_event[0]),
                        ),
                    )
                plan = "\n".join(
                    str(row[3])
                    for row in builder.db.execute(
                        "EXPLAIN QUERY PLAN " + REFINEMENT_BATCH_SQL,
                        (-1, -1, 257),
                    )
                )
                self.assertIn("SEARCH candidates USING PRIMARY KEY", plan)
                self.assertNotIn("SCAN candidates", plan)
                with patch.object(
                    builder, "_load_cached_shingles", side_effect=fake_shingles
                ):
                    self.assertFalse(builder.refine_candidates(max_new_pairs=4_321))
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM refined").fetchone()[0],
                    4_321,
                )
                self.assertEqual(
                    json.loads(
                        builder.db.execute(
                            "SELECT value FROM metadata WHERE key='refinement_cursor'"
                        ).fetchone()[0]
                    ),
                    {
                        "left_document": 4_321,
                        "right_document": 104_321,
                        "processed_pairs": 4_321,
                    },
                )

            with fixture.builder(output, batch_size=257) as builder:
                with patch.object(
                    builder, "_load_cached_shingles", side_effect=fake_shingles
                ):
                    self.assertFalse(builder.refine_candidates(max_new_pairs=7_777))
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM refined").fetchone()[0],
                    12_098,
                )

            with fixture.builder(output, batch_size=257) as builder:
                with patch.object(
                    builder, "_load_cached_shingles", side_effect=fake_shingles
                ):
                    self.assertTrue(builder.refine_candidates())
                self.assertEqual(builder._phase(), "union")
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM refined").fetchone()[0],
                    len(pairs),
                )
                self.assertIsNone(
                    builder.db.execute(
                        """
                        SELECT 1 FROM candidates AS c
                        LEFT JOIN refined AS r USING(left_document,right_document)
                        WHERE r.left_document IS NULL LIMIT 1
                        """
                    ).fetchone()
                )

    def test_operational_preflight_fails_closed_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = NearFixture(root)
            failing_config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            failing_config["operational_preflight"][
                "maximum_peak_process_rss_bytes"
            ] = 1
            failing_path = root / "failing-operational-preflight.json"
            write_json(failing_path, failing_config)
            failed_output = root / "failed-operational-preflight"
            with fixture.builder(failed_output, failing_path) as builder:
                result = builder.run(stop_after_phase="cache")
                self.assertEqual(result["phase"], "refine")
                with self.assertRaisesRegex(
                    EnglishNearDedupError,
                    "Operational preflight status is not production-passing",
                ):
                    builder.refine_candidates(max_new_pairs=1)
            failed_result = json.loads(
                (
                    failed_output
                    / "operational-preflight-v1"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(failed_result["status"], "fail")
            self.assertIn("peak_rss_exceeds_limit", failed_result["failures"])
            self.assertEqual(
                builder.db_path.exists(),
                True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            output = fixture.root / "tampered-operational-preflight"
            with fixture.builder(output) as builder:
                result = builder.run(stop_after_phase="cache")
                self.assertEqual(result["phase"], "refine")
                self.assertFalse(builder.refine_candidates(max_new_pairs=1))
            result_path = output / "operational-preflight-v1" / "result.json"
            original = result_path.read_bytes()
            stale = json.loads(original)
            stale["identity"]["report_inventory_sha256"] = "0" * 64
            write_json(result_path, stale)
            write_sha256_sidecar(result_path)
            with self.assertRaisesRegex(
                EnglishNearDedupError,
                "Operational preflight identity is not production-passing",
            ):
                with fixture.builder(output):
                    pass
            result_path.write_bytes(original)
            write_sha256_sidecar(result_path)
            inconsistent_sample = json.loads(original)
            inconsistent_sample["sample"]["sample_pairs_sha256"] = "0" * 64
            write_json(result_path, inconsistent_sample)
            write_sha256_sidecar(result_path)
            with self.assertRaisesRegex(
                EnglishNearDedupError,
                "Operational preflight sample sample_pairs_sha256 mismatch",
            ):
                with fixture.builder(output):
                    pass
            result_path.write_bytes(original)
            write_sha256_sidecar(result_path)
            inconsistent_eta = json.loads(original)
            inconsistent_eta["measurements"][
                "projected_refinement_seconds"
            ] += 1.0
            write_json(result_path, inconsistent_eta)
            write_sha256_sidecar(result_path)
            with self.assertRaisesRegex(
                EnglishNearDedupError,
                "Operational preflight rate/ETA formulas are inconsistent",
            ):
                with fixture.builder(output):
                    pass
            result_path.write_bytes(original)
            write_sha256_sidecar(result_path)
            result_path.write_bytes(result_path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                EnglishNearDedupError, "Operational preflight checksum mismatch"
            ):
                with fixture.builder(output):
                    pass

    def test_live_refinement_resource_gate_rechecks_rss_and_disk(self) -> None:
        for resource_field, failure in (
            ("peak_process_rss_bytes", "live_refinement_peak_rss_exceeds_limit"),
            (
                "filesystem_free_bytes",
                "live_refinement_disk_projection_exceeds_free_space",
            ),
        ):
            with self.subTest(resource_field=resource_field), tempfile.TemporaryDirectory() as temporary:
                fixture = NearFixture(Path(temporary))
                output = fixture.root / f"live-refine-{resource_field}"
                with fixture.builder(output) as builder:
                    result = builder.run(stop_after_phase="cache")
                    self.assertEqual(result["phase"], "refine")
                    self.assertFalse(builder.refine_candidates(max_new_pairs=1))
                    resources = builder._resource_snapshot()
                    if resource_field == "peak_process_rss_bytes":
                        resources[resource_field] = (
                            int(
                                builder.config["operational_preflight"][
                                    "maximum_peak_process_rss_bytes"
                                ]
                            )
                            + 1
                        )
                    else:
                        resources[resource_field] = 0
                    with patch.object(
                        builder, "_resource_snapshot", return_value=resources
                    ):
                        with self.assertRaisesRegex(
                            EnglishNearDedupError,
                            "Live refinement operational gate failed",
                        ):
                            builder.refine_candidates(max_new_pairs=1)
                    operational = json.loads(
                        (output / ".work" / "OPERATIONAL.json").read_text(
                            encoding="utf-8"
                        )
                    )["phases"]["refine"]
                    self.assertEqual(operational["status"], "fail")
                    self.assertIn(failure, operational["failures"])
                    self.assertFalse((output / "clusters.jsonl.zst").exists())

    def test_posting_overflow_fails_closed_without_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            config["candidate_signature"]["maximum_posting_documents"] = 1
            config_path = fixture.root / "overflow-config.json"
            write_json(config_path, config)
            output = fixture.root / "overflow"
            with fixture.builder(output, config_path) as builder:
                with self.assertRaisesRegex(EnglishNearDedupError, "fail-closed limit"):
                    builder.run()
            self.assertTrue((output / ".work" / "FAILURE.json").is_file())
            self.assertFalse((output / "clusters.jsonl.zst").exists())

    def test_cumulative_candidate_overflow_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            config["candidate_signature"]["maximum_unique_candidate_pairs"] = 1
            config_path = fixture.root / "candidate-cap.json"
            write_json(config_path, config)
            output = fixture.root / "candidate-cap"
            with fixture.builder(output, config_path) as builder:
                with self.assertRaisesRegex(EnglishNearDedupError, "candidate pairs"):
                    builder.run()
            failure = json.loads(
                (output / ".work" / "FAILURE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["error"], "global_candidate_pair_overflow")
            self.assertFalse((output / "clusters.jsonl.zst").exists())

    def test_resume_rejects_changed_algorithm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NearFixture(Path(temporary))
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            config_path = fixture.root / "pinned-config.json"
            write_json(config_path, config)
            output = fixture.root / "identity"
            with fixture.builder(output, config_path) as builder:
                result = builder.run(max_new_inventory_archives=1)
                self.assertFalse(result["complete"])
            config["refinement"]["minimum_jaccard_numerator"] = 3
            write_json(config_path, config)
            with self.assertRaisesRegex(EnglishNearDedupError, "Resume identity mismatch"):
                with fixture.builder(output, config_path):
                    pass

    def test_raw_signature_is_deterministic_and_refinement_is_complete(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        base = " ".join(f"token{index}" for index in range(100))
        near = base.replace("token50", "replacement")
        first, first_count = candidate_bands(base, config)
        second, second_count = candidate_bands(base, config)
        self.assertEqual(first, second)
        self.assertEqual(first_count, second_count)
        self.assertTrue(set(first) & set(candidate_bands(near, config)[0]))
        left = exact_shingle_hashes(base, config["refinement"]["shingle_hash_seed"])
        right = exact_shingle_hashes(near, config["refinement"]["shingle_hash_seed"])
        intersection, union = jaccard_counts(left, right)
        self.assertGreaterEqual(intersection * 5, union * 4)


if __name__ == "__main__":
    unittest.main()
