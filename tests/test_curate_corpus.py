from __future__ import annotations

import array
import fcntl
import hashlib
import json
import math
import queue
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_guard import BenchmarkGuard
from build_english_near_clusters import EnglishNearDedupBuilder
import curate_corpus as curate_corpus_module
from curate_corpus import (
    BENCHMARK_CONTENT_CLUSTER_SQL,
    BENCHMARK_FINAL_CLUSTER_SQL,
    CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT,
    CURATION_STORAGE_PREFLIGHT_VERSION,
    CurationBuilder,
    CurationError,
    QUOTA_CANDIDATE_AFTER_SQL,
    QUOTA_CANDIDATE_SQL,
    QUOTA_SELECTION_INDEX,
    assign_group_split,
    detect_output_mount,
    english_source_identity,
    file_sha256,
    iter_merged_quota_candidates,
    iter_jsonl_zst,
    select_sqlite_journal_policy,
    split_thresholds,
    stable_digest,
)
from curation_policy import (
    DEFAULT_POLICY,
    FAST_CANONICAL_POLICY,
    FAST_CANONICAL_PROFILE,
    load_policy,
)
from preprocess_raw_stream import FINGERPRINT_VERSION, POLICY_DESCRIPTOR, POLICY_SHA256
from quota_tracker import write_record


TOKENIZER_REVISION = "1" * 40
FINEWEB_REVISION = "2" * 40
WIKIPEDIA_REVISION = "3" * 40


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(path: Path, manifest: dict[str, object]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    (path.parent / "manifest.sha256").write_bytes(
        hashlib.sha256(payload).hexdigest().encode("ascii") + b"  manifest.json\n"
    )


def make_row(
    archive: str,
    bucket: str,
    index: int,
    member_index: int,
    *,
    content_key: str,
    normalized_key: str | None = None,
    tokens: int = 3,
    repo_id: str | None = None,
    url: str | None = None,
    article_id: str | None = None,
    flags: list[str] | None = None,
    benchmark_reason: str | None = None,
) -> dict[str, object]:
    member_path = f"files/{member_index:08d}.txt"
    doc_id = hashlib.sha256(f"{archive}\0{member_path}".encode()).hexdigest()
    provenance: dict[str, object] = {}
    if repo_id is not None:
        provenance.update(repo_id=repo_id, repo_path=repo_id, language="Python" if bucket == "python" else "Rust")
    if url is not None:
        provenance["url"] = url
    if article_id is not None:
        provenance["id"] = article_id
    quality_flags = sorted(set(flags or []))
    if benchmark_reason:
        quality_flags = sorted(set([*quality_flags, "benchmark_contamination"]))
    return {
        "record_version": 1,
        "fingerprint_version": FINGERPRINT_VERSION,
        "doc_id": doc_id,
        "bucket": bucket,
        "archive": archive,
        "archive_index": index,
        "manifest_index": member_index,
        "member_path": member_path,
        "size_bytes": 100 + member_index,
        "starcoder2_tokens": tokens,
        "content_sha256": hashlib.sha256(f"content:{content_key}".encode()).hexdigest(),
        "normalized_sha256": hashlib.sha256(
            f"normalized:{normalized_key or content_key}".encode()
        ).hexdigest(),
        "near_sketch": [],
        "metrics": {},
        "quality_flags": quality_flags,
        "benchmark_reason": benchmark_reason,
        "provenance": provenance,
    }


class CorpusFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging = root / "staging" / "preprocess"
        self.policy = load_policy(DEFAULT_POLICY)
        self.guard = BenchmarkGuard(PROJECT_ROOT / "configs" / "mbpp_denylist.json")
        self.quotas = {(split, category): 2 for split in ("train", "validation", "test") for category in ("python", "other_code", "english")}
        self.collection_targets = {
            "python": 1,
            "other_code": 1,
            "fineweb_edu": 1,
            "wikipedia": 1,
        }
        self.quota_path = root / "tiny-quotas.json"
        write_json(
            self.quota_path,
            {
                "version": 1,
                "quotas": [
                    {
                        "name": f"collection/{bucket}",
                        "phase": "collection",
                        "category": (
                            bucket if bucket in ("python", "other_code") else "english"
                        ),
                        **(
                            {"language_group": bucket}
                            if bucket in ("fineweb_edu", "wikipedia")
                            else {}
                        ),
                        "token_field": "exact_tokens",
                        "target": target,
                    }
                    for bucket, target in sorted(self.collection_targets.items())
                ]
                + [
                    {
                        "name": "collection/english",
                        "phase": "collection",
                        "category": "english",
                        "token_field": "exact_tokens",
                        "target": (
                            self.collection_targets["fineweb_edu"]
                            + self.collection_targets["wikipedia"]
                        ),
                    }
                ]
                + [
                    {
                        "name": f"final/{split}/{category}",
                        "phase": "final",
                        "split": split,
                        "category": category,
                        "token_field": "exact_tokens",
                        "target": target,
                    }
                    for (split, category), target in sorted(self.quotas.items())
                ],
            },
        )
        self._write_identities()
        self.rows_by_archive: dict[tuple[str, int], list[dict[str, object]]] = {}
        self.quota_record_paths: dict[tuple[str, int], Path] = {}
        self.near_clusters: dict[str, str] = {}
        self._make_rows()
        self._write_inputs()

    def _write_identities(self) -> None:
        tokenizer_dir = self.root / "tokenizer" / "starcoder2"
        tokenizer_dir.mkdir(parents=True)
        tokenizer_payload = b'{"fixture":true}\n'
        (tokenizer_dir / "tokenizer.json").write_bytes(tokenizer_payload)
        write_json(
            tokenizer_dir / "TOKENIZER_MANIFEST.json",
            {
                "manifest_version": 1,
                "repo_id": "bigcode/starcoder2-tokenizer",
                "resolved_revision": TOKENIZER_REVISION,
                "files": {
                    "tokenizer.json": {
                        "bytes": len(tokenizer_payload),
                        "sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
                    }
                },
                "validation": {"vocab_size": 49_152},
            },
        )
        trusted = self.policy["trusted_code_source"]
        write_json(
            self.root / "manifests" / "STACK_V3_SOURCE.json",
            {
                "manifest_version": 3,
                "repo_id": trusted["repo_id"],
                "resolved_revision": trusted["resolved_revision"],
                "source_shard_count": trusted["source_shard_count"],
                "source_shards_sha256": trusted["source_shards_sha256"],
            },
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

    def _repo_for_split(self, split: str) -> str:
        thresholds = split_thresholds(self.quotas)
        for number in range(100_000):
            repo = f"repo-{split}-{number}"
            group = stable_digest("code-repository", repo)
            if assign_group_split(self.policy["selection"]["seed"], group, thresholds) == split:
                return repo
        raise AssertionError(f"could not find repo for {split}")

    def _url_for_split(self, split: str) -> str:
        thresholds = split_thresholds(self.quotas)
        for number in range(100_000):
            url = f"https://example.test/{split}/{number}"
            identity = f"fineweb_edu:url:{url}"
            group = stable_digest("english-source", identity)
            if assign_group_split(self.policy["selection"]["seed"], group, thresholds) == split:
                return url
        raise AssertionError(f"could not find URL for {split}")

    def _append(self, bucket: str, archive_index: int, **kwargs: object) -> dict[str, object]:
        archive = {
            "python": f"raw/python/part-{archive_index:06d}.tar.zst",
            "other_code": f"raw/other_code/part-{archive_index:06d}.tar.zst",
            "fineweb_edu": f"raw/english/fineweb_edu/part-{archive_index:06d}.tar.zst",
            "wikipedia": f"raw/english/wikipedia/part-{archive_index:06d}.tar.zst",
        }[bucket]
        rows = self.rows_by_archive.setdefault((bucket, archive_index), [])
        row = make_row(archive, bucket, archive_index, len(rows), **kwargs)
        rows.append(row)
        return row

    def _make_rows(self) -> None:
        # Every split has one globally grouped repository containing both code
        # domains. These are the quota candidates used to prove cross-bucket
        # repository grouping.
        for split in ("train", "validation", "test"):
            repo = self._repo_for_split(split)
            self._append("python", 0, content_key=f"py-{split}", repo_id=repo)
            self._append("other_code", 0, content_key=f"other-{split}", repo_id=repo)
            url = self._url_for_split(split)
            row = self._append("fineweb_edu", 0, content_key=f"english-{split}", url=url)
            self.near_clusters[str(row["doc_id"])] = f"unique-{row['doc_id']}"

        # Exact duplicates cross both archives and code buckets; normalized
        # duplicates are byte-distinct. Only one canonical may survive each
        # global residual cluster.
        exact_a = self._append("python", 0, content_key="exact-pair", repo_id="repo-extra-a")
        exact_b = self._append("other_code", 1, content_key="exact-pair", repo_id="repo-extra-b")
        norm_a = self._append("python", 0, content_key="norm-a", normalized_key="norm-pair", repo_id="repo-extra-a")
        norm_b = self._append("python", 1, content_key="norm-b", normalized_key="norm-pair", repo_id="repo-extra-b")

        # A benchmark hit contaminates its whole normalized cluster, including
        # a row that was not directly detected.
        benchmark = self._append(
            "python",
            0,
            content_key="benchmark-a",
            normalized_key="benchmark-cluster",
            repo_id="repo-benchmark",
            benchmark_reason="mbpp-exact-code",
        )
        benchmark_exact_copy = self._append(
            "other_code",
            1,
            content_key="benchmark-a",
            normalized_key="benchmark-cluster",
            repo_id="repo-benchmark-exact-copy",
        )
        benchmark_copy = self._append(
            "python",
            1,
            content_key="benchmark-b",
            normalized_key="benchmark-cluster",
            repo_id="repo-benchmark-copy",
        )
        quality = self._append(
            "other_code", 0, content_key="quality-reject", repo_id="repo-quality", flags=["too_short"]
        )

        # Cross-source English near duplicate. The complete mapping covers all
        # English documents and picks at most one canonical from this cluster.
        fine = self._append("fineweb_edu", 0, content_key="fine-near", url="https://near.test/a")
        wiki = self._append("wikipedia", 0, content_key="wiki-near", article_id="42")
        for row in (fine, wiki):
            self.near_clusters[str(row["doc_id"])] = "cross-source-near"
        # Ensure all other English rows have singleton cluster identities.
        for (bucket, _index), rows in self.rows_by_archive.items():
            if bucket in ("fineweb_edu", "wikipedia"):
                for row in rows:
                    self.near_clusters.setdefault(str(row["doc_id"]), f"unique-{row['doc_id']}")

        self.special = {
            "exact": {str(exact_a["doc_id"]), str(exact_b["doc_id"])},
            "normalized": {str(norm_a["doc_id"]), str(norm_b["doc_id"])},
            "benchmark": str(benchmark["doc_id"]),
            "benchmark_exact_copy": str(benchmark_exact_copy["doc_id"]),
            "benchmark_copy": str(benchmark_copy["doc_id"]),
            "quality": str(quality["doc_id"]),
            "near": {str(fine["doc_id"]), str(wiki["doc_id"])},
        }

    def _source(self, bucket: str) -> str:
        trusted = self.policy["trusted_code_source"]
        return {
            "python": f"{trusted['repo_id']}@{trusted['resolved_revision']}",
            "other_code": f"{trusted['repo_id']}@{trusted['resolved_revision']}",
            "fineweb_edu": f"HuggingFaceFW/fineweb-edu@{FINEWEB_REVISION}#sample-10BT",
            "wikipedia": f"wikimedia/wikipedia@{WIKIPEDIA_REVISION}#20231101.en",
        }[bucket]

    def _write_inputs(self) -> None:
        for (bucket, archive_index), rows in self.rows_by_archive.items():
            fingerprint = self.staging / "fingerprints" / bucket / f"part-{archive_index:06d}.jsonl.zst"
            fingerprint.parent.mkdir(parents=True, exist_ok=True)
            with fingerprint.open("wb") as raw:
                raw.write(zstandard.ZstdCompressor(level=1, write_checksum=True).compress(
                    b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
                ))
            archive = str(rows[0]["archive"])
            raw_archive = self.root / archive
            raw_archive.parent.mkdir(parents=True, exist_ok=True)
            raw_archive.write_bytes(
                (f"fixture:{bucket}:{archive_index}\n".encode("utf-8") * 64)
            )
            quota_shard_id = f"fixture-{bucket}-{archive_index:06d}"
            quota_record = {
                "phase": "collection",
                "category": (
                    bucket if bucket in ("python", "other_code") else "english"
                ),
                **(
                    {"language_group": bucket}
                    if bucket in ("fineweb_edu", "wikipedia")
                    else {}
                ),
                "shard_id": quota_shard_id,
                "source": self._source(bucket),
                "documents": len(rows),
                "clean_bytes": sum(int(row["size_bytes"]) for row in rows),
                "exact_tokens": sum(int(row["starcoder2_tokens"]) for row in rows),
            }
            quota_path, _written = write_record(self.root, quota_record)
            self.quota_record_paths[(bucket, archive_index)] = quota_path
            report = {
                "report_version": 1,
                "fingerprint_version": FINGERPRINT_VERSION,
                "policy_sha256": POLICY_SHA256,
                "archive": archive,
                "archive_sha256": file_sha256(raw_archive),
                "archive_compressed_bytes": raw_archive.stat().st_size,
                "bucket": bucket,
                "index": archive_index,
                "quota_shard_id": quota_shard_id,
                "source": self._source(bucket),
                "fingerprint_file": str(fingerprint.relative_to(self.staging)),
                "fingerprint_sha256": file_sha256(fingerprint),
                "documents": len(rows),
                "clean_bytes": sum(int(row["size_bytes"]) for row in rows),
                "exact_tokens": sum(int(row["starcoder2_tokens"]) for row in rows),
                "benchmark_hits": sum(bool(row["benchmark_reason"]) for row in rows),
                "quality_flag_counts": {},
                "language_documents": {},
                "language_tokens": {},
            }
            write_json(
                self.staging / "reports" / bucket / f"part-{archive_index:06d}.json", report
            )
        totals = {
            bucket: sum(
                sum(int(row["starcoder2_tokens"]) for row in rows)
                for (row_bucket, _index), rows in self.rows_by_archive.items()
                if row_bucket == bucket
            )
            for bucket in self.collection_targets
        }
        write_json(
            self.root / "state" / "COLLECTION_COMPLETE.json",
            {
                "source": self._source("python"),
                "benchmark_guard_sha256": self.guard.manifest_sha256,
                "python_tokens": totals["python"],
                "other_code_tokens": totals["other_code"],
                "targets": {
                    bucket: self.collection_targets[bucket]
                    for bucket in ("python", "other_code")
                },
            },
        )
        for bucket, filename in (
            ("fineweb_edu", "ENGLISH_FINEWEB_EDU_COMPLETE.json"),
            ("wikipedia", "ENGLISH_WIKIPEDIA_COMPLETE.json"),
        ):
            write_json(
                self.root / "state" / filename,
                {
                    "source": self._source(bucket),
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "benchmark_guard_sha256": self.guard.manifest_sha256,
                    "english_tokens": totals[bucket],
                    "target": self.collection_targets[bucket],
                },
            )
        near_output = self.root / "english-near-v1"
        near_output.mkdir(parents=True, exist_ok=True)
        self.near_path = near_output / "clusters.jsonl.zst"
        payload = b"".join(
            json.dumps({"doc_id": doc_id, "cluster_id": cluster}, sort_keys=True).encode() + b"\n"
            for doc_id, cluster in sorted(self.near_clusters.items())
        )
        self.near_path.write_bytes(zstandard.ZstdCompressor(level=1).compress(payload))
        self._write_near_manifest()

    def _write_near_manifest(self) -> None:
        config_path = curate_corpus_module.DEFAULT_ENGLISH_NEAR_CONFIG
        builder_path = curate_corpus_module.ENGLISH_NEAR_BUILDER
        config_raw = config_path.read_bytes()
        config = json.loads(config_raw)
        config_sha = hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in config.items() if not key.startswith("_")}
            )
        ).hexdigest()
        quota_records = []
        for (bucket, index), path in sorted(self.quota_record_paths.items()):
            if bucket not in ("fineweb_edu", "wikipedia"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            quota_records.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "sha256": file_sha256(path),
                    "shard_id": record["shard_id"],
                    "bucket": bucket,
                    "index": index,
                    "documents": record["documents"],
                    "clean_bytes": record["clean_bytes"],
                    "exact_tokens": record["exact_tokens"],
                    "source": record["source"],
                }
            )
        report_inputs = []
        collection_buckets = {}
        for bucket, marker_name in (
            ("fineweb_edu", "ENGLISH_FINEWEB_EDU_COMPLETE.json"),
            ("wikipedia", "ENGLISH_WIKIPEDIA_COMPLETE.json"),
        ):
            bucket_records = [row for row in quota_records if row["bucket"] == bucket]
            bucket_reports = []
            for path in sorted((self.staging / "reports" / bucket).glob("part-*.json")):
                report = json.loads(path.read_text(encoding="utf-8"))
                bucket_reports.append(report)
                report_inputs.append(
                    {
                        "report_path": str(path.relative_to(self.staging)),
                        "report_sha256": file_sha256(path),
                        "archive": report["archive"],
                        "archive_sha256": report["archive_sha256"],
                        "fingerprint_file": report["fingerprint_file"],
                        "fingerprint_sha256": report["fingerprint_sha256"],
                        "documents": report["documents"],
                    }
                )
            totals = {
                "archives": len(bucket_records),
                "documents": sum(int(row["documents"]) for row in bucket_records),
                "clean_bytes": sum(int(row["clean_bytes"]) for row in bucket_records),
                "exact_tokens": sum(int(row["exact_tokens"]) for row in bucket_records),
            }
            report_totals = {
                "archives": len(bucket_reports),
                "documents": sum(int(row["documents"]) for row in bucket_reports),
                "clean_bytes": sum(int(row["clean_bytes"]) for row in bucket_reports),
                "exact_tokens": sum(int(row["exact_tokens"]) for row in bucket_reports),
            }
            marker_path = self.root / "state" / marker_name
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            collection_buckets[bucket] = {
                "target_exact_tokens": self.collection_targets[bucket],
                "completion_marker": {
                    "path": str(marker_path.relative_to(self.root)),
                    "sha256": file_sha256(marker_path),
                    **marker,
                },
                "finalized_totals": totals,
                "archive_indices_sha256": hashlib.sha256(
                    canonical_json_bytes(sorted(int(row["index"]) for row in bucket_records))
                ).hexdigest(),
                "report_totals": report_totals,
            }
        collection = {
            "evidence_version": 1,
            "quota_config_path": str(self.quota_path),
            "quota_config_sha256": file_sha256(self.quota_path),
            "quota_record_inventory_sha256": hashlib.sha256(
                canonical_json_bytes(quota_records)
            ).hexdigest(),
            "quota_records": quota_records,
            "buckets": collection_buckets,
        }
        report_projection = [
            {
                "path": row["report_path"],
                "sha256": row["report_sha256"],
                "fingerprint_sha256": row["fingerprint_sha256"],
            }
            for row in report_inputs
        ]
        report_inventory_sha = hashlib.sha256(
            canonical_json_bytes(report_projection)
        ).hexdigest()
        source_manifests = {
            "TOKENIZER_MANIFEST.json": {
                "sha256": file_sha256(
                    self.root / "tokenizer" / "starcoder2" / "TOKENIZER_MANIFEST.json"
                ),
                "resolved_revision": TOKENIZER_REVISION,
            },
            "FINEWEB_EDU_SOURCE.json": {
                "sha256": file_sha256(self.root / "manifests" / "FINEWEB_EDU_SOURCE.json"),
                "resolved_revision": FINEWEB_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
            "WIKIPEDIA_SOURCE.json": {
                "sha256": file_sha256(self.root / "manifests" / "WIKIPEDIA_SOURCE.json"),
                "resolved_revision": WIKIPEDIA_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
        }
        storage = {
            "filesystem_type": "ext4",
            "mount_point": "/fixture",
            "mount_source": "/dev/fixture",
            "mount_options": "rw",
            "detection": "fixture",
            "classification": "proven-local",
            "sqlite_journal_mode_configured": config["storage"]["sqlite_journal_mode"],
            "sqlite_journal_mode_requested": config["storage"]["sqlite_journal_mode"],
            "sqlite_journal_mode_request_source": "config",
            "sqlite_journal_mode_selected": "wal",
            "sqlite_journal_mode_actual": "wal",
            "policy": {
                "network_or_unknown_action": "delete",
                "wal_on_non_allowlisted_action": "fail_closed",
                "wal_local_filesystem_allowlist": config["storage"][
                    "wal_local_filesystem_allowlist"
                ],
            },
        }
        calibration_config_path = (
            curate_corpus_module.DEFAULT_ENGLISH_NEAR_CALIBRATION_CONFIG
        )
        calibration_config_raw = calibration_config_path.read_bytes()
        calibration_config = json.loads(calibration_config_raw)
        sample_manifest = [
            {
                "doc_id": hashlib.sha256(b"fixture-calibration-document").hexdigest(),
                "bucket": "fineweb_edu",
            }
        ]
        sampling_counts = {}
        for bucket in ("fineweb_edu", "wikipedia"):
            documents = sum(
                int(row["documents"])
                for row in report_inputs
                if f"/{bucket}/" in str(row["archive"])
            )
            sampling_counts[bucket] = {
                "fingerprint_rows_scanned": documents,
                "eligible_fingerprint_candidates": documents,
                "documents_selected": documents,
            }
        documents_selected = sum(
            row["documents_selected"] for row in sampling_counts.values()
        )
        calibration_input = {
            "kind": "immutable_real_english_sample",
            "full_report_inventory_sha256": report_inventory_sha,
            "preprocess_manifest_sha256": file_sha256(
                self.staging / "PREPROCESS_MANIFEST.json"
            ),
            "curation_policy_sha256": hashlib.sha256(
                canonical_json_bytes(self.policy)
            ).hexdigest(),
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "source_manifests": source_manifests,
            "collection_completeness_sha256": hashlib.sha256(
                canonical_json_bytes(collection)
            ).hexdigest(),
            "collection_completeness": collection,
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
            "sampling_counts": sampling_counts,
            "selected_reports": report_inputs,
            "documents_selected": documents_selected,
        }
        calibration_identity = {
            "harness_sha256": file_sha256(
                curate_corpus_module.ENGLISH_NEAR_CALIBRATION
            ),
            "production_builder_sha256": file_sha256(builder_path),
            "calibration_algorithm": calibration_config["calibration_algorithm"],
            "calibration_seed": calibration_config["seed"],
            "production_config_file_sha256": hashlib.sha256(config_raw).hexdigest(),
            "production_config_canonical_sha256": config_sha,
            "calibration_config_file_sha256": hashlib.sha256(
                calibration_config_raw
            ).hexdigest(),
            "calibration_config_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(calibration_config)
            ).hexdigest(),
            "input": calibration_input,
            "sample_manifest_sha256": hashlib.sha256(
                canonical_json_bytes(sample_manifest)
            ).hexdigest(),
            "runtime": {
                "python": "fixture-python",
                "xxhash": "fixture-xxhash",
                "zstandard": zstandard.__version__,
            },
        }
        calibration_result = {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "identity": calibration_identity,
            "production_threshold": {
                "minimum_jaccard_numerator": config["refinement"][
                    "minimum_jaccard_numerator"
                ],
                "minimum_jaccard_denominator": config["refinement"][
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
                "sample_manifest": sample_manifest,
            },
        }
        calibration_path = self.root / "audits/english-near-calibration-v1.json"
        write_json(calibration_path, calibration_result)
        calibration_raw = calibration_path.read_bytes()
        calibration_sha = hashlib.sha256(calibration_raw).hexdigest()
        sidecar_path = calibration_path.with_name(calibration_path.name + ".sha256")
        sidecar_path.write_text(
            f"{calibration_sha}  {calibration_path.name}\n", encoding="utf-8"
        )
        calibration_evidence = {
            "contract_version": 1,
            "result_path": str(calibration_path.relative_to(self.root)),
            "result_sha256": calibration_sha,
            "result_bytes": len(calibration_raw),
            "sidecar_path": str(sidecar_path.relative_to(self.root)),
            "sidecar_sha256": file_sha256(sidecar_path),
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_failures": [],
            "identity_sha256": hashlib.sha256(
                canonical_json_bytes(calibration_identity)
            ).hexdigest(),
            "identity": calibration_identity,
        }
        identity_builder = EnglishNearDedupBuilder(
            root=self.root,
            staging_root=self.staging,
            output=self.root / ".near-identity-contract-not-created",
            config_path=config_path,
            policy_path=DEFAULT_POLICY,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            quota_config_path=self.quota_path,
            calibration_result_path=calibration_path,
            batch_size=2,
        )
        with identity_builder as entered_builder:
            identity = json.loads(json.dumps(entered_builder.identity))
        self.near_builder_identity = identity
        cluster_sizes: dict[str, int] = {}
        cluster_buckets: dict[str, set[str]] = {}
        english_docs = 0
        for (bucket, _index), rows in self.rows_by_archive.items():
            if bucket not in ("fineweb_edu", "wikipedia"):
                continue
            for row in rows:
                cluster = self.near_clusters[str(row["doc_id"])]
                cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
                cluster_buckets.setdefault(cluster, set()).add(bucket)
                english_docs += 1
        mapping_sha = file_sha256(self.near_path)
        thresholds = config["operational_preflight"]
        total_candidates = 10
        expected_pairs = min(thresholds["requested_pairs"], total_candidates)
        elapsed = 0.01
        rate = round(expected_pairs / elapsed, 6)
        measured_growth = 100
        bytes_per_pair = measured_growth / expected_pairs
        projected_growth = math.ceil(bytes_per_pair * total_candidates)
        projected_safety = math.ceil(
            projected_growth
            * thresholds["disk_projection_safety_numerator"]
            / thresholds["disk_projection_safety_denominator"]
        )
        required_free = (
            projected_safety + thresholds["minimum_post_refinement_free_bytes"]
        )
        cache_bytes = 1024
        resources_before = {
            "filesystem_total_bytes": required_free * 2,
            "filesystem_free_bytes": required_free + 1024,
            "sqlite_state_bytes": 4096,
            "refinement_cache_bytes": cache_bytes,
            "peak_process_rss_bytes": 1024,
        }
        resources_after = {
            **resources_before,
            "sqlite_state_bytes": 4196,
            "peak_process_rss_bytes": 2048,
        }
        item_size = array.array("Q").itemsize
        parent_bytes = (english_docs + 1) * item_size
        parent_with_safety = math.ceil(
            parent_bytes
            * thresholds["union_parent_memory_safety_numerator"]
            / thresholds["union_parent_memory_safety_denominator"]
        )
        preflight_identity = {
            "contract_version": 1,
            "builder_sha256": identity["builder_sha256"],
            "builder_identity_sha256": hashlib.sha256(
                canonical_json_bytes(identity)
            ).hexdigest(),
            "config_file_sha256": identity["config_file_sha256"],
            "config_sha256": identity["config_sha256"],
            "calibration_evidence_sha256": hashlib.sha256(
                canonical_json_bytes(identity["calibration_evidence"])
            ).hexdigest(),
            "report_inventory_sha256": identity["report_inventory_sha256"],
            "preprocess_manifest_sha256": identity["preprocess_manifest_sha256"],
            "curation_policy_sha256": identity["curation_policy_sha256"],
            "benchmark_guard_sha256": identity["benchmark_guard_sha256"],
            "collection_completeness_sha256": hashlib.sha256(
                canonical_json_bytes(identity["collection_completeness"])
            ).hexdigest(),
            "candidate_pairs_total": total_candidates,
            "documents_total": english_docs,
            "candidate_blocks_total": 1,
            "candidate_blocks_committed": 1,
            "phase_at_measurement": "refine",
            "candidate_cursor_at_measurement": {"band": 0, "key": "00" * 8},
            "refinement_cursor_at_measurement": None,
            "cache_archives": identity["report_count"],
            "cache_bytes": cache_bytes,
            "cache_inventory_sha256": hashlib.sha256(
                b"fixture-cache-inventory"
            ).hexdigest(),
            "runtime_storage": identity["runtime"]["storage"],
        }
        sample = {
            "algorithm": thresholds["sampling_algorithm"],
            "seed": thresholds["sampling_seed"],
            "requested_pairs": thresholds["requested_pairs"],
            "expected_pairs": expected_pairs,
            "measured_pairs": expected_pairs,
            "sample_pairs_sha256": hashlib.sha256(
                canonical_json_bytes(
                    [
                        {"left_document": index, "right_document": index + 1}
                        for index in range(expected_pairs)
                    ]
                )
            ).hexdigest(),
            "accepted_pairs": 0,
            "sample_limited_by_total_candidates": True,
        }
        measurements = {
            "sample_elapsed_seconds": elapsed,
            "measurement_batches": math.ceil(expected_pairs / 2),
            "measurement_batch_size": 2,
            "refinement_pairs_per_second": rate,
            "candidate_pairs_total": total_candidates,
            "projected_refinement_seconds": round(total_candidates / rate, 3),
            "measurement_sqlite_growth_bytes": measured_growth,
            "measurement_sqlite_bytes_per_pair": round(bytes_per_pair, 6),
            "projected_additional_refinement_sqlite_bytes": projected_growth,
            "projected_additional_with_safety_bytes": projected_safety,
            "required_filesystem_free_bytes": required_free,
            "union_parent_item_bytes": item_size,
            "union_parent_array_projected_bytes": parent_bytes,
            "union_parent_array_with_safety_bytes": parent_with_safety,
            "union_projected_peak_process_rss_bytes": (
                resources_before["peak_process_rss_bytes"] + parent_with_safety
            ),
            "resources_before": resources_before,
            "resources_after": resources_after,
        }
        preflight_result = {
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
            "identity": preflight_identity,
            "thresholds": thresholds,
            "sample": sample,
            "measurements": measurements,
            "statistical_scope": "fixture bounded operational evidence",
        }
        preflight_dir = self.near_path.parent / "operational-preflight-v1"
        preflight_path = preflight_dir / "result.json"
        write_json(preflight_path, preflight_result)
        preflight_raw = preflight_path.read_bytes()
        preflight_sha = hashlib.sha256(preflight_raw).hexdigest()
        preflight_sidecar = preflight_dir / "result.json.sha256"
        preflight_sidecar.write_text(
            f"{preflight_sha}  result.json\n", encoding="ascii"
        )
        preflight_evidence = {
            "contract_version": 1,
            "result_path": "operational-preflight-v1/result.json",
            "result_sha256": preflight_sha,
            "result_bytes": len(preflight_raw),
            "sidecar_path": "operational-preflight-v1/result.json.sha256",
            "sidecar_sha256": file_sha256(preflight_sidecar),
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
            "identity_sha256": hashlib.sha256(
                canonical_json_bytes(preflight_identity)
            ).hexdigest(),
            "identity": preflight_identity,
            "thresholds": thresholds,
            "sample": sample,
            "measurements": measurements,
        }
        manifest = {
            "manifest_version": 1,
            "mapping_record_version": 1,
            "production_ready": True,
            "identity": identity,
            "refinement_operational_preflight": preflight_evidence,
            "algorithm": {
                "config": str(config_path),
                "config_file_sha256": identity["config_file_sha256"],
                "config_sha256": identity["config_sha256"],
                "name": curate_corpus_module.ENGLISH_NEAR_ALGORITHM,
                "raw_text_candidate_pass": True,
                "full_shingle_refinement": True,
                "posting_overflow_action": "fail_closed_without_truncation",
            },
            "inputs": {
                "report_inventory_sha256": report_inventory_sha,
                "reports": report_inputs,
            },
            "completeness_and_leakage_audit": {
                "english_documents_inventory": english_docs,
                "english_documents_mapped": english_docs,
                "mapping_missing_documents": 0,
                "mapping_unknown_documents": 0,
                "mapping_duplicate_documents": 0,
                "clusters": len(cluster_sizes),
                "singleton_clusters": sum(size == 1 for size in cluster_sizes.values()),
                "cross_source_clusters": sum(
                    len(buckets) > 1 for buckets in cluster_buckets.values()
                ),
                "normalized_hashes_in_multiple_clusters": 0,
                "invalid_cluster_roots": 0,
            },
            "database_integrity_check": "ok",
            "mapping": {
                "path": "clusters.jsonl.zst",
                "sha256": mapping_sha,
                "bytes": self.near_path.stat().st_size,
                "records": english_docs,
                "ordered_by": "frozen_inventory_ordinal",
                "singleton_clusters_included": True,
            },
        }
        sign_manifest(self.near_path.parent / "manifest.json", manifest)

    def build(
        self,
        output: Path,
        max_new_archives: int | None = None,
        sqlite_journal_mode: str = "auto",
    ) -> dict[str, object]:
        with CurationBuilder(
            root=self.root,
            staging_root=self.staging,
            output=output,
            policy_path=DEFAULT_POLICY,
            quota_path=self.quota_path,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            english_near_clusters=self.near_path,
            allow_missing_english_near_dedup=False,
            batch_size=3,
            sqlite_journal_mode=sqlite_journal_mode,
        ) as builder:
            return builder.run(max_new_archives=max_new_archives)

    def build_fast(
        self, output: Path, max_new_archives: int | None = None
    ) -> dict[str, object]:
        with CurationBuilder(
            root=self.root,
            staging_root=self.staging,
            output=output,
            policy_path=FAST_CANONICAL_POLICY,
            quota_path=self.quota_path,
            denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            english_near_clusters=None,
            allow_missing_english_near_dedup=False,
            batch_size=3,
        ) as builder:
            return builder.run(max_new_archives=max_new_archives)


class CurateCorpusTest(unittest.TestCase):
    def test_verified_compressed_jsonl_authenticates_consumed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl.zst"
            payload = b'{"row":1}\n{"row":2}\n'
            path.write_bytes(zstandard.ZstdCompressor().compress(payload))
            checksum = file_sha256(path)

            self.assertEqual(
                list(iter_jsonl_zst(path, expected_sha256=checksum)),
                [{"row": 1}, {"row": 2}],
            )
            with self.assertRaisesRegex(CurationError, "checksum mismatch"):
                list(iter_jsonl_zst(path, expected_sha256="0" * 64))

    def test_cross_client_lease_is_atomic_exclusive_and_releasable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curated"
            output.mkdir()
            lease, owner, lock = curate_corpus_module.acquire_cross_client_lease(output)
            self.assertTrue(lease.is_file())
            self.assertEqual(
                json.loads(lease.read_text(encoding="utf-8")), owner
            )
            self.assertFalse(
                any(
                    path.name.startswith(".curation.lease-candidate-")
                    for path in output.iterdir()
                )
            )
            with self.assertRaisesRegex(CurationError, "Another curation process holds"):
                curate_corpus_module.acquire_cross_client_lease(output)

            curate_corpus_module.release_cross_client_lease(output, lease, owner, lock)
            self.assertFalse(lease.exists())
            second_lease, second_owner, second_lock = (
                curate_corpus_module.acquire_cross_client_lease(output)
            )
            curate_corpus_module.release_cross_client_lease(
                output, second_lease, second_owner, second_lock
            )

    def test_cross_client_lease_recovery_is_explicit_and_archival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curated"
            output.mkdir()
            lease, owner, lock = curate_corpus_module.acquire_cross_client_lease(output)
            lock.close()  # Simulate process death: durable owner remains.
            with self.assertRaisesRegex(CurationError, "live process"):
                curate_corpus_module.acquire_cross_client_lease(
                    output, recover_stale_owner_token=owner["owner_token"]
                )

            stale = {**owner, "hostname": "retired-test-pod"}
            stale["owner_token"] = hashlib.sha256(
                canonical_json_bytes(
                    {key: value for key, value in stale.items() if key != "owner_token"}
                )
            ).hexdigest()
            write_json(lease, stale)
            replacement_lease, replacement_owner, replacement_lock = (
                curate_corpus_module.acquire_cross_client_lease(
                    output, recover_stale_owner_token=stale["owner_token"]
                )
            )
            archives = list((output / ".stale-curation-leases").glob("*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8")), stale)
            claims = list((output / ".curation-recovery-claims").glob("*.json"))
            self.assertEqual([path.stem for path in claims], [stale["owner_token"]])
            with self.assertRaisesRegex(
                CurationError, "Another curation process holds"
            ):
                curate_corpus_module.acquire_cross_client_lease(
                    output, recover_stale_owner_token=stale["owner_token"]
                )
            curate_corpus_module.release_cross_client_lease(
                output, replacement_lease, replacement_owner, replacement_lock
            )

    def test_cross_client_lease_release_refuses_owner_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curated"
            output.mkdir()
            lease, owner, lock = curate_corpus_module.acquire_cross_client_lease(output)
            original = lease.read_bytes()
            drifted = {**owner, "started_unix_ns": owner["started_unix_ns"] + 1}
            drifted["owner_token"] = hashlib.sha256(
                canonical_json_bytes(
                    {
                        key: value
                        for key, value in drifted.items()
                        if key != "owner_token"
                    }
                )
            ).hexdigest()
            write_json(lease, drifted)
            with self.assertRaisesRegex(CurationError, "ownership changed"):
                curate_corpus_module.release_cross_client_lease(
                    output, lease, owner, lock
                )
            lease.write_bytes(original)
            reacquired_lock = (output / ".curation.lock").open("a+")
            fcntl.flock(reacquired_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            curate_corpus_module.release_cross_client_lease(
                output, lease, owner, reacquired_lock
            )

    def test_concurrent_stale_recovery_claim_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curated"
            output.mkdir()
            lease, owner, lock = curate_corpus_module.acquire_cross_client_lease(output)
            lock.close()  # Simulated crash leaves only the durable owner.
            stale = {**owner, "hostname": "retired-concurrent-test-pod"}
            stale["owner_token"] = hashlib.sha256(
                canonical_json_bytes(
                    {key: value for key, value in stale.items() if key != "owner_token"}
                )
            ).hexdigest()
            write_json(lease, stale)

            rendezvous = threading.Barrier(2)
            outcomes: queue.Queue[tuple[str, object]] = queue.Queue()
            original_publish = curate_corpus_module.publish_stale_recovery_claim

            def synchronized_publish(
                claim_output: Path, previous: dict[str, object]
            ) -> Path:
                rendezvous.wait(timeout=5)
                return original_publish(claim_output, previous)

            def recover() -> None:
                try:
                    result = (
                        curate_corpus_module._acquire_durable_cross_client_lease(
                            output,
                            recover_stale_owner_token=stale["owner_token"],
                        )
                    )
                except BaseException as error:
                    outcomes.put(("error", error))
                else:
                    outcomes.put(("success", result))

            with mock.patch.object(
                curate_corpus_module,
                "publish_stale_recovery_claim",
                side_effect=synchronized_publish,
            ):
                threads = [threading.Thread(target=recover) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
                    self.assertFalse(thread.is_alive())

            results = [outcomes.get_nowait() for _ in range(2)]
            successes = [value for status, value in results if status == "success"]
            errors = [value for status, value in results if status == "error"]
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(errors), 1)
            self.assertRegex(str(errors[0]), "permanent recovery claim")
            winner_lease, winner_owner = successes[0]
            self.assertEqual(winner_lease, lease)
            self.assertEqual(
                json.loads(lease.read_text(encoding="utf-8")), winner_owner
            )

            cleanup_lock = (output / ".curation.lock").open("a+")
            fcntl.flock(cleanup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            curate_corpus_module.release_cross_client_lease(
                output, lease, winner_owner, cleanup_lock
            )

    def test_bounded_subphase_faults_resume_to_identical_selection(self) -> None:
        targets = (
            "inventory.archive.python.000000",
            "canonicalize.near_map_load",
            "canonicalize.near_map_apply",
            "canonicalize.benchmark_exact",
            "canonicalize.exact_choice",
            "canonicalize.canonical_map",
            "selection.groups",
            "selection.quota.test.python",
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            fresh = Path(temporary) / "fresh"
            fresh_result = fixture.build(fresh)
            fresh_manifest_bytes = (fresh / "manifest.json").read_bytes()

            def logical_rows(output: Path) -> dict[str, list[tuple[object, ...]]]:
                connection = sqlite3.connect(output / ".work" / "curation.sqlite3")
                try:
                    queries = {
                        "reasons": "SELECT hex(doc_id), reason FROM reasons ORDER BY doc_id, reason",
                        "near_map": "SELECT hex(doc_id), hex(cluster) FROM near_map ORDER BY doc_id",
                        "exact_choice": "SELECT hex(content_hash), hex(canonical_rank), hex(canonical_doc_id) FROM exact_choice ORDER BY content_hash",
                        "final_choice": "SELECT hex(final_cluster), hex(canonical_rank), hex(canonical_doc_id) FROM final_choice ORDER BY final_cluster",
                        "canonical_map": "SELECT hex(doc_id), hex(canonical_doc_id) FROM canonical_map ORDER BY doc_id",
                        "groups": "SELECT hex(group_id), split FROM groups ORDER BY group_id",
                        "selected": "SELECT hex(doc_id), split, selected_tokens FROM selected ORDER BY doc_id",
                        "phase_progress": "SELECT subphase, status, cursor_json, processed_rows, processed_tokens, committed_batches, details_json FROM phase_progress ORDER BY subphase",
                    }
                    return {
                        name: [tuple(row) for row in connection.execute(query)]
                        for name, query in queries.items()
                    }
                finally:
                    connection.close()

            expected_rows = logical_rows(fresh)
            expected_decisions = {
                shard["path"]: (fresh / shard["path"]).read_bytes()
                for shard in fresh_result["manifest"]["decision_shards"]
            }

            for target in targets:
                with self.subTest(target=target):
                    output = Path(temporary) / target.replace(".", "-")
                    with CurationBuilder(
                        root=fixture.root,
                        staging_root=fixture.staging,
                        output=output,
                        policy_path=DEFAULT_POLICY,
                        quota_path=fixture.quota_path,
                        denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                        english_near_clusters=fixture.near_path,
                        allow_missing_english_near_dedup=False,
                        batch_size=3,
                    ) as builder:
                        original = builder._after_bounded_commit

                        def interrupt(
                            subphase: str,
                            progress: dict[str, object],
                            *,
                            wanted: str = target,
                        ) -> None:
                            original(subphase, progress)
                            if (
                                subphase == wanted
                                and progress["committed_batches"] == 1
                            ):
                                raise RuntimeError(f"injected after {wanted} commit")

                        with mock.patch.object(
                            builder, "_after_bounded_commit", side_effect=interrupt
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError, "injected after"
                            ):
                                builder.run()

                    checkpoint = json.loads(
                        (output / ".work" / "CHECKPOINT.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    interrupted = {
                        row["subphase"]: row for row in checkpoint["subphases"]
                    }[target]
                    self.assertEqual(interrupted["status"], "running")
                    self.assertEqual(interrupted["committed_batches"], 1)
                    self.assertGreater(interrupted["processed_rows"], 0)

                    resumed = fixture.build(output)
                    self.assertTrue(resumed["complete"])
                    self.assertEqual(
                        (output / "manifest.json").read_bytes(),
                        fresh_manifest_bytes,
                    )
                    self.assertEqual(logical_rows(output), expected_rows)
                    for relative, wanted in expected_decisions.items():
                        self.assertEqual((output / relative).read_bytes(), wanted)

    def test_completed_group_subphase_rejects_an_unbacked_extra_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "tampered-groups"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=3,
            ) as builder:
                builder.run(stop_after_phase="canonicalized")
                with mock.patch.object(
                    builder,
                    "_select_quota_bounded",
                    side_effect=RuntimeError("stop after groups"),
                ), self.assertRaisesRegex(RuntimeError, "stop after groups"):
                    builder.assign_splits_and_quotas()
                progress = builder._progress("selection.groups")
                self.assertIsNotNone(progress)
                self.assertEqual(progress["status"], "complete")
                builder.db.execute(
                    "INSERT INTO groups(group_id, split) VALUES (?, 'train')",
                    (hashlib.sha256(b"unbacked-extra-group").digest(),),
                )
                builder.db.commit()

            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=3,
            ) as builder, self.assertRaisesRegex(
                RuntimeError, "Completed split-group table changed"
            ):
                builder.assign_splits_and_quotas()

    def test_storage_preflight_and_transaction_bounds_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated"
            fixture.build(output)
            checkpoint = json.loads(
                (output / ".work" / "CHECKPOINT.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["checkpoint_version"], 2)
            self.assertEqual(checkpoint["database_version"], 4)
            storage = checkpoint["storage"]
            self.assertEqual(storage["preflight"]["status"], "pass")
            self.assertLessEqual(storage["maximum_transaction_rows"], 3)
            self.assertGreater(storage["committed_transactions"], 0)
            self.assertLessEqual(
                max(
                    storage["maximum_journal_bytes"],
                    storage["maximum_wal_bytes"],
                ),
                storage["preflight"]["contract"][
                    "transaction_sidecar_limit_bytes"
                ],
            )
            self.assertTrue(
                all(row["status"] == "complete" for row in checkpoint["subphases"])
            )

    def test_schema_v2_resume_seeds_v4_durable_counters_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated"
            partial = fixture.build(output, max_new_archives=1)
            self.assertFalse(partial["complete"])
            database = output / ".work" / "curation.sqlite3"
            connection = sqlite3.connect(database)
            try:
                current_contract = json.loads(
                    connection.execute(
                        "SELECT value FROM metadata "
                        "WHERE key='curation_storage_contract'"
                    ).fetchone()[0]
                )
                current_contract[
                    "projected_additional_bytes_per_document"
                ] = 1_024
                current_contract["contract_version"] = 1
                del current_contract["projection_basis"]
                del current_contract["projection_method"]
                del current_contract["sqlite_temp_relative_path"]
                del current_contract["sqlite_temp_same_device_as_database"]
                preflight = json.loads(
                    connection.execute(
                        "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
                    ).fetchone()[0]
                )
                preflight["contract"] = current_contract
                preflight["preflight_version"] = 1
                del preflight["measurement_reason"]
                del preflight["sqlite_temp_relative_path"]
                del preflight["sqlite_temp_free_bytes_at_measurement"]
                del preflight["sqlite_temp_total_bytes_at_measurement"]
                del preflight["sqlite_temp_same_device_as_database"]
                expected_documents = int(preflight["documents_expected"])
                projected = expected_documents * 1_024 * 2
                preflight["projected_additional_bytes_with_safety"] = projected
                preflight["required_free_bytes_at_measurement"] = (
                    projected
                    + int(current_contract["transaction_sidecar_limit_bytes"])
                    + int(current_contract["minimum_free_bytes_after_projection"])
                )
                connection.execute(
                    "UPDATE metadata SET value='2' WHERE key='database_version'"
                )
                connection.execute(
                    "UPDATE metadata SET value=? "
                    "WHERE key='curation_storage_contract'",
                    (json.dumps(current_contract, sort_keys=True),),
                )
                connection.execute(
                    "UPDATE storage_metrics SET preflight_json=? WHERE singleton=1",
                    (canonical_json_bytes(preflight).decode("utf-8"),),
                )
                connection.execute("DROP TABLE durable_counts")
                connection.commit()
            finally:
                connection.close()

            resumed = fixture.build(output)
            self.assertTrue(resumed["complete"])
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    json.loads(
                        connection.execute(
                            "SELECT value FROM metadata WHERE key='database_version'"
                        ).fetchone()[0]
                    ),
                    4,
                )
                durable = tuple(
                    connection.execute(
                        """
                        SELECT archives, documents, selected_documents, output_archives
                        FROM durable_counts WHERE singleton=1
                        """
                    ).fetchone()
                )
                actual = tuple(
                    connection.execute(
                        """
                        SELECT (SELECT COUNT(*) FROM archives),
                               (SELECT COUNT(*) FROM documents),
                               (SELECT COUNT(*) FROM selected),
                               (SELECT COUNT(*) FROM output_archives)
                        """
                    ).fetchone()
                )
                self.assertEqual(durable, actual)
                migrations = connection.execute(
                    "SELECT COUNT(*) FROM events WHERE event='database_migrated'"
                ).fetchone()[0]
                self.assertEqual(migrations, 1)
                migrated_contract = json.loads(
                    connection.execute(
                        "SELECT value FROM metadata "
                        "WHERE key='curation_storage_contract'"
                    ).fetchone()[0]
                )
                self.assertEqual(
                    migrated_contract["contract_version"],
                    CURATION_STORAGE_PREFLIGHT_VERSION,
                )
                self.assertEqual(
                    migrated_contract["projected_additional_bytes_per_document"],
                    CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT,
                )
                self.assertEqual(
                    migrated_contract["sqlite_temp_relative_path"],
                    ".work/sqlite-tmp",
                )
                self.assertTrue(
                    migrated_contract["sqlite_temp_same_device_as_database"]
                )
                migrated_preflight = json.loads(
                    connection.execute(
                        "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
                    ).fetchone()[0]
                )
                self.assertEqual(
                    migrated_preflight["preflight_version"],
                    CURATION_STORAGE_PREFLIGHT_VERSION,
                )
                self.assertEqual(
                    migrated_preflight["measurement_reason"][
                        "from_database_version"
                    ],
                    2,
                )
                self.assertEqual(
                    len(
                        migrated_preflight["measurement_reason"][
                            "previous_preflight_sha256"
                        ]
                    ),
                    64,
                )
            finally:
                connection.close()

    def test_schema_v3_storage_preflight_is_remeasured_for_v4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated-v3-migration"
            fixture.build(output, max_new_archives=1)
            database = output / ".work" / "curation.sqlite3"
            connection = sqlite3.connect(database)
            try:
                contract = json.loads(
                    connection.execute(
                        "SELECT value FROM metadata "
                        "WHERE key='curation_storage_contract'"
                    ).fetchone()[0]
                )
                contract["contract_version"] = 1
                contract["projected_additional_bytes_per_document"] = 1_024
                del contract["projection_basis"]
                del contract["projection_method"]
                preflight = json.loads(
                    connection.execute(
                        "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
                    ).fetchone()[0]
                )
                preflight["preflight_version"] = 1
                preflight["contract"] = contract
                del preflight["measurement_reason"]
                projected = int(preflight["documents_expected"]) * 1_024 * 2
                preflight["projected_additional_bytes_with_safety"] = projected
                preflight["required_free_bytes_at_measurement"] = (
                    projected
                    + int(contract["transaction_sidecar_limit_bytes"])
                    + int(contract["minimum_free_bytes_after_projection"])
                )
                connection.execute(
                    "UPDATE metadata SET value='3' WHERE key='database_version'"
                )
                connection.execute(
                    "UPDATE metadata SET value=? "
                    "WHERE key='curation_storage_contract'",
                    (json.dumps(contract, sort_keys=True),),
                )
                connection.execute(
                    "UPDATE storage_metrics SET preflight_json=? WHERE singleton=1",
                    (canonical_json_bytes(preflight).decode("utf-8"),),
                )
                connection.commit()
            finally:
                connection.close()

            resumed = fixture.build(output)
            self.assertTrue(resumed["complete"])
            connection = sqlite3.connect(database)
            try:
                migrated = json.loads(
                    connection.execute(
                        "SELECT preflight_json FROM storage_metrics WHERE singleton=1"
                    ).fetchone()[0]
                )
                self.assertEqual(
                    migrated["preflight_version"],
                    CURATION_STORAGE_PREFLIGHT_VERSION,
                )
                self.assertEqual(
                    migrated["contract"][
                        "projected_additional_bytes_per_document"
                    ],
                    CURATION_PROJECTED_ADDITIONAL_BYTES_PER_DOCUMENT,
                )
                self.assertEqual(
                    migrated["measurement_reason"]["reason"],
                    "database_contract_migration",
                )
                self.assertEqual(
                    migrated["measurement_reason"]["from_database_version"],
                    3,
                )
                self.assertEqual(
                    len(
                        migrated["measurement_reason"][
                            "previous_preflight_sha256"
                        ]
                    ),
                    64,
                )
            finally:
                connection.close()

    def test_schema_v1_public_phase_resumes_match_fresh_v4_output(self) -> None:
        phases = (
            "inventory",
            "inventory_complete",
            "canonicalized",
            "selected",
            "emitting",
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            fresh = Path(temporary) / "fresh-v4"
            fresh_result = fixture.build(fresh)
            fresh_manifest = (fresh / "manifest.json").read_bytes()
            fresh_decisions = {
                shard["path"]: (fresh / shard["path"]).read_bytes()
                for shard in fresh_result["manifest"]["decision_shards"]
            }

            for phase in phases:
                with self.subTest(phase=phase):
                    output = Path(temporary) / f"legacy-v1-{phase}"
                    if phase == "inventory":
                        fixture.build(output, max_new_archives=1)
                    else:
                        stop = None if phase == "emitting" else phase
                        with CurationBuilder(
                            root=fixture.root,
                            staging_root=fixture.staging,
                            output=output,
                            policy_path=DEFAULT_POLICY,
                            quota_path=fixture.quota_path,
                            denylist_path=PROJECT_ROOT
                            / "configs"
                            / "mbpp_denylist.json",
                            english_near_clusters=fixture.near_path,
                            allow_missing_english_near_dedup=False,
                            batch_size=3,
                        ) as builder:
                            if phase == "emitting":
                                builder.run(stop_after_phase="selected")
                                builder._advance("selected", "emitting", {})
                                first_report = min(
                                    (
                                        report
                                        for *_prefix, report in builder.report_inventory
                                    ),
                                    key=lambda report: report["archive"],
                                )
                                builder._emit_archive(first_report)
                            else:
                                builder.run(stop_after_phase=stop)

                    database = output / ".work" / "curation.sqlite3"
                    connection = sqlite3.connect(database)
                    try:
                        connection.execute(
                            "UPDATE metadata SET value='1' "
                            "WHERE key='database_version'"
                        )
                        connection.execute(
                            "DELETE FROM metadata "
                            "WHERE key='curation_storage_contract'"
                        )
                        for table in (
                            "phase_progress",
                            "storage_metrics",
                            "durable_counts",
                            "benchmark_content_clusters",
                            "benchmark_final_clusters",
                        ):
                            connection.execute(f"DROP TABLE {table}")
                        connection.commit()
                    finally:
                        connection.close()

                    resumed = fixture.build(output)
                    self.assertTrue(resumed["complete"])
                    self.assertEqual(
                        (output / "manifest.json").read_bytes(), fresh_manifest
                    )
                    for relative, wanted in fresh_decisions.items():
                        self.assertEqual((output / relative).read_bytes(), wanted)

    def test_storage_preflight_fails_before_canonical_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=3,
            ) as builder, mock.patch.object(
                curate_corpus_module.shutil,
                "disk_usage",
                return_value=type(
                    "DiskUsage", (), {"total": 100, "used": 99, "free": 1}
                )(),
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient curation"):
                    builder.run()
                self.assertEqual(builder._phase(), "inventory")
                self.assertEqual(
                    builder.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    0,
                )

    def test_postcommit_wal_bound_violation_permanently_blocks_resume(self) -> None:
        mount = {
            "detector": "fixture",
            "mount_point": "/fixture-local",
            "filesystem_type": "xfs",
            "source": "/dev/fixture",
            "device": "1:1",
            "options": ["rw"],
            "classification": "local",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            curate_corpus_module, "detect_output_mount", return_value=mount
        ):
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=3,
                sqlite_journal_mode="wal",
            ) as builder:
                builder.ingest_inventory(max_new_archives=1)
                limit = builder.storage_contract[
                    "transaction_sidecar_limit_bytes"
                ]
                injected = {
                    "database_bytes": 1,
                    "journal_bytes": 0,
                    "wal_bytes": limit + 1,
                    "free_bytes": 10_000_000_000,
                    "filesystem_total_bytes": 20_000_000_000,
                    "sqlite_temp_free_bytes": 10_000_000_000,
                    "sqlite_temp_total_bytes": 20_000_000_000,
                }
                with mock.patch.object(
                    builder, "_storage_snapshot", return_value=injected
                ), self.assertRaisesRegex(RuntimeError, "new output generation"):
                    builder._bound_wal_after_commit()

            with self.assertRaisesRegex(RuntimeError, "permanently violated"):
                with CurationBuilder(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    output=output,
                    policy_path=DEFAULT_POLICY,
                    quota_path=fixture.quota_path,
                    denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                    english_near_clusters=fixture.near_path,
                    allow_missing_english_near_dedup=False,
                    batch_size=3,
                    sqlite_journal_mode="wal",
                ):
                    pass

    def test_busy_wal_checkpoint_cannot_hide_committed_bound_violation(self) -> None:
        mount = {
            "detector": "fixture",
            "mount_point": "/fixture-local",
            "filesystem_type": "xfs",
            "source": "/dev/fixture",
            "device": "1:1",
            "options": ["rw"],
            "classification": "local",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            curate_corpus_module, "detect_output_mount", return_value=mount
        ):
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated-busy-checkpoint"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=3,
                sqlite_journal_mode="wal",
            ) as builder:
                builder.ingest_inventory(max_new_archives=1)
                limit = int(
                    builder.storage_contract["transaction_sidecar_limit_bytes"]
                )
                injected = {
                    "database_bytes": 1,
                    "journal_bytes": 0,
                    "wal_bytes": limit + 1,
                    "free_bytes": 10_000_000_000,
                    "filesystem_total_bytes": 20_000_000_000,
                    "sqlite_temp_free_bytes": 10_000_000_000,
                    "sqlite_temp_total_bytes": 20_000_000_000,
                }
                with mock.patch.object(
                    builder, "_storage_snapshot", return_value=injected
                ), mock.patch.object(
                    builder, "_size_or_zero", return_value=limit + 1
                ), mock.patch.object(
                    builder, "_truncate_wal", return_value=(1, 8, 0)
                ), self.assertRaisesRegex(RuntimeError, "checkpoint"):
                    builder._bound_wal_after_commit()

            with self.assertRaisesRegex(RuntimeError, "permanently violated"):
                with CurationBuilder(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    output=output,
                    policy_path=DEFAULT_POLICY,
                    quota_path=fixture.quota_path,
                    denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                    english_near_clusters=fixture.near_path,
                    allow_missing_english_near_dedup=False,
                    batch_size=3,
                    sqlite_journal_mode="wal",
                ):
                    pass

    def test_large_two_bucket_quota_merge_is_indexed_streaming_and_deterministic(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(
                f"""
                CREATE TABLE documents (
                    doc_id BLOB PRIMARY KEY,
                    bucket TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    source_group BLOB NOT NULL,
                    selection_rank BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE groups (
                    group_id BLOB PRIMARY KEY,
                    split TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE reasons (
                    doc_id BLOB NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(doc_id, reason)
                ) WITHOUT ROWID;
                CREATE INDEX {QUOTA_SELECTION_INDEX}
                    ON documents(bucket, selection_rank, doc_id);
                """
            )
            groups = [
                (hashlib.sha256(f"group-{index}".encode()).digest(), split)
                for index, split in enumerate(("train", "validation", "train"))
            ]
            connection.executemany(
                "INSERT INTO groups(group_id, split) VALUES (?, ?)", groups
            )
            rows = []
            rejected = []
            expected = []
            for index in range(24_000):
                bucket = "fineweb_edu" if index % 2 == 0 else "wikipedia"
                doc_id = hashlib.sha256(f"document-{index}".encode()).digest()
                # Deliberate cross-bucket rank ties prove doc_id is the stable
                # total-order tiebreaker, not SQLite cursor scheduling.
                rank = hashlib.sha256(f"rank-{index // 2}".encode()).digest()
                group_id, split = groups[index % len(groups)]
                tokens = index % 11 + 1
                rows.append((doc_id, bucket, tokens, group_id, rank))
                is_rejected = index % 97 == 0
                if is_rejected:
                    rejected.append((doc_id, "synthetic-reject"))
                elif split == "train":
                    expected.append((rank, doc_id, tokens))
            connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)", rows
            )
            connection.executemany(
                "INSERT INTO reasons(doc_id, reason) VALUES (?, ?)", rejected
            )
            connection.commit()

            for bucket in ("fineweb_edu", "wikipedia"):
                plan = [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + QUOTA_CANDIDATE_SQL,
                        (bucket, "train"),
                    )
                ]
                self.assertTrue(
                    any(QUOTA_SELECTION_INDEX in detail for detail in plan), plan
                )
                self.assertFalse(
                    any("TEMP B-TREE" in detail.upper() for detail in plan), plan
                )

                after_plan = [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + QUOTA_CANDIDATE_AFTER_SQL,
                        (bucket, "train", b"\x40" * 32, b"\x40" * 32),
                    )
                ]
                self.assertTrue(
                    any(
                        QUOTA_SELECTION_INDEX in detail
                        and "selection_rank" in detail
                        and ">" in detail
                        for detail in after_plan
                    ),
                    after_plan,
                )
                self.assertFalse(
                    any("TEMP B-TREE" in detail.upper() for detail in after_plan),
                    after_plan,
                )

            expected.sort(key=lambda row: (row[0], row[1]))
            expected_rows = [(doc_id, tokens) for _rank, doc_id, tokens in expected]
            first = list(
                iter_merged_quota_candidates(
                    connection,
                    split="train",
                    buckets=("wikipedia", "fineweb_edu"),
                )
            )
            second = list(
                iter_merged_quota_candidates(
                    connection,
                    split="train",
                    buckets=("fineweb_edu", "wikipedia"),
                )
            )
            self.assertEqual(first, expected_rows)
            self.assertEqual(second, expected_rows)

            midpoint = expected[len(expected) // 2]
            suffix = list(
                curate_corpus_module.iter_merged_quota_candidate_rows(
                    connection,
                    split="train",
                    buckets=("wikipedia", "fineweb_edu"),
                    after=(midpoint[0], midpoint[1]),
                )
            )
            self.assertEqual(suffix, expected[len(expected) // 2 + 1 :])
        finally:
            connection.close()

    def test_decision_lookup_combines_status_joins_and_preserves_optional_fields(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(
                """
                CREATE TABLE documents (
                    doc_id BLOB PRIMARY KEY,
                    source_group BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE reasons (
                    doc_id BLOB NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(doc_id, reason)
                ) WITHOUT ROWID;
                CREATE TABLE canonical_map (
                    doc_id BLOB PRIMARY KEY,
                    canonical_doc_id BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE groups (
                    group_id BLOB PRIMARY KEY,
                    split TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE selected (
                    doc_id BLOB PRIMARY KEY,
                    split TEXT NOT NULL,
                    selected_tokens INTEGER NOT NULL
                ) WITHOUT ROWID;
                """
            )
            kept = hashlib.sha256(b"kept").digest()
            rejected = hashlib.sha256(b"rejected").digest()
            canonical = hashlib.sha256(b"canonical").digest()
            group = hashlib.sha256(b"group").digest()
            missing_group = hashlib.sha256(b"missing-group").digest()
            connection.executemany(
                "INSERT INTO documents(doc_id, source_group) VALUES (?, ?)",
                ((kept, group), (rejected, missing_group)),
            )
            connection.execute(
                "INSERT INTO reasons(doc_id, reason) VALUES (?, ?)",
                (rejected, "quality:synthetic"),
            )
            connection.execute(
                "INSERT INTO canonical_map(doc_id, canonical_doc_id) VALUES (?, ?)",
                (kept, canonical),
            )
            connection.execute(
                "INSERT INTO groups(group_id, split) VALUES (?, ?)",
                (group, "train"),
            )
            connection.execute(
                "INSERT INTO selected(doc_id, split, selected_tokens) VALUES (?, ?, ?)",
                (kept, "train", 17),
            )
            builder = object.__new__(CurationBuilder)
            builder.connection = connection
            statements: list[str] = []
            connection.set_trace_callback(statements.append)
            status = builder._decision_rows(
                [{"doc_id": kept.hex()}, {"doc_id": rejected.hex()}]
            )
            connection.set_trace_callback(None)

            self.assertEqual(
                status[kept],
                {
                    "reasons": [],
                    "canonical_doc_id": canonical.hex(),
                    "assigned_split": "train",
                    "group_id": group.hex(),
                    "split": "train",
                    "selected_tokens": 17,
                },
            )
            self.assertEqual(status[rejected], {"reasons": ["quality:synthetic"]})
            self.assertEqual(
                sum(statement.lstrip().upper().startswith("SELECT") for statement in statements),
                2,
            )
        finally:
            connection.close()

    def test_sparse_benchmark_cluster_scans_start_from_reason_index(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(
                """
                CREATE TABLE documents (
                    doc_id BLOB PRIMARY KEY,
                    content_hash BLOB NOT NULL,
                    final_cluster BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE reasons (
                    doc_id BLOB NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(doc_id, reason)
                ) WITHOUT ROWID;
                CREATE INDEX reasons_reason ON reasons(reason);
                """
            )
            documents = []
            reasons = []
            wanted_content = []
            wanted_final = []
            for index in range(1_000):
                doc_id = hashlib.sha256(f"doc-{index}".encode()).digest()
                content = hashlib.sha256(f"content-{index // 2}".encode()).digest()
                final = hashlib.sha256(f"final-{index // 3}".encode()).digest()
                documents.append((doc_id, content, final))
                if index % 10 == 0:
                    reasons.append((doc_id, "quality:synthetic"))
                if index in (111, 333, 777):
                    reasons.append((doc_id, "benchmark:mbpp"))
                    wanted_content.append(content)
                    wanted_final.append(final)
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?)", documents)
            connection.executemany("INSERT INTO reasons VALUES (?, ?)", reasons)

            for sql, expected in (
                (BENCHMARK_CONTENT_CLUSTER_SQL, sorted(set(wanted_content))),
                (BENCHMARK_FINAL_CLUSTER_SQL, sorted(set(wanted_final))),
            ):
                plan = [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + sql, (b"", 10_000)
                    )
                ]
                self.assertTrue(
                    any("reasons_reason" in detail for detail in plan), plan
                )
                self.assertFalse(
                    any("SCAN d" in detail for detail in plan), plan
                )
                self.assertEqual(
                    [bytes(row[0]) for row in connection.execute(sql, (b"", 10_000))],
                    expected,
                )
        finally:
            connection.close()

    def test_cross_archive_dedup_group_safe_splits_exact_quotas_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            resumed = Path(temporary) / "curated-resumed"
            partial = fixture.build(resumed, max_new_archives=1)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["phase"], "inventory")
            result = fixture.build(resumed)
            self.assertTrue(result["complete"])
            manifest = result["manifest"]
            self.assertTrue(manifest["production_ready"])
            self.assertTrue(manifest["english_near_dedup_complete"])
            self.assertNotIn("english_near_dedup_status", manifest)
            self.assertNotIn("english_near_dedup_policy", manifest["identity"])
            self.assertTrue(manifest["raw_archives_hashed_for_integrity"])
            self.assertFalse(manifest["raw_archive_payloads_parsed_by_curation"])
            self.assertEqual(
                manifest["identity"]["english_near_artifact"]["contract_version"],
                2,
            )
            near_artifact = manifest["identity"]["english_near_artifact"]
            self.assertEqual(
                near_artifact["refinement_operational_preflight"]["status"],
                "pass",
            )
            self.assertEqual(
                near_artifact["manifest"]["sha256"],
                file_sha256(fixture.near_path.parent / "manifest.json"),
            )
            near_identity = manifest["identity"]["english_near_artifact"]["identity"]
            calibration = near_identity["calibration_evidence"]
            self.assertEqual(calibration, fixture.near_builder_identity["calibration_evidence"])
            self.assertEqual(
                calibration["identity"]["production_builder_sha256"],
                near_identity["builder_sha256"],
            )
            completeness = manifest["collection_completeness"]
            self.assertTrue(completeness["complete"])
            self.assertFalse(completeness["legacy_dedup_index_required"])
            self.assertEqual(completeness, manifest["identity"]["collection_completeness"])
            self.assertEqual(
                set(completeness["per_bucket"]),
                {"python", "other_code", "fineweb_edu", "wikipedia"},
            )
            self.assertEqual(
                completeness["quota_records"]["count"],
                completeness["reports"]["count"],
            )
            self.assertEqual(manifest["quota_unit"], "pre_packing_starcoder2_content_tokens")
            self.assertEqual(
                manifest["leakage_audit"]["content_hashes_in_multiple_splits"], 0
            )
            self.assertEqual(
                manifest["leakage_audit"]["canonical_clusters_in_multiple_splits"], 0
            )
            self.assertEqual(manifest["leakage_audit"]["source_groups_in_multiple_splits"], 0)
            self.assertGreater(manifest["leakage_audit"]["cross_bucket_code_repo_groups"], 0)
            self.assertTrue(all(row["selected_tokens"] == row["target_tokens"] == 2 for row in manifest["quotas"]))
            self.assertTrue(all(row["terminal_prefix_documents"] == 1 for row in manifest["quotas"]))

            decisions: dict[str, dict[str, object]] = {}
            for shard in manifest["decision_shards"]:
                for row in iter_jsonl_zst(resumed / shard["path"]):
                    decisions[str(row["doc_id"])] = row

            for cluster in (fixture.special["exact"], fixture.special["normalized"], fixture.special["near"]):
                kept_or_canonical = [
                    decisions[doc_id]
                    for doc_id in cluster
                    if not any("duplicate" in reason for reason in decisions[doc_id]["reasons"])
                ]
                self.assertEqual(len(kept_or_canonical), 1)
            self.assertIn(
                "benchmark_cluster_contamination",
                decisions[fixture.special["benchmark_copy"]]["reasons"],
            )
            self.assertIn("quality:too_short", decisions[fixture.special["quality"]]["reasons"])

            selected = [row for row in decisions.values() if row["decision"] == "keep"]
            repo_splits: dict[str, set[str]] = {}
            hashes_by_split: dict[str, set[str]] = {}
            for row in selected:
                repo = row["provenance"].get("repo_id")
                if repo:
                    repo_splits.setdefault(str(repo), set()).add(str(row["split"]))
                hashes_by_split.setdefault(str(row["split"]), set()).add(str(row["content_sha256"]))
                self.assertEqual(row["selected_tokens"], 2)
                self.assertTrue(row["terminal_quota_prefix"])
            self.assertTrue(all(len(splits) == 1 for splits in repo_splits.values()))
            for left in ("train", "validation", "test"):
                for right in ("train", "validation", "test"):
                    if left < right:
                        self.assertFalse(hashes_by_split.get(left, set()) & hashes_by_split.get(right, set()))

            # A clean rebuild has byte-identical decision shards and manifest,
            # proving resume does not alter deterministic selection.
            fresh = Path(temporary) / "curated-fresh"
            fresh_result = fixture.build(fresh)
            self.assertEqual(result["manifest"], fresh_result["manifest"])
            for shard in manifest["decision_shards"]:
                self.assertEqual(file_sha256(resumed / shard["path"]), file_sha256(fresh / shard["path"]))

    def test_fast_profile_is_production_exact_normalized_and_near_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = Path(temporary) / "curated-fast"
            partial = fixture.build_fast(output, max_new_archives=1)
            self.assertFalse(partial["complete"])
            result = fixture.build_fast(output)
            self.assertTrue(result["complete"])
            manifest = result["manifest"]

            self.assertTrue(manifest["production_ready"])
            self.assertFalse(manifest["english_near_dedup_complete"])
            self.assertEqual(
                manifest["english_near_dedup_status"],
                "disabled_by_fast_profile",
            )
            self.assertEqual(manifest["curation_profile"], FAST_CANONICAL_PROFILE)
            self.assertEqual(
                manifest["identity"]["curation_profile"], FAST_CANONICAL_PROFILE
            )
            self.assertEqual(manifest["identity"]["format_version"], 6)
            self.assertIsNone(manifest["identity"]["english_near_artifact"])
            self.assertIsNone(
                manifest["identity"]["english_near_clusters_sha256"]
            )
            self.assertIn(
                FAST_CANONICAL_PROFILE["known_limitations"][0],
                manifest["known_provenance_limitations"],
            )
            self.assertTrue(
                all(
                    row["selected_tokens"] == row["target_tokens"] == 2
                    for row in manifest["quotas"]
                )
            )
            self.assertEqual(len(manifest["quotas"]), 9)

            leakage = manifest["leakage_audit"]
            for key in (
                "content_hashes_in_multiple_splits",
                "normalized_hashes_in_multiple_splits",
                "canonical_clusters_in_multiple_splits",
                "source_groups_in_multiple_splits",
                "content_hashes_with_multiple_selected_documents",
                "normalized_hashes_with_multiple_selected_documents",
            ):
                self.assertEqual(leakage[key], 0, key)
            fast_audit = manifest["fast_profile_audit"]
            self.assertFalse(fast_audit["fuzzy_near_dedup_performed"])
            for key in (
                "near_map_rows",
                "near_mapping_subphases",
                "english_near_duplicate_reasons",
                "content_hashes_in_multiple_splits",
                "normalized_hashes_in_multiple_splits",
                "content_hashes_with_multiple_selected_documents",
                "normalized_hashes_with_multiple_selected_documents",
                "source_groups_in_multiple_splits",
            ):
                self.assertEqual(fast_audit[key], 0, key)
            self.assertNotIn(
                "english_near_duplicate", manifest["reason_document_counts"]
            )

            decisions: dict[str, dict[str, object]] = {}
            for shard in manifest["decision_shards"]:
                for row in iter_jsonl_zst(output / shard["path"]):
                    decisions[str(row["doc_id"])] = row

            exact = [decisions[doc_id] for doc_id in fixture.special["exact"]]
            self.assertEqual({str(row["bucket"]) for row in exact}, {"python", "other_code"})
            self.assertEqual(
                sum("residual_exact_duplicate" in row["reasons"] for row in exact),
                1,
            )
            normalized = [
                decisions[doc_id] for doc_id in fixture.special["normalized"]
            ]
            self.assertEqual(
                sum(
                    "residual_normalized_duplicate" in row["reasons"]
                    for row in normalized
                ),
                1,
            )
            self.assertEqual(
                len({str(row["content_sha256"]) for row in normalized}), 2
            )

            # The fixture near map links this byte- and normalized-distinct
            # cross-source pair. The fast profile must ignore that artifact and
            # retain both as canonical documents (quota overflow is not dedup).
            near = [decisions[doc_id] for doc_id in fixture.special["near"]]
            for row in near:
                self.assertFalse(
                    any("duplicate" in str(reason) for reason in row["reasons"])
                )
                self.assertEqual(row["canonical_doc_id"], row["doc_id"])

            for key in ("benchmark_exact_copy", "benchmark_copy"):
                self.assertIn(
                    "benchmark_cluster_contamination",
                    decisions[fixture.special[key]]["reasons"],
                )
            self.assertIn(
                "quality:too_short",
                decisions[fixture.special["quality"]]["reasons"],
            )

            # Every assigned split group is still the stable repository or
            # stable English source identity promised by the profile.
            repo_splits: dict[str, set[str]] = {}
            for row in decisions.values():
                group_id = row.get("split_group_id")
                if group_id is None:
                    continue
                provenance = row["provenance"]
                repo_id = provenance.get("repo_id")
                if repo_id:
                    expected = stable_digest("code-repository", str(repo_id)).hex()
                    repo_splits.setdefault(str(repo_id), set()).add(
                        str(row["assigned_split"])
                    )
                else:
                    identity = english_source_identity(
                        str(row["bucket"]), provenance
                    )
                    self.assertIsNotNone(identity)
                    expected = stable_digest("english-source", identity).hex()
                self.assertEqual(group_id, expected)
            self.assertTrue(all(len(splits) == 1 for splits in repo_splits.values()))

            connection = sqlite3.connect(output / ".work" / "curation.sqlite3")
            try:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM near_map").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM phase_progress "
                        "WHERE subphase LIKE 'canonicalize.near_map_%'"
                    ).fetchone()[0],
                    0,
                )
                benchmark_doc = bytes.fromhex(fixture.special["benchmark"])
                benchmark_hashes = connection.execute(
                    "SELECT content_hash, final_cluster FROM documents WHERE doc_id=?",
                    (benchmark_doc,),
                ).fetchone()
                self.assertIsNotNone(benchmark_hashes)
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM benchmark_content_clusters "
                        "WHERE content_hash=?",
                        (benchmark_hashes[0],),
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM benchmark_final_clusters "
                        "WHERE final_cluster=?",
                        (benchmark_hashes[1],),
                    ).fetchone()[0],
                    1,
                )
                for normalized_hash, final_cluster in connection.execute(
                    "SELECT normalized_hash, final_cluster FROM documents"
                ):
                    self.assertEqual(
                        bytes(final_cluster),
                        stable_digest(
                            "fast-global-normalized", bytes(normalized_hash)
                        ),
                    )
            finally:
                connection.close()

    def test_fast_profile_rejects_near_inputs_and_diagnostic_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            common = {
                "root": fixture.root,
                "staging_root": fixture.staging,
                "policy_path": FAST_CANONICAL_POLICY,
                "quota_path": fixture.quota_path,
                "denylist_path": PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                "batch_size": 3,
            }
            with self.assertRaisesRegex(
                CurationError, "must not consume an English near mapping"
            ):
                CurationBuilder(
                    output=fixture.root / "fast-with-near",
                    english_near_clusters=fixture.near_path,
                    allow_missing_english_near_dedup=False,
                    **common,
                )
            with self.assertRaisesRegex(CurationError, "diagnostic-only"):
                CurationBuilder(
                    output=fixture.root / "fast-with-diagnostic",
                    english_near_clusters=None,
                    allow_missing_english_near_dedup=True,
                    **common,
                )
            with self.assertRaisesRegex(
                CurationError, "requires --sqlite-local-work-root"
            ):
                CurationBuilder(
                    output=fixture.root / "fast-handoff-without-local-store",
                    english_near_clusters=None,
                    allow_missing_english_near_dedup=False,
                    defer_raw_archive_integrity_until_finalize=True,
                    fast_all_eligible_handoff=True,
                    **common,
                )

    def test_fast_profile_interrupted_resume_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            fresh = Path(temporary) / "fast-fresh"
            fresh_result = fixture.build_fast(fresh)
            expected_manifest = (fresh / "manifest.json").read_bytes()
            expected_decisions = {
                shard["path"]: (fresh / shard["path"]).read_bytes()
                for shard in fresh_result["manifest"]["decision_shards"]
            }

            output = Path(temporary) / "fast-resumed"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=FAST_CANONICAL_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=None,
                allow_missing_english_near_dedup=False,
                batch_size=3,
            ) as builder:
                original = builder._after_bounded_commit

                def interrupt(
                    subphase: str, progress: dict[str, object]
                ) -> None:
                    original(subphase, progress)
                    if (
                        subphase == "canonicalize.final_choice"
                        and progress["committed_batches"] == 1
                    ):
                        raise RuntimeError("injected fast final-choice interruption")

                with mock.patch.object(
                    builder, "_after_bounded_commit", side_effect=interrupt
                ), self.assertRaisesRegex(RuntimeError, "injected fast"):
                    builder.run()

            resumed = fixture.build_fast(output)
            self.assertTrue(resumed["complete"])
            self.assertEqual((output / "manifest.json").read_bytes(), expected_manifest)
            for relative, payload in expected_decisions.items():
                self.assertEqual((output / relative).read_bytes(), payload)

    def test_english_near_mapping_is_required_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=fixture.root / "missing-near",
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=None,
                allow_missing_english_near_dedup=False,
                batch_size=10,
            ) as builder:
                with self.assertRaisesRegex(RuntimeError, "near-cluster mapping"):
                    builder.run()

    def test_missing_near_diagnostic_override_remains_nonproduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "diagnostic-missing-near"
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=output,
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=None,
                allow_missing_english_near_dedup=True,
                batch_size=3,
            ) as builder:
                result = builder.run()
            manifest = result["manifest"]
            self.assertFalse(manifest["production_ready"])
            self.assertFalse(manifest["english_near_dedup_complete"])
            self.assertNotIn("english_near_dedup_status", manifest)
            self.assertNotIn("english_near_dedup_policy", manifest["identity"])

    def test_english_near_sibling_contract_fails_closed(self) -> None:
        def missing_manifest(fixture: CorpusFixture) -> None:
            (fixture.near_path.parent / "manifest.json").unlink()

        def missing_checksum(fixture: CorpusFixture) -> None:
            (fixture.near_path.parent / "manifest.sha256").unlink()

        def tampered_checksum(fixture: CorpusFixture) -> None:
            (fixture.near_path.parent / "manifest.sha256").write_text(
                "0" * 64 + "  manifest.json\n", encoding="ascii"
            )

        def tampered_manifest(fixture: CorpusFixture) -> None:
            path = fixture.near_path.parent / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["production_ready"] = False
            write_json(path, payload)

        def incomplete_audit(fixture: CorpusFixture) -> None:
            path = fixture.near_path.parent / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["completeness_and_leakage_audit"][
                "mapping_unknown_documents"
            ]
            sign_manifest(path, payload)

        def missing_calibration_evidence(fixture: CorpusFixture) -> None:
            path = fixture.near_path.parent / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["identity"]["calibration_evidence"]
            sign_manifest(path, payload)

        def missing_operational_preflight(fixture: CorpusFixture) -> None:
            path = fixture.near_path.parent / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["refinement_operational_preflight"]
            sign_manifest(path, payload)

        def missing_operational_result(fixture: CorpusFixture) -> None:
            (
                fixture.near_path.parent
                / "operational-preflight-v1"
                / "result.json"
            ).unlink()

        def tampered_operational_result(fixture: CorpusFixture) -> None:
            path = (
                fixture.near_path.parent
                / "operational-preflight-v1"
                / "result.json"
            )
            with path.open("ab") as handle:
                handle.write(b"tamper")

        def tampered_calibration_result(fixture: CorpusFixture) -> None:
            evidence = fixture.near_builder_identity["calibration_evidence"]
            calibration = fixture.root / evidence["result_path"]
            with calibration.open("ab") as handle:
                handle.write(b"tamper")

        def forged_calibration_builder(fixture: CorpusFixture) -> None:
            manifest_path = fixture.near_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = manifest["identity"]["calibration_evidence"]
            calibration_path = fixture.root / evidence["result_path"]
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["identity"]["production_builder_sha256"] = hashlib.sha256(
                b"forged-production-builder"
            ).hexdigest()
            write_json(calibration_path, calibration)
            calibration_sha = file_sha256(calibration_path)
            sidecar_path = fixture.root / evidence["sidecar_path"]
            sidecar_path.write_text(
                f"{calibration_sha}  {calibration_path.name}\n", encoding="utf-8"
            )
            evidence.update(
                result_sha256=calibration_sha,
                result_bytes=calibration_path.stat().st_size,
                sidecar_sha256=file_sha256(sidecar_path),
                identity=calibration["identity"],
                identity_sha256=hashlib.sha256(
                    canonical_json_bytes(calibration["identity"])
                ).hexdigest(),
            )
            sign_manifest(manifest_path, manifest)

        cases = (
            ("missing_manifest", missing_manifest, "manifest"),
            ("missing_checksum", missing_checksum, "manifest checksum"),
            ("tampered_checksum", tampered_checksum, "manifest.sha256 mismatch"),
            ("tampered_manifest", tampered_manifest, "manifest.sha256 mismatch"),
            ("incomplete_audit", incomplete_audit, "audit is incomplete"),
            (
                "missing_calibration_evidence",
                missing_calibration_evidence,
                "identity is incomplete",
            ),
            (
                "missing_operational_preflight",
                missing_operational_preflight,
                "operational preflight evidence is incomplete",
            ),
            (
                "missing_operational_result",
                missing_operational_result,
                "Missing English near-dedup operational preflight result",
            ),
            (
                "tampered_operational_result",
                tampered_operational_result,
                "operational preflight result identity mismatch",
            ),
            (
                "tampered_calibration_result",
                tampered_calibration_result,
                "calibration result identity mismatch",
            ),
            (
                "forged_calibration_builder",
                forged_calibration_builder,
                "production_builder_sha256 mismatch",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = CorpusFixture(Path(temporary))
                mutate(fixture)
                with self.assertRaisesRegex(RuntimeError, message):
                    fixture.build(fixture.root / "curated")

    def test_english_near_operational_preflight_rejects_resigned_stale_evidence(
        self,
    ) -> None:
        def resign(fixture: CorpusFixture, mutate: object) -> None:
            manifest_path = fixture.near_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = manifest["refinement_operational_preflight"]
            result_path = fixture.near_path.parent / evidence["result_path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            mutate(result)
            write_json(result_path, result)
            result_raw = result_path.read_bytes()
            result_sha = hashlib.sha256(result_raw).hexdigest()
            sidecar_path = fixture.near_path.parent / evidence["sidecar_path"]
            sidecar_path.write_text(
                f"{result_sha}  result.json\n", encoding="ascii"
            )
            for field in ("identity", "thresholds", "sample", "measurements"):
                evidence[field] = result[field]
            evidence.update(
                result_sha256=result_sha,
                result_bytes=len(result_raw),
                sidecar_sha256=file_sha256(sidecar_path),
                identity_sha256=hashlib.sha256(
                    canonical_json_bytes(result["identity"])
                ).hexdigest(),
            )
            sign_manifest(manifest_path, manifest)

        cases = (
            (
                "threshold",
                lambda result: result["thresholds"].__setitem__(
                    "requested_pairs",
                    result["thresholds"]["requested_pairs"] + 1,
                ),
                "thresholds mismatch",
            ),
            (
                "formula",
                lambda result: result["measurements"].__setitem__(
                    "projected_additional_refinement_sqlite_bytes",
                    result["measurements"][
                        "projected_additional_refinement_sqlite_bytes"
                    ]
                    + 1,
                ),
                "formula projected_additional_refinement_sqlite_bytes mismatch",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = CorpusFixture(Path(temporary))
                resign(fixture, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    fixture.build(fixture.root / "curated")

    def test_english_near_five_file_publication_is_resnapshotted_before_mapping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            with CurationBuilder(
                root=fixture.root,
                staging_root=fixture.staging,
                output=fixture.root / "curated",
                policy_path=DEFAULT_POLICY,
                quota_path=fixture.quota_path,
                denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                english_near_clusters=fixture.near_path,
                allow_missing_english_near_dedup=False,
                batch_size=10,
            ) as builder:
                result_path = (
                    fixture.near_path.parent
                    / "operational-preflight-v1"
                    / "result.json"
                )
                with result_path.open("ab") as handle:
                    handle.write(b"tamper-after-construction")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "publication changed before mapping ingestion",
                ):
                    builder.run()

    def test_english_near_manifest_identity_drift_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated"
            partial = fixture.build(output, max_new_archives=1)
            self.assertFalse(partial["complete"])
            manifest_path = fixture.near_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["identity"]["runtime"]["python"] = "fixture-python-drift"
            sign_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "operational preflight identity builder_identity_sha256 mismatch",
            ):
                fixture.build(output)

    def test_fingerprint_checksum_mismatch_fails_before_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            fingerprint = next((fixture.staging / "fingerprints").glob("*/*.jsonl.zst"))
            with fingerprint.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(
                RuntimeError,
                "[Ff]ingerprint checksum mismatch",
            ):
                with CurationBuilder(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    output=fixture.root / "corrupt",
                    policy_path=DEFAULT_POLICY,
                    quota_path=fixture.quota_path,
                    denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                    english_near_clusters=fixture.near_path,
                    allow_missing_english_near_dedup=False,
                    batch_size=10,
                ) as builder:
                    builder.run()

    def test_raw_archive_checksum_mismatch_fails_before_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            raw_archive = fixture.root / "raw/python/part-000000.tar.zst"
            original = raw_archive.read_bytes()
            # Preserve the byte count so this specifically exercises the
            # SHA-256 gate rather than the cheaper size check.
            raw_archive.write_bytes(b"X" + original[1:])
            with self.assertRaisesRegex(
                RuntimeError,
                "raw archive checksum mismatch",
            ):
                fixture.build(fixture.root / "corrupt-raw")

    def test_completeness_gate_rejects_incomplete_or_ambiguous_inputs(self) -> None:
        def missing_report(fixture: CorpusFixture) -> None:
            next((fixture.staging / "reports").glob("*/*.json")).unlink()

        def extra_report(fixture: CorpusFixture) -> None:
            source = next((fixture.staging / "reports" / "python").glob("*.json"))
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["index"] = 999_999
            payload["archive"] = "raw/python/part-999999.tar.zst"
            payload["quota_shard_id"] = "fixture-python-999999"
            write_json(fixture.staging / "reports" / "python" / "part-999999.json", payload)

        def duplicate_report(fixture: CorpusFixture) -> None:
            source = next((fixture.staging / "reports" / "python").glob("*.json"))
            duplicate = fixture.staging / "reports" / "python" / "duplicate.json"
            duplicate.write_bytes(source.read_bytes())

        def quota_mismatch(fixture: CorpusFixture) -> None:
            report = next((fixture.staging / "reports").glob("*/*.json"))
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["exact_tokens"] += 1
            write_json(report, payload)

        def preprocess_error(fixture: CorpusFixture) -> None:
            write_json(
                fixture.staging / "errors" / "python" / "part-000000.json",
                {"error": "fixture"},
            )

        def pending_archive(fixture: CorpusFixture) -> None:
            path = fixture.root / "raw" / "python" / ".part-000002-fixture.tar.zst"
            path.write_bytes(b"pending")

        def extra_raw_archive(fixture: CorpusFixture) -> None:
            path = fixture.root / "raw" / "python" / "part-999999.tar.zst"
            path.write_bytes(b"not-finalized-in-the-ledger")

        cases = (
            ("missing_report", missing_report, "missing fingerprint reports"),
            ("extra_report", extra_report, "no finalized quota record"),
            ("duplicate_report", duplicate_report, "not at its canonical path"),
            ("quota_mismatch", quota_mismatch, "Report/quota exact_tokens mismatch"),
            ("preprocess_error", preprocess_error, "Preprocessing error records remain"),
            ("pending_archive", pending_archive, "Pending raw archive inputs remain"),
            ("extra_raw_archive", extra_raw_archive, "no finalized quota record"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = CorpusFixture(Path(temporary))
                mutate(fixture)
                with self.assertRaisesRegex(RuntimeError, message):
                    fixture.build(fixture.root / "curated")

    def test_collection_authority_drift_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated"
            partial = fixture.build(output, max_new_archives=1)
            self.assertFalse(partial["complete"])
            record_path = fixture.quota_record_paths[("python", 0)]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["audit_note"] = "semantically neutral but identity-changing"
            write_json(record_path, record)
            with self.assertRaisesRegex(
                RuntimeError,
                "Resume identity mismatch for collection_completeness",
            ):
                fixture.build(output)

    def test_mount_detection_and_journal_policy_are_fail_safe(self) -> None:
        mountinfo = "\n".join(
            (
                "20 1 8:1 / / rw,relatime - ext4 /dev/root rw",
                "21 20 259:1 / /local-nvme rw,noatime - xfs /dev/nvme1n1 rw",
                "22 20 0:42 / /workspace rw,relatime - nfs4 server:/volume rw",
                "23 20 0:43 / /mystery rw,relatime - fuse.unknown mystery rw",
            )
        )
        local = detect_output_mount(
            Path("/local-nvme/curation-v1"),
            system="Linux",
            linux_mountinfo=mountinfo,
        )
        network = detect_output_mount(
            Path("/workspace/dataset/curated"),
            system="Linux",
            linux_mountinfo=mountinfo,
        )
        unknown = detect_output_mount(
            Path("/mystery/curated"),
            system="Linux",
            linux_mountinfo=mountinfo,
        )
        darwin = detect_output_mount(
            Path("/Users/research/curated"),
            system="Darwin",
            darwin_mounts=(
                "/dev/disk1s1 on / (apfs, sealed, local, read-only)\n"
                "/dev/disk1s5 on /System/Volumes/Data (apfs, local, journaled)\n"
            ),
            darwin_df=(
                "Filesystem 512-blocks Used Available Capacity Mounted on\n"
                "/dev/disk1s5 1000 100 900 10% /System/Volumes/Data\n"
            ),
        )
        self.assertEqual(local["classification"], "local")
        self.assertEqual(local["mount_point"], "/local-nvme")
        self.assertEqual(network["classification"], "network")
        self.assertEqual(network["filesystem_type"], "nfs4")
        self.assertEqual(unknown["classification"], "unknown")
        self.assertEqual(darwin["classification"], "local")
        self.assertEqual(darwin["mount_point"], "/System/Volumes/Data")
        self.assertEqual(
            select_sqlite_journal_policy(local, "auto")["selected_mode"],
            "wal",
        )
        self.assertEqual(
            select_sqlite_journal_policy(network, "auto")["selected_mode"],
            "delete",
        )
        self.assertEqual(
            select_sqlite_journal_policy(unknown, "auto")["selected_mode"],
            "delete",
        )
        self.assertEqual(
            select_sqlite_journal_policy(local, "delete")["selected_mode"],
            "delete",
        )
        with self.assertRaisesRegex(RuntimeError, "positively identified local"):
            select_sqlite_journal_policy(network, "wal")
        with self.assertRaisesRegex(RuntimeError, "exactly one of"):
            select_sqlite_journal_policy(local, "truncate")

    def test_database_work_subtree_must_match_frozen_output_mount(self) -> None:
        first_mount = {
            "detector": "fixture",
            "mount_point": "/workspace",
            "filesystem_type": "nfs4",
            "source": "server:/expected",
            "device": "0:42",
            "options": ["hard", "local_lock=none", "rw"],
            "classification": "network",
        }
        different_mount = {
            **first_mount,
            "mount_point": "/workspace/dataset/curated/selection-fast-v1/.work",
            "source": "/dev/ephemeral",
            "filesystem_type": "ext4",
            "device": "8:1",
            "classification": "local",
        }
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated-mount-drift"
            with mock.patch.object(
                curate_corpus_module,
                "detect_output_mount",
                side_effect=(first_mount, first_mount, different_mount),
            ):
                builder = CurationBuilder(
                    root=fixture.root,
                    staging_root=fixture.staging,
                    output=output,
                    policy_path=DEFAULT_POLICY,
                    quota_path=fixture.quota_path,
                    denylist_path=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
                    english_near_clusters=fixture.near_path,
                    allow_missing_english_near_dedup=False,
                    batch_size=3,
                    sqlite_journal_mode="delete",
                )
                with self.assertRaisesRegex(
                    CurationError, "database work mount differs"
                ):
                    with builder:
                        pass

    def test_explicit_delete_override_is_applied_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated-delete"
            result = fixture.build(output, sqlite_journal_mode="delete")
            policy = result["manifest"]["identity"]["sqlite_runtime"][
                "journal_policy"
            ]
            self.assertEqual(policy["requested_mode"], "delete")
            self.assertEqual(policy["selected_mode"], "delete")
            connection = sqlite3.connect(output / ".work" / "curation.sqlite3")
            try:
                self.assertEqual(
                    str(connection.execute("PRAGMA journal_mode").fetchone()[0]).casefold(),
                    "delete",
                )
            finally:
                connection.close()

    def test_journal_mode_and_mount_evidence_drift_reject_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated"
            partial = fixture.build(
                output,
                max_new_archives=1,
                sqlite_journal_mode="delete",
            )
            self.assertFalse(partial["complete"])
            with self.assertRaisesRegex(
                RuntimeError,
                "journal mode mismatch|Resume identity mismatch for sqlite_runtime",
            ):
                fixture.build(output, sqlite_journal_mode="auto")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = CorpusFixture(Path(temporary))
            output = fixture.root / "curated"
            first_mount = {
                "detector": "fixture",
                "mount_point": "/local-a",
                "filesystem_type": "xfs",
                "source": "/dev/a",
                "device": "1:1",
                "options": ["rw"],
                "classification": "local",
            }
            second_mount = {**first_mount, "source": "/dev/b", "device": "1:2"}
            with mock.patch.object(
                curate_corpus_module,
                "detect_output_mount",
                return_value=first_mount,
            ):
                partial = fixture.build(
                    output,
                    max_new_archives=1,
                    sqlite_journal_mode="delete",
                )
                self.assertFalse(partial["complete"])
            with mock.patch.object(
                curate_corpus_module,
                "detect_output_mount",
                return_value=second_mount,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Resume identity mismatch for sqlite_runtime",
                ):
                    fixture.build(output, sqlite_journal_mode="delete")


if __name__ == "__main__":
    unittest.main()
