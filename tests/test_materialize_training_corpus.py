from __future__ import annotations

import array
import hashlib
import io
import json
import math
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zstandard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from pretrain.data import validate_packed_manifest, validate_training_order
from pretrain.materialize import (
    CorpusMaterializer,
    FAST_CURATION_PROFILE,
    MaterializationConfig,
    MaterializationError,
    canonical_sha256,
    file_sha256,
    iter_jsonl_zst,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_zst(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        compressor = zstandard.ZstdCompressor(level=1, threads=0, write_checksum=True)
        with compressor.stream_writer(raw, closefd=False) as output:
            for row in rows:
                output.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )


class MaterializationFixture:
    REVISION = "1" * 40
    VOCAB_SIZE = 32

    def __init__(self, root: Path, *, malformed_prefix: bool = False) -> None:
        self.root = root
        self.preprocess = root / "staging" / "preprocess"
        self.selection = root / "curated" / "selection-v1"
        self.tokenizer_root = root / "tokenizer" / "starcoder2"
        self.policy_path = root / "policy.json"
        self.quota_path = root / "quotas.json"
        self.denylist_path = root / "denylist.json"
        self.malformed_prefix = malformed_prefix
        self._write_tokenizer()
        self._write_supporting_identities()
        self._write_corpus_and_selection()

    @property
    def config(self) -> MaterializationConfig:
        # Packing yields six rows. Order v4 selects the strict 2/2/1 domain
        # allocation (five rows, 20 inputs) and audits one packed-surplus row.
        return MaterializationConfig(
            sequence_length=4,
            rows_per_shard=1,
            construction_seed=17,
            order_seed=29,
            frozen_global_microbatch_rows=1,
            frozen_gradient_accumulation_steps=1,
            expected_train_input_tokens=20,
            expected_validation_input_tokens=20,
            expected_test_input_tokens=20,
            train_input_token_tolerance=0,
            validation_input_token_tolerance=0,
            test_input_token_tolerance=0,
            enforce_input_weights=True,
            expected_vocab_size=self.VOCAB_SIZE,
            expected_eos_token_id=0,
        )

    def materializer(
        self,
        output: Path,
        *,
        config: MaterializationConfig | None = None,
        tokenizer_batch_documents: int = 256,
        tokenizer_batch_bytes: int = 64 * 1024 * 1024,
        tokenizer_max_document_bytes: int | None = None,
        fault_injector: Any | None = None,
    ) -> CorpusMaterializer:
        return CorpusMaterializer(
            raw_root=self.root,
            preprocess_root=self.preprocess,
            selection_root=self.selection,
            tokenizer_root=self.tokenizer_root,
            policy_path=self.policy_path,
            quota_path=self.quota_path,
            benchmark_denylist_path=self.denylist_path,
            output_root=output,
            config=config or self.config,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_bytes=tokenizer_batch_bytes,
            tokenizer_max_document_bytes=tokenizer_max_document_bytes,
            fault_injector=fault_injector,
        )

    def _write_tokenizer(self) -> None:
        self.tokenizer_root.mkdir(parents=True)
        vocabulary = {"<eos>": 0, "<unk>": 1}
        for index, token in enumerate("abcdefghijklmnopqrstuvwxyz", 2):
            vocabulary[token] = index
        for index in range(len(vocabulary), self.VOCAB_SIZE):
            vocabulary[f"unused-{index}"] = index
        tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer_path = self.tokenizer_root / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        manifest = {
            "manifest_version": 1,
            "repo_id": "fixture/tokenizer",
            "resolved_revision": self.REVISION,
            "files": {
                "tokenizer.json": {
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": file_sha256(tokenizer_path),
                }
            },
            "validation": {
                "vocab_size": self.VOCAB_SIZE,
                "eos_token": "<eos>",
                "eos_token_id": 0,
            },
        }
        write_json(self.tokenizer_root / "TOKENIZER_MANIFEST.json", manifest)

    def _write_supporting_identities(self) -> None:
        self.policy = {
            "policy_version": 1,
            "selection": {
                "selection_version": 1,
                "seed": "fixture",
                "terminal_document_action": "keep_token_prefix_to_exact_quota",
            },
        }
        write_json(self.policy_path, self.policy)
        write_json(self.quota_path, {"version": 1, "fixture": True})
        write_json(self.denylist_path, {"manifest_version": 1, "benchmark": "fixture"})
        write_json(
            self.preprocess / "PREPROCESS_MANIFEST.json",
            {"manifest_version": 1, "raw_data_mutated": False},
        )
        self.source_descriptors: dict[str, dict[str, str]] = {}
        for name in (
            "STACK_V3_SOURCE.json",
            "FINEWEB_EDU_SOURCE.json",
            "WIKIPEDIA_SOURCE.json",
        ):
            path = self.root / "manifests" / name
            payload: dict[str, Any] = {
                "manifest_version": 1,
                "repo_id": f"fixture/{name.removesuffix('_SOURCE.json').lower()}",
                "resolved_revision": self.REVISION,
            }
            if name != "STACK_V3_SOURCE.json":
                payload.update(
                    dataset_config="fixture",
                    tokenizer_revision=self.REVISION,
                )
            write_json(path, payload)
            descriptor = {
                "sha256": file_sha256(path),
                "resolved_revision": self.REVISION,
            }
            if name != "STACK_V3_SOURCE.json":
                descriptor["tokenizer_revision"] = self.REVISION
            self.source_descriptors[name] = descriptor

    def _english_near_artifact(
        self,
        *,
        input_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        audit = {
            "english_documents_inventory": 14,
            "english_documents_mapped": 14,
            "mapping_missing_documents": 0,
            "mapping_unknown_documents": 0,
            "mapping_duplicate_documents": 0,
            "clusters": 14,
            "singleton_clusters": 14,
            "cross_source_clusters": 0,
            "normalized_hashes_in_multiple_clusters": 0,
            "invalid_cluster_roots": 0,
        }
        near_collection = {
            "evidence_version": 1,
            "quota_config_path": "configs/data_quotas.json",
            "quota_config_sha256": file_sha256(self.quota_path),
            "quota_record_inventory_sha256": hashlib.sha256(
                b"fixture-near-quota-inventory"
            ).hexdigest(),
            "quota_records": 2,
            "buckets": {
                "fineweb_edu": {"archives": 1, "documents": 7},
                "wikipedia": {"archives": 1, "documents": 7},
            },
        }
        english_reports = [
            row for row in input_reports if str(row["archive"]).startswith("raw/english/")
        ]
        selected_reports = [
            {
                "report_path": row["report"],
                "report_sha256": row["report_sha256"],
                "archive": row["archive"],
                "archive_sha256": row["archive_sha256"],
                "fingerprint_file": row["fingerprint_file"],
                "fingerprint_sha256": row["fingerprint_sha256"],
                "documents": row["documents"],
            }
            for row in english_reports
        ]
        near_report_inventory_sha = canonical_sha256(
            [
                {
                    "path": row["report"],
                    "sha256": row["report_sha256"],
                    "fingerprint_sha256": row["fingerprint_sha256"],
                }
                for row in english_reports
            ]
        )
        english_sources: dict[str, Any] = {
            name: descriptor
            for name, descriptor in self.source_descriptors.items()
            if name != "STACK_V3_SOURCE.json"
        }
        tokenizer_manifest = self.tokenizer_root / "TOKENIZER_MANIFEST.json"
        english_sources["TOKENIZER_MANIFEST.json"] = {
            "sha256": file_sha256(tokenizer_manifest),
            "resolved_revision": self.REVISION,
        }
        near_builder_path = PROJECT_ROOT / "scripts" / "build_english_near_clusters.py"
        near_config_path = PROJECT_ROOT / "configs" / "english_near_dedup.json"
        near_config_raw = near_config_path.read_bytes()
        near_config = json.loads(near_config_raw)
        near_builder_sha = file_sha256(near_builder_path)
        near_config_file_sha = hashlib.sha256(near_config_raw).hexdigest()
        near_config_sha = canonical_sha256(near_config)
        calibration_harness = (
            PROJECT_ROOT / "scripts" / "calibrate_english_near_dedup.py"
        )
        calibration_config_path = (
            PROJECT_ROOT / "configs" / "english_near_dedup_calibration.json"
        )
        calibration_config_raw = calibration_config_path.read_bytes()
        calibration_config = json.loads(calibration_config_raw)
        sample_manifest = [
            {
                "doc_id": hashlib.sha256(b"calibration-sample").hexdigest(),
                "bucket": "fineweb_edu",
            }
        ]
        calibration_input = {
            "kind": "immutable_real_english_sample",
            "full_report_inventory_sha256": near_report_inventory_sha,
            "preprocess_manifest_sha256": file_sha256(
                self.preprocess / "PREPROCESS_MANIFEST.json"
            ),
            "curation_policy_sha256": canonical_sha256(self.policy),
            "benchmark_guard_sha256": file_sha256(self.denylist_path),
            "source_manifests": english_sources,
            "collection_completeness_sha256": canonical_sha256(near_collection),
            "collection_completeness": near_collection,
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
            "sampling_counts": {
                "fineweb_edu": {"archives": 1, "documents": 7},
                "wikipedia": {"archives": 1, "documents": 7},
            },
            "selected_reports": selected_reports,
            "documents_selected": 14,
        }
        calibration_identity = {
            "harness_sha256": file_sha256(calibration_harness),
            "production_builder_sha256": near_builder_sha,
            "calibration_algorithm": calibration_config["calibration_algorithm"],
            "calibration_seed": calibration_config["seed"],
            "production_config_file_sha256": near_config_file_sha,
            "production_config_canonical_sha256": near_config_sha,
            "calibration_config_file_sha256": hashlib.sha256(
                calibration_config_raw
            ).hexdigest(),
            "calibration_config_canonical_sha256": canonical_sha256(
                calibration_config
            ),
            "input": calibration_input,
            "sample_manifest_sha256": canonical_sha256(sample_manifest),
            "runtime": {"python": "fixture-python"},
        }
        calibration_result = {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_overrides": {},
            "acceptance_failures": [],
            "identity": calibration_identity,
            "production_threshold": {
                "minimum_jaccard_numerator": near_config["refinement"][
                    "minimum_jaccard_numerator"
                ],
                "minimum_jaccard_denominator": near_config["refinement"][
                    "minimum_jaccard_denominator"
                ],
            },
            "acceptance": calibration_config["acceptance"],
            "sampling": {
                "documents_input": 14,
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
            "identity_sha256": canonical_sha256(calibration_identity),
            "identity": calibration_identity,
        }
        near_identity = {
            "format_version": 1,
            "builder_sha256": near_builder_sha,
            "config_file_sha256": near_config_file_sha,
            "config_sha256": near_config_sha,
            "curation_policy_sha256": canonical_sha256(self.policy),
            "preprocess_manifest_sha256": file_sha256(
                self.preprocess / "PREPROCESS_MANIFEST.json"
            ),
            "benchmark_guard_sha256": file_sha256(self.denylist_path),
            "report_inventory_sha256": near_report_inventory_sha,
            "report_count": 2,
            "source_manifests": english_sources,
            "collection_completeness": near_collection,
            "calibration_evidence": calibration_evidence,
            "runtime": {
                "python": "fixture-python",
                "sqlite": "fixture-sqlite",
                "xxhash": "fixture-xxhash",
                "zstandard": zstandard.__version__,
                "storage": {
                    "filesystem_type": "ext4",
                    "mount_point": "/fixture",
                    "mount_source": "/dev/fixture",
                    "mount_options": "rw",
                    "detection": "fixture",
                    "classification": "proven-local",
                    "sqlite_journal_mode_configured": near_config["storage"][
                        "sqlite_journal_mode"
                    ],
                    "sqlite_journal_mode_requested": near_config["storage"][
                        "sqlite_journal_mode"
                    ],
                    "sqlite_journal_mode_request_source": "config",
                    "sqlite_journal_mode_selected": "wal",
                    "sqlite_journal_mode_actual": "wal",
                    "policy": {
                        "network_or_unknown_action": "delete",
                        "wal_on_non_allowlisted_action": "fail_closed",
                        "wal_local_filesystem_allowlist": near_config["storage"][
                            "wal_local_filesystem_allowlist"
                        ],
                    },
                },
            },
        }
        near_root = self.root / "staging" / "english-near"
        mapping_path = near_root / "clusters.jsonl.zst"
        write_jsonl_zst(
            mapping_path,
            [
                {
                    "doc_id": hashlib.sha256(
                        f"fixture-near-document-{index}".encode()
                    ).hexdigest(),
                    "cluster_id": f"fixture-cluster-{index}",
                }
                for index in range(14)
            ],
        )
        mapping = {
            "path": "clusters.jsonl.zst",
            "sha256": file_sha256(mapping_path),
            "bytes": mapping_path.stat().st_size,
            "records": 14,
            "ordered_by": "frozen_inventory_ordinal",
            "singleton_clusters_included": True,
        }
        thresholds = near_config["operational_preflight"]
        total_candidates = 10
        expected_pairs = min(thresholds["requested_pairs"], total_candidates)
        elapsed = 0.01
        rate = round(expected_pairs / elapsed, 6)
        measured_growth = 100
        bytes_per_pair = measured_growth / expected_pairs
        projected_growth = math.ceil(bytes_per_pair * total_candidates)
        projected_with_safety = math.ceil(
            projected_growth
            * thresholds["disk_projection_safety_numerator"]
            / thresholds["disk_projection_safety_denominator"]
        )
        required_free = (
            projected_with_safety
            + thresholds["minimum_post_refinement_free_bytes"]
        )
        cache_bytes = 1024
        before = {
            "filesystem_total_bytes": required_free * 2,
            "filesystem_free_bytes": required_free + 1024,
            "sqlite_state_bytes": 4096,
            "refinement_cache_bytes": cache_bytes,
            "peak_process_rss_bytes": 1024,
        }
        after = {
            **before,
            "sqlite_state_bytes": 4196,
            "peak_process_rss_bytes": 2048,
        }
        item_size = array.array("Q").itemsize
        parent_bytes = 15 * item_size
        parent_with_safety = math.ceil(
            parent_bytes
            * thresholds["union_parent_memory_safety_numerator"]
            / thresholds["union_parent_memory_safety_denominator"]
        )
        preflight_identity = {
            "contract_version": 1,
            "builder_sha256": near_identity["builder_sha256"],
            "builder_identity_sha256": canonical_sha256(near_identity),
            "config_file_sha256": near_identity["config_file_sha256"],
            "config_sha256": near_identity["config_sha256"],
            "calibration_evidence_sha256": canonical_sha256(
                calibration_evidence
            ),
            "report_inventory_sha256": near_identity["report_inventory_sha256"],
            "preprocess_manifest_sha256": near_identity["preprocess_manifest_sha256"],
            "curation_policy_sha256": near_identity["curation_policy_sha256"],
            "benchmark_guard_sha256": near_identity["benchmark_guard_sha256"],
            "collection_completeness_sha256": canonical_sha256(near_collection),
            "candidate_pairs_total": total_candidates,
            "documents_total": 14,
            "candidate_blocks_total": 1,
            "candidate_blocks_committed": 1,
            "phase_at_measurement": "refine",
            "candidate_cursor_at_measurement": {"band": 0, "key": "00" * 8},
            "refinement_cursor_at_measurement": None,
            "cache_archives": 2,
            "cache_bytes": cache_bytes,
            "cache_inventory_sha256": hashlib.sha256(
                b"fixture-cache-inventory"
            ).hexdigest(),
            "runtime_storage": near_identity["runtime"]["storage"],
        }
        sample = {
            "algorithm": thresholds["sampling_algorithm"],
            "seed": thresholds["sampling_seed"],
            "requested_pairs": thresholds["requested_pairs"],
            "expected_pairs": expected_pairs,
            "measured_pairs": expected_pairs,
            "sample_pairs_sha256": canonical_sha256(
                [
                    {"left_document": index, "right_document": index + 1}
                    for index in range(expected_pairs)
                ]
            ),
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
            "projected_additional_with_safety_bytes": projected_with_safety,
            "required_filesystem_free_bytes": required_free,
            "union_parent_item_bytes": item_size,
            "union_parent_array_projected_bytes": parent_bytes,
            "union_parent_array_with_safety_bytes": parent_with_safety,
            "union_projected_peak_process_rss_bytes": (
                before["peak_process_rss_bytes"] + parent_with_safety
            ),
            "resources_before": before,
            "resources_after": after,
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
        preflight_path = near_root / "operational-preflight-v1" / "result.json"
        write_json(preflight_path, preflight_result)
        preflight_raw = preflight_path.read_bytes()
        preflight_sha = hashlib.sha256(preflight_raw).hexdigest()
        preflight_sidecar = preflight_path.with_name("result.json.sha256")
        preflight_sidecar.write_text(
            f"{preflight_sha}  result.json\n", encoding="ascii"
        )
        preflight = {
            "contract_version": 1,
            "result_path": "operational-preflight-v1/result.json",
            "result_sha256": preflight_sha,
            "result_bytes": len(preflight_raw),
            "sidecar_path": "operational-preflight-v1/result.json.sha256",
            "sidecar_sha256": file_sha256(preflight_sidecar),
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
            "identity_sha256": canonical_sha256(preflight_identity),
            "identity": preflight_identity,
            "thresholds": thresholds,
            "sample": sample,
            "measurements": measurements,
        }
        near_manifest = {
            "manifest_version": 1,
            "mapping_record_version": 1,
            "production_ready": True,
            "identity": near_identity,
            "refinement_operational_preflight": preflight,
            "algorithm": {
                "config": str(near_config_path),
                "config_file_sha256": near_config_file_sha,
                "config_sha256": near_config_sha,
                "name": near_config["algorithm"],
                "compact_preprocess_sketch_role": (
                    "validated-integrity-only-not-candidate-authority"
                ),
                "raw_text_candidate_pass": True,
                "full_shingle_refinement": True,
                "jaccard_threshold": (
                    near_config["refinement"]["minimum_jaccard_numerator"]
                    / near_config["refinement"]["minimum_jaccard_denominator"]
                ),
                "ideal_independent_minhash_candidate_recall_at_threshold": 0.99,
                "statistical_limitation": "fixture statistical limitation",
                "hash_limitation": "fixture hash limitation",
                "posting_overflow_action": "fail_closed_without_truncation",
            },
            "inputs": {
                "report_inventory_sha256": near_report_inventory_sha,
                "reports": selected_reports,
            },
            "candidate_stats": {
                "blocks": 1,
                "maximum_posting_documents": 2,
                "raw_posting_pairs": total_candidates,
                "length_pruned_pairs": 0,
                "unique_candidate_pairs": total_candidates,
                "accepted_near_pairs": 0,
            },
            "completeness_and_leakage_audit": audit,
            "database_integrity_check": "ok",
            "mapping": mapping,
        }
        near_manifest_path = near_root / "manifest.json"
        write_json(near_manifest_path, near_manifest)
        near_manifest_raw = near_manifest_path.read_bytes()
        near_manifest_sha = hashlib.sha256(near_manifest_raw).hexdigest()
        near_manifest_sidecar = near_root / "manifest.sha256"
        near_manifest_sidecar.write_text(
            f"{near_manifest_sha}  manifest.json\n", encoding="ascii"
        )
        return {
            "contract_version": 2,
            "publication_root": str(near_root.relative_to(self.root)),
            "manifest": {
                "path": str(near_manifest_path.relative_to(self.root)),
                "sha256": near_manifest_sha,
                "bytes": len(near_manifest_raw),
                "sidecar_path": str(near_manifest_sidecar.relative_to(self.root)),
                "sidecar_sha256": file_sha256(near_manifest_sidecar),
            },
            "identity": near_identity,
            "mapping": mapping,
            "refinement_operational_preflight": preflight,
            "database_integrity_check": "ok",
            "completeness_and_leakage_audit": audit,
        }

    @staticmethod
    def _archive_definition(bucket: str) -> tuple[str, str]:
        if bucket == "python":
            return "raw/python/part-000000.tar.zst", "Python"
        if bucket == "other_code":
            return "raw/other_code/part-000000.tar.zst", "Rust"
        if bucket == "wikipedia":
            return "raw/english/wikipedia/part-000000.tar.zst", "English"
        return "raw/english/fineweb_edu/part-000000.tar.zst", "English"

    def _document_rows(
        self, bucket: str, archive: str
    ) -> tuple[list[tuple[str, bytes, dict[str, Any]]], list[dict[str, Any]]]:
        category = (
            "english" if bucket in ("fineweb_edu", "wikipedia") else bucket
        )
        documents: list[tuple[str, bytes, dict[str, Any]]] = []
        decisions: list[dict[str, Any]] = []
        row_index = 0
        bucket_tokens = {
            "python": ("a", "b", "c"),
            "other_code": ("d", "e", "f"),
            "fineweb_edu": ("g", "h", "i"),
            "wikipedia": ("j", "k", "l"),
        }
        for split, token in zip(
            ("train", "validation", "test"), bucket_tokens[bucket]
        ):
            for document_in_split, count in enumerate((3, 8)):
                member = f"files/{row_index:08d}.txt"
                content = " ".join([token] * count).encode("utf-8")
                provenance: dict[str, Any]
                if bucket in ("fineweb_edu", "wikipedia"):
                    provenance = {
                        "url": f"https://example.test/{split}/{document_in_split}",
                        "language": "English",
                    }
                else:
                    provenance = {
                        "repo_id": f"repo-{split}",
                        "repo_path": f"repo-{split}",
                        "language": "Python" if bucket == "python" else "Rust",
                    }
                raw_manifest = {
                    "member_path": member,
                    **provenance,
                    "size_bytes": len(content),
                    "starcoder2_tokens": count,
                }
                selected = count if document_in_split == 0 else count - 1
                token_prefix = [0, selected]
                if self.malformed_prefix and bucket == "python" and split == "train" and document_in_split == 1:
                    token_prefix = [0, selected - 1]
                doc_id = hashlib.sha256(f"{archive}\0{member}".encode()).hexdigest()
                decisions.append(
                    {
                        "record_version": 1,
                        "doc_id": doc_id,
                        "bucket": bucket,
                        "category": category,
                        "archive": archive,
                        "archive_index": 0,
                        "manifest_index": row_index,
                        "member_path": member,
                        "decision": "keep",
                        "split": split,
                        "assigned_split": split,
                        "source_tokens": count,
                        "selected_tokens": selected,
                        "token_prefix": token_prefix,
                        "terminal_quota_prefix": selected < count,
                        "canonical_doc_id": doc_id,
                        "split_group_id": hashlib.sha256(
                            f"group:{split}:{category}".encode()
                        ).hexdigest(),
                        "reasons": [],
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "normalized_sha256": hashlib.sha256(content.lower()).hexdigest(),
                        "quality_flags": [],
                        "benchmark_reason": None,
                        "provenance": provenance,
                    }
                )
                documents.append((member, content, raw_manifest))
                row_index += 1
        member = f"files/{row_index:08d}.txt"
        content = b"d d"
        provenance = (
            {"url": "https://example.test/rejected", "language": "English"}
            if bucket in ("fineweb_edu", "wikipedia")
            else {
                "repo_id": "repo-rejected",
                "repo_path": "repo-rejected",
                "language": "Python" if bucket == "python" else "Rust",
            }
        )
        raw_manifest = {
            "member_path": member,
            **provenance,
            "size_bytes": len(content),
            "starcoder2_tokens": 2,
        }
        doc_id = hashlib.sha256(f"{archive}\0{member}".encode()).hexdigest()
        decisions.append(
            {
                "record_version": 1,
                "doc_id": doc_id,
                "bucket": bucket,
                "category": category,
                "archive": archive,
                "archive_index": 0,
                "manifest_index": row_index,
                "member_path": member,
                "decision": "reject",
                "split": None,
                "assigned_split": "train",
                "source_tokens": 2,
                "selected_tokens": 0,
                "token_prefix": None,
                "terminal_quota_prefix": False,
                "canonical_doc_id": doc_id,
                "split_group_id": hashlib.sha256(b"rejected-group").hexdigest(),
                "reasons": ["quality:fixture"],
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "normalized_sha256": hashlib.sha256(content.lower()).hexdigest(),
                "quality_flags": ["fixture"],
                "benchmark_reason": None,
                "provenance": provenance,
            }
        )
        documents.append((member, content, raw_manifest))
        return documents, decisions

    @staticmethod
    def _write_raw_archive(
        path: Path, documents: list[tuple[str, bytes, dict[str, Any]]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for _member, _content, row in documents
        )
        with path.open("wb") as raw:
            compressor = zstandard.ZstdCompressor(
                level=1, threads=0, write_checksum=True
            )
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
                ) as archive:
                    for member_name, content, _row in documents:
                        info = tarfile.TarInfo(member_name)
                        info.size = len(content)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        archive.addfile(info, io.BytesIO(content))
                    info = tarfile.TarInfo("_manifest.jsonl")
                    info.size = len(manifest)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(manifest))

    def _write_corpus_and_selection(self) -> None:
        input_reports: list[dict[str, Any]] = []
        decision_shards: list[dict[str, Any]] = []
        report_payloads: dict[str, dict[str, Any]] = {}
        for bucket in ("fineweb_edu", "other_code", "python", "wikipedia"):
            archive_relative, _language = self._archive_definition(bucket)
            documents, decisions = self._document_rows(bucket, archive_relative)
            if bucket == "wikipedia":
                for decision in decisions:
                    if decision["decision"] == "keep":
                        decision.update(
                            decision="reject",
                            split=None,
                            selected_tokens=0,
                            token_prefix=None,
                            terminal_quota_prefix=False,
                            reasons=["quality:fixture-wikipedia"],
                        )
            archive_path = self.root / archive_relative
            self._write_raw_archive(archive_path, documents)
            fingerprint_relative = f"fingerprints/{bucket}/part-000000.jsonl.zst"
            fingerprint_path = self.preprocess / fingerprint_relative
            fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
            fingerprint_path.write_bytes(f"fixture:{bucket}\n".encode())
            report_relative = f"reports/{bucket}/part-000000.json"
            report_path = self.preprocess / report_relative
            report = {
                "report_version": 1,
                "bucket": bucket,
                "index": 0,
                "archive": archive_relative,
                "archive_sha256": file_sha256(archive_path),
                "archive_compressed_bytes": archive_path.stat().st_size,
                "fingerprint_file": fingerprint_relative,
                "fingerprint_sha256": file_sha256(fingerprint_path),
                "documents": len(documents),
                "clean_bytes": sum(len(content) for _member, content, _row in documents),
                "exact_tokens": sum(row["starcoder2_tokens"] for _m, _c, row in documents),
            }
            write_json(report_path, report)
            report_payloads[bucket] = report
            decision_relative = f"decisions/{bucket}/part-000000.jsonl.zst"
            decision_path = self.selection / decision_relative
            write_jsonl_zst(decision_path, decisions)
            input_reports.append(
                {
                    "report": report_relative,
                    "report_sha256": file_sha256(report_path),
                    "archive": archive_relative,
                    "archive_sha256": report["archive_sha256"],
                    "fingerprint_file": fingerprint_relative,
                    "fingerprint_sha256": report["fingerprint_sha256"],
                    "documents": report["documents"],
                    "content_tokens": report["exact_tokens"],
                }
            )
            decision_shards.append(
                {
                    "archive": archive_relative,
                    "path": decision_relative,
                    "sha256": file_sha256(decision_path),
                    "records": len(decisions),
                }
            )
        input_reports.sort(key=lambda row: row["report"])
        decision_shards.sort(key=lambda row: row["archive"])
        report_projection = [
            {"path": row["report"], "sha256": row["report_sha256"]}
            for row in input_reports
        ]
        quotas = [
            {
                "split": split,
                "category": domain,
                "unit": "pre_packing_starcoder2_content_tokens",
                "target_tokens": 10,
                "selected_tokens": 10,
                "documents": 2,
                "terminal_prefix_documents": 1,
            }
            for split in ("train", "validation", "test")
            for domain in ("python", "other_code", "english")
        ]
        tokenizer_manifest_path = self.tokenizer_root / "TOKENIZER_MANIFEST.json"
        marker_files = []
        for bucket in sorted(report_payloads):
            marker_path = self.root / "state" / f"{bucket}_complete.json"
            write_json(marker_path, {"complete": True, "bucket": bucket})
            marker_files.append(
                {
                    "path": str(marker_path.relative_to(self.root)),
                    "sha256": file_sha256(marker_path),
                    "buckets": [bucket],
                }
            )
        per_bucket = {
            bucket: {
                "archives": 1,
                "documents": report["documents"],
                "clean_bytes": report["clean_bytes"],
                "exact_tokens": report["exact_tokens"],
                "target_exact_tokens": 1,
            }
            for bucket, report in sorted(report_payloads.items())
        }
        inventory_stub = [
            {"bucket": bucket, "archive": report["archive"]}
            for bucket, report in sorted(report_payloads.items())
        ]
        completeness = {
            "format_version": 1,
            "complete": True,
            "legacy_dedup_index_required": False,
            "pending_inputs": 0,
            "preprocess_error_records": 0,
            "collection_targets_exact_tokens": {
                bucket: 1 for bucket in sorted(report_payloads)
            },
            "quota_records": {
                "count": len(report_payloads),
                "inventory_sha256": canonical_sha256(inventory_stub),
            },
            "completion_markers": {
                "count": len(marker_files),
                "inventory_sha256": canonical_sha256(marker_files),
                "files": marker_files,
            },
            "raw_archives": {
                "count": len(report_payloads),
                "inventory_sha256": canonical_sha256(inventory_stub),
            },
            "reports": {
                "count": len(report_payloads),
                "inventory_sha256": canonical_sha256(report_projection),
            },
            "fingerprints": {
                "count": len(report_payloads),
                "inventory_sha256": canonical_sha256(inventory_stub),
            },
            "per_bucket": per_bucket,
        }
        near_artifact = self._english_near_artifact(input_reports=input_reports)
        near_mapping_sha256 = near_artifact["mapping"]["sha256"]
        identity = {
            "format_version": 5,
            "policy_sha256": canonical_sha256(self.policy),
            "quota_config_sha256": file_sha256(self.quota_path),
            "benchmark_guard_sha256": file_sha256(self.denylist_path),
            "preprocess_manifest_sha256": file_sha256(
                self.preprocess / "PREPROCESS_MANIFEST.json"
            ),
            "report_inventory_sha256": canonical_sha256(report_projection),
            "report_count": len(input_reports),
            "english_near_clusters_sha256": near_mapping_sha256,
            "english_near_artifact": near_artifact,
            "sqlite_runtime": {
                "sqlite_version": "3.45.1",
                "journal_policy": {
                    "policy_version": 1,
                    "requested_mode": "delete",
                    "selected_mode": "delete",
                    "mount": {
                        "detector": "fixture",
                        "mount_point": "/fixture",
                        "filesystem_type": "xfs",
                        "source": "/dev/fixture",
                        "device": "1:1",
                        "options": ["rw"],
                        "classification": "local",
                    },
                },
            },
            "curation_storage_contract": {
                "contract_version": 2,
                "progress_version": 1,
                "maximum_transaction_rows": 10_000,
                "transaction_sidecar_limit_bytes": 655_360_000,
                "projected_additional_bytes_per_document": 3_072,
                "disk_safety_numerator": 2,
                "disk_safety_denominator": 1,
                "minimum_free_bytes_after_projection": 2_000_000_000,
                "sqlite_temp_store": "FILE",
                "sqlite_temp_relative_path": ".work/sqlite-tmp",
                "sqlite_temp_same_device_as_database": True,
            },
            "tokenizer_manifest_sha256": file_sha256(tokenizer_manifest_path),
            "tokenizer_revision": self.REVISION,
            "tokenizer_files_validated": ["tokenizer.json"],
            "source_manifests": self.source_descriptors,
            "collection_completeness": completeness,
            "legacy_stack_tokenizer_binding": (
                "collector_configuration_not_source_manifest_field"
            ),
        }
        manifest = {
            "manifest_version": 1,
            "decision_record_version": 1,
            "identity": identity,
            "collection_completeness": completeness,
            "selection_policy": self.policy["selection"],
            "production_ready": True,
            "raw_archives_opened": False,
            "raw_archives_hashed_for_integrity": True,
            "raw_archive_payloads_parsed_by_curation": False,
            "quota_unit": "pre_packing_starcoder2_content_tokens",
            "english_near_dedup_complete": True,
            "leakage_audit": {
                "content_hashes_in_multiple_splits": 0,
                "canonical_clusters_in_multiple_splits": 0,
                "source_groups_in_multiple_splits": 0,
                "cross_bucket_code_repo_groups_in_multiple_splits": 0,
            },
            "quotas": quotas,
            "input_reports": input_reports,
            "decision_shards": decision_shards,
            "decision_inventory_sha256": canonical_sha256(decision_shards),
        }
        write_json(self.selection / "manifest.json", manifest)
        (self.selection / "manifest.sha256").write_text(
            f"{file_sha256(self.selection / 'manifest.json')}  manifest.json\n",
            encoding="ascii",
        )


def directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class MaterializeTrainingCorpusTest(unittest.TestCase):
    @staticmethod
    def _convert_fixture_to_fast_profile(
        fixture: MaterializationFixture,
    ) -> dict[str, Any]:
        policy = {
            "policy_version": 2,
            "curation_profile": FAST_CURATION_PROFILE,
            "selection": {
                **fixture.policy["selection"],
                "english_near_dedup": {
                    "required_for_production": False,
                    "mapping_format": None,
                    "missing_mapping_action": "disabled_by_fast_profile",
                },
            },
            "buckets": {
                bucket: {
                    "local_near_dedup": False,
                    "near_duplicate_authority": "disabled_by_fast_profile",
                    "split_grouping": (
                        "stable_english_source_after_canonicalization"
                    ),
                }
                for bucket in ("fineweb_edu", "wikipedia")
            },
        }
        fixture.policy = policy
        write_json(fixture.policy_path, policy)
        manifest_path = fixture.selection / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["identity"].update(
            format_version=6,
            policy_sha256=canonical_sha256(policy),
            english_near_clusters_sha256=None,
            english_near_artifact=None,
            curation_profile=FAST_CURATION_PROFILE,
        )
        manifest.update(
            selection_policy=policy["selection"],
            english_near_dedup_complete=False,
            english_near_dedup_status="disabled_by_fast_profile",
            curation_profile=FAST_CURATION_PROFILE,
            known_provenance_limitations=list(
                FAST_CURATION_PROFILE["known_limitations"]
            ),
        )
        manifest["leakage_audit"].update(
            normalized_hashes_in_multiple_splits=0,
            content_hashes_with_multiple_selected_documents=0,
            normalized_hashes_with_multiple_selected_documents=0,
        )
        manifest["fast_profile_audit"] = {
            "audit_version": 1,
            "fuzzy_near_dedup_performed": False,
            "near_map_rows": 0,
            "near_mapping_subphases": 0,
            "english_near_duplicate_reasons": 0,
            "content_hashes_in_multiple_splits": 0,
            "normalized_hashes_in_multiple_splits": 0,
            "content_hashes_with_multiple_selected_documents": 0,
            "normalized_hashes_with_multiple_selected_documents": 0,
            "source_groups_in_multiple_splits": 0,
        }
        write_json(manifest_path, manifest)
        (fixture.selection / "manifest.sha256").write_text(
            f"{file_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        return manifest

    def test_end_to_end_preserves_prefix_boundaries_splits_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            output = Path(temporary) / "materialized"
            result = fixture.materializer(output).run()
            self.assertTrue(result["complete"])
            self.assertFalse((output / ".materialization-journal.json").exists())
            self.assertEqual(
                set(result["manifest"]["splits"]),
                {"train", "validation", "test"},
            )
            self.assertEqual(
                set(result["manifest"]["provenance"]),
                {"source", "policy", "tokenizer", "fingerprints"},
            )
            source_provenance = json.loads(
                (output / "provenance/source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                source_provenance["curation_storage_contract"],
                json.loads(
                    (fixture.selection / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["identity"]["curation_storage_contract"],
            )

            for split in ("train", "validation", "test"):
                order = validate_training_order(output / "orders" / split / "manifest.json")
                self.assertEqual(order["format_version"], 4)
                self.assertEqual(order["split"], split)
                self.assertEqual(order["rows"], 5)
                self.assertEqual(order["rows_per_domain"], {
                    "python": 2,
                    "other_code": 2,
                    "english": 1,
                })
                self.assertEqual(order["packed_available_rows"], 6)
                self.assertEqual(order["packed_surplus_rows"], 1)
                for domain in ("python", "other_code", "english"):
                    manifest = validate_packed_manifest(
                        output / "packed" / split / domain / "manifest.json"
                    )
                    self.assertEqual(manifest["documents"], 2)
                    self.assertEqual(manifest["source_content_tokens"], 10)
                    self.assertEqual(manifest["stream_tokens"], 12)
                    self.assertEqual(manifest["rows"], 2)

                    index_manifest_path = (
                        output
                        / "provenance"
                        / "documents"
                        / split
                        / domain
                        / "manifest.json"
                    )
                    index_manifest = json.loads(index_manifest_path.read_text())
                    self.assertEqual(index_manifest["documents"], 2)
                    self.assertEqual(index_manifest["selected_content_tokens"], 10)
                    self.assertEqual(index_manifest["logical_stream_tokens"], 12)

            python_manifest = json.loads(
                (output / "packed" / "train" / "python" / "manifest.json").read_text()
            )
            token_path = output / "packed" / "train" / "python" / python_manifest["shards"][0]["tokens"]["path"]
            start_path = output / "packed" / "train" / "python" / python_manifest["shards"][0]["starts"]["path"]
            tokens = np.fromfile(token_path, dtype="<u2")
            starts = np.unpackbits(
                np.fromfile(start_path, dtype=np.uint8), bitorder="little"
            )
            # The selected prefix is three `a` tokens followed by one EOS;
            # the next document starts in the lookahead column and is masked.
            self.assertEqual(tokens.tolist(), [2, 2, 2, 0, 2])
            self.assertEqual(starts[:5].tolist(), [1, 0, 0, 0, 1])

            index_manifest = json.loads(
                (
                    output
                    / "provenance/documents/train/python/manifest.json"
                ).read_text()
            )
            rows = [
                row
                for shard in index_manifest["shards"]
                for row in iter_jsonl_zst(output / shard["path"])
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row["logical_stream_start"] for row in rows], [0, 4]
            )
            self.assertEqual(
                [row["logical_eos_position"] for row in rows], [3, 11]
            )
            self.assertTrue(all(row["language"] == "Python" for row in rows))

            # Re-opening a complete output performs the full integrity pass.
            reopened = fixture.materializer(output).run()
            self.assertTrue(reopened["complete"])
            self.assertEqual(reopened["manifest"], result["manifest"])

    def test_fast_v6_profile_materializes_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            source_manifest = self._convert_fixture_to_fast_profile(fixture)
            kept = [
                row
                for descriptor in source_manifest["decision_shards"]
                for row in iter_jsonl_zst(fixture.selection / descriptor["path"])
                if row["decision"] == "keep"
            ]
            self.assertEqual(
                len(kept), len({row["content_sha256"] for row in kept})
            )
            self.assertEqual(
                len(kept), len({row["normalized_sha256"] for row in kept})
            )
            output = Path(temporary) / "materialized"
            result = fixture.materializer(output).run()
            self.assertTrue(result["complete"])
            source = json.loads(
                (output / "provenance/source.json").read_text(encoding="utf-8")
            )
            fingerprints = json.loads(
                (output / "provenance/fingerprints.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(source["curation_profile"], FAST_CURATION_PROFILE)
            self.assertEqual(
                source["known_provenance_limitations"],
                source_manifest["known_provenance_limitations"],
            )
            self.assertIsNone(fingerprints["english_near_artifact"])
            self.assertFalse(fingerprints["english_near_dedup_complete"])

        cases = (
            (
                "near-artifact",
                lambda manifest: manifest["identity"].__setitem__(
                    "english_near_clusters_sha256", hashlib.sha256(b"near").hexdigest()
                ),
                "must not carry English near artifacts",
            ),
            (
                "profile-drift",
                lambda manifest: manifest["identity"]["curation_profile"].__setitem__(
                    "canonicalization", "exact-only"
                ),
                "Unsupported fast curation profile",
            ),
            (
                "audit-drift",
                lambda manifest: manifest["fast_profile_audit"].__setitem__(
                    "normalized_hashes_with_multiple_selected_documents", 1
                ),
                "Fast curation audit is unsafe",
            ),
            (
                "missing-limitation",
                lambda manifest: manifest.__setitem__(
                    "known_provenance_limitations", []
                ),
                "limitations were not preserved",
            ),
        )
        for label, mutate, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = MaterializationFixture(Path(temporary) / "source")
                self._convert_fixture_to_fast_profile(fixture)
                manifest_path = fixture.selection / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                write_json(manifest_path, manifest)
                (fixture.selection / "manifest.sha256").write_text(
                    f"{file_sha256(manifest_path)}  manifest.json\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(MaterializationError, error):
                    fixture.materializer(Path(temporary) / "output")

    def test_packing_can_finish_before_gpu_geometry_and_finalize_later(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            output = Path(temporary) / "materialized"
            deferred = replace(
                fixture.config,
                frozen_global_microbatch_rows=None,
                frozen_gradient_accumulation_steps=None,
            )
            with self.assertRaisesRegex(
                MaterializationError, "GPU smoke"
            ):
                fixture.materializer(
                    Path(temporary) / "missing-geometry",
                    config=deferred,
                ).run()
            staged = fixture.materializer(output, config=deferred).run(
                stop_after_packing=True
            )
            self.assertFalse(staged["complete"])
            self.assertEqual(staged["phase"], "packed")
            self.assertFalse((output / "orders").exists())
            self.assertFalse((output / "manifest.json").exists())
            before = directory_bytes(output / "packed") | {
                f"documents/{key}": value
                for key, value in directory_bytes(
                    output / "provenance" / "documents"
                ).items()
            }

            completed = fixture.materializer(output).run()
            self.assertTrue(completed["complete"])
            after = directory_bytes(output / "packed") | {
                f"documents/{key}": value
                for key, value in directory_bytes(
                    output / "provenance" / "documents"
                ).items()
            }
            self.assertEqual(after, before)

            incompatible = replace(
                fixture.config,
                frozen_global_microbatch_rows=2,
                frozen_gradient_accumulation_steps=1,
            )
            with self.assertRaisesRegex(
                MaterializationError, "identity mismatch"
            ):
                fixture.materializer(output, config=incompatible).run()

    def test_interruption_between_writer_checkpoints_resumes_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            fresh = Path(temporary) / "fresh"
            resumed = Path(temporary) / "resumed"
            fixture.materializer(fresh).run()

            fired = False

            def interrupt(event: str, _payload: Any) -> None:
                nonlocal fired
                if event == "writer_checkpoint" and not fired:
                    fired = True
                    raise RuntimeError("injected interruption")

            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                fixture.materializer(
                    resumed,
                    tokenizer_batch_documents=1,
                    fault_injector=interrupt,
                ).run()
            journal = json.loads(
                (resumed / ".materialization-journal.json").read_text()
            )
            # The bridge journal may lag, while one PackedShardWriter journal
            # has already committed the archive. Resume reconciles that vector.
            self.assertEqual(journal["state"]["completed_archives"], 0)
            completed = fixture.materializer(resumed).run()
            self.assertTrue(completed["complete"])
            self.assertEqual(directory_bytes(resumed), directory_bytes(fresh))

    def test_tokenizer_byte_batches_are_hard_bounded_and_output_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            default_output = Path(temporary) / "default"
            bounded_output = Path(temporary) / "bounded"
            fixture.materializer(default_output).run()
            bounded = fixture.materializer(
                bounded_output,
                tokenizer_batch_bytes=16,
                tokenizer_max_document_bytes=16,
            ).run()
            self.assertTrue(bounded["complete"])
            self.assertEqual(directory_bytes(bounded_output), directory_bytes(default_output))

        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            with self.assertRaisesRegex(
                MaterializationError, "exceeds tokenizer max-document byte limit"
            ):
                fixture.materializer(
                    Path(temporary) / "oversized",
                    tokenizer_batch_bytes=16,
                    tokenizer_max_document_bytes=4,
                ).run()

    def test_controlled_stop_resumes_without_replaying_committed_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            output = Path(temporary) / "materialized"
            partial = fixture.materializer(output).run(max_archives=1)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["completed_archives"], 1)
            self.assertEqual(partial["runtime"]["archives"], 1)
            self.assertGreater(
                partial["runtime"]["source_content_tokens_per_second"], 0
            )
            final = fixture.materializer(output).run()
            self.assertTrue(final["complete"])

    def test_malformed_terminal_prefix_fails_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(
                Path(temporary) / "source", malformed_prefix=True
            )
            with self.assertRaisesRegex(MaterializationError, "Kept decision"):
                fixture.materializer(Path(temporary) / "output").run()

    def test_changed_raw_archive_and_nonproduction_selection_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            archive = fixture.root / "raw/python/part-000000.tar.zst"
            with archive.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(MaterializationError, "raw archive"):
                fixture.materializer(Path(temporary) / "tampered-output").run()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            manifest_path = fixture.selection / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["production_ready"] = False
            write_json(manifest_path, manifest)
            (fixture.selection / "manifest.sha256").write_text(
                f"{file_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(MaterializationError, "production_ready"):
                fixture.materializer(Path(temporary) / "unsafe-output")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            manifest_path = fixture.selection / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["collection_completeness"]["complete"] = False
            manifest["identity"]["collection_completeness"]["complete"] = False
            write_json(manifest_path, manifest)
            (fixture.selection / "manifest.sha256").write_text(
                f"{file_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                MaterializationError, "Collection completeness complete"
            ):
                fixture.materializer(Path(temporary) / "incomplete-output")

    def test_curation_v5_integrity_runtime_and_near_identity_fail_closed(self) -> None:
        def set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
            cursor = payload
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value

        cases = (
            (
                "identity-version",
                ("identity", "format_version"),
                3,
                "Unsupported curation identity",
            ),
            (
                "raw-hash-evidence",
                ("raw_archives_hashed_for_integrity",),
                False,
                "raw_archives_hashed_for_integrity mismatch",
            ),
            (
                "raw-payload-parsing",
                ("raw_archive_payloads_parsed_by_curation",),
                True,
                "raw_archive_payloads_parsed_by_curation mismatch",
            ),
            (
                "near-mapping-cross-identity",
                ("identity", "english_near_artifact", "mapping", "sha256"),
                hashlib.sha256(b"different-near-mapping").hexdigest(),
                "english_near_clusters_sha256 differs",
            ),
            (
                "near-policy-cross-identity",
                (
                    "identity",
                    "english_near_artifact",
                    "identity",
                    "curation_policy_sha256",
                ),
                hashlib.sha256(b"different-policy").hexdigest(),
                "curation_policy_sha256 differs",
            ),
            (
                "calibration-gate",
                (
                    "identity",
                    "english_near_artifact",
                    "identity",
                    "calibration_evidence",
                    "status",
                ),
                "fail",
                "Calibration evidence status is not production-passing",
            ),
            (
                "calibration-identity-checksum",
                (
                    "identity",
                    "english_near_artifact",
                    "identity",
                    "calibration_evidence",
                    "identity_sha256",
                ),
                hashlib.sha256(b"different-calibration-identity").hexdigest(),
                "Calibration identity checksum mismatch",
            ),
            (
                "sqlite-runtime-policy",
                ("identity", "sqlite_runtime", "journal_policy", "selected_mode"),
                "wal",
                "SQLite journal policy is internally inconsistent",
            ),
            (
                "storage-contract-version",
                ("identity", "curation_storage_contract", "contract_version"),
                1,
                "storage contract mismatch for contract_version",
            ),
            (
                "storage-contract-sidecar-formula",
                (
                    "identity",
                    "curation_storage_contract",
                    "transaction_sidecar_limit_bytes",
                ),
                268_435_456,
                "storage contract mismatch for transaction_sidecar_limit_bytes",
            ),
        )
        for label, path, value, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = MaterializationFixture(Path(temporary) / "source")
                manifest_path = fixture.selection / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                set_nested(manifest, path, value)
                write_json(manifest_path, manifest)
                (fixture.selection / "manifest.sha256").write_text(
                    f"{file_sha256(manifest_path)}  manifest.json\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(MaterializationError, error):
                    fixture.materializer(Path(temporary) / "output")

    def test_materializer_reopens_five_file_english_authority(self) -> None:
        def tamper_mapping(fixture: MaterializationFixture) -> None:
            path = fixture.root / "staging/english-near/clusters.jsonl.zst"
            path.write_bytes(path.read_bytes() + b"tamper")

        cases = (
            (
                "missing-manifest-sidecar",
                lambda fixture: (
                    fixture.root / "staging/english-near/manifest.sha256"
                ).unlink(),
                "Missing English near-dedup manifest sidecar",
            ),
            (
                "missing-preflight-result",
                lambda fixture: (
                    fixture.root
                    / "staging/english-near/operational-preflight-v1/result.json"
                ).unlink(),
                "Missing operational preflight result",
            ),
            (
                "tampered-preflight-sidecar",
                lambda fixture: (
                    fixture.root
                    / "staging/english-near/operational-preflight-v1/result.json.sha256"
                ).write_text("0" * 64 + "  result.json\n", encoding="ascii"),
                "Operational preflight sidecar mismatch",
            ),
            (
                "tampered-mapping",
                tamper_mapping,
                "mapping identity mismatch",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = MaterializationFixture(Path(temporary) / "source")
                mutate(fixture)
                with self.assertRaisesRegex(MaterializationError, message):
                    fixture.materializer(Path(temporary) / "output")

    def test_materializer_rejects_resigned_stale_preflight_formulas(self) -> None:
        def set_nested(
            payload: dict[str, Any], path: tuple[str, ...], value: Any
        ) -> None:
            cursor = payload
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value

        def resign(fixture: MaterializationFixture, mutate: Any) -> None:
            near_root = fixture.root / "staging/english-near"
            result_path = near_root / "operational-preflight-v1/result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            mutate(result)
            write_json(result_path, result)
            result_raw = result_path.read_bytes()
            result_sha = hashlib.sha256(result_raw).hexdigest()
            preflight_sidecar = result_path.with_name("result.json.sha256")
            preflight_sidecar.write_text(
                f"{result_sha}  result.json\n", encoding="ascii"
            )

            near_manifest_path = near_root / "manifest.json"
            near_manifest = json.loads(
                near_manifest_path.read_text(encoding="utf-8")
            )
            evidence = near_manifest["refinement_operational_preflight"]
            for field in ("identity", "thresholds", "sample", "measurements"):
                evidence[field] = result[field]
            evidence.update(
                result_sha256=result_sha,
                result_bytes=len(result_raw),
                sidecar_sha256=file_sha256(preflight_sidecar),
                identity_sha256=canonical_sha256(result["identity"]),
            )
            write_json(near_manifest_path, near_manifest)
            near_manifest_raw = near_manifest_path.read_bytes()
            near_manifest_sha = hashlib.sha256(near_manifest_raw).hexdigest()
            near_sidecar = near_root / "manifest.sha256"
            near_sidecar.write_text(
                f"{near_manifest_sha}  manifest.json\n", encoding="ascii"
            )

            selection_path = fixture.selection / "manifest.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            artifact = selection["identity"]["english_near_artifact"]
            artifact["refinement_operational_preflight"] = evidence
            artifact["manifest"].update(
                sha256=near_manifest_sha,
                bytes=len(near_manifest_raw),
                sidecar_sha256=file_sha256(near_sidecar),
            )
            write_json(selection_path, selection)
            (fixture.selection / "manifest.sha256").write_text(
                f"{file_sha256(selection_path)}  manifest.json\n", encoding="ascii"
            )

        cases = (
            (
                "threshold",
                lambda result: result["thresholds"].__setitem__(
                    "requested_pairs",
                    result["thresholds"]["requested_pairs"] + 1,
                ),
                "thresholds changed",
            ),
            (
                "rate-formula",
                lambda result: result["measurements"].__setitem__(
                    "projected_refinement_seconds",
                    result["measurements"]["projected_refinement_seconds"] + 1,
                ),
                "formula differs for projected_refinement_seconds",
            ),
            (
                "union-formula",
                lambda result: result["measurements"].__setitem__(
                    "union_parent_array_projected_bytes",
                    result["measurements"]["union_parent_array_projected_bytes"]
                    + 8,
                ),
                "union formula differs for union_parent_array_projected_bytes",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = MaterializationFixture(Path(temporary) / "source")
                resign(fixture, mutate)
                with self.assertRaisesRegex(MaterializationError, message):
                    fixture.materializer(Path(temporary) / "output")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            calibration = (
                fixture.root / "audits/english-near-calibration-v1.json"
            )
            with calibration.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(
                MaterializationError, "Calibration result identity mismatch"
            ):
                fixture.materializer(Path(temporary) / "output")

        cross_authority_cases = (
            (
                "production-builder",
                ("production_builder_sha256",),
                hashlib.sha256(b"different-production-builder").hexdigest(),
                "production builder/config differs",
            ),
            (
                "calibration-input-policy",
                ("input", "curation_policy_sha256"),
                hashlib.sha256(b"different-calibration-policy").hexdigest(),
                "input authority differs for curation_policy_sha256",
            ),
        )
        for label, path, value, error in cross_authority_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = MaterializationFixture(Path(temporary) / "source")
                calibration_path = (
                    fixture.root / "audits/english-near-calibration-v1.json"
                )
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                set_nested(calibration["identity"], path, value)
                write_json(calibration_path, calibration)
                calibration_sha = file_sha256(calibration_path)
                sidecar_path = calibration_path.with_name(
                    calibration_path.name + ".sha256"
                )
                sidecar_path.write_text(
                    f"{calibration_sha}  {calibration_path.name}\n", encoding="utf-8"
                )

                selection_path = fixture.selection / "manifest.json"
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                evidence = selection["identity"]["english_near_artifact"][
                    "identity"
                ]["calibration_evidence"]
                evidence.update(
                    result_sha256=calibration_sha,
                    result_bytes=calibration_path.stat().st_size,
                    sidecar_sha256=file_sha256(sidecar_path),
                    identity=calibration["identity"],
                    identity_sha256=canonical_sha256(calibration["identity"]),
                )
                write_json(selection_path, selection)
                (fixture.selection / "manifest.sha256").write_text(
                    f"{file_sha256(selection_path)}  manifest.json\n", encoding="ascii"
                )
                with self.assertRaisesRegex(MaterializationError, error):
                    fixture.materializer(Path(temporary) / "output")

    def test_document_position_index_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MaterializationFixture(Path(temporary) / "source")
            output = Path(temporary) / "materialized"
            fixture.materializer(output).run()
            index_manifest = json.loads(
                (
                    output
                    / "provenance/documents/train/python/manifest.json"
                ).read_text()
            )
            shard = output / index_manifest["shards"][0]["path"]
            with shard.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(
                MaterializationError, "Document-index shard checksum mismatch"
            ):
                fixture.materializer(output).run()


if __name__ == "__main__":
    unittest.main()
