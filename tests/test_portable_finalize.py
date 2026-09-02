from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import zstandard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from pretrain.data import DOMAIN_ORDER, PackedShardWriter, validate_training_order
from pretrain.materialize import (
    CURSOR_FORMAT,
    CURSOR_VERSION,
    DOCUMENT_INDEX_FORMAT,
    DOCUMENT_INDEX_VERSION,
    FORMAT,
    FORMAT_VERSION,
    JOURNAL_NAME,
    canonical_sha256,
    file_sha256,
)
from pretrain.portable_finalize import (
    PORTABLE_FINALIZATION_FORMAT,
    PortableFinalizationConfig,
    PortableFinalizationError,
    PortablePackedFinalizer,
    _aligned_train_rows,
)
from pretrain.raw_token_cache import CACHE_FORMAT, CACHE_FORMAT_VERSION, CACHE_PROFILE
from pretrain.raw_token_cache_inventory import NON_AUTHORITIES
from pretrain.selection_contract import (
    ALL_ELIGIBLE_BITMAP_FORMAT,
    ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
    ALL_ELIGIBLE_SELECTION_STRATEGY,
)
from scripts.qualify_training_corpus import QualificationConfig, qualify_corpus


SPLITS = ("train", "validation", "test")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_inventory_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_zst_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(zstandard.ZstdCompressor(level=1, write_checksum=True).compress(raw))


class PortableFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpus = root / "packed-v1"
        self.selection = root / "selection-v7"
        self.tokenizer = root / "tokenizer" / "starcoder2"
        self.cache_inventory = root / "token-cache-inventory"
        self.policy_path = root / "policy.json"
        self.quota_path = root / "quotas.json"
        self.denylist_path = root / "denylist.json"
        self.model_config = root / "model-config.json"
        self.scratch = root / "scratch"
        self.receipts = root / "receipts"
        for path in (
            self.corpus,
            self.selection,
            self.tokenizer,
            self.cache_inventory,
            self.scratch,
            self.receipts,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.sequence_length = 4
        self.vocab_size = 16
        self.rows = {"python": 12, "other_code": 12, "english": 6}
        self._write_supporting_files()
        self.tokenizer_sha = self._write_tokenizer()
        self.selection_payload, self.selection_sha, authorities = self._write_selection()
        self.cache_payload, cache_descriptor = self._write_cache_inventory(authorities)
        self.archive_inventory_sha = canonical_sha256(
            [item["materialization_identity"] for item in authorities]
        )
        self.identity = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "selection_manifest_sha256": self.selection_sha,
            "collection_completeness_sha256": canonical_sha256(
                self.selection_payload["collection_completeness"]
            ),
            "decision_inventory_sha256": self.selection_payload[
                "decision_inventory_sha256"
            ],
            "archive_inventory_sha256": self.archive_inventory_sha,
            "tokenizer_manifest_sha256": self.tokenizer_sha,
            "curation_policy_sha256": canonical_sha256(
                json.loads(self.policy_path.read_text())
            ),
            "quota_config_sha256": file_sha256(self.quota_path),
            "benchmark_guard_sha256": file_sha256(self.denylist_path),
            "preprocess_manifest_sha256": self.selection_payload["identity"][
                "preprocess_manifest_sha256"
            ],
            "packing_configuration": {
                "sequence_length": self.sequence_length,
                "rows_per_shard": 5,
                "construction_seed": 1234,
                "expected_vocab_size": self.vocab_size,
                "expected_eos_token_id": 0,
            },
            "raw_token_cache": cache_descriptor,
        }
        self._write_packed_and_indexes(authorities)

    def config(self) -> PortableFinalizationConfig:
        return PortableFinalizationConfig(
            corpus_root=self.corpus,
            selection_root=self.selection,
            tokenizer_root=self.tokenizer,
            cache_inventory_root=self.cache_inventory,
            policy_path=self.policy_path,
            quota_path=self.quota_path,
            benchmark_denylist_path=self.denylist_path,
            order_seed=100,
            maximum_train_input_tokens=200,
            maximum_validation_input_tokens=200,
            maximum_test_input_tokens=200,
            expected_optimizer_batch_rows=30,
            world_size=6,
            verify_packed_payloads=True,
        )

    def _write_supporting_files(self) -> None:
        write_json(
            self.policy_path,
            {
                "policy_version": 2,
                "selection": {"selection_version": 1},
                "curation_profile": {"name": "fixture"},
            },
        )
        write_json(self.quota_path, {"version": 1, "fixture": True})
        write_json(self.denylist_path, {"manifest_version": 1, "benchmark": "MBPP"})
        write_json(
            self.model_config,
            {
                "vocab_size": self.vocab_size,
                "dim": 8,
                "hidden_dim": 16,
                "n_layers": 1,
                "n_heads": 2,
                "n_kv_heads": 1,
                "max_seq_len": self.sequence_length,
                "norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "initializer_range": 0.02,
                "tie_word_embeddings": False,
                "attention_backend": "auto",
                "loss_chunk_size": 4,
                "activation_checkpointing": False,
            },
        )

    def _write_tokenizer(self) -> str:
        vocab = {"<eos>": 0, "<unk>": 1}
        vocab.update({f"token-{index}": index for index in range(2, self.vocab_size)})
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer_path = self.tokenizer / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        manifest = {
            "manifest_version": 1,
            "repo_id": "fixture/tokenizer",
            "resolved_revision": "1" * 40,
            "files": {
                "tokenizer.json": {
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": file_sha256(tokenizer_path),
                }
            },
            "validation": {
                "vocab_size": self.vocab_size,
                "eos_token": "<eos>",
                "eos_token_id": 0,
            },
        }
        path = self.tokenizer / "TOKENIZER_MANIFEST.json"
        write_json(path, manifest)
        return file_sha256(path)

    def _write_selection(
        self,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        reports: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        authorities: list[dict[str, Any]] = []
        bucket_by_domain = {
            "python": "python",
            "other_code": "other_code",
            "english": "fineweb_edu",
        }
        ordinal = 0
        for split in SPLITS:
            for domain in DOMAIN_ORDER:
                # Selection-v7 decisions are in lexicographic raw-archive order.
                # Keep the fixture path prefix ordinal-first even though the
                # independent bucket authority still records the real bucket.
                archive_path = f"raw/part-{ordinal:06d}.tar.zst"
                report_path = f"reports/{bucket_by_domain[domain]}/part-{ordinal:06d}.json"
                fingerprint_path = (
                    f"fingerprints/{bucket_by_domain[domain]}/part-{ordinal:06d}.jsonl.zst"
                )
                content_tokens = self.rows[domain] * self.sequence_length
                archive_sha = digest(f"archive:{ordinal}".encode())
                report_sha = digest(f"report:{ordinal}".encode())
                fingerprint_sha = digest(f"fingerprint:{ordinal}".encode())
                decision_relative = f"decisions/part-{ordinal:06d}.bitmap"
                decision_path = self.selection / decision_relative
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                decision_path.write_bytes(b"\x01")
                decision = {
                    "archive": archive_path,
                    "path": decision_relative,
                    "sha256": file_sha256(decision_path),
                    "records": 1,
                    "format": ALL_ELIGIBLE_BITMAP_FORMAT,
                    "format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
                    "bytes": 1,
                    "kept_documents": 1,
                }
                report = {
                    "archive": archive_path,
                    "archive_sha256": archive_sha,
                    "report": report_path,
                    "report_sha256": report_sha,
                    "fingerprint_file": fingerprint_path,
                    "fingerprint_sha256": fingerprint_sha,
                    "documents": 1,
                    "content_tokens": content_tokens,
                }
                reports.append(report)
                decisions.append(decision)
                materialization_identity = {
                    "ordinal": ordinal,
                    "archive": archive_path,
                    "archive_index": ordinal,
                    "bucket": bucket_by_domain[domain],
                    "domain": domain,
                    "raw_sha256": archive_sha,
                    "report": report_path,
                    "report_sha256": report_sha,
                    "fingerprint": fingerprint_path,
                    "fingerprint_sha256": fingerprint_sha,
                    "decision": decision_relative,
                    "decision_sha256": decision["sha256"],
                    "documents": 1,
                    "clean_bytes": content_tokens * 2,
                    "content_tokens": content_tokens,
                    "decision_format": ALL_ELIGIBLE_BITMAP_FORMAT,
                    "decision_format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
                    "decision_bytes": 1,
                    "decision_kept_documents": 1,
                }
                authorities.append(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "domain": domain,
                        "bucket": bucket_by_domain[domain],
                        "archive": archive_path,
                        "archive_sha": archive_sha,
                        "report": report_path,
                        "report_sha": report_sha,
                        "fingerprint": fingerprint_path,
                        "fingerprint_sha": fingerprint_sha,
                        "decision": decision,
                        "content_tokens": content_tokens,
                        "materialization_identity": materialization_identity,
                    }
                )
                ordinal += 1
        decision_inventory = canonical_sha256(decisions)
        completeness = {
            "format_version": 1,
            "complete": True,
            "pending_inputs": 0,
            "preprocess_error_records": 0,
        }
        leakage = {
            "content_hashes_in_multiple_splits": 0,
            "canonical_clusters_in_multiple_splits": 0,
            "source_groups_in_multiple_splits": 0,
            "cross_bucket_code_repo_groups_in_multiple_splits": 0,
            "normalized_hashes_in_multiple_splits": 0,
        }
        policy = json.loads(self.policy_path.read_text())
        payload = {
            "production_ready": True,
            "identity": {
                "format_version": 7,
                "tokenizer_manifest_sha256": self.tokenizer_sha,
                "tokenizer_revision": "1" * 40,
                "policy_sha256": canonical_sha256(policy),
                "quota_config_sha256": file_sha256(self.quota_path),
                "benchmark_guard_sha256": file_sha256(self.denylist_path),
                "preprocess_manifest_sha256": digest(b"preprocess"),
                "report_inventory_sha256": canonical_sha256(
                    [
                        {"path": row["report"], "sha256": row["report_sha256"]}
                        for row in reports
                    ]
                ),
                "english_near_clusters_sha256": None,
                "english_near_artifact": None,
                "source_manifests": {
                    "STACK_V3_SOURCE.json": {
                        "sha256": digest(b"stack-source"),
                        "resolved_revision": "2" * 40,
                    }
                },
                "sqlite_runtime": {
                    "sqlite_version": "fixture",
                    "journal_policy": {"selected_mode": "wal"},
                },
                "curation_storage_contract": {"contract_version": 3},
                "raw_archive_integrity_policy": (
                    "deferred-full-sha256-mandatory-before-publication"
                ),
                "sqlite_execution": {"mode": "local-wal"},
                "fast_all_eligible_handoff": {"contract_version": 1},
                "source_curation": {"format_version": 1},
                "curation_profile": {"name": "fixture"},
            },
            "selection_strategy": ALL_ELIGIBLE_SELECTION_STRATEGY,
            "decision_format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "decision_format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "decision_inventory_sha256": decision_inventory,
            "decision_shards": decisions,
            "input_reports": reports,
            "collection_completeness": completeness,
            "raw_archives_hashed_for_integrity": True,
            "raw_archive_payloads_parsed_by_curation": False,
            "known_provenance_limitations": ["fixture limitation"],
            "english_near_dedup_complete": False,
            "english_near_dedup_status": "disabled_by_fast_profile",
            "selection_policy": policy["selection"],
            "selection_profile": {
                "split_authority": "frozen_leakage_safe_source_groups"
            },
            "leakage_audit": leakage,
            "publication_scope": "production-durable-snapshot",
            "training_input_budget_authority": "packed order v4",
            "selected_totals": {"documents": 9},
            "reference_quotas": {"profile": "fixture"},
        }
        path = self.selection / "manifest.json"
        write_json(path, payload)
        sha = file_sha256(path)
        (self.selection / "manifest.sha256").write_text(
            f"{sha}  manifest.json\n", encoding="ascii"
        )
        return payload, sha, authorities

    def _write_cache_inventory(
        self, authorities: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        tokenizer = {
            "repo_id": "fixture/tokenizer",
            "resolved_revision": "1" * 40,
            "manifest_sha256": self.tokenizer_sha,
            "vocabulary_sha256": digest(b"vocabulary"),
            "vocab_size": self.vocab_size,
            "eos_token": "<eos>",
            "eos_token_id": 0,
            "eos_present_in_payload": False,
        }
        entries = []
        for authority in authorities:
            ordinal = authority["ordinal"]
            bucket = authority["bucket"]
            cache_dir = f"archives/{bucket}/part-{ordinal:06d}"
            entries.append(
                {
                    "ordinal": ordinal,
                    "cache_directory": cache_dir,
                    "cache_manifest": {
                        "path": f"{cache_dir}/manifest.json",
                        "bytes": 1,
                        "sha256": digest(f"cache:{ordinal}".encode()),
                    },
                    "cache_sidecar": {
                        "path": f"{cache_dir}/manifest.sha256",
                        "bytes": 1,
                        "sha256": digest(f"sidecar:{ordinal}".encode()),
                    },
                    "archive": {
                        "path": authority["archive"],
                        "bucket": bucket,
                        "index": ordinal,
                        "bytes": 100 + ordinal,
                        "sha256": authority["archive_sha"],
                    },
                    "preprocess_report": {
                        "path": authority["report"],
                        "bytes": 10,
                        "sha256": authority["report_sha"],
                    },
                    "fingerprint": {
                        "path": authority["fingerprint"],
                        "bytes": 11,
                        "sha256": authority["fingerprint_sha"],
                    },
                    "documents": {
                        "records": 1,
                        "clean_bytes": authority["content_tokens"] * 2,
                        "content_tokens": authority["content_tokens"],
                    },
                }
            )
        inventory_sha = digest(canonical_inventory_bytes(entries))
        payload = {
            "format": "raw-token-cache-inventory",
            "format_version": 1,
            "inventory_complete": True,
            "training_ready": False,
            "selection": {
                "identity_format_version": 7,
                "manifest_sha256": self.selection_sha,
            },
            "cache": {
                "format": CACHE_FORMAT,
                "format_version": CACHE_FORMAT_VERSION,
                "profile": CACHE_PROFILE,
            },
            "tokenizer": tokenizer,
            "archive_count": len(entries),
            "archive_inventory_sha256": inventory_sha,
            "archives": entries,
            "non_authorities": NON_AUTHORITIES,
            "builder": {
                "implementation": "pretrain.raw_token_cache_inventory",
                "implementation_sha256": digest(b"builder"),
            },
        }
        manifest_raw = canonical_inventory_bytes(payload)
        manifest_path = self.cache_inventory / "manifest.json"
        manifest_path.write_bytes(manifest_raw)
        manifest_sha = digest(manifest_raw)
        sidecar_raw = f"{manifest_sha}  manifest.json\n".encode("ascii")
        sidecar_path = self.cache_inventory / "manifest.sha256"
        sidecar_path.write_bytes(sidecar_raw)
        descriptor = {
            "format": "raw-token-cache-inventory",
            "format_version": 1,
            "manifest": {
                "path": "manifest.json",
                "bytes": len(manifest_raw),
                "sha256": manifest_sha,
            },
            "sidecar": {
                "path": "manifest.sha256",
                "bytes": len(sidecar_raw),
                "sha256": digest(sidecar_raw),
            },
            "selection_manifest_sha256": self.selection_sha,
            "archive_count": len(entries),
            "archive_inventory_sha256": inventory_sha,
        }
        return payload, descriptor

    def _write_packed_and_indexes(self, authorities: list[dict[str, Any]]) -> None:
        cursors: dict[str, Any] = {}
        for authority in authorities:
            split = authority["split"]
            domain = authority["domain"]
            ordinal = authority["ordinal"]
            content_tokens = authority["content_tokens"]
            index_relative = (
                Path("provenance")
                / "documents"
                / split
                / domain
                / f"archive-{ordinal:06d}.jsonl.zst"
            ).as_posix()
            index_path = self.corpus / index_relative
            unique = digest(f"{split}:{domain}".encode())
            write_zst_jsonl(
                index_path,
                [
                    {
                        "record_version": DOCUMENT_INDEX_VERSION,
                        "doc_id": digest(f"doc:{split}:{domain}".encode()),
                        "canonical_doc_id": digest(
                            f"canonical:{split}:{domain}".encode()
                        ),
                        "split_group_id": unique,
                        "split": split,
                        "domain": domain,
                        "bucket": authority["bucket"],
                        "language": "Python" if domain == "python" else domain,
                        "source_archive": authority["archive"],
                        "source_archive_ordinal": ordinal,
                        "source_manifest_index": 0,
                        "source_member": f"member-{ordinal}",
                        "source_tokens": content_tokens,
                        "selected_content_tokens": content_tokens,
                        "terminal_quota_prefix": False,
                        "logical_stream_start": 0,
                        "logical_content_end_exclusive": content_tokens,
                        "logical_eos_position": content_tokens,
                    }
                ],
            )
            index_descriptor = {
                "archive_ordinal": ordinal,
                "path": index_relative,
                "bytes": index_path.stat().st_size,
                "sha256": file_sha256(index_path),
                "records": 1,
            }
            cursor = {
                "format": CURSOR_FORMAT,
                "version": CURSOR_VERSION,
                "archive_inventory_sha256": self.archive_inventory_sha,
                "split": split,
                "domain": domain,
                "next_archive": len(authorities),
                "selected_documents": 1,
                "selected_content_tokens": content_tokens,
                "terminal_prefix_documents": 0,
                "document_index_shards": [index_descriptor],
            }
            cursors[f"{split}/{domain}"] = cursor
            packed_dir = self.corpus / "packed" / split / domain
            writer = PackedShardWriter(
                packed_dir,
                domain=domain,
                split=split,
                sequence_length=self.sequence_length,
                vocab_size=self.vocab_size,
                eos_token_id=0,
                tokenizer_manifest_sha256=self.tokenizer_sha,
                rows_per_shard=5,
                construction_seed=1234,
                curation_policy_sha256=self.identity["curation_policy_sha256"],
                selection_manifest_sha256=self.selection_sha,
            )
            writer.add_document(
                [2 + index % (self.vocab_size - 2) for index in range(content_tokens)]
            )
            packed = writer.finish(source_cursor=cursor)
            self.assert_equal(packed["rows"], self.rows[domain])
            index_manifest = {
                "format": DOCUMENT_INDEX_FORMAT,
                "format_version": DOCUMENT_INDEX_VERSION,
                "split": split,
                "domain": domain,
                "selection_manifest_sha256": self.selection_sha,
                "tokenizer_manifest_sha256": self.tokenizer_sha,
                "sequence_length": self.sequence_length,
                "documents": 1,
                "selected_content_tokens": content_tokens,
                "logical_stream_tokens": content_tokens + 1,
                "shards": [index_descriptor],
            }
            write_json(index_path.parent / "manifest.json", index_manifest)
        journal = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "identity": self.identity,
            "state": {
                "phase": "packed",
                "completed_archives": len(authorities),
                "archive_count": len(authorities),
                "writer_cursors": cursors,
            },
        }
        write_json(self.corpus / JOURNAL_NAME, journal)

    def write_restore_receipt(self) -> Path:
        audit = self.root / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        packed_files = sorted(
            (
                path.relative_to(self.corpus).as_posix(),
                path.stat().st_size,
            )
            for path in self.corpus.rglob("*")
            if path.is_file()
        )
        packed_tsv = audit / "packed-v1-files.tsv"
        packed_tsv.write_text(
            "".join(f"{relative}\t{size}\n" for relative, size in packed_files),
            encoding="utf-8",
        )
        manifest_paths = [
            self.corpus / "packed" / split / domain / "manifest.json"
            for split in SPLITS
            for domain in DOMAIN_ORDER
        ]
        manifest_paths.append(self.corpus / JOURNAL_NAME)
        manifest_inventory = audit / "packed-manifests.sha256"
        manifest_inventory.write_text(
            "".join(
                f"{file_sha256(path)}  {path.relative_to(self.corpus).as_posix()}\n"
                for path in sorted(
                    manifest_paths,
                    key=lambda item: item.relative_to(self.corpus).as_posix(),
                )
            ),
            encoding="utf-8",
        )
        packed_total_bytes = sum(size for _, size in packed_files)
        summary = {
            "source_root": "/fixture/final/packed-v1",
            "generated_at_utc": "2026-09-02T12:00:00Z",
            "file_count": len(packed_files),
            "total_bytes": packed_total_bytes,
            "selection_manifest_sha256": self.selection_sha,
        }
        summary_path = audit / "source-summary.json"
        write_json(summary_path, summary)
        payloads = [
            path
            for path in self.corpus.glob("packed/*/*/*.bin")
            if path.is_file()
        ]
        receipt = {
            "format": "transcendent-logic-pretraining-data-restore",
            "format_version": 1,
            "status": "ready",
            "source_uri": (
                "s3://transcendent-logic-data-618079239540/"
                "coding-llm/pretraining/2026-09-02-packed-v1/"
            ),
            "bucket_owner": "618079239540",
            "region": "eu-central-1",
            "restored_at_utc": "2026-09-02T13:00:00Z",
            "remote_object_count": len(packed_files) + 4,
            "remote_total_bytes": packed_total_bytes + 1_000,
            "remote_inventory_sha256": digest(b"remote inventory"),
            "selection_manifest_sha256": self.selection_sha,
            "summary_sha256": file_sha256(summary_path),
            "packed_tsv_sha256": file_sha256(packed_tsv),
            "packed_manifests_sha256": file_sha256(manifest_inventory),
            "packed_file_count": len(packed_files),
            "packed_total_bytes": packed_total_bytes,
            "payload_file_count": len(payloads),
            "payload_total_bytes": sum(path.stat().st_size for path in payloads),
            "payload_sha256_verified": True,
            "optional_global_sha256_verified": False,
        }
        receipt_path = self.root / ".RESTORE_READY.json"
        receipt_path.write_bytes(canonical_inventory_bytes(receipt))
        return receipt_path

    @staticmethod
    def assert_equal(found: Any, expected: Any) -> None:
        if found != expected:
            raise AssertionError(f"fixture mismatch: {found!r} != {expected!r}")


class PortableFinalizerTest(unittest.TestCase):
    def test_production_train_cap_is_exactly_update_aligned(self) -> None:
        rows = _aligned_train_rows(
            maximum_input_tokens=52_580_000_000,
            sequence_length=4_096,
            optimizer_update_rows=192,
        )
        self.assertEqual(rows, 12_836_736)
        self.assertEqual(rows // 192, 66_858)
        self.assertEqual(rows * 4_096, 52_579_270_656)
        self.assertEqual(52_580_000_000 - rows * 4_096, 729_344)

    def test_heldout_orders_are_exact_balanced_and_preserve_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            journal_path = fixture.corpus / JOURNAL_NAME
            journal_before = journal_path.read_bytes()

            heldout = PortablePackedFinalizer(fixture.config()).run(mode="heldout")

            self.assertFalse(heldout["complete"])
            self.assertEqual(heldout["phase"], "heldout-orders")
            self.assertEqual(
                heldout["orders"]["validation"]["input_tokens"], 120
            )
            self.assertEqual(heldout["orders"]["test"]["input_tokens"], 120)
            self.assertFalse((fixture.corpus / "orders/train").exists())
            self.assertFalse((fixture.corpus / "manifest.json").exists())
            self.assertEqual(journal_path.read_bytes(), journal_before)

            for split in ("validation", "test"):
                order = validate_training_order(
                    fixture.corpus / "orders" / split / "manifest.json"
                )
                self.assertEqual(order["rows_per_domain"], self.fixture_weights(30))
                self.assertEqual(order["input_token_budget"]["actual_total"], 120)
                self.assertIsNone(
                    order["training_consumption"]["frozen_global_microbatch_rows"]
                )
            self.assertEqual(journal_path.read_bytes(), journal_before)

    def test_authenticated_restore_receipt_can_replace_payload_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            receipt = fixture.write_restore_receipt()
            config = dataclasses.replace(
                fixture.config(),
                restore_ready_path=receipt,
                verify_packed_payloads=False,
            )

            result = PortablePackedFinalizer(config).run(mode="heldout")

            self.assertEqual(result["phase"], "heldout-orders")
            self.assertTrue((fixture.corpus / JOURNAL_NAME).exists())

    def test_restore_receipt_rejects_changed_packed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            receipt = fixture.write_restore_receipt()
            manifest = fixture.corpus / "packed/train/python/manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            config = dataclasses.replace(
                fixture.config(),
                restore_ready_path=receipt,
                verify_packed_payloads=False,
            )

            with self.assertRaisesRegex(
                PortableFinalizationError,
                "Restore-authenticated packed artifact changed",
            ):
                PortablePackedFinalizer(config).run(mode="heldout")
            self.assertFalse((fixture.corpus / "orders").exists())

    @staticmethod
    def fixture_weights(rows: int) -> dict[str, int]:
        return {"python": rows * 2 // 5, "other_code": rows * 2 // 5, "english": rows // 5}

    def test_wrong_geometry_is_rejected_before_train_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            with self.assertRaisesRegex(
                PortableFinalizationError, "frozen optimizer batch"
            ):
                PortablePackedFinalizer(fixture.config()).run(
                    mode="final",
                    global_microbatch_rows=6,
                    gradient_accumulation_steps=4,
                )
            self.assertFalse((fixture.corpus / "orders/train").exists())
            self.assertTrue((fixture.corpus / JOURNAL_NAME).exists())

    def test_final_publication_matches_corpus_qualification_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            config = dataclasses.replace(
                fixture.config(), maximum_train_input_tokens=120
            )
            result = PortablePackedFinalizer(config).run(
                mode="final",
                global_microbatch_rows=6,
                gradient_accumulation_steps=5,
            )
            self.assertTrue(result["complete"])
            self.assertFalse((fixture.corpus / JOURNAL_NAME).exists())
            top = result["manifest"]
            self.assertEqual(
                top["finalization"]["format"], PORTABLE_FINALIZATION_FORMAT
            )
            self.assertEqual(
                top["order_configuration"]["expected_train_input_tokens"], 120
            )
            train = validate_training_order(
                fixture.corpus / "orders/train/manifest.json"
            )
            self.assertEqual(train["input_token_budget"]["actual_total"], 120)
            self.assertEqual(train["training_consumption"]["optimizer_updates"], 1)
            self.assertEqual(train["training_consumption"]["dropped_tail_rows"], 0)

            receipt = qualify_corpus(
                QualificationConfig(
                    corpus_root=fixture.corpus,
                    tokenizer_root=fixture.tokenizer,
                    output=fixture.receipts / "qualified",
                    model_config=fixture.model_config,
                    expected_targets={
                        "train": 120,
                        "validation": 120,
                        "test": 120,
                    },
                    sample_rows_per_domain=2,
                    sample_seed=7,
                    world_size=6,
                    scratch_directory=fixture.scratch,
                    split_identity_batch_rows=4,
                )
            )
            self.assertEqual(receipt["status"], "pass")

    def test_tampered_document_index_fails_before_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            index_path = next(
                (fixture.corpus / "provenance/documents").rglob("*.jsonl.zst")
            )
            index_path.write_bytes(index_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                PortableFinalizationError, "Document-index checksum mismatch"
            ):
                PortablePackedFinalizer(fixture.config()).run(mode="heldout")
            self.assertFalse((fixture.corpus / "orders").exists())

    def test_tampered_selection_decision_fails_before_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            decision = next((fixture.selection / "decisions").iterdir())
            decision.write_bytes(b"\x00")
            with self.assertRaisesRegex(
                PortableFinalizationError, "Selection decision changed"
            ):
                PortablePackedFinalizer(fixture.config()).run(mode="heldout")
            self.assertFalse((fixture.corpus / "orders").exists())

    def test_heldout_finalization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            config = fixture.config()
            first = PortablePackedFinalizer(config).run(mode="heldout")
            validation_before = (
                fixture.corpus / "orders/validation/manifest.json"
            ).read_bytes()
            second = PortablePackedFinalizer(config).run(mode="heldout")
            self.assertFalse(first["complete"])
            self.assertFalse(second["complete"])
            self.assertEqual(
                (fixture.corpus / "orders/validation/manifest.json").read_bytes(),
                validation_before,
            )


if __name__ == "__main__":
    unittest.main()
