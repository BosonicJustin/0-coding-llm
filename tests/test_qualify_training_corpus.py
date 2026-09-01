from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from typing import Any

import zstandard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from pretrain.data import DOMAIN_ORDER, PackedShardWriter, build_training_order
from pretrain.materialize import (
    DOCUMENT_INDEX_FORMAT,
    DOCUMENT_INDEX_VERSION,
    FORMAT as CORPUS_FORMAT,
    FORMAT_VERSION as CORPUS_FORMAT_VERSION,
)
from pretrain.model import ModelConfig
from scripts.qualify_training_corpus import (
    QualificationConfig,
    QualificationError,
    main,
    publish_receipt,
    qualify_corpus,
)


SPLITS = ("train", "validation", "test")
SELECTION_SHA = hashlib.sha256(b"selection").hexdigest()
POLICY_SHA = hashlib.sha256(b"policy").hexdigest()


def race_publish_worker(
    output: str,
    value: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    ready.put(value)
    start.wait(timeout=10)
    try:
        publish_receipt(Path(output), {"status": value})
        results.put((value, "published"))
    except Exception as exc:
        results.put((value, type(exc).__name__))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl_zst(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(payload))


class QualificationFixture:
    def __init__(self, root: Path, *, cross_split_group: bool = False) -> None:
        self.root = root
        self.corpus = root / "corpus"
        self.tokenizer = root / "tokenizer"
        self.model_config = root / "model-config.json"
        self.receipts = root / "receipts"
        self.scratch = root / "scratch"
        for path in (self.corpus, self.tokenizer, self.receipts, self.scratch):
            path.mkdir(parents=True, exist_ok=True)
        self.vocab_size = 16
        self.sequence_length = 4
        self.eos_token_id = 0
        self.tokenizer_manifest_sha256 = self._build_tokenizer()
        self._build_model_config()
        self._build_corpus(cross_split_group=cross_split_group)

    def _build_tokenizer(self) -> str:
        vocabulary = {"<|endoftext|>": 0, "<unk>": 1}
        vocabulary.update({f"token-{index}": index for index in range(2, self.vocab_size)})
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.save(str(self.tokenizer / "tokenizer.json"))
        tokenizer_payload = self.tokenizer / "tokenizer.json"
        manifest = {
            "manifest_version": 1,
            "repo_id": "fixture/tokenizer",
            "resolved_revision": "0" * 40,
            "files": {
                "tokenizer.json": {
                    "bytes": tokenizer_payload.stat().st_size,
                    "sha256": sha256(tokenizer_payload),
                }
            },
            "validation": {
                "vocab_size": self.vocab_size,
                "eos_token": "<|endoftext|>",
                "eos_token_id": self.eos_token_id,
            },
        }
        manifest_path = self.tokenizer / "TOKENIZER_MANIFEST.json"
        write_json(manifest_path, manifest)
        return sha256(manifest_path)

    def _build_model_config(self) -> None:
        config = ModelConfig(
            vocab_size=self.vocab_size,
            dim=8,
            hidden_dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            max_seq_len=self.sequence_length,
            loss_chunk_size=4,
        )
        write_json(self.model_config, dataclasses.asdict(config))

    def _document_index(
        self,
        *,
        split: str,
        domain: str,
        archive_ordinal: int,
        content_tokens: int,
        group_id: str,
    ) -> dict[str, Any]:
        archive = f"raw/{domain}/part-{archive_ordinal:06d}.tar.zst"
        member = f"documents/{split}-{domain}.txt"
        row = {
            "record_version": DOCUMENT_INDEX_VERSION,
            "doc_id": hashlib.sha256(f"doc:{split}:{domain}".encode()).hexdigest(),
            "canonical_doc_id": hashlib.sha256(
                f"canonical:{split}:{domain}".encode()
            ).hexdigest(),
            "split_group_id": group_id,
            "split": split,
            "domain": domain,
            "bucket": "fineweb_edu" if domain == "english" else domain,
            "language": "English" if domain == "english" else domain,
            "source_archive": archive,
            "source_archive_ordinal": archive_ordinal,
            "source_manifest_index": 0,
            "source_member": member,
            "source_tokens": content_tokens,
            "selected_content_tokens": content_tokens,
            "terminal_quota_prefix": False,
            "logical_stream_start": 0,
            "logical_content_end_exclusive": content_tokens,
            "logical_eos_position": content_tokens,
        }
        shard_relative = (
            Path("provenance")
            / "documents"
            / split
            / domain
            / f"archive-{archive_ordinal:06d}.jsonl.zst"
        )
        shard_path = self.corpus / shard_relative
        write_jsonl_zst(shard_path, [row])
        manifest = {
            "format": DOCUMENT_INDEX_FORMAT,
            "format_version": DOCUMENT_INDEX_VERSION,
            "split": split,
            "domain": domain,
            "selection_manifest_sha256": SELECTION_SHA,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "sequence_length": self.sequence_length,
            "documents": 1,
            "selected_content_tokens": content_tokens,
            "logical_stream_tokens": content_tokens + 1,
            "shards": [
                {
                    "archive_ordinal": archive_ordinal,
                    "path": shard_relative.as_posix(),
                    "bytes": shard_path.stat().st_size,
                    "sha256": sha256(shard_path),
                    "records": 1,
                }
            ],
        }
        manifest_path = shard_path.parent / "manifest.json"
        write_json(manifest_path, manifest)
        return {
            "path": manifest_path.relative_to(self.corpus).as_posix(),
            "sha256": sha256(manifest_path),
            "documents": 1,
            "logical_stream_tokens": content_tokens + 1,
        }

    def _build_corpus(self, *, cross_split_group: bool) -> None:
        row_targets = {"python": 12, "other_code": 12, "english": 6}
        splits: dict[str, Any] = {}
        train_python_group = hashlib.sha256(b"group:train:python").hexdigest()
        archive_ordinal = 0
        for split_index, split in enumerate(SPLITS):
            packed_descriptors: dict[str, Any] = {}
            packed_paths: dict[str, Path] = {}
            for domain in DOMAIN_ORDER:
                rows = row_targets[domain]
                content_tokens = rows * self.sequence_length
                packed_dir = self.corpus / "packed" / split / domain
                writer = PackedShardWriter(
                    packed_dir,
                    domain=domain,
                    split=split,
                    sequence_length=self.sequence_length,
                    vocab_size=self.vocab_size,
                    eos_token_id=self.eos_token_id,
                    tokenizer_manifest_sha256=self.tokenizer_manifest_sha256,
                    curation_policy_sha256=POLICY_SHA,
                    selection_manifest_sha256=SELECTION_SHA,
                    rows_per_shard=5,
                    construction_seed=1234,
                )
                writer.add_document(
                    [2 + (index % (self.vocab_size - 2)) for index in range(content_tokens)]
                )
                packed = writer.finish()
                packed_path = packed_dir / "manifest.json"
                packed_paths[domain] = packed_path
                group_id = hashlib.sha256(f"group:{split}:{domain}".encode()).hexdigest()
                if cross_split_group and split == "validation" and domain == "python":
                    group_id = train_python_group
                document_index = self._document_index(
                    split=split,
                    domain=domain,
                    archive_ordinal=archive_ordinal,
                    content_tokens=content_tokens,
                    group_id=group_id,
                )
                archive_ordinal += 1
                packed_descriptors[domain] = {
                    "path": packed_path.relative_to(self.corpus).as_posix(),
                    "sha256": sha256(packed_path),
                    "rows": packed["rows"],
                    "documents": packed["documents"],
                    "source_content_tokens": packed["source_content_tokens"],
                    "document_index": document_index,
                }
            order_dir = self.corpus / "orders" / split
            kwargs: dict[str, Any] = {}
            if split == "train":
                kwargs.update(
                    frozen_global_microbatch_rows=6,
                    frozen_gradient_accumulation_steps=1,
                )
            order = build_training_order(
                packed_paths,
                order_dir,
                seed=100 + split_index,
                expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
                expected_total_input_tokens=120,
                input_token_tolerance=0,
                **kwargs,
            )
            order_path = order_dir / "manifest.json"
            splits[split] = {
                "packed": packed_descriptors,
                "order": {
                    "path": order_path.relative_to(self.corpus).as_posix(),
                    "sha256": sha256(order_path),
                    "format_version": order["format_version"],
                    "rows": order["rows"],
                    "packed_available_rows": order["packed_available_rows"],
                    "packed_surplus_rows": order["packed_surplus_rows"],
                    "authorized_input_tokens": order["input_token_budget"]["actual_total"],
                },
            }

        raw_archives = [
            {
                "archive": f"raw/{domain}/part-{ordinal:06d}.tar.zst",
                "sha256": hashlib.sha256(f"raw:{ordinal}".encode()).hexdigest(),
                "documents": 1,
                "content_tokens": (
                    {"python": 12, "other_code": 12, "english": 6}[domain]
                    * self.sequence_length
                ),
            }
            for ordinal, (_split, domain) in enumerate(
                (split, domain) for split in SPLITS for domain in DOMAIN_ORDER
            )
        ]
        provenance_payloads = {
            "source": {
                "format_version": 1,
                "selection_manifest_sha256": SELECTION_SHA,
                "raw_archives": raw_archives,
            },
            "policy": {
                "format_version": 1,
                "curation_policy_sha256": POLICY_SHA,
            },
            "tokenizer": {
                "format_version": 1,
                "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            },
            "fingerprints": {"format_version": 1},
        }
        provenance: dict[str, Any] = {}
        for name, payload in provenance_payloads.items():
            path = self.corpus / "provenance" / f"{name}.json"
            write_json(path, payload)
            provenance[name] = {
                "path": path.relative_to(self.corpus).as_posix(),
                "sha256": sha256(path),
            }
        leakage = {
            "content_hashes_in_multiple_splits": 0,
            "canonical_clusters_in_multiple_splits": 0,
            "source_groups_in_multiple_splits": 0,
            "cross_bucket_code_repo_groups_in_multiple_splits": 0,
            "normalized_hashes_in_multiple_splits": 0,
        }
        manifest = {
            "format": CORPUS_FORMAT,
            "format_version": CORPUS_FORMAT_VERSION,
            "identity": {
                "format": CORPUS_FORMAT,
                "format_version": CORPUS_FORMAT_VERSION,
                "selection_manifest_sha256": SELECTION_SHA,
                "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
                "curation_policy_sha256": POLICY_SHA,
                "packing_configuration": {
                    "sequence_length": self.sequence_length,
                    "rows_per_shard": 5,
                    "construction_seed": 1234,
                    "expected_vocab_size": self.vocab_size,
                    "expected_eos_token_id": self.eos_token_id,
                },
            },
            "order_configuration": {
                "order_seeds": {
                    "train": 100,
                    "validation": 101,
                    "test": 102,
                },
                "frozen_global_microbatch_rows": 6,
                "frozen_gradient_accumulation_steps": 1,
                "expected_train_input_tokens": 120,
                "expected_validation_input_tokens": 120,
                "expected_test_input_tokens": 120,
                "train_input_token_tolerance": 0,
                "validation_input_token_tolerance": 0,
                "test_input_token_tolerance": 0,
                "enforce_input_weights": True,
                "expected_input_weights": {
                    "python": 0.4,
                    "other_code": 0.4,
                    "english": 0.2,
                },
            },
            "source_cursor": {"next_archive": archive_ordinal, "archive_count": archive_ordinal},
            "split_isolation": {
                "authoritative_assignment": "frozen_leakage_safe_source_groups",
                "physical_outputs_separate": True,
                "curation_leakage_audit": leakage,
            },
            "splits": splits,
            "provenance": provenance,
        }
        manifest_path = self.corpus / "manifest.json"
        write_json(manifest_path, manifest)
        (self.corpus / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii"
        )

    def config(self, *, output_name: str = "corpus-v2") -> QualificationConfig:
        return QualificationConfig(
            corpus_root=self.corpus,
            tokenizer_root=self.tokenizer,
            output=self.receipts / output_name,
            model_config=self.model_config,
            expected_targets={"train": 120, "validation": 120, "test": 120},
            sample_rows_per_domain=3,
            sample_seed=7,
            world_size=6,
            scratch_directory=self.scratch,
            split_identity_batch_rows=4,
        )

    def argv(self, *, output_name: str = "corpus-v2") -> list[str]:
        return [
            "--corpus-root",
            str(self.corpus),
            "--tokenizer-root",
            str(self.tokenizer),
            "--model-config",
            str(self.model_config),
            "--output",
            str(self.receipts / output_name),
            "--expected-train-input-tokens",
            "120",
            "--expected-validation-input-tokens",
            "120",
            "--expected-test-input-tokens",
            "120",
            "--sample-rows-per-domain",
            "3",
            "--sample-seed",
            "7",
            "--scratch-directory",
            str(self.scratch),
            "--split-identity-batch-rows",
            "4",
        ]


class QualifyTrainingCorpusTest(unittest.TestCase):
    def test_runbook_uses_a_dedicated_cpu_torch_environment(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        requirements = (
            project_root / "requirements-qualification.txt"
        ).read_text(encoding="utf-8")
        runbook = (
            project_root / "docs/operations/training-corpus-qualification.md"
        ).read_text(encoding="utf-8")
        self.assertNotRegex(requirements, r"(?m)^torch[<=>]")
        self.assertIn("/opt/coding-model-qualification-venv/bin/python", runbook)
        self.assertIn("https://download.pytorch.org/whl/cpu", runbook)
        self.assertIn("Do **not** invoke", runbook)
        self.assertIn("/opt/coding-model-data-venv/bin/python", runbook)

    def test_complete_corpus_passes_and_publishes_authenticated_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            self.assertEqual(main(fixture.argv()), 0)
            receipt_path = fixture.receipts / "corpus-v2" / "qualification.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "pass")
            self.assertTrue(receipt["checks"]["boundary_bits_and_loss_masks_consistent"])
            self.assertEqual(
                receipt["split_identity_audit"]["cross_split_collisions"],
                {"source": 0, "split_group": 0},
            )
            self.assertEqual(
                receipt["splits"]["train"]["order"]["rows_per_domain"],
                {"python": 12, "other_code": 12, "english": 6},
            )
            sample = receipt["splits"]["train"]["deterministic_samples"]["python"]
            self.assertEqual(sample["sampled_rows"], 3)
            sidecar = receipt_path.with_name("qualification.json.sha256")
            self.assertEqual(
                sidecar.read_text(encoding="ascii"),
                f"{sha256(receipt_path)}  qualification.json\n",
            )

    def test_cross_split_group_collision_fails_and_still_publishes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary), cross_split_group=True)
            self.assertEqual(main(fixture.argv(output_name="failed")), 1)
            receipt_path = fixture.receipts / "failed" / "qualification.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "fail")
            self.assertIn("occur in multiple splits", receipt["error"])
            self.assertEqual(
                receipt["details"]["split_identity_audit"][
                    "cross_split_collisions"
                ]["split_group"],
                1,
            )
            self.assertEqual(
                (fixture.receipts / "failed/qualification.json.sha256").read_text(
                    encoding="ascii"
                ),
                f"{sha256(receipt_path)}  qualification.json\n",
            )

    def test_receipt_generation_is_write_once_and_never_publishes_a_torn_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = root / "generation-1"
            receipt, sidecar = publish_receipt(generation, {"status": "pass"})
            self.assertTrue(receipt.is_file())
            self.assertTrue(sidecar.is_file())
            original = {path.name: path.read_bytes() for path in generation.iterdir()}
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                publish_receipt(generation, {"status": "fail"})
            self.assertEqual(
                {path.name: path.read_bytes() for path in generation.iterdir()},
                original,
            )

            torn = root / "externally-torn-generation"
            torn.mkdir()
            (torn / "qualification.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                publish_receipt(torn, {"status": "pass"})
            self.assertFalse((torn / "qualification.json.sha256").exists())

    def test_concurrent_receipt_publishers_install_exactly_one_complete_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("fork")
            output = Path(temporary) / "raced-generation"
            ready = context.Queue()
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=race_publish_worker,
                    args=(str(output), value, ready, start, results),
                )
                for value in ("pass-a", "pass-b")
            ]
            for process in processes:
                process.start()
            self.assertEqual({ready.get(timeout=10), ready.get(timeout=10)}, {"pass-a", "pass-b"})
            start.set()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            outcomes = {results.get(timeout=10), results.get(timeout=10)}
            self.assertEqual(
                sorted(outcome for _value, outcome in outcomes),
                ["FileExistsError", "published"],
            )
            receipt = output / "qualification.json"
            sidecar = output / "qualification.json.sha256"
            self.assertTrue(receipt.is_file())
            self.assertEqual(
                sidecar.read_text(encoding="ascii"),
                f"{sha256(receipt)}  qualification.json\n",
            )

    def test_packed_payload_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            payload = fixture.corpus / "packed/train/python/shard-000000.tokens.bin"
            raw = bytearray(payload.read_bytes())
            raw[0] ^= 1
            payload.write_bytes(raw)
            with self.assertRaisesRegex(QualificationError, "Checksum mismatch"):
                qualify_corpus(fixture.config())

    def test_model_vocabulary_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            payload = json.loads(fixture.model_config.read_text(encoding="utf-8"))
            payload["vocab_size"] = fixture.vocab_size + 1
            write_json(fixture.model_config, payload)
            with self.assertRaisesRegex(QualificationError, "differs from model"):
                qualify_corpus(fixture.config())


if __name__ == "__main__":
    unittest.main()
