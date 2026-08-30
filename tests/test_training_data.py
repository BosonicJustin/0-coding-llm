from __future__ import annotations

import hashlib
import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import (
    DOMAIN_ORDER,
    IGNORE_INDEX,
    DistributedBatchSampler,
    PackedBatchCollator,
    PackedShardDataset,
    PackedShardWriter,
    build_training_order,
    create_training_dataloader,
    decode_reference,
    evaluation_order_geometry,
    frozen_training_geometry,
    validate_packed_manifest,
    validate_training_order,
)


FAKE_TOKENIZER_MANIFEST_SHA256 = hashlib.sha256(
    b"synthetic-unit-test-tokenizer-manifest"
).hexdigest()
ALTERNATE_TOKENIZER_MANIFEST_SHA256 = hashlib.sha256(
    b"different-synthetic-unit-test-tokenizer-manifest"
).hexdigest()
FAKE_CURATION_POLICY_SHA256 = hashlib.sha256(b"synthetic-curation-policy").hexdigest()
FAKE_SELECTION_MANIFEST_SHA256 = hashlib.sha256(
    b"synthetic-selection-manifest"
).hexdigest()


def build_domain(
    root: Path,
    domain: str,
    rows: int,
    *,
    sequence_length: int = 4,
    tokenizer_manifest_sha256: str = FAKE_TOKENIZER_MANIFEST_SHA256,
) -> Path:
    output = root / domain
    writer = PackedShardWriter(
        output,
        domain=domain,
        split="train",
        sequence_length=sequence_length,
        vocab_size=256,
        eos_token_id=0,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        rows_per_shard=2,
        construction_seed=17,
    )
    # One stream with rows*T content tokens plus EOS produces exactly `rows`
    # fixed rows and one lookahead-only tail token.
    writer.add_document([(index % 200) + 1 for index in range(rows * sequence_length)])
    manifest = writer.finish()
    assert manifest["rows"] == rows
    return output / "manifest.json"


def build_three_domains(root: Path) -> dict[str, Path]:
    counts = {"python": 4, "other_code": 4, "english": 2}
    return {domain: build_domain(root, domain, counts[domain]) for domain in DOMAIN_ORDER}


class PackedShardTest(unittest.TestCase):
    @staticmethod
    def _resumable_writer(output: Path, *, resume: bool = False, **overrides):
        arguments = {
            "domain": "python",
            "split": "train",
            "sequence_length": 4,
            "vocab_size": 256,
            "eos_token_id": 0,
            "tokenizer_manifest_sha256": FAKE_TOKENIZER_MANIFEST_SHA256,
            "rows_per_shard": 2,
            "construction_seed": 17,
            "curation_policy_sha256": FAKE_CURATION_POLICY_SHA256,
            "selection_manifest_sha256": FAKE_SELECTION_MANIFEST_SHA256,
            "resume": resume,
        }
        arguments.update(overrides)
        return PackedShardWriter(output, **arguments)

    @staticmethod
    def _directory_bytes(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_boundaries_positions_and_labels_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
                rows_per_shard=2,
            )
            writer.add_document([10, 11])
            writer.add_document([20, 21, 22])
            writer.add_document([30, 31, 32, 33])
            manifest = writer.finish()
            self.assertEqual(manifest["rows"], 2)
            validate_packed_manifest(output / "manifest.json")

            dataset = PackedShardDataset(output / "manifest.json")
            row = dataset[0]
            batch = PackedBatchCollator(4)(
                [{**row, "domain_id": 0, "sample_reference": 0}]
            )
            self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 0, 20]])
            self.assertEqual(batch["labels"].tolist(), [[11, 0, IGNORE_INDEX, 21]])
            self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2, 0]])
            self.assertEqual(batch["document_ids"].tolist(), [[0, 0, 0, 1]])
            self.assertEqual(batch["num_loss_tokens"].item(), 3)

    def test_literal_eos_token_does_not_create_a_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document([7, 0, 8, 9, 10])
            writer.finish()
            dataset = PackedShardDataset(output / "manifest.json")
            batch = PackedBatchCollator(4)(
                [{**dataset[0], "domain_id": 0, "sample_reference": 0}]
            )
            self.assertEqual(batch["input_ids"].tolist(), [[7, 0, 8, 9]])
            self.assertEqual(batch["labels"].tolist(), [[0, 8, 9, 10]])
            self.assertEqual(batch["document_ids"].tolist(), [[0, 0, 0, 0]])

    def test_long_document_rows_overlap_only_at_target_input_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document(list(range(1, 10)))
            manifest = writer.finish()
            self.assertEqual(manifest["rows"], 2)
            dataset = PackedShardDataset(output / "manifest.json")
            first = dataset[0]["tokens"].tolist()
            second = dataset[1]["tokens"].tolist()
            self.assertEqual(first, [1, 2, 3, 4, 5])
            self.assertEqual(second, [5, 6, 7, 8, 9])
            self.assertEqual(first[-1], second[0])

    def test_checksum_and_size_validation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = build_domain(root, "python", 2)
            validate_packed_manifest(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tokens_path = manifest_path.parent / manifest["shards"][0]["tokens"]["path"]
            with tokens_path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(IOError, "Size mismatch"):
                validate_packed_manifest(manifest_path)

    def test_writer_refuses_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            output.mkdir()
            (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                PackedShardWriter(
                    output,
                    domain="python",
                    split="train",
                    sequence_length=4,
                    vocab_size=100,
                    eos_token_id=0,
                    tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
                )

    def test_interrupted_multishard_resume_is_byte_identical(self) -> None:
        documents = [
            [document * 3 + 1, document * 3 + 2, document * 3 + 3]
            for document in range(12)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = root / "uninterrupted"
            resumed_output = root / "resumed"

            baseline = self._resumable_writer(uninterrupted)
            for index, document in enumerate(documents):
                baseline.add_document(
                    document,
                    source_cursor={"source": "fixture", "next_document": index + 1},
                )
            baseline_manifest = baseline.finish()

            class ForcedInterruption(RuntimeError):
                pass

            with self.assertRaises(ForcedInterruption):
                with self._resumable_writer(resumed_output) as interrupted:
                    for index, document in enumerate(documents[:4]):
                        interrupted.add_document(
                            document,
                            source_cursor={
                                "source": "fixture",
                                "next_document": index + 1,
                            },
                        )
                    # Move multiple shards beyond the durable cursor. Resume
                    # must truncate the old open-shard prefix and remove every
                    # uncommitted successor before replaying these documents.
                    for document in documents[4:8]:
                        interrupted.add_document(document)
                    raise ForcedInterruption("synthetic process death")

            journal = json.loads(
                (resumed_output / ".packing-journal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(journal["state"]["completed_shards"]), 1)
            self.assertEqual(journal["state"]["open_shard"]["rows"], 1)

            resumed = self._resumable_writer(resumed_output, resume=True)
            self.assertTrue(resumed.resumed)
            self.assertEqual(
                resumed.source_cursor,
                {"source": "fixture", "next_document": 4},
            )
            for index, document in enumerate(documents[4:], start=4):
                resumed.add_document(
                    document,
                    source_cursor={"source": "fixture", "next_document": index + 1},
                )
            resumed_manifest = resumed.finish()

            self.assertEqual(resumed_manifest, baseline_manifest)
            self.assertEqual(
                self._directory_bytes(resumed_output),
                self._directory_bytes(uninterrupted),
            )
            self.assertFalse((resumed_output / ".packing-journal.json").exists())
            validate_packed_manifest(resumed_output / "manifest.json")

    def test_resume_fails_closed_on_construction_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"

            class ForcedInterruption(RuntimeError):
                pass

            with self.assertRaises(ForcedInterruption):
                with self._resumable_writer(output) as writer:
                    writer.add_document(
                        [1, 2, 3],
                        source_cursor={"source": "fixture", "next_document": 1},
                    )
                    raise ForcedInterruption

            mismatches = (
                {"tokenizer_manifest_sha256": ALTERNATE_TOKENIZER_MANIFEST_SHA256},
                {"curation_policy_sha256": hashlib.sha256(b"other-policy").hexdigest()},
                {
                    "selection_manifest_sha256": hashlib.sha256(
                        b"other-selection"
                    ).hexdigest()
                },
                {"construction_seed": 18},
            )
            for override in mismatches:
                with self.subTest(override=override):
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        self._resumable_writer(output, resume=True, **override)

    def test_resume_verifies_committed_shard_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"

            class ForcedInterruption(RuntimeError):
                pass

            with self.assertRaises(ForcedInterruption):
                with self._resumable_writer(output) as writer:
                    for index in range(3):
                        writer.add_document(
                            [index * 3 + 1, index * 3 + 2, index * 3 + 3],
                            source_cursor={"next_document": index + 1},
                        )
                    raise ForcedInterruption
            committed = output / "shard-000000.tokens.bin"
            with committed.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))
            with self.assertRaisesRegex(IOError, "checksum mismatch"):
                self._resumable_writer(output, resume=True)

    def test_tokenizer_manifest_hash_is_mandatory_and_strict(self) -> None:
        invalid_hashes = (
            None,
            "a" * 63,
            "A" * 64,
            "g" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, tokenizer_hash in enumerate(invalid_hashes):
                with self.subTest(tokenizer_hash=tokenizer_hash):
                    with self.assertRaisesRegex(ValueError, "64-character lowercase hex"):
                        PackedShardWriter(
                            root / str(index),
                            domain="python",
                            split="train",
                            sequence_length=4,
                            vocab_size=100,
                            eos_token_id=0,
                            tokenizer_manifest_sha256=tokenizer_hash,  # type: ignore[arg-type]
                        )

    def test_parsed_manifest_requires_valid_tokenizer_hash(self) -> None:
        for mutation in (None, "F" * 64):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                manifest_path = build_domain(Path(temporary), "python", 1)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if mutation is None:
                    del manifest["tokenizer_manifest_sha256"]
                else:
                    manifest["tokenizer_manifest_sha256"] = mutation
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tokenizer_manifest_sha256"):
                    validate_packed_manifest(manifest_path, verify_checksums=False)

    def test_parsed_manifest_rejects_unknown_starts_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = build_domain(Path(temporary), "python", 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["starts_encoding"] = "numpy-packbits-big"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "segment-start encoding"):
                PackedShardDataset(manifest_path)

    def test_tail_accounting_distinguishes_lookahead_from_unstored_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document([1, 2, 3, 4, 5, 6])
            manifest = writer.finish()
            self.assertEqual(manifest["stream_tokens"], 7)
            self.assertEqual(manifest["input_tokens"], 4)
            self.assertEqual(manifest["unused_as_input_tail_tokens"], 3)
            self.assertEqual(manifest["unstored_tail_tokens"], 2)
            self.assertNotIn("discarded_tail_tokens", manifest)
            validate_packed_manifest(output / "manifest.json")

    def test_tail_accounting_with_no_emitted_rows_marks_every_token_unstored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document([1, 2])
            manifest = writer.finish()
            self.assertEqual(manifest["rows"], 0)
            self.assertEqual(manifest["stream_tokens"], 3)
            self.assertEqual(manifest["unused_as_input_tail_tokens"], 3)
            self.assertEqual(manifest["unstored_tail_tokens"], 3)
            validate_packed_manifest(output / "manifest.json")

    def test_semantic_payload_validation_rejects_corruption(self) -> None:
        mutations = ("token", "first_start", "unused_start_bits", "loss_counters")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                manifest_path = build_domain(Path(temporary), "python", 2)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                shard = manifest["shards"][0]
                if mutation == "token":
                    token_path = manifest_path.parent / shard["tokens"]["path"]
                    with token_path.open("r+b") as handle:
                        handle.write(np.asarray([256], dtype="<u2").tobytes())
                    expected_error = "outside"
                elif mutation == "first_start":
                    starts_path = manifest_path.parent / shard["starts"]["path"]
                    with starts_path.open("r+b") as handle:
                        value = handle.read(1)[0]
                        handle.seek(0)
                        handle.write(bytes([value & 0xFE]))
                    expected_error = "does not begin"
                elif mutation == "unused_start_bits":
                    starts_path = manifest_path.parent / shard["starts"]["path"]
                    with starts_path.open("r+b") as handle:
                        value = handle.read(1)[0]
                        handle.seek(0)
                        handle.write(bytes([value | 0x80]))
                    expected_error = "Unused high start bits"
                else:
                    manifest["valid_loss_tokens"] -= 1
                    manifest["masked_boundary_labels"] += 1
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    expected_error = "recomputed"
                with self.assertRaisesRegex(ValueError, expected_error):
                    validate_packed_manifest(manifest_path, verify_checksums=False)

    def test_segment_start_requires_preceding_eos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=4,
                vocab_size=100,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document([1, 2])
            writer.add_document([3, 4, 5])
            writer.finish()
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            token_path = output / manifest["shards"][0]["tokens"]["path"]
            # Column 3 begins the second document, so column 2 must be EOS.
            with token_path.open("r+b") as handle:
                handle.seek(2 * np.dtype("<u2").itemsize)
                handle.write(np.asarray([9], dtype="<u2").tobytes())
            with self.assertRaisesRegex(ValueError, "not preceded by EOS"):
                validate_packed_manifest(manifest_path, verify_checksums=False)


class TrainingOrderTest(unittest.TestCase):
    def test_full_shuffle_is_exact_deterministic_and_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            first = root / "order-first"
            second = root / "order-second"
            third = root / "order-third"
            expected_weights = {"python": 0.4, "other_code": 0.4, "english": 0.2}
            build_training_order(manifests, first, seed=123, expected_weights=expected_weights)
            build_training_order(manifests, second, seed=123, expected_weights=expected_weights)
            build_training_order(manifests, third, seed=124, expected_weights=expected_weights)
            order_manifest = validate_training_order(first / "manifest.json")
            self.assertEqual(
                order_manifest["tokenizer_manifest_sha256"],
                FAKE_TOKENIZER_MANIFEST_SHA256,
            )
            self.assertEqual(order_manifest["expected_input_token_weights"], expected_weights)

            first_order = np.fromfile(first / "order.bin", dtype="<u8")
            second_order = np.fromfile(second / "order.bin", dtype="<u8")
            third_order = np.fromfile(third / "order.bin", dtype="<u8")
            np.testing.assert_array_equal(first_order, second_order)
            self.assertFalse(np.array_equal(first_order, third_order))
            self.assertEqual(len(np.unique(first_order)), 10)
            decoded = [decode_reference(value) for value in first_order]
            self.assertEqual([domain for domain, _ in decoded].count(0), 4)
            self.assertEqual([domain for domain, _ in decoded].count(1), 4)
            self.assertEqual([domain for domain, _ in decoded].count(2), 2)
            self.assertNotEqual(decoded, sorted(decoded))

    def test_wrong_domain_mixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = {
                "python": build_domain(root / "packed", "python", 6),
                "other_code": build_domain(root / "packed", "other_code", 2),
                "english": build_domain(root / "packed", "english", 2),
            }
            with self.assertRaisesRegex(ValueError, "mixture is wrong"):
                build_training_order(
                    manifests,
                    root / "order",
                    seed=1,
                    expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
                )

    def test_mixed_tokenizer_manifests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            manifests["english"] = build_domain(
                root / "alternate",
                "english",
                2,
                tokenizer_manifest_sha256=ALTERNATE_TOKENIZER_MANIFEST_SHA256,
            )
            with self.assertRaisesRegex(ValueError, "tokenizer_manifest_sha256"):
                build_training_order(manifests, root / "order", seed=1)

    def test_absolute_input_token_budget_is_enforced_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            weights = {"python": 0.4, "other_code": 0.4, "english": 0.2}
            with self.assertRaisesRegex(ValueError, "Insufficient packed rows"):
                build_training_order(
                    manifests,
                    root / "wrong-order",
                    seed=1,
                    expected_total_input_tokens=50,
                    expected_weights=weights,
                )
            manifest = build_training_order(
                manifests,
                root / "order",
                seed=1,
                expected_weights=weights,
                expected_total_input_tokens=33,
            )
            self.assertEqual(
                manifest["expected_input_token_weights"],
                {"python": 0.4, "other_code": 0.4, "english": 0.2},
            )
            self.assertEqual(
                manifest["input_token_budget"],
                {
                    "policy": (
                        "packed_model_inputs_including_eos_excluding_lookahead_duplicates"
                    ),
                    "direction": "at_or_below",
                    "expected_total": 33,
                    "actual_total": 32,
                    "available_total": 32,
                    "packed_available_total": 40,
                    "packed_surplus_total": 8,
                    "accounting": "available-order-inputs",
                    "delta": -1,
                    "tolerance": 3,
                },
            )
            self.assertEqual(manifest["rows_per_domain"], {
                "python": 3,
                "other_code": 3,
                "english": 2,
            })
            self.assertEqual(manifest["packed_available_rows"], 10)
            self.assertEqual(manifest["packed_surplus_rows"], 2)
            validate_training_order(root / "order" / "manifest.json")

    def test_capped_subset_is_deterministic_and_uses_real_source_row_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            weights = {"python": 0.4, "other_code": 0.4, "english": 0.2}
            for name, seed in (("first", 73), ("second", 73), ("third", 74)):
                build_training_order(
                    manifests,
                    root / name,
                    seed=seed,
                    expected_weights=weights,
                    expected_total_input_tokens=33,
                )
            first = np.fromfile(root / "first" / "order.bin", dtype="<u8")
            second = np.fromfile(root / "second" / "order.bin", dtype="<u8")
            third = np.fromfile(root / "third" / "order.bin", dtype="<u8")
            np.testing.assert_array_equal(first, second)
            self.assertFalse(np.array_equal(first, third))
            self.assertEqual(len(np.unique(first)), 8)
            decoded = [decode_reference(value) for value in first]
            self.assertEqual([domain for domain, _row in decoded].count(0), 3)
            self.assertEqual([domain for domain, _row in decoded].count(1), 3)
            self.assertEqual([domain for domain, _row in decoded].count(2), 2)
            self.assertTrue(all(0 <= row < (4, 4, 2)[domain] for domain, row in decoded))
            validate_training_order(root / "first" / "manifest.json")

    def test_capped_order_rejects_duplicate_and_out_of_range_source_rows(self) -> None:
        for mutation, expected_error in (
            ("duplicate", "duplicate"),
            ("out_of_range", "out of range"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifests = build_three_domains(root / "packed")
                build_training_order(
                    manifests,
                    root / "order",
                    seed=17,
                    expected_weights={
                        "python": 0.4,
                        "other_code": 0.4,
                        "english": 0.2,
                    },
                    expected_total_input_tokens=33,
                )
                order_path = root / "order" / "order.bin"
                values = np.fromfile(order_path, dtype="<u8")
                if mutation == "duplicate":
                    values[0] = values[1]
                else:
                    values[0] = np.uint64(4)
                values.tofile(order_path)
                manifest_path = root / "order" / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["order"]["sha256"] = hashlib.sha256(
                    order_path.read_bytes()
                ).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected_error):
                    validate_training_order(manifest_path)

    def test_frozen_training_geometry_is_all_or_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            with self.assertRaisesRegex(ValueError, "supplied together"):
                build_training_order(
                    manifests,
                    root / "partial-microbatch",
                    seed=1,
                    frozen_global_microbatch_rows=2,
                )
            with self.assertRaisesRegex(ValueError, "supplied together"):
                build_training_order(
                    manifests,
                    root / "partial-accumulation",
                    seed=1,
                    frozen_gradient_accumulation_steps=2,
                )
            build_training_order(manifests, root / "unfrozen", seed=1)
            with self.assertRaisesRegex(ValueError, "no frozen"):
                frozen_training_geometry(root / "unfrozen" / "manifest.json")

    def test_unfrozen_evaluation_geometry_uses_complete_microbatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "validation", seed=1)
            manifest_path = root / "validation" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["split"] = "validation"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            geometry = evaluation_order_geometry(
                manifest_path,
                global_microbatch_rows=3,
                verify_checksum=False,
            )
            self.assertEqual(geometry["available_rows"], 10)
            self.assertEqual(geometry["consumed_rows"], 9)
            self.assertEqual(geometry["dropped_tail_rows"], 1)
            self.assertEqual(geometry["available_global_microbatches"], 3)

            sampler = DistributedBatchSampler(
                manifest_path,
                global_microbatch_rows=3,
                gradient_accumulation_steps=1,
                verify_checksum=False,
            )
            try:
                self.assertEqual(len(sampler), 3)
                self.assertEqual(sampler.trainable_rows, 9)
            finally:
                sampler.close()

    def test_frozen_optimizer_updates_record_exact_consumed_token_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            manifest = build_training_order(
                manifests,
                root / "order",
                seed=3,
                expected_weights={
                    "python": 0.4,
                    "other_code": 0.4,
                    "english": 0.2,
                },
                expected_total_input_tokens=40,
                frozen_global_microbatch_rows=2,
                frozen_gradient_accumulation_steps=2,
            )
            consumption = manifest["training_consumption"]
            self.assertEqual(consumption["frozen_global_microbatch_rows"], 2)
            self.assertEqual(consumption["frozen_gradient_accumulation_steps"], 2)
            self.assertEqual(consumption["frozen_optimizer_update_rows"], 4)
            self.assertEqual(consumption["available_global_microbatches"], 4)
            self.assertEqual(consumption["consumed_global_microbatches"], 4)
            self.assertEqual(consumption["dropped_global_microbatches"], 0)
            self.assertEqual(consumption["optimizer_updates"], 2)
            self.assertEqual(consumption["consumed_rows"], 8)
            self.assertEqual(consumption["dropped_tail_rows"], 0)
            self.assertEqual(consumption["available_input_tokens"], 32)
            self.assertEqual(consumption["consumed_input_tokens"], 32)
            self.assertEqual(consumption["dropped_input_tokens"], 0)
            self.assertEqual(consumption["available_supervised_tokens"], 32)
            self.assertEqual(consumption["consumed_supervised_tokens"], 32)
            self.assertEqual(consumption["dropped_supervised_tokens"], 0)
            self.assertEqual(sum(consumption["consumed_rows_per_domain"].values()), 8)
            self.assertEqual(
                manifest["input_token_budget"],
                {
                    "policy": (
                        "packed_model_inputs_including_eos_excluding_lookahead_duplicates"
                    ),
                    "direction": "at_or_below",
                    "expected_total": 40,
                    "actual_total": 32,
                    "available_total": 32,
                    "packed_available_total": 40,
                    "packed_surplus_total": 8,
                    "accounting": "consumed-complete-optimizer-update-inputs",
                    "delta": -8,
                    "tolerance": 15,
                },
            )
            validate_training_order(root / "order" / "manifest.json")
            geometry = frozen_training_geometry(root / "order" / "manifest.json")
            self.assertEqual(geometry["optimizer_update_rows"], 4)
            self.assertEqual(geometry["optimizer_updates"], 2)
            self.assertEqual(geometry["consumed_input_tokens"], 32)

            sampler = DistributedBatchSampler(
                root / "order" / "manifest.json",
                global_microbatch_rows=2,
                gradient_accumulation_steps=2,
            )
            self.assertEqual(len(sampler), 4)
            self.assertEqual(sampler.trainable_rows, 8)
            with self.assertRaisesRegex(ValueError, "optimizer-update boundary"):
                sampler.state_dict(completed_global_microbatches=1)
            state = sampler.state_dict(completed_global_microbatches=2)
            self.assertEqual(state["completed_optimizer_updates"], 1)
            self.assertEqual(state["next_order_row"], 4)
            sampler.close()
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                DistributedBatchSampler(
                    root / "order" / "manifest.json",
                    global_microbatch_rows=2,
                    gradient_accumulation_steps=1,
                )
            with self.assertRaisesRegex(ValueError, "immutable order"):
                DistributedBatchSampler(
                    root / "order" / "manifest.json",
                    global_microbatch_rows=5,
                    gradient_accumulation_steps=2,
                )

    def test_frozen_consumption_domain_counts_are_proved_from_order_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(
                manifests,
                root / "order",
                seed=7,
                frozen_global_microbatch_rows=2,
                frozen_gradient_accumulation_steps=2,
            )
            manifest_path = root / "order" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            consumption = manifest["training_consumption"]
            consumed = consumption["consumed_rows_per_domain"]
            totals = manifest["rows_per_domain"]
            receiver = next(domain for domain in DOMAIN_ORDER if consumed[domain] < totals[domain])
            donor = next(
                domain
                for domain in DOMAIN_ORDER
                if domain != receiver and consumed[domain] > 0
            )
            consumed[donor] -= 1
            consumed[receiver] += 1
            for domain in DOMAIN_ORDER:
                consumption["consumed_input_tokens_per_domain"][domain] = (
                    consumed[domain] * manifest["sequence_length"]
                )
                consumption["realized_consumed_input_token_weights"][domain] = (
                    consumed[domain] / consumption["consumed_rows"]
                )
                consumption["dropped_rows_per_domain"][domain] = (
                    totals[domain] - consumed[domain]
                )
                consumption["dropped_input_tokens_per_domain"][domain] = (
                    consumption["dropped_rows_per_domain"][domain]
                    * manifest["sequence_length"]
                )
                consumption["consumed_supervised_tokens_per_domain"][domain] = (
                    consumed[domain] * manifest["sequence_length"]
                )
                consumption["dropped_supervised_tokens_per_domain"][domain] = (
                    consumption["dropped_rows_per_domain"][domain]
                    * manifest["sequence_length"]
                )
                consumption["realized_consumed_supervised_token_weights"][domain] = (
                    consumed[domain] / consumption["consumed_rows"]
                )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen order prefix"):
                validate_training_order(manifest_path)

    def test_order_schema_rejects_wrong_bit_allocation_and_legacy_versions(self) -> None:
        for mutation, expected_error in (
            ({"encoding": {"domain_bits": 7, "row_bits": 57}}, "bit allocation"),
            ({"format_version": 2}, "before v4"),
            ({"format_version": 3}, "before v4"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifests = build_three_domains(root / "packed")
                build_training_order(manifests, root / "order", seed=1)
                manifest_path = root / "order" / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(mutation)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected_error):
                    DistributedBatchSampler(manifest_path, global_microbatch_rows=2)

    def test_default_sampler_rejects_same_size_order_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=1)
            order_path = root / "order" / "order.bin"
            original_size = order_path.stat().st_size
            with order_path.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 1]))
            self.assertEqual(order_path.stat().st_size, original_size)
            with self.assertRaisesRegex(IOError, "Order checksum mismatch"):
                DistributedBatchSampler(
                    root / "order" / "manifest.json", global_microbatch_rows=2
                )

    def test_resume_state_binds_every_data_identity_and_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=42)
            order_manifest = root / "order" / "manifest.json"
            initial = DistributedBatchSampler(
                order_manifest, global_microbatch_rows=2, world_size=1
            )
            all_batches = list(initial)
            state = initial.state_dict(completed_global_microbatches=2)
            initial.close()

            resumed = DistributedBatchSampler(
                order_manifest,
                global_microbatch_rows=2,
                world_size=1,
                resume_state=state,
            )
            self.assertEqual(resumed.start_global_microbatch, 2)
            self.assertEqual(list(resumed), all_batches[2:])
            resumed.close()

            mutations = {
                "order_manifest_sha256": "0" * 64,
                "order_payload_sha256": "1" * 64,
                "tokenizer_manifest_sha256": "2" * 64,
                "global_microbatch_rows": 4,
                "gradient_accumulation_steps": 2,
                "completed_optimizer_updates": (
                    state["completed_optimizer_updates"] + 1
                ),
                "world_size": 2,
                "next_order_row": state["next_order_row"] + 1,
                "consumed_input_tokens": state["consumed_input_tokens"] + 1,
            }
            for field, value in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(state)
                    changed[field] = value
                    with self.assertRaisesRegex(ValueError, "Sampler resume"):
                        DistributedBatchSampler(
                            order_manifest,
                            global_microbatch_rows=2,
                            world_size=1,
                            resume_state=changed,
                        )

            changed = copy.deepcopy(state)
            changed["dataset_manifest_sha256"]["python"] = "3" * 64
            with self.assertRaisesRegex(ValueError, "dataset_manifest_sha256 mismatch"):
                DistributedBatchSampler(
                    order_manifest,
                    global_microbatch_rows=2,
                    resume_state=changed,
                )

            missing = copy.deepcopy(state)
            del missing["next_order_row"]
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                DistributedBatchSampler(
                    order_manifest,
                    global_microbatch_rows=2,
                    resume_state=missing,
                )

    def test_dataloader_accepts_validated_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=5)
            order_manifest = root / "order" / "manifest.json"
            initial = DistributedBatchSampler(order_manifest, global_microbatch_rows=2)
            state = initial.state_dict(completed_global_microbatches=1)
            initial.close()
            loader, sampler = create_training_dataloader(
                order_manifest,
                global_microbatch_rows=2,
                resume_state=state,
                num_workers=0,
                pin_memory=False,
            )
            expected = np.fromfile(root / "order" / "order.bin", dtype="<u8")[2:4]
            batch = next(iter(loader))
            np.testing.assert_array_equal(
                batch["sample_references"].numpy(), expected.astype(np.int64)
            )
            sampler.close()
            loader.dataset.close()

    def test_tampered_expected_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(
                manifests,
                root / "order",
                seed=1,
                expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            )
            manifest_path = root / "order" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expected_input_token_weights"] = {
                "python": 0.6,
                "other_code": 0.2,
                "english": 0.2,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixture is wrong"):
                validate_training_order(manifest_path)

    def test_distributed_ranks_are_disjoint_and_resume_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=42)
            order_manifest = root / "order" / "manifest.json"
            rank_zero = DistributedBatchSampler(
                order_manifest, global_microbatch_rows=4, rank=0, world_size=2
            )
            rank_one = DistributedBatchSampler(
                order_manifest, global_microbatch_rows=4, rank=1, world_size=2
            )
            zero_batches = list(rank_zero)
            one_batches = list(rank_one)
            self.assertEqual(len(zero_batches), 2)
            self.assertEqual(rank_zero.dropped_rows, 2)
            order = np.fromfile(root / "order" / "order.bin", dtype="<u8")
            for batch_index, (left, right) in enumerate(zip(zero_batches, one_batches, strict=True)):
                self.assertTrue(set(left).isdisjoint(right))
                expected = set(int(value) for value in order[batch_index * 4 : (batch_index + 1) * 4])
                self.assertEqual(set(left) | set(right), expected)

            resumed = DistributedBatchSampler(
                order_manifest,
                global_microbatch_rows=4,
                rank=0,
                world_size=2,
                start_global_microbatch=1,
            )
            self.assertEqual(list(resumed), zero_batches[1:])
            state = rank_zero.state_dict(completed_global_microbatches=1)
            self.assertEqual(state["completed_global_microbatches"], 1)
            self.assertEqual(state["dropped_rows"], 2)

    def test_dataloader_uses_order_and_emits_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=99)
            loader, sampler = create_training_dataloader(
                root / "order" / "manifest.json",
                global_microbatch_rows=5,
                num_workers=0,
                pin_memory=False,
                verify_payload_checksums=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual(len(batches), 2)
            references = torch.cat([batch["sample_references"] for batch in batches])
            self.assertEqual(len(torch.unique(references)), 10)
            for batch in batches:
                self.assertEqual(tuple(batch["input_ids"].shape), (5, 4))
                self.assertEqual(tuple(batch["labels"].shape), (5, 4))
                self.assertEqual(tuple(batch["position_ids"].shape), (5, 4))
                self.assertEqual(tuple(batch["document_ids"].shape), (5, 4))
                self.assertTrue(torch.all(batch["position_ids"][:, 0] == 0))

    @unittest.skipIf(
        sys.platform == "darwin",
        "sandboxed macOS blocks PyTorch's torch_shm_manager; production is Linux",
    )
    def test_dataloader_workers_can_reopen_memory_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = build_three_domains(root / "packed")
            build_training_order(manifests, root / "order", seed=101)
            loader, _ = create_training_dataloader(
                root / "order" / "manifest.json",
                global_microbatch_rows=5,
                num_workers=2,
                pin_memory=False,
            )
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["input_ids"].shape), (5, 4))


if __name__ == "__main__":
    unittest.main()
