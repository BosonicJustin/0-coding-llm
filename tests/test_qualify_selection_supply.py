from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pretrain.data import DOMAIN_ORDER, PackedShardWriter
from pretrain.materialize import MaterializationError, SPLITS
from pretrain.raw_token_cache_inventory import RawTokenCacheInventoryError
from scripts.qualify_selection_supply import (
    DEFAULT_TARGETS,
    geometry_independent_available_rows,
    main,
    qualify_selection_supply,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "qualify_selection_supply.py"


def _write_selection(
    root: Path,
    available_rows: dict[tuple[str, str], int],
    *,
    sequence_length: int,
    documents: int = 1,
) -> dict[str, object]:
    root.mkdir(parents=True)
    selected_totals: list[dict[str, object]] = []
    reference_quotas: list[dict[str, object]] = []
    selected_documents = 0
    for split in SPLITS:
        for domain in DOMAIN_ORDER:
            rows = available_rows[(split, domain)]
            # With this content count, floor((content + documents - 1) / T)
            # is exactly `rows`, including rows == 0.
            content_tokens = rows * sequence_length + 1 - documents
            if content_tokens < 1:
                content_tokens = 1
            selected_totals.append(
                {
                    "split": split,
                    "category": domain,
                    "unit": "pre_packing_starcoder2_content_tokens",
                    "documents": documents,
                    "selected_tokens": content_tokens,
                    "terminal_prefix_documents": 0,
                }
            )
            reference_quotas.append(
                {
                    "split": split,
                    "category": domain,
                    "unit": "pre_packing_starcoder2_content_tokens",
                    "reference_target_tokens": content_tokens,
                    "observed_tokens": content_tokens,
                    "shortfall_tokens": 0,
                    "surplus_tokens": 0,
                    "selection_authority": False,
                }
            )
            selected_documents += documents
    manifest: dict[str, object] = {
        "identity": {"format_version": 7},
        "production_ready": True,
        "selection_strategy": "all_eligible_canonical_documents",
        "decision_format": "all-eligible-keep-bitmap",
        "decision_format_version": 1,
        "selected_totals": selected_totals,
        "reference_quotas": reference_quotas,
        "documents": {
            "input": selected_documents,
            "accepted_canonical_before_selection": selected_documents,
            "selected": selected_documents,
            "quota_overflow": 0,
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (root / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    return manifest


def _all_rows(value: int) -> dict[tuple[str, str], int]:
    return {(split, domain): value for split in SPLITS for domain in DOMAIN_ORDER}


class SelectionSupplyQualificationTests(unittest.TestCase):
    def test_default_targets_have_frozen_largest_remainder_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            available = _all_rows(20_000_000)
            _write_selection(selection, available, sequence_length=4_096)

            report = qualify_selection_supply(selection)

        self.assertEqual(report["status"], "pass")
        by_split = {row["split"]: row for row in report["splits"]}
        self.assertEqual(by_split["train"]["selectable_rows_cap"], 12_836_914)
        self.assertEqual(
            by_split["train"]["required_rows_per_domain"],
            {"python": 5_134_766, "other_code": 5_134_765, "english": 2_567_383},
        )
        self.assertEqual(by_split["train"]["target_rounding_shortfall_tokens"], 256)
        for split in ("validation", "test"):
            self.assertEqual(by_split[split]["selectable_rows_cap"], 122_070)
            self.assertEqual(
                by_split[split]["required_rows_per_domain"],
                {"python": 48_828, "other_code": 48_828, "english": 24_414},
            )
            self.assertEqual(
                by_split[split]["target_rounding_shortfall_tokens"], 1_280
            )
        self.assertEqual(
            report["mixture"]["allocation"],
            "largest_remainder_stable_domain_order",
        )

    def test_exact_capacity_passes_and_one_row_shortfall_fails(self) -> None:
        targets = {split: 80 for split in SPLITS}
        requirements = {"python": 4, "other_code": 4, "english": 2}
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            available = {
                (split, domain): requirements[domain]
                for split in SPLITS
                for domain in DOMAIN_ORDER
            }
            available[("validation", "english")] -= 1
            _write_selection(selection, available, sequence_length=8)

            report = qualify_selection_supply(
                selection, sequence_length=8, targets=targets
            )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(
            report["failures"],
            [
                {
                    "split": "validation",
                    "domain": "english",
                    "available_rows": 1,
                    "required_rows": 2,
                    "shortfall_rows": 1,
                    "shortfall_input_tokens": 8,
                }
            ],
        )

    def test_cli_returns_one_and_emits_machine_readable_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            _write_selection(selection, _all_rows(1), sequence_length=8)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--selection-root",
                    str(selection),
                    "--sequence-length",
                    "8",
                    "--expected-train-input-tokens",
                    "80",
                    "--expected-validation-input-tokens",
                    "80",
                    "--expected-test-input-tokens",
                    "80",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertTrue(payload["failures"])

    def test_manifest_mutation_after_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            _write_selection(selection, _all_rows(4), sequence_length=8)
            manifest = json.loads((selection / "manifest.json").read_text())
            manifest["selected_totals"][0]["selected_tokens"] += 1
            (selection / "manifest.json").write_text(json.dumps(manifest))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--selection-root",
                        str(selection),
                        "--sequence-length",
                        "8",
                        "--expected-train-input-tokens",
                        "80",
                        "--expected-validation-input-tokens",
                        "80",
                        "--expected-test-input-tokens",
                        "80",
                    ]
                )
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    def test_nonproduction_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            manifest = _write_selection(
                selection, _all_rows(4), sequence_length=8
            )
            manifest["production_ready"] = False
            raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
            (selection / "manifest.json").write_bytes(raw)
            (selection / "manifest.sha256").write_text(
                f"{hashlib.sha256(raw).hexdigest()}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                RawTokenCacheInventoryError,
                "production v7 all-eligible selection",
            ):
                qualify_selection_supply(
                    selection,
                    sequence_length=8,
                    targets={split: 80 for split in SPLITS},
                )

    def test_strict_selected_total_validator_rejects_terminal_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            manifest = _write_selection(
                selection, _all_rows(4), sequence_length=8
            )
            manifest["selected_totals"][0]["terminal_prefix_documents"] = 1
            raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
            (selection / "manifest.json").write_bytes(raw)
            (selection / "manifest.sha256").write_text(
                f"{hashlib.sha256(raw).hexdigest()}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                MaterializationError, "complete documents"
            ):
                qualify_selection_supply(
                    selection,
                    sequence_length=8,
                    targets={split: 80 for split in SPLITS},
                )

    def test_available_row_formula_matches_actual_no_padding_writer(self) -> None:
        cases = ([7], [3, 4], [8], [8, 8], [1, 1, 1, 1])
        for lengths in cases:
            with (
                self.subTest(lengths=lengths),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary) / "packed"
                writer = PackedShardWriter(
                    output,
                    domain="python",
                    split="train",
                    sequence_length=4,
                    vocab_size=16,
                    eos_token_id=0,
                    tokenizer_manifest_sha256="a" * 64,
                    rows_per_shard=8,
                )
                for length in lengths:
                    writer.add_document([1] * length)
                packed = writer.finish()
                predicted = geometry_independent_available_rows(
                    selected_content_tokens=sum(lengths),
                    documents=len(lengths),
                    sequence_length=4,
                )
                self.assertEqual(predicted, packed["rows"])

    def test_targets_must_cover_all_splits_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection"
            _write_selection(selection, _all_rows(4), sequence_length=8)
            with self.assertRaisesRegex(ValueError, "targets must contain exactly"):
                qualify_selection_supply(
                    selection,
                    sequence_length=8,
                    targets={"train": DEFAULT_TARGETS["train"]},
                )


if __name__ == "__main__":
    unittest.main()
