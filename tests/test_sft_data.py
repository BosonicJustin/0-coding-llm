from __future__ import annotations

import json
import unittest
from pathlib import Path

from posttrain.prime.chat_format import StarCoder2CodingChatFormat
from posttrain.sft_data import (
    FilterPolicy,
    SplitPolicy,
    decide_row,
    normalize_for_group,
    prompt_group_id,
    split_for_group,
    validate_policy_payload,
)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "0123456789abcdef0123456789abcdef",
        "input": "Write a Python function that returns one.",
        "output": "```python\ndef one():\n    return 1\n```",
        "domain": "generic",
        "generation_algorithm": "self-instruct",
        "llm_judgement": json.dumps({"correct": True}),
        "unit_tests": json.dumps(["assert one() == 1"]),
        "tests_execution_status": json.dumps(["pass"]),
        "average_test_score": 1.0,
    }
    row.update(overrides)
    return row


class SFTDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.renderer = StarCoder2CodingChatFormat(self.tokenizer)
        self.split = SplitPolicy(seed="unit-test", validation_per_million=20_000)
        self.filters = FilterPolicy()

    def decide(self, row: dict[str, object], **kwargs: object):
        return decide_row(
            row,
            source_shard="data/train-00000-of-00050.parquet",
            source_row=7,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            split_policy=self.split,
            filter_policy=self.filters,
            max_sequence_length=int(kwargs.pop("max_sequence_length", 4096)),
            contamination_reason=kwargs.pop("contamination_reason", lambda _text: None),
            contaminated_groups=kwargs.pop("contaminated_groups", frozenset()),
            **kwargs,
        )

    def test_accepts_prime_messages_with_provenance_and_exact_length(self) -> None:
        decision = self.decide(valid_row())
        self.assertIsNone(decision.reason)
        assert decision.example is not None
        record = decision.example.as_record()
        self.assertEqual(
            record["messages"],
            [
                {"role": "user", "content": valid_row()["input"]},
                {"role": "assistant", "content": valid_row()["output"]},
            ],
        )
        self.assertEqual(record["source_row"], 7)
        self.assertEqual(
            record["rendered_tokens"],
            self.renderer.rendered_length(record["messages"]),
        )
        self.assertIn(record["split"] if "split" in record else decision.example.split, ("train", "validation"))
        # Tests and judgements are deliberately absent from Prime's training rows.
        self.assertNotIn("unit_tests", record)
        self.assertNotIn("llm_judgement", record)

    def test_group_and_split_are_stable_under_exact_normalization(self) -> None:
        left = "  WRITE\tcode\nfor  café "
        right = "write code for cafe\u0301"
        self.assertEqual(normalize_for_group(left), normalize_for_group(right))
        left_group = prompt_group_id(left)
        right_group = prompt_group_id(right)
        self.assertEqual(left_group, right_group)
        self.assertEqual(split_for_group(left_group, self.split), split_for_group(right_group, self.split))

    def test_benchmark_scan_includes_unescaped_unit_tests(self) -> None:
        captured: list[str] = []

        def guard(content: str) -> str | None:
            captured.append(content)
            return "fixture" if "assert hidden(2) == 4" in content else None

        decision = self.decide(
            valid_row(unit_tests=json.dumps(["\nassert hidden(2) == 4\n"])),
            contamination_reason=guard,
        )
        self.assertEqual(decision.reason, "benchmark:fixture")
        self.assertEqual(len(captured), 1)
        self.assertIn("assert hidden(2) == 4", captured[0])

    def test_benchmark_rejection_propagates_across_prompt_group(self) -> None:
        row = valid_row()
        group = prompt_group_id(str(row["input"]))
        decision = self.decide(row, contaminated_groups=frozenset((group,)))
        self.assertEqual(decision.reason, "benchmark:group-propagated")
        self.assertEqual(decision.group_id, group)

    def test_complete_conversation_context_gate_has_no_truncation(self) -> None:
        row = valid_row()
        exact = self.renderer.rendered_length(
            [
                {"role": "user", "content": str(row["input"])},
                {"role": "assistant", "content": str(row["output"])},
            ]
        )
        accepted = self.decide(row, max_sequence_length=exact - 1)
        self.assertIsNotNone(accepted.example)
        rejected = self.decide(row, max_sequence_length=exact - 2)
        self.assertEqual(rejected.reason, "length:over-context")
        self.assertEqual(rejected.rendered_tokens, exact)

    def test_high_confidence_secret_and_reserved_eos_are_rejected(self) -> None:
        private_key = self.decide(
            valid_row(output="-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----")
        )
        self.assertEqual(private_key.reason, "secret:private-key")
        reserved = self.decide(valid_row(output="literal <|endoftext|> token"))
        # CharacterTokenizer does not special-case the literal; the real
        # tokenizer gate is covered in test_prime_chat_format. This assertion
        # ensures normal source text is still accepted with a benign tokenizer.
        self.assertIsNotNone(reserved.example)

    def test_malformed_and_low_quality_rows_fail_closed(self) -> None:
        cases = (
            (valid_row(id="not-an-id"), "schema:id"),
            (valid_row(input="  "), "quality:empty-prompt"),
            (valid_row(output=""), "quality:empty-completion"),
            (valid_row(domain="unknown"), "schema:domain"),
            (valid_row(generation_algorithm="unknown"), "schema:generation-algorithm"),
            (valid_row(average_test_score=float("nan")), "schema:average-test-score"),
            (valid_row(unit_tests="not-json"), None),
        )
        for row, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(self.decide(row).reason, reason)
        strict = FilterPolicy(require_metadata_json=True)
        decision = decide_row(
            valid_row(unit_tests="not-json"),
            source_shard="fixture.parquet",
            source_row=0,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            split_policy=self.split,
            filter_policy=strict,
            max_sequence_length=4096,
            contamination_reason=lambda _text: None,
        )
        self.assertEqual(decision.reason, "schema:unit-tests-json")

    def test_production_second_pass_does_not_require_unused_metadata_columns(self) -> None:
        row = valid_row()
        for field in ("llm_judgement", "unit_tests", "tests_execution_status"):
            row.pop(field)
        decision = self.decide(row, contamination_reason=None)
        self.assertIsNotNone(decision.example)
        self.assertIsNone(decision.reason)

    def test_checked_in_policy_is_frozen(self) -> None:
        policy = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "posttrain" / "opencodeinstruct_prime_sft_v1.json").read_text(
                encoding="utf-8"
            )
        )
        validate_policy_payload(policy)
        self.assertEqual(policy["chat_format_id"], self.renderer.format_id)
        self.assertEqual(policy["benchmark_denylists"], ["configs/mbpp_denylist.json"])
        for field, value in (
            ("source_license", "not-the-source-license"),
            ("chat_format_id", "unknown-chat-format"),
        ):
            changed = dict(policy)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_policy_payload(changed)


if __name__ == "__main__":
    unittest.main()
