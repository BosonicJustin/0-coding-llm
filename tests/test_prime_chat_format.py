from __future__ import annotations

import unittest
from dataclasses import dataclass

from posttrain.prime.chat_format import (
    EOS_TOKEN_ID,
    FORMAT_ID,
    MESSAGE_SEPARATOR,
    ROLE_OPENERS,
    ChatFormatError,
    StarCoder2CodingChatFormat,
)
from posttrain.prime.renderer import StarCoder2CodingRenderer


class CharacterTokenizer:
    """Reversible fake whose ordinary character IDs never overlap EOS 0."""

    name_or_path = "bigcode/starcoder2-tokenizer"
    eos_token_id = EOS_TOKEN_ID
    unk_token_id = None

    def __init__(self) -> None:
        self.encode_kwargs: list[dict[str, object]] = []

    def encode(self, text: str, **kwargs: object) -> list[int]:
        self.encode_kwargs.append(dict(kwargs))
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


@dataclass
class Encoding:
    ids: list[int]


class LowLevelTokenizer(CharacterTokenizer):
    """Models tokenizers.Tokenizer's object-with-ids return surface."""

    eos_token_id = None

    def encode(self, text: str) -> Encoding:  # type: ignore[override]
        return Encoding([ord(character) + 1 for character in text])


class PairMergeTokenizer(CharacterTokenizer):
    """Makes monolithic and segmented BPE counts intentionally differ."""

    def encode(self, text: str, **kwargs: object) -> list[int]:
        self.encode_kwargs.append(dict(kwargs))
        return [index + 1 for index in range((len(text) + 1) // 2)]


class ReservedTokenTokenizer(CharacterTokenizer):
    def encode(self, text: str, **kwargs: object) -> list[int]:
        if "reserved-eos" in text:
            return [EOS_TOKEN_ID]
        return super().encode(text, **kwargs)


class WrongEosTokenizer(CharacterTokenizer):
    eos_token_id = 7


class PrimeChatFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.chat = StarCoder2CodingChatFormat(self.tokenizer)

    def test_frozen_identity_and_ascii_scaffold(self) -> None:
        self.assertEqual(FORMAT_ID, "starcoder2-coding-chat-v1")
        self.assertEqual(self.chat.format_id, FORMAT_ID)
        self.assertEqual(self.chat.eos_token_id, 0)
        for value in [*ROLE_OPENERS.values(), MESSAGE_SEPARATOR]:
            value.encode("ascii")
        self.assertTrue(self.tokenizer.encode_kwargs)
        self.assertTrue(
            all(call == {"add_special_tokens": False} for call in self.tokenizer.encode_kwargs)
        )

    def test_single_turn_ids_attribution_and_masks(self) -> None:
        messages = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ]
        rendered = self.chat.render(messages)

        expected_without_eos = self.chat.render_training_text("prompt", "answer")
        decoded = self.tokenizer.decode(
            [token_id for token_id in rendered.token_ids if token_id != EOS_TOKEN_ID]
        )
        self.assertEqual(decoded, expected_without_eos)
        self.assertEqual(rendered.token_ids[-1], EOS_TOKEN_ID)
        self.assertTrue(rendered.sampled_mask[-1])
        self.assertTrue(rendered.is_content[-1])
        self.assertEqual(rendered.message_indices[-1], 1)
        self.assertEqual(rendered.message_roles, ["user", "assistant"])
        self.assertEqual(rendered.message_tool_names, [None, None])
        spans = rendered.message_token_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            rendered.tokens_per_message(),
            [
                sum(index == 0 for index in rendered.message_indices),
                sum(index == 1 for index in rendered.message_indices),
            ],
        )
        self.assertEqual(
            rendered.tokens_per_message(sampled_only=True),
            [0, len("answer") + 1],
        )

        answer_start = len(rendered.token_ids) - len("answer") - 1
        self.assertTrue(all(rendered.sampled_mask[answer_start:]))
        self.assertTrue(all(rendered.is_content[answer_start:]))
        self.assertFalse(any(rendered.sampled_mask[:answer_start]))

        separator_start = len(ROLE_OPENERS["user"]) + len("prompt")
        self.assertEqual(rendered.message_indices[separator_start], -1)
        self.assertFalse(rendered.is_content[separator_start])

    def test_generation_prompt_is_scaffold_and_unattributed(self) -> None:
        rendered = self.chat.render(
            [{"role": "user", "content": "write code"}],
            add_generation_prompt=True,
        )
        suffix = self.chat.separator_ids + self.chat.generation_prompt_ids
        self.assertEqual(rendered.token_ids[-len(suffix) :], suffix)
        self.assertTrue(all(index == -1 for index in rendered.message_indices[-len(suffix) :]))
        self.assertFalse(any(rendered.sampled_mask))
        self.assertFalse(any(rendered.is_content[-len(suffix) :]))

    def test_multi_turn_render_matches_prompt_completion_bridge(self) -> None:
        renderer = StarCoder2CodingRenderer(self.tokenizer)
        first_prompt = renderer.render_ids(
            [{"role": "user", "content": "q1"}],
            add_generation_prompt=True,
        )
        first_completion = self.tokenizer.encode("a1", add_special_tokens=False) + [0]
        bridge = renderer.bridge_to_next_turn(
            first_prompt,
            first_completion,
            [{"role": "user", "content": "q2"}],
        )
        self.assertIsNotNone(bridge)
        assert bridge is not None

        full = renderer.render(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
            add_generation_prompt=True,
        )
        self.assertEqual(bridge.token_ids, full.token_ids)
        self.assertFalse(any(bridge.sampled_mask))
        self.assertEqual(bridge.message_roles, ["user"])
        prior_length = len(first_prompt) + len(first_completion)
        self.assertEqual(
            bridge.message_indices[:prior_length],
            [-1] * prior_length,
        )

    def test_bridge_synthesizes_only_unambiguous_missing_close(self) -> None:
        renderer = StarCoder2CodingRenderer(self.tokenizer)
        prompt = renderer.render_ids(
            [{"role": "user", "content": "q"}], add_generation_prompt=True
        )
        completion = self.tokenizer.encode("truncated", add_special_tokens=False)
        bridge = renderer.bridge_to_next_turn(
            prompt, completion, [{"role": "user", "content": "next"}]
        )
        self.assertIsNotNone(bridge)
        assert bridge is not None
        self.assertEqual(bridge.token_ids[len(prompt) + len(completion)], EOS_TOKEN_ID)

        self.assertIsNone(
            renderer.bridge_to_next_turn(
                prompt,
                [11, EOS_TOKEN_ID, 12],
                [{"role": "user", "content": "next"}],
            )
        )
        self.assertIsNone(
            renderer.bridge_to_next_turn(
                prompt,
                [11],
                [{"role": "assistant", "content": "not safe"}],
            )
        )
        self.assertIsNone(
            renderer.bridge_to_next_turn(
                prompt[:-1],
                [11],
                [{"role": "user", "content": "next"}],
            )
        )

    def test_parse_response_stops_at_first_eos(self) -> None:
        tokens = self.tokenizer.encode("ok", add_special_tokens=False)
        trailing = self.tokenizer.encode("ignored", add_special_tokens=False)
        parsed = self.chat.parse_response(tokens + [EOS_TOKEN_ID] + trailing)
        self.assertEqual(parsed.content, "ok")
        self.assertIsNone(parsed.reasoning_content)
        self.assertEqual(parsed.tool_calls, [])

    def test_exact_length_api_preserves_segmented_tokenization(self) -> None:
        tokenizer = PairMergeTokenizer()
        chat = StarCoder2CodingChatFormat(tokenizer)
        messages = [
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "def"},
        ]
        exact = chat.rendered_length(messages)
        monolithic = len(
            tokenizer.encode(chat.render_training_text("abc", "def"), add_special_tokens=False)
        ) + 1
        self.assertNotEqual(exact, monolithic)
        self.assertEqual(exact, chat.rendered_training_length("abc", "def"))

    def test_low_level_tokenizer_encoding_object_is_supported(self) -> None:
        chat = StarCoder2CodingChatFormat(LowLevelTokenizer())
        self.assertGreater(
            chat.rendered_training_length("prompt", "answer"),
            1,
        )

    def test_fail_closed_validation(self) -> None:
        invalid_conversations = [
            [],
            [{"role": "assistant", "content": "answer"}],
            [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}],
            [{"role": "user", "content": ""}],
            [{"role": "tool", "content": "value"}],
            [{"role": "user", "content": ["not", "plain", "text"]}],
        ]
        for messages in invalid_conversations:
            with self.subTest(messages=messages), self.assertRaises(ChatFormatError):
                self.chat.render(messages)  # type: ignore[arg-type]
        with self.assertRaises(ChatFormatError):
            self.chat.render(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                add_generation_prompt=True,
            )

    def test_reserved_eos_tools_and_wrong_tokenizer_are_rejected(self) -> None:
        with self.assertRaisesRegex(ChatFormatError, "reserved EOS"):
            StarCoder2CodingChatFormat(ReservedTokenTokenizer()).render(
                [{"role": "user", "content": "reserved-eos"}]
            )
        with self.assertRaisesRegex(ChatFormatError, "eos_token_id"):
            StarCoder2CodingChatFormat(WrongEosTokenizer())
        with self.assertRaisesRegex(ChatFormatError, "no tool-call grammar"):
            StarCoder2CodingRenderer(self.tokenizer).render(
                [{"role": "user", "content": "q"}],
                tools=[{"name": "run"}],
            )


if __name__ == "__main__":
    unittest.main()
