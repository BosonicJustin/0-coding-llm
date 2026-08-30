"""Prime ``renderers`` protocol adapter for the frozen StarCoder2 format.

The class is intentionally duck-typed: it has no import-time dependency on
Prime's package, while its methods and return fields satisfy the official
``Renderer`` protocol at the pinned integration revision.  This lets the same
source be copied into a pinned ``renderers`` checkout by the installation
script without making the training runtime depend on this repository's path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

if __package__ == "renderers":  # Copied into the pinned upstream checkout.
    # Select the copied canonical source even if an unrelated/stale
    # ``posttrain`` package happens to be importable in the environment.
    from renderers.starcoder2_chat_format import (  # type: ignore[no-redef]
        EOS_TOKEN_ID,
        ChatFormatError,
        ParsedChatResponse,
        RenderedChat,
        StarCoder2CodingChatFormat,
        TokenizerLike,
    )
else:
    # Normal repository import (dataset curation, unit tests, editable use).
    from posttrain.prime.chat_format import (
        EOS_TOKEN_ID,
        ChatFormatError,
        ParsedChatResponse,
        RenderedChat,
        StarCoder2CodingChatFormat,
        TokenizerLike,
    )

try:  # Exact Prime dataclasses when running inside the patched package.
    from renderers.base import (
        ParsedResponse as _PrimeParsedResponse,
        RenderedTokens as _PrimeRenderedTokens,
    )
except ModuleNotFoundError:  # Framework-neutral curation and local tests.
    _PrimeParsedResponse = ParsedChatResponse
    _PrimeRenderedTokens = RenderedChat


def _prime_rendered(rendered: RenderedChat) -> Any:
    """Convert canonical output to Prime's rich attribution type when present."""
    if isinstance(rendered, _PrimeRenderedTokens):
        return rendered
    return _PrimeRenderedTokens(
        token_ids=rendered.token_ids,
        message_indices=rendered.message_indices,
        sampled_mask=rendered.sampled_mask,
        is_content=rendered.is_content,
        message_roles=rendered.message_roles,
        message_tool_names=rendered.message_tool_names,
        multi_modal_data=None,
    )


class StarCoder2CodingRenderer:
    """Typed, deterministic renderer for this experiment's coding model."""

    is_multimodal = False

    def __init__(self, tokenizer: TokenizerLike, config: Any | None = None):
        self._format = StarCoder2CodingChatFormat(tokenizer)
        self._tokenizer = tokenizer
        self.config = config
        # The format has no reasoning channel; preserving all history is the
        # only meaningful and bridge-stable policy.
        self.effective_thinking_retention = "all"

    @staticmethod
    def _reject_tools(tools: Sequence[Mapping[str, Any]] | None) -> None:
        if tools:
            raise ChatFormatError(
                "starcoder2-coding-chat-v1 has no tool-call grammar; tools must be empty"
            )

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> Any:
        self._reject_tools(tools)
        return _prime_rendered(
            self._format.render(
                messages, add_generation_prompt=add_generation_prompt
            )
        )

    def render_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        self._reject_tools(tools)
        return self._format.render_ids(
            messages, add_generation_prompt=add_generation_prompt
        )

    def parse_response(
        self,
        token_ids: Sequence[int],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        self._reject_tools(tools)
        parsed = self._format.parse_response(token_ids)
        if isinstance(parsed, _PrimeParsedResponse):
            return parsed
        return _PrimeParsedResponse(
            content=parsed.content,
            reasoning_content=None,
            tool_calls=[],
        )

    def get_stop_token_ids(self) -> list[int]:
        return [EOS_TOKEN_ID]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any | None:
        """Safely extend a completed turn without re-tokenizing its output.

        This deliberately accepts only one new user message.  Any ambiguous
        history, tool use, assistant extension, or non-canonical prior prompt
        returns ``None`` so Prime falls back to a full re-render.
        """
        self._reject_tools(tools)
        if not previous_prompt_ids or len(new_messages) != 1:
            return None
        message = new_messages[0]
        if not isinstance(message, Mapping) or message.get("role") != "user":
            return None

        generation_prompt = self._format.generation_prompt_ids
        if (
            len(previous_prompt_ids) < len(generation_prompt)
            or previous_prompt_ids[-len(generation_prompt) :] != generation_prompt
        ):
            return None

        # A completion may be naturally stopped (one final EOS) or truncated
        # (no EOS, in which case the canonical close is safe to synthesize).
        # EOS anywhere else is ambiguous and refuses the optimization.
        eos_positions = [
            index
            for index, token_id in enumerate(previous_completion_ids)
            if token_id == EOS_TOKEN_ID
        ]
        if eos_positions and eos_positions != [len(previous_completion_ids) - 1]:
            return None

        previous = list(previous_prompt_ids) + list(previous_completion_ids)
        if not eos_positions:
            previous.append(EOS_TOKEN_ID)

        try:
            extension = self._format.render(
                new_messages, add_generation_prompt=True
            )
        except ChatFormatError:
            return None

        separator = self._format.separator_ids
        token_ids = previous + separator + extension.token_ids
        prior_scaffold = len(previous) + len(separator)
        return _prime_rendered(
            RenderedChat(
                token_ids=token_ids,
                message_indices=[-1] * prior_scaffold + extension.message_indices,
                # The entire bridge becomes the next prompt; none of it is sampled
                # in that next step, including the prior completion.
                sampled_mask=[False] * len(token_ids),
                is_content=[False] * prior_scaffold + extension.is_content,
                message_roles=extension.message_roles,
                message_tool_names=extension.message_tool_names,
            )
        )


__all__ = ["StarCoder2CodingRenderer"]
