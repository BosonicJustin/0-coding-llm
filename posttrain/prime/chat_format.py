"""Frozen StarCoder2-tokenizer chat format for SFT and later post-training.

This module deliberately has no PrimeRL, Transformers, or PyTorch dependency.
It is the one canonical implementation used by the dataset curator for exact
length checks and by the Prime ``renderers`` adapter at training/inference.

Format ``starcoder2-coding-chat-v1``
-----------------------------------

Every role opener is ordinary ASCII text encoded with the existing tokenizer::

    <|system|>\nSYSTEM
    <|user|>\nPROMPT
    <|assistant|>\nANSWER<EOS>

Messages are separated by one independently encoded ``"\n"``.  The
generation prompt is ``<|assistant|>\n`` and is not attributed to a message.
There are no added vocabulary entries.  Token ID 0 is the only turn-close and
generation-stop token.  An assistant body and its EOS token are sampled;
openers and separators are injected scaffold.

Each scaffold/body component is encoded independently.  That segmentation is
part of the frozen format: it makes body attribution exact without depending
on optional character offsets and keeps dataset length accounting identical to
PrimeRL rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

FORMAT_ID = "starcoder2-coding-chat-v1"
EOS_TOKEN_ID = 0

ROLE_OPENERS: dict[str, str] = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}
MESSAGE_SEPARATOR = "\n"


class ChatFormatError(ValueError):
    """Conversation or tokenizer violates the frozen format contract."""


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer surface accepted by the canonical renderer.

    Hugging Face tokenizers return ``list[int]`` from ``encode``.  The lower
    level ``tokenizers.Tokenizer`` returns an object with an ``ids`` member;
    both are accepted so offline curation does not need Transformers.
    """

    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any: ...

    def decode(self, token_ids: Sequence[int], *args: Any, **kwargs: Any) -> str: ...


@dataclass
class RenderedChat:
    """Framework-neutral result with the structural Prime renderer fields."""

    token_ids: list[int] = field(default_factory=list)
    message_indices: list[int] = field(default_factory=list)
    sampled_mask: list[bool] = field(default_factory=list)
    is_content: list[bool] = field(default_factory=list)
    message_roles: list[str] = field(default_factory=list)
    message_tool_names: list[str | None] = field(default_factory=list)
    multi_modal_data: None = None

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        parallel = {
            "message_indices": len(self.message_indices),
            "sampled_mask": len(self.sampled_mask),
            "is_content": len(self.is_content),
        }
        mismatches = {name: size for name, size in parallel.items() if size != length}
        if mismatches:
            raise ChatFormatError(
                f"render metadata must match {length} token IDs; got {mismatches}"
            )
        if len(self.message_tool_names) != len(self.message_roles):
            raise ChatFormatError(
                "message_tool_names must be parallel to message_roles"
            )

    def tokens_per_message(
        self,
        n_messages: int | None = None,
        *,
        sampled_only: bool = False,
    ) -> list[int]:
        """Prime-compatible per-message token counts."""
        count = len(self.message_roles) if n_messages is None else n_messages
        count = min(max(count, 0), len(self.message_roles))
        result = [0] * count
        for offset, message_index in enumerate(self.message_indices):
            if not 0 <= message_index < count:
                continue
            if sampled_only and not self.sampled_mask[offset]:
                continue
            result[message_index] += 1
        return result

    def message_token_spans(self) -> list[tuple[int, int] | None]:
        """Prime-compatible half-open token span for every message."""
        first = [-1] * len(self.message_roles)
        last = [-1] * len(self.message_roles)
        for offset, message_index in enumerate(self.message_indices):
            if not 0 <= message_index < len(self.message_roles):
                continue
            if first[message_index] == -1:
                first[message_index] = offset
            last[message_index] = offset
        return [
            None if start == -1 else (start, end + 1)
            for start, end in zip(first, last)
        ]


@dataclass(frozen=True)
class ParsedChatResponse:
    """Prime-compatible parsed text response (this format has no tool syntax)."""

    content: str
    reasoning_content: None = None
    tool_calls: list[Any] = field(default_factory=list)


def _normalise_ids(encoded: Any, *, component: str) -> list[int]:
    raw_ids = getattr(encoded, "ids", encoded)
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise ChatFormatError(
            f"tokenizer.encode returned an unsupported value for {component}: "
            f"{type(encoded).__name__}"
        )
    ids: list[int] = []
    for offset, value in enumerate(raw_ids):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ChatFormatError(
                f"tokenizer returned non-integer ID at {component}[{offset}]"
            )
        if value < 0:
            raise ChatFormatError(
                f"tokenizer returned negative ID at {component}[{offset}]"
            )
        ids.append(int(value))
    if EOS_TOKEN_ID in ids:
        raise ChatFormatError(
            f"{component} encoded to reserved EOS ID {EOS_TOKEN_ID}; reject or "
            "clean literal tokenizer control tokens before publication"
        )
    return ids


def _encode(tokenizer: TokenizerLike, text: str, *, component: str) -> list[int]:
    if not text:
        return []
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        # ``tokenizers.Tokenizer.encode`` has no ``add_special_tokens`` kwarg
        # on some supported versions.  It does not add model BOS/EOS tokens.
        encoded = tokenizer.encode(text)
    return _normalise_ids(encoded, component=component)


def _decode(tokenizer: TokenizerLike, token_ids: Sequence[int]) -> str:
    variants = (
        {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        {"skip_special_tokens": False},
        {},
    )
    last_type_error: TypeError | None = None
    for kwargs in variants:
        try:
            decoded = tokenizer.decode(list(token_ids), **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
        if not isinstance(decoded, str):
            raise ChatFormatError(
                f"tokenizer.decode returned {type(decoded).__name__}, expected str"
            )
        return decoded
    raise ChatFormatError("tokenizer.decode rejected all supported call forms") from last_type_error


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ChatFormatError("messages must be a non-empty sequence of mappings")
    if not messages:
        raise ChatFormatError("messages must not be empty")

    normalised: list[tuple[str, str]] = []
    saw_conversation = False
    expected_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ChatFormatError(f"messages[{index}] must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_OPENERS:
            raise ChatFormatError(
                f"messages[{index}].role must be one of {sorted(ROLE_OPENERS)}"
            )
        if not isinstance(content, str):
            raise ChatFormatError(
                f"messages[{index}].content must be a plain string"
            )
        if not content:
            raise ChatFormatError(f"messages[{index}].content must not be empty")

        if role == "system":
            if saw_conversation:
                raise ChatFormatError("system messages are allowed only at the beginning")
        else:
            saw_conversation = True
            if role != expected_role:
                raise ChatFormatError(
                    f"messages[{index}] must have role {expected_role!r}; got {role!r}"
                )
            expected_role = "assistant" if role == "user" else "user"
        normalised.append((role, content))

    if not saw_conversation:
        raise ChatFormatError("at least one user message is required")
    return normalised


class StarCoder2CodingChatFormat:
    """Canonical tokenizer-bound implementation of the frozen format."""

    format_id = FORMAT_ID
    eos_token_id = EOS_TOKEN_ID
    is_multimodal = False

    def __init__(self, tokenizer: TokenizerLike):
        if not isinstance(tokenizer, TokenizerLike):
            raise ChatFormatError("tokenizer must provide encode and decode methods")
        declared_eos = getattr(tokenizer, "eos_token_id", EOS_TOKEN_ID)
        if declared_eos is not None and declared_eos != EOS_TOKEN_ID:
            raise ChatFormatError(
                f"tokenizer eos_token_id must be {EOS_TOKEN_ID}; got {declared_eos}"
            )
        self.tokenizer = tokenizer
        self._separator_ids = _encode(
            tokenizer, MESSAGE_SEPARATOR, component="message separator"
        )
        self._opener_ids = {
            role: _encode(tokenizer, text, component=f"{role} opener")
            for role, text in ROLE_OPENERS.items()
        }
        if not self._opener_ids["assistant"]:
            raise ChatFormatError("assistant opener must encode to at least one token")

    @property
    def separator_ids(self) -> list[int]:
        return list(self._separator_ids)

    @property
    def generation_prompt_ids(self) -> list[int]:
        return list(self._opener_ids["assistant"])

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> RenderedChat:
        normalised = _validate_messages(messages)
        if add_generation_prompt and normalised[-1][0] != "user":
            raise ChatFormatError(
                "add_generation_prompt requires a conversation ending in user"
            )

        token_ids: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit(
            ids: Sequence[int],
            message_index: int,
            *,
            is_sampled: bool,
            is_content: bool,
        ) -> None:
            token_ids.extend(ids)
            indices.extend([message_index] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        for index, (role, content) in enumerate(normalised):
            if index:
                emit(
                    self._separator_ids,
                    -1,
                    is_sampled=False,
                    is_content=False,
                )
            emit(
                self._opener_ids[role],
                index,
                is_sampled=False,
                is_content=False,
            )
            body_ids = _encode(
                self.tokenizer,
                content,
                component=f"messages[{index}].content",
            )
            assistant = role == "assistant"
            emit(
                body_ids,
                index,
                is_sampled=assistant,
                is_content=True,
            )
            if assistant:
                emit(
                    [EOS_TOKEN_ID],
                    index,
                    is_sampled=True,
                    # For assistant-attributed tokens Prime defines content
                    # and sampled masks identically; EOS is model-emitted.
                    is_content=True,
                )

        if add_generation_prompt:
            emit(
                self._separator_ids,
                -1,
                is_sampled=False,
                is_content=False,
            )
            emit(
                self._opener_ids["assistant"],
                -1,
                is_sampled=False,
                is_content=False,
            )

        roles = [role for role, _ in normalised]
        return RenderedChat(
            token_ids=token_ids,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=roles,
            message_tool_names=[None] * len(roles),
        )

    def render_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages, add_generation_prompt=add_generation_prompt
        ).token_ids

    def render_training_text(self, prompt: str, completion: str) -> str:
        """Human-readable single-turn template text without terminal EOS.

        This method exists for manifests and inspection.  Do **not** tokenize
        its return value to make a length decision: the frozen format encodes
        scaffold and message bodies in separate BPE calls so attribution is
        exact and training matches inference.  Use :meth:`rendered_length`.
        """
        _validate_messages(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            )
        )
        return (
            ROLE_OPENERS["user"]
            + prompt
            + MESSAGE_SEPARATOR
            + ROLE_OPENERS["assistant"]
            + completion
        )

    def rendered_training_length(self, prompt: str, completion: str) -> int:
        """Exact token count for one user/assistant SFT pair, including EOS."""
        return self.rendered_length(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            )
        )

    def rendered_length(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> int:
        """Exact token count used by SFT filtering and packing."""
        return len(
            self.render_ids(
                messages, add_generation_prompt=add_generation_prompt
            )
        )

    def parse_response(self, token_ids: Sequence[int]) -> ParsedChatResponse:
        ids: list[int] = []
        for index, value in enumerate(token_ids):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ChatFormatError(
                    f"response token_ids[{index}] must be a non-negative integer"
                )
            if value == EOS_TOKEN_ID:
                break
            ids.append(int(value))
        return ParsedChatResponse(content=_decode(self.tokenizer, ids))


# Shorter compatibility spelling used by early local experiments.  The long
# name is canonical and records that this is the frozen coding-chat format.
StarCoder2ChatFormat = StarCoder2CodingChatFormat


__all__ = [
    "ChatFormatError",
    "EOS_TOKEN_ID",
    "FORMAT_ID",
    "MESSAGE_SEPARATOR",
    "ParsedChatResponse",
    "ROLE_OPENERS",
    "RenderedChat",
    "StarCoder2CodingChatFormat",
    "StarCoder2ChatFormat",
    "TokenizerLike",
]
