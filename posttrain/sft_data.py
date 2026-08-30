"""Framework-neutral contracts for curating instruction-tuning examples.

The raw OpenCodeInstruct snapshot is an acquisition artifact, not training
input.  This module contains the deterministic, dependency-light decisions
used to project it into an immutable Hugging Face/PrimeRL ``messages`` corpus.
It deliberately does no file I/O so the decisions can be unit-tested without
PyArrow, Transformers, or PrimeRL.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


POLICY_VERSION = 1
SOURCE_REPO_ID = "nvidia/OpenCodeInstruct"
SOURCE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
SOURCE_LICENSE = "CC-BY-4.0"
CHAT_FORMAT_ID = "starcoder2-coding-chat-v1"
TOKENIZER_REPO_ID = "bigcode/starcoder2-tokenizer"
TOKENIZER_REVISION = "9cfe60e28fd01cc1391ecd2146a34cda7534efeb"
TOKENIZER_VOCAB_SIZE = 49_152
TOKENIZER_EOS_ID = 0
EXPECTED_SOURCE_ROWS = 5_000_000
EXPECTED_SOURCE_SHARDS = 50
EXPECTED_SOURCE_BYTES = 6_861_113_102

REQUIRED_COLUMNS = (
    "id",
    "input",
    "output",
    "domain",
    "generation_algorithm",
    "llm_judgement",
    "unit_tests",
    "tests_execution_status",
    "average_test_score",
)
ALLOWED_DOMAINS = frozenset(("generic", "algorithmic"))
ALLOWED_GENERATION_ALGORITHMS = frozenset(("self-instruct", "evol-instruct"))
HEX_ID_RE = re.compile(r"^[0-9a-f]{32}$")
WHITESPACE_RE = re.compile(r"\s+")

# These signatures are intentionally high precision.  We would rather leave a
# suspicious example for a later manual/entropy audit than delete ordinary code
# containing names such as ``api_key`` or documentation placeholders.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
)


class TextRenderer(Protocol):
    """Minimal renderer surface needed by the curator."""

    format_id: str
    eos_token_id: int

    def rendered_length(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Return the exact segmented-tokenization length, including EOS."""


class TextTokenizer(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SplitPolicy:
    seed: str
    validation_per_million: int = 2_000

    def __post_init__(self) -> None:
        if not self.seed:
            raise ValueError("split seed must not be empty")
        if not 0 < self.validation_per_million < 1_000_000:
            raise ValueError("validation_per_million must be between 1 and 999999")


@dataclass(frozen=True)
class FilterPolicy:
    max_prompt_bytes: int = 1_000_000
    max_completion_bytes: int = 1_000_000
    min_average_test_score: float = 0.0
    # Parsing three large, unused metadata JSON strings across five million
    # rows is deliberately opt-in.  The source Arrow schema still requires
    # those fields to be strings; malformed metadata is retained only because
    # it never enters the published training record.
    require_metadata_json: bool = False
    reject_high_confidence_secrets: bool = True

    def __post_init__(self) -> None:
        if self.max_prompt_bytes <= 0 or self.max_completion_bytes <= 0:
            raise ValueError("text byte limits must be positive")
        if not 0.0 <= self.min_average_test_score <= 1.0:
            raise ValueError("min_average_test_score must be in [0, 1]")


@dataclass(frozen=True)
class CuratedExample:
    example_id: str
    group_id: str
    split: str
    messages: tuple[dict[str, str], dict[str, str]]
    domain: str
    generation_algorithm: str
    average_test_score: float
    rendered_tokens: int
    source_shard: str
    source_row: int

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "group_id": self.group_id,
            "messages": list(self.messages),
            "domain": self.domain,
            "generation_algorithm": self.generation_algorithm,
            "average_test_score": self.average_test_score,
            "rendered_tokens": self.rendered_tokens,
            "source_revision": SOURCE_REVISION,
            "source_shard": self.source_shard,
            "source_row": self.source_row,
        }


@dataclass(frozen=True)
class Decision:
    example: CuratedExample | None
    reason: str | None
    example_id: str
    group_id: str | None
    rendered_tokens: int | None

    def __post_init__(self) -> None:
        if (self.example is None) == (self.reason is None):
            raise ValueError("decision must be accepted or have exactly one rejection reason")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_group(text: str) -> str:
    """Normalize conservatively for leakage-safe exact prompt grouping.

    This is not fuzzy deduplication and does not discard examples.  It only
    ensures byte/case/whitespace variants of the same prompt receive the same
    split.
    """

    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def prompt_group_id(prompt: str) -> str:
    return "prompt-sha256:" + sha256_bytes(normalize_for_group(prompt).encode("utf-8"))


def split_for_group(group_id: str, policy: SplitPolicy) -> str:
    value = hashlib.sha256(f"{policy.seed}\0{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(value[:8], "big") % 1_000_000
    return "validation" if bucket < policy.validation_per_million else "train"


def _string_field(row: Mapping[str, Any], name: str) -> str | None:
    value = row.get(name)
    return value if isinstance(value, str) else None


def _parse_json_list(value: str) -> list[Any] | None:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, list) else None


def _parse_json_object(value: str) -> dict[str, Any] | None:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _encoded_ids(tokenizer: TextTokenizer, text: str) -> Sequence[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TypeError("tokenizer.encode must return a sequence or an object with .ids")
    return encoded


def rendered_token_count(
    prompt: str,
    completion: str,
    *,
    tokenizer: TextTokenizer,
    renderer: TextRenderer,
) -> int:
    if renderer.eos_token_id != TOKENIZER_EOS_ID:
        raise ValueError(
            f"renderer EOS {renderer.eos_token_id} does not match frozen tokenizer EOS {TOKENIZER_EOS_ID}"
        )
    del tokenizer  # the renderer is bound to the exact tokenizer instance
    messages = (
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    )
    # Segmented tokenization prevents a BPE merge across the generation
    # boundary.  Concatenating template text and encoding it once is subtly
    # inconsistent with inference and can be off by one at scaffold/content
    # boundaries.
    return renderer.rendered_length(messages)


def high_confidence_secret_reason(text: str) -> str | None:
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return f"secret:{name}"
    return None


def validate_policy_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "policy_version": POLICY_VERSION,
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "tokenizer_repo_id": TOKENIZER_REPO_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "tokenizer_eos_id": TOKENIZER_EOS_ID,
        "chat_format_id": CHAT_FORMAT_ID,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"policy {field} must equal {expected!r}")
    if payload.get("max_sequence_length") != 4_096:
        raise ValueError("policy max_sequence_length must equal 4096")
    if payload.get("overlength_policy") != "drop-complete-conversation":
        raise ValueError("only drop-complete-conversation is supported")
    denylist = payload.get("benchmark_denylists")
    if not isinstance(denylist, list) or not denylist:
        raise ValueError("at least one frozen benchmark denylist is required")


def decide_row(
    row: Mapping[str, Any],
    *,
    source_shard: str,
    source_row: int,
    tokenizer: TextTokenizer,
    renderer: TextRenderer,
    split_policy: SplitPolicy,
    filter_policy: FilterPolicy,
    max_sequence_length: int,
    contamination_reason: Callable[[str], str | None] | None,
    contaminated_groups: frozenset[str] = frozenset(),
) -> Decision:
    """Validate, decontaminate, length-check, and split one raw row."""

    raw_id = row.get("id")
    example_id = raw_id if isinstance(raw_id, str) else ""

    def reject(reason: str, *, group_id: str | None = None, tokens: int | None = None) -> Decision:
        return Decision(None, reason, example_id, group_id, tokens)

    if not HEX_ID_RE.fullmatch(example_id):
        return reject("schema:id")
    prompt = _string_field(row, "input")
    completion = _string_field(row, "output")
    if prompt is None:
        return reject("schema:input")
    if completion is None:
        return reject("schema:output")
    if not prompt.strip():
        return reject("quality:empty-prompt")
    if not completion.strip():
        return reject("quality:empty-completion")
    if "\x00" in prompt or "\x00" in completion:
        return reject("quality:nul-byte")
    if len(prompt.encode("utf-8")) > filter_policy.max_prompt_bytes:
        return reject("quality:prompt-bytes")
    if len(completion.encode("utf-8")) > filter_policy.max_completion_bytes:
        return reject("quality:completion-bytes")

    domain = _string_field(row, "domain")
    algorithm = _string_field(row, "generation_algorithm")
    if domain not in ALLOWED_DOMAINS:
        return reject("schema:domain")
    if algorithm not in ALLOWED_GENERATION_ALGORITHMS:
        return reject("schema:generation-algorithm")

    try:
        score = float(row.get("average_test_score"))
    except (TypeError, ValueError):
        return reject("schema:average-test-score")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return reject("schema:average-test-score")
    if score < filter_policy.min_average_test_score:
        return reject("quality:test-score")

    judgement: str | None = None
    unit_tests: str | None = None
    statuses: str | None = None
    tests_value: list[Any] | None = None
    # The production two-pass curator has already scanned unit tests for
    # benchmark contamination.  Do not materialize or parse the three large
    # metadata columns again unless a strict JSON audit or a standalone direct
    # contamination callback explicitly needs them.
    if filter_policy.require_metadata_json or contamination_reason is not None:
        judgement = _string_field(row, "llm_judgement")
        unit_tests = _string_field(row, "unit_tests")
        statuses = _string_field(row, "tests_execution_status")
        if judgement is None or unit_tests is None or statuses is None:
            return reject("schema:metadata")
    if filter_policy.require_metadata_json:
        assert judgement is not None and unit_tests is not None and statuses is not None
        if _parse_json_object(judgement) is None:
            return reject("schema:llm-judgement-json")
        statuses_value = _parse_json_list(statuses)
        tests_value = _parse_json_list(unit_tests)
        if tests_value is None:
            return reject("schema:unit-tests-json")
        if statuses_value is None:
            return reject("schema:test-status-json")
        if len(tests_value) != len(statuses_value):
            return reject("schema:test-count-mismatch")

    group_id = prompt_group_id(prompt)
    if group_id in contaminated_groups:
        return reject("benchmark:group-propagated", group_id=group_id)
    if contamination_reason is not None:
        # Standalone row decisions still support a direct benchmark scan.  The
        # production two-pass curator passes ``None`` here: its authenticated
        # first pass already scanned every source row and propagated every hit
        # through the global prompt-group authority.  Avoiding a duplicate
        # JSON parse and fingerprint scan materially reduces the five-million
        # row tokenization pass without weakening the gate.
        assert unit_tests is not None
        if tests_value is None:
            tests_value = _parse_json_list(unit_tests)
        tests_for_scan = (
            "\n".join(str(value) for value in tests_value)
            if tests_value is not None
            else unit_tests
        )
        contamination = contamination_reason(
            "\n\n".join((prompt, completion, tests_for_scan))
        )
        if contamination is not None:
            return reject(f"benchmark:{contamination}", group_id=group_id)

    if filter_policy.reject_high_confidence_secrets:
        secret = high_confidence_secret_reason(prompt + "\n" + completion)
        if secret is not None:
            return reject(secret, group_id=group_id)

    try:
        tokens = rendered_token_count(
            prompt,
            completion,
            tokenizer=tokenizer,
            renderer=renderer,
        )
    except (TypeError, ValueError):
        # Literal tokenizer control tokens (notably <|endoftext|>) and an
        # otherwise unrenderable conversation are data-quality rejections,
        # not reasons to lose the other ~100k rows in the source shard.
        return reject("quality:rendering", group_id=group_id)
    if tokens - 1 > max_sequence_length:
        return reject("length:over-context", group_id=group_id, tokens=tokens)
    # At least one assistant content target plus EOS must remain after shifting.
    if tokens < 2:
        return reject("length:no-target", group_id=group_id, tokens=tokens)

    split = split_for_group(group_id, split_policy)
    example = CuratedExample(
        example_id=example_id,
        group_id=group_id,
        split=split,
        messages=(
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ),
        domain=domain,
        generation_algorithm=algorithm,
        average_test_score=score,
        rendered_tokens=tokens,
        source_shard=source_shard,
        source_row=source_row,
    )
    return Decision(example, None, example_id, group_id, tokens)
