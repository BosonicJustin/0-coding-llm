"""Pretokenized packed data for deterministic PyTorch pre-training.

The hot training path deliberately contains no text parsing, tokenization, tar
decompression, filtering, or database access.  A packed row contains ``T + 1``
uint16 token IDs: the first ``T`` are model inputs and the final token is the
look-ahead target for the last input.  A compact bitset records which tokens
start independent attention segments.

The segment-start bitset, rather than the EOS token ID, is authoritative.  It
lets literal special-token text remain ordinary document content while still
guaranteeing that labels and attention never cross a document boundary.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


FORMAT_VERSION = 2
PACKING_JOURNAL_FORMAT_VERSION = 1
ORDER_FORMAT_VERSION = 4
SAMPLER_STATE_FORMAT_VERSION = 2
IGNORE_INDEX = -100
TOKEN_DTYPE = np.dtype("<u2")
REFERENCE_DTYPE = np.dtype("<u8")
DOMAIN_ORDER = ("python", "other_code", "english")
DOMAIN_BITS = 8
ROW_BITS = 64 - DOMAIN_BITS
MAX_ROW_ID = (1 << ROW_BITS) - 1
STARTS_ENCODING = "numpy-packbits-little"
ORDER_SHUFFLE = "full-uniform-permutation-without-replacement"
ORDER_CAPPED_SHUFFLE = (
    "independent-full-domain-permutations-select-prefix-then-global-uniform-"
    "permutation-without-replacement"
)
ORDER_RNG = "numpy.random.PCG64"
ORDER_CAPPED_SEED_DERIVATION = "sha256(seed-nul-namespace)-first-128-bits-big-endian"
SAMPLER_STATE_FORMAT = "distributed-packed-batch-sampler-state"
PACKING_JOURNAL_FORMAT = "packed-shard-construction-journal"
PACKING_JOURNAL_NAME = ".packing-journal.json"
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_BYTE_POPCOUNT = np.fromiter(
    (value.bit_count() for value in range(256)),
    dtype=np.uint8,
    count=256,
)


def _require_sha256(value: Any, *, field: str, source: str | Path) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(
            f"Invalid {field!r} in {source}: expected a 64-character lowercase hex SHA-256"
        )
    return value


def _optional_sha256(value: Any, *, field: str, source: str | Path) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field=field, source=source)


def _normalize_domain_weights(
    weights: Mapping[str, float],
    *,
    source: str | Path,
) -> dict[str, float]:
    if set(weights) != set(DOMAIN_ORDER):
        raise ValueError(f"Expected weights in {source} must contain exactly {DOMAIN_ORDER}")
    values = {domain: float(weights[domain]) for domain in DOMAIN_ORDER}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"Expected weights in {source} must be finite and non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError(f"Expected weights in {source} must have positive total weight")
    return {domain: values[domain] / total for domain in DOMAIN_ORDER}


def _validate_row_mixture(
    counts: Mapping[str, int],
    normalized_weights: Mapping[str, float],
    *,
    source: str | Path,
) -> None:
    total_rows = sum(counts.values())
    for domain in DOMAIN_ORDER:
        expected = total_rows * normalized_weights[domain]
        if abs(counts[domain] - expected) > 1.0:
            raise ValueError(
                f"Packed row mixture is wrong for {domain} in {source}: found "
                f"{counts[domain]}, expected approximately {expected:.3f}"
            )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on filesystems that support it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_mapping(
    value: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    """Return a JSON-only copy so a cursor cannot mutate after checkpointing."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")

    def validate_keys(item: Any) -> None:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{field} must use string keys at every nesting level")
            for nested in item.values():
                validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate_keys(nested)

    validate_keys(value)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain only finite JSON values") from error
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise ValueError(f"{field} must be a JSON object with string keys")
    return decoded


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def encode_reference(domain_id: int, row_id: int) -> np.uint64:
    if not 0 <= domain_id < (1 << DOMAIN_BITS):
        raise ValueError(f"domain_id is out of range: {domain_id}")
    if not 0 <= row_id <= MAX_ROW_ID:
        raise ValueError(f"row_id is out of range: {row_id}")
    return np.uint64((domain_id << ROW_BITS) | row_id)


def decode_reference(reference: int | np.integer[Any]) -> tuple[int, int]:
    value = int(reference)
    return value >> ROW_BITS, value & MAX_ROW_ID


@dataclass(frozen=True)
class _Shard:
    index: int
    rows: int
    tokens_path: Path
    starts_path: Path
    tokens_sha256: str
    starts_sha256: str


class PackedShardWriter:
    """Stream tokenized documents into immutable, fixed-row binary shards.

    ``add_document`` expects content token IDs without an automatically added
    EOS.  The writer appends exactly one EOS boundary token to every document.
    Full rows are emitted without padding. Tail accounting distinguishes the
    final lookahead-only token from tokens omitted from every stored row.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        domain: str,
        split: str,
        sequence_length: int,
        vocab_size: int,
        eos_token_id: int,
        tokenizer_manifest_sha256: str,
        rows_per_shard: int = 131_072,
        construction_seed: int | None = None,
        curation_policy_sha256: str | None = None,
        selection_manifest_sha256: str | None = None,
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.domain = domain
        self.split = split
        self.sequence_length = int(sequence_length)
        self.tokens_per_row = self.sequence_length + 1
        self.starts_bytes_per_row = math.ceil(self.tokens_per_row / 8)
        self.vocab_size = int(vocab_size)
        self.eos_token_id = int(eos_token_id)
        self.rows_per_shard = int(rows_per_shard)
        self.tokenizer_manifest_sha256 = _require_sha256(
            tokenizer_manifest_sha256,
            field="tokenizer_manifest_sha256",
            source="PackedShardWriter",
        )
        self.construction_seed = (
            None if construction_seed is None else int(construction_seed)
        )
        self.curation_policy_sha256 = _optional_sha256(
            curation_policy_sha256,
            field="curation_policy_sha256",
            source="PackedShardWriter",
        )
        self.selection_manifest_sha256 = _optional_sha256(
            selection_manifest_sha256,
            field="selection_manifest_sha256",
            source="PackedShardWriter",
        )

        if self.domain not in DOMAIN_ORDER:
            raise ValueError(f"Unsupported domain {domain!r}; expected one of {DOMAIN_ORDER}")
        if not self.split:
            raise ValueError("split must not be empty")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if not 1 <= self.vocab_size <= np.iinfo(TOKEN_DTYPE).max + 1:
            raise ValueError("vocab_size must fit in uint16")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be inside the vocabulary")
        if self.rows_per_shard < 1:
            raise ValueError("rows_per_shard must be positive")

        self._tokens: list[int] = []
        self._starts: list[bool] = []
        self._buffer_offset = 0
        self._documents = 0
        self._source_content_tokens = 0
        self._rows = 0
        self._valid_loss_tokens = 0
        self._masked_boundary_labels = 0
        self._shards: list[dict[str, Any]] = []
        self._tokens_handle: Any | None = None
        self._starts_handle: Any | None = None
        self._temporary_tokens: Path | None = None
        self._temporary_starts: Path | None = None
        self._rows_in_open_shard = 0
        self._tokens_digest: Any | None = None
        self._starts_digest: Any | None = None
        self._source_cursor: dict[str, Any] | None = None
        self._finished = False
        self.resumed = False
        self._journal_path = self.output_dir / PACKING_JOURNAL_NAME

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self._remove_stale_journal_temporaries()
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Packed-shard construction is already complete in {self.output_dir}"
                )
            self._resume_finished_manifest(manifest_path)
            return

        entries = list(self.output_dir.iterdir())
        if entries:
            if not resume or not self._journal_path.exists():
                raise FileExistsError(
                    f"Refusing to write into non-empty packed-shard directory {self.output_dir}"
                )
            self._resume_from_journal()
            self.resumed = True
        else:
            self._write_journal(source_cursor=None)

    def __enter__(self) -> "PackedShardWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None and not self._finished:
            self.finish()
        elif exc_type is not None:
            # Preserve the last committed journal state. Bytes written after
            # that state are deliberately left in place and rolled back on
            # resume; deleting the part file could also delete committed rows.
            self._close_temporary_shard(remove=False)

    @property
    def source_cursor(self) -> dict[str, Any] | None:
        """Caller-owned cursor committed by the latest durable checkpoint.

        The cursor must identify the *next* document the caller should feed.
        It is intentionally opaque to the writer so archive/member, JSONL row,
        or database key cursors can all use the same construction protocol.
        """

        if self._source_cursor is None:
            return None
        return _canonical_json_mapping(self._source_cursor, field="source_cursor")

    def _identity(self) -> dict[str, Any]:
        return {
            "output_format_version": FORMAT_VERSION,
            "domain": self.domain,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "rows_per_shard": self.rows_per_shard,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "curation_policy_sha256": self.curation_policy_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "construction_seed": self.construction_seed,
        }

    def _remove_stale_journal_temporaries(self) -> None:
        prefix = f".{PACKING_JOURNAL_NAME}."
        removed = False
        for path in self.output_dir.iterdir():
            if path.name.startswith(prefix):
                path.unlink()
                removed = True
        if removed:
            _fsync_directory(self.output_dir)

    def _resume_finished_manifest(self, manifest_path: Path) -> None:
        manifest, parsed_shards = _parse_packed_manifest(manifest_path)
        expected = self._identity()
        found = {
            "output_format_version": manifest.get("format_version"),
            "domain": manifest.get("domain"),
            "split": manifest.get("split"),
            "sequence_length": manifest.get("sequence_length"),
            "vocab_size": manifest.get("vocab_size"),
            "eos_token_id": manifest.get("eos_token_id"),
            "rows_per_shard": manifest.get("rows_per_shard"),
            "tokenizer_manifest_sha256": manifest.get("tokenizer_manifest_sha256"),
            "curation_policy_sha256": manifest.get("curation_policy_sha256"),
            "selection_manifest_sha256": manifest.get("selection_manifest_sha256"),
            "construction_seed": manifest.get("construction_seed"),
        }
        if found != expected:
            differing = sorted(key for key in expected if found.get(key) != expected[key])
            raise ValueError(
                "Packed construction identity mismatch for completed output: "
                + ", ".join(differing)
            )
        allowed_paths = {manifest_path, self._journal_path}
        for shard, payload in zip(parsed_shards, manifest["shards"], strict=True):
            allowed_paths.update((shard.tokens_path, shard.starts_path))
            for path, kind, width in (
                (shard.tokens_path, "tokens", self.tokens_per_row * TOKEN_DTYPE.itemsize),
                (shard.starts_path, "starts", self.starts_bytes_per_row),
            ):
                expected_bytes = shard.rows * width
                if payload[kind].get("bytes") != expected_bytes:
                    raise ValueError(f"Invalid completed {kind} byte count in {manifest_path}")
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise IOError(f"Missing or truncated completed packing shard: {path}")
        unexpected_paths = [
            path.name for path in self.output_dir.iterdir() if path not in allowed_paths
        ]
        if unexpected_paths:
            raise ValueError(f"Completed packed output contains unknown files: {unexpected_paths}")
        self._shards = list(manifest["shards"])
        self._documents = int(manifest["documents"])
        self._source_content_tokens = int(manifest["source_content_tokens"])
        self._rows = int(manifest["rows"])
        self._valid_loss_tokens = int(manifest["valid_loss_tokens"])
        self._masked_boundary_labels = int(manifest["masked_boundary_labels"])
        cursor = manifest.get("construction_last_source_cursor")
        if cursor is not None:
            cursor = _canonical_json_mapping(cursor, field="construction_last_source_cursor")
        self._source_cursor = cursor
        self._finished = True
        self.resumed = True
        if self._journal_path.exists():
            self._journal_path.unlink()
            _fsync_directory(self.output_dir)

    @staticmethod
    def _state_integer(state: Mapping[str, Any], key: str) -> int:
        value = state.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid non-negative integer {key!r} in packing journal")
        return value

    @staticmethod
    def _hash_prefix(path: Path, byte_count: int) -> tuple[str, Any]:
        digest = hashlib.sha256()
        remaining = byte_count
        with path.open("rb") as handle:
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"Packing part is shorter than its journal state: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest(), digest

    @staticmethod
    def _generated_shard(path: Path) -> tuple[int, str, bool] | None:
        final_match = re.fullmatch(r"shard-(\d{6})\.(tokens|starts)\.bin", path.name)
        if final_match:
            return int(final_match.group(1)), final_match.group(2), True
        part_match = re.fullmatch(
            r"\.shard-(\d{6})\.(tokens|starts)\.bin\.part", path.name
        )
        if part_match:
            return int(part_match.group(1)), part_match.group(2), False
        return None

    def _resume_from_journal(self) -> None:
        journal = _load_json(self._journal_path)
        if journal.get("format") != PACKING_JOURNAL_FORMAT:
            raise ValueError(f"Unsupported packing journal format in {self._journal_path}")
        if journal.get("format_version") != PACKING_JOURNAL_FORMAT_VERSION:
            raise ValueError(f"Unsupported packing journal version in {self._journal_path}")
        found_identity = journal.get("identity")
        if found_identity != self._identity():
            expected = self._identity()
            found = found_identity if isinstance(found_identity, dict) else {}
            differing = sorted(key for key in expected if found.get(key) != expected[key])
            raise ValueError(
                "Packed construction identity mismatch: " + ", ".join(differing)
            )
        state = journal.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state in {self._journal_path}")

        completed_shards = state.get("completed_shards")
        if not isinstance(completed_shards, list):
            raise ValueError(f"Invalid completed_shards in {self._journal_path}")
        self._shards = []
        for expected_index, shard in enumerate(completed_shards):
            if not isinstance(shard, dict) or shard.get("index") != expected_index:
                raise ValueError("Packing journal shard indices must be contiguous")
            rows = shard.get("rows")
            if rows != self.rows_per_shard:
                raise ValueError("Only full completed shards may appear in a packing journal")
            for kind in ("tokens", "starts"):
                payload = shard.get(kind)
                expected_name = f"shard-{expected_index:06d}.{kind}.bin"
                if not isinstance(payload, dict) or payload.get("path") != expected_name:
                    raise ValueError(f"Invalid {kind} shard path in packing journal")
                expected_bytes = (
                    rows * self.tokens_per_row * TOKEN_DTYPE.itemsize
                    if kind == "tokens"
                    else rows * self.starts_bytes_per_row
                )
                if payload.get("bytes") != expected_bytes:
                    raise ValueError(f"Invalid {kind} shard byte count in packing journal")
                expected_hash = _require_sha256(
                    payload.get("sha256"),
                    field=f"completed_shards[{expected_index}].{kind}.sha256",
                    source=self._journal_path,
                )
                path = self.output_dir / expected_name
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise IOError(f"Missing or truncated committed packing shard: {path}")
                if _sha256(path) != expected_hash:
                    raise IOError(f"Committed packing shard checksum mismatch: {path}")
            self._shards.append(shard)

        self._documents = self._state_integer(state, "documents")
        self._source_content_tokens = self._state_integer(state, "source_content_tokens")
        self._rows = self._state_integer(state, "rows")
        self._valid_loss_tokens = self._state_integer(state, "valid_loss_tokens")
        self._masked_boundary_labels = self._state_integer(
            state, "masked_boundary_labels"
        )
        if self._valid_loss_tokens + self._masked_boundary_labels != self._rows * self.sequence_length:
            raise ValueError("Packing journal loss counters are inconsistent")

        carry_tokens = state.get("carry_tokens")
        carry_starts = state.get("carry_starts")
        if not isinstance(carry_tokens, list) or not isinstance(carry_starts, list):
            raise ValueError("Invalid carry buffers in packing journal")
        if len(carry_tokens) != len(carry_starts) or len(carry_tokens) > self.sequence_length:
            raise ValueError("Packing journal carry buffers have inconsistent lengths")
        for token_id in carry_tokens:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError("Packing journal contains an invalid carry token")
        if any(not isinstance(value, bool) for value in carry_starts):
            raise ValueError("Packing journal contains an invalid carry start bit")
        stream_tokens = self._source_content_tokens + self._documents
        if stream_tokens != self._rows * self.sequence_length + len(carry_tokens):
            raise ValueError("Packing journal stream counters do not match its carry")
        self._tokens = list(carry_tokens)
        self._starts = list(carry_starts)
        self._buffer_offset = 0

        cursor = state.get("source_cursor")
        if cursor is not None:
            cursor = _canonical_json_mapping(cursor, field="source_cursor")
        self._source_cursor = cursor

        generated: dict[tuple[int, str, bool], Path] = {}
        allowed_nonshards = {PACKING_JOURNAL_NAME}
        for path in self.output_dir.iterdir():
            parsed = self._generated_shard(path)
            if parsed is None:
                if path.name not in allowed_nonshards:
                    raise FileExistsError(
                        f"Unknown file prevents fail-closed packing resume: {path}"
                    )
                continue
            if parsed in generated:
                raise ValueError(f"Duplicate generated packing path identity: {path}")
            generated[parsed] = path

        for index in range(len(self._shards)):
            for kind in ("tokens", "starts"):
                if (index, kind, True) not in generated:
                    raise IOError(f"Committed packing shard file is missing at index {index}")
                if (index, kind, False) in generated:
                    raise ValueError("Committed packing shard also has a temporary part")

        open_state = state.get("open_shard")
        keep_paths = {
            self.output_dir / shard[kind]["path"]
            for shard in self._shards
            for kind in ("tokens", "starts")
        }
        if open_state is not None:
            if not isinstance(open_state, dict):
                raise ValueError("Invalid open_shard in packing journal")
            open_index = len(self._shards)
            if open_state.get("index") != open_index:
                raise ValueError("Open packing shard index is not contiguous")
            open_rows = open_state.get("rows")
            if (
                not isinstance(open_rows, int)
                or isinstance(open_rows, bool)
                or not 1 <= open_rows < self.rows_per_shard
            ):
                raise ValueError("Invalid open packing shard row count")
            expected_total_rows = len(self._shards) * self.rows_per_shard + open_rows
            if self._rows != expected_total_rows:
                raise ValueError("Open packing shard rows disagree with total row counter")
            self._rows_in_open_shard = open_rows
            digests: dict[str, Any] = {}
            for kind in ("tokens", "starts"):
                payload = open_state.get(kind)
                expected_name = f".shard-{open_index:06d}.{kind}.bin.part"
                if not isinstance(payload, dict) or payload.get("path") != expected_name:
                    raise ValueError(f"Invalid open {kind} path in packing journal")
                expected_bytes = (
                    open_rows * self.tokens_per_row * TOKEN_DTYPE.itemsize
                    if kind == "tokens"
                    else open_rows * self.starts_bytes_per_row
                )
                if payload.get("bytes") != expected_bytes:
                    raise ValueError(f"Invalid open {kind} byte count in packing journal")
                expected_hash = _require_sha256(
                    payload.get("sha256"),
                    field=f"open_shard.{kind}.sha256",
                    source=self._journal_path,
                )
                part_path = self.output_dir / expected_name
                final_path = self.output_dir / f"shard-{open_index:06d}.{kind}.bin"
                candidates = [path for path in (part_path, final_path) if path.exists()]
                if len(candidates) != 1:
                    raise IOError(
                        f"Expected exactly one recoverable open {kind} shard, found {candidates}"
                    )
                candidate = candidates[0]
                if candidate.stat().st_size < expected_bytes:
                    raise IOError(f"Recoverable packing shard is truncated: {candidate}")
                found_hash, digest = self._hash_prefix(candidate, expected_bytes)
                if found_hash != expected_hash:
                    raise IOError(f"Open packing shard checksum mismatch: {candidate}")
                if candidate != part_path:
                    os.replace(candidate, part_path)
                with part_path.open("r+b") as handle:
                    handle.truncate(expected_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                keep_paths.add(part_path)
                digests[kind] = digest
            self._temporary_tokens = self.output_dir / (
                f".shard-{open_index:06d}.tokens.bin.part"
            )
            self._temporary_starts = self.output_dir / (
                f".shard-{open_index:06d}.starts.bin.part"
            )
            self._tokens_handle = self._temporary_tokens.open("ab")
            self._starts_handle = self._temporary_starts.open("ab")
            self._tokens_digest = digests["tokens"]
            self._starts_digest = digests["starts"]
        else:
            if self._rows != len(self._shards) * self.rows_per_shard:
                raise ValueError("Packing journal total rows disagree with completed shards")

        removed = False
        for path in list(self.output_dir.iterdir()):
            if self._generated_shard(path) is not None and path not in keep_paths:
                path.unlink()
                removed = True
        if removed or open_state is not None:
            _fsync_directory(self.output_dir)

    def _journal_payload(
        self,
        source_cursor: dict[str, Any] | None,
    ) -> dict[str, Any]:
        carry_tokens = self._tokens[self._buffer_offset :]
        carry_starts = self._starts[self._buffer_offset :]
        open_shard: dict[str, Any] | None = None
        if self._rows_in_open_shard:
            if (
                self._temporary_tokens is None
                or self._temporary_starts is None
                or self._tokens_digest is None
                or self._starts_digest is None
            ):
                raise AssertionError("Open packing shard state is incomplete")
            open_shard = {
                "index": len(self._shards),
                "rows": self._rows_in_open_shard,
                "tokens": {
                    "path": self._temporary_tokens.name,
                    "bytes": self._rows_in_open_shard
                    * self.tokens_per_row
                    * TOKEN_DTYPE.itemsize,
                    "sha256": self._tokens_digest.copy().hexdigest(),
                },
                "starts": {
                    "path": self._temporary_starts.name,
                    "bytes": self._rows_in_open_shard * self.starts_bytes_per_row,
                    "sha256": self._starts_digest.copy().hexdigest(),
                },
            }
        return {
            "format": PACKING_JOURNAL_FORMAT,
            "format_version": PACKING_JOURNAL_FORMAT_VERSION,
            "identity": self._identity(),
            "state": {
                "source_cursor": source_cursor,
                "documents": self._documents,
                "source_content_tokens": self._source_content_tokens,
                "rows": self._rows,
                "valid_loss_tokens": self._valid_loss_tokens,
                "masked_boundary_labels": self._masked_boundary_labels,
                "carry_tokens": carry_tokens,
                "carry_starts": carry_starts,
                "completed_shards": self._shards,
                "open_shard": open_shard,
            },
        }

    def _sync_open_shard(self) -> None:
        for handle in (self._tokens_handle, self._starts_handle):
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())

    def _write_journal(self, source_cursor: dict[str, Any] | None) -> None:
        self._sync_open_shard()
        _atomic_json(self._journal_path, self._journal_payload(source_cursor))
        self._source_cursor = source_cursor

    def checkpoint(self, source_cursor: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically commit all documents through ``source_cursor``.

        ``source_cursor`` must point to the next source document. The caller
        must checkpoint only after every preceding document has been passed to
        ``add_document``. A resume exposes the exact committed cursor and rolls
        back any later shard bytes left by an interrupted process.
        """

        if self._finished:
            raise RuntimeError("Cannot checkpoint after finish()")
        normalized = _canonical_json_mapping(source_cursor, field="source_cursor")
        self._write_journal(normalized)
        return self.source_cursor or {}

    def add_document(
        self,
        token_ids: Sequence[int] | np.ndarray[Any, Any],
        *,
        source_cursor: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError("Cannot add documents after finish()")
        if isinstance(token_ids, np.ndarray):
            values = token_ids.reshape(-1).tolist()
        else:
            values = list(token_ids)
        if not values:
            raise ValueError("Empty documents are not packable")
        for token_id in values:
            if not isinstance(token_id, (int, np.integer)):
                raise TypeError(f"Token ID must be an integer, got {type(token_id).__name__}")
            if not 0 <= int(token_id) < self.vocab_size:
                raise ValueError(f"Token ID {token_id} is outside [0, {self.vocab_size})")

        self._tokens.extend(int(value) for value in values)
        self._starts.extend([True] + [False] * (len(values) - 1))
        self._tokens.append(self.eos_token_id)
        self._starts.append(False)
        self._documents += 1
        self._source_content_tokens += len(values)
        self._drain_rows()
        if source_cursor is not None:
            self.checkpoint(source_cursor)

    def _open_shard(self) -> None:
        if self._tokens_handle is not None:
            return
        shard_index = len(self._shards)
        token_name = f"shard-{shard_index:06d}.tokens.bin"
        starts_name = f"shard-{shard_index:06d}.starts.bin"
        self._temporary_tokens = self.output_dir / f".{token_name}.part"
        self._temporary_starts = self.output_dir / f".{starts_name}.part"
        self._tokens_handle = self._temporary_tokens.open("xb")
        self._starts_handle = self._temporary_starts.open("xb")
        self._tokens_digest = hashlib.sha256()
        self._starts_digest = hashlib.sha256()
        self._rows_in_open_shard = 0

    def _drain_rows(self) -> None:
        while len(self._tokens) - self._buffer_offset >= self.tokens_per_row:
            self._open_shard()
            start = self._buffer_offset
            end = start + self.tokens_per_row
            row_tokens = np.asarray(self._tokens[start:end], dtype=TOKEN_DTYPE)
            row_starts = np.asarray(self._starts[start:end], dtype=np.bool_)
            # Every physical row is an independent attention container, even
            # when it begins in the middle of a long source document.
            row_starts[0] = True
            packed_starts = np.packbits(row_starts, bitorder="little")
            if packed_starts.size != self.starts_bytes_per_row:
                raise AssertionError("Unexpected packed starts width")

            token_bytes = row_tokens.tobytes(order="C")
            start_bytes = packed_starts.tobytes(order="C")
            self._tokens_handle.write(token_bytes)
            self._starts_handle.write(start_bytes)
            self._tokens_digest.update(token_bytes)
            self._starts_digest.update(start_bytes)
            masked = int(np.count_nonzero(row_starts[1:]))
            self._masked_boundary_labels += masked
            self._valid_loss_tokens += self.sequence_length - masked
            self._rows += 1
            self._rows_in_open_shard += 1
            # Advance by T, retaining the look-ahead token as the first input
            # token of the next row.
            self._buffer_offset += self.sequence_length

            if self._rows_in_open_shard == self.rows_per_shard:
                self._finalize_open_shard()
            if self._buffer_offset >= self.sequence_length * 256:
                self._tokens = self._tokens[self._buffer_offset :]
                self._starts = self._starts[self._buffer_offset :]
                self._buffer_offset = 0

    def _close_temporary_shard(self, *, remove: bool) -> None:
        for handle_name in ("_tokens_handle", "_starts_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        if remove:
            for path in (self._temporary_tokens, self._temporary_starts):
                if path is not None:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

    def _finalize_open_shard(self) -> None:
        if self._tokens_handle is None or self._starts_handle is None:
            return
        if self._rows_in_open_shard < 1:
            raise AssertionError("Cannot finalize an empty shard")
        for handle in (self._tokens_handle, self._starts_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._tokens_handle = None
        self._starts_handle = None

        shard_index = len(self._shards)
        token_name = f"shard-{shard_index:06d}.tokens.bin"
        starts_name = f"shard-{shard_index:06d}.starts.bin"
        token_path = self.output_dir / token_name
        starts_path = self.output_dir / starts_name
        assert self._temporary_tokens is not None and self._temporary_starts is not None
        os.replace(self._temporary_tokens, token_path)
        os.replace(self._temporary_starts, starts_path)
        expected_token_bytes = self._rows_in_open_shard * self.tokens_per_row * TOKEN_DTYPE.itemsize
        expected_start_bytes = self._rows_in_open_shard * self.starts_bytes_per_row
        if token_path.stat().st_size != expected_token_bytes:
            raise IOError(f"Incorrect token shard size for {token_path}")
        if starts_path.stat().st_size != expected_start_bytes:
            raise IOError(f"Incorrect starts shard size for {starts_path}")
        if self._tokens_digest is None or self._starts_digest is None:
            raise AssertionError("Open packing shard digest state is missing")
        self._shards.append(
            {
                "index": shard_index,
                "rows": self._rows_in_open_shard,
                "tokens": {
                    "path": token_name,
                    "bytes": expected_token_bytes,
                    "sha256": self._tokens_digest.hexdigest(),
                },
                "starts": {
                    "path": starts_name,
                    "bytes": expected_start_bytes,
                    "sha256": self._starts_digest.hexdigest(),
                },
            }
        )
        self._rows_in_open_shard = 0
        self._temporary_tokens = None
        self._temporary_starts = None
        self._tokens_digest = None
        self._starts_digest = None
        _fsync_directory(self.output_dir)

    def finish(
        self,
        *,
        source_cursor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._finished:
            return _load_json(self.output_dir / "manifest.json")
        if source_cursor is not None:
            self.checkpoint(source_cursor)
        self._drain_rows()
        self._finalize_open_shard()
        stream_tokens = self._source_content_tokens + self._documents
        unused_as_input_tail_tokens = stream_tokens - self._rows * self.sequence_length
        stored_unique_stream_tokens = (
            self._rows * self.sequence_length + 1 if self._rows else 0
        )
        unstored_tail_tokens = stream_tokens - stored_unique_stream_tokens
        manifest: dict[str, Any] = {
            "format": "packed-document-causal",
            "format_version": FORMAT_VERSION,
            "domain": self.domain,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "tokens_per_row": self.tokens_per_row,
            "token_dtype": TOKEN_DTYPE.str,
            "starts_encoding": STARTS_ENCODING,
            "starts_bytes_per_row": self.starts_bytes_per_row,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "rows_per_shard": self.rows_per_shard,
            "rows": self._rows,
            "input_tokens": self._rows * self.sequence_length,
            "valid_loss_tokens": self._valid_loss_tokens,
            "masked_boundary_labels": self._masked_boundary_labels,
            "documents": self._documents,
            "source_content_tokens": self._source_content_tokens,
            "stream_tokens": stream_tokens,
            "unused_as_input_tail_tokens": unused_as_input_tail_tokens,
            "unstored_tail_tokens": unstored_tail_tokens,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "curation_policy_sha256": self.curation_policy_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "construction_seed": self.construction_seed,
            "construction_last_source_cursor": self._source_cursor,
            "shards": self._shards,
        }
        _atomic_json(self.output_dir / "manifest.json", manifest)
        try:
            self._journal_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(self.output_dir)
        self._finished = True
        return manifest


def _parse_packed_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[_Shard]]:
    manifest = _load_json(manifest_path)
    if manifest.get("format") != "packed-document-causal":
        raise ValueError(f"Unsupported packed format in {manifest_path}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported packed format version in {manifest_path}")
    required_nonnegative = (
        "sequence_length",
        "tokens_per_row",
        "starts_bytes_per_row",
        "vocab_size",
        "rows",
        "input_tokens",
        "valid_loss_tokens",
        "masked_boundary_labels",
        "documents",
        "source_content_tokens",
        "stream_tokens",
        "unused_as_input_tail_tokens",
        "unstored_tail_tokens",
    )
    for key in required_nonnegative:
        if not isinstance(manifest.get(key), int) or manifest[key] < 0:
            raise ValueError(f"Invalid {key!r} in {manifest_path}")
    sequence_length = manifest["sequence_length"]
    if sequence_length < 2 or manifest["tokens_per_row"] != sequence_length + 1:
        raise ValueError(f"Invalid row width in {manifest_path}")
    expected_starts_width = math.ceil((sequence_length + 1) / 8)
    if manifest["starts_bytes_per_row"] != expected_starts_width:
        raise ValueError(f"Invalid starts width in {manifest_path}")
    if manifest.get("token_dtype") != TOKEN_DTYPE.str:
        raise ValueError(f"Unsupported token dtype in {manifest_path}")
    if manifest.get("starts_encoding") != STARTS_ENCODING:
        raise ValueError(
            f"Unsupported segment-start encoding in {manifest_path}: "
            f"expected {STARTS_ENCODING!r}"
        )
    if manifest.get("domain") not in DOMAIN_ORDER:
        raise ValueError(f"Invalid domain in {manifest_path}")
    rows_per_shard = manifest.get("rows_per_shard")
    if (
        not isinstance(rows_per_shard, int)
        or isinstance(rows_per_shard, bool)
        or rows_per_shard < 1
    ):
        raise ValueError(f"Invalid rows_per_shard in {manifest_path}")
    if not 1 <= manifest["vocab_size"] <= np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"Invalid vocabulary size in {manifest_path}")
    eos_token_id = manifest.get("eos_token_id")
    if not isinstance(eos_token_id, int) or not 0 <= eos_token_id < manifest["vocab_size"]:
        raise ValueError(f"Invalid EOS token ID in {manifest_path}")
    if manifest["input_tokens"] != manifest["rows"] * sequence_length:
        raise ValueError(f"Invalid input token count in {manifest_path}")
    if (
        manifest["valid_loss_tokens"] + manifest["masked_boundary_labels"]
        != manifest["input_tokens"]
    ):
        raise ValueError(f"Loss-token counters are inconsistent in {manifest_path}")
    if manifest["stream_tokens"] != manifest["source_content_tokens"] + manifest["documents"]:
        raise ValueError(f"Source/EOS stream counters are inconsistent in {manifest_path}")
    expected_unused_tail = manifest["stream_tokens"] - manifest["input_tokens"]
    if (
        expected_unused_tail < 0
        or expected_unused_tail > sequence_length
        or manifest["unused_as_input_tail_tokens"] != expected_unused_tail
    ):
        raise ValueError(f"Invalid unused input tail count in {manifest_path}")
    stored_unique_stream_tokens = manifest["input_tokens"] + 1 if manifest["rows"] else 0
    expected_unstored_tail = manifest["stream_tokens"] - stored_unique_stream_tokens
    if manifest["unstored_tail_tokens"] != expected_unstored_tail:
        raise ValueError(f"Invalid unstored tail count in {manifest_path}")
    _require_sha256(
        manifest.get("tokenizer_manifest_sha256"),
        field="tokenizer_manifest_sha256",
        source=manifest_path,
    )
    for field in ("curation_policy_sha256", "selection_manifest_sha256"):
        _optional_sha256(manifest.get(field), field=field, source=manifest_path)
    construction_cursor = manifest.get("construction_last_source_cursor")
    if construction_cursor is not None:
        _canonical_json_mapping(
            construction_cursor,
            field="construction_last_source_cursor",
        )

    shard_payloads = manifest.get("shards")
    if not isinstance(shard_payloads, list):
        raise ValueError(f"Invalid shard list in {manifest_path}")
    shards: list[_Shard] = []
    total_rows = 0
    for expected_index, payload in enumerate(shard_payloads):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid shard entry in {manifest_path}")
        if payload.get("index") != expected_index:
            raise ValueError(f"Non-contiguous shard index in {manifest_path}")
        rows = payload.get("rows")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or not 0 < rows <= rows_per_shard
        ):
            raise ValueError(f"Invalid shard row count in {manifest_path}")
        if expected_index < len(shard_payloads) - 1 and rows != rows_per_shard:
            raise ValueError(f"Only the final shard may be partial in {manifest_path}")
        tokens = payload.get("tokens")
        starts = payload.get("starts")
        if not isinstance(tokens, dict) or not isinstance(starts, dict):
            raise ValueError(f"Invalid shard payload metadata in {manifest_path}")
        token_relative = tokens.get("path")
        starts_relative = starts.get("path")
        if not isinstance(token_relative, str) or not token_relative:
            raise ValueError(f"Invalid token shard path in {manifest_path}")
        if not isinstance(starts_relative, str) or not starts_relative:
            raise ValueError(f"Invalid starts shard path in {manifest_path}")
        token_path = manifest_path.parent / token_relative
        starts_path = manifest_path.parent / starts_relative
        shards.append(
            _Shard(
                index=expected_index,
                rows=rows,
                tokens_path=token_path,
                starts_path=starts_path,
                tokens_sha256=_require_sha256(
                    tokens.get("sha256"),
                    field=f"shards[{expected_index}].tokens.sha256",
                    source=manifest_path,
                ),
                starts_sha256=_require_sha256(
                    starts.get("sha256"),
                    field=f"shards[{expected_index}].starts.sha256",
                    source=manifest_path,
                ),
            )
        )
        total_rows += rows
    if total_rows != manifest["rows"]:
        raise ValueError(f"Shard rows do not sum to manifest rows in {manifest_path}")
    return manifest, shards


def validate_packed_manifest(
    manifest_path: str | Path,
    *,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate schema, payload semantics, sizes, and optional checksums.

    The semantic pass is always performed. It streams each mmap in bounded
    chunks and proves that token IDs fit the declared vocabulary, every row
    begins an attention segment, unused start-bit padding is zero, and the
    recorded supervised/masked label counts match the payload.
    """

    path = Path(manifest_path)
    manifest, shards = _parse_packed_manifest(path)
    token_bytes_per_row = manifest["tokens_per_row"] * TOKEN_DTYPE.itemsize
    starts_bytes_per_row = manifest["starts_bytes_per_row"]
    vocab_size = manifest["vocab_size"]
    tokens_per_row = manifest["tokens_per_row"]
    valid_loss_tokens = 0
    masked_boundary_labels = 0
    semantic_chunk_rows = 4_096
    final_byte_bits = tokens_per_row % 8
    unused_high_bits_mask = (
        ((0xFF << final_byte_bits) & 0xFF) if final_byte_bits else 0
    )
    for shard, payload in zip(shards, manifest["shards"], strict=True):
        expected_tokens = shard.rows * token_bytes_per_row
        expected_starts = shard.rows * starts_bytes_per_row
        for file_path, expected, recorded in (
            (shard.tokens_path, expected_tokens, payload["tokens"].get("bytes")),
            (shard.starts_path, expected_starts, payload["starts"].get("bytes")),
        ):
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            actual = file_path.stat().st_size
            if actual != expected or recorded != expected:
                raise IOError(
                    f"Size mismatch for {file_path}: expected {expected}, recorded {recorded}, "
                    f"found {actual}"
                )
        if verify_checksums:
            if _sha256(shard.tokens_path) != shard.tokens_sha256:
                raise IOError(f"Checksum mismatch for {shard.tokens_path}")
            if _sha256(shard.starts_path) != shard.starts_sha256:
                raise IOError(f"Checksum mismatch for {shard.starts_path}")

        tokens_map = np.memmap(
            shard.tokens_path,
            dtype=TOKEN_DTYPE,
            mode="r",
            shape=(shard.rows, tokens_per_row),
        )
        starts_map = np.memmap(
            shard.starts_path,
            dtype=np.uint8,
            mode="r",
            shape=(shard.rows, starts_bytes_per_row),
        )
        try:
            for begin in range(0, shard.rows, semantic_chunk_rows):
                end = min(begin + semantic_chunk_rows, shard.rows)
                tokens = np.asarray(tokens_map[begin:end])
                starts = np.asarray(starts_map[begin:end])
                if np.any(tokens >= vocab_size):
                    local_row, column = np.argwhere(tokens >= vocab_size)[0]
                    token_id = int(tokens[local_row, column])
                    raise ValueError(
                        f"Token ID {token_id} is outside [0, {vocab_size}) in "
                        f"{shard.tokens_path} row {begin + int(local_row)}, column {int(column)}"
                    )
                missing_first_start = (starts[:, 0] & np.uint8(1)) == 0
                if np.any(missing_first_start):
                    local_row = int(np.flatnonzero(missing_first_start)[0])
                    raise ValueError(
                        f"Packed row does not begin an attention segment in "
                        f"{shard.starts_path} row {begin + local_row}"
                    )
                if unused_high_bits_mask:
                    nonzero_padding = (
                        starts[:, -1] & np.uint8(unused_high_bits_mask)
                    ) != 0
                    if np.any(nonzero_padding):
                        local_row = int(np.flatnonzero(nonzero_padding)[0])
                        raise ValueError(
                            f"Unused high start bits are nonzero in {shard.starts_path} "
                            f"row {begin + local_row}"
                        )
                unpacked_starts = np.unpackbits(
                    starts,
                    axis=1,
                    count=tokens_per_row,
                    bitorder="little",
                )
                invalid_boundaries = np.logical_and(
                    unpacked_starts[:, 1:] != 0,
                    tokens[:, :-1] != manifest["eos_token_id"],
                )
                if np.any(invalid_boundaries):
                    local_row, target_column = np.argwhere(invalid_boundaries)[0]
                    target_column = int(target_column) + 1
                    raise ValueError(
                        f"Segment start is not preceded by EOS in {shard.starts_path} "
                        f"row {begin + int(local_row)}, column {target_column}"
                    )
                # Since bit zero is required for every row and unused high
                # bits are zero, all remaining set bits correspond exactly to
                # targets that must be masked at document boundaries.
                set_start_bits = int(_BYTE_POPCOUNT[starts].sum(dtype=np.uint64))
                chunk_rows = end - begin
                chunk_masked = set_start_bits - chunk_rows
                masked_boundary_labels += chunk_masked
                valid_loss_tokens += chunk_rows * manifest["sequence_length"] - chunk_masked
        finally:
            for mapping in (tokens_map, starts_map):
                mmap_handle = getattr(mapping, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()

    for field, found, expected in (
        ("valid_loss_tokens", manifest.get("valid_loss_tokens"), valid_loss_tokens),
        (
            "masked_boundary_labels",
            manifest.get("masked_boundary_labels"),
            masked_boundary_labels,
        ),
    ):
        if found != expected:
            raise ValueError(
                f"Manifest {field} mismatch in {path}: recorded {found!r}, recomputed {expected}"
            )
    return manifest


class PackedShardDataset(Dataset[dict[str, Any]]):
    """Random-access mmap reader for one domain/split packed manifest."""

    def __init__(self, manifest_path: str | Path, *, verify_checksums: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        if verify_checksums:
            validate_packed_manifest(self.manifest_path, verify_checksums=True)
        self.manifest, self._shards = _parse_packed_manifest(self.manifest_path)
        self.sequence_length = self.manifest["sequence_length"]
        self.tokens_per_row = self.manifest["tokens_per_row"]
        self.starts_bytes_per_row = self.manifest["starts_bytes_per_row"]
        self.domain = self.manifest["domain"]
        self.split = self.manifest["split"]
        self._ends: list[int] = []
        total = 0
        for shard in self._shards:
            total += shard.rows
            self._ends.append(total)
        self._maps: dict[int, tuple[np.memmap[Any, Any], np.memmap[Any, Any]]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def close(self) -> None:
        for token_map, starts_map in self._maps.values():
            for mapping in (token_map, starts_map):
                mmap_handle = getattr(mapping, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
        self._maps.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.manifest["rows"]

    def _open_maps(self, shard_index: int) -> tuple[np.memmap[Any, Any], np.memmap[Any, Any]]:
        maps = self._maps.get(shard_index)
        if maps is not None:
            return maps
        shard = self._shards[shard_index]
        tokens = np.memmap(
            shard.tokens_path,
            dtype=TOKEN_DTYPE,
            mode="r",
            shape=(shard.rows, self.tokens_per_row),
        )
        starts = np.memmap(
            shard.starts_path,
            dtype=np.uint8,
            mode="r",
            shape=(shard.rows, self.starts_bytes_per_row),
        )
        self._maps[shard_index] = (tokens, starts)
        return tokens, starts

    def __getitem__(self, row_id: int) -> dict[str, Any]:
        if row_id < 0:
            row_id += len(self)
        if not 0 <= row_id < len(self):
            raise IndexError(row_id)
        shard_index = bisect.bisect_right(self._ends, row_id)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local_row = row_id - shard_start
        tokens_map, starts_map = self._open_maps(shard_index)
        # Copy just one compact row so worker output is writable and independent
        # of the mmap object's lifetime during collation/prefetch.
        return {
            "tokens": np.array(tokens_map[local_row], copy=True),
            "starts": np.array(starts_map[local_row], copy=True),
            "row_id": row_id,
        }


class DomainMixtureDataset(Dataset[dict[str, Any]]):
    """Route encoded global sample references to per-domain mmap datasets."""

    def __init__(
        self,
        manifests: Mapping[str, str | Path],
        *,
        verify_checksums: bool = False,
    ) -> None:
        if set(manifests) != set(DOMAIN_ORDER):
            raise ValueError(f"Expected exactly the domains {DOMAIN_ORDER}, found {tuple(manifests)}")
        self.datasets = [
            PackedShardDataset(manifests[name], verify_checksums=verify_checksums)
            for name in DOMAIN_ORDER
        ]
        first = self.datasets[0].manifest
        for dataset, expected_name in zip(self.datasets, DOMAIN_ORDER, strict=True):
            if dataset.domain != expected_name:
                raise ValueError(
                    f"Manifest for {expected_name!r} declares domain {dataset.domain!r}"
                )
            for key in (
                "sequence_length",
                "tokens_per_row",
                "vocab_size",
                "eos_token_id",
                "split",
                "tokenizer_manifest_sha256",
            ):
                if dataset.manifest.get(key) != first.get(key):
                    raise ValueError(f"All domain manifests must agree on {key}")
        self.sequence_length = first["sequence_length"]
        self.split = first["split"]

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def __getitem__(self, reference: int | np.integer[Any]) -> dict[str, Any]:
        domain_id, row_id = decode_reference(reference)
        if not 0 <= domain_id < len(self.datasets):
            raise IndexError(f"Unknown encoded domain ID {domain_id}")
        row = self.datasets[domain_id][row_id]
        row["domain_id"] = domain_id
        row["sample_reference"] = int(reference)
        return row


class PackedBatchCollator:
    """Build model inputs, shifted labels, positions, and document IDs."""

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = int(sequence_length)
        self.tokens_per_row = self.sequence_length + 1

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        tokens = np.stack([row["tokens"] for row in rows], axis=0)
        starts_bytes = np.stack([row["starts"] for row in rows], axis=0)
        if tokens.shape[1] != self.tokens_per_row:
            raise ValueError(f"Expected {self.tokens_per_row} tokens per row, found {tokens.shape[1]}")
        starts = np.unpackbits(
            starts_bytes,
            axis=1,
            count=self.tokens_per_row,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        if not np.all(starts[:, 0]):
            raise ValueError("Every packed row must begin a new attention segment")

        input_starts = starts[:, : self.sequence_length]
        indices = np.arange(self.sequence_length, dtype=np.int32)[None, :]
        last_start = np.maximum.accumulate(np.where(input_starts, indices, 0), axis=1)
        positions = indices - last_start
        document_ids = np.cumsum(input_starts, axis=1, dtype=np.int32) - 1

        input_ids = tokens[:, : self.sequence_length].astype(np.int64, copy=False)
        labels = tokens[:, 1 : self.tokens_per_row].astype(np.int64, copy=True)
        # If the target token starts a new segment, predicting it would leak a
        # training relationship across documents.  The preceding content token
        # still predicts EOS because EOS belongs to the preceding segment.
        labels[starts[:, 1 : self.tokens_per_row]] = IGNORE_INDEX

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "position_ids": torch.from_numpy(positions.astype(np.int64, copy=False)),
            "document_ids": torch.from_numpy(document_ids.astype(np.int64, copy=False)),
            "domain_ids": torch.tensor([int(row["domain_id"]) for row in rows], dtype=torch.int64),
            "row_ids": torch.tensor([int(row["row_id"]) for row in rows], dtype=torch.int64),
            "sample_references": torch.tensor(
                [int(row["sample_reference"]) for row in rows], dtype=torch.int64
            ),
            "num_loss_tokens": torch.tensor(
                int(np.count_nonzero(labels != IGNORE_INDEX)), dtype=torch.int64
            ),
        }


def _count_supervised_tokens_for_references(
    references: np.ndarray[Any, np.dtype[np.uint64]],
    manifests: Mapping[str, str | Path],
    *,
    sequence_length: int,
) -> dict[str, int]:
    """Count exact non-boundary labels for arbitrary packed-row references.

    Callers choose the smaller selected/surplus or consumed/dropped side when
    possible, then derive its complement from each packed manifest's audited
    total. This keeps exact accounting cheap for the expected EOS surplus and
    complete-update suffix without weakening arbitrary-subset validation.
    """

    counts = {domain: 0 for domain in DOMAIN_ORDER}
    if not len(references):
        return counts
    datasets: dict[str, PackedShardDataset] = {}
    try:
        for reference in references:
            domain_id, row_id = decode_reference(reference)
            if not 0 <= domain_id < len(DOMAIN_ORDER):
                raise ValueError(f"Order contains invalid domain ID {domain_id}")
            domain = DOMAIN_ORDER[domain_id]
            dataset = datasets.get(domain)
            if dataset is None:
                dataset = PackedShardDataset(manifests[domain])
                if dataset.sequence_length != sequence_length:
                    raise ValueError("Packed dataset sequence length changed during order build")
                datasets[domain] = dataset
            starts = dataset[row_id]["starts"]
            if not int(starts[0]) & 1:
                raise ValueError(
                    f"Packed row does not begin an attention segment for {domain} "
                    f"row {row_id}"
                )
            final_byte_bits = (sequence_length + 1) % 8
            if final_byte_bits:
                unused_high_bits_mask = (0xFF << final_byte_bits) & 0xFF
                if int(starts[-1]) & unused_high_bits_mask:
                    raise ValueError(
                        f"Packed row has nonzero unused start bits for {domain} "
                        f"row {row_id}"
                    )
            start_count = int(_BYTE_POPCOUNT[starts].sum(dtype=np.uint64))
            # Bit zero marks the row's first attention segment and does not
            # mask a label. Every other set bit corresponds to one masked
            # target in the T-token label row.
            supervised = sequence_length - (start_count - 1)
            if not 0 <= supervised <= sequence_length:
                raise ValueError(
                    f"Invalid segment-start payload for {domain} row {row_id}"
                )
            counts[domain] += supervised
    finally:
        for dataset in datasets.values():
            dataset.close()
    return counts


def _strict_weighted_row_counts(
    total_rows: int,
    normalized_weights: Mapping[str, float],
) -> dict[str, int]:
    """Allocate integer rows by largest remainder with stable domain ties."""

    if total_rows < 1:
        raise ValueError("A weighted order must contain at least one row")
    fractions = {
        domain: Fraction(str(normalized_weights[domain])) for domain in DOMAIN_ORDER
    }
    weight_total = sum(fractions.values(), start=Fraction(0, 1))
    exact = {
        domain: Fraction(total_rows, 1) * fractions[domain] / weight_total
        for domain in DOMAIN_ORDER
    }
    counts = {domain: exact[domain].numerator // exact[domain].denominator for domain in DOMAIN_ORDER}
    remainder = total_rows - sum(counts.values())
    ranked = sorted(
        DOMAIN_ORDER,
        key=lambda domain: (-(exact[domain] - counts[domain]), DOMAIN_ORDER.index(domain)),
    )
    for domain in ranked[:remainder]:
        counts[domain] += 1
    _validate_row_mixture(counts, normalized_weights, source="weighted row allocation")
    return counts


def _derived_order_rng(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{int(seed)}\0{namespace}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:16], "big")
    return np.random.Generator(np.random.PCG64(derived_seed))


def _encoded_domain_references(domain_id: int, row_ids: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    references = np.asarray(row_ids, dtype=REFERENCE_DTYPE).copy()
    references |= np.uint64(domain_id << ROW_BITS)
    return references


def build_training_order(
    manifests: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    seed: int,
    expected_weights: Mapping[str, float] | None = None,
    expected_total_input_tokens: int | None = None,
    input_token_tolerance: int | None = None,
    frozen_global_microbatch_rows: int | None = None,
    frozen_gradient_accumulation_steps: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic row order, optionally selecting a strict cap.

    With ``expected_total_input_tokens=None`` every packed row is retained,
    preserving the original all-row behavior.  With a token target, the order
    contains the largest strict-weight row subset at or below the target.  A
    frozen training order is additionally quantized to complete optimizer
    updates.  Packed rows outside this selected order remain immutable source
    data and are recorded as *packed surplus*, never as a training tail.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty order directory {output}")
    if set(manifests) != set(DOMAIN_ORDER):
        raise ValueError(f"Expected exactly the domains {DOMAIN_ORDER}")
    if (frozen_global_microbatch_rows is None) != (
        frozen_gradient_accumulation_steps is None
    ):
        raise ValueError(
            "frozen_global_microbatch_rows and "
            "frozen_gradient_accumulation_steps must be supplied together"
        )
    for field, value in (
        ("frozen_global_microbatch_rows", frozen_global_microbatch_rows),
        ("frozen_gradient_accumulation_steps", frozen_gradient_accumulation_steps),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{field} must be a positive integer")

    loaded: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    common: dict[str, Any] | None = None
    for domain in DOMAIN_ORDER:
        path = Path(manifests[domain])
        manifest, _ = _parse_packed_manifest(path)
        if manifest["domain"] != domain:
            raise ValueError(f"{path} declares {manifest['domain']!r}, expected {domain!r}")
        if common is None:
            common = manifest
        else:
            for key in (
                "split",
                "sequence_length",
                "vocab_size",
                "eos_token_id",
                "tokenizer_manifest_sha256",
            ):
                if manifest.get(key) != common.get(key):
                    raise ValueError(f"All manifests must agree on {key}")
        loaded[domain] = manifest
        manifest_hashes[domain] = _sha256(path)
    assert common is not None

    sequence_length = int(common["sequence_length"])
    packed_available_rows_per_domain = {
        domain: int(loaded[domain]["rows"]) for domain in DOMAIN_ORDER
    }
    packed_available_rows = sum(packed_available_rows_per_domain.values())
    if packed_available_rows < 1:
        raise ValueError("Cannot build an empty training order")
    if any(count > MAX_ROW_ID for count in packed_available_rows_per_domain.values()):
        raise ValueError("A packed domain contains too many addressable rows")
    packed_available_input_tokens_per_domain = {
        domain: packed_available_rows_per_domain[domain] * sequence_length
        for domain in DOMAIN_ORDER
    }
    packed_available_input_tokens = sum(
        packed_available_input_tokens_per_domain.values()
    )
    packed_available_supervised_tokens_per_domain = {
        domain: int(loaded[domain]["valid_loss_tokens"]) for domain in DOMAIN_ORDER
    }
    packed_available_supervised_tokens = sum(
        packed_available_supervised_tokens_per_domain.values()
    )

    normalized_weights: dict[str, float] | None = None
    if expected_weights is not None:
        normalized_weights = _normalize_domain_weights(
            expected_weights, source="build_training_order"
        )
    optimizer_update_rows = (
        None
        if frozen_global_microbatch_rows is None
        else frozen_global_microbatch_rows * int(frozen_gradient_accumulation_steps)
    )
    capped = expected_total_input_tokens is not None
    if capped:
        if normalized_weights is None:
            raise ValueError(
                "expected_total_input_tokens requires expected_weights for a strict subset"
            )
        if (
            not isinstance(expected_total_input_tokens, int)
            or isinstance(expected_total_input_tokens, bool)
            or expected_total_input_tokens < 1
        ):
            raise ValueError("expected_total_input_tokens must be a positive integer")
        maximum_rows = expected_total_input_tokens // sequence_length
        selected_total_rows = (
            maximum_rows
            if optimizer_update_rows is None
            else (maximum_rows // optimizer_update_rows) * optimizer_update_rows
        )
        if selected_total_rows < 1:
            raise ValueError(
                "Input-token target is smaller than one selectable row/update"
            )
        selected_rows_per_domain = _strict_weighted_row_counts(
            selected_total_rows, normalized_weights
        )
        insufficient = {
            domain: {
                "required": selected_rows_per_domain[domain],
                "available": packed_available_rows_per_domain[domain],
            }
            for domain in DOMAIN_ORDER
            if selected_rows_per_domain[domain] > packed_available_rows_per_domain[domain]
        }
        if insufficient:
            raise ValueError(f"Insufficient packed rows for strict weighted cap: {insufficient}")
        if input_token_tolerance is None:
            input_token_tolerance = (
                sequence_length - 1
                if optimizer_update_rows is None
                else optimizer_update_rows * sequence_length - 1
            )
        if not isinstance(input_token_tolerance, int) or input_token_tolerance < 0:
            raise ValueError("input_token_tolerance must be a non-negative integer")
        input_token_delta = (
            selected_total_rows * sequence_length - expected_total_input_tokens
        )
        if input_token_delta > 0 or input_token_delta < -input_token_tolerance:
            raise ValueError(
                "Packed input-token budget is wrong: found "
                f"{selected_total_rows * sequence_length}, expected a value at or below "
                f"{expected_total_input_tokens} and no more than "
                f"{input_token_tolerance} tokens under it"
            )
    else:
        if input_token_tolerance is not None:
            raise ValueError("input_token_tolerance requires expected_total_input_tokens")
        selected_rows_per_domain = dict(packed_available_rows_per_domain)
        selected_total_rows = packed_available_rows
        input_token_delta = None
        if normalized_weights is not None:
            _validate_row_mixture(
                selected_rows_per_domain,
                normalized_weights,
                source="build_training_order",
            )

    selected_chunks: list[np.ndarray[Any, Any]] = []
    surplus_chunks: list[np.ndarray[Any, Any]] = []
    if capped:
        for domain_id, domain in enumerate(DOMAIN_ORDER):
            row_ids = np.arange(
                packed_available_rows_per_domain[domain], dtype=REFERENCE_DTYPE
            )
            _derived_order_rng(int(seed), f"domain:{domain}").shuffle(row_ids)
            selected_count = selected_rows_per_domain[domain]
            selected_chunks.append(
                _encoded_domain_references(domain_id, row_ids[:selected_count])
            )
            surplus_chunks.append(
                _encoded_domain_references(domain_id, row_ids[selected_count:])
            )
        references = np.concatenate(selected_chunks)
        _derived_order_rng(int(seed), "global:selected").shuffle(references)
        surplus_references = np.concatenate(surplus_chunks)
        shuffle_policy = ORDER_CAPPED_SHUFFLE
        seed_derivation: str | None = ORDER_CAPPED_SEED_DERIVATION
    else:
        references = np.empty(selected_total_rows, dtype=REFERENCE_DTYPE)
        offset = 0
        for domain_id, domain in enumerate(DOMAIN_ORDER):
            count = selected_rows_per_domain[domain]
            references[offset : offset + count] = _encoded_domain_references(
                domain_id, np.arange(count, dtype=REFERENCE_DTYPE)
            )
            offset += count
        np.random.Generator(np.random.PCG64(int(seed))).shuffle(references)
        surplus_references = np.empty(0, dtype=REFERENCE_DTYPE)
        shuffle_policy = ORDER_SHUFFLE
        seed_derivation = None

    packed_surplus_rows_per_domain = {
        domain: packed_available_rows_per_domain[domain] - selected_rows_per_domain[domain]
        for domain in DOMAIN_ORDER
    }
    packed_surplus_rows = sum(packed_surplus_rows_per_domain.values())
    packed_surplus_input_tokens_per_domain = {
        domain: packed_surplus_rows_per_domain[domain] * sequence_length
        for domain in DOMAIN_ORDER
    }
    packed_surplus_input_tokens = sum(packed_surplus_input_tokens_per_domain.values())
    if not capped:
        selected_supervised_tokens_per_domain = dict(
            packed_available_supervised_tokens_per_domain
        )
    elif packed_surplus_rows <= selected_total_rows:
        packed_surplus_supervised_tokens_per_domain = (
            _count_supervised_tokens_for_references(
                surplus_references, manifests, sequence_length=sequence_length
            )
        )
        selected_supervised_tokens_per_domain = {
            domain: packed_available_supervised_tokens_per_domain[domain]
            - packed_surplus_supervised_tokens_per_domain[domain]
            for domain in DOMAIN_ORDER
        }
    else:
        selected_supervised_tokens_per_domain = _count_supervised_tokens_for_references(
            references, manifests, sequence_length=sequence_length
        )
    packed_surplus_supervised_tokens_per_domain = {
        domain: packed_available_supervised_tokens_per_domain[domain]
        - selected_supervised_tokens_per_domain[domain]
        for domain in DOMAIN_ORDER
    }
    if any(value < 0 for value in packed_surplus_supervised_tokens_per_domain.values()):
        raise ValueError("Selected supervised tokens exceed packed availability")
    packed_surplus_supervised_tokens = sum(
        packed_surplus_supervised_tokens_per_domain.values()
    )
    selected_supervised_tokens = sum(selected_supervised_tokens_per_domain.values())
    selected_input_tokens_per_domain = {
        domain: selected_rows_per_domain[domain] * sequence_length
        for domain in DOMAIN_ORDER
    }
    selected_input_tokens = selected_total_rows * sequence_length
    realized_input_token_weights = {
        domain: selected_rows_per_domain[domain] / selected_total_rows
        for domain in DOMAIN_ORDER
    }
    realized_valid_loss_token_weights = {
        domain: (
            selected_supervised_tokens_per_domain[domain] / selected_supervised_tokens
            if selected_supervised_tokens
            else 0.0
        )
        for domain in DOMAIN_ORDER
    }

    if frozen_global_microbatch_rows is None:
        consumed_rows: int | None = None
        dropped_tail_rows: int | None = None
        available_global_microbatches: int | None = None
        consumed_global_microbatches: int | None = None
        dropped_global_microbatches: int | None = None
        dropped_partial_microbatch_rows: int | None = None
        optimizer_updates: int | None = None
        consumed_rows_per_domain: dict[str, int] | None = None
        consumed_input_tokens_per_domain: dict[str, int] | None = None
        consumed_supervised_tokens_per_domain: dict[str, int] | None = None
        realized_consumed_input_token_weights: dict[str, float] | None = None
        realized_consumed_supervised_token_weights: dict[str, float] | None = None
        dropped_rows_per_domain: dict[str, int] | None = None
        dropped_input_tokens_per_domain: dict[str, int] | None = None
        dropped_supervised_tokens_per_domain: dict[str, int] | None = None
        consumed_supervised_tokens: int | None = None
        dropped_supervised_tokens: int | None = None
        accounted_total_input_tokens = selected_input_tokens
    else:
        assert optimizer_update_rows is not None
        optimizer_updates = selected_total_rows // optimizer_update_rows
        if optimizer_updates < 1:
            raise ValueError("frozen optimizer-update geometry is larger than the order")
        consumed_rows = optimizer_updates * optimizer_update_rows
        dropped_tail_rows = selected_total_rows - consumed_rows
        available_global_microbatches = (
            selected_total_rows // frozen_global_microbatch_rows
        )
        consumed_global_microbatches = (
            optimizer_updates * int(frozen_gradient_accumulation_steps)
        )
        dropped_global_microbatches = (
            available_global_microbatches - consumed_global_microbatches
        )
        dropped_partial_microbatch_rows = (
            selected_total_rows % frozen_global_microbatch_rows
        )
        accounted_total_input_tokens = consumed_rows * sequence_length
        consumed_domain_ids = references[:consumed_rows] >> np.uint64(ROW_BITS)
        consumed_rows_per_domain = {
            domain: int(np.count_nonzero(consumed_domain_ids == domain_id))
            for domain_id, domain in enumerate(DOMAIN_ORDER)
        }
        consumed_input_tokens_per_domain = {
            domain: consumed_rows_per_domain[domain] * sequence_length
            for domain in DOMAIN_ORDER
        }
        dropped_rows_per_domain = {
            domain: selected_rows_per_domain[domain] - consumed_rows_per_domain[domain]
            for domain in DOMAIN_ORDER
        }
        dropped_input_tokens_per_domain = {
            domain: dropped_rows_per_domain[domain] * sequence_length
            for domain in DOMAIN_ORDER
        }
        dropped_supervised_tokens_per_domain = _count_supervised_tokens_for_references(
            references[consumed_rows:], manifests, sequence_length=sequence_length
        )
        consumed_supervised_tokens_per_domain = {
            domain: selected_supervised_tokens_per_domain[domain]
            - dropped_supervised_tokens_per_domain[domain]
            for domain in DOMAIN_ORDER
        }
        consumed_supervised_tokens = sum(consumed_supervised_tokens_per_domain.values())
        dropped_supervised_tokens = sum(dropped_supervised_tokens_per_domain.values())
        realized_consumed_input_token_weights = {
            domain: consumed_rows_per_domain[domain] / consumed_rows
            for domain in DOMAIN_ORDER
        }
        realized_consumed_supervised_token_weights = {
            domain: (
                consumed_supervised_tokens_per_domain[domain] / consumed_supervised_tokens
                if consumed_supervised_tokens
                else 0.0
            )
            for domain in DOMAIN_ORDER
        }

    order_name = "order.bin"
    temporary = output / f".{order_name}.part"
    order_path = output / order_name
    with temporary.open("xb") as handle:
        handle.write(references.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, order_path)
    order_manifest = {
        "format": "global-packed-row-order",
        "format_version": ORDER_FORMAT_VERSION,
        "reference_dtype": REFERENCE_DTYPE.str,
        "encoding": {"domain_bits": DOMAIN_BITS, "row_bits": ROW_BITS},
        "domain_order": list(DOMAIN_ORDER),
        "rows_per_domain": selected_rows_per_domain,
        "rows": selected_total_rows,
        "packed_available_rows": packed_available_rows,
        "packed_available_rows_per_domain": packed_available_rows_per_domain,
        "packed_available_input_tokens": packed_available_input_tokens,
        "packed_available_input_tokens_per_domain": (
            packed_available_input_tokens_per_domain
        ),
        "packed_available_supervised_tokens": packed_available_supervised_tokens,
        "packed_available_supervised_tokens_per_domain": (
            packed_available_supervised_tokens_per_domain
        ),
        "packed_surplus_rows": packed_surplus_rows,
        "packed_surplus_rows_per_domain": packed_surplus_rows_per_domain,
        "packed_surplus_input_tokens": packed_surplus_input_tokens,
        "packed_surplus_input_tokens_per_domain": packed_surplus_input_tokens_per_domain,
        "packed_surplus_supervised_tokens": packed_surplus_supervised_tokens,
        "packed_surplus_supervised_tokens_per_domain": (
            packed_surplus_supervised_tokens_per_domain
        ),
        "split": common["split"],
        "sequence_length": sequence_length,
        "vocab_size": common["vocab_size"],
        "eos_token_id": common["eos_token_id"],
        "seed": int(seed),
        "rng": ORDER_RNG,
        "seed_derivation": seed_derivation,
        "numpy_version": np.__version__,
        "shuffle": shuffle_policy,
        "row_selection": (
            "strict-weighted-token-cap" if capped else "all-packed-rows"
        ),
        "tokenizer_manifest_sha256": common["tokenizer_manifest_sha256"],
        "expected_input_token_weights": normalized_weights,
        "input_tokens_per_domain": selected_input_tokens_per_domain,
        "realized_input_token_weights": realized_input_token_weights,
        "valid_loss_tokens_per_domain": selected_supervised_tokens_per_domain,
        "realized_valid_loss_token_weights": realized_valid_loss_token_weights,
        "training_consumption": {
            "policy": (
                "runtime-training-geometry-not-frozen"
                if frozen_global_microbatch_rows is None
                else "complete-frozen-optimizer-updates-no-padding"
            ),
            "frozen_global_microbatch_rows": frozen_global_microbatch_rows,
            "frozen_gradient_accumulation_steps": frozen_gradient_accumulation_steps,
            "frozen_optimizer_update_rows": optimizer_update_rows,
            "order_rows": selected_total_rows,
            "consumed_rows": consumed_rows,
            "dropped_tail_rows": dropped_tail_rows,
            "available_global_microbatches": available_global_microbatches,
            "consumed_global_microbatches": consumed_global_microbatches,
            "dropped_global_microbatches": dropped_global_microbatches,
            "dropped_partial_microbatch_rows": dropped_partial_microbatch_rows,
            "optimizer_updates": optimizer_updates,
            "available_input_tokens": selected_input_tokens,
            "available_supervised_tokens": selected_supervised_tokens,
            "consumed_input_tokens": (
                None if consumed_rows is None else accounted_total_input_tokens
            ),
            "dropped_input_tokens": (
                None if dropped_tail_rows is None else dropped_tail_rows * sequence_length
            ),
            "consumed_supervised_tokens": consumed_supervised_tokens,
            "dropped_supervised_tokens": dropped_supervised_tokens,
            "consumed_rows_per_domain": consumed_rows_per_domain,
            "consumed_input_tokens_per_domain": consumed_input_tokens_per_domain,
            "consumed_supervised_tokens_per_domain": consumed_supervised_tokens_per_domain,
            "realized_consumed_input_token_weights": realized_consumed_input_token_weights,
            "realized_consumed_supervised_token_weights": (
                realized_consumed_supervised_token_weights
            ),
            "dropped_rows_per_domain": dropped_rows_per_domain,
            "dropped_input_tokens_per_domain": dropped_input_tokens_per_domain,
            "dropped_supervised_tokens_per_domain": dropped_supervised_tokens_per_domain,
        },
        "input_token_budget": {
            "policy": "packed_model_inputs_including_eos_excluding_lookahead_duplicates",
            "direction": "at_or_below",
            "expected_total": expected_total_input_tokens,
            "actual_total": accounted_total_input_tokens,
            "available_total": selected_input_tokens,
            "packed_available_total": packed_available_input_tokens,
            "packed_surplus_total": packed_surplus_input_tokens,
            "accounting": (
                "available-order-inputs"
                if frozen_global_microbatch_rows is None
                else "consumed-complete-optimizer-update-inputs"
            ),
            "delta": input_token_delta,
            "tolerance": input_token_tolerance,
        },
        "dataset_manifests": {
            domain: {
                "path": os.path.relpath(Path(manifests[domain]), output),
                "sha256": manifest_hashes[domain],
            }
            for domain in DOMAIN_ORDER
        },
        "order": {
            "path": order_name,
            "bytes": order_path.stat().st_size,
            "sha256": _sha256(order_path),
        },
    }
    _atomic_json(output / "manifest.json", order_manifest)
    return order_manifest


def _load_order_manifest(
    manifest_path: Path,
    *,
    verify_checksum: bool,
) -> tuple[dict[str, Any], Path]:
    manifest = _load_json(manifest_path)
    if manifest.get("format") != "global-packed-row-order":
        raise ValueError(f"Unsupported order format in {manifest_path}")
    found_version = manifest.get("format_version")
    if found_version != ORDER_FORMAT_VERSION:
        migration = (
            "; order versions before v4 do not freeze optimizer-update geometry "
            "and must be rebuilt"
            if found_version in (2, 3)
            else ""
        )
        raise ValueError(
            f"Unsupported order format version {found_version!r} in {manifest_path}; "
            f"expected {ORDER_FORMAT_VERSION}{migration}"
        )
    if manifest.get("domain_order") != list(DOMAIN_ORDER):
        raise ValueError(f"Order domain mapping differs from {DOMAIN_ORDER}")
    if manifest.get("reference_dtype") != REFERENCE_DTYPE.str:
        raise ValueError("Unsupported order reference dtype")
    if manifest.get("encoding") != {"domain_bits": DOMAIN_BITS, "row_bits": ROW_BITS}:
        raise ValueError(
            "Unsupported order reference bit allocation; expected "
            f"domain_bits={DOMAIN_BITS}, row_bits={ROW_BITS}"
        )
    _require_sha256(
        manifest.get("tokenizer_manifest_sha256"),
        field="tokenizer_manifest_sha256",
        source=manifest_path,
    )
    rows = manifest.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("Invalid order row count")
    if not isinstance(manifest.get("split"), str) or not manifest["split"]:
        raise ValueError("Invalid order split")
    if not isinstance(manifest.get("sequence_length"), int) or manifest["sequence_length"] < 2:
        raise ValueError("Invalid order sequence length")
    vocab_size = manifest.get("vocab_size")
    if not isinstance(vocab_size, int) or not 1 <= vocab_size <= np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError("Invalid order vocabulary size")
    eos_token_id = manifest.get("eos_token_id")
    if not isinstance(eos_token_id, int) or not 0 <= eos_token_id < vocab_size:
        raise ValueError("Invalid order EOS token ID")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Invalid order construction seed")
    if manifest.get("rng") != ORDER_RNG:
        raise ValueError(f"Unsupported order RNG; expected {ORDER_RNG}")
    row_selection = manifest.get("row_selection")
    if row_selection == "all-packed-rows":
        if manifest.get("shuffle") != ORDER_SHUFFLE or manifest.get("seed_derivation") is not None:
            raise ValueError("All-row order shuffle identity is invalid")
    elif row_selection == "strict-weighted-token-cap":
        if (
            manifest.get("shuffle") != ORDER_CAPPED_SHUFFLE
            or manifest.get("seed_derivation") != ORDER_CAPPED_SEED_DERIVATION
        ):
            raise ValueError("Capped order shuffle identity is invalid")
    else:
        raise ValueError("Order row-selection policy is invalid")
    if not isinstance(manifest.get("numpy_version"), str) or not manifest["numpy_version"]:
        raise ValueError("Invalid order NumPy version")

    expected_counts = manifest.get("rows_per_domain")
    if not isinstance(expected_counts, dict) or set(expected_counts) != set(DOMAIN_ORDER):
        raise ValueError("Order manifest has an invalid rows_per_domain mapping")
    if any(
        not isinstance(expected_counts[domain], int)
        or isinstance(expected_counts[domain], bool)
        or expected_counts[domain] < 0
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Order manifest has invalid per-domain row counts")
    if sum(expected_counts.values()) != rows:
        raise ValueError("Order per-domain row counts do not sum to total rows")

    packed_available_counts = manifest.get("packed_available_rows_per_domain")
    packed_surplus_counts = manifest.get("packed_surplus_rows_per_domain")
    for field, value in (
        ("packed_available_rows_per_domain", packed_available_counts),
        ("packed_surplus_rows_per_domain", packed_surplus_counts),
    ):
        if (
            not isinstance(value, dict)
            or set(value) != set(DOMAIN_ORDER)
            or any(
                not isinstance(value[domain], int)
                or isinstance(value[domain], bool)
                or value[domain] < 0
                for domain in DOMAIN_ORDER
            )
        ):
            raise ValueError(f"Order {field} mapping is invalid")
    assert isinstance(packed_available_counts, dict)
    assert isinstance(packed_surplus_counts, dict)
    if any(
        packed_available_counts[domain]
        != expected_counts[domain] + packed_surplus_counts[domain]
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Packed selected/available/surplus row counts are inconsistent")
    packed_available_rows = sum(packed_available_counts.values())
    packed_surplus_rows = sum(packed_surplus_counts.values())
    if (
        manifest.get("packed_available_rows") != packed_available_rows
        or manifest.get("packed_surplus_rows") != packed_surplus_rows
        or packed_available_rows != rows + packed_surplus_rows
    ):
        raise ValueError("Packed global row counts are inconsistent")
    sequence_length = manifest["sequence_length"]
    expected_available_input_per_domain = {
        domain: packed_available_counts[domain] * sequence_length
        for domain in DOMAIN_ORDER
    }
    expected_surplus_input_per_domain = {
        domain: packed_surplus_counts[domain] * sequence_length
        for domain in DOMAIN_ORDER
    }
    if (
        manifest.get("packed_available_input_tokens_per_domain")
        != expected_available_input_per_domain
        or manifest.get("packed_surplus_input_tokens_per_domain")
        != expected_surplus_input_per_domain
        or manifest.get("packed_available_input_tokens")
        != sum(expected_available_input_per_domain.values())
        or manifest.get("packed_surplus_input_tokens")
        != sum(expected_surplus_input_per_domain.values())
    ):
        raise ValueError("Packed input-token accounting is inconsistent")

    dataset_manifests = manifest.get("dataset_manifests")
    if not isinstance(dataset_manifests, dict) or set(dataset_manifests) != set(DOMAIN_ORDER):
        raise ValueError("Order manifest has an invalid dataset_manifests mapping")
    for domain in DOMAIN_ORDER:
        payload = dataset_manifests[domain]
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid packed-manifest identity for {domain}")
        relative_path = payload.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Invalid packed-manifest path for {domain}")
        _require_sha256(
            payload.get("sha256"),
            field=f"dataset_manifests.{domain}.sha256",
            source=manifest_path,
        )

    valid_loss_tokens_per_domain = manifest.get("valid_loss_tokens_per_domain")
    if (
        not isinstance(valid_loss_tokens_per_domain, dict)
        or set(valid_loss_tokens_per_domain) != set(DOMAIN_ORDER)
        or any(
            not isinstance(valid_loss_tokens_per_domain[domain], int)
            or isinstance(valid_loss_tokens_per_domain[domain], bool)
            or valid_loss_tokens_per_domain[domain] < 0
            for domain in DOMAIN_ORDER
        )
    ):
        raise ValueError("Order valid-loss-token counts are invalid")
    packed_available_supervised = manifest.get(
        "packed_available_supervised_tokens_per_domain"
    )
    packed_surplus_supervised = manifest.get(
        "packed_surplus_supervised_tokens_per_domain"
    )
    for field, value in (
        ("packed_available_supervised_tokens_per_domain", packed_available_supervised),
        ("packed_surplus_supervised_tokens_per_domain", packed_surplus_supervised),
    ):
        if (
            not isinstance(value, dict)
            or set(value) != set(DOMAIN_ORDER)
            or any(
                not isinstance(value[domain], int)
                or isinstance(value[domain], bool)
                or value[domain] < 0
                for domain in DOMAIN_ORDER
            )
        ):
            raise ValueError(f"Order {field} mapping is invalid")
    assert isinstance(packed_available_supervised, dict)
    assert isinstance(packed_surplus_supervised, dict)
    if any(
        packed_available_supervised[domain]
        != valid_loss_tokens_per_domain[domain] + packed_surplus_supervised[domain]
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Packed supervised-token accounting is inconsistent")
    if (
        manifest.get("packed_available_supervised_tokens")
        != sum(packed_available_supervised.values())
        or manifest.get("packed_surplus_supervised_tokens")
        != sum(packed_surplus_supervised.values())
    ):
        raise ValueError("Packed supervised-token totals are inconsistent")

    consumption = manifest.get("training_consumption")
    if not isinstance(consumption, dict):
        raise ValueError("Order manifest is missing training_consumption")
    if consumption.get("order_rows") != rows:
        raise ValueError("Training-consumption order row count is inconsistent")
    available_input_tokens = rows * manifest["sequence_length"]
    if consumption.get("available_input_tokens") != available_input_tokens:
        raise ValueError("Training-consumption available token count is inconsistent")
    available_supervised_tokens = sum(valid_loss_tokens_per_domain.values())
    if consumption.get("available_supervised_tokens") != available_supervised_tokens:
        raise ValueError("Training-consumption available supervised tokens are inconsistent")
    frozen_global_microbatch_rows = consumption.get(
        "frozen_global_microbatch_rows"
    )
    frozen_gradient_accumulation_steps = consumption.get(
        "frozen_gradient_accumulation_steps"
    )
    frozen_optimizer_update_rows = consumption.get("frozen_optimizer_update_rows")
    if (frozen_global_microbatch_rows is None) != (
        frozen_gradient_accumulation_steps is None
    ):
        raise ValueError("Frozen training geometry is only partially declared")
    if frozen_global_microbatch_rows is None:
        if frozen_optimizer_update_rows is not None:
            raise ValueError("Unfrozen training consumption declares update rows")
        if consumption.get("policy") != "runtime-training-geometry-not-frozen":
            raise ValueError("Unfrozen training-consumption policy is invalid")
        nullable_fields = (
            "consumed_rows",
            "dropped_tail_rows",
            "available_global_microbatches",
            "consumed_global_microbatches",
            "dropped_global_microbatches",
            "dropped_partial_microbatch_rows",
            "optimizer_updates",
            "consumed_input_tokens",
            "dropped_input_tokens",
            "consumed_supervised_tokens",
            "dropped_supervised_tokens",
            "consumed_rows_per_domain",
            "consumed_input_tokens_per_domain",
            "consumed_supervised_tokens_per_domain",
            "realized_consumed_input_token_weights",
            "realized_consumed_supervised_token_weights",
            "dropped_rows_per_domain",
            "dropped_input_tokens_per_domain",
            "dropped_supervised_tokens_per_domain",
        )
        if any(consumption.get(field) is not None for field in nullable_fields):
            raise ValueError("Unfrozen training consumption declares frozen counters")
    else:
        if (
            not isinstance(frozen_global_microbatch_rows, int)
            or isinstance(frozen_global_microbatch_rows, bool)
            or frozen_global_microbatch_rows < 1
            or not isinstance(frozen_gradient_accumulation_steps, int)
            or isinstance(frozen_gradient_accumulation_steps, bool)
            or frozen_gradient_accumulation_steps < 1
        ):
            raise ValueError("Invalid frozen training geometry")
        optimizer_update_rows = (
            frozen_global_microbatch_rows * frozen_gradient_accumulation_steps
        )
        if frozen_optimizer_update_rows != optimizer_update_rows:
            raise ValueError("Frozen optimizer-update row count is inconsistent")
        if consumption.get("policy") != "complete-frozen-optimizer-updates-no-padding":
            raise ValueError("Frozen training-consumption policy is invalid")
        optimizer_updates = rows // optimizer_update_rows
        consumed_rows = optimizer_updates * optimizer_update_rows
        dropped_rows = rows - consumed_rows
        if optimizer_updates < 1:
            raise ValueError("Frozen optimizer update is larger than the order")
        available_global_microbatches = rows // frozen_global_microbatch_rows
        consumed_global_microbatches = (
            optimizer_updates * frozen_gradient_accumulation_steps
        )
        expected_scalars = {
            "available_global_microbatches": available_global_microbatches,
            "consumed_global_microbatches": consumed_global_microbatches,
            "dropped_global_microbatches": (
                available_global_microbatches - consumed_global_microbatches
            ),
            "dropped_partial_microbatch_rows": (
                rows % frozen_global_microbatch_rows
            ),
            "optimizer_updates": optimizer_updates,
            "consumed_rows": consumed_rows,
            "dropped_tail_rows": dropped_rows,
            "consumed_input_tokens": consumed_rows * manifest["sequence_length"],
            "dropped_input_tokens": dropped_rows * manifest["sequence_length"],
        }
        for field, expected in expected_scalars.items():
            if consumption.get(field) != expected:
                raise ValueError(f"Training-consumption {field} is inconsistent")
        consumed_counts = consumption.get("consumed_rows_per_domain")
        consumed_tokens = consumption.get("consumed_input_tokens_per_domain")
        consumed_supervised = consumption.get(
            "consumed_supervised_tokens_per_domain"
        )
        consumed_weights = consumption.get("realized_consumed_input_token_weights")
        consumed_supervised_weights = consumption.get(
            "realized_consumed_supervised_token_weights"
        )
        dropped_counts = consumption.get("dropped_rows_per_domain")
        dropped_tokens = consumption.get("dropped_input_tokens_per_domain")
        dropped_supervised = consumption.get(
            "dropped_supervised_tokens_per_domain"
        )
        for field, value in (
            ("consumed_rows_per_domain", consumed_counts),
            ("consumed_input_tokens_per_domain", consumed_tokens),
            ("consumed_supervised_tokens_per_domain", consumed_supervised),
            ("realized_consumed_input_token_weights", consumed_weights),
            (
                "realized_consumed_supervised_token_weights",
                consumed_supervised_weights,
            ),
            ("dropped_rows_per_domain", dropped_counts),
            ("dropped_input_tokens_per_domain", dropped_tokens),
            ("dropped_supervised_tokens_per_domain", dropped_supervised),
        ):
            if not isinstance(value, dict) or set(value) != set(DOMAIN_ORDER):
                raise ValueError(f"Training-consumption {field} mapping is invalid")
        assert isinstance(consumed_counts, dict)
        assert isinstance(consumed_tokens, dict)
        assert isinstance(consumed_supervised, dict)
        assert isinstance(consumed_weights, dict)
        assert isinstance(consumed_supervised_weights, dict)
        assert isinstance(dropped_counts, dict)
        assert isinstance(dropped_tokens, dict)
        assert isinstance(dropped_supervised, dict)
        total_consumed_supervised = consumption.get("consumed_supervised_tokens")
        total_dropped_supervised = consumption.get("dropped_supervised_tokens")
        if (
            not isinstance(total_consumed_supervised, int)
            or isinstance(total_consumed_supervised, bool)
            or total_consumed_supervised < 0
            or not isinstance(total_dropped_supervised, int)
            or isinstance(total_dropped_supervised, bool)
            or total_dropped_supervised < 0
            or total_consumed_supervised + total_dropped_supervised
            != available_supervised_tokens
        ):
            raise ValueError("Training-consumption supervised-token totals are invalid")
        for domain in DOMAIN_ORDER:
            if (
                not isinstance(consumed_counts[domain], int)
                or isinstance(consumed_counts[domain], bool)
                or not 0 <= consumed_counts[domain] <= expected_counts[domain]
            ):
                raise ValueError("Training-consumption per-domain row count is invalid")
            if consumed_tokens[domain] != (
                consumed_counts[domain] * manifest["sequence_length"]
            ):
                raise ValueError("Training-consumption per-domain token count is invalid")
            if dropped_counts[domain] != expected_counts[domain] - consumed_counts[domain]:
                raise ValueError("Training-consumption dropped per-domain count is invalid")
            if dropped_tokens[domain] != (
                dropped_counts[domain] * manifest["sequence_length"]
            ):
                raise ValueError("Training-consumption dropped token count is invalid")
            available_domain_supervised = manifest["valid_loss_tokens_per_domain"][domain]
            for value in (consumed_supervised[domain], dropped_supervised[domain]):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(
                        "Training-consumption per-domain supervised count is invalid"
                    )
            if (
                consumed_supervised[domain] + dropped_supervised[domain]
                != available_domain_supervised
            ):
                raise ValueError(
                    "Training-consumption per-domain supervised count is inconsistent"
                )
            if not math.isclose(
                float(consumed_weights[domain]),
                consumed_counts[domain] / consumed_rows,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("Training-consumption per-domain weight is invalid")
            expected_supervised_weight = (
                consumed_supervised[domain] / total_consumed_supervised
                if total_consumed_supervised
                else 0.0
            )
            if not math.isclose(
                float(consumed_supervised_weights[domain]),
                expected_supervised_weight,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "Training-consumption per-domain supervised weight is invalid"
                )
        if sum(consumed_counts.values()) != consumed_rows:
            raise ValueError("Training-consumption per-domain rows do not sum")
        if sum(consumed_supervised.values()) != total_consumed_supervised:
            raise ValueError("Training-consumption supervised tokens do not sum")
        if sum(dropped_supervised.values()) != total_dropped_supervised:
            raise ValueError("Training-consumption dropped supervised tokens do not sum")

    budget = manifest.get("input_token_budget")
    if (
        not isinstance(budget, dict)
        or budget.get("policy")
        != "packed_model_inputs_including_eos_excluding_lookahead_duplicates"
        or budget.get("direction") != "at_or_below"
        or budget.get("available_total") != available_input_tokens
        or budget.get("packed_available_total")
        != manifest["packed_available_input_tokens"]
        or budget.get("packed_surplus_total")
        != manifest["packed_surplus_input_tokens"]
    ):
        raise ValueError("Order input-token budget schema is invalid")
    expected_accounting = (
        "available-order-inputs"
        if frozen_global_microbatch_rows is None
        else "consumed-complete-optimizer-update-inputs"
    )
    accounted_tokens = (
        available_input_tokens
        if frozen_global_microbatch_rows is None
        else consumption["consumed_input_tokens"]
    )
    if (
        budget.get("accounting") != expected_accounting
        or budget.get("actual_total") != accounted_tokens
    ):
        raise ValueError("Order input-token budget accounting is invalid")
    expected_weights = manifest.get("expected_input_token_weights")
    target = budget.get("expected_total")
    if target is None:
        if (
            row_selection != "all-packed-rows"
            or budget.get("delta") is not None
            or budget.get("tolerance") is not None
            or packed_surplus_rows != 0
        ):
            raise ValueError("Untargeted order selection/budget is invalid")
    else:
        if (
            row_selection != "strict-weighted-token-cap"
            or not isinstance(target, int)
            or isinstance(target, bool)
            or target < 1
            or not isinstance(budget.get("tolerance"), int)
            or isinstance(budget.get("tolerance"), bool)
            or budget["tolerance"] < 0
            or expected_weights is None
        ):
            raise ValueError("Targeted order selection/budget is invalid")
        normalized_weights = _normalize_domain_weights(
            expected_weights, source=manifest_path
        )
        maximum_rows = target // sequence_length
        update_rows = consumption["frozen_optimizer_update_rows"]
        selected_rows = (
            maximum_rows
            if update_rows is None
            else (maximum_rows // update_rows) * update_rows
        )
        if rows != selected_rows or expected_counts != _strict_weighted_row_counts(
            selected_rows, normalized_weights
        ):
            raise ValueError("Targeted order row allocation is invalid")
        delta = accounted_tokens - target
        if (
            budget.get("delta") != delta
            or delta > 0
            or delta < -budget["tolerance"]
        ):
            raise ValueError("Targeted order token delta is invalid")

    order_payload = manifest.get("order")
    if not isinstance(order_payload, dict):
        raise ValueError("Order payload metadata is invalid")
    relative_order_path = order_payload.get("path")
    if relative_order_path != "order.bin":
        raise ValueError("Order payload path must be the canonical 'order.bin'")
    expected_order_sha256 = _require_sha256(
        order_payload.get("sha256"), field="order.sha256", source=manifest_path
    )
    order_path = manifest_path.parent / relative_order_path
    expected_bytes = rows * REFERENCE_DTYPE.itemsize
    if not order_path.is_file() or order_path.stat().st_size != expected_bytes:
        raise IOError(f"Order file size mismatch: {order_path}")
    if order_payload.get("bytes") != expected_bytes:
        raise ValueError("Recorded order byte count is inconsistent")
    if verify_checksum and _sha256(order_path) != expected_order_sha256:
        raise IOError(f"Order checksum mismatch: {order_path}")
    return manifest, order_path


def validate_training_order(
    manifest_path: str | Path,
    *,
    verify_dataset_manifest_hashes: bool = True,
) -> dict[str, Any]:
    """Validate the order and prove every selected source row occurs once.

    The duplicate check uses one boolean array per domain rather than sorting
    the full 100+ MB order, so validation memory scales at roughly one bit
    represented as one byte per packed-available row. Unselected bits must
    reconcile exactly to the manifest's packed-surplus accounting.
    """

    path = Path(manifest_path)
    manifest, order_path = _load_order_manifest(path, verify_checksum=True)
    expected_counts = manifest.get("rows_per_domain", {})
    if set(expected_counts) != set(DOMAIN_ORDER):
        raise ValueError("Order manifest has an invalid rows_per_domain mapping")
    if any(
        not isinstance(expected_counts[domain], int) or expected_counts[domain] < 0
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Order manifest has invalid per-domain row counts")
    if sum(expected_counts.values()) != manifest["rows"]:
        raise ValueError("Order per-domain row counts do not sum to total rows")

    expected_weights = manifest.get("expected_input_token_weights")
    if expected_weights is not None:
        normalized_weights = _normalize_domain_weights(
            expected_weights,
            source=path,
        )
        for domain in DOMAIN_ORDER:
            if not math.isclose(
                float(expected_weights[domain]),
                normalized_weights[domain],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"Expected input-token weights are not normalized in {path}")
        _validate_row_mixture(expected_counts, normalized_weights, source=path)

    sequence_length = manifest.get("sequence_length")
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("Order manifest has an invalid sequence length")
    available_input_tokens = manifest["rows"] * sequence_length
    expected_input_tokens_per_domain = {
        domain: expected_counts[domain] * sequence_length for domain in DOMAIN_ORDER
    }
    if manifest.get("input_tokens_per_domain") != expected_input_tokens_per_domain:
        raise ValueError("Order input_tokens_per_domain is inconsistent")
    realized_input_weights = manifest.get("realized_input_token_weights", {})
    if set(realized_input_weights) != set(DOMAIN_ORDER) or any(
        not math.isclose(
            float(realized_input_weights[domain]),
            expected_input_tokens_per_domain[domain] / available_input_tokens,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Order realized input-token weights are inconsistent")

    budget = manifest.get("input_token_budget")
    if not isinstance(budget, dict) or budget.get("policy") != (
        "packed_model_inputs_including_eos_excluding_lookahead_duplicates"
    ):
        raise ValueError("Order input-token budget policy is invalid")
    if budget.get("direction") != "at_or_below":
        raise ValueError("Order input-token budget direction is invalid")
    if budget.get("available_total") != available_input_tokens:
        raise ValueError("Order available input-token budget is inconsistent")
    if budget.get("packed_available_total") != manifest[
        "packed_available_input_tokens"
    ]:
        raise ValueError("Order packed-available input-token budget is inconsistent")
    if budget.get("packed_surplus_total") != manifest[
        "packed_surplus_input_tokens"
    ]:
        raise ValueError("Order packed-surplus input-token budget is inconsistent")
    consumption = manifest["training_consumption"]
    frozen_global_microbatch_rows = consumption["frozen_global_microbatch_rows"]
    expected_accounting = (
        "available-order-inputs"
        if frozen_global_microbatch_rows is None
        else "consumed-complete-optimizer-update-inputs"
    )
    if budget.get("accounting") != expected_accounting:
        raise ValueError("Order input-token budget accounting is invalid")
    accounted_input_tokens = (
        available_input_tokens
        if frozen_global_microbatch_rows is None
        else consumption["consumed_input_tokens"]
    )
    if budget.get("actual_total") != accounted_input_tokens:
        raise ValueError("Order actual input-token budget is inconsistent")
    budget_target = budget.get("expected_total")
    budget_delta = budget.get("delta")
    budget_tolerance = budget.get("tolerance")
    if budget_target is None:
        if budget_delta is not None or budget_tolerance is not None:
            raise ValueError("Untargeted order has invalid budget delta/tolerance")
        if manifest.get("row_selection") != "all-packed-rows":
            raise ValueError("Untargeted order must select every packed row")
        if (
            manifest["packed_surplus_rows"] != 0
            or manifest["packed_available_rows"] != manifest["rows"]
        ):
            raise ValueError("Untargeted order has an unexpected packed surplus")
    else:
        if not isinstance(budget_target, int) or budget_target < 1:
            raise ValueError("Order expected input-token budget is invalid")
        if not isinstance(budget_tolerance, int) or budget_tolerance < 0:
            raise ValueError("Order input-token tolerance is invalid")
        expected_delta = accounted_input_tokens - budget_target
        if (
            budget_delta != expected_delta
            or expected_delta > 0
            or expected_delta < -budget_tolerance
        ):
            raise ValueError("Order input-token budget delta is invalid")
        if manifest.get("row_selection") != "strict-weighted-token-cap":
            raise ValueError("Targeted order has an invalid row-selection policy")
        if expected_weights is None:
            raise ValueError("Targeted order is missing strict input-token weights")
        maximum_rows = budget_target // sequence_length
        frozen_update_rows = consumption["frozen_optimizer_update_rows"]
        expected_selected_rows = (
            maximum_rows
            if frozen_update_rows is None
            else (maximum_rows // frozen_update_rows) * frozen_update_rows
        )
        if manifest["rows"] != expected_selected_rows:
            raise ValueError(
                "Targeted order does not contain the largest permitted row prefix"
            )
        expected_weighted_counts = _strict_weighted_row_counts(
            expected_selected_rows, normalized_weights
        )
        if expected_counts != expected_weighted_counts:
            raise ValueError("Targeted order does not use the strict weighted allocation")

    valid_loss_tokens_per_domain = manifest.get("valid_loss_tokens_per_domain", {})
    if set(valid_loss_tokens_per_domain) != set(DOMAIN_ORDER) or any(
        not isinstance(valid_loss_tokens_per_domain[domain], int)
        or valid_loss_tokens_per_domain[domain] < 0
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Order valid-loss-token counts are invalid")
    total_valid_loss_tokens = sum(valid_loss_tokens_per_domain.values())
    realized_valid_weights = manifest.get("realized_valid_loss_token_weights", {})
    if set(realized_valid_weights) != set(DOMAIN_ORDER) or any(
        not math.isclose(
            float(realized_valid_weights[domain]),
            (
                valid_loss_tokens_per_domain[domain] / total_valid_loss_tokens
                if total_valid_loss_tokens
                else 0.0
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for domain in DOMAIN_ORDER
    ):
        raise ValueError("Order realized valid-loss-token weights are inconsistent")
    packed_available_counts = manifest["packed_available_rows_per_domain"]
    seen = {
        domain_id: np.zeros(int(packed_available_counts[domain]), dtype=np.bool_)
        for domain_id, domain in enumerate(DOMAIN_ORDER)
    }
    observed_consumed_counts = {domain: 0 for domain in DOMAIN_ORDER}
    consumed_rows = consumption["consumed_rows"]
    order = np.memmap(
        order_path,
        dtype=REFERENCE_DTYPE,
        mode="r",
        shape=(manifest["rows"],),
    )
    chunk_rows = 1_000_000
    for begin in range(0, manifest["rows"], chunk_rows):
        chunk = np.asarray(order[begin : begin + chunk_rows])
        domain_ids = chunk >> np.uint64(ROW_BITS)
        if np.any(domain_ids >= len(DOMAIN_ORDER)):
            raise ValueError(f"Order contains an invalid domain ID near row {begin}")
        row_ids = chunk & np.uint64(MAX_ROW_ID)
        if consumed_rows is not None and begin < consumed_rows:
            consumed_in_chunk = min(len(chunk), consumed_rows - begin)
            prefix_domain_ids = domain_ids[:consumed_in_chunk]
            for domain_id, domain in enumerate(DOMAIN_ORDER):
                observed_consumed_counts[domain] += int(
                    np.count_nonzero(prefix_domain_ids == domain_id)
                )
        for domain_id in range(len(DOMAIN_ORDER)):
            selected = row_ids[domain_ids == domain_id]
            if not selected.size:
                continue
            domain_seen = seen[domain_id]
            if int(selected.max()) >= len(domain_seen):
                raise ValueError(
                    f"Order row ID is out of range for {DOMAIN_ORDER[domain_id]} near row {begin}"
                )
            unique = np.unique(selected)
            if unique.size != selected.size or np.any(domain_seen[unique]):
                raise ValueError(
                    f"Order contains a duplicate {DOMAIN_ORDER[domain_id]} row near row {begin}"
                )
            domain_seen[unique] = True
    for domain_id, domain in enumerate(DOMAIN_ORDER):
        if int(np.count_nonzero(seen[domain_id])) != expected_counts[domain]:
            raise ValueError(
                f"Order selected-row count differs for {domain}"
            )
    if consumed_rows is not None and (
        observed_consumed_counts != consumption["consumed_rows_per_domain"]
    ):
        raise ValueError(
            "Training-consumption per-domain counts differ from the frozen order prefix"
        )

    if verify_dataset_manifest_hashes:
        resolved = _resolve_order_manifests(path)
        if consumed_rows is not None:
            observed_dropped_supervised = _count_supervised_tokens_for_references(
                np.asarray(order[consumed_rows:]),
                resolved,
                sequence_length=sequence_length,
            )
            if observed_dropped_supervised != consumption[
                "dropped_supervised_tokens_per_domain"
            ]:
                raise ValueError(
                    "Training-consumption supervised counts differ from the frozen "
                    "order suffix"
                )
        for domain in DOMAIN_ORDER:
            packed_manifest, _ = _parse_packed_manifest(resolved[domain])
            expected_fields = {
                "domain": domain,
                "rows": packed_available_counts[domain],
                "input_tokens": manifest[
                    "packed_available_input_tokens_per_domain"
                ][domain],
                "valid_loss_tokens": manifest[
                    "packed_available_supervised_tokens_per_domain"
                ][domain],
                "split": manifest["split"],
                "sequence_length": sequence_length,
                "vocab_size": manifest["vocab_size"],
                "eos_token_id": manifest["eos_token_id"],
                "tokenizer_manifest_sha256": manifest["tokenizer_manifest_sha256"],
            }
            for field, expected in expected_fields.items():
                if packed_manifest.get(field) != expected:
                    raise ValueError(
                        f"Order {field} differs from packed manifest for {domain}"
                    )

        # A capped order is an arbitrary deterministic subset of each packed
        # domain.  Prove its exact supervised-token accounting from the actual
        # selected row IDs (or, when cheaper, from the complementary surplus).
        # This permits unselected packed rows while retaining the same
        # fail-closed bounds, uniqueness, and loss-accounting guarantees as an
        # all-row order.
        packed_surplus_rows = manifest["packed_surplus_rows"]
        if packed_surplus_rows <= manifest["rows"]:
            surplus_chunks: list[np.ndarray[Any, Any]] = []
            for domain_id, _domain in enumerate(DOMAIN_ORDER):
                row_ids = np.flatnonzero(~seen[domain_id]).astype(
                    REFERENCE_DTYPE, copy=False
                )
                surplus_chunks.append(
                    _encoded_domain_references(domain_id, row_ids)
                )
            surplus_references = np.concatenate(surplus_chunks)
            observed_surplus_supervised = _count_supervised_tokens_for_references(
                surplus_references,
                resolved,
                sequence_length=sequence_length,
            )
            observed_selected_supervised = {
                domain: manifest[
                    "packed_available_supervised_tokens_per_domain"
                ][domain]
                - observed_surplus_supervised[domain]
                for domain in DOMAIN_ORDER
            }
        else:
            observed_selected_supervised = _count_supervised_tokens_for_references(
                np.asarray(order),
                resolved,
                sequence_length=sequence_length,
            )
            observed_surplus_supervised = {
                domain: manifest[
                    "packed_available_supervised_tokens_per_domain"
                ][domain]
                - observed_selected_supervised[domain]
                for domain in DOMAIN_ORDER
            }
        if observed_selected_supervised != valid_loss_tokens_per_domain:
            raise ValueError(
                "Order selected supervised-token counts differ from source rows"
            )
        if observed_surplus_supervised != manifest[
            "packed_surplus_supervised_tokens_per_domain"
        ]:
            raise ValueError(
                "Order packed-surplus supervised-token counts differ from source rows"
            )
    return manifest


def _frozen_training_geometry_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    consumption = manifest["training_consumption"]
    global_microbatch_rows = consumption["frozen_global_microbatch_rows"]
    accumulation = consumption["frozen_gradient_accumulation_steps"]
    if global_microbatch_rows is None or accumulation is None:
        raise ValueError(
            "Training order has no frozen optimizer-update geometry; rebuild it "
            "with global microbatch rows and gradient accumulation"
        )
    return {
        "global_microbatch_rows": global_microbatch_rows,
        "gradient_accumulation_steps": accumulation,
        "optimizer_update_rows": consumption["frozen_optimizer_update_rows"],
        "optimizer_updates": consumption["optimizer_updates"],
        "consumed_global_microbatches": consumption[
            "consumed_global_microbatches"
        ],
        "sequence_length": manifest["sequence_length"],
        "consumed_rows": consumption["consumed_rows"],
        "dropped_rows": consumption["dropped_tail_rows"],
        "consumed_input_tokens": consumption["consumed_input_tokens"],
        "dropped_input_tokens": consumption["dropped_input_tokens"],
        "consumed_supervised_tokens": consumption["consumed_supervised_tokens"],
        "dropped_supervised_tokens": consumption["dropped_supervised_tokens"],
        "consumed_input_tokens_per_domain": dict(
            consumption["consumed_input_tokens_per_domain"]
        ),
        "consumed_supervised_tokens_per_domain": dict(
            consumption["consumed_supervised_tokens_per_domain"]
        ),
    }


def frozen_training_geometry(
    order_manifest_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """Return the immutable optimizer-update contract for a production run.

    Development orders without frozen geometry are deliberately rejected: a
    trainer must never infer gradient accumulation after the 52.58B-token
    accounting prefix has been authorized.
    """

    path = Path(order_manifest_path)
    manifest, _ = _load_order_manifest(path, verify_checksum=verify_checksum)
    return _frozen_training_geometry_from_manifest(manifest)


def evaluation_order_geometry(
    order_manifest_path: str | Path,
    *,
    global_microbatch_rows: int,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """Return a deterministic whole-microbatch prefix for held-out evaluation.

    Validation/test orders are intentionally not frozen to an optimizer
    accumulation geometry. Evaluation uses the training run's global
    microbatch rows, consumes only complete global microbatches, and records an
    explicit tail instead of pretending the held-out order is a train order.
    """

    if (
        not isinstance(global_microbatch_rows, int)
        or isinstance(global_microbatch_rows, bool)
        or global_microbatch_rows < 1
    ):
        raise ValueError("global_microbatch_rows must be a positive integer")
    path = Path(order_manifest_path)
    manifest, _ = _load_order_manifest(path, verify_checksum=verify_checksum)
    if manifest["split"] not in ("validation", "test"):
        raise ValueError("Evaluation order must identify validation or test split")
    consumption = manifest["training_consumption"]
    if consumption["frozen_global_microbatch_rows"] is not None:
        raise ValueError(
            "Held-out evaluation order unexpectedly freezes optimizer geometry"
        )
    rows = int(manifest["rows"])
    global_microbatches = rows // global_microbatch_rows
    consumed_rows = global_microbatches * global_microbatch_rows
    if global_microbatches < 1:
        raise ValueError("Evaluation order is smaller than one global microbatch")
    sequence_length = int(manifest["sequence_length"])
    return {
        "split": manifest["split"],
        "global_microbatch_rows": global_microbatch_rows,
        "available_rows": rows,
        "consumed_rows": consumed_rows,
        "dropped_tail_rows": rows - consumed_rows,
        "available_global_microbatches": global_microbatches,
        "sequence_length": sequence_length,
        "consumed_input_tokens": consumed_rows * sequence_length,
        "dropped_input_tokens": (rows - consumed_rows) * sequence_length,
        "vocab_size": manifest["vocab_size"],
        "tokenizer_manifest_sha256": manifest["tokenizer_manifest_sha256"],
    }


class DistributedBatchSampler(Sampler[list[int]]):
    """Stateless global-order sampler with disjoint data-parallel ownership.

    Resume uses the number of *completed* global microbatches, not the sampler's
    internal iterator position. This remains correct when DataLoader workers
    have prefetched future batches that were never consumed by the optimizer.
    """

    def __init__(
        self,
        order_manifest_path: str | Path,
        *,
        global_microbatch_rows: int,
        gradient_accumulation_steps: int = 1,
        rank: int = 0,
        world_size: int = 1,
        start_global_microbatch: int = 0,
        resume_state: Mapping[str, Any] | None = None,
        verify_checksum: bool = True,
    ) -> None:
        self.order_manifest_path = Path(order_manifest_path)
        self.manifest, order_path = _load_order_manifest(
            self.order_manifest_path, verify_checksum=verify_checksum
        )
        self.global_microbatch_rows = int(global_microbatch_rows)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_global_microbatch = int(start_global_microbatch)
        if resume_state is not None and self.start_global_microbatch != 0:
            raise ValueError(
                "resume_state and a nonzero start_global_microbatch are mutually exclusive"
            )
        if self.global_microbatch_rows < 1:
            raise ValueError("global_microbatch_rows must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid distributed rank/world_size")
        if self.global_microbatch_rows % self.world_size:
            raise ValueError("global_microbatch_rows must be divisible by world_size")
        if self.start_global_microbatch < 0:
            raise ValueError("start_global_microbatch must be non-negative")
        consumption = self.manifest["training_consumption"]
        frozen_global_microbatch_rows = consumption[
            "frozen_global_microbatch_rows"
        ]
        frozen_gradient_accumulation_steps = consumption[
            "frozen_gradient_accumulation_steps"
        ]
        self.training_geometry = (
            None
            if frozen_global_microbatch_rows is None
            else _frozen_training_geometry_from_manifest(self.manifest)
        )
        if (
            frozen_global_microbatch_rows is not None
            and self.global_microbatch_rows != frozen_global_microbatch_rows
        ):
            raise ValueError(
                "global_microbatch_rows differs from the immutable order: found "
                f"{self.global_microbatch_rows}, expected "
                f"{frozen_global_microbatch_rows}"
            )
        if (
            frozen_gradient_accumulation_steps is not None
            and self.gradient_accumulation_steps
            != frozen_gradient_accumulation_steps
        ):
            raise ValueError(
                "gradient_accumulation_steps differs from the immutable order: found "
                f"{self.gradient_accumulation_steps}, expected "
                f"{frozen_gradient_accumulation_steps}"
            )
        self.optimizer_update_rows = (
            self.global_microbatch_rows * self.gradient_accumulation_steps
        )
        self.local_batch_size = self.global_microbatch_rows // self.world_size
        if frozen_global_microbatch_rows is None:
            self.total_optimizer_updates = (
                self.manifest["rows"] // self.optimizer_update_rows
            )
            self.total_global_microbatches = (
                self.total_optimizer_updates * self.gradient_accumulation_steps
            )
            self.trainable_rows = (
                self.total_optimizer_updates * self.optimizer_update_rows
            )
        else:
            self.total_optimizer_updates = consumption["optimizer_updates"]
            self.total_global_microbatches = consumption[
                "consumed_global_microbatches"
            ]
            self.trainable_rows = consumption["consumed_rows"]
        self.dropped_rows = self.manifest["rows"] - self.trainable_rows
        if self.start_global_microbatch > self.total_global_microbatches:
            raise ValueError("start_global_microbatch is beyond the end of the order")
        if self.start_global_microbatch % self.gradient_accumulation_steps:
            raise ValueError(
                "start_global_microbatch must be aligned to an optimizer-update boundary"
            )
        self.order_manifest_sha256 = _sha256(self.order_manifest_path)
        self.order_payload_sha256 = self.manifest["order"]["sha256"]
        self.data_identity = (
            f"order-manifest-sha256:{self.order_manifest_sha256};"
            f"order-payload-sha256:{self.order_payload_sha256}"
        )
        self.dataset_manifest_sha256 = {
            domain: self.manifest["dataset_manifests"][domain]["sha256"]
            for domain in DOMAIN_ORDER
        }
        self.tokenizer_manifest_sha256 = self.manifest["tokenizer_manifest_sha256"]
        self._iteration_started = False
        if resume_state is not None:
            self.load_state_dict(resume_state)
        self._order = np.memmap(
            order_path,
            dtype=REFERENCE_DTYPE,
            mode="r",
            shape=(self.manifest["rows"],),
        )

    def __len__(self) -> int:
        return self.total_global_microbatches - self.start_global_microbatch

    def __iter__(self) -> Iterator[list[int]]:
        self._iteration_started = True
        local_begin = self.rank * self.local_batch_size
        local_end = local_begin + self.local_batch_size
        for global_microbatch in range(
            self.start_global_microbatch, self.total_global_microbatches
        ):
            begin = global_microbatch * self.global_microbatch_rows
            global_references = self._order[begin : begin + self.global_microbatch_rows]
            yield [int(value) for value in global_references[local_begin:local_end]]

    def state_dict(self, *, completed_global_microbatches: int) -> dict[str, Any]:
        if (
            not isinstance(completed_global_microbatches, int)
            or isinstance(completed_global_microbatches, bool)
            or completed_global_microbatches < self.start_global_microbatch
            or completed_global_microbatches > self.total_global_microbatches
        ):
            raise ValueError("completed_global_microbatches is out of range")
        if completed_global_microbatches % self.gradient_accumulation_steps:
            raise ValueError(
                "completed_global_microbatches must be an optimizer-update boundary"
            )
        completed_optimizer_updates = (
            completed_global_microbatches // self.gradient_accumulation_steps
        )
        next_order_row = completed_global_microbatches * self.global_microbatch_rows
        return {
            "format": SAMPLER_STATE_FORMAT,
            "format_version": SAMPLER_STATE_FORMAT_VERSION,
            "order_manifest_sha256": self.order_manifest_sha256,
            "order_payload_sha256": self.order_payload_sha256,
            "dataset_manifest_sha256": dict(self.dataset_manifest_sha256),
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "completed_global_microbatches": int(completed_global_microbatches),
            "completed_optimizer_updates": completed_optimizer_updates,
            "next_order_row": next_order_row,
            "consumed_input_tokens": next_order_row * self.manifest["sequence_length"],
            "global_microbatch_rows": self.global_microbatch_rows,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_update_rows": self.optimizer_update_rows,
            "world_size": self.world_size,
            "sequence_length": self.manifest["sequence_length"],
            "total_global_microbatches": self.total_global_microbatches,
            "total_optimizer_updates": self.total_optimizer_updates,
            "trainable_rows": self.trainable_rows,
            "dropped_rows": self.dropped_rows,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate and apply an exact, rank-independent resume cursor.

        The cursor is an absolute position in the immutable global order. It
        intentionally does not contain ``rank``: every rank resumes the same
        global microbatch and derives its disjoint local slice at runtime.
        """

        if self._iteration_started:
            raise RuntimeError("Cannot change sampler resume state after iteration started")
        if not isinstance(state, Mapping):
            raise TypeError("Sampler resume state must be a mapping")
        expected_keys = {
            "format",
            "format_version",
            "order_manifest_sha256",
            "order_payload_sha256",
            "dataset_manifest_sha256",
            "tokenizer_manifest_sha256",
            "completed_global_microbatches",
            "completed_optimizer_updates",
            "next_order_row",
            "consumed_input_tokens",
            "global_microbatch_rows",
            "gradient_accumulation_steps",
            "optimizer_update_rows",
            "world_size",
            "sequence_length",
            "total_global_microbatches",
            "total_optimizer_updates",
            "trainable_rows",
            "dropped_rows",
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys.difference(state))
            extra = sorted(set(state).difference(expected_keys))
            raise ValueError(
                f"Sampler resume state schema mismatch: missing={missing}, extra={extra}"
            )
        integer_fields = (
            "format_version",
            "completed_global_microbatches",
            "completed_optimizer_updates",
            "next_order_row",
            "consumed_input_tokens",
            "global_microbatch_rows",
            "gradient_accumulation_steps",
            "optimizer_update_rows",
            "world_size",
            "sequence_length",
            "total_global_microbatches",
            "total_optimizer_updates",
            "trainable_rows",
            "dropped_rows",
        )
        if any(type(state[field]) is not int for field in integer_fields):
            raise ValueError("Sampler resume integer fields must be plain integers")
        if type(state["dataset_manifest_sha256"]) is not dict:
            raise ValueError("Sampler resume dataset identity must be a plain dictionary")
        for field in (
            "format",
            "order_manifest_sha256",
            "order_payload_sha256",
            "tokenizer_manifest_sha256",
        ):
            if type(state[field]) is not str:
                raise ValueError(f"Sampler resume {field} must be a string")
        expected_static: dict[str, Any] = {
            "format": SAMPLER_STATE_FORMAT,
            "format_version": SAMPLER_STATE_FORMAT_VERSION,
            "order_manifest_sha256": self.order_manifest_sha256,
            "order_payload_sha256": self.order_payload_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "global_microbatch_rows": self.global_microbatch_rows,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_update_rows": self.optimizer_update_rows,
            "world_size": self.world_size,
            "sequence_length": self.manifest["sequence_length"],
            "total_global_microbatches": self.total_global_microbatches,
            "total_optimizer_updates": self.total_optimizer_updates,
            "trainable_rows": self.trainable_rows,
            "dropped_rows": self.dropped_rows,
        }
        for field, expected in expected_static.items():
            if state[field] != expected:
                raise ValueError(
                    f"Sampler resume {field} mismatch: found {state[field]!r}, "
                    f"expected {expected!r}"
                )
        completed = state["completed_global_microbatches"]
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or not 0 <= completed <= self.total_global_microbatches
        ):
            raise ValueError("Sampler resume completed_global_microbatches is out of range")
        if completed % self.gradient_accumulation_steps:
            raise ValueError(
                "Sampler resume cursor is not an optimizer-update boundary"
            )
        expected_completed_updates = completed // self.gradient_accumulation_steps
        if state["completed_optimizer_updates"] != expected_completed_updates:
            raise ValueError(
                "Sampler resume completed_optimizer_updates does not match the cursor"
            )
        expected_next_row = completed * self.global_microbatch_rows
        if state["next_order_row"] != expected_next_row:
            raise ValueError(
                "Sampler resume next_order_row does not match the completed batch cursor"
            )
        expected_consumed_tokens = expected_next_row * self.manifest["sequence_length"]
        if state["consumed_input_tokens"] != expected_consumed_tokens:
            raise ValueError(
                "Sampler resume consumed_input_tokens does not match the order cursor"
            )
        self.start_global_microbatch = completed

    def close(self) -> None:
        mmap_handle = getattr(self._order, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


def _resolve_order_manifests(order_manifest_path: Path) -> dict[str, Path]:
    manifest, _ = _load_order_manifest(order_manifest_path, verify_checksum=False)
    resolved: dict[str, Path] = {}
    for domain in DOMAIN_ORDER:
        payload = manifest["dataset_manifests"].get(domain, {})
        path = (order_manifest_path.parent / payload.get("path", "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != payload.get("sha256"):
            raise IOError(f"Packed dataset manifest changed after order creation: {path}")
        packed_manifest, _ = _parse_packed_manifest(path)
        if (
            packed_manifest["tokenizer_manifest_sha256"]
            != manifest["tokenizer_manifest_sha256"]
        ):
            raise ValueError(
                f"Packed dataset tokenizer manifest differs from the training order: {path}"
            )
        resolved[domain] = path
    return resolved


def load_diagnostic_order_prefix(
    order_manifest_path: str | Path,
    *,
    rows: int,
    verify_payload_checksums: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a small frozen-order prefix without changing training geometry.

    This is intentionally a diagnostic API for fixed-batch overfit and decode
    inspection. Production iteration must use ``create_training_dataloader``;
    allowing an arbitrary diagnostic batch size there would weaken the frozen
    optimizer-update contract.
    """

    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("rows must be a positive integer")
    order_manifest = Path(order_manifest_path)
    manifest, order_path = _load_order_manifest(order_manifest, verify_checksum=True)
    consumption = manifest["training_consumption"]
    trainable_rows = consumption["consumed_rows"]
    if trainable_rows is None:
        trainable_rows = manifest["rows"]
    if rows > trainable_rows:
        raise ValueError(
            f"Diagnostic prefix requests {rows} rows, only {trainable_rows} are trainable"
        )
    manifests = _resolve_order_manifests(order_manifest)
    dataset = DomainMixtureDataset(
        manifests,
        verify_checksums=verify_payload_checksums,
    )
    order = np.memmap(
        order_path,
        dtype=REFERENCE_DTYPE,
        mode="r",
        shape=(manifest["rows"],),
    )
    try:
        samples = [dataset[reference] for reference in order[:rows]]
        return PackedBatchCollator(dataset.sequence_length)(samples)
    finally:
        mmap_handle = getattr(order, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        dataset.close()


def create_training_dataloader(
    order_manifest_path: str | Path,
    *,
    global_microbatch_rows: int,
    gradient_accumulation_steps: int = 1,
    rank: int = 0,
    world_size: int = 1,
    start_global_microbatch: int = 0,
    resume_state: Mapping[str, Any] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    multiprocessing_context: str | None = "spawn",
    verify_order_checksum: bool = True,
    verify_payload_checksums: bool = False,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DistributedBatchSampler]:
    """Construct the native PyTorch loader and its explicit resume sampler.

    The compact ``order.bin`` is checksum-verified by default independently of
    the much larger packed payloads. Set ``verify_payload_checksums`` only for
    an explicit full-corpus integrity pass. A ``resume_state`` is preferred to
    a bare offset because it binds that cursor to all data identities.
    """

    order_path = Path(order_manifest_path)
    sampler = DistributedBatchSampler(
        order_path,
        global_microbatch_rows=global_microbatch_rows,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank=rank,
        world_size=world_size,
        start_global_microbatch=start_global_microbatch,
        resume_state=resume_state,
        verify_checksum=verify_order_checksum,
    )
    try:
        manifests = _resolve_order_manifests(order_path)
        dataset = DomainMixtureDataset(
            manifests,
            verify_checksums=verify_payload_checksums,
        )
    except BaseException:
        sampler.close()
        raise
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": PackedBatchCollator(dataset.sequence_length),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": int(num_workers) > 0 and bool(persistent_workers),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context
    try:
        loader = DataLoader(**kwargs)
    except BaseException:
        dataset.close()
        sampler.close()
        raise
    return loader, sampler
