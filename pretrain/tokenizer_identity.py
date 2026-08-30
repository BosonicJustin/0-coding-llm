"""Fail-closed identity checks for the frozen pretraining tokenizer.

The tokenizer manifest authenticates the exact source files.  The vocabulary
digest separately authenticates the semantic token-to-ID mapping so two
tokenizers with the same vocabulary size cannot be substituted silently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKENIZER_MANIFEST_NAME = "TOKENIZER_MANIFEST.json"


class TokenizerIdentityError(ValueError):
    """Raised when a tokenizer identity cannot be authenticated."""


@dataclass(frozen=True)
class TokenizerIdentity:
    """Authenticated identities for one immutable tokenizer directory."""

    manifest_sha256: str
    vocabulary_sha256: str
    vocab_size: int
    manifest_path: Path
    manifest_bytes: bytes
    manifest: Mapping[str, Any]


def require_sha256(value: Any, *, field: str) -> str:
    """Return a validated lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TokenizerIdentityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def vocabulary_sha256(vocabulary: Mapping[str, int]) -> str:
    """Hash a canonical token-to-ID mapping independent of file serialization."""

    if not vocabulary:
        raise TokenizerIdentityError("Tokenizer vocabulary must not be empty")
    if any(not isinstance(token, str) for token in vocabulary):
        raise TokenizerIdentityError("Tokenizer vocabulary contains a non-string token")
    identifiers = list(vocabulary.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in identifiers):
        raise TokenizerIdentityError("Tokenizer vocabulary contains a non-integer ID")
    if set(identifiers) != set(range(len(vocabulary))):
        raise TokenizerIdentityError(
            f"Tokenizer IDs must be exactly 0..{len(vocabulary) - 1}"
        )
    ordered = sorted(vocabulary.items(), key=lambda item: (item[1], item[0]))
    payload = json.dumps(
        ordered,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_tokenizer_identity(
    tokenizer_root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_vocabulary_sha256: str | None = None,
    expected_vocab_size: int | None = None,
) -> TokenizerIdentity:
    """Authenticate a tokenizer manifest, every declared file, and vocabulary.

    Expected identities are optional for discovery, but callers at a trust
    boundary (training resume and model export) must pass the identities they
    obtained from the immutable data order or checkpoint.
    """

    root = Path(tokenizer_root)
    if root.is_symlink() or not root.is_dir():
        raise TokenizerIdentityError(f"Tokenizer path is not a directory: {root}")
    manifest_path = root / TOKENIZER_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TokenizerIdentityError(
            f"Tokenizer source manifest is missing: {manifest_path}"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerIdentityError(
            f"Cannot read tokenizer source manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise TokenizerIdentityError("Tokenizer source manifest root must be an object")
    if manifest.get("manifest_version") != 1:
        raise TokenizerIdentityError("Unsupported tokenizer source manifest version")

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = require_sha256(
            expected_manifest_sha256,
            field="expected tokenizer manifest SHA-256",
        )
        if manifest_sha256 != expected_manifest_sha256:
            raise TokenizerIdentityError(
                "Tokenizer manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, found {manifest_sha256}"
            )

    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TokenizerIdentityError(
            "Tokenizer source manifest must contain a non-empty files object"
        )
    if "tokenizer.json" not in files:
        raise TokenizerIdentityError("Tokenizer source manifest does not pin tokenizer.json")
    for name in files:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name == TOKENIZER_MANIFEST_NAME
        ):
            raise TokenizerIdentityError(
                f"Unsafe tokenizer source manifest file name: {name!r}"
            )
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != TOKENIZER_MANIFEST_NAME
    }
    declared_file_names = set(files)
    if actual_files != declared_file_names:
        missing = sorted(declared_file_names - actual_files)
        extra = sorted(actual_files - declared_file_names)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"undeclared {extra}")
        raise TokenizerIdentityError(
            "Tokenizer source directory differs from its closed-world manifest: "
            + "; ".join(details)
        )
    for name, descriptor in sorted(files.items()):
        if not isinstance(descriptor, Mapping):
            raise TokenizerIdentityError(
                f"Tokenizer source manifest record for {name!r} must be an object"
            )
        expected_bytes = descriptor.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise TokenizerIdentityError(
                f"Tokenizer source manifest bytes for {name!r} must be non-negative"
            )
        expected_file_sha256 = require_sha256(
            descriptor.get("sha256"),
            field=f"tokenizer source manifest SHA-256 for {name!r}",
        )
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise TokenizerIdentityError(f"Tokenizer source file is missing: {artifact}")
        actual_bytes = artifact.stat().st_size
        if actual_bytes != expected_bytes:
            raise TokenizerIdentityError(
                f"Tokenizer source manifest size mismatch for {name!r}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        actual_file_sha256 = sha256_file(artifact)
        if actual_file_sha256 != expected_file_sha256:
            raise TokenizerIdentityError(
                f"Tokenizer source manifest checksum mismatch for {name!r}: "
                f"expected {expected_file_sha256}, found {actual_file_sha256}"
            )

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Install tokenizers to verify tokenizer identity") from exc
    try:
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    except Exception as exc:
        raise TokenizerIdentityError(
            f"Cannot load pinned tokenizer.json from {root}: {exc}"
        ) from exc
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    vocabulary_digest = vocabulary_sha256(vocabulary)
    vocab_size = len(vocabulary)
    if (
        expected_vocab_size is not None
        and (
            isinstance(expected_vocab_size, bool)
            or not isinstance(expected_vocab_size, int)
            or expected_vocab_size < 1
        )
    ):
        raise TokenizerIdentityError(
            "Expected tokenizer vocabulary size must be a positive integer"
        )
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise TokenizerIdentityError(
            f"Tokenizer vocabulary size mismatch: expected {expected_vocab_size}, "
            f"found {vocab_size}"
        )
    validation = manifest.get("validation")
    if isinstance(validation, Mapping) and "vocab_size" in validation:
        if validation.get("vocab_size") != vocab_size:
            raise TokenizerIdentityError(
                "Tokenizer vocabulary size disagrees with source manifest validation"
            )
    if expected_vocabulary_sha256 is not None:
        expected_vocabulary_sha256 = require_sha256(
            expected_vocabulary_sha256,
            field="expected tokenizer vocabulary SHA-256",
        )
        if vocabulary_digest != expected_vocabulary_sha256:
            raise TokenizerIdentityError(
                "Tokenizer vocabulary SHA-256 mismatch: "
                f"expected {expected_vocabulary_sha256}, found {vocabulary_digest}"
            )
    return TokenizerIdentity(
        manifest_sha256=manifest_sha256,
        vocabulary_sha256=vocabulary_digest,
        vocab_size=vocab_size,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
    )
