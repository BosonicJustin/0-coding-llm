#!/usr/bin/env python3
"""Download, pin, checksum, and validate the experiment tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_REPO = "bigcode/starcoder2-tokenizer"
EXPECTED_VOCAB_SIZE = 49_152
REQUIRED_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<repo_name>",
    "<file_sep>",
)
MANIFEST_NAME = "TOKENIZER_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def existing_revision(output: Path, repo_id: str) -> str | None:
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != repo_id:
        raise ValueError(
            f"{output} already contains a tokenizer from {manifest.get('repo_id')!r}, "
            f"not {repo_id!r}"
        )
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"Invalid existing manifest: {manifest_path}")
    return revision


def validate_tokenizer(output: Path) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Tokenizer validation requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(output), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Expected a {EXPECTED_VOCAB_SIZE}-token vocabulary, found {len(tokenizer)}"
        )
    vocabulary = tokenizer.get_vocab()
    missing = [token for token in REQUIRED_SPECIAL_TOKENS if token not in vocabulary]
    if missing:
        raise ValueError(f"Tokenizer is missing required special tokens: {', '.join(missing)}")

    samples = (
        "def fib(n: int) -> int:\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
        "fn main() { println!(\"Καλημέρα 🦀\"); }\n",
        "English prose, tabs\tand whitespace.\n",
    )
    for sample in samples:
        token_ids = tokenizer.encode(sample, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != sample:
            raise ValueError(f"Tokenizer failed byte-preserving round trip for {sample!r}")

    return {
        "vocab_size": len(tokenizer),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "special_token_ids": {token: vocabulary[token] for token in REQUIRED_SPECIAL_TOKENS},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/dataset/tokenizer/starcoder2"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--revision",
        help="branch, tag, or preferably a 40-character commit SHA; defaults to main on first run",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub before downloading the tokenizer") from exc

        pinned = existing_revision(args.output, args.repo_id)
        requested_revision = args.revision or pinned or "main"
        info = HfApi().model_info(args.repo_id, revision=requested_revision)
        resolved_revision = info.sha
        if not isinstance(resolved_revision, str) or len(resolved_revision) != 40:
            raise RuntimeError(f"Hugging Face returned an invalid revision: {resolved_revision!r}")
        if pinned is not None and resolved_revision != pinned:
            raise ValueError(
                f"Existing tokenizer is pinned to {pinned}, but {requested_revision!r} resolves to "
                f"{resolved_revision}. Use a new output directory to change tokenizer revisions."
            )

        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=resolved_revision,
            local_dir=args.output,
            allow_patterns=("*.json", "*.txt"),
        )
        validation = validate_tokenizer(args.output)
        files = {}
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != MANIFEST_NAME:
                files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "manifest_version": 1,
            "repo_id": args.repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "files": files,
            "validation": validation,
        }
        atomic_json(args.output / MANIFEST_NAME, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
