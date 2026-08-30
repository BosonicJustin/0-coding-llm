#!/usr/bin/env python3
"""Count a normalized shard and atomically add it to the quota ledger.

Supported inputs are JSONL/JSONL.GZ and text/text.gz. Exact counting accepts
either a tokenizer.json file or a local Hugging Face tokenizer directory. No
network download is attempted. The main streaming collector records counts
directly, so this command is primarily an audit tool.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from quota_tracker import DEFAULT_CONFIG, estimate_tokens, load_config, write_record


def nested_value(item: dict[str, Any], dotted_field: str) -> Any:
    value: Any = item
    for part in dotted_field.split("."):
        value = value[part]
    return value


def iter_jsonl(path: Path, field: str) -> Iterator[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                value = nested_value(item, field)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: cannot read {field!r}: {exc}") from exc
            if not isinstance(value, str):
                raise ValueError(f"{path}:{line_number}: {field!r} is not a string")
            yield value


def iter_text(path: Path) -> Iterator[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        yield handle.read()


def detect_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz")):
        return "jsonl"
    if name.endswith((".txt", ".txt.gz")):
        return "text"
    raise ValueError(f"Cannot infer the format of {path}; pass --format")


def batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class ExactTokenizer:
    def __init__(self, path: Path):
        if path.is_file():
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError("Exact counting requires tokenizers") from exc
            self.tokenizer = Tokenizer.from_file(str(path))
            self.kind = "tokenizers"
        elif path.is_dir():
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("A tokenizer directory requires transformers") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False, use_fast=True
            )
            self.kind = "transformers"
        else:
            raise ValueError(f"Tokenizer does not exist: {path}")

    def lengths(self, texts: list[str]) -> list[int]:
        if self.kind == "tokenizers":
            return [
                len(encoding.ids)
                for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False)
            ]
        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(token_ids) for token_ids in encodings["input_ids"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("collection", "final"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--language-group")
    parser.add_argument("--shard-id", help="defaults to the input filename")
    parser.add_argument("--source")
    parser.add_argument("--format", choices=("auto", "jsonl", "text"), default="auto")
    parser.add_argument("--text-field", default="content")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer.json or local HF tokenizer directory")
    parser.add_argument(
        "--eos-per-document",
        type=int,
        default=1,
        help="tokens inserted between documents by the eventual packer (default: 1)",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if not args.path.is_file():
            raise ValueError(f"Input does not exist or is not a file: {args.path}")
        if args.phase == "final" and args.split is None:
            raise ValueError("--split is required for final counts")
        if args.phase == "collection" and args.tokenizer is None:
            raise ValueError("--tokenizer is required because collection thresholds are exact")
        if args.eos_per_document < 0:
            raise ValueError("--eos-per-document cannot be negative")
        config = load_config(args.config)
        input_format = detect_format(args.path) if args.format == "auto" else args.format
        if input_format == "jsonl":
            documents = iter_jsonl(args.path, args.text_field)
        else:
            documents = iter_text(args.path)

        tokenizer = ExactTokenizer(args.tokenizer) if args.tokenizer else None
        document_count = 0
        clean_bytes = 0
        exact_tokens = 0
        for batch in batched(documents, args.batch_size):
            document_count += len(batch)
            clean_bytes += sum(len(text.encode("utf-8")) for text in batch)
            if tokenizer is not None:
                exact_tokens += sum(tokenizer.lengths(batch))
                exact_tokens += len(batch) * args.eos_per_document

        estimated_tokens = estimate_tokens(clean_bytes, args.category, config)
        record = {
            "phase": args.phase,
            "split": args.split,
            "category": args.category,
            "language_group": args.language_group,
            "shard_id": args.shard_id or args.path.name,
            "source": args.source or str(args.path),
            "documents": document_count,
            "clean_bytes": clean_bytes,
            "estimated_tokens": estimated_tokens,
            "exact_tokens": exact_tokens if tokenizer is not None else None,
        }
        record = {key: value for key, value in record.items() if value is not None}
        record_path, created = write_record(args.root, record, replace=args.replace)
        print(
            json.dumps(
                {
                    "record": str(record_path),
                    "created": created,
                    "documents": document_count,
                    "clean_bytes": clean_bytes,
                    "estimated_tokens": estimated_tokens,
                    "exact_tokens": exact_tokens if tokenizer is not None else None,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
