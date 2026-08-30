#!/usr/bin/env python3
"""Smoke-test audited raw archives through tokenization and the CPU loader.

This is an integration check, not the final filtering/deduplication/selection
builder. It reads immutable raw archives and completed preprocessing reports,
writes only to a disposable staging directory, and never touches SQLite.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import (
    DOMAIN_ORDER,
    PackedShardWriter,
    build_training_order,
    create_training_dataloader,
    validate_packed_manifest,
    validate_training_order,
)


RAW_BUCKETS = {
    "python": ("python",),
    "other_code": ("other_code",),
    "english": ("fineweb_edu", "wikipedia"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_audited_archive(root: Path, domain: str) -> tuple[dict[str, Any], Path]:
    staging = root / "staging" / "preprocess"
    for bucket in RAW_BUCKETS[domain]:
        candidates: list[tuple[int, dict[str, Any], Path]] = []
        for report_path in (staging / "reports" / bucket).glob("part-*.json"):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            archive_value = report.get("archive")
            fingerprint_value = report.get("fingerprint_file")
            if not isinstance(archive_value, str) or not isinstance(fingerprint_value, str):
                continue
            archive_path = root / archive_value
            fingerprint_path = staging / fingerprint_value
            if archive_path.is_file() and fingerprint_path.is_file():
                compressed_bytes = int(
                    report.get("archive_compressed_bytes") or archive_path.stat().st_size
                )
                candidates.append((compressed_bytes, report, archive_path))
        if candidates:
            _, report, archive_path = min(candidates, key=lambda item: (item[0], str(item[2])))
            return report, archive_path
    raise FileNotFoundError(f"No completed audited archive is available for {domain}")


def iter_archive_documents(
    archive_path: Path,
    *,
    max_document_bytes: int,
) -> Iterator[tuple[str, str]]:
    """Yield bounded UTF-8 document members before the archive manifest."""

    manifest_seen = False
    with archive_path.open("rb") as raw_handle:
        reader = zstandard.ZstdDecompressor().stream_reader(
            raw_handle,
            read_across_frames=True,
            closefd=False,
        )
        try:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        raise ValueError(f"Unexpected non-file member {member.name!r}")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"Could not read member {member.name!r}")
                    if member.name == "_manifest.jsonl":
                        manifest_seen = True
                        break
                    content = extracted.read()
                    if len(content) != member.size:
                        raise IOError(
                            f"Short tar member {member.name!r}: {len(content)} != {member.size}"
                        )
                    if len(content) > max_document_bytes:
                        continue
                    yield member.name, content.decode("utf-8", errors="strict")
        finally:
            reader.close()
    # A consumer may stop early once it has enough tokens, in which case the
    # streaming reader intentionally never reaches the trailing manifest.
    if not manifest_seen:
        raise ValueError(f"Archive is missing its trailing manifest: {archive_path}")


def build_domain_smoke(
    *,
    root: Path,
    output: Path,
    domain: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
    sequence_length: int,
    min_rows: int,
    max_documents: int,
    max_document_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    report, archive_path = select_audited_archive(root, domain)
    domain_output = output / "packed" / domain
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("Pinned tokenizer is missing <|endoftext|>")
    writer = PackedShardWriter(
        domain_output,
        domain=domain,
        split="smoke",
        sequence_length=sequence_length,
        vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
        eos_token_id=int(eos_token_id),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        rows_per_shard=max(1, min_rows),
        construction_seed=0,
    )
    document_count = 0
    source_tokens = 0
    sampled_members: list[str] = []
    for member_name, text in iter_archive_documents(
        archive_path,
        max_document_bytes=max_document_bytes,
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not token_ids:
            continue
        writer.add_document(token_ids)
        document_count += 1
        source_tokens += len(token_ids)
        if len(sampled_members) < 3:
            sampled_members.append(member_name)
        if source_tokens + document_count >= min_rows * sequence_length + 1:
            break
        if document_count >= max_documents:
            break
    manifest = writer.finish()
    if manifest["rows"] < min_rows:
        raise RuntimeError(
            f"{archive_path} produced only {manifest['rows']} packed rows from "
            f"{document_count} bounded documents; raise --max-documents or "
            "--max-document-bytes"
        )
    validate_packed_manifest(domain_output / "manifest.json", verify_checksums=True)
    return domain_output / "manifest.json", {
        "raw_bucket": report["bucket"],
        "archive": report["archive"],
        "preprocess_report_tokens": report["exact_tokens"],
        "documents_sampled": document_count,
        "source_tokens_sampled": source_tokens,
        "packed_rows": manifest["rows"],
        "packed_input_tokens": manifest["input_tokens"],
        "valid_loss_tokens": manifest["valid_loss_tokens"],
        "sample_member_paths": sampled_members,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--min-rows-per-domain", type=int, default=2)
    parser.add_argument("--max-documents", type=int, default=256)
    parser.add_argument("--max-document-bytes", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--keep", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.sequence_length < 2 or args.min_rows_per_domain < 1:
            raise ValueError("sequence length and minimum rows must be positive")
        if args.max_documents < 1 or args.max_document_bytes < 1 or args.workers < 0:
            raise ValueError("document limits must be positive and workers non-negative")
        tokenizer_directory = args.root / "tokenizer" / "starcoder2"
        tokenizer_json = tokenizer_directory / "tokenizer.json"
        tokenizer_manifest = tokenizer_directory / "TOKENIZER_MANIFEST.json"
        if not tokenizer_json.is_file() or not tokenizer_manifest.is_file():
            raise FileNotFoundError(f"Missing pinned tokenizer under {tokenizer_directory}")
        tokenizer_payload = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
        if tokenizer_payload.get("validation", {}).get("vocab_size") != 49_152:
            raise ValueError("Pinned tokenizer manifest does not declare vocabulary 49,152")
        if tokenizer_payload.get("validation", {}).get("eos_token_id") != 0:
            raise ValueError("Pinned tokenizer manifest does not declare EOS ID 0")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-data.txt for the tokenizer smoke test") from exc
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        tokenizer_hash = file_sha256(tokenizer_manifest)

        temporary_parent = args.temporary_root or args.root / "staging"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        work_directory = Path(
            tempfile.mkdtemp(prefix="raw-to-training-smoke-", dir=temporary_parent)
        )
        summary: dict[str, Any] = {
            "work_directory": str(work_directory),
            "kept": bool(args.keep),
            "tokenizer_manifest_sha256": tokenizer_hash,
            "sequence_length": args.sequence_length,
            "domains": {},
        }
        loader = None
        sampler = None
        try:
            manifests: dict[str, Path] = {}
            for domain in DOMAIN_ORDER:
                manifest_path, domain_summary = build_domain_smoke(
                    root=args.root,
                    output=work_directory,
                    domain=domain,
                    tokenizer=tokenizer,
                    tokenizer_manifest_sha256=tokenizer_hash,
                    sequence_length=args.sequence_length,
                    min_rows=args.min_rows_per_domain,
                    max_documents=args.max_documents,
                    max_document_bytes=args.max_document_bytes,
                )
                manifests[domain] = manifest_path
                summary["domains"][domain] = domain_summary

            order_directory = work_directory / "order"
            build_training_order(
                manifests,
                order_directory,
                seed=0,
                frozen_global_microbatch_rows=1,
                frozen_gradient_accumulation_steps=1,
            )
            validate_training_order(order_directory / "manifest.json")
            loader, sampler = create_training_dataloader(
                order_directory / "manifest.json",
                global_microbatch_rows=1,
                num_workers=args.workers,
                pin_memory=False,
                persistent_workers=False,
                verify_payload_checksums=True,
            )
            batches = 0
            loaded_input_tokens = 0
            loaded_loss_tokens = 0
            for batch in loader:
                batches += 1
                loaded_input_tokens += int(batch["input_ids"].numel())
                loaded_loss_tokens += int(batch["num_loss_tokens"])
            summary["loader"] = {
                "batches": batches,
                "input_tokens": loaded_input_tokens,
                "valid_loss_tokens": loaded_loss_tokens,
                "workers": args.workers,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
        finally:
            if loader is not None:
                dataset = getattr(loader, "dataset", None)
                if dataset is not None and hasattr(dataset, "close"):
                    dataset.close()
            if sampler is not None:
                sampler.close()
            del loader
            gc.collect()
            if not args.keep:
                import shutil

                shutil.rmtree(work_directory)
        return 0
    except (FileNotFoundError, IOError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
