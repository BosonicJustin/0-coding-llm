#!/usr/bin/env python3
"""Certify a complete raw-token cache against one immutable v7 selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.raw_token_cache import (  # noqa: E402
    CacheConfig,
    RawTokenCacheError,
    load_cache_job,
)
from pretrain.raw_token_cache_inventory import (  # noqa: E402
    InventorySource,
    RawTokenCacheInventoryError,
    publish_raw_token_cache_inventory,
)
from pretrain.raw_token_cache_reader import (  # noqa: E402
    ArchiveAuthority,
    FileAuthority,
    RawTokenCacheReadError,
    TokenizerAuthority,
)
from pretrain.selection_contract import (  # noqa: E402
    ALL_ELIGIBLE_BITMAP_FORMAT,
    ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
    ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
    ALL_ELIGIBLE_SELECTION_STRATEGY,
)
from pretrain.tokenizer_identity import (  # noqa: E402
    TokenizerIdentityError,
    verify_tokenizer_identity,
)


DEFAULT_ROOT = Path("/workspace/dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        help="default: ROOT/staging/preprocess",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        help="default: ROOT/curated/selection-v7",
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        help="default: ROOT/tokenizer/starcoder2",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="default: ROOT/token-cache/raw-all-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: ROOT/token-cache/inventories/selection-v7",
    )
    parser.add_argument("--expected-vocab-size", type=int, default=49_152)
    return parser


def _load_selection(selection_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = selection_root / "manifest.json"
    sidecar_path = selection_root / "manifest.sha256"
    if (
        selection_root.is_symlink()
        or not selection_root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sidecar_path.is_symlink()
        or not sidecar_path.is_file()
    ):
        raise RawTokenCacheInventoryError(
            f"Selection root lacks non-symlink manifest authority: {selection_root}"
        )
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected_sidecar = f"{digest}  manifest.json\n"
    if sidecar_path.read_text(encoding="ascii") != expected_sidecar:
        raise RawTokenCacheInventoryError("Selection manifest sidecar mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTokenCacheInventoryError(f"Invalid selection manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise RawTokenCacheInventoryError("Selection manifest root is not an object")
    identity = payload.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("format_version") != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
        or payload.get("production_ready") is not True
        or payload.get("selection_strategy") != ALL_ELIGIBLE_SELECTION_STRATEGY
        or payload.get("decision_format") != ALL_ELIGIBLE_BITMAP_FORMAT
        or payload.get("decision_format_version")
        != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
    ):
        raise RawTokenCacheInventoryError(
            "Cache inventory publication requires a production v7 all-eligible selection"
        )
    return payload, digest


def _tokenizer_authority(
    tokenizer_root: Path, *, expected_vocab_size: int
) -> TokenizerAuthority:
    identity = verify_tokenizer_identity(
        tokenizer_root, expected_vocab_size=expected_vocab_size
    )
    manifest = identity.manifest
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise RawTokenCacheInventoryError("Tokenizer manifest has no validation object")
    return TokenizerAuthority(
        repo_id=manifest.get("repo_id"),
        resolved_revision=manifest.get("resolved_revision"),
        manifest_sha256=identity.manifest_sha256,
        vocabulary_sha256=identity.vocabulary_sha256,
        vocab_size=identity.vocab_size,
        eos_token=validation.get("eos_token"),
        eos_token_id=validation.get("eos_token_id"),
    )


def _sources(
    *,
    root: Path,
    preprocess_root: Path,
    selection: dict[str, Any],
    expected_vocab_size: int,
) -> list[InventorySource]:
    reports = selection.get("input_reports")
    decisions = selection.get("decision_shards")
    if not isinstance(reports, list) or not reports or not isinstance(decisions, list):
        raise RawTokenCacheInventoryError(
            "Selection has no non-empty report/decision inventories"
        )
    report_by_archive: dict[str, dict[str, Any]] = {}
    for descriptor in reports:
        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("archive"), str
        ):
            raise RawTokenCacheInventoryError("Invalid selection report descriptor")
        archive = descriptor["archive"]
        if archive in report_by_archive:
            raise RawTokenCacheInventoryError("Duplicate selection report archive")
        report_by_archive[archive] = descriptor
    decision_archives = [
        descriptor.get("archive") if isinstance(descriptor, dict) else None
        for descriptor in decisions
    ]
    if decision_archives != sorted(report_by_archive):
        raise RawTokenCacheInventoryError(
            "Selection decisions are not the complete canonical archive order"
        )
    config = CacheConfig(
        expected_vocab_size=expected_vocab_size,
        minimum_free_bytes=1,
    )
    sources: list[InventorySource] = []
    for ordinal, decision in enumerate(decisions):
        assert isinstance(decision, dict)
        if (
            decision.get("format") != ALL_ELIGIBLE_BITMAP_FORMAT
            or decision.get("format_version")
            != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
        ):
            raise RawTokenCacheInventoryError(
                "Selection contains a non-v7 decision descriptor"
            )
        descriptor = report_by_archive[str(decision["archive"])]
        report_relative = descriptor.get("report")
        if not isinstance(report_relative, str):
            raise RawTokenCacheInventoryError("Selection report path is invalid")
        job = load_cache_job(
            root,
            preprocess_root,
            preprocess_root / report_relative,
            config=config,
        )
        expected = {
            "archive": job.archive_relative,
            "archive_sha256": job.archive_sha256,
            "report": job.report_relative,
            "report_sha256": job.report_sha256,
            "fingerprint_file": job.fingerprint_relative,
            "fingerprint_sha256": job.fingerprint_sha256,
            "documents": job.documents,
            "content_tokens": job.exact_tokens,
        }
        if any(descriptor.get(key) != value for key, value in expected.items()):
            raise RawTokenCacheInventoryError(
                f"Selection/report authority mismatch for {job.archive_relative}"
            )
        if decision.get("records") != job.documents:
            raise RawTokenCacheInventoryError(
                f"Decision/report record mismatch for {job.archive_relative}"
            )
        sources.append(
            InventorySource(
                ordinal=ordinal,
                archive=ArchiveAuthority(
                    path=job.archive_relative,
                    bucket=job.bucket,
                    index=job.index,
                    bytes=job.archive_compressed_bytes,
                    sha256=job.archive_sha256,
                ),
                preprocess_report=FileAuthority(
                    path=job.report_relative,
                    bytes=job.report_bytes,
                    sha256=job.report_sha256,
                ),
                fingerprint=FileAuthority(
                    path=job.fingerprint_relative,
                    bytes=job.fingerprint_path.stat().st_size,
                    sha256=job.fingerprint_sha256,
                ),
                records=job.documents,
                clean_bytes=job.clean_bytes,
                content_tokens=job.exact_tokens,
            )
        )
    return sources


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    preprocess_root = args.preprocess_root or args.root / "staging" / "preprocess"
    selection_root = args.selection_root or args.root / "curated" / "selection-v7"
    tokenizer_root = args.tokenizer_root or args.root / "tokenizer" / "starcoder2"
    cache_root = args.cache_root or args.root / "token-cache" / "raw-all-v1"
    output = args.output or args.root / "token-cache" / "inventories" / "selection-v7"
    try:
        selection, selection_sha = _load_selection(selection_root)
        tokenizer = _tokenizer_authority(
            tokenizer_root, expected_vocab_size=args.expected_vocab_size
        )
        sources = _sources(
            root=args.root,
            preprocess_root=preprocess_root,
            selection=selection,
            expected_vocab_size=args.expected_vocab_size,
        )
        inventory = publish_raw_token_cache_inventory(
            cache_root=cache_root,
            inventory_root=output,
            dataset_root=args.root,
            preprocess_root=preprocess_root,
            tokenizer_root=tokenizer_root,
            selection_manifest_sha256=selection_sha,
            selection_format_version=ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
            tokenizer=tokenizer,
            sources=sources,
        )
        print(
            json.dumps(
                {
                    "event": "raw_token_cache_inventory_published",
                    "output": str(output),
                    "selection_manifest_sha256": selection_sha,
                    "manifest_sha256": inventory.manifest_sha256,
                    "sidecar_sha256": inventory.sidecar_sha256,
                    "archives": len(inventory.entries),
                    "content_tokens": sum(
                        entry.authority.content_tokens for entry in inventory.entries
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        RawTokenCacheError,
        RawTokenCacheReadError,
        RawTokenCacheInventoryError,
        TokenizerIdentityError,
    ) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
