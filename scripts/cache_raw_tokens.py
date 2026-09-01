#!/usr/bin/env python3
"""Build or benchmark immutable per-archive raw-document token caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.raw_token_cache import (  # noqa: E402
    CacheConfig,
    RawTokenCacheError,
    discover_cache_jobs,
    load_cache_job,
    run_cache_jobs,
)


DEFAULT_ROOT = Path("/workspace/dataset")
DEFAULT_PREPROCESS_ROOT = DEFAULT_ROOT / "staging" / "preprocess"
DEFAULT_TOKENIZER_ROOT = DEFAULT_ROOT / "tokenizer" / "starcoder2"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "token-cache" / "raw-all-v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument(
        "--max-documents-per-archive", type=_positive_int, default=2_000_000
    )
    parser.add_argument(
        "--max-document-bytes", type=_positive_int, default=16 * 1024 * 1024
    )
    parser.add_argument(
        "--max-document-tokens", type=_positive_int, default=8 * 1024 * 1024
    )
    parser.add_argument("--batch-documents", type=_positive_int, default=64)
    parser.add_argument("--batch-bytes", type=_positive_int, default=32 * 1024 * 1024)
    parser.add_argument("--batch-tokens", type=_positive_int, default=2 * 1024 * 1024)
    parser.add_argument(
        "--minimum-free-gib",
        type=_positive_float,
        default=10.0,
        help="free-space reserve left after estimated pending outputs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="discover all finalized preprocess reports and build their caches"
    )
    _add_common(build)
    build.add_argument(
        "--max-archives",
        type=_positive_int,
        help="bounded invocation; process only the first N canonical reports",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="build exactly one complete archive and report end-to-end throughput",
    )
    _add_common(benchmark)
    benchmark.set_defaults(workers=1, minimum_free_gib=1.0)
    benchmark.add_argument(
        "--report",
        type=Path,
        required=True,
        help="canonical reports/<bucket>/part-NNNNNN.json to benchmark",
    )
    return parser


def _config(args: argparse.Namespace) -> CacheConfig:
    return CacheConfig(
        max_documents_per_archive=args.max_documents_per_archive,
        max_document_bytes=args.max_document_bytes,
        max_document_tokens=args.max_document_tokens,
        tokenizer_batch_documents=args.batch_documents,
        tokenizer_batch_bytes=args.batch_bytes,
        tokenizer_batch_tokens=args.batch_tokens,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = _config(args)
        if args.command == "benchmark":
            job = load_cache_job(
                args.root,
                args.preprocess_root,
                args.report,
                config=config,
            )
            target = job.target(args.output_root.absolute())
            if target.exists() or target.is_symlink():
                raise RawTokenCacheError(
                    "Benchmark target already exists; use a fresh output root so the "
                    "completed-cache verification fast path is not measured"
                )
            jobs = [job]
        else:
            jobs = discover_cache_jobs(
                args.root,
                args.preprocess_root,
                config=config,
            )
            if args.max_archives is not None:
                jobs = jobs[: args.max_archives]
        if not jobs:
            raise RawTokenCacheError("No finalized preprocess reports were discovered")
        started = time.monotonic()
        results = run_cache_jobs(
            jobs,
            args.output_root,
            args.tokenizer,
            config=config,
            workers=args.workers,
        )
        elapsed = max(time.monotonic() - started, 1e-9)
        summary = {
            "event": "raw_token_cache_complete",
            "mode": args.command,
            "archives": len(results),
            "built": sum(result.status == "built" for result in results),
            "verified": sum(result.status == "verified" for result in results),
            "documents": sum(result.documents for result in results),
            "content_tokens": sum(result.content_tokens for result in results),
            "elapsed_seconds": round(elapsed, 6),
            "content_tokens_per_second": round(
                sum(result.content_tokens for result in results) / elapsed, 3
            ),
            "results": [result.__dict__ for result in results],
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, RawTokenCacheError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
