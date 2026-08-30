#!/usr/bin/env python3
"""Benchmark one immutable raw archive without touching production staging."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import resource
import time
from pathlib import Path

from preprocess_raw_stream import (
    BUCKET_PATHS,
    DEFAULT_DENYLIST,
    ArchiveCandidate,
    file_sha256,
    fingerprint_path,
    initialize_worker,
    load_quota_records,
    process_archive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--bucket", choices=tuple(BUCKET_PATHS), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--analysis-batch-size", type=int, default=64)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--log-every-documents", type=int, default=100_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.analysis_batch_size < 1 or args.index < 0:
        raise SystemExit("workers/batch size must be positive and index non-negative")
    if args.staging_root.exists() and any(args.staging_root.iterdir()):
        raise SystemExit(f"Refusing non-empty benchmark staging root: {args.staging_root}")
    args.staging_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    quota = load_quota_records(args.root).get((args.bucket, args.index))
    if quota is None:
        raise SystemExit(f"No committed quota record for {args.bucket}/{args.index:06d}")
    archive = args.root / "raw" / BUCKET_PATHS[args.bucket] / f"part-{args.index:06d}.tar.zst"
    if not archive.is_file():
        raise SystemExit(f"Missing finalized archive: {archive}")
    candidate = ArchiveCandidate(
        bucket=args.bucket,
        index=args.index,
        path=archive,
        relative_path=str(archive.relative_to(args.root)),
        quota_record=quota,
    )
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(str(args.benchmark_denylist),),
    ) as executor:
        report = process_archive(
            candidate,
            args.staging_root,
            executor,
            args.log_every_documents,
            args.analysis_batch_size,
            args.scratch_root,
        )
    fingerprint = fingerprint_path(args.staging_root, args.bucket, args.index)
    verified_sha256 = file_sha256(fingerprint)
    if verified_sha256 != report["fingerprint_sha256"]:
        raise RuntimeError("Streaming fingerprint SHA-256 did not match a full reread")
    payload = {
        "benchmark_version": 1,
        "workers": args.workers,
        "analysis_batch_size": args.analysis_batch_size,
        "wall_seconds_including_pool_startup_and_checksum_verify": round(
            time.monotonic() - started, 3
        ),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "fingerprint_sha256_verified": True,
        "report": report,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
