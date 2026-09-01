#!/usr/bin/env python3
"""Bounded CPU/memory benchmark for the v7 all-eligible keep bitmap encoder."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from publish_all_eligible_selection import build_all_eligible_keep_bitmap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--rejected-every", type=int, default=20)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.records < 1 or args.rejected_every < 1:
        parser.error("--records and --rejected-every must be positive")
    rejected = set(range(0, args.records, args.rejected_every))
    started = time.perf_counter()
    payload, kept = build_all_eligible_keep_bitmap(args.records, rejected)
    elapsed = time.perf_counter() - started
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    maximum_resident_set_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    result = {
        "format": "all-eligible-keep-bitmap-benchmark",
        "format_version": 1,
        "records": args.records,
        "rejected_documents": len(rejected),
        "kept_documents": kept,
        "bitmap_bytes": len(payload),
        "elapsed_seconds": elapsed,
        "records_per_second": args.records / elapsed,
        "maximum_resident_set_bytes": maximum_resident_set_bytes,
        "scope": "bitmap_encoding_only; excludes SQLite and source-file SHA-256 I/O",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.max_seconds is not None and elapsed > args.max_seconds:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
