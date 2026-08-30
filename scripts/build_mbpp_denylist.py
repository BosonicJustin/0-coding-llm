#!/usr/bin/env python3
"""Build a non-reversible MBPP content denylist from canonical JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_guard import build_mbpp_manifest


DEFAULT_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    manifest = build_mbpp_manifest(rows, hashlib.sha256(raw).hexdigest(), args.source_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"items": manifest["items"], "output": str(args.output), "source_sha256": manifest["source_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
