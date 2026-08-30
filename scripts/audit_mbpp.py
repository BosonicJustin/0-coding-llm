#!/usr/bin/env python3
"""Audit raw Python archives for canonical MBPP content fingerprints."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from benchmark_guard import BenchmarkGuard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--denylist", type=Path, default=PROJECT_ROOT / "configs" / "mbpp_denylist.json")
    args = parser.parse_args()
    guard = BenchmarkGuard(args.denylist)
    archives = sorted((args.root / "raw" / "python").glob("part-*.tar.zst"))
    findings: list[dict[str, str]] = []
    files = 0
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Install zstandard before auditing archives") from exc
    for archive in archives:
        with archive.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as stream:
                    for member in stream:
                        if not member.isfile() or member.name == "_manifest.jsonl":
                            continue
                        extracted = stream.extractfile(member)
                        if extracted is None:
                            continue
                        files += 1
                        reason = guard.contamination_reason(
                            "Python", extracted.read().decode("utf-8")
                        )
                        if reason:
                            findings.append(
                                {
                                    "archive": str(archive),
                                    "member": member.name,
                                    "reason": reason,
                                }
                            )
    print(json.dumps({"archives": len(archives), "files": files, "findings": findings}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
