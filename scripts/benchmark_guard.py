#!/usr/bin/env python3
"""Content fingerprints used to keep evaluation benchmarks out of training data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


FUNCTION_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")


def normalized_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_code_digest(text: str) -> str:
    return digest("\n".join(normalized_lines(text)))


def code_pair_digests(text: str) -> set[str]:
    lines = normalized_lines(text)
    return {
        digest(f"{left}\n{right}")
        for left, right in zip(lines, lines[1:])
        if len(left) + len(right) >= 40
    }


def strong_line_digests(text: str) -> set[str]:
    return {digest(line) for line in normalized_lines(text) if len(line) >= 32}


def build_mbpp_manifest(rows: Iterable[dict[str, Any]], source_sha256: str, source_url: str) -> dict[str, Any]:
    exact_codes: set[str] = set()
    code_pairs: set[str] = set()
    strong_lines: set[str] = set()
    function_names: set[str] = set()
    count = 0
    for row in rows:
        count += 1
        code = str(row.get("code") or "")
        if code:
            exact_codes.add(normalized_code_digest(code))
            code_pairs.update(code_pair_digests(code))
            function_names.update(FUNCTION_RE.findall(code))
        text = str(row.get("text") or "")
        strong_lines.update(strong_line_digests(text))
        for field in ("test_list", "challenge_test_list"):
            for test in row.get(field) or []:
                strong_lines.update(strong_line_digests(str(test)))
    return {
        "manifest_version": 1,
        "benchmark": "MBPP",
        "source_url": source_url,
        "source_sha256": source_sha256,
        "items": count,
        "exact_code_sha256": sorted(exact_codes),
        "code_line_pair_sha256": sorted(code_pairs),
        "strong_line_sha256": sorted(strong_lines),
        "function_names": sorted(function_names),
    }


class BenchmarkGuard:
    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if payload.get("manifest_version") != 1 or payload.get("benchmark") != "MBPP":
            raise ValueError(f"Unsupported benchmark denylist: {path}")
        if int(payload.get("items", 0)) != 974:
            raise ValueError(f"Expected 974 canonical MBPP items in {path}")
        self.exact_codes = set(payload.get("exact_code_sha256") or [])
        self.code_pairs = set(payload.get("code_line_pair_sha256") or [])
        self.strong_lines = set(payload.get("strong_line_sha256") or [])
        self.function_names = set(payload.get("function_names") or [])
        if not self.exact_codes or not self.code_pairs or not self.strong_lines:
            raise ValueError(f"Incomplete benchmark denylist: {path}")

    def contamination_reason(self, language: str, content: str) -> str | None:
        if language not in ("Python", "English"):
            return None
        lowered = content.lower()
        if "mostly basic python problems" in lowered or re.search(r"\bmbpp\b", lowered):
            return "mbpp-marker"
        if all(marker in lowered for marker in ('"task_id"', '"test_list"', '"challenge_test_list"')):
            return "mbpp-json-shape"
        lines = normalized_lines(content)
        if digest("\n".join(lines)) in self.exact_codes:
            return "mbpp-exact-code"
        if {digest(line) for line in lines if len(line) >= 32} & self.strong_lines:
            return "mbpp-problem-or-test"
        defined = set(FUNCTION_RE.findall(content))
        pair_matches = len(
            {
                digest(f"{left}\n{right}")
                for left, right in zip(lines, lines[1:])
                if len(left) + len(right) >= 40
            }
            & self.code_pairs
        )
        if pair_matches >= 2 or (pair_matches >= 1 and defined & self.function_names):
            return "mbpp-embedded-code"
        return None
