#!/usr/bin/env python3
"""Resumable, shard-level token quota tracking for the data pipeline.

Each record is a separate JSON file written with an atomic rename. This is more
robust on a network volume than a single frequently rewritten ledger, and a
stable shard ID makes retries idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data_quotas.json"
RECORD_VERSION = 1


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("version") != 1:
        raise ValueError(f"Unsupported quota config version: {config.get('version')!r}")
    if not isinstance(config.get("quotas"), list):
        raise ValueError("Quota config must contain a 'quotas' list")
    return config


def estimate_tokens(clean_bytes: int, category: str, config: dict[str, Any]) -> int:
    ratios = config.get("estimation", {}).get("bytes_per_token", {})
    try:
        bytes_per_token = float(ratios[category])
    except KeyError as exc:
        raise ValueError(f"No bytes_per_token estimate configured for {category!r}") from exc
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    return int(round(clean_bytes / bytes_per_token))


def _record_key(record: dict[str, Any]) -> str:
    identity = "\0".join(
        str(record.get(key) or "")
        for key in ("phase", "split", "category", "language_group", "shard_id")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return digest


def record_path(root: Path, record: dict[str, Any]) -> Path:
    return root / "state" / "quota_records" / record["phase"] / f"{_record_key(record)}.json"


def write_record(root: Path, record: dict[str, Any], replace: bool = False) -> tuple[Path, bool]:
    required = ("phase", "category", "shard_id")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"Record is missing required fields: {', '.join(missing)}")

    normalized = {"record_version": RECORD_VERSION, **record}
    for field in ("documents", "clean_bytes", "estimated_tokens", "exact_tokens"):
        if field in normalized and normalized[field] is not None:
            normalized[field] = int(normalized[field])
            if normalized[field] < 0:
                raise ValueError(f"{field} cannot be negative")

    path = record_path(root, normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalized, sort_keys=True, indent=2) + "\n"

    if path.exists():
        old_payload = path.read_text(encoding="utf-8")
        if old_payload == payload:
            return path, False
        if not replace:
            raise FileExistsError(
                f"A different record already exists for shard {record['shard_id']!r}: {path}. "
                "Use --replace only after verifying the old record is wrong."
            )

    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path, True


def iter_records(root: Path) -> Iterable[dict[str, Any]]:
    records_root = root / "state" / "quota_records"
    if not records_root.exists():
        return
    for path in sorted(records_root.glob("*/*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read quota record {path}: {exc}") from exc
        if record.get("record_version") != RECORD_VERSION:
            raise RuntimeError(f"Unsupported record version in {path}")
        yield record


def _matches(record: dict[str, Any], quota: dict[str, Any]) -> bool:
    for key in ("phase", "split", "category", "language_group"):
        if key in quota and record.get(key) != quota[key]:
            return False
    return True


def quota_status(
    root: Path, config: dict[str, Any], phase: str | None = None
) -> list[dict[str, Any]]:
    records = list(iter_records(root))
    result: list[dict[str, Any]] = []
    for quota in config["quotas"]:
        if phase is not None and quota.get("phase") != phase:
            continue
        token_field = quota["token_field"]
        matching = [record for record in records if _matches(record, quota)]
        current = sum(int(record.get(token_field) or 0) for record in matching)
        target = int(quota["target"])
        result.append(
            {
                **quota,
                "current": current,
                "remaining": max(0, target - current),
                "fraction": current / target if target else 1.0,
                "reached": current >= target,
                "records": len(matching),
            }
        )
    return result


def _human_tokens(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def print_status(rows: list[dict[str, Any]], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("No matching quotas.")
        return
    name_width = max(len(row["name"]) for row in rows)
    print(f"{'quota':<{name_width}}  {'current':>10}  {'target':>10}  {'done':>7}  status")
    for row in rows:
        percent = min(row["fraction"] * 100.0, 999.99)
        marker = "REACHED" if row["reached"] else "collect"
        print(
            f"{row['name']:<{name_width}}  {_human_tokens(row['current']):>10}  "
            f"{_human_tokens(row['target']):>10}  {percent:6.2f}%  {marker}"
        )


def command_record(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    estimated_tokens = args.estimated_tokens
    if estimated_tokens is None and args.clean_bytes is not None:
        estimated_tokens = estimate_tokens(args.clean_bytes, args.category, config)
    record = {
        "phase": args.phase,
        "split": args.split,
        "category": args.category,
        "language_group": args.language_group,
        "shard_id": args.shard_id,
        "source": args.source,
        "documents": args.documents,
        "clean_bytes": args.clean_bytes,
        "estimated_tokens": estimated_tokens,
        "exact_tokens": args.exact_tokens,
    }
    record = {key: value for key, value in record.items() if value is not None}
    path, created = write_record(args.root, record, replace=args.replace)
    print(f"{'recorded' if created else 'already recorded'}: {path}")
    return 0


def command_status(args: argparse.Namespace, check: bool) -> int:
    config = load_config(args.config)
    rows = quota_status(args.root, config, phase=args.phase)
    print_status(rows, as_json=args.json)
    if check:
        if not rows:
            return 2
        return 0 if all(row["reached"] for row in rows) else 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="atomically record one processed shard")
    record_parser.add_argument("--phase", choices=("collection", "final"), required=True)
    record_parser.add_argument("--split", choices=("train", "validation", "test"))
    record_parser.add_argument("--category", required=True)
    record_parser.add_argument("--language-group")
    record_parser.add_argument("--shard-id", required=True)
    record_parser.add_argument("--source")
    record_parser.add_argument("--documents", type=int)
    record_parser.add_argument("--clean-bytes", type=int)
    record_parser.add_argument("--estimated-tokens", type=int)
    record_parser.add_argument("--exact-tokens", type=int)
    record_parser.add_argument("--replace", action="store_true")
    record_parser.set_defaults(function=command_record)

    for name, help_text in (
        ("status", "show progress without affecting the exit status"),
        ("check", "exit 0 only when every matching quota has been reached"),
    ):
        status_parser = subparsers.add_parser(name, help=help_text)
        status_parser.add_argument("--phase", choices=("collection", "final"))
        status_parser.add_argument("--json", action="store_true")
        status_parser.set_defaults(
            function=lambda args, is_check=(name == "check"): command_status(args, is_check)
        )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
