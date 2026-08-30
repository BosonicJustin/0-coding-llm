#!/usr/bin/env python3
"""Build a complete, restart-safe cross-source English near-duplicate map.

The compact 16-value sketches emitted by ``preprocess_raw_stream.py`` are
sampled and too small to be a defensible production near-dedup authority.
They are validated here as integrity records, but are *not* used to decide
candidate recall.  Production candidates come from a second pass over the
immutable raw FineWeb-Edu and Wikipedia archives.  Candidate pairs are then
refined with the Jaccard similarity of the complete set of hashed five-word
shingles before deterministic disk-backed connected components are formed.

The authoritative restart journal is SQLite.  Work is committed one input
archive, candidate posting, refinement batch, or union batch at a time.  The
only public result is an atomically published ``clusters.jsonl.zst`` plus a
checksummed manifest.  Every English document, including singletons, appears
exactly once.
"""

from __future__ import annotations

import argparse
import array
import collections
import fcntl
import hashlib
import io
import json
import math
import os
import re
import resource
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Sequence

import xxhash
import zstandard

from benchmark_guard import BenchmarkGuard
from curation_policy import DEFAULT_POLICY, canonical_sha256, load_policy
from preprocess_raw_stream import (
    FINGERPRINT_VERSION,
    NEAR_SKETCH_SIZE,
    POLICY_SHA256,
    WORD_RE,
    bottom_k_sketch,
    file_sha256,
    near_features,
    normalize_content,
)


FORMAT_VERSION = 1
DATABASE_VERSION = 1
MAPPING_RECORD_VERSION = 1
DEFAULT_ROOT = Path("/workspace/dataset")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "english_near_dedup.json"
DEFAULT_CALIBRATION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "english_near_dedup_calibration.json"
)
CALIBRATION_HARNESS = Path(__file__).resolve().with_name(
    "calibrate_english_near_dedup.py"
)
DEFAULT_DENYLIST = Path(__file__).resolve().parents[1] / "configs" / "mbpp_denylist.json"
DEFAULT_QUOTA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data_quotas.json"
ENGLISH_BUCKETS = ("fineweb_edu", "wikipedia")
HEX = frozenset("0123456789abcdef")
MOUNT_ESCAPE_RE = re.compile(r"\\([0-7]{3})")
CANDIDATE_DOCUMENTS_FOR_ARCHIVE_SQL = """
    SELECT c.left_document
    FROM documents AS d INDEXED BY documents_archive_ordinal
    JOIN candidates AS c ON c.left_document=d.ordinal
    WHERE d.archive=?
    UNION
    SELECT c.right_document
    FROM documents AS d INDEXED BY documents_archive_ordinal
    JOIN candidates AS c INDEXED BY candidates_by_right
      ON c.right_document=d.ordinal
    WHERE d.archive=?
"""
REFINEMENT_BATCH_SQL = """
    SELECT left_document,right_document FROM candidates
    WHERE (left_document,right_document) > (?,?)
    ORDER BY left_document,right_document LIMIT ?
"""
NEXT_CANDIDATE_BLOCK_SQL = """
    SELECT band,band_key,COUNT(*) AS documents
    FROM bands
    WHERE (band,band_key) > (?,?)
    GROUP BY band,band_key HAVING COUNT(*) > 1
    ORDER BY band,band_key LIMIT 1
"""
UNION_PARENT_BATCH_SQL = """
    SELECT document FROM parents
    WHERE document>? ORDER BY document LIMIT ?
"""
UNION_DOCUMENT_BATCH_SQL = """
    SELECT ordinal FROM documents
    WHERE ordinal>? ORDER BY ordinal LIMIT ?
"""


class EnglishNearDedupError(RuntimeError):
    """An integrity, identity, completeness, or fail-closed safety error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def safe_relative_path(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EnglishNearDedupError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise EnglishNearDedupError(f"Unsafe {field}: {value!r}")
    resolved_base = base.resolve()
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(resolved_base):
        raise EnglishNearDedupError(f"Symlink escape in {field}: {value!r}")
    return resolved


def _unescape_mount_field(value: str) -> str:
    return MOUNT_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise EnglishNearDedupError(f"Cannot locate an existing parent for {path}")
    return candidate


def _device_mount_point(path: Path) -> Path:
    current = path.resolve(strict=True)
    device = current.stat().st_dev
    while current != current.parent and current.parent.stat().st_dev == device:
        current = current.parent
    return current


def parse_linux_mountinfo(text: str, path: Path) -> dict[str, str] | None:
    """Return longest-prefix mount evidence from Linux mountinfo text."""
    resolved = path.resolve(strict=False)
    matches: list[tuple[int, dict[str, str]]] = []
    for line in text.splitlines():
        try:
            left, right = line.split(" - ", 1)
            left_fields = left.split()
            right_fields = right.split()
            mount_point = Path(_unescape_mount_field(left_fields[4])).resolve(
                strict=False
            )
            if resolved != mount_point and not resolved.is_relative_to(mount_point):
                continue
            evidence = {
                "filesystem_type": right_fields[0].lower(),
                "mount_point": str(mount_point),
                "mount_source": _unescape_mount_field(right_fields[1]),
                "mount_options": left_fields[5],
                "detection": "linux-proc-self-mountinfo",
            }
            matches.append((len(mount_point.parts), evidence))
        except (IndexError, ValueError, OSError):
            continue
    return max(matches, key=lambda item: item[0])[1] if matches else None


def parse_bsd_mount_output(
    text: str, *, mount_source: str, mount_point: str
) -> dict[str, str] | None:
    """Resolve BSD/macOS ``mount`` output, preferring the source from df."""
    candidates: list[tuple[int, dict[str, str]]] = []
    for line in text.splitlines():
        match = re.fullmatch(r"(.+?) on (.+?) \(([^,()]+)(?:, (.*))?\)", line)
        if match is None:
            continue
        source, mounted_at, filesystem_type, options = match.groups()
        if source != mount_source and mounted_at != mount_point:
            continue
        evidence = {
            "filesystem_type": filesystem_type.casefold(),
            "mount_point": mounted_at,
            "mount_source": source,
            "mount_options": options or "",
            "detection": "darwin-df-and-mount",
        }
        candidates.append((1 if source == mount_source else 0, evidence))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def detect_filesystem(path: Path) -> dict[str, str]:
    probe = _nearest_existing_path(path)
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.is_file():
        evidence = parse_linux_mountinfo(
            mountinfo.read_text(encoding="utf-8", errors="strict"), probe
        )
        if evidence is not None:
            return evidence
    if sys.platform == "darwin":
        try:
            df_result = subprocess.run(
                ["df", "-P", str(probe)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            fields = df_result.stdout.splitlines()[-1].split()
            mount_result = subprocess.run(
                ["mount"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            evidence = parse_bsd_mount_output(
                mount_result.stdout,
                mount_source=fields[0],
                mount_point=fields[-1],
            )
            if evidence is not None:
                return evidence
        except (IndexError, OSError, subprocess.SubprocessError):
            pass
    inferred_mount = _device_mount_point(probe)
    return {
        "filesystem_type": "unknown",
        "mount_point": str(inferred_mount),
        "mount_source": "unknown",
        "mount_options": "unknown",
        "detection": "unavailable-fail-safe",
    }


def select_sqlite_journal_mode(
    filesystem_type: str,
    requested: str,
    local_allowlist: Sequence[str],
) -> tuple[str, str]:
    """Choose a journal mode without ever placing WAL on unproven storage."""
    normalized_type = filesystem_type.casefold()
    allowlist = {item.casefold() for item in local_allowlist}
    if requested not in {"auto", "wal", "delete"}:
        raise EnglishNearDedupError(f"Unsupported SQLite journal request: {requested}")
    if normalized_type in allowlist:
        return ("wal" if requested in {"auto", "wal"} else "delete"), "proven-local"
    if requested == "wal":
        raise EnglishNearDedupError(
            f"Refusing SQLite WAL on non-allowlisted filesystem {filesystem_type!r}; "
            "use DELETE or local storage"
        )
    classification = "unknown" if normalized_type == "unknown" else "non-local"
    return "delete", classification


def parse_digest(value: Any, field: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise EnglishNearDedupError(f"{field} must be a lowercase SHA-256")
    return bytes.fromhex(value)


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        try:
            for line_number, line in enumerate(text, 1):
                if not line.strip():
                    raise EnglishNearDedupError(f"Blank JSONL row in {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EnglishNearDedupError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EnglishNearDedupError(f"Non-object row in {path}:{line_number}")
                yield row
        finally:
            text.close()


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnglishNearDedupError(f"Missing English near-dedup config: {path}") from exc
    if not isinstance(config, dict) or config.get("config_version") != 1:
        raise EnglishNearDedupError("Unsupported English near-dedup config")
    if config.get("algorithm") != "raw-text-doph-lsh-plus-full-shingle-jaccard-v1":
        raise EnglishNearDedupError("Unexpected English near-dedup algorithm")
    if config.get("normalization") != "NFKC+casefold+collapse-whitespace":
        raise EnglishNearDedupError("English normalization must match preprocessing")
    if config.get("shingle_words") != 5:
        raise EnglishNearDedupError("Only pinned five-word shingles are supported")
    candidate = config.get("candidate_signature")
    if not isinstance(candidate, dict):
        raise EnglishNearDedupError("candidate_signature must be an object")
    tables = candidate.get("tables")
    if not isinstance(tables, list) or not tables:
        raise EnglishNearDedupError("At least one candidate signature table is required")
    total_bands = 0
    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            raise EnglishNearDedupError(f"Candidate table {index} must be an object")
        bins = table.get("bins")
        bands = table.get("bands")
        rows = table.get("rows_per_band")
        if not all(isinstance(item, int) and item > 0 for item in (bins, bands, rows)):
            raise EnglishNearDedupError(f"Invalid candidate table {index} dimensions")
        if bins != bands * rows:
            raise EnglishNearDedupError(f"Candidate table {index} bins != bands * rows")
        for field in ("hash_seed", "densify_seed"):
            value = table.get(field)
            if not isinstance(value, int) or not 0 <= value < (1 << 64):
                raise EnglishNearDedupError(f"Invalid candidate table {index} {field}")
        total_bands += bands
    for field in (
        "band_key_seed",
        "maximum_posting_documents",
        "maximum_unique_candidate_pairs",
    ):
        value = candidate.get(field)
        if not isinstance(value, int) or value < 1:
            raise EnglishNearDedupError(f"Invalid candidate_signature.{field}")
    if candidate.get("posting_overflow_action") != "fail_closed":
        raise EnglishNearDedupError("Oversized candidate postings must fail closed")
    refinement = config.get("refinement")
    if not isinstance(refinement, dict):
        raise EnglishNearDedupError("refinement must be an object")
    numerator = refinement.get("minimum_jaccard_numerator")
    denominator = refinement.get("minimum_jaccard_denominator")
    if not all(isinstance(value, int) and value > 0 for value in (numerator, denominator)):
        raise EnglishNearDedupError("Invalid exact Jaccard threshold")
    if numerator > denominator:
        raise EnglishNearDedupError("Jaccard threshold cannot exceed one")
    seed = refinement.get("shingle_hash_seed")
    if not isinstance(seed, int) or not 0 <= seed < (1 << 64):
        raise EnglishNearDedupError("Invalid refinement shingle seed")
    preflight = config.get("operational_preflight")
    if not isinstance(preflight, dict) or preflight.get("contract_version") != 1:
        raise EnglishNearDedupError("Invalid operational_preflight contract")
    if (
        preflight.get("sampling_algorithm")
        != "deterministic-keyspace-stratified-successor-v1"
    ):
        raise EnglishNearDedupError("Unsupported operational preflight sampler")
    integer_fields = (
        "requested_pairs",
        "strata",
        "successors_per_probe",
        "maximum_probe_passes",
        "maximum_projected_refinement_seconds",
        "maximum_peak_process_rss_bytes",
        "union_parent_memory_safety_numerator",
        "union_parent_memory_safety_denominator",
        "disk_projection_safety_numerator",
        "disk_projection_safety_denominator",
        "minimum_post_refinement_free_bytes",
    )
    for field in integer_fields:
        value = preflight.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise EnglishNearDedupError(
                f"Invalid operational_preflight.{field}"
            )
    sampling_seed = preflight.get("sampling_seed")
    if (
        not isinstance(sampling_seed, int)
        or isinstance(sampling_seed, bool)
        or not 0 <= sampling_seed < (1 << 64)
    ):
        raise EnglishNearDedupError(
            "Invalid operational_preflight.sampling_seed"
        )
    minimum_rate = preflight.get("minimum_production_pairs_per_second")
    if (
        not isinstance(minimum_rate, (int, float))
        or isinstance(minimum_rate, bool)
        or not math.isfinite(float(minimum_rate))
        or float(minimum_rate) <= 0.0
    ):
        raise EnglishNearDedupError(
            "Invalid operational_preflight.minimum_production_pairs_per_second"
        )
    if (
        preflight["disk_projection_safety_numerator"]
        < preflight["disk_projection_safety_denominator"]
    ):
        raise EnglishNearDedupError(
            "Operational preflight disk safety multiplier must be at least one"
        )
    if (
        preflight["union_parent_memory_safety_numerator"]
        < preflight["union_parent_memory_safety_denominator"]
    ):
        raise EnglishNearDedupError(
            "Operational preflight union-memory safety multiplier must be at least one"
        )
    storage = config.get("storage")
    if not isinstance(storage, dict):
        raise EnglishNearDedupError("storage must be an object")
    if storage.get("sqlite_journal_mode") not in {"auto", "wal", "delete"}:
        raise EnglishNearDedupError("Invalid storage.sqlite_journal_mode")
    allowlist = storage.get("wal_local_filesystem_allowlist")
    if (
        not isinstance(allowlist, list)
        or not allowlist
        or allowlist != sorted(set(allowlist))
        or not all(isinstance(value, str) and value for value in allowlist)
    ):
        raise EnglishNearDedupError("Invalid WAL filesystem allowlist")
    if storage.get("network_or_unknown_action") != "delete":
        raise EnglishNearDedupError("Network/unknown filesystems must use DELETE")
    if storage.get("wal_on_non_allowlisted_action") != "fail_closed":
        raise EnglishNearDedupError("WAL on non-allowlisted filesystems must fail closed")
    config["_total_bands"] = total_bands
    return config


def shingle_bytes(text: str, width: int = 5) -> Iterator[bytes]:
    normalized = normalize_content(text, "fineweb_edu")
    words = WORD_RE.findall(normalized)
    if len(words) < width:
        for word in words:
            yield word.encode("utf-8")
        return
    for index in range(len(words) - width + 1):
        yield " ".join(words[index : index + width]).encode("utf-8")


def exact_shingle_hashes(text: str, seed: int, width: int = 5) -> list[int]:
    return sorted(
        {
            xxhash.xxh3_64_intdigest(shingle, seed=seed)
            for shingle in shingle_bytes(text, width)
        }
    )


def densified_one_permutation_signature(
    shingles: Iterable[bytes], *, bins: int, hash_seed: int, densify_seed: int
) -> tuple[int, ...]:
    """Return an aligned one-permutation signature with deterministic densification."""
    values: list[int | None] = [None] * bins
    for shingle in shingles:
        value = xxhash.xxh3_64_intdigest(shingle, seed=hash_seed)
        bucket = value % bins
        rank = value // bins
        current = values[bucket]
        if current is None or rank < current:
            values[bucket] = rank
    nonempty = [index for index, value in enumerate(values) if value is not None]
    if not nonempty:
        # Empty documents are quality-rejected later but must still map safely.
        return tuple(
            xxhash.xxh3_64_intdigest(struct.pack(">H", index), seed=densify_seed)
            for index in range(bins)
        )
    for index, value in enumerate(values):
        if value is not None:
            continue
        offset = 1
        while values[(index + offset) % bins] is None:
            offset += 1
        donor = (index + offset) % bins
        donor_value = int(values[donor])
        values[index] = xxhash.xxh3_64_intdigest(
            struct.pack(">HHQ", index, donor, donor_value), seed=densify_seed
        )
    return tuple(int(value) for value in values)


def candidate_bands(text: str, config: dict[str, Any]) -> tuple[list[tuple[int, bytes]], int]:
    """Build raw-text LSH bands and the complete refinement shingle cardinality."""
    width = int(config["shingle_words"])
    refinement = config["refinement"]
    exact_values: set[int] = set()
    table_bins: list[list[int | None]] = [
        [None] * int(table["bins"])
        for table in config["candidate_signature"]["tables"]
    ]
    for shingle in shingle_bytes(text, width):
        exact_values.add(
            xxhash.xxh3_64_intdigest(
                shingle, seed=int(refinement["shingle_hash_seed"])
            )
        )
        for table, values in zip(
            config["candidate_signature"]["tables"], table_bins, strict=True
        ):
            hashed = xxhash.xxh3_64_intdigest(shingle, seed=int(table["hash_seed"]))
            bucket = hashed % len(values)
            rank = hashed // len(values)
            current = values[bucket]
            if current is None or rank < current:
                values[bucket] = rank

    def densify(values: list[int | None], seed: int) -> tuple[int, ...]:
        nonempty = [index for index, value in enumerate(values) if value is not None]
        if not nonempty:
            return tuple(
                xxhash.xxh3_64_intdigest(struct.pack(">H", index), seed=seed)
                for index in range(len(values))
            )
        for index, value in enumerate(values):
            if value is not None:
                continue
            offset = 1
            while values[(index + offset) % len(values)] is None:
                offset += 1
            donor = (index + offset) % len(values)
            values[index] = xxhash.xxh3_64_intdigest(
                struct.pack(">HHQ", index, donor, int(values[donor])), seed=seed
            )
        return tuple(int(value) for value in values)

    result: list[tuple[int, bytes]] = []
    global_band = 0
    key_seed = int(config["candidate_signature"]["band_key_seed"])
    for table_index, (table, values) in enumerate(
        zip(config["candidate_signature"]["tables"], table_bins, strict=True)
    ):
        signature = densify(values, int(table["densify_seed"]))
        rows = int(table["rows_per_band"])
        for band in range(int(table["bands"])):
            values = signature[band * rows : (band + 1) * rows]
            payload = struct.pack(">HH", table_index, band) + b"".join(
                struct.pack(">Q", value) for value in values
            )
            result.append(
                (
                    global_band,
                    xxhash.xxh3_64_digest(
                        payload, seed=(key_seed + global_band) % (1 << 64)
                    ),
                )
            )
            global_band += 1
    if global_band != int(config["_total_bands"]):
        raise AssertionError("candidate band count changed")
    return result, len(exact_values)


def encode_uint64s(values: Sequence[int]) -> bytes:
    packed = array.array("Q", values)
    if sys.byteorder != "big":
        packed.byteswap()
    return packed.tobytes()


def decode_uint64s(payload: bytes) -> array.array[int]:
    if len(payload) % 8:
        raise EnglishNearDedupError("Corrupt cached shingle payload")
    values = array.array("Q")
    values.frombytes(payload)
    if sys.byteorder != "big":
        values.byteswap()
    return values


def jaccard_counts(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    left_index = right_index = intersection = 0
    while left_index < len(left) and right_index < len(right):
        left_value = left[left_index]
        right_value = right[right_index]
        if left_value == right_value:
            intersection += 1
            left_index += 1
            right_index += 1
        elif left_value < right_value:
            left_index += 1
        else:
            right_index += 1
    return intersection, len(left) + len(right) - intersection


def refinement_accepts(
    intersection: int, union: int, config: dict[str, Any]
) -> bool:
    """Apply the pinned production threshold with integer-only arithmetic."""
    numerator = int(config["refinement"]["minimum_jaccard_numerator"])
    denominator = int(config["refinement"]["minimum_jaccard_denominator"])
    return union == 0 or intersection * denominator >= union * numerator


@dataclass(frozen=True)
class ReportInput:
    relative_report: str
    report_path: Path
    report_sha256: str
    report: dict[str, Any]


class EnglishNearDedupBuilder:
    def __init__(
        self,
        *,
        root: Path,
        staging_root: Path,
        output: Path,
        config_path: Path = DEFAULT_CONFIG,
        policy_path: Path = DEFAULT_POLICY,
        denylist_path: Path = DEFAULT_DENYLIST,
        quota_config_path: Path = DEFAULT_QUOTA_CONFIG,
        batch_size: int = 5_000,
        progress_interval_seconds: int = 60,
        sqlite_journal_mode: str | None = None,
        calibration_result_path: Path | None = None,
        identity_probe_only: bool = False,
    ) -> None:
        if not 1 <= batch_size <= 100_000:
            raise EnglishNearDedupError("batch_size must be between 1 and 100,000")
        if not 1 <= progress_interval_seconds <= 3_600:
            raise EnglishNearDedupError(
                "progress_interval_seconds must be between 1 and 3,600"
            )
        self.root = root
        self.staging_root = staging_root
        self.output = output
        self.config_path = config_path
        self.policy_path = policy_path
        self.denylist_path = denylist_path
        self.quota_config_path = quota_config_path
        self.batch_size = batch_size
        self.progress_interval_seconds = progress_interval_seconds
        if not isinstance(identity_probe_only, bool):
            raise EnglishNearDedupError("identity_probe_only must be boolean")
        if identity_probe_only and calibration_result_path is not None:
            raise EnglishNearDedupError(
                "An identity-only calibration probe cannot consume calibration evidence"
            )
        self.identity_probe_only = identity_probe_only
        self.calibration_result_path = calibration_result_path
        self.config = load_config(config_path)
        self.config_file_sha = file_sha256(config_path)
        self.config_sha = canonical_sha256(
            {key: value for key, value in self.config.items() if not key.startswith("_")}
        )
        self.sqlite_journal_mode_override = sqlite_journal_mode
        self.storage_evidence = self._current_storage_evidence()
        self.policy = load_policy(policy_path)
        self.policy_sha = canonical_sha256(self.policy)
        self.guard = BenchmarkGuard(denylist_path)
        self.preprocess_manifest_path = staging_root / "PREPROCESS_MANIFEST.json"
        self.preprocess_manifest = self._validate_preprocess_manifest()
        self.preprocess_manifest_sha = file_sha256(self.preprocess_manifest_path)
        self.sources, self.source_manifests = self._load_source_identities()
        self.collection_completeness = self._load_collection_completeness()
        self.reports = self._freeze_reports()
        self._validate_report_completeness()
        self.report_inventory_sha = hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "path": item.relative_report,
                        "sha256": item.report_sha256,
                        "fingerprint_sha256": item.report["fingerprint_sha256"],
                    }
                    for item in self.reports
                ]
            )
        ).hexdigest()
        self.calibration_evidence = (
            None
            if self.identity_probe_only
            else self._load_calibration_evidence(calibration_result_path)
        )
        if self.calibration_evidence is not None:
            self.calibration_result_path = self.root / str(
                self.calibration_evidence["result_path"]
            )
        self.identity = {
            "format_version": FORMAT_VERSION,
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "config_file_sha256": self.config_file_sha,
            "config_sha256": self.config_sha,
            "curation_policy_sha256": self.policy_sha,
            "preprocess_manifest_sha256": self.preprocess_manifest_sha,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "report_inventory_sha256": self.report_inventory_sha,
            "report_count": len(self.reports),
            "source_manifests": self.source_manifests,
            "collection_completeness": self.collection_completeness,
            "calibration_evidence": self.calibration_evidence,
            "runtime": {
                "python": sys.version.split()[0],
                "sqlite": sqlite3.sqlite_version,
                "xxhash": str(getattr(xxhash, "__version__", getattr(xxhash, "VERSION", "unknown"))),
                "zstandard": zstandard.__version__,
                "storage": self.storage_evidence,
            },
        }
        self.work = output / ".work"
        self.db_path = self.work / "english_near.sqlite3"
        self.connection: sqlite3.Connection | None = None
        self.lock_handle: BinaryIO | None = None
        self._last_progress_at = 0.0
        self.operational_metrics: dict[str, dict[str, Any]] = {}
        self.refinement_preflight_evidence: dict[str, Any] | None = None

    def _current_storage_evidence(self) -> dict[str, Any]:
        configured_journal = str(self.config["storage"]["sqlite_journal_mode"])
        requested_journal = (
            self.sqlite_journal_mode_override or configured_journal
        )
        filesystem = detect_filesystem(self.output)
        selected_journal, storage_classification = select_sqlite_journal_mode(
            filesystem["filesystem_type"],
            requested_journal,
            self.config["storage"]["wal_local_filesystem_allowlist"],
        )
        return {
            **filesystem,
            "classification": storage_classification,
            "sqlite_journal_mode_configured": configured_journal,
            "sqlite_journal_mode_requested": requested_journal,
            "sqlite_journal_mode_request_source": (
                "cli"
                if self.sqlite_journal_mode_override is not None
                else "config"
            ),
            "sqlite_journal_mode_selected": selected_journal,
            "policy": {
                "network_or_unknown_action": "delete",
                "wal_on_non_allowlisted_action": "fail_closed",
                "wal_local_filesystem_allowlist": self.config["storage"][
                    "wal_local_filesystem_allowlist"
                ],
            },
        }

    @staticmethod
    def _json_without_duplicate_keys(raw: bytes, label: str) -> dict[str, Any]:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise EnglishNearDedupError(
                        f"Duplicate key {key!r} in {label}"
                    )
                result[key] = value
            return result

        try:
            payload = json.loads(raw, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as exc:
            raise EnglishNearDedupError(f"Invalid {label}: {exc}") from exc
        if not isinstance(payload, dict):
            raise EnglishNearDedupError(f"{label} must be a JSON object")
        return payload

    def _load_calibration_evidence(
        self, calibration_result_path: Path | None
    ) -> dict[str, Any]:
        if calibration_result_path is None:
            raise EnglishNearDedupError(
                "Missing required production calibration result; pass "
                "--calibration-result"
            )
        if calibration_result_path.is_symlink() or not calibration_result_path.is_file():
            raise EnglishNearDedupError(
                f"Calibration result must be a regular file: {calibration_result_path}"
            )
        result_path = calibration_result_path.resolve(strict=True)
        root = self.root.resolve(strict=True)
        if not result_path.is_relative_to(root):
            raise EnglishNearDedupError(
                "Calibration result must live under the immutable dataset root"
            )
        checksum_path = result_path.with_name(result_path.name + ".sha256")
        if checksum_path.is_symlink() or not checksum_path.is_file():
            raise EnglishNearDedupError(
                f"Missing or unsafe calibration checksum: {checksum_path}"
            )
        result_raw = result_path.read_bytes()
        result_sha = hashlib.sha256(result_raw).hexdigest()
        expected_sidecar = f"{result_sha}  {result_path.name}\n".encode("utf-8")
        sidecar_raw = checksum_path.read_bytes()
        if sidecar_raw != expected_sidecar:
            raise EnglishNearDedupError("Calibration result checksum mismatch")
        result = self._json_without_duplicate_keys(
            result_raw, "English near-dedup calibration result"
        )
        required_gate = {
            "result_version": 1,
            "status": "pass",
            "production_configuration_unchanged": True,
            "production_gate_eligible": True,
            "production_gate_noneligibility_reasons": [],
            "acceptance_profile": "pinned-production",
            "sampling_profile": "pinned-production",
            "acceptance_overrides": {},
            "acceptance_failures": [],
        }
        for field, expected in required_gate.items():
            if result.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Calibration result {field} is not production-passing"
                )

        identity = result.get("identity")
        if not isinstance(identity, dict):
            raise EnglishNearDedupError("Calibration result identity is missing")
        if not CALIBRATION_HARNESS.is_file():
            raise EnglishNearDedupError(
                f"Missing calibration harness: {CALIBRATION_HARNESS}"
            )
        calibration_raw = DEFAULT_CALIBRATION_CONFIG.read_bytes()
        calibration_config = self._json_without_duplicate_keys(
            calibration_raw, "English near-dedup calibration config"
        )
        expected_identity = {
            "harness_sha256": file_sha256(CALIBRATION_HARNESS),
            "production_builder_sha256": file_sha256(Path(__file__).resolve()),
            "calibration_algorithm": calibration_config.get(
                "calibration_algorithm"
            ),
            "calibration_seed": calibration_config.get("seed"),
            "production_config_file_sha256": self.config_file_sha,
            "production_config_canonical_sha256": self.config_sha,
            "calibration_config_file_sha256": hashlib.sha256(
                calibration_raw
            ).hexdigest(),
            "calibration_config_canonical_sha256": canonical_sha256(
                calibration_config
            ),
        }
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Calibration identity mismatch for {field}"
                )

        input_identity = identity.get("input")
        if not isinstance(input_identity, dict):
            raise EnglishNearDedupError("Calibration input identity is missing")
        completeness_sha = hashlib.sha256(
            canonical_json_bytes(self.collection_completeness)
        ).hexdigest()
        expected_input_identity = {
            "kind": "immutable_real_english_sample",
            "full_report_inventory_sha256": self.report_inventory_sha,
            "preprocess_manifest_sha256": self.preprocess_manifest_sha,
            "curation_policy_sha256": self.policy_sha,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "source_manifests": self.source_manifests,
            "collection_completeness_sha256": completeness_sha,
            "collection_completeness": self.collection_completeness,
        }
        for field, expected in expected_input_identity.items():
            if input_identity.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Calibration input identity mismatch for {field}"
                )

        sampling = calibration_config.get("sampling")
        if not isinstance(sampling, dict):
            raise EnglishNearDedupError("Pinned calibration sampling config is invalid")
        expected_sampling = {
            "selection_seed": calibration_config.get("seed"),
            "maximum_archives_per_bucket": sampling.get(
                "maximum_archives_per_bucket"
            ),
            "maximum_documents_per_bucket": sampling.get(
                "maximum_documents_per_bucket"
            ),
            "minimum_source_words": sampling.get("minimum_source_words"),
        }
        for field, expected in expected_sampling.items():
            if input_identity.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Calibration sampling identity mismatch for {field}"
                )

        selected_reports = input_identity.get("selected_reports")
        if not isinstance(selected_reports, list) or not selected_reports:
            raise EnglishNearDedupError(
                "Calibration input identity has no selected reports"
            )
        current_reports = {
            str(item.report["archive"]): {
                "report_path": item.relative_report,
                "report_sha256": item.report_sha256,
                "archive": item.report["archive"],
                "archive_sha256": item.report["archive_sha256"],
                "fingerprint_file": item.report["fingerprint_file"],
                "fingerprint_sha256": item.report["fingerprint_sha256"],
                "documents": item.report["documents"],
            }
            for item in self.reports
        }
        seen_selected: set[str] = set()
        for selected in selected_reports:
            if not isinstance(selected, dict):
                raise EnglishNearDedupError(
                    "Calibration selected-report identity must be an object"
                )
            archive = selected.get("archive")
            if (
                not isinstance(archive, str)
                or archive in seen_selected
                or current_reports.get(archive) != selected
            ):
                raise EnglishNearDedupError(
                    "Calibration selected-report identity mismatch"
                )
            seen_selected.add(archive)

        acceptance = calibration_config.get("acceptance")
        if result.get("acceptance") != acceptance:
            raise EnglishNearDedupError("Calibration acceptance policy mismatch")
        expected_threshold = {
            "minimum_jaccard_numerator": self.config["refinement"][
                "minimum_jaccard_numerator"
            ],
            "minimum_jaccard_denominator": self.config["refinement"][
                "minimum_jaccard_denominator"
            ],
        }
        if result.get("production_threshold") != expected_threshold:
            raise EnglishNearDedupError("Calibration production threshold mismatch")
        sampling_result = result.get("sampling")
        if not isinstance(sampling_result, dict):
            raise EnglishNearDedupError("Calibration sampling result is missing")
        sample_manifest = sampling_result.get("sample_manifest")
        if not isinstance(sample_manifest, list) or not sample_manifest:
            raise EnglishNearDedupError("Calibration sample manifest is missing")
        if hashlib.sha256(canonical_json_bytes(sample_manifest)).hexdigest() != identity.get(
            "sample_manifest_sha256"
        ):
            raise EnglishNearDedupError("Calibration sample manifest identity mismatch")
        if sampling_result.get("documents_input") != input_identity.get(
            "documents_selected"
        ):
            raise EnglishNearDedupError("Calibration selected-document count mismatch")

        relative_result = str(result_path.relative_to(root))
        relative_sidecar = str(checksum_path.relative_to(root))
        return {
            "contract_version": 1,
            "result_path": relative_result,
            "result_sha256": result_sha,
            "result_bytes": len(result_raw),
            "sidecar_path": relative_sidecar,
            "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
            "result_version": result["result_version"],
            "status": result["status"],
            "production_gate_eligible": result["production_gate_eligible"],
            "acceptance_profile": result["acceptance_profile"],
            "sampling_profile": result["sampling_profile"],
            "acceptance_failures": result["acceptance_failures"],
            "identity_sha256": canonical_sha256(identity),
            "identity": identity,
        }

    def _assert_calibration_evidence_unchanged(self) -> None:
        if self.identity_probe_only or self.calibration_evidence is None:
            raise EnglishNearDedupError(
                "Production clustering has no calibration evidence"
            )
        current = self._load_calibration_evidence(self.calibration_result_path)
        if current != self.calibration_evidence:
            raise EnglishNearDedupError(
                "Calibration evidence changed after the run identity was frozen"
            )

    def _validate_preprocess_manifest(self) -> dict[str, Any]:
        if not self.preprocess_manifest_path.is_file():
            raise EnglishNearDedupError(
                f"Missing preprocess manifest: {self.preprocess_manifest_path}"
            )
        payload = json.loads(self.preprocess_manifest_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_version": 1,
            "fingerprint_version": FINGERPRINT_VERSION,
            "policy_sha256": POLICY_SHA256,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "raw_data_mutated": False,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise EnglishNearDedupError(f"Preprocess manifest {field} mismatch")
        return payload

    def _load_source_identities(self) -> tuple[dict[str, str], dict[str, Any]]:
        tokenizer = self.root / "tokenizer" / "starcoder2" / "TOKENIZER_MANIFEST.json"
        if not tokenizer.is_file():
            raise EnglishNearDedupError(f"Missing tokenizer manifest: {tokenizer}")
        tokenizer_payload = json.loads(tokenizer.read_text(encoding="utf-8"))
        tokenizer_revision = tokenizer_payload.get("resolved_revision")
        if not isinstance(tokenizer_revision, str) or len(tokenizer_revision) != 40:
            raise EnglishNearDedupError("Invalid tokenizer resolved revision")
        sources: dict[str, str] = {}
        manifests: dict[str, Any] = {
            "TOKENIZER_MANIFEST.json": {
                "sha256": file_sha256(tokenizer),
                "resolved_revision": tokenizer_revision,
            }
        }
        for bucket, filename in (
            ("fineweb_edu", "FINEWEB_EDU_SOURCE.json"),
            ("wikipedia", "WIKIPEDIA_SOURCE.json"),
        ):
            path = self.root / "manifests" / filename
            if not path.is_file():
                raise EnglishNearDedupError(f"Missing source manifest: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            for field in ("repo_id", "resolved_revision", "dataset_config"):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise EnglishNearDedupError(f"Invalid {filename} field {field}")
            if payload.get("tokenizer_revision") != tokenizer_revision:
                raise EnglishNearDedupError(f"{filename} tokenizer revision mismatch")
            sources[bucket] = (
                f"{payload['repo_id']}@{payload['resolved_revision']}#{payload['dataset_config']}"
            )
            manifests[filename] = {
                "sha256": file_sha256(path),
                "resolved_revision": payload["resolved_revision"],
                "tokenizer_revision": payload["tokenizer_revision"],
            }
        return sources, manifests

    def _load_collection_completeness(self) -> dict[str, Any]:
        """Prove every finalized English archive has authoritative ledger evidence."""
        if not self.quota_config_path.is_file():
            raise EnglishNearDedupError(
                f"Missing collection quota config: {self.quota_config_path}"
            )
        quota_raw = self.quota_config_path.read_bytes()
        quota_config = json.loads(quota_raw)
        if quota_config.get("version") != 1 or not isinstance(
            quota_config.get("quotas"), list
        ):
            raise EnglishNearDedupError("Unsupported collection quota config")
        targets: dict[str, int] = {}
        for bucket in ENGLISH_BUCKETS:
            matches = [
                row
                for row in quota_config["quotas"]
                if row.get("phase") == "collection"
                and row.get("category") == "english"
                and row.get("language_group") == bucket
                and "split" not in row
            ]
            if (
                len(matches) != 1
                or matches[0].get("token_field") != "exact_tokens"
                or not isinstance(matches[0].get("target"), int)
                or matches[0]["target"] <= 0
            ):
                raise EnglishNearDedupError(
                    f"Expected one positive exact-token collection quota for {bucket}"
                )
            targets[bucket] = int(matches[0]["target"])

        records_root = self.root / "state" / "quota_records" / "collection"
        if not records_root.is_dir():
            raise EnglishNearDedupError(
                f"Missing finalized collection quota ledger: {records_root}"
            )
        records_by_bucket: dict[str, dict[int, dict[str, Any]]] = {
            bucket: {} for bucket in ENGLISH_BUCKETS
        }
        record_inventory: list[dict[str, Any]] = []
        seen_shards: set[str] = set()
        for path in sorted(records_root.glob("*.json")):
            raw = path.read_bytes()
            record = json.loads(raw)
            bucket = record.get("language_group")
            if (
                record.get("phase") != "collection"
                or record.get("category") != "english"
                or bucket not in ENGLISH_BUCKETS
            ):
                continue
            if record.get("record_version") != 1:
                raise EnglishNearDedupError(f"Unsupported quota record: {path}")
            shard_id = record.get("shard_id")
            if not isinstance(shard_id, str) or shard_id in seen_shards:
                raise EnglishNearDedupError(f"Duplicate/invalid quota shard ID: {path}")
            seen_shards.add(shard_id)
            match = re.search(r"-(\d{6})$", shard_id)
            if match is None:
                raise EnglishNearDedupError(f"Quota shard lacks archive index: {path}")
            index = int(match.group(1))
            if index in records_by_bucket[str(bucket)]:
                raise EnglishNearDedupError(
                    f"Duplicate finalized {bucket} archive index {index:06d}"
                )
            if record.get("source") != self.sources[str(bucket)]:
                raise EnglishNearDedupError(f"Quota record source mismatch: {path}")
            for field in ("documents", "clean_bytes", "exact_tokens"):
                if not isinstance(record.get(field), int) or record[field] <= 0:
                    raise EnglishNearDedupError(f"Invalid quota record {field}: {path}")
            descriptor = {
                "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shard_id": shard_id,
                "bucket": bucket,
                "index": index,
                "documents": int(record["documents"]),
                "clean_bytes": int(record["clean_bytes"]),
                "exact_tokens": int(record["exact_tokens"]),
                "source": record["source"],
            }
            records_by_bucket[str(bucket)][index] = descriptor
            record_inventory.append(descriptor)

        tokenizer_revision = self.source_manifests["TOKENIZER_MANIFEST.json"][
            "resolved_revision"
        ]
        marker_names = {
            "fineweb_edu": "ENGLISH_FINEWEB_EDU_COMPLETE.json",
            "wikipedia": "ENGLISH_WIKIPEDIA_COMPLETE.json",
        }
        raw_directories = {
            "fineweb_edu": self.root / "raw" / "english" / "fineweb_edu",
            "wikipedia": self.root / "raw" / "english" / "wikipedia",
        }
        bucket_evidence: dict[str, Any] = {}
        for bucket in ENGLISH_BUCKETS:
            marker_path = self.root / "state" / marker_names[bucket]
            if not marker_path.is_file():
                raise EnglishNearDedupError(
                    f"Missing authoritative {bucket} collection-complete marker: {marker_path}"
                )
            marker_raw = marker_path.read_bytes()
            marker = json.loads(marker_raw)
            expected_marker = {
                "source": self.sources[bucket],
                "tokenizer_revision": tokenizer_revision,
                "benchmark_guard_sha256": self.guard.manifest_sha256,
                "target": targets[bucket],
            }
            for field, expected in expected_marker.items():
                if marker.get(field) != expected:
                    raise EnglishNearDedupError(
                        f"{bucket} completion marker {field} mismatch"
                    )
            if not isinstance(marker.get("english_tokens"), int):
                raise EnglishNearDedupError(
                    f"{bucket} completion marker has invalid english_tokens"
                )
            records = records_by_bucket[bucket]
            if not records:
                raise EnglishNearDedupError(f"No finalized quota records for {bucket}")
            totals = {
                "archives": len(records),
                "documents": sum(row["documents"] for row in records.values()),
                "clean_bytes": sum(row["clean_bytes"] for row in records.values()),
                "exact_tokens": sum(row["exact_tokens"] for row in records.values()),
            }
            if totals["exact_tokens"] != marker["english_tokens"]:
                raise EnglishNearDedupError(
                    f"{bucket} marker/ledger token mismatch: "
                    f"{marker['english_tokens']} != {totals['exact_tokens']}"
                )
            if totals["exact_tokens"] < targets[bucket]:
                raise EnglishNearDedupError(
                    f"{bucket} finalized ledger is below its collection target"
                )
            raw_directory = raw_directories[bucket]
            raw_paths = sorted(raw_directory.glob("part-*.tar.zst")) if raw_directory.exists() else []
            raw_indices: dict[int, Path] = {}
            for raw_path in raw_paths:
                match = re.fullmatch(r"part-(\d{6})\.tar\.zst", raw_path.name)
                if match is None:
                    raise EnglishNearDedupError(f"Unexpected finalized archive name: {raw_path}")
                index = int(match.group(1))
                if index in raw_indices:
                    raise EnglishNearDedupError(f"Duplicate finalized raw archive: {raw_path}")
                raw_indices[index] = raw_path
            pending = sorted(raw_directory.glob(".part-*.tar.zst")) if raw_directory.exists() else []
            if pending:
                raise EnglishNearDedupError(
                    f"{bucket} has {len(pending)} pending archives despite completion marker"
                )
            if set(raw_indices) != set(records):
                missing_raw = sorted(set(records) - set(raw_indices))
                missing_ledger = sorted(set(raw_indices) - set(records))
                raise EnglishNearDedupError(
                    f"{bucket} raw/ledger archive mismatch "
                    f"(missing_raw={missing_raw}, missing_ledger={missing_ledger})"
                )
            bucket_evidence[bucket] = {
                "target_exact_tokens": targets[bucket],
                "completion_marker": {
                    "path": str(marker_path.relative_to(self.root)),
                    "sha256": hashlib.sha256(marker_raw).hexdigest(),
                    **marker,
                },
                "finalized_totals": totals,
                "archive_indices_sha256": hashlib.sha256(
                    canonical_json_bytes(sorted(records))
                ).hexdigest(),
            }
        record_inventory.sort(key=lambda row: (row["bucket"], row["index"]))
        return {
            "evidence_version": 1,
            "quota_config_path": str(self.quota_config_path),
            "quota_config_sha256": hashlib.sha256(quota_raw).hexdigest(),
            "quota_record_inventory_sha256": hashlib.sha256(
                canonical_json_bytes(record_inventory)
            ).hexdigest(),
            "quota_records": record_inventory,
            "buckets": bucket_evidence,
        }

    def _validate_report_completeness(self) -> None:
        reports_by_bucket: dict[str, dict[int, dict[str, Any]]] = {
            bucket: {} for bucket in ENGLISH_BUCKETS
        }
        for item in self.reports:
            report = item.report
            bucket = str(report["bucket"])
            index = report.get("index")
            if not isinstance(index, int) or index in reports_by_bucket[bucket]:
                raise EnglishNearDedupError(
                    f"Duplicate/invalid report index for {bucket}: {item.report_path}"
                )
            reports_by_bucket[bucket][index] = report
        evidence_records: dict[str, dict[int, dict[str, Any]]] = {
            bucket: {} for bucket in ENGLISH_BUCKETS
        }
        for record in self.collection_completeness["quota_records"]:
            evidence_records[str(record["bucket"])][int(record["index"])] = record
        for bucket in ENGLISH_BUCKETS:
            reports = reports_by_bucket[bucket]
            records = evidence_records[bucket]
            if set(reports) != set(records):
                missing_reports = sorted(set(records) - set(reports))
                unknown_reports = sorted(set(reports) - set(records))
                raise EnglishNearDedupError(
                    f"{bucket} finalized report coverage incomplete "
                    f"(missing_reports={missing_reports}, unknown_reports={unknown_reports})"
                )
            for index, record in records.items():
                report = reports[index]
                expected_archive = {
                    "fineweb_edu": f"raw/english/fineweb_edu/part-{index:06d}.tar.zst",
                    "wikipedia": f"raw/english/wikipedia/part-{index:06d}.tar.zst",
                }[bucket]
                expected_report = (
                    self.staging_root
                    / "reports"
                    / bucket
                    / f"part-{index:06d}.json"
                )
                matching_item = next(
                    item for item in self.reports if item.report is report
                )
                if matching_item.report_path != expected_report:
                    raise EnglishNearDedupError(
                        f"{bucket} report path/index mismatch at part-{index:06d}"
                    )
                expected = {
                    "archive": expected_archive,
                    "fingerprint_file": (
                        f"fingerprints/{bucket}/part-{index:06d}.jsonl.zst"
                    ),
                    "quota_shard_id": record["shard_id"],
                    "source": record["source"],
                    "documents": record["documents"],
                    "clean_bytes": record["clean_bytes"],
                    "exact_tokens": record["exact_tokens"],
                }
                for field, value in expected.items():
                    if report.get(field) != value:
                        raise EnglishNearDedupError(
                            f"{bucket} report/ledger {field} mismatch at part-{index:06d}"
                        )
            report_totals = {
                "archives": len(reports),
                "documents": sum(int(row["documents"]) for row in reports.values()),
                "clean_bytes": sum(int(row["clean_bytes"]) for row in reports.values()),
                "exact_tokens": sum(int(row["exact_tokens"]) for row in reports.values()),
            }
            if report_totals != self.collection_completeness["buckets"][bucket][
                "finalized_totals"
            ]:
                raise EnglishNearDedupError(
                    f"{bucket} report totals do not match finalized quota ledger"
                )
            self.collection_completeness["buckets"][bucket][
                "report_totals"
            ] = report_totals

    def _freeze_reports(self) -> list[ReportInput]:
        report_root = self.staging_root / "reports"
        paths = sorted(
            path
            for bucket in ENGLISH_BUCKETS
            for path in (report_root / bucket).glob("part-*.json")
        )
        if not paths:
            raise EnglishNearDedupError("No completed English fingerprint reports found")
        result: list[ReportInput] = []
        seen_archives: set[str] = set()
        for path in paths:
            raw = path.read_bytes()
            report = json.loads(raw)
            bucket = report.get("bucket")
            if bucket not in ENGLISH_BUCKETS:
                raise EnglishNearDedupError(f"Unexpected report bucket: {path}")
            if report.get("report_version") != 1 or report.get("fingerprint_version") != FINGERPRINT_VERSION:
                raise EnglishNearDedupError(f"Unsupported report version: {path}")
            if report.get("policy_sha256") != POLICY_SHA256:
                raise EnglishNearDedupError(f"Fingerprint policy mismatch: {path}")
            if report.get("source") != self.sources[bucket]:
                raise EnglishNearDedupError(f"Source identity mismatch: {path}")
            archive = report.get("archive")
            if not isinstance(archive, str) or archive in seen_archives:
                raise EnglishNearDedupError(f"Duplicate or invalid archive identity: {path}")
            seen_archives.add(archive)
            parse_digest(report.get("archive_sha256"), "archive_sha256")
            parse_digest(report.get("fingerprint_sha256"), "fingerprint_sha256")
            for field in ("documents", "clean_bytes", "exact_tokens", "archive_compressed_bytes"):
                if not isinstance(report.get(field), int) or report[field] <= 0:
                    raise EnglishNearDedupError(f"Invalid report {field}: {path}")
            raw_archive = safe_relative_path(self.root, archive, "report archive")
            fingerprint = safe_relative_path(
                self.staging_root, report.get("fingerprint_file"), "fingerprint_file"
            )
            if not raw_archive.is_file():
                raise EnglishNearDedupError(f"Missing immutable raw archive: {raw_archive}")
            if raw_archive.stat().st_size != report["archive_compressed_bytes"]:
                raise EnglishNearDedupError(f"Raw archive size mismatch: {raw_archive}")
            if not fingerprint.is_file():
                raise EnglishNearDedupError(f"Missing fingerprint shard: {fingerprint}")
            relative = str(path.relative_to(self.staging_root))
            result.append(
                ReportInput(relative, path, hashlib.sha256(raw).hexdigest(), report)
            )
        return result

    def __enter__(self) -> "EnglishNearDedupBuilder":
        if self.identity_probe_only:
            raise EnglishNearDedupError(
                "An identity-only calibration probe cannot execute clustering"
            )
        self._assert_calibration_evidence_unchanged()
        current_storage = self._current_storage_evidence()
        frozen_without_actual = {
            key: value
            for key, value in self.storage_evidence.items()
            if key != "sqlite_journal_mode_actual"
        }
        if current_storage != frozen_without_actual:
            raise EnglishNearDedupError(
                "Output mount or SQLite journal policy changed before database open"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (self.work / ".lock").open("a+b")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_handle.close()
            self.lock_handle = None
            raise EnglishNearDedupError(f"Another near-dedup process owns {self.work}") from exc
        try:
            selected_journal = str(
                self.storage_evidence["sqlite_journal_mode_selected"]
            )
            sidecars = [
                path
                for path in (
                    Path(f"{self.db_path}-wal"),
                    Path(f"{self.db_path}-shm"),
                )
                if path.exists()
            ]
            if selected_journal == "delete" and sidecars:
                raise EnglishNearDedupError(
                    "Rollback-journal near-dedup output contains WAL sidecars; "
                    "refusing a possibly live or incompletely copied database: "
                    + ", ".join(str(path) for path in sidecars)
                )
            if self.db_path.is_symlink() or (
                self.db_path.exists() and not self.db_path.is_file()
            ):
                raise EnglishNearDedupError(
                    f"Unsafe near-dedup database path: {self.db_path}"
                )
            database_exists = self.db_path.exists()
            nonempty_database = (
                database_exists and self.db_path.stat().st_size > 0
            )
            if not database_exists and sidecars:
                raise EnglishNearDedupError(
                    "Near-dedup WAL sidecars exist without a database"
                )
            self.connection = sqlite3.connect(self.db_path, timeout=120)
            self.connection.row_factory = sqlite3.Row
            current_journal = str(
                self.connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            metadata_exists = self.connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='metadata'"
            ).fetchone() is not None
            if metadata_exists:
                if current_journal != selected_journal:
                    raise EnglishNearDedupError(
                        "Existing near-dedup database journal mode mismatch: "
                        f"found {current_journal}, expected {selected_journal}"
                    )
                metadata_rows = int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM metadata"
                    ).fetchone()[0]
                )
                if metadata_rows == 0:
                    raise EnglishNearDedupError(
                        "Non-empty near-dedup database has no identity metadata"
                    )
                self.storage_evidence[
                    "sqlite_journal_mode_actual"
                ] = current_journal
                # Validate frozen mount, mode, calibration and corpus identity
                # before any schema write or journal conversion.
                self._initialize_identity()
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.execute("PRAGMA foreign_keys=ON")
                self.connection.execute("PRAGMA temp_store=FILE")
                self.connection.execute("PRAGMA cache_size=-524288")
                self.connection.execute("PRAGMA mmap_size=8589934592")
                self._create_schema()
            else:
                if nonempty_database:
                    raise EnglishNearDedupError(
                        "Non-empty near-dedup database has no identity metadata"
                    )
                actual_journal = str(
                    self.connection.execute(
                        f"PRAGMA journal_mode={selected_journal.upper()}"
                    ).fetchone()[0]
                ).casefold()
                if actual_journal != selected_journal:
                    raise EnglishNearDedupError(
                        "SQLite did not activate the selected journal mode: "
                        f"{actual_journal!r} != {selected_journal!r}"
                    )
                self.storage_evidence[
                    "sqlite_journal_mode_actual"
                ] = actual_journal
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.execute("PRAGMA foreign_keys=ON")
                self.connection.execute("PRAGMA temp_store=FILE")
                self.connection.execute("PRAGMA cache_size=-524288")
                self.connection.execute("PRAGMA mmap_size=8589934592")
                self._create_schema()
                self._initialize_identity()
            self._load_operational_metrics()
            if (self.output / "operational-preflight-v1").exists():
                self.refinement_preflight_evidence = (
                    self._load_refinement_preflight(verify_cache_files=True)
                )
            self._last_progress_at = time.monotonic()
            self._sync_checkpoint()
            return self
        except BaseException:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            self.lock_handle.close()
            self.lock_handle = None
            raise

    def __exit__(self, *_args: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock_handle is not None:
            self.lock_handle.close()
            self.lock_handle = None

    @property
    def db(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("builder must be used as a context manager")
        return self.connection

    def _load_operational_metrics(self) -> None:
        path = self.work / "OPERATIONAL.json"
        checksum_path = self.work / "OPERATIONAL.json.sha256"
        if not path.exists() and not checksum_path.exists():
            return
        try:
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if checksum_path.read_bytes() != (
                f"{checksum}  {path.name}\n".encode("ascii")
            ):
                raise ValueError("checksum mismatch")
            payload = self._json_without_duplicate_keys(
                raw, "near-dedup operational metrics"
            )
            if (
                payload.get("operational_version") != 1
                or payload.get("identity_sha256")
                != canonical_sha256(self.identity)
                or not isinstance(payload.get("phases"), dict)
            ):
                raise ValueError("identity or schema mismatch")
            self.operational_metrics = dict(payload["phases"])
        except (EnglishNearDedupError, OSError, ValueError) as exc:
            # This is a rebuildable progress projection, never the restart
            # authority. A clean controlled stop republishes a valid pair.
            self.operational_metrics = {}
            print(
                json.dumps(
                    {
                        "event": "english_near_operational_projection_reset",
                        "reason": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    def _resource_snapshot(self) -> dict[str, int]:
        filesystem = os.statvfs(self.work)
        fragment = int(filesystem.f_frsize or filesystem.f_bsize)
        sqlite_state_bytes = sum(
            path.stat().st_size
            for path in (
                self.db_path,
                Path(f"{self.db_path}-wal"),
                Path(f"{self.db_path}-shm"),
                Path(f"{self.db_path}-journal"),
            )
            if path.is_file()
        )
        cache_bytes = 0
        if self.connection is not None:
            for row in self.db.execute("SELECT cache_file FROM cache_archives"):
                path = self.work / str(row[0])
                if path.is_file():
                    cache_bytes += path.stat().st_size
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
        return {
            "filesystem_total_bytes": int(filesystem.f_blocks) * fragment,
            "filesystem_free_bytes": int(filesystem.f_bavail) * fragment,
            "sqlite_state_bytes": sqlite_state_bytes,
            "refinement_cache_bytes": cache_bytes,
            "peak_process_rss_bytes": peak_rss,
        }

    def _union_memory_projection(
        self, document_count: int, baseline_peak_rss_bytes: int
    ) -> dict[str, int]:
        """Return the pinned conservative dense-parent allocation projection.

        Union uses one unsigned-64 parent per frozen ordinal.  The configured
        multiplier accounts for allocator/page overhead and the bounded SQL
        row/update batches.  This is checked before allocation; measured peak
        RSS is checked again after allocation and after every committed batch.
        """
        if document_count < 0 or baseline_peak_rss_bytes < 0:
            raise EnglishNearDedupError("Invalid union memory projection input")
        item_size = array.array("Q").itemsize
        parent_bytes = (document_count + 1) * item_size
        config = self.config["operational_preflight"]
        parent_with_safety = math.ceil(
            parent_bytes
            * int(config["union_parent_memory_safety_numerator"])
            / int(config["union_parent_memory_safety_denominator"])
        )
        return {
            "union_parent_item_bytes": item_size,
            "union_parent_array_projected_bytes": parent_bytes,
            "union_parent_array_with_safety_bytes": parent_with_safety,
            "union_projected_peak_process_rss_bytes": (
                baseline_peak_rss_bytes + parent_with_safety
            ),
        }

    def _publish_operational_metric(
        self,
        phase: str,
        metric: dict[str, Any],
        *,
        force: bool = False,
        resources: dict[str, int] | None = None,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress_at < self.progress_interval_seconds:
            return
        self._last_progress_at = now
        published = {
            **metric,
            "phase": phase,
            "recorded_unix_seconds": round(time.time(), 3),
            "resources": resources or self._resource_snapshot(),
        }
        self.operational_metrics[phase] = published
        payload = {
            "operational_version": 1,
            "identity_sha256": canonical_sha256(self.identity),
            "phases": self.operational_metrics,
        }
        path = self.work / "OPERATIONAL.json"
        raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        atomic_bytes(path, raw)
        atomic_bytes(
            self.work / "OPERATIONAL.json.sha256",
            hashlib.sha256(raw).hexdigest().encode("ascii")
            + f"  {path.name}\n".encode("ascii"),
        )
        print(
            json.dumps(
                {"event": "english_near_progress", **published},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports(
                report_path TEXT PRIMARY KEY,
                report_sha256 TEXT NOT NULL,
                archive TEXT NOT NULL UNIQUE,
                bucket TEXT NOT NULL,
                fingerprint_file TEXT NOT NULL,
                fingerprint_sha256 TEXT NOT NULL,
                documents INTEGER NOT NULL,
                tokens INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS documents(
                ordinal INTEGER PRIMARY KEY,
                doc_id BLOB NOT NULL UNIQUE,
                bucket TEXT NOT NULL,
                archive TEXT NOT NULL,
                manifest_index INTEGER NOT NULL,
                member_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                content_hash BLOB NOT NULL,
                normalized_hash BLOB NOT NULL,
                audit_sketch BLOB NOT NULL,
                shingle_count INTEGER,
                normalized_representative INTEGER NOT NULL DEFAULT 0,
                cluster_root INTEGER,
                UNIQUE(archive, manifest_index),
                UNIQUE(archive, member_path)
            );
            CREATE TABLE IF NOT EXISTS signature_archives(
                archive TEXT PRIMARY KEY,
                archive_sha256 TEXT NOT NULL,
                documents INTEGER NOT NULL,
                representatives INTEGER NOT NULL,
                band_rows INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS bands(
                band INTEGER NOT NULL,
                band_key BLOB NOT NULL,
                document INTEGER NOT NULL,
                PRIMARY KEY(band, band_key, document),
                FOREIGN KEY(document) REFERENCES documents(ordinal)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates(
                left_document INTEGER NOT NULL,
                right_document INTEGER NOT NULL,
                PRIMARY KEY(left_document, right_document)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidate_blocks(
                band INTEGER NOT NULL,
                band_key BLOB NOT NULL,
                posting_documents INTEGER NOT NULL,
                raw_pairs INTEGER NOT NULL,
                length_pruned_pairs INTEGER NOT NULL,
                PRIMARY KEY(band, band_key)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS cache_archives(
                archive TEXT PRIMARY KEY,
                cache_file TEXT NOT NULL,
                cache_sha256 TEXT NOT NULL,
                cached_documents INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS shingle_cache(
                document INTEGER PRIMARY KEY,
                cache_file TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                shingle_count INTEGER NOT NULL,
                FOREIGN KEY(document) REFERENCES documents(ordinal)
            );
            CREATE TABLE IF NOT EXISTS refined(
                left_document INTEGER NOT NULL,
                right_document INTEGER NOT NULL,
                intersection_count INTEGER NOT NULL,
                union_count INTEGER NOT NULL,
                accepted INTEGER NOT NULL,
                PRIMARY KEY(left_document, right_document)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS edges(
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                left_document INTEGER NOT NULL,
                right_document INTEGER NOT NULL,
                kind TEXT NOT NULL,
                UNIQUE(left_document, right_document, kind)
            );
            CREATE TABLE IF NOT EXISTS parents(
                document INTEGER PRIMARY KEY,
                parent INTEGER NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self.db.commit()

    def _initialize_identity(self) -> None:
        encoded = {key: json.dumps(value, sort_keys=True) for key, value in self.identity.items()}
        existing = dict(self.db.execute("SELECT key, value FROM metadata"))
        if not existing:
            with self.db:
                self.db.executemany(
                    "INSERT INTO metadata(key,value) VALUES (?,?)",
                    [
                        ("database_version", json.dumps(DATABASE_VERSION)),
                        ("phase", json.dumps("inventory")),
                        *encoded.items(),
                    ],
                )
                self._event("initialized", {"reports": len(self.reports)})
            return
        required_keys = {"database_version", "phase", *encoded}
        optional_cursor_keys = {
            "candidate_cursor",
            "candidate_blocks_total",
            "refinement_cursor",
            "union_edge_cursor",
            "union_merges",
            "union_flatten_cursor",
            "union_document_cursor",
        }
        missing = sorted(required_keys - existing.keys())
        unexpected = sorted(existing.keys() - required_keys - optional_cursor_keys)
        if missing or unexpected:
            raise EnglishNearDedupError(
                "Near-dedup metadata key-set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if json.loads(existing.get("database_version", "null")) != DATABASE_VERSION:
            raise EnglishNearDedupError("Near-dedup database version mismatch")
        for key, value in encoded.items():
            if existing.get(key) != value:
                raise EnglishNearDedupError(f"Resume identity mismatch for {key}")
        try:
            phase = json.loads(existing["phase"])
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            raise EnglishNearDedupError("Invalid near-dedup resume phase") from exc
        phases = (
            "inventory",
            "signatures",
            "candidates",
            "cache",
            "refine",
            "union",
            "emit",
            "complete",
        )
        if phase not in phases:
            raise EnglishNearDedupError(f"Invalid near-dedup resume phase: {phase!r}")

        candidate_cursor = existing.get("candidate_cursor")
        if candidate_cursor is not None:
            if phases.index(phase) < phases.index("candidates"):
                raise EnglishNearDedupError(
                    "Candidate cursor exists before the candidates phase"
                )
            try:
                payload = json.loads(candidate_cursor)
                band = payload["band"]
                key = payload["key"]
                if (
                    set(payload) != {"band", "key"}
                    or not isinstance(band, int)
                    or isinstance(band, bool)
                    or not 0 <= band < int(self.config["_total_bands"])
                    or not isinstance(key, str)
                    or len(key) != 16
                    or any(character not in HEX for character in key)
                ):
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EnglishNearDedupError(
                    "Invalid near-dedup candidate cursor metadata"
                ) from exc

        candidate_blocks_total = existing.get("candidate_blocks_total")
        if candidate_blocks_total is not None:
            if phases.index(phase) < phases.index("candidates"):
                raise EnglishNearDedupError(
                    "Candidate-block total exists before the candidates phase"
                )
            try:
                total = json.loads(candidate_blocks_total)
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total < 0
                ):
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise EnglishNearDedupError(
                    "Invalid near-dedup candidate-block total metadata"
                ) from exc

        refinement_cursor = existing.get("refinement_cursor")
        if refinement_cursor is not None:
            if phases.index(phase) < phases.index("refine"):
                raise EnglishNearDedupError(
                    "Refinement cursor exists before the refine phase"
                )
            try:
                payload = json.loads(refinement_cursor)
                left = payload["left_document"]
                right = payload["right_document"]
                processed_pairs = payload["processed_pairs"]
                if (
                    set(payload)
                    != {"left_document", "right_document", "processed_pairs"}
                    or not isinstance(left, int)
                    or isinstance(left, bool)
                    or left < 1
                    or not isinstance(right, int)
                    or isinstance(right, bool)
                    or right <= left
                    or not isinstance(processed_pairs, int)
                    or isinstance(processed_pairs, bool)
                    or processed_pairs < 1
                ):
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EnglishNearDedupError(
                    "Invalid near-dedup refinement cursor metadata"
                ) from exc

        union_values = (
            existing.get("union_edge_cursor"),
            existing.get("union_merges"),
        )
        if any(value is not None for value in union_values):
            if not all(value is not None for value in union_values):
                raise EnglishNearDedupError(
                    "Near-dedup union cursor metadata is incomplete"
                )
            if phases.index(phase) < phases.index("union"):
                raise EnglishNearDedupError(
                    "Union cursor exists before the union phase"
                )
            try:
                cursor, merges = (json.loads(value) for value in union_values)
                if (
                    not isinstance(cursor, int)
                    or isinstance(cursor, bool)
                    or cursor < 0
                    or not isinstance(merges, int)
                    or isinstance(merges, bool)
                    or merges < 0
                    or merges > cursor
                ):
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise EnglishNearDedupError(
                    "Invalid near-dedup union cursor metadata"
                ) from exc

        flatten_cursor = existing.get("union_flatten_cursor")
        document_cursor = existing.get("union_document_cursor")
        if document_cursor is not None and flatten_cursor is None:
            raise EnglishNearDedupError(
                "Union document cursor exists without a completed parent cursor"
            )
        for key, value, cursor_key, total_key in (
            (
                "union_flatten_cursor",
                flatten_cursor,
                "document",
                "parent_documents_total",
            ),
            (
                "union_document_cursor",
                document_cursor,
                "ordinal",
                "documents_total",
            ),
        ):
            if value is None:
                continue
            if phases.index(phase) < phases.index("union"):
                raise EnglishNearDedupError(f"{key} exists before the union phase")
            try:
                payload = json.loads(value)
                cursor_value = payload[cursor_key]
                processed_documents = payload["processed_documents"]
                total_documents = payload[total_key]
                if (
                    set(payload)
                    != {cursor_key, "processed_documents", total_key}
                    or any(
                        not isinstance(item, int)
                        or isinstance(item, bool)
                        or item < 0
                        for item in (
                            cursor_value,
                            processed_documents,
                            total_documents,
                        )
                    )
                    or processed_documents > total_documents
                    or (processed_documents == 0) != (cursor_value == 0)
                    or (
                        key == "union_document_cursor"
                        and cursor_value != processed_documents
                    )
                ):
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EnglishNearDedupError(
                    f"Invalid near-dedup {key} metadata"
                ) from exc
        if flatten_cursor is not None and document_cursor is not None:
            flatten_payload = json.loads(flatten_cursor)
            if flatten_payload["processed_documents"] != flatten_payload[
                "parent_documents_total"
            ]:
                raise EnglishNearDedupError(
                    "Union document assignment began before parent flattening completed"
                )
        if phases.index(phase) > phases.index("union"):
            for key, value in (
                ("union_flatten_cursor", flatten_cursor),
                ("union_document_cursor", document_cursor),
            ):
                if value is None:
                    raise EnglishNearDedupError(
                        f"Completed union is missing {key} metadata"
                    )
                payload = json.loads(value)
                if payload["processed_documents"] != payload[
                    next(
                        field
                        for field in payload
                        if field.endswith("_total")
                    )
                ]:
                    raise EnglishNearDedupError(
                        f"Completed union has incomplete {key} metadata"
                    )

    def _phase(self) -> str:
        row = self.db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()
        return str(json.loads(row[0]))

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO events(event,payload) VALUES (?,?)",
            (event, canonical_json_bytes(payload).decode("utf-8")),
        )

    def _advance(self, expected: str, new: str, payload: dict[str, Any]) -> None:
        if self._phase() != expected:
            raise EnglishNearDedupError(f"Cannot advance {self._phase()} as {expected}")
        with self.db:
            self.db.execute("UPDATE metadata SET value=? WHERE key='phase'", (json.dumps(new),))
            self._event(new, payload)
        self._sync_checkpoint()

    def _sync_checkpoint(self) -> None:
        if self.connection is None:
            return
        events = [
            {"sequence": int(row[0]), "event": row[1], "payload": json.loads(row[2])}
            for row in self.db.execute(
                "SELECT sequence,event,payload FROM events ORDER BY sequence"
            )
        ]
        atomic_bytes(
            self.work / "journal.jsonl",
            b"".join(canonical_json_bytes(row) + b"\n" for row in events),
        )
        counts = {
            table: int(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "reports",
                "documents",
                "signature_archives",
                "bands",
                "candidates",
                "cache_archives",
                "refined",
                "edges",
            )
        }
        atomic_json(
            self.work / "CHECKPOINT.json",
            {
                "checkpoint_version": 1,
                "phase": self._phase(),
                "identity": self.identity,
                "refinement_operational_preflight": self.refinement_preflight_evidence,
                "counts": counts,
                "last_event_sequence": events[-1]["sequence"] if events else 0,
            },
        )

    def ingest_inventory(self, max_new_archives: int | None = None) -> bool:
        if self._phase() != "inventory":
            return True
        committed = {
            row[0]: row[1]
            for row in self.db.execute("SELECT report_path,report_sha256 FROM reports")
        }
        added = 0
        for item in self.reports:
            if item.relative_report in committed:
                if committed[item.relative_report] != item.report_sha256:
                    raise EnglishNearDedupError(
                        f"Committed report changed: {item.relative_report}"
                    )
                continue
            self._ingest_report(item)
            added += 1
            self._sync_checkpoint()
            if max_new_archives is not None and added >= max_new_archives:
                return False
        if int(self.db.execute("SELECT COUNT(*) FROM reports").fetchone()[0]) != len(self.reports):
            raise EnglishNearDedupError("Inventory did not ingest the frozen report set")
        with self.db:
            self.db.executescript(
                """
                CREATE INDEX IF NOT EXISTS documents_archive_index
                    ON documents(archive, manifest_index);
                CREATE INDEX IF NOT EXISTS documents_normalized
                    ON documents(normalized_hash, ordinal);
                UPDATE documents SET normalized_representative=1
                WHERE ordinal IN (
                    SELECT MIN(ordinal) FROM documents GROUP BY normalized_hash
                );
                """
            )
        totals = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(tokens),0),SUM(normalized_representative) FROM documents"
        ).fetchone()
        self._advance(
            "inventory",
            "signatures",
            {
                "documents": int(totals[0]),
                "tokens": int(totals[1]),
                "normalized_representatives": int(totals[2]),
            },
        )
        return True

    def _ingest_report(self, item: ReportInput) -> None:
        report = item.report
        fingerprint = safe_relative_path(
            self.staging_root, report["fingerprint_file"], "fingerprint_file"
        )
        if file_sha256(fingerprint) != report["fingerprint_sha256"]:
            raise EnglishNearDedupError(f"Fingerprint checksum mismatch: {fingerprint}")
        document_batch: list[tuple[Any, ...]] = []
        documents = clean_bytes = tokens = 0

        def flush() -> None:
            if not document_batch:
                return
            self.db.executemany(
                """
                INSERT INTO documents(
                    doc_id,bucket,archive,manifest_index,member_path,size_bytes,tokens,
                    content_hash,normalized_hash,audit_sketch
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                document_batch,
            )
            document_batch.clear()

        try:
            self.db.execute("BEGIN IMMEDIATE")
            for expected_index, row in enumerate(iter_jsonl_zst(fingerprint)):
                if row.get("record_version") != 1 or row.get("fingerprint_version") != FINGERPRINT_VERSION:
                    raise EnglishNearDedupError(f"Unsupported fingerprint row: {fingerprint}")
                if row.get("bucket") != report["bucket"] or row.get("archive") != report["archive"]:
                    raise EnglishNearDedupError(f"Report/fingerprint identity mismatch: {fingerprint}")
                if row.get("archive_index") != report.get("index"):
                    raise EnglishNearDedupError(f"Archive index mismatch: {fingerprint}")
                if row.get("manifest_index") != expected_index:
                    raise EnglishNearDedupError(f"Non-contiguous fingerprint rows: {fingerprint}")
                member_path = row.get("member_path")
                if not isinstance(member_path, str) or not member_path:
                    raise EnglishNearDedupError(f"Invalid member path: {fingerprint}")
                expected_doc = hashlib.sha256(
                    f"{report['archive']}\0{member_path}".encode("utf-8")
                ).hexdigest()
                if row.get("doc_id") != expected_doc:
                    raise EnglishNearDedupError(f"Unstable document ID: {fingerprint}")
                doc_id = parse_digest(row.get("doc_id"), "doc_id")
                content_hash = parse_digest(row.get("content_sha256"), "content_sha256")
                normalized_hash = parse_digest(
                    row.get("normalized_sha256"), "normalized_sha256"
                )
                size = row.get("size_bytes")
                token_count = row.get("starcoder2_tokens")
                if not isinstance(size, int) or size <= 0:
                    raise EnglishNearDedupError(f"Invalid fingerprint size: {fingerprint}")
                if not isinstance(token_count, int) or token_count <= 0:
                    raise EnglishNearDedupError(f"Invalid fingerprint token count: {fingerprint}")
                sketch = row.get("near_sketch")
                if (
                    not isinstance(sketch, list)
                    or len(sketch) > NEAR_SKETCH_SIZE
                    or sketch != sorted(set(sketch))
                    or not all(
                        isinstance(value, str)
                        and len(value) == 16
                        and all(character in HEX for character in value)
                        for value in sketch
                    )
                ):
                    raise EnglishNearDedupError(f"Invalid compact near sketch: {fingerprint}")
                flags = row.get("quality_flags")
                if (
                    not isinstance(flags, list)
                    or flags != sorted(set(flags))
                    or not all(isinstance(flag, str) for flag in flags)
                ):
                    raise EnglishNearDedupError(f"Invalid quality flags: {fingerprint}")
                benchmark_reason = row.get("benchmark_reason")
                if bool(benchmark_reason) != ("benchmark_contamination" in flags):
                    raise EnglishNearDedupError(f"Benchmark identity mismatch: {fingerprint}")
                provenance = row.get("provenance")
                if not isinstance(provenance, dict):
                    raise EnglishNearDedupError(f"Missing provenance: {fingerprint}")
                identity_fields = (
                    ("url", "id")
                    if report["bucket"] == "fineweb_edu"
                    else ("id", "url", "title")
                )
                if not any(
                    provenance.get(field) is not None
                    and str(provenance[field]).strip()
                    for field in identity_fields
                ):
                    raise EnglishNearDedupError(
                        f"Missing stable English source identity: {fingerprint} row {expected_index}"
                    )
                if not isinstance(row.get("metrics"), dict):
                    raise EnglishNearDedupError(f"Missing quality metrics: {fingerprint}")
                document_batch.append(
                    (
                        doc_id,
                        report["bucket"],
                        report["archive"],
                        expected_index,
                        member_path,
                        size,
                        token_count,
                        content_hash,
                        normalized_hash,
                        b"".join(bytes.fromhex(value) for value in sketch),
                    )
                )
                documents += 1
                clean_bytes += size
                tokens += token_count
                if len(document_batch) >= self.batch_size:
                    flush()
            flush()
            if (
                documents != report["documents"]
                or clean_bytes != report["clean_bytes"]
                or tokens != report["exact_tokens"]
            ):
                raise EnglishNearDedupError(
                    f"Fingerprint totals do not match report: {fingerprint}"
                )
            self.db.execute(
                """
                INSERT INTO reports(
                    report_path,report_sha256,archive,bucket,fingerprint_file,
                    fingerprint_sha256,documents,tokens
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    item.relative_report,
                    item.report_sha256,
                    report["archive"],
                    report["bucket"],
                    report["fingerprint_file"],
                    report["fingerprint_sha256"],
                    documents,
                    tokens,
                ),
            )
            self._event(
                "inventory_archive",
                {"archive": report["archive"], "documents": documents, "tokens": tokens},
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def _report_by_archive(self, archive: str) -> dict[str, Any]:
        for item in self.reports:
            if item.report["archive"] == archive:
                return item.report
        raise EnglishNearDedupError(f"Unknown frozen archive: {archive}")

    def _archive_documents(self, archive: str) -> list[sqlite3.Row]:
        return list(
            self.db.execute(
                """
                SELECT ordinal,doc_id,bucket,manifest_index,member_path,size_bytes,tokens,
                       content_hash,normalized_hash,audit_sketch,
                       normalized_representative,shingle_count
                FROM documents WHERE archive=? ORDER BY manifest_index
                """,
                (archive,),
            )
        )

    def _stream_verified_archive(
        self,
        report: dict[str, Any],
        callback: Callable[[sqlite3.Row, str], None],
    ) -> None:
        raw_path = safe_relative_path(self.root, report["archive"], "archive")
        if raw_path.stat().st_size != report["archive_compressed_bytes"]:
            raise EnglishNearDedupError(f"Archive size changed: {raw_path}")
        if file_sha256(raw_path) != report["archive_sha256"]:
            raise EnglishNearDedupError(f"Archive checksum changed: {raw_path}")
        expected = self._archive_documents(report["archive"])
        next_index = 0
        manifest_seen = False
        with raw_path.open("rb") as raw:
            decompressor = zstandard.ZstdDecompressor().stream_reader(
                raw, read_across_frames=True, closefd=False
            )
            try:
                with tarfile.open(fileobj=decompressor, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            raise EnglishNearDedupError(
                                f"Unexpected non-file tar member {member.name}: {raw_path}"
                            )
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise EnglishNearDedupError(f"Cannot read {member.name}: {raw_path}")
                        if member.name == "_manifest.jsonl":
                            if manifest_seen:
                                raise EnglishNearDedupError(f"Multiple internal manifests: {raw_path}")
                            manifest_seen = True
                            manifest_rows = 0
                            for manifest_index, line in enumerate(extracted):
                                if manifest_index >= len(expected):
                                    raise EnglishNearDedupError(
                                        f"Internal manifest has extra rows: {raw_path}"
                                    )
                                row = json.loads(line)
                                document = expected[manifest_index]
                                if (
                                    row.get("member_path") != document["member_path"]
                                    or row.get("size_bytes") != document["size_bytes"]
                                    or row.get("starcoder2_tokens") != document["tokens"]
                                ):
                                    raise EnglishNearDedupError(
                                        f"Internal manifest mismatch at {manifest_index}: {raw_path}"
                                    )
                                manifest_rows += 1
                            if manifest_rows != len(expected):
                                raise EnglishNearDedupError(
                                    f"Internal manifest row count mismatch: {raw_path} "
                                    f"({manifest_rows}/{len(expected)})"
                                )
                            continue
                        if manifest_seen:
                            raise EnglishNearDedupError(f"Document after manifest: {raw_path}")
                        if next_index >= len(expected):
                            raise EnglishNearDedupError(f"Archive has extra documents: {raw_path}")
                        document = expected[next_index]
                        if member.name != document["member_path"]:
                            raise EnglishNearDedupError(
                                f"Raw/fingerprint member order mismatch at {next_index}: {raw_path}"
                            )
                        content = extracted.read()
                        if len(content) != document["size_bytes"] or len(content) != member.size:
                            raise EnglishNearDedupError(f"Raw document size mismatch: {member.name}")
                        if hashlib.sha256(content).digest() != document["content_hash"]:
                            raise EnglishNearDedupError(f"Raw content hash mismatch: {member.name}")
                        try:
                            text = content.decode("utf-8", errors="strict")
                        except UnicodeDecodeError as exc:
                            raise EnglishNearDedupError(
                                f"Raw document is not UTF-8: {member.name}"
                            ) from exc
                        normalized = normalize_content(text, str(document["bucket"]))
                        if hashlib.sha256(normalized.encode("utf-8")).digest() != document["normalized_hash"]:
                            raise EnglishNearDedupError(
                                f"Raw normalized hash mismatch: {member.name}"
                            )
                        recomputed_sketch = b"".join(
                            bytes.fromhex(value)
                            for value in bottom_k_sketch(
                                near_features(normalized, str(document["bucket"]))
                            )
                        )
                        if recomputed_sketch != document["audit_sketch"]:
                            raise EnglishNearDedupError(
                                f"Compact audit sketch mismatch: {member.name}"
                            )
                        callback(document, text)
                        next_index += 1
            finally:
                decompressor.close()
        if not manifest_seen or next_index != len(expected):
            raise EnglishNearDedupError(
                f"Archive completeness mismatch: {raw_path} ({next_index}/{len(expected)})"
            )

    def build_signatures(self, max_new_archives: int | None = None) -> bool:
        if self._phase() != "signatures":
            return True
        committed = {
            row[0] for row in self.db.execute("SELECT archive FROM signature_archives")
        }
        added = 0
        for item in self.reports:
            report = item.report
            archive = str(report["archive"])
            if archive in committed:
                continue
            band_batch: list[tuple[int, bytes, int]] = []
            count_batch: list[tuple[int, int]] = []
            representatives = band_rows = documents = 0

            def flush() -> None:
                if count_batch:
                    self.db.executemany(
                        "UPDATE documents SET shingle_count=? WHERE ordinal=?", count_batch
                    )
                    count_batch.clear()
                if band_batch:
                    band_batch.sort(key=lambda row: (row[0], row[1], row[2]))
                    self.db.executemany(
                        "INSERT INTO bands(band,band_key,document) VALUES (?,?,?)",
                        band_batch,
                    )
                    band_batch.clear()

            def analyze(document: sqlite3.Row, text: str) -> None:
                nonlocal representatives, band_rows, documents
                bands, shingle_count = candidate_bands(text, self.config)
                count_batch.append((shingle_count, int(document["ordinal"])))
                documents += 1
                if int(document["normalized_representative"]):
                    representatives += 1
                    for band, key in bands:
                        band_batch.append((band, key, int(document["ordinal"])))
                        band_rows += 1
                if len(count_batch) >= self.batch_size or len(band_batch) >= self.batch_size:
                    flush()

            try:
                self.db.execute("BEGIN IMMEDIATE")
                self._stream_verified_archive(report, analyze)
                flush()
                self.db.execute(
                    """
                    INSERT INTO signature_archives(
                        archive,archive_sha256,documents,representatives,band_rows
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        archive,
                        report["archive_sha256"],
                        documents,
                        representatives,
                        band_rows,
                    ),
                )
                self._event(
                    "signature_archive",
                    {
                        "archive": archive,
                        "documents": documents,
                        "representatives": representatives,
                        "band_rows": band_rows,
                    },
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            added += 1
            self._sync_checkpoint()
            if max_new_archives is not None and added >= max_new_archives:
                return False
        if int(self.db.execute("SELECT COUNT(*) FROM signature_archives").fetchone()[0]) != len(self.reports):
            raise EnglishNearDedupError("Raw signature pass did not cover every report")
        missing = int(
            self.db.execute("SELECT COUNT(*) FROM documents WHERE shingle_count IS NULL").fetchone()[0]
        )
        expected_bands = int(
            self.db.execute(
                "SELECT SUM(normalized_representative) FROM documents"
            ).fetchone()[0]
        ) * int(self.config["_total_bands"])
        actual_bands = int(self.db.execute("SELECT COUNT(*) FROM bands").fetchone()[0])
        if missing or actual_bands != expected_bands:
            raise EnglishNearDedupError(
                f"Signature completeness failure: missing={missing}, bands={actual_bands}/{expected_bands}"
            )
        self._advance(
            "signatures",
            "candidates",
            {"band_rows": actual_bands, "total_bands": self.config["_total_bands"]},
        )
        return True

    def generate_candidates(self, max_new_blocks: int | None = None) -> bool:
        if self._phase() != "candidates":
            return True
        started = time.monotonic()
        maximum_posting = int(
            self.config["candidate_signature"]["maximum_posting_documents"]
        )
        maximum_candidates = int(
            self.config["candidate_signature"]["maximum_unique_candidate_pairs"]
        )
        candidate_count = int(
            self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        )
        initial_candidate_count = candidate_count
        initial_block_count = int(
            self.db.execute("SELECT COUNT(*) FROM candidate_blocks").fetchone()[0]
        )
        numerator = int(self.config["refinement"]["minimum_jaccard_numerator"])
        denominator = int(self.config["refinement"]["minimum_jaccard_denominator"])
        new_blocks = 0
        cursor_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='candidate_cursor'"
        ).fetchone()
        if cursor_row:
            cursor_payload = json.loads(cursor_row[0])
            cursor_band = int(cursor_payload["band"])
            cursor_key = bytes.fromhex(cursor_payload["key"])
        else:
            cursor_band, cursor_key = -1, b""
        total_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='candidate_blocks_total'"
        ).fetchone()
        if total_row is None:
            candidate_blocks_total = int(
                self.db.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT 1 FROM bands
                        GROUP BY band,band_key HAVING COUNT(*) > 1
                    )
                    """
                ).fetchone()[0]
            )
            with self.db:
                self.db.execute(
                    "INSERT INTO metadata(key,value) VALUES (?,?)",
                    ("candidate_blocks_total", json.dumps(candidate_blocks_total)),
                )
        else:
            candidate_blocks_total = int(json.loads(total_row[0]))
        if not 0 <= initial_block_count <= candidate_blocks_total:
            raise EnglishNearDedupError(
                "Committed candidate blocks exceed the frozen block total"
            )
        invocation_raw_pairs = 0

        def publish_progress(*, force: bool, complete: bool = False) -> None:
            elapsed = max(time.monotonic() - started, 1e-9)
            committed_blocks = initial_block_count + new_blocks
            remaining_blocks = max(0, candidate_blocks_total - committed_blocks)
            block_rate = new_blocks / elapsed
            current_fraction = (
                1.0
                if candidate_blocks_total == 0
                else committed_blocks / candidate_blocks_total
            )
            estimated_remaining = (
                0.0
                if complete
                else (remaining_blocks / block_rate if block_rate > 0.0 else None)
            )
            self._publish_operational_metric(
                "candidates",
                {
                    "complete": complete,
                    "invocation_elapsed_seconds": round(elapsed, 6),
                    "invocation_candidate_blocks": new_blocks,
                    "invocation_raw_posting_pairs": invocation_raw_pairs,
                    "invocation_new_unique_candidate_pairs": (
                        candidate_count - initial_candidate_count
                    ),
                    "candidate_blocks_committed": committed_blocks,
                    "candidate_blocks_total": candidate_blocks_total,
                    "candidate_blocks_remaining": remaining_blocks,
                    "candidate_blocks_per_second": round(block_rate, 6),
                    "candidate_pairs_committed": candidate_count,
                    "raw_posting_pairs_per_second": round(
                        invocation_raw_pairs / elapsed, 6
                    ),
                    "candidate_pairs_per_second": round(
                        (candidate_count - initial_candidate_count) / elapsed, 6
                    ),
                    "exact_candidate_block_fraction": round(current_fraction, 9),
                    "estimated_remaining_seconds": (
                        None
                        if estimated_remaining is None
                        else round(estimated_remaining, 3)
                    ),
                    "eta_basis": "exact-duplicate-block-count/current-process-rate",
                },
                force=force,
            )

        while True:
            group = self.db.execute(
                NEXT_CANDIDATE_BLOCK_SQL,
                (cursor_band, cursor_key),
            ).fetchone()
            if group is None:
                break
            block_identity = (int(group["band"]), bytes(group["band_key"]))
            posting_count = int(group["documents"])
            if posting_count > maximum_posting:
                failure = {
                    "error": "candidate_posting_overflow",
                    "action": "fail_closed",
                    "band": block_identity[0],
                    "band_key": block_identity[1].hex(),
                    "posting_documents": posting_count,
                    "maximum_posting_documents": maximum_posting,
                    "message": (
                        "No documents were silently truncated. Increase the pinned limit only "
                        "after inspecting this band, or strengthen the candidate signature in a "
                        "new config/output generation."
                    ),
                }
                atomic_json(self.work / "FAILURE.json", failure)
                raise EnglishNearDedupError(
                    f"Candidate band posting {posting_count} exceeds fail-closed limit "
                    f"{maximum_posting} (band={block_identity[0]}, key={block_identity[1].hex()})"
                )
            posting = list(
                self.db.execute(
                    """
                    SELECT b.document,d.shingle_count
                    FROM bands AS b JOIN documents AS d ON d.ordinal=b.document
                    WHERE b.band=? AND b.band_key=? ORDER BY b.document
                    """,
                    block_identity,
                )
            )
            raw_pairs = length_pruned = 0
            candidate_batch: list[tuple[int, int]] = []
            try:
                self.db.execute("BEGIN IMMEDIATE")
                changes_before = self.db.total_changes
                for left_index, left in enumerate(posting):
                    left_count = int(left["shingle_count"])
                    for right in posting[left_index + 1 :]:
                        raw_pairs += 1
                        right_count = int(right["shingle_count"])
                        smaller = min(left_count, right_count)
                        larger = max(left_count, right_count)
                        # Jaccard(A,B) <= min(|A|,|B|)/max(|A|,|B|). This
                        # exact upper bound removes impossible pairs with zero
                        # false negatives at the configured final threshold.
                        if larger and smaller * denominator < larger * numerator:
                            length_pruned += 1
                            continue
                        candidate_batch.append(
                            (int(left["document"]), int(right["document"]))
                        )
                        if len(candidate_batch) >= self.batch_size:
                            self.db.executemany(
                                "INSERT OR IGNORE INTO candidates VALUES (?,?)",
                                candidate_batch,
                            )
                            candidate_batch.clear()
                if candidate_batch:
                    self.db.executemany(
                        "INSERT OR IGNORE INTO candidates VALUES (?,?)", candidate_batch
                    )
                new_candidates = self.db.total_changes - changes_before
                if candidate_count + new_candidates > maximum_candidates:
                    failure = {
                        "error": "global_candidate_pair_overflow",
                        "action": "fail_closed",
                        "candidate_pairs_before_block": candidate_count,
                        "new_unique_pairs_in_block": new_candidates,
                        "maximum_unique_candidate_pairs": maximum_candidates,
                        "band": block_identity[0],
                        "band_key": block_identity[1].hex(),
                        "message": (
                            "No candidate pairs were silently discarded; this whole block was "
                            "rolled back. Strengthen the pinned LSH config in a new generation."
                        ),
                    }
                    atomic_json(self.work / "FAILURE.json", failure)
                    raise EnglishNearDedupError(
                        f"Unique candidate pairs would exceed fail-closed limit "
                        f"{maximum_candidates}"
                    )
                self.db.execute(
                    "INSERT INTO candidate_blocks VALUES (?,?,?,?,?)",
                    (
                        block_identity[0],
                        block_identity[1],
                        posting_count,
                        raw_pairs,
                        length_pruned,
                    ),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES ('candidate_cursor',?)",
                    (
                        json.dumps(
                            {"band": block_identity[0], "key": block_identity[1].hex()},
                            sort_keys=True,
                        ),
                    ),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            candidate_count += new_candidates
            cursor_band, cursor_key = block_identity
            new_blocks += 1
            invocation_raw_pairs += raw_pairs
            publish_progress(force=False)
            if max_new_blocks is not None and new_blocks >= max_new_blocks:
                publish_progress(force=True)
                self._sync_checkpoint()
                return False
        self.db.executescript(
            """
            CREATE INDEX IF NOT EXISTS candidates_by_right
                ON candidates(right_document,left_document);
            CREATE INDEX IF NOT EXISTS documents_archive_ordinal ON documents(archive,ordinal);
            """
        )
        self.db.commit()
        candidate_count = int(self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        block_stats = self.db.execute(
            """
            SELECT COUNT(*),COALESCE(MAX(posting_documents),0),
                   COALESCE(SUM(raw_pairs),0),COALESCE(SUM(length_pruned_pairs),0)
            FROM candidate_blocks
            """
        ).fetchone()
        publish_progress(force=True, complete=True)
        self._advance(
            "candidates",
            "cache",
            {
                "candidate_pairs": candidate_count,
                "candidate_blocks": int(block_stats[0]),
                "maximum_posting_observed": int(block_stats[1]),
                "raw_posting_pairs": int(block_stats[2]),
                "length_pruned_pairs": int(block_stats[3]),
            },
        )
        return True

    def _candidate_documents_for_archive(self, archive: str) -> set[int]:
        return {
            int(row[0])
            for row in self.db.execute(
                CANDIDATE_DOCUMENTS_FOR_ARCHIVE_SQL,
                (archive, archive),
            )
        }

    def build_refinement_cache(self, max_new_archives: int | None = None) -> bool:
        if self._phase() != "cache":
            return True
        committed = {
            row[0]: (row[1], row[2])
            for row in self.db.execute(
                "SELECT archive,cache_file,cache_sha256 FROM cache_archives"
            )
        }
        seed = int(self.config["refinement"]["shingle_hash_seed"])
        width = int(self.config["shingle_words"])
        added = 0
        for item in self.reports:
            report = item.report
            archive = str(report["archive"])
            if archive in committed:
                cache_path = self.work / committed[archive][0]
                if not cache_path.is_file() or file_sha256(cache_path) != committed[archive][1]:
                    raise EnglishNearDedupError(f"Committed cache changed: {cache_path}")
                continue
            wanted = self._candidate_documents_for_archive(archive)
            cache_relative = Path("cache") / str(report["bucket"]) / (
                Path(archive).stem.replace(".tar", "") + ".shingles.zstframes"
            )
            cache_path = self.work / cache_relative
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_path.name}.", dir=cache_path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            entries: list[tuple[int, str, int, int, int]] = []
            offset = 0
            cache_output: BinaryIO | None = None

            def cache(document: sqlite3.Row, text: str) -> None:
                nonlocal offset
                ordinal = int(document["ordinal"])
                if ordinal not in wanted:
                    return
                values = exact_shingle_hashes(text, seed, width)
                if len(values) != int(document["shingle_count"]):
                    raise EnglishNearDedupError(
                        f"Raw shingle cardinality changed for ordinal {ordinal}"
                    )
                frame = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(
                    encode_uint64s(values)
                )
                if cache_output is None:
                    raise AssertionError("cache output was not opened")
                cache_output.write(frame)
                entries.append((ordinal, str(cache_relative), offset, len(frame), len(values)))
                offset += len(frame)

            try:
                # Create the empty file before callbacks so zero-candidate
                # archives still publish a durable, checksummed completion.
                with temporary.open("wb") as handle:
                    cache_output = handle
                    self._stream_verified_archive(report, cache)
                    handle.flush()
                    os.fsync(handle.fileno())
                    cache_output = None
                os.replace(temporary, cache_path)
                fsync_directory(cache_path.parent)
                checksum = file_sha256(cache_path)
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.executemany(
                        """
                        INSERT INTO shingle_cache(
                            document,cache_file,byte_offset,byte_length,shingle_count
                        ) VALUES (?,?,?,?,?)
                        """,
                        entries,
                    )
                    self.db.execute(
                        "INSERT INTO cache_archives VALUES (?,?,?,?)",
                        (archive, str(cache_relative), checksum, len(entries)),
                    )
                    self._event(
                        "cache_archive",
                        {"archive": archive, "cached_documents": len(entries), "bytes": offset},
                    )
                    self.db.commit()
                except BaseException:
                    self.db.rollback()
                    raise
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            added += 1
            self._sync_checkpoint()
            if max_new_archives is not None and added >= max_new_archives:
                return False
        candidate_docs = int(
            self.db.execute(
                "SELECT COUNT(*) FROM (SELECT left_document AS d FROM candidates UNION SELECT right_document FROM candidates)"
            ).fetchone()[0]
        )
        cached_docs = int(self.db.execute("SELECT COUNT(*) FROM shingle_cache").fetchone()[0])
        if cached_docs != candidate_docs:
            raise EnglishNearDedupError(
                f"Refinement cache coverage mismatch: {cached_docs}/{candidate_docs}"
            )
        self._advance(
            "cache",
            "refine",
            {"candidate_documents": candidate_docs, "cached_documents": cached_docs},
        )
        return True

    def _load_cached_shingles(
        self,
        document: int,
        cache: collections.OrderedDict[int, array.array[int]],
        maximum_cached: int = 256,
    ) -> array.array[int]:
        existing = cache.pop(document, None)
        if existing is not None:
            cache[document] = existing
            return existing
        row = self.db.execute(
            """
            SELECT cache_file,byte_offset,byte_length,shingle_count
            FROM shingle_cache WHERE document=?
            """,
            (document,),
        ).fetchone()
        if row is None:
            raise EnglishNearDedupError(f"Missing candidate shingle cache for {document}")
        path = self.work / row["cache_file"]
        with path.open("rb") as handle:
            handle.seek(int(row["byte_offset"]))
            frame = handle.read(int(row["byte_length"]))
        values = decode_uint64s(zstandard.ZstdDecompressor().decompress(frame))
        if len(values) != int(row["shingle_count"]):
            raise EnglishNearDedupError(f"Corrupt candidate cache for {document}")
        cache[document] = values
        while len(cache) > maximum_cached:
            cache.popitem(last=False)
        return values

    def _refinement_preflight_identity(
        self, total_candidates: int, *, verify_cache_files: bool
    ) -> dict[str, Any]:
        cache_inventory: list[dict[str, Any]] = []
        cache_bytes = 0
        for row in self.db.execute(
            """
            SELECT archive,cache_file,cache_sha256,cached_documents
            FROM cache_archives ORDER BY archive
            """
        ):
            path = self.work / str(row["cache_file"])
            if not path.is_file() or path.is_symlink():
                raise EnglishNearDedupError(
                    f"Missing or unsafe refinement cache: {path}"
                )
            size = path.stat().st_size
            if verify_cache_files and file_sha256(path) != row["cache_sha256"]:
                raise EnglishNearDedupError(
                    f"Refinement cache checksum changed: {path}"
                )
            cache_bytes += size
            cache_inventory.append(
                {
                    "archive": row["archive"],
                    "cache_file": row["cache_file"],
                    "cache_sha256": row["cache_sha256"],
                    "cached_documents": int(row["cached_documents"]),
                    "bytes": size,
                }
            )
        if len(cache_inventory) != len(self.reports):
            raise EnglishNearDedupError(
                "Operational preflight cache archive coverage is incomplete"
            )
        total_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='candidate_blocks_total'"
        ).fetchone()
        if total_row is None:
            raise EnglishNearDedupError(
                "Operational preflight lacks candidate-block authority"
            )
        candidate_cursor_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='candidate_cursor'"
        ).fetchone()
        candidate_blocks_total = int(json.loads(total_row[0]))
        candidate_blocks_committed = int(
            self.db.execute("SELECT COUNT(*) FROM candidate_blocks").fetchone()[0]
        )
        if candidate_blocks_committed != candidate_blocks_total:
            raise EnglishNearDedupError(
                "Operational preflight candidate-block coverage is incomplete"
            )
        documents_total = int(
            self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        return {
            "contract_version": 1,
            "builder_sha256": self.identity["builder_sha256"],
            "builder_identity_sha256": canonical_sha256(self.identity),
            "config_file_sha256": self.config_file_sha,
            "config_sha256": self.config_sha,
            "calibration_evidence_sha256": canonical_sha256(
                self.calibration_evidence
            ),
            "report_inventory_sha256": self.report_inventory_sha,
            "preprocess_manifest_sha256": self.preprocess_manifest_sha,
            "curation_policy_sha256": self.policy_sha,
            "benchmark_guard_sha256": self.guard.manifest_sha256,
            "collection_completeness_sha256": canonical_sha256(
                self.collection_completeness
            ),
            "candidate_pairs_total": total_candidates,
            "documents_total": documents_total,
            "candidate_blocks_total": candidate_blocks_total,
            "candidate_blocks_committed": candidate_blocks_committed,
            "phase_at_measurement": "refine",
            "candidate_cursor_at_measurement": (
                None
                if candidate_cursor_row is None
                else json.loads(candidate_cursor_row[0])
            ),
            "refinement_cursor_at_measurement": None,
            "cache_archives": len(cache_inventory),
            "cache_bytes": cache_bytes,
            "cache_inventory_sha256": canonical_sha256(cache_inventory),
            "runtime_storage": self.storage_evidence,
        }

    def _sample_refinement_preflight_pairs(
        self, total_candidates: int
    ) -> list[tuple[int, int]]:
        if total_candidates == 0:
            return []
        config = self.config["operational_preflight"]
        target = min(int(config["requested_pairs"]), total_candidates)
        bounds = self.db.execute(
            """
            SELECT MIN(left_document),MAX(left_document),MAX(right_document)
            FROM candidates
            """
        ).fetchone()
        minimum_left = int(bounds[0])
        maximum_left = int(bounds[1])
        maximum_right = int(bounds[2])
        strata = min(int(config["strata"]), target)
        successors = int(config["successors_per_probe"])
        seed = int(config["sampling_seed"])
        selected: set[tuple[int, int]] = set()
        for pass_index in range(int(config["maximum_probe_passes"])):
            for stratum in range(strata):
                target_left = minimum_left + (
                    (2 * stratum + 1) * (maximum_left - minimum_left + 1)
                ) // (2 * strata)
                target_right = xxhash.xxh3_64_intdigest(
                    struct.pack(">II", pass_index, stratum), seed=seed
                ) % (maximum_right + 1)
                rows = self.db.execute(
                    """
                    SELECT left_document,right_document FROM candidates
                    WHERE (left_document,right_document) >= (?,?)
                    ORDER BY left_document,right_document LIMIT ?
                    """,
                    (target_left, target_right, successors),
                )
                for row in rows:
                    selected.add((int(row[0]), int(row[1])))
                    if len(selected) == target:
                        return sorted(selected)
            if len(selected) == target:
                break
        return sorted(selected)

    def _run_refinement_operational_preflight(
        self, total_candidates: int
    ) -> dict[str, Any]:
        if self._phase() != "refine":
            raise EnglishNearDedupError(
                "Refinement operational preflight must run at the refine boundary"
            )
        if self.db.execute("SELECT 1 FROM refined LIMIT 1").fetchone() is not None:
            raise EnglishNearDedupError(
                "Refinement started before its operational preflight"
            )
        identity = self._refinement_preflight_identity(
            total_candidates, verify_cache_files=True
        )
        config = dict(self.config["operational_preflight"])
        pairs = self._sample_refinement_preflight_pairs(total_candidates)
        expected_sample = min(int(config["requested_pairs"]), total_candidates)
        sample_sha = canonical_sha256(
            [
                {"left_document": left, "right_document": right}
                for left, right in pairs
            ]
        )
        before = self._resource_snapshot()
        union_projection = self._union_memory_projection(
            int(identity["documents_total"]), before["peak_process_rss_bytes"]
        )
        results: list[tuple[int, int, int, int, int]] = []
        cache: collections.OrderedDict[int, array.array[int]] = collections.OrderedDict()
        accepted = 0
        elapsed = 0.0
        measured_growth = 0
        with tempfile.TemporaryDirectory(
            prefix=".refinement-preflight-measurement.", dir=self.work
        ) as temporary:
            measurement_db = Path(temporary) / "measurement.sqlite3"
            measurement = sqlite3.connect(measurement_db)
            try:
                selected_journal = str(
                    self.storage_evidence["sqlite_journal_mode_selected"]
                )
                actual = str(
                    measurement.execute(
                        f"PRAGMA journal_mode={selected_journal.upper()}"
                    ).fetchone()[0]
                ).casefold()
                if actual != selected_journal:
                    raise EnglishNearDedupError(
                        "Operational preflight could not mirror SQLite journal mode"
                    )
                measurement.execute("PRAGMA synchronous=FULL")
                measurement.executescript(
                    """
                    CREATE TABLE refined(
                        left_document INTEGER NOT NULL,
                        right_document INTEGER NOT NULL,
                        intersection_count INTEGER NOT NULL,
                        union_count INTEGER NOT NULL,
                        accepted INTEGER NOT NULL,
                        PRIMARY KEY(left_document,right_document)
                    ) WITHOUT ROWID;
                    CREATE TABLE edges(
                        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        left_document INTEGER NOT NULL,
                        right_document INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        UNIQUE(left_document,right_document,kind)
                    );
                    """
                )
                measurement.commit()
                baseline = sum(
                    path.stat().st_size
                    for path in Path(temporary).iterdir()
                    if path.is_file()
                )
                started = time.monotonic()
                for offset in range(0, len(pairs), self.batch_size):
                    batch: list[tuple[int, int, int, int, int]] = []
                    for left_id, right_id in pairs[offset : offset + self.batch_size]:
                        left = self._load_cached_shingles(left_id, cache)
                        right = self._load_cached_shingles(right_id, cache)
                        intersection, union = jaccard_counts(left, right)
                        decision = int(
                            refinement_accepts(intersection, union, self.config)
                        )
                        accepted += decision
                        batch.append(
                            (left_id, right_id, intersection, union, decision)
                        )
                    with measurement:
                        measurement.executemany(
                            "INSERT INTO refined VALUES (?,?,?,?,?)", batch
                        )
                        measurement.executemany(
                            """
                            INSERT INTO edges(left_document,right_document,kind)
                            VALUES (?,?,'near')
                            """,
                            [
                                (left, right)
                                for left, right, _intersection, _union, decision in batch
                                if decision
                            ],
                        )
                    results.extend(batch)
                elapsed = max(time.monotonic() - started, 1e-9)
            finally:
                measurement.close()
            measured_growth = max(
                0,
                sum(
                    path.stat().st_size
                    for path in Path(temporary).iterdir()
                    if path.is_file()
                )
                - baseline,
            )
        after = self._resource_snapshot()
        # Publish formula inputs at fixed precision first, then derive every
        # projection from those published inputs.  The loader can therefore
        # recompute the complete rate/ETA/disk arithmetic exactly instead of
        # trusting mutually inconsistent, checksum-rewritten numbers.
        published_elapsed = max(round(elapsed, 6), 0.000001)
        pairs_per_second = (
            round(len(results) / published_elapsed, 6) if results else None
        )
        bytes_per_pair = measured_growth / len(results) if results else 0.0
        projected_seconds = (
            0.0
            if total_candidates == 0
            else (
                total_candidates / float(pairs_per_second)
                if pairs_per_second is not None and pairs_per_second > 0.0
                else None
            )
        )
        projected_growth = int(math.ceil(bytes_per_pair * total_candidates))
        safety_growth = math.ceil(
            projected_growth
            * int(config["disk_projection_safety_numerator"])
            / int(config["disk_projection_safety_denominator"])
        )
        required_free = (
            safety_growth + int(config["minimum_post_refinement_free_bytes"])
        )
        failures: list[str] = []
        if len(results) != expected_sample:
            failures.append("representative_sample_incomplete")
        if (
            total_candidates >= int(config["requested_pairs"])
            and (
                pairs_per_second is None
                or pairs_per_second
                < float(config["minimum_production_pairs_per_second"])
            )
        ):
            failures.append("refinement_throughput_below_minimum")
        if (
            projected_seconds is None
            or projected_seconds
            > int(config["maximum_projected_refinement_seconds"])
        ):
            failures.append("projected_refinement_eta_exceeds_window")
        if (
            after["peak_process_rss_bytes"]
            > int(config["maximum_peak_process_rss_bytes"])
        ):
            failures.append("peak_rss_exceeds_limit")
        if union_projection["union_projected_peak_process_rss_bytes"] > int(
            config["maximum_peak_process_rss_bytes"]
        ):
            failures.append("projected_union_peak_rss_exceeds_limit")
        if after["filesystem_free_bytes"] < required_free:
            failures.append("projected_refinement_disk_exceeds_free_space")
        result = {
            "result_version": 1,
            "status": "pass" if not failures else "fail",
            "production_gate_eligible": not failures,
            "failures": failures,
            "identity": identity,
            "thresholds": config,
            "sample": {
                "algorithm": config["sampling_algorithm"],
                "seed": config["sampling_seed"],
                "requested_pairs": config["requested_pairs"],
                "expected_pairs": expected_sample,
                "measured_pairs": len(results),
                "sample_pairs_sha256": sample_sha,
                "accepted_pairs": accepted,
                "sample_limited_by_total_candidates": (
                    total_candidates < int(config["requested_pairs"])
                ),
            },
            "measurements": {
                "sample_elapsed_seconds": published_elapsed,
                "measurement_batches": (
                    math.ceil(len(results) / self.batch_size) if results else 0
                ),
                "measurement_batch_size": self.batch_size,
                "refinement_pairs_per_second": (
                    None
                    if pairs_per_second is None
                    else pairs_per_second
                ),
                "candidate_pairs_total": total_candidates,
                "projected_refinement_seconds": (
                    None
                    if projected_seconds is None
                    else round(projected_seconds, 3)
                ),
                "measurement_sqlite_growth_bytes": measured_growth,
                "measurement_sqlite_bytes_per_pair": round(bytes_per_pair, 6),
                "projected_additional_refinement_sqlite_bytes": projected_growth,
                "projected_additional_with_safety_bytes": safety_growth,
                "required_filesystem_free_bytes": required_free,
                **union_projection,
                "resources_before": before,
                "resources_after": after,
            },
            "statistical_scope": (
                "Keyspace-stratified successor sampling covers deterministic regions of "
                "the frozen candidate primary-key space and exercises the real shingle "
                "cache plus SQLite writes. It is bounded operational evidence, not a "
                "random-sample confidence bound or a guarantee against later skew."
            ),
        }
        return result

    def _publish_refinement_preflight(self, result: dict[str, Any]) -> dict[str, Any]:
        destination = self.output / "operational-preflight-v1"
        if destination.exists():
            raise EnglishNearDedupError(
                f"Operational preflight destination already exists: {destination}"
            )
        payload = json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with tempfile.TemporaryDirectory(
            prefix=".operational-preflight-v1.", dir=self.output
        ) as temporary:
            temporary_path = Path(temporary)
            result_path = temporary_path / "result.json"
            checksum_path = temporary_path / "result.json.sha256"
            result_path.write_bytes(payload)
            checksum_path.write_bytes(
                hashlib.sha256(payload).hexdigest().encode("ascii")
                + b"  result.json\n"
            )
            for path in (result_path, checksum_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            fsync_directory(temporary_path)
            os.rename(temporary_path, destination)
            fsync_directory(self.output)
        return self._load_refinement_preflight(verify_cache_files=False)

    def _load_refinement_preflight(
        self, *, verify_cache_files: bool
    ) -> dict[str, Any]:
        directory = self.output / "operational-preflight-v1"
        if directory.is_symlink() or not directory.is_dir():
            raise EnglishNearDedupError(
                f"Missing or unsafe operational preflight: {directory}"
            )
        result_path = directory / "result.json"
        checksum_path = directory / "result.json.sha256"
        if any(path.is_symlink() or not path.is_file() for path in (result_path, checksum_path)):
            raise EnglishNearDedupError(
                "Operational preflight result/checksum is missing or unsafe"
            )
        raw = result_path.read_bytes()
        result_sha = hashlib.sha256(raw).hexdigest()
        sidecar_raw = checksum_path.read_bytes()
        if sidecar_raw != f"{result_sha}  result.json\n".encode("ascii"):
            raise EnglishNearDedupError("Operational preflight checksum mismatch")
        result = self._json_without_duplicate_keys(
            raw, "refinement operational preflight result"
        )
        if set(result) != {
            "result_version",
            "status",
            "production_gate_eligible",
            "failures",
            "identity",
            "thresholds",
            "sample",
            "measurements",
            "statistical_scope",
        }:
            raise EnglishNearDedupError(
                "Operational preflight result schema is incomplete"
            )
        candidate_event = self.db.execute(
            "SELECT payload FROM events WHERE event='cache' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if candidate_event is None:
            raise EnglishNearDedupError(
                "Operational preflight lacks candidate-count authority"
            )
        total_candidates = int(json.loads(candidate_event[0])["candidate_pairs"])
        expected_identity = self._refinement_preflight_identity(
            total_candidates, verify_cache_files=verify_cache_files
        )
        expected_gate = {
            "result_version": 1,
            "status": "pass",
            "production_gate_eligible": True,
            "failures": [],
            "identity": expected_identity,
            "thresholds": self.config["operational_preflight"],
        }
        for field, expected in expected_gate.items():
            if result.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Operational preflight {field} is not production-passing"
                )
        sample = result.get("sample")
        measurements = result.get("measurements")
        if not isinstance(sample, dict) or set(sample) != {
            "algorithm",
            "seed",
            "requested_pairs",
            "expected_pairs",
            "measured_pairs",
            "sample_pairs_sha256",
            "accepted_pairs",
            "sample_limited_by_total_candidates",
        }:
            raise EnglishNearDedupError(
                "Operational preflight sample schema is incomplete"
            )
        if not isinstance(measurements, dict) or set(measurements) != {
            "sample_elapsed_seconds",
            "measurement_batches",
            "measurement_batch_size",
            "refinement_pairs_per_second",
            "candidate_pairs_total",
            "projected_refinement_seconds",
            "measurement_sqlite_growth_bytes",
            "measurement_sqlite_bytes_per_pair",
            "projected_additional_refinement_sqlite_bytes",
            "projected_additional_with_safety_bytes",
            "required_filesystem_free_bytes",
            "union_parent_item_bytes",
            "union_parent_array_projected_bytes",
            "union_parent_array_with_safety_bytes",
            "union_projected_peak_process_rss_bytes",
            "resources_before",
            "resources_after",
        }:
            raise EnglishNearDedupError(
                "Operational preflight measurement schema is incomplete"
            )
        thresholds = self.config["operational_preflight"]
        expected_pairs = min(int(thresholds["requested_pairs"]), total_candidates)
        expected_sample_rows = self._sample_refinement_preflight_pairs(total_candidates)
        expected_sample_sha = canonical_sha256(
            [
                {"left_document": left, "right_document": right}
                for left, right in expected_sample_rows
            ]
        )
        parse_digest(sample.get("sample_pairs_sha256"), "preflight sample sha256")
        for field, expected in {
            "algorithm": thresholds["sampling_algorithm"],
            "seed": thresholds["sampling_seed"],
            "requested_pairs": thresholds["requested_pairs"],
            "expected_pairs": expected_pairs,
            "measured_pairs": expected_pairs,
            "sample_pairs_sha256": expected_sample_sha,
            "sample_limited_by_total_candidates": (
                total_candidates < int(thresholds["requested_pairs"])
            ),
        }.items():
            if sample.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Operational preflight sample {field} mismatch"
                )
        accepted_pairs = sample.get("accepted_pairs")
        if (
            not isinstance(accepted_pairs, int)
            or isinstance(accepted_pairs, bool)
            or not 0 <= accepted_pairs <= expected_pairs
        ):
            raise EnglishNearDedupError(
                "Operational preflight accepted-pair count is invalid"
            )
        if measurements.get("candidate_pairs_total") != total_candidates:
            raise EnglishNearDedupError(
                "Operational preflight candidate total mismatch"
            )
        expected_batches = (
            math.ceil(expected_pairs / self.batch_size) if expected_pairs else 0
        )
        for field, expected in {
            "measurement_batches": expected_batches,
            "measurement_batch_size": self.batch_size,
        }.items():
            value = measurements.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value != expected
            ):
                raise EnglishNearDedupError(
                    f"Operational preflight {field} is invalid or inconsistent"
                )
        elapsed = measurements.get("sample_elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise EnglishNearDedupError(
                "Operational preflight elapsed time is invalid"
            )
        rate = measurements.get("refinement_pairs_per_second")
        projected_seconds = measurements.get("projected_refinement_seconds")
        if total_candidates == 0:
            if rate is not None or projected_seconds != 0.0:
                raise EnglishNearDedupError(
                    "Operational preflight zero-candidate metrics are invalid"
                )
        elif (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(float(rate))
            or float(rate) <= 0.0
            or not isinstance(projected_seconds, (int, float))
            or isinstance(projected_seconds, bool)
            or not math.isfinite(float(projected_seconds))
            or float(projected_seconds) < 0.0
        ):
            raise EnglishNearDedupError(
                "Operational preflight rate/ETA metrics are invalid"
            )
        expected_rate = (
            None if expected_pairs == 0 else round(expected_pairs / float(elapsed), 6)
        )
        expected_projected_seconds = (
            0.0
            if total_candidates == 0
            else round(total_candidates / float(expected_rate), 3)
        )
        if rate != expected_rate or projected_seconds != expected_projected_seconds:
            raise EnglishNearDedupError(
                "Operational preflight rate/ETA formulas are inconsistent"
            )
        measured_growth = measurements.get("measurement_sqlite_growth_bytes")
        if (
            not isinstance(measured_growth, int)
            or isinstance(measured_growth, bool)
            or measured_growth < 0
        ):
            raise EnglishNearDedupError(
                "Operational preflight measured SQLite growth is invalid"
            )
        expected_bytes_per_pair = (
            measured_growth / expected_pairs if expected_pairs else 0.0
        )
        expected_projected_growth = int(
            math.ceil(expected_bytes_per_pair * total_candidates)
        )
        expected_safety_growth = math.ceil(
            expected_projected_growth
            * int(thresholds["disk_projection_safety_numerator"])
            / int(thresholds["disk_projection_safety_denominator"])
        )
        expected_required_free = expected_safety_growth + int(
            thresholds["minimum_post_refinement_free_bytes"]
        )
        for field, expected in {
            "measurement_sqlite_bytes_per_pair": round(
                expected_bytes_per_pair, 6
            ),
            "projected_additional_refinement_sqlite_bytes": expected_projected_growth,
            "projected_additional_with_safety_bytes": expected_safety_growth,
            "required_filesystem_free_bytes": expected_required_free,
        }.items():
            if measurements.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Operational preflight disk formula {field} is inconsistent"
                )
        for resource_name in ("resources_before", "resources_after"):
            resources = measurements.get(resource_name)
            if not isinstance(resources, dict) or set(resources) != {
                "filesystem_total_bytes",
                "filesystem_free_bytes",
                "sqlite_state_bytes",
                "refinement_cache_bytes",
                "peak_process_rss_bytes",
            } or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in resources.values()
            ):
                raise EnglishNearDedupError(
                    f"Operational preflight {resource_name} is invalid"
                )
            if resources["filesystem_free_bytes"] > resources["filesystem_total_bytes"]:
                raise EnglishNearDedupError(
                    f"Operational preflight {resource_name} free space exceeds total"
                )
            if resources["refinement_cache_bytes"] != int(
                expected_identity["cache_bytes"]
            ):
                raise EnglishNearDedupError(
                    f"Operational preflight {resource_name} cache bytes mismatch"
                )
        resources_before = measurements["resources_before"]
        resources_after = measurements["resources_after"]
        if resources_after["peak_process_rss_bytes"] < resources_before[
            "peak_process_rss_bytes"
        ]:
            raise EnglishNearDedupError(
                "Operational preflight peak RSS regressed"
            )
        expected_union_projection = self._union_memory_projection(
            int(expected_identity["documents_total"]),
            resources_before["peak_process_rss_bytes"],
        )
        for field, expected in expected_union_projection.items():
            if measurements.get(field) != expected:
                raise EnglishNearDedupError(
                    f"Operational preflight union formula {field} is inconsistent"
                )
        if total_candidates >= int(thresholds["requested_pairs"]) and float(rate) < float(
            thresholds["minimum_production_pairs_per_second"]
        ):
            raise EnglishNearDedupError(
                "Operational preflight throughput no longer passes thresholds"
            )
        if projected_seconds is None or float(projected_seconds) > int(
            thresholds["maximum_projected_refinement_seconds"]
        ):
            raise EnglishNearDedupError(
                "Operational preflight projected ETA no longer passes thresholds"
            )
        if resources_after["peak_process_rss_bytes"] > int(
            thresholds["maximum_peak_process_rss_bytes"]
        ):
            raise EnglishNearDedupError(
                "Operational preflight peak RSS no longer passes thresholds"
            )
        if measurements["union_projected_peak_process_rss_bytes"] > int(
            thresholds["maximum_peak_process_rss_bytes"]
        ):
            raise EnglishNearDedupError(
                "Operational preflight union peak RSS projection no longer passes thresholds"
            )
        required_free = measurements.get("required_filesystem_free_bytes")
        if (
            not isinstance(required_free, int)
            or isinstance(required_free, bool)
            or required_free < 0
            or resources_after["filesystem_free_bytes"] < required_free
        ):
            raise EnglishNearDedupError(
                "Operational preflight disk projection no longer passes thresholds"
            )
        return {
            "contract_version": 1,
            "result_path": "operational-preflight-v1/result.json",
            "result_sha256": result_sha,
            "result_bytes": len(raw),
            "sidecar_path": "operational-preflight-v1/result.json.sha256",
            "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
            "status": result["status"],
            "production_gate_eligible": result["production_gate_eligible"],
            "failures": result["failures"],
            "identity_sha256": canonical_sha256(result["identity"]),
            "identity": result["identity"],
            "thresholds": result["thresholds"],
            "sample": result["sample"],
            "measurements": result["measurements"],
        }

    def _ensure_refinement_operational_preflight(
        self, total_candidates: int
    ) -> dict[str, Any]:
        if self.refinement_preflight_evidence is not None:
            current = self._load_refinement_preflight(verify_cache_files=False)
            if current != self.refinement_preflight_evidence:
                raise EnglishNearDedupError(
                    "Operational preflight evidence changed after validation"
                )
            return current
        destination = self.output / "operational-preflight-v1"
        if destination.exists():
            evidence = self._load_refinement_preflight(verify_cache_files=True)
        else:
            result = self._run_refinement_operational_preflight(total_candidates)
            evidence = self._publish_refinement_preflight(result)
        self.refinement_preflight_evidence = evidence
        if evidence["status"] != "pass" or evidence["failures"]:
            raise EnglishNearDedupError(
                "Refinement operational preflight did not pass"
            )
        return evidence

    def refine_candidates(self, max_new_pairs: int | None = None) -> bool:
        if self._phase() != "refine":
            return True
        started = time.monotonic()
        cache: collections.OrderedDict[int, array.array[int]] = collections.OrderedDict()
        cursor_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='refinement_cursor'"
        ).fetchone()
        if cursor_row is None:
            cursor = (-1, -1)
            committed_pairs = 0
        else:
            cursor_payload = json.loads(cursor_row[0])
            cursor = (
                int(cursor_payload["left_document"]),
                int(cursor_payload["right_document"]),
            )
            committed_pairs = int(cursor_payload["processed_pairs"])
        candidate_event = self.db.execute(
            """
            SELECT payload FROM events WHERE event='cache'
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        if candidate_event is None:
            raise EnglishNearDedupError(
                "Missing committed candidate-count authority for refinement"
            )
        total_candidates = int(json.loads(candidate_event[0])["candidate_pairs"])
        if not 0 <= committed_pairs <= total_candidates:
            raise EnglishNearDedupError(
                "Refinement cursor count exceeds the frozen candidate count"
            )
        preflight = self._ensure_refinement_operational_preflight(total_candidates)
        initial_resources = self._resource_snapshot()
        last_refined_row = self.db.execute(
            """
            SELECT left_document,right_document FROM refined
            ORDER BY left_document DESC,right_document DESC LIMIT 1
            """
        ).fetchone()
        last_refined = (
            None
            if last_refined_row is None
            else (int(last_refined_row[0]), int(last_refined_row[1]))
        )
        if (cursor_row is None and last_refined is not None) or (
            cursor_row is not None and last_refined != cursor
        ):
            raise EnglishNearDedupError(
                "Refinement cursor does not match committed refined prefix"
            )
        if cursor_row is not None and self.db.execute(
            """
            SELECT 1 FROM candidates
            WHERE left_document=? AND right_document=?
            """,
            cursor,
        ).fetchone() is None:
            raise EnglishNearDedupError(
                "Refinement cursor is not present in the frozen candidate set"
            )
        processed = 0

        thresholds = preflight["thresholds"]
        preflight_bytes_per_pair = float(
            preflight["measurements"]["measurement_sqlite_bytes_per_pair"]
        )

        def required_free_bytes(remaining_pairs: int) -> int:
            projected = math.ceil(preflight_bytes_per_pair * remaining_pairs)
            with_safety = math.ceil(
                projected
                * int(thresholds["disk_projection_safety_numerator"])
                / int(thresholds["disk_projection_safety_denominator"])
            )
            return with_safety + int(
                thresholds["minimum_post_refinement_free_bytes"]
            )

        def enforce_live_resources(
            resources: dict[str, int], *, context: str
        ) -> None:
            failures: list[str] = []
            if resources["peak_process_rss_bytes"] > int(
                thresholds["maximum_peak_process_rss_bytes"]
            ):
                failures.append("live_refinement_peak_rss_exceeds_limit")
            required_free = required_free_bytes(
                max(0, total_candidates - committed_pairs)
            )
            if resources["filesystem_free_bytes"] < required_free:
                failures.append("live_refinement_disk_projection_exceeds_free_space")
            if not failures:
                return
            self._publish_operational_metric(
                "refine",
                {
                    "status": "fail",
                    "failures": failures,
                    "complete": False,
                    "resource_gate_context": context,
                    "refinement_pairs_committed": committed_pairs,
                    "refinement_pairs_total": total_candidates,
                    "refinement_pairs_remaining": max(
                        0, total_candidates - committed_pairs
                    ),
                    "required_filesystem_free_bytes": required_free,
                    "maximum_peak_process_rss_bytes": int(
                        thresholds["maximum_peak_process_rss_bytes"]
                    ),
                    "preflight_identity_sha256": preflight["identity_sha256"],
                },
                force=True,
                resources=resources,
            )
            raise EnglishNearDedupError(
                "Live refinement operational gate failed: " + ",".join(failures)
            )

        enforce_live_resources(initial_resources, context="before-refinement")

        def publish_progress(
            *,
            force: bool,
            complete: bool = False,
            resources: dict[str, int] | None = None,
        ) -> None:
            if (
                not force
                and time.monotonic() - self._last_progress_at
                < self.progress_interval_seconds
            ):
                return
            elapsed = max(time.monotonic() - started, 1e-9)
            remaining = max(0, total_candidates - committed_pairs)
            rate = processed / elapsed
            resources = resources or self._resource_snapshot()
            sqlite_growth = max(
                0,
                resources["sqlite_state_bytes"]
                - initial_resources["sqlite_state_bytes"],
            )
            bytes_per_pair = sqlite_growth / processed if processed else None
            estimated_disk = (
                None
                if bytes_per_pair is None
                else int(round(bytes_per_pair * remaining))
            )
            self._publish_operational_metric(
                "refine",
                {
                    "status": "pass" if complete else "running",
                    "failures": [],
                    "complete": complete,
                    "invocation_elapsed_seconds": round(elapsed, 6),
                    "invocation_refinement_pairs": processed,
                    "refinement_pairs_committed": committed_pairs,
                    "refinement_pairs_total": total_candidates,
                    "refinement_pairs_remaining": remaining,
                    "refinement_pairs_per_second": round(rate, 6),
                    "estimated_remaining_seconds": (
                        0.0
                        if complete
                        else (round(remaining / rate, 3) if rate > 0.0 else None)
                    ),
                    "observed_sqlite_growth_bytes": sqlite_growth,
                    "observed_sqlite_bytes_per_pair": (
                        None
                        if bytes_per_pair is None
                        else round(bytes_per_pair, 6)
                    ),
                    "estimated_additional_refinement_sqlite_bytes": estimated_disk,
                    "required_filesystem_free_bytes": required_free_bytes(remaining),
                    "maximum_peak_process_rss_bytes": int(
                        thresholds["maximum_peak_process_rss_bytes"]
                    ),
                    "preflight_identity_sha256": preflight["identity_sha256"],
                    "eta_basis": "current-process-sustained-refinement-rate",
                },
                force=force,
                resources=resources,
            )

        while True:
            limit = self.batch_size
            if max_new_pairs is not None:
                remaining = max_new_pairs - processed
                if remaining <= 0:
                    resources = self._resource_snapshot()
                    enforce_live_resources(resources, context="controlled-stop")
                    publish_progress(force=True, resources=resources)
                    self._sync_checkpoint()
                    return False
                limit = min(limit, remaining)
            rows = list(
                self.db.execute(
                    REFINEMENT_BATCH_SQL,
                    (*cursor, limit),
                )
            )
            if not rows:
                break
            batch: list[tuple[int, int, int, int, int]] = []
            for row in rows:
                left_id = int(row["left_document"])
                right_id = int(row["right_document"])
                left = self._load_cached_shingles(left_id, cache)
                right = self._load_cached_shingles(right_id, cache)
                intersection, union = jaccard_counts(left, right)
                accepted = int(refinement_accepts(intersection, union, self.config))
                batch.append((left_id, right_id, intersection, union, accepted))
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany("INSERT INTO refined VALUES (?,?,?,?,?)", batch)
                self.db.executemany(
                    """
                    INSERT OR IGNORE INTO edges(left_document,right_document,kind)
                    VALUES (?,?, 'near')
                    """,
                    [
                        (left, right)
                        for left, right, _intersection, _union, accepted in batch
                        if accepted
                    ],
                )
                cursor = (int(batch[-1][0]), int(batch[-1][1]))
                committed_pairs += len(batch)
                self.db.execute(
                    """
                    INSERT OR REPLACE INTO metadata(key,value)
                    VALUES ('refinement_cursor',?)
                    """,
                    (
                        json.dumps(
                            {
                                "left_document": cursor[0],
                                "right_document": cursor[1],
                                "processed_pairs": committed_pairs,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            processed += len(batch)
            resources = self._resource_snapshot()
            enforce_live_resources(resources, context="committed-batch")
            publish_progress(force=False, resources=resources)
        candidates = int(self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        refined = int(self.db.execute("SELECT COUNT(*) FROM refined").fetchone()[0])
        missing = self.db.execute(
            """
            SELECT 1 FROM candidates AS c
            LEFT JOIN refined AS r USING(left_document,right_document)
            WHERE r.left_document IS NULL LIMIT 1
            """
        ).fetchone()
        unknown = self.db.execute(
            """
            SELECT 1 FROM refined AS r
            LEFT JOIN candidates AS c USING(left_document,right_document)
            WHERE c.left_document IS NULL LIMIT 1
            """
        ).fetchone()
        if refined != candidates or missing is not None or unknown is not None:
            raise EnglishNearDedupError(
                "Refinement coverage mismatch: "
                f"refined={refined}, candidates={candidates}, "
                f"missing={int(missing is not None)}, unknown={int(unknown is not None)}"
            )
        # Equal preprocessing-normalized content is a certain duplicate edge,
        # even when only one representative entered probabilistic LSH.
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                """
                INSERT OR IGNORE INTO edges(left_document,right_document,kind)
                SELECT grouped.root,grouped.ordinal,'normalized'
                FROM (
                    SELECT ordinal,MIN(ordinal) OVER (PARTITION BY normalized_hash) AS root,
                           COUNT(*) OVER (PARTITION BY normalized_hash) AS members
                    FROM documents
                ) AS grouped
                WHERE grouped.members > 1 AND grouped.ordinal != grouped.root
                """
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        accepted = int(self.db.execute("SELECT COUNT(*) FROM refined WHERE accepted=1").fetchone()[0])
        edges = int(self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        resources = self._resource_snapshot()
        enforce_live_resources(resources, context="refinement-complete")
        publish_progress(force=True, complete=True, resources=resources)
        self._advance(
            "refine",
            "union",
            {"candidate_pairs": candidates, "accepted_near_edges": accepted, "all_edges": edges},
        )
        return True

    def union_edges(
        self,
        max_new_edges: int | None = None,
        max_new_finalization_documents: int | None = None,
    ) -> bool:
        if self._phase() != "union":
            return True
        for name, value in (
            ("max_new_edges", max_new_edges),
            (
                "max_new_finalization_documents",
                max_new_finalization_documents,
            ),
        ):
            if value is not None and value < 1:
                raise EnglishNearDedupError(f"{name} must be positive when supplied")

        document_stats = self.db.execute(
            "SELECT COUNT(*),COALESCE(MAX(ordinal),0) FROM documents"
        ).fetchone()
        document_count = int(document_stats[0])
        maximum_ordinal = int(document_stats[1])
        if document_count != maximum_ordinal:
            raise EnglishNearDedupError(
                "Frozen inventory ordinals are not dense; cannot use durable union accelerator"
            )
        edge_totals = self.db.execute(
            "SELECT COUNT(*),COALESCE(MAX(edge_id),0) FROM edges"
        ).fetchone()
        total_edges = int(edge_totals[0])
        maximum_edge_id = int(edge_totals[1])
        cursor_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='union_edge_cursor'"
        ).fetchone()
        cursor = int(json.loads(cursor_row[0])) if cursor_row else 0
        merged_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='union_merges'"
        ).fetchone()
        merges = int(json.loads(merged_row[0])) if merged_row else 0
        if cursor > maximum_edge_id:
            raise EnglishNearDedupError(
                "Union edge cursor exceeds the frozen edge inventory"
            )

        rss_limit = int(
            self.config["operational_preflight"]["maximum_peak_process_rss_bytes"]
        )
        initial_resources = self._resource_snapshot()
        memory_projection = self._union_memory_projection(
            document_count, initial_resources["peak_process_rss_bytes"]
        )

        def publish_progress(
            stage: str,
            *,
            force: bool,
            stage_started_at: float,
            stage_committed: int,
            stage_total: int,
            stage_invocation_units: int,
            complete: bool = False,
            resources: dict[str, int] | None = None,
        ) -> None:
            snapshot = resources or self._resource_snapshot()
            elapsed = max(time.monotonic() - stage_started_at, 1e-9)
            rate = stage_invocation_units / elapsed
            remaining = max(0, stage_total - stage_committed)
            self._publish_operational_metric(
                "union",
                {
                    "status": "pass" if complete else "running",
                    "failures": [],
                    "complete": complete,
                    "union_stage": stage,
                    "invocation_elapsed_seconds": round(elapsed, 6),
                    "invocation_stage_units": stage_invocation_units,
                    "stage_units_committed": stage_committed,
                    "stage_units_total": stage_total,
                    "stage_units_remaining": remaining,
                    "stage_units_per_second": round(rate, 6),
                    "estimated_remaining_seconds": (
                        0.0
                        if complete or remaining == 0
                        else (round(remaining / rate, 3) if rate > 0.0 else None)
                    ),
                    "eta_basis": "exact-stage-unit-count/current-process-rate",
                    "maximum_peak_process_rss_bytes": rss_limit,
                    **memory_projection,
                },
                force=force,
                resources=snapshot,
            )

        def fail_rss(
            failure: str, stage: str, resources: dict[str, int]
        ) -> None:
            self._publish_operational_metric(
                "union",
                {
                    "status": "fail",
                    "failures": [failure],
                    "complete": False,
                    "union_stage": stage,
                    "maximum_peak_process_rss_bytes": rss_limit,
                    **memory_projection,
                },
                force=True,
                resources=resources,
            )
            raise EnglishNearDedupError(
                f"Union operational RSS gate failed: {failure}"
            )

        if memory_projection["union_projected_peak_process_rss_bytes"] > rss_limit:
            fail_rss(
                "projected_union_peak_rss_exceeds_limit",
                "pre-allocation",
                initial_resources,
            )

        flatten_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='union_flatten_cursor'"
        ).fetchone()
        document_row = self.db.execute(
            "SELECT value FROM metadata WHERE key='union_document_cursor'"
        ).fetchone()
        current_parent_documents = int(
            self.db.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
        )
        # A parent array is needed while consuming edges or flattening parents,
        # but not when resuming the final SQL-only document-root assignment.
        needs_parent_array = cursor < maximum_edge_id or (
            flatten_row is None and current_parent_documents > 0
        )
        if flatten_row is not None:
            flatten_probe = json.loads(flatten_row[0])
            needs_parent_array = needs_parent_array or (
                int(flatten_probe["processed_documents"])
                < int(flatten_probe["parent_documents_total"])
            )
        parent_array: array.array[int] | None = None
        if needs_parent_array:
            # The SQLite table remains authoritative.  This is one compact
            # 8-byte value per ordinal; no all-parent Python list is created.
            parent_array = array.array("Q", range(maximum_ordinal + 1))
            for row in self.db.execute("SELECT document,parent FROM parents"):
                parent_array[int(row[0])] = int(row[1])
            allocated_resources = self._resource_snapshot()
            if allocated_resources["peak_process_rss_bytes"] > rss_limit:
                fail_rss(
                    "measured_union_peak_rss_exceeds_limit",
                    "parent-array-loaded",
                    allocated_resources,
                )

        def find(document: int) -> int:
            if parent_array is None:
                raise AssertionError("Union parent array is unavailable")
            current = document
            while parent_array[current] != current:
                current = int(parent_array[current])
            # Compress only the queried endpoint.  This keeps per-batch changed
            # state bounded even for a pathological deep component.
            parent_array[document] = current
            return current

        def union(left: int, right: int, changed: set[int]) -> bool:
            if parent_array is None:
                raise AssertionError("Union parent array is unavailable")
            left_root = find(left)
            right_root = find(right)
            changed.update((left, right, left_root, right_root))
            if left_root == right_root:
                return False
            root, child = sorted((left_root, right_root))
            parent_array[child] = root
            changed.add(child)
            return True

        edge_started = time.monotonic()
        processed_edges = 0
        committed_edge_rows = int(
            self.db.execute(
                "SELECT COUNT(*) FROM edges WHERE edge_id<=?", (cursor,)
            ).fetchone()[0]
        )
        while cursor < maximum_edge_id:
            limit = self.batch_size
            if max_new_edges is not None:
                remaining = max_new_edges - processed_edges
                if remaining <= 0:
                    publish_progress(
                        "edges",
                        force=True,
                        stage_started_at=edge_started,
                        stage_committed=committed_edge_rows,
                        stage_total=total_edges,
                        stage_invocation_units=processed_edges,
                    )
                    self._sync_checkpoint()
                    return False
                limit = min(limit, remaining)
            rows = list(
                self.db.execute(
                    """
                    SELECT edge_id,left_document,right_document FROM edges
                    WHERE edge_id>? ORDER BY edge_id LIMIT ?
                    """,
                    (cursor, limit),
                )
            )
            if not rows:
                raise EnglishNearDedupError(
                    "Union edge keyset ended before its frozen maximum edge id"
                )
            changed: set[int] = set()
            try:
                self.db.execute("BEGIN IMMEDIATE")
                for row in rows:
                    if union(
                        int(row["left_document"]),
                        int(row["right_document"]),
                        changed,
                    ):
                        merges += 1
                    cursor = int(row["edge_id"])
                self.db.executemany(
                    """
                    INSERT INTO parents(document,parent) VALUES (?,?)
                    ON CONFLICT(document) DO UPDATE SET parent=excluded.parent
                    """,
                    (
                        (document, int(parent_array[document]))
                        for document in sorted(changed)
                    ),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES ('union_edge_cursor',?)",
                    (json.dumps(cursor),),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES ('union_merges',?)",
                    (json.dumps(merges),),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            processed_edges += len(rows)
            committed_edge_rows += len(rows)
            resources = self._resource_snapshot()
            if resources["peak_process_rss_bytes"] > rss_limit:
                fail_rss(
                    "measured_union_peak_rss_exceeds_limit", "edges", resources
                )
            publish_progress(
                "edges",
                force=False,
                stage_started_at=edge_started,
                stage_committed=committed_edge_rows,
                stage_total=total_edges,
                stage_invocation_units=processed_edges,
                resources=resources,
            )
        if cursor != maximum_edge_id:
            raise EnglishNearDedupError(
                f"Union edge coverage mismatch: cursor={cursor}, max_edge_id={maximum_edge_id}"
            )

        parent_stats = self.db.execute(
            "SELECT COUNT(*),COALESCE(MAX(document),0) FROM parents"
        ).fetchone()
        parent_documents_total = int(parent_stats[0])
        maximum_parent_document = int(parent_stats[1])
        if flatten_row is None:
            flatten_cursor = 0
            flattened_documents = 0
        else:
            flatten_payload = json.loads(flatten_row[0])
            flatten_cursor = int(flatten_payload["document"])
            flattened_documents = int(flatten_payload["processed_documents"])
            if int(flatten_payload["parent_documents_total"]) != parent_documents_total:
                raise EnglishNearDedupError(
                    "Union parent inventory changed after flattening began"
                )
        finalization_units = 0
        flatten_invocation_units = 0
        flatten_started = time.monotonic()
        while flattened_documents < parent_documents_total:
            limit = self.batch_size
            if max_new_finalization_documents is not None:
                remaining = max_new_finalization_documents - finalization_units
                if remaining <= 0:
                    publish_progress(
                        "flatten-parents",
                        force=True,
                        stage_started_at=flatten_started,
                        stage_committed=flattened_documents,
                        stage_total=parent_documents_total,
                        stage_invocation_units=flatten_invocation_units,
                    )
                    self._sync_checkpoint()
                    return False
                limit = min(limit, remaining)
            rows = list(
                self.db.execute(
                    UNION_PARENT_BATCH_SQL,
                    (flatten_cursor, limit),
                )
            )
            if not rows:
                raise EnglishNearDedupError(
                    "Union parent keyset ended before its frozen row count"
                )
            updates: list[tuple[int, int]] = []
            for row in rows:
                document = int(row[0])
                updates.append((find(document), document))
                flatten_cursor = document
            flattened_documents += len(updates)
            payload = {
                "document": flatten_cursor,
                "processed_documents": flattened_documents,
                "parent_documents_total": parent_documents_total,
            }
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(
                    "UPDATE parents SET parent=? WHERE document=?", updates
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES ('union_flatten_cursor',?)",
                    (json.dumps(payload, sort_keys=True),),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            finalization_units += len(updates)
            flatten_invocation_units += len(updates)
            resources = self._resource_snapshot()
            if resources["peak_process_rss_bytes"] > rss_limit:
                fail_rss(
                    "measured_union_peak_rss_exceeds_limit",
                    "flatten-parents",
                    resources,
                )
            publish_progress(
                "flatten-parents",
                force=False,
                stage_started_at=flatten_started,
                stage_committed=flattened_documents,
                stage_total=parent_documents_total,
                stage_invocation_units=flatten_invocation_units,
                resources=resources,
            )
        if parent_documents_total == 0 and flatten_row is None:
            with self.db:
                self.db.execute(
                    "INSERT INTO metadata(key,value) VALUES ('union_flatten_cursor',?)",
                    (
                        json.dumps(
                            {
                                "document": 0,
                                "processed_documents": 0,
                                "parent_documents_total": 0,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
        if flatten_cursor != maximum_parent_document:
            raise EnglishNearDedupError(
                "Union parent flatten cursor does not cover the parent inventory"
            )
        unflattened = self.db.execute(
            """
            SELECT 1 FROM parents AS child
            JOIN parents AS parent ON parent.document=child.parent
            WHERE parent.parent!=parent.document LIMIT 1
            """
        ).fetchone()
        if unflattened is not None:
            raise EnglishNearDedupError("Union parent flattening is incomplete")

        if document_row is None:
            document_cursor = 0
            rooted_documents = 0
        else:
            document_payload = json.loads(document_row[0])
            document_cursor = int(document_payload["ordinal"])
            rooted_documents = int(document_payload["processed_documents"])
            if int(document_payload["documents_total"]) != document_count:
                raise EnglishNearDedupError(
                    "Union document inventory changed after root assignment began"
                )
        document_invocation_units = 0
        document_started = time.monotonic()
        while rooted_documents < document_count:
            limit = self.batch_size
            if max_new_finalization_documents is not None:
                remaining = max_new_finalization_documents - finalization_units
                if remaining <= 0:
                    publish_progress(
                        "assign-document-roots",
                        force=True,
                        stage_started_at=document_started,
                        stage_committed=rooted_documents,
                        stage_total=document_count,
                        stage_invocation_units=document_invocation_units,
                    )
                    self._sync_checkpoint()
                    return False
                limit = min(limit, remaining)
            rows = list(
                self.db.execute(
                    UNION_DOCUMENT_BATCH_SQL,
                    (document_cursor, limit),
                )
            )
            if not rows:
                raise EnglishNearDedupError(
                    "Union document keyset ended before its frozen row count"
                )
            lower = document_cursor
            document_cursor = int(rows[-1][0])
            rooted_documents += len(rows)
            payload = {
                "ordinal": document_cursor,
                "processed_documents": rooted_documents,
                "documents_total": document_count,
            }
            try:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute(
                    """
                    UPDATE documents SET cluster_root=COALESCE(
                        (SELECT parent FROM parents
                         WHERE parents.document=documents.ordinal),
                        ordinal
                    )
                    WHERE ordinal>? AND ordinal<=?
                    """,
                    (lower, document_cursor),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES ('union_document_cursor',?)",
                    (json.dumps(payload, sort_keys=True),),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            finalization_units += len(rows)
            document_invocation_units += len(rows)
            resources = self._resource_snapshot()
            if resources["peak_process_rss_bytes"] > rss_limit:
                fail_rss(
                    "measured_union_peak_rss_exceeds_limit",
                    "assign-document-roots",
                    resources,
                )
            publish_progress(
                "assign-document-roots",
                force=False,
                stage_started_at=document_started,
                stage_committed=rooted_documents,
                stage_total=document_count,
                stage_invocation_units=document_invocation_units,
                resources=resources,
            )
        if document_cursor != maximum_ordinal:
            raise EnglishNearDedupError(
                "Union document cursor does not cover the frozen inventory"
            )
        missing_roots = int(
            self.db.execute(
                "SELECT COUNT(*) FROM documents WHERE cluster_root IS NULL"
            ).fetchone()[0]
        )
        if missing_roots:
            raise EnglishNearDedupError(f"Missing {missing_roots} final cluster roots")
        final_resources = self._resource_snapshot()
        if final_resources["peak_process_rss_bytes"] > rss_limit:
            fail_rss(
                "measured_union_peak_rss_exceeds_limit",
                "complete",
                final_resources,
            )
        publish_progress(
            "complete",
            force=True,
            stage_started_at=document_started,
            stage_committed=document_count,
            stage_total=document_count,
            stage_invocation_units=document_invocation_units,
            complete=True,
            resources=final_resources,
        )
        self._advance(
            "union",
            "emit",
            {
                "edges": total_edges,
                "successful_component_unions": merges,
                "parent_documents": parent_documents_total,
                "rooted_documents": rooted_documents,
                "operational_rss_gate": {
                    "status": "pass",
                    "failures": [],
                    "maximum_peak_process_rss_bytes": rss_limit,
                    "observed_peak_process_rss_bytes": final_resources[
                        "peak_process_rss_bytes"
                    ],
                    **memory_projection,
                },
            },
        )
        return True

    def _audit_clusters(self) -> dict[str, Any]:
        documents = int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        clusters = int(
            self.db.execute("SELECT COUNT(DISTINCT cluster_root) FROM documents").fetchone()[0]
        )
        singleton_clusters = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT cluster_root FROM documents GROUP BY cluster_root HAVING COUNT(*)=1
                )
                """
            ).fetchone()[0]
        )
        cross_source = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT cluster_root FROM documents GROUP BY cluster_root
                    HAVING COUNT(DISTINCT bucket)>1
                )
                """
            ).fetchone()[0]
        )
        normalized_leakage = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT normalized_hash FROM documents GROUP BY normalized_hash
                    HAVING COUNT(DISTINCT cluster_root)>1
                )
                """
            ).fetchone()[0]
        )
        invalid_roots = int(
            self.db.execute(
                """
                SELECT COUNT(*) FROM documents AS d
                LEFT JOIN documents AS root ON root.ordinal=d.cluster_root
                WHERE root.ordinal IS NULL
                """
            ).fetchone()[0]
        )
        if normalized_leakage or invalid_roots:
            raise EnglishNearDedupError(
                f"Cluster audit failed: normalized_leakage={normalized_leakage}, "
                f"invalid_roots={invalid_roots}"
            )
        return {
            "english_documents_inventory": documents,
            "english_documents_mapped": documents,
            "mapping_missing_documents": 0,
            "mapping_unknown_documents": 0,
            "mapping_duplicate_documents": 0,
            "clusters": clusters,
            "singleton_clusters": singleton_clusters,
            "cross_source_clusters": cross_source,
            "normalized_hashes_in_multiple_clusters": normalized_leakage,
            "invalid_cluster_roots": invalid_roots,
        }

    def _verify_mapping_file(self, path: Path) -> int:
        """Re-read the published bytes and compare every row to authoritative state."""
        expected = self.db.execute(
            """
            SELECT d.doc_id,root.doc_id AS root_doc_id
            FROM documents AS d JOIN documents AS root ON root.ordinal=d.cluster_root
            ORDER BY d.ordinal
            """
        )
        count = 0
        iterator = iter_jsonl_zst(path)
        for expected_row in expected:
            try:
                actual = next(iterator)
            except StopIteration as exc:
                raise EnglishNearDedupError(
                    f"Published mapping ended early after {count} rows"
                ) from exc
            wanted = {
                "doc_id": bytes(expected_row["doc_id"]).hex(),
                "cluster_id": f"english-near-v1-{bytes(expected_row['root_doc_id']).hex()}",
            }
            if actual != wanted:
                raise EnglishNearDedupError(
                    f"Published mapping row {count} does not match authoritative state"
                )
            count += 1
        try:
            next(iterator)
        except StopIteration:
            return count
        raise EnglishNearDedupError("Published mapping has rows outside the English inventory")

    def emit(self) -> dict[str, Any]:
        self._assert_calibration_evidence_unchanged()
        if self.refinement_preflight_evidence is None:
            self.refinement_preflight_evidence = self._load_refinement_preflight(
                verify_cache_files=True
            )
        elif self._load_refinement_preflight(
            verify_cache_files=True
        ) != self.refinement_preflight_evidence:
            raise EnglishNearDedupError(
                "Refinement operational preflight changed before publication"
            )
        if self._phase() == "complete":
            manifest_path = self.output / "manifest.json"
            checksum_path = self.output / "manifest.sha256"
            if any(
                path.is_symlink() or not path.is_file()
                for path in (manifest_path, checksum_path)
            ):
                raise EnglishNearDedupError(
                    "Published near-dedup manifest/checksum is missing or unsafe"
                )
            manifest_raw = manifest_path.read_bytes()
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            if checksum_path.read_bytes() != (
                f"{manifest_sha}  manifest.json\n".encode("ascii")
            ):
                raise EnglishNearDedupError(
                    "Published near-dedup manifest checksum changed"
                )
            manifest = self._json_without_duplicate_keys(
                manifest_raw, "published near-dedup manifest"
            )
            if (
                manifest.get("production_ready") is not True
                or manifest.get("identity") != self.identity
                or manifest.get("refinement_operational_preflight")
                != self.refinement_preflight_evidence
            ):
                raise EnglishNearDedupError(
                    "Published near-dedup manifest authority changed"
                )
            mapping = manifest.get("mapping")
            if not isinstance(mapping, dict) or mapping.get("path") != str(
                self.config["output"]["mapping_filename"]
            ):
                raise EnglishNearDedupError(
                    "Published near-dedup mapping identity changed"
                )
            mapping_path = self.output / str(mapping["path"])
            if mapping_path.is_symlink() or not mapping_path.is_file():
                raise EnglishNearDedupError(
                    "Published near-dedup mapping is missing or unsafe"
                )
            if (
                mapping_path.stat().st_size != mapping.get("bytes")
                or file_sha256(mapping_path) != mapping.get("sha256")
            ):
                raise EnglishNearDedupError("Published mapping checksum changed")
            return manifest
        if self._phase() != "emit":
            raise EnglishNearDedupError(f"Cannot emit mapping in phase {self._phase()}")
        audit = self._audit_clusters()
        mapping_path = self.output / str(self.config["output"]["mapping_filename"])
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{mapping_path.name}.", dir=mapping_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        records = 0
        try:
            with temporary.open("wb") as raw:
                compressor = zstandard.ZstdCompressor(
                    level=3, threads=1, write_checksum=True
                ).stream_writer(raw, closefd=False)
                try:
                    rows = self.db.execute(
                        """
                        SELECT d.doc_id,root.doc_id AS root_doc_id
                        FROM documents AS d JOIN documents AS root ON root.ordinal=d.cluster_root
                        ORDER BY d.ordinal
                        """
                    )
                    for row in rows:
                        payload = canonical_json_bytes(
                            {
                                "doc_id": bytes(row["doc_id"]).hex(),
                                "cluster_id": f"english-near-v1-{bytes(row['root_doc_id']).hex()}",
                            }
                        ) + b"\n"
                        compressor.write(payload)
                        records += 1
                    compressor.flush(zstandard.FLUSH_FRAME)
                    compressor.close()
                finally:
                    try:
                        compressor.close()
                    except Exception:
                        pass
                raw.flush()
                os.fsync(raw.fileno())
            if records != audit["english_documents_inventory"]:
                raise EnglishNearDedupError(
                    f"Mapping emission coverage mismatch: {records}/{audit['english_documents_inventory']}"
                )
            checksum = file_sha256(temporary)
            os.replace(temporary, mapping_path)
            fsync_directory(mapping_path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        verified_records = self._verify_mapping_file(mapping_path)
        if verified_records != records:
            raise EnglishNearDedupError(
                f"Post-publication mapping audit mismatch: {verified_records}/{records}"
            )

        candidate_stats = self.db.execute(
            """
            SELECT COUNT(*),COALESCE(MAX(posting_documents),0),
                   COALESCE(SUM(raw_pairs),0),COALESCE(SUM(length_pruned_pairs),0)
            FROM candidate_blocks
            """
        ).fetchone()
        refinement_stats = self.db.execute(
            """
            SELECT COUNT(*),COALESCE(SUM(accepted),0) FROM refined
            """
        ).fetchone()
        rows_per_band = int(
            self.config["candidate_signature"]["tables"][0]["rows_per_band"]
        )
        bands = int(self.config["_total_bands"])
        threshold = (
            int(self.config["refinement"]["minimum_jaccard_numerator"])
            / int(self.config["refinement"]["minimum_jaccard_denominator"])
        )
        idealized_recall = 1.0 - (1.0 - threshold**rows_per_band) ** bands
        input_reports = [
            {
                "report_path": item.relative_report,
                "report_sha256": item.report_sha256,
                "archive": item.report["archive"],
                "archive_sha256": item.report["archive_sha256"],
                "fingerprint_file": item.report["fingerprint_file"],
                "fingerprint_sha256": item.report["fingerprint_sha256"],
                "documents": item.report["documents"],
            }
            for item in self.reports
        ]
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "mapping_record_version": MAPPING_RECORD_VERSION,
            "production_ready": True,
            "identity": self.identity,
            "refinement_operational_preflight": self.refinement_preflight_evidence,
            "algorithm": {
                "config": str(self.config_path),
                "config_file_sha256": self.config_file_sha,
                "config_sha256": self.config_sha,
                "name": self.config["algorithm"],
                "compact_preprocess_sketch_role": "validated-integrity-only-not-candidate-authority",
                "raw_text_candidate_pass": True,
                "full_shingle_refinement": True,
                "jaccard_threshold": threshold,
                "ideal_independent_minhash_candidate_recall_at_threshold": round(
                    idealized_recall, 9
                ),
                "statistical_limitation": (
                    "Candidate recall is probabilistic. The reported idealized value assumes "
                    "independent MinHash rows; densified one-permutation rows are correlated, so "
                    "the value is not a formal lower bound. Full-shingle refinement eliminates "
                    "LSH false-positive merges but cannot recover a candidate the LSH misses."
                ),
                "hash_limitation": (
                    "Refinement compares complete sets of 64-bit xxh3 shingle hashes. A hash "
                    "collision is possible, although unlikely; it is not described as exact "
                    "string-set Jaccard."
                ),
                "posting_overflow_action": "fail_closed_without_truncation",
            },
            "inputs": {
                "report_inventory_sha256": self.report_inventory_sha,
                "reports": input_reports,
            },
            "candidate_stats": {
                "blocks": int(candidate_stats[0]),
                "maximum_posting_documents": int(candidate_stats[1]),
                "raw_posting_pairs": int(candidate_stats[2]),
                "length_pruned_pairs": int(candidate_stats[3]),
                "unique_candidate_pairs": int(refinement_stats[0]),
                "accepted_near_pairs": int(refinement_stats[1]),
            },
            "completeness_and_leakage_audit": audit,
            "database_integrity_check": str(
                self.db.execute("PRAGMA integrity_check").fetchone()[0]
            ),
            "mapping": {
                "path": mapping_path.name,
                "sha256": checksum,
                "bytes": mapping_path.stat().st_size,
                "records": records,
                "ordered_by": "frozen_inventory_ordinal",
                "singleton_clusters_included": True,
            },
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        atomic_bytes(self.output / "manifest.json", manifest_payload)
        atomic_bytes(
            self.output / "manifest.sha256",
            hashlib.sha256(manifest_payload).hexdigest().encode("ascii") + b"  manifest.json\n",
        )
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("UPDATE metadata SET value=? WHERE key='phase'", (json.dumps("complete"),))
            self._event(
                "complete", {"documents": records, "clusters": audit["clusters"], "sha256": checksum}
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self._sync_checkpoint()
        return manifest

    def run(
        self,
        *,
        max_new_inventory_archives: int | None = None,
        max_new_signature_archives: int | None = None,
        max_new_candidate_blocks: int | None = None,
        max_new_cache_archives: int | None = None,
        max_new_refinement_pairs: int | None = None,
        max_new_union_edges: int | None = None,
        max_new_union_finalization_documents: int | None = None,
        stop_after_phase: str | None = None,
    ) -> dict[str, Any]:
        self._assert_calibration_evidence_unchanged()
        phases: list[tuple[str, Callable[[], bool]]] = [
            ("inventory", lambda: self.ingest_inventory(max_new_inventory_archives)),
            ("signatures", lambda: self.build_signatures(max_new_signature_archives)),
            ("candidates", lambda: self.generate_candidates(max_new_candidate_blocks)),
            ("cache", lambda: self.build_refinement_cache(max_new_cache_archives)),
            ("refine", lambda: self.refine_candidates(max_new_refinement_pairs)),
            (
                "union",
                lambda: self.union_edges(
                    max_new_union_edges,
                    max_new_union_finalization_documents,
                ),
            ),
        ]
        for name, action in phases:
            if self._phase() == "complete":
                break
            complete = action()
            if not complete:
                return {
                    "complete": False,
                    "phase": self._phase(),
                    "operational": self.operational_metrics,
                }
            if stop_after_phase == name:
                return {
                    "complete": False,
                    "phase": self._phase(),
                    "operational": self.operational_metrics,
                }
        manifest = self.emit()
        return {
            "complete": True,
            "phase": "complete",
            "manifest": manifest,
            "operational": self.operational_metrics,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_QUOTA_CONFIG)
    parser.add_argument(
        "--calibration-result",
        type=Path,
        required=True,
        help=(
            "passed production calibration JSON; the matching .sha256 sidecar "
            "is mandatory and both are pinned into the run identity"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=60,
        help="atomic operational telemetry/log interval (1..3600 seconds)",
    )
    parser.add_argument(
        "--sqlite-journal-mode",
        choices=("auto", "wal", "delete"),
        help=(
            "Audited override. WAL is rejected unless the detected filesystem "
            "is in the pinned local allowlist."
        ),
    )
    parser.add_argument("--max-new-inventory-archives", type=int)
    parser.add_argument("--max-new-signature-archives", type=int)
    parser.add_argument("--max-new-candidate-blocks", type=int)
    parser.add_argument("--max-new-cache-archives", type=int)
    parser.add_argument("--max-new-refinement-pairs", type=int)
    parser.add_argument("--max-new-union-edges", type=int)
    parser.add_argument("--max-new-union-finalization-documents", type=int)
    parser.add_argument(
        "--stop-after-phase",
        choices=("inventory", "signatures", "candidates", "cache", "refine", "union"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    staging = args.staging_root or args.root / "staging" / "preprocess"
    output = args.output or args.root / "staging" / "english-near"
    try:
        with EnglishNearDedupBuilder(
            root=args.root,
            staging_root=staging,
            output=output,
            config_path=args.config,
            policy_path=args.policy,
            denylist_path=args.denylist,
            quota_config_path=args.quota_config,
            calibration_result_path=args.calibration_result,
            batch_size=args.batch_size,
            progress_interval_seconds=args.progress_interval_seconds,
            sqlite_journal_mode=args.sqlite_journal_mode,
        ) as builder:
            result = builder.run(
                max_new_inventory_archives=args.max_new_inventory_archives,
                max_new_signature_archives=args.max_new_signature_archives,
                max_new_candidate_blocks=args.max_new_candidate_blocks,
                max_new_cache_archives=args.max_new_cache_archives,
                max_new_refinement_pairs=args.max_new_refinement_pairs,
                max_new_union_edges=args.max_new_union_edges,
                max_new_union_finalization_documents=(
                    args.max_new_union_finalization_documents
                ),
                stop_after_phase=args.stop_after_phase,
            )
    except (EnglishNearDedupError, json.JSONDecodeError, sqlite3.Error, OSError) as exc:
        print(f"English near-dedup failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
