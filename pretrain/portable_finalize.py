"""Finalize an authenticated packed-only corpus after a portable handoff.

The normal :class:`pretrain.materialize.CorpusMaterializer` deliberately binds
raw archives, preprocess reports, decisions, and token-cache payloads before it
writes a single packed byte.  Those cold-path inputs need not travel to a GPU
pod once packing has completed.  This module consumes the *packed* durable
journal plus the compact selection, cache-inventory, tokenizer, and document-
index authorities and publishes order-v4/final corpus metadata without
re-tokenizing, re-packing, or pretending to revalidate absent raw payloads.

Two phases are intentional:

``heldout``
    Authenticate the handoff and publish only validation/test orders.  Their
    targets are the largest exact 40/40/20 whole-row budgets at or below the
    configured caps.  No training geometry is guessed.

``final``
    Require the measured global-microbatch and accumulation geometry, publish
    the update-aligned train order, provenance, and the top-level manifest,
    then remove the packed journal last.

The original journal is never rewritten.  Every visible artifact is either a
complete atomic rename or a pre-existing artifact that is re-authenticated.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pretrain import data as training_data
from pretrain.materialize import (
    BUCKET_DOMAIN,
    CURSOR_FORMAT,
    CURSOR_VERSION,
    DOCUMENT_INDEX_FORMAT,
    DOCUMENT_INDEX_VERSION,
    FORMAT,
    FORMAT_VERSION,
    JOURNAL_NAME,
    SPLITS,
    atomic_bytes,
    atomic_json,
    canonical_sha256,
    file_sha256,
    iter_jsonl_zst,
)
from pretrain.raw_token_cache_inventory import (
    INVENTORY_FORMAT,
    INVENTORY_FORMAT_VERSION,
    INVENTORY_MANIFEST_FILE,
    INVENTORY_SIDECAR_FILE,
    NON_AUTHORITIES,
)
from pretrain.selection_contract import (
    ALL_ELIGIBLE_BITMAP_FORMAT,
    ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
    ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION,
    ALL_ELIGIBLE_SELECTION_STRATEGY,
)
from pretrain.tokenizer_identity import verify_tokenizer_identity


PORTABLE_FINALIZATION_FORMAT = "portable-packed-corpus-finalization"
PORTABLE_FINALIZATION_VERSION = 1
RESTORE_READY_FORMAT = "transcendent-logic-pretraining-data-restore"
RESTORE_READY_VERSION = 1
RESTORE_READY_NAME = ".RESTORE_READY.json"
RESTORE_SOURCE_URI = (
    "s3://transcendent-logic-data-618079239540/"
    "coding-llm/pretraining/2026-09-02-packed-v1/"
)
RESTORE_BUCKET_OWNER = "618079239540"
RESTORE_REGION = "eu-central-1"
RESTORE_SUMMARY_RELATIVE = "audit/source-summary.json"
RESTORE_PACKED_TSV_RELATIVE = "audit/packed-v1-files.tsv"
RESTORE_MANIFEST_INVENTORY_RELATIVE = "audit/packed-manifests.sha256"
EXPECTED_WEIGHTS = {"python": 0.4, "other_code": 0.4, "english": 0.2}
HEX = frozenset("0123456789abcdef")


class PortableFinalizationError(RuntimeError):
    """Raised when a packed-only handoff cannot be finalized safely."""


@dataclass(frozen=True)
class PortableFinalizationConfig:
    corpus_root: Path
    selection_root: Path
    tokenizer_root: Path
    cache_inventory_root: Path
    policy_path: Path
    quota_path: Path
    benchmark_denylist_path: Path
    restore_ready_path: Path | None = None
    order_seed: int = 1_234
    maximum_train_input_tokens: int = 52_580_000_000
    maximum_validation_input_tokens: int = 500_000_000
    maximum_test_input_tokens: int = 500_000_000
    expected_optimizer_batch_rows: int = 192
    world_size: int = 6
    verify_packed_payloads: bool = False

    def validate(self) -> None:
        for field, value, minimum in (
            ("order_seed", self.order_seed, 0),
            ("maximum_train_input_tokens", self.maximum_train_input_tokens, 1),
            (
                "maximum_validation_input_tokens",
                self.maximum_validation_input_tokens,
                1,
            ),
            ("maximum_test_input_tokens", self.maximum_test_input_tokens, 1),
            (
                "expected_optimizer_batch_rows",
                self.expected_optimizer_batch_rows,
                1,
            ),
            ("world_size", self.world_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise PortableFinalizationError(
                    f"{field} must be an integer >= {minimum}"
                )
        if not isinstance(self.verify_packed_payloads, bool):
            raise PortableFinalizationError("verify_packed_payloads must be boolean")
        if self.restore_ready_path is None and not self.verify_packed_payloads:
            raise PortableFinalizationError(
                "Either an authenticated --restore-ready receipt or full packed-"
                "payload verification is required"
            )


@dataclass(frozen=True)
class _ArchiveAuthority:
    ordinal: int
    archive: str
    index: int
    bucket: str
    raw_bytes: int
    raw_sha256: str
    report_path: str
    report_bytes: int
    report_sha256: str
    fingerprint_path: str
    fingerprint_bytes: int
    fingerprint_sha256: str
    decision_path: str
    decision_bytes: int
    decision_sha256: str
    documents: int
    clean_bytes: int
    content_tokens: int
    kept_documents: int

    @property
    def domain(self) -> str:
        return BUCKET_DOMAIN[self.bucket]

    def materialization_identity(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "archive": self.archive,
            "archive_index": self.index,
            "bucket": self.bucket,
            "domain": self.domain,
            "raw_sha256": self.raw_sha256,
            "report": self.report_path,
            "report_sha256": self.report_sha256,
            "fingerprint": self.fingerprint_path,
            "fingerprint_sha256": self.fingerprint_sha256,
            "decision": self.decision_path,
            "decision_sha256": self.decision_sha256,
            "documents": self.documents,
            "clean_bytes": self.clean_bytes,
            "content_tokens": self.content_tokens,
            "decision_format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "decision_format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "decision_bytes": self.decision_bytes,
            "decision_kept_documents": self.kept_documents,
        }

    def source_provenance(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "sha256": self.raw_sha256,
            "documents": self.documents,
            "content_tokens": self.content_tokens,
        }

    def fingerprint_provenance(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "report": self.report_path,
            "report_sha256": self.report_sha256,
            "fingerprint": self.fingerprint_path,
            "fingerprint_sha256": self.fingerprint_sha256,
            "decision": self.decision_path,
            "decision_sha256": self.decision_sha256,
            "decision_format": ALL_ELIGIBLE_BITMAP_FORMAT,
            "decision_format_version": ALL_ELIGIBLE_BITMAP_FORMAT_VERSION,
            "decision_bytes": self.decision_bytes,
            "decision_kept_documents": self.kept_documents,
        }


def _plain_int(value: Any, *, minimum: int = 0, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PortableFinalizationError(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise PortableFinalizationError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PortableFinalizationError(f"Unsafe {field}: {value!r}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise PortableFinalizationError(f"Unsafe {field}: {value!r}")
    return value


def _regular_file(path: Path, *, field: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PortableFinalizationError(f"Missing {field}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PortableFinalizationError(
            f"{field} must be a regular non-symlink file: {path}"
        )
    return path


def _safe_file(root: Path, relative: Any, *, field: str) -> Path:
    name = _safe_relative(relative, field=field)
    root = root.resolve(strict=True)
    path = root.joinpath(*PurePosixPath(name).parts)
    _regular_file(path, field=field)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PortableFinalizationError(f"{field} escapes {root}: {name!r}") from exc
    if resolved != path.absolute():
        raise PortableFinalizationError(f"{field} traverses a symlink: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    _regular_file(path, field="JSON artifact")

    def duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PortableFinalizationError(
                    f"Duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PortableFinalizationError(f"Non-finite JSON number {value!r} in {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=duplicate_guard,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableFinalizationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PortableFinalizationError(f"Expected a JSON object in {path}")
    return value


def _canonical_inventory_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortableFinalizationError("Inventory is not canonical JSON") from exc


def _verify_sidecar(root: Path, *, manifest_name: str, sidecar_name: str) -> tuple[dict[str, Any], str]:
    manifest_path = _regular_file(root / manifest_name, field="manifest")
    sidecar_path = _regular_file(root / sidecar_name, field="manifest sidecar")
    digest = file_sha256(manifest_path)
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise PortableFinalizationError(f"Cannot read {sidecar_path}: {exc}") from exc
    if sidecar != f"{digest}  {manifest_name}\n":
        raise PortableFinalizationError(f"Manifest sidecar mismatch: {sidecar_path}")
    return _load_json(manifest_path), digest


def _parse_sha256_inventory(path: Path) -> dict[str, str]:
    """Parse one strict, newline-terminated ``sha256sum`` inventory."""

    raw = _regular_file(path, field="SHA-256 inventory").read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise PortableFinalizationError(f"Invalid SHA-256 inventory framing: {path}")
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw[:-1].split(b"\n"), 1):
        if len(raw_line) < 67 or raw_line[64:66] != b"  ":
            raise PortableFinalizationError(
                f"Invalid SHA-256 inventory line {line_number}: {path}"
            )
        try:
            digest = raw_line[:64].decode("ascii")
            relative = raw_line[66:].decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise PortableFinalizationError(
                f"Invalid SHA-256 inventory encoding at line {line_number}: {path}"
            ) from exc
        _sha256(digest, field=f"SHA-256 inventory line {line_number}")
        relative = _safe_relative(
            relative, field=f"SHA-256 inventory path at line {line_number}"
        )
        if relative in result:
            raise PortableFinalizationError(
                f"Duplicate SHA-256 inventory path {relative!r}: {path}"
            )
        result[relative] = digest
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _largest_feasible_rows(
    available: Mapping[str, int],
    *,
    maximum_rows: int,
    row_quantum: int,
) -> int:
    """Return the largest strict-weight row count fitting every domain."""

    if maximum_rows < 1 or row_quantum < 1:
        raise PortableFinalizationError("Order row caps and quanta must be positive")
    maximum_units = maximum_rows // row_quantum
    low = 0
    high = maximum_units
    while low < high:
        middle = (low + high + 1) // 2
        total = middle * row_quantum
        counts = training_data._strict_weighted_row_counts(  # type: ignore[attr-defined]
            total, EXPECTED_WEIGHTS
        )
        if all(counts[domain] <= available[domain] for domain in training_data.DOMAIN_ORDER):
            low = middle
        else:
            high = middle - 1
    rows = low * row_quantum
    if rows < 1:
        raise PortableFinalizationError(
            "Packed supply cannot form one strict 40/40/20 order quantum"
        )
    return rows


def _aligned_train_rows(
    *,
    maximum_input_tokens: int,
    sequence_length: int,
    optimizer_update_rows: int,
) -> int:
    """Return the largest complete-update row count under an input-token cap."""

    for field, value in (
        ("maximum_input_tokens", maximum_input_tokens),
        ("sequence_length", sequence_length),
        ("optimizer_update_rows", optimizer_update_rows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PortableFinalizationError(f"{field} must be a positive integer")
    maximum_rows = maximum_input_tokens // sequence_length
    rows = (maximum_rows // optimizer_update_rows) * optimizer_update_rows
    if rows < optimizer_update_rows:
        raise PortableFinalizationError(
            "Train cap is smaller than one complete optimizer update"
        )
    return rows


class PortablePackedFinalizer:
    """Authenticate and finalize a packed-only portable corpus."""

    def __init__(self, config: PortableFinalizationConfig) -> None:
        config.validate()
        self.config = config
        self.corpus_root = config.corpus_root.absolute()
        self.selection_root = config.selection_root.absolute()
        self.tokenizer_root = config.tokenizer_root.absolute()
        self.cache_inventory_root = config.cache_inventory_root.absolute()
        for root, field in (
            (self.corpus_root, "corpus root"),
            (self.selection_root, "selection root"),
            (self.tokenizer_root, "tokenizer root"),
            (self.cache_inventory_root, "cache inventory root"),
        ):
            if root.is_symlink() or not root.is_dir():
                raise PortableFinalizationError(
                    f"{field} must be a non-symlink directory: {root}"
                )
        self.journal_path = self.corpus_root / JOURNAL_NAME
        self.journal: dict[str, Any] | None = None
        self.journal_sha256: str | None = None
        self.identity: dict[str, Any] = {}
        self.selection: dict[str, Any] = {}
        self.selection_sha256 = ""
        self.tokenizer_manifest: dict[str, Any] = {}
        self.cache_inventory: dict[str, Any] = {}
        self.cache_inventory_sha256 = ""
        self.cache_sidecar_sha256 = ""
        self.archives: list[_ArchiveAuthority] = []
        self.cursors: dict[tuple[str, str], dict[str, Any]] = {}
        self.packed: dict[tuple[str, str], dict[str, Any]] = {}
        self.document_indexes: dict[tuple[str, str], dict[str, Any]] = {}
        self.restore_receipt: dict[str, Any] | None = None

    def _load_journal(self) -> None:
        if not self.journal_path.exists():
            raise PortableFinalizationError(
                f"Packed handoff journal is missing: {self.journal_path}"
            )
        journal = _load_json(self.journal_path)
        if set(journal) != {"format", "format_version", "identity", "state"}:
            raise PortableFinalizationError("Packed journal top-level schema mismatch")
        if journal.get("format") != FORMAT or journal.get("format_version") != FORMAT_VERSION:
            raise PortableFinalizationError("Unsupported packed journal format/version")
        identity = journal.get("identity")
        state = journal.get("state")
        if not isinstance(identity, dict) or not isinstance(state, dict):
            raise PortableFinalizationError("Packed journal identity/state is invalid")
        if state.get("phase") != "packed":
            raise PortableFinalizationError(
                f"Portable finalization requires phase 'packed', found {state.get('phase')!r}"
            )
        archive_count = _plain_int(
            state.get("archive_count"), minimum=1, field="journal archive_count"
        )
        if state.get("completed_archives") != archive_count:
            raise PortableFinalizationError("Packed journal does not prove complete packing")
        writer_cursors = state.get("writer_cursors")
        expected_keys = {
            f"{split}/{domain}"
            for split in SPLITS
            for domain in training_data.DOMAIN_ORDER
        }
        if not isinstance(writer_cursors, dict) or set(writer_cursors) != expected_keys:
            raise PortableFinalizationError(
                "Packed journal must contain exactly nine writer cursors"
            )
        archive_inventory_sha = _sha256(
            identity.get("archive_inventory_sha256"),
            field="journal archive inventory",
        )
        for split in SPLITS:
            for domain in training_data.DOMAIN_ORDER:
                cursor = writer_cursors[f"{split}/{domain}"]
                if not isinstance(cursor, dict):
                    raise PortableFinalizationError(
                        f"Invalid writer cursor for {split}/{domain}"
                    )
                expected = {
                    "format": CURSOR_FORMAT,
                    "version": CURSOR_VERSION,
                    "archive_inventory_sha256": archive_inventory_sha,
                    "split": split,
                    "domain": domain,
                    "next_archive": archive_count,
                }
                if any(cursor.get(key) != value for key, value in expected.items()):
                    raise PortableFinalizationError(
                        f"Writer cursor identity mismatch for {split}/{domain}"
                    )
                for field in (
                    "selected_documents",
                    "selected_content_tokens",
                    "terminal_prefix_documents",
                ):
                    _plain_int(cursor.get(field), field=f"cursor {field}")
                if cursor.get("terminal_prefix_documents") != 0:
                    raise PortableFinalizationError(
                        "Selection-v7 packed handoff contains terminal document prefixes"
                    )
                if not isinstance(cursor.get("document_index_shards"), list):
                    raise PortableFinalizationError(
                        f"Writer cursor has no index inventory for {split}/{domain}"
                    )
                self.cursors[(split, domain)] = dict(cursor)
        self.journal = journal
        self.journal_sha256 = file_sha256(self.journal_path)
        self.identity = dict(identity)

    def _authenticate_restore_receipt(self) -> None:
        path = self.config.restore_ready_path
        if path is None:
            return
        receipt_path = path.absolute()
        if (
            receipt_path.name != RESTORE_READY_NAME
            or receipt_path.parent.resolve(strict=True)
            != self.corpus_root.parent.resolve(strict=True)
        ):
            raise PortableFinalizationError(
                "Restore receipt must be the sibling .RESTORE_READY.json for "
                f"{self.corpus_root}"
            )
        receipt_raw = _regular_file(
            receipt_path, field="S3 restore readiness receipt"
        ).read_bytes()
        receipt = _load_json(receipt_path)
        if receipt_raw != _canonical_inventory_bytes(receipt):
            raise PortableFinalizationError("Restore receipt is not canonical JSON")
        expected_fields = {
            "format",
            "format_version",
            "status",
            "source_uri",
            "bucket_owner",
            "region",
            "restored_at_utc",
            "remote_object_count",
            "remote_total_bytes",
            "remote_inventory_sha256",
            "selection_manifest_sha256",
            "summary_sha256",
            "packed_tsv_sha256",
            "packed_manifests_sha256",
            "packed_file_count",
            "packed_total_bytes",
            "payload_file_count",
            "payload_total_bytes",
            "payload_sha256_verified",
            "optional_global_sha256_verified",
        }
        if (
            set(receipt) != expected_fields
            or receipt.get("format") != RESTORE_READY_FORMAT
            or receipt.get("format_version") != RESTORE_READY_VERSION
            or receipt.get("status") != "ready"
            or receipt.get("source_uri") != RESTORE_SOURCE_URI
            or receipt.get("bucket_owner") != RESTORE_BUCKET_OWNER
            or receipt.get("region") != RESTORE_REGION
            or not isinstance(receipt.get("restored_at_utc"), str)
            or not receipt["restored_at_utc"].endswith("Z")
            or receipt.get("payload_sha256_verified") is not True
            or not isinstance(receipt.get("optional_global_sha256_verified"), bool)
            or receipt.get("selection_manifest_sha256")
            != self.identity.get("selection_manifest_sha256")
        ):
            raise PortableFinalizationError(
                f"Restore receipt is invalid or bound to another corpus: {path}"
            )
        for field in (
            "remote_inventory_sha256",
            "selection_manifest_sha256",
            "summary_sha256",
            "packed_tsv_sha256",
            "packed_manifests_sha256",
        ):
            _sha256(receipt.get(field), field=f"restore receipt {field}")
        for field in (
            "remote_object_count",
            "remote_total_bytes",
            "packed_file_count",
            "packed_total_bytes",
            "payload_file_count",
            "payload_total_bytes",
        ):
            _plain_int(receipt.get(field), minimum=1, field=f"restore receipt {field}")
        if (
            receipt["remote_object_count"] < receipt["packed_file_count"]
            or receipt["remote_total_bytes"] < receipt["packed_total_bytes"]
            or receipt["payload_total_bytes"] > receipt["packed_total_bytes"]
        ):
            raise PortableFinalizationError("Restore receipt byte/count totals are invalid")

        restore_root = receipt_path.parent
        summary_path = _safe_file(
            restore_root,
            RESTORE_SUMMARY_RELATIVE,
            field="restore source summary",
        )
        if file_sha256(summary_path) != receipt["summary_sha256"]:
            raise PortableFinalizationError("Restore source-summary checksum mismatch")
        summary = _load_json(summary_path)
        if (
            set(summary)
            != {
                "source_root",
                "generated_at_utc",
                "file_count",
                "total_bytes",
                "selection_manifest_sha256",
            }
            or not isinstance(summary.get("source_root"), str)
            or not summary["source_root"].endswith("/final/packed-v1")
            or not isinstance(summary.get("generated_at_utc"), str)
            or not summary["generated_at_utc"].endswith("Z")
            or summary.get("file_count") != receipt["packed_file_count"]
            or summary.get("total_bytes") != receipt["packed_total_bytes"]
            or summary.get("selection_manifest_sha256")
            != self.identity.get("selection_manifest_sha256")
        ):
            raise PortableFinalizationError("Restore source-summary contract mismatch")

        tsv_path = _safe_file(
            restore_root,
            RESTORE_PACKED_TSV_RELATIVE,
            field="restore packed-file inventory",
        )
        if file_sha256(tsv_path) != receipt["packed_tsv_sha256"]:
            raise PortableFinalizationError("Restore packed-file inventory changed")

        manifest_inventory_path = _safe_file(
            restore_root,
            RESTORE_MANIFEST_INVENTORY_RELATIVE,
            field="restore packed-manifest inventory",
        )
        if file_sha256(manifest_inventory_path) != receipt["packed_manifests_sha256"]:
            raise PortableFinalizationError("Restore packed-manifest inventory changed")
        manifest_inventory = _parse_sha256_inventory(manifest_inventory_path)
        expected_inventory_paths = {JOURNAL_NAME}
        expected_inventory_paths.update(
            f"packed/{split}/{domain}/manifest.json"
            for split in SPLITS
            for domain in training_data.DOMAIN_ORDER
        )
        if set(manifest_inventory) != expected_inventory_paths:
            raise PortableFinalizationError(
                "Restore packed-manifest inventory must contain exactly the journal "
                "and nine packed manifests"
            )
        for relative, expected_sha256 in sorted(manifest_inventory.items()):
            artifact = _safe_file(
                self.corpus_root,
                relative,
                field=f"restore-authenticated packed artifact {relative}",
            )
            if file_sha256(artifact) != expected_sha256:
                raise PortableFinalizationError(
                    f"Restore-authenticated packed artifact changed: {artifact}"
                )
        if manifest_inventory[JOURNAL_NAME] != self.journal_sha256:
            raise PortableFinalizationError(
                "Restore inventory is bound to another packed journal"
            )
        self.restore_receipt = receipt

    def _authenticate_selection(self) -> None:
        selection, digest = _verify_sidecar(
            self.selection_root,
            manifest_name="manifest.json",
            sidecar_name="manifest.sha256",
        )
        if digest != self.identity.get("selection_manifest_sha256"):
            raise PortableFinalizationError(
                "Selection manifest differs from the packed journal identity"
            )
        identity = selection.get("identity")
        if (
            not isinstance(identity, dict)
            or identity.get("format_version") != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
            or selection.get("production_ready") is not True
            or selection.get("selection_strategy")
            != ALL_ELIGIBLE_SELECTION_STRATEGY
            or selection.get("decision_format") != ALL_ELIGIBLE_BITMAP_FORMAT
            or selection.get("decision_format_version")
            != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
        ):
            raise PortableFinalizationError(
                "Packed handoff requires a production selection-v7 authority"
            )
        expected_bindings = {
            "tokenizer_manifest_sha256": "tokenizer_manifest_sha256",
            "policy_sha256": "curation_policy_sha256",
            "quota_config_sha256": "quota_config_sha256",
            "benchmark_guard_sha256": "benchmark_guard_sha256",
            "preprocess_manifest_sha256": "preprocess_manifest_sha256",
        }
        for selection_key, journal_key in expected_bindings.items():
            if identity.get(selection_key) != self.identity.get(journal_key):
                raise PortableFinalizationError(
                    f"Selection/journal binding mismatch: {selection_key}"
                )
        if selection.get("decision_inventory_sha256") != self.identity.get(
            "decision_inventory_sha256"
        ):
            raise PortableFinalizationError("Selection decision inventory changed")
        completeness = selection.get("collection_completeness")
        if (
            not isinstance(completeness, dict)
            or canonical_sha256(completeness)
            != self.identity.get("collection_completeness_sha256")
        ):
            raise PortableFinalizationError("Collection-completeness authority changed")
        decisions = selection.get("decision_shards")
        if not isinstance(decisions, list) or not decisions:
            raise PortableFinalizationError("Selection has no decision inventory")
        if canonical_sha256(decisions) != selection.get("decision_inventory_sha256"):
            raise PortableFinalizationError("Selection decision digest is inconsistent")
        for ordinal, descriptor in enumerate(decisions):
            if not isinstance(descriptor, dict):
                raise PortableFinalizationError(
                    f"Selection decision {ordinal} is not an object"
                )
            decision_path = _safe_file(
                self.selection_root,
                descriptor.get("path"),
                field=f"selection decision {ordinal}",
            )
            if (
                decision_path.stat().st_size != descriptor.get("bytes")
                or file_sha256(decision_path) != descriptor.get("sha256")
            ):
                raise PortableFinalizationError(
                    f"Selection decision changed: {decision_path}"
                )
        self.selection = selection
        self.selection_sha256 = digest

    def _authenticate_configs(self) -> None:
        policy = _load_json(self.config.policy_path.absolute())
        if canonical_sha256(policy) != self.identity.get("curation_policy_sha256"):
            raise PortableFinalizationError("Curation policy differs from packed identity")
        for path, key, label in (
            (self.config.quota_path, "quota_config_sha256", "quota configuration"),
            (
                self.config.benchmark_denylist_path,
                "benchmark_guard_sha256",
                "benchmark denylist",
            ),
        ):
            candidate = _regular_file(path.absolute(), field=label)
            if file_sha256(candidate) != self.identity.get(key):
                raise PortableFinalizationError(f"{label} differs from packed identity")

    def _authenticate_tokenizer(self) -> None:
        expected = _sha256(
            self.identity.get("tokenizer_manifest_sha256"),
            field="tokenizer manifest",
        )
        packing = self.identity.get("packing_configuration")
        if not isinstance(packing, dict):
            raise PortableFinalizationError("Journal has no packing configuration")
        try:
            tokenizer = verify_tokenizer_identity(
                self.tokenizer_root,
                expected_manifest_sha256=expected,
                expected_vocab_size=_plain_int(
                    packing.get("expected_vocab_size"),
                    minimum=1,
                    field="packed vocabulary size",
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise PortableFinalizationError(
                f"Tokenizer authentication failed: {exc}"
            ) from exc
        validation = tokenizer.manifest.get("validation")
        if (
            not isinstance(validation, dict)
            or validation.get("eos_token_id")
            != packing.get("expected_eos_token_id")
        ):
            raise PortableFinalizationError("Tokenizer EOS differs from packed identity")
        self.tokenizer_manifest = dict(tokenizer.manifest)

    def _authenticate_cache_inventory(self) -> None:
        manifest_path = self.cache_inventory_root / INVENTORY_MANIFEST_FILE
        sidecar_path = self.cache_inventory_root / INVENTORY_SIDECAR_FILE
        manifest_raw = _regular_file(
            manifest_path, field="cache inventory manifest"
        ).read_bytes()
        sidecar_raw = _regular_file(
            sidecar_path, field="cache inventory sidecar"
        ).read_bytes()
        digest = hashlib.sha256(manifest_raw).hexdigest()
        if sidecar_raw != f"{digest}  {INVENTORY_MANIFEST_FILE}\n".encode("ascii"):
            raise PortableFinalizationError("Cache inventory sidecar mismatch")
        inventory = _load_json(manifest_path)
        if _canonical_inventory_bytes(inventory) != manifest_raw:
            raise PortableFinalizationError("Cache inventory JSON is not canonical")
        if (
            inventory.get("format") != INVENTORY_FORMAT
            or inventory.get("format_version") != INVENTORY_FORMAT_VERSION
            or inventory.get("inventory_complete") is not True
            or inventory.get("training_ready") is not False
            or inventory.get("non_authorities") != NON_AUTHORITIES
        ):
            raise PortableFinalizationError("Cache inventory contract mismatch")
        selection = inventory.get("selection")
        if (
            not isinstance(selection, dict)
            or selection.get("identity_format_version")
            != ALL_ELIGIBLE_IDENTITY_FORMAT_VERSION
            or selection.get("manifest_sha256") != self.selection_sha256
        ):
            raise PortableFinalizationError(
                "Cache inventory is bound to another selection"
            )
        tokenizer = inventory.get("tokenizer")
        if (
            not isinstance(tokenizer, dict)
            or tokenizer.get("manifest_sha256")
            != self.identity.get("tokenizer_manifest_sha256")
        ):
            raise PortableFinalizationError(
                "Cache inventory is bound to another tokenizer"
            )
        entries = inventory.get("archives")
        archive_count = _plain_int(
            inventory.get("archive_count"), minimum=1, field="cache archive_count"
        )
        if not isinstance(entries, list) or len(entries) != archive_count:
            raise PortableFinalizationError("Cache archive inventory is incomplete")
        if canonical_sha256(entries) != inventory.get("archive_inventory_sha256"):
            # raw-token-cache inventories include a trailing newline in their
            # canonical digest domain, unlike materialization identities.
            inventory_digest = hashlib.sha256(_canonical_inventory_bytes(entries)).hexdigest()
            if inventory_digest != inventory.get("archive_inventory_sha256"):
                raise PortableFinalizationError("Cache archive inventory digest mismatch")
        descriptor = {
            "format": INVENTORY_FORMAT,
            "format_version": INVENTORY_FORMAT_VERSION,
            "manifest": {
                "path": INVENTORY_MANIFEST_FILE,
                "bytes": len(manifest_raw),
                "sha256": digest,
            },
            "sidecar": {
                "path": INVENTORY_SIDECAR_FILE,
                "bytes": len(sidecar_raw),
                "sha256": hashlib.sha256(sidecar_raw).hexdigest(),
            },
            "selection_manifest_sha256": self.selection_sha256,
            "archive_count": archive_count,
            "archive_inventory_sha256": inventory["archive_inventory_sha256"],
        }
        if self.identity.get("raw_token_cache") != descriptor:
            raise PortableFinalizationError(
                "Cache inventory differs from the packed journal authority"
            )
        self.cache_inventory = inventory
        self.cache_inventory_sha256 = digest
        self.cache_sidecar_sha256 = hashlib.sha256(sidecar_raw).hexdigest()

    def _build_archive_authority(self) -> None:
        reports = self.selection.get("input_reports")
        decisions = self.selection.get("decision_shards")
        entries = self.cache_inventory.get("archives")
        if not isinstance(reports, list) or not isinstance(decisions, list):
            raise PortableFinalizationError("Selection report/decision inventory missing")
        if not isinstance(entries, list) or len(entries) != len(decisions):
            raise PortableFinalizationError("Cache/selection archive counts differ")
        report_by_archive: dict[str, dict[str, Any]] = {}
        for report in reports:
            if not isinstance(report, dict) or not isinstance(report.get("archive"), str):
                raise PortableFinalizationError("Invalid selection report descriptor")
            if report["archive"] in report_by_archive:
                raise PortableFinalizationError("Duplicate selection report archive")
            report_by_archive[report["archive"]] = report
        decision_archives = [
            decision.get("archive") if isinstance(decision, dict) else None
            for decision in decisions
        ]
        if decision_archives != sorted(report_by_archive):
            raise PortableFinalizationError(
                "Selection decisions are not in canonical archive order"
            )
        archives: list[_ArchiveAuthority] = []
        for ordinal, (decision, entry) in enumerate(zip(decisions, entries, strict=True)):
            if not isinstance(decision, dict) or not isinstance(entry, dict):
                raise PortableFinalizationError("Invalid archive authority descriptor")
            if entry.get("ordinal") != ordinal:
                raise PortableFinalizationError("Cache archive ordinals are not contiguous")
            archive = entry.get("archive")
            report_authority = entry.get("preprocess_report")
            fingerprint = entry.get("fingerprint")
            documents = entry.get("documents")
            if not all(
                isinstance(item, dict)
                for item in (archive, report_authority, fingerprint, documents)
            ):
                raise PortableFinalizationError("Cache archive authority schema mismatch")
            assert isinstance(archive, dict)
            assert isinstance(report_authority, dict)
            assert isinstance(fingerprint, dict)
            assert isinstance(documents, dict)
            name = archive.get("path")
            report = report_by_archive.get(str(name))
            if report is None or decision.get("archive") != name:
                raise PortableFinalizationError(
                    "Cache and selection archive identities differ"
                )
            expected = {
                "archive_sha256": archive.get("sha256"),
                "report": report_authority.get("path"),
                "report_sha256": report_authority.get("sha256"),
                "fingerprint_file": fingerprint.get("path"),
                "fingerprint_sha256": fingerprint.get("sha256"),
                "documents": documents.get("records"),
                "content_tokens": documents.get("content_tokens"),
            }
            if any(report.get(key) != value for key, value in expected.items()):
                raise PortableFinalizationError(
                    f"Cache/report binding differs for archive {name!r}"
                )
            if decision.get("records") != documents.get("records"):
                raise PortableFinalizationError(
                    f"Decision/report records differ for archive {name!r}"
                )
            bucket = archive.get("bucket")
            if bucket not in BUCKET_DOMAIN:
                raise PortableFinalizationError(f"Unknown raw bucket {bucket!r}")
            authority = _ArchiveAuthority(
                ordinal=ordinal,
                archive=_safe_relative(name, field="raw archive path"),
                index=_plain_int(archive.get("index"), field="raw archive index"),
                bucket=str(bucket),
                raw_bytes=_plain_int(
                    archive.get("bytes"), minimum=1, field="raw archive bytes"
                ),
                raw_sha256=_sha256(archive.get("sha256"), field="raw archive"),
                report_path=_safe_relative(
                    report_authority.get("path"), field="preprocess report path"
                ),
                report_bytes=_plain_int(
                    report_authority.get("bytes"),
                    minimum=1,
                    field="preprocess report bytes",
                ),
                report_sha256=_sha256(
                    report_authority.get("sha256"), field="preprocess report"
                ),
                fingerprint_path=_safe_relative(
                    fingerprint.get("path"), field="fingerprint path"
                ),
                fingerprint_bytes=_plain_int(
                    fingerprint.get("bytes"), minimum=1, field="fingerprint bytes"
                ),
                fingerprint_sha256=_sha256(
                    fingerprint.get("sha256"), field="fingerprint"
                ),
                decision_path=_safe_relative(
                    decision.get("path"), field="decision path"
                ),
                decision_bytes=_plain_int(
                    decision.get("bytes"), minimum=1, field="decision bytes"
                ),
                decision_sha256=_sha256(decision.get("sha256"), field="decision"),
                documents=_plain_int(
                    documents.get("records"), minimum=1, field="archive documents"
                ),
                clean_bytes=_plain_int(
                    documents.get("clean_bytes"),
                    minimum=1,
                    field="archive clean bytes",
                ),
                content_tokens=_plain_int(
                    documents.get("content_tokens"),
                    minimum=1,
                    field="archive content tokens",
                ),
                kept_documents=_plain_int(
                    decision.get("kept_documents"), field="kept documents"
                ),
            )
            if (
                decision.get("format") != ALL_ELIGIBLE_BITMAP_FORMAT
                or decision.get("format_version")
                != ALL_ELIGIBLE_BITMAP_FORMAT_VERSION
                or authority.kept_documents > authority.documents
            ):
                raise PortableFinalizationError(
                    f"Invalid selection-v7 decision for {authority.archive}"
                )
            archives.append(authority)
        digest = canonical_sha256(
            [archive.materialization_identity() for archive in archives]
        )
        if digest != self.identity.get("archive_inventory_sha256"):
            raise PortableFinalizationError(
                "Reconstructed archive inventory differs from the packed journal"
            )
        state = self.journal["state"] if self.journal is not None else {}
        if state.get("archive_count") != len(archives):
            raise PortableFinalizationError("Journal/archive inventory counts differ")
        self.archives = archives

    def _validate_packed_structure(self, path: Path) -> dict[str, Any]:
        try:
            if self.config.verify_packed_payloads:
                return training_data.validate_packed_manifest(
                    path, verify_checksums=True
                )
            manifest, shards = training_data._parse_packed_manifest(path)  # type: ignore[attr-defined]
        except (OSError, TypeError, ValueError) as exc:
            raise PortableFinalizationError(f"Invalid packed manifest {path}: {exc}") from exc
        expected_names = {"manifest.json"}
        for shard in shards:
            expected_names.update((shard.tokens_path.name, shard.starts_path.name))
            for payload_path, expected_bytes in (
                (
                    shard.tokens_path,
                    shard.rows
                    * manifest["tokens_per_row"]
                    * training_data.TOKEN_DTYPE.itemsize,
                ),
                (
                    shard.starts_path,
                    shard.rows * manifest["starts_bytes_per_row"],
                ),
            ):
                _regular_file(payload_path, field="packed payload")
                if payload_path.stat().st_size != expected_bytes:
                    raise PortableFinalizationError(
                        f"Packed payload size mismatch: {payload_path}"
                    )
        found_names = {entry.name for entry in path.parent.iterdir()}
        if found_names != expected_names:
            raise PortableFinalizationError(
                f"Packed directory is not closed-world: {path.parent}"
            )
        return manifest

    def _authenticate_packed(self) -> None:
        packing = self.identity["packing_configuration"]
        for split in SPLITS:
            for domain in training_data.DOMAIN_ORDER:
                path = self.corpus_root / "packed" / split / domain / "manifest.json"
                manifest = self._validate_packed_structure(path)
                expected = {
                    "split": split,
                    "domain": domain,
                    "sequence_length": packing.get("sequence_length"),
                    "rows_per_shard": packing.get("rows_per_shard"),
                    "construction_seed": packing.get("construction_seed"),
                    "vocab_size": packing.get("expected_vocab_size"),
                    "eos_token_id": packing.get("expected_eos_token_id"),
                    "tokenizer_manifest_sha256": self.identity.get(
                        "tokenizer_manifest_sha256"
                    ),
                    "curation_policy_sha256": self.identity.get(
                        "curation_policy_sha256"
                    ),
                    "selection_manifest_sha256": self.selection_sha256,
                }
                if any(manifest.get(key) != value for key, value in expected.items()):
                    raise PortableFinalizationError(
                        f"Packed manifest identity mismatch for {split}/{domain}"
                    )
                cursor = self.cursors[(split, domain)]
                if manifest.get("construction_last_source_cursor") != cursor:
                    raise PortableFinalizationError(
                        f"Packed manifest cursor differs for {split}/{domain}"
                    )
                if (
                    manifest.get("documents") != cursor["selected_documents"]
                    or manifest.get("source_content_tokens")
                    != cursor["selected_content_tokens"]
                ):
                    raise PortableFinalizationError(
                        f"Packed totals differ from cursor for {split}/{domain}"
                    )
                self.packed[(split, domain)] = manifest

    def _authenticate_document_indexes(self) -> None:
        archive_count = len(self.archives)
        for split in SPLITS:
            for domain in training_data.DOMAIN_ORDER:
                path = (
                    self.corpus_root
                    / "provenance"
                    / "documents"
                    / split
                    / domain
                    / "manifest.json"
                )
                manifest = _load_json(path)
                expected = {
                    "format": DOCUMENT_INDEX_FORMAT,
                    "format_version": DOCUMENT_INDEX_VERSION,
                    "split": split,
                    "domain": domain,
                    "selection_manifest_sha256": self.selection_sha256,
                    "tokenizer_manifest_sha256": self.identity[
                        "tokenizer_manifest_sha256"
                    ],
                    "sequence_length": self.identity["packing_configuration"][
                        "sequence_length"
                    ],
                }
                if any(manifest.get(key) != value for key, value in expected.items()):
                    raise PortableFinalizationError(
                        f"Document-index identity mismatch for {split}/{domain}"
                    )
                cursor = self.cursors[(split, domain)]
                shards = manifest.get("shards")
                if shards != cursor["document_index_shards"]:
                    raise PortableFinalizationError(
                        f"Document-index inventory differs from cursor for {split}/{domain}"
                    )
                logical_position = 0
                selected_tokens = 0
                records = 0
                last_ordinal = -1
                for descriptor in shards:
                    if not isinstance(descriptor, dict):
                        raise PortableFinalizationError("Invalid document-index descriptor")
                    ordinal = _plain_int(
                        descriptor.get("archive_ordinal"),
                        field="document-index archive ordinal",
                    )
                    if ordinal <= last_ordinal or ordinal >= archive_count:
                        raise PortableFinalizationError(
                            "Document-index archive order is invalid"
                        )
                    last_ordinal = ordinal
                    expected_relative = (
                        Path("provenance")
                        / "documents"
                        / split
                        / domain
                        / f"archive-{ordinal:06d}.jsonl.zst"
                    ).as_posix()
                    if descriptor.get("path") != expected_relative:
                        raise PortableFinalizationError(
                            "Document-index path is not canonical"
                        )
                    shard_path = _safe_file(
                        self.corpus_root,
                        expected_relative,
                        field="document-index shard",
                    )
                    if (
                        shard_path.stat().st_size != descriptor.get("bytes")
                        or file_sha256(shard_path) != descriptor.get("sha256")
                    ):
                        raise PortableFinalizationError(
                            f"Document-index checksum mismatch: {shard_path}"
                        )
                    archive = self.archives[ordinal]
                    shard_records = 0
                    try:
                        rows = iter_jsonl_zst(shard_path)
                        for row in rows:
                            if (
                                row.get("record_version") != DOCUMENT_INDEX_VERSION
                                or row.get("split") != split
                                or row.get("domain") != domain
                                or row.get("bucket") != archive.bucket
                                or row.get("source_archive") != archive.archive
                                or row.get("source_archive_ordinal") != ordinal
                            ):
                                raise PortableFinalizationError(
                                    f"Document-index source mismatch: {shard_path}"
                                )
                            for field in ("doc_id", "canonical_doc_id", "split_group_id"):
                                _sha256(row.get(field), field=f"document index {field}")
                            source_tokens = _plain_int(
                                row.get("source_tokens"),
                                minimum=1,
                                field="document source tokens",
                            )
                            selected = _plain_int(
                                row.get("selected_content_tokens"),
                                minimum=1,
                                field="document selected tokens",
                            )
                            if (
                                selected != source_tokens
                                or row.get("terminal_quota_prefix") is not False
                                or row.get("logical_stream_start") != logical_position
                                or row.get("logical_content_end_exclusive")
                                != logical_position + selected
                                or row.get("logical_eos_position")
                                != logical_position + selected
                            ):
                                raise PortableFinalizationError(
                                    f"Document-index logical position mismatch: {shard_path}"
                                )
                            source_index = row.get("source_manifest_index")
                            if (
                                isinstance(source_index, bool)
                                or not isinstance(source_index, int)
                                or not 0 <= source_index < archive.documents
                                or not isinstance(row.get("source_member"), str)
                                or not row["source_member"]
                                or not isinstance(row.get("language"), str)
                                or not row["language"].strip()
                            ):
                                raise PortableFinalizationError(
                                    f"Invalid document-index record: {shard_path}"
                                )
                            logical_position += selected + 1
                            selected_tokens += selected
                            records += 1
                            shard_records += 1
                    except PortableFinalizationError:
                        raise
                    except (OSError, UnicodeError, ValueError) as exc:
                        raise PortableFinalizationError(
                            f"Cannot scan document-index shard {shard_path}: {exc}"
                        ) from exc
                    if descriptor.get("records") != shard_records or shard_records < 1:
                        raise PortableFinalizationError(
                            f"Document-index record count mismatch: {shard_path}"
                        )
                totals = {
                    "documents": records,
                    "selected_content_tokens": selected_tokens,
                    "logical_stream_tokens": logical_position,
                }
                if any(manifest.get(key) != value for key, value in totals.items()):
                    raise PortableFinalizationError(
                        f"Document-index totals are invalid for {split}/{domain}"
                    )
                packed = self.packed[(split, domain)]
                if (
                    records != packed["documents"]
                    or selected_tokens != packed["source_content_tokens"]
                    or logical_position != packed["stream_tokens"]
                ):
                    raise PortableFinalizationError(
                        f"Document-index/packed totals differ for {split}/{domain}"
                    )
                self.document_indexes[(split, domain)] = manifest

    def authenticate(self) -> None:
        self._load_journal()
        self._authenticate_restore_receipt()
        self._authenticate_selection()
        self._authenticate_configs()
        self._authenticate_tokenizer()
        self._authenticate_cache_inventory()
        self._build_archive_authority()
        self._authenticate_packed()
        self._authenticate_document_indexes()

    def _available_rows(self, split: str) -> dict[str, int]:
        return {
            domain: int(self.packed[(split, domain)]["rows"])
            for domain in training_data.DOMAIN_ORDER
        }

    def _target_rows(
        self,
        split: str,
        *,
        global_microbatch_rows: int | None,
        gradient_accumulation_steps: int | None,
    ) -> int:
        sequence_length = int(
            self.identity["packing_configuration"]["sequence_length"]
        )
        caps = {
            "train": self.config.maximum_train_input_tokens,
            "validation": self.config.maximum_validation_input_tokens,
            "test": self.config.maximum_test_input_tokens,
        }
        if split == "train":
            if global_microbatch_rows is None or gradient_accumulation_steps is None:
                raise PortableFinalizationError("Train target requires frozen geometry")
            optimizer_rows = global_microbatch_rows * gradient_accumulation_steps
            rows = _aligned_train_rows(
                maximum_input_tokens=caps[split],
                sequence_length=sequence_length,
                optimizer_update_rows=optimizer_rows,
            )
            counts = training_data._strict_weighted_row_counts(  # type: ignore[attr-defined]
                rows, EXPECTED_WEIGHTS
            )
            available = self._available_rows(split)
            insufficient = {
                domain: {"required": counts[domain], "available": available[domain]}
                for domain in training_data.DOMAIN_ORDER
                if counts[domain] > available[domain]
            }
            if insufficient:
                raise PortableFinalizationError(
                    "Packed train supply cannot satisfy the authorized 52.58B "
                    f"strict-weight order: {insufficient}"
                )
            return rows
        else:
            quantum = 5
        return _largest_feasible_rows(
            self._available_rows(split),
            maximum_rows=caps[split] // sequence_length,
            row_quantum=quantum,
        )

    def _order_budget(
        self,
        split: str,
        *,
        rows: int,
    ) -> tuple[int, int, int]:
        sequence_length = int(
            self.identity["packing_configuration"]["sequence_length"]
        )
        actual = rows * sequence_length
        if split == "train":
            expected = self.config.maximum_train_input_tokens
            tolerance = expected - actual
            if tolerance < 0:
                raise PortableFinalizationError("Train order exceeds its token cap")
            return expected, actual, tolerance
        return actual, actual, 0

    def _validate_order(
        self,
        split: str,
        *,
        expected_tokens: int,
        actual_tokens: int,
        tolerance_tokens: int,
        global_microbatch_rows: int | None,
        gradient_accumulation_steps: int | None,
    ) -> dict[str, Any]:
        path = self.corpus_root / "orders" / split / "manifest.json"
        try:
            manifest = training_data.validate_training_order(path)
        except (OSError, TypeError, ValueError) as exc:
            raise PortableFinalizationError(f"Invalid {split} order: {exc}") from exc
        expected_geometry = (
            (global_microbatch_rows, gradient_accumulation_steps)
            if split == "train"
            else (None, None)
        )
        consumption = manifest.get("training_consumption")
        if not isinstance(consumption, dict):
            raise PortableFinalizationError(f"{split} order has no consumption contract")
        found_geometry = (
            consumption.get("frozen_global_microbatch_rows"),
            consumption.get("frozen_gradient_accumulation_steps"),
        )
        expected = {
            "format_version": training_data.ORDER_FORMAT_VERSION,
            "split": split,
            "seed": self.config.order_seed + SPLITS.index(split),
            "expected_input_token_weights": EXPECTED_WEIGHTS,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise PortableFinalizationError(f"{split} order identity mismatch")
        if found_geometry != expected_geometry:
            raise PortableFinalizationError(f"{split} order geometry mismatch")
        budget = manifest.get("input_token_budget")
        if (
            not isinstance(budget, dict)
            or budget.get("expected_total") != expected_tokens
            or budget.get("actual_total") != actual_tokens
            or budget.get("tolerance") != tolerance_tokens
        ):
            raise PortableFinalizationError(f"{split} order token target mismatch")
        return manifest

    def _build_order(
        self,
        split: str,
        *,
        global_microbatch_rows: int | None,
        gradient_accumulation_steps: int | None,
    ) -> dict[str, Any]:
        rows = self._target_rows(
            split,
            global_microbatch_rows=global_microbatch_rows,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        expected_tokens, actual_tokens, tolerance_tokens = self._order_budget(
            split, rows=rows
        )
        orders_root = self.corpus_root / "orders"
        orders_root.mkdir(parents=True, exist_ok=True)
        final = orders_root / split
        staging = orders_root / f".{split}.portable-part"
        if final.exists():
            if staging.exists():
                raise PortableFinalizationError(
                    f"Both final and staging order exist for {split}"
                )
            return self._validate_order(
                split,
                expected_tokens=expected_tokens,
                actual_tokens=actual_tokens,
                tolerance_tokens=tolerance_tokens,
                global_microbatch_rows=global_microbatch_rows,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        if staging.exists():
            if staging.is_symlink() or not staging.is_dir():
                raise PortableFinalizationError(f"Unsafe order staging path: {staging}")
            shutil.rmtree(staging)
        manifests = {
            domain: self.corpus_root
            / "packed"
            / split
            / domain
            / "manifest.json"
            for domain in training_data.DOMAIN_ORDER
        }
        kwargs: dict[str, Any] = {}
        if split == "train":
            kwargs.update(
                frozen_global_microbatch_rows=global_microbatch_rows,
                frozen_gradient_accumulation_steps=gradient_accumulation_steps,
            )
        try:
            training_data.build_training_order(
                manifests,
                staging,
                seed=self.config.order_seed + SPLITS.index(split),
                expected_weights=EXPECTED_WEIGHTS,
                expected_total_input_tokens=expected_tokens,
                input_token_tolerance=tolerance_tokens,
                **kwargs,
            )
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            raise PortableFinalizationError(
                f"Cannot build {split} order: {exc}"
            ) from exc
        manifest = self._validate_staged_order(
            staging / "manifest.json",
            split=split,
            expected_tokens=expected_tokens,
            actual_tokens=actual_tokens,
            tolerance_tokens=tolerance_tokens,
            global_microbatch_rows=global_microbatch_rows,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        os.replace(staging, final)
        _fsync_directory(orders_root)
        return manifest

    def _validate_staged_order(
        self,
        path: Path,
        *,
        split: str,
        expected_tokens: int,
        actual_tokens: int,
        tolerance_tokens: int,
        global_microbatch_rows: int | None,
        gradient_accumulation_steps: int | None,
    ) -> dict[str, Any]:
        # Temporarily validate the staged path using the same exact invariants
        # as a published order; relative dataset paths are already final-safe
        # because both directories are siblings under orders/.
        try:
            manifest = training_data.validate_training_order(path)
        except (OSError, TypeError, ValueError) as exc:
            raise PortableFinalizationError(f"Invalid staged {split} order: {exc}") from exc
        expected_geometry = (
            (global_microbatch_rows, gradient_accumulation_steps)
            if split == "train"
            else (None, None)
        )
        consumption = manifest["training_consumption"]
        if (
            manifest.get("split") != split
            or manifest.get("seed") != self.config.order_seed + SPLITS.index(split)
            or manifest.get("expected_input_token_weights") != EXPECTED_WEIGHTS
            or manifest["input_token_budget"].get("expected_total")
            != expected_tokens
            or manifest["input_token_budget"].get("actual_total") != actual_tokens
            or manifest["input_token_budget"].get("tolerance")
            != tolerance_tokens
            or (
                consumption.get("frozen_global_microbatch_rows"),
                consumption.get("frozen_gradient_accumulation_steps"),
            )
            != expected_geometry
        ):
            raise PortableFinalizationError(f"Staged {split} order identity mismatch")
        return manifest

    def _provenance_payloads(self) -> dict[str, dict[str, Any]]:
        selection_identity = self.selection["identity"]
        source: dict[str, Any] = {
            "format_version": 1,
            "selection_manifest_sha256": self.selection_sha256,
            "curation_identity_format_version": selection_identity["format_version"],
            "curation_profile": selection_identity.get("curation_profile"),
            "known_provenance_limitations": self.selection.get(
                "known_provenance_limitations", []
            ),
            "raw_archives_hashed_for_integrity": self.selection[
                "raw_archives_hashed_for_integrity"
            ],
            "raw_archive_payloads_parsed_by_curation": self.selection[
                "raw_archive_payloads_parsed_by_curation"
            ],
            "curation_sqlite_runtime": selection_identity["sqlite_runtime"],
            "curation_storage_contract": selection_identity[
                "curation_storage_contract"
            ],
            "collection_completeness_sha256": canonical_sha256(
                self.selection["collection_completeness"]
            ),
            "collection_completeness": self.selection["collection_completeness"],
            "source_manifests": selection_identity["source_manifests"],
            "raw_archives": [archive.source_provenance() for archive in self.archives],
            "publication_scope": self.selection["publication_scope"],
            "raw_archive_integrity_policy": selection_identity[
                "raw_archive_integrity_policy"
            ],
            "curation_sqlite_execution": selection_identity["sqlite_execution"],
            "fast_all_eligible_handoff": selection_identity[
                "fast_all_eligible_handoff"
            ],
            "source_curation": selection_identity["source_curation"],
        }
        policy = {
            "format_version": 1,
            "curation_policy_sha256": selection_identity["policy_sha256"],
            "quota_config_sha256": selection_identity["quota_config_sha256"],
            "benchmark_guard_sha256": selection_identity["benchmark_guard_sha256"],
            "selection_policy": self.selection["selection_policy"],
            "leakage_audit": self.selection["leakage_audit"],
            "curation_profile": selection_identity.get("curation_profile"),
            "selection_strategy": self.selection["selection_strategy"],
            "selection_profile": self.selection["selection_profile"],
            "training_input_budget_authority": self.selection[
                "training_input_budget_authority"
            ],
            "selected_totals": self.selection["selected_totals"],
            "reference_quotas": self.selection["reference_quotas"],
        }
        tokenizer = {
            "format_version": 1,
            "tokenizer_manifest_sha256": selection_identity[
                "tokenizer_manifest_sha256"
            ],
            "resolved_revision": self.tokenizer_manifest["resolved_revision"],
            "validation": self.tokenizer_manifest["validation"],
            "files": self.tokenizer_manifest["files"],
        }
        fingerprints = {
            "format_version": 1,
            "preprocess_manifest_sha256": selection_identity[
                "preprocess_manifest_sha256"
            ],
            "report_inventory_sha256": selection_identity["report_inventory_sha256"],
            "english_near_clusters_sha256": selection_identity.get(
                "english_near_clusters_sha256"
            ),
            "english_near_artifact": selection_identity.get("english_near_artifact"),
            "english_near_dedup_complete": self.selection[
                "english_near_dedup_complete"
            ],
            "english_near_dedup_status": self.selection.get(
                "english_near_dedup_status"
            ),
            "curation_profile": selection_identity.get("curation_profile"),
            "decision_inventory_sha256": self.selection[
                "decision_inventory_sha256"
            ],
            "archives": [
                archive.fingerprint_provenance() for archive in self.archives
            ],
        }
        raw_cache = {
            "format_version": 1,
            "acceleration_only": True,
            "training_ready": False,
            "inventory": self.identity["raw_token_cache"],
            "cache_contract": self.cache_inventory["cache"],
            "tokenizer": self.cache_inventory["tokenizer"],
            "archives": self.cache_inventory["archives"],
            "non_authorities": self.cache_inventory["non_authorities"],
        }
        handoff = {
            "format": PORTABLE_FINALIZATION_FORMAT,
            "format_version": PORTABLE_FINALIZATION_VERSION,
            "packed_journal_sha256": self.journal_sha256,
            "selection_manifest_sha256": self.selection_sha256,
            "cache_inventory_manifest_sha256": self.cache_inventory_sha256,
            "cache_inventory_sidecar_sha256": self.cache_sidecar_sha256,
            "tokenizer_manifest_sha256": self.identity["tokenizer_manifest_sha256"],
            "raw_preprocess_and_cache_payloads_required": False,
            "raw_preprocess_and_cache_payloads_revalidated": False,
            "packed_payload_verification": (
                "full-local-sha256-and-semantics"
                if self.config.verify_packed_payloads
                else "authenticated-s3-restore-receipt-plus-local-size-contract"
            ),
            "restore_receipt": (
                None
                if self.config.restore_ready_path is None
                else {
                    "path": str(self.config.restore_ready_path.absolute()),
                    "sha256": file_sha256(self.config.restore_ready_path.absolute()),
                    "remote_inventory_sha256": self.restore_receipt[
                        "remote_inventory_sha256"
                    ],
                    "payload_sha256_verified": True,
                }
            ),
            "packed_manifests_authenticated": 9,
            "document_index_manifests_authenticated": 9,
        }
        return {
            "source.json": source,
            "policy.json": policy,
            "tokenizer.json": tokenizer,
            "fingerprints.json": fingerprints,
            "raw_token_cache.json": raw_cache,
            "handoff.json": handoff,
        }

    def _write_provenance(self) -> dict[str, dict[str, Any]]:
        root = self.corpus_root / "provenance"
        root.mkdir(parents=True, exist_ok=True)
        result: dict[str, dict[str, Any]] = {}
        for name, payload in sorted(self._provenance_payloads().items()):
            path = root / name
            if path.exists():
                if _load_json(path) != payload:
                    raise PortableFinalizationError(
                        f"Existing provenance differs: {path}"
                    )
            else:
                atomic_json(path, payload)
            result[name.removesuffix(".json")] = {
                "path": f"provenance/{name}",
                "sha256": file_sha256(path),
            }
        return result

    def _order_configuration(
        self,
        orders: Mapping[str, Mapping[str, Any]],
        *,
        global_microbatch_rows: int,
        gradient_accumulation_steps: int,
    ) -> dict[str, Any]:
        return {
            "order_seeds": {
                split: self.config.order_seed + offset
                for offset, split in enumerate(SPLITS)
            },
            "frozen_global_microbatch_rows": global_microbatch_rows,
            "frozen_gradient_accumulation_steps": gradient_accumulation_steps,
            "expected_train_input_tokens": orders["train"]["input_token_budget"][
                "expected_total"
            ],
            "expected_validation_input_tokens": orders["validation"][
                "input_token_budget"
            ]["expected_total"],
            "expected_test_input_tokens": orders["test"]["input_token_budget"][
                "expected_total"
            ],
            "train_input_token_tolerance": orders["train"]["input_token_budget"][
                "tolerance"
            ],
            "validation_input_token_tolerance": orders["validation"][
                "input_token_budget"
            ]["tolerance"],
            "test_input_token_tolerance": orders["test"]["input_token_budget"][
                "tolerance"
            ],
            "enforce_input_weights": True,
            "expected_input_weights": dict(EXPECTED_WEIGHTS),
        }

    def _final_manifest(
        self,
        orders: Mapping[str, Mapping[str, Any]],
        provenance: Mapping[str, Mapping[str, Any]],
        *,
        global_microbatch_rows: int,
        gradient_accumulation_steps: int,
    ) -> dict[str, Any]:
        splits: dict[str, Any] = {}
        for split in SPLITS:
            packed_descriptors: dict[str, Any] = {}
            for domain in training_data.DOMAIN_ORDER:
                packed_path = (
                    self.corpus_root / "packed" / split / domain / "manifest.json"
                )
                index_path = (
                    self.corpus_root
                    / "provenance"
                    / "documents"
                    / split
                    / domain
                    / "manifest.json"
                )
                packed = self.packed[(split, domain)]
                index = self.document_indexes[(split, domain)]
                packed_descriptors[domain] = {
                    "path": packed_path.relative_to(self.corpus_root).as_posix(),
                    "sha256": file_sha256(packed_path),
                    "rows": packed["rows"],
                    "documents": packed["documents"],
                    "source_content_tokens": packed["source_content_tokens"],
                    "document_index": {
                        "path": index_path.relative_to(self.corpus_root).as_posix(),
                        "sha256": file_sha256(index_path),
                        "documents": index["documents"],
                        "logical_stream_tokens": index["logical_stream_tokens"],
                    },
                }
            order_path = self.corpus_root / "orders" / split / "manifest.json"
            order = orders[split]
            splits[split] = {
                "packed": packed_descriptors,
                "order": {
                    "path": order_path.relative_to(self.corpus_root).as_posix(),
                    "sha256": file_sha256(order_path),
                    "format_version": order["format_version"],
                    "rows": order["rows"],
                    "packed_available_rows": order["packed_available_rows"],
                    "packed_surplus_rows": order["packed_surplus_rows"],
                    "authorized_input_tokens": order["input_token_budget"][
                        "actual_total"
                    ],
                },
            }
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "identity": self.identity,
            "order_configuration": self._order_configuration(
                orders,
                global_microbatch_rows=global_microbatch_rows,
                gradient_accumulation_steps=gradient_accumulation_steps,
            ),
            "source_cursor": {
                "next_archive": len(self.archives),
                "archive_count": len(self.archives),
            },
            "split_isolation": {
                "authoritative_assignment": self.selection["selection_profile"][
                    "split_authority"
                ],
                "physical_outputs_separate": True,
                "curation_leakage_audit": self.selection["leakage_audit"],
            },
            "splits": splits,
            "provenance": dict(provenance),
            "finalization": {
                "format": PORTABLE_FINALIZATION_FORMAT,
                "format_version": PORTABLE_FINALIZATION_VERSION,
                "reused_packed_payloads_without_repacking": True,
                "packed_journal_sha256": self.journal_sha256,
            },
        }

    def _validate_completed(self) -> dict[str, Any]:
        manifest_path = self.corpus_root / "manifest.json"
        sidecar_path = self.corpus_root / "manifest.sha256"
        manifest = _load_json(manifest_path)
        digest = file_sha256(manifest_path)
        if _regular_file(sidecar_path, field="corpus manifest sidecar").read_text(
            encoding="ascii"
        ) != f"{digest}  manifest.json\n":
            raise PortableFinalizationError("Corpus manifest sidecar mismatch")
        if (
            manifest.get("format") != FORMAT
            or manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("identity") != self.identity
        ):
            raise PortableFinalizationError("Completed corpus identity mismatch")
        for split in SPLITS:
            split_payload = manifest.get("splits", {}).get(split)
            if not isinstance(split_payload, dict):
                raise PortableFinalizationError(f"Missing completed split {split}")
            packed_descriptors = split_payload.get("packed")
            if (
                not isinstance(packed_descriptors, dict)
                or set(packed_descriptors) != set(training_data.DOMAIN_ORDER)
            ):
                raise PortableFinalizationError(
                    f"Completed {split} packed inventory is invalid"
                )
            for domain in training_data.DOMAIN_ORDER:
                descriptor = packed_descriptors[domain]
                if not isinstance(descriptor, dict):
                    raise PortableFinalizationError(
                        f"Completed {split}/{domain} packed descriptor is invalid"
                    )
                packed_path = _safe_file(
                    self.corpus_root,
                    descriptor.get("path"),
                    field=f"{split}/{domain} packed manifest",
                )
                if file_sha256(packed_path) != descriptor.get("sha256"):
                    raise PortableFinalizationError(
                        f"Completed {split}/{domain} packed manifest changed"
                    )
                packed, _ = training_data._parse_packed_manifest(  # type: ignore[attr-defined]
                    packed_path
                )
                for field in ("rows", "documents", "source_content_tokens"):
                    if descriptor.get(field) != packed.get(field):
                        raise PortableFinalizationError(
                            f"Completed {split}/{domain} {field} differs"
                        )
                index_descriptor = descriptor.get("document_index")
                if not isinstance(index_descriptor, dict):
                    raise PortableFinalizationError(
                        f"Completed {split}/{domain} document index is missing"
                    )
                index_path = _safe_file(
                    self.corpus_root,
                    index_descriptor.get("path"),
                    field=f"{split}/{domain} document-index manifest",
                )
                if file_sha256(index_path) != index_descriptor.get("sha256"):
                    raise PortableFinalizationError(
                        f"Completed {split}/{domain} document-index manifest changed"
                    )
            order_descriptor = split_payload.get("order")
            if not isinstance(order_descriptor, dict):
                raise PortableFinalizationError(f"Missing completed {split} order")
            order_path = _safe_file(
                self.corpus_root,
                order_descriptor.get("path"),
                field=f"{split} order manifest",
            )
            if file_sha256(order_path) != order_descriptor.get("sha256"):
                raise PortableFinalizationError(f"Completed {split} order changed")
            training_data.validate_training_order(order_path)
        for descriptor in manifest.get("provenance", {}).values():
            if not isinstance(descriptor, dict):
                raise PortableFinalizationError("Invalid provenance descriptor")
            path = _safe_file(
                self.corpus_root, descriptor.get("path"), field="provenance artifact"
            )
            if file_sha256(path) != descriptor.get("sha256"):
                raise PortableFinalizationError(f"Provenance changed: {path}")
        if self.journal_path.exists():
            self.journal_path.unlink()
            _fsync_directory(self.corpus_root)
        return manifest

    def _validate_geometry(
        self, global_microbatch_rows: int, gradient_accumulation_steps: int
    ) -> None:
        for field, value in (
            ("global_microbatch_rows", global_microbatch_rows),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
        ):
            _plain_int(value, minimum=1, field=field)
        if global_microbatch_rows % self.config.world_size:
            raise PortableFinalizationError(
                "Global microbatch rows must be divisible by world size"
            )
        if (
            global_microbatch_rows * gradient_accumulation_steps
            != self.config.expected_optimizer_batch_rows
        ):
            raise PortableFinalizationError(
                "Geometry does not preserve the frozen optimizer batch: "
                f"{global_microbatch_rows} * {gradient_accumulation_steps} != "
                f"{self.config.expected_optimizer_batch_rows}"
            )

    def _exclusive_lock(self) -> Any:
        class _Lock:
            def __init__(inner_self, path: Path) -> None:
                inner_self.path = path
                inner_self.handle: Any = None

            def __enter__(inner_self) -> None:
                inner_self.handle = inner_self.path.open("a+b")
                try:
                    fcntl.flock(
                        inner_self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError as exc:
                    inner_self.handle.close()
                    raise PortableFinalizationError(
                        f"Another materializer/finalizer holds {inner_self.path}"
                    ) from exc

            def __exit__(inner_self, exc_type: Any, exc: Any, traceback: Any) -> None:
                assert inner_self.handle is not None
                fcntl.flock(inner_self.handle.fileno(), fcntl.LOCK_UN)
                inner_self.handle.close()

        self.corpus_root.parent.mkdir(parents=True, exist_ok=True)
        return _Lock(
            self.corpus_root.parent
            / f".{self.corpus_root.name}.materialize.lock"
        )

    def run(
        self,
        *,
        mode: str,
        global_microbatch_rows: int | None = None,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, Any]:
        if mode not in ("heldout", "final"):
            raise PortableFinalizationError("mode must be 'heldout' or 'final'")
        if mode == "heldout" and (
            global_microbatch_rows is not None
            or gradient_accumulation_steps is not None
        ):
            raise PortableFinalizationError(
                "Held-out mode must not receive training geometry"
            )
        if mode == "final":
            if global_microbatch_rows is None or gradient_accumulation_steps is None:
                raise PortableFinalizationError(
                    "Final mode requires global microbatch rows and accumulation"
                )
            self._validate_geometry(
                global_microbatch_rows, gradient_accumulation_steps
            )
        with self._exclusive_lock():
            final_manifest = self.corpus_root / "manifest.json"
            final_sidecar = self.corpus_root / "manifest.sha256"
            if final_manifest.exists() and final_sidecar.exists():
                # A completed portable corpus records the original packed
                # journal hash in authenticated provenance; the journal itself
                # is intentionally absent.
                manifest = _load_json(final_manifest)
                identity = manifest.get("identity")
                if not isinstance(identity, dict):
                    raise PortableFinalizationError("Completed corpus has no identity")
                self.identity = dict(identity)
                if mode == "final":
                    order_configuration = manifest.get("order_configuration")
                    if (
                        not isinstance(order_configuration, dict)
                        or order_configuration.get("frozen_global_microbatch_rows")
                        != global_microbatch_rows
                        or order_configuration.get(
                            "frozen_gradient_accumulation_steps"
                        )
                        != gradient_accumulation_steps
                    ):
                        raise PortableFinalizationError(
                            "Completed corpus was finalized with another geometry"
                        )
                self.selection_sha256 = str(identity.get("selection_manifest_sha256"))
                self._authenticate_selection()
                self._authenticate_configs()
                self._authenticate_tokenizer()
                self._authenticate_cache_inventory()
                self._build_completed_archive_authority(manifest)
                return {"complete": True, "manifest": self._validate_completed()}
            if final_sidecar.exists() and not final_manifest.exists():
                raise PortableFinalizationError(
                    "Corpus has a manifest sidecar without manifest.json"
                )
            self.authenticate()
            heldout = {
                split: self._build_order(
                    split,
                    global_microbatch_rows=None,
                    gradient_accumulation_steps=None,
                )
                for split in ("validation", "test")
            }
            if mode == "heldout":
                return {
                    "complete": False,
                    "phase": "heldout-orders",
                    "orders": {
                        split: {
                            "rows": order["rows"],
                            "input_tokens": order["input_token_budget"]["actual_total"],
                            "packed_surplus_rows": order["packed_surplus_rows"],
                        }
                        for split, order in heldout.items()
                    },
                    "train_order_published": False,
                    "journal_preserved": True,
                }
            assert global_microbatch_rows is not None
            assert gradient_accumulation_steps is not None
            train = self._build_order(
                "train",
                global_microbatch_rows=global_microbatch_rows,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            orders = {"train": train, **heldout}
            provenance = self._write_provenance()
            manifest = self._final_manifest(
                orders,
                provenance,
                global_microbatch_rows=global_microbatch_rows,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if final_manifest.exists():
                if _load_json(final_manifest) != manifest:
                    raise PortableFinalizationError(
                        "Existing incomplete corpus manifest differs"
                    )
            else:
                atomic_json(final_manifest, manifest)
            expected_sidecar = (
                f"{file_sha256(final_manifest)}  manifest.json\n".encode("ascii")
            )
            if final_sidecar.exists():
                if final_sidecar.read_bytes() != expected_sidecar:
                    raise PortableFinalizationError("Existing corpus sidecar differs")
            else:
                atomic_bytes(final_sidecar, expected_sidecar)
            return {"complete": True, "manifest": self._validate_completed()}

    def _build_completed_archive_authority(self, manifest: Mapping[str, Any]) -> None:
        """Rehydrate only what completed-manifest validation needs on rerun."""

        source_descriptor = manifest.get("provenance", {}).get("source")
        if not isinstance(source_descriptor, dict):
            raise PortableFinalizationError("Completed corpus has no source provenance")
        source_path = _safe_file(
            self.corpus_root,
            source_descriptor.get("path"),
            field="source provenance",
        )
        if file_sha256(source_path) != source_descriptor.get("sha256"):
            raise PortableFinalizationError("Completed source provenance changed")
        source = _load_json(source_path)
        raw_archives = source.get("raw_archives")
        if not isinstance(raw_archives, list) or not raw_archives:
            raise PortableFinalizationError("Completed source inventory is missing")
        # The full cache/selection reconciliation already passed before the
        # immutable top-level publication.  On an idempotent completed rerun,
        # retain count semantics without inventing missing cold-path fields.
        self.archives = [
            _ArchiveAuthority(
                ordinal=ordinal,
                archive=str(item["archive"]),
                index=ordinal,
                bucket="python",
                raw_bytes=1,
                raw_sha256=str(item["sha256"]),
                report_path="completed",
                report_bytes=1,
                report_sha256="0" * 64,
                fingerprint_path="completed",
                fingerprint_bytes=1,
                fingerprint_sha256="0" * 64,
                decision_path="completed",
                decision_bytes=1,
                decision_sha256="0" * 64,
                documents=int(item["documents"]),
                clean_bytes=1,
                content_tokens=int(item["content_tokens"]),
                kept_documents=0,
            )
            for ordinal, item in enumerate(raw_archives)
        ]


__all__ = [
    "EXPECTED_WEIGHTS",
    "PORTABLE_FINALIZATION_FORMAT",
    "PORTABLE_FINALIZATION_VERSION",
    "PortableFinalizationConfig",
    "PortableFinalizationError",
    "PortablePackedFinalizer",
]
