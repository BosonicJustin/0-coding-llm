#!/usr/bin/env python3
"""Bounded, reproducible recall calibration for English near-dedup LSH.

This program never edits the production near-dedup configuration. It samples a
bounded set of immutable real English documents (or reads a small JSONL
fixture), creates seeded perturbations whose measured full-shingle Jaccard
falls into pinned bins around the production threshold, and measures whether
the pinned candidate LSH retrieves them. Results are published even when the
acceptance gate fails, with a non-zero exit status and a checksum sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Iterator, Sequence

import xxhash
import zstandard

from build_english_near_clusters import (
    DEFAULT_CONFIG as DEFAULT_PRODUCTION_CONFIG,
    DEFAULT_DENYLIST,
    DEFAULT_QUOTA_CONFIG,
    EnglishNearDedupBuilder,
    EnglishNearDedupError,
    atomic_bytes,
    candidate_bands,
    exact_shingle_hashes,
    file_sha256,
    iter_jsonl_zst,
    jaccard_counts,
    load_config as load_production_config,
    refinement_accepts,
    safe_relative_path,
)
from curation_policy import DEFAULT_POLICY, canonical_sha256
from preprocess_raw_stream import FINGERPRINT_VERSION, WORD_RE, normalize_content


FORMAT_VERSION = 1
DEFAULT_CALIBRATION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "english_near_dedup_calibration.json"
)
DEFAULT_ROOT = Path("/workspace/dataset")
PRODUCTION_BUILDER = Path(__file__).resolve().with_name(
    "build_english_near_clusters.py"
)
ENGLISH_BUCKETS = ("fineweb_edu", "wikipedia")
HEX = frozenset("0123456789abcdef")


class CalibrationError(RuntimeError):
    """A calibration input, identity, generation, or publication error."""


@dataclass(frozen=True)
class CalibrationDocument:
    doc_id: str
    bucket: str
    text: str
    source_identity: str
    archive: str | None = None
    manifest_index: int | None = None
    member_path: str | None = None
    content_sha256: str | None = None
    normalized_sha256: str | None = None
    truncated_to_words: int | None = None


@dataclass(frozen=True)
class SampleFingerprint:
    doc_id: str
    bucket: str
    archive: str
    manifest_index: int
    member_path: str
    size_bytes: int
    starcoder2_tokens: int
    word_count: int
    content_sha256: str
    normalized_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    )


def validate_input_identity(identity: dict[str, Any], document_count: int) -> str:
    if not isinstance(identity, dict):
        raise CalibrationError("Calibration input identity must be an object")
    kind = identity.get("kind")
    if kind not in {"jsonl_fixture", "immutable_real_english_sample"}:
        raise CalibrationError(f"Unsupported calibration input identity kind: {kind!r}")
    if identity.get("documents_selected") != document_count:
        raise CalibrationError("Calibration input identity document count mismatch")
    if kind == "jsonl_fixture":
        if not is_sha256(identity.get("sha256")):
            raise CalibrationError("Fixture input identity lacks a valid checksum")
        return str(kind)
    for field in (
        "full_report_inventory_sha256",
        "preprocess_manifest_sha256",
        "curation_policy_sha256",
        "benchmark_guard_sha256",
        "collection_completeness_sha256",
    ):
        if not is_sha256(identity.get(field)):
            raise CalibrationError(f"Real input identity lacks valid {field}")
    completeness = identity.get("collection_completeness")
    if not isinstance(completeness, dict) or hashlib.sha256(
        canonical_json_bytes(completeness)
    ).hexdigest() != identity.get("collection_completeness_sha256"):
        raise CalibrationError("Real input collection-completeness identity mismatch")
    if set(completeness.get("buckets", {})) != set(ENGLISH_BUCKETS):
        raise CalibrationError("Real input completeness evidence lacks an English bucket")
    reports = identity.get("selected_reports")
    if not isinstance(reports, list) or not reports:
        raise CalibrationError("Real input identity has no selected reports")
    for report in reports:
        if not isinstance(report, dict) or not all(
            is_sha256(report.get(field))
            for field in ("report_sha256", "archive_sha256", "fingerprint_sha256")
        ):
            raise CalibrationError("Real input selected-report identity is invalid")
    if not isinstance(identity.get("source_manifests"), dict):
        raise CalibrationError("Real input source-manifest identity is invalid")
    return str(kind)


def stable_priority(seed: str, namespace: str, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).digest(),
        "big",
    )


def load_calibration_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(f"Missing calibration config: {path}") from exc
    if not isinstance(config, dict) or config.get("config_version") != 1:
        raise CalibrationError("Unsupported calibration config")
    if config.get("calibration_algorithm") != "deterministic-real-text-perturbation-v1":
        raise CalibrationError("Unexpected calibration algorithm")
    seed = config.get("seed")
    if not isinstance(seed, str) or not seed:
        raise CalibrationError("Calibration seed must be non-empty")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise CalibrationError("sampling must be an object")
    for field in (
        "maximum_archives_per_bucket",
        "maximum_documents_per_bucket",
        "maximum_fixture_documents",
        "minimum_source_words",
        "maximum_source_words",
    ):
        if not isinstance(sampling.get(field), int) or sampling[field] < 1:
            raise CalibrationError(f"Invalid sampling.{field}")
    if sampling["minimum_source_words"] > sampling["maximum_source_words"]:
        raise CalibrationError("minimum_source_words exceeds maximum_source_words")
    perturbations = config.get("perturbations")
    if not isinstance(perturbations, dict):
        raise CalibrationError("perturbations must be an object")
    operations = perturbations.get("operations")
    allowed = {
        "append_donor_fragment",
        "truncate_tail",
        "replace_contiguous_with_donor",
    }
    if not isinstance(operations, list) or not operations or set(operations) - allowed:
        raise CalibrationError("Unsupported perturbation operation")
    for field in (
        "variants_per_document_per_bin",
        "maximum_search_evaluations_per_operation",
    ):
        if not isinstance(perturbations.get(field), int) or perturbations[field] < 1:
            raise CalibrationError(f"Invalid perturbations.{field}")
    bins = perturbations.get("bins")
    if not isinstance(bins, list) or not bins:
        raise CalibrationError("At least one Jaccard bin is required")
    names: set[str] = set()
    previous_max = -1.0
    for row in bins:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise CalibrationError("Invalid Jaccard bin")
        if row["name"] in names:
            raise CalibrationError(f"Duplicate Jaccard bin {row['name']}")
        names.add(row["name"])
        minimum = row.get("minimum_numerator")
        maximum = row.get("maximum_numerator")
        denominator = row.get("denominator")
        if not all(isinstance(value, int) and value > 0 for value in (minimum, maximum, denominator)):
            raise CalibrationError(f"Invalid Jaccard bin {row['name']}")
        if minimum > maximum or maximum > denominator:
            raise CalibrationError(f"Invalid Jaccard bounds for {row['name']}")
        minimum_float = minimum / denominator
        maximum_float = maximum / denominator
        if minimum_float <= previous_max:
            raise CalibrationError("Jaccard bins must be strictly ordered and non-overlapping")
        previous_max = maximum_float
    acceptance = config.get("acceptance")
    if not isinstance(acceptance, dict):
        raise CalibrationError("acceptance must be an object")
    for field in (
        "minimum_eligible_documents",
        "minimum_pairs_at_or_above_production_threshold",
        "maximum_generation_failures",
        "maximum_exact_refinement_decision_errors",
    ):
        if not isinstance(acceptance.get(field), int) or acceptance[field] < 0:
            raise CalibrationError(f"Invalid acceptance.{field}")
    recall = acceptance.get("minimum_candidate_recall")
    if not isinstance(recall, (int, float)) or not 0 <= float(recall) <= 1:
        raise CalibrationError("minimum_candidate_recall must be in [0,1]")
    if acceptance.get("recall_acceptance_statistic") != "one-sided-wilson-lower-bound":
        raise CalibrationError(
            "Recall acceptance must use the pinned one-sided Wilson lower bound"
        )
    confidence = acceptance.get("recall_confidence_level")
    if not isinstance(confidence, (int, float)) or not 0.5 < float(confidence) < 1:
        raise CalibrationError("recall_confidence_level must be in (0.5,1)")
    if acceptance.get("require_every_bin") is not True:
        raise CalibrationError("Every pinned calibration bin must remain required")
    return config


def iter_fixture(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".zst":
        yield from iter_jsonl_zst(path)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CalibrationError(f"Blank fixture row {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"Invalid fixture JSON {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise CalibrationError(f"Non-object fixture row {path}:{line_number}")
            yield row


def load_fixture_documents(
    path: Path, *, seed: str, maximum_documents: int
) -> tuple[list[CalibrationDocument], dict[str, Any]]:
    if not path.is_file():
        raise CalibrationError(f"Missing calibration fixture: {path}")
    heap: list[tuple[int, str, CalibrationDocument]] = []
    seen: set[str] = set()
    scanned = 0
    for index, row in enumerate(iter_fixture(path)):
        scanned += 1
        doc_id = row.get("doc_id")
        text = row.get("text")
        bucket = row.get("bucket", "fixture")
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
            raise CalibrationError(f"Invalid/duplicate fixture doc_id at row {index}")
        if not isinstance(text, str) or not text.strip():
            raise CalibrationError(f"Fixture text must be non-empty at row {index}")
        if not isinstance(bucket, str) or not bucket:
            raise CalibrationError(f"Invalid fixture bucket at row {index}")
        seen.add(doc_id)
        document = CalibrationDocument(
            doc_id=doc_id,
            bucket=bucket,
            text=text,
            source_identity=f"fixture:{path.name}:{index}",
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            normalized_sha256=hashlib.sha256(
                normalize_content(text, "fineweb_edu").encode("utf-8")
            ).hexdigest(),
        )
        priority = stable_priority(seed, "fixture-document", doc_id)
        entry = (-priority, doc_id, document)
        if len(heap) < maximum_documents:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    documents = [entry[2] for entry in heap]
    documents.sort(key=lambda item: (stable_priority(seed, "fixture-document", item.doc_id), item.doc_id))
    if not documents:
        raise CalibrationError("Calibration fixture contains no documents")
    identity = {
        "kind": "jsonl_fixture",
        "filename": path.name,
        "sha256": file_sha256(path),
        "rows_scanned": scanned,
        "documents_selected": len(documents),
        "selection_seed": seed,
        "maximum_documents": maximum_documents,
    }
    return documents, identity


def _validate_sample_fingerprint(
    row: dict[str, Any], report: dict[str, Any], expected_index: int
) -> SampleFingerprint:
    if row.get("record_version") != 1 or row.get("fingerprint_version") != FINGERPRINT_VERSION:
        raise CalibrationError("Unsupported sampled fingerprint record")
    if row.get("bucket") != report["bucket"] or row.get("archive") != report["archive"]:
        raise CalibrationError("Sampled report/fingerprint identity mismatch")
    if row.get("manifest_index") != expected_index:
        raise CalibrationError("Sampled fingerprint indices are not contiguous")
    member_path = row.get("member_path")
    if not isinstance(member_path, str) or not member_path:
        raise CalibrationError("Invalid sampled member path")
    expected_doc = hashlib.sha256(
        f"{report['archive']}\0{member_path}".encode("utf-8")
    ).hexdigest()
    doc_id = row.get("doc_id")
    if doc_id != expected_doc:
        raise CalibrationError("Unstable sampled document ID")
    for field in ("content_sha256", "normalized_sha256"):
        value = row.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in HEX for character in value)
        ):
            raise CalibrationError(f"Invalid sampled {field}")
    size = row.get("size_bytes")
    tokens = row.get("starcoder2_tokens")
    if not isinstance(size, int) or size <= 0 or not isinstance(tokens, int) or tokens <= 0:
        raise CalibrationError("Invalid sampled document size/token count")
    metrics = row.get("metrics")
    words = metrics.get("words") if isinstance(metrics, dict) else None
    if not isinstance(words, int) or words < 0:
        raise CalibrationError("Invalid sampled fingerprint word count")
    return SampleFingerprint(
        doc_id=doc_id,
        bucket=str(report["bucket"]),
        archive=str(report["archive"]),
        manifest_index=expected_index,
        member_path=member_path,
        size_bytes=size,
        starcoder2_tokens=tokens,
        word_count=words,
        content_sha256=str(row["content_sha256"]),
        normalized_sha256=str(row["normalized_sha256"]),
    )


def _select_lowest(
    items: Iterable[Any], maximum: int, priority: Any, identity: Any
) -> list[Any]:
    heap: list[tuple[int, str, Any]] = []
    for item in items:
        item_priority = int(priority(item))
        item_identity = str(identity(item))
        entry = (-item_priority, item_identity, item)
        if len(heap) < maximum:
            heapq.heappush(heap, entry)
        elif item_priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    result = [row[2] for row in heap]
    result.sort(key=lambda item: (priority(item), identity(item)))
    return result


def sample_real_documents(
    *,
    root: Path,
    staging_root: Path,
    production_config_path: Path,
    policy_path: Path,
    denylist_path: Path,
    quota_config_path: Path,
    seed: str,
    maximum_archives_per_bucket: int,
    maximum_documents_per_bucket: int,
    minimum_source_words: int,
    identity_output_hint: Path,
) -> tuple[list[CalibrationDocument], dict[str, Any]]:
    if (
        maximum_archives_per_bucket < 1
        or maximum_documents_per_bucket < 1
        or minimum_source_words < 1
    ):
        raise CalibrationError("Real-sample bounds must be positive")
    # Constructor performs the same policy, source, tokenizer, preprocess and
    # frozen-report identity checks as the production builder, but it creates
    # no directory or database unless entered as a context manager.
    try:
        identity_builder = EnglishNearDedupBuilder(
            root=root,
            staging_root=staging_root,
            output=identity_output_hint,
            config_path=production_config_path,
            policy_path=policy_path,
            denylist_path=denylist_path,
            quota_config_path=quota_config_path,
            identity_probe_only=True,
        )
    except EnglishNearDedupError as exc:
        raise CalibrationError(str(exc)) from exc
    selected_reports: list[Any] = []
    for bucket in ENGLISH_BUCKETS:
        bucket_reports = [item for item in identity_builder.reports if item.report["bucket"] == bucket]
        selected_reports.extend(
            _select_lowest(
                bucket_reports,
                maximum_archives_per_bucket,
                lambda item: stable_priority(seed, "real-archive", item.report_sha256),
                lambda item: item.report["archive"],
            )
        )
    if not selected_reports:
        raise CalibrationError("No English reports were selected")

    selected_fingerprints: list[SampleFingerprint] = []
    selected_report_identities: list[dict[str, Any]] = []
    sampling_counts: dict[str, dict[str, int]] = {}
    for bucket in ENGLISH_BUCKETS:
        document_heap: list[tuple[int, str, SampleFingerprint]] = []
        rows_scanned = 0
        eligible_candidates = 0
        for item in selected_reports:
            report = item.report
            if report["bucket"] != bucket:
                continue
            fingerprint = safe_relative_path(
                staging_root, report["fingerprint_file"], "fingerprint_file"
            )
            if file_sha256(fingerprint) != report["fingerprint_sha256"]:
                raise CalibrationError(f"Fingerprint checksum mismatch: {fingerprint}")
            rows = 0
            clean_bytes = tokens = 0
            for expected_index, row in enumerate(iter_jsonl_zst(fingerprint)):
                sample = _validate_sample_fingerprint(row, report, expected_index)
                if sample.word_count >= minimum_source_words:
                    eligible_candidates += 1
                    priority = stable_priority(seed, "real-document", sample.doc_id)
                    entry = (-priority, sample.doc_id, sample)
                    if len(document_heap) < maximum_documents_per_bucket:
                        heapq.heappush(document_heap, entry)
                    elif priority < -document_heap[0][0]:
                        heapq.heapreplace(document_heap, entry)
                rows += 1
                rows_scanned += 1
                clean_bytes += sample.size_bytes
                tokens += sample.starcoder2_tokens
            if (
                rows != report["documents"]
                or clean_bytes != report["clean_bytes"]
                or tokens != report["exact_tokens"]
            ):
                raise CalibrationError(f"Fingerprint totals mismatch: {fingerprint}")
            selected_report_identities.append(
                {
                    "report_path": item.relative_report,
                    "report_sha256": item.report_sha256,
                    "archive": report["archive"],
                    "archive_sha256": report["archive_sha256"],
                    "fingerprint_file": report["fingerprint_file"],
                    "fingerprint_sha256": report["fingerprint_sha256"],
                    "documents": report["documents"],
                }
            )
        selected = [row[2] for row in document_heap]
        if not selected:
            raise CalibrationError(
                f"Selected {bucket} reports contain no document with at least "
                f"{minimum_source_words} words"
            )
        selected.sort(
            key=lambda item: (
                stable_priority(seed, "real-document", item.doc_id), item.doc_id
            )
        )
        selected_fingerprints.extend(selected)
        sampling_counts[bucket] = {
            "fingerprint_rows_scanned": rows_scanned,
            "eligible_fingerprint_candidates": eligible_candidates,
            "documents_selected": len(selected),
        }
    if not selected_fingerprints:
        raise CalibrationError("Selected reports contain no documents")

    fingerprints_by_archive: dict[str, dict[int, SampleFingerprint]] = {}
    for sample in selected_fingerprints:
        fingerprints_by_archive.setdefault(sample.archive, {})[sample.manifest_index] = sample
    reports_by_archive = {
        str(item.report["archive"]): item.report for item in selected_reports
    }
    documents: list[CalibrationDocument] = []
    for archive_identity, wanted in sorted(fingerprints_by_archive.items()):
        report = reports_by_archive[archive_identity]
        raw_path = safe_relative_path(root, archive_identity, "archive")
        if raw_path.stat().st_size != report["archive_compressed_bytes"]:
            raise CalibrationError(f"Raw archive size mismatch: {raw_path}")
        if file_sha256(raw_path) != report["archive_sha256"]:
            raise CalibrationError(f"Raw archive checksum mismatch: {raw_path}")
        document_index = 0
        manifest_seen = False
        manifest_rows = 0
        with raw_path.open("rb") as raw:
            decompressor = zstandard.ZstdDecompressor().stream_reader(
                raw, read_across_frames=True, closefd=False
            )
            try:
                with tarfile.open(fileobj=decompressor, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            raise CalibrationError(f"Unexpected tar member: {member.name}")
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise CalibrationError(f"Cannot read tar member: {member.name}")
                        if member.name == "_manifest.jsonl":
                            if manifest_seen:
                                raise CalibrationError(f"Multiple internal manifests: {raw_path}")
                            manifest_seen = True
                            for manifest_index, line in enumerate(extracted):
                                manifest_rows += 1
                                if manifest_index in wanted:
                                    row = json.loads(line)
                                    sample = wanted[manifest_index]
                                    if (
                                        row.get("member_path") != sample.member_path
                                        or row.get("size_bytes") != sample.size_bytes
                                        or row.get("starcoder2_tokens") != sample.starcoder2_tokens
                                    ):
                                        raise CalibrationError(
                                            f"Internal manifest mismatch: {raw_path}:{manifest_index}"
                                        )
                            continue
                        if manifest_seen:
                            raise CalibrationError(f"Document after manifest: {raw_path}")
                        if document_index in wanted:
                            sample = wanted[document_index]
                            if member.name != sample.member_path or member.size != sample.size_bytes:
                                raise CalibrationError(
                                    f"Raw/fingerprint member mismatch: {raw_path}:{document_index}"
                                )
                            content = extracted.read()
                            if len(content) != sample.size_bytes:
                                raise CalibrationError(f"Short raw document: {sample.member_path}")
                            if hashlib.sha256(content).hexdigest() != sample.content_sha256:
                                raise CalibrationError(f"Raw content hash mismatch: {sample.member_path}")
                            try:
                                text = content.decode("utf-8", errors="strict")
                            except UnicodeDecodeError as exc:
                                raise CalibrationError(
                                    f"Sample document is not UTF-8: {sample.member_path}"
                                ) from exc
                            normalized_sha = hashlib.sha256(
                                normalize_content(text, sample.bucket).encode("utf-8")
                            ).hexdigest()
                            if normalized_sha != sample.normalized_sha256:
                                raise CalibrationError(
                                    f"Raw normalized hash mismatch: {sample.member_path}"
                                )
                            documents.append(
                                CalibrationDocument(
                                    doc_id=sample.doc_id,
                                    bucket=sample.bucket,
                                    text=text,
                                    source_identity=(
                                        f"{sample.bucket}:{sample.archive}:{sample.manifest_index}"
                                    ),
                                    archive=sample.archive,
                                    manifest_index=sample.manifest_index,
                                    member_path=sample.member_path,
                                    content_sha256=sample.content_sha256,
                                    normalized_sha256=sample.normalized_sha256,
                                )
                            )
                        document_index += 1
            finally:
                decompressor.close()
        if (
            not manifest_seen
            or document_index != report["documents"]
            or manifest_rows != report["documents"]
        ):
            raise CalibrationError(
                f"Raw archive completeness mismatch: {raw_path} "
                f"(docs={document_index}, manifest={manifest_rows}, expected={report['documents']})"
            )
    if len(documents) != len(selected_fingerprints):
        raise CalibrationError(
            f"Real sample extraction coverage mismatch: {len(documents)}/{len(selected_fingerprints)}"
        )
    documents.sort(key=lambda item: (stable_priority(seed, "real-document", item.doc_id), item.doc_id))
    selected_report_identities.sort(key=lambda row: row["archive"])
    identity = {
        "kind": "immutable_real_english_sample",
        "full_report_inventory_sha256": identity_builder.report_inventory_sha,
        "preprocess_manifest_sha256": identity_builder.identity["preprocess_manifest_sha256"],
        "curation_policy_sha256": identity_builder.policy_sha,
        "benchmark_guard_sha256": identity_builder.guard.manifest_sha256,
        "source_manifests": identity_builder.source_manifests,
        "collection_completeness_sha256": hashlib.sha256(
            canonical_json_bytes(identity_builder.collection_completeness)
        ).hexdigest(),
        "collection_completeness": identity_builder.collection_completeness,
        "selection_seed": seed,
        "maximum_archives_per_bucket": maximum_archives_per_bucket,
        "maximum_documents_per_bucket": maximum_documents_per_bucket,
        "minimum_source_words": minimum_source_words,
        "sampling_counts": sampling_counts,
        "selected_reports": selected_report_identities,
        "documents_selected": len(documents),
    }
    return documents, identity


def _words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_content(text, "fineweb_edu"))


def prepare_documents(
    documents: Sequence[CalibrationDocument],
    *,
    seed: str,
    minimum_words: int,
    maximum_words: int,
) -> tuple[list[CalibrationDocument], dict[str, int]]:
    eligible: list[CalibrationDocument] = []
    too_short = 0
    truncated = 0
    duplicate_normalized = 0
    seen_normalized: set[str] = set()
    for document in documents:
        words = _words(document.text)
        if len(words) < minimum_words:
            too_short += 1
            continue
        if len(words) > maximum_words:
            words = words[:maximum_words]
            truncated += 1
        text = " ".join(words)
        normalized_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if normalized_sha256 in seen_normalized:
            duplicate_normalized += 1
            continue
        seen_normalized.add(normalized_sha256)
        eligible.append(
            replace(
                document,
                text=text,
                truncated_to_words=len(words),
                normalized_sha256=normalized_sha256,
            )
        )
    eligible.sort(key=lambda item: (stable_priority(seed, "eligible", item.doc_id), item.doc_id))
    return eligible, {
        "too_short": too_short,
        "truncated": truncated,
        "duplicate_normalized": duplicate_normalized,
    }


def _donor_tokens(tokens: Sequence[str], magnitude: int, variant_seed: str) -> list[str]:
    if not tokens:
        tokens = ("additional", "reference", "material", "document", "section")
    result: list[str] = []
    for index in range(magnitude):
        source = tokens[index % len(tokens)]
        # Preserve donor language while ensuring a long appended/replaced span
        # does not collapse to a tiny repeated shingle set.
        cycle = index // len(tokens)
        result.append(source if cycle == 0 else f"{source}calibration{cycle}")
    return result


def apply_perturbation(
    base: Sequence[str],
    donor: Sequence[str],
    *,
    operation: str,
    magnitude: int,
    variant_seed: str,
) -> list[str]:
    if magnitude < 1:
        raise CalibrationError("Perturbation magnitude must be positive")
    if operation == "append_donor_fragment":
        return [*base, *_donor_tokens(donor, magnitude, variant_seed)]
    if operation == "truncate_tail":
        if magnitude >= len(base):
            return []
        return list(base[:-magnitude])
    if operation == "replace_contiguous_with_donor":
        if magnitude >= len(base):
            return _donor_tokens(donor, len(base), variant_seed)
        available = len(base) - magnitude + 1
        anchor = stable_priority(variant_seed, "replace-anchor", str(len(base))) % available
        replacement = _donor_tokens(donor, magnitude, variant_seed)
        return [*base[:anchor], *replacement, *base[anchor + magnitude :]]
    raise CalibrationError(f"Unsupported perturbation operation: {operation}")


def _in_bin(intersection: int, union: int, row: dict[str, Any]) -> bool:
    if union == 0:
        return False
    denominator = int(row["denominator"])
    return (
        intersection * denominator >= union * int(row["minimum_numerator"])
        and intersection * denominator <= union * int(row["maximum_numerator"])
    )


def _distance_to_bin_midpoint(intersection: int, union: int, row: dict[str, Any]) -> float:
    if union == 0:
        return math.inf
    midpoint = (
        int(row["minimum_numerator"]) + int(row["maximum_numerator"])
    ) / (2 * int(row["denominator"]))
    return abs(intersection / union - midpoint)


def find_perturbation(
    *,
    base_tokens: Sequence[str],
    donor_tokens: Sequence[str],
    base_hashes: Sequence[int],
    bin_row: dict[str, Any],
    operations: Sequence[str],
    maximum_evaluations: int,
    production_config: dict[str, Any],
    variant_seed: str,
) -> tuple[str, list[str], int, int, int] | None:
    preferred = stable_priority(variant_seed, "operation", bin_row["name"]) % len(operations)
    ordered_operations = [*operations[preferred:], *operations[:preferred]]
    exact_seed = int(production_config["refinement"]["shingle_hash_seed"])
    midpoint = (
        int(bin_row["minimum_numerator"]) + int(bin_row["maximum_numerator"])
    ) / (2 * int(bin_row["denominator"]))
    best: tuple[float, str, list[str], int, int, int] | None = None
    for operation in ordered_operations:
        if operation == "append_donor_fragment":
            maximum_magnitude = max(2, len(base_tokens))
        elif operation == "truncate_tail":
            maximum_magnitude = max(1, len(base_tokens) - 5)
        else:
            maximum_magnitude = max(1, len(base_tokens) // 2)
        evaluated: dict[int, tuple[list[str], int, int]] = {}

        def evaluate(magnitude: int) -> tuple[list[str], int, int]:
            if magnitude not in evaluated:
                variant = apply_perturbation(
                    base_tokens,
                    donor_tokens,
                    operation=operation,
                    magnitude=magnitude,
                    variant_seed=variant_seed,
                )
                hashes = exact_shingle_hashes(" ".join(variant), exact_seed)
                intersection, union = jaccard_counts(base_hashes, hashes)
                evaluated[magnitude] = (variant, intersection, union)
            return evaluated[magnitude]

        low, high = 1, maximum_magnitude
        while low <= high and len(evaluated) < maximum_evaluations:
            magnitude = (low + high) // 2
            variant, intersection, union = evaluate(magnitude)
            distance = _distance_to_bin_midpoint(intersection, union, bin_row)
            candidate = (distance, operation, variant, magnitude, intersection, union)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            similarity = intersection / union if union else 1.0
            if similarity > midpoint:
                low = magnitude + 1
            else:
                high = magnitude - 1
        center = min(
            evaluated,
            key=lambda magnitude: _distance_to_bin_midpoint(
                evaluated[magnitude][1], evaluated[magnitude][2], bin_row
            ),
        )
        radius = 1
        while len(evaluated) < maximum_evaluations and (
            center - radius >= 1 or center + radius <= maximum_magnitude
        ):
            for magnitude in (center - radius, center + radius):
                if (
                    1 <= magnitude <= maximum_magnitude
                    and magnitude not in evaluated
                    and len(evaluated) < maximum_evaluations
                ):
                    variant, intersection, union = evaluate(magnitude)
                    distance = _distance_to_bin_midpoint(intersection, union, bin_row)
                    candidate = (distance, operation, variant, magnitude, intersection, union)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
            radius += 1
        matching = [
            (magnitude, *value)
            for magnitude, value in evaluated.items()
            if _in_bin(value[1], value[2], bin_row)
        ]
        if matching:
            magnitude, variant, intersection, union = min(
                matching,
                key=lambda item: (
                    _distance_to_bin_midpoint(item[2], item[3], bin_row), item[0]
                ),
            )
            return operation, variant, magnitude, intersection, union
    return None


def one_sided_wilson_interval(
    successes: int, trials: int, confidence_level: float
) -> tuple[float, float]:
    """Return a one-sided Wilson lower bound and the matching upper bound.

    The lower bound is the acceptance statistic.  The upper value is included
    only to make the finite-sample uncertainty visible in the published
    result.  No normal approximation is reported when there are no trials.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise CalibrationError("Invalid binomial counts for Wilson interval")
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(confidence_level)
    observed = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (observed + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            observed * (1.0 - observed) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def calibrate(
    *,
    documents: Sequence[CalibrationDocument],
    input_identity: dict[str, Any],
    production_config_path: Path,
    calibration_config_path: Path,
    output_path: Path,
    minimum_candidate_recall: float | None = None,
    minimum_pairs: int | None = None,
    minimum_documents: int | None = None,
) -> dict[str, Any]:
    checksum_path = output_path.with_name(output_path.name + ".sha256")
    harness_path = Path(__file__).resolve()
    builder_path = PRODUCTION_BUILDER.resolve()
    protected = {
        production_config_path.resolve(),
        calibration_config_path.resolve(),
        harness_path,
        builder_path,
    }
    if output_path.resolve() in protected or checksum_path.resolve() in protected:
        raise CalibrationError("Calibration output would overwrite a pinned input")
    production_config_file_sha = file_sha256(production_config_path)
    calibration_config_file_sha = file_sha256(calibration_config_path)
    harness_sha = file_sha256(harness_path)
    production_builder_sha = file_sha256(builder_path)
    production = load_production_config(production_config_path)
    calibration = load_calibration_config(calibration_config_path)
    input_kind = validate_input_identity(input_identity, len(documents))
    seed = str(calibration["seed"])
    sampling = calibration["sampling"]
    maximum_documents = max(
        int(sampling["maximum_fixture_documents"]),
        len(ENGLISH_BUCKETS) * int(sampling["maximum_documents_per_bucket"]),
    )
    if len(documents) > maximum_documents:
        raise CalibrationError(
            f"Calibration input exceeds pinned bound: {len(documents)} > {maximum_documents}"
        )
    eligible, exclusions = prepare_documents(
        documents,
        seed=seed,
        minimum_words=int(sampling["minimum_source_words"]),
        maximum_words=int(sampling["maximum_source_words"]),
    )
    acceptance = dict(calibration["acceptance"])
    acceptance_overrides: dict[str, int | float] = {}
    if minimum_candidate_recall is not None:
        if not 0 <= minimum_candidate_recall <= 1:
            raise CalibrationError("CLI minimum candidate recall must be in [0,1]")
        acceptance["minimum_candidate_recall"] = minimum_candidate_recall
        acceptance_overrides["minimum_candidate_recall"] = minimum_candidate_recall
    if minimum_pairs is not None:
        if minimum_pairs < 0:
            raise CalibrationError("CLI minimum pairs cannot be negative")
        acceptance["minimum_pairs_at_or_above_production_threshold"] = minimum_pairs
        acceptance_overrides[
            "minimum_pairs_at_or_above_production_threshold"
        ] = minimum_pairs
    if minimum_documents is not None:
        if minimum_documents < 0:
            raise CalibrationError("CLI minimum documents cannot be negative")
        acceptance["minimum_eligible_documents"] = minimum_documents
        acceptance_overrides["minimum_eligible_documents"] = minimum_documents

    exact_seed = int(production["refinement"]["shingle_hash_seed"])
    base_state: dict[str, tuple[list[str], list[int], set[tuple[int, bytes]]]] = {}
    for document in eligible:
        tokens = document.text.split()
        hashes = exact_shingle_hashes(document.text, exact_seed)
        bands = set(candidate_bands(document.text, production)[0])
        base_state[document.doc_id] = (tokens, hashes, bands)

    pairs: list[dict[str, Any]] = []
    generation_failures: list[dict[str, str]] = []
    bins = calibration["perturbations"]["bins"]
    operations = list(calibration["perturbations"]["operations"])
    variants_per_bin = int(
        calibration["perturbations"]["variants_per_document_per_bin"]
    )
    maximum_evaluations = int(
        calibration["perturbations"]["maximum_search_evaluations_per_operation"]
    )
    for document_index, document in enumerate(eligible):
        base_tokens, base_hashes, base_bands = base_state[document.doc_id]
        donors = [
            candidate
            for candidate in eligible[document_index + 1 :] + eligible[:document_index]
            if candidate.normalized_sha256 != document.normalized_sha256
        ]
        donor = donors[0] if donors else document
        donor_tokens = donor.text.split()
        for bin_row in bins:
            for variant_index in range(variants_per_bin):
                variant_seed = hashlib.sha256(
                    f"{seed}\0{document.doc_id}\0{bin_row['name']}\0{variant_index}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                found = find_perturbation(
                    base_tokens=base_tokens,
                    donor_tokens=donor_tokens,
                    base_hashes=base_hashes,
                    bin_row=bin_row,
                    operations=operations,
                    maximum_evaluations=maximum_evaluations,
                    production_config=production,
                    variant_seed=variant_seed,
                )
                if found is None:
                    generation_failures.append(
                        {
                            "doc_id": document.doc_id,
                            "bin": str(bin_row["name"]),
                            "variant_index": str(variant_index),
                        }
                    )
                    continue
                operation, variant_tokens, magnitude, intersection, union = found
                variant_text = " ".join(variant_tokens)
                variant_bands = set(candidate_bands(variant_text, production)[0])
                shared_bands = len(base_bands & variant_bands)
                accepted = refinement_accepts(intersection, union, production)
                production_numerator = int(
                    production["refinement"]["minimum_jaccard_numerator"]
                )
                production_denominator = int(
                    production["refinement"]["minimum_jaccard_denominator"]
                )
                independently_expected = (
                    union == 0
                    or intersection * production_denominator
                    >= union * production_numerator
                )
                pairs.append(
                    {
                        "base_doc_id": document.doc_id,
                        "base_bucket": document.bucket,
                        "bin": bin_row["name"],
                        "variant_index": variant_index,
                        "variant_id": variant_seed,
                        "variant_normalized_sha256": hashlib.sha256(
                            variant_text.encode("utf-8")
                        ).hexdigest(),
                        "operation": operation,
                        "edit_words": magnitude,
                        "base_words": len(base_tokens),
                        "variant_words": len(variant_tokens),
                        "intersection_shingles": intersection,
                        "union_shingles": union,
                        "jaccard": round(intersection / union if union else 1.0, 9),
                        "candidate_hit": bool(shared_bands),
                        "shared_bands": shared_bands,
                        "exact_refinement_accepted": accepted,
                        "exact_refinement_expected": independently_expected,
                    }
                )

    production_numerator = int(production["refinement"]["minimum_jaccard_numerator"])
    production_denominator = int(
        production["refinement"]["minimum_jaccard_denominator"]
    )
    controls: list[dict[str, Any]] = []
    used_control_pairs: set[tuple[str, str]] = set()
    if len(eligible) > 1:
        for left_document in eligible:
            _left_tokens, left_hashes, left_bands = base_state[left_document.doc_id]
            choices: list[tuple[float, int, str, CalibrationDocument, int, int]] = []
            for right_document in eligible:
                if right_document.doc_id == left_document.doc_id:
                    continue
                pair_id = tuple(sorted((left_document.doc_id, right_document.doc_id)))
                if pair_id in used_control_pairs:
                    continue
                _right_tokens, right_hashes, _right_bands = base_state[
                    right_document.doc_id
                ]
                intersection, union = jaccard_counts(left_hashes, right_hashes)
                if (
                    union == 0
                    or intersection * production_denominator
                    >= union * production_numerator
                ):
                    continue
                choices.append(
                    (
                        intersection / union,
                        stable_priority(
                            seed,
                            "negative-control",
                            f"{left_document.doc_id}\0{right_document.doc_id}",
                        ),
                        right_document.doc_id,
                        right_document,
                        intersection,
                        union,
                    )
                )
            if not choices:
                continue
            (
                _similarity,
                _priority,
                _right_id,
                right_document,
                intersection,
                union,
            ) = min(choices, key=lambda item: item[:3])
            pair_id = tuple(sorted((left_document.doc_id, right_document.doc_id)))
            used_control_pairs.add(pair_id)
            _right_tokens, _right_hashes, right_bands = base_state[right_document.doc_id]
            accepted = refinement_accepts(intersection, union, production)
            controls.append(
                {
                    "left_doc_id": left_document.doc_id,
                    "right_doc_id": right_document.doc_id,
                    "intersection_shingles": intersection,
                    "union_shingles": union,
                    "jaccard": round(intersection / union if union else 1.0, 9),
                    "candidate_hit": bool(left_bands & right_bands),
                    "shared_bands": len(left_bands & right_bands),
                    "exact_refinement_accepted": accepted,
                    "exact_refinement_expected": False,
                }
            )

    bin_stats: list[dict[str, Any]] = []
    for bin_row in bins:
        rows = [row for row in pairs if row["bin"] == bin_row["name"]]
        hits = sum(bool(row["candidate_hit"]) for row in rows)
        accepted = sum(bool(row["exact_refinement_accepted"]) for row in rows)
        bin_stats.append(
            {
                "name": bin_row["name"],
                "bounds": {
                    "minimum_numerator": bin_row["minimum_numerator"],
                    "maximum_numerator": bin_row["maximum_numerator"],
                    "denominator": bin_row["denominator"],
                },
                "generated_pairs": len(rows),
                "generation_failures": sum(
                    failure["bin"] == bin_row["name"] for failure in generation_failures
                ),
                "candidate_hits": hits,
                "candidate_recall": round(hits / len(rows), 9) if rows else None,
                "exact_refinement_accepts": accepted,
                "minimum_actual_jaccard": min(
                    (float(row["jaccard"]) for row in rows), default=None
                ),
                "maximum_actual_jaccard": max(
                    (float(row["jaccard"]) for row in rows), default=None
                ),
            }
        )

    target_pairs = [
        row for row in pairs if bool(row["exact_refinement_expected"])
    ]
    target_hits = sum(bool(row["candidate_hit"]) for row in target_pairs)
    target_recall = target_hits / len(target_pairs) if target_pairs else 0.0
    recall_lower, recall_upper = one_sided_wilson_interval(
        target_hits,
        len(target_pairs),
        float(acceptance["recall_confidence_level"]),
    )
    exact_errors = sum(
        row["exact_refinement_accepted"] != row["exact_refinement_expected"]
        for row in [*pairs, *controls]
    )
    failures: list[str] = []
    if len(eligible) < int(acceptance["minimum_eligible_documents"]):
        failures.append(
            f"eligible_documents={len(eligible)} < {acceptance['minimum_eligible_documents']}"
        )
    if len(target_pairs) < int(
        acceptance["minimum_pairs_at_or_above_production_threshold"]
    ):
        failures.append(
            "pairs_at_or_above_threshold="
            f"{len(target_pairs)} < "
            f"{acceptance['minimum_pairs_at_or_above_production_threshold']}"
        )
    if recall_lower < float(acceptance["minimum_candidate_recall"]):
        failures.append(
            f"candidate_recall_wilson_lower_bound={recall_lower:.9f} < "
            f"{float(acceptance['minimum_candidate_recall']):.9f}"
        )
    if exact_errors > int(acceptance["maximum_exact_refinement_decision_errors"]):
        failures.append(
            f"exact_refinement_decision_errors={exact_errors} > "
            f"{acceptance['maximum_exact_refinement_decision_errors']}"
        )
    if len(generation_failures) > int(acceptance["maximum_generation_failures"]):
        failures.append(
            f"generation_failures={len(generation_failures)} > "
            f"{acceptance['maximum_generation_failures']}"
        )
    if acceptance["require_every_bin"]:
        missing_bins = [row["name"] for row in bin_stats if not row["generated_pairs"]]
        if missing_bins:
            failures.append(f"missing_perturbation_bins={','.join(missing_bins)}")

    sample_manifest = [
        {
            "doc_id": document.doc_id,
            "bucket": document.bucket,
            "source_identity": document.source_identity,
            "content_sha256": document.content_sha256,
            "normalized_sha256": document.normalized_sha256,
            "calibration_words": document.truncated_to_words,
            "archive": document.archive,
            "manifest_index": document.manifest_index,
            "member_path": document.member_path,
        }
        for document in eligible
    ]
    if file_sha256(production_config_path) != production_config_file_sha:
        raise CalibrationError("Production config changed during calibration")
    if file_sha256(calibration_config_path) != calibration_config_file_sha:
        raise CalibrationError("Calibration config changed during calibration")
    if file_sha256(harness_path) != harness_sha:
        raise CalibrationError("Calibration harness changed during calibration")
    if file_sha256(builder_path) != production_builder_sha:
        raise CalibrationError("Production builder changed during calibration")

    sampling_matches_pinned = (
        input_kind == "immutable_real_english_sample"
        and input_identity.get("maximum_archives_per_bucket")
        == sampling["maximum_archives_per_bucket"]
        and input_identity.get("maximum_documents_per_bucket")
        == sampling["maximum_documents_per_bucket"]
        and input_identity.get("minimum_source_words")
        == sampling["minimum_source_words"]
    )
    noneligibility_reasons: list[str] = []
    if failures:
        noneligibility_reasons.append("acceptance_failed")
    if acceptance_overrides:
        noneligibility_reasons.append("cli_acceptance_override")
    if input_kind != "immutable_real_english_sample":
        noneligibility_reasons.append("fixture_input")
    elif not sampling_matches_pinned:
        noneligibility_reasons.append("non_pinned_sampling_bounds")
    production_gate_eligible = not noneligibility_reasons
    result: dict[str, Any] = {
        "result_version": 1,
        "status": "pass" if not failures else "fail",
        "production_configuration_unchanged": True,
        "production_gate_eligible": production_gate_eligible,
        "production_gate_noneligibility_reasons": noneligibility_reasons,
        "identity": {
            "harness_sha256": harness_sha,
            "production_builder_sha256": production_builder_sha,
            "calibration_algorithm": calibration["calibration_algorithm"],
            "calibration_seed": calibration["seed"],
            "production_config_file_sha256": production_config_file_sha,
            "production_config_canonical_sha256": canonical_sha256(
                {key: value for key, value in production.items() if not key.startswith("_")}
            ),
            "calibration_config_file_sha256": calibration_config_file_sha,
            "calibration_config_canonical_sha256": canonical_sha256(calibration),
            "input": input_identity,
            "sample_manifest_sha256": hashlib.sha256(
                canonical_json_bytes(sample_manifest)
            ).hexdigest(),
            "runtime": {
                "python": sys.version.split()[0],
                "xxhash": str(
                    getattr(xxhash, "__version__", getattr(xxhash, "VERSION", "unknown"))
                ),
                "zstandard": zstandard.__version__,
            },
        },
        "production_threshold": {
            "minimum_jaccard_numerator": production["refinement"][
                "minimum_jaccard_numerator"
            ],
            "minimum_jaccard_denominator": production["refinement"][
                "minimum_jaccard_denominator"
            ],
        },
        "acceptance": acceptance,
        "acceptance_profile": (
            "pinned-production" if not acceptance_overrides else "cli-override-non-production"
        ),
        "acceptance_overrides": acceptance_overrides,
        "sampling_profile": (
            "pinned-production" if sampling_matches_pinned else "non-production"
        ),
        "acceptance_failures": failures,
        "sampling": {
            "documents_input": len(documents),
            "eligible_documents": len(eligible),
            "excluded_too_short": exclusions["too_short"],
            "excluded_duplicate_normalized": exclusions["duplicate_normalized"],
            "truncated_to_bound": exclusions["truncated"],
            "sample_manifest": sample_manifest,
        },
        "summary": {
            "perturbation_pairs": len(pairs),
            "generation_failures": len(generation_failures),
            "pairs_at_or_above_production_threshold": len(target_pairs),
            "candidate_hits_at_or_above_threshold": target_hits,
            "candidate_recall_at_or_above_threshold": round(target_recall, 9),
            "candidate_recall_one_sided_wilson_interval": {
                "confidence_level": acceptance["recall_confidence_level"],
                "lower": round(recall_lower, 9),
                "upper": round(recall_upper, 9),
            },
            "exact_refinement_decision_errors": exact_errors,
            "unrelated_control_pairs": len(controls),
            "unrelated_control_candidate_hits": sum(
                bool(row["candidate_hit"]) for row in controls
            ),
            "unrelated_control_exact_accepts": sum(
                bool(row["exact_refinement_accepted"]) for row in controls
            ),
        },
        "bins": bin_stats,
        "pairs": pairs,
        "generation_failure_details": generation_failures,
        "unrelated_controls": controls,
        "statistical_scope": (
            "This bounded synthetic-perturbation calibration measures retrieval on the "
            "selected inputs and edit families only. Passing does not prove corpus-wide recall "
            "or justify silently changing the pinned production configuration."
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_bytes(output_path, payload)
    atomic_bytes(
        checksum_path,
        hashlib.sha256(payload).hexdigest().encode("ascii")
        + f"  {output_path.name}\n".encode("utf-8"),
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, default=DEFAULT_PRODUCTION_CONFIG)
    parser.add_argument("--calibration-config", type=Path, default=DEFAULT_CALIBRATION_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_QUOTA_CONFIG)
    parser.add_argument("--maximum-archives-per-bucket", type=int)
    parser.add_argument("--maximum-documents-per-bucket", type=int)
    parser.add_argument("--maximum-fixture-documents", type=int)
    parser.add_argument("--minimum-candidate-recall", type=float)
    parser.add_argument("--minimum-pairs", type=int)
    parser.add_argument("--minimum-documents", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        calibration_config = load_calibration_config(args.calibration_config)
        seed = str(calibration_config["seed"])
        sampling = calibration_config["sampling"]
        maximum_fixture_documents = (
            args.maximum_fixture_documents
            if args.maximum_fixture_documents is not None
            else int(sampling["maximum_fixture_documents"])
        )
        if not 1 <= maximum_fixture_documents <= int(sampling["maximum_fixture_documents"]):
            raise CalibrationError(
                "maximum fixture documents must be positive and no larger than "
                "the pinned calibration bound"
            )
        if args.input_jsonl is not None:
            if args.input_jsonl.resolve() in {
                args.output.resolve(),
                args.output.with_name(args.output.name + ".sha256").resolve(),
            }:
                raise CalibrationError("Calibration output cannot overwrite its fixture")
            documents, input_identity = load_fixture_documents(
                args.input_jsonl,
                seed=seed,
                maximum_documents=maximum_fixture_documents,
            )
        else:
            maximum_archives = (
                args.maximum_archives_per_bucket
                if args.maximum_archives_per_bucket is not None
                else int(sampling["maximum_archives_per_bucket"])
            )
            maximum_documents = (
                args.maximum_documents_per_bucket
                if args.maximum_documents_per_bucket is not None
                else int(sampling["maximum_documents_per_bucket"])
            )
            if maximum_archives < 1 or maximum_documents < 1:
                raise CalibrationError("real sample bounds must be positive")
            staging = args.staging_root or args.root / "staging" / "preprocess"
            documents, input_identity = sample_real_documents(
                root=args.root,
                staging_root=staging,
                production_config_path=args.production_config,
                policy_path=args.policy,
                denylist_path=args.denylist,
                quota_config_path=args.quota_config,
                seed=seed,
                maximum_archives_per_bucket=maximum_archives,
                maximum_documents_per_bucket=maximum_documents,
                minimum_source_words=int(sampling["minimum_source_words"]),
                identity_output_hint=args.output.parent / ".identity-only-not-created",
            )
        result = calibrate(
            documents=documents,
            input_identity=input_identity,
            production_config_path=args.production_config,
            calibration_config_path=args.calibration_config,
            output_path=args.output,
            minimum_candidate_recall=args.minimum_candidate_recall,
            minimum_pairs=args.minimum_pairs,
            minimum_documents=args.minimum_documents,
        )
    except (
        CalibrationError,
        EnglishNearDedupError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"English near-dedup calibration failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "production_gate_eligible": result["production_gate_eligible"],
                "production_gate_noneligibility_reasons": result[
                    "production_gate_noneligibility_reasons"
                ],
                "output": str(args.output),
                "candidate_recall_at_or_above_threshold": result["summary"][
                    "candidate_recall_at_or_above_threshold"
                ],
                "candidate_recall_wilson_lower_bound": result["summary"][
                    "candidate_recall_one_sided_wilson_interval"
                ]["lower"],
                "pairs_at_or_above_threshold": result["summary"][
                    "pairs_at_or_above_production_threshold"
                ],
                "acceptance_failures": result["acceptance_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
