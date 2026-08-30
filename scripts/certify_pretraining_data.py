#!/usr/bin/env python3
"""Fully validate one local packed order and publish immutable launch evidence.

Unlike the production launcher's fast metadata preflight, this command reads
and checksums every referenced order/token/start payload and performs the full
semantic validators. Run it once per final local train and validation copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for import_root in (str(PROJECT_ROOT), str(SCRIPTS_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import launch_pretraining as launch  # noqa: E402
from pretrain import data as training_data  # noqa: E402


EVIDENCE_FORMAT = "production-pretraining-full-data-validation"
EVIDENCE_VERSION = 1


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _source_identity() -> dict[str, Any]:
    import numpy as np

    data_source = Path(training_data.__file__).resolve(strict=True)
    launcher_source = Path(launch.__file__).resolve(strict=True)
    certifier_source = Path(__file__).resolve(strict=True)
    return {
        "evidence_contract_version": EVIDENCE_VERSION,
        **launch._interpreter_identity(),
        "numpy_version": np.__version__,
        "order_format_version": training_data.ORDER_FORMAT_VERSION,
        "packed_format_version": training_data.FORMAT_VERSION,
        "pretrain_data_path": str(data_source),
        "pretrain_data_sha256": _sha256(data_source),
        "launcher_path": str(launcher_source),
        "launcher_sha256": _sha256(launcher_source),
        "certifier_path": str(certifier_source),
        "certifier_sha256": _sha256(certifier_source),
    }


def certify_order(
    *,
    order_manifest: Path,
    expected_split: str,
    local_data_root: Path,
    global_microbatch_rows: int | None,
) -> dict[str, Any]:
    source_before = _source_identity()
    before = launch.inspect_order(
        order_manifest,
        expected_split=expected_split,
        local_data_root=local_data_root,
        global_microbatch_rows=global_microbatch_rows,
    )
    validated_order = training_data.validate_training_order(before.path)
    packed_results: dict[str, dict[str, Any]] = {}
    for domain in training_data.DOMAIN_ORDER:
        packed_path = Path(before.packed_manifest_paths[domain])
        packed_results[domain] = training_data.validate_packed_manifest(
            packed_path, verify_checksums=True
        )
    after = launch.inspect_order(
        order_manifest,
        expected_split=expected_split,
        local_data_root=local_data_root,
        global_microbatch_rows=global_microbatch_rows,
    )
    if before.metadata_inventory != after.metadata_inventory:
        raise launch.PreflightError(
            "Local order or packed-file identity changed during full validation"
        )
    if before.metadata_inventory_sha256 != after.metadata_inventory_sha256:
        raise launch.PreflightError("Metadata inventory digest changed during validation")
    source_after = _source_identity()
    if source_before != source_after:
        raise launch.PreflightError(
            "Validator implementation or runtime identity changed during validation"
        )
    return {
        "format": EVIDENCE_FORMAT,
        "format_version": EVIDENCE_VERSION,
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": expected_split,
        "order_manifest": before.path,
        "metadata_inventory": before.metadata_inventory,
        "metadata_inventory_sha256": before.metadata_inventory_sha256,
        "validator": source_after,
        "checks": {
            "order_payload_checksum": True,
            "order_reference_semantics": True,
            "packed_manifest_identities": True,
            "packed_payload_checksums": True,
            "packed_payload_semantics": True,
            "stable_file_identity_during_validation": True,
        },
        "summary": {
            "order_rows": validated_order["rows"],
            "order_payload_sha256": validated_order["order"]["sha256"],
            "packed_payload_files": before.packed_payload_files,
            "packed_payload_bytes": before.packed_payload_bytes,
            "domains": {
                domain: {
                    "rows": packed_results[domain]["rows"],
                    "input_tokens": packed_results[domain]["input_tokens"],
                    "valid_loss_tokens": packed_results[domain]["valid_loss_tokens"],
                    "masked_boundary_labels": packed_results[domain][
                        "masked_boundary_labels"
                    ],
                }
                for domain in training_data.DOMAIN_ORDER
            },
        },
    }


def publish_evidence(path: Path, evidence: Mapping[str, Any]) -> tuple[Path, Path]:
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    sidecar = destination.with_name(f"{destination.name}.sha256")
    if (
        destination.exists()
        or destination.is_symlink()
        or sidecar.exists()
        or sidecar.is_symlink()
    ):
        raise launch.PreflightError(
            f"Refusing to overwrite full-validation evidence or sidecar: {destination}"
        )
    payload = json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_bytes(destination, payload)
    _atomic_bytes(sidecar, f"{digest}  {destination.name}\n".encode("ascii"))
    return destination, sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_manifest", type=Path)
    parser.add_argument(
        "--expected-split", choices=("train", "validation"), required=True
    )
    parser.add_argument("--local-data-root", type=Path, required=True)
    parser.add_argument(
        "--global-microbatch-rows",
        type=int,
        help="required for the intentionally unfrozen validation order",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.expected_split == "train" and args.global_microbatch_rows is not None:
            raise launch.PreflightError(
                "--global-microbatch-rows is only valid for held-out validation evidence"
            )
        if (
            args.expected_split == "validation"
            and args.global_microbatch_rows is None
        ):
            raise launch.PreflightError(
                "--global-microbatch-rows is required for validation evidence"
            )
        output_parent = args.output.parent.resolve(strict=True)
        local_root = args.local_data_root.resolve(strict=True)
        if output_parent == local_root or output_parent.is_relative_to(local_root):
            raise launch.PreflightError(
                "Full-validation evidence must not be written into the immutable local data root"
            )
        evidence = certify_order(
            order_manifest=args.order_manifest,
            expected_split=args.expected_split,
            local_data_root=args.local_data_root,
            global_microbatch_rows=args.global_microbatch_rows,
        )
        destination, sidecar = publish_evidence(args.output, evidence)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "evidence": str(destination),
                    "sidecar": str(sidecar),
                    "metadata_inventory_sha256": evidence[
                        "metadata_inventory_sha256"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileNotFoundError,
        IOError,
        KeyError,
        TypeError,
        ValueError,
        launch.PreflightError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
