"""Fail-closed acceptance of geometry when only the final-order soak is waived.

The waiver is deliberately narrower than the normal geometry qualification.  It
does not waive the one-GPU baselines, the six-GPU grid, production hardware,
data identity, checkpoint/resume, validation, or telemetry gates.  It only
permits the selected grid measurement to stand in for a second run over the
final 52.58B-token order.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from pretrain import data as training_data
from pretrain import geometry_qualification as qualification


WAIVER_FORMAT = "pretraining-accepted-geometry-with-waiver"
WAIVER_VERSION = 1
WAIVER_SCOPE = "separate-final-order-soak-only"
REQUIRED_SAFEGUARDS = (
    "durable-checkpoint-rotation-and-resume-readiness",
    "validation-at-step-zero-and-frozen-cadence",
    "online-wandb-from-step-one",
    "attended-opening-through-first-validation-and-checkpoint",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_GRID_SOURCE_KEYS = frozenset(
    {
        "qualification_module",
        "qualification_cli",
        "trainer",
        "model",
        "data",
        "tokenizer_identity",
        "production_launcher",
    }
)
_PRODUCER_SOURCE_KEYS = frozenset(
    {"waiver_module", "waiver_cli", "geometry_evidence", "run_authority"}
)
_WANDB_METRIC_MINIMUMS = {
    "train/": 50,
    "perf/": 8,
    "system/": 8,
    "checkpoint/": 5,
}


class GeometryWaiverError(ValueError):
    """The separate-final-soak waiver cannot be proven from durable evidence."""


def _fail(message: str) -> None:
    raise GeometryWaiverError(message)


def _plain_int(value: Any, *, minimum: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, *, field: str, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        _fail(f"{field} must be a decimal value")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise GeometryWaiverError(f"{field} is not a decimal value") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        _fail(f"{field} must be finite and {qualifier}")
    return parsed


def _descriptor_content(descriptor: Mapping[str, Any], *, label: str) -> tuple[int, str]:
    try:
        byte_count = descriptor["bytes"]
        digest = descriptor["sha256"]
    except KeyError as exc:
        raise GeometryWaiverError(f"{label} descriptor is incomplete") from exc
    _plain_int(byte_count, minimum=1, field=f"{label}.bytes")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _fail(f"{label}.sha256 is invalid")
    return byte_count, digest


def _same_content(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    if _descriptor_content(left, label=f"{label}.left") != _descriptor_content(
        right, label=f"{label}.right"
    ):
        _fail(f"{label} content differs")


def _read_json_artifact(
    descriptor: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    try:
        return qualification.read_bound_json(descriptor, label=label)
    except (OSError, ValueError) as exc:
        raise GeometryWaiverError(f"Invalid {label}: {exc}") from exc


def _current_sources(waiver_cli: Path) -> dict[str, dict[str, Any]]:
    modules = {
        "waiver_module": "pretrain.geometry_waiver",
        "geometry_evidence": "pretrain.geometry_evidence",
        "run_authority": "pretrain.run_authority",
    }
    records: dict[str, dict[str, Any]] = {
        "waiver_cli": qualification.artifact(waiver_cli, label="geometry waiver CLI")
    }
    for key, module_name in modules.items():
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            _fail(f"Cannot locate {module_name} source")
        records[key] = qualification.artifact(Path(source), label=f"{key} source")
    return records


def _validate_current_sources(sources: Mapping[str, Any]) -> None:
    if set(sources) != _PRODUCER_SOURCE_KEYS:
        _fail("Waiver producer source inventory is incomplete")
    expected = _current_sources(Path(str(sources["waiver_cli"]["path"])))
    for key in sorted(_PRODUCER_SOURCE_KEYS):
        if sources.get(key) != expected[key]:
            _fail(f"Current {key} source differs from the waiver receipt")


def _validate_grid_runtime_compatibility(grid_plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = grid_plan.get("runtime")
    if not isinstance(runtime, Mapping):
        _fail("Grid plan lacks runtime identity")
    sources = runtime.get("sources")
    python = runtime.get("python")
    if not isinstance(sources, Mapping) or not isinstance(python, Mapping):
        _fail("Grid runtime identity is incomplete")
    if not _STABLE_GRID_SOURCE_KEYS.issubset(sources):
        _fail("Grid plan lacks stable source identities")

    from scripts import launch_pretraining, qualify_training_geometry
    from pretrain import data, model, tokenizer_identity, train

    current_paths = {
        "qualification_module": Path(qualification.__file__),
        "qualification_cli": Path(qualify_training_geometry.__file__),
        "trainer": Path(train.__file__),
        "model": Path(model.__file__),
        "data": Path(data.__file__),
        "tokenizer_identity": Path(tokenizer_identity.__file__),
        "production_launcher": Path(launch_pretraining.__file__),
    }
    compatible: dict[str, Any] = {}
    for key, path in current_paths.items():
        frozen = sources.get(key)
        if not isinstance(frozen, Mapping):
            _fail(f"Grid plan source {key} is invalid")
        current = qualification.artifact(path, label=f"current {key}")
        _same_content(frozen, current, label=f"frozen/current {key}")
        compatible[key] = {
            "grid": dict(frozen),
            "current": current,
            "content_identical": True,
        }

    resolved = python.get("resolved")
    if not isinstance(resolved, Mapping):
        _fail("Grid Python identity lacks resolved executable")
    current_python = qualification.artifact(
        Path(sys.executable).resolve(strict=True), label="current Python"
    )
    _same_content(resolved, current_python, label="frozen/current Python executable")
    if python.get("version") != ".".join(str(v) for v in sys.version_info[:3]):
        _fail("Current Python version differs from the grid")
    if python.get("implementation") != sys.implementation.name:
        _fail("Current Python implementation differs from the grid")
    return {
        "stable_sources": compatible,
        "python": {
            "grid": dict(python),
            "current": current_python,
            "content_identical": True,
        },
        "allowed_newer_sources": ["geometry_evidence", "run_authority"],
    }


def _hardware_compatibility(
    *, final_identity: Mapping[str, Any], grid_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare hardware identities while allowing only the newer Git object IDs.

    The qualification script, requirements, package lock, GPU topology and every
    runtime field remain exact.  Git commit/tree/archive necessarily change when
    only the waiver authority/evidence code is added after a frozen grid checkout.
    The strict final hardware receipt is still checked against current clean Git by
    the normal run-authority path.
    """

    final = json.loads(json.dumps(dict(final_identity)))
    frozen = json.loads(json.dumps(dict(grid_identity)))
    if final.pop("scope", None) != "final-launch-authorizing" or frozen.pop(
        "scope", None
    ) != "geometry-only-provisional":
        _fail("Hardware receipt scopes are not strict-final versus geometry-only-grid")
    final_digest = final.pop("identity_sha256", None)
    frozen_digest = frozen.pop("identity_sha256", None)
    if (
        not isinstance(final_digest, str)
        or _SHA256.fullmatch(final_digest) is None
        or not isinstance(frozen_digest, str)
        or _SHA256.fullmatch(frozen_digest) is None
    ):
        _fail("Hardware geometry identity digest is invalid")
    try:
        final_source = final["identity"]["source"]
        frozen_source = frozen["identity"]["source"]
    except (KeyError, TypeError) as exc:
        raise GeometryWaiverError("Hardware geometry identities lack source records") from exc
    allowed_git_differences = ("git_commit", "git_tree", "git_head_archive_sha256")
    for field in allowed_git_differences:
        final_source.pop(field, None)
        frozen_source.pop(field, None)
    if final_source.get("git_clean") is not True or frozen_source.get("git_clean") is not True:
        _fail("Both grid and final hardware identities must come from clean Git trees")
    try:
        final_lock = final["identity"]["package_lock"]
        frozen_lock = frozen["identity"]["package_lock"]
        final_lock["lock"].pop("path", None)
        frozen_lock["lock"].pop("path", None)
        final_lock["python"].pop("executable", None)
        frozen_lock["python"].pop("executable", None)
    except (KeyError, TypeError) as exc:
        raise GeometryWaiverError(
            "Hardware geometry identities lack package/runtime content identity"
        ) from exc
    if final != frozen:
        _fail(
            "Strict final hardware differs from grid hardware beyond permitted Git object IDs"
        )
    allowed_differences = [
        *(f"source.{field}" for field in allowed_git_differences),
        "package_lock.lock.path",
        "package_lock.python.executable",
    ]
    return {
        "status": "compatible",
        "comparison": "exact-except-git-commit-tree-and-head-archive",
        "stable_identity_sha256": qualification.canonical_sha256(final),
        "allowed_differences": allowed_differences,
        "grid_identity_sha256": frozen_digest,
        "final_identity_sha256": final_digest,
    }


def _inspect_order(path: Path, *, split: str) -> dict[str, Any]:
    try:
        manifest, payload_path = training_data._load_order_manifest(  # type: ignore[attr-defined]
            path, verify_checksum=True
        )
    except (OSError, ValueError) as exc:
        raise GeometryWaiverError(f"Invalid {split} order: {exc}") from exc
    if manifest.get("split") != split:
        _fail(f"Expected {split} order, found {manifest.get('split')!r}")
    record: dict[str, Any] = {
        "manifest": qualification.artifact(path, label=f"{split} order manifest"),
        "payload": qualification.artifact(payload_path, label=f"{split} order payload"),
        "sequence_length": manifest.get("sequence_length"),
        "vocab_size": manifest.get("vocab_size"),
        "eos_token_id": manifest.get("eos_token_id"),
        "tokenizer_manifest_sha256": manifest.get("tokenizer_manifest_sha256"),
        "rows": manifest.get("rows"),
    }
    if split == "train":
        record["geometry"] = training_data.frozen_training_geometry(
            path, verify_checksum=True
        )
    return record


def _validate_orders(
    *,
    train_path: Path,
    validation_path: Path,
    grid_plan: Mapping[str, Any],
    accepted: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = _inspect_order(train_path, split="train")
    validation = _inspect_order(validation_path, split="validation")
    common = grid_plan["inputs"]["common"]
    for field in ("sequence_length", "vocab_size", "eos_token_id"):
        if train[field] != common[field] or validation[field] != common[field]:
            _fail(f"Final orders differ from grid data contract on {field}")
    tokenizer_sha = common["tokenizer"]["manifest_sha256"]
    if (
        train["tokenizer_manifest_sha256"] != tokenizer_sha
        or validation["tokenizer_manifest_sha256"] != tokenizer_sha
    ):
        _fail("Final orders differ from the grid tokenizer")

    geometry = train["geometry"]
    if (
        geometry.get("global_microbatch_rows")
        != accepted.get("global_microbatch_rows")
        or geometry.get("gradient_accumulation_steps")
        != accepted.get("gradient_accumulation_steps")
        or geometry.get("optimizer_update_rows") != qualification.UPDATE_ROWS
        or geometry.get("optimizer_updates")
        != qualification.FINAL_TRAIN_EXPECTED_OPTIMIZER_UPDATES
        or geometry.get("consumed_rows") != qualification.FINAL_TRAIN_EXPECTED_ROWS
        or geometry.get("consumed_input_tokens")
        != qualification.FINAL_TRAIN_EXPECTED_CONSUMED_INPUT_TOKENS
        or geometry.get("dropped_rows") != 0
    ):
        _fail("Final train order is not the exact selected 52.58B geometry")

    train_payload = _read_json_artifact(train["manifest"], label="train manifest")
    if (
        train_payload.get("expected_input_token_weights")
        != {"python": 0.4, "other_code": 0.4, "english": 0.2}
        or train_payload.get("input_token_budget", {}).get("expected_total")
        != qualification.FINAL_TRAIN_TARGET_INPUT_TOKENS
    ):
        _fail("Final train order is not the approved 40/40/20 target")
    for domain in training_data.DOMAIN_ORDER:
        domain_record = train_payload.get("dataset_manifests", {}).get(domain)
        if not isinstance(domain_record, Mapping) or not isinstance(
            domain_record.get("path"), str
        ):
            _fail(f"Final train order lacks {domain} packed-manifest binding")
        current = qualification.artifact(
            train_path.parent / domain_record["path"],
            label=f"final train {domain} packed manifest",
        )
        frozen = grid_plan["inputs"]["packed"][domain]["manifest"]
        _same_content(current, frozen, label=f"final/grid {domain} packed manifest")

    frozen_validation = grid_plan["inputs"]["validation_order"]["manifest"]
    _same_content(
        validation["manifest"], frozen_validation, label="final/grid validation manifest"
    )
    validation["geometry"] = training_data.evaluation_order_geometry(
        validation_path,
        global_microbatch_rows=int(accepted["global_microbatch_rows"]),
        verify_checksum=True,
    )
    if validation["geometry"]["available_global_microbatches"] < int(
        grid_plan["settings"]["eval_batches"]
    ):
        _fail("Validation order is too short for the qualified evaluation cadence")
    return train, validation


def _option(args: list[str], name: str, *, required: bool = True) -> str | None:
    positions = [index for index, value in enumerate(args) if value == name]
    if not positions:
        if required:
            _fail(f"W&B metadata lacks {name}")
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        _fail(f"W&B metadata has malformed or repeated {name}")
    return args[positions[0] + 1]


def _flag(args: list[str], name: str) -> bool:
    return args.count(name) == 1


def _wandb_run_id(path: Path) -> str:
    match = re.fullmatch(r"run-(?:\d{8}_\d{6}-)?([A-Za-z0-9]+)", path.name)
    if match is None:
        _fail(f"Cannot derive W&B run ID from {path}")
    return match.group(1)


def _validate_wandb_phase(
    *,
    metadata_descriptor: Mapping[str, Any],
    summary_descriptor: Mapping[str, Any],
    event_descriptor: Mapping[str, Any],
    expected_step: int,
    order_manifest: Path,
    checkpoint: Path,
    reference_checkpoint: Path | None,
) -> dict[str, Any]:
    metadata = _read_json_artifact(metadata_descriptor, label="W&B metadata")
    summary = _read_json_artifact(summary_descriptor, label="W&B summary")
    args = metadata.get("args")
    if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
        _fail("W&B metadata args are invalid")
    if metadata.get("gpu_count") != 6 or Path(str(metadata.get("program", ""))).name != (
        "overfit_single_chunk.py"
    ):
        _fail("W&B metadata is not from the real six-GPU overfit program")
    required_options = {
        "--order-manifest": str(order_manifest.resolve(strict=True)),
        "--model-size": "tiny",
        "--device": "cuda",
        "--parameter-dtype": "bfloat16",
        "--precision": "bfloat16",
        "--steps": "1000",
        "--batch-size": "6",
        "--wandb-mode": "online",
    }
    for name, expected in required_options.items():
        found = _option(args, name)
        if name == "--order-manifest":
            try:
                matches = Path(str(found)).resolve(strict=True) == Path(expected)
            except OSError:
                matches = False
        else:
            matches = found == expected
        if not matches:
            _fail(f"W&B metadata {name} differs from real six-GPU evidence")
    if not _flag(args, "--compile"):
        _fail("W&B metadata does not prove the compiled overfit trajectory")

    if expected_step == 500:
        if _option(args, "--stop-after-step") != "500":
            _fail("Phase-one W&B metadata does not stop exactly at step 500")
        if _option(args, "--resume", required=False) is not None:
            _fail("Phase-one W&B metadata unexpectedly resumes")
    elif expected_step == 1000:
        if _option(args, "--stop-after-step", required=False) is not None:
            _fail("Phase-two W&B metadata unexpectedly stops early")
        resume = _option(args, "--resume")
        reference = _option(args, "--exact-reference-checkpoint")
        try:
            resume_matches = Path(str(resume)).resolve(strict=True) == checkpoint.resolve(
                strict=True
            )
            reference_matches = (
                reference_checkpoint is not None
                and Path(str(reference)).resolve(strict=True)
                == reference_checkpoint.resolve(strict=True)
            )
        except OSError:
            resume_matches = reference_matches = False
        if not resume_matches or not reference_matches:
            _fail("Phase-two W&B metadata is bound to another resume trajectory")
    else:
        _fail("Only the exact 500->1000 restart phases are accepted")

    expected_tokens = expected_step * 6 * 4096
    exact_metrics = {
        "train/step": expected_step,
        "train/world_size": 6,
        "train/global_microbatch_rows": 6,
        "train/local_microbatch_rows": 1,
        "train/gradient_accumulation_steps": 1,
        "train/input_tokens": expected_tokens,
        "checkpoint/completed": 1,
        "checkpoint/has_previous_generation": 1,
        "checkpoint/train_input_tokens": expected_tokens,
    }
    for key, expected in exact_metrics.items():
        if summary.get(key) != expected:
            _fail(f"W&B summary {key} does not prove step {expected_step}")
    counts = {
        prefix: sum(1 for key in summary if key.startswith(prefix))
        for prefix in _WANDB_METRIC_MINIMUMS
    }
    for prefix, minimum in _WANDB_METRIC_MINIMUMS.items():
        if counts[prefix] < minimum:
            _fail(f"W&B summary has too few {prefix} telemetry fields")
    event_path = Path(str(event_descriptor.get("path", "")))
    current_event = qualification.artifact(event_path, label="W&B event log")
    if current_event != dict(event_descriptor) or current_event["bytes"] < 1:
        _fail("W&B event log changed or is empty")
    run_dir = Path(str(metadata_descriptor["path"])).parent.parent
    if Path(str(summary_descriptor["path"])).parent.parent != run_dir:
        _fail("W&B metadata and summary are from different run directories")
    run_id = _wandb_run_id(run_dir)
    if event_path.parent != run_dir or event_path.name != f"run-{run_id}.wandb":
        _fail("W&B event log does not match the phase run ID")
    return {
        "run_id": run_id,
        "step": expected_step,
        "input_tokens": expected_tokens,
        "metric_family_counts": counts,
        "metric_keys_sha256": qualification.canonical_sha256(sorted(summary)),
    }


def _validate_overfit_result(
    descriptor: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    payload = _read_json_artifact(descriptor, label=label)
    if (
        payload.get("format") != "single-packed-chunk-overfit-qualification"
        or payload.get("format_version") != 2
        or payload.get("status") != "passed"
        or payload.get("failures") != []
        or payload.get("rank") != 0
        or payload.get("world_size") != 6
        or payload.get("steps") != 1000
        or payload.get("target_steps") != 1000
        or payload.get("partial_until_step") is not None
        or payload.get("global_input_shape") != [6, 4096]
        or payload.get("local_input_shape") != [1, 4096]
    ):
        _fail(f"{label} is not a completed real six-H100 1000-step result")
    boundaries = payload.get("packed_boundaries")
    if not isinstance(boundaries, Mapping) or (
        boundaries.get("contract") != "packed-document-causal-boundaries-v1"
        or boundaries.get("rows") != 6
        or boundaries.get("sequence_length") != 4096
        or _plain_int(
            boundaries.get("in_row_document_boundaries"),
            minimum=1,
            field=f"{label}.in_row_document_boundaries",
        )
        != boundaries.get("masked_boundary_targets")
    ):
        _fail(f"{label} does not prove packed document-boundary masking")
    initial = _decimal(payload.get("initial_loss"), field=f"{label}.initial_loss")
    final = _decimal(payload.get("final_loss"), field=f"{label}.final_loss")
    ratio = _decimal(payload.get("loss_ratio"), field=f"{label}.loss_ratio")
    required_ratio = _decimal(
        payload.get("required_loss_ratio"), field=f"{label}.required_loss_ratio"
    )
    required_final = _decimal(
        payload.get("required_final_loss"), field=f"{label}.required_final_loss"
    )
    if not final < initial or ratio > required_ratio or final > required_final:
        _fail(f"{label} does not meet its memorization thresholds")
    return payload


def _validate_prior_evidence(prior: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "overfit_order",
        "overfit_producer_source",
        "uninterrupted",
        "restart",
        "grid_validation_and_telemetry",
    }
    if set(prior) != required:
        _fail("Prior-evidence inventory is incomplete")
    order = prior.get("overfit_order")
    if not isinstance(order, Mapping):
        _fail("Overfit order evidence is invalid")
    order_record = _inspect_order(Path(str(order["manifest"]["path"])), split="train")
    if order_record["manifest"] != order.get("manifest") or order_record["payload"] != order.get(
        "payload"
    ):
        _fail("Overfit order changed after waiver publication")

    source = prior.get("overfit_producer_source")
    if not isinstance(source, Mapping):
        _fail("Overfit producer source evidence is invalid")
    if qualification.artifact(Path(str(source["path"])), label="overfit producer") != dict(
        source
    ):
        _fail("Overfit producer source changed")

    uninterrupted = prior.get("uninterrupted")
    restart = prior.get("restart")
    if not isinstance(uninterrupted, Mapping) or not isinstance(restart, Mapping):
        _fail("Overfit trajectory evidence is incomplete")
    control = _validate_overfit_result(
        uninterrupted["result"], label="uninterrupted overfit result"
    )
    resumed = _validate_overfit_result(restart["result"], label="resumed overfit result")
    control_checkpoint = qualification.artifact(
        Path(str(uninterrupted["checkpoint"]["path"])),
        label="uninterrupted overfit checkpoint",
    )
    resumed_checkpoint = qualification.artifact(
        Path(str(restart["checkpoint"]["path"])), label="resumed overfit checkpoint"
    )
    if control_checkpoint != uninterrupted["checkpoint"] or resumed_checkpoint != restart[
        "checkpoint"
    ]:
        _fail("Overfit checkpoint bytes changed")
    order_path = Path(str(order["manifest"]["path"])).resolve(strict=True)
    for label, result, checkpoint in (
        ("uninterrupted", control, control_checkpoint),
        ("resumed", resumed, resumed_checkpoint),
    ):
        try:
            result_order = Path(str(result["order_manifest"])).resolve(strict=True)
            result_checkpoint = Path(str(result["checkpoint"])).resolve(strict=True)
        except OSError as exc:
            raise GeometryWaiverError(f"{label} result path is invalid") from exc
        if result_order != order_path or result_checkpoint != Path(checkpoint["path"]):
            _fail(f"{label} result names another order or checkpoint")
    shared_fields = (
        "global_batch_sha256",
        "fixed_batch_identity",
        "model_config",
        "parameter_count",
        "tokenizer_manifest_sha256",
        "tokenizer_vocabulary_sha256",
        "global_input_shape",
        "local_input_shape",
        "packed_boundaries",
        "initial_loss",
        "final_loss",
        "loss_ratio",
    )
    if any(control.get(field) != resumed.get(field) for field in shared_fields):
        _fail("Restart and uninterrupted overfit trajectories differ")
    if control.get("exact_resume") != {"requested": False}:
        _fail("Uninterrupted overfit result has unexpected resume evidence")
    exact = resumed.get("exact_resume")
    if not isinstance(exact, Mapping) or (
        exact.get("requested") is not True
        or exact.get("exact_match") is not True
        or exact.get("tolerant_match") is not True
        or exact.get("accepted_match") is not True
        or exact.get("mismatches") != []
        or exact.get("tolerant_mismatches") != []
        or exact.get("components") != exact.get("reference_components")
    ):
        _fail("Restart result does not prove exact 500->1000 equivalence")
    try:
        actual = Path(str(exact["actual_checkpoint"])).resolve(strict=True)
        reference = Path(str(exact["reference_checkpoint"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise GeometryWaiverError("Restart comparison checkpoint paths are invalid") from exc
    if actual != Path(resumed_checkpoint["path"]) or reference != Path(
        control_checkpoint["path"]
    ):
        _fail("Restart comparison names another checkpoint pair")
    diagnostics = exact.get("numeric_diagnostics")
    if not isinstance(diagnostics, Mapping):
        _fail("Restart comparison lacks numeric diagnostics")
    for key, value in diagnostics.items():
        if key.endswith(
            ("mismatch_count", "nonfinite_count", "out_of_tolerance_count")
        ) and value != 0:
            _fail(f"Restart numeric diagnostic {key} is non-zero")

    phases = restart.get("wandb")
    if not isinstance(phases, Mapping) or set(phases) != {"phase_one", "phase_two"}:
        _fail("Restart W&B evidence must contain exactly two phases")
    phase_summaries: dict[str, Any] = {}
    for key, step, reference_path in (
        ("phase_one", 500, None),
        ("phase_two", 1000, Path(control_checkpoint["path"])),
    ):
        phase = phases[key]
        if not isinstance(phase, Mapping) or set(phase) != {
            "metadata",
            "summary",
            "event_log",
            "validation",
        }:
            _fail(f"Restart W&B {key} evidence has the wrong schema")
        summary = _validate_wandb_phase(
            metadata_descriptor=phase["metadata"],
            summary_descriptor=phase["summary"],
            event_descriptor=phase["event_log"],
            expected_step=step,
            order_manifest=order_path,
            checkpoint=Path(resumed_checkpoint["path"]),
            reference_checkpoint=reference_path,
        )
        if summary != phase["validation"]:
            _fail(f"Restart W&B {key} validation summary is inconsistent")
        phase_summaries[key] = summary
    if phase_summaries["phase_one"]["run_id"] != phase_summaries["phase_two"]["run_id"]:
        _fail("Restart W&B phases do not continue the same run ID")
    grid = prior.get("grid_validation_and_telemetry")
    if not isinstance(grid, Mapping) or set(grid) != {
        "selected_candidate_result",
        "validation_events",
        "checkpoint_events",
        "wandb_log_events",
        "resume_verified",
    }:
        _fail("Grid validation/telemetry evidence is malformed")
    for field in ("validation_events", "checkpoint_events", "wandb_log_events"):
        _plain_int(grid.get(field), minimum=1, field=f"grid evidence {field}")
    if grid.get("resume_verified") is not True:
        _fail("Selected grid candidate did not verify resume")
    return {
        "uninterrupted_steps": 1000,
        "restart_boundary_step": 500,
        "restart_final_step": 1000,
        "restart_exact_match": True,
        "wandb_run_id": phase_summaries["phase_one"]["run_id"],
        "grid_validation_events": grid["validation_events"],
        "grid_checkpoint_events": grid["checkpoint_events"],
        "grid_wandb_log_events": grid["wandb_log_events"],
    }


def _decision_record(*, decision_by: str, decision_utc: str, rationale: str) -> dict[str, Any]:
    if not all(
        isinstance(value, str) and value.strip()
        for value in (decision_by, decision_utc, rationale)
    ):
        _fail("Waiver decision_by, decision_utc, and rationale must be non-empty")
    try:
        timestamp = datetime.fromisoformat(decision_utc)
    except ValueError as exc:
        raise GeometryWaiverError("Waiver decision_utc is invalid") from exc
    if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc).astimezone(
        timestamp.tzinfo
    ):
        _fail("Waiver decision_utc must be timezone-aware and not in the future")
    return {
        "decision": "approved",
        "decision_by": decision_by.strip(),
        "decision_utc": decision_utc,
        "rationale": rationale.strip(),
        "scope": WAIVER_SCOPE,
        "required_full_run_safeguards": list(REQUIRED_SAFEGUARDS),
    }


def collect_geometry_waiver(
    *,
    grid_result: Path,
    hardware_contract: Path,
    train_order_manifest: Path,
    validation_order_manifest: Path,
    uninterrupted_result: Path,
    resumed_result: Path,
    overfit_order_manifest: Path,
    phase_one_metadata: Path,
    phase_one_summary: Path,
    phase_one_event_log: Path,
    phase_two_metadata: Path,
    phase_two_summary: Path,
    phase_two_event_log: Path,
    waiver_cli: Path,
    decision_by: str,
    decision_utc: str,
    rationale: str,
) -> dict[str, Any]:
    """Collect a self-authenticating waiver receipt from existing evidence."""

    try:
        grid_loader = qualification._load_validated_grid_result  # type: ignore[attr-defined]
        grid, grid_bound, grid_plan_bound = grid_loader(grid_result)
        grid_plan = qualification.read_bound_json(
            grid_plan_bound["artifact"], label="authenticated geometry grid plan"
        )
        hardware_loader = qualification._validate_hardware_contract  # type: ignore[attr-defined]
        hardware, hardware_bound, hardware_identity = hardware_loader(hardware_contract)
    except (OSError, ValueError) as exc:
        raise GeometryWaiverError(f"Grid or hardware authority is invalid: {exc}") from exc
    grid_identity = grid_plan["inputs"]["hardware"]["geometry_identity"]
    hardware_compatibility = _hardware_compatibility(
        final_identity=hardware_identity,
        grid_identity=grid_identity,
    )
    runtime_compatibility = _validate_grid_runtime_compatibility(grid_plan)
    candidate = grid.get("accepted_candidate")
    accepted_result = grid.get("accepted_result")
    if not isinstance(candidate, Mapping) or not isinstance(accepted_result, Mapping):
        _fail("Passing grid does not identify an accepted candidate/result")
    selected_bound = accepted_result.get("artifact")
    if not isinstance(selected_bound, Mapping):
        _fail("Grid accepted result binding is invalid")
    selected = qualification.read_bound_json(
        selected_bound["artifact"], label="selected grid candidate result"
    )
    measurements = selected.get("measurements")
    if not isinstance(measurements, Mapping):
        _fail("Selected grid candidate lacks measurements")
    external = measurements.get("throughput_measurement")
    if not isinstance(external, Mapping):
        _fail("Selected grid candidate lacks external validation/telemetry evidence")

    accepted = {
        "global_microbatch_rows": candidate["global_microbatch_rows"],
        "gradient_accumulation_steps": candidate["gradient_accumulation_steps"],
        "workers": grid_plan["settings"]["workers"],
        "overfit_batch_rows": candidate["global_microbatch_rows"],
        "compile_model": candidate["compile_model"],
        "activation_checkpointing": True,
        "precision": "bfloat16",
        "parameter_dtype": "float32",
    }
    train, validation = _validate_orders(
        train_path=train_order_manifest,
        validation_path=validation_order_manifest,
        grid_plan=grid_plan,
        accepted=accepted,
    )
    overfit_order = _inspect_order(overfit_order_manifest, split="train")
    control_descriptor = qualification.artifact(
        uninterrupted_result, label="uninterrupted overfit result"
    )
    resume_descriptor = qualification.artifact(resumed_result, label="resumed overfit result")
    control = _validate_overfit_result(control_descriptor, label="uninterrupted overfit result")
    resumed = _validate_overfit_result(resume_descriptor, label="resumed overfit result")
    control_checkpoint = qualification.artifact(
        Path(str(control["checkpoint"])), label="uninterrupted overfit checkpoint"
    )
    resumed_checkpoint = qualification.artifact(
        Path(str(resumed["checkpoint"])), label="resumed overfit checkpoint"
    )
    producer_source = qualification.artifact(
        Path(str(_read_json_artifact(
            qualification.artifact(phase_one_metadata, label="phase-one W&B metadata"),
            label="phase-one W&B metadata",
        )["program"])),
        label="overfit producer source",
    )
    prior: dict[str, Any] = {
        "overfit_order": {
            "manifest": overfit_order["manifest"],
            "payload": overfit_order["payload"],
        },
        "overfit_producer_source": producer_source,
        "uninterrupted": {
            "result": control_descriptor,
            "checkpoint": control_checkpoint,
        },
        "restart": {
            "result": resume_descriptor,
            "checkpoint": resumed_checkpoint,
            "wandb": {
                "phase_one": {
                    "metadata": qualification.artifact(
                        phase_one_metadata, label="phase-one W&B metadata"
                    ),
                    "summary": qualification.artifact(
                        phase_one_summary, label="phase-one W&B summary"
                    ),
                    "event_log": qualification.artifact(
                        phase_one_event_log, label="phase-one W&B event log"
                    ),
                },
                "phase_two": {
                    "metadata": qualification.artifact(
                        phase_two_metadata, label="phase-two W&B metadata"
                    ),
                    "summary": qualification.artifact(
                        phase_two_summary, label="phase-two W&B summary"
                    ),
                    "event_log": qualification.artifact(
                        phase_two_event_log, label="phase-two W&B event log"
                    ),
                },
            },
        },
        "grid_validation_and_telemetry": {
            "selected_candidate_result": selected_bound,
            "validation_events": external.get("validation_events"),
            "checkpoint_events": external.get("checkpoint_events"),
            "wandb_log_events": external.get("wandb_log_events"),
            "resume_verified": external.get("resume_verified"),
        },
    }
    for phase in prior["restart"]["wandb"].values():
        step = 500 if phase is prior["restart"]["wandb"]["phase_one"] else 1000
        phase["validation"] = _validate_wandb_phase(
            metadata_descriptor=phase["metadata"],
            summary_descriptor=phase["summary"],
            event_descriptor=phase["event_log"],
            expected_step=step,
            order_manifest=overfit_order_manifest,
            checkpoint=Path(resumed_checkpoint["path"]),
            reference_checkpoint=(
                None if step == 500 else Path(control_checkpoint["path"])
            ),
        )

    receipt: dict[str, Any] = {
        "format": WAIVER_FORMAT,
        "format_version": WAIVER_VERSION,
        "status": "pass",
        "hardware_contract_sha256": hardware_bound["artifact"]["sha256"],
        "train_order_manifest_sha256": train["manifest"]["sha256"],
        "validation_order_manifest_sha256": validation["manifest"]["sha256"],
        "accepted": accepted,
        "measurements": dict(measurements),
        "final_soak_waived": True,
        "waiver": _decision_record(
            decision_by=decision_by,
            decision_utc=decision_utc,
            rationale=rationale,
        ),
        "qualification": {
            "hardware": {
                "bound": hardware_bound,
                "expected": hardware,
                "geometry_identity": hardware_identity,
                "grid_geometry_identity": grid_identity,
                "compatibility": hardware_compatibility,
            },
            "single_gpu_baselines": grid_plan["inputs"]["single_gpu_baselines"],
            "grid_plan": grid_plan_bound,
            "grid_result": grid_bound,
            "selected_candidate": dict(candidate),
            "selected_candidate_result": selected_bound,
            "selected_candidate_measurements_sha256": qualification.canonical_sha256(
                measurements
            ),
            "train_order": train,
            "validation_order": validation,
            "grid_runtime_compatibility": runtime_compatibility,
            "prior_evidence": prior,
            "producer_sources": _current_sources(waiver_cli),
            "secrets_recorded": False,
        },
    }
    validate_geometry_waiver_receipt(receipt)
    return receipt


def validate_geometry_waiver_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_hardware_contract_sha256: str | None = None,
    expected_train_order: Mapping[str, Any] | None = None,
    expected_validation_order: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-authenticate all evidence frozen into a waiver receipt."""

    required_top = {
        "format",
        "format_version",
        "status",
        "hardware_contract_sha256",
        "train_order_manifest_sha256",
        "validation_order_manifest_sha256",
        "accepted",
        "measurements",
        "final_soak_waived",
        "waiver",
        "qualification",
    }
    if set(receipt) != required_top or (
        receipt.get("format") != WAIVER_FORMAT
        or receipt.get("format_version") != WAIVER_VERSION
        or receipt.get("status") != "pass"
        or receipt.get("final_soak_waived") is not True
    ):
        _fail("Unsupported or non-passing geometry waiver receipt")
    for field in (
        "hardware_contract_sha256",
        "train_order_manifest_sha256",
        "validation_order_manifest_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _fail(f"Waiver {field} is invalid")
    if (
        expected_hardware_contract_sha256 is not None
        and receipt["hardware_contract_sha256"] != expected_hardware_contract_sha256
    ):
        _fail("Waiver is bound to another final hardware contract")
    waiver = receipt.get("waiver")
    if not isinstance(waiver, Mapping) or waiver != _decision_record(
        decision_by=str(waiver.get("decision_by", "")),
        decision_utc=str(waiver.get("decision_utc", "")),
        rationale=str(waiver.get("rationale", "")),
    ):
        _fail("Waiver decision metadata is non-canonical")
    record = receipt.get("qualification")
    required_record = {
        "hardware",
        "single_gpu_baselines",
        "grid_plan",
        "grid_result",
        "selected_candidate",
        "selected_candidate_result",
        "selected_candidate_measurements_sha256",
        "train_order",
        "validation_order",
        "grid_runtime_compatibility",
        "prior_evidence",
        "producer_sources",
        "secrets_recorded",
    }
    if not isinstance(record, Mapping) or set(record) != required_record or record.get(
        "secrets_recorded"
    ) is not False:
        _fail("Waiver qualification provenance is incomplete")
    try:
        grid_loader = qualification._load_validated_grid_result  # type: ignore[attr-defined]
        grid, grid_bound, grid_plan_bound = grid_loader(
            Path(str(record["grid_result"]["artifact"]["path"]))
        )
        grid_plan = qualification.read_bound_json(
            grid_plan_bound["artifact"], label="waiver grid plan"
        )
        hardware_loader = qualification._validate_hardware_contract  # type: ignore[attr-defined]
        hardware, hardware_bound, hardware_identity = hardware_loader(
            Path(str(record["hardware"]["bound"]["artifact"]["path"]))
        )
    except (KeyError, OSError, ValueError) as exc:
        raise GeometryWaiverError(f"Waiver grid/hardware evidence is invalid: {exc}") from exc
    if grid_bound != record["grid_result"] or grid_plan_bound != record["grid_plan"]:
        _fail("Grid result or plan changed after waiver publication")
    if hardware_bound != record["hardware"]["bound"] or hardware != record["hardware"][
        "expected"
    ]:
        _fail("Final hardware contract changed after waiver publication")
    if receipt["hardware_contract_sha256"] != hardware_bound["artifact"]["sha256"]:
        _fail("Top-level hardware digest differs from bound strict hardware")
    grid_identity = grid_plan["inputs"]["hardware"]["geometry_identity"]
    compatibility = _hardware_compatibility(
        final_identity=hardware_identity,
        grid_identity=grid_identity,
    )
    if (
        hardware_identity != record["hardware"]["geometry_identity"]
        or grid_identity != record["hardware"]["grid_geometry_identity"]
        or compatibility != record["hardware"].get("compatibility")
    ):
        _fail("Final hardware and provisional grid hardware identities differ")
    compatibility = _validate_grid_runtime_compatibility(grid_plan)
    if compatibility != record["grid_runtime_compatibility"]:
        _fail("Frozen-grid/current-runtime compatibility record is inconsistent")
    if record["single_gpu_baselines"] != grid_plan["inputs"]["single_gpu_baselines"]:
        _fail("Waiver is bound to another single-GPU baseline receipt")

    candidate = grid.get("accepted_candidate")
    accepted_result = grid.get("accepted_result")
    if not isinstance(candidate, Mapping) or not isinstance(accepted_result, Mapping):
        _fail("Grid has no accepted candidate")
    selected_bound = accepted_result["artifact"]
    selected = qualification.read_bound_json(
        selected_bound["artifact"], label="waiver selected grid result"
    )
    measurements = selected.get("measurements")
    expected_accepted = {
        "global_microbatch_rows": candidate["global_microbatch_rows"],
        "gradient_accumulation_steps": candidate["gradient_accumulation_steps"],
        "workers": grid_plan["settings"]["workers"],
        "overfit_batch_rows": candidate["global_microbatch_rows"],
        "compile_model": candidate["compile_model"],
        "activation_checkpointing": True,
        "precision": "bfloat16",
        "parameter_dtype": "float32",
    }
    if (
        receipt.get("accepted") != expected_accepted
        or receipt.get("measurements") != measurements
        or record["selected_candidate"] != candidate
        or record["selected_candidate_result"] != selected_bound
        or record["selected_candidate_measurements_sha256"]
        != qualification.canonical_sha256(measurements)
    ):
        _fail("Waiver selected candidate/result/measurements binding is invalid")

    train, validation = _validate_orders(
        train_path=Path(str(record["train_order"]["manifest"]["path"])),
        validation_path=Path(str(record["validation_order"]["manifest"]["path"])),
        grid_plan=grid_plan,
        accepted=expected_accepted,
    )
    if train != record["train_order"] or validation != record["validation_order"]:
        _fail("Final train or validation order changed after waiver publication")
    if (
        receipt["train_order_manifest_sha256"] != train["manifest"]["sha256"]
        or receipt["validation_order_manifest_sha256"]
        != validation["manifest"]["sha256"]
    ):
        _fail("Top-level final-order digests are inconsistent")
    if expected_train_order is not None and (
        expected_train_order.get("manifest") != train["manifest"]
        or expected_train_order.get("payload") != train["payload"]
    ):
        _fail("Waiver is bound to another authoritative train order")
    if expected_validation_order is not None and (
        expected_validation_order.get("manifest") != validation["manifest"]
        or expected_validation_order.get("payload") != validation["payload"]
    ):
        _fail("Waiver is bound to another authoritative validation order")
    prior_summary = _validate_prior_evidence(record["prior_evidence"])
    grid_prior = record["prior_evidence"]["grid_validation_and_telemetry"]
    external = measurements.get("throughput_measurement")
    if not isinstance(external, Mapping) or (
        grid_prior["selected_candidate_result"] != selected_bound
        or any(
            grid_prior[field] != external.get(field)
            for field in (
                "validation_events",
                "checkpoint_events",
                "wandb_log_events",
                "resume_verified",
            )
        )
    ):
        _fail("Prior validation/telemetry claim differs from selected grid evidence")
    _validate_current_sources(record["producer_sources"])
    return {
        "status": "pass",
        "format": WAIVER_FORMAT,
        "final_soak_waived": True,
        "waiver_scope": WAIVER_SCOPE,
        "hardware_identity_sha256": hardware_identity["identity_sha256"],
        "grid_compatible_hardware_identity_sha256": compatibility[
            "stable_identity_sha256"
        ],
        "grid_result_sha256": grid_bound["artifact"]["sha256"],
        "selected_candidate": dict(candidate),
        "train_order_manifest_sha256": train["manifest"]["sha256"],
        "train_order_payload_sha256": train["payload"]["sha256"],
        "validation_order_manifest_sha256": validation["manifest"]["sha256"],
        "validation_order_payload_sha256": validation["payload"]["sha256"],
        "prior_evidence": prior_summary,
    }
