"""Strict launch-time validation for the external six-GPU geometry soak.

The run-authority package authenticates the accepted-geometry receipt.  This
module additionally requires that receipt to identify and arithmetically
reconcile an external monotonic wall-clock interval rather than the trainer's
compute-only per-step metric, and enforces production soak/memory/scale gates.
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


MINIMUM_SOAK_STEPS = 100
MINIMUM_SCALING_EFFICIENCY = Decimal("0.70")
MAXIMUM_DATA_WAIT_FRACTION = Decimal("0.05")
MINIMUM_FREE_MEMORY_BYTES = 8 * 1024**3
MINIMUM_FREE_MEMORY_FRACTION = Decimal("0.10")
MAXIMUM_THROUGHPUT_RELATIVE_ERROR = Decimal("0.000001")
THROUGHPUT_SCOPE = (
    "end-to-end-including-validation-checkpoint-wandb-and-resume"
)
THROUGHPUT_TIMER = "time.monotonic_ns"
THROUGHPUT_COUNTER = "trainer.consumed_input_tokens"


class GeometryEvidenceError(ValueError):
    """The authenticated authority contains inadequate geometry evidence."""


def _plain_int(value: Any, *, minimum: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise GeometryEvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, *, field: str, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise GeometryEvidenceError(
            f"{field} must be a canonical decimal string or integer"
        )
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise GeometryEvidenceError(f"{field} is not a decimal") from exc
    if not result.is_finite() or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise GeometryEvidenceError(f"{field} must be finite and {qualifier}")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_artifact(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if set(descriptor) != {"path", "bytes", "sha256"}:
        raise GeometryEvidenceError(
            f"{label} descriptor must contain exactly path, bytes, and sha256"
        )
    path_value = descriptor.get("path")
    byte_count = descriptor.get("bytes")
    digest = descriptor.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise GeometryEvidenceError(f"{label} path must be absolute")
    _plain_int(byte_count, minimum=1, field=f"{label} bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GeometryEvidenceError(f"{label} sha256 is invalid")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise GeometryEvidenceError(f"{label} is not a regular non-symlink file: {path}")
    payload = path.read_bytes()
    if len(payload) != byte_count or _sha256_bytes(payload) != digest:
        raise GeometryEvidenceError(f"{label} changed after authority publication")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GeometryEvidenceError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GeometryEvidenceError(f"{label} root must be an object")
    return value


def validate_authority_geometry_soak(
    authority_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the production geometry evidence in an authority."""

    try:
        geometry = authority_payload["geometry"]
        hardware = authority_payload["hardware"]
        training = authority_payload["training"]
        data = authority_payload["data"]
    except KeyError as exc:
        raise GeometryEvidenceError("Run authority lacks geometry soak inputs") from exc
    if not all(isinstance(value, Mapping) for value in (geometry, hardware, training, data)):
        raise GeometryEvidenceError("Run-authority geometry soak inputs must be objects")

    descriptor = geometry.get("artifact")
    embedded_receipt = geometry.get("receipt")
    if not isinstance(descriptor, Mapping) or not isinstance(embedded_receipt, Mapping):
        raise GeometryEvidenceError("Run authority lacks an authenticated geometry receipt")
    receipt = _read_json_artifact(descriptor, label="accepted-geometry receipt")
    if receipt != dict(embedded_receipt):
        raise GeometryEvidenceError(
            "Accepted-geometry receipt differs from the authority's embedded copy"
        )
    final_soak_waived = receipt.get("final_soak_waived") is True
    waiver_validation: dict[str, Any] | None = None
    if final_soak_waived:
        try:
            from pretrain.geometry_waiver import validate_geometry_waiver_receipt

            train_order = data.get("train_order")
            validation_order = data.get("validation_order")
            hardware_contract = hardware.get("contract")
            if not all(
                isinstance(value, Mapping)
                for value in (train_order, validation_order, hardware_contract)
            ):
                raise GeometryEvidenceError(
                    "Waived geometry requires authoritative hardware/train/validation bindings"
                )
            waiver_validation = validate_geometry_waiver_receipt(
                receipt,
                expected_hardware_contract_sha256=str(hardware_contract["sha256"]),
                expected_train_order=train_order,
                expected_validation_order=validation_order,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise GeometryEvidenceError(
                f"Separate-final-soak waiver evidence is invalid: {exc}"
            ) from exc
    accepted = receipt.get("accepted")
    measurements = receipt.get("measurements")
    expected_hardware = hardware.get("expected")
    frozen_geometry = training.get("frozen_geometry")
    train_order = data.get("train_order")
    if not all(
        isinstance(value, Mapping)
        for value in (accepted, measurements, expected_hardware, frozen_geometry, train_order)
    ):
        raise GeometryEvidenceError("Geometry receipt authority objects are incomplete")

    global_rows = _plain_int(
        accepted.get("global_microbatch_rows"),
        minimum=1,
        field="accepted.global_microbatch_rows",
    )
    accumulation = _plain_int(
        accepted.get("gradient_accumulation_steps"),
        minimum=1,
        field="accepted.gradient_accumulation_steps",
    )
    sequence_length = _plain_int(
        train_order.get("sequence_length"),
        minimum=1,
        field="train order sequence_length",
    )
    tokens_per_update = global_rows * accumulation * sequence_length

    soak_steps = _plain_int(
        measurements.get("soak_steps"),
        minimum=MINIMUM_SOAK_STEPS,
        field="measurements.soak_steps",
    )
    throughput = _decimal(
        measurements.get("aggregate_input_tokens_per_second"),
        field="measurements.aggregate_input_tokens_per_second",
    )
    scaling = _decimal(
        measurements.get("scaling_efficiency"),
        field="measurements.scaling_efficiency",
    )
    if scaling > 1 or scaling < MINIMUM_SCALING_EFFICIENCY:
        raise GeometryEvidenceError(
            f"Scaling efficiency must be in [{MINIMUM_SCALING_EFFICIENCY}, 1]"
        )
    data_wait = _decimal(
        measurements.get("data_wait_fraction"),
        field="measurements.data_wait_fraction",
        allow_zero=True,
    )
    if data_wait > MAXIMUM_DATA_WAIT_FRACTION:
        raise GeometryEvidenceError(
            f"Data-wait fraction exceeds {MAXIMUM_DATA_WAIT_FRACTION}"
        )

    gpu_memory = _plain_int(
        expected_hardware.get("gpu_memory_bytes"),
        minimum=1,
        field="hardware gpu_memory_bytes",
    )
    peak_allocated = _plain_int(
        measurements.get("peak_memory_allocated_bytes_per_gpu"),
        minimum=1,
        field="measurements.peak_memory_allocated_bytes_per_gpu",
    )
    peak_reserved = _plain_int(
        measurements.get("peak_memory_reserved_bytes_per_gpu"),
        minimum=1,
        field="measurements.peak_memory_reserved_bytes_per_gpu",
    )
    minimum_free = _plain_int(
        measurements.get("minimum_free_memory_bytes_per_gpu"),
        minimum=1,
        field="measurements.minimum_free_memory_bytes_per_gpu",
    )
    fraction_margin = int(
        (Decimal(gpu_memory) * MINIMUM_FREE_MEMORY_FRACTION).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    required_free = max(MINIMUM_FREE_MEMORY_BYTES, fraction_margin)
    if not peak_allocated <= peak_reserved <= gpu_memory:
        raise GeometryEvidenceError(
            "GPU memory measurements must satisfy allocated <= reserved <= physical"
        )
    if minimum_free < required_free:
        raise GeometryEvidenceError(
            f"Minimum free GPU memory {minimum_free} is below required {required_free}"
        )
    if peak_reserved + minimum_free > gpu_memory:
        raise GeometryEvidenceError(
            "Peak reserved plus minimum free GPU memory exceeds physical memory"
        )

    external = measurements.get("throughput_measurement")
    required_external_keys = {
        "scope",
        "timer",
        "counter",
        "start_consumed_input_tokens",
        "end_consumed_input_tokens",
        "elapsed_wall_time_ns",
        "validation_events",
        "checkpoint_events",
        "wandb_log_events",
        "resume_verified",
    }
    if not isinstance(external, Mapping) or set(external) != required_external_keys:
        raise GeometryEvidenceError(
            "measurements.throughput_measurement must contain the exact external-soak schema"
        )
    if external["scope"] != THROUGHPUT_SCOPE:
        raise GeometryEvidenceError("Throughput scope is not end-to-end")
    if external["timer"] != THROUGHPUT_TIMER or external["counter"] != THROUGHPUT_COUNTER:
        raise GeometryEvidenceError("Throughput must use the external monotonic timer/counter")
    if external["resume_verified"] is not True:
        raise GeometryEvidenceError("The six-rank soak must verify checkpoint resume")
    for field in ("validation_events", "checkpoint_events", "wandb_log_events"):
        _plain_int(external[field], minimum=1, field=f"throughput_measurement.{field}")
    start_tokens = _plain_int(
        external["start_consumed_input_tokens"],
        minimum=0,
        field="throughput_measurement.start_consumed_input_tokens",
    )
    end_tokens = _plain_int(
        external["end_consumed_input_tokens"],
        minimum=1,
        field="throughput_measurement.end_consumed_input_tokens",
    )
    elapsed_ns = _plain_int(
        external["elapsed_wall_time_ns"],
        minimum=1,
        field="throughput_measurement.elapsed_wall_time_ns",
    )
    if end_tokens <= start_tokens:
        raise GeometryEvidenceError("External soak input-token counter did not advance")
    if start_tokens % tokens_per_update or end_tokens % tokens_per_update:
        raise GeometryEvidenceError("External soak counters are not optimizer-update aligned")
    token_delta = end_tokens - start_tokens
    if token_delta != soak_steps * tokens_per_update:
        raise GeometryEvidenceError(
            "External soak token delta does not match soak_steps and frozen geometry"
        )
    authorized_tokens = _plain_int(
        frozen_geometry.get("consumed_input_tokens"),
        minimum=1,
        field="frozen consumed_input_tokens",
    )
    if end_tokens > authorized_tokens:
        raise GeometryEvidenceError("External soak counter exceeds the training order")
    calculated = Decimal(token_delta) * Decimal(1_000_000_000) / Decimal(elapsed_ns)
    relative_error = abs(throughput - calculated) / calculated
    if relative_error > MAXIMUM_THROUGHPUT_RELATIVE_ERROR:
        raise GeometryEvidenceError(
            "Reported throughput does not match external token delta / wall time"
        )

    result = {
        "status": "pass",
        "soak_steps": soak_steps,
        "tokens_per_update": tokens_per_update,
        "input_token_delta": token_delta,
        "elapsed_wall_time_ns": elapsed_ns,
        "calculated_input_tokens_per_second": format(calculated, "f"),
        "reported_input_tokens_per_second": format(throughput, "f"),
        "scaling_efficiency": format(scaling, "f"),
        "data_wait_fraction": format(data_wait, "f"),
        "required_free_memory_bytes_per_gpu": required_free,
        "minimum_free_memory_bytes_per_gpu": minimum_free,
    }
    if final_soak_waived:
        result.update(
            final_soak_waived=True,
            waiver_validation=waiver_validation,
        )
    return result
