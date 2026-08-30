#!/usr/bin/env python3
"""Authenticate, prewarm, and launch the frozen six-GPU PrimeRL SFT job.

The wrapper has three deliberately separate modes. ``verify-only`` performs
no writes. ``prewarm`` lets one process materialize Hugging Face dataset caches
and publishes a cache marker. ``launch`` repeats all integrity gates, requires
that marker, and replaces this process with PrimeRL. The checked-in contract
only permits Prime's dry run; a later real run needs a new reviewed contract
and a dataset-bound training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from pretrain.hf_export import (  # noqa: E402
    EXPORT_FORMAT,
    EXPORT_FORMAT_VERSION,
    EXPORT_MANIFEST_NAME,
    EXPORT_MANIFEST_SIDECAR_NAME,
    SOURCE_TOKENIZER_MANIFEST_NAME,
    TOKENIZER_EXPORT_FORMAT,
    TOKENIZER_EXPORT_MANIFEST_VERSION,
    TOKENIZER_MANIFEST_NAME,
)
# Some Python environments install an unrelated top-level ``scripts``
# package.  The curation module intentionally imports sibling utilities as
# ``scripts.*``. Load the repository modules through a temporary namespace,
# then restore *every* prior scripts module so importing this wrapper cannot
# poison the host interpreter.
_prior_scripts_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == "scripts" or name.startswith("scripts.")
}
for _name in tuple(_prior_scripts_modules):
    sys.modules.pop(_name, None)
_local_scripts = types.ModuleType("scripts")
_local_scripts.__path__ = [str(SCRIPTS_ROOT)]  # type: ignore[attr-defined]
sys.modules["scripts"] = _local_scripts
try:
    renderer_patch = importlib.import_module("scripts.apply_prime_renderer_patch")
    _prepare_prime_sft = importlib.import_module("scripts.prepare_prime_sft")
finally:
    for _name in tuple(sys.modules):
        if _name == "scripts" or _name.startswith("scripts."):
            sys.modules.pop(_name, None)
    sys.modules.update(_prior_scripts_modules)

CHAT_FORMAT_ID = _prepare_prime_sft.CHAT_FORMAT_ID
SFTPreparationError = _prepare_prime_sft.SFTPreparationError
file_sha256 = _prepare_prime_sft.file_sha256
validate_published_dataset = _prepare_prime_sft.validate_published_dataset


CONTRACT_PATH = PROJECT_ROOT / "configs" / "posttrain" / "prime_sft_launch_v1.json"
PREWARM_MARKER = "PREWARM_COMPLETE.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIB = 1024**3


class PrimeSFTPreflightError(RuntimeError):
    """Raised when an immutable launch gate does not hold."""


@dataclass(frozen=True)
class LaunchContract:
    config_path: Path
    config_sha256: str
    prime_rl_commit: str
    renderers_commit: str
    cache_default_root: Path
    cache_minimum_free_bytes: int
    cache_dataset_size_multiplier: int
    cache_reserve_bytes: int


@dataclass(frozen=True)
class PrimeConfig:
    path: Path
    sha256: str
    payload: dict[str, Any]
    dataset_root: Path
    model_root: Path
    dry_run: bool


@dataclass(frozen=True)
class CachePlan:
    root: Path
    hf_home: Path
    datasets_cache: Path
    required_free_bytes: int
    available_free_bytes: int
    prewarmed: bool


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrimeSFTPreflightError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimeSFTPreflightError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PrimeSFTPreflightError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise PrimeSFTPreflightError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrimeSFTPreflightError(f"{label} must be a full lowercase Git commit")
    return value


def _simple_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise PrimeSFTPreflightError(f"{label} must be a plain file name")
    return value


def _positive_integer(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise PrimeSFTPreflightError(f"{label} must be a {comparator} integer")
    return value


def _verify_file_record(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) - {
        "path",
        "bytes",
        "sha256",
        "tensors",
        "format",
        "manifest_version",
        "declared_files_verified",
    }:
        raise PrimeSFTPreflightError(f"{label} contains unexpected fields")
    name = _simple_name(record.get("path"), label=f"{label}.path")
    expected_bytes = _positive_integer(record.get("bytes"), label=f"{label}.bytes")
    expected_sha = _sha256(record.get("sha256"), label=f"{label}.sha256")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise PrimeSFTPreflightError(f"Missing or unsafe authenticated file: {path}")
    if path.stat().st_size != expected_bytes:
        raise PrimeSFTPreflightError(f"Authenticated byte size changed: {path}")
    if file_sha256(path) != expected_sha:
        raise PrimeSFTPreflightError(f"Authenticated checksum changed: {path}")
    return path


def load_launch_contract(path: Path = CONTRACT_PATH) -> LaunchContract:
    payload = _load_json(path, label="Prime SFT launch contract")
    if set(payload) != {
        "format_version",
        "kind",
        "config_path",
        "config_sha256",
        "prime_rl_commit",
        "renderers_commit",
        "cache",
    }:
        raise PrimeSFTPreflightError("Prime SFT launch contract fields changed")
    if payload.get("format_version") != 1 or payload.get("kind") != "prime_sft_launch_contract":
        raise PrimeSFTPreflightError("Unsupported Prime SFT launch contract")
    cache = payload.get("cache")
    if not isinstance(cache, dict) or set(cache) != {
        "default_root",
        "minimum_free_bytes",
        "dataset_size_multiplier",
        "reserve_bytes",
    }:
        raise PrimeSFTPreflightError("Prime SFT cache contract fields changed")
    relative_config = Path(str(payload.get("config_path", "")))
    if relative_config.is_absolute() or ".." in relative_config.parts:
        raise PrimeSFTPreflightError("Prime SFT config path must stay beneath the repository")
    config_path = (PROJECT_ROOT / relative_config).resolve(strict=True)
    if PROJECT_ROOT.resolve() not in config_path.parents:
        raise PrimeSFTPreflightError("Prime SFT config escaped the repository")
    prime_commit = _git_commit(payload.get("prime_rl_commit"), label="prime_rl_commit")
    renderer_commit = _git_commit(payload.get("renderers_commit"), label="renderers_commit")
    if prime_commit != renderer_patch.PRIME_RL_COMMIT:
        raise PrimeSFTPreflightError("Launch and renderer-installer PrimeRL pins disagree")
    if renderer_commit != renderer_patch.RENDERERS_COMMIT:
        raise PrimeSFTPreflightError("Launch and renderer-installer renderer pins disagree")
    default_root = Path(str(cache.get("default_root", "")))
    if not default_root.is_absolute():
        raise PrimeSFTPreflightError("Default Hugging Face cache root must be absolute")
    return LaunchContract(
        config_path=config_path,
        config_sha256=_sha256(payload.get("config_sha256"), label="config_sha256"),
        prime_rl_commit=prime_commit,
        renderers_commit=renderer_commit,
        cache_default_root=default_root,
        cache_minimum_free_bytes=_positive_integer(
            cache.get("minimum_free_bytes"), label="cache.minimum_free_bytes"
        ),
        cache_dataset_size_multiplier=_positive_integer(
            cache.get("dataset_size_multiplier"), label="cache.dataset_size_multiplier"
        ),
        cache_reserve_bytes=_positive_integer(
            cache.get("reserve_bytes"), label="cache.reserve_bytes"
        ),
    )


def load_prime_config(contract: LaunchContract, path: Path | None = None) -> PrimeConfig:
    config_path = (path or contract.config_path).expanduser().resolve(strict=True)
    if config_path != contract.config_path:
        raise PrimeSFTPreflightError(
            f"Config must be the contracted file {contract.config_path}; got {config_path}"
        )
    if config_path.is_symlink() or not config_path.is_file():
        raise PrimeSFTPreflightError(f"Missing or unsafe Prime SFT config: {config_path}")
    config_sha = file_sha256(config_path)
    if config_sha != contract.config_sha256:
        raise PrimeSFTPreflightError(
            f"Prime SFT TOML changed: expected {contract.config_sha256}, got {config_sha}"
        )
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PrimeSFTPreflightError(f"Cannot parse Prime SFT TOML: {exc}") from exc
    expected_top_level = {
        "max_steps",
        "dry_run",
        "output_dir",
        "run",
        "deployment",
        "model",
        "tokenizer",
        "renderer",
        "data",
        "val",
        "optim",
        "scheduler",
        "ckpt",
        "monitors",
        "log",
    }
    if set(payload) != expected_top_level:
        raise PrimeSFTPreflightError("Prime SFT TOML top-level contract changed")
    dry_run = payload.get("dry_run")
    if not isinstance(dry_run, bool):
        raise PrimeSFTPreflightError("Prime SFT dry_run must be boolean")
    deployment = payload.get("deployment", {})
    model = payload.get("model", {})
    tokenizer = payload.get("tokenizer", {})
    renderer = payload.get("renderer", {})
    data = payload.get("data", {})
    validation = payload.get("val", {}).get("data", {})
    if deployment != {
        "type": "single_node",
        "gpus_per_node": 6,
        "num_train_gpus": 6,
        "num_infer_gpus": 0,
    }:
        raise PrimeSFTPreflightError("Prime SFT must remain a one-node, six-training-rank job")
    required_model = {
        "impl": "custom",
        "seq_len": 4096,
        "cp": 1,
        "ep": 1,
        "optimization_dtype": "bfloat16",
        "reduce_dtype": "bfloat16",
    }
    if any(model.get(key) != value for key, value in required_model.items()):
        raise PrimeSFTPreflightError("Prime custom-Llama boundary/dtype contract changed")
    if tokenizer.get("trust_remote_code") is not False:
        raise PrimeSFTPreflightError("Prime tokenizer must refuse remote code")
    if renderer != {"name": CHAT_FORMAT_ID}:
        raise PrimeSFTPreflightError("Prime renderer contract changed")
    expected_loss_mask = {"system": False, "user": False, "assistant": True, "tool": False}
    for label, section, split in (
        ("training", data, "train"),
        ("validation", validation, "validation"),
    ):
        if section.get("type") != "sft" or section.get("splits") != [split]:
            raise PrimeSFTPreflightError(f"Prime {label} dataset split contract changed")
        if section.get("seq_len") != 4096 or section.get("micro_batch_size") != 1:
            raise PrimeSFTPreflightError(f"Prime {label} sequence/microbatch contract changed")
        if section.get("num_workers") != 1:
            raise PrimeSFTPreflightError(f"Prime {label} must use one worker per rank")
        if section.get("loss_mask") != expected_loss_mask:
            raise PrimeSFTPreflightError(f"Prime {label} assistant-only labels changed")
    if data.get("batch_size") != 48 or 48 % (6 * data["micro_batch_size"]) != 0:
        raise PrimeSFTPreflightError("Prime global batch must produce eight accumulation rounds")
    if data.get("shuffle") is not True or validation.get("shuffle") is not False:
        raise PrimeSFTPreflightError("Prime train/validation shuffle contract changed")
    if payload.get("log", {}).get("log_data") is not False:
        raise PrimeSFTPreflightError("Prime must not log training examples")
    dataset_root = Path(str(data.get("name", "")))
    validation_root = Path(str(validation.get("name", "")))
    model_root = Path(str(model.get("name", "")))
    tokenizer_root = Path(str(tokenizer.get("name", "")))
    if not dataset_root.is_absolute() or validation_root != dataset_root:
        raise PrimeSFTPreflightError("Prime train/validation dataset paths must be one absolute tree")
    if not model_root.is_absolute() or tokenizer_root != model_root:
        raise PrimeSFTPreflightError("Prime model/tokenizer paths must be one absolute HF export")
    if not dry_run:
        # The current checked-in hash cannot reach this branch. Keep the
        # invariant explicit so a future reviewed real-run contract cannot
        # accidentally lose the quality-approval gate.
        _positive_integer(payload.get("max_steps"), label="max_steps")
    return PrimeConfig(
        path=config_path,
        sha256=config_sha,
        payload=payload,
        dataset_root=dataset_root,
        model_root=model_root,
        dry_run=dry_run,
    )


def validate_hf_export(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise PrimeSFTPreflightError(f"Missing or unsafe HF model export: {root}")
    root = root.resolve(strict=True)
    manifest_path = root / EXPORT_MANIFEST_NAME
    manifest = _load_json(manifest_path, label="native HF export manifest")
    sidecar_path = root / EXPORT_MANIFEST_SIDECAR_NAME
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise PrimeSFTPreflightError(f"Missing or unsafe HF export sidecar: {sidecar_path}")
    manifest_sha = file_sha256(manifest_path)
    expected_sidecar = f"{manifest_sha}  {EXPORT_MANIFEST_NAME}\n"
    if sidecar_path.read_text(encoding="ascii") != expected_sidecar:
        raise PrimeSFTPreflightError("HF export manifest sidecar is invalid")
    if (
        manifest.get("manifest_version") != EXPORT_FORMAT_VERSION
        or manifest.get("format") != EXPORT_FORMAT
        or manifest.get("architecture") != "LlamaForCausalLM"
    ):
        raise PrimeSFTPreflightError("Unsupported native-to-HF export contract")

    file_records = manifest.get("files")
    weights = manifest.get("weights")
    tokenizer = manifest.get("tokenizer")
    native_config = manifest.get("native_model_config")
    if not isinstance(file_records, list) or not isinstance(weights, dict):
        raise PrimeSFTPreflightError("HF export has no authenticated file inventory")
    if not isinstance(tokenizer, dict) or not isinstance(native_config, dict):
        raise PrimeSFTPreflightError("HF export tokenizer/model identities are missing")
    if weights.get("format") != "safetensors":
        raise PrimeSFTPreflightError("HF export weights must be safetensors")
    weight_records = weights.get("files")
    if not isinstance(weight_records, list) or not weight_records:
        raise PrimeSFTPreflightError("HF export safetensors inventory is empty")
    names: set[str] = set()
    tensor_names: set[str] = set()
    total_weight_bytes = 0
    for index, record in enumerate(weight_records):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256", "tensors"}:
            raise PrimeSFTPreflightError(f"Invalid safetensors record {index}")
        path = _verify_file_record(root, record, label=f"weights.files[{index}]")
        if path.suffix != ".safetensors" or path.name in names:
            raise PrimeSFTPreflightError("HF safetensors paths are invalid or duplicated")
        tensors = record.get("tensors")
        if not isinstance(tensors, list) or not tensors or any(
            not isinstance(name, str) or not name for name in tensors
        ):
            raise PrimeSFTPreflightError(f"Invalid safetensors tensor inventory {index}")
        if tensor_names.intersection(tensors):
            raise PrimeSFTPreflightError("A tensor is assigned to multiple safetensors shards")
        tensor_names.update(tensors)
        names.add(path.name)
        total_weight_bytes += path.stat().st_size
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise PrimeSFTPreflightError(
                "HF export verification requires safetensors in the Prime environment"
            ) from exc
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                stored_tensors = set(handle.keys())
                metadata = handle.metadata()
        except Exception as exc:
            raise PrimeSFTPreflightError(f"Cannot open authenticated safetensors file {path}: {exc}") from exc
        if stored_tensors != set(tensors) or metadata != {"format": "pt"}:
            raise PrimeSFTPreflightError(f"Safetensors header inventory changed: {path}")
    for index, record in enumerate(file_records):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise PrimeSFTPreflightError(f"Invalid HF non-weight record {index}")
        path = _verify_file_record(root, record, label=f"files[{index}]")
        if path.name in names:
            raise PrimeSFTPreflightError("HF export file inventory contains a duplicate")
        names.add(path.name)
    expected_names = names | {EXPORT_MANIFEST_NAME, EXPORT_MANIFEST_SIDECAR_NAME}
    actual_names = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise PrimeSFTPreflightError(f"HF export contains an unsafe or non-file entry: {path}")
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise PrimeSFTPreflightError("HF export exact file tree differs from its manifest")

    is_sharded = weights.get("is_sharded")
    index_name = weights.get("index")
    if not isinstance(is_sharded, bool):
        raise PrimeSFTPreflightError("HF export sharding flag is invalid")
    if is_sharded != (len(weight_records) > 1):
        raise PrimeSFTPreflightError("HF export sharding inventory is inconsistent")
    if is_sharded:
        if index_name != "model.safetensors.index.json" or index_name not in names:
            raise PrimeSFTPreflightError("Sharded HF export has no authenticated index")
        index_payload = _load_json(root / index_name, label="safetensors index")
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != tensor_names:
            raise PrimeSFTPreflightError("Safetensors index tensor inventory changed")
        if set(weight_map.values()) != {record["path"] for record in weight_records}:
            raise PrimeSFTPreflightError("Safetensors index shard inventory changed")
    elif index_name is not None:
        raise PrimeSFTPreflightError("Unsharded HF export unexpectedly names an index")

    tokenizer_record = tokenizer.get("export_manifest")
    if not isinstance(tokenizer_record, dict) or set(tokenizer_record) != {
        "path",
        "bytes",
        "sha256",
        "format",
        "manifest_version",
    }:
        raise PrimeSFTPreflightError("HF tokenizer export-manifest record is invalid")
    if (
        tokenizer_record.get("path") != TOKENIZER_MANIFEST_NAME
        or tokenizer_record.get("format") != TOKENIZER_EXPORT_FORMAT
        or tokenizer_record.get("manifest_version") != TOKENIZER_EXPORT_MANIFEST_VERSION
    ):
        raise PrimeSFTPreflightError("HF tokenizer export-manifest identity changed")
    _verify_file_record(root, tokenizer_record, label="tokenizer.export_manifest")
    tokenizer_manifest = _load_json(root / TOKENIZER_MANIFEST_NAME, label="exported tokenizer manifest")
    if (
        tokenizer_manifest.get("manifest_version") != TOKENIZER_EXPORT_MANIFEST_VERSION
        or tokenizer_manifest.get("format") != TOKENIZER_EXPORT_FORMAT
    ):
        raise PrimeSFTPreflightError("Exported tokenizer manifest contract changed")
    tokenizer_files = tokenizer_manifest.get("files")
    tokenizer_validation = tokenizer_manifest.get("validation")
    if not isinstance(tokenizer_files, dict) or not isinstance(tokenizer_validation, dict):
        raise PrimeSFTPreflightError("Exported tokenizer manifest is incomplete")
    for name, record in tokenizer_files.items():
        if not isinstance(record, dict):
            raise PrimeSFTPreflightError(f"Invalid tokenizer file record: {name}")
        normalized = {"path": name, **record}
        _verify_file_record(root, normalized, label=f"tokenizer.files[{name}]")
    vocab_size = _positive_integer(tokenizer_validation.get("vocab_size"), label="tokenizer vocab_size")
    model_max_length = _positive_integer(
        tokenizer_validation.get("model_max_length"), label="tokenizer model_max_length"
    )
    special_ids = tokenizer_validation.get("special_token_ids")
    if not isinstance(special_ids, dict) or special_ids.get("eos_token_id") != 0:
        raise PrimeSFTPreflightError("Exported tokenizer EOS identity changed")
    if tokenizer.get("special_token_ids") != special_ids:
        raise PrimeSFTPreflightError("Outer and inner tokenizer special-token IDs disagree")
    vocab_sha = _sha256(tokenizer_validation.get("vocab_sha256"), label="tokenizer vocab_sha256")
    if tokenizer.get("vocab_sha256") != vocab_sha:
        raise PrimeSFTPreflightError("Outer and inner tokenizer vocabulary identities disagree")
    source_manifest = tokenizer.get("source_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != {
        "path",
        "bytes",
        "sha256",
        "declared_files_verified",
    }:
        raise PrimeSFTPreflightError("HF export has no authenticated source tokenizer manifest")
    if source_manifest.get("path") != SOURCE_TOKENIZER_MANIFEST_NAME:
        raise PrimeSFTPreflightError("HF source tokenizer manifest path changed")
    _positive_integer(
        source_manifest.get("declared_files_verified"),
        label="tokenizer.source_manifest.declared_files_verified",
    )
    _verify_file_record(root, source_manifest, label="tokenizer.source_manifest")
    if tokenizer.get("source_manifest_sha256") != source_manifest.get("sha256"):
        raise PrimeSFTPreflightError("HF source tokenizer manifest identities disagree")
    source_tokenizer_payload = _load_json(
        root / SOURCE_TOKENIZER_MANIFEST_NAME,
        label="source tokenizer manifest",
    )
    source_tokenizer_files = source_tokenizer_payload.get("files")
    if not isinstance(source_tokenizer_files, dict):
        raise PrimeSFTPreflightError("Source tokenizer manifest has no file inventory")
    source_tokenizer_json = source_tokenizer_files.get("tokenizer.json")
    if not isinstance(source_tokenizer_json, dict):
        raise PrimeSFTPreflightError("Source tokenizer manifest does not authenticate tokenizer.json")
    source_tokenizer_json_sha = _sha256(
        source_tokenizer_json.get("sha256"),
        label="source tokenizer tokenizer.json SHA-256",
    )
    inner_source = tokenizer_manifest.get("source")
    if not isinstance(inner_source, dict) or inner_source.get("manifest") != source_manifest:
        raise PrimeSFTPreflightError("Exported tokenizer source provenance changed")
    if inner_source.get("vocab_sha256") != vocab_sha:
        raise PrimeSFTPreflightError("Exported tokenizer source vocabulary identity changed")
    if tokenizer.get("vocab_size") != vocab_size or tokenizer.get("model_max_length") != model_max_length:
        raise PrimeSFTPreflightError("Outer and inner tokenizer manifests disagree")
    if native_config.get("vocab_size") != vocab_size or native_config.get("max_seq_len") != model_max_length:
        raise PrimeSFTPreflightError("HF export model and tokenizer geometry disagree")
    config_payload = _load_json(root / "config.json", label="HF config")
    if (
        config_payload.get("model_type") != "llama"
        or config_payload.get("vocab_size") != vocab_size
        or config_payload.get("max_position_embeddings") != model_max_length
        or config_payload.get("architectures") != ["LlamaForCausalLM"]
    ):
        raise PrimeSFTPreflightError("HF Llama config disagrees with the export manifest")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "vocab_size": vocab_size,
        "max_sequence_length": model_max_length,
        "weight_bytes": total_weight_bytes,
        "source_tokenizer_manifest_sha256": source_manifest["sha256"],
        "source_tokenizer_json_sha256": source_tokenizer_json_sha,
        "source_tokenizer_vocabulary_sha256": vocab_sha,
    }


def validate_tokenizer_binding(
    dataset: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, str]:
    """Prove curation and the exported checkpoint use identical tokenizer bytes."""

    dataset_tokenizer = dataset.get("identity", {}).get("tokenizer")
    if not isinstance(dataset_tokenizer, dict):
        raise PrimeSFTPreflightError("Curated dataset has no tokenizer identity")
    dataset_manifest_sha = _sha256(
        dataset_tokenizer.get("tokenizer_manifest_sha256"),
        label="dataset tokenizer manifest SHA-256",
    )
    dataset_json_sha = _sha256(
        dataset_tokenizer.get("tokenizer_json_sha256"),
        label="dataset tokenizer.json SHA-256",
    )
    model_manifest_sha = _sha256(
        model.get("source_tokenizer_manifest_sha256"),
        label="model source tokenizer manifest SHA-256",
    )
    model_json_sha = _sha256(
        model.get("source_tokenizer_json_sha256"),
        label="model source tokenizer.json SHA-256",
    )
    if dataset_manifest_sha != model_manifest_sha:
        raise PrimeSFTPreflightError(
            "Curated dataset and HF model export use different tokenizer manifests"
        )
    if dataset_json_sha != model_json_sha:
        raise PrimeSFTPreflightError(
            "Curated dataset and HF model export use different tokenizer.json bytes"
        )
    return {
        "tokenizer_manifest_sha256": dataset_manifest_sha,
        "tokenizer_json_sha256": dataset_json_sha,
    }


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PrimeSFTPreflightError(f"git {' '.join(args)} failed in {checkout}: {detail}")
    return result.stdout


def validate_prime_runtime(prime_rl_checkout: Path, contract: LaunchContract) -> dict[str, Any]:
    prime = Path(prime_rl_checkout).expanduser()
    if prime.is_symlink() or not prime.is_dir():
        raise PrimeSFTPreflightError(f"Missing or unsafe PrimeRL checkout: {prime}")
    prime = prime.resolve(strict=True)
    renderers = prime / "deps" / "renderers"
    if renderers.is_symlink() or not renderers.is_dir():
        raise PrimeSFTPreflightError(f"Missing or unsafe Prime renderer dependency: {renderers}")
    if _git(prime, "rev-parse", "HEAD").strip() != contract.prime_rl_commit:
        raise PrimeSFTPreflightError("PrimeRL checkout is not at the contracted commit")
    if _git(renderers, "rev-parse", "HEAD").strip() != contract.renderers_commit:
        raise PrimeSFTPreflightError("Renderer checkout is not at the contracted commit")
    try:
        patch_state = renderer_patch.install(
            renderers,
            prime_rl_checkout=prime,
            check_only=True,
        )
    except renderer_patch.IntegrationError as exc:
        raise PrimeSFTPreflightError(f"Prime renderer verification failed: {exc}") from exc
    if patch_state != "already-installed":
        raise PrimeSFTPreflightError(
            "The frozen renderer patch is valid but not installed; run apply_prime_renderer_patch.py first"
        )

    expected_renderer_status = {
        "renderers/__init__.py": " M",
        "renderers/base.py": " M",
        "renderers/configs.py": " M",
        "renderers/starcoder2_chat_format.py": "??",
        "renderers/starcoder2_coding.py": "??",
    }
    actual_renderer_status: dict[str, str] = {}
    for line in _git(renderers, "status", "--porcelain", "--untracked-files=all").splitlines():
        if len(line) < 4:
            raise PrimeSFTPreflightError("Cannot parse renderer Git status")
        actual_renderer_status[line[3:]] = line[:2]
    if actual_renderer_status != expected_renderer_status:
        raise PrimeSFTPreflightError("Renderer checkout contains changes beyond the frozen integration")
    prime_status = _git(
        prime,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).splitlines()
    if any(len(line) < 4 or line[3:] != "deps/renderers" for line in prime_status):
        raise PrimeSFTPreflightError("PrimeRL checkout contains changes outside deps/renderers")
    return {
        "prime_rl_checkout": str(prime),
        "renderers_checkout": str(renderers),
        "prime_rl_commit": contract.prime_rl_commit,
        "renderers_commit": contract.renderers_commit,
        "renderer_patch": patch_state,
    }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PrimeSFTPreflightError(f"No existing ancestor for cache path: {path}")
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise PrimeSFTPreflightError(f"Unsafe cache ancestor: {candidate}")
    return candidate


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def cache_environment(cache_root: Path) -> dict[str, str]:
    root = cache_root.resolve(strict=False)
    hf_home = root / "huggingface"
    return {
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _cache_inventory(datasets_cache: Path) -> list[dict[str, Any]]:
    if datasets_cache.is_symlink() or not datasets_cache.is_dir():
        raise PrimeSFTPreflightError(f"Missing or unsafe HF datasets cache: {datasets_cache}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(datasets_cache.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PrimeSFTPreflightError(f"HF datasets cache contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PrimeSFTPreflightError(f"HF datasets cache contains a non-file: {path}")
        inventory.append(
            {
                "path": path.relative_to(datasets_cache).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not inventory:
        raise PrimeSFTPreflightError("HF datasets cache is empty after prewarm")
    return inventory


def validate_prewarm_marker(
    cache_root: Path,
    *,
    config: PrimeConfig,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any] | None:
    marker_path = cache_root / PREWARM_MARKER
    if not marker_path.exists():
        return None
    marker = _load_json(marker_path, label="Prime SFT prewarm marker")
    expected_keys = {
        "format_version",
        "kind",
        "config_sha256",
        "dataset_manifest_sha256",
        "model_manifest_sha256",
        "cache_environment",
        "cache_inventory",
        "splits",
        "tokenizer",
        "completed_at_utc",
    }
    if set(marker) != expected_keys or marker.get("format_version") != 1 or marker.get("kind") != "prime_sft_hf_cache_prewarm":
        raise PrimeSFTPreflightError("Prime SFT prewarm marker contract is invalid")
    if marker.get("config_sha256") != config.sha256:
        raise PrimeSFTPreflightError("Prewarm marker belongs to another Prime config")
    if marker.get("dataset_manifest_sha256") != dataset["dataset_manifest_sha256"]:
        raise PrimeSFTPreflightError("Prewarm marker belongs to another curated dataset")
    if marker.get("model_manifest_sha256") != model["manifest_sha256"]:
        raise PrimeSFTPreflightError("Prewarm marker belongs to another HF model export")
    env = cache_environment(cache_root)
    if marker.get("cache_environment") != {
        "HF_HOME": env["HF_HOME"],
        "HF_DATASETS_CACHE": env["HF_DATASETS_CACHE"],
        "HF_HUB_CACHE": env["HF_HUB_CACHE"],
    }:
        raise PrimeSFTPreflightError("Prewarm marker cache paths changed")
    cache_inventory = marker.get("cache_inventory")
    if (
        not isinstance(cache_inventory, list)
        or cache_inventory != _cache_inventory(Path(env["HF_DATASETS_CACHE"]))
    ):
        raise PrimeSFTPreflightError("Prewarmed HF datasets cache inventory changed")
    expected_rows = {
        "train": dataset["completion"]["train_rows"],
        "validation": dataset["completion"]["validation_rows"],
    }
    splits = marker.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(expected_rows):
        raise PrimeSFTPreflightError("Prewarm marker split inventory changed")
    for split, rows in expected_rows.items():
        record = splits.get(split)
        if (
            not isinstance(record, dict)
            or set(record) != {"rows", "fingerprint"}
            or record.get("rows") != rows
            or not isinstance(record.get("fingerprint"), str)
            or not record["fingerprint"]
        ):
            raise PrimeSFTPreflightError(f"Prewarm marker {split} record is invalid")
    if marker.get("tokenizer") != {
        "vocab_size": model["vocab_size"],
        "model_max_length": model["max_sequence_length"],
    }:
        raise PrimeSFTPreflightError("Prewarm tokenizer identity changed")
    return marker


def inspect_cache(
    cache_root: Path,
    *,
    contract: LaunchContract,
    config: PrimeConfig,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    prime_root: Path,
) -> CachePlan:
    cache_root = Path(cache_root).expanduser()
    if not cache_root.is_absolute():
        raise PrimeSFTPreflightError("Hugging Face cache root must be absolute")
    if cache_root.is_symlink():
        raise PrimeSFTPreflightError(f"Hugging Face cache root cannot be a symlink: {cache_root}")
    cache_root = cache_root.resolve(strict=False)
    for protected in (config.dataset_root.resolve(), config.model_root.resolve(), prime_root.resolve()):
        if _paths_overlap(cache_root, protected):
            raise PrimeSFTPreflightError(
                f"Hugging Face cache must be separate from authenticated input: {protected}"
            )
    ancestor = _nearest_existing_directory(cache_root)
    marker = validate_prewarm_marker(
        cache_root,
        config=config,
        dataset=dataset,
        model=model,
    ) if cache_root.is_dir() else None
    required_unwarmed = max(
        contract.cache_minimum_free_bytes,
        int(dataset["data_bytes"]) * contract.cache_dataset_size_multiplier
        + contract.cache_reserve_bytes,
    )
    required = contract.cache_reserve_bytes if marker is not None else required_unwarmed
    available = shutil.disk_usage(ancestor).free
    if available < required:
        raise PrimeSFTPreflightError(
            f"Hugging Face cache filesystem has {available / GIB:.1f} GiB free; "
            f"requires at least {required / GIB:.1f} GiB"
        )
    return CachePlan(
        root=cache_root,
        hf_home=cache_root / "huggingface",
        datasets_cache=cache_root / "huggingface" / "datasets",
        required_free_bytes=required,
        available_free_bytes=available,
        prewarmed=marker is not None,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prewarm_hf_cache(
    plan: CachePlan,
    *,
    config: PrimeConfig,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    if plan.prewarmed:
        marker = validate_prewarm_marker(
            plan.root,
            config=config,
            dataset=dataset,
            model=model,
        )
        assert marker is not None
        return marker
    plan.datasets_cache.mkdir(parents=True, exist_ok=True)
    (plan.hf_home / "hub").mkdir(parents=True, exist_ok=True)
    env = cache_environment(plan.root)
    os.environ.update(env)
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PrimeSFTPreflightError(
            "Prewarm requires datasets and transformers in the Prime environment"
        ) from exc
    split_records: dict[str, dict[str, Any]] = {}
    expected_rows = {
        "train": dataset["completion"]["train_rows"],
        "validation": dataset["completion"]["validation_rows"],
    }
    for split, rows in expected_rows.items():
        loaded = load_dataset(
            str(config.dataset_root),
            None,
            split=split,
            cache_dir=env["HF_DATASETS_CACHE"],
        )
        if len(loaded) != rows:
            raise PrimeSFTPreflightError(
                f"Prewarmed {split} has {len(loaded):,} rows; expected {rows:,}"
            )
        fingerprint = getattr(loaded, "_fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise PrimeSFTPreflightError(f"Prewarmed {split} has no HF fingerprint")
        split_records[split] = {"rows": len(loaded), "fingerprint": fingerprint}
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != model["vocab_size"] or tokenizer.model_max_length != model["max_sequence_length"]:
        raise PrimeSFTPreflightError("Prewarmed tokenizer disagrees with its authenticated manifest")
    marker = {
        "format_version": 1,
        "kind": "prime_sft_hf_cache_prewarm",
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "cache_environment": {
            "HF_HOME": env["HF_HOME"],
            "HF_DATASETS_CACHE": env["HF_DATASETS_CACHE"],
            "HF_HUB_CACHE": env["HF_HUB_CACHE"],
        },
        "cache_inventory": _cache_inventory(plan.datasets_cache),
        "splits": split_records,
        "tokenizer": {
            "vocab_size": model["vocab_size"],
            "model_max_length": model["max_sequence_length"],
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(plan.root / PREWARM_MARKER, marker)
    validated = validate_prewarm_marker(
        plan.root,
        config=config,
        dataset=dataset,
        model=model,
    )
    assert validated is not None
    return validated


def validate_training_approval(
    approval_path: Path | None,
    *,
    config: PrimeConfig,
    dataset: Mapping[str, Any],
) -> dict[str, Any] | None:
    policy = dataset["manifest"]["identity"]["policy"]
    threshold = policy["filter"]["min_average_test_score"]
    if config.dry_run:
        return None
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise PrimeSFTPreflightError(
            "Real SFT is blocked: the curated dataset still has the provisional "
            "min_average_test_score=0.0 policy"
        )
    if approval_path is None:
        raise PrimeSFTPreflightError("Real SFT requires --training-approval")
    approval_path = Path(approval_path).expanduser()
    if approval_path.is_symlink() or not approval_path.is_file():
        raise PrimeSFTPreflightError(f"Missing or unsafe Prime SFT training approval: {approval_path}")
    approval_path = approval_path.resolve(strict=True)
    approval = _load_json(approval_path, label="Prime SFT training approval")
    if set(approval) != {
        "format_version",
        "kind",
        "status",
        "dataset_manifest_sha256",
        "curation_identity_sha256",
        "policy_sha256",
        "quality_audit",
        "decision",
        "approved_at_utc",
        "approval_sha256",
    }:
        raise PrimeSFTPreflightError("Prime SFT training approval fields changed")
    body = dict(approval)
    recorded_sha = _sha256(body.pop("approval_sha256", None), label="approval_sha256")
    if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != recorded_sha:
        raise PrimeSFTPreflightError("Prime SFT training approval digest is invalid")
    if approval.get("format_version") != 1 or approval.get("kind") != "prime_sft_training_approval" or approval.get("status") != "approved":
        raise PrimeSFTPreflightError("Prime SFT training approval status is invalid")
    try:
        approved_at = datetime.fromisoformat(str(approval.get("approved_at_utc")))
    except ValueError as exc:
        raise PrimeSFTPreflightError("Prime SFT approval timestamp is invalid") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        raise PrimeSFTPreflightError("Prime SFT approval timestamp must be timezone-aware")
    identity = dataset["identity"]
    if (
        approval.get("dataset_manifest_sha256") != dataset["dataset_manifest_sha256"]
        or approval.get("curation_identity_sha256") != dataset["curation_identity_sha256"]
        or approval.get("policy_sha256") != identity["policy_sha256"]
    ):
        raise PrimeSFTPreflightError("Prime SFT training approval belongs to another dataset/policy")
    decision = approval.get("decision")
    if decision != {
        "min_average_test_score": threshold,
        "accepted_train_rows": dataset["completion"]["train_rows"],
    }:
        raise PrimeSFTPreflightError("Prime SFT approval decision disagrees with the curated artifact")
    audit = approval.get("quality_audit")
    if not isinstance(audit, dict) or set(audit) != {"path", "bytes", "sha256"}:
        raise PrimeSFTPreflightError("Prime SFT quality-audit record is invalid")
    audit_name = _simple_name(audit.get("path"), label="quality_audit.path")
    audit_path = _verify_file_record(
        approval_path.parent,
        {**audit, "path": audit_name},
        label="quality_audit",
    )
    audit_payload = _load_json(audit_path, label="OpenCodeInstruct quality audit")
    if set(audit_payload) != {
        "format_version",
        "kind",
        "source_repo_id",
        "source_revision",
        "score_field",
        "source_rows",
        "measurement",
        "created_at_utc",
    }:
        raise PrimeSFTPreflightError("OpenCodeInstruct quality-audit fields changed")
    manifest_identity = dataset["manifest"]["identity"]
    if (
        audit_payload.get("format_version") != 1
        or audit_payload.get("kind") != "opencodeinstruct_quality_audit"
        or audit_payload.get("source_repo_id") != manifest_identity.get("source_repo_id")
        or audit_payload.get("source_revision") != manifest_identity.get("source_revision")
        or audit_payload.get("score_field") != "average_test_score"
        or audit_payload.get("source_rows") != dataset["completion"].get("source_rows")
        or not isinstance(audit_payload.get("measurement"), dict)
        or not audit_payload["measurement"]
    ):
        raise PrimeSFTPreflightError("OpenCodeInstruct quality-audit identity is invalid")
    try:
        audit_created_at = datetime.fromisoformat(str(audit_payload.get("created_at_utc")))
    except ValueError as exc:
        raise PrimeSFTPreflightError("OpenCodeInstruct quality-audit timestamp is invalid") from exc
    if audit_created_at.tzinfo is None or audit_created_at.utcoffset() is None:
        raise PrimeSFTPreflightError("OpenCodeInstruct quality-audit timestamp must be timezone-aware")
    return approval


def build_launch_command(prime_root: Path, config: PrimeConfig) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise PrimeSFTPreflightError("uv is required to execute pinned PrimeRL")
    return [uv, "--directory", str(prime_root), "run", "sft", "@", str(config.path)]


def execute_prime(
    prime_root: Path,
    config: PrimeConfig,
    plan: CachePlan,
    *,
    execve: Callable[[str, list[str], Mapping[str, str]], NoReturn] = os.execve,
) -> NoReturn:
    command = build_launch_command(prime_root, config)
    environment = dict(os.environ)
    environment.update(cache_environment(plan.root))
    execve(command[0], command, environment)
    raise AssertionError("os.execve unexpectedly returned")


def run(
    *,
    mode: str,
    prime_rl_checkout: Path,
    config_path: Path | None = None,
    cache_root: Path | None = None,
    training_approval: Path | None = None,
    execve: Callable[[str, list[str], Mapping[str, str]], NoReturn] = os.execve,
) -> dict[str, Any] | NoReturn:
    if mode not in {"verify-only", "prewarm", "launch"}:
        raise PrimeSFTPreflightError(f"Unsupported mode: {mode}")
    contract = load_launch_contract()
    config = load_prime_config(contract, config_path)
    try:
        dataset = validate_published_dataset(config.dataset_root)
    except SFTPreparationError as exc:
        raise PrimeSFTPreflightError(f"Curated SFT dataset verification failed: {exc}") from exc
    model = validate_hf_export(config.model_root)
    validate_tokenizer_binding(dataset, model)
    if dataset["manifest"]["identity"]["max_sequence_length"] != model["max_sequence_length"]:
        raise PrimeSFTPreflightError("Curated dataset and HF model context lengths disagree")
    if model["max_sequence_length"] != config.payload["model"]["seq_len"]:
        raise PrimeSFTPreflightError("HF model and Prime config context lengths disagree")
    runtime = validate_prime_runtime(prime_rl_checkout, contract)
    validate_training_approval(training_approval, config=config, dataset=dataset)
    selected_cache = cache_root or contract.cache_default_root
    plan = inspect_cache(
        selected_cache,
        contract=contract,
        config=config,
        dataset=dataset,
        model=model,
        prime_root=Path(runtime["prime_rl_checkout"]),
    )
    report = {
        "mode": mode,
        "dry_run": config.dry_run,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "prime_rl_commit": runtime["prime_rl_commit"],
        "renderers_commit": runtime["renderers_commit"],
        "cache_root": str(plan.root),
        "cache_prewarmed": plan.prewarmed,
        "cache_free_bytes": plan.available_free_bytes,
        "cache_required_free_bytes": plan.required_free_bytes,
    }
    if mode == "verify-only":
        return report
    if mode == "prewarm":
        prewarm_hf_cache(plan, config=config, dataset=dataset, model=model)
        report["cache_prewarmed"] = True
        return report
    if not plan.prewarmed:
        raise PrimeSFTPreflightError(
            "Six-rank launch requires a completed one-process prewarm; run --mode prewarm first"
        )
    return execute_prime(
        Path(runtime["prime_rl_checkout"]),
        config,
        plan,
        execve=execve,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-only", "prewarm", "launch"), required=True)
    parser.add_argument("--prime-rl-checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--training-approval", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            mode=args.mode,
            prime_rl_checkout=args.prime_rl_checkout,
            config_path=args.config,
            cache_root=args.cache_root,
            training_approval=args.training_approval,
        )
    except PrimeSFTPreflightError as exc:
        print(f"Prime SFT preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
