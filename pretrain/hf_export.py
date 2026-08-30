"""Export native pretraining checkpoints as Hugging Face Llama models.

The native architecture in :mod:`model` is parameter-compatible with
``transformers.LlamaForCausalLM``.  This module deliberately performs a
closed-world conversion: every expected native key and tensor shape must be
present, no extra key is accepted, and the tokenizer vocabulary must match the
model vocabulary exactly.

PyTorch checkpoint files are pickle containers.  Only export checkpoints
created by this experiment or another trusted source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model import ModelConfig
from pretrain.tokenizer_identity import (
    TOKENIZER_MANIFEST_NAME,
    TokenizerIdentityError,
    require_sha256,
    verify_tokenizer_identity,
    vocabulary_sha256,
)


EXPORT_MANIFEST_NAME = "NATIVE_EXPORT_MANIFEST.json"
EXPORT_MANIFEST_SIDECAR_NAME = "NATIVE_EXPORT_MANIFEST.sha256"
EXPORT_FORMAT = "native-pytorch-to-hf-llama"
EXPORT_FORMAT_VERSION = 1
SUPPORTED_CHECKPOINT_FORMAT = "native-pytorch-pretrain"
SUPPORTED_CHECKPOINT_VERSIONS = frozenset({4, 5})
SOURCE_TOKENIZER_MANIFEST_NAME = "SOURCE_TOKENIZER_MANIFEST.json"
TOKENIZER_EXPORT_FORMAT = "transformers-tokenizer-export"
TOKENIZER_EXPORT_MANIFEST_VERSION = 1


class ExportError(ValueError):
    """Raised when an input cannot be safely exported."""


@dataclass(frozen=True)
class LoadedNativeCheckpoint:
    """Validated checkpoint payload needed by the exporter."""

    state_dict: Mapping[str, torch.Tensor]
    config: ModelConfig
    source_kind: str
    checkpoint_format: str | None
    checkpoint_format_version: int | None
    tokenizer_manifest_sha256: str
    tokenizer_vocabulary_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    """Durably flush one export artifact without following a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExportError(f"Cannot open export artifact for fsync {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExportError(f"Export artifact is not a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_export_tree(root: Path) -> None:
    """Flush every flat HF export artifact before publishing its directory."""

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ExportError(f"HF export contains an unsafe or non-file artifact: {path}")
        _fsync_regular_file(path)
    _fsync_directory(root)


def _strict_model_config(
    value: ModelConfig | Mapping[str, Any], *, label: str
) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    if not isinstance(value, Mapping):
        raise ExportError(f"{label} must be a ModelConfig or mapping")
    if any(not isinstance(name, str) for name in value):
        raise ExportError(f"{label} field names must be strings")

    expected = {field.name for field in dataclasses.fields(ModelConfig)}
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ExportError(f"{label} fields do not match ModelConfig: {'; '.join(details)}")

    normalized = dict(value)
    integer_fields = {
        "vocab_size",
        "dim",
        "hidden_dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "max_seq_len",
        "loss_chunk_size",
    }
    for name in integer_fields:
        field_value = normalized[name]
        if isinstance(field_value, bool) or not isinstance(field_value, int):
            raise ExportError(f"{label}.{name} must be an integer")
    for name in ("norm_eps", "rope_theta", "initializer_range"):
        field_value = normalized[name]
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise ExportError(f"{label}.{name} must be numeric")
        normalized[name] = float(field_value)
    if not isinstance(normalized["tie_word_embeddings"], bool):
        raise ExportError(f"{label}.tie_word_embeddings must be boolean")
    if not isinstance(normalized["activation_checkpointing"], bool):
        raise ExportError(f"{label}.activation_checkpointing must be boolean")
    if not isinstance(normalized["attention_backend"], str):
        raise ExportError(f"{label}.attention_backend must be a string")
    try:
        return ModelConfig(**normalized)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"Invalid {label}: {exc}") from exc


def load_model_config_json(path: str | Path) -> ModelConfig:
    """Load an exact native ``ModelConfig`` JSON document."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Cannot read model config {config_path}: {exc}") from exc
    if isinstance(payload, Mapping) and set(payload) == {"model_config"}:
        payload = payload["model_config"]
    return _strict_model_config(payload, label=f"model config {config_path}")


def _is_tensor_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
    )


def load_native_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    expected_tokenizer_manifest_sha256: str | None = None,
    expected_tokenizer_vocabulary_sha256: str | None = None,
) -> LoadedNativeCheckpoint:
    """Load a full native checkpoint or a directly saved state dictionary.

    A version-5 trainer checkpoint embeds both tokenizer identities. A legacy
    checkpoint or direct state dictionary is unbound, so the caller must
    provide both expected tokenizer identities out of band. A direct state
    dictionary also requires ``model_config``.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ExportError(f"Checkpoint is not a regular file: {path}")
    supplied_config = (
        None
        if model_config is None
        else _strict_model_config(model_config, label="supplied model config")
    )
    if (expected_tokenizer_manifest_sha256 is None) != (
        expected_tokenizer_vocabulary_sha256 is None
    ):
        raise ExportError(
            "Expected tokenizer manifest and vocabulary SHA-256 values must be "
            "supplied together"
        )
    if expected_tokenizer_manifest_sha256 is not None:
        try:
            expected_tokenizer_manifest_sha256 = require_sha256(
                expected_tokenizer_manifest_sha256,
                field="expected tokenizer manifest SHA-256",
            )
            expected_tokenizer_vocabulary_sha256 = require_sha256(
                expected_tokenizer_vocabulary_sha256,
                field="expected tokenizer vocabulary SHA-256",
            )
        except TokenizerIdentityError as exc:
            raise ExportError(str(exc)) from exc

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("Install safetensors to load this checkpoint") from exc
        try:
            root: Any = load_file(str(path), device="cpu")
        except Exception as exc:
            raise ExportError(f"Cannot load safetensors checkpoint {path}: {exc}") from exc
    else:
        # Native pretraining checkpoints include Python and NumPy RNG objects,
        # so weights_only=True cannot read the full payload. mmap avoids a
        # second eager copy of the multi-gigabyte checkpoint.
        try:
            root = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except Exception as exc:
            raise ExportError(f"Cannot load checkpoint {path}: {exc}") from exc

    checkpoint_format: str | None = None
    checkpoint_version: int | None = None
    embedded_config: ModelConfig | None = None
    embedded_manifest_sha256: str | None = None
    embedded_vocabulary_sha256: str | None = None
    if _is_tensor_state_dict(root):
        state_dict = root
        source_kind = "direct_state_dict"
    elif isinstance(root, Mapping) and "model" in root:
        state_dict = root["model"]
        if not _is_tensor_state_dict(state_dict):
            raise ExportError("Checkpoint payload['model'] is not a non-empty tensor state dict")
        source_kind = "checkpoint_model_payload"
        raw_format = root.get("format")
        raw_version = root.get("format_version")
        if raw_format is not None:
            if raw_format != SUPPORTED_CHECKPOINT_FORMAT:
                raise ExportError(f"Unsupported checkpoint format {raw_format!r}")
            if isinstance(raw_version, bool) or not isinstance(raw_version, int):
                raise ExportError("Checkpoint format_version must be an integer")
            if raw_version not in SUPPORTED_CHECKPOINT_VERSIONS:
                raise ExportError(
                    f"Unsupported {raw_format!r} version {raw_version}; supported versions are "
                    f"{sorted(SUPPORTED_CHECKPOINT_VERSIONS)}"
                )
            checkpoint_format = raw_format
            checkpoint_version = raw_version
        elif raw_version is not None:
            raise ExportError("Checkpoint has format_version but no format")
        if "model_config" in root:
            embedded_config = _strict_model_config(
                root["model_config"], label="embedded checkpoint model_config"
            )
        if checkpoint_version == 5:
            try:
                embedded_manifest_sha256 = require_sha256(
                    root.get("tokenizer_manifest_sha256"),
                    field="checkpoint tokenizer_manifest_sha256",
                )
                embedded_vocabulary_sha256 = require_sha256(
                    root.get("tokenizer_vocabulary_sha256"),
                    field="checkpoint tokenizer_vocabulary_sha256",
                )
            except TokenizerIdentityError as exc:
                raise ExportError(str(exc)) from exc
        elif checkpoint_version == 4:
            try:
                if root.get("tokenizer_manifest_sha256") is not None:
                    embedded_manifest_sha256 = require_sha256(
                        root.get("tokenizer_manifest_sha256"),
                        field="legacy checkpoint tokenizer_manifest_sha256",
                    )
                if root.get("tokenizer_vocabulary_sha256") is not None:
                    embedded_vocabulary_sha256 = require_sha256(
                        root.get("tokenizer_vocabulary_sha256"),
                        field="legacy checkpoint tokenizer_vocabulary_sha256",
                    )
            except TokenizerIdentityError as exc:
                raise ExportError(str(exc)) from exc
    else:
        raise ExportError(
            "Checkpoint root must be a direct tensor state dict or contain payload['model']"
        )

    if embedded_config is not None and supplied_config is not None:
        if dataclasses.asdict(embedded_config) != dataclasses.asdict(supplied_config):
            raise ExportError("Supplied model config disagrees with checkpoint model_config")
    resolved_config = embedded_config or supplied_config
    if resolved_config is None:
        raise ExportError(
            "Checkpoint does not embed model_config; supply the exact native model config"
        )
    if checkpoint_version == 5:
        assert embedded_manifest_sha256 is not None
        assert embedded_vocabulary_sha256 is not None
        if expected_tokenizer_manifest_sha256 is not None and (
            embedded_manifest_sha256 != expected_tokenizer_manifest_sha256
            or embedded_vocabulary_sha256 != expected_tokenizer_vocabulary_sha256
        ):
            raise ExportError(
                "Explicit tokenizer identity disagrees with version-5 checkpoint metadata"
            )
        resolved_manifest_sha256 = embedded_manifest_sha256
        resolved_vocabulary_sha256 = embedded_vocabulary_sha256
    else:
        if expected_tokenizer_manifest_sha256 is None:
            raise ExportError(
                "Checkpoint is not tokenizer-bound format version 5; supply explicit "
                "authenticated expected tokenizer manifest and vocabulary SHA-256 values"
            )
        assert expected_tokenizer_vocabulary_sha256 is not None
        if (
            embedded_manifest_sha256 is not None
            and embedded_manifest_sha256 != expected_tokenizer_manifest_sha256
        ):
            raise ExportError(
                "Explicit tokenizer manifest identity disagrees with legacy "
                "checkpoint metadata"
            )
        if (
            embedded_vocabulary_sha256 is not None
            and embedded_vocabulary_sha256 != expected_tokenizer_vocabulary_sha256
        ):
            raise ExportError(
                "Explicit tokenizer vocabulary identity disagrees with legacy "
                "checkpoint metadata"
            )
        resolved_manifest_sha256 = expected_tokenizer_manifest_sha256
        resolved_vocabulary_sha256 = expected_tokenizer_vocabulary_sha256
    return LoadedNativeCheckpoint(
        state_dict=state_dict,
        config=resolved_config,
        source_kind=source_kind,
        checkpoint_format=checkpoint_format,
        checkpoint_format_version=checkpoint_version,
        tokenizer_manifest_sha256=resolved_manifest_sha256,
        tokenizer_vocabulary_sha256=resolved_vocabulary_sha256,
    )


def native_to_hf_key_map(config: ModelConfig) -> dict[str, str]:
    """Return the complete, deterministic native-to-Llama parameter map."""

    mapping = {"tok_embeddings.weight": "model.embed_tokens.weight"}
    for index in range(config.n_layers):
        native = f"layers.{index}"
        hf = f"model.layers.{index}"
        mapping.update(
            {
                f"{native}.attention.q_proj.weight": f"{hf}.self_attn.q_proj.weight",
                f"{native}.attention.k_proj.weight": f"{hf}.self_attn.k_proj.weight",
                f"{native}.attention.v_proj.weight": f"{hf}.self_attn.v_proj.weight",
                f"{native}.attention.o_proj.weight": f"{hf}.self_attn.o_proj.weight",
                f"{native}.feed_forward.gate_proj.weight": f"{hf}.mlp.gate_proj.weight",
                f"{native}.feed_forward.up_proj.weight": f"{hf}.mlp.up_proj.weight",
                f"{native}.feed_forward.down_proj.weight": f"{hf}.mlp.down_proj.weight",
                f"{native}.attention_norm.weight": f"{hf}.input_layernorm.weight",
                f"{native}.ffn_norm.weight": f"{hf}.post_attention_layernorm.weight",
            }
        )
    mapping.update({"norm.weight": "model.norm.weight", "lm_head.weight": "lm_head.weight"})
    return mapping


def _native_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "tok_embeddings.weight": (config.vocab_size, config.dim),
        "norm.weight": (config.dim,),
        "lm_head.weight": (config.vocab_size, config.dim),
    }
    kv_dim = config.n_kv_heads * config.head_dim
    for index in range(config.n_layers):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.attention.q_proj.weight": (config.dim, config.dim),
                f"{prefix}.attention.k_proj.weight": (kv_dim, config.dim),
                f"{prefix}.attention.v_proj.weight": (kv_dim, config.dim),
                f"{prefix}.attention.o_proj.weight": (config.dim, config.dim),
                f"{prefix}.feed_forward.gate_proj.weight": (
                    config.hidden_dim,
                    config.dim,
                ),
                f"{prefix}.feed_forward.up_proj.weight": (
                    config.hidden_dim,
                    config.dim,
                ),
                f"{prefix}.feed_forward.down_proj.weight": (
                    config.dim,
                    config.hidden_dim,
                ),
                f"{prefix}.attention_norm.weight": (config.dim,),
                f"{prefix}.ffn_norm.weight": (config.dim,),
            }
        )
    return shapes


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.layout != torch.strided or right.layout != torch.strided:
        return False
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
    )


def validate_native_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: ModelConfig,
    *,
    check_finite: bool = True,
) -> torch.dtype:
    """Validate the exact key, shape, layout, dtype, and tying contract."""

    expected_shapes = _native_shapes(config)
    actual_keys = set(state_dict)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if extra:
            details.append(f"unexpected keys {extra}")
        raise ExportError(f"Native state dict does not match ModelConfig: {'; '.join(details)}")

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    dtypes: set[torch.dtype] = set()
    for name, expected_shape in expected_shapes.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ExportError(f"State value {name!r} is not a tensor")
        if tensor.device.type == "meta":
            raise ExportError(f"State tensor {name!r} is still on the meta device")
        if tensor.layout != torch.strided:
            raise ExportError(f"State tensor {name!r} must use strided layout")
        if tuple(tensor.shape) != expected_shape:
            raise ExportError(
                f"State tensor {name!r} has shape {tuple(tensor.shape)}, expected {expected_shape}"
            )
        if tensor.dtype not in supported_dtypes:
            raise ExportError(
                f"State tensor {name!r} has unsupported dtype {tensor.dtype}; expected one of "
                f"{sorted(str(dtype) for dtype in supported_dtypes)}"
            )
        dtypes.add(tensor.dtype)
        if check_finite and not bool(torch.isfinite(tensor).all()):
            raise ExportError(f"State tensor {name!r} contains NaN or infinity")
    if len(dtypes) != 1:
        raise ExportError(
            f"All native parameters must share one dtype; found {sorted(str(value) for value in dtypes)}"
        )

    embeddings = state_dict["tok_embeddings.weight"]
    output = state_dict["lm_head.weight"]
    shares_storage = _same_tensor_storage(embeddings, output)
    if config.tie_word_embeddings:
        if not shares_storage and not torch.equal(embeddings, output):
            raise ExportError(
                "tie_word_embeddings=True but input and output embedding weights differ"
            )
    elif shares_storage:
        raise ExportError(
            "tie_word_embeddings=False but input and output embedding weights share storage"
        )

    owners: dict[tuple[str, int], str] = {}
    for name in expected_shapes:
        tensor = state_dict[name]
        identity = _storage_identity(tensor)
        if identity is None:
            continue
        previous = owners.get(identity)
        if previous is None:
            owners[identity] = name
            continue
        allowed_tied_pair = config.tie_word_embeddings and {
            previous,
            name,
        } == {"tok_embeddings.weight", "lm_head.weight"}
        if not allowed_tied_pair:
            raise ExportError(
                f"Native parameters {previous!r} and {name!r} unexpectedly share storage"
            )
    return next(iter(dtypes))


def map_native_state_dict(
    state_dict: Mapping[str, torch.Tensor], config: ModelConfig
) -> dict[str, torch.Tensor]:
    """Map a previously validated native state dict into HF Llama names."""

    mapping = native_to_hf_key_map(config)
    return {hf_name: state_dict[native_name] for native_name, hf_name in mapping.items()}


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int] | None:
    if tensor.layout != torch.strided:
        return None
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _cpu_contiguous_without_shared_storage(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prepare tensors for safetensors without copying ordinary CPU weights."""

    result: dict[str, torch.Tensor] = {}
    seen_storage: set[tuple[str, int]] = set()
    for name, source in state_dict.items():
        tensor = source.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        identity = _storage_identity(tensor)
        if identity is not None and identity in seen_storage:
            # safetensors intentionally rejects aliased storage. This only
            # duplicates tied/shared weights in the serialized representation;
            # HF restores the configured tying when it loads the model.
            tensor = tensor.clone()
            identity = _storage_identity(tensor)
        if identity is not None:
            seen_storage.add(identity)
        result[name] = tensor
    return result


def _validate_vocab(vocabulary: Mapping[str, int], *, expected_size: int, label: str) -> None:
    if len(vocabulary) != expected_size:
        raise ExportError(
            f"{label} has {len(vocabulary)} tokens, but the model has {expected_size} embeddings"
        )
    identifiers = list(vocabulary.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in identifiers):
        raise ExportError(f"{label} contains a non-integer token ID")
    if set(identifiers) != set(range(expected_size)):
        raise ExportError(f"{label} token IDs must be exactly 0..{expected_size - 1}")


def _vocab_sha256(vocabulary: Mapping[str, int]) -> str:
    try:
        return vocabulary_sha256(vocabulary)
    except TokenizerIdentityError as exc:
        raise ExportError(str(exc)) from exc


def _file_integrity(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_and_verify_source_tokenizer_manifest(
    tokenizer_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_vocabulary_sha256: str,
    expected_vocab_size: int,
) -> tuple[bytes, Mapping[str, Any]]:
    """Load a source manifest only after verifying every declared source file.

    The source manifest is provenance for the input tokenizer. It must never be
    reused as the integrity manifest for the derived HF export because
    ``save_pretrained`` rewrites tokenizer metadata such as model_max_length.
    """

    try:
        identity = verify_tokenizer_identity(
            tokenizer_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_vocabulary_sha256=expected_vocabulary_sha256,
            expected_vocab_size=expected_vocab_size,
        )
    except TokenizerIdentityError as exc:
        raise ExportError(str(exc)) from exc
    return identity.manifest_bytes, identity.manifest


def _save_and_validate_tokenizer(
    tokenizer_path: Path,
    destination: Path,
    *,
    config: ModelConfig,
    expected_manifest_sha256: str,
    expected_vocabulary_sha256: str,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Install transformers to export the tokenizer") from exc

    if not tokenizer_path.is_dir():
        raise ExportError(f"Tokenizer path is not a directory: {tokenizer_path}")
    source_manifest_record = _load_and_verify_source_tokenizer_manifest(
        tokenizer_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_vocabulary_sha256=expected_vocabulary_sha256,
        expected_vocab_size=config.vocab_size,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, ValueError) as exc:
        raise ExportError(f"Cannot load local tokenizer {tokenizer_path}: {exc}") from exc
    source_vocab = tokenizer.get_vocab()
    _validate_vocab(
        source_vocab,
        expected_size=config.vocab_size,
        label=f"Tokenizer {tokenizer_path}",
    )
    source_vocab_sha256 = _vocab_sha256(source_vocab)
    if source_vocab_sha256 != expected_vocabulary_sha256:
        raise ExportError(
            "Transformers tokenizer vocabulary disagrees with the authenticated "
            "tokenizer.json vocabulary"
        )
    source_special_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    source_model_max_length = tokenizer.model_max_length

    # Align tokenizer metadata with the model context without adding a pad or
    # any other token. The vocabulary itself remains byte-for-byte equivalent
    # under the token->ID hash below.
    tokenizer.model_max_length = config.max_seq_len
    tokenizer.save_pretrained(str(destination))

    try:
        reloaded = AutoTokenizer.from_pretrained(
            str(destination),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, ValueError) as exc:
        raise ExportError(f"Exported tokenizer cannot be reloaded: {exc}") from exc
    exported_vocab = reloaded.get_vocab()
    _validate_vocab(
        exported_vocab,
        expected_size=config.vocab_size,
        label="Exported tokenizer",
    )
    if exported_vocab != source_vocab or _vocab_sha256(exported_vocab) != source_vocab_sha256:
        raise ExportError("Tokenizer vocabulary changed while saving the HF export")
    exported_special_ids = {
        "bos_token_id": reloaded.bos_token_id,
        "eos_token_id": reloaded.eos_token_id,
        "pad_token_id": reloaded.pad_token_id,
        "unk_token_id": reloaded.unk_token_id,
    }
    if exported_special_ids != source_special_ids:
        raise ExportError("Tokenizer special-token IDs changed while saving the HF export")
    if reloaded.model_max_length != config.max_seq_len:
        raise ExportError("Exported tokenizer model_max_length does not match ModelConfig")

    # Detect a concurrent source mutation across the load/save window before
    # publishing any model derived from those bytes.
    try:
        final_source_identity = verify_tokenizer_identity(
            tokenizer_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_vocabulary_sha256=expected_vocabulary_sha256,
            expected_vocab_size=config.vocab_size,
        )
    except TokenizerIdentityError as exc:
        raise ExportError(f"Tokenizer source changed during export: {exc}") from exc

    exported_files: dict[str, dict[str, Any]] = {}
    for path in sorted(destination.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            raise ExportError(f"Tokenizer export created a non-file artifact: {path}")
        exported_files[path.name] = _file_integrity(path)

    source_manifest_bytes, source_manifest_payload = source_manifest_record
    if final_source_identity.manifest_bytes != source_manifest_bytes:
        raise ExportError("Tokenizer source manifest changed during export")
    preserved_path = destination / SOURCE_TOKENIZER_MANIFEST_NAME
    _write_bytes(preserved_path, source_manifest_bytes)
    source_manifest = {
        "path": SOURCE_TOKENIZER_MANIFEST_NAME,
        "bytes": len(source_manifest_bytes),
        "sha256": _sha256_bytes(source_manifest_bytes),
        "declared_files_verified": len(source_manifest_payload["files"]),
    }

    exported_manifest = {
        "manifest_version": TOKENIZER_EXPORT_MANIFEST_VERSION,
        "format": TOKENIZER_EXPORT_FORMAT,
        "source": {
            # This immutable snapshot authenticates the input files only. The
            # derived files below have independent hashes over their exported
            # bytes after model_max_length was rewritten.
            "manifest": source_manifest,
            "vocab_sha256": source_vocab_sha256,
        },
        "transformation": {
            "kind": "transformers.save_pretrained",
            "source_model_max_length": source_model_max_length,
            "exported_model_max_length": reloaded.model_max_length,
            "vocabulary_preserved": True,
            "special_token_ids_preserved": True,
        },
        "files": exported_files,
        "validation": {
            "vocab_size": len(exported_vocab),
            "vocab_sha256": source_vocab_sha256,
            "model_max_length": reloaded.model_max_length,
            "special_token_ids": exported_special_ids,
        },
    }
    exported_manifest_path = destination / TOKENIZER_MANIFEST_NAME
    _write_json(exported_manifest_path, exported_manifest)

    return {
        "source_path": str(tokenizer_path.resolve()),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest["sha256"],
        "export_manifest": {
            "path": TOKENIZER_MANIFEST_NAME,
            "bytes": exported_manifest_path.stat().st_size,
            "sha256": _sha256_file(exported_manifest_path),
            "format": TOKENIZER_EXPORT_FORMAT,
            "manifest_version": TOKENIZER_EXPORT_MANIFEST_VERSION,
        },
        "vocab_size": len(exported_vocab),
        "vocab_sha256": source_vocab_sha256,
        "model_max_length": reloaded.model_max_length,
        "special_token_ids": exported_special_ids,
    }


def _save_hf_config(
    destination: Path,
    *,
    config: ModelConfig,
    dtype: torch.dtype,
) -> None:
    try:
        from transformers import AutoConfig, AutoTokenizer, LlamaConfig
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Install transformers to export the model config") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(destination),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    constructor_parameters = inspect.signature(LlamaConfig).parameters
    versioned_arguments: dict[str, Any]
    if "rope_parameters" in constructor_parameters:
        versioned_arguments = {
            "rope_parameters": {
                "rope_theta": config.rope_theta,
                "rope_type": "default",
            },
            "dtype": dtype,
        }
    else:
        versioned_arguments = {
            "rope_theta": config.rope_theta,
            "torch_dtype": dtype,
        }
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.hidden_dim,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_seq_len,
        hidden_act="silu",
        initializer_range=config.initializer_range,
        rms_norm_eps=config.norm_eps,
        use_cache=True,
        tie_word_embeddings=config.tie_word_embeddings,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **versioned_arguments,
    )
    hf_config.architectures = ["LlamaForCausalLM"]
    hf_config.save_pretrained(str(destination))
    reloaded = AutoConfig.from_pretrained(
        str(destination), local_files_only=True, trust_remote_code=False
    )
    expected = {
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.dim,
        "intermediate_size": config.hidden_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
    }
    for field, expected_value in expected.items():
        actual = getattr(reloaded, field, None)
        if actual != expected_value:
            raise ExportError(
                f"Reloaded HF config field {field!r} is {actual!r}, expected {expected_value!r}"
            )
    rope_parameters = getattr(reloaded, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        actual_theta = rope_parameters.get("rope_theta")
        actual_rope_type = rope_parameters.get("rope_type")
        if actual_rope_type != "default":
            raise ExportError(
                f"Reloaded HF config has rope_type {actual_rope_type!r}, expected 'default'"
            )
    else:
        actual_theta = getattr(reloaded, "rope_theta", None)
    if actual_theta != config.rope_theta:
        raise ExportError(
            f"Reloaded HF config RoPE theta is {actual_theta!r}, expected {config.rope_theta!r}"
        )
    actual_dtype = getattr(reloaded, "dtype", None)
    if actual_dtype is None:
        actual_dtype = getattr(reloaded, "torch_dtype", None)
    if actual_dtype != dtype:
        raise ExportError(
            f"Reloaded HF config dtype is {actual_dtype!r}, expected {dtype!r}"
        )
    head_dim = getattr(reloaded, "head_dim", config.head_dim)
    if head_dim != config.head_dim:
        raise ExportError(
            f"Reloaded HF config head_dim is {head_dim!r}, expected {config.head_dim!r}"
        )
    if reloaded.vocab_size != len(tokenizer):
        raise ExportError("HF config and tokenizer vocabulary sizes differ")


def _save_safetensor_weights(
    destination: Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    max_shard_size: int | str,
) -> dict[str, Any]:
    try:
        from huggingface_hub import split_torch_state_dict_into_shards
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "Install huggingface_hub and safetensors to export model weights"
        ) from exc
    try:
        split = split_torch_state_dict_into_shards(
            dict(state_dict),
            filename_pattern="model{suffix}.safetensors",
            max_shard_size=max_shard_size,
        )
    except (TypeError, ValueError) as exc:
        raise ExportError(f"Invalid max_shard_size {max_shard_size!r}: {exc}") from exc

    files: list[dict[str, Any]] = []
    for filename, names in split.filename_to_tensors.items():
        path = destination / filename
        try:
            save_file(
                {name: state_dict[name] for name in names},
                str(path),
                metadata={"format": "pt"},
            )
        except Exception as exc:
            raise ExportError(f"Cannot write safetensors shard {filename}: {exc}") from exc
        files.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "tensors": list(names),
            }
        )
    index_name: str | None = None
    if split.is_sharded:
        index_name = "model.safetensors.index.json"
        _write_json(
            destination / index_name,
            {
                "metadata": split.metadata,
                "weight_map": split.tensor_to_filename,
            },
        )
    return {
        "format": "safetensors",
        "is_sharded": bool(split.is_sharded),
        "index": index_name,
        "total_tensor_bytes": int(split.metadata["total_size"]),
        "files": files,
    }


def _inventory_files(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in exclude:
            result.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return result


def export_native_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: ModelConfig | Mapping[str, Any],
    *,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    expected_tokenizer_manifest_sha256: str,
    expected_tokenizer_vocabulary_sha256: str,
    max_shard_size: int | str = "5GB",
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish an HF Llama-compatible model directory."""

    resolved_config = _strict_model_config(config, label="model config")
    try:
        resolved_manifest_sha256 = require_sha256(
            expected_tokenizer_manifest_sha256,
            field="expected tokenizer manifest SHA-256",
        )
        resolved_vocabulary_sha256 = require_sha256(
            expected_tokenizer_vocabulary_sha256,
            field="expected tokenizer vocabulary SHA-256",
        )
    except TokenizerIdentityError as exc:
        raise ExportError(str(exc)) from exc
    dtype = validate_native_state_dict(state_dict, resolved_config)
    hf_state = _cpu_contiguous_without_shared_storage(
        map_native_state_dict(state_dict, resolved_config)
    )
    output = Path(output_dir)
    tokenizer = Path(tokenizer_path)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ExportError(f"Refusing to overwrite existing output path: {output}")

    lock = parent / f".{output.name}.export.lock"
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ExportError(f"Another exporter holds the output lock: {lock}") from exc
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=parent))
    except BaseException:
        os.close(lock_descriptor)
        lock.unlink(missing_ok=True)
        raise
    published = False
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())

        tokenizer_manifest = _save_and_validate_tokenizer(
            tokenizer,
            temporary,
            config=resolved_config,
            expected_manifest_sha256=resolved_manifest_sha256,
            expected_vocabulary_sha256=resolved_vocabulary_sha256,
        )
        _save_hf_config(temporary, config=resolved_config, dtype=dtype)
        _write_json(temporary / "native_config.json", dataclasses.asdict(resolved_config))
        weights_manifest = _save_safetensor_weights(
            temporary,
            hf_state,
            max_shard_size=max_shard_size,
        )
        non_weight_files = _inventory_files(
            temporary,
            exclude={
                EXPORT_MANIFEST_NAME,
                *(entry["path"] for entry in weights_manifest["files"]),
            },
        )
        manifest = {
            "manifest_version": EXPORT_FORMAT_VERSION,
            "format": EXPORT_FORMAT,
            "architecture": "LlamaForCausalLM",
            "source": dict(source or {"kind": "in_memory_state_dict"}),
            "native_model_config": dataclasses.asdict(resolved_config),
            "dtype": str(dtype).removeprefix("torch."),
            "parameter_entries": len(hf_state),
            "serialized_parameter_elements": sum(
                tensor.numel() for tensor in hf_state.values()
            ),
            "tokenizer": tokenizer_manifest,
            "weights": weights_manifest,
            "files": non_weight_files,
        }
        _write_json(temporary / EXPORT_MANIFEST_NAME, manifest)
        _write_text(
            temporary / EXPORT_MANIFEST_SIDECAR_NAME,
            f"{_sha256_file(temporary / EXPORT_MANIFEST_NAME)}  {EXPORT_MANIFEST_NAME}\n",
        )
        # Third-party serializers do not promise fsync. Flush every tokenizer,
        # config, manifest, and safetensors file before the atomic directory
        # rename so a completed export remains durable across abrupt host loss.
        _fsync_export_tree(temporary)
        if output.exists():
            raise ExportError(f"Output path appeared during export: {output}")
        os.rename(temporary, output)
        published = True
        _fsync_directory(parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
        try:
            lock.unlink()
            _fsync_directory(parent)
        except FileNotFoundError:
            pass


def export_native_checkpoint(
    checkpoint_path: str | Path,
    *,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    expected_tokenizer_manifest_sha256: str | None = None,
    expected_tokenizer_vocabulary_sha256: str | None = None,
    max_shard_size: int | str = "5GB",
) -> dict[str, Any]:
    """Load a native checkpoint and atomically export it as HF Llama."""

    checkpoint = Path(checkpoint_path)
    loaded = load_native_checkpoint(
        checkpoint,
        model_config=model_config,
        expected_tokenizer_manifest_sha256=expected_tokenizer_manifest_sha256,
        expected_tokenizer_vocabulary_sha256=expected_tokenizer_vocabulary_sha256,
    )
    source = {
        "kind": loaded.source_kind,
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256_file(checkpoint),
        "checkpoint_format": loaded.checkpoint_format,
        "checkpoint_format_version": loaded.checkpoint_format_version,
        "tokenizer_manifest_sha256": loaded.tokenizer_manifest_sha256,
        "tokenizer_vocabulary_sha256": loaded.tokenizer_vocabulary_sha256,
    }
    return export_native_state_dict(
        loaded.state_dict,
        loaded.config,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        expected_tokenizer_manifest_sha256=loaded.tokenizer_manifest_sha256,
        expected_tokenizer_vocabulary_sha256=loaded.tokenizer_vocabulary_sha256,
        max_shard_size=max_shard_size,
        source=source,
    )
