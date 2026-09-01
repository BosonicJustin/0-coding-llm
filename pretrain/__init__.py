"""Pre-training components with a dependency-safe lazy public API.

The repository's data-preparation environment intentionally does not install
PyTorch.  Importing a lightweight submodule such as
``pretrain.raw_token_cache`` must therefore not eagerly import
``pretrain.data``.  PEP 562 module attribute resolution preserves the existing
``from pretrain import PackedShardWriter`` API while loading the PyTorch data
module only when one of those training symbols is actually requested.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


_DATA_EXPORTS = (
    "IGNORE_INDEX",
    "DOMAIN_ORDER",
    "DistributedBatchSampler",
    "DomainMixtureDataset",
    "PackedBatchCollator",
    "PackedShardDataset",
    "PackedShardWriter",
    "build_training_order",
    "create_training_dataloader",
    "validate_packed_manifest",
    "validate_training_order",
)

if TYPE_CHECKING:
    from .data import (
        DOMAIN_ORDER,
        IGNORE_INDEX,
        DistributedBatchSampler,
        DomainMixtureDataset,
        PackedBatchCollator,
        PackedShardDataset,
        PackedShardWriter,
        build_training_order,
        create_training_dataloader,
        validate_packed_manifest,
        validate_training_order,
    )

__all__ = [
    *_DATA_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name not in _DATA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    data_module = importlib.import_module(f"{__name__}.data")
    value = getattr(data_module, name)
    # Match ordinary eager-import behavior after first resolution and avoid a
    # repeated module lookup for hot public symbols.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
