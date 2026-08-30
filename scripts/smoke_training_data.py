#!/usr/bin/env python3
"""Build a tiny packed corpus and exercise multiprocessing data loading."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import (
    DOMAIN_ORDER,
    PackedShardWriter,
    build_training_order,
    create_training_dataloader,
)


FAKE_TOKENIZER_MANIFEST_SHA256 = hashlib.sha256(
    b"synthetic-smoke-test-tokenizer-manifest"
).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifests = {}
        for domain, rows in zip(DOMAIN_ORDER, (4, 4, 2), strict=True):
            output = root / "packed" / domain
            writer = PackedShardWriter(
                output,
                domain=domain,
                split="train",
                sequence_length=16,
                vocab_size=256,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
                rows_per_shard=2,
            )
            writer.add_document([(index % 200) + 1 for index in range(rows * 16)])
            writer.finish()
            manifests[domain] = output / "manifest.json"
        build_training_order(
            manifests,
            root / "order",
            seed=123,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            frozen_global_microbatch_rows=5,
            frozen_gradient_accumulation_steps=1,
        )
        loader, _ = create_training_dataloader(
            root / "order" / "manifest.json",
            global_microbatch_rows=5,
            num_workers=args.workers,
            pin_memory=False,
        )
        batches = list(loader)
        assert len(batches) == 2
        assert all(tuple(batch["input_ids"].shape) == (5, 16) for batch in batches)
        print(f"ok: loaded {len(batches)} batches with {args.workers} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
