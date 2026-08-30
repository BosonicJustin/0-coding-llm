#!/usr/bin/env python3
"""Exercise realistic 4096-token packed mmap shards and report CPU throughput."""

from __future__ import annotations

import argparse
import gc
import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import (
    DOMAIN_ORDER,
    PackedShardWriter,
    build_training_order,
    create_training_dataloader,
    validate_packed_manifest,
    validate_training_order,
)


FAKE_TOKENIZER_MANIFEST_SHA256 = hashlib.sha256(
    b"synthetic-loader-benchmark-tokenizer-manifest"
).hexdigest()


def remove_tree_with_retries(path: Path, attempts: int = 6) -> None:
    """Remove our temporary NFS tree after mmap workers release file handles."""

    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=Path("/tmp"),
        help="parent directory for an automatically removed benchmark corpus",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--total-rows", type=int, default=1000)
    parser.add_argument("--global-microbatch-rows", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rows-per-shard", type=int, default=256)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep generated files and print their directory",
    )
    args = parser.parse_args()
    if args.total_rows < 5 or args.total_rows % 5:
        parser.error("--total-rows must be a positive multiple of 5")
    if args.global_microbatch_rows < 1 or args.gradient_accumulation_steps < 1:
        parser.error("training geometry values must be positive")
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix="training-loader-benchmark-", dir=args.temporary_root)
    )
    try:
        counts = {
            "python": args.total_rows * 2 // 5,
            "other_code": args.total_rows * 2 // 5,
            "english": args.total_rows // 5,
        }
        manifests = {}
        build_started = time.perf_counter()
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            output = work / "packed" / domain
            writer = PackedShardWriter(
                output,
                domain=domain,
                split="train",
                sequence_length=args.sequence_length,
                vocab_size=49_152,
                eos_token_id=0,
                tokenizer_manifest_sha256=FAKE_TOKENIZER_MANIFEST_SHA256,
                rows_per_shard=args.rows_per_shard,
                construction_seed=args.seed,
            )
            # T-1 content tokens plus EOS make one segment exactly T tokens.
            # N+1 such documents produce N full T+1 rows.
            document = [domain_index + 1] * (args.sequence_length - 1)
            for _ in range(counts[domain] + 1):
                writer.add_document(document)
            manifest = writer.finish()
            if manifest["rows"] != counts[domain]:
                raise AssertionError(f"Unexpected row count for {domain}")
            manifests[domain] = output / "manifest.json"
            validate_packed_manifest(manifests[domain], verify_checksums=True)
        build_training_order(
            manifests,
            work / "order",
            seed=args.seed,
            expected_weights={"python": 0.4, "other_code": 0.4, "english": 0.2},
            frozen_global_microbatch_rows=args.global_microbatch_rows,
            frozen_gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        validate_training_order(work / "order" / "manifest.json")
        build_seconds = time.perf_counter() - build_started

        loader, sampler = create_training_dataloader(
            work / "order" / "manifest.json",
            global_microbatch_rows=args.global_microbatch_rows,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.workers,
            pin_memory=False,
            persistent_workers=False,
            verify_payload_checksums=False,
        )
        load_started = time.perf_counter()
        batches = 0
        supervised_tokens = 0
        iterator = iter(loader)
        for batch in iterator:
            batches += 1
            supervised_tokens += int(batch["num_loss_tokens"])
        load_seconds = time.perf_counter() - load_started
        input_tokens = batches * args.global_microbatch_rows * args.sequence_length
        payload_bytes = sum(
            path.stat().st_size
            for path in (work / "packed").rglob("*.bin")
        )
        print(f"work_directory={work}")
        print(f"rows_declared={args.total_rows}")
        print(f"rows_loaded={batches * args.global_microbatch_rows}")
        print(f"rows_dropped={sampler.dropped_rows}")
        print(f"input_tokens={input_tokens}")
        print(f"supervised_tokens={supervised_tokens}")
        print(f"payload_bytes={payload_bytes}")
        print(f"build_and_checksum_seconds={build_seconds:.3f}")
        print(f"load_seconds={load_seconds:.3f}")
        print(f"input_tokens_per_second={input_tokens / load_seconds:.0f}")
        print(f"rows_per_second={input_tokens / args.sequence_length / load_seconds:.1f}")
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        loader.dataset.close()
        sampler.close()
        del iterator, loader, sampler
        gc.collect()
        return 0
    finally:
        if args.keep:
            print(f"kept={work}")
        else:
            remove_tree_with_retries(work)


if __name__ == "__main__":
    raise SystemExit(main())
