from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
import unittest
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import CausalLM
from pretrain.data import (
    DistributedBatchSampler,
    create_training_dataloader,
    frozen_training_geometry,
)
from pretrain.train import (
    TrainConfig,
    Trainer,
    capture_rng_state,
    seed_everything,
    tiny_model_config,
    wrap_distributed_model,
)
from scripts.overfit_single_chunk import build_synthetic_order


WORLD_SIZE = 2
VOCAB_SIZE = 64
SEQUENCE_LENGTH = 8
MODEL_SEED = 20_260_830
MASKED_TARGETS = ((0, 5), (2, 7))
RESUME_STEPS = 2
RESUME_GLOBAL_MICROBATCH_ROWS = WORLD_SIZE
RESUME_ACCUMULATION_STEPS = 2


def _batch_for(rank: int, microbatch: int) -> dict[str, torch.Tensor]:
    """Return one deterministic row with rank-dependent supervision density."""

    offset = 1 + rank * 17 + microbatch * 9
    input_ids = torch.tensor(
        [[1 + ((offset + index * 5) % (VOCAB_SIZE - 1)) for index in range(SEQUENCE_LENGTH)]],
        dtype=torch.int64,
    )
    labels = torch.tensor(
        [
            [
                1 + ((offset + (index + 1) * 5) % (VOCAB_SIZE - 1))
                for index in range(SEQUENCE_LENGTH)
            ]
        ],
        dtype=torch.int64,
    )
    masked = MASKED_TARGETS[rank][microbatch]
    if masked:
        labels[0, -masked:] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.arange(SEQUENCE_LENGTH, dtype=torch.int64).unsqueeze(0),
        "document_ids": torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int64),
        "domain_ids": torch.tensor([rank], dtype=torch.int64),
        "num_loss_tokens": labels.ne(-100).sum(),
    }


def _global_batches() -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for microbatch in range(2):
        rank_batches = [_batch_for(rank, microbatch) for rank in range(WORLD_SIZE)]
        batch = {
            key: torch.cat([rank_batch[key] for rank_batch in rank_batches], dim=0)
            for key in (
                "input_ids",
                "labels",
                "position_ids",
                "document_ids",
                "domain_ids",
            )
        }
        batch["num_loss_tokens"] = batch["labels"].ne(-100).sum()
        batches.append(batch)
    return batches


def _train_config() -> TrainConfig:
    return TrainConfig(
        max_steps=1,
        global_microbatch_rows=WORLD_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=4e-3,
        min_learning_rate=4e-4,
        warmup_steps=1,
        weight_decay=0.0,
        max_grad_norm=1_000_000.0,
        precision="float32",
        seed=1234,
        deterministic=True,
        checkpoint_every=0,
        fused_adamw=False,
    )


def _resume_train_config() -> TrainConfig:
    """Return a two-update AdamW trajectory frozen by the synthetic order."""

    return TrainConfig(
        max_steps=RESUME_STEPS,
        global_microbatch_rows=RESUME_GLOBAL_MICROBATCH_ROWS,
        gradient_accumulation_steps=RESUME_ACCUMULATION_STEPS,
        learning_rate=2e-3,
        min_learning_rate=2e-4,
        warmup_steps=1,
        weight_decay=0.01,
        max_grad_norm=10.0,
        precision="float32",
        seed=1234,
        deterministic=True,
        checkpoint_every=0,
        fused_adamw=False,
    )


def _selected_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("train/") and not key.endswith("perplexity")
    }


def _ddp_worker(rank: int, rendezvous_uri: str, output_root: str) -> None:
    """Exercise the real production DDP wrapper and collect one rank result."""

    root = Path(output_root)
    try:
        # Keep this correctness gate small and avoid multiplying BLAS worker
        # pools inside the two spawned ranks.
        torch.set_num_threads(1)
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        elif sys.platform.startswith("linux"):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            backend="gloo",
            init_method=rendezvous_uri,
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )

        seed_everything(MODEL_SEED, deterministic=True)
        model = CausalLM(
            tiny_model_config(vocab_size=VOCAB_SIZE, max_seq_len=SEQUENCE_LENGTH),
            device="cpu",
            dtype=torch.float32,
        )
        distributed_model = wrap_distributed_model(model, device=torch.device("cpu"))
        optimizer = torch.optim.SGD(
            distributed_model.parameters(),
            lr=_train_config().learning_rate,
            momentum=0.9,
        )
        trainer = Trainer(
            distributed_model,
            _train_config(),
            device="cpu",
            data_identity="two-rank-gloo-token-normalization-v1",
            optimizer=optimizer,
        )
        metrics = trainer.train([_batch_for(rank, 0), _batch_for(rank, 1)])
        dist.barrier()
        torch.save(
            {
                "rank": rank,
                "local_loss_tokens": sum(
                    int(_batch_for(rank, index)["num_loss_tokens"])
                    for index in range(2)
                ),
                "model": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in trainer.raw_model.state_dict().items()
                },
                "optimizer": trainer.optimizer.state_dict(),
                "state": dataclasses.asdict(trainer.state),
                "metrics": _selected_metrics(metrics),
            },
            root / f"rank-{rank}.pt",
        )
        dist.barrier()
    except BaseException:
        (root / f"rank-{rank}.error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _local_order_references(
    order_manifest: Path,
    *,
    rank: int,
    start_global_microbatch: int,
) -> list[int]:
    """Read one rank's next immutable-order slice without touching any RNG."""

    sampler = DistributedBatchSampler(
        order_manifest,
        global_microbatch_rows=RESUME_GLOBAL_MICROBATCH_ROWS,
        gradient_accumulation_steps=RESUME_ACCUMULATION_STEPS,
        rank=rank,
        world_size=WORLD_SIZE,
        start_global_microbatch=start_global_microbatch,
    )
    try:
        if len(sampler) == 0:
            return []
        return list(next(iter(sampler)))
    finally:
        sampler.close()


def _resume_worker(
    rank: int,
    phase: str,
    rendezvous_uri: str,
    output_root: str,
    order_manifest_path: str,
    checkpoint_path: str,
) -> None:
    """Run one fresh-process phase of the distributed resume trajectory."""

    root = Path(output_root)
    order_manifest = Path(order_manifest_path)
    checkpoint = Path(checkpoint_path)
    loader = None
    sampler = None
    identity_sampler = None
    try:
        torch.set_num_threads(1)
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        elif sys.platform.startswith("linux"):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            backend="gloo",
            init_method=rendezvous_uri,
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )

        geometry = frozen_training_geometry(order_manifest)
        identity_sampler = DistributedBatchSampler(
            order_manifest,
            global_microbatch_rows=RESUME_GLOBAL_MICROBATCH_ROWS,
            gradient_accumulation_steps=RESUME_ACCUMULATION_STEPS,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        data_identity = identity_sampler.data_identity
        identity_sampler.close()
        identity_sampler = None

        seed_everything(MODEL_SEED, deterministic=True)
        model = CausalLM(
            tiny_model_config(vocab_size=VOCAB_SIZE, max_seq_len=SEQUENCE_LENGTH),
            device="cpu",
            dtype=torch.float32,
        )
        distributed_model = wrap_distributed_model(model, device=torch.device("cpu"))
        trainer = Trainer(
            distributed_model,
            _resume_train_config(),
            device="cpu",
            data_identity=data_identity,
            checkpoint_path=checkpoint,
            training_geometry=geometry,
        )
        if phase == "resume":
            trainer.load_checkpoint(checkpoint)
        elif phase not in ("uninterrupted", "partial"):
            raise ValueError(f"Unknown distributed resume phase: {phase!r}")

        start_state = dataclasses.asdict(trainer.state)
        start_references = _local_order_references(
            order_manifest,
            rank=rank,
            start_global_microbatch=trainer.state.completed_microbatches,
        )
        resume_boundary_references = _local_order_references(
            order_manifest,
            rank=rank,
            start_global_microbatch=RESUME_ACCUMULATION_STEPS,
        )

        loader, sampler = create_training_dataloader(
            order_manifest,
            global_microbatch_rows=RESUME_GLOBAL_MICROBATCH_ROWS,
            gradient_accumulation_steps=RESUME_ACCUMULATION_STEPS,
            rank=rank,
            world_size=WORLD_SIZE,
            start_global_microbatch=trainer.state.completed_microbatches,
            num_workers=0,
            pin_memory=False,
        )
        metrics = trainer.train(
            loader,
            until_step=1 if phase == "partial" else None,
        )
        sampler.close()
        loader.dataset.close()
        sampler = None
        loader = None

        if phase == "partial":
            # This is the actual all-rank checkpoint collective and rank-zero
            # atomic publication used by production Trainer instances.
            trainer.save_checkpoint()

        next_references = _local_order_references(
            order_manifest,
            rank=rank,
            start_global_microbatch=trainer.state.completed_microbatches,
        )
        dist.barrier()
        final_rng = capture_rng_state()
        torch.save(
            {
                "phase": phase,
                "rank": rank,
                "start_state": start_state,
                "start_references": start_references,
                "resume_boundary_references": resume_boundary_references,
                "next_references": next_references,
                "model": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in trainer.raw_model.state_dict().items()
                },
                "optimizer": trainer.optimizer.state_dict(),
                "state": dataclasses.asdict(trainer.state),
                "metrics": _selected_metrics(metrics),
                "rng": final_rng,
            },
            root / f"{phase}-rank-{rank}.pt",
        )
        dist.barrier()
    except BaseException:
        (root / f"{phase}-rank-{rank}.error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        if sampler is not None:
            sampler.close()
        if loader is not None:
            loader.dataset.close()
        if identity_sampler is not None:
            identity_sampler.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _assert_nested_close(
    test: unittest.TestCase,
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(left, np.ndarray):
        test.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        test.assertIsInstance(right, Mapping)
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_close(test, left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(
                test, left_item, right_item, rtol=rtol, atol=atol
            )
    elif isinstance(left, float):
        test.assertIsInstance(right, float)
        test.assertTrue(np.isclose(left, right, rtol=rtol, atol=atol))
    else:
        test.assertEqual(type(left), type(right))
        test.assertEqual(left, right)


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(),
    "the real distributed gate requires the Gloo backend",
)
class DistributedTrainingGateTest(unittest.TestCase):
    def _run_resume_phase(
        self,
        *,
        root: Path,
        phase: str,
        order_manifest: Path,
        checkpoint: Path,
    ) -> list[dict[str, Any]]:
        rendezvous = (root / f"{phase}-gloo-rendezvous").resolve().as_uri()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_resume_worker,
                args=(
                    rank,
                    phase,
                    rendezvous,
                    str(root),
                    str(order_manifest),
                    str(checkpoint),
                ),
                name=f"gloo-resume-{phase}-rank-{rank}",
            )
            for rank in range(WORLD_SIZE)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 90.0
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        hung = [process for process in processes if process.is_alive()]
        for process in hung:
            process.terminate()
        for process in hung:
            process.join(5.0)
        self.assertFalse(hung, f"two-process Gloo {phase} phase timed out")

        errors = sorted(root.glob(f"{phase}-rank-*.error.txt"))
        error_text = "\n".join(
            path.read_text(encoding="utf-8") for path in errors
        )
        self.assertEqual(
            [process.exitcode for process in processes],
            [0, 0],
            error_text,
        )
        return [
            torch.load(
                root / f"{phase}-rank-{rank}.pt",
                weights_only=False,
            )
            for rank in range(WORLD_SIZE)
        ]

    def test_accumulated_token_normalization_matches_global_reference(self) -> None:
        """Real DDP must weight target tokens, not rank-local mean losses."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendezvous = (root / "gloo-rendezvous").resolve().as_uri()
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=_ddp_worker,
                    args=(rank, rendezvous, str(root)),
                    name=f"gloo-correctness-rank-{rank}",
                )
                for rank in range(WORLD_SIZE)
            ]
            for process in processes:
                process.start()

            deadline = time.monotonic() + 60.0
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            hung = [process for process in processes if process.is_alive()]
            for process in hung:
                process.terminate()
            for process in hung:
                process.join(5.0)
            self.assertFalse(hung, "two-process Gloo correctness gate timed out")

            errors = sorted(root.glob("rank-*.error.txt"))
            error_text = "\n".join(path.read_text(encoding="utf-8") for path in errors)
            self.assertEqual(
                [process.exitcode for process in processes],
                [0, 0],
                error_text,
            )
            rank_results = [
                torch.load(root / f"rank-{rank}.pt", weights_only=False)
                for rank in range(WORLD_SIZE)
            ]

            # The fixture deliberately gives each rank a different denominator;
            # averaging rank-local mean losses would therefore be incorrect.
            self.assertNotEqual(
                rank_results[0]["local_loss_tokens"],
                rank_results[1]["local_loss_tokens"],
            )

            # A synchronized DDP step must leave every replica and optimizer
            # exactly equal, with the same globally reduced metrics and cursors.
            for key in ("model", "optimizer", "state", "metrics"):
                _assert_nested_close(
                    self,
                    rank_results[0][key],
                    rank_results[1][key],
                    rtol=0.0,
                    atol=0.0,
                )

            seed_everything(MODEL_SEED, deterministic=True)
            reference_model = CausalLM(
                tiny_model_config(
                    vocab_size=VOCAB_SIZE,
                    max_seq_len=SEQUENCE_LENGTH,
                ),
                device="cpu",
                dtype=torch.float32,
            )
            reference = Trainer(
                reference_model,
                _train_config(),
                device="cpu",
                data_identity="single-process-global-batch-reference-v1",
                optimizer=torch.optim.SGD(
                    reference_model.parameters(),
                    lr=_train_config().learning_rate,
                    momentum=0.9,
                ),
            )
            reference_metrics = reference.train(_global_batches())

            _assert_nested_close(
                self,
                rank_results[0]["model"],
                reference_model.state_dict(),
                rtol=1e-5,
                atol=1e-7,
            )
            _assert_nested_close(
                self,
                rank_results[0]["optimizer"],
                reference.optimizer.state_dict(),
                rtol=1e-5,
                atol=1e-7,
            )
            self.assertEqual(
                rank_results[0]["state"], dataclasses.asdict(reference.state)
            )
            for key in ("train/loss", "train/loss_sum", "train/loss_tokens"):
                self.assertTrue(
                    np.isclose(
                        rank_results[0]["metrics"][key],
                        reference_metrics[key],
                        rtol=1e-7,
                        atol=1e-7,
                    ),
                    key,
                )

    def test_checkpoint_process_restart_resume_matches_uninterrupted(self) -> None:
        """A real two-rank restart must resume the exact immutable-order cursor."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order_manifest = build_synthetic_order(
                root / "fixture",
                sequence_length=SEQUENCE_LENGTH,
                vocab_size=VOCAB_SIZE,
                global_microbatch_rows=RESUME_GLOBAL_MICROBATCH_ROWS,
                gradient_accumulation_steps=RESUME_ACCUMULATION_STEPS,
            )
            geometry = frozen_training_geometry(order_manifest)
            self.assertEqual(geometry["optimizer_updates"], RESUME_STEPS)
            checkpoint = root / "distributed-checkpoint.pt"

            uninterrupted = self._run_resume_phase(
                root=root,
                phase="uninterrupted",
                order_manifest=order_manifest,
                checkpoint=checkpoint,
            )
            partial = self._run_resume_phase(
                root=root,
                phase="partial",
                order_manifest=order_manifest,
                checkpoint=checkpoint,
            )
            self.assertTrue(checkpoint.is_file())
            checkpoint_payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(checkpoint_payload["format_version"], 4)
            self.assertEqual(checkpoint_payload["world_size"], WORLD_SIZE)
            self.assertEqual(
                checkpoint_payload["train_state"]["completed_steps"], 1
            )
            self.assertEqual(len(checkpoint_payload["rng_states"]), WORLD_SIZE)
            self.assertTrue(
                all(
                    state["torch_cuda"] is None
                    for state in checkpoint_payload["rng_states"]
                )
            )

            resumed = self._run_resume_phase(
                root=root,
                phase="resume",
                order_manifest=order_manifest,
                checkpoint=checkpoint,
            )

            # Every phase must leave synchronized rank replicas, optimizer
            # state, counters, and globally reduced non-timing metrics.
            for results in (uninterrupted, partial, resumed):
                for key in ("model", "optimizer", "state", "metrics"):
                    _assert_nested_close(
                        self,
                        results[0][key],
                        results[1][key],
                        rtol=0.0,
                        atol=0.0,
                    )

            expected_boundary = partial[0]["state"]
            self.assertEqual(expected_boundary["completed_steps"], 1)
            self.assertEqual(
                expected_boundary["completed_microbatches"],
                RESUME_ACCUMULATION_STEPS,
            )
            for rank in range(WORLD_SIZE):
                self.assertEqual(resumed[rank]["start_state"], expected_boundary)
                self.assertEqual(
                    resumed[rank]["start_references"],
                    partial[rank]["next_references"],
                )
                self.assertEqual(
                    resumed[rank]["start_references"],
                    resumed[rank]["resume_boundary_references"],
                )
                self.assertEqual(
                    uninterrupted[rank]["resume_boundary_references"],
                    resumed[rank]["start_references"],
                )

            # Rank slices at the resume boundary are disjoint and together
            # cover exactly one global microbatch.
            rank_zero_next = set(resumed[0]["start_references"])
            rank_one_next = set(resumed[1]["start_references"])
            self.assertTrue(rank_zero_next.isdisjoint(rank_one_next))
            self.assertEqual(
                len(rank_zero_next | rank_one_next),
                RESUME_GLOBAL_MICROBATCH_ROWS,
            )

            # Fresh processes resumed from the durable checkpoint must finish
            # bit-for-bit equal to the uninterrupted distributed trajectory.
            for rank in range(WORLD_SIZE):
                for key in ("model", "optimizer", "state", "metrics", "rng"):
                    _assert_nested_close(
                        self,
                        uninterrupted[rank][key],
                        resumed[rank][key],
                        rtol=0.0,
                        atol=0.0,
                    )
                self.assertEqual(uninterrupted[rank]["next_references"], [])
                self.assertEqual(resumed[rank]["next_references"], [])


if __name__ == "__main__":
    unittest.main()
