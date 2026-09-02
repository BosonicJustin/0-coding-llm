from __future__ import annotations

import copy
import contextlib
import dataclasses
import importlib
import io
import json
import random
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.model import CausalLM
from pretrain.data import create_training_dataloader, frozen_training_geometry
from pretrain.train import (
    CheckpointLease,
    CompositeLogger,
    ConsoleLogger,
    FixedBatchStream,
    GracefulStopController,
    TrainConfig,
    Trainer as ProductionTrainer,
    ValidationRunner,
    WandbLogger,
    _accumulate_domain_metrics,
    _atomic_torch_save,
    _bind_wandb_run_id,
    _finalize_training_run,
    _raise_if_distributed_stage_failed,
    _resolve_device,
    batch_fingerprint,
    build_optimizer,
    capture_rng_state,
    evaluate_fixed_batch,
    initialize_optional_metric_logger,
    learning_rate_for_step,
    preserve_host_rng_state,
    reconcile_checkpoint_temporaries,
    restore_rng_state,
    seed_everything,
    sha256_file,
    tiny_model_config,
    validate_deterministic_cuda_environment,
    verify_order_payload_checksum,
)


TEST_TOKENIZER_MANIFEST_SHA256 = "a" * 64
TEST_TOKENIZER_VOCABULARY_SHA256 = "b" * 64


def Trainer(*args, **kwargs):
    """Construct a trainer bound to the deterministic synthetic test tokenizer."""

    kwargs.setdefault(
        "tokenizer_manifest_sha256", TEST_TOKENIZER_MANIFEST_SHA256
    )
    kwargs.setdefault(
        "tokenizer_vocabulary_sha256", TEST_TOKENIZER_VOCABULARY_SHA256
    )
    return ProductionTrainer(*args, **kwargs)


OVERFIT_SCRIPT = PROJECT_ROOT / "scripts" / "overfit_single_chunk.py"
OVERFIT_SPEC = importlib.util.spec_from_file_location("overfit_single_chunk", OVERFIT_SCRIPT)
assert OVERFIT_SPEC is not None and OVERFIT_SPEC.loader is not None
OVERFIT_MODULE = importlib.util.module_from_spec(OVERFIT_SPEC)
OVERFIT_SPEC.loader.exec_module(OVERFIT_MODULE)
build_synthetic_order = OVERFIT_MODULE.build_synthetic_order
load_one_packed_batch = OVERFIT_MODULE.load_one_packed_batch


class MemoryLogger:
    def __init__(self) -> None:
        self.metrics: list[dict[str, float | int]] = []

    def log(self, metrics) -> None:
        self.metrics.append(dict(metrics))

    def finish(self) -> None:
        pass


class FailingLogger:
    def log(self, metrics) -> None:
        del metrics
        raise RuntimeError("intentional logger failure")

    def finish(self) -> None:
        pass


class RngConsumingLogger:
    def log(self, metrics) -> None:
        del metrics
        random.random()
        np.random.rand()
        torch.rand(5)

    def finish(self) -> None:
        pass


class StopAtStepLogger:
    def __init__(self, controller: GracefulStopController, step: int = 1) -> None:
        self.controller = controller
        self.step = step

    def log(self, metrics) -> None:
        if metrics.get("train/step") == self.step and "train/loss" in metrics:
            self.controller.request(int(signal.SIGTERM))

    def finish(self) -> None:
        pass


class StopOnValidationLogger:
    def __init__(self, controller: GracefulStopController) -> None:
        self.controller = controller
        self.metrics: list[dict[str, float | int]] = []

    def log(self, metrics) -> None:
        payload = dict(metrics)
        self.metrics.append(payload)
        if "validation/loss" in payload:
            self.controller.request(int(signal.SIGTERM))

    def finish(self) -> None:
        pass


def make_batch(*, masked_targets: int = 0) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [[1, 7, 4, 9, 3, 11, 6, 2]],
        dtype=torch.int64,
    )
    labels = torch.tensor(
        [[7, 4, 9, 3, 11, 6, 2, 5]],
        dtype=torch.int64,
    )
    if masked_targets:
        labels[0, -masked_targets:] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.arange(8, dtype=torch.int64).unsqueeze(0),
        "document_ids": torch.zeros((1, 8), dtype=torch.int64),
        "num_loss_tokens": labels.ne(-100).sum(),
    }


def assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


class TrainingHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(123, deterministic=True)
        self.model_config = tiny_model_config(vocab_size=64, max_seq_len=8)

    def test_tiny_model_uses_compile_compatible_attention_heads(self) -> None:
        self.assertGreaterEqual(self.model_config.head_dim, 16)
        self.assertEqual(self.model_config.dim, 64)
        self.assertEqual(self.model_config.hidden_dim, 176)

    def test_domain_metric_accumulation_matches_expected_totals(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        domain_ids = torch.tensor([0, 2, 1, 0], device=device)
        row_loss_sums = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        row_loss_tokens = torch.tensor([2, 3, 4, 5], device=device)
        row_input_tokens = torch.tensor([8, 8, 8, 8], device=device)
        loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
        loss_tokens = torch.zeros(3, dtype=torch.int64, device=device)
        input_tokens = torch.zeros(3, dtype=torch.int64, device=device)
        rows = torch.zeros(3, dtype=torch.int64, device=device)

        deterministic_before = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            _accumulate_domain_metrics(
                domain_ids=domain_ids,
                row_loss_sums=row_loss_sums,
                row_loss_tokens=row_loss_tokens,
                row_input_tokens=row_input_tokens,
                loss_sums=loss_sums,
                loss_tokens=loss_tokens,
                input_tokens=input_tokens,
                rows=rows,
            )
        finally:
            torch.use_deterministic_algorithms(deterministic_before)

        torch.testing.assert_close(
            loss_sums.cpu(), torch.tensor([5.0, 3.0, 2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            loss_tokens.cpu(), torch.tensor([7, 4, 3], dtype=torch.int64)
        )
        torch.testing.assert_close(
            input_tokens.cpu(), torch.tensor([16, 8, 8], dtype=torch.int64)
        )
        torch.testing.assert_close(
            rows.cpu(), torch.tensor([2, 1, 1], dtype=torch.int64)
        )

    def test_deterministic_cuda_requires_pinned_cublas_workspace(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                validate_deterministic_cuda_environment(
                    "cuda:0",
                    deterministic=True,
                )

    def test_unindexed_cuda_maps_each_of_six_local_ranks_to_its_own_gpu(self) -> None:
        devices = [
            _resolve_device("cuda", local_rank=local_rank)
            for local_rank in range(6)
        ]
        self.assertEqual(
            devices,
            [torch.device(f"cuda:{local_rank}") for local_rank in range(6)],
        )

    def test_rank_local_setup_failure_is_synchronized(self) -> None:
        injected = OSError("rank-one local read failed")
        with self.assertRaisesRegex(OSError, "rank-one"):
            _raise_if_distributed_stage_failed(
                injected,
                stage="resume",
                rank=0,
                world_size=1,
            )

        def report_peer_failure(statuses, local_status):
            self.assertIsNone(local_status)
            statuses[:] = [None, "rank 1: OSError: rank-one local read failed"]

        with mock.patch(
            "pretrain.train.dist.all_gather_object",
            side_effect=report_peer_failure,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Distributed setup stage 'resume' failed"
            ):
                _raise_if_distributed_stage_failed(
                    None,
                    stage="resume",
                    rank=0,
                    world_size=2,
                )
            validate_deterministic_cuda_environment("cpu", deterministic=True)
            validate_deterministic_cuda_environment("cuda:0", deterministic=False)
        for workspace in (":4096:8", ":16:8"):
            with mock.patch.dict(
                "os.environ",
                {"CUBLAS_WORKSPACE_CONFIG": workspace},
                clear=True,
            ):
                validate_deterministic_cuda_environment(
                    "cuda:0",
                    deterministic=True,
                )

    def test_rng_helpers_touch_only_the_rank_local_cuda_device(self) -> None:
        cuda_rng = torch.arange(16, dtype=torch.uint8)
        with (
            mock.patch("pretrain.train.torch.cuda.is_available", return_value=True),
            mock.patch("pretrain.train.torch.cuda.current_device", return_value=3),
            mock.patch("pretrain.train.torch.cuda.manual_seed") as manual_seed,
            mock.patch(
                "pretrain.train.torch.cuda.manual_seed_all",
                side_effect=AssertionError("must not seed peer devices"),
            ),
            mock.patch(
                "pretrain.train.torch.cuda.get_rng_state",
                return_value=cuda_rng.clone(),
            ) as get_rng_state,
            mock.patch(
                "pretrain.train.torch.cuda.get_rng_state_all",
                side_effect=AssertionError("must not capture peer devices"),
            ),
        ):
            seed_everything(314, deterministic=True)
            state = capture_rng_state()

        manual_seed.assert_called_once_with(314)
        get_rng_state.assert_called_once_with(3)
        self.assertEqual(state["torch_cuda"]["device_index"], 3)
        self.assertTrue(torch.equal(state["torch_cuda"]["state"], cuda_rng))

        with (
            mock.patch("pretrain.train.torch.cuda.is_available", return_value=True),
            mock.patch("pretrain.train.torch.cuda.current_device", return_value=3),
            mock.patch("pretrain.train.torch.cuda.set_rng_state") as set_rng_state,
            mock.patch(
                "pretrain.train.torch.cuda.set_rng_state_all",
                side_effect=AssertionError("must not restore peer devices"),
            ),
        ):
            restore_rng_state(state)
        set_rng_state.assert_called_once()
        restored_state, restored_device = set_rng_state.call_args.args
        self.assertTrue(torch.equal(restored_state, cuda_rng))
        self.assertEqual(restored_device, 3)

    def test_rng_restore_rejects_changed_rank_local_cuda_assignment(self) -> None:
        state = capture_rng_state()
        state["torch_cuda"] = {
            "device_index": 3,
            "state": torch.arange(8, dtype=torch.uint8),
        }
        with (
            mock.patch("pretrain.train.torch.cuda.is_available", return_value=True),
            mock.patch("pretrain.train.torch.cuda.current_device", return_value=2),
            self.assertRaisesRegex(RuntimeError, "Rank-local CUDA device changed"),
        ):
            restore_rng_state(state)

    def test_trainer_rejects_noncurrent_explicit_cuda_device(self) -> None:
        with (
            mock.patch("pretrain.train.torch.cuda.is_available", return_value=True),
            mock.patch("pretrain.train.torch.cuda.current_device", return_value=0),
            self.assertRaisesRegex(ValueError, "must be the process's current device"),
        ):
            Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cuda:1",
                data_identity="cuda-device-mismatch",
            )

    def train_config(
        self,
        *,
        max_steps: int = 4,
        accumulation: int = 1,
        global_microbatch_rows: int = 1,
    ) -> TrainConfig:
        return TrainConfig(
            max_steps=max_steps,
            global_microbatch_rows=global_microbatch_rows,
            gradient_accumulation_steps=accumulation,
            learning_rate=4e-3,
            min_learning_rate=4e-4,
            warmup_steps=1,
            weight_decay=0.0,
            max_grad_norm=10.0,
            precision="float32",
            seed=123,
            deterministic=True,
            checkpoint_every=0,
            fused_adamw=False,
        )

    def test_wandb_disabled_does_not_import_optional_dependency(self) -> None:
        with mock.patch.object(
            importlib,
            "import_module",
            side_effect=AssertionError("wandb import must remain lazy"),
        ):
            logger = WandbLogger(mode="disabled")
            logger.log({"train/step": 1})
            logger.finish()

    def test_wandb_enabled_without_package_has_clear_error(self) -> None:
        with mock.patch.object(importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "requirements-wandb"):
                WandbLogger(mode="offline")

    def test_console_logger_emits_strict_finite_json(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ConsoleLogger().log({"train/step": 3, "train/loss": 1.25})
        self.assertEqual(
            json.loads(output.getvalue()),
            {"train/step": 3, "train/loss": 1.25},
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must be finite"
            ):
                ConsoleLogger().log({"train/step": 3, "train/loss": value})

    def test_composite_finishes_a_logger_disabled_after_log_failure(self) -> None:
        failed = mock.Mock()
        failed.log.side_effect = OSError("tracker write failed")
        healthy = mock.Mock()
        logger = CompositeLogger(failed, healthy)

        logger.log({"train/step": 1, "train/loss": 2.0})
        logger.log({"train/step": 2, "train/loss": 1.0})
        logger.finish()

        failed.log.assert_called_once()
        self.assertEqual(healthy.log.call_count, 2)
        failed.finish.assert_called_once()
        healthy.finish.assert_called_once()

    def test_optional_tracking_initialization_falls_back_without_rng_or_failure(self) -> None:
        before = capture_rng_state()

        def fail_to_initialize():
            random.random()
            np.random.rand()
            torch.rand(3)
            raise OSError("tracker service unavailable")

        with preserve_host_rng_state():
            logger, error = initialize_optional_metric_logger(fail_to_initialize)
        self.assertEqual(type(logger).__name__, "NullLogger")
        self.assertEqual(error, "OSError: tracker service unavailable")
        after = capture_rng_state()
        self.assertEqual(before["python"], after["python"])
        np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
        self.assertTrue(torch.equal(before["torch_cpu"], after["torch_cpu"]))

    def test_wandb_offline_reuses_explicit_run_id(self) -> None:
        run = mock.Mock(id="stable-run-id")
        wandb = mock.Mock()
        wandb.init.return_value = run
        with mock.patch.object(importlib, "import_module", return_value=wandb):
            logger = WandbLogger(
                mode="offline",
                project="test-project",
                run_id="stable-run-id",
            )
        self.assertEqual(logger.run_id, "stable-run-id")
        self.assertEqual(wandb.init.call_args.kwargs["mode"], "offline")
        self.assertEqual(wandb.init.call_args.kwargs["id"], "stable-run-id")
        self.assertEqual(wandb.init.call_args.kwargs["resume"], "allow")
        logger.log({"train/step": 3, "train/loss": 1.0})
        logger.log({"train/step": 3, "validation/loss": 1.1})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            logger.log({"train/step": 3, "validation/loss": float("nan")})
        run.define_metric.assert_has_calls(
            [
                mock.call("train/step"),
                mock.call("*", step_metric="train/step"),
            ]
        )
        self.assertEqual(run.log.call_count, 2)
        for call in run.log.call_args_list:
            self.assertNotIn("step", call.kwargs)
        logger.finish()
        run.finish.assert_called_once()

    def test_wandb_uploads_only_explicit_small_evidence_files(self) -> None:
        run = mock.Mock(id="evidence-run-id")
        artifact = mock.Mock()
        wandb = mock.Mock()
        wandb.init.return_value = run
        wandb.Artifact.return_value = artifact
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "manifest.json"
            evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
            with mock.patch.object(importlib, "import_module", return_value=wandb):
                logger = WandbLogger(mode="online", project="test-project")
            logger.log_evidence_artifact(
                name="authority-abc123",
                files={"train-order-manifest.json": evidence},
                aliases=("latest", "abc123"),
            )
        wandb.Artifact.assert_called_once_with(
            name="authority-abc123", type="pretraining-authority"
        )
        artifact.add_file.assert_called_once_with(
            str(evidence.resolve()), name="train-order-manifest.json"
        )
        run.log_artifact.assert_called_once_with(
            artifact, aliases=["latest", "abc123"]
        )

    def test_new_wandb_run_id_invalidates_same_step_checkpoint_cache(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            trainer.save_checkpoint()
            self.assertIsNotNone(trainer._last_checkpoint)
            _bind_wandb_run_id(trainer, "new-stable-run-id")
            self.assertIsNone(trainer._last_checkpoint)
            trainer.save_checkpoint()
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(
                payload["metadata"]["wandb_run_id"],
                "new-stable-run-id",
            )

    def test_learning_rate_warmup_and_decay_endpoints(self) -> None:
        config = self.train_config(max_steps=5)
        self.assertEqual(learning_rate_for_step(config, 0), config.learning_rate)
        self.assertEqual(learning_rate_for_step(config, 4), config.min_learning_rate)

    def test_train_config_rejects_nonfinite_optimizer_hyperparameters(self) -> None:
        base = self.train_config(max_steps=5)
        for field, value in (
            ("learning_rate", float("nan")),
            ("min_learning_rate", float("inf")),
            ("weight_decay", float("nan")),
            ("beta1", float("inf")),
            ("adam_eps", float("nan")),
            ("max_grad_norm", float("inf")),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "must be finite"
            ):
                dataclasses.replace(base, **{field: value})

    def test_adamw_decay_groups_are_complete_disjoint_and_shape_based(self) -> None:
        model = CausalLM(self.model_config, dtype=torch.float32)
        config = dataclasses.replace(
            self.train_config(max_steps=1),
            weight_decay=0.1,
        )
        optimizer = build_optimizer(
            model,
            config,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(optimizer.param_groups), 2)
        decay = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        no_decay = {
            id(parameter) for parameter in optimizer.param_groups[1]["params"]
        }
        expected = {id(parameter) for parameter in model.parameters()}
        self.assertFalse(decay & no_decay)
        self.assertEqual(decay | no_decay, expected)
        for parameter in model.parameters():
            self.assertEqual(id(parameter) in decay, parameter.ndim >= 2)
        self.assertEqual(
            optimizer.param_groups[0]["weight_decay"],
            config.weight_decay,
        )
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_fixed_stream_clones_and_fingerprints_full_batch(self) -> None:
        batch = make_batch()
        stream = FixedBatchStream(batch)
        before = stream.identity
        batch["input_ids"][0, 0] = 63
        self.assertEqual(stream.identity, before)
        self.assertNotEqual(batch_fingerprint(batch), before.split(":", 1)[1])
        iterator = iter(stream)
        self.assertIs(next(iterator), next(iterator))

    def test_metrics_and_loss_fall_on_fixed_batch(self) -> None:
        batch = make_batch()
        stream = FixedBatchStream(batch)
        logger = MemoryLogger()
        model = CausalLM(self.model_config, dtype=torch.float32)
        initial = evaluate_fixed_batch(model, stream.batch, device="cpu")
        trainer = Trainer(
            model,
            self.train_config(max_steps=12),
            device="cpu",
            data_identity=stream.identity,
            logger=logger,
        )
        metrics = trainer.train(stream)
        final = evaluate_fixed_batch(model, stream.batch, device="cpu")
        self.assertLess(final, initial * 0.8)
        self.assertEqual(len(logger.metrics), 12)
        expected_metrics = {
            "train/step",
            "train/loss",
            "train/token_loss",
            "train/loss_sum",
            "train/perplexity",
            "train/learning_rate",
            "train/grad_norm",
            "train/rows",
            "train/input_tokens",
            "train/supervised_tokens",
            "train/loss_tokens",
            "train/microbatches",
            "train/update_rows",
            "train/update_input_tokens",
            "train/update_supervised_tokens",
            "train/update_supervision_fraction",
            "train/progress_fraction",
            "train/progress_percent",
            "train/remaining_steps",
            "train/grad_norm_to_clip_threshold",
            "train/gradient_was_clipped",
            "perf/input_tokens_per_second",
            "perf/input_tokens_per_second_ema",
            "perf/loss_tokens_per_second",
            "perf/step_seconds",
            "perf/step_seconds_ema",
            "perf/estimated_remaining_seconds",
            "perf/data_wait_seconds",
            "perf/data_wait_fraction",
        }
        self.assertTrue(expected_metrics.issubset(metrics))
        self.assertEqual(metrics["train/input_tokens"], 12 * 8)
        self.assertEqual(metrics["train/loss_tokens"], 12 * 8)
        self.assertEqual(metrics["train/progress_fraction"], 1.0)
        self.assertEqual(metrics["train/remaining_steps"], 0)
        self.assertEqual(metrics["train/update_supervision_fraction"], 1.0)
        self.assertGreaterEqual(metrics["perf/data_wait_fraction"], 0.0)
        self.assertLessEqual(metrics["perf/data_wait_fraction"], 1.0)

    def test_distributed_timing_reduces_wait_fraction_independently(self) -> None:
        trainer = Trainer(
            CausalLM(self.model_config, dtype=torch.float32),
            self.train_config(max_steps=1),
            device="cpu",
            data_identity="timing-test",
        )
        trainer.world_size = 2

        def reduce_max(values, *, op):
            self.assertIs(op, torch.distributed.ReduceOp.MAX)
            values.copy_(torch.tensor([10.0, 8.0, 8.0 / 9.0]))

        with mock.patch.object(torch.distributed, "all_reduce", side_effect=reduce_max):
            elapsed, wait, fraction = trainer._max_step_timings(9.0, 8.0)
        self.assertEqual(elapsed, 10.0)
        self.assertEqual(wait, 8.0)
        self.assertAlmostEqual(fraction, 8.0 / 9.0)

    def test_token_normalized_accumulation_matches_concatenated_batch(self) -> None:
        first = make_batch(masked_targets=0)
        second = make_batch(masked_targets=5)
        second["input_ids"] = (second["input_ids"] + 13) % 64
        concatenated = {
            key: torch.cat((first[key], second[key]), dim=0)
            for key in ("input_ids", "labels", "position_ids", "document_ids")
        }
        concatenated["num_loss_tokens"] = concatenated["labels"].ne(-100).sum()

        seed_everything(99, deterministic=True)
        accumulated_model = CausalLM(self.model_config, dtype=torch.float32)
        seed_everything(99, deterministic=True)
        concatenated_model = CausalLM(self.model_config, dtype=torch.float32)
        accumulated_config = self.train_config(max_steps=1, accumulation=2)
        concatenated_config = self.train_config(
            max_steps=1,
            accumulation=1,
            global_microbatch_rows=2,
        )
        accumulated = Trainer(
            accumulated_model,
            accumulated_config,
            device="cpu",
            data_identity="two-microbatches",
        )
        combined = Trainer(
            concatenated_model,
            concatenated_config,
            device="cpu",
            data_identity="one-combined-batch",
        )
        accumulated.train(iter((first, second)))
        combined.train(iter((concatenated,)))
        for left, right in zip(
            accumulated_model.parameters(),
            concatenated_model.parameters(),
            strict=True,
        ):
            # Separate backward calls change floating-point summation order,
            # but must produce the same token-weighted optimizer update.  The
            # compile-compatible debug model has wider reduction dimensions,
            # so allow a few float32 ULPs without weakening the semantic gate.
            torch.testing.assert_close(left, right, rtol=1e-4, atol=5e-5)

    def test_checkpoint_resume_is_exact_and_restores_rng(self) -> None:
        batch = make_batch(masked_targets=2)
        stream = FixedBatchStream(batch)
        config = self.train_config(max_steps=4, accumulation=2)

        seed_everything(777, deterministic=True)
        uninterrupted_model = CausalLM(self.model_config, dtype=torch.float32)
        uninterrupted = Trainer(
            uninterrupted_model,
            config,
            device="cpu",
            data_identity=stream.identity,
        )
        uninterrupted.train(stream)
        uninterrupted_optimizer = uninterrupted.optimizer.state_dict()
        uninterrupted_rng = capture_rng_state()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "nested" / "last.pt"
            seed_everything(777, deterministic=True)
            partial_model = CausalLM(self.model_config, dtype=torch.float32)
            partial = Trainer(
                partial_model,
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
                checkpoint_metadata={"wandb_run_id": "stable-run-id"},
            )
            partial.train(stream, until_step=2)
            partial.save_checkpoint()
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(any(checkpoint.parent.glob(f".{checkpoint.name}.*")))

            # Deliberately disturb all RNGs and model initialization before load.
            random.random()
            np.random.rand()
            torch.rand(3)
            resumed_model = CausalLM(self.model_config, dtype=torch.float32)
            resumed = Trainer(
                resumed_model,
                config,
                device="cpu",
                data_identity=stream.identity,
            )
            resumed.load_checkpoint(checkpoint)
            self.assertEqual(
                resumed.checkpoint_metadata["wandb_run_id"],
                "stable-run-id",
            )
            resumed.train(stream)

            for expected, actual in zip(
                uninterrupted_model.parameters(), resumed_model.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            assert_nested_equal(self, uninterrupted_optimizer, resumed.optimizer.state_dict())
            self.assertEqual(uninterrupted.state, resumed.state)
            resumed_rng = capture_rng_state()
            self.assertEqual(uninterrupted_rng["python"], resumed_rng["python"])
            np.testing.assert_array_equal(
                uninterrupted_rng["numpy"][1], resumed_rng["numpy"][1]
            )
            self.assertTrue(
                torch.equal(uninterrupted_rng["torch_cpu"], resumed_rng["torch_cpu"])
            )

    def test_checkpoint_rejects_changed_fixed_batch(self) -> None:
        first = FixedBatchStream(make_batch())
        changed_batch = make_batch()
        changed_batch["input_ids"][0, 0] = 42
        second = FixedBatchStream(changed_batch)
        config = self.train_config(max_steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            model = CausalLM(self.model_config, dtype=torch.float32)
            trainer = Trainer(
                model,
                config,
                device="cpu",
                data_identity=first.identity,
                checkpoint_path=checkpoint,
            )
            trainer.train(first)
            trainer.save_checkpoint()
            other = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=second.identity,
            )
            with self.assertRaisesRegex(ValueError, "data_identity mismatch"):
                other.load_checkpoint(checkpoint)

    def test_checkpointing_requires_both_tokenizer_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Checkpointing requires tokenizer"):
                ProductionTrainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    self.train_config(max_steps=1),
                    device="cpu",
                    data_identity="unbound-checkpoint-must-fail",
                    checkpoint_path=Path(temporary) / "last.pt",
                )

    def test_checkpoint_rejects_changed_same_size_tokenizer_vocabulary(self) -> None:
        stream = FixedBatchStream(make_batch())
        config = self.train_config(max_steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            source = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            source.save_checkpoint()
            changed = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                tokenizer_vocabulary_sha256="c" * 64,
            )
            with self.assertRaisesRegex(
                ValueError, "tokenizer_vocabulary_sha256 mismatch"
            ):
                changed.load_checkpoint(checkpoint)

    def test_checkpoint_rejects_changed_global_microbatch_rows(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            source = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            source.train(stream)
            source.save_checkpoint()
            changed = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1, global_microbatch_rows=2),
                device="cpu",
                data_identity=stream.identity,
            )
            with self.assertRaisesRegex(ValueError, "train_trajectory_config mismatch"):
                changed.load_checkpoint(checkpoint)

    def test_checkpoint_rejects_resume_under_planned_six_rank_world(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            source = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            source.save_checkpoint()

            # Simulate opening a one-rank checkpoint after relaunching on the
            # planned six-rank pod. Global rows are valid for that topology, but
            # exact resume must reject the changed world before model,
            # optimizer, or RNG restoration.
            with mock.patch(
                "pretrain.train._distributed_rank_world", return_value=(0, 6)
            ):
                changed_world = Trainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    self.train_config(max_steps=1, global_microbatch_rows=6),
                    device="cpu",
                    data_identity=stream.identity,
                )
            with self.assertRaisesRegex(ValueError, "world_size mismatch"):
                changed_world.load_checkpoint(checkpoint)

    def test_logger_failure_cannot_prevent_scheduled_checkpoint(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            config = dataclasses.replace(
                self.train_config(max_steps=1),
                checkpoint_every=1,
            )
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                logger=FailingLogger(),
                checkpoint_path=checkpoint,
            )
            trainer.train(stream)
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(trainer.state.completed_steps, 1)

    def test_checkpoint_rotation_keeps_previous_generation(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            previous = Path(temporary) / "last.previous.pt"
            config = dataclasses.replace(
                self.train_config(max_steps=2),
                checkpoint_every=1,
            )
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            trainer.train(stream)
            trainer.save_checkpoint()  # same step/path is deliberately a no-op
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(previous.is_file())
            latest_state = torch.load(checkpoint, weights_only=False)["train_state"]
            previous_state = torch.load(previous, weights_only=False)["train_state"]
            self.assertEqual(latest_state["completed_steps"], 2)
            self.assertEqual(previous_state["completed_steps"], 1)
            self.assertFalse(any(Path(temporary).glob(".*.link")))

    def test_new_latest_is_durable_before_previous_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            previous = Path(temporary) / "last.previous.pt"
            _atomic_torch_save({"generation": 0}, checkpoint)
            _atomic_torch_save({"generation": 1}, checkpoint)
            real_replace = __import__("os").replace

            def fail_previous_rotation(source, destination):
                if Path(destination) == previous:
                    raise OSError("injected previous rotation failure")
                return real_replace(source, destination)

            with mock.patch("pretrain.train.os.replace", side_effect=fail_previous_rotation):
                with self.assertRaisesRegex(OSError, "injected"):
                    _atomic_torch_save({"generation": 2}, checkpoint)
            self.assertEqual(
                torch.load(checkpoint, weights_only=False)["generation"], 2
            )
            self.assertEqual(
                torch.load(previous, weights_only=False)["generation"], 0
            )
            self.assertFalse(
                any(Path(temporary).glob(".*.previous-link-*"))
            )

    def test_previous_recovery_preserves_known_good_rollback_generation(self) -> None:
        stream = FixedBatchStream(make_batch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            previous = Path(temporary) / "last.previous.pt"
            config = dataclasses.replace(
                self.train_config(max_steps=2),
                checkpoint_every=1,
            )
            source = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            source.train(stream)
            self.assertEqual(
                torch.load(checkpoint, weights_only=False)["train_state"][
                    "completed_steps"
                ],
                2,
            )
            self.assertEqual(
                torch.load(previous, weights_only=False)["train_state"][
                    "completed_steps"
                ],
                1,
            )

            recovered = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
            )
            recovered.load_checkpoint(previous)
            self.assertTrue(recovered._preserve_previous_on_next_save)
            recovered.train(stream)
            self.assertFalse(recovered._preserve_previous_on_next_save)
            self.assertEqual(
                torch.load(checkpoint, weights_only=False)["train_state"][
                    "completed_steps"
                ],
                2,
            )
            # The rejected old latest was not rotated over the explicitly
            # selected rollback generation.
            self.assertEqual(
                torch.load(previous, weights_only=False)["train_state"][
                    "completed_steps"
                ],
                1,
            )

    def test_distributed_checkpoint_write_failure_is_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity="synthetic",
                checkpoint_path=checkpoint,
            )
            trainer.world_size = 2
            trainer.rank = 0
            rng = capture_rng_state()
            statuses: list[list[str | None]] = []

            def observe_status(status, *, src):
                self.assertEqual(src, 0)
                statuses.append(list(status))

            with (
                mock.patch.object(trainer, "_collect_rng_states", return_value=[rng, rng]),
                mock.patch(
                    "pretrain.train._atomic_torch_save",
                    side_effect=OSError("injected network-volume failure"),
                ),
                mock.patch(
                    "pretrain.train.dist.broadcast_object_list",
                    side_effect=observe_status,
                ),
                mock.patch("pretrain.train.dist.all_reduce"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Rank-zero checkpoint commit failed"
                ):
                    trainer.save_checkpoint()
            self.assertEqual(len(statuses), 1)
            self.assertIn("injected network-volume failure", statuses[0][0])
            self.assertIsNone(trainer._last_checkpoint)

    def test_logger_cannot_perturb_training_rng(self) -> None:
        stream = FixedBatchStream(make_batch())
        config = self.train_config(max_steps=2)
        seed_everything(55, deterministic=True)
        plain_model = CausalLM(self.model_config, dtype=torch.float32)
        plain = Trainer(
            plain_model,
            config,
            device="cpu",
            data_identity=stream.identity,
        )
        plain.train(stream)
        plain_rng = capture_rng_state()

        seed_everything(55, deterministic=True)
        logged_model = CausalLM(self.model_config, dtype=torch.float32)
        logged = Trainer(
            logged_model,
            config,
            device="cpu",
            data_identity=stream.identity,
            logger=RngConsumingLogger(),
        )
        logged.train(stream)
        logged_rng = capture_rng_state()
        for expected, actual in zip(
            plain_model.parameters(), logged_model.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(plain_rng["python"], logged_rng["python"])
        np.testing.assert_array_equal(plain_rng["numpy"][1], logged_rng["numpy"][1])
        self.assertTrue(torch.equal(plain_rng["torch_cpu"], logged_rng["torch_cpu"]))

    def test_validation_is_repeatable_distributed_order_and_preserves_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            order = build_synthetic_order(
                Path(temporary),
                sequence_length=8,
                vocab_size=64,
                split="validation",
            )
            loader, sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            try:
                runner = ValidationRunner(
                    loader,
                    device="cpu",
                    precision="float32",
                    max_batches=2,
                )
                model = CausalLM(self.model_config, dtype=torch.float32).train()
                before = capture_rng_state()
                first = runner.evaluate(model, train_step=7)
                after = capture_rng_state()
                second = runner.evaluate(model, train_step=7)
                self.assertTrue(model.training)
                self.assertEqual(first["validation/loss"], second["validation/loss"])
                self.assertEqual(first["validation/batches"], 2)
                self.assertEqual(first["validation/rows"], 4)
                self.assertEqual(
                    first["validation/input_tokens"],
                    sum(
                        int(first[f"validation/{domain}/input_tokens"])
                        for domain in ("python", "other_code", "english")
                    ),
                )
                self.assertEqual(
                    first["validation/rows"],
                    sum(
                        int(first[f"validation/{domain}/rows"])
                        for domain in ("python", "other_code", "english")
                    ),
                )
                self.assertEqual(
                    first["validation/supervised_tokens"],
                    sum(
                        int(first[f"validation/{domain}/supervised_tokens"])
                        for domain in ("python", "other_code", "english")
                    ),
                )
                reconstructed_loss_sum = sum(
                    float(first.get(f"validation/{domain}/loss", 0.0))
                    * int(first[f"validation/{domain}/supervised_tokens"])
                    for domain in ("python", "other_code", "english")
                )
                self.assertTrue(
                    torch.isclose(
                        torch.tensor(reconstructed_loss_sum),
                        torch.tensor(float(first["validation/loss_sum"])),
                        rtol=1e-6,
                        atol=1e-6,
                    )
                )
                self.assertEqual(before["python"], after["python"])
                np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
                self.assertTrue(torch.equal(before["torch_cpu"], after["torch_cpu"]))
            finally:
                sampler.close()
                loader.dataset.close()

    def test_validation_rejects_divergent_per_domain_loss_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            order = build_synthetic_order(
                Path(temporary),
                sequence_length=8,
                vocab_size=64,
                split="validation",
            )
            loader, sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            try:
                runner = ValidationRunner(
                    loader,
                    device="cpu",
                    precision="float32",
                    max_batches=1,
                )
                model = CausalLM(self.model_config, dtype=torch.float32)
                original_forward = model.forward

                def divergent_forward(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    assert output.loss_sums_per_row is not None
                    return output._replace(
                        loss_sums_per_row=output.loss_sums_per_row + 1.0
                    )

                with (
                    mock.patch.object(model, "forward", side_effect=divergent_forward),
                    self.assertRaisesRegex(ValueError, "domain loss sums"),
                ):
                    runner.evaluate(model, train_step=0)
            finally:
                sampler.close()
                loader.dataset.close()

    def test_step_zero_validation_is_checkpointed_and_not_repeated_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_order = build_synthetic_order(
                root / "validation",
                sequence_length=8,
                vocab_size=64,
                split="validation",
            )
            validation_loader, validation_sampler = create_training_dataloader(
                validation_order,
                global_microbatch_rows=1,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            checkpoint = root / "last.pt"
            controller = GracefulStopController()
            logger = StopOnValidationLogger(controller)
            config = dataclasses.replace(
                self.train_config(max_steps=1),
                eval_at_start=True,
            )
            runner = ValidationRunner(
                validation_loader,
                device="cpu",
                precision="float32",
                max_batches=1,
            )
            stream = FixedBatchStream(make_batch())
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=stream.identity,
                logger=logger,
                checkpoint_path=checkpoint,
                validation_runner=runner,
                stop_controller=controller,
            )
            try:
                self.assertEqual(trainer.train(stream), {})
                self.assertEqual(trainer.state.completed_steps, 0)
                self.assertEqual(trainer.stop_signal, int(signal.SIGTERM))
                validation_records = [
                    metrics
                    for metrics in logger.metrics
                    if "validation/loss" in metrics
                ]
                self.assertEqual(len(validation_records), 1)
                checkpoint_index = next(
                    index
                    for index, metrics in enumerate(logger.metrics)
                    if metrics.get("checkpoint/completed") == 1
                )
                stop_index = next(
                    index
                    for index, metrics in enumerate(logger.metrics)
                    if "system/graceful_stop_signal" in metrics
                )
                self.assertLess(checkpoint_index, stop_index)
                payload = torch.load(checkpoint, weights_only=False)
                self.assertIs(
                    payload["metadata"]["validation_at_start_completed"],
                    True,
                )

                resume_logger = MemoryLogger()
                resumed = Trainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    config,
                    device="cpu",
                    data_identity=stream.identity,
                    logger=resume_logger,
                    checkpoint_path=checkpoint,
                    validation_runner=runner,
                )
                resumed.load_checkpoint(checkpoint)
                self.assertFalse(resumed._validation_at_start_due())
                self.assertEqual(resumed.train(stream, until_step=0), {})
                self.assertFalse(
                    any("validation/loss" in metrics for metrics in resume_logger.metrics)
                )
            finally:
                validation_sampler.close()
                validation_loader.dataset.close()

    def test_validation_runs_on_schedule_and_is_checkpoint_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_order = build_synthetic_order(
                root / "train",
                sequence_length=8,
                vocab_size=64,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
            )
            validation_order = build_synthetic_order(
                root / "validation",
                sequence_length=8,
                vocab_size=64,
                split="validation",
            )
            validation_loader, validation_sampler = create_training_dataloader(
                validation_order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            train_loader, train_sampler = create_training_dataloader(
                train_order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            checkpoint = root / "last.pt"
            logger = MemoryLogger()
            config = dataclasses.replace(
                self.train_config(
                    max_steps=5,
                    accumulation=1,
                    global_microbatch_rows=2,
                ),
                eval_every=2,
            )
            runner = ValidationRunner(
                validation_loader,
                device="cpu",
                precision="float32",
                max_batches=2,
            )
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity=train_sampler.data_identity,
                logger=logger,
                checkpoint_path=checkpoint,
                training_geometry=frozen_training_geometry(train_order),
                validation_runner=runner,
            )
            try:
                trainer.train(train_loader)
                trainer.save_checkpoint()
                validation_steps = [
                    int(metrics["train/step"])
                    for metrics in logger.metrics
                    if "validation/loss" in metrics
                ]
                self.assertEqual(validation_steps, [2, 4, 5])
                self.assertEqual(trainer.state.last_validated_step, 5)
                self.assertEqual(
                    torch.load(checkpoint, weights_only=False)["train_state"][
                        "last_validated_step"
                    ],
                    5,
                )

                changed_runner = ValidationRunner(
                    validation_loader,
                    device="cpu",
                    precision="float32",
                    max_batches=1,
                )
                changed = Trainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    config,
                    device="cpu",
                    data_identity=train_sampler.data_identity,
                    training_geometry=frozen_training_geometry(train_order),
                    validation_runner=changed_runner,
                )
                with self.assertRaisesRegex(
                    ValueError, "validation_configuration mismatch"
                ):
                    changed.load_checkpoint(checkpoint)
            finally:
                train_sampler.close()
                train_loader.dataset.close()
                validation_sampler.close()
                validation_loader.dataset.close()

    def test_final_preemption_checkpoint_catches_up_validation_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_order = build_synthetic_order(
                root / "train",
                sequence_length=8,
                vocab_size=64,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
            )
            validation_order = build_synthetic_order(
                root / "validation",
                sequence_length=8,
                vocab_size=64,
                split="validation",
            )
            train_loader, train_sampler = create_training_dataloader(
                train_order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            validation_loader, validation_sampler = create_training_dataloader(
                validation_order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            checkpoint = root / "last.pt"
            controller = GracefulStopController()
            config = dataclasses.replace(
                self.train_config(
                    max_steps=5,
                    accumulation=1,
                    global_microbatch_rows=2,
                ),
                eval_every=5,
            )
            runner = ValidationRunner(
                validation_loader,
                device="cpu",
                precision="float32",
                max_batches=2,
            )
            interrupted_model = CausalLM(self.model_config, dtype=torch.float32)
            interrupted = Trainer(
                interrupted_model,
                config,
                device="cpu",
                data_identity=train_sampler.data_identity,
                logger=StopAtStepLogger(controller, step=5),
                checkpoint_path=checkpoint,
                training_geometry=frozen_training_geometry(train_order),
                validation_runner=runner,
                stop_controller=controller,
            )
            resumed_loader = None
            resumed_sampler = None
            try:
                interrupted.train(train_loader)
                self.assertEqual(interrupted.stop_signal, int(signal.SIGTERM))
                saved = torch.load(checkpoint, weights_only=False)
                self.assertEqual(saved["train_state"]["completed_steps"], 5)
                self.assertEqual(saved["train_state"]["last_validated_step"], 0)
                expected_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in interrupted_model.named_parameters()
                }

                resume_logger = MemoryLogger()
                resumed = Trainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    config,
                    device="cpu",
                    data_identity=train_sampler.data_identity,
                    logger=resume_logger,
                    checkpoint_path=checkpoint,
                    training_geometry=frozen_training_geometry(train_order),
                    validation_runner=runner,
                )
                resumed.load_checkpoint(checkpoint)
                resumed_loader, resumed_sampler = create_training_dataloader(
                    train_order,
                    global_microbatch_rows=2,
                    gradient_accumulation_steps=1,
                    start_global_microbatch=resumed.state.completed_microbatches,
                    num_workers=0,
                    pin_memory=False,
                )
                resumed.train(resumed_loader)
                validation_steps = [
                    metrics["train/step"]
                    for metrics in resume_logger.metrics
                    if "validation/loss" in metrics
                ]
                self.assertEqual(validation_steps, [5])
                self.assertEqual(resumed.state.last_validated_step, 5)
                for name, parameter in resumed.raw_model.named_parameters():
                    self.assertTrue(torch.equal(parameter, expected_parameters[name]))
                self.assertEqual(
                    torch.load(checkpoint, weights_only=False)["train_state"][
                        "last_validated_step"
                    ],
                    5,
                )
            finally:
                train_sampler.close()
                train_loader.dataset.close()
                if resumed_loader is not None and resumed_sampler is not None:
                    resumed_sampler.close()
                    resumed_loader.dataset.close()
                validation_sampler.close()
                validation_loader.dataset.close()

    def test_fixed_batch_stream_accepts_authenticated_shared_identity(self) -> None:
        batch = make_batch()
        stream = FixedBatchStream(
            batch,
            identity="fixed-global-batch-sha256:" + "a" * 64,
        )
        self.assertEqual(
            stream.identity,
            "fixed-global-batch-sha256:" + "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            FixedBatchStream(batch, identity="")

    def test_graceful_stop_checkpoints_clean_boundary_and_resumes_exactly(self) -> None:
        stream = FixedBatchStream(make_batch(masked_targets=2))
        config = self.train_config(max_steps=4, accumulation=2)
        seed_everything(919, deterministic=True)
        full_model = CausalLM(self.model_config, dtype=torch.float32)
        full = Trainer(
            full_model,
            config,
            device="cpu",
            data_identity=stream.identity,
        )
        full.train(stream)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            controller = GracefulStopController()
            seed_everything(919, deterministic=True)
            interrupted_model = CausalLM(self.model_config, dtype=torch.float32)
            interrupted = Trainer(
                interrupted_model,
                config,
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
                logger=StopAtStepLogger(controller),
                stop_controller=controller,
            )
            interrupted.train(stream)
            self.assertEqual(interrupted.state.completed_steps, 1)
            self.assertEqual(interrupted.stop_signal, int(signal.SIGTERM))
            self.assertTrue(checkpoint.is_file())

            resumed_model = CausalLM(self.model_config, dtype=torch.float32)
            resumed = Trainer(
                resumed_model,
                config,
                device="cpu",
                data_identity=stream.identity,
            )
            resumed.load_checkpoint(checkpoint)
            resumed.train(stream)
            for expected, actual in zip(
                full_model.parameters(), resumed_model.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            assert_nested_equal(
                self,
                full.optimizer.state_dict(),
                resumed.optimizer.state_dict(),
            )

    def test_pending_stop_is_checkpointed_then_reported_before_any_update(self) -> None:
        stream = FixedBatchStream(make_batch())
        controller = GracefulStopController()
        controller.request(int(signal.SIGTERM))
        logger = MemoryLogger()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity=stream.identity,
                logger=logger,
                checkpoint_path=checkpoint,
                stop_controller=controller,
            )
            self.assertEqual(trainer.train(stream), {})
            self.assertEqual(trainer.state.completed_steps, 0)
            self.assertEqual(
                torch.load(checkpoint, weights_only=False)["train_state"][
                    "completed_steps"
                ],
                0,
            )
            self.assertEqual(
                [
                    metrics.get("checkpoint/completed", 0)
                    for metrics in logger.metrics
                ],
                [1, 0],
            )
            self.assertEqual(
                logger.metrics[-1]["system/graceful_stop_signal"],
                int(signal.SIGTERM),
            )

    def test_external_supervisor_stop_request_and_finalization_window(self) -> None:
        stream = FixedBatchStream(make_batch())
        controller = GracefulStopController()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "stop-request"
            checkpoint = root / "last.pt"
            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                self.train_config(max_steps=1),
                device="cpu",
                data_identity=stream.identity,
                checkpoint_path=checkpoint,
                stop_controller=controller,
            )

            original_save = trainer.save_checkpoint

            def save_then_receive_preemption(path=None):
                original_save(path)
                request.write_text(str(int(signal.SIGTERM)), encoding="ascii")

            with (
                mock.patch.dict(
                    "os.environ",
                    {"PRETRAIN_STOP_REQUEST_FILE": str(request)},
                    clear=False,
                ),
                mock.patch.object(
                    trainer,
                    "save_checkpoint",
                    side_effect=save_then_receive_preemption,
                ),
            ):
                exit_code = _finalize_training_run(trainer)
            self.assertEqual(exit_code, 128 + int(signal.SIGTERM))
            self.assertEqual(trainer.stop_signal, int(signal.SIGTERM))
            self.assertTrue(checkpoint.is_file())

    def test_compile_setting_must_match_model_wrapper(self) -> None:
        config = dataclasses.replace(self.train_config(max_steps=1), compile_model=True)
        with self.assertRaisesRegex(ValueError, "compile_model"):
            Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity="compile-mismatch",
            )

    def test_production_loader_resume_does_not_perturb_model_rng(self) -> None:
        config = self.train_config(
            max_steps=5,
            accumulation=1,
            global_microbatch_rows=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = build_synthetic_order(
                root,
                sequence_length=8,
                vocab_size=64,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
            )
            geometry = frozen_training_geometry(order)
            self.assertLess(
                geometry["consumed_supervised_tokens"],
                geometry["consumed_input_tokens"],
            )
            data_identity = (
                f"order-manifest-sha256:{sha256_file(order)};"
                f"order-payload-sha256:{verify_order_payload_checksum(order)}"
            )

            seed_everything(42, deterministic=True)
            full_model = CausalLM(self.model_config, dtype=torch.float32)
            full = Trainer(
                full_model,
                config,
                device="cpu",
                data_identity=data_identity,
                training_geometry=geometry,
            )
            full_loader, full_sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                num_workers=0,
                pin_memory=False,
            )
            try:
                full.train(full_loader)
            finally:
                full_sampler.close()
                full_loader.dataset.close()
            self.assertEqual(full.state.consumed_rows, geometry["consumed_rows"])
            self.assertEqual(
                full.state.consumed_input_tokens,
                geometry["consumed_input_tokens"],
            )
            self.assertEqual(
                full.state.consumed_supervised_tokens,
                geometry["consumed_supervised_tokens"],
            )
            full_rng = capture_rng_state()

            checkpoint = root / "checkpoint.pt"
            seed_everything(42, deterministic=True)
            partial_model = CausalLM(self.model_config, dtype=torch.float32)
            partial = Trainer(
                partial_model,
                config,
                device="cpu",
                data_identity=data_identity,
                checkpoint_path=checkpoint,
                training_geometry=geometry,
            )
            partial_loader, partial_sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                num_workers=0,
                pin_memory=False,
            )
            try:
                partial.train(partial_loader, until_step=2)
                partial.save_checkpoint()
            finally:
                partial_sampler.close()
                partial_loader.dataset.close()

            resumed_model = CausalLM(self.model_config, dtype=torch.float32)
            resumed = Trainer(
                resumed_model,
                config,
                device="cpu",
                data_identity=data_identity,
                training_geometry=geometry,
            )
            resumed.load_checkpoint(checkpoint)
            resumed_loader, resumed_sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                start_global_microbatch=resumed.state.completed_microbatches,
                num_workers=0,
                pin_memory=False,
            )
            try:
                resumed.train(resumed_loader)
            finally:
                resumed_sampler.close()
                resumed_loader.dataset.close()

            for expected, actual in zip(
                full_model.parameters(), resumed_model.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            resumed_rng = capture_rng_state()
            self.assertTrue(torch.equal(full_rng["torch_cpu"], resumed_rng["torch_cpu"]))

    def test_frozen_geometry_rejects_trainer_and_loader_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = build_synthetic_order(
                root,
                sequence_length=8,
                vocab_size=64,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
            )
            geometry = frozen_training_geometry(order)
            config = self.train_config(
                max_steps=5,
                accumulation=1,
                global_microbatch_rows=2,
            )
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                Trainer(
                    CausalLM(self.model_config, dtype=torch.float32),
                    dataclasses.replace(config, gradient_accumulation_steps=2),
                    device="cpu",
                    data_identity="irrelevant",
                    training_geometry=geometry,
                )

            trainer = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity="wrong-order-identity",
                training_geometry=geometry,
            )
            loader, sampler = create_training_dataloader(
                order,
                global_microbatch_rows=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
            )
            try:
                with self.assertRaisesRegex(ValueError, "data_identity mismatch"):
                    trainer.train(loader)
                self.assertEqual(trainer.state, type(trainer.state)())
            finally:
                sampler.close()
                loader.dataset.close()

            checkpoint = root / "geometry-checkpoint.pt"
            source = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity="frozen-order",
                checkpoint_path=checkpoint,
                training_geometry=geometry,
            )
            source.save_checkpoint()
            changed_geometry = copy.deepcopy(geometry)
            changed_geometry["dropped_rows"] += 1
            changed_geometry["dropped_input_tokens"] += 8
            changed = Trainer(
                CausalLM(self.model_config, dtype=torch.float32),
                config,
                device="cpu",
                data_identity="frozen-order",
                training_geometry=changed_geometry,
            )
            with self.assertRaisesRegex(ValueError, "training_geometry mismatch"):
                changed.load_checkpoint(checkpoint)

    def test_single_chunk_comes_through_real_packed_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            order = build_synthetic_order(
                Path(temporary),
                sequence_length=8,
                vocab_size=64,
                global_microbatch_rows=2,
                gradient_accumulation_steps=2,
            )
            batch = load_one_packed_batch(order, batch_size=1)
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 8))
        self.assertEqual(tuple(batch["labels"].shape), (1, 8))
        self.assertGreater(int(batch["num_loss_tokens"]), 0)
        self.assertEqual(batch["input_ids"].dtype, torch.int64)
        self.assertGreater(int(batch["document_ids"].max()), 0)
        self.assertTrue(torch.any(batch["position_ids"][:, 1:] == 0))
        self.assertTrue(torch.any(batch["labels"] == -100))

    def test_order_payload_checksum_is_verified_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            order_manifest = build_synthetic_order(
                Path(temporary),
                sequence_length=8,
                vocab_size=64,
            )
            self.assertEqual(len(verify_order_payload_checksum(order_manifest)), 64)
            order_path = order_manifest.parent / "order.bin"
            with order_path.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(IOError, "checksum mismatch"):
                verify_order_payload_checksum(order_manifest)

    def test_checkpoint_lease_and_exact_temporary_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            unrelated = checkpoint.parent / ".last.pt.user-note"
            stale_checkpoint = (
                checkpoint.parent / ".last.pt.checkpoint-part-stale"
            )
            stale_previous_link = (
                checkpoint.parent
                / ".last.previous.pt.previous-link-123-456"
            )
            unrelated.write_text("keep", encoding="utf-8")
            stale_checkpoint.write_bytes(b"partial")
            stale_previous_link.write_bytes(b"link")
            first = CheckpointLease(checkpoint)
            second = CheckpointLease(checkpoint)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "holds checkpoint lease"):
                    second.acquire()
                removed = reconcile_checkpoint_temporaries(checkpoint)
                self.assertEqual(
                    {path.name for path in removed},
                    {stale_checkpoint.name, stale_previous_link.name},
                )
                self.assertTrue(unrelated.is_file())
                self.assertTrue(first.path.is_file())
            finally:
                first.release()
            second.acquire()
            second.release()


if __name__ == "__main__":
    unittest.main()
