from __future__ import annotations

import json
import io
import os
import signal
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest import mock

from pretrain import geometry_evidence
from pretrain import geometry_qualification as qualification
from scripts import qualify_training_geometry as cli


GIB = 1024**3


class GeometryQualificationTest(unittest.TestCase):
    def _baseline(
        self,
        root: Path,
        *,
        hardware_identity_sha256: str = "a" * 64,
        sequence_length: int = 16,
    ) -> tuple[Path, dict[str, object]]:
        candidates: dict[str, object] = {}
        results: dict[str, object] = {}
        plan: dict[str, object] = {
            "format": qualification.BASELINE_PLAN_FORMAT,
            "format_version": qualification.FORMAT_VERSION,
            "mode": "single-gpu-baselines",
            "inputs": {
                "hardware": {
                    "geometry_identity": {
                        "identity_sha256": hardware_identity_sha256
                    }
                },
                "common": {"sequence_length": sequence_length},
            },
        }
        plan["plan_sha256"] = qualification.canonical_sha256(plan)
        plan_bound = qualification.publish_json_new(root / "baseline-plan.json", plan)
        shared_order_sha256 = "d" * 64
        for candidate in qualification.CANDIDATES:
            token_delta = (
                100
                * candidate.local_microbatch_rows
                * candidate.gradient_accumulation_steps
                * sequence_length
            )
            elapsed = token_delta * 1_000_000
            candidates[candidate.candidate_id] = {
                "global_microbatch_rows": candidate.local_microbatch_rows,
                "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
                "compile_model": candidate.compile_model,
                "gpu_count": 1,
                "scope": geometry_evidence.THROUGHPUT_SCOPE,
                "timer": geometry_evidence.THROUGHPUT_TIMER,
                "counter": geometry_evidence.THROUGHPUT_COUNTER,
                "start_consumed_input_tokens": 0,
                "end_consumed_input_tokens": token_delta,
                "elapsed_wall_time_ns": elapsed,
                "aggregate_input_tokens_per_second": "1000",
            }
            result_payload = {
                "format": qualification.BASELINE_CANDIDATE_RESULT_FORMAT,
                "format_version": qualification.FORMAT_VERSION,
                "status": "pass",
                "plan_sha256": plan["plan_sha256"],
                "candidate": candidate.as_dict(),
                "order": {"order": {"sha256": shared_order_sha256}},
                "measurement": candidates[candidate.candidate_id],
            }
            result_path = root / f"{candidate.candidate_id}.json"
            results[candidate.candidate_id] = qualification.publish_json_new(
                result_path, result_payload
            )
        payload: dict[str, object] = {
            "format": qualification.BASELINE_FORMAT,
            "format_version": qualification.FORMAT_VERSION,
            "status": "pass",
            "hardware_identity_sha256": hardware_identity_sha256,
            "producer_plan": plan_bound,
            "exact_shared_order_payload_sha256": shared_order_sha256,
            "results": results,
            "candidates": candidates,
        }
        path = root / "baselines.json"
        bound = qualification.publish_json_new(path, payload)
        return path, bound

    def test_grid_is_exactly_equal_effective_batch_and_commands_freeze_geometry(
        self,
    ) -> None:
        self.assertEqual(len(qualification.CANDIDATES), 6)
        self.assertEqual(
            {candidate.optimizer_update_rows for candidate in qualification.CANDIDATES},
            {192},
        )
        self.assertEqual(
            qualification.FINAL_TRAIN_EXPECTED_ROWS,
            qualification.FINAL_TRAIN_EXPECTED_OPTIMIZER_UPDATES * 192,
        )
        self.assertEqual(
            qualification.FINAL_TRAIN_EXPECTED_CONSUMED_INPUT_TOKENS,
            52_579_270_656,
        )
        self.assertLessEqual(
            qualification.FINAL_TRAIN_TARGET_INPUT_TOKENS
            - qualification.FINAL_TRAIN_EXPECTED_CONSUMED_INPUT_TOKENS,
            192 * 4_096 - 1,
        )
        settings = qualification.SoakSettings()
        for candidate in qualification.CANDIDATES:
            command = qualification.render_torchrun_command(
                python_executable=Path("/python"),
                order_manifest=Path("/order.json"),
                validation_order_manifest=Path("/validation.json"),
                tokenizer_root=Path("/tokenizer"),
                checkpoint=Path("/checkpoints/last.pt"),
                candidate=candidate,
                settings=settings,
                run_name="safe-name",
                run_group="safe-group",
                resume=True,
            )
            self.assertIn("--deterministic", command)
            self.assertIn("--activation-checkpointing", command)
            self.assertIn("--fused-adamw", command)
            self.assertEqual(
                command[command.index("--global-microbatch-rows") + 1],
                str(candidate.global_microbatch_rows),
            )
            self.assertEqual(
                command[command.index("--gradient-accumulation-steps") + 1],
                str(candidate.gradient_accumulation_steps),
            )
            self.assertEqual(command.count("--compile"), int(candidate.compile_model))
            self.assertEqual(command[-2:], ["--resume", "/checkpoints/last.pt"])

    def test_online_wandb_mode_and_run_name_are_explicitly_forwarded(self) -> None:
        settings = qualification.SoakSettings(
            wandb_mode="online",
            wandb_project="live-project",
            wandb_run_name_prefix="h100-qualification",
        )
        settings.validate()
        command = qualification.render_torchrun_command(
            python_executable=Path("/python"),
            order_manifest=Path("/order.json"),
            validation_order_manifest=Path("/validation.json"),
            tokenizer_root=Path("/tokenizer"),
            checkpoint=Path("/last.pt"),
            candidate=qualification.CANDIDATES[0],
            settings=settings,
            run_name="h100-qualification-candidate",
            run_group="group",
            resume=False,
        )
        self.assertEqual(command[command.index("--wandb-mode") + 1], "online")
        self.assertEqual(
            command[command.index("--wandb-project") + 1], "live-project"
        )
        self.assertEqual(
            command[command.index("--wandb-run-name") + 1],
            "h100-qualification-candidate",
        )
        with self.assertRaisesRegex(
            qualification.GeometryQualificationError, "disabled"
        ):
            qualification.SoakSettings(wandb_mode="disabled").validate()

    def test_phase_observation_uses_external_counter_and_requests_one_stop(self) -> None:
        observation = qualification.PhaseObservation()
        stops: list[bool] = []
        clock = iter((1_000, 2_000))
        for step in range(1, 8):
            payload = {
                "train/step": step,
                "train/input_tokens": step * 192 * 16,
                "perf/input_tokens_per_second": 999999,
                "perf/data_wait_fraction": 0.01,
                "system/cuda_peak_memory_allocated_bytes": 10,
                "system/cuda_peak_memory_reserved_bytes": 12,
            }
            observation.feed(
                json.dumps(payload),
                clock_ns=lambda: next(clock),
                start_after_step=5,
                request_stop_after_step=7,
                publish_stop=lambda: stops.append(True),
            )
        observation.feed(
            json.dumps({"train/step": 7, "system/graceful_stop_signal": 10}),
            clock_ns=lambda: 3_000,
            start_after_step=5,
            request_stop_after_step=7,
            publish_stop=lambda: stops.append(True),
        )
        self.assertEqual(observation.start_monotonic_ns, 1_000)
        self.assertEqual(observation.start_consumed_input_tokens, 5 * 192 * 16)
        self.assertEqual(observation.stop_request_monotonic_ns, 2_000)
        self.assertEqual(stops, [True])
        self.assertEqual(observation.graceful_stop_events, 1)

    def test_timed_metric_without_data_wait_fraction_fails_closed(self) -> None:
        observation = qualification.PhaseObservation(
            start_monotonic_ns=1,
            start_step=1,
            start_consumed_input_tokens=1,
        )
        with self.assertRaisesRegex(
            qualification.GeometryQualificationError, "data_wait_fraction"
        ):
            observation.feed(
                json.dumps(
                    {
                        "train/step": 2,
                        "train/input_tokens": 2,
                        "perf/input_tokens_per_second": 1,
                    }
                ),
                clock_ns=lambda: 2,
                start_after_step=None,
                request_stop_after_step=None,
                publish_stop=None,
            )

    def test_real_supervisor_requests_stop_and_durably_captures_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "worker.py"
            worker.write_text(
                """import json, os, pathlib, time
stop = pathlib.Path(os.environ['PRETRAIN_STOP_REQUEST_FILE'])
for step in range(1, 20):
    print(json.dumps({
        'train/step': step,
        'train/input_tokens': step * 3072,
        'perf/input_tokens_per_second': 1,
        'perf/data_wait_fraction': 0.01,
        'system/cuda_peak_memory_allocated_bytes': 10,
        'system/cuda_peak_memory_reserved_bytes': 12,
    }), flush=True)
    if stop.exists():
        print(json.dumps({
            'train/step': step,
            'system/graceful_stop_signal': 10,
        }), flush=True)
        break
    time.sleep(0.01)
""",
                encoding="utf-8",
            )
            run = qualification.supervise_phase(
                command=[sys.executable, str(worker)],
                environment=os.environ,
                log_path=root / "phase.log",
                stop_request_path=root / "stop",
                working_directory=root,
                settings=qualification.SoakSettings(
                    measurement_warmup_steps=1,
                    minimum_soak_steps=2,
                    stop_after_soak_steps=1,
                    order_buffer_steps=2,
                    eval_every=1,
                    phase_timeout_seconds=10,
                    graceful_shutdown_seconds=5,
                    gpu_poll_interval_seconds="0.01",
                ),
                start_after_step=1,
                request_stop_after_step=3,
                gpu_sampler=lambda: [(80 * GIB, 70 * GIB)] * 6,
            )
            self.assertEqual(run.return_code, 0)
            self.assertEqual(run.observation.start_step, 1)
            self.assertGreaterEqual(run.observation.last_step, 3)
            self.assertEqual(run.observation.graceful_stop_events, 1)
            self.assertGreater(run.gpu_memory.samples, 0)
            self.assertEqual(
                qualification.artifact(root / "phase.log", label="phase log"),
                run.log,
            )

    def test_nvidia_memory_normalization_is_conservative(self) -> None:
        rows = [(82 * GIB, 20 * GIB)] * 6
        self.assertEqual(
            qualification._normalized_nvidia_free_memory(  # noqa: SLF001
                rows, physical_memory_bytes=80 * GIB
            ),
            18 * GIB,
        )
        memory = qualification.GPUMemoryObservation()
        memory.add(rows)
        memory.add([(82 * GIB, 19 * GIB)] * 6)
        self.assertEqual(memory.samples, 2)
        self.assertEqual(memory.minimum_free_bytes_per_gpu, 19 * GIB)

    def test_grid_storage_admission_reserves_all_twelve_generations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            network = Path(temporary)
            root = network / "grid"
            root.mkdir()
            plan = {
                "mode": "grid",
                "settings": vars(qualification.SoakSettings()),
                "inputs": {
                    "hardware": {
                        "expected": {
                            "qualification": {
                                "host": {
                                    "storage": {
                                        "network": {
                                            "path": str(network.resolve()),
                                            "classification": "network",
                                            "read_only": False,
                                            "device": network.stat().st_dev,
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
            estimate = qualification.SoakSettings().checkpoint_generation_bytes
            with mock.patch.object(
                qualification.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(
                    total=500 * GIB, used=1, free=500 * GIB
                ),
            ):
                result = qualification.verify_output_storage(root, plan=plan)
            self.assertEqual(result["projected_checkpoint_generations"], 12)
            self.assertEqual(result["projected_checkpoint_bytes"], 12 * estimate)
            self.assertGreater(result["required_free_bytes"], 12 * estimate)

    def test_storage_admission_rejects_checkpoint_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            network = Path(temporary)
            root = network / "grid"
            checkpoint_parent = (
                root
                / "candidates"
                / qualification.CANDIDATES[0].candidate_id
            )
            checkpoint_parent.mkdir(parents=True)
            outside = network / "outside"
            outside.mkdir()
            (checkpoint_parent / "checkpoints").symlink_to(
                outside, target_is_directory=True
            )
            plan = {
                "mode": "grid",
                "settings": vars(qualification.SoakSettings()),
                "inputs": {
                    "hardware": {
                        "expected": {
                            "qualification": {
                                "host": {
                                    "storage": {
                                        "network": {
                                            "path": str(network.resolve()),
                                            "classification": "network",
                                            "read_only": False,
                                            "device": network.stat().st_dev,
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
            with self.assertRaisesRegex(
                qualification.GeometryQualificationError, "symlink"
            ):
                qualification.verify_output_storage(root, plan=plan)

    def test_live_runtime_is_bound_to_environment_interpreter_and_gpu_uuids(self) -> None:
        required = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "WANDB_MODE": "offline",
        }
        environment = {**required, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}
        devices = [
            {
                "physical_index": index,
                "uuid": f"GPU-{index:032x}",
                "name": "Fixture GPU",
                "nvidia_smi_memory_bytes": 80 * GIB,
            }
            for index in range(6)
        ]
        hardware = {
            "gpu_model": "Fixture GPU",
            "gpu_memory_bytes": 80 * GIB,
            "compute_capability": [9, 0],
            "multiprocessor_count": 100,
            "torch_version": "2.7.1+cu128",
            "cuda_runtime_version": "12.8",
            "qualification": {
                "host": {
                    "environment": {
                        "required": required,
                        "cuda_visible_devices": ["0", "1", "2", "3", "4", "5"],
                    }
                },
                "gpu": {
                    "devices": devices,
                    "nvidia_smi_executable": {"path": "/usr/bin/nvidia-smi"},
                },
            },
        }
        runtime = types.SimpleNamespace(
            cuda_devices=6,
            world_size=6,
            bf16_supported_devices=list(range(6)),
            torch_version="2.7.1+cu128",
            cuda_runtime="12.8",
            python_executable=sys.executable,
            python_executable_sha256="f" * 64,
            cuda_device_profiles=[
                {
                    "name": "Fixture GPU",
                    "total_memory_bytes": 80 * GIB,
                    "compute_capability": [9, 0],
                    "multiprocessor_count": 100,
                }
                for _ in range(6)
            ],
        )
        plan = {
            "output_root": "/unused",
            "inputs": {"hardware": {"expected": hardware}},
            "runtime": {
                "python": {
                    "invocation_path": sys.executable,
                    "resolved": {"sha256": "f" * 64},
                }
            },
        }
        identities = [
            {
                "index": index,
                "uuid": f"GPU-{index:032x}",
                "name": "Fixture GPU",
                "memory_bytes": 80 * GIB,
            }
            for index in range(6)
        ]
        with (
            mock.patch.object(
                qualification,
                "verify_plan_artifacts",
                return_value={"artifact_descriptors_verified": 1},
            ),
            mock.patch(
                "scripts.launch_pretraining.inspect_runtime", return_value=runtime
            ),
            mock.patch.object(
                qualification,
                "sample_nvidia_smi_memory",
                return_value=[(80 * GIB, 70 * GIB)] * 6,
            ),
            mock.patch.object(
                qualification,
                "sample_nvidia_smi_identity",
                return_value=identities,
            ),
            mock.patch.object(
                qualification,
                "verify_output_storage",
                return_value={"status": "pass"},
            ),
        ):
            result = qualification.verify_live_runtime(
                plan=plan, environment=environment
            )
            self.assertEqual(result["status"], "pass")
            changed = dict(environment, CUDA_VISIBLE_DEVICES="5,4,3,2,1,0")
            with self.assertRaisesRegex(
                qualification.GeometryQualificationError, "CUDA_VISIBLE_DEVICES"
            ):
                qualification.verify_live_runtime(plan=plan, environment=changed)

    def test_baseline_receipt_reconciles_and_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self._baseline(root)
            rates, _ = qualification.load_single_gpu_baselines(
                path,
                hardware_identity_sha256="a" * 64,
                sequence_length=16,
            )
            self.assertEqual(set(rates), {
                candidate.candidate_id for candidate in qualification.CANDIDATES
            })
            self.assertEqual(set(rates.values()), {Decimal("1000")})
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                qualification.GeometryQualificationError, "sidecar"
            ):
                qualification.load_single_gpu_baselines(
                    path,
                    hardware_identity_sha256="a" * 64,
                    sequence_length=16,
                )

    def test_baseline_shorter_than_one_hundred_updates_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self._baseline(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            candidate = qualification.CANDIDATES[0]
            entry = payload["candidates"][candidate.candidate_id]
            one_update_tokens = (
                candidate.local_microbatch_rows
                * candidate.gradient_accumulation_steps
                * 16
            )
            entry["end_consumed_input_tokens"] = one_update_tokens
            entry["elapsed_wall_time_ns"] = one_update_tokens * 1_000_000
            short = root / "short-baseline.json"
            qualification.publish_json_new(short, payload)
            with self.assertRaisesRegex(
                qualification.GeometryQualificationError, "at least 100"
            ):
                qualification.load_single_gpu_baselines(
                    short,
                    hardware_identity_sha256="a" * 64,
                    sequence_length=16,
                )

    def test_command_and_cli_error_redaction_reject_secret_values(self) -> None:
        self.assertEqual(
            qualification.command_display(["python", "-m", "pretrain.train"]),
            "python -m pretrain.train",
        )
        with self.assertRaisesRegex(
            qualification.GeometryQualificationError, "credential-bearing"
        ):
            qualification._assert_secret_free_command(  # noqa: SLF001
                ["python", "https://user:password@example.invalid/path"], {}
            )
        with self.assertRaisesRegex(
            qualification.GeometryQualificationError, "inherited secret"
        ):
            qualification._assert_secret_free_command(  # noqa: SLF001
                ["python", "/tmp/value-abc123"], {"WANDB_API_KEY": "abc123"}
            )

    def test_manual_baseline_draft_is_validated_before_immutable_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self._baseline(root)
            draft = root / "draft.json"
            draft.write_bytes(source.read_bytes())
            output = root / "sealed.json"
            with mock.patch.object(
                qualification,
                "_validate_hardware_contract",
                return_value=(
                    {"status": "accepted"},
                    {"artifact": {"sha256": "a" * 64}},
                    {
                        "scope": "geometry-only-provisional",
                        "identity": {},
                        "identity_sha256": "a" * 64,
                    },
                ),
            ):
                bound = qualification.seal_single_gpu_baselines(
                    draft=draft,
                    output=output,
                    hardware_contract=root / "hardware.json",
                    sequence_length=16,
                )
            self.assertEqual(
                qualification.verified_json_with_sidecar(
                    output, label="sealed baseline"
                )[1],
                bound,
            )
            with mock.patch.object(
                qualification,
                "_validate_hardware_contract",
                return_value=(
                    {"status": "accepted"},
                    {"artifact": {"sha256": "a" * 64}},
                    {
                        "scope": "geometry-only-provisional",
                        "identity": {},
                        "identity_sha256": "a" * 64,
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    qualification.GeometryQualificationError, "overwrite"
                ):
                    qualification.seal_single_gpu_baselines(
                        draft=draft,
                        output=output,
                        hardware_contract=root / "hardware.json",
                        sequence_length=16,
                    )

    def test_plan_root_is_write_once_and_rejects_another_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "qualification"
            resolved_root = root.resolve(strict=False)
            plan = {"output_root": str(resolved_root)}
            plan["plan_sha256"] = qualification.canonical_sha256(plan)
            first = qualification.prepare_plan_root(root, plan)
            second = qualification.prepare_plan_root(root, plan)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(
                qualification.GeometryQualificationError, "another qualification"
            ):
                qualification.prepare_plan_root(
                    root,
                    {
                        "output_root": str(resolved_root),
                        "plan_sha256": qualification.canonical_sha256(
                            {"output_root": str(resolved_root), "different": True}
                        ),
                        "different": True,
                    },
                )

    def test_grid_runs_all_candidates_and_selects_fastest_passing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, baseline_bound = self._baseline(root)
            run_root = root / "grid"
            run_root.mkdir()
            plan = {
                "mode": "grid",
                "candidates": [
                    candidate.as_dict() for candidate in qualification.CANDIDATES
                ],
                "settings": vars(qualification.SoakSettings()),
                "inputs": {
                    "single_gpu_baselines": baseline_bound,
                    "hardware": {
                        "bound": {"artifact": {"sha256": "a" * 64}},
                        "expected": {"gpu_memory_bytes": 80 * GIB},
                        "geometry_identity": {
                            "identity_sha256": "a" * 64
                        },
                    },
                    "common": {"sequence_length": 16},
                },
            }
            plan["plan_sha256"] = qualification.canonical_sha256(plan)
            qualification.publish_json_new(run_root / "PLAN.json", plan)
            shared_order = {
                "manifest": {"path": "/manifest", "bytes": 1, "sha256": "d" * 64},
                "order": {"path": "/order", "bytes": 1, "sha256": "e" * 64},
                "geometry": {},
            }
            calls: list[str] = []

            def candidate_runner(**kwargs: object):
                candidate = kwargs["candidate"]
                candidate_root = kwargs["root"]
                assert isinstance(candidate, qualification.CandidateSpec)
                assert isinstance(candidate_root, Path)
                calls.append(candidate.candidate_id)
                rate = 1000 + len(calls)
                payload = {
                    "format": qualification.CANDIDATE_RESULT_FORMAT,
                    "format_version": qualification.FORMAT_VERSION,
                    "status": "pass",
                    "plan_sha256": plan["plan_sha256"],
                    "candidate": candidate.as_dict(),
                    "failures": [],
                    "measurements": {
                        "aggregate_input_tokens_per_second": str(rate)
                    },
                }
                candidate_root.mkdir(parents=True, exist_ok=True)
                bound = qualification.publish_json_new(
                    candidate_root / "RESULT.json", payload
                )
                return payload, bound

            with mock.patch.object(
                qualification, "ensure_grid_order", return_value=shared_order
            ):
                payload, _ = qualification.run_grid(
                    root=run_root,
                    plan=plan,
                    candidate_runner=candidate_runner,
                    runtime_verifier=lambda **_: {"status": "pass"},
                )
            self.assertEqual(len(calls), 6)
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(
                payload["accepted_candidate"], qualification.CANDIDATES[-1].as_dict()
            )
            with (
                mock.patch.object(
                    qualification,
                    "verify_grid_plan_authority",
                    return_value={"status": "pass"},
                ),
                mock.patch.object(
                    qualification, "_candidate_failures", return_value=[]
                ),
            ):
                loaded, _, _ = qualification._load_validated_grid_result(  # noqa: SLF001
                    run_root / "GRID-RESULT.json"
                )
            self.assertEqual(loaded, payload)

    def test_automatic_baselines_run_all_six_and_publish_sealed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "baselines"
            root.mkdir()
            sequence_length = 16
            plan = {
                "format": qualification.BASELINE_PLAN_FORMAT,
                "format_version": qualification.FORMAT_VERSION,
                "mode": "single-gpu-baselines",
                "inputs": {
                    "common": {"sequence_length": sequence_length},
                    "hardware": {
                        "geometry_identity": {
                            "identity_sha256": "b" * 64
                        }
                    },
                },
            }
            plan["plan_sha256"] = qualification.canonical_sha256(plan)
            qualification.publish_json_new(root / "PLAN.json", plan)
            calls: list[str] = []
            shared_order = {
                "manifest": {"sha256": "c" * 64},
                "order": {"sha256": "d" * 64},
                "geometry": {},
            }

            def runner(**kwargs: object):
                candidate = kwargs["candidate"]
                candidate_root = kwargs["root"]
                assert isinstance(candidate, qualification.CandidateSpec)
                assert isinstance(candidate_root, Path)
                calls.append(candidate.candidate_id)
                updates = 100
                token_delta = (
                    updates
                    * candidate.local_microbatch_rows
                    * candidate.gradient_accumulation_steps
                    * sequence_length
                )
                measurement = {
                    "global_microbatch_rows": candidate.local_microbatch_rows,
                    "gradient_accumulation_steps": (
                        candidate.gradient_accumulation_steps
                    ),
                    "compile_model": candidate.compile_model,
                    "gpu_count": 1,
                    "scope": geometry_evidence.THROUGHPUT_SCOPE,
                    "timer": geometry_evidence.THROUGHPUT_TIMER,
                    "counter": geometry_evidence.THROUGHPUT_COUNTER,
                    "start_consumed_input_tokens": 0,
                    "end_consumed_input_tokens": token_delta,
                    "elapsed_wall_time_ns": token_delta * 1_000_000,
                    "aggregate_input_tokens_per_second": "1000",
                }
                payload = {
                    "format": qualification.BASELINE_CANDIDATE_RESULT_FORMAT,
                    "format_version": qualification.FORMAT_VERSION,
                    "status": "pass",
                    "plan_sha256": plan["plan_sha256"],
                    "candidate": candidate.as_dict(),
                    "order": {"order": {"sha256": "d" * 64}},
                    "measurement": measurement,
                }
                candidate_root.mkdir(parents=True)
                return payload, qualification.publish_json_new(
                    candidate_root / "RESULT.json", payload
                )

            with (
                mock.patch.object(
                    qualification,
                    "verify_baseline_plan_authority",
                    return_value={"status": "pass"},
                ),
                mock.patch.object(
                    qualification,
                    "ensure_baseline_order",
                    return_value=shared_order,
                ),
            ):
                payload, bound = qualification.run_single_gpu_baselines(
                    root=root,
                    plan=plan,
                    candidate_runner=runner,
                    runtime_verifier=lambda **_: {"status": "pass"},
                )
            self.assertEqual(calls, [item.candidate_id for item in qualification.CANDIDATES])
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(
                payload["hardware_identity_sha256"], "b" * 64
            )
            self.assertEqual(
                bound,
                qualification.verified_json_with_sidecar(
                    root / "single-gpu-baselines.json",
                    label="automatic baselines",
                )[1],
            )

    def test_baseline_candidate_measures_real_one_rank_counter_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "candidate"
            candidate = qualification.CANDIDATES[0]
            sequence_length = 16
            tokens_per_update = qualification.BASELINE_UPDATE_ROWS * sequence_length
            validation = root / "validation.json"
            validation.write_text("{}", encoding="utf-8")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            order_manifest = root / "order.json"
            order_manifest.write_text("{}", encoding="utf-8")
            order = {
                "manifest": qualification.artifact(order_manifest, label="order"),
                "order": {"path": "/unused", "bytes": 1, "sha256": "1" * 64},
                "geometry": {},
            }
            trainer_source = Path(qualification.training_data.__file__).parent / "train.py"
            plan = {
                "plan_sha256": "2" * 64,
                "baseline_gpu": {
                    "visible_index": 0,
                    "physical_index": 0,
                    "uuid": "GPU-0",
                },
                "settings": vars(qualification.SoakSettings()),
                "runtime": {
                    "python": {"invocation_path": sys.executable},
                    "sources": {
                        "trainer": {"path": str(trainer_source), "sha256": "8" * 64},
                        "model": {"sha256": "9" * 64},
                        "data": {"sha256": "a" * 64},
                    },
                },
                "inputs": {
                    "validation_order": {"manifest": {"path": str(validation)}},
                    "common": {
                        "sequence_length": sequence_length,
                        "tokenizer": {"root": str(tokenizer)},
                    },
                    "hardware": {
                        "expected": {
                            "qualification": {
                                "host": {
                                    "environment": {
                                        "cuda_visible_devices": [
                                            "0",
                                            "1",
                                            "2",
                                            "3",
                                            "4",
                                            "5",
                                        ]
                                    }
                                },
                                "gpu": {
                                    "nvidia_smi_executable": {
                                        "path": "/usr/bin/nvidia-smi"
                                    },
                                    "devices": [
                                        {"nvidia_smi_memory_bytes": 80 * GIB}
                                        for _ in range(6)
                                    ],
                                },
                            }
                        }
                    },
                },
            }
            phase_calls = 0

            def phase_runner(**kwargs: object) -> qualification.PhaseRun:
                nonlocal phase_calls
                phase_calls += 1
                self.assertEqual(kwargs["expected_gpu_count"], 1)
                self.assertEqual(kwargs["environment"]["CUDA_VISIBLE_DEVICES"], "0")
                command = kwargs["command"]
                self.assertIn("--nproc-per-node=1", command)
                self.assertEqual("--resume" in command, phase_calls == 2)
                log = kwargs["log_path"]
                stop = kwargs["stop_request_path"]
                assert isinstance(log, Path) and isinstance(stop, Path)
                log.write_text(
                    f"exitcode : {128 + int(signal.SIGUSR1)} (pid: 123)\n",
                    encoding="utf-8",
                )
                stop.write_text(f"{int(signal.SIGUSR1)}\n", encoding="ascii")
                checkpoint = candidate_root / "checkpoints" / "last.pt"
                if phase_calls == 2:
                    (candidate_root / "checkpoints" / "last.previous.pt").write_bytes(
                        checkpoint.read_bytes()
                    )
                checkpoint.write_bytes(f"checkpoint-{phase_calls}".encode("ascii"))
                wandb = candidate_root / "checkpoints" / "wandb"
                wandb.mkdir(exist_ok=True)
                (wandb / "offline-run").write_bytes(b"wandb")
                last_step = 55 if phase_calls == 1 else 105
                completed_ns = (
                    26_600_000_000 if phase_calls == 1 else 52_200_000_000
                )
                stop_ns = 26_000_000_000 if phase_calls == 1 else 52_000_000_000
                return qualification.PhaseRun(
                    return_code=1,
                    completed_monotonic_ns=completed_ns,
                    observation=qualification.PhaseObservation(
                        start_monotonic_ns=1_000_000_000,
                        start_step=5,
                        start_consumed_input_tokens=5 * tokens_per_update,
                        last_step=last_step,
                        last_consumed_input_tokens=last_step * tokens_per_update,
                        validation_events_after_start=1,
                        wandb_log_events_after_start=50,
                        data_wait_samples=50,
                        maximum_data_wait_fraction=Decimal("0.01"),
                        peak_memory_allocated_bytes=60 * GIB,
                        peak_memory_reserved_bytes=64 * GIB,
                        stop_request_monotonic_ns=stop_ns,
                        stop_requested_after_step=last_step,
                        json_metric_records=50,
                        graceful_stop_events=1,
                    ),
                    gpu_memory=qualification.GPUMemoryObservation(
                        gpu_count=1,
                        total_bytes_per_gpu=(80 * GIB,),
                        minimum_free_bytes_per_gpu=12 * GIB,
                        samples=4,
                    ),
                    log=qualification.artifact(log, label="baseline log"),
                )

            def inspect_checkpoint(path: Path, **kwargs: object) -> dict[str, object]:
                self.assertEqual(kwargs["expected_world_size"], 1)
                completed_steps = 55 if path.name == "last.previous.pt" else 105
                expected_implementation = {
                    "trainer_class": "pretrain.train.Trainer",
                    "model_class": "pretrain.model.CausalLM",
                    "trainer_source_sha256": "8" * 64,
                    "model_source_sha256": "9" * 64,
                    "data_source_sha256": "a" * 64,
                }
                return {
                    **qualification.artifact(path, label="checkpoint"),
                    "completed_steps": completed_steps,
                    "consumed_input_tokens": completed_steps * tokens_per_update,
                    "world_size": 1,
                    "wandb_run_id_sha256": "b" * 64,
                    "data_identity_sha256": "c" * 64,
                    "training_geometry_sha256": qualification.canonical_sha256({}),
                    "trajectory_sha256": "d" * 64,
                    "runtime_signature_sha256": "e" * 64,
                    "implementation_signature_sha256": (
                        qualification.canonical_sha256(expected_implementation)
                    ),
                }

            with mock.patch.object(
                qualification,
                "_boot_id",
                return_value="12345678-1234-1234-1234-123456789abc",
            ):
                payload, _ = qualification.run_baseline_candidate(
                    root=candidate_root,
                    plan=plan,
                    candidate=candidate,
                    order=order,
                    phase_runner=phase_runner,
                    checkpoint_inspector=inspect_checkpoint,
                    binding_verifier=lambda **_: {"status": "pass"},
                    order_verifier=lambda _: {"order": order["order"]},
                )
            self.assertEqual(phase_calls, 2)
            self.assertEqual(payload["status"], "pass")
            self.assertTrue(payload["evidence"]["resume_verified"])
            self.assertEqual(payload["measurement"]["gpu_count"], 1)
            self.assertEqual(
                payload["measurement"]["aggregate_input_tokens_per_second"],
                "1000",
            )

    def test_baseline_preview_is_one_rank_and_uses_local_recipe(self) -> None:
        plan = {
            "output_root": "/output",
            "plan_sha256": "a" * 64,
            "settings": vars(qualification.SoakSettings()),
            "runtime": {"python": {"invocation_path": "/python"}},
            "inputs": {
                "validation_order": {"manifest": {"path": "/validation"}},
                "common": {"tokenizer": {"root": "/tokenizer"}},
            },
        }
        previews = cli._preview_baseline_commands(plan)  # noqa: SLF001
        self.assertEqual(len(previews), 6)
        for candidate, preview in zip(
            qualification.CANDIDATES, previews, strict=True
        ):
            command = preview["command"]
            self.assertIn("--nproc-per-node=1", command)
            self.assertEqual(
                command[command.index("--global-microbatch-rows") + 1],
                str(candidate.local_microbatch_rows),
            )

    def test_candidate_resumes_only_after_authenticated_phase_one_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "candidate"
            candidate = qualification.CANDIDATES[0]
            sequence_length = 16
            tokens_per_update = 192 * sequence_length
            trainer_source = Path(qualification.training_data.__file__).parent / "train.py"
            validation = root / "validation.json"
            validation.write_text("{}", encoding="utf-8")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            order_manifest = root / "order.json"
            order_manifest.write_text("{}", encoding="utf-8")
            order = {
                "manifest": qualification.artifact(order_manifest, label="order"),
                "order": {"path": "/unused", "bytes": 1, "sha256": "1" * 64},
                "geometry": {},
            }
            plan = {
                "plan_sha256": "2" * 64,
                "settings": vars(qualification.SoakSettings()),
                "runtime": {
                    "python": {"invocation_path": sys.executable},
                    "sources": {
                        "trainer": {"path": str(trainer_source), "sha256": "8" * 64},
                        "model": {"sha256": "9" * 64},
                        "data": {"sha256": "a" * 64},
                    },
                },
                "inputs": {
                    "validation_order": {"manifest": {"path": str(validation)}},
                    "common": {
                        "sequence_length": sequence_length,
                        "tokenizer": {"root": str(tokenizer)},
                    },
                    "hardware": {"expected": {"gpu_memory_bytes": 80 * GIB}},
                },
            }
            phase_calls: list[bool] = []
            fail_phase_two_once = True

            def phase_runner(**kwargs: object) -> qualification.PhaseRun:
                nonlocal fail_phase_two_once
                resume = "--resume" in kwargs["command"]
                phase_calls.append(resume)
                if resume and fail_phase_two_once:
                    fail_phase_two_once = False
                    raise RuntimeError("simulated interruption before phase two started")
                log_path = kwargs["log_path"]
                stop_path = kwargs["stop_request_path"]
                assert isinstance(log_path, Path) and isinstance(stop_path, Path)
                log_path.write_text(
                    f"exitcode : {128 + int(signal.SIGUSR1)} (pid: 123)\n",
                    encoding="utf-8",
                )
                stop_path.write_text(f"{int(signal.SIGUSR1)}\n", encoding="ascii")
                checkpoints = candidate_root / "checkpoints"
                checkpoint = checkpoints / "last.pt"
                previous = checkpoints / "last.previous.pt"
                if resume:
                    previous.write_bytes(checkpoint.read_bytes())
                    checkpoint.write_bytes(b"final")
                    wandb = checkpoints / "wandb"
                    wandb.mkdir()
                    (wandb / "offline-run").write_bytes(b"wandb")
                    observation = qualification.PhaseObservation(
                        start_monotonic_ns=1_000_000_000,
                        start_step=5,
                        start_consumed_input_tokens=5 * tokens_per_update,
                        last_step=105,
                        last_consumed_input_tokens=105 * tokens_per_update,
                        validation_events_after_start=2,
                        wandb_log_events_after_start=50,
                        data_wait_samples=50,
                        maximum_data_wait_fraction=Decimal("0.01"),
                        peak_memory_allocated_bytes=60 * GIB,
                        peak_memory_reserved_bytes=64 * GIB,
                        stop_request_monotonic_ns=64_000_000_000,
                        stop_requested_after_step=105,
                        json_metric_records=50,
                        graceful_stop_events=1,
                    )
                    completed = 65_000_000_000
                else:
                    checkpoint.write_bytes(b"mid")
                    observation = qualification.PhaseObservation(
                        start_monotonic_ns=1_000_000_000,
                        start_step=5,
                        start_consumed_input_tokens=5 * tokens_per_update,
                        last_step=55,
                        last_consumed_input_tokens=55 * tokens_per_update,
                        validation_events_after_start=2,
                        wandb_log_events_after_start=50,
                        data_wait_samples=50,
                        maximum_data_wait_fraction=Decimal("0.01"),
                        peak_memory_allocated_bytes=60 * GIB,
                        peak_memory_reserved_bytes=64 * GIB,
                        stop_request_monotonic_ns=1_500_000_000,
                        stop_requested_after_step=55,
                        json_metric_records=50,
                        graceful_stop_events=1,
                    )
                    completed = 2_000_000_000
                return qualification.PhaseRun(
                    return_code=1,
                    completed_monotonic_ns=completed,
                    observation=observation,
                    gpu_memory=qualification.GPUMemoryObservation(
                        gpu_count=6,
                        total_bytes_per_gpu=(80 * GIB,) * 6,
                        minimum_free_bytes_per_gpu=12 * GIB,
                        samples=4,
                    ),
                    log=qualification.artifact(log_path, label="phase log"),
                )

            def inspect_checkpoint(
                path: Path,
                *,
                expected_order_sha256: str,
                expected_order_payload_sha256: str,
            ) -> dict[str, object]:
                del expected_order_sha256, expected_order_payload_sha256
                final = path.name == "last.pt"
                step = 105 if final else 55
                return {
                    **qualification.artifact(path, label="checkpoint"),
                    "completed_steps": step,
                    "completed_microbatches": step * 32,
                    "consumed_rows": step * 192,
                    "consumed_input_tokens": step * tokens_per_update,
                    "last_validated_step": step - 5,
                    "world_size": 6,
                    "wandb_run_id_sha256": "3" * 64,
                    "data_identity_sha256": "4" * 64,
                    "training_geometry_sha256": qualification.canonical_sha256({}),
                    "trajectory_sha256": "6" * 64,
                    "runtime_signature_sha256": "7" * 64,
                    "implementation_signature_sha256": qualification.canonical_sha256(
                        {
                            "trainer_class": "pretrain.train.Trainer",
                            "model_class": "pretrain.model.CausalLM",
                            "trainer_source_sha256": "8" * 64,
                            "model_source_sha256": "9" * 64,
                            "data_source_sha256": "a" * 64,
                        }
                    ),
                }

            with mock.patch.object(qualification, "_boot_id", return_value="boot"):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    qualification.run_candidate(
                        root=candidate_root,
                        plan=plan,
                        candidate=candidate,
                        order=order,
                        baseline_rate=Decimal("1000"),
                        phase_runner=phase_runner,
                        checkpoint_inspector=inspect_checkpoint,
                        binding_verifier=lambda **_: {"status": "pass"},
                        order_verifier=lambda _: {"order": order["order"]},
                        gpu_sampler=lambda: [(80 * GIB, 12 * GIB)] * 6,
                        clock_ns=lambda: 70_000_000_000,
                    )
                payload, _ = qualification.run_candidate(
                    root=candidate_root,
                    plan=plan,
                    candidate=candidate,
                    order=order,
                    baseline_rate=Decimal("1000"),
                    phase_runner=phase_runner,
                    checkpoint_inspector=inspect_checkpoint,
                    binding_verifier=lambda **_: {"status": "pass"},
                    order_verifier=lambda _: {"order": order["order"]},
                    gpu_sampler=lambda: [(80 * GIB, 12 * GIB)] * 6,
                    clock_ns=lambda: 70_000_000_000,
                )
            self.assertEqual(phase_calls, [False, True, True])
            self.assertEqual(payload["status"], "pass")
            self.assertTrue(payload["measurements"]["throughput_measurement"]["resume_verified"])

    def test_final_soak_publishes_exact_accepted_receipt_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path, baseline_bound = self._baseline(root)
            del baseline_path
            run_root = root / "final"
            run_root.mkdir()
            candidate = qualification.CANDIDATES[0]
            update_tokens = 192 * 16
            token_delta = 100 * update_tokens
            elapsed = token_delta * 1_000_000
            dummy_order = root / "train-order.json"
            dummy_order.write_text("{}", encoding="utf-8")
            order_descriptor = qualification.artifact(dummy_order, label="order")
            plan = {
                "mode": "final-soak",
                "candidate": candidate.as_dict(),
                "settings": vars(qualification.SoakSettings()),
                "inputs": {
                    "single_gpu_baselines": baseline_bound,
                    "hardware": {
                        "bound": {"artifact": {"sha256": "a" * 64}},
                        "expected": {"gpu_memory_bytes": 80 * GIB},
                        "geometry_identity": {
                            "identity_sha256": "a" * 64
                        },
                    },
                    "common": {"sequence_length": 16},
                    "train_order": {
                        "manifest": order_descriptor,
                        "geometry": {
                            "consumed_input_tokens": 1_000_000_000,
                            "global_microbatch_rows": 6,
                            "gradient_accumulation_steps": 32,
                        },
                    },
                    "validation_order": {"manifest": {"sha256": "b" * 64}},
                    "grid_result": {"artifact": {"sha256": "c" * 64}},
                },
                "runtime": {"sources": {}},
            }
            plan["plan_sha256"] = qualification.canonical_sha256(plan)
            measurements = {
                "aggregate_input_tokens_per_second": "1000",
                "peak_memory_allocated_bytes_per_gpu": 60 * GIB,
                "peak_memory_reserved_bytes_per_gpu": 64 * GIB,
                "minimum_free_memory_bytes_per_gpu": 12 * GIB,
                "checkpoint_seconds": "10",
                "data_wait_fraction": "0.01",
                "scaling_efficiency": "0.8",
                "soak_steps": 100,
                "throughput_measurement": {
                    "scope": geometry_evidence.THROUGHPUT_SCOPE,
                    "timer": geometry_evidence.THROUGHPUT_TIMER,
                    "counter": geometry_evidence.THROUGHPUT_COUNTER,
                    "start_consumed_input_tokens": 0,
                    "end_consumed_input_tokens": token_delta,
                    "elapsed_wall_time_ns": elapsed,
                    "validation_events": 2,
                    "checkpoint_events": 2,
                    "wandb_log_events": 100,
                    "resume_verified": True,
                },
            }
            calls = 0

            def runner(**kwargs: object):
                nonlocal calls
                calls += 1
                candidate_root = kwargs["root"]
                assert isinstance(candidate_root, Path)
                candidate_root.mkdir(parents=True, exist_ok=True)
                payload = {
                    "status": "pass",
                    "plan_sha256": plan["plan_sha256"],
                    "candidate": candidate.as_dict(),
                    "measurements": measurements,
                }
                bound = qualification.publish_json_new(
                    candidate_root / "RESULT.json", payload
                )
                return payload, bound

            receipt_path = run_root / "accepted-geometry.json"
            receipt, bound = qualification.run_final_soak(
                root=run_root,
                plan=plan,
                receipt_path=receipt_path,
                candidate_runner=runner,
                runtime_verifier=lambda **_: {"status": "pass"},
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(calls, 1)
            same, same_bound = qualification.run_final_soak(
                root=run_root,
                plan=plan,
                receipt_path=receipt_path,
                candidate_runner=lambda **_: self.fail("idempotent rerun executed training"),
                runtime_verifier=lambda **_: {"status": "pass"},
            )
            self.assertEqual(same, receipt)
            self.assertEqual(same_bound, bound)

    def test_cli_grid_preview_has_six_two_phase_commands(
        self,
    ) -> None:
        plan = {
            "output_root": "/output",
            "plan_sha256": "a" * 64,
            "settings": vars(qualification.SoakSettings()),
            "runtime": {"python": {"invocation_path": "/python"}},
            "inputs": {
                "validation_order": {"manifest": {"path": "/validation"}},
                "common": {"tokenizer": {"root": "/tokenizer"}},
            },
        }
        commands = cli._preview_grid_commands(plan)  # noqa: SLF001
        self.assertEqual(len(commands), 6)
        self.assertTrue(all("--resume" in row["phase_two"] for row in commands))

    def test_cli_grid_preflight_is_cpu_only_and_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "must-not-exist"
            plan = {"mode": "grid", "output_root": str(root), "plan_sha256": "a" * 64}
            arguments = [
                "preflight-grid",
                "--output-root",
                str(root),
                "--python-packed-manifest",
                "/python.json",
                "--other-code-packed-manifest",
                "/other.json",
                "--english-packed-manifest",
                "/english.json",
                "--validation-order-manifest",
                "/validation.json",
                "--tokenizer",
                "/tokenizer",
                "--hardware-contract",
                "/hardware.json",
                "--single-gpu-baselines",
                "/baseline.json",
            ]
            output = io.StringIO()
            with (
                mock.patch.object(cli, "_grid_plan", return_value=plan),
                mock.patch.object(cli, "_preview_grid_commands", return_value=[]),
                mock.patch.object(
                    qualification,
                    "verify_output_storage",
                    return_value={"status": "pass"},
                ),
                mock.patch.object(qualification, "run_grid") as run_grid,
                redirect_stdout(output),
            ):
                self.assertEqual(cli.main(arguments), 0)
            run_grid.assert_not_called()
            self.assertFalse(root.exists())
            self.assertEqual(
                json.loads(output.getvalue())["status"],
                "preflight-pass-no-gpu-action",
            )

    def test_cli_grid_threshold_failure_has_nonzero_exit(self) -> None:
        plan = {"mode": "grid", "output_root": "/output", "plan_sha256": "a" * 64}
        arguments = [
            "run-grid",
            "--output-root",
            "/output",
            "--python-packed-manifest",
            "/python.json",
            "--other-code-packed-manifest",
            "/other.json",
            "--english-packed-manifest",
            "/english.json",
            "--validation-order-manifest",
            "/validation.json",
            "--tokenizer",
            "/tokenizer",
            "--hardware-contract",
            "/hardware.json",
            "--single-gpu-baselines",
            "/baseline.json",
        ]
        output = io.StringIO()
        with (
            mock.patch.object(cli, "_grid_plan", return_value=plan),
            mock.patch.object(qualification, "prepare_plan_root"),
            mock.patch.object(qualification, "QualificationLock"),
            mock.patch.object(
                qualification,
                "run_grid",
                return_value=(
                    {"status": "fail"},
                    {"artifact": {"sha256": "b" * 64}},
                ),
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(cli.main(arguments), 3)
        self.assertEqual(json.loads(output.getvalue())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
