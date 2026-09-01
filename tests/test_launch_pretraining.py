from __future__ import annotations

import contextlib
import hashlib
import io
import signal
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import launch_pretraining as launch
import certify_pretraining_data as certifier
from pretrain.data import DOMAIN_ORDER, PackedShardWriter, build_training_order


TOKENIZER_SHA256 = hashlib.sha256(b"launch-preflight-tokenizer").hexdigest()
WEIGHTS = {"python": 0.4, "other_code": 0.4, "english": 0.2}


def _build_domain(root: Path, domain: str, *, split: str, rows: int) -> Path:
    output = root / domain
    writer = PackedShardWriter(
        output,
        domain=domain,
        split=split,
        sequence_length=4,
        vocab_size=256,
        eos_token_id=0,
        tokenizer_manifest_sha256=TOKENIZER_SHA256,
        rows_per_shard=2,
        construction_seed=11,
    )
    writer.add_document([(index % 200) + 1 for index in range(rows * 4)])
    writer.finish()
    return output / "manifest.json"


def _build_order(
    local_root: Path, *, split: str, name: str, frozen: bool | None = None
) -> Path:
    packed = local_root / "packed" / split
    counts = {"python": 4, "other_code": 4, "english": 2}
    manifests = {
        domain: _build_domain(packed, domain, split=split, rows=counts[domain])
        for domain in DOMAIN_ORDER
    }
    output = local_root / "orders" / name
    if frozen is None:
        frozen = split == "train"
    geometry = (
        {
            "frozen_global_microbatch_rows": 2,
            "frozen_gradient_accumulation_steps": 1,
        }
        if frozen
        else {}
    )
    build_training_order(
        manifests,
        output,
        seed=17 if split == "train" else 19,
        expected_weights=WEIGHTS,
        expected_total_input_tokens=40,
        **geometry,
    )
    return output / "manifest.json"


class OrderPreflightTest(unittest.TestCase):
    def test_metadata_only_inspection_accepts_complete_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            train_path = _build_order(local, split="train", name="train")
            validation_path = _build_order(
                local, split="validation", name="validation"
            )
            original_sha256 = launch._sha256

            def metadata_sha256(path: Path) -> str:
                self.assertEqual(path.name, "manifest.json")
                return original_sha256(path)

            with mock.patch.object(launch, "_sha256", side_effect=metadata_sha256):
                train = launch.inspect_order(
                    train_path, expected_split="train", local_data_root=local
                )
                validation = launch.inspect_order(
                    validation_path,
                    expected_split="validation",
                    local_data_root=local,
                    global_microbatch_rows=train.geometry[
                        "global_microbatch_rows"
                    ],
                )
            launch.validate_order_pair(
                train, validation, world_size=2, eval_batches=2
            )
            self.assertEqual(train.payload_bytes_read, 0)
            self.assertGreater(train.packed_payload_bytes, 0)
            self.assertEqual(train.packed_payload_files, 10)

    def test_wrong_split_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            train_path = _build_order(local, split="train", name="train")
            with self.assertRaisesRegex(launch.PreflightError, "Evaluation order"):
                launch.inspect_order(
                    train_path,
                    expected_split="validation",
                    local_data_root=local,
                    global_microbatch_rows=2,
                )

    def test_frozen_held_out_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            validation_path = _build_order(
                local,
                split="validation",
                name="validation-frozen",
                frozen=True,
            )
            with self.assertRaisesRegex(
                launch.PreflightError, "unexpectedly freezes optimizer geometry"
            ):
                launch.inspect_order(
                    validation_path,
                    expected_split="validation",
                    local_data_root=local,
                    global_microbatch_rows=2,
                )

    def test_truncated_packed_payload_is_caught_by_stat_without_reading_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            train_path = _build_order(local, split="train", name="train")
            token_shard = next((local / "packed" / "train" / "python").glob("*.tokens.bin"))
            with token_shard.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(launch.PreflightError, "Token shard size mismatch"):
                launch.inspect_order(
                    train_path, expected_split="train", local_data_root=local
                )

    def test_pair_rejects_incompatible_geometry_and_world_size(self) -> None:
        geometry = {
            "global_microbatch_rows": 4,
            "gradient_accumulation_steps": 2,
            "sequence_length": 4096,
            "consumed_global_microbatches": 8,
        }
        evaluation_geometry = {
            "global_microbatch_rows": 4,
            "sequence_length": 4096,
            "available_global_microbatches": 8,
        }
        common = dict(
            format_version=4,
            sequence_length=4096,
            vocab_size=49152,
            eos_token_id=0,
            tokenizer_manifest_sha256="a" * 64,
            geometry=geometry,
            order_payload_bytes=8,
            packed_manifest_paths={},
            packed_payload_files=0,
            packed_payload_bytes=0,
        )
        train = launch.OrderInspection(path="/local/train.json", split="train", **common)
        validation = launch.OrderInspection(
            path="/local/validation.json",
            split="validation",
            **{**common, "geometry": evaluation_geometry},
        )
        with self.assertRaisesRegex(launch.PreflightError, "not divisible"):
            launch.validate_order_pair(
                train, validation, world_size=3, eval_batches=1
            )
        changed = replace(
            validation,
            geometry={**evaluation_geometry, "global_microbatch_rows": 2},
        )
        with self.assertRaisesRegex(launch.PreflightError, "Validation evaluation geometry"):
            launch.validate_order_pair(train, changed, world_size=2, eval_batches=1)


class StorageAndResumeTest(unittest.TestCase):
    @staticmethod
    def _mount(
        path: Path,
        *,
        device: int,
        classification: str,
        mount_read_only: bool | None = None,
    ) -> launch.MountEvidence:
        if mount_read_only is None:
            mount_read_only = classification == "local-or-block"
        return launch.MountEvidence(
            path=str(path.resolve()),
            device=device,
            mount_point=str(path.resolve()),
            filesystem_type="ext4" if classification != "network" else "nfs4",
            mount_source="fixture",
            read_only=mount_read_only,
            mount_read_only=mount_read_only,
            classification=classification,
        )

    def test_local_payload_root_requires_read_only_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            durable = root / "durable"
            checkpoints = durable / "checkpoints"
            local.mkdir()
            checkpoints.mkdir(parents=True)
            actual_device = durable.stat().st_dev

            def mounts(path: Path) -> launch.MountEvidence:
                if path.resolve() == local.resolve():
                    return self._mount(
                        path,
                        device=actual_device + 1,
                        classification="local-or-block",
                        mount_read_only=False,
                    )
                return self._mount(path, device=actual_device, classification="network")

            with self.assertRaisesRegex(launch.PreflightError, "read-only mount"):
                launch.inspect_storage_and_checkpoint(
                    local_data_root=local,
                    durable_checkpoint_root=durable,
                    checkpoint=checkpoints / "last.pt",
                    resume_generation="none",
                    checkpoint_generation_bytes=100,
                    disk_usage=lambda _path: SimpleNamespace(free=1_000),
                    mount_inspector=mounts,
                )

    def test_mountinfo_read_only_flag_is_explicit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mountinfo = (
                f"36 25 0:30 / {root} ro,relatime - ext4 /dev/nvme0n1p1 ro\n"
            )
            with mock.patch.object(launch.sys, "platform", "linux"):
                evidence = launch.inspect_mount(root, mountinfo_text=mountinfo)
            self.assertTrue(evidence.read_only)
            self.assertTrue(evidence.mount_read_only)
            self.assertEqual(evidence.classification, "local-or-block")

    def test_new_lineage_requires_two_generation_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            durable = root / "durable"
            checkpoints = durable / "checkpoints"
            local.mkdir()
            checkpoints.mkdir(parents=True)
            actual_device = durable.stat().st_dev

            def mounts(path: Path) -> launch.MountEvidence:
                if path.resolve() == local.resolve():
                    return self._mount(
                        path,
                        device=actual_device + 1,
                        classification="local-or-block",
                    )
                return self._mount(path, device=actual_device, classification="network")

            usage = lambda _path: SimpleNamespace(free=199)
            with self.assertRaisesRegex(launch.PreflightError, "two generations"):
                launch.inspect_storage_and_checkpoint(
                    local_data_root=local,
                    durable_checkpoint_root=durable,
                    checkpoint=checkpoints / "last.pt",
                    resume_generation="none",
                    checkpoint_generation_bytes=100,
                    disk_usage=usage,
                    mount_inspector=mounts,
                )

    def test_previous_generation_is_selected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            durable = root / "durable"
            checkpoints = durable / "checkpoints"
            local.mkdir()
            checkpoints.mkdir(parents=True)
            checkpoint = checkpoints / "last.pt"
            checkpoint.write_bytes(b"latest")
            previous = checkpoints / "last.previous.pt"
            previous.write_bytes(b"previous")
            actual_device = durable.stat().st_dev

            def mounts(path: Path) -> launch.MountEvidence:
                if path.resolve() == local.resolve():
                    return self._mount(
                        path,
                        device=actual_device + 1,
                        classification="local-or-block",
                    )
                return self._mount(path, device=actual_device, classification="network")

            _, _, result = launch.inspect_storage_and_checkpoint(
                local_data_root=local,
                durable_checkpoint_root=durable,
                checkpoint=checkpoint,
                resume_generation="previous",
                checkpoint_generation_bytes=100,
                disk_usage=lambda _path: SimpleNamespace(free=1_000),
                mount_inspector=mounts,
            )
            self.assertEqual(result.resume_path, str(previous.resolve()))
            self.assertEqual(result.checkpoint, str(checkpoint.resolve()))
            self.assertTrue(all(result.filesystem_capabilities.values()))
            self.assertFalse(result.local_resume_staging)


class FullValidationEvidenceTest(unittest.TestCase):
    def test_execute_refuses_missing_evidence_but_dry_run_reports_it(self) -> None:
        missing = {"status": "missing", "required_for_execute": True}
        self.assertFalse(
            launch.require_full_validation_for_mode(
                execute=False,
                train_evidence=missing,
                validation_evidence=missing,
            )
        )
        with self.assertRaisesRegex(launch.PreflightError, "--execute requires"):
            launch.require_full_validation_for_mode(
                execute=True,
                train_evidence=missing,
                validation_evidence={"status": "pass"},
            )

    def test_receipt_is_bound_to_exact_local_payload_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            durable = root / "durable"
            local.mkdir()
            durable.mkdir()
            train_path = _build_order(local, split="train", name="train")
            evidence = certifier.certify_order(
                order_manifest=train_path,
                expected_split="train",
                local_data_root=local,
                global_microbatch_rows=None,
            )
            receipt, _ = certifier.publish_evidence(
                durable / "train-validation.json", evidence
            )
            inspection = launch.inspect_order(
                train_path, expected_split="train", local_data_root=local
            )
            verified = launch.verify_full_validation_evidence(
                receipt, expected_order=inspection, durable_root=durable
            )
            self.assertEqual(verified["status"], "pass")

            token_shard = next((local / "packed" / "train" / "python").glob("*.tokens.bin"))
            with token_shard.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 1]))
            changed = launch.inspect_order(
                train_path, expected_split="train", local_data_root=local
            )
            with self.assertRaisesRegex(launch.PreflightError, "changed after full validation"):
                launch.verify_full_validation_evidence(
                    receipt, expected_order=changed, durable_root=durable
                )

    def test_receipt_sidecar_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            durable = root / "durable"
            local.mkdir()
            durable.mkdir()
            train_path = _build_order(local, split="train", name="train")
            evidence = certifier.certify_order(
                order_manifest=train_path,
                expected_split="train",
                local_data_root=local,
                global_microbatch_rows=None,
            )
            receipt, sidecar = certifier.publish_evidence(
                durable / "train-validation.json", evidence
            )
            sidecar.write_text("0" * 64 + "  train-validation.json\n", encoding="ascii")
            inspection = launch.inspect_order(
                train_path, expected_split="train", local_data_root=local
            )
            with self.assertRaisesRegex(launch.PreflightError, "checksum mismatches"):
                launch.verify_full_validation_evidence(
                    receipt, expected_order=inspection, durable_root=durable
                )


class RuntimeAndCommandTest(unittest.TestCase):
    def test_execute_requires_authority_before_runtime_or_data_access(self) -> None:
        arguments = [
            "--train-order-manifest",
            "/missing/train.json",
            "--validation-order-manifest",
            "/missing/validation.json",
            "--tokenizer",
            "/missing/tokenizer",
            "--local-data-root",
            "/missing/local",
            "--durable-checkpoint-root",
            "/missing/durable",
            "--checkpoint",
            "/missing/durable/last.pt",
            "--checkpoint-generation-bytes",
            "1",
            "--nproc-per-node",
            "6",
            "--model-size",
            "1.3b",
            "--activation-checkpointing",
            "--checkpoint-every",
            "1",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
            "--execute",
        ]
        standard_error = io.StringIO()
        with (
            contextlib.redirect_stderr(standard_error),
            self.assertRaises(SystemExit) as raised,
        ):
            launch.main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--execute requires --run-authority", standard_error.getvalue())

    def test_production_selection_requires_explicit_six_rank_checkpointing(self) -> None:
        self.assertTrue(
            launch.validate_production_launch_selection(
                model_size="1.3b",
                nproc_per_node=6,
                activation_checkpointing=True,
            )
        )
        for ranks, checkpointing, pattern in (
            (5, True, "six DDP ranks"),
            (6, None, "explicit --activation-checkpointing"),
            (6, False, "explicit --activation-checkpointing"),
        ):
            with self.subTest(ranks=ranks, checkpointing=checkpointing):
                with self.assertRaisesRegex(launch.PreflightError, pattern):
                    launch.validate_production_launch_selection(
                        model_size="1.3b",
                        nproc_per_node=ranks,
                        activation_checkpointing=checkpointing,
                    )
        self.assertFalse(
            launch.validate_production_launch_selection(
                model_size="tiny",
                nproc_per_node=2,
                activation_checkpointing=None,
            )
        )

    def test_execute_authority_is_bound_to_exact_current_argv(self) -> None:
        invocation = [
            "/opt/venv/bin/python",
            "/workspace/repo/scripts/launch_pretraining.py",
            "--run-authority",
            "/durable/authority.json",
            "--execute",
        ]
        digest = launch.canonical_sha256(invocation)
        validated = {
            "status": "valid",
            "launcher_argv_sha256": digest,
            "authorization_sha256": "a" * 64,
        }
        with mock.patch.object(
            launch, "validate_run_authority", return_value=validated
        ) as validator:
            result = launch.validate_launch_authority(
                Path("/durable/authority.json"),
                invocation_argv=invocation,
            )
        validator.assert_called_once_with(Path("/durable/authority.json"))
        self.assertEqual(result["current_launcher_argv_sha256"], digest)

        changed = [*invocation[:-1], "--dry-run"]
        with (
            mock.patch.object(
                launch, "validate_run_authority", return_value=validated
            ),
            self.assertRaisesRegex(launch.PreflightError, "differs"),
        ):
            launch.validate_launch_authority(
                Path("/durable/authority.json"),
                invocation_argv=changed,
            )

        with (
            mock.patch.object(
                launch,
                "validate_run_authority",
                side_effect=launch.RunAuthorityError("mutated recipe"),
            ),
            self.assertRaisesRegex(launch.PreflightError, "mutated recipe"),
        ):
            launch.validate_launch_authority(
                Path("/durable/authority.json"),
                invocation_argv=invocation,
            )

    def test_planned_six_gpu_single_node_launch_contract(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 6

            @staticmethod
            def device(index: int):
                return contextlib.nullcontext(index)

            @staticmethod
            def is_bf16_supported(*, including_emulation: bool = True) -> bool:
                if including_emulation:
                    raise AssertionError("preflight must require native BF16")
                return True

            @staticmethod
            def get_device_name(index: int) -> str:
                return "Homogeneous-GPU"

            @staticmethod
            def get_device_capability(index: int) -> tuple[int, int]:
                del index
                return (9, 0)

            @staticmethod
            def get_device_properties(index: int):
                del index
                return SimpleNamespace(
                    total_memory=80 * 1024**3,
                    multi_processor_count=132,
                )

            @staticmethod
            def mem_get_info() -> tuple[int, int]:
                return (79 * 1024**3, 80 * 1024**3)

        fake_torch = SimpleNamespace(
            __version__="9.9",
            version=SimpleNamespace(cuda="99.0"),
            cuda=FakeCuda(),
            distributed=SimpleNamespace(
                is_available=lambda: True, is_nccl_available=lambda: True
            ),
        )
        runtime = launch.inspect_runtime(
            nproc_per_node=6,
            torch_module=fake_torch,
            environment={},
        )
        self.assertEqual(runtime.world_size, 6)
        self.assertEqual(runtime.cuda_devices, 6)
        self.assertEqual(runtime.bf16_supported_devices, list(range(6)))
        self.assertEqual(
            [profile["total_memory_bytes"] for profile in runtime.cuda_device_profiles],
            [80 * 1024**3] * 6,
        )
        self.assertEqual(
            [
                profile["available_memory_bytes"]
                for profile in runtime.cuda_device_profiles
            ],
            [79 * 1024**3] * 6,
        )
        memory = launch.inspect_model_memory(
            runtime,
            model_size="1.3b",
            vocab_size=49152,
            max_seq_len=4096,
        )
        self.assertEqual(memory.parameter_count, 1_283_557_376)
        self.assertEqual(
            memory.persistent_training_state_bytes,
            memory.parameter_count * 16,
        )
        self.assertEqual(memory.minimum_device_memory_bytes, 32 * 1024**3)
        self.assertEqual(
            memory.smallest_available_device_memory_bytes,
            79 * 1024**3,
        )
        self.assertTrue(memory.measured_full_topology_smoke_required)

        common = dict(
            format_version=4,
            sequence_length=4096,
            vocab_size=49152,
            eos_token_id=0,
            tokenizer_manifest_sha256="a" * 64,
            order_payload_bytes=8,
            packed_manifest_paths={},
            packed_payload_files=0,
            packed_payload_bytes=0,
        )
        train = launch.OrderInspection(
            path="/local/train.json",
            split="train",
            geometry={"global_microbatch_rows": 12},
            **common,
        )
        validation = launch.OrderInspection(
            path="/local/validation.json",
            split="validation",
            geometry={
                "global_microbatch_rows": 12,
                "available_global_microbatches": 8,
            },
            **common,
        )
        launch.validate_order_pair(
            train,
            validation,
            world_size=runtime.world_size,
            eval_batches=8,
        )

        checkpoint = launch.CheckpointInspection(
            checkpoint="/durable/checkpoints/last.pt",
            previous="/durable/checkpoints/last.previous.pt",
            resume_generation="none",
            resume_path=None,
            latest_exists=False,
            latest_bytes=None,
            previous_exists=False,
            previous_bytes=None,
            free_bytes=1_000,
            required_free_bytes=200,
            filesystem_capabilities={"atomic_replace": True},
            resume_read_storage="durable-checkpoint-filesystem",
            local_resume_staging=False,
            guidance="fixture",
        )
        command = launch.render_torchrun_command(
            runtime=runtime,
            train_order=train,
            validation_order=validation,
            checkpoint=checkpoint,
            tokenizer_path="/local/tokenizer/starcoder2",
            model_size="1.3b",
            workers=4,
            checkpoint_every=100,
            eval_every=50,
            eval_batches=8,
            eval_at_start=True,
            wandb_mode="disabled",
            wandb_project="project",
            wandb_entity=None,
            wandb_run_name="six-gpu-smoke",
            wandb_group=None,
            wandb_tags=(),
            extra_trainer_args=(),
            activation_checkpointing=True,
        )
        self.assertIn("--standalone", command)
        self.assertIn("--nnodes=1", command)
        self.assertIn("--nproc-per-node=6", command)
        self.assertIn("--max-restarts=0", command)
        self.assertIn("--activation-checkpointing", command)
        self.assertEqual(
            command[command.index("--tokenizer") + 1],
            "/local/tokenizer/starcoder2",
        )
        self.assertEqual(command[command.index("--device") + 1], "cuda")

        incompatible_train = replace(
            train,
            geometry={"global_microbatch_rows": 8},
        )
        incompatible_validation = replace(
            validation,
            geometry={
                "global_microbatch_rows": 8,
                "available_global_microbatches": 8,
            },
        )
        with self.assertRaisesRegex(launch.PreflightError, "not divisible"):
            launch.validate_order_pair(
                incompatible_train,
                incompatible_validation,
                world_size=runtime.world_size,
                eval_batches=8,
            )

    def test_cuda_bf16_world_size_probe(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

            @staticmethod
            def device(index: int):
                return contextlib.nullcontext(index)

            @staticmethod
            def is_bf16_supported(*, including_emulation: bool = True) -> bool:
                if including_emulation:
                    raise AssertionError("preflight must require native BF16")
                return True

            @staticmethod
            def get_device_name(index: int) -> str:
                return "Homogeneous-GPU"

            @staticmethod
            def get_device_capability(index: int) -> tuple[int, int]:
                del index
                return (9, 0)

            @staticmethod
            def get_device_properties(index: int):
                del index
                return SimpleNamespace(
                    total_memory=80 * 1024**3,
                    multi_processor_count=132,
                )

            @staticmethod
            def mem_get_info() -> tuple[int, int]:
                return (79 * 1024**3, 80 * 1024**3)

        fake_torch = SimpleNamespace(
            __version__="9.9",
            version=SimpleNamespace(cuda="99.0"),
            cuda=FakeCuda(),
            distributed=SimpleNamespace(
                is_available=lambda: True, is_nccl_available=lambda: True
            ),
        )
        result = launch.inspect_runtime(
            nproc_per_node=2,
            torch_module=fake_torch,
            environment={},
        )
        self.assertEqual(result.world_size, 2)
        self.assertEqual(result.bf16_supported_devices, [0, 1])
        self.assertTrue(result.deterministic_algorithms)
        self.assertEqual(result.cublas_workspace_config, ":4096:8")
        self.assertEqual(result.python_hash_seed, "0")
        with self.assertRaisesRegex(launch.PreflightError, "exact match"):
            launch.inspect_runtime(
                nproc_per_node=1,
                torch_module=fake_torch,
                environment={},
            )
        with self.assertRaisesRegex(launch.PreflightError, "CUBLAS_WORKSPACE_CONFIG"):
            launch.inspect_runtime(
                nproc_per_node=2,
                torch_module=fake_torch,
                environment={"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            )
        with self.assertRaisesRegex(launch.PreflightError, "PYTHONHASHSEED"):
            launch.inspect_runtime(
                nproc_per_node=2,
                torch_module=fake_torch,
                environment={"PYTHONHASHSEED": "random"},
            )

    def test_runtime_rejects_heterogeneous_or_24gb_1p3b_ddp(self) -> None:
        class HeterogeneousCuda:
            is_available = staticmethod(lambda: True)
            device_count = staticmethod(lambda: 2)
            device = staticmethod(contextlib.nullcontext)
            is_bf16_supported = staticmethod(
                lambda *, including_emulation=True: not including_emulation
            )
            get_device_name = staticmethod(lambda index: f"GPU-{index}")
            get_device_capability = staticmethod(lambda index: (9, 0))
            get_device_properties = staticmethod(
                lambda index: SimpleNamespace(
                    total_memory=80 * 1024**3,
                    multi_processor_count=132,
                )
            )
            mem_get_info = staticmethod(
                lambda: (79 * 1024**3, 80 * 1024**3)
            )

        fake_torch = SimpleNamespace(
            __version__="9.9",
            version=SimpleNamespace(cuda="99.0"),
            cuda=HeterogeneousCuda(),
            distributed=SimpleNamespace(
                is_available=lambda: True, is_nccl_available=lambda: True
            ),
        )
        with self.assertRaisesRegex(launch.PreflightError, "heterogeneous"):
            launch.inspect_runtime(
                nproc_per_node=2,
                torch_module=fake_torch,
                environment={},
            )

        low_memory_runtime = launch.RuntimeInspection(
            python_executable="/opt/venv/bin/python",
            python_executable_sha256="b" * 64,
            python_version="3.12.0",
            python_implementation="cpython",
            torch_version="9.9",
            cuda_runtime="99.0",
            cuda_devices=6,
            world_size=6,
            bf16_supported_devices=list(range(6)),
            cuda_device_profiles=[
                {
                    "index": index,
                    "name": "24GB-GPU",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 24 * 1024**3,
                    "available_memory_bytes": 24 * 1024**3,
                    "allocator_total_memory_bytes": 24 * 1024**3,
                    "multiprocessor_count": 128,
                }
                for index in range(6)
            ],
            launcher_module="torch.distributed.run",
            deterministic_algorithms=True,
            cublas_workspace_config=":4096:8",
            python_hash_seed="0",
        )
        with self.assertRaisesRegex(
            launch.PreflightError, "production admission floor"
        ):
            launch.inspect_model_memory(
                low_memory_runtime,
                model_size="1.3b",
                vocab_size=49152,
                max_seq_len=4096,
            )

    def test_runtime_and_command_ignore_poisoned_path_torchrun(self) -> None:
        class FakeCuda:
            is_available = staticmethod(lambda: True)
            device_count = staticmethod(lambda: 1)
            device = staticmethod(contextlib.nullcontext)
            is_bf16_supported = staticmethod(
                lambda *, including_emulation=True: not including_emulation
            )
            get_device_name = staticmethod(lambda index: "Homogeneous-GPU")
            get_device_capability = staticmethod(lambda index: (9, 0))
            get_device_properties = staticmethod(
                lambda index: SimpleNamespace(
                    total_memory=80 * 1024**3,
                    multi_processor_count=132,
                )
            )
            mem_get_info = staticmethod(
                lambda: (79 * 1024**3, 80 * 1024**3)
            )

        fake_torch = SimpleNamespace(
            __version__="9.9",
            version=SimpleNamespace(cuda="99.0"),
            cuda=FakeCuda(),
            distributed=SimpleNamespace(
                is_available=lambda: True, is_nccl_available=lambda: True
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            poisoned = Path(temporary) / "torchrun"
            poisoned.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            poisoned.chmod(0o755)
            runtime = launch.inspect_runtime(
                nproc_per_node=1,
                torch_module=fake_torch,
                environment={"PATH": temporary},
            )
        self.assertEqual(runtime.python_executable, str(Path(sys.executable).absolute()))
        self.assertEqual(runtime.launcher_module, "torch.distributed.run")
        self.assertNotEqual(runtime.python_executable, str(poisoned))

    def test_interpreter_identity_preserves_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "base-python"
            target.write_bytes(b"authenticated interpreter\n")
            target.chmod(0o755)
            environment_python = root / "venv" / "bin" / "python"
            environment_python.parent.mkdir(parents=True)
            environment_python.symlink_to(target)
            with mock.patch.object(launch.sys, "executable", str(environment_python)):
                identity = launch._interpreter_identity()
        self.assertEqual(identity["python_executable"], str(environment_python))
        self.assertNotEqual(identity["python_executable"], str(target))
        self.assertEqual(
            identity["python_executable_sha256"],
            hashlib.sha256(b"authenticated interpreter\n").hexdigest(),
        )

    def test_rendered_torchrun_command_owns_production_contract(self) -> None:
        runtime = launch.RuntimeInspection(
            python_executable="/opt/venv/bin/python",
            python_executable_sha256="b" * 64,
            python_version="3.12.0",
            python_implementation="cpython",
            torch_version="9.9",
            cuda_runtime="99.0",
            cuda_devices=2,
            world_size=2,
            bf16_supported_devices=[0, 1],
            cuda_device_profiles=[
                {
                    "index": index,
                    "name": "Homogeneous-GPU",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 80 * 1024**3,
                    "available_memory_bytes": 79 * 1024**3,
                    "allocator_total_memory_bytes": 80 * 1024**3,
                    "multiprocessor_count": 132,
                }
                for index in range(2)
            ],
            launcher_module="torch.distributed.run",
            deterministic_algorithms=True,
            cublas_workspace_config=":4096:8",
            python_hash_seed="0",
        )
        train = launch.OrderInspection(
            path="/local/train/manifest.json",
            split="train",
            format_version=4,
            sequence_length=4096,
            vocab_size=49152,
            eos_token_id=0,
            tokenizer_manifest_sha256="a" * 64,
            geometry={"global_microbatch_rows": 2},
            order_payload_bytes=8,
            packed_manifest_paths={},
            packed_payload_files=0,
            packed_payload_bytes=0,
        )
        validation = replace(
            train,
            path="/local/validation/manifest.json",
            split="validation",
        )
        checkpoint = launch.CheckpointInspection(
            checkpoint="/durable/checkpoints/last.pt",
            previous="/durable/checkpoints/last.previous.pt",
            resume_generation="latest",
            resume_path="/durable/checkpoints/last.pt",
            latest_exists=True,
            latest_bytes=100,
            previous_exists=True,
            previous_bytes=100,
            free_bytes=1_000,
            required_free_bytes=200,
            filesystem_capabilities={"atomic_replace": True},
            resume_read_storage="durable-checkpoint-filesystem",
            local_resume_staging=False,
            guidance="fixture",
        )
        command = launch.render_torchrun_command(
            runtime=runtime,
            train_order=train,
            validation_order=validation,
            checkpoint=checkpoint,
            tokenizer_path="/local/tokenizer/starcoder2",
            model_size="1.3b",
            workers=4,
            checkpoint_every=100,
            eval_every=50,
            eval_batches=8,
            eval_at_start=True,
            wandb_mode="offline",
            wandb_project="project",
            wandb_entity=None,
            wandb_run_name="run",
            wandb_group=None,
            wandb_tags=("production",),
            extra_trainer_args=("--compile",),
            activation_checkpointing=True,
        )
        self.assertEqual(
            command[:3],
            [runtime.python_executable, "-m", runtime.launcher_module],
        )
        self.assertIn("--deterministic", command)
        self.assertIn("--validation-order-manifest", command)
        self.assertIn("--resume", command)
        self.assertIn("--compile", command)
        self.assertNotIn("--steps", command)

    def test_supervisor_starts_exact_argv_and_injects_stop_channel(self) -> None:
        command = [
            str(Path(sys.executable).resolve()),
            "-m",
            "torch.distributed.run",
            "--standalone",
        ]
        environment = {
            "PATH": "/poisoned",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WANDB_MODE": "disabled",
        }
        process = mock.Mock()
        process.wait.return_value = 0
        with mock.patch.object(launch.subprocess, "Popen", return_value=process) as popen:
            exit_code = launch.supervise_torchrun(
                command,
                environment,
                graceful_shutdown_timeout_seconds=60,
            )
        self.assertEqual(exit_code, 0)
        launched_environment = popen.call_args.kwargs["env"]
        self.assertEqual(popen.call_args.args[0], command)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(launched_environment["PATH"], "/poisoned")
        self.assertIn("PRETRAIN_STOP_REQUEST_FILE", launched_environment)
        process.wait.assert_called_once_with(timeout=0.25)

    def test_execute_handoff_reexecs_clean_context_supervisor(self) -> None:
        python_executable = str(Path(sys.executable).resolve())
        command = [python_executable, "-m", "torch.distributed.run"]
        environment = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with mock.patch.object(launch.os, "execvpe") as execvpe:
            with self.assertRaisesRegex(AssertionError, "unexpectedly returned"):
                launch.exec_clean_supervisor(
                    python_executable=python_executable,
                    command=command,
                    environment=environment,
                    graceful_shutdown_timeout_seconds=1800,
                )
        argv = execvpe.call_args.args[1]
        self.assertEqual(execvpe.call_args.args[0], python_executable)
        self.assertEqual(execvpe.call_args.args[2], environment)
        self.assertEqual(argv[0], python_executable)
        self.assertEqual(argv[2], "--internal-supervise-torchrun")
        self.assertEqual(argv[3:5], ["1800", "--"])
        self.assertEqual(argv[5:], command)

    def test_supervisor_holds_torchrun_open_for_worker_checkpoint(self) -> None:
        observed: dict[str, object] = {}

        class SignalProcess:
            def __init__(self, environment):
                self.environment = environment
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    __import__("os").kill(__import__("os").getpid(), signal.SIGTERM)
                    raise launch.subprocess.TimeoutExpired("torchrun", timeout)
                request = Path(
                    self.environment["PRETRAIN_STOP_REQUEST_FILE"]
                )
                observed["request"] = request.read_text(encoding="ascii")
                return 1

            def terminate(self):
                self.fail("supervisor terminated torchrun before its deadline")

            def kill(self):
                self.fail("supervisor killed torchrun before its deadline")

            @staticmethod
            def fail(message):
                raise AssertionError(message)

        def start_process(command, *, env, start_new_session):
            del command
            self.assertTrue(start_new_session)
            return SignalProcess(env)

        with mock.patch.object(launch.subprocess, "Popen", side_effect=start_process):
            exit_code = launch.supervise_torchrun(
                [str(Path(sys.executable).resolve()), "-m", "torch.distributed.run"],
                {},
                graceful_shutdown_timeout_seconds=60,
            )
        self.assertEqual(exit_code, 128 + int(signal.SIGTERM))
        self.assertEqual(observed["request"], f"{int(signal.SIGTERM)}\n")

    def test_supervisor_kills_entire_child_session_after_grace_expires(self) -> None:
        class StuckProcess:
            pid = 43210

            def __init__(self):
                self.initial_wait = True
                self.terminated = False

            def wait(self, timeout=None):
                if timeout == 0.25 and self.initial_wait:
                    self.initial_wait = False
                    __import__("os").kill(__import__("os").getpid(), signal.SIGTERM)
                    raise launch.subprocess.TimeoutExpired("torchrun", timeout)
                if timeout == 35:
                    raise launch.subprocess.TimeoutExpired("torchrun", timeout)
                self.assertEqual(timeout, None)
                return -int(signal.SIGKILL)

            def terminate(self):
                self.terminated = True

            @staticmethod
            def assertEqual(found, expected):
                if found != expected:
                    raise AssertionError(f"found {found!r}, expected {expected!r}")

        process = StuckProcess()
        with (
            mock.patch.object(launch.subprocess, "Popen", return_value=process),
            mock.patch.object(launch.time, "monotonic", side_effect=(100.0, 102.0)),
            mock.patch.object(launch.os, "killpg") as killpg,
        ):
            exit_code = launch.supervise_torchrun(
                [str(Path(sys.executable).resolve()), "-m", "torch.distributed.run"],
                {},
                graceful_shutdown_timeout_seconds=1,
            )
        self.assertTrue(process.terminated)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(exit_code, 128 + int(signal.SIGTERM))

    def test_protected_passthrough_flags_are_rejected(self) -> None:
        for values in (
            ["--", "--steps", "10"],
            ["--step", "10"],
            ["--checkpoint=/tmp/wrong.pt"],
            ["--graceful-shutdown-timeout-seconds=1"],
            ["--no-activation-checkpointing"],
            ["--verify-packed-payloads"],
        ):
            with self.subTest(values=values), self.assertRaisesRegex(
                launch.PreflightError, "owned by the launcher"
            ):
                launch.validate_extra_trainer_args(values)

    def test_wandb_modes_are_lazy_and_online_requires_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            disabled = launch.inspect_wandb(
                mode="disabled",
                checkpoint_parent=root,
                durable_root=root,
                find_spec=lambda _name: self.fail("disabled mode imported W&B"),
            )
            self.assertFalse(disabled["dependency_checked"])
            with self.assertRaisesRegex(launch.PreflightError, "WANDB_API_KEY"):
                launch.inspect_wandb(
                    mode="online",
                    checkpoint_parent=root,
                    durable_root=root,
                    environment={},
                    find_spec=lambda _name: object(),
                    package_version=lambda _name: "0.18.7",
                    credentials_present=lambda _environment: False,
                )


if __name__ == "__main__":
    unittest.main()
