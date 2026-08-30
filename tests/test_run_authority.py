from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pretrain import run_authority as authority


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


class RunAuthorityUnitTests(unittest.TestCase):
    def test_clean_git_identity_rejects_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"], check=True
            )
            tracked = root / "tracked.txt"
            tracked.write_text("immutable\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            identity = authority.inspect_clean_git(root)
            self.assertTrue(identity["clean"])
            self.assertRegex(identity["commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(identity["head_archive_sha256"], r"^[0-9a-f]{64}$")
            (root / "untracked.txt").write_text("mutation\n", encoding="utf-8")
            with self.assertRaisesRegex(authority.RunAuthorityError, "not clean"):
                authority.inspect_clean_git(root)

    def test_hardware_contract_requires_exactly_six_ddp_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            payload = {
                "format": authority.HARDWARE_FORMAT,
                "format_version": 1,
                "status": "accepted",
                "topology": "single-node",
                "world_size": 5,
                "gpu_count": 5,
                "gpu_model": "Fixture GPU",
                "gpu_memory_bytes": 80 * 1024**3,
                "compute_capability": [9, 0],
                "multiprocessor_count": 100,
                "driver_version": "fixture",
                "cuda_runtime_version": "fixture",
                "cudnn_version": "fixture",
                "nccl_version": "fixture",
                "torch_version": "fixture",
                "bf16_supported": True,
                "distributed_strategy": "ddp",
            }
            _write_json(path, payload)
            with self.assertRaisesRegex(authority.RunAuthorityError, "exactly six"):
                authority.inspect_hardware_contract(path)

    def test_cost_cap_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(authority.RunAuthorityError, "exceeds authorized cap"):
            authority._economics(
                consumed_input_tokens=52_580_000_000,
                measured_input_tokens_per_second="10000",
                hourly_cost_usd="18",
                total_cost_cap_usd="10",
            )

    def test_training_package_floor_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(authority.RunAuthorityError, "torch>=2.6"):
            authority._validate_training_package_versions(
                {"torch": "2.5.1", "numpy": "2.2.0", "tokenizers": "0.21.0"}
            )

    def test_publication_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authority.json"
            authority.publish_run_authority(output, {"example": True})
            with self.assertRaisesRegex(authority.RunAuthorityError, "overwrite"):
                authority.publish_run_authority(output, {"example": True})

    def test_launcher_argv_rejects_wrong_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "scripts").mkdir(parents=True)
            launcher = project / "scripts" / "launch_pretraining.py"
            launcher.write_text("# fixture\n", encoding="utf-8")
            train_manifest = root / "train.json"
            validation_manifest = root / "validation.json"
            train_cert = root / "train-cert.json"
            validation_cert = root / "validation-cert.json"
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            for path in (train_manifest, validation_manifest, train_cert, validation_cert):
                path.write_text("{}", encoding="utf-8")
            argv = [
                sys.executable,
                str(launcher),
                "--train-order-manifest",
                str(train_manifest),
                "--validation-order-manifest",
                str(validation_manifest),
                "--tokenizer",
                str(tokenizer),
                "--train-data-evidence",
                str(train_cert),
                "--validation-data-evidence",
                str(validation_cert),
                "--nproc-per-node",
                "5",
                "--model-size",
                "1.3b",
                "--workers",
                "4",
                "--checkpoint-every",
                "10",
                "--eval-every",
                "10",
                "--eval-batches",
                "2",
                "--log-every",
                "1",
                "--wandb-mode",
                "offline",
                "--eval-at-start",
                "--execute",
                "--",
                "--learning-rate",
                "0.0003",
                "--weight-decay",
                "0.1",
                "--beta1",
                "0.9",
                "--beta2",
                "0.95",
                "--adam-eps",
                "0.00000001",
                "--max-grad-norm",
                "1",
                "--min-learning-rate",
                "0.00003",
                "--warmup-steps",
                "1",
                "--seed",
                "1234",
                "--activation-checkpointing",
                "--fused-adamw",
            ]
            argv_file = root / "argv.json"
            _write_json(argv_file, argv)
            recipe = {
                "seed": 1234,
                "wandb_mode": "offline",
                "activation_checkpointing": True,
                "compile_model": False,
                "optimizer": {
                    "fused": True,
                    "learning_rate": "0.0003",
                    "weight_decay": "0.1",
                    "beta1": "0.9",
                    "beta2": "0.95",
                    "epsilon": "0.00000001",
                    "max_grad_norm": "1",
                },
                "schedule": {"minimum_learning_rate": "0.00003", "warmup_steps": 1},
                "cadence": {
                    "checkpoint_every": 10,
                    "eval_every": 10,
                    "eval_batches": 2,
                    "log_every": 1,
                },
            }
            order = lambda path: {"manifest": {"path": str(path.resolve())}}
            certification = lambda path: {
                "artifact": {"path": str(path.resolve())}
            }
            with self.assertRaisesRegex(authority.RunAuthorityError, "nproc-per-node"):
                authority.inspect_launcher_argv(
                    argv_file,
                    project_root=project,
                    train_order=order(train_manifest),
                    validation_order=order(validation_manifest),
                    tokenizer_root=tokenizer,
                    train_certification=certification(train_cert),
                    validation_certification=certification(validation_cert),
                    recipe=recipe,
                    geometry={"receipt": {"accepted": {"workers": 4}}},
                )

    def test_certified_inventory_detects_payload_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            payload = root / "order.bin"
            packed = root / "tokens.bin"
            manifest.write_text("{}", encoding="utf-8")
            payload.write_bytes(b"order")
            packed.write_bytes(b"packed")

            def record(path: Path, kind: str, *, actual: bool) -> dict[str, object]:
                metadata = path.stat()
                digest = authority.sha256_file(path)
                result: dict[str, object] = {
                    "path": str(path.resolve()),
                    "kind": kind,
                    "bytes": metadata.st_size,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mtime_ns": metadata.st_mtime_ns,
                    "ctime_ns": metadata.st_ctime_ns,
                    "declared_sha256": digest,
                }
                if actual:
                    result["actual_sha256"] = digest
                return result

            inventory = {
                "order_manifest": record(manifest, "order-manifest", actual=True),
                "order_payload": record(payload, "order-payload", actual=False),
                "packed_files": [record(packed, "packed-tokens", actual=False)],
            }
            order = {
                "manifest": authority._artifact(manifest, label="manifest"),
                "payload": authority._artifact(payload, label="payload"),
            }
            authority._validate_certified_inventory(inventory, order=order)
            packed.write_bytes(b"mutate")
            with self.assertRaisesRegex(authority.RunAuthorityError, "identity changed"):
                authority._validate_certified_inventory(inventory, order=order)


class RunAuthorityMutationTest(unittest.TestCase):
    def test_validate_recollects_inputs_and_detects_recipe_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "scripts").mkdir(parents=True)
            launcher_source = project / "scripts" / "launch_pretraining.py"
            launcher_source.write_text("# fixture\n", encoding="utf-8")
            package_lock = root / "package-lock.json"
            _write_json(package_lock, {"fixture": True})
            hardware = root / "hardware.json"
            hardware_payload = {
                "format": authority.HARDWARE_FORMAT,
                "format_version": 1,
                "status": "accepted",
                "topology": "single-node",
                "world_size": 6,
                "gpu_count": 6,
                "gpu_model": "Fixture GPU",
                "gpu_memory_bytes": 80 * 1024**3,
                "compute_capability": [9, 0],
                "multiprocessor_count": 100,
                "driver_version": "fixture",
                "cuda_runtime_version": "fixture",
                "cudnn_version": "fixture",
                "nccl_version": "fixture",
                "torch_version": "test-torch",
                "bf16_supported": True,
                "distributed_strategy": "ddp",
            }
            _write_json(hardware, hardware_payload)
            train_manifest = root / "train-manifest.json"
            validation_manifest = root / "validation-manifest.json"
            train_payload = root / "train-order.bin"
            validation_payload = root / "validation-order.bin"
            train_payload.write_bytes(b"train-order")
            validation_payload.write_bytes(b"validation-order")
            _write_json(train_manifest, {"split": "train"})
            _write_json(validation_manifest, {"split": "validation"})
            tokenizer_digest = hashlib.sha256(b"{}").hexdigest()

            def fake_order(path: str | Path, *, expected_split: str) -> dict[str, object]:
                manifest = train_manifest if expected_split == "train" else validation_manifest
                payload = train_payload if expected_split == "train" else validation_payload
                result: dict[str, object] = {
                    "manifest": authority._artifact(manifest, label="fixture manifest"),
                    "payload": authority._artifact(payload, label="fixture order"),
                    "split": expected_split,
                    "sequence_length": 4096,
                    "vocab_size": 49_152,
                    "eos_token_id": 0,
                    "tokenizer_manifest_sha256": tokenizer_digest,
                    "shuffle_seed": 1234,
                    "shuffle_rng": "fixture",
                    "rows": 60,
                    "geometry": None,
                }
                if expected_split == "train":
                    result["geometry"] = {
                        "global_microbatch_rows": 6,
                        "gradient_accumulation_steps": 2,
                        "optimizer_update_rows": 12,
                        "optimizer_updates": 100,
                        "consumed_global_microbatches": 200,
                        "sequence_length": 4096,
                        "consumed_rows": 1200,
                        "dropped_rows": 0,
                        "consumed_input_tokens": 4_915_200,
                        "dropped_input_tokens": 0,
                        "consumed_supervised_tokens": 4_000_000,
                        "dropped_supervised_tokens": 0,
                        "consumed_input_tokens_per_domain": {},
                        "consumed_supervised_tokens_per_domain": {},
                    }
                return result

            geometry = root / "geometry.json"
            geometry_payload = {
                "format": authority.GEOMETRY_FORMAT,
                "format_version": 1,
                "status": "pass",
                "hardware_contract_sha256": authority.sha256_file(hardware),
                "train_order_manifest_sha256": authority.sha256_file(train_manifest),
                "validation_order_manifest_sha256": authority.sha256_file(validation_manifest),
                "accepted": {
                    "global_microbatch_rows": 6,
                    "gradient_accumulation_steps": 2,
                    "workers": 4,
                    "overfit_batch_rows": 6,
                    "compile_model": False,
                    "activation_checkpointing": True,
                    "precision": "bfloat16",
                    "parameter_dtype": "float32",
                },
                "measurements": {
                    "aggregate_input_tokens_per_second": "1000",
                    "peak_memory_allocated_bytes_per_gpu": 1,
                    "peak_memory_reserved_bytes_per_gpu": 1,
                    "minimum_free_memory_bytes_per_gpu": 1,
                    "checkpoint_seconds": "1",
                    "data_wait_fraction": "0",
                    "scaling_efficiency": "0.9",
                    "soak_steps": 10,
                },
            }
            _write_json(geometry, geometry_payload)
            _write_sidecar(geometry)
            recipe = root / "recipe.json"
            recipe_payload = {
                "format": authority.RECIPE_FORMAT,
                "format_version": 1,
                "model_size": "1.3b",
                "precision": "bfloat16",
                "parameter_dtype": "float32",
                "deterministic": True,
                "seed": 1234,
                "optimizer": {
                    "name": "AdamW",
                    "fused": True,
                    "learning_rate": "0.0003",
                    "weight_decay": "0.1",
                    "beta1": "0.9",
                    "beta2": "0.95",
                    "epsilon": "0.00000001",
                    "max_grad_norm": "1",
                },
                "schedule": {
                    "name": "warmup-cosine",
                    "minimum_learning_rate": "0.00003",
                    "warmup_steps": 1,
                },
                "cadence": {
                    "checkpoint_every": 10,
                    "eval_every": 10,
                    "eval_batches": 2,
                    "eval_at_start": True,
                    "log_every": 1,
                },
                "wandb_mode": "offline",
                "activation_checkpointing": True,
                "compile_model": False,
            }
            _write_json(recipe, recipe_payload)
            train_cert = root / "train-cert.json"
            validation_cert = root / "validation-cert.json"
            _write_json(train_cert, {"fixture": "train"})
            _write_json(validation_cert, {"fixture": "validation"})
            _write_sidecar(train_cert)
            _write_sidecar(validation_cert)
            tokenizer_root = root / "tokenizer"
            tokenizer_root.mkdir()
            tokenizer_manifest = tokenizer_root / "TOKENIZER_MANIFEST.json"
            tokenizer_manifest.write_text("{}", encoding="utf-8")
            argv_file = root / "launcher-argv.json"
            _write_json(argv_file, [sys.executable, str(launcher_source), "--execute"])

            def fake_certification(
                path, *, expected_split, order, project_root, environment
            ):
                del order, project_root, environment
                bound = authority._verified_sidecar(path, label="fixture certification")
                return {**bound, "receipt": {"split": expected_split}}

            def fake_launcher(path, **kwargs):
                del kwargs
                argv = json.loads(Path(path).read_text(encoding="utf-8"))
                return {
                    "argv_file": authority._artifact(path, label="fixture argv"),
                    "argv": argv,
                    "argv_sha256": authority.canonical_sha256(argv),
                    "launcher_source": authority._artifact(
                        launcher_source, label="fixture launcher"
                    ),
                }

            package_identity = {
                "lock": authority._artifact(package_lock, label="fixture lock"),
                "python": {"fixture": True},
                "packages_sha256": "b" * 64,
                "package_count": 3,
                "torch_version": "test-torch",
                "numpy_version": "test-numpy",
            }
            git_identity = {
                "project_root": str(project.resolve()),
                "commit": "c" * 40,
                "tree": "d" * 40,
                "head_archive_sha256": "a" * 64,
                "head_archive_bytes": 10240,
                "branch": "main",
                "origin_sha256": None,
                "clean": True,
            }
            tokenizer_identity = SimpleNamespace(
                manifest_path=tokenizer_manifest,
                manifest_sha256=tokenizer_digest,
                vocabulary_sha256="e" * 64,
                vocab_size=49_152,
            )
            patches = (
                mock.patch.object(authority, "inspect_clean_git", return_value=git_identity),
                mock.patch.object(authority, "inspect_package_lock", return_value=package_identity),
                mock.patch.object(authority, "inspect_order", side_effect=fake_order),
                mock.patch.object(
                    authority, "inspect_certification", side_effect=fake_certification
                ),
                mock.patch.object(
                    authority, "verify_tokenizer_identity", return_value=tokenizer_identity
                ),
                mock.patch.object(authority, "inspect_launcher_argv", side_effect=fake_launcher),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                payload = authority.collect_run_authority(
                    project_root=project,
                    package_lock=package_lock,
                    container_image_digest=f"sha256:{'f' * 64}",
                    hardware_contract=hardware,
                    geometry_receipt=geometry,
                    train_order_manifest=train_manifest,
                    validation_order_manifest=validation_manifest,
                    train_certification=train_cert,
                    validation_certification=validation_cert,
                    tokenizer_root=tokenizer_root,
                    training_recipe=recipe,
                    launcher_argv_json=argv_file,
                    measured_input_tokens_per_second="1000",
                    hourly_cost_usd="18",
                    total_cost_cap_usd="100",
                )
                output = root / "authority.json"
                authority.publish_run_authority(output, payload)
                self.assertEqual(authority.validate_run_authority(output)["status"], "valid")
                recipe_payload["optimizer"]["learning_rate"] = "0.0002"
                _write_json(recipe, recipe_payload)
                with self.assertRaisesRegex(
                    authority.RunAuthorityError, "no longer matches"
                ):
                    authority.validate_run_authority(output)


if __name__ == "__main__":
    unittest.main()
