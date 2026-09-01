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


def _qualified_hardware_payload(
    *,
    package_identity: dict[str, object],
    git_identity: dict[str, object],
    qualification_script: Path,
    requirements_train: Path,
    requirements_wandb: Path,
) -> dict[str, object]:
    gpu_uuids = [f"GPU-{index:032x}" for index in range(authority.WORLD_SIZE)]
    devices = [
        {
            "visible_index": index,
            "physical_index": index,
            "uuid": uuid,
            "name": "Fixture GPU",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "compute_capability": [9, 0],
            "total_memory_bytes": 80 * 1024**3,
            "available_memory_bytes": 79 * 1024**3,
            "multiprocessor_count": 100,
            "bf16_supported": True,
        }
        for index, uuid in enumerate(gpu_uuids)
    ]
    return {
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
        "driver_version": "fixture-driver",
        "cuda_runtime_version": "12.8",
        "cudnn_version": "9.8.0",
        "nccl_version": "2.25.1",
        "torch_version": package_identity["torch_version"],
        "bf16_supported": True,
        "distributed_strategy": "ddp",
        "created_utc": "2026-09-01T00:00:00+00:00",
        "qualification": {
            "format": authority.POD_QUALIFICATION_FORMAT,
            "format_version": authority.POD_QUALIFICATION_VERSION,
            "status": "pass",
            "nvlink_policy": "observe",
            "gpu": {
                "devices": devices,
                "driver_version": "fixture-driver",
                "cuda_runtime_version": "12.8",
                "cudnn_version": "9.8.0",
                "nccl_version": "2.25.1",
                "torch_version": package_identity["torch_version"],
                "nccl_smoke": {
                    "status": "pass",
                    "backend": "nccl",
                    "world_size": 6,
                    "completed_ranks": 6,
                    "all_reduce_sum": 21,
                    "all_reduce_dtype": "bfloat16",
                    "local_ranks": list(range(6)),
                    "device_uuids_in_rank_order": gpu_uuids,
                    "rank_results_sha256": "a" * 64,
                },
            },
            "host": {
                "data": {"status": "pass"},
                "environment": {"secrets_recorded": False},
                "wandb": {"credential_value_recorded": False},
                "package_lock": package_identity,
            },
            "source": {
                "qualification_script": authority._artifact(
                    qualification_script, label="fixture qualifier"
                ),
                "requirements_train": authority._artifact(
                    requirements_train, label="fixture train requirements"
                ),
                "requirements_wandb": authority._artifact(
                    requirements_wandb, label="fixture W&B requirements"
                ),
                "git": git_identity,
                "argv": [
                    str(qualification_script),
                    "verify",
                    "--package-lock",
                    str(package_identity["lock"]["path"]),
                ],
            },
        },
    }


def _write_corpus_qualification(
    *,
    root: Path,
    corpus_root: Path,
    tokenizer_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    tokenizer_manifest_sha256: str,
    vocabulary_sha256: str,
    vocab_size: int = 49_152,
    eos_token_id: int = 0,
    sequence_length: int = 4096,
) -> Path:
    corpus_manifest = corpus_root / "manifest.json"
    _write_json(corpus_manifest, {"fixture": "corpus"})
    corpus_manifest_sidecar = corpus_root / "manifest.sha256"
    corpus_manifest_sidecar.write_text(
        f"{authority.sha256_file(corpus_manifest)}  manifest.json\n",
        encoding="ascii",
    )
    test_manifest = corpus_root / "test-manifest.json"
    _write_json(test_manifest, {"split": "test"})
    order_paths = {
        "train": train_manifest,
        "validation": validation_manifest,
        "test": test_manifest,
    }
    model = authority._validate_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        activation_checkpointing=False,
    )["config"]
    project_root = Path(authority.__file__).resolve().parents[1]
    validator_paths = {
        "qualification": project_root / "scripts" / "qualify_training_corpus.py",
        "pretrain_data": project_root / "pretrain" / "data.py",
        "materialize_contract": project_root / "pretrain" / "materialize.py",
        "tokenizer_identity": project_root / "pretrain" / "tokenizer_identity.py",
        "model": project_root / "pretrain" / "model.py",
        "model_config_loader": project_root / "pretrain" / "hf_export.py",
    }
    provenance_root = corpus_root / "provenance"
    provenance_root.mkdir(exist_ok=True)
    provenance: dict[str, object] = {}
    for name in ("source", "policy", "tokenizer", "fingerprints"):
        provenance_path = provenance_root / f"{name}.json"
        _write_json(provenance_path, {"fixture": name})
        provenance[name] = {
            "path": provenance_path.relative_to(corpus_root).as_posix(),
            "bytes": provenance_path.stat().st_size,
            "sha256": authority.sha256_file(provenance_path),
        }
    provenance["bindings"] = {
        "selection_manifest_sha256": "1" * 64,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "materialization_curation_policy_sha256": "2" * 64,
        "source_curation_policy_sha256": "3" * 64,
        "split_authority": "completed-curation-decision-shards",
        "authenticated_leakage_audit": {
            "content_hashes_in_multiple_splits": 0,
            "canonical_clusters_in_multiple_splits": 0,
            "source_groups_in_multiple_splits": 0,
            "cross_bucket_code_repo_groups_in_multiple_splits": 0,
        },
        "raw_archive_count": 1,
        "raw_archive_inventory_sha256": "4" * 64,
    }
    receipt = {
        "format": authority.CORPUS_QUALIFICATION_FORMAT,
        "format_version": authority.CORPUS_QUALIFICATION_VERSION,
        "status": "pass",
        "started_utc": "2026-09-01T00:00:00+00:00",
        "completed_utc": "2026-09-01T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "checks": {name: True for name in authority.CORPUS_QUALIFICATION_CHECKS},
        "acceptance_policy": {
            "world_size": 6,
            "expected_input_tokens": dict(authority.EXPECTED_INPUT_TOKEN_TARGETS),
            "expected_input_token_weights": dict(authority.EXPECTED_INPUT_TOKEN_WEIGHTS),
            "maximum_target_shortfall_fraction": 1e-3,
            "mixture_absolute_tolerance": 1e-6,
            "sample_rows_per_domain": 8,
            "sample_seed": 20_260_901,
            "full_packed_payload_checksums": True,
            "full_packed_semantic_scan": True,
            "full_order_uniqueness_scan": True,
            "full_document_index_scan": True,
            "exact_disk_backed_split_identity_scan": True,
        },
        "corpus": {
            "manifest_path": str(corpus_manifest.resolve()),
            "manifest_bytes": corpus_manifest.stat().st_size,
            "manifest_sha256": authority.sha256_file(corpus_manifest),
            "sidecar_path": str(corpus_manifest_sidecar.resolve()),
            "sidecar_sha256": authority.sha256_file(corpus_manifest_sidecar),
            "stable_tree_inventory": authority._current_tree_identity(
                corpus_root, label="fixture corpus"
            ),
        },
        "tokenizer": {
            "root": str(tokenizer_root.resolve()),
            "manifest_path": str(
                (tokenizer_root / "TOKENIZER_MANIFEST.json").resolve()
            ),
            "manifest_sha256": tokenizer_manifest_sha256,
            "vocabulary_sha256": vocabulary_sha256,
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "stable_tree_inventory": authority._current_tree_identity(
                tokenizer_root, label="fixture tokenizer"
            ),
        },
        "model": {
            "source": "pretrain.model.ModelConfig defaults",
            "sha256": None,
            "config": model,
        },
        "validator": {
            "sources": {
                name: authority._artifact(path, label=f"fixture validator {name}")
                for name, path in validator_paths.items()
            },
            "runtime": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": sys.version,
                "numpy_version": "2.2.6",
                "torch_version": "2.7.1+cpu",
                "sqlite_version": "3.46.0",
                "tokenizers_version": "0.21.4",
                "zstandard_version": "0.23.0",
                "packed_format_version": authority.training_data.FORMAT_VERSION,
                "order_format_version": authority.training_data.ORDER_FORMAT_VERSION,
            },
        },
        "common_data_contract": {
            "sequence_length": sequence_length,
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        },
        "provenance": provenance,
        "splits": {
            split: {
                "order": {
                    "manifest": path.relative_to(corpus_root).as_posix(),
                    "manifest_sha256": authority.sha256_file(path),
                },
                "packed": {
                    domain: {"fixture": True}
                    for domain in authority.training_data.DOMAIN_ORDER
                },
                "deterministic_samples": {
                    domain: {"fixture": True}
                    for domain in authority.training_data.DOMAIN_ORDER
                },
            }
            for split, path in order_paths.items()
        },
        "document_indexes": {
            split: {
                domain: {"fixture": True}
                for domain in authority.training_data.DOMAIN_ORDER
            }
            for split in order_paths
        },
        "split_identity_audit": {
            "method": "exact-sqlite-without-rowid-split-bitmask-union",
            "identity_kinds": ["source", "split_group"],
            "cross_split_collisions": {"source": 0, "split_group": 0},
            "identity_occurrences": {"source": 3, "split_group": 3},
            "unique_identities": {"source": 3, "split_group": 3},
            "collision_examples": {"source": [], "split_group": []},
            "scratch_database_bytes": 4096,
        },
    }
    output = root / "corpus-qualification" / "qualification.json"
    output.parent.mkdir()
    _write_json(output, receipt)
    _write_sidecar(output)
    return output


def _minimal_corpus_qualification(
    root: Path,
    *,
    extra_corpus_files: dict[str, bytes] | None = None,
) -> tuple[Path, Path, Path]:
    corpus_root = root / "corpus"
    corpus_root.mkdir()
    train_manifest = corpus_root / "train-manifest.json"
    validation_manifest = corpus_root / "validation-manifest.json"
    _write_json(train_manifest, {"split": "train"})
    _write_json(validation_manifest, {"split": "validation"})
    for relative, payload in (extra_corpus_files or {}).items():
        destination = corpus_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    tokenizer_root = root / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer_manifest = tokenizer_root / "TOKENIZER_MANIFEST.json"
    tokenizer_manifest.write_text("{}", encoding="utf-8")
    tokenizer_digest = authority.sha256_file(tokenizer_manifest)
    qualification = _write_corpus_qualification(
        root=root,
        corpus_root=corpus_root,
        tokenizer_root=tokenizer_root,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        tokenizer_manifest_sha256=tokenizer_digest,
        vocabulary_sha256="e" * 64,
    )
    return qualification, corpus_root, tokenizer_root


class RunAuthorityUnitTests(unittest.TestCase):
    def test_corpus_qualification_accepts_real_qualifier_generation(self) -> None:
        from scripts.qualify_training_corpus import publish_receipt, qualify_corpus
        from tests.test_qualify_training_corpus import QualificationFixture

        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            config = fixture.config()
            payload = qualify_corpus(config)
            receipt, _ = publish_receipt(config.output, payload)
            with mock.patch.object(
                authority,
                "EXPECTED_INPUT_TOKEN_TARGETS",
                dict(config.expected_targets),
            ), mock.patch.object(
                authority,
                "MINIMUM_SAMPLE_ROWS_PER_DOMAIN",
                config.sample_rows_per_domain,
            ):
                inspected = authority.inspect_corpus_qualification(receipt)
            self.assertEqual(inspected["receipt"]["status"], "pass")

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
            _write_sidecar(path)
            with self.assertRaisesRegex(authority.RunAuthorityError, "exactly six"):
                authority.inspect_hardware_contract(path)

    def test_hardware_contract_requires_passing_qualified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "scripts").mkdir(parents=True)
            qualifier = project / "scripts" / "qualify_runpod_pod.py"
            qualifier.write_text("# qualifier\n", encoding="utf-8")
            train_requirements = project / "requirements-train.txt"
            train_requirements.write_text("torch==fixture\n", encoding="utf-8")
            wandb_requirements = project / "requirements-wandb.txt"
            wandb_requirements.write_text("wandb==fixture\n", encoding="utf-8")
            lock = root / "lock.json"
            _write_json(lock, {"fixture": True})
            executable = Path(sys.executable).resolve()
            package = {
                "lock": authority._artifact(lock, label="fixture lock"),
                "python": {
                    "implementation": "cpython",
                    "version": "3.12.11",
                    "executable": str(executable),
                    "executable_bytes": executable.stat().st_size,
                    "executable_sha256": authority.sha256_file(executable),
                },
                "packages_sha256": "1" * 64,
                "package_count": 3,
                "torch_version": "2.7.1+cu128",
                "numpy_version": "2.2.6",
            }
            git = {
                "project_root": str(project.resolve()),
                "commit": "2" * 40,
                "tree": "3" * 40,
                "head_archive_sha256": "4" * 64,
                "head_archive_bytes": 10240,
                "branch": "main",
                "origin_sha256": None,
                "clean": True,
            }
            payload = _qualified_hardware_payload(
                package_identity=package,
                git_identity=git,
                qualification_script=qualifier,
                requirements_train=train_requirements,
                requirements_wandb=wandb_requirements,
            )
            payload["qualification"]["status"] = "fail"
            path = root / "hardware.json"
            _write_json(path, payload)
            _write_sidecar(path)
            with self.assertRaisesRegex(authority.RunAuthorityError, "did not pass"):
                authority.inspect_hardware_contract(path)

            payload["qualification"]["status"] = "pass"
            _write_json(path, payload)
            _write_sidecar(path)
            inspected = authority.inspect_hardware_contract(path)
            changed_package = dict(package)
            changed_package["packages_sha256"] = "5" * 64
            with self.assertRaisesRegex(authority.RunAuthorityError, "packages_sha256"):
                authority._validate_qualification_provenance(
                    hardware=inspected,
                    environment=changed_package,
                    git=git,
                    project_root=project,
                )

    def test_corpus_qualification_rejects_failed_or_torn_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qualification, _, _ = _minimal_corpus_qualification(Path(temporary))
            receipt = json.loads(qualification.read_text(encoding="utf-8"))
            receipt["status"] = "fail"
            _write_json(qualification, receipt)
            _write_sidecar(qualification)
            with self.assertRaisesRegex(authority.RunAuthorityError, "did not pass"):
                authority.inspect_corpus_qualification(qualification)

            receipt["status"] = "pass"
            _write_json(qualification, receipt)
            qualification.with_name(f"{qualification.name}.sha256").write_text(
                f"{'0' * 64}  {qualification.name}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(authority.RunAuthorityError, "SHA-256 sidecar"):
                authority.inspect_corpus_qualification(qualification)

    def test_corpus_qualification_rejects_document_index_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "provenance/documents/test/python/archive-000000.jsonl.zst"
            qualification, corpus_root, _ = _minimal_corpus_qualification(
                root, extra_corpus_files={relative: b"certified document index"}
            )
            authority.inspect_corpus_qualification(qualification)
            (corpus_root / relative).write_bytes(b"mutated document index")
            with self.assertRaisesRegex(
                authority.RunAuthorityError, "corpus tree identity changed"
            ):
                authority.inspect_corpus_qualification(qualification)

    def test_corpus_qualification_rejects_added_corpus_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qualification, corpus_root, _ = _minimal_corpus_qualification(
                Path(temporary)
            )
            authority.inspect_corpus_qualification(qualification)
            (corpus_root / "unexpected.bin").write_bytes(b"not qualified")
            with self.assertRaisesRegex(
                authority.RunAuthorityError, "corpus tree identity changed"
            ):
                authority.inspect_corpus_qualification(qualification)

    def test_corpus_qualification_rejects_loose_policy_or_changed_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qualification, _, _ = _minimal_corpus_qualification(Path(temporary))
            receipt = json.loads(qualification.read_text(encoding="utf-8"))
            receipt["acceptance_policy"]["maximum_target_shortfall_fraction"] = 0.5
            _write_json(qualification, receipt)
            _write_sidecar(qualification)
            with self.assertRaisesRegex(authority.RunAuthorityError, "too loose"):
                authority.inspect_corpus_qualification(qualification)

            receipt["acceptance_policy"]["maximum_target_shortfall_fraction"] = 1e-3
            receipt["validator"]["sources"]["qualification"]["sha256"] = "f" * 64
            _write_json(qualification, receipt)
            _write_sidecar(qualification)
            with self.assertRaisesRegex(
                authority.RunAuthorityError, "validator source changed"
            ):
                authority.inspect_corpus_qualification(qualification)

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
                "--run-authority",
                str(root / "authority.json"),
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
                "--wandb-mode",
                "offline",
                "--eval-at-start",
                "--activation-checkpointing",
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
                "--log-every",
                "1",
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
            argv[argv.index("--nproc-per-node") + 1] = "6"
            _write_json(argv_file, argv)
            inspected = authority.inspect_launcher_argv(
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
            self.assertEqual(inspected["argv"], argv)

    def test_publication_must_match_self_authorized_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intended = root / "intended.json"
            payload = {
                "format": authority.AUTHORITY_FORMAT,
                "launcher": {
                    "argv": [
                        sys.executable,
                        "/fixture/launch_pretraining.py",
                        "--run-authority",
                        str(intended),
                    ]
                },
            }
            with self.assertRaisesRegex(
                authority.RunAuthorityError, "Publication path differs"
            ):
                authority.publish_run_authority(root / "wrong.json", payload)

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
            corpus_root = root / "corpus"
            corpus_root.mkdir()
            (project / "scripts").mkdir(parents=True)
            launcher_source = project / "scripts" / "launch_pretraining.py"
            launcher_source.write_text("# fixture\n", encoding="utf-8")
            qualification_script = project / "scripts" / "qualify_runpod_pod.py"
            qualification_script.write_text("# fixture qualifier\n", encoding="utf-8")
            requirements_train = project / "requirements-train.txt"
            requirements_train.write_text("torch==test\n", encoding="utf-8")
            requirements_wandb = project / "requirements-wandb.txt"
            requirements_wandb.write_text("wandb==test\n", encoding="utf-8")
            package_lock = root / "package-lock.json"
            _write_json(package_lock, {"fixture": True})
            package_identity = {
                "lock": authority._artifact(package_lock, label="fixture lock"),
                "python": {
                    "implementation": "cpython",
                    "version": "3.12.11",
                    "executable": str(Path(sys.executable).resolve()),
                    "executable_bytes": Path(sys.executable).resolve().stat().st_size,
                    "executable_sha256": authority.sha256_file(
                        Path(sys.executable).resolve()
                    ),
                },
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
            hardware = root / "hardware.json"
            hardware_payload = _qualified_hardware_payload(
                package_identity=package_identity,
                git_identity=git_identity,
                qualification_script=qualification_script,
                requirements_train=requirements_train,
                requirements_wandb=requirements_wandb,
            )
            _write_json(hardware, hardware_payload)
            _write_sidecar(hardware)
            train_manifest = corpus_root / "train-manifest.json"
            validation_manifest = corpus_root / "validation-manifest.json"
            train_payload = corpus_root / "train-order.bin"
            validation_payload = corpus_root / "validation-order.bin"
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
            output = root / "authority.json"
            argv_file = root / "launcher-argv.json"
            _write_json(
                argv_file,
                [
                    sys.executable,
                    str(launcher_source),
                    "--run-authority",
                    str(output),
                    "--execute",
                ],
            )

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

            tokenizer_identity = SimpleNamespace(
                manifest_path=tokenizer_manifest,
                manifest_sha256=tokenizer_digest,
                vocabulary_sha256="e" * 64,
                vocab_size=49_152,
            )
            corpus_qualification = _write_corpus_qualification(
                root=root,
                corpus_root=corpus_root,
                tokenizer_root=tokenizer_root,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                tokenizer_manifest_sha256=tokenizer_digest,
                vocabulary_sha256="e" * 64,
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
                    corpus_qualification=corpus_qualification,
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
