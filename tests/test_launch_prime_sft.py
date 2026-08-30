from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import launch_prime_sft as launch
from posttrain.sft_data import FilterPolicy, SplitPolicy


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_synthetic_hf_export(root: Path) -> dict[str, object]:
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    weight = root / "model.safetensors"
    save_file(
        {"model.embed_tokens.weight": torch.zeros((2, 2))},
        str(weight),
        metadata={"format": "pt"},
    )
    config = root / "config.json"
    config.write_bytes(
        _json_bytes(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "vocab_size": 8,
                "max_position_embeddings": 4096,
            }
        )
    )
    tokenizer_json = root / "tokenizer.json"
    tokenizer_json.write_bytes(b'{"synthetic":true}\n')
    source_manifest_path = root / "SOURCE_TOKENIZER_MANIFEST.json"
    source_manifest_path.write_bytes(
        _json_bytes(
            {
                "files": {
                    "tokenizer.json": {
                        key: value
                        for key, value in _record(tokenizer_json).items()
                        if key != "path"
                    }
                }
            }
        )
    )
    source_manifest = {
        **_record(source_manifest_path),
        "declared_files_verified": 1,
    }
    vocab_sha = "a" * 64
    special_ids = {
        "bos_token_id": None,
        "eos_token_id": 0,
        "pad_token_id": None,
        "unk_token_id": 0,
    }
    tokenizer_manifest = {
        "manifest_version": 1,
        "format": "transformers-tokenizer-export",
        "source": {"manifest": source_manifest, "vocab_sha256": vocab_sha},
        "files": {"tokenizer.json": {key: value for key, value in _record(tokenizer_json).items() if key != "path"}},
        "validation": {
            "vocab_size": 8,
            "vocab_sha256": vocab_sha,
            "model_max_length": 4096,
            "special_token_ids": special_ids,
        },
    }
    tokenizer_manifest_path = root / "TOKENIZER_MANIFEST.json"
    tokenizer_manifest_path.write_bytes(_json_bytes(tokenizer_manifest))
    non_weights = [
        _record(path)
        for path in (config, source_manifest_path, tokenizer_json, tokenizer_manifest_path)
    ]
    tokenizer_record = {
        **_record(tokenizer_manifest_path),
        "format": "transformers-tokenizer-export",
        "manifest_version": 1,
    }
    manifest = {
        "manifest_version": 1,
        "format": "native-pytorch-to-hf-llama",
        "architecture": "LlamaForCausalLM",
        "native_model_config": {"vocab_size": 8, "max_seq_len": 4096},
        "tokenizer": {
            "export_manifest": tokenizer_record,
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_manifest["sha256"],
            "vocab_size": 8,
            "vocab_sha256": vocab_sha,
            "model_max_length": 4096,
            "special_token_ids": special_ids,
        },
        "weights": {
            "format": "safetensors",
            "is_sharded": False,
            "index": None,
            "total_tensor_bytes": 4,
            "files": [{**_record(weight), "tensors": ["model.embed_tokens.weight"]}],
        },
        "files": non_weights,
    }
    manifest_path = root / "NATIVE_EXPORT_MANIFEST.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "NATIVE_EXPORT_MANIFEST.sha256").write_text(
        f"{manifest_sha}  NATIVE_EXPORT_MANIFEST.json\n",
        encoding="ascii",
    )
    return manifest


def _dataset_identity(*, threshold: float = 0.0) -> dict[str, object]:
    return {
        "dataset_manifest_sha256": "d" * 64,
        "curation_identity_sha256": "c" * 64,
        "identity": {
            "policy_sha256": "p" * 64,
            "tokenizer": {
                "tokenizer_manifest_sha256": "1" * 64,
                "tokenizer_json_sha256": "2" * 64,
            },
        },
        "manifest": {
            "identity": {
                "max_sequence_length": 4096,
                "source_repo_id": "nvidia/OpenCodeInstruct",
                "source_revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
                "policy": {"filter": {"min_average_test_score": threshold}},
            }
        },
        "completion": {"train_rows": 10, "validation_rows": 2, "source_rows": 12},
        "data_bytes": 1024,
    }


def _model_identity() -> dict[str, object]:
    return {
        "manifest_sha256": "m" * 64,
        "vocab_size": 49152,
        "max_sequence_length": 4096,
        "source_tokenizer_manifest_sha256": "1" * 64,
        "source_tokenizer_json_sha256": "2" * 64,
        "source_tokenizer_vocabulary_sha256": "3" * 64,
    }


class HuggingFaceExportPreflightTests(unittest.TestCase):
    def test_exact_export_tree_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            _write_synthetic_hf_export(root)
            result = launch.validate_hf_export(root)
            self.assertEqual(result["vocab_size"], 8)
            self.assertEqual(result["max_sequence_length"], 4096)
            self.assertEqual(
                result["manifest_sha256"],
                hashlib.sha256((root / "NATIVE_EXPORT_MANIFEST.json").read_bytes()).hexdigest(),
            )

            with (root / "model.safetensors").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(launch.PrimeSFTPreflightError, "byte size changed"):
                launch.validate_hf_export(root)

    def test_rogue_model_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            _write_synthetic_hf_export(root)
            (root / "pytorch_model.bin").write_bytes(b"rogue")
            with self.assertRaisesRegex(launch.PrimeSFTPreflightError, "exact file tree"):
                launch.validate_hf_export(root)


class LaunchContractTests(unittest.TestCase):
    def test_import_restores_preexisting_scripts_namespace(self) -> None:
        fake_scripts = types.ModuleType("scripts")
        fake_child = types.ModuleType("scripts.keep")
        with mock.patch.dict(
            sys.modules,
            {"scripts": fake_scripts, "scripts.keep": fake_child},
            clear=False,
        ):
            module_name = "_launch_prime_sft_import_regression"
            spec = importlib.util.spec_from_file_location(module_name, launch.__file__)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)
            self.assertIs(sys.modules["scripts"], fake_scripts)
            self.assertIs(sys.modules["scripts.keep"], fake_child)

    def test_checked_in_toml_is_exact_six_rank_dry_run(self) -> None:
        contract = launch.load_launch_contract()
        config = launch.load_prime_config(contract)
        self.assertTrue(config.dry_run)
        self.assertEqual(config.payload["deployment"]["num_train_gpus"], 6)
        self.assertEqual(config.payload["model"]["impl"], "custom")
        self.assertEqual(config.payload["renderer"]["name"], "starcoder2-coding-chat-v1")
        self.assertEqual(config.sha256, contract.config_sha256)

    def test_real_run_rejects_provisional_quality_policy(self) -> None:
        config = launch.load_prime_config(launch.load_launch_contract())
        real = dataclasses.replace(config, dry_run=False)
        with self.assertRaisesRegex(
            launch.PrimeSFTPreflightError,
            "provisional min_average_test_score=0.0",
        ):
            launch.validate_training_approval(None, config=real, dataset=_dataset_identity())

    def test_dataset_and_model_tokenizer_identity_must_match(self) -> None:
        dataset = _dataset_identity()
        model = _model_identity()
        self.assertEqual(
            launch.validate_tokenizer_binding(dataset, model),
            {
                "tokenizer_manifest_sha256": "1" * 64,
                "tokenizer_json_sha256": "2" * 64,
            },
        )
        changed = {**model, "source_tokenizer_json_sha256": "4" * 64}
        with self.assertRaisesRegex(
            launch.PrimeSFTPreflightError,
            "different tokenizer.json bytes",
        ):
            launch.validate_tokenizer_binding(dataset, changed)

    def test_training_approval_binds_dataset_policy_and_quality_audit(self) -> None:
        config = dataclasses.replace(
            launch.load_prime_config(launch.load_launch_contract()), dry_run=False
        )
        dataset = _dataset_identity(threshold=0.5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "quality-audit-v2.json"
            audit.write_bytes(
                _json_bytes(
                    {
                        "format_version": 1,
                        "kind": "opencodeinstruct_quality_audit",
                        "source_repo_id": "nvidia/OpenCodeInstruct",
                        "source_revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
                        "score_field": "average_test_score",
                        "source_rows": 12,
                        "measurement": {"distribution_measured": True},
                        "created_at_utc": "2026-08-30T00:00:00+00:00",
                    }
                )
            )
            body = {
                "format_version": 1,
                "kind": "prime_sft_training_approval",
                "status": "approved",
                "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
                "curation_identity_sha256": dataset["curation_identity_sha256"],
                "policy_sha256": dataset["identity"]["policy_sha256"],
                "quality_audit": _record(audit),
                "decision": {"min_average_test_score": 0.5, "accepted_train_rows": 10},
                "approved_at_utc": "2026-08-30T00:00:00+00:00",
            }
            approval = {
                **body,
                "approval_sha256": hashlib.sha256(_json_bytes(body)).hexdigest(),
            }
            approval_path = root / "TRAINING_APPROVAL.json"
            approval_path.write_bytes(_json_bytes(approval))
            validated = launch.validate_training_approval(
                approval_path,
                config=config,
                dataset=dataset,
            )
            self.assertEqual(validated, approval)


class CachePrewarmTests(unittest.TestCase):
    def test_one_process_prewarms_both_splits_and_reuses_marker(self) -> None:
        config = launch.load_prime_config(launch.load_launch_contract())
        dataset = _dataset_identity()
        model = _model_identity()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=False):
            root = Path(temporary) / "cache"
            plan = launch.CachePlan(
                root=root,
                hf_home=root / "huggingface",
                datasets_cache=root / "huggingface" / "datasets",
                required_free_bytes=1,
                available_free_bytes=2,
                prewarmed=False,
            )
            calls: list[str] = []

            class FakeDataset:
                def __init__(self, split: str) -> None:
                    self._split = split
                    self._fingerprint = f"fp-{split}"

                def __len__(self) -> int:
                    return dataset["completion"][f"{self._split}_rows"]

            def fake_load_dataset(_name, _subset, *, split, cache_dir):
                calls.append(split)
                self.assertEqual(Path(cache_dir), plan.datasets_cache.resolve())
                cache = Path(cache_dir)
                cache.mkdir(parents=True, exist_ok=True)
                (cache / f"{split}.arrow").write_bytes(f"cached-{split}".encode())
                return FakeDataset(split)

            fake_datasets = types.ModuleType("datasets")
            fake_datasets.load_dataset = fake_load_dataset
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                    __len__=lambda self: model["vocab_size"],
                    model_max_length=model["max_sequence_length"],
                )
            )

            class FakeTokenizer:
                model_max_length = model["max_sequence_length"]

                def __len__(self) -> int:
                    return model["vocab_size"]

            fake_transformers.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: FakeTokenizer()
            with mock.patch.dict(
                sys.modules,
                {"datasets": fake_datasets, "transformers": fake_transformers},
            ):
                marker = launch.prewarm_hf_cache(
                    plan,
                    config=config,
                    dataset=dataset,
                    model=model,
                )
            self.assertEqual(calls, ["train", "validation"])
            self.assertTrue((root / launch.PREWARM_MARKER).is_file())
            self.assertEqual(marker["splits"]["train"]["rows"], 10)
            warm_plan = dataclasses.replace(plan, prewarmed=True)
            repeated = launch.prewarm_hf_cache(
                warm_plan,
                config=config,
                dataset=dataset,
                model=model,
            )
            self.assertEqual(repeated, marker)
            self.assertEqual(calls, ["train", "validation"])

            cached_train = plan.datasets_cache / "train.arrow"
            original_size = cached_train.stat().st_size
            cached_train.write_bytes(b"x" * original_size)
            self.assertEqual(cached_train.stat().st_size, original_size)
            with self.assertRaisesRegex(
                launch.PrimeSFTPreflightError,
                "cache inventory changed",
            ):
                launch.validate_prewarm_marker(
                    root,
                    config=config,
                    dataset=dataset,
                    model=model,
                )

    def test_six_rank_launch_requires_prewarm_then_execs_with_explicit_cache_env(self) -> None:
        contract = launch.load_launch_contract()
        config = launch.load_prime_config(contract)
        dataset = _dataset_identity()
        model = _model_identity()
        runtime = {
            "prime_rl_checkout": "/opt/prime/prime-rl",
            "prime_rl_commit": contract.prime_rl_commit,
            "renderers_commit": contract.renderers_commit,
        }
        cold = launch.CachePlan(
            root=Path("/workspace/posttraining-cache/prime-sft-v1"),
            hf_home=Path("/workspace/posttraining-cache/prime-sft-v1/huggingface"),
            datasets_cache=Path("/workspace/posttraining-cache/prime-sft-v1/huggingface/datasets"),
            required_free_bytes=1,
            available_free_bytes=2,
            prewarmed=False,
        )
        common_patches = (
            mock.patch.object(launch, "validate_published_dataset", return_value=dataset),
            mock.patch.object(launch, "validate_hf_export", return_value=model),
            mock.patch.object(launch, "validate_prime_runtime", return_value=runtime),
        )
        with common_patches[0], common_patches[1], common_patches[2], mock.patch.object(
            launch, "inspect_cache", return_value=cold
        ):
            with self.assertRaisesRegex(launch.PrimeSFTPreflightError, "one-process prewarm"):
                launch.run(mode="launch", prime_rl_checkout=Path("/opt/prime/prime-rl"))

        warm = dataclasses.replace(cold, prewarmed=True)
        observed: dict[str, object] = {}

        class Executed(Exception):
            pass

        def fake_exec(path: str, argv: list[str], env: dict[str, str]):
            observed.update(path=path, argv=argv, env=env)
            raise Executed

        with (
            mock.patch.object(launch, "validate_published_dataset", return_value=dataset),
            mock.patch.object(launch, "validate_hf_export", return_value=model),
            mock.patch.object(launch, "validate_prime_runtime", return_value=runtime),
            mock.patch.object(launch, "inspect_cache", return_value=warm),
            mock.patch.object(launch, "build_launch_command", return_value=["/usr/bin/true"]),
            self.assertRaises(Executed),
        ):
            launch.run(
                mode="launch",
                prime_rl_checkout=Path("/opt/prime/prime-rl"),
                execve=fake_exec,
            )
        environment = observed["env"]
        self.assertEqual(
            environment["HF_HOME"],
            "/workspace/posttraining-cache/prime-sft-v1/huggingface",
        )
        self.assertEqual(
            environment["HF_DATASETS_CACHE"],
            "/workspace/posttraining-cache/prime-sft-v1/huggingface/datasets",
        )
        self.assertEqual(environment["HF_DATASETS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
