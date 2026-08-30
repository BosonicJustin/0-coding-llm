from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import CausalLM, ModelConfig
import pretrain.hf_export as hf_export
from pretrain.hf_export import (
    EXPORT_MANIFEST_NAME,
    EXPORT_MANIFEST_SIDECAR_NAME,
    SOURCE_TOKENIZER_MANIFEST_NAME,
    TOKENIZER_EXPORT_FORMAT,
    TOKENIZER_MANIFEST_NAME,
    ExportError,
    export_native_checkpoint,
    export_native_state_dict,
    load_native_checkpoint,
    map_native_state_dict,
    native_to_hf_key_map,
    validate_native_state_dict,
)
from pretrain.tokenizer_identity import verify_tokenizer_identity, vocabulary_sha256


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        dim=16,
        hidden_dim=40,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        norm_eps=1e-5,
        rope_theta=10_000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_backend="sdpa",
        loss_chunk_size=4,
        activation_checkpointing=False,
    )


def write_tokenizer(
    path: Path, *, vocab_size: int, token_prefix: str = "token"
) -> dict[str, int]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover - handled by caller skip
        raise unittest.SkipTest(f"tokenizer dependencies are unavailable: {exc}") from exc

    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        **{f"{token_prefix}_{index}": index for index in range(4, vocab_size)},
    }
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.model_max_length = 999
    path.mkdir(parents=True)
    tokenizer.save_pretrained(path)
    files = {
        artifact.name: {
            "bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        for artifact in sorted(path.iterdir(), key=lambda item: item.name)
        if artifact.is_file()
    }
    (path / TOKENIZER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "repo_id": "fixture/starcoder2-tokenizer",
                "resolved_revision": "f" * 40,
                "files": files,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return vocabulary


class HuggingFaceExportValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(765)
        self.config = tiny_config()
        self.model = CausalLM(self.config, dtype=torch.float32).eval()

    def test_key_map_is_complete_and_exact(self) -> None:
        mapping = native_to_hf_key_map(self.config)
        self.assertEqual(set(mapping), set(self.model.state_dict()))
        self.assertEqual(len(mapping), 1 + 9 * self.config.n_layers + 2)
        self.assertEqual(mapping["tok_embeddings.weight"], "model.embed_tokens.weight")
        self.assertEqual(
            mapping["layers.1.attention.k_proj.weight"],
            "model.layers.1.self_attn.k_proj.weight",
        )
        self.assertEqual(
            mapping["layers.0.feed_forward.down_proj.weight"],
            "model.layers.0.mlp.down_proj.weight",
        )
        self.assertEqual(mapping["norm.weight"], "model.norm.weight")
        self.assertEqual(mapping["lm_head.weight"], "lm_head.weight")
        self.assertEqual(
            set(map_native_state_dict(self.model.state_dict(), self.config)),
            set(mapping.values()),
        )

    def test_vocabulary_identity_is_order_independent_but_mapping_sensitive(self) -> None:
        first = {"alpha": 0, "beta": 1, "gamma": 2}
        reordered = {"gamma": 2, "alpha": 0, "beta": 1}
        same_size_substitution = {"alpha": 0, "beta": 1, "delta": 2}
        self.assertEqual(vocabulary_sha256(first), vocabulary_sha256(reordered))
        self.assertNotEqual(
            vocabulary_sha256(first),
            vocabulary_sha256(same_size_substitution),
        )

    def test_state_validation_fails_closed_on_keys_shapes_dtypes_and_nonfinite(self) -> None:
        state = dict(self.model.state_dict())
        validate_native_state_dict(state, self.config)

        missing = dict(state)
        missing.pop("norm.weight")
        with self.assertRaisesRegex(ExportError, "missing keys"):
            validate_native_state_dict(missing, self.config)

        extra = dict(state)
        extra["unknown.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ExportError, "unexpected keys"):
            validate_native_state_dict(extra, self.config)

        wrong_shape = dict(state)
        wrong_shape["norm.weight"] = torch.zeros(self.config.dim + 1)
        with self.assertRaisesRegex(ExportError, "has shape"):
            validate_native_state_dict(wrong_shape, self.config)

        mixed_dtype = dict(state)
        mixed_dtype["norm.weight"] = mixed_dtype["norm.weight"].to(torch.float16)
        with self.assertRaisesRegex(ExportError, "share one dtype"):
            validate_native_state_dict(mixed_dtype, self.config)

        nonfinite = dict(state)
        nonfinite["norm.weight"] = nonfinite["norm.weight"].clone()
        nonfinite["norm.weight"][0] = float("nan")
        with self.assertRaisesRegex(ExportError, "NaN or infinity"):
            validate_native_state_dict(nonfinite, self.config)

        aliased = dict(state)
        aliased["layers.0.attention.o_proj.weight"] = aliased[
            "layers.0.attention.q_proj.weight"
        ]
        with self.assertRaisesRegex(ExportError, "unexpectedly share storage"):
            validate_native_state_dict(aliased, self.config)

    def test_full_payload_and_direct_state_dict_config_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = root / "direct.pt"
            torch.save(self.model.state_dict(), direct)
            with self.assertRaisesRegex(ExportError, "supply the exact native model config"):
                load_native_checkpoint(direct)
            with self.assertRaisesRegex(ExportError, "not tokenizer-bound"):
                load_native_checkpoint(direct, model_config=self.config)
            loaded_direct = load_native_checkpoint(
                direct,
                model_config=self.config,
                expected_tokenizer_manifest_sha256="a" * 64,
                expected_tokenizer_vocabulary_sha256="b" * 64,
            )
            self.assertEqual(loaded_direct.source_kind, "direct_state_dict")

            full = root / "full.pt"
            torch.save(
                {
                    "format": "native-pytorch-pretrain",
                    "format_version": 5,
                    "tokenizer_manifest_sha256": "a" * 64,
                    "tokenizer_vocabulary_sha256": "b" * 64,
                    "model_config": dataclasses.asdict(self.config),
                    "model": self.model.state_dict(),
                },
                full,
            )
            loaded_full = load_native_checkpoint(full)
            self.assertEqual(loaded_full.config, self.config)
            self.assertEqual(loaded_full.checkpoint_format_version, 5)
            self.assertEqual(loaded_full.tokenizer_manifest_sha256, "a" * 64)
            self.assertEqual(loaded_full.tokenizer_vocabulary_sha256, "b" * 64)

            mismatch = dataclasses.replace(self.config, max_seq_len=32)
            with self.assertRaisesRegex(ExportError, "disagrees"):
                load_native_checkpoint(full, model_config=mismatch)

            bad_version = root / "bad-version.pt"
            torch.save(
                {
                    "format": "native-pytorch-pretrain",
                    "format_version": 999,
                    "model_config": dataclasses.asdict(self.config),
                    "model": self.model.state_dict(),
                },
                bad_version,
            )
            with self.assertRaisesRegex(ExportError, "Unsupported.*version"):
                load_native_checkpoint(bad_version)

            legacy = root / "legacy-v4.pt"
            torch.save(
                {
                    "format": "native-pytorch-pretrain",
                    "format_version": 4,
                    "model_config": dataclasses.asdict(self.config),
                    "model": self.model.state_dict(),
                },
                legacy,
            )
            with self.assertRaisesRegex(ExportError, "not tokenizer-bound"):
                load_native_checkpoint(legacy)
            loaded_legacy = load_native_checkpoint(
                legacy,
                expected_tokenizer_manifest_sha256="a" * 64,
                expected_tokenizer_vocabulary_sha256="b" * 64,
            )
            self.assertEqual(loaded_legacy.checkpoint_format_version, 4)


class HuggingFaceExportIntegrationTest(unittest.TestCase):
    def test_safetensors_export_reloads_with_same_tokenizer_and_logits(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self.skipTest(f"transformers is unavailable: {exc}")

        torch.manual_seed(987)
        config = tiny_config()
        native = CausalLM(config, dtype=torch.float32).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer"
            vocabulary = write_tokenizer(tokenizer_path, vocab_size=config.vocab_size)
            tokenizer_identity = verify_tokenizer_identity(tokenizer_path)
            source_manifest_bytes = (tokenizer_path / TOKENIZER_MANIFEST_NAME).read_bytes()
            source_manifest = json.loads(source_manifest_bytes)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "format": "native-pytorch-pretrain",
                    "format_version": 5,
                    "tokenizer_manifest_sha256": tokenizer_identity.manifest_sha256,
                    "tokenizer_vocabulary_sha256": (
                        tokenizer_identity.vocabulary_sha256
                    ),
                    "model_config": dataclasses.asdict(config),
                    "model": native.state_dict(),
                },
                checkpoint,
            )
            output = root / "hf-model"
            synced_artifacts: list[str] = []
            fsync_regular_file = hf_export._fsync_regular_file

            def record_fsync(path: Path) -> None:
                synced_artifacts.append(path.name)
                fsync_regular_file(path)

            with mock.patch.object(
                hf_export,
                "_fsync_regular_file",
                side_effect=record_fsync,
            ):
                manifest = export_native_checkpoint(
                    checkpoint,
                    tokenizer_path=tokenizer_path,
                    output_dir=output,
                    max_shard_size="1MB",
                )

            self.assertTrue((output / "model.safetensors").is_file())
            self.assertEqual(
                set(synced_artifacts),
                {path.name for path in output.iterdir() if path.is_file()},
            )
            self.assertEqual(len(synced_artifacts), len(set(synced_artifacts)))
            self.assertFalse(any(output.glob("*.bin")))
            self.assertTrue((output / EXPORT_MANIFEST_NAME).is_file())
            self.assertEqual(
                (output / EXPORT_MANIFEST_SIDECAR_NAME).read_text(encoding="utf-8"),
                f"{hashlib.sha256((output / EXPORT_MANIFEST_NAME).read_bytes()).hexdigest()}  "
                f"{EXPORT_MANIFEST_NAME}\n",
            )
            self.assertEqual(manifest["format"], "native-pytorch-to-hf-llama")
            self.assertEqual(manifest["weights"]["format"], "safetensors")
            self.assertEqual(manifest["tokenizer"]["vocab_size"], config.vocab_size)
            self.assertEqual(
                manifest["tokenizer"]["vocab_sha256"],
                tokenizer_identity.vocabulary_sha256,
            )
            self.assertEqual(
                manifest["source"]["sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )

            preserved_source = output / SOURCE_TOKENIZER_MANIFEST_NAME
            self.assertEqual(preserved_source.read_bytes(), source_manifest_bytes)
            self.assertEqual(
                manifest["tokenizer"]["source_manifest"]["sha256"],
                hashlib.sha256(source_manifest_bytes).hexdigest(),
            )
            exported_manifest_path = output / TOKENIZER_MANIFEST_NAME
            exported_manifest = json.loads(exported_manifest_path.read_bytes())
            self.assertEqual(exported_manifest["format"], TOKENIZER_EXPORT_FORMAT)
            self.assertEqual(
                manifest["tokenizer"]["export_manifest"]["sha256"],
                hashlib.sha256(exported_manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                exported_manifest["source"]["manifest"]["path"],
                SOURCE_TOKENIZER_MANIFEST_NAME,
            )
            self.assertNotIn(TOKENIZER_MANIFEST_NAME, exported_manifest["files"])
            self.assertNotIn(SOURCE_TOKENIZER_MANIFEST_NAME, exported_manifest["files"])
            for name, record in exported_manifest["files"].items():
                artifact = output / name
                self.assertEqual(record["bytes"], artifact.stat().st_size)
                self.assertEqual(
                    record["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
                )
            self.assertNotEqual(
                source_manifest["files"]["tokenizer_config.json"]["sha256"],
                exported_manifest["files"]["tokenizer_config.json"]["sha256"],
            )

            exported_tokenizer = AutoTokenizer.from_pretrained(
                output, local_files_only=True, trust_remote_code=False
            )
            self.assertEqual(exported_tokenizer.get_vocab(), vocabulary)
            self.assertEqual(exported_tokenizer.model_max_length, config.max_seq_len)
            self.assertEqual(len(exported_tokenizer), config.vocab_size)

            hf_model = AutoModelForCausalLM.from_pretrained(
                output,
                local_files_only=True,
                trust_remote_code=False,
            ).eval()
            self.assertEqual(hf_model.config.model_type, "llama")
            self.assertEqual(hf_model.config.vocab_size, len(exported_tokenizer))
            input_ids = torch.tensor([[1, 7, 9, 3, 12, 4]], dtype=torch.int64)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            document_ids = torch.zeros_like(input_ids)
            with torch.no_grad():
                native_logits = native(input_ids, position_ids, document_ids).logits
                hf_logits = hf_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits
            torch.testing.assert_close(native_logits, hf_logits, rtol=2e-5, atol=2e-6)

            with self.assertRaisesRegex(ExportError, "Refusing to overwrite"):
                export_native_checkpoint(
                    checkpoint,
                    tokenizer_path=tokenizer_path,
                    output_dir=output,
                )

    def test_wrong_same_size_tokenizer_is_rejected_by_identity(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"transformers is unavailable: {exc}")

        config = tiny_config()
        native = CausalLM(config, dtype=torch.float32).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_path = root / "expected-tokenizer"
            substituted_path = root / "substituted-tokenizer"
            expected_vocab = write_tokenizer(
                expected_path,
                vocab_size=config.vocab_size,
                token_prefix="expected",
            )
            substituted_vocab = write_tokenizer(
                substituted_path,
                vocab_size=config.vocab_size,
                token_prefix="substituted",
            )
            expected_identity = verify_tokenizer_identity(expected_path)
            substituted_identity = verify_tokenizer_identity(substituted_path)
            self.assertEqual(len(expected_vocab), len(substituted_vocab))
            self.assertNotEqual(
                expected_identity.vocabulary_sha256,
                substituted_identity.vocabulary_sha256,
            )

            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "format": "native-pytorch-pretrain",
                    "format_version": 5,
                    "tokenizer_manifest_sha256": expected_identity.manifest_sha256,
                    "tokenizer_vocabulary_sha256": (
                        expected_identity.vocabulary_sha256
                    ),
                    "model_config": dataclasses.asdict(config),
                    "model": native.state_dict(),
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ExportError, "manifest SHA-256 mismatch"):
                export_native_checkpoint(
                    checkpoint,
                    tokenizer_path=substituted_path,
                    output_dir=root / "must-not-exist",
                )
            self.assertFalse((root / "must-not-exist").exists())

    def test_stale_source_tokenizer_manifest_is_rejected(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"transformers is unavailable: {exc}")

        torch.manual_seed(321)
        config = tiny_config()
        native = CausalLM(config, dtype=torch.float32).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer"
            write_tokenizer(tokenizer_path, vocab_size=config.vocab_size)
            tokenizer_identity = verify_tokenizer_identity(tokenizer_path)
            with (tokenizer_path / "tokenizer_config.json").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(ExportError, "manifest size mismatch"):
                export_native_state_dict(
                    native.state_dict(),
                    config,
                    tokenizer_path=tokenizer_path,
                    output_dir=root / "hf-model",
                    expected_tokenizer_manifest_sha256=(
                        tokenizer_identity.manifest_sha256
                    ),
                    expected_tokenizer_vocabulary_sha256=(
                        tokenizer_identity.vocabulary_sha256
                    ),
                )

    def test_small_shard_limit_writes_hf_index_and_reloads(self) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            self.skipTest(f"transformers is unavailable: {exc}")

        torch.manual_seed(654)
        config = tiny_config()
        native = CausalLM(config, dtype=torch.float32).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer"
            write_tokenizer(tokenizer_path, vocab_size=config.vocab_size)
            tokenizer_identity = verify_tokenizer_identity(tokenizer_path)
            checkpoint = root / "state.pt"
            torch.save(native.state_dict(), checkpoint)
            output = root / "sharded"
            manifest = export_native_checkpoint(
                checkpoint,
                tokenizer_path=tokenizer_path,
                output_dir=output,
                model_config=config,
                expected_tokenizer_manifest_sha256=(
                    tokenizer_identity.manifest_sha256
                ),
                expected_tokenizer_vocabulary_sha256=(
                    tokenizer_identity.vocabulary_sha256
                ),
                max_shard_size="2KB",
            )
            self.assertTrue(manifest["weights"]["is_sharded"])
            index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(index["weight_map"]),
                set(native_to_hf_key_map(config).values()),
            )
            reloaded = AutoModelForCausalLM.from_pretrained(
                output, local_files_only=True, trust_remote_code=False
            )
            self.assertEqual(
                set(reloaded.state_dict()),
                set(native_to_hf_key_map(config).values()),
            )


if __name__ == "__main__":
    unittest.main()
