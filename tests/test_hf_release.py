from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_export import (  # noqa: E402
    EXPORT_FORMAT,
    EXPORT_FORMAT_VERSION,
    EXPORT_MANIFEST_NAME,
    EXPORT_MANIFEST_SIDECAR_NAME,
    SOURCE_TOKENIZER_MANIFEST_NAME,
    TOKENIZER_EXPORT_FORMAT,
    TOKENIZER_MANIFEST_NAME,
)
from pretrain.hf_release import (  # noqa: E402
    GENERATION_CONFIG_NAME,
    HUB_ATTRIBUTES_NAME,
    MODEL_CARD_NAME,
    RELEASE_MANIFEST_NAME,
    RELEASE_PROVENANCE_NAME,
    ReleaseError,
    prepare_release_package,
    validate_release_package,
    verify_source_export,
)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def create_sealed_export(root: Path) -> Path:
    root.mkdir()
    write_json(
        root / "config.json",
        {
            "_name_or_path": "/root/private/native-export",
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "eos_token_id": 0,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "pad_token_id": None,
            "vocab_size": 49152,
        },
    )
    write_json(
        root / "native_config.json",
        {
            "vocab_size": 49152,
            "max_seq_len": 4096,
            "attention_backend": "auto",
        },
    )
    write_json(root / "tokenizer.json", {"fixture": True})
    write_json(
        root / "tokenizer_config.json",
        {
            "name_or_path": "/workspace/private/tokenizer",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "model_max_length": 4096,
        },
    )
    tokenizer_files = {
        name: {key: value for key, value in record(root / name).items() if key != "path"}
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    source_tokenizer = {
        "repo_id": "bigcode/starcoder2-tokenizer",
        "resolved_revision": "9cfe60e28fd01cc1391ecd2146a34cda7534efeb",
        "files": {"tokenizer.json": tokenizer_files["tokenizer.json"]},
    }
    write_json(root / SOURCE_TOKENIZER_MANIFEST_NAME, source_tokenizer)
    write_json(
        root / TOKENIZER_MANIFEST_NAME,
        {
            "format": TOKENIZER_EXPORT_FORMAT,
            "manifest_version": 1,
            "files": tokenizer_files,
        },
    )
    first = root / "model-00001-of-00002.safetensors"
    second = root / "model-00002-of-00002.safetensors"
    first.write_bytes(b"fixture shard one")
    second.write_bytes(b"fixture shard two")
    write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": first.stat().st_size + second.stat().st_size},
            "weight_map": {
                "model.embed_tokens.weight": first.name,
                "lm_head.weight": second.name,
            },
        },
    )
    weight_records = []
    for path, tensors in (
        (first, ["model.embed_tokens.weight"]),
        (second, ["lm_head.weight"]),
    ):
        weight_records.append({**record(path), "tensors": tensors})
    nonweights = [
        record(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path not in (first, second)
    ]
    manifest = {
        "format": EXPORT_FORMAT,
        "manifest_version": EXPORT_FORMAT_VERSION,
        "architecture": "LlamaForCausalLM",
        "dtype": "float32",
        "source": {
            "kind": "checkpoint_model_payload",
            "path": "/workspace/private/checkpoints/last.pt",
            "sha256": "a" * 64,
        },
        "tokenizer": {"vocab_sha256": "b" * 64},
        "weights": {
            "format": "safetensors",
            "is_sharded": True,
            "index": "model.safetensors.index.json",
            "files": weight_records,
        },
        "files": nonweights,
    }
    write_json(root / EXPORT_MANIFEST_NAME, manifest)
    digest = hashlib.sha256((root / EXPORT_MANIFEST_NAME).read_bytes()).hexdigest()
    (root / EXPORT_MANIFEST_SIDECAR_NAME).write_text(
        f"{digest}  {EXPORT_MANIFEST_NAME}\n",
        encoding="ascii",
    )
    return root


class HuggingFaceReleaseTest(unittest.TestCase):
    def create_assets(self, root: Path, *, public: bool) -> tuple[Path, Path, Path]:
        card = root / "card.md"
        if public:
            card.write_text(
                "---\nlicense: custom\nlibrary_name: transformers\n---\n# Fixture\n",
                encoding="utf-8",
            )
        else:
            card.write_text(
                "---\nlibrary_name: transformers\n---\n"
                "<!-- RELEASE_BLOCKER: resolve -->\n# {{MODEL_NAME}}\n",
                encoding="utf-8",
            )
        generation = root / "generation.json"
        write_json(generation, {"bos_token_id": 0, "eos_token_id": 0})
        attributes = root / "attributes"
        attributes.write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
            encoding="ascii",
        )
        return card, generation, attributes

    def test_stage_is_sanitized_authenticated_and_keeps_source_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_sealed_export(root / "source")
            card, generation, attributes = self.create_assets(root, public=False)
            source_config = (source / "config.json").read_bytes()
            output = root / "release"
            manifest = prepare_release_package(
                source,
                output,
                model_card=card,
                generation_config=generation,
                hub_attributes=attributes,
                file_mode="hardlink",
            )

            self.assertEqual(manifest["publication_state"], "draft")
            validate_release_package(output, require_public_ready=False)
            with self.assertRaisesRegex(ReleaseError, "Public release is blocked"):
                validate_release_package(output, require_public_ready=True)
            self.assertEqual((source / "config.json").read_bytes(), source_config)
            self.assertEqual(
                (source / "model-00001-of-00002.safetensors").stat().st_ino,
                (output / "model-00001-of-00002.safetensors").stat().st_ino,
            )
            self.assertTrue((output / "model.safetensors.index.json").is_file())
            for internal in (
                EXPORT_MANIFEST_NAME,
                EXPORT_MANIFEST_SIDECAR_NAME,
                SOURCE_TOKENIZER_MANIFEST_NAME,
                TOKENIZER_MANIFEST_NAME,
            ):
                self.assertFalse((output / internal).exists())
            self.assertNotIn("_name_or_path", json.loads((output / "config.json").read_text()))
            self.assertNotIn(
                "name_or_path",
                json.loads((output / "tokenizer_config.json").read_text()),
            )
            self.assertNotIn(
                "/workspace/private",
                (output / RELEASE_PROVENANCE_NAME).read_text(encoding="utf-8"),
            )
            self.assertTrue((output / MODEL_CARD_NAME).is_file())
            self.assertTrue((output / GENERATION_CONFIG_NAME).is_file())
            self.assertTrue((output / HUB_ATTRIBUTES_NAME).is_file())
            self.assertTrue((output / RELEASE_MANIFEST_NAME).is_file())

    def test_completed_card_can_be_staged_public_ready_with_copied_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_sealed_export(root / "source")
            card, generation, attributes = self.create_assets(root, public=True)
            output = root / "release"
            manifest = prepare_release_package(
                source,
                output,
                model_card=card,
                generation_config=generation,
                hub_attributes=attributes,
                file_mode="copy",
            )
            self.assertEqual(manifest["publication_state"], "public-ready")
            validate_release_package(output, require_public_ready=True)
            self.assertNotEqual(
                (source / "model-00001-of-00002.safetensors").stat().st_ino,
                (output / "model-00001-of-00002.safetensors").stat().st_ino,
            )

    def test_tamper_and_generation_identity_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_sealed_export(root / "source")
            card, generation, attributes = self.create_assets(root, public=False)
            bad_generation = root / "bad-generation.json"
            write_json(bad_generation, {"bos_token_id": None, "eos_token_id": 0})
            with self.assertRaisesRegex(ReleaseError, "must exactly mirror BOS/EOS"):
                prepare_release_package(
                    source,
                    root / "bad-release",
                    model_card=card,
                    generation_config=bad_generation,
                    hub_attributes=attributes,
                )

            output = root / "release"
            prepare_release_package(
                source,
                output,
                model_card=card,
                generation_config=generation,
                hub_attributes=attributes,
            )
            with (output / MODEL_CARD_NAME).open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ReleaseError, "byte size does not match"):
                validate_release_package(output, require_public_ready=False)

    def test_source_manifest_or_tree_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = create_sealed_export(Path(temporary) / "source")
            (source / "unexpected.txt").write_text("not authenticated", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseError, "inventory differs"):
                verify_source_export(source)


if __name__ == "__main__":
    unittest.main()
