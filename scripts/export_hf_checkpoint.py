#!/usr/bin/env python3
"""Export a trusted native pretraining checkpoint as a Hugging Face Llama model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_export import (  # noqa: E402
    ExportError,
    export_native_checkpoint,
    load_model_config_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "PyTorch .pt checkpoints are pickle containers. Only pass a checkpoint "
            "created by this experiment or another trusted source. The output path "
            "must not already exist."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="native trainer .pt checkpoint, direct state-dict .pt, or .safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="pinned local tokenizer directory used for pretraining",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory to publish in Hugging Face from_pretrained format",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help=(
            "exact native ModelConfig JSON; required for direct state dicts and "
            "checked against embedded metadata for full trainer checkpoints"
        ),
    )
    parser.add_argument(
        "--expected-tokenizer-manifest-sha256",
        help=(
            "authenticated TOKENIZER_MANIFEST.json SHA-256; required with "
            "--expected-tokenizer-vocabulary-sha256 for direct state dicts and "
            "legacy checkpoint formats, optional as an additional check for v5"
        ),
    )
    parser.add_argument(
        "--expected-tokenizer-vocabulary-sha256",
        help=(
            "authenticated canonical token-to-ID vocabulary SHA-256; required "
            "with --expected-tokenizer-manifest-sha256 for direct state dicts "
            "and legacy checkpoint formats"
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="maximum safetensors shard size, e.g. 5GB or 500MB (default: 5GB)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model_config = (
            None
            if args.model_config is None
            else load_model_config_json(args.model_config)
        )
        manifest = export_native_checkpoint(
            args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_dir=args.output,
            model_config=model_config,
            expected_tokenizer_manifest_sha256=(
                args.expected_tokenizer_manifest_sha256
            ),
            expected_tokenizer_vocabulary_sha256=(
                args.expected_tokenizer_vocabulary_sha256
            ),
            max_shard_size=args.max_shard_size,
        )
    except (ExportError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
