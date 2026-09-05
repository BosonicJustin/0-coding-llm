#!/usr/bin/env python3
"""Create a sanitized Hugging Face release candidate from a sealed HF export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import ReleaseError, prepare_release_package  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-card",
        type=Path,
        default=PROJECT_ROOT / "release/huggingface/README.template.md",
        help="completed model card or the checked-in draft template",
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=PROJECT_ROOT / "release/huggingface/generation_config.json",
    )
    parser.add_argument(
        "--hub-attributes",
        type=Path,
        default=PROJECT_ROOT / "release/huggingface/.gitattributes",
    )
    parser.add_argument(
        "--file-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help=(
            "hardlink avoids duplicating weight bytes and fails across filesystems; "
            "copy duplicates every public artifact explicitly"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = prepare_release_package(
            args.source_export,
            args.output,
            model_card=args.model_card,
            generation_config=args.generation_config,
            hub_attributes=args.hub_attributes,
            file_mode=args.file_mode,
        )
    except (OSError, ReleaseError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
