#!/usr/bin/env python3
"""Preflight, upload, or verify an authenticated Hugging Face release package.

No network or Hub mutation occurs unless the explicit ``upload`` or
``verify-remote`` subcommand is selected. Tokens are read only from an
environment variable and are never accepted as command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import ReleaseError, validate_release_package  # noqa: E402


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _token(variable: str) -> str:
    token = os.environ.get(variable)
    if not token:
        raise ReleaseError(f"Environment variable {variable} is not set")
    return token


def _load_smoke(root: Path, *, device: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ReleaseError(
            "Install requirements-release.txt and the qualified PyTorch wheel for load smoke"
        ) from exc
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if (
        config.model_type != "llama"
        or config.vocab_size != 49_152
        or config.max_position_embeddings != 4_096
        or tokenizer.eos_token_id != 0
        or tokenizer.bos_token_id != 0
        or tokenizer.pad_token_id is not None
        or len(tokenizer) != 49_152
    ):
        raise ReleaseError("Loaded config/tokenizer differs from the frozen release contract")
    model = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).eval()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    encoded = tokenizer(
        "def add(a, b):\n    return",
        return_tensors="pt",
        add_special_tokens=False,
    ).to(resolved_device)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits
        if logits.shape != (1, encoded.input_ids.shape[1], 49_152):
            raise ReleaseError(f"Unexpected smoke logits shape: {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise ReleaseError("Load smoke produced non-finite logits")
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    if generated.shape[1] not in (encoded.input_ids.shape[1], encoded.input_ids.shape[1] + 1):
        raise ReleaseError("Generation smoke returned an invalid sequence length")
    return {
        "architecture": type(model).__name__,
        "device": str(resolved_device),
        "logits_finite": True,
        "generated_tokens": int(generated.shape[1] - encoded.input_ids.shape[1]),
    }


def _add_common_release(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--load-smoke",
        action="store_true",
        help="also instantiate all weights and run one deterministic next-token smoke",
    )
    parser.add_argument("--device", default="cpu", help="device for --load-smoke")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="verify local bytes without network access")
    _add_common_release(preflight)
    preflight.add_argument(
        "--public",
        action="store_true",
        help="require resolved model-card placeholders and explicit license metadata",
    )

    upload = commands.add_parser("upload", help="create a new model repo and upload one commit")
    _add_common_release(upload)
    upload.add_argument("--repo-id", required=True, help="exact namespace/model repository ID")
    upload.add_argument("--visibility", choices=("private", "public"), default="private")
    upload.add_argument(
        "--confirm-upload",
        required=True,
        help="must exactly repeat --repo-id; prevents accidental upload",
    )
    upload.add_argument("--token-env", default="HF_TOKEN")
    upload.add_argument("--commit-message", default="Publish authenticated base-model release")

    verify = commands.add_parser(
        "verify-remote",
        help="download one immutable Hub commit into a new directory and verify every byte",
    )
    _add_common_release(verify)
    verify.add_argument("--repo-id", required=True)
    verify.add_argument("--revision", required=True, help="exact 40-character Hub commit SHA")
    verify.add_argument("--download-dir", type=Path, required=True)
    verify.add_argument("--token-env", default="HF_TOKEN")
    verify.add_argument("--public", action="store_true")
    return parser


def _preflight(args: argparse.Namespace, *, release_dir: Path | None = None) -> dict[str, Any]:
    root = args.release_dir if release_dir is None else release_dir
    require_public = bool(getattr(args, "public", False)) or getattr(args, "visibility", None) == "public"
    manifest = validate_release_package(root, require_public_ready=require_public)
    result: dict[str, Any] = {
        "verified": True,
        "publication_state": manifest["publication_state"],
        "source_native_export_manifest_sha256": manifest[
            "source_native_export_manifest_sha256"
        ],
    }
    if args.load_smoke:
        result["load_smoke"] = _load_smoke(root, device=args.device)
    return result


def _upload(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm_upload != args.repo_id:
        raise ReleaseError("--confirm-upload must exactly repeat --repo-id")
    local = _preflight(args)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ReleaseError("Install requirements-release.txt before upload") from exc
    api = HfApi(token=_token(args.token_env))
    created = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.visibility == "private",
        exist_ok=False,
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.release_dir),
        commit_message=args.commit_message,
    )
    return {
        **local,
        "uploaded": True,
        "repo_id": args.repo_id,
        "visibility": args.visibility,
        "repo_url": str(created),
        "commit_sha": getattr(commit, "oid", None),
        "commit_url": getattr(commit, "commit_url", None),
        "next_step": "run verify-remote at commit_sha before announcing or changing visibility",
    }


def _verify_remote(args: argparse.Namespace) -> dict[str, Any]:
    if _COMMIT_SHA.fullmatch(args.revision) is None:
        raise ReleaseError("--revision must be an immutable 40-character lowercase commit SHA")
    if args.download_dir.exists():
        raise ReleaseError(f"Refusing to reuse remote-verification directory: {args.download_dir}")
    local_manifest = validate_release_package(
        args.release_dir,
        require_public_ready=args.public,
    )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ReleaseError("Install requirements-release.txt before remote verification") from exc
    downloaded = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            token=_token(args.token_env),
            local_dir=args.download_dir,
        )
    )
    manifest = validate_release_package(
        downloaded,
        require_public_ready=args.public,
        allowed_extra={".cache"},
    )
    if manifest != local_manifest:
        raise ReleaseError("Remote release manifest differs from the local release candidate")
    result: dict[str, Any] = {
        "verified": True,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "publication_state": manifest["publication_state"],
    }
    if args.load_smoke:
        result["load_smoke"] = _load_smoke(downloaded, device=args.device)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = _preflight(args)
        elif args.command == "upload":
            result = _upload(args)
        else:
            result = _verify_remote(args)
    except (OSError, ReleaseError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
