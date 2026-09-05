#!/usr/bin/env python3
"""Refresh exactly three authenticated metadata files in an existing HF model repo.

This recovery path is for a repository that has already been renamed.  It
validates a fresh release staging tree, authenticates every unchanged remote
file, verifies safetensors only through Hub LFS SHA-256 metadata, and commits
only the model card, release manifest, and manifest sidecar.  It never downloads
or submits model weights and never changes repository visibility.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import (  # noqa: E402
    MODEL_CARD_NAME,
    ReleaseError,
    validate_release_package,
)
from scripts.rename_hf_release import (  # noqa: E402
    MODEL_REPO_TYPE,
    _COMMIT_SHA,
    _METADATA_COMMIT_PATHS,
    _token,
    _value,
    _verify_remote_against_staging,
    _visibility_info,
)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.confirm_repo_id != args.repo_id:
        raise ReleaseError("--confirm-repo-id must exactly repeat --repo-id")
    if args.confirm_visibility != args.visibility:
        raise ReleaseError("--confirm-visibility must exactly repeat --visibility")
    if args.repo_id.count("/") != 1 or any(not part for part in args.repo_id.split("/")):
        raise ReleaseError("--repo-id must be exactly NAMESPACE/NAME")
    if _COMMIT_SHA.fullmatch(args.expected_parent_sha) is None:
        raise ReleaseError("--expected-parent-sha must be an exact lowercase 40-character SHA")


def _validate_model_card(
    release_dir: Path,
    *,
    repo_id: str,
    expected_model_name: str,
) -> None:
    card = (release_dir / MODEL_CARD_NAME).read_text(encoding="utf-8")
    if repo_id not in card:
        raise ReleaseError("model card does not contain the confirmed repository ID")
    headings = re.findall(r"(?m)^# ([^#\n].*)$", card)
    if expected_model_name not in headings:
        raise ReleaseError(
            f"model card has no exact H1 heading for {expected_model_name!r}"
        )


def refresh_release_metadata(args: argparse.Namespace) -> Mapping[str, Any]:
    _validate_arguments(args)
    release_dir = args.release_dir.expanduser().resolve(strict=True)
    manifest = validate_release_package(release_dir, require_public_ready=False)
    _validate_model_card(
        release_dir,
        repo_id=args.repo_id,
        expected_model_name=args.expected_model_name,
    )
    token = _token(args.token_env)

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise ReleaseError("Install requirements-release.txt before refreshing metadata") from exc

    api = HfApi(token=token)
    if not api.repo_exists(
        repo_id=args.repo_id,
        repo_type=MODEL_REPO_TYPE,
        token=token,
    ):
        raise ReleaseError(f"repository does not exist or is inaccessible: {args.repo_id}")
    before = _visibility_info(
        api,
        repo_id=args.repo_id,
        visibility=args.visibility,
    )
    before_sha = _value(before, "sha")
    if before_sha != args.expected_parent_sha:
        raise ReleaseError(
            f"repository HEAD changed: expected {args.expected_parent_sha}, found {before_sha}"
        )

    before_weights = _verify_remote_against_staging(
        api,
        repo_id=args.repo_id,
        revision=before_sha,
        release_dir=release_dir,
        manifest=manifest,
        token=token,
        verify_metadata_commit=False,
    )

    operation_paths = tuple(_METADATA_COMMIT_PATHS)
    if set(operation_paths) != {
        "README.md",
        "HF_RELEASE_MANIFEST.json",
        "HF_RELEASE_MANIFEST.sha256",
    } or any(path.endswith(".safetensors") for path in operation_paths):
        raise ReleaseError("internal error: unsafe metadata commit operation list")
    operations = [
        CommitOperationAdd(
            path_in_repo=filename,
            path_or_fileobj=release_dir / filename,
        )
        for filename in operation_paths
    ]
    commit = api.create_commit(
        repo_id=args.repo_id,
        operations=operations,
        commit_message=args.commit_message,
        repo_type=MODEL_REPO_TYPE,
        parent_commit=before_sha,
        token=token,
    )
    new_sha = _value(commit, "oid")
    if not isinstance(new_sha, str) or _COMMIT_SHA.fullmatch(new_sha) is None:
        raise ReleaseError("Hub metadata commit returned no immutable commit SHA")

    after = _visibility_info(
        api,
        repo_id=args.repo_id,
        visibility=args.visibility,
    )
    if _value(after, "sha") != new_sha:
        raise ReleaseError("repository HEAD does not equal the metadata commit SHA")
    after_weights = _verify_remote_against_staging(
        api,
        repo_id=args.repo_id,
        revision=new_sha,
        release_dir=release_dir,
        manifest=manifest,
        token=token,
        verify_metadata_commit=True,
    )
    if after_weights != before_weights:
        raise ReleaseError("weight metadata changed after the metadata-only commit")

    return {
        "refreshed": True,
        "repo_id": args.repo_id,
        "old_commit_sha": before_sha,
        "new_commit_sha": new_sha,
        "commit_url": _value(commit, "commit_url"),
        "visibility": args.visibility,
        "metadata_files_committed": list(operation_paths),
        "weight_files": [
            {"path": path, "bytes": size, "sha256": sha}
            for path, size, sha in after_weights
        ],
        "weight_files_downloaded": False,
        "weight_files_uploaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--confirm-repo-id", required=True)
    parser.add_argument("--visibility", choices=("private", "public"), required=True)
    parser.add_argument("--confirm-visibility", required=True)
    parser.add_argument("--expected-parent-sha", required=True)
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--commit-message",
        default="Refresh authenticated model release metadata",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = refresh_release_metadata(args)
    except (OSError, ReleaseError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
