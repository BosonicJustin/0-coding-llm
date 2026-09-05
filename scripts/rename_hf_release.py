#!/usr/bin/env python3
"""Rename a private Hugging Face model and commit only authenticated card metadata.

The Hub rename is server-side.  This command never places a safetensors file in
an upload operation.  It verifies the existing immutable revision against a
freshly validated release directory, performs the rename, and then commits only
``README.md``, ``HF_RELEASE_MANIFEST.json``, and
``HF_RELEASE_MANIFEST.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import (  # noqa: E402
    MODEL_CARD_NAME,
    RELEASE_MANIFEST_NAME,
    RELEASE_MANIFEST_SIDECAR_NAME,
    ReleaseError,
    validate_release_package,
)


MODEL_REPO_TYPE = "model"
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_METADATA_COMMIT_PATHS = (
    MODEL_CARD_NAME,
    RELEASE_MANIFEST_NAME,
    RELEASE_MANIFEST_SIDECAR_NAME,
)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _token(variable: str) -> str:
    token = os.environ.get(variable)
    if token is None or not token.strip():
        raise ReleaseError(f"Environment variable {variable} is not set")
    if token != token.strip():
        raise ReleaseError(f"Environment variable {variable} has surrounding whitespace")
    return token


def _validate_repo_ids(args: argparse.Namespace) -> None:
    if args.confirm_from_repo_id != args.from_repo_id:
        raise ReleaseError("--confirm-from-repo-id must exactly repeat --from-repo-id")
    if args.confirm_to_repo_id != args.to_repo_id:
        raise ReleaseError("--confirm-to-repo-id must exactly repeat --to-repo-id")
    if args.from_repo_id == args.to_repo_id:
        raise ReleaseError("source and destination repository IDs must differ")
    for label, repo_id in (
        ("source", args.from_repo_id),
        ("destination", args.to_repo_id),
    ):
        if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
            raise ReleaseError(f"{label} repository ID must be exactly NAMESPACE/NAME")
    source_owner = args.from_repo_id.split("/", 1)[0]
    destination_owner = args.to_repo_id.split("/", 1)[0]
    if source_owner != destination_owner:
        raise ReleaseError("this command only permits a rename within one namespace")
    if _COMMIT_SHA.fullmatch(args.expected_parent_sha) is None:
        raise ReleaseError("--expected-parent-sha must be an exact lowercase 40-character SHA")


def _validate_model_card(
    release_dir: Path,
    *,
    from_repo_id: str,
    to_repo_id: str,
    expected_model_name: str,
) -> None:
    card = (release_dir / MODEL_CARD_NAME).read_text(encoding="utf-8")
    if from_repo_id in card:
        raise ReleaseError("model card still contains the source repository ID")
    if to_repo_id not in card:
        raise ReleaseError("model card does not contain the destination repository ID")
    headings = re.findall(r"(?m)^# ([^#\n].*)$", card)
    if expected_model_name not in headings:
        raise ReleaseError(
            f"model card has no exact H1 heading for {expected_model_name!r}"
        )


def _release_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ReleaseError("release manifest has no file records")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReleaseError("release manifest contains an invalid file record")
        path = record.get("path")
        if not isinstance(path, str) or not path or path in result:
            raise ReleaseError("release manifest contains an invalid or duplicate path")
        result[path] = record
    return result


def _remote_tree(api: Any, *, repo_id: str, revision: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type=MODEL_REPO_TYPE,
        revision=revision,
        recursive=True,
        expand=True,
    ):
        path = _value(item, "path")
        if not isinstance(path, str) or not path or path in result:
            raise ReleaseError("Hub returned an invalid or duplicate repository path")
        result[path] = item
    return result


def _download_sha256(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    filename: str,
    token: str,
    download_root: Path,
) -> tuple[int, str]:
    path = Path(
        api.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=MODEL_REPO_TYPE,
            revision=revision,
            token=token,
            local_dir=download_root,
            force_download=True,
        )
    )
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"downloaded Hub file is missing or unsafe: {filename}")
    return path.stat().st_size, _sha256_file(path)


def _verify_remote_against_staging(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    release_dir: Path,
    manifest: Mapping[str, Any],
    token: str,
    verify_metadata_commit: bool,
) -> tuple[tuple[str, int, str], ...]:
    records = _release_records(manifest)
    staged_paths = {
        path.name
        for path in release_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    tree = _remote_tree(api, repo_id=repo_id, revision=revision)
    if set(tree) != staged_paths:
        raise ReleaseError(
            "Hub file inventory differs from staging: "
            f"missing={sorted(staged_paths - set(tree))}, "
            f"extra={sorted(set(tree) - staged_paths)}"
        )

    weights: list[tuple[str, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="hf-rename-verify-") as temporary:
        download_root = Path(temporary)
        for filename, record in sorted(records.items()):
            if filename == MODEL_CARD_NAME:
                continue
            expected_size = record.get("bytes")
            expected_sha = record.get("sha256")
            if type(expected_size) is not int or not isinstance(expected_sha, str):
                raise ReleaseError(f"invalid staged manifest record: {filename}")
            if filename.endswith(".safetensors"):
                item = tree[filename]
                actual_size = _value(item, "size")
                lfs = _value(item, "lfs")
                actual_sha = _value(lfs, "sha256")
                lfs_size = _value(lfs, "size")
                if (
                    actual_size != expected_size
                    or lfs_size != expected_size
                    or actual_sha != expected_sha
                ):
                    raise ReleaseError(
                        f"Hub weight metadata differs from staging: {filename}"
                    )
                weights.append((filename, expected_size, expected_sha))
                continue
            actual_size, actual_sha = _download_sha256(
                api,
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                token=token,
                download_root=download_root,
            )
            if actual_size != expected_size or actual_sha != expected_sha:
                raise ReleaseError(f"Hub file differs from staging: {filename}")

        if verify_metadata_commit:
            for filename in _METADATA_COMMIT_PATHS:
                actual_size, actual_sha = _download_sha256(
                    api,
                    repo_id=repo_id,
                    revision=revision,
                    filename=filename,
                    token=token,
                    download_root=download_root,
                )
                staged = release_dir / filename
                if actual_size != staged.stat().st_size or actual_sha != _sha256_file(staged):
                    raise ReleaseError(f"committed metadata differs from staging: {filename}")

    if not weights:
        raise ReleaseError("release has no safetensors weight files")
    return tuple(weights)


def _private_info(api: Any, *, repo_id: str, revision: str | None = None) -> Any:
    info = api.model_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
    )
    if _value(info, "private") is not True:
        raise ReleaseError(f"repository must remain private: {repo_id}")
    return info


def rename_release(args: argparse.Namespace) -> Mapping[str, Any]:
    _validate_repo_ids(args)
    release_dir = args.release_dir.expanduser().resolve(strict=True)
    manifest = validate_release_package(release_dir, require_public_ready=False)
    _validate_model_card(
        release_dir,
        from_repo_id=args.from_repo_id,
        to_repo_id=args.to_repo_id,
        expected_model_name=args.expected_model_name,
    )
    token = _token(args.token_env)

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise ReleaseError("Install requirements-release.txt before renaming") from exc

    api = HfApi(token=token)
    before = _private_info(api, repo_id=args.from_repo_id)
    before_sha = _value(before, "sha")
    if before_sha != args.expected_parent_sha:
        raise ReleaseError(
            f"source HEAD changed: expected {args.expected_parent_sha}, found {before_sha}"
        )
    if api.repo_exists(
        repo_id=args.to_repo_id,
        repo_type=MODEL_REPO_TYPE,
        token=token,
    ):
        raise ReleaseError(f"destination repository already exists: {args.to_repo_id}")

    before_weights = _verify_remote_against_staging(
        api,
        repo_id=args.from_repo_id,
        revision=before_sha,
        release_dir=release_dir,
        manifest=manifest,
        token=token,
        verify_metadata_commit=False,
    )

    api.move_repo(
        from_id=args.from_repo_id,
        to_id=args.to_repo_id,
        repo_type=MODEL_REPO_TYPE,
        token=token,
    )
    moved = _private_info(api, repo_id=args.to_repo_id)
    moved_sha = _value(moved, "sha")
    if moved_sha != before_sha:
        raise ReleaseError(
            f"server-side rename changed HEAD: before={before_sha}, after={moved_sha}"
        )
    moved_weights = _verify_remote_against_staging(
        api,
        repo_id=args.to_repo_id,
        revision=moved_sha,
        release_dir=release_dir,
        manifest=manifest,
        token=token,
        verify_metadata_commit=False,
    )
    if moved_weights != before_weights:
        raise ReleaseError("weight metadata changed during server-side rename")

    operation_paths = tuple(_METADATA_COMMIT_PATHS)
    if set(operation_paths) != set(_METADATA_COMMIT_PATHS) or any(
        path.endswith(".safetensors") for path in operation_paths
    ):
        raise ReleaseError("internal error: unsafe metadata commit operation list")
    operations = [
        CommitOperationAdd(
            path_in_repo=filename,
            path_or_fileobj=release_dir / filename,
        )
        for filename in operation_paths
    ]
    commit = api.create_commit(
        repo_id=args.to_repo_id,
        operations=operations,
        commit_message=args.commit_message,
        repo_type=MODEL_REPO_TYPE,
        parent_commit=moved_sha,
        token=token,
    )
    new_sha = _value(commit, "oid")
    if not isinstance(new_sha, str) or _COMMIT_SHA.fullmatch(new_sha) is None:
        raise ReleaseError("Hub metadata commit returned no immutable commit SHA")

    after = _private_info(api, repo_id=args.to_repo_id)
    if _value(after, "sha") != new_sha:
        raise ReleaseError("destination HEAD does not equal the metadata commit SHA")
    after_weights = _verify_remote_against_staging(
        api,
        repo_id=args.to_repo_id,
        revision=new_sha,
        release_dir=release_dir,
        manifest=manifest,
        token=token,
        verify_metadata_commit=True,
    )
    if after_weights != before_weights:
        raise ReleaseError("weight metadata changed after the metadata-only commit")

    return {
        "renamed": True,
        "from_repo_id": args.from_repo_id,
        "to_repo_id": args.to_repo_id,
        "old_commit_sha": before_sha,
        "new_commit_sha": new_sha,
        "commit_url": _value(commit, "commit_url"),
        "visibility": "private",
        "metadata_files_committed": list(operation_paths),
        "weight_files": [
            {"path": path, "bytes": size, "sha256": sha}
            for path, size, sha in after_weights
        ],
        "weight_files_uploaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--from-repo-id", required=True)
    parser.add_argument("--to-repo-id", required=True)
    parser.add_argument("--confirm-from-repo-id", required=True)
    parser.add_argument("--confirm-to-repo-id", required=True)
    parser.add_argument("--expected-parent-sha", required=True)
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--commit-message",
        default="Rename model and refresh authenticated release metadata",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = rename_release(args)
    except (OSError, ReleaseError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
