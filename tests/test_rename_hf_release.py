from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import (  # noqa: E402
    RELEASE_MANIFEST_NAME,
    RELEASE_MANIFEST_SIDECAR_NAME,
    ReleaseError,
    validate_release_package,
)
from scripts import rename_hf_release  # noqa: E402


OLD_REPO = "BosonicJustin/transcendent-logic-1.3b-base"
NEW_REPO = "BosonicJustin/transcendent-logic-model"
OLD_SHA = "1" * 40
NEW_SHA = "2" * 40
MODEL_NAME = "Transcendent Logic Model"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _make_release(root: Path) -> Path:
    root.mkdir()
    files = {
        ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        "README.md": (
            "---\nlicense: other\nlibrary_name: transformers\n---\n"
            f"# {MODEL_NAME}\n\n"
            f"Load from `{NEW_REPO}`.\n"
        ).encode(),
        "config.json": _json_bytes({"model_type": "llama"}),
        "generation_config.json": _json_bytes({"bos_token_id": 0, "eos_token_id": 0}),
        "model-00001-of-00002.safetensors": b"first fixture weight shard",
        "model-00002-of-00002.safetensors": b"second fixture weight shard",
        "model.safetensors.index.json": _json_bytes(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        "RELEASE_PROVENANCE.json": _json_bytes({"format": "fixture"}),
        "tokenizer.json": _json_bytes({"fixture": True}),
    }
    for filename, payload in files.items():
        (root / filename).write_bytes(payload)
    records = [
        {
            "path": filename,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
        for filename, payload in sorted(files.items())
    ]
    manifest = {
        "format": "hugging-face-model-release",
        "format_version": 1,
        "publication_state": "public-ready",
        "publication_blockers": [],
        "source_native_export_manifest_sha256": "a" * 64,
        "files": records,
    }
    manifest_bytes = _json_bytes(manifest)
    (root / RELEASE_MANIFEST_NAME).write_bytes(manifest_bytes)
    (root / RELEASE_MANIFEST_SIDECAR_NAME).write_text(
        f"{_sha256(manifest_bytes)}  {RELEASE_MANIFEST_NAME}\n",
        encoding="ascii",
    )
    validate_release_package(root, require_public_ready=False)
    return root


class FakeOperationAdd:
    def __init__(self, path_in_repo: str, path_or_fileobj: str | Path | bytes) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeHub:
    def __init__(self, release: Path) -> None:
        self.repo_id = OLD_REPO
        self.sha = OLD_SHA
        self.private = True
        self.destination_exists = False
        self.moves: list[tuple[str, str]] = []
        self.commits: list[dict[str, object]] = []
        self.files = {
            path.name: path.read_bytes()
            for path in release.iterdir()
            if path.is_file()
        }
        old_card = self.files["README.md"].decode().replace(MODEL_NAME, "Old Model")
        old_card = old_card.replace(NEW_REPO, OLD_REPO)
        self.files["README.md"] = old_card.encode()
        self.files[RELEASE_MANIFEST_NAME] = b"old release manifest\n"
        self.files[RELEASE_MANIFEST_SIDECAR_NAME] = b"old manifest sidecar\n"
        self.weight_sha_overrides: dict[str, str] = {}
        self.mutate_weights_during_move = False
        self.mutate_visibility_during_move = False
        self.mutate_visibility_during_commit = False

    def model_info(
        self,
        repo_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> object:
        del files_metadata
        if repo_id != self.repo_id:
            raise AssertionError(f"unknown repo: {repo_id}")
        if revision is not None and revision != self.sha:
            raise AssertionError(f"unknown revision: {revision}")
        return types.SimpleNamespace(private=self.private, sha=self.sha)

    def repo_exists(self, *, repo_id: str, repo_type: str, token: str) -> bool:
        self._auth(repo_type, token)
        return self.destination_exists or repo_id == self.repo_id

    def list_repo_tree(self, **kwargs: object) -> list[object]:
        self._auth(str(kwargs["repo_type"]), None)
        if kwargs["repo_id"] != self.repo_id or kwargs["revision"] != self.sha:
            raise AssertionError("tree requested for unknown repository revision")
        result = []
        for filename, payload in sorted(self.files.items()):
            lfs = None
            if filename.endswith(".safetensors"):
                digest = self.weight_sha_overrides.get(filename, _sha256(payload))
                lfs = types.SimpleNamespace(size=len(payload), sha256=digest)
            result.append(
                types.SimpleNamespace(path=filename, size=len(payload), lfs=lfs)
            )
        return result

    def hf_hub_download(self, **kwargs: object) -> str:
        self._auth(str(kwargs["repo_type"]), str(kwargs["token"]))
        if kwargs["repo_id"] != self.repo_id or kwargs["revision"] != self.sha:
            raise AssertionError("download requested for unknown repository revision")
        filename = str(kwargs["filename"])
        if filename.endswith(".safetensors"):
            raise AssertionError("the rename command must never download weights")
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)

    def move_repo(self, **kwargs: object) -> None:
        self._auth(str(kwargs["repo_type"]), str(kwargs["token"]))
        if kwargs["from_id"] != self.repo_id or kwargs["to_id"] != NEW_REPO:
            raise AssertionError("unexpected move")
        self.moves.append((str(kwargs["from_id"]), str(kwargs["to_id"])))
        self.repo_id = str(kwargs["to_id"])
        if self.mutate_visibility_during_move:
            self.private = not self.private
        if self.mutate_weights_during_move:
            filename = "model-00001-of-00002.safetensors"
            self.weight_sha_overrides[filename] = "f" * 64

    def create_commit(self, **kwargs: object) -> object:
        self._auth(str(kwargs["repo_type"]), None)
        if kwargs["repo_id"] != self.repo_id:
            raise AssertionError("commit targeted an unknown repository")
        if kwargs["parent_commit"] != self.sha:
            raise AssertionError("commit did not pin current parent")
        operations = list(kwargs["operations"])
        paths = [operation.path_in_repo for operation in operations]
        self.commits.append({**kwargs, "paths": paths})
        for operation in operations:
            source = operation.path_or_fileobj
            self.files[operation.path_in_repo] = (
                source if isinstance(source, bytes) else Path(source).read_bytes()
            )
        self.sha = NEW_SHA
        if self.mutate_visibility_during_commit:
            self.private = not self.private
        return types.SimpleNamespace(
            oid=NEW_SHA,
            commit_url=f"https://huggingface.co/{self.repo_id}/commit/{NEW_SHA}",
        )

    @staticmethod
    def _auth(repo_type: str, token: str | None) -> None:
        if repo_type != "model":
            raise AssertionError("unexpected repo type")
        if token is not None and token != "secret-token":
            raise AssertionError("unexpected token")


def _args(release: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "release_dir": release,
        "from_repo_id": OLD_REPO,
        "to_repo_id": NEW_REPO,
        "confirm_from_repo_id": OLD_REPO,
        "confirm_to_repo_id": NEW_REPO,
        "visibility": "private",
        "confirm_visibility": "private",
        "expected_parent_sha": OLD_SHA,
        "expected_model_name": MODEL_NAME,
        "token_env": "TEST_HF_TOKEN",
        "commit_message": "Rename model and metadata",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RenameHuggingFaceReleaseTest(unittest.TestCase):
    def _run(self, fake: FakeHub, args: argparse.Namespace) -> dict[str, object]:
        module = types.SimpleNamespace(
            CommitOperationAdd=FakeOperationAdd,
            HfApi=lambda token: fake if token == "secret-token" else None,
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": module}), mock.patch.dict(
            os.environ,
            {"TEST_HF_TOKEN": "secret-token"},
            clear=False,
        ):
            return dict(rename_hf_release.rename_release(args))

    def test_rename_commits_only_three_metadata_files_and_preserves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = FakeHub(release)
            result = self._run(fake, _args(release))

        expected_paths = [
            "README.md",
            "HF_RELEASE_MANIFEST.json",
            "HF_RELEASE_MANIFEST.sha256",
        ]
        self.assertEqual(fake.moves, [(OLD_REPO, NEW_REPO)])
        self.assertEqual(len(fake.commits), 1)
        self.assertEqual(fake.commits[0]["paths"], expected_paths)
        self.assertFalse(any(path.endswith(".safetensors") for path in expected_paths))
        self.assertEqual(fake.commits[0]["parent_commit"], OLD_SHA)
        self.assertEqual(result["old_commit_sha"], OLD_SHA)
        self.assertEqual(result["new_commit_sha"], NEW_SHA)
        self.assertEqual(result["visibility"], "private")
        self.assertEqual(result["weight_files_uploaded"], False)
        self.assertEqual(len(result["weight_files"]), 2)

    def test_public_visibility_is_preserved_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = FakeHub(release)
            fake.private = False
            result = self._run(
                fake,
                _args(
                    release,
                    visibility="public",
                    confirm_visibility="public",
                ),
            )

        self.assertEqual(fake.moves, [(OLD_REPO, NEW_REPO)])
        self.assertEqual(len(fake.commits), 1)
        self.assertEqual(result["visibility"], "public")

    def test_confirmation_parent_visibility_and_destination_guards_fail_before_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            cases = (
                ({"confirm_to_repo_id": "BosonicJustin/wrong"}, "confirm-to-repo-id"),
                ({"confirm_visibility": "public"}, "confirm-visibility"),
                ({"expected_parent_sha": "3" * 40}, "source HEAD changed"),
                ({"expected_parent_sha": "not-a-sha"}, "40-character SHA"),
            )
            for overrides, message in cases:
                fake = FakeHub(release)
                with self.subTest(message=message), self.assertRaisesRegex(ReleaseError, message):
                    self._run(fake, _args(release, **overrides))
                self.assertEqual(fake.moves, [])

            fake = FakeHub(release)
            fake.private = False
            with self.assertRaisesRegex(ReleaseError, "visibility differs"):
                self._run(fake, _args(release))
            self.assertEqual(fake.moves, [])

            fake = FakeHub(release)
            fake.destination_exists = True
            with self.assertRaisesRegex(ReleaseError, "destination repository already exists"):
                self._run(fake, _args(release))
            self.assertEqual(fake.moves, [])

    def test_mismatched_remote_nonweight_or_weight_fails_before_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = FakeHub(release)
            fake.files["config.json"] = b"different config"
            with self.assertRaisesRegex(ReleaseError, "Hub file differs from staging"):
                self._run(fake, _args(release))
            self.assertEqual(fake.moves, [])

            fake = FakeHub(release)
            fake.weight_sha_overrides["model-00001-of-00002.safetensors"] = "e" * 64
            with self.assertRaisesRegex(ReleaseError, "weight metadata differs"):
                self._run(fake, _args(release))
            self.assertEqual(fake.moves, [])

    def test_weight_metadata_change_during_move_blocks_metadata_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = FakeHub(release)
            fake.mutate_weights_during_move = True
            with self.assertRaisesRegex(ReleaseError, "weight metadata differs"):
                self._run(fake, _args(release))
            self.assertEqual(fake.moves, [(OLD_REPO, NEW_REPO)])
            self.assertEqual(fake.commits, [])

    def test_visibility_is_rechecked_after_move_and_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")

            fake = FakeHub(release)
            fake.private = False
            fake.mutate_visibility_during_move = True
            with self.assertRaisesRegex(ReleaseError, "visibility differs"):
                self._run(
                    fake,
                    _args(
                        release,
                        visibility="public",
                        confirm_visibility="public",
                    ),
                )
            self.assertEqual(fake.commits, [])

            fake = FakeHub(release)
            fake.private = False
            fake.mutate_visibility_during_commit = True
            with self.assertRaisesRegex(ReleaseError, "visibility differs"):
                self._run(
                    fake,
                    _args(
                        release,
                        visibility="public",
                        confirm_visibility="public",
                    ),
                )
            self.assertEqual(len(fake.commits), 1)

    def test_model_card_must_name_destination_and_exact_model_title(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            card = release / "README.md"
            payload = card.read_text().replace(NEW_REPO, OLD_REPO)
            card.write_text(payload)
            with self.assertRaisesRegex(ReleaseError, "source repository ID"):
                rename_hf_release._validate_model_card(
                    release,
                    from_repo_id=OLD_REPO,
                    to_repo_id=NEW_REPO,
                    expected_model_name=MODEL_NAME,
                )

            card.write_text(
                payload.replace(OLD_REPO, NEW_REPO).replace(
                    f"# {MODEL_NAME}", "# Wrong Model"
                )
            )
            with self.assertRaisesRegex(ReleaseError, "no exact H1 heading"):
                rename_hf_release._validate_model_card(
                    release,
                    from_repo_id=OLD_REPO,
                    to_repo_id=NEW_REPO,
                    expected_model_name=MODEL_NAME,
                )


if __name__ == "__main__":
    unittest.main()
