from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.hf_release import ReleaseError  # noqa: E402
from scripts import refresh_hf_release_metadata  # noqa: E402
from tests.test_rename_hf_release import (  # noqa: E402
    FakeHub,
    FakeOperationAdd,
    MODEL_NAME,
    NEW_REPO,
    NEW_SHA,
    OLD_SHA,
    _make_release,
)


def _args(release: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "release_dir": release,
        "repo_id": NEW_REPO,
        "confirm_repo_id": NEW_REPO,
        "visibility": "public",
        "confirm_visibility": "public",
        "expected_parent_sha": OLD_SHA,
        "expected_model_name": MODEL_NAME,
        "token_env": "TEST_HF_TOKEN",
        "commit_message": "Refresh model metadata",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RefreshHuggingFaceMetadataTest(unittest.TestCase):
    def _fake(self, release: Path) -> FakeHub:
        fake = FakeHub(release)
        fake.repo_id = NEW_REPO
        fake.private = False
        return fake

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
            return dict(refresh_hf_release_metadata.refresh_release_metadata(args))

    def test_refresh_allows_only_three_metadata_differences_and_never_moves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = self._fake(release)
            for filename in (
                "README.md",
                "HF_RELEASE_MANIFEST.json",
                "HF_RELEASE_MANIFEST.sha256",
            ):
                self.assertNotEqual(fake.files[filename], (release / filename).read_bytes())

            result = self._run(fake, _args(release))

        expected_paths = [
            "README.md",
            "HF_RELEASE_MANIFEST.json",
            "HF_RELEASE_MANIFEST.sha256",
        ]
        self.assertEqual(fake.moves, [])
        self.assertEqual(len(fake.commits), 1)
        self.assertEqual(fake.commits[0]["paths"], expected_paths)
        self.assertEqual(fake.commits[0]["parent_commit"], OLD_SHA)
        self.assertFalse(any(path.endswith(".safetensors") for path in expected_paths))
        self.assertEqual(result["old_commit_sha"], OLD_SHA)
        self.assertEqual(result["new_commit_sha"], NEW_SHA)
        self.assertEqual(result["visibility"], "public")
        self.assertEqual(result["weight_files_downloaded"], False)
        self.assertEqual(result["weight_files_uploaded"], False)
        self.assertEqual(len(result["weight_files"]), 2)

    def test_confirmation_parent_and_visibility_fail_closed_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            checks = (
                ({"confirm_repo_id": "BosonicJustin/wrong"}, "confirm-repo-id"),
                ({"confirm_visibility": "private"}, "confirm-visibility"),
                ({"expected_parent_sha": "3" * 40}, "repository HEAD changed"),
                ({"expected_parent_sha": "not-a-sha"}, "40-character SHA"),
            )
            for overrides, message in checks:
                fake = self._fake(release)
                with self.subTest(message=message), self.assertRaisesRegex(
                    ReleaseError, message
                ):
                    self._run(fake, _args(release, **overrides))
                self.assertEqual(fake.commits, [])

            fake = self._fake(release)
            fake.private = True
            with self.assertRaisesRegex(ReleaseError, "visibility differs"):
                self._run(fake, _args(release))
            self.assertEqual(fake.commits, [])

    def test_nonmetadata_or_weight_mismatch_blocks_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")
            fake = self._fake(release)
            fake.files["config.json"] = b"changed config"
            with self.assertRaisesRegex(ReleaseError, "Hub file differs from staging"):
                self._run(fake, _args(release))
            self.assertEqual(fake.commits, [])

            fake = self._fake(release)
            fake.weight_sha_overrides["model-00001-of-00002.safetensors"] = "d" * 64
            with self.assertRaisesRegex(ReleaseError, "weight metadata differs"):
                self._run(fake, _args(release))
            self.assertEqual(fake.commits, [])

    def test_post_commit_head_visibility_metadata_and_weights_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = _make_release(Path(temporary) / "release")

            fake = self._fake(release)
            fake.mutate_visibility_during_commit = True
            with self.assertRaisesRegex(ReleaseError, "visibility differs"):
                self._run(fake, _args(release))
            self.assertEqual(len(fake.commits), 1)

            fake = self._fake(release)
            original_commit = fake.create_commit

            def mutate_weight_after_commit(**kwargs: object) -> object:
                result = original_commit(**kwargs)
                fake.weight_sha_overrides[
                    "model-00001-of-00002.safetensors"
                ] = "e" * 64
                return result

            fake.create_commit = mutate_weight_after_commit  # type: ignore[method-assign]
            with self.assertRaisesRegex(ReleaseError, "weight metadata differs"):
                self._run(fake, _args(release))
            self.assertEqual(len(fake.commits), 1)

            fake = self._fake(release)
            original_commit = fake.create_commit

            def corrupt_metadata_after_commit(**kwargs: object) -> object:
                result = original_commit(**kwargs)
                fake.files["README.md"] += b"corrupt\n"
                return result

            fake.create_commit = corrupt_metadata_after_commit  # type: ignore[method-assign]
            with self.assertRaisesRegex(ReleaseError, "committed metadata differs"):
                self._run(fake, _args(release))
            self.assertEqual(len(fake.commits), 1)


if __name__ == "__main__":
    unittest.main()
