from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import curate_corpus as curate_corpus_module  # noqa: E402
import launch_fast_all_eligible_curation as launcher  # noqa: E402
from curate_corpus import (  # noqa: E402
    CURATION_STORAGE_PROJECTION_BASIS,
    FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
)
from curation_policy import FAST_CANONICAL_POLICY  # noqa: E402
from tests.test_curate_corpus import CorpusFixture  # noqa: E402


class FakeProcess:
    def __init__(self, stdout: bytes, returncode: int) -> None:
        self._stdout = stdout
        self.returncode = returncode
        self.signals: list[int] = []

    def communicate(self) -> tuple[bytes, None]:
        return self._stdout, None

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)


class FastCurationLauncherTest(unittest.TestCase):
    def fixture(self, base: Path) -> tuple[CorpusFixture, launcher.LaunchConfig]:
        fixture = CorpusFixture(base / "generation")
        (fixture.root / "logs").mkdir()
        local_parent = base / "pod-local"
        local_parent.mkdir()
        config = launcher.resolve_config(
            generation_root=fixture.root,
            local_work_root=local_parent / "curation-v2",
            quotas=fixture.quota_path,
            policy=FAST_CANONICAL_POLICY,
            benchmark_denylist=PROJECT_ROOT / "configs" / "mbpp_denylist.json",
            log_path=fixture.root / "logs" / "curation-v2.log",
            result_path=fixture.root / "logs" / "curation-v2.result.json",
        )
        return fixture, config

    @contextmanager
    def production_probes(self, config: launcher.LaunchConfig):
        def mount(path: Path) -> dict[str, object]:
            local = str(path).startswith(str(config.local_work_root.parent))
            return {
                "detector": "fixture",
                "mount_point": "/local" if local else "/workspace",
                "filesystem_type": "ext4" if local else "nfs4",
                "source": "/dev/nvme0n1" if local else "server:/volume",
                "device": "local-device" if local else "network-device",
                "options": [],
                "classification": "local" if local else "network",
            }

        def admission(**values: object) -> dict[str, object]:
            documents = int(values["expected_documents"])
            per_document = int(values["projected_bytes_per_document"])
            numerator = int(values["safety_numerator"])
            denominator = int(values["safety_denominator"])
            projected = (
                documents * per_document * numerator + denominator - 1
            ) // denominator
            reclaimable = int(values["reclaimable_existing_bytes"])
            remaining = max(0, projected - reclaimable)
            required = (
                remaining
                + int(values["transaction_sidecar_bytes"])
                + int(values["minimum_free_bytes"])
            )
            return {
                "status": "pass",
                "filesystem_type": values["filesystem_type"],
                "ram_backed": False,
                "expected_documents": documents,
                "projected_bytes_per_document": per_document,
                "safety_numerator": numerator,
                "safety_denominator": denominator,
                "projection_basis": values["projection_basis"],
                "projected_database_bytes_with_safety": projected,
                "reclaimable_existing_bytes": reclaimable,
                "remaining_database_bytes_with_safety": remaining,
                "transaction_sidecar_bytes": values["transaction_sidecar_bytes"],
                "minimum_free_bytes": values["minimum_free_bytes"],
                "required_free_bytes": required,
                "observed_free_bytes": required + 1,
                "filesystem_total_bytes": required * 2,
                "cgroup_memory": {},
            }

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(launcher, "detect_output_mount", side_effect=mount)
            )
            stack.enter_context(
                mock.patch.object(
                    curate_corpus_module, "detect_output_mount", side_effect=mount
                )
            )
            stack.enter_context(
                mock.patch.object(launcher, "storage_admission", side_effect=admission)
            )
            yield

    def make_success_result(self, config: launcher.LaunchConfig) -> bytes:
        snapshot = (
            config.output
            / ".work"
            / "sqlite-snapshots-v1"
            / "snapshot-000000000001"
        )
        snapshot.mkdir(parents=True)
        database = snapshot / "curation.sqlite3"
        checkpoint = snapshot / "CHECKPOINT.json"
        database.write_bytes(b"fixture-sqlite")
        checkpoint.write_text('{"fixture":true}\n', encoding="utf-8")
        database_descriptor = {
            "path": str(database.resolve()),
            "bytes": database.stat().st_size,
            "sha256": launcher.file_sha256(database),
        }
        checkpoint_descriptor = {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": launcher.file_sha256(checkpoint),
        }
        identity = {"fixture": True}
        manifest = {
            "generation": 1,
            "identity": identity,
            "identity_sha256": launcher.canonical_sha256(identity),
            "database": database_descriptor,
            "authority_artifacts": {"CHECKPOINT.json": checkpoint_descriptor},
        }
        manifest_path = snapshot / "manifest.json"
        manifest_raw = (
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        manifest_path.write_bytes(manifest_raw)
        manifest_sha = launcher._sha256_bytes(manifest_raw)
        (snapshot / "manifest.json.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="ascii"
        )
        result = {
            "complete": True,
            "phase": "canonicalized",
            "ready_for_all_eligible_publication": True,
            "execution_profile": FAST_ALL_ELIGIBLE_HANDOFF_PROFILE,
            "authority": {
                "selected_documents": 0,
                "decision_archives": 0,
                "quota_subphases": 0,
                "fuzzy_near_map_rows": 0,
            },
            "source_snapshot": {
                "generation": 1,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha,
                "database": database_descriptor,
                "checkpoint": checkpoint_descriptor,
            },
        }
        return json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    def test_preflight_freezes_profile_command_and_storage_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, config = self.fixture(Path(temporary))
            with self.production_probes(config):
                report = launcher.preflight_only(config)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["resume_state"], "fresh")
            self.assertEqual(
                report["storage_admission"]["projected_bytes_per_document"],
                1_322,
            )
            self.assertEqual(
                report["storage_admission"]["projection_basis"],
                CURATION_STORAGE_PROJECTION_BASIS,
            )
            argv = report["child_argv"]
            self.assertIn("--fast-all-eligible-handoff", argv)
            self.assertIn("--defer-raw-archive-integrity-until-finalize", argv)
            self.assertEqual(argv[argv.index("--batch-size") + 1], "100000")
            self.assertFalse(config.authority_path.exists())
            self.assertFalse(config.output.exists())

    def test_incomplete_collection_fails_before_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, config = self.fixture(Path(temporary))
            (fixture.root / "state" / "COLLECTION_COMPLETE.json").unlink()
            with self.production_probes(config), self.assertRaisesRegex(
                launcher.FastCurationLaunchError,
                "Collection/preprocess authority",
            ):
                launcher.preflight_only(config)
            self.assertFalse(config.authority_path.exists())

    def test_unowned_existing_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, config = self.fixture(Path(temporary))
            config.output.mkdir(parents=True)
            with self.production_probes(config), self.assertRaisesRegex(
                launcher.FastCurationLaunchError,
                "output exists without immutable launcher authority",
            ):
                launcher.preflight_only(config)

    def test_nonblocking_launcher_lock_rejects_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, config = self.fixture(Path(temporary))
            with launcher.exclusive_lock(config.lock_path, label="test owner"):
                with self.production_probes(config), self.assertRaisesRegex(
                    launcher.FastCurationLaunchError, "Another process holds"
                ):
                    launcher.preflight_only(config)

    def test_child_failure_preserves_exit_code_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, config = self.fixture(Path(temporary))

            def factory(*_args: object, **_kwargs: object) -> FakeProcess:
                return FakeProcess(b'{"failed":true}\n', 7)

            with self.production_probes(config):
                code, report = launcher.launch(config, popen_factory=factory)
            self.assertEqual(code, 7)
            self.assertEqual(report["child_returncode"], 7)
            self.assertTrue(config.authority_path.is_file())
            self.assertFalse(config.result_path.exists())
            log = config.log_path.read_text(encoding="utf-8")
            self.assertIn('"child_returncode":7', log)
            self.assertIn('{"failed":true}', log)

    def test_success_publishes_receipt_and_identical_rerun_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, config = self.fixture(Path(temporary))
            calls = 0

            def factory(*_args: object, **_kwargs: object) -> FakeProcess:
                nonlocal calls
                calls += 1
                return FakeProcess(self.make_success_result(config), 0)

            with self.production_probes(config):
                code, first = launcher.launch(config, popen_factory=factory)
                second_code, second = launcher.launch(
                    config,
                    popen_factory=lambda *_args, **_kwargs: self.fail(
                        "completed rerun spawned a child"
                    ),
                )
            self.assertEqual(code, 0)
            self.assertFalse(first["already_complete"])
            self.assertEqual(second_code, 0)
            self.assertTrue(second["already_complete"])
            self.assertEqual(calls, 1)
            self.assertTrue(config.result_path.is_file())
            self.assertTrue(config.completion_path.is_file())


if __name__ == "__main__":
    unittest.main()
