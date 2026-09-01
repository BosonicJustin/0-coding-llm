from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pretrain.materialize import FORMAT as MATERIALIZATION_FORMAT
from pretrain.materialize import FORMAT_VERSION as MATERIALIZATION_FORMAT_VERSION
from pretrain.materialize import JOURNAL_NAME
from pretrain.materialize import MaterializationError
from pretrain.materialize import canonical_sha256
from pretrain.postcuration_orchestrator import (
    PostCurationConfig,
    PostCurationOrchestrationError,
    PostCurationOrchestrator,
    _existing_lock_is_held,
)
from scripts.launch_fast_all_eligible_curation import LaunchConfig
from tests import test_launch_fast_all_eligible_curation as launch_test
from scripts import orchestrate_postcuration as orchestrate_cli


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class Harness(PostCurationOrchestrator):
    def __init__(self, config: PostCurationConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        root = config.generation_root
        self.launch = LaunchConfig(
            generation_root=root,
            staging_root=root / "staging" / "preprocess",
            output=root / "curated" / "all-eligible-source-v2",
            local_work_root=root.parent / "local-curation",
            quotas=root / "quotas.json",
            policy=root / "policy.json",
            benchmark_denylist=root / "denylist.json",
            log_path=root / "curation.log",
            result_path=root / "curation.result.json",
            authority_path=root / "authority.json",
            completion_path=root / "completion.json",
            lock_path=root / "curation.lock",
            preprocess_lock_path=root / "preprocess.lock",
        )
        self.selection_state: tuple[dict[str, object], str] | None = (
            {"decision_shards": [{"archive": "raw/python/part-000000.tar.zst"}]},
            "a" * 64,
        )
        self.inventory_state: object | None = SimpleNamespace(
            entries=(object(), object()),
            manifest_sha256="b" * 64,
            selection_manifest_sha256="a" * 64,
        )
        self.cache_state = {
            "status": "complete",
            "archives_complete": 2,
            "archives_total": 2,
            "content_tokens_complete": 10,
            "content_tokens_total": 10,
            "publisher_lock_held": False,
        }
        self.materialization_states = [
            {"status": "ready", "phase": "not-started", "lock_held": False}
        ]
        self.materializer = SimpleNamespace(
            identity={
                "format": MATERIALIZATION_FORMAT,
                "format_version": MATERIALIZATION_FORMAT_VERSION,
                "selection_manifest_sha256": "a" * 64,
                "tokenizer_manifest_sha256": "b" * 64,
                "curation_policy_sha256": "c" * 64,
                "packing_configuration": {
                    "sequence_length": 4_096,
                    "rows_per_shard": 131_072,
                    "construction_seed": 1_234,
                    "expected_vocab_size": 49_152,
                    "expected_eos_token_id": 0,
                },
            },
            archives=(1, 2),
        )

    def _inspect_curation(self) -> dict[str, object]:
        return {
            "status": "complete",
            "launcher_identity_sha256": "c" * 64,
            "result_sha256": "d" * 64,
            "snapshot": {"generation": 1},
            "source_db": str(self.config.generation_root / "snapshot/curation.sqlite3"),
            "source_checkpoint": str(
                self.config.generation_root / "snapshot/CHECKPOINT.json"
            ),
            "source_snapshot_manifest": str(
                self.config.generation_root / "snapshot/manifest.json"
            ),
            "launch_config": self.launch,
        }

    def _inspect_selection(
        self, _curation: object
    ) -> tuple[dict[str, object], str] | None:
        return self.selection_state

    def _cache_status(self, selection: object) -> tuple[dict[str, object], list[object]]:
        return self.cache_state, []

    def _inspect_inventory(self, selection_sha256: str) -> object | None:
        return self.inventory_state

    def _build_materializer(self, curation: object) -> object:
        return self.materializer

    def _inspect_materialization(self, materializer: object) -> dict[str, object]:
        if len(self.materialization_states) > 1:
            return self.materialization_states.pop(0)
        return self.materialization_states[0]


class PostCurationOrchestratorTests(unittest.TestCase):
    def config(self, root: Path) -> PostCurationConfig:
        return PostCurationConfig.for_generation(root)

    def passing_supply(self) -> dict[str, object]:
        return {"status": "pass", "failures": [], "splits": []}

    def test_default_paths_and_geometry_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)
            self.assertEqual(config.selection_root, root / "curated/selection-v7")
            self.assertEqual(config.cache_root, root / "token-cache/raw-all-v1")
            orchestrator = Harness(config)
            report = orchestrator._base_report()
            self.assertFalse(report["guardrails"]["gpu_actions_allowed"])
            self.assertFalse(
                report["guardrails"]["geometry_dependent_orders_allowed"]
            )

    def test_empty_generation_is_read_only_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = list(root.iterdir())
            report = PostCurationOrchestrator(self.config(root)).run()
            self.assertEqual(report["status"], "waiting")
            self.assertEqual(report["next_stage"], "curation")
            self.assertEqual(list(root.iterdir()), before)

    def test_real_launcher_receipt_and_snapshot_are_accepted_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            helper = launch_test.FastCurationLauncherTest()
            fixture, launch_config = helper.fixture(Path(temporary))

            def factory(*_args: object, **_kwargs: object) -> launch_test.FakeProcess:
                return launch_test.FakeProcess(
                    helper.make_success_result(launch_config), 0
                )

            with helper.production_probes(launch_config):
                code, launch_report = launch_test.launcher.launch(
                    launch_config, popen_factory=factory
                )
            self.assertEqual(code, 0)
            self.assertTrue(launch_report["complete"])
            before = sorted(path.as_posix() for path in fixture.root.rglob("*"))
            observed = PostCurationOrchestrator(
                self.config(fixture.root)
            )._inspect_curation()
            after = sorted(path.as_posix() for path in fixture.root.rglob("*"))
            self.assertEqual(observed["status"], "complete")
            self.assertEqual(observed["snapshot"]["generation"], 1)
            self.assertEqual(before, after)

    def test_dry_run_plans_selection_without_calling_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            called: list[list[str]] = []
            harness = Harness(
                self.config(Path(temporary)),
                runner=lambda argv: called.append(list(argv)) or 0,
            )
            harness.selection_state = None
            report = harness.run()
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["next_stage"], "selection")
            self.assertEqual(called, [])
            self.assertFalse(report["actions"][0]["executed"])

    def test_execute_publishes_selection_then_supply_failure_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)), execute=True)
            harness.selection_state = None

            def runner(argv: object) -> int:
                harness.selection_state = (
                    {"decision_shards": [{"archive": "raw/python/part-0"}]},
                    "a" * 64,
                )
                return 0

            harness.runner = runner
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value={"status": "fail", "failures": [{"domain": "python"}]},
            ):
                report = harness.run()
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["next_stage"], "selection-supply-remediation")
            self.assertEqual(report["actions"][0]["stage"], "publish-selection-v7")

    def test_cache_in_progress_waits_without_inventory_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)), execute=True)
            harness.inventory_state = None
            harness.cache_state = {
                **harness.cache_state,
                "status": "waiting",
                "archives_complete": 1,
                "publisher_lock_held": True,
            }
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                report = harness.run()
            self.assertEqual(report["status"], "waiting")
            self.assertEqual(report["next_stage"], "raw-token-cache")
            self.assertEqual(report["actions"], [])

    def test_dry_run_stops_before_inventory_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            harness.inventory_state = None
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                report = harness.run()
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["next_stage"], "cache-inventory")
            command = report["actions"][0]["argv"]
            self.assertIn("publish_raw_token_cache_inventory.py", command[1])

    def test_execute_advances_inventory_and_packing_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stages: list[str] = []
            harness = Harness(self.config(Path(temporary)), execute=True)
            harness.inventory_state = None
            harness.materialization_states = [
                {"status": "ready", "phase": "not-started", "lock_held": False},
                {
                    "status": "complete",
                    "phase": "packed",
                    "completed_archives": 2,
                    "archive_count": 2,
                    "lock_held": False,
                },
            ]

            def runner(argv: list[str]) -> int:
                if "publish_raw_token_cache_inventory.py" in argv[1]:
                    stages.append("inventory")
                    harness.inventory_state = SimpleNamespace(
                        entries=(object(), object()),
                        manifest_sha256="b" * 64,
                        selection_manifest_sha256="a" * 64,
                    )
                elif "materialize_training_corpus.py" in argv[1]:
                    stages.append("packing")
                return 0

            harness.runner = runner
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                report = harness.run()
            self.assertEqual(stages, ["inventory", "packing"])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                [action["stage"] for action in report["actions"]],
                ["publish-cache-inventory", "materialize-stop-after-packing"],
            )

    def test_packing_command_never_contains_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            command = harness._materialization_command(harness._inspect_curation())
            self.assertIn("--stop-after-packing", command)
            self.assertNotIn("--global-microbatch-rows", command)
            self.assertNotIn("--gradient-accumulation-steps", command)

    def test_already_packed_is_terminal_and_does_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)), execute=True)
            harness.materialization_states = [
                {
                    "status": "complete",
                    "phase": "packed",
                    "completed_archives": 2,
                    "archive_count": 2,
                    "lock_held": False,
                }
            ]
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                report = harness.run()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["actions"], [])

    def test_nonzero_child_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(
                self.config(Path(temporary)), execute=True, runner=lambda argv: 17
            )
            harness.inventory_state = None
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "status 17"
                ):
                    harness.run()

    def test_runner_os_error_is_wrapped_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            def fail(_argv: object) -> int:
                raise OSError("exec boundary failed")

            harness = Harness(
                self.config(Path(temporary)), execute=True, runner=fail
            )
            harness.inventory_state = None
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError,
                    "runner failed: OSError: exec boundary failed",
                ):
                    harness.run()

    def test_runner_boolean_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(
                self.config(Path(temporary)), execute=True, runner=lambda _argv: True
            )
            harness.inventory_state = None
            with mock.patch(
                "pretrain.postcuration_orchestrator.qualify_selection_supply",
                return_value=self.passing_supply(),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "non-integer status"
                ):
                    harness.run()

    def test_cli_wraps_constructor_os_error_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with (
                mock.patch.object(
                    orchestrate_cli,
                    "PostCurationOrchestrator",
                    side_effect=OSError("constructor boundary failed"),
                ),
                redirect_stdout(output),
            ):
                code = orchestrate_cli.main(
                    ["--generation-root", temporary]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertEqual(
                payload["error_type"], "PostCurationOrchestrationError"
            )
            self.assertIn("constructor boundary failed", payload["error"])

    def test_existing_lock_probe_does_not_create_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "missing.lock"
            self.assertFalse(_existing_lock_is_held(lock, label="test lock"))
            self.assertFalse(lock.exists())

    def test_existing_lock_probe_detects_held_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "held.lock"
            lock.touch()
            with lock.open("r+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(_existing_lock_is_held(lock, label="test lock"))

    def test_existing_lock_probe_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "hostile.lock"
            os.mkfifo(lock)
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "regular non-symlink"
            ):
                _existing_lock_is_held(lock, label="test lock")

    def test_existing_lock_probe_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.touch()
            lock = root / "hostile.lock"
            lock.symlink_to(target)
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "regular non-symlink"
            ):
                _existing_lock_is_held(lock, label="test lock")

    def test_selection_poison_blocks_even_when_manifest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            write_json(
                harness.config.selection_root / ".work" / "POISONED.json",
                {
                    "format": "all-eligible-publication-poison",
                    "format_version": 1,
                    "reason": "source authority changed",
                },
            )
            (harness.config.selection_root / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "poisoned"
            ):
                PostCurationOrchestrator._selection_lock_held(harness)

    def test_selection_must_match_exact_curation_snapshot_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            database = {"path": "/snapshot/db", "bytes": 10, "sha256": "d" * 64}
            checkpoint = {
                "path": "/snapshot/checkpoint",
                "bytes": 11,
                "sha256": "e" * 64,
            }
            curation = {
                "source_identity_sha256": "f" * 64,
                "source_snapshot": {
                    "generation": 7,
                    "manifest_path": "/snapshot/manifest.json",
                    "manifest_sha256": "1" * 64,
                    "database": database,
                    "checkpoint": checkpoint,
                },
            }
            source = {
                "identity_sha256": "f" * 64,
                "database": database,
                "checkpoint": checkpoint,
                "snapshot": {
                    "generation": 7,
                    "manifest_path": "/snapshot/manifest.json",
                    "manifest_sha256": "1" * 64,
                    "identity_sha256": "f" * 64,
                },
            }
            selection = {"identity": {"source_curation": source}}
            PostCurationOrchestrator._authenticate_selection_curation(
                harness, selection, curation
            )
            source["database"] = {**database, "sha256": "2" * 64}
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "another curation snapshot database"
            ):
                PostCurationOrchestrator._authenticate_selection_curation(
                    harness, selection, curation
                )

    def test_existing_selection_requires_publication_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            root = harness.config.selection_root
            root.mkdir(parents=True)
            (root / "manifest.json").write_text("{}")
            (root / "manifest.sha256").write_text("fixture")
            database = {"path": "/snapshot/db", "bytes": 10, "sha256": "d" * 64}
            checkpoint = {
                "path": "/snapshot/checkpoint",
                "bytes": 11,
                "sha256": "e" * 64,
            }
            curation = {
                "source_identity_sha256": "f" * 64,
                "source_snapshot": {
                    "generation": 7,
                    "manifest_path": "/snapshot/manifest.json",
                    "manifest_sha256": "1" * 64,
                    "database": database,
                    "checkpoint": checkpoint,
                },
            }
            selection = {
                "identity": {
                    "source_curation": {
                        "identity_sha256": "f" * 64,
                        "database": database,
                        "checkpoint": checkpoint,
                        "snapshot": {
                            "generation": 7,
                            "manifest_path": "/snapshot/manifest.json",
                            "manifest_sha256": "1" * 64,
                            "identity_sha256": "f" * 64,
                        },
                    }
                }
            }
            with mock.patch(
                "pretrain.postcuration_orchestrator._load_selection",
                return_value=(selection, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError,
                    "selection publication checkpoint",
                ):
                    PostCurationOrchestrator._inspect_selection(
                        harness, curation
                    )

    def test_selection_publication_checkpoint_identity_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            source = {
                "database": {"sha256": "d" * 64},
                "checkpoint": {"sha256": "e" * 64},
            }
            identity = {"source_curation": source, "format_version": 7}
            selection = {
                "identity": identity,
                "selection_strategy": "all-eligible",
                "selection_profile": {"profile": "fixture"},
                "selected_totals": [],
                "reference_quotas": [],
                "decision_shards": [],
            }
            write_json(
                harness.config.selection_root
                / ".work"
                / "PUBLICATION_CHECKPOINT.json",
                {
                    "checkpoint_version": 1,
                    "publication_identity_sha256": canonical_sha256(identity),
                    "source_database_sha256": "d" * 64,
                    "source_checkpoint_sha256": "e" * 64,
                    "selection_strategy": "all-eligible",
                    "selection_profile": {"profile": "fixture"},
                    "selected_totals": [],
                    "reference_quotas": [],
                    "completed_shards": [],
                },
            )
            PostCurationOrchestrator._authenticate_selection_checkpoint(
                harness, selection
            )
            selection["selection_profile"] = {"profile": "changed"}
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "checkpoint identity mismatch"
            ):
                PostCurationOrchestrator._authenticate_selection_checkpoint(
                    harness, selection
                )

    def test_cache_status_counts_only_atomic_completed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = Harness(self.config(root))
            sources = [
                SimpleNamespace(
                    archive=SimpleNamespace(bucket="python", index=0),
                    content_tokens=11,
                    authority=mock.Mock(return_value=object()),
                ),
                SimpleNamespace(
                    archive=SimpleNamespace(bucket="other_code", index=2),
                    content_tokens=19,
                    authority=mock.Mock(return_value=object()),
                ),
            ]
            completed = harness.config.cache_root / "archives/python/part-000000"
            completed.mkdir(parents=True)
            manifest_raw = b"{}\n"
            (completed / "manifest.json").write_bytes(manifest_raw)
            digest = hashlib.sha256(manifest_raw).hexdigest()
            (completed / "manifest.sha256").write_text(
                f"{digest}  manifest.json\n"
            )
            reader = mock.MagicMock()
            with (
                mock.patch(
                    "pretrain.postcuration_orchestrator._sources",
                    return_value=sources,
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator._tokenizer_authority",
                    return_value=object(),
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator.RawTokenCacheReader.open",
                    return_value=reader,
                ),
            ):
                status, found = PostCurationOrchestrator._cache_status(harness, {})
            self.assertEqual(found, sources)
            self.assertEqual(status["archives_complete"], 1)
            self.assertEqual(status["archives_total"], 2)
            self.assertEqual(status["content_tokens_complete"], 11)
            self.assertEqual(status["status"], "waiting")
            sources[0].authority.assert_called_once()

    def test_cache_status_rejects_incomplete_published_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = Harness(self.config(root))
            source = SimpleNamespace(
                archive=SimpleNamespace(bucket="python", index=0),
                content_tokens=11,
                authority=mock.Mock(return_value=object()),
            )
            target = harness.config.cache_root / "archives/python/part-000000"
            target.mkdir(parents=True)
            (target / "manifest.json").write_text("{}")
            with (
                mock.patch(
                    "pretrain.postcuration_orchestrator._sources",
                    return_value=[source],
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator._tokenizer_authority",
                    return_value=object(),
                ),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "manifest sidecar"
                ):
                    PostCurationOrchestrator._cache_status(harness, {})

    def test_cache_status_rejects_extra_closed_world_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = Harness(self.config(root))
            source = SimpleNamespace(
                archive=SimpleNamespace(
                    bucket="python", index=0, path="raw/python/part-000000.tar.zst"
                ),
                content_tokens=11,
                authority=mock.Mock(return_value=object()),
            )
            extra = harness.config.cache_root / "archives/python/part-000001"
            extra.mkdir(parents=True)
            with (
                mock.patch(
                    "pretrain.postcuration_orchestrator._sources",
                    return_value=[source],
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator._tokenizer_authority",
                    return_value=object(),
                ),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "not closed-world"
                ):
                    PostCurationOrchestrator._cache_status(harness, {})

    def test_cache_status_rejects_stale_staging_without_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = Harness(self.config(root))
            source = SimpleNamespace(
                archive=SimpleNamespace(
                    bucket="python", index=0, path="raw/python/part-000000.tar.zst"
                ),
                content_tokens=11,
                authority=mock.Mock(return_value=object()),
            )
            stage = (
                harness.config.cache_root
                / "archives/python/.part-000000.building-hostile"
            )
            stage.mkdir(parents=True)
            with (
                mock.patch(
                    "pretrain.postcuration_orchestrator._sources",
                    return_value=[source],
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator._tokenizer_authority",
                    return_value=object(),
                ),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "Stale raw-token-cache staging"
                ):
                    PostCurationOrchestrator._cache_status(harness, {})

    def test_cache_status_wraps_reader_os_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = Harness(self.config(root))
            source = SimpleNamespace(
                archive=SimpleNamespace(
                    bucket="python", index=0, path="raw/python/part-000000.tar.zst"
                ),
                content_tokens=11,
                authority=mock.Mock(return_value=object()),
            )
            target = harness.config.cache_root / "archives/python/part-000000"
            target.mkdir(parents=True)
            manifest_raw = b"{}\n"
            (target / "manifest.json").write_bytes(manifest_raw)
            digest = hashlib.sha256(manifest_raw).hexdigest()
            (target / "manifest.sha256").write_text(
                f"{digest}  manifest.json\n"
            )
            with (
                mock.patch(
                    "pretrain.postcuration_orchestrator._sources",
                    return_value=[source],
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator._tokenizer_authority",
                    return_value=object(),
                ),
                mock.patch(
                    "pretrain.postcuration_orchestrator.RawTokenCacheReader.open",
                    side_effect=OSError("mmap boundary failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    PostCurationOrchestrationError, "mmap boundary failed"
                ):
                    PostCurationOrchestrator._cache_status(harness, {})

    def test_minimal_packed_journal_is_not_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            output = harness.config.materialization_output
            output.mkdir(parents=True)
            write_json(
                output / JOURNAL_NAME,
                {
                    "format": MATERIALIZATION_FORMAT,
                    "format_version": MATERIALIZATION_FORMAT_VERSION,
                    "identity": harness.materializer.identity,
                    "state": {
                        "phase": "packed",
                        "completed_archives": 2,
                        "archive_count": 2,
                        "writer_cursors": {},
                    },
                },
            )
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "exact writer cursor inventory"
            ):
                PostCurationOrchestrator._inspect_materialization(
                    harness, harness.materializer
                )

    def test_packed_terminal_state_uses_full_payload_and_index_validators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            output = harness.config.materialization_output
            cursors: dict[str, dict[str, object]] = {}
            for split in ("train", "validation", "test"):
                for domain in ("python", "other_code", "english"):
                    (output / "packed" / split / domain).mkdir(parents=True)
                    (
                        output
                        / "provenance"
                        / "documents"
                        / split
                        / domain
                    ).mkdir(parents=True)
                    cursors[f"{split}/{domain}"] = {
                        "next_archive": 2,
                        "split": split,
                        "domain": domain,
                        "selected_documents": 3,
                        "selected_content_tokens": 17,
                    }
            write_json(
                output / JOURNAL_NAME,
                {
                    "format": MATERIALIZATION_FORMAT,
                    "format_version": MATERIALIZATION_FORMAT_VERSION,
                    "identity": harness.materializer.identity,
                    "state": {
                        "phase": "packed",
                        "completed_archives": 2,
                        "archive_count": 2,
                        "writer_cursors": cursors,
                    },
                },
            )
            harness.materializer._validate_cursor = mock.Mock(
                side_effect=lambda cursor, **_kwargs: cursor
            )
            harness.materializer._validate_document_index_manifest = mock.Mock(
                return_value={}
            )

            def packed_manifest(path: Path, **_kwargs: object) -> dict[str, object]:
                domain = path.parent.name
                split = path.parent.parent.name
                packing = harness.materializer.identity["packing_configuration"]
                return {
                    "split": split,
                    "domain": domain,
                    "sequence_length": packing["sequence_length"],
                    "rows_per_shard": packing["rows_per_shard"],
                    "construction_seed": packing["construction_seed"],
                    "vocab_size": packing["expected_vocab_size"],
                    "eos_token_id": packing["expected_eos_token_id"],
                    "tokenizer_manifest_sha256": harness.materializer.identity[
                        "tokenizer_manifest_sha256"
                    ],
                    "curation_policy_sha256": harness.materializer.identity[
                        "curation_policy_sha256"
                    ],
                    "selection_manifest_sha256": harness.materializer.identity[
                        "selection_manifest_sha256"
                    ],
                    "documents": 3,
                    "source_content_tokens": 17,
                    "construction_last_source_cursor": cursors[
                        f"{split}/{domain}"
                    ],
                }

            with mock.patch(
                "pretrain.postcuration_orchestrator.validate_packed_manifest",
                side_effect=packed_manifest,
            ) as validator:
                state = PostCurationOrchestrator._inspect_materialization(
                    harness, harness.materializer
                )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["packed_outputs"], 9)
            self.assertEqual(validator.call_count, 9)
            self.assertEqual(
                harness.materializer._validate_document_index_manifest.call_count,
                9,
            )

    def test_materialization_rejects_geometry_order_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            (harness.config.materialization_output / "orders").mkdir(parents=True)
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "Geometry-dependent"
            ):
                PostCurationOrchestrator._inspect_materialization(
                    harness, harness.materializer
                )

    def test_minimal_final_manifest_is_not_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            output = harness.config.materialization_output
            manifest = output / "manifest.json"
            write_json(
                manifest,
                {
                    "format": MATERIALIZATION_FORMAT,
                    "format_version": MATERIALIZATION_FORMAT_VERSION,
                    "identity": harness.materializer.identity,
                },
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (output / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
            harness.materializer._validate_completed = mock.Mock(
                side_effect=MaterializationError("manifest is incomplete")
            )
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "manifest is incomplete"
            ):
                PostCurationOrchestrator._inspect_materialization(
                    harness, harness.materializer
                )
            harness.materializer._validate_completed.assert_called_once_with()

    def test_final_validation_never_cleans_a_racing_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = Harness(self.config(Path(temporary)))
            output = harness.config.materialization_output
            manifest = output / "manifest.json"
            write_json(
                manifest,
                {
                    "format": MATERIALIZATION_FORMAT,
                    "format_version": MATERIALIZATION_FORMAT_VERSION,
                    "identity": harness.materializer.identity,
                },
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (output / "manifest.sha256").write_text(
                f"{digest}  manifest.json\n"
            )
            journal = output / JOURNAL_NAME

            def racing_validator() -> None:
                write_json(journal, {"phase": "racing"})
                if harness.materializer.journal_path.exists():
                    harness.materializer.journal_path.unlink()

            harness.materializer._validate_completed = racing_validator
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "appeared during final validation"
            ):
                PostCurationOrchestrator._inspect_materialization(
                    harness, harness.materializer
                )
            self.assertTrue(journal.is_file())

    def test_config_rejects_invalid_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = PostCurationConfig.for_generation(
                temporary, expected_train_input_tokens=0
            )
            with self.assertRaisesRegex(
                PostCurationOrchestrationError, "positive integer"
            ):
                PostCurationOrchestrator(config)


if __name__ == "__main__":
    unittest.main()
