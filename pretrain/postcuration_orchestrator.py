"""Fail-closed orchestration for CPU work after fast curation.

The orchestrator deliberately owns no data-transformation implementation.  It
authenticates the durable hand-off artifacts and invokes the repository's
existing publishers/materializer in their resumable modes.  Inspection is
read-only; writes are possible only when ``execute=True``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pretrain.data import DOMAIN_ORDER, validate_packed_manifest
from pretrain.materialize import (
    FORMAT as MATERIALIZATION_FORMAT,
    FORMAT_VERSION as MATERIALIZATION_FORMAT_VERSION,
    JOURNAL_NAME,
    SPLITS,
    CorpusMaterializer,
    MaterializationConfig,
    MaterializationError,
    canonical_sha256,
)
from pretrain.raw_token_cache import MANIFEST_FILE as CACHE_MANIFEST_FILE
from pretrain.raw_token_cache import SIDECAR_FILE as CACHE_SIDECAR_FILE
from pretrain.raw_token_cache import RawTokenCacheError
from pretrain.raw_token_cache_inventory import (
    RawTokenCacheInventoryError,
    load_raw_token_cache_inventory,
)
from pretrain.raw_token_cache_reader import (
    RawTokenCacheReadError,
    RawTokenCacheReader,
)
from pretrain.selection_contract import ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS
from scripts.launch_fast_all_eligible_curation import (
    AUTHORITY_NAME,
    COMPLETION_NAME,
    LaunchConfig,
    FastCurationLaunchError,
    _expected_completion,
    _load_authority,
    _safe_json_file,
    _sha256_bytes,
    _validate_result,
    resolve_config,
)
from scripts.publish_raw_token_cache_inventory import (
    _load_selection,
    _sources,
    _tokenizer_authority,
)
from scripts.qualify_selection_supply import qualify_selection_supply


FORMAT = "post-curation-cpu-orchestration"
FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


class PostCurationOrchestrationError(RuntimeError):
    """An authority or stage invariant could not be proven."""


class PrerequisitePending(RuntimeError):
    """A valid prerequisite has not been published yet."""

    def __init__(self, stage: str, detail: str):
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


class _ReadOnlyJournalSentinel:
    """Prevent a reused materializer validator from performing cleanup writes."""

    @staticmethod
    def exists() -> bool:
        return False

    @staticmethod
    def unlink() -> None:  # pragma: no cover - defensive future-proofing
        raise PostCurationOrchestrationError(
            "Read-only materialization validation attempted journal cleanup"
        )


Runner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class PostCurationConfig:
    generation_root: Path
    selection_root: Path
    tokenizer_root: Path
    cache_root: Path
    cache_inventory_root: Path
    materialization_output: Path
    sequence_length: int = 4_096
    rows_per_shard: int = 131_072
    construction_seed: int = 1_234
    order_seed: int = 1_234
    expected_train_input_tokens: int = 52_580_000_000
    expected_validation_input_tokens: int = 500_000_000
    expected_test_input_tokens: int = 500_000_000
    expected_vocab_size: int = 49_152
    expected_eos_token_id: int = 0

    @classmethod
    def for_generation(
        cls,
        generation_root: str | Path,
        *,
        selection_root: str | Path | None = None,
        tokenizer_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        cache_inventory_root: str | Path | None = None,
        materialization_output: str | Path | None = None,
        **overrides: Any,
    ) -> "PostCurationConfig":
        root = Path(generation_root).absolute()
        return cls(
            generation_root=root,
            selection_root=Path(
                selection_root or root / "curated" / "selection-v7"
            ).absolute(),
            tokenizer_root=Path(
                tokenizer_root or root / "tokenizer" / "starcoder2"
            ).absolute(),
            cache_root=Path(
                cache_root or root / "token-cache" / "raw-all-v1"
            ).absolute(),
            cache_inventory_root=Path(
                cache_inventory_root
                or root / "token-cache" / "inventories" / "selection-v7"
            ).absolute(),
            materialization_output=Path(
                materialization_output or root / "final" / "packed-v1"
            ).absolute(),
            **overrides,
        )

    def validate(self) -> None:
        if self.generation_root.is_symlink() or not self.generation_root.is_dir():
            raise PostCurationOrchestrationError(
                "generation_root must be a real existing directory"
            )
        integer_fields = {
            "sequence_length": self.sequence_length,
            "rows_per_shard": self.rows_per_shard,
            "expected_train_input_tokens": self.expected_train_input_tokens,
            "expected_validation_input_tokens": self.expected_validation_input_tokens,
            "expected_test_input_tokens": self.expected_test_input_tokens,
            "expected_vocab_size": self.expected_vocab_size,
        }
        for field, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise PostCurationOrchestrationError(
                    f"{field} must be a positive integer"
                )
        if self.sequence_length < 2:
            raise PostCurationOrchestrationError(
                "sequence_length must be at least 2"
            )
        for field, value in (
            ("construction_seed", self.construction_seed),
            ("order_seed", self.order_seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PostCurationOrchestrationError(
                    f"{field} must be a non-negative integer"
                )
        if (
            not isinstance(self.expected_eos_token_id, int)
            or isinstance(self.expected_eos_token_id, bool)
            or self.expected_eos_token_id < 0
        ):
            raise PostCurationOrchestrationError(
                "expected_eos_token_id must be a non-negative integer"
            )
        if self.expected_eos_token_id >= self.expected_vocab_size:
            raise PostCurationOrchestrationError(
                "expected_eos_token_id must be smaller than expected_vocab_size"
            )


def _default_runner(argv: Sequence[str]) -> int:
    completed = subprocess.run(list(argv), cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostCurationOrchestrationError(
            f"{label} must be a regular non-symlink file: {path}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostCurationOrchestrationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            _stable_regular_file_bytes(
                path, label=label, maximum_bytes=64 * 1024 * 1024
            ),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostCurationOrchestrationError(f"Invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PostCurationOrchestrationError(f"{label} must be a JSON object")
    return value


def _existing_lock_is_held(path: Path, *, label: str) -> bool:
    """Probe an existing advisory lock without creating or changing the file."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PostCurationOrchestrationError(
            f"Cannot inspect {label} {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PostCurationOrchestrationError(
            f"{label} is not a regular non-symlink file: {path}"
        )
    # O_NONBLOCK is essential here: a hostile FIFO must never hang inspection
    # between the lstat and open. O_NOFOLLOW closes the corresponding symlink
    # replacement race where the platform supports it.
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PostCurationOrchestrationError(
            f"Cannot inspect {label} {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"{label} changed while being opened: {path}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise PostCurationOrchestrationError(
                f"{label} changed identity while being opened: {path}"
            )
        # Every writer uses an exclusive lock.  Probe it with a shared lock so
        # the read-only descriptor remains portable to flock implementations
        # that reject LOCK_EX unless the file was opened for writing.
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"Cannot probe {label} {path}: {exc}"
            ) from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"Cannot release probe for {label} {path}: {exc}"
            ) from exc
        return False
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"Cannot close {label} {path}: {exc}"
            ) from exc


def _stable_regular_file_bytes(
    path: Path, *, label: str, maximum_bytes: int
) -> bytes:
    """Read a small authority file without following or racing path swaps."""

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PostCurationOrchestrationError(
            f"Cannot open {label} {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            raise PostCurationOrchestrationError(
                f"{label} has an unsafe type or size: {path}"
            )
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.read(
                descriptor, min(1024 * 1024, before.st_size - len(payload))
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"{label} disappeared while being read: {path}"
            ) from exc
        identity = lambda value: (  # noqa: E731 - compact immutable projection
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or identity(before) != identity(after)
            or identity(before) != identity(current)
            or stat.S_ISLNK(current.st_mode)
        ):
            raise PostCurationOrchestrationError(
                f"{label} changed while being read: {path}"
            )
        return bytes(payload)
    except OSError as exc:
        raise PostCurationOrchestrationError(
            f"Cannot read {label} {path}: {exc}"
        ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise PostCurationOrchestrationError(
                f"Cannot close {label} {path}: {exc}"
            ) from exc


def _require_exact_directory_children(
    path: Path, *, expected: set[str], label: str
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PostCurationOrchestrationError(
            f"{label} must be a real directory: {path}"
        )
    observed: set[str] = set()
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            raise PostCurationOrchestrationError(
                f"Unsafe {label} child: {child}"
            )
        observed.add(child.name)
    if observed != expected:
        raise PostCurationOrchestrationError(
            f"{label} is not closed-world: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


class PostCurationOrchestrator:
    """Inspect and optionally advance the CPU-only post-curation stages."""

    def __init__(
        self,
        config: PostCurationConfig,
        *,
        execute: bool = False,
        runner: Runner | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        config.validate()
        if not isinstance(execute, bool):
            raise PostCurationOrchestrationError("execute must be boolean")
        self.config = config
        self.execute = execute
        self.runner = runner or _default_runner
        self.python_executable = str(
            Path(python_executable or sys.executable).absolute()
        )
        self.actions: list[dict[str, Any]] = []

    def _base_report(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "mode": "execute" if self.execute else "dry-run",
            "guardrails": {
                "default_read_only": True,
                "gpu_actions_allowed": False,
                "geometry_dependent_orders_allowed": False,
                "materialization_mode": "stop-after-packing",
            },
            "paths": {
                "generation_root": str(self.config.generation_root),
                "selection_root": str(self.config.selection_root),
                "cache_root": str(self.config.cache_root),
                "cache_inventory_root": str(self.config.cache_inventory_root),
                "materialization_output": str(
                    self.config.materialization_output
                ),
            },
            "stages": {},
            "actions": self.actions,
        }

    @staticmethod
    def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
        if not isinstance(value, dict):
            raise PostCurationOrchestrationError(f"{field} must be an object")
        return value

    def _curation_config_from_authority(
        self, authority: Mapping[str, Any]
    ) -> LaunchConfig:
        identity = self._require_mapping(authority.get("identity"), field="authority.identity")
        paths = self._require_mapping(identity.get("paths"), field="authority.identity.paths")
        inputs = self._require_mapping(identity.get("inputs"), field="authority.identity.inputs")
        try:
            quotas = Path(
                str(
                    self._require_mapping(inputs["quotas"], field="inputs.quotas")[
                        "path"
                    ]
                )
            )
            policy = Path(
                str(
                    self._require_mapping(inputs["policy"], field="inputs.policy")[
                        "path"
                    ]
                )
            )
            denylist = Path(
                str(
                    self._require_mapping(
                        inputs["benchmark_denylist"], field="inputs.benchmark_denylist"
                    )["path"]
                )
            )
            local = Path(str(paths["local_work_root"]))
            log_path = Path(str(paths["log"]))
            result_path = Path(str(paths["result"]))
        except KeyError as exc:
            raise PostCurationOrchestrationError(
                f"Frozen curation authority is missing {exc}"
            ) from exc
        try:
            resolved = resolve_config(
                generation_root=self.config.generation_root,
                local_work_root=local,
                quotas=quotas,
                policy=policy,
                benchmark_denylist=denylist,
                log_path=log_path,
                result_path=result_path,
            )
        except (FastCurationLaunchError, OSError, ValueError) as exc:
            raise PostCurationOrchestrationError(
                f"Cannot reconstruct frozen curation authority: {exc}"
            ) from exc
        expected_paths = {
            "generation_root": str(resolved.generation_root),
            "staging_root": str(resolved.staging_root),
            "output": str(resolved.output),
            "local_work_root": str(resolved.local_work_root),
            "log": str(resolved.log_path),
            "result": str(resolved.result_path),
        }
        if dict(paths) != expected_paths:
            raise PostCurationOrchestrationError(
                "Frozen curation paths do not reconstruct exactly"
            )
        for name, path in (
            ("quotas", resolved.quotas),
            ("policy", resolved.policy),
            ("benchmark_denylist", resolved.benchmark_denylist),
        ):
            descriptor = self._require_mapping(inputs[name], field=f"inputs.{name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if descriptor != {"path": str(path), "sha256": digest}:
                raise PostCurationOrchestrationError(
                    f"Frozen curation input changed: {name}"
                )
        return resolved

    def _inspect_curation(self) -> dict[str, Any]:
        authority_path = self.config.generation_root / AUTHORITY_NAME
        if not authority_path.exists() and not authority_path.is_symlink():
            raise PrerequisitePending("curation", "curation authority is not published")
        try:
            authority = _load_authority(authority_path)
        except (FastCurationLaunchError, OSError, ValueError) as exc:
            raise PostCurationOrchestrationError(
                f"Invalid curation authority: {exc}"
            ) from exc
        if authority is None:
            raise PrerequisitePending("curation", "curation authority is not published")
        launch_config = self._curation_config_from_authority(authority)
        if launch_config.result_path.is_symlink():
            raise PostCurationOrchestrationError(
                f"Unsafe curation result: {launch_config.result_path}"
            )
        if not launch_config.result_path.exists():
            raise PrerequisitePending("curation", "immutable curation result is not published")
        try:
            result_raw, _result_payload = _safe_json_file(
                launch_config.result_path, label="curation result"
            )
            authority_identity = self._require_mapping(
                authority["identity"], field="authority.identity"
            )
            source = self._require_mapping(
                authority_identity.get("source"), field="authority.identity.source"
            )
            result, snapshot = _validate_result(
                launch_config,
                result_raw,
                expected_source_identity=self._require_mapping(
                    source.get("curation_identity"),
                    field="authority.identity.source.curation_identity",
                ),
            )
        except (FastCurationLaunchError, OSError, ValueError) as exc:
            raise PostCurationOrchestrationError(
                f"Invalid immutable curation result/snapshot: {exc}"
            ) from exc
        if launch_config.completion_path.is_symlink():
            raise PostCurationOrchestrationError(
                f"Unsafe curation completion receipt: {launch_config.completion_path}"
            )
        if not launch_config.completion_path.exists():
            raise PrerequisitePending(
                "curation", "curation completion receipt is not published"
            )
        try:
            _completion_raw, completion = _safe_json_file(
                launch_config.completion_path, label="curation completion receipt"
            )
        except (FastCurationLaunchError, OSError, ValueError) as exc:
            raise PostCurationOrchestrationError(
                f"Invalid curation completion receipt: {exc}"
            ) from exc
        expected = _expected_completion(
            launch_config,
            identity_sha256=str(authority["identity_sha256"]),
            result_sha256=_sha256_bytes(result_raw),
            snapshot=snapshot,
        )
        if set(completion) != set(expected) | {"created_utc"}:
            raise PostCurationOrchestrationError(
                "Curation completion receipt schema mismatch"
            )
        projected = {key: value for key, value in completion.items() if key != "created_utc"}
        if projected != expected or not isinstance(completion.get("created_utc"), str):
            raise PostCurationOrchestrationError(
                "Curation completion receipt authority mismatch"
            )
        source_snapshot = self._require_mapping(
            result.get("source_snapshot"), field="curation result source_snapshot"
        )
        database = self._require_mapping(
            source_snapshot.get("database"), field="curation snapshot database"
        )
        checkpoint = self._require_mapping(
            source_snapshot.get("checkpoint"), field="curation snapshot checkpoint"
        )
        authority_source = self._require_mapping(
            self._require_mapping(
                authority.get("identity"), field="authority.identity"
            ).get("source"),
            field="authority.identity.source",
        )
        curation_identity = self._require_mapping(
            authority_source.get("curation_identity"),
            field="authority.identity.source.curation_identity",
        )
        return {
            "status": "complete",
            "launcher_identity_sha256": authority["identity_sha256"],
            "result_sha256": _sha256_bytes(result_raw),
            "snapshot": snapshot,
            "source_identity_sha256": canonical_sha256(curation_identity),
            "source_snapshot": {
                "generation": source_snapshot.get("generation"),
                "manifest_path": source_snapshot.get("manifest_path"),
                "manifest_sha256": source_snapshot.get("manifest_sha256"),
                "database": dict(database),
                "checkpoint": dict(checkpoint),
            },
            "source_db": str(database["path"]),
            "source_checkpoint": str(checkpoint["path"]),
            "source_snapshot_manifest": str(source_snapshot["manifest_path"]),
            "launch_config": launch_config,
        }

    def _selection_command(self, curation: Mapping[str, Any]) -> list[str]:
        launch = curation["launch_config"]
        assert isinstance(launch, LaunchConfig)
        return [
            self.python_executable,
            str(SCRIPTS_ROOT / "publish_all_eligible_selection.py"),
            "--root",
            str(launch.generation_root),
            "--staging-root",
            str(launch.staging_root),
            "--source-db",
            str(curation["source_db"]),
            "--source-checkpoint",
            str(curation["source_checkpoint"]),
            "--source-snapshot-manifest",
            str(curation["source_snapshot_manifest"]),
            "--output",
            str(self.config.selection_root),
            "--policy",
            str(launch.policy),
            "--quotas",
            str(launch.quotas),
            "--benchmark-denylist",
            str(launch.benchmark_denylist),
        ]

    def _selection_lock_held(self) -> bool:
        root = self.config.selection_root
        poison = root / ".work" / "POISONED.json"
        if poison.exists() or poison.is_symlink():
            if poison.is_symlink() or not poison.is_file():
                raise PostCurationOrchestrationError(
                    f"Unsafe selection publication poison marker: {poison}"
                )
            payload = _json_file(poison, label="selection publication poison marker")
            reason = payload.get("reason")
            detail = reason if isinstance(reason, str) and reason else "unknown reason"
            raise PostCurationOrchestrationError(
                f"Selection publication is poisoned: {detail}"
            )
        return _existing_lock_is_held(
            root / ".all-eligible-publication.lock",
            label="selection publication lock",
        )

    def _authenticate_selection_checkpoint(
        self, selection: Mapping[str, Any]
    ) -> None:
        """Bind a completed manifest to the publisher's durable checkpoint."""

        checkpoint_path = (
            self.config.selection_root / ".work" / "PUBLICATION_CHECKPOINT.json"
        )
        checkpoint = _json_file(
            checkpoint_path, label="selection publication checkpoint"
        )
        identity = self._require_mapping(
            selection.get("identity"), field="selection.identity"
        )
        source = self._require_mapping(
            identity.get("source_curation"),
            field="selection.identity.source_curation",
        )
        database = self._require_mapping(
            source.get("database"), field="selection source database"
        )
        source_checkpoint = self._require_mapping(
            source.get("checkpoint"), field="selection source checkpoint"
        )
        expected = {
            "checkpoint_version": 1,
            "publication_identity_sha256": canonical_sha256(identity),
            "source_database_sha256": database.get("sha256"),
            "source_checkpoint_sha256": source_checkpoint.get("sha256"),
            "selection_strategy": selection.get("selection_strategy"),
            "selection_profile": selection.get("selection_profile"),
            "selected_totals": selection.get("selected_totals"),
            "reference_quotas": selection.get("reference_quotas"),
        }
        if set(checkpoint) != set(expected) | {"completed_shards"} or any(
            checkpoint.get(key) != value for key, value in expected.items()
        ):
            raise PostCurationOrchestrationError(
                "Selection publication checkpoint identity mismatch"
            )
        completed = checkpoint.get("completed_shards")
        decisions = selection.get("decision_shards")
        if not isinstance(completed, list) or not isinstance(decisions, list):
            raise PostCurationOrchestrationError(
                "Selection publication checkpoint has no shard inventory"
            )
        try:
            projected = [
                {
                    key: row[key]
                    for key in sorted(ALL_ELIGIBLE_BITMAP_DESCRIPTOR_KEYS)
                }
                for row in completed
                if isinstance(row, dict)
            ]
        except KeyError as exc:
            raise PostCurationOrchestrationError(
                f"Selection checkpoint shard is missing {exc}"
            ) from exc
        if len(projected) != len(completed) or projected != decisions:
            raise PostCurationOrchestrationError(
                "Selection manifest differs from its publication checkpoint"
            )

    def _authenticate_selection_curation(
        self,
        selection: Mapping[str, Any],
        curation: Mapping[str, Any],
    ) -> None:
        identity = self._require_mapping(
            selection.get("identity"), field="selection.identity"
        )
        source = self._require_mapping(
            identity.get("source_curation"),
            field="selection.identity.source_curation",
        )
        expected_snapshot = self._require_mapping(
            curation.get("source_snapshot"), field="curation.source_snapshot"
        )
        selection_snapshot = self._require_mapping(
            source.get("snapshot"), field="selection source snapshot"
        )
        if source.get("database") != expected_snapshot.get("database"):
            raise PostCurationOrchestrationError(
                "Selection is bound to another curation snapshot database"
            )
        if source.get("checkpoint") != expected_snapshot.get("checkpoint"):
            raise PostCurationOrchestrationError(
                "Selection is bound to another curation snapshot checkpoint"
            )
        expected_snapshot_projection = {
            "generation": expected_snapshot.get("generation"),
            "manifest_path": expected_snapshot.get("manifest_path"),
            "manifest_sha256": expected_snapshot.get("manifest_sha256"),
            "identity_sha256": curation.get("source_identity_sha256"),
        }
        if any(
            selection_snapshot.get(key) != value
            for key, value in expected_snapshot_projection.items()
        ):
            raise PostCurationOrchestrationError(
                "Selection is bound to another exact curation snapshot"
            )
        if source.get("identity_sha256") != curation.get("source_identity_sha256"):
            raise PostCurationOrchestrationError(
                "Selection source identity differs from frozen curation"
            )

    def _inspect_selection(
        self, curation: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str] | None:
        if self._selection_lock_held():
            raise PrerequisitePending(
                "selection", "selection publisher still holds its lock"
            )
        root = self.config.selection_root
        if root.is_symlink():
            raise PostCurationOrchestrationError(f"Unsafe selection root: {root}")
        manifest = root / "manifest.json"
        sidecar = root / "manifest.sha256"
        if manifest.is_symlink() or sidecar.is_symlink():
            raise PostCurationOrchestrationError(
                "Selection manifest authority contains a symlink"
            )
        if not manifest.exists() and not sidecar.exists():
            if root.exists() and not root.is_dir():
                raise PostCurationOrchestrationError(
                    f"Selection root is not a directory: {root}"
                )
            return None
        try:
            selection, digest = _load_selection(root)
        except (
            OSError,
            ValueError,
            RawTokenCacheError,
            RawTokenCacheInventoryError,
        ) as exc:
            raise PostCurationOrchestrationError(
                f"Invalid selection-v7 publication: {exc}"
            ) from exc
        self._authenticate_selection_curation(selection, curation)
        self._authenticate_selection_checkpoint(selection)
        return selection, digest

    def _run_action(self, stage: str, argv: Sequence[str]) -> None:
        action = {
            "stage": stage,
            "argv": [str(value) for value in argv],
            "executed": self.execute,
        }
        self.actions.append(action)
        if not self.execute:
            return
        try:
            code = self.runner(action["argv"])
        except Exception as exc:
            raise PostCurationOrchestrationError(
                f"{stage} runner failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(code, int) or isinstance(code, bool):
            raise PostCurationOrchestrationError(
                f"{stage} runner returned a non-integer status: {code!r}"
            )
        action["returncode"] = code
        if code != 0:
            raise PostCurationOrchestrationError(
                f"{stage} exited with status {code}"
            )

    def _cache_status(
        self, selection: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[Any]]:
        try:
            sources = _sources(
                root=self.config.generation_root,
                preprocess_root=self.config.generation_root / "staging" / "preprocess",
                selection=dict(selection),
                expected_vocab_size=self.config.expected_vocab_size,
            )
        except (
            OSError,
            ValueError,
            RawTokenCacheError,
            RawTokenCacheInventoryError,
        ) as exc:
            raise PostCurationOrchestrationError(
                f"Cannot reconcile selection with raw cache jobs: {exc}"
            ) from exc
        root = self.config.cache_root
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise PostCurationOrchestrationError(f"Unsafe raw-token-cache root: {root}")
        lock_held = _existing_lock_is_held(
            root / ".raw-token-cache.lock", label="raw-token-cache lock"
        )
        try:
            tokenizer = _tokenizer_authority(
                self.config.tokenizer_root,
                expected_vocab_size=self.config.expected_vocab_size,
            )
        except Exception as exc:
            raise PostCurationOrchestrationError(
                f"Cannot authenticate raw-token-cache tokenizer authority: {exc}"
            ) from exc

        expected_targets = {
            (
                Path("archives")
                / source.archive.bucket
                / f"part-{source.archive.index:06d}"
            ).as_posix(): source
            for source in sources
        }
        if len(expected_targets) != len(sources):
            raise PostCurationOrchestrationError(
                "Selection resolves to duplicate raw-token-cache targets"
            )
        observed_targets: dict[str, Path] = {}
        staging: list[str] = []
        if root.exists():
            allowed_root_entries = {"archives", ".raw-token-cache.lock"}
            unexpected_root = sorted(
                entry.name
                for entry in root.iterdir()
                if entry.name not in allowed_root_entries
            )
            if unexpected_root:
                raise PostCurationOrchestrationError(
                    "Raw-token-cache root is not closed-world: "
                    f"unexpected={unexpected_root}"
                )
            archives_root = root / "archives"
            if archives_root.exists() or archives_root.is_symlink():
                if archives_root.is_symlink() or not archives_root.is_dir():
                    raise PostCurationOrchestrationError(
                        f"Unsafe raw-token-cache archives root: {archives_root}"
                    )
                expected_buckets = {
                    source.archive.bucket for source in sources
                }
                for bucket in archives_root.iterdir():
                    if (
                        bucket.name not in expected_buckets
                        or bucket.is_symlink()
                        or not bucket.is_dir()
                    ):
                        raise PostCurationOrchestrationError(
                            f"Unsafe or unexpected raw-token-cache bucket: {bucket}"
                        )
                    expected_parts = {
                        f"part-{source.archive.index:06d}"
                        for source in sources
                        if source.archive.bucket == bucket.name
                    }
                    for candidate in bucket.iterdir():
                        relative = candidate.relative_to(root).as_posix()
                        if candidate.name in expected_parts:
                            if candidate.is_symlink() or not candidate.is_dir():
                                raise PostCurationOrchestrationError(
                                    "Unsafe raw-token-cache archive directory: "
                                    f"{candidate}"
                                )
                            observed_targets[relative] = candidate
                            continue
                        if any(
                            candidate.name.startswith(f".{part}.building-")
                            for part in expected_parts
                        ):
                            if candidate.is_symlink() or not candidate.is_dir():
                                raise PostCurationOrchestrationError(
                                    f"Unsafe raw-token-cache staging path: {candidate}"
                                )
                            staging.append(relative)
                            continue
                        raise PostCurationOrchestrationError(
                            "Raw-token-cache archive set is not closed-world: "
                            f"unexpected={relative}"
                        )
        extra = sorted(set(observed_targets) - set(expected_targets))
        if extra:
            raise PostCurationOrchestrationError(
                f"Raw-token-cache archive set is not closed-world: extra={extra}"
            )
        if staging and not lock_held:
            raise PostCurationOrchestrationError(
                "Stale raw-token-cache staging exists without an active publisher: "
                f"{sorted(staging)}"
            )

        completed = 0
        completed_tokens = 0
        for source in sources:
            relative = (
                Path("archives")
                / source.archive.bucket
                / f"part-{source.archive.index:06d}"
            ).as_posix()
            target = observed_targets.get(relative)
            if target is None:
                continue
            manifest_raw = _stable_regular_file_bytes(
                target / CACHE_MANIFEST_FILE,
                label="raw-token-cache manifest",
                maximum_bytes=4 * 1024 * 1024,
            )
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            sidecar_raw = _stable_regular_file_bytes(
                target / CACHE_SIDECAR_FILE,
                label="raw-token-cache manifest sidecar",
                maximum_bytes=1024,
            )
            if sidecar_raw != (
                f"{manifest_sha}  {CACHE_MANIFEST_FILE}\n".encode("ascii")
            ):
                raise PostCurationOrchestrationError(
                    f"Raw-token-cache manifest sidecar mismatch: {target}"
                )
            try:
                authority = source.authority(
                    cache_manifest_bytes=len(manifest_raw),
                    cache_manifest_sha256=manifest_sha,
                    tokenizer=tokenizer,
                )
                with RawTokenCacheReader.open(
                    target,
                    authority,
                    dataset_root=self.config.generation_root,
                    preprocess_root=(
                        self.config.generation_root / "staging" / "preprocess"
                    ),
                    tokenizer_root=self.config.tokenizer_root,
                ) as reader:
                    reader.verify_unchanged()
            except (
                OSError,
                RawTokenCacheReadError,
                RawTokenCacheInventoryError,
                ValueError,
            ) as exc:
                raise PostCurationOrchestrationError(
                    f"Invalid raw-token-cache target for {source.archive.path}: {exc}"
                ) from exc
            completed += 1
            completed_tokens += int(source.content_tokens)
        return (
            {
                "status": (
                    "complete" if (
                        completed == len(sources)
                        and not lock_held
                        and not staging
                    ) else "waiting"
                ),
                "archives_complete": completed,
                "archives_total": len(sources),
                "content_tokens_complete": completed_tokens,
                "content_tokens_total": sum(
                    int(source.content_tokens) for source in sources
                ),
                "publisher_lock_held": lock_held,
                "staging_directories": len(staging),
            },
            sources,
        )

    def _inventory_command(self) -> list[str]:
        return [
            self.python_executable,
            str(SCRIPTS_ROOT / "publish_raw_token_cache_inventory.py"),
            "--root",
            str(self.config.generation_root),
            "--preprocess-root",
            str(self.config.generation_root / "staging" / "preprocess"),
            "--selection-root",
            str(self.config.selection_root),
            "--tokenizer-root",
            str(self.config.tokenizer_root),
            "--cache-root",
            str(self.config.cache_root),
            "--output",
            str(self.config.cache_inventory_root),
            "--expected-vocab-size",
            str(self.config.expected_vocab_size),
        ]

    def _inspect_inventory(self, selection_sha256: str) -> Any | None:
        root = self.config.cache_inventory_root
        if not root.exists() and not root.is_symlink():
            return None
        try:
            inventory = load_raw_token_cache_inventory(
                inventory_root=root, cache_root=self.config.cache_root
            )
        except (OSError, ValueError, RawTokenCacheInventoryError) as exc:
            raise PostCurationOrchestrationError(
                f"Invalid closed-world raw-token-cache inventory: {exc}"
            ) from exc
        if inventory.selection_manifest_sha256 != selection_sha256:
            raise PostCurationOrchestrationError(
                "Cache inventory is bound to another selection manifest"
            )
        return inventory

    def _materialization_config(self) -> MaterializationConfig:
        return MaterializationConfig(
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.rows_per_shard,
            construction_seed=self.config.construction_seed,
            order_seed=self.config.order_seed,
            frozen_global_microbatch_rows=None,
            frozen_gradient_accumulation_steps=None,
            expected_train_input_tokens=self.config.expected_train_input_tokens,
            expected_validation_input_tokens=(
                self.config.expected_validation_input_tokens
            ),
            expected_test_input_tokens=self.config.expected_test_input_tokens,
            expected_vocab_size=self.config.expected_vocab_size,
            expected_eos_token_id=self.config.expected_eos_token_id,
        )

    def _build_materializer(self, curation: Mapping[str, Any]) -> CorpusMaterializer:
        launch = curation["launch_config"]
        assert isinstance(launch, LaunchConfig)
        try:
            return CorpusMaterializer(
                raw_root=self.config.generation_root,
                preprocess_root=self.config.generation_root / "staging" / "preprocess",
                selection_root=self.config.selection_root,
                tokenizer_root=self.config.tokenizer_root,
                policy_path=launch.policy,
                quota_path=launch.quotas,
                benchmark_denylist_path=launch.benchmark_denylist,
                output_root=self.config.materialization_output,
                config=self._materialization_config(),
                raw_token_cache_root=self.config.cache_root,
                raw_token_cache_inventory_root=self.config.cache_inventory_root,
            )
        except (OSError, ValueError, MaterializationError) as exc:
            raise PostCurationOrchestrationError(
                f"Cache-backed materialization preflight failed: {exc}"
            ) from exc

    def _materialization_command(self, curation: Mapping[str, Any]) -> list[str]:
        launch = curation["launch_config"]
        assert isinstance(launch, LaunchConfig)
        return [
            self.python_executable,
            str(SCRIPTS_ROOT / "materialize_training_corpus.py"),
            "--root",
            str(self.config.generation_root),
            "--preprocess-root",
            str(self.config.generation_root / "staging" / "preprocess"),
            "--selection-root",
            str(self.config.selection_root),
            "--tokenizer-root",
            str(self.config.tokenizer_root),
            "--raw-token-cache-root",
            str(self.config.cache_root),
            "--raw-token-cache-inventory-root",
            str(self.config.cache_inventory_root),
            "--output",
            str(self.config.materialization_output),
            "--curation-policy",
            str(launch.policy),
            "--quota-config",
            str(launch.quotas),
            "--benchmark-denylist",
            str(launch.benchmark_denylist),
            "--sequence-length",
            str(self.config.sequence_length),
            "--rows-per-shard",
            str(self.config.rows_per_shard),
            "--construction-seed",
            str(self.config.construction_seed),
            "--order-seed",
            str(self.config.order_seed),
            "--expected-train-input-tokens",
            str(self.config.expected_train_input_tokens),
            "--expected-validation-input-tokens",
            str(self.config.expected_validation_input_tokens),
            "--expected-test-input-tokens",
            str(self.config.expected_test_input_tokens),
            "--expected-vocab-size",
            str(self.config.expected_vocab_size),
            "--expected-eos-token-id",
            str(self.config.expected_eos_token_id),
            "--stop-after-packing",
        ]

    def _inspect_materialization(self, materializer: CorpusMaterializer) -> dict[str, Any]:
        root = self.config.materialization_output
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise PostCurationOrchestrationError(
                f"Unsafe materialization output: {root}"
            )
        lock_held = _existing_lock_is_held(
            root.parent / f".{root.name}.materialize.lock",
            label="materialization lock",
        )
        if not root.exists():
            return {"status": "ready", "phase": "not-started", "lock_held": lock_held}
        allowed_names = {
            JOURNAL_NAME,
            "packed",
            "orders",
            "provenance",
            "manifest.json",
            "manifest.sha256",
        }
        unexpected = sorted(
            path.name for path in root.iterdir() if path.name not in allowed_names
        )
        if unexpected:
            raise PostCurationOrchestrationError(
                f"Unknown materialization outputs: {unexpected}"
            )
        manifest_path = root / "manifest.json"
        sidecar_path = root / "manifest.sha256"
        if (
            manifest_path.exists()
            or manifest_path.is_symlink()
            or sidecar_path.exists()
            or sidecar_path.is_symlink()
        ):
            if (
                manifest_path.is_symlink()
                or sidecar_path.is_symlink()
                or not manifest_path.is_file()
                or not sidecar_path.is_file()
            ):
                raise PostCurationOrchestrationError(
                    "Unsafe finalized materialization manifest"
                )
            journal_path = root / JOURNAL_NAME
            if journal_path.exists() or journal_path.is_symlink():
                raise PostCurationOrchestrationError(
                    "Final materialization still has an unpublished authority journal"
                )
            validator = getattr(materializer, "_validate_completed", None)
            if not callable(validator):
                raise PostCurationOrchestrationError(
                    "Materializer does not expose its full final-output validator"
                )
            had_journal_attribute = hasattr(materializer, "journal_path")
            original_journal_path = getattr(materializer, "journal_path", None)
            try:
                materializer.journal_path = _ReadOnlyJournalSentinel()
                validated_manifest = validator()
            except (OSError, ValueError, MaterializationError) as exc:
                raise PostCurationOrchestrationError(
                    f"Invalid finalized materialization: {exc}"
                ) from exc
            finally:
                if had_journal_attribute:
                    materializer.journal_path = original_journal_path
                else:
                    delattr(materializer, "journal_path")
            if journal_path.exists() or journal_path.is_symlink():
                raise PostCurationOrchestrationError(
                    "Materialization journal appeared during final validation"
                )
            lock_held = _existing_lock_is_held(
                root.parent / f".{root.name}.materialize.lock",
                label="materialization lock",
            )
            manifest_raw = _stable_regular_file_bytes(
                manifest_path,
                label="final materialization manifest",
                maximum_bytes=64 * 1024 * 1024,
            )
            sidecar_raw = _stable_regular_file_bytes(
                sidecar_path,
                label="final materialization manifest sidecar",
                maximum_bytes=1024,
            )
            digest = hashlib.sha256(manifest_raw).hexdigest()
            if sidecar_raw != f"{digest}  manifest.json\n".encode("ascii"):
                raise PostCurationOrchestrationError(
                    "Final materialization manifest sidecar changed after validation"
                )
            try:
                observed_manifest = json.loads(manifest_raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PostCurationOrchestrationError(
                    f"Final materialization manifest changed after validation: {exc}"
                ) from exc
            if observed_manifest != validated_manifest:
                raise PostCurationOrchestrationError(
                    "Final materialization manifest changed during validation"
                )
            return {
                "status": "waiting" if lock_held else "complete",
                "phase": "finalized-outside-orchestrator",
                "lock_held": lock_held,
                "manifest_sha256": digest,
            }
        orders = root / "orders"
        if orders.exists() or orders.is_symlink():
            raise PostCurationOrchestrationError(
                "Geometry-dependent order state exists; CPU orchestrator will not touch it"
            )
        journal_path = root / JOURNAL_NAME
        if not journal_path.exists() and not journal_path.is_symlink():
            if any(root.iterdir()):
                raise PostCurationOrchestrationError(
                    "Materialization output is non-empty without an authority journal"
                )
            return {"status": "ready", "phase": "not-started", "lock_held": lock_held}
        journal = _json_file(journal_path, label="materialization journal")
        if set(journal) != {"format", "format_version", "identity", "state"}:
            raise PostCurationOrchestrationError(
                "Materialization journal schema mismatch"
            )
        if (
            journal.get("format") != MATERIALIZATION_FORMAT
            or journal.get("format_version") != MATERIALIZATION_FORMAT_VERSION
            or journal.get("identity") != materializer.identity
        ):
            raise PostCurationOrchestrationError(
                "Materialization journal identity mismatch"
            )
        state = self._require_mapping(
            journal.get("state"), field="materialization journal state"
        )
        if set(state) != {
            "phase",
            "completed_archives",
            "archive_count",
            "writer_cursors",
        }:
            raise PostCurationOrchestrationError(
                "Materialization journal state schema mismatch"
            )
        phase = state.get("phase")
        completed = state.get("completed_archives")
        archive_count = state.get("archive_count")
        writer_cursors = self._require_mapping(
            state.get("writer_cursors"),
            field="materialization journal writer_cursors",
        )
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or not isinstance(archive_count, int)
            or isinstance(archive_count, bool)
            or archive_count != len(materializer.archives)
            or completed < 0
            or completed > archive_count
        ):
            raise PostCurationOrchestrationError(
                "Materialization journal archive counters are invalid"
            )
        if phase == "packed" and completed == archive_count:
            if orders.exists() or orders.is_symlink():
                raise PostCurationOrchestrationError(
                    "Geometry-dependent order state exists beside packed hand-off"
                )
            expected_cursor_keys = {
                f"{split}/{domain}" for split in SPLITS for domain in DOMAIN_ORDER
            }
            if set(writer_cursors) != expected_cursor_keys:
                raise PostCurationOrchestrationError(
                    "Packed hand-off journal lacks the exact writer cursor inventory"
                )
            packing = self._require_mapping(
                materializer.identity.get("packing_configuration"),
                field="materializer packing_configuration",
            )
            _require_exact_directory_children(
                root / "packed",
                expected=set(SPLITS),
                label="packed split inventory",
            )
            _require_exact_directory_children(
                root / "provenance",
                expected={"documents"},
                label="packed provenance inventory",
            )
            _require_exact_directory_children(
                root / "provenance" / "documents",
                expected=set(SPLITS),
                label="document-index split inventory",
            )
            for split in SPLITS:
                _require_exact_directory_children(
                    root / "packed" / split,
                    expected=set(DOMAIN_ORDER),
                    label=f"packed domain inventory for {split}",
                )
                _require_exact_directory_children(
                    root / "provenance" / "documents" / split,
                    expected=set(DOMAIN_ORDER),
                    label=f"document-index domain inventory for {split}",
                )
            packed_outputs = 0
            document_indexes = 0
            for split in SPLITS:
                for domain in DOMAIN_ORDER:
                    packed_path = (
                        root / "packed" / split / domain / "manifest.json"
                    )
                    try:
                        packed_manifest = validate_packed_manifest(
                            packed_path, verify_checksums=True
                        )
                    except (OSError, ValueError) as exc:
                        raise PostCurationOrchestrationError(
                            f"Invalid packed hand-off {split}/{domain}: {exc}"
                        ) from exc
                    expected_packed_identity = {
                        "split": split,
                        "domain": domain,
                        "sequence_length": packing.get("sequence_length"),
                        "rows_per_shard": packing.get("rows_per_shard"),
                        "construction_seed": packing.get("construction_seed"),
                        "vocab_size": packing.get("expected_vocab_size"),
                        "eos_token_id": packing.get("expected_eos_token_id"),
                        "tokenizer_manifest_sha256": materializer.identity.get(
                            "tokenizer_manifest_sha256"
                        ),
                        "curation_policy_sha256": materializer.identity.get(
                            "curation_policy_sha256"
                        ),
                        "selection_manifest_sha256": materializer.identity.get(
                            "selection_manifest_sha256"
                        ),
                    }
                    if any(
                        packed_manifest.get(key) != value
                        for key, value in expected_packed_identity.items()
                    ):
                        raise PostCurationOrchestrationError(
                            f"Packed output identity mismatch for {split}/{domain}"
                        )
                    cursor_validator = getattr(
                        materializer, "_validate_cursor", None
                    )
                    index_validator = getattr(
                        materializer, "_validate_document_index_manifest", None
                    )
                    if not callable(cursor_validator) or not callable(index_validator):
                        raise PostCurationOrchestrationError(
                            "Materializer lacks full packed hand-off validators"
                        )
                    try:
                        cursor = cursor_validator(
                            packed_manifest.get(
                                "construction_last_source_cursor"
                            ),
                            split=split,
                            domain=domain,
                        )
                        if (
                            cursor.get("next_archive") != archive_count
                            or writer_cursors[f"{split}/{domain}"] != cursor
                            or cursor.get("selected_documents")
                            != packed_manifest.get("documents")
                            or cursor.get("selected_content_tokens")
                            != packed_manifest.get("source_content_tokens")
                        ):
                            raise MaterializationError(
                                "packed cursor differs from terminal journal/data"
                            )
                        index_validator(
                            root
                            / "provenance"
                            / "documents"
                            / split
                            / domain
                            / "manifest.json",
                            split=split,
                            domain=domain,
                            packed_manifest=packed_manifest,
                        )
                    except (OSError, ValueError, MaterializationError) as exc:
                        raise PostCurationOrchestrationError(
                            "Invalid packed hand-off cursor/document index for "
                            f"{split}/{domain}: {exc}"
                        ) from exc
                    packed_outputs += 1
                    document_indexes += 1
            return {
                "status": "waiting" if lock_held else "complete",
                "phase": "packed",
                "completed_archives": completed,
                "archive_count": archive_count,
                "lock_held": lock_held,
                "packed_outputs": packed_outputs,
                "document_indexes": document_indexes,
            }
        if phase == "packing" and completed <= archive_count:
            expected_cursor_keys = {
                f"{split}/{domain}" for split in SPLITS for domain in DOMAIN_ORDER
            }
            if writer_cursors and set(writer_cursors) != expected_cursor_keys:
                raise PostCurationOrchestrationError(
                    "Packing journal has a partial writer cursor inventory"
                )
            cursor_positions: list[int] = []
            cursor_validator = getattr(materializer, "_validate_cursor", None)
            if writer_cursors and not callable(cursor_validator):
                raise PostCurationOrchestrationError(
                    "Materializer lacks its writer cursor validator"
                )
            for key, raw_cursor in writer_cursors.items():
                split, domain = key.split("/", 1)
                try:
                    cursor = cursor_validator(
                        raw_cursor, split=split, domain=domain
                    )
                except (OSError, ValueError, MaterializationError) as exc:
                    raise PostCurationOrchestrationError(
                        f"Invalid packing writer cursor {key}: {exc}"
                    ) from exc
                cursor_positions.append(int(cursor["next_archive"]))
            if completed != (min(cursor_positions) if cursor_positions else 0):
                raise PostCurationOrchestrationError(
                    "Packing journal counter differs from writer cursors"
                )
            return {
                "status": "waiting" if lock_held else "resumable",
                "phase": "packing",
                "completed_archives": completed,
                "archive_count": archive_count,
                "lock_held": lock_held,
            }
        raise PostCurationOrchestrationError(
            f"Refusing materialization phase outside CPU packing authority: {phase!r}"
        )

    def _run(self) -> dict[str, Any]:
        report = self._base_report()
        stages = report["stages"]
        try:
            curation = self._inspect_curation()
        except PrerequisitePending as pending:
            stages[pending.stage] = {"status": "waiting", "detail": pending.detail}
            return {
                **report,
                "status": "waiting",
                "next_stage": pending.stage,
            }
        stages["curation"] = {
            key: value
            for key, value in curation.items()
            if key not in {"launch_config", "source_db", "source_checkpoint"}
        }

        selection_lock = self._selection_lock_held()
        if selection_lock:
            stages["selection"] = {
                "status": "waiting",
                "publisher_lock_held": True,
            }
            return {**report, "status": "waiting", "next_stage": "selection"}
        try:
            selection_state = self._inspect_selection(curation)
        except PrerequisitePending as pending:
            stages["selection"] = {
                "status": "waiting",
                "publisher_lock_held": True,
                "detail": pending.detail,
            }
            return {**report, "status": "waiting", "next_stage": "selection"}
        if selection_state is None:
            stages["selection"] = {
                "status": "ready",
                "publisher_lock_held": False,
            }
            self._run_action("publish-selection-v7", self._selection_command(curation))
            if not self.execute:
                return {**report, "status": "ready", "next_stage": "selection"}
            if self._selection_lock_held():
                raise PostCurationOrchestrationError(
                    "Selection publisher exited while retaining its publication lock"
                )
            try:
                selection_state = self._inspect_selection(curation)
            except PrerequisitePending as pending:
                stages["selection"] = {
                    "status": "waiting",
                    "publisher_lock_held": True,
                    "detail": pending.detail,
                }
                return {
                    **report,
                    "status": "waiting",
                    "next_stage": "selection",
                }
            if selection_state is None:
                raise PostCurationOrchestrationError(
                    "Selection publisher exited successfully without publishing a manifest"
                )
        selection, selection_sha = selection_state
        stages["selection"] = {
            "status": "complete",
            "manifest_sha256": selection_sha,
            "archives": len(selection["decision_shards"]),
        }

        try:
            supply = qualify_selection_supply(
                self.config.selection_root,
                sequence_length=self.config.sequence_length,
                targets={
                    "train": self.config.expected_train_input_tokens,
                    "validation": self.config.expected_validation_input_tokens,
                    "test": self.config.expected_test_input_tokens,
                },
            )
        except (OSError, ValueError, MaterializationError, RawTokenCacheInventoryError) as exc:
            raise PostCurationOrchestrationError(
                f"Selection supply qualification failed: {exc}"
            ) from exc
        stages["selection_supply"] = supply
        if supply.get("status") != "pass":
            return {
                **report,
                "status": "blocked",
                "next_stage": "selection-supply-remediation",
            }

        cache_status, _sources_unused = self._cache_status(selection)
        stages["raw_token_cache"] = cache_status
        if cache_status["status"] != "complete":
            stages["cache_inventory"] = {"status": "waiting"}
            return {
                **report,
                "status": "waiting",
                "next_stage": "raw-token-cache",
            }

        inventory = self._inspect_inventory(selection_sha)
        if inventory is None:
            stages["cache_inventory"] = {"status": "waiting"}
            self._run_action("publish-cache-inventory", self._inventory_command())
            if not self.execute:
                return {
                    **report,
                    "status": "ready",
                    "next_stage": "cache-inventory",
                }
            inventory = self._inspect_inventory(selection_sha)
            if inventory is None:
                raise PostCurationOrchestrationError(
                    "Cache inventory publisher exited successfully without publication"
                )
        stages["cache_inventory"] = {
            "status": "complete",
            "manifest_sha256": inventory.manifest_sha256,
            "archives": len(inventory.entries),
        }

        materializer = self._build_materializer(curation)
        materialization = self._inspect_materialization(materializer)
        stages["materialization"] = materialization
        if materialization["status"] == "complete":
            return {**report, "status": "complete", "next_stage": None}
        if materialization.get("lock_held"):
            return {
                **report,
                "status": "waiting",
                "next_stage": "materialization",
            }
        self._run_action(
            "materialize-stop-after-packing",
            self._materialization_command(curation),
        )
        if not self.execute:
            return {
                **report,
                "status": "ready",
                "next_stage": "materialization",
            }
        materialization = self._inspect_materialization(materializer)
        stages["materialization"] = materialization
        if materialization["status"] != "complete" or materialization["phase"] != "packed":
            raise PostCurationOrchestrationError(
                "Materializer exited successfully without a complete packed hand-off"
            )
        return {**report, "status": "complete", "next_stage": None}

    def run(self) -> dict[str, Any]:
        """Return only structured states or the orchestrator's fail-closed error."""

        try:
            return self._run()
        except PostCurationOrchestrationError:
            raise
        except Exception as exc:
            raise PostCurationOrchestrationError(
                "Unhandled post-curation OS/runtime boundary failure: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "PostCurationConfig",
    "PostCurationOrchestrationError",
    "PostCurationOrchestrator",
    "PrerequisitePending",
]
