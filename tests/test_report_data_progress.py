from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "report_data_progress.py"
SPEC = importlib.util.spec_from_file_location("report_data_progress_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def rewrite_cache_manifest(
    path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    raw = canonical_json(manifest)
    manifest_path.write_bytes(raw)
    (path / "manifest.sha256").write_text(
        f"{hashlib.sha256(raw).hexdigest()}  manifest.json\n", encoding="ascii"
    )


def safe_tree_state(root: Path) -> dict[str, tuple[int, int, int, int]]:
    result: dict[str, tuple[int, int, int, int]] = {}
    for path in sorted([root, *root.rglob("*")]):
        metadata = path.lstat()
        result[str(path.relative_to(root))] = (
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    return result


class ProgressFixture:
    def __init__(self, root: Path):
        self.root = root
        self.generation = root / "generation"
        self.preprocess = self.generation / "staging" / "preprocess"
        self.tokenizer_root = self.generation / "tokenizer" / "starcoder2"
        self.curation_output = root / "curation-output"
        self.work = self.curation_output / ".work"
        self.work.mkdir(parents=True)
        (self.curation_output / MODULE.CURATION_LOCK_FILE).write_bytes(b"")
        lease = {
            "lease_version": 1,
            "hostname": platform.node(),
            "pid": 101,
            "started_unix_ns": 1_000_000_000,
            "output": str(self.curation_output.resolve()),
        }
        lease["owner_token"] = hashlib.sha256(canonical_json(lease)).hexdigest()
        write_json(
            self.curation_output / MODULE.CURATION_LEASE_FILE,
            lease,
        )
        self.checkpoint = self.work / "CHECKPOINT.json"
        self.journal = self.work / "journal.jsonl"
        self.cache = root / "raw-cache"
        self.cache.mkdir()
        (self.cache / MODULE.CACHE_LOCK_FILE).write_bytes(b"")
        self.authority = {
            bucket: {
                "archives": 1,
                "documents": 2,
                "clean_bytes": 20,
                "exact_tokens": 100,
                "target_exact_tokens": 90,
            }
            for bucket in MODULE.BUCKETS
        }
        self._write_tokenizer()
        self.source_authority: dict[tuple[str, int], dict[str, object]] = {}
        self.inventory_descriptors: dict[str, dict[str, object]] = {}
        self.marker_files: list[dict[str, object]] = []
        self._write_generation()
        self.checkpoint_payload = self.make_checkpoint()
        self.publish_checkpoint()
        self.write_cache("python", 0)

    def _write_tokenizer(self) -> None:
        self.tokenizer_root.mkdir(parents=True)
        vocabulary = {"<|endoftext|>": 0}
        vocabulary.update(
            {f"token-{identifier:05d}": identifier for identifier in range(1, 49_152)}
        )
        tokenizer = {"model": {"vocab": vocabulary}, "added_tokens": []}
        tokenizer_path = self.tokenizer_root / "tokenizer.json"
        tokenizer_path.write_bytes(canonical_json(tokenizer))
        ordered = sorted(vocabulary.items(), key=lambda item: (item[1], item[0]))
        vocabulary_sha = hashlib.sha256(
            json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        manifest = {
            "manifest_version": 1,
            "repo_id": "bigcode/starcoder2-tokenizer",
            "requested_revision": "c" * 40,
            "resolved_revision": "c" * 40,
            "files": {
                "tokenizer.json": {
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                }
            },
            "validation": {
                "vocab_size": 49_152,
                "bos_token": None,
                "bos_token_id": None,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 0,
                "special_token_ids": {"<|endoftext|>": 0},
            },
        }
        manifest_path = self.tokenizer_root / "TOKENIZER_MANIFEST.json"
        write_json(manifest_path, manifest)
        self.tokenizer_descriptor = {
            "repo_id": "bigcode/starcoder2-tokenizer",
            "resolved_revision": "c" * 40,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "vocabulary_sha256": vocabulary_sha,
            "vocab_size": 49_152,
            "eos_token": "<|endoftext|>",
            "eos_token_id": 0,
            "eos_present_in_payload": False,
        }

    def _write_generation(self) -> None:
        report_identity = []
        fingerprint_identity = []
        raw_identity = []
        for bucket in MODULE.BUCKETS:
            index = 0
            archive_relative = (
                f"{MODULE.BUCKET_RAW_PATHS[bucket]}/part-{index:06d}.tar.zst"
            )
            archive_path = self.generation / archive_relative
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(f"archive:{bucket}\n".encode("ascii"))
            fingerprint_relative = (
                f"fingerprints/{bucket}/part-{index:06d}.jsonl.zst"
            )
            fingerprint_path = self.preprocess / fingerprint_relative
            fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
            fingerprint_path.write_bytes(f"fingerprint:{bucket}\n".encode("ascii"))
            report_relative = f"reports/{bucket}/part-{index:06d}.json"
            report_path = self.preprocess / report_relative
            report = {
                "report_version": 1,
                "fingerprint_version": 1,
                "policy_sha256": "9" * 64,
                "archive": archive_relative,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "archive_compressed_bytes": archive_path.stat().st_size,
                "bucket": bucket,
                "index": index,
                "quota_shard_id": f"fixture-{bucket}-{index:06d}",
                "source": f"fixture:{bucket}",
                "fingerprint_file": fingerprint_relative,
                "fingerprint_sha256": hashlib.sha256(
                    fingerprint_path.read_bytes()
                ).hexdigest(),
                "documents": 2,
                "clean_bytes": 20,
                "exact_tokens": 100,
            }
            write_json(report_path, report)
            report_raw = report_path.read_bytes()
            report_sha = hashlib.sha256(report_raw).hexdigest()
            source = {
                "archive": {
                    "path": archive_relative,
                    "bucket": bucket,
                    "index": index,
                    "bytes": archive_path.stat().st_size,
                    "sha256": report["archive_sha256"],
                },
                "preprocess_report": {
                    "path": report_relative,
                    "bytes": len(report_raw),
                    "sha256": report_sha,
                    "report_version": 1,
                },
                "fingerprint": {
                    "path": fingerprint_relative,
                    "bytes": fingerprint_path.stat().st_size,
                    "sha256": report["fingerprint_sha256"],
                    "fingerprint_version": 1,
                },
            }
            self.source_authority[(bucket, index)] = {
                "source": source,
                "documents": {
                    "records": 2,
                    "clean_bytes": 20,
                    "content_tokens": 100,
                },
                "report_sha256": report_sha,
            }
            report_identity.append({"path": report_relative, "sha256": report_sha})
            fingerprint_identity.append(
                {"path": fingerprint_relative, "sha256": report["fingerprint_sha256"]}
            )
            raw_identity.append(
                {
                    "archive": archive_relative,
                    "bytes": archive_path.stat().st_size,
                    "quota_shard_id": report["quota_shard_id"],
                    "sha256": report["archive_sha256"],
                }
            )
        for name, payload in (
            ("reports", report_identity),
            ("fingerprints", fingerprint_identity),
            ("raw_archives", raw_identity),
        ):
            self.inventory_descriptors[name] = {
                "count": len(payload),
                "inventory_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
            }
        self.inventory_descriptors["quota_records"] = {
            "count": len(report_identity),
            "inventory_sha256": "a" * 64,
        }
        marker_specs = (
            ("COLLECTION_COMPLETE.json", ["python", "other_code"]),
            ("ENGLISH_FINEWEB_EDU_COMPLETE.json", ["fineweb_edu"]),
            ("ENGLISH_WIKIPEDIA_COMPLETE.json", ["wikipedia"]),
        )
        for name, buckets in marker_specs:
            path = self.generation / name
            write_json(path, {"complete": True, "buckets": buckets})
            self.marker_files.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "buckets": buckets,
                }
            )

    def completeness(self) -> dict[str, object]:
        archives = sum(item["archives"] for item in self.authority.values())
        return {
            "format_version": 1,
            "complete": True,
            "legacy_dedup_index_required": False,
            "pending_inputs": 0,
            "preprocess_error_records": 0,
            "collection_targets_exact_tokens": {
                bucket: item["target_exact_tokens"]
                for bucket, item in self.authority.items()
            },
            "quota_records": dict(self.inventory_descriptors["quota_records"]),
            "completion_markers": {
                "count": 3,
                "inventory_sha256": hashlib.sha256(
                    canonical_json(self.marker_files)
                ).hexdigest(),
                "files": self.marker_files,
            },
            "raw_archives": dict(self.inventory_descriptors["raw_archives"]),
            "reports": dict(self.inventory_descriptors["reports"]),
            "fingerprints": dict(self.inventory_descriptors["fingerprints"]),
            "per_bucket": self.authority,
        }

    def inventory_subphase(
        self,
        bucket: str,
        index: int,
        *,
        status: str,
        rows: int,
        tokens: int,
        expected_rows: int = 2,
        expected_tokens: int = 100,
    ) -> dict[str, object]:
        expected = self.source_authority[(bucket, index)]
        result = {
            "subphase": f"inventory.archive.{bucket}.{index:06d}",
            "status": status,
            "cursor": {"input_rows": rows},
            "processed_rows": rows,
            "processed_tokens": tokens,
            "committed_batches": 1,
            "details": {
                "archive": f"raw/{bucket}/part-{index:06d}.tar.zst",
                "report_sha256": expected["report_sha256"],
                "expected_documents": expected_rows,
                "expected_tokens": expected_tokens,
                "expected_clean_bytes": 20,
                "clean_bytes": 20 if status == "complete" else 10,
            },
        }
        if status == "complete":
            result["details"]["validated_documents"] = rows
            result["details"]["validated_tokens"] = tokens
        result["details"]["archive"] = expected["source"]["archive"]["path"]
        return result

    def make_checkpoint(self) -> dict[str, object]:
        return {
            "checkpoint_version": 2,
            "database_version": 4,
            "phase": "inventory",
            "identity": {
                "collection_completeness": self.completeness(),
                "report_count": 4,
                "report_inventory_sha256": self.inventory_descriptors["reports"][
                    "inventory_sha256"
                ],
            },
            "last_event_sequence": 2,
            "counts": {
                "archives": 1,
                "documents": 3,
                "selected_documents": 0,
                "output_archives": 0,
            },
            "subphases": [
                self.inventory_subphase(
                    "python", 0, status="complete", rows=2, tokens=100
                ),
                self.inventory_subphase(
                    "other_code", 0, status="running", rows=1, tokens=40
                ),
            ],
            "storage": {
                "preflight": {
                    "status": "pass",
                    "documents_expected": 8,
                },
                "violation": {},
            },
        }

    def publish_checkpoint(self) -> None:
        write_json(self.checkpoint, self.checkpoint_payload)
        self.journal.write_text(
            json.dumps({"sequence": 1, "event": "initialized", "payload": {}})
            + "\n"
            + json.dumps(
                {
                    "sequence": 2,
                    "event": "archive_ingested",
                    "payload": {
                        "archive": self.source_authority[("python", 0)]["source"][
                            "archive"
                        ]["path"],
                        "documents": 2,
                        "tokens": 100,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tokenizer(self, *, revision: str = "c" * 40) -> dict[str, object]:
        return {**self.tokenizer_descriptor, "resolved_revision": revision}

    def cache_manifest(
        self,
        bucket: str,
        index: int,
        *,
        records: int = 2,
        tokens: int = 100,
        revision: str = "c" * 40,
    ) -> dict[str, object]:
        config = {
            "expected_vocab_size": 49_152,
            "max_documents_per_archive": 2_000_000,
            "max_document_bytes": 16 * 1024 * 1024,
            "max_document_tokens": 8 * 1024 * 1024,
            "tokenizer_batch_documents": 64,
            "tokenizer_batch_bytes": 32 * 1024 * 1024,
            "tokenizer_batch_tokens": 2 * 1024 * 1024,
            "max_manifest_member_bytes": 8 * 1024 * 1024 * 1024,
            "max_json_line_bytes": 1024 * 1024,
            "minimum_free_bytes": 10 * 1024 * 1024 * 1024,
        }
        contract_sha = hashlib.sha256(
            json.dumps(
                MODULE.TOKENIZATION_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "format": "raw-document-token-cache",
            "format_version": 1,
            "profile": "all-raw-documents-content-only-v1",
            "cache_complete": True,
            "training_ready": False,
            "tokenization": MODULE.TOKENIZATION_CONTRACT,
            "non_authorities": MODULE.NON_AUTHORITIES,
            "source": self.source_authority[(bucket, index)]["source"],
            "tokenizer": self.tokenizer(revision=revision),
            "documents": {
                "records": records,
                "clean_bytes": 20,
                "content_tokens": tokens,
                "alignment": {
                    "authority": "manifest-index+raw-manifest+preprocess-fingerprint",
                    "record_sha256": "4" * 64,
                    "offset_items": records + 1,
                    "offset_zero": 0,
                    "terminal_offset": tokens,
                },
            },
            "payloads": {
                "tokens": {
                    "path": "tokens.u16",
                    "dtype": "uint16",
                    "endianness": "little",
                    "items": tokens,
                    "bytes": tokens * 2,
                    "sha256": "5" * 64,
                    "minimum_id": 0,
                    "maximum_id": 12,
                },
                "offsets": {
                    "path": "offsets.u64",
                    "dtype": "uint64",
                    "endianness": "little",
                    "items": records + 1,
                    "bytes": (records + 1) * 8,
                    "sha256": "6" * 64,
                },
            },
            "builder": {
                "implementation": "pretrain.raw_token_cache",
                "implementation_sha256": "7" * 64,
                "config": config,
                "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
                "tokenization_contract_sha256": contract_sha,
            },
        }

    def write_cache(
        self,
        bucket: str,
        index: int,
        *,
        records: int = 2,
        tokens: int = 100,
        revision: str = "c" * 40,
    ) -> Path:
        target = self.cache / "archives" / bucket / f"part-{index:06d}"
        target.mkdir(parents=True, exist_ok=True)
        manifest = self.cache_manifest(
            bucket, index, records=records, tokens=tokens, revision=revision
        )
        raw = canonical_json(manifest)
        (target / "manifest.json").write_bytes(raw)
        (target / "manifest.sha256").write_text(
            f"{hashlib.sha256(raw).hexdigest()}  manifest.json\n", encoding="ascii"
        )
        (target / "tokens.u16").write_bytes(bytes(tokens * 2))
        (target / "offsets.u64").write_bytes(bytes((records + 1) * 8))
        return target

    def complete_everything(self) -> None:
        subphases = []
        for bucket in MODULE.BUCKETS:
            subphases.append(
                self.inventory_subphase(
                    bucket, 0, status="complete", rows=2, tokens=100
                )
            )
            if bucket != "python":
                self.write_cache(bucket, 0)
        self.checkpoint_payload["phase"] = "canonicalized"
        self.checkpoint_payload["identity"]["fast_all_eligible_handoff"] = dict(
            MODULE.FAST_ALL_ELIGIBLE_HANDOFF_PROFILE
        )
        self.checkpoint_payload["counts"] = {
            "archives": 4,
            "documents": 8,
            "selected_documents": 0,
            "output_archives": 0,
        }
        self.checkpoint_payload["subphases"] = subphases
        self.checkpoint_payload["last_event_sequence"] = 5
        self.publish_checkpoint()
        events = [{"sequence": 1, "event": "initialized", "payload": {}}]
        for sequence in range(2, 6):
            bucket = MODULE.BUCKETS[sequence - 2]
            events.append(
                {
                    "sequence": sequence,
                    "event": "archive_ingested",
                    "payload": {
                        "archive": self.source_authority[(bucket, 0)]["source"][
                            "archive"
                        ]["path"],
                        "documents": 2,
                        "tokens": 100,
                    },
                }
            )
        self.journal.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )


def live_process_probe(fixture: ProgressFixture) -> dict[str, object]:
    return {
        "available": True,
        "processes": [
            {
                "pid": 101,
                "cwd": str(PROJECT_ROOT),
                "argv": [
                    "python3",
                    str(PROJECT_ROOT / "scripts/curate_corpus.py"),
                    "--root",
                    str(fixture.generation),
                    "--output",
                    str(fixture.curation_output),
                ],
                "executable": sys.executable,
                "start_ticks": 1001,
            },
            {
                "pid": 102,
                "cwd": str(PROJECT_ROOT),
                "argv": [
                    "python3",
                    str(PROJECT_ROOT / "scripts/cache_raw_tokens.py"),
                    "build",
                    "--root",
                    str(fixture.generation),
                    "--output-root",
                    str(fixture.cache),
                ],
                "executable": sys.executable,
                "start_ticks": 1002,
            },
        ],
        "skipped": 0,
    }


def live_lock_probe(paths: Sequence[Path]) -> dict[str, object]:
    holders = {}
    for path in paths:
        resolved = str(path.resolve(strict=True))
        if path.name == MODULE.CURATION_LOCK_FILE:
            holders[resolved] = [101]
        elif path.name == MODULE.CACHE_LOCK_FILE:
            holders[resolved] = [102]
        else:
            holders[resolved] = []
    return {"available": True, "holders": holders}


def tmux_probe() -> dict[str, object]:
    return {
        "available": True,
        "sessions": [
            {
                "name": "curation",
                "panes": [{"pid": 101, "dead": False, "command": "python"}],
            },
            {
                "name": "cache",
                "panes": [{"pid": 102, "dead": False, "command": "python"}],
            },
        ],
    }


def clean_memory_probe() -> dict[str, object]:
    return {
        "available": True,
        "scope": "cgroup-v2",
        "current_bytes": 100,
        "limit_bytes": 1000,
        "working_set_estimate_bytes": 80,
        "events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
    }


class DataProgressReporterTest(unittest.TestCase):
    def report(self, fixture: ProgressFixture, **overrides: object) -> dict[str, object]:
        options: dict[str, object] = {
            "generation_root": fixture.generation,
            "curation_checkpoint": fixture.checkpoint,
            "curation_output_root": fixture.curation_output,
            "curation_journal": fixture.journal,
            "cache_root": fixture.cache,
            "minimum_free_bytes": 1,
            "process_probe": lambda: live_process_probe(fixture),
            "lock_probe": live_lock_probe,
            "tmux_probe": tmux_probe,
            "memory_probe": clean_memory_probe,
            "expected_tmux_sessions": ("curation", "cache"),
            "now": time.time(),
        }
        options.update(overrides)
        return MODULE.build_report(**options)

    def test_reports_exact_authority_relative_progress_and_runtime_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            report = self.report(fixture)
        self.assertEqual(report["status"], "healthy")
        self.assertTrue(report["read_only"])
        self.assertEqual(report["authority"]["documents"], 8)
        self.assertEqual(report["authority"]["content_tokens"], 400)
        self.assertEqual(report["curation"]["counts"]["documents"]["fraction"], "3/8")
        self.assertEqual(report["curation"]["counts"]["documents"]["percent"], 37.5)
        self.assertEqual(
            report["curation"]["counts"]["content_tokens"]["percent"], 35.0
        )
        self.assertEqual(
            report["raw_token_cache"]["counts"]["content_tokens"]["fraction"],
            "100/400",
        )
        self.assertEqual(
            report["raw_token_cache"]["per_bucket"]["python"]["content_tokens"][
                "percent"
            ],
            100.0,
        )
        self.assertEqual(
            report["runtime"]["processes"]["matches"]["curation"][0]["pid"],
            101,
        )
        self.assertTrue(report["runtime"]["filesystems"][0]["reserve_pass"])

    def test_accepts_manifest_emitted_by_real_raw_cache_builder(self) -> None:
        module_name = "reporter_real_raw_cache_fixture"
        raw_cache_test = PROJECT_ROOT / "tests" / "test_raw_token_cache.py"
        raw_spec = importlib.util.spec_from_file_location(module_name, raw_cache_test)
        assert raw_spec is not None and raw_spec.loader is not None
        raw_module = importlib.util.module_from_spec(raw_spec)
        sys.modules[module_name] = raw_module
        try:
            raw_spec.loader.exec_module(raw_module)
        except ModuleNotFoundError as exc:
            sys.modules.pop(module_name, None)
            if exc.name in {"zstandard", "tokenizers"}:
                self.skipTest(
                    f"raw cache integration dependency unavailable: {exc.name}"
                )
            raise
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = raw_module.RawTokenCacheFixture(root)
            cache_root = root / "built-cache"
            raw_module.run_cache_jobs(
                [source.job()],
                cache_root,
                source.tokenizer_root,
                config=source.config,
            )
            target = source.job().target(cache_root)
            item, _metadata = MODULE._cache_manifest(
                target, bucket="python", index=0
            )
            expected_tokens = source.job().exact_tokens
        self.assertEqual(item["records"], 2)
        self.assertEqual(item["content_tokens"], expected_tokens)

    def test_reporter_does_not_modify_any_inspected_file_or_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            before = safe_tree_state(fixture.root)
            self.report(fixture)
            after = safe_tree_state(fixture.root)
        self.assertEqual(after, before)

    def test_complete_fast_handoff_and_cache_need_no_live_process_and_may_be_old(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.complete_everything()
            old = time.time() - 100_000
            os.utime(fixture.checkpoint, (old, old))
            for manifest in fixture.cache.rglob("manifest.json"):
                os.utime(manifest, (old, old))
            report = self.report(
                fixture,
                now=time.time(),
                process_probe=lambda: {"available": True, "processes": []},
                expected_tmux_sessions=(),
            )
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["curation"]["terminal"])
        self.assertTrue(report["raw_token_cache"]["complete"])
        self.assertEqual(
            report["raw_token_cache"]["counts"]["content_tokens"]["percent"],
            100.0,
        )

    def test_duplicate_checkpoint_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint.write_text(
                '{"checkpoint_version":2,"checkpoint_version":2}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(MODULE.DataProgressError, "duplicate JSON key"):
                self.report(fixture)

    def test_stale_active_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            modified = fixture.checkpoint.stat().st_mtime
            with self.assertRaisesRegex(MODULE.DataProgressError, "checkpoint is stale"):
                self.report(
                    fixture,
                    now=modified + 101,
                    checkpoint_stale_seconds=100,
                    cache_stale_seconds=10_000,
                )

    def test_checkpoint_journal_sequence_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint_payload["last_event_sequence"] = 3
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(MODULE.DataProgressError, "incoherent"):
                self.report(fixture)

    def test_malformed_collection_authority_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint_payload["identity"]["collection_completeness"][
                "pending_inputs"
            ] = 1
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(MODULE.DataProgressError, "not closed and clean"):
                self.report(fixture)

    def test_json_bool_never_substitutes_for_authenticated_integer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint_payload["checkpoint_version"] = True
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "checkpoint version must be an integer"
            ):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint_payload["identity"]["collection_completeness"][
                "pending_inputs"
            ] = False
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(MODULE.DataProgressError, "not closed and clean"):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            report_path = fixture.preprocess / "reports/python/part-000000.json"
            generation_report = json.loads(report_path.read_bytes())
            generation_report["index"] = False
            write_json(report_path, generation_report)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "wrong preprocess report part identity"
            ):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"

            def mutate(manifest: dict[str, object]) -> None:
                manifest["source"]["archive"]["index"] = False

            rewrite_cache_manifest(target, mutate)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "cache archive identity/path mismatch"
            ):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"

            def mutate(manifest: dict[str, object]) -> None:
                manifest["documents"]["alignment"]["offset_zero"] = False

            rewrite_cache_manifest(target, mutate)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "cache alignment authority is malformed"
            ):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"

            def mutate(manifest: dict[str, object]) -> None:
                builder = manifest["builder"]
                builder["config"]["expected_vocab_size"] = True
                builder["config_sha256"] = hashlib.sha256(
                    canonical_json(builder["config"])
                ).hexdigest()

            rewrite_cache_manifest(target, mutate)
            with self.assertRaisesRegex(
                MODULE.DataProgressError,
                "cache builder config expected_vocab_size must be an integer",
            ):
                self.report(fixture)

    def test_fast_handoff_profile_rejects_bool_for_contract_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            profile = dict(MODULE.FAST_ALL_ELIGIBLE_HANDOFF_PROFILE)
            profile["contract_version"] = True
            fixture.checkpoint_payload["identity"]["fast_all_eligible_handoff"] = profile
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(MODULE.DataProgressError, "profile identity"):
                self.report(fixture)

    def test_inventory_token_counter_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.checkpoint_payload["subphases"][0]["processed_tokens"] = 99
            fixture.publish_checkpoint()
            with self.assertRaisesRegex(MODULE.DataProgressError, "incomplete"):
                self.report(fixture)

    def test_bad_cache_manifest_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            sidecar = (
                fixture.cache
                / "archives/python/part-000000/manifest.sha256"
            )
            sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
            with self.assertRaisesRegex(MODULE.DataProgressError, "sidecar mismatch"):
                self.report(fixture)

    def test_cache_payload_size_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            token_file = fixture.cache / "archives/python/part-000000/tokens.u16"
            token_file.write_bytes(b"short")
            with self.assertRaisesRegex(MODULE.DataProgressError, "payload size differs"):
                self.report(fixture)

    def test_mixed_cache_tokenizer_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.write_cache("other_code", 0, revision="f" * 40)
            with self.assertRaisesRegex(MODULE.DataProgressError, "tokenizer authority"):
                self.report(fixture)

    def test_cache_authority_overshoot_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"
            for path in target.iterdir():
                path.unlink()
            target.rmdir()
            fixture.write_cache("python", 0, tokens=101)
            with self.assertRaisesRegex(MODULE.DataProgressError, "generation/report"):
                self.report(fixture)

    def test_wrong_cache_part_cannot_substitute_for_authoritative_part(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            original = fixture.cache / "archives/python/part-000000"
            substituted = fixture.cache / "archives/python/part-000001"
            original.rename(substituted)

            def mutate(manifest: dict[str, object]) -> None:
                source = manifest["source"]
                source["archive"]["index"] = 1
                source["archive"]["path"] = "raw/python/part-000001.tar.zst"
                source["preprocess_report"]["path"] = (
                    "reports/python/part-000001.json"
                )
                source["fingerprint"]["path"] = (
                    "fingerprints/python/part-000001.jsonl.zst"
                )

            rewrite_cache_manifest(substituted, mutate)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "no authoritative generation part identity"
            ):
                self.report(fixture)

    def test_cache_source_digest_cannot_self_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"

            def mutate(manifest: dict[str, object]) -> None:
                manifest["source"]["fingerprint"]["sha256"] = "0" * 64

            rewrite_cache_manifest(target, mutate)
            with self.assertRaisesRegex(MODULE.DataProgressError, "generation/report"):
                self.report(fixture)

    def test_missing_generation_report_part_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            report = fixture.preprocess / "reports/python/part-000000.json"
            report.unlink()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "report totals|reports inventory"
            ):
                self.report(fixture)

    def test_authoritative_tokenizer_rejects_self_asserted_cache_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            target = fixture.cache / "archives/python/part-000000"

            def mutate(manifest: dict[str, object]) -> None:
                manifest["tokenizer"]["resolved_revision"] = "f" * 40

            rewrite_cache_manifest(target, mutate)
            with self.assertRaisesRegex(MODULE.DataProgressError, "tokenizer authority"):
                self.report(fixture)

    def test_complete_cache_rejects_in_flight_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.complete_everything()
            stage = (
                fixture.cache
                / "archives/python/.part-000000.building-stale-generation"
            )
            stage.mkdir()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "in-flight cache directory"
            ):
                self.report(fixture)

    def test_missing_cache_part_never_reports_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.complete_everything()
            target = fixture.cache / "archives/wikipedia/part-000000"
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
            report = self.report(fixture)
        self.assertFalse(report["raw_token_cache"]["complete"])
        self.assertNotEqual(report["status"], "complete")

    def test_wrong_in_flight_part_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            stage = fixture.cache / "archives/python/.part-999999.building-live"
            stage.mkdir()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "in-flight cache directory"
            ):
                self.report(fixture)

    def test_duplicate_in_flight_part_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            bucket = fixture.cache / "archives/other_code"
            bucket.mkdir(parents=True)
            (bucket / ".part-000000.building-first").mkdir()
            (bucket / ".part-000000.building-second").mkdir()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "in-flight cache directory"
            ):
                self.report(fixture)

    def test_cache_global_inventory_change_during_scan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            original = MODULE._cache_manifest
            changed = False

            def racing_manifest(
                target: Path, *, bucket: str, index: int
            ) -> tuple[dict[str, object], os.stat_result]:
                nonlocal changed
                result = original(target, bucket=bucket, index=index)
                if not changed:
                    changed = True
                    (fixture.cache / "archives/python/part-000001").mkdir()
                return result

            with mock.patch.object(
                MODULE, "_cache_manifest", side_effect=racing_manifest
            ):
                with self.assertRaisesRegex(
                    MODULE.DataProgressError, "inventory changed during global scan"
                ):
                    self.report(fixture)

    def test_process_output_never_contains_raw_arguments_or_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            secret = "super-secret-token-value"
            evidence = live_process_probe(fixture)
            evidence["processes"][0]["argv"].extend(["--api-key", secret])
            evidence["processes"][1]["argv"].extend(["--password", secret])
            report = self.report(fixture, process_probe=lambda: evidence)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(secret, serialized)
        self.assertNotIn('"argv"', serialized)
        self.assertNotIn('"command"', serialized)
        self.assertEqual(
            report["runtime"]["processes"]["matches"]["curation"],
            [
                {
                    "pid": 101,
                    "script": "curate_corpus.py",
                    "ownership": "lease+flock+proc",
                }
            ],
        )

    def test_process_liveness_requires_exact_configured_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            wrong_output = fixture.root / "wrong-curation-output"
            wrong_output.mkdir()
            evidence = live_process_probe(fixture)
            curation_argv = evidence["processes"][0]["argv"]
            curation_argv[curation_argv.index("--output") + 1] = str(wrong_output)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease/lock owner is not the exact"
            ):
                self.report(fixture, process_probe=lambda: evidence)

    def test_process_liveness_requires_exact_script_and_generation_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            wrong_root = fixture.root / "wrong-generation"
            wrong_root.mkdir()
            evidence = live_process_probe(fixture)
            curation_argv = evidence["processes"][0]["argv"]
            curation_argv[curation_argv.index("--root") + 1] = str(wrong_root)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease/lock owner is not the exact"
            ):
                self.report(fixture, process_probe=lambda: evidence)

    def test_process_argv_without_authoritative_lock_owner_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))

            def forged_lock_probe(paths: Sequence[Path]) -> dict[str, object]:
                evidence = live_lock_probe(paths)
                curation_lock = str(
                    (fixture.curation_output / MODULE.CURATION_LOCK_FILE).resolve()
                )
                evidence["holders"][curation_lock] = [999]
                return evidence

            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease PID differs from advisory-lock owner"
            ):
                self.report(fixture, lock_probe=forged_lock_probe)

    def test_unavailable_lock_proof_is_unknown_never_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            report = self.report(
                fixture,
                lock_probe=lambda _paths: {
                    "available": False,
                    "reason": "no authoritative lock inventory",
                },
            )
        self.assertEqual(report["status"], "warning")
        self.assertEqual(
            report["runtime"]["processes"]["matches"],
            {"curation": [], "raw_token_cache": []},
        )
        self.assertIn("lock_ownership_unavailable", report["warnings"])
        self.assertIn("liveness_authority_unavailable", report["warnings"])

    def test_liveness_rejects_missing_or_wrong_curator_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            (fixture.curation_output / MODULE.CURATION_LEASE_FILE).unlink()
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "cannot open curation ownership lease"
            ):
                self.report(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            lease_path = fixture.curation_output / MODULE.CURATION_LEASE_FILE
            lease = json.loads(lease_path.read_bytes())
            lease["lease_version"] = True
            authority = {
                key: value for key, value in lease.items() if key != "owner_token"
            }
            lease["owner_token"] = hashlib.sha256(
                canonical_json(authority)
            ).hexdigest()
            write_json(lease_path, lease)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "curation lease version must be an integer"
            ):
                self.report(fixture)

    def test_liveness_rejects_non_python_executable_and_process_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            evidence = live_process_probe(fixture)
            evidence["processes"][0]["executable"] = "/bin/sh"
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease/lock owner is not the exact"
            ):
                self.report(fixture, process_probe=lambda: evidence)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            calls = 0

            def racing_process_probe() -> dict[str, object]:
                nonlocal calls
                calls += 1
                evidence = live_process_probe(fixture)
                if calls > 1:
                    evidence["processes"][0]["start_ticks"] = 2001
                return evidence

            with self.assertRaisesRegex(
                MODULE.DataProgressError, "process ownership changed while inspected"
            ):
                self.report(fixture, process_probe=racing_process_probe)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            spoof = fixture.root / "fake-curate_corpus.py"
            spoof.write_text("# not the configured script\n", encoding="utf-8")
            evidence = live_process_probe(fixture)
            evidence["processes"][0]["argv"][1] = str(spoof)
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease/lock owner is not the exact"
            ):
                self.report(fixture, process_probe=lambda: evidence)

    def test_checkpoint_must_belong_to_configured_curation_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            unrelated_output = fixture.root / "unrelated-output"
            unrelated_output.mkdir()
            with self.assertRaisesRegex(MODULE.DataProgressError, "outside.*output root"):
                self.report(fixture, curation_output_root=unrelated_output)

    def test_fast_handoff_profile_must_match_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.complete_everything()
            del fixture.checkpoint_payload["identity"]["fast_all_eligible_handoff"][
                "publisher"
            ]
            fixture.publish_checkpoint()
            fixture.journal.write_text(
                "".join(
                    json.dumps(
                        {
                            "sequence": index + 1,
                            "event": "initialized" if index == 0 else "archive_ingested",
                            "payload": (
                                {}
                                if index == 0
                                else {
                                    "archive": fixture.source_authority[
                                        (MODULE.BUCKETS[index - 1], 0)
                                    ]["source"]["archive"]["path"],
                                    "documents": 2,
                                    "tokens": 100,
                                }
                            ),
                        }
                    )
                    + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.DataProgressError, "profile identity"):
                self.report(fixture)

    def test_terminal_curation_rejects_running_subphase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            fixture.complete_everything()
            fixture.checkpoint_payload["subphases"][0]["status"] = "running"
            fixture.publish_checkpoint()
            fixture.journal.write_text(
                "".join(
                    json.dumps(
                        {
                            "sequence": index + 1,
                            "event": "initialized" if index == 0 else "archive_ingested",
                            "payload": (
                                {}
                                if index == 0
                                else {
                                    "archive": fixture.source_authority[
                                        (MODULE.BUCKETS[index - 1], 0)
                                    ]["source"]["archive"]["path"],
                                    "documents": 2,
                                    "tokens": 100,
                                }
                            ),
                        }
                    )
                    + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.DataProgressError, "running subphase"):
                self.report(fixture)

    def test_stable_file_reader_detects_in_place_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority.json"
            path.write_bytes(b'{"value":1}\n')
            original_fstat = MODULE.os.fstat
            calls = 0

            def racing_fstat(descriptor: int) -> os.stat_result:
                nonlocal calls
                result = original_fstat(descriptor)
                calls += 1
                if calls == 1:
                    path.write_bytes(b'{"value":2}\n')
                return result

            with mock.patch.object(MODULE.os, "fstat", side_effect=racing_fstat):
                with self.assertRaisesRegex(MODULE.DataProgressError, "changed while read"):
                    MODULE._stable_file_bytes(
                        path, label="test authority", maximum_bytes=1024
                    )

    def test_missing_required_process_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            with self.assertRaisesRegex(
                MODULE.DataProgressError, "lease/lock owner is not the exact"
            ):
                self.report(
                    fixture,
                    process_probe=lambda: {"available": True, "processes": []},
                )

    def test_nonzero_oom_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            with self.assertRaisesRegex(MODULE.DataProgressError, "OOM evidence"):
                self.report(
                    fixture,
                    memory_probe=lambda: {
                        "available": True,
                        "events": {"oom": 0, "oom_kill": 1},
                    },
                )

    def test_missing_required_tmux_session_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            with self.assertRaisesRegex(MODULE.DataProgressError, "tmux session is missing"):
                self.report(
                    fixture,
                    tmux_probe=lambda: {"available": True, "sessions": []},
                )

    def test_filesystem_reserve_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            with self.assertRaisesRegex(MODULE.DataProgressError, "safety reserve failed"):
                self.report(fixture, minimum_free_bytes=10**30)

    def test_unavailable_optional_host_probes_are_explicit_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProgressFixture(Path(temporary))
            report = self.report(
                fixture,
                process_probe=lambda: {"available": False, "reason": "no proc"},
                tmux_probe=lambda: {"available": False, "reason": "no tmux"},
                memory_probe=lambda: {"available": False, "reason": "no cgroup"},
            )
        self.assertEqual(report["status"], "warning")
        self.assertEqual(
            report["warnings"],
            [
                "process_health_unavailable",
                "liveness_authority_unavailable",
                "tmux_health_unavailable",
                "memory_health_unavailable",
            ],
        )


if __name__ == "__main__":
    unittest.main()
