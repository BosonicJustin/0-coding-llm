from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pretrain.materialize import MaterializationError, file_sha256
from pretrain.raw_token_cache import CacheConfig, discover_cache_jobs, run_cache_jobs
from pretrain.raw_token_cache_inventory import (
    InventorySource,
    publish_raw_token_cache_inventory,
)
from pretrain.raw_token_cache_reader import (
    ArchiveAuthority,
    FileAuthority,
    TokenizerAuthority,
)
from pretrain.tokenizer_identity import verify_tokenizer_identity
from tests import test_materialize_training_corpus as materialize_fixture


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class RawTokenCacheMaterializerTest(unittest.TestCase):
    def _prepare(
        self, root: Path
    ) -> tuple[materialize_fixture.MaterializationFixture, Path, Path]:
        fixture = materialize_fixture.MaterializationFixture(
            root / "source",
            tokenizer_repo_id="bigcode/starcoder2-tokenizer",
        )
        materialize_fixture.MaterializeTrainingCorpusTest._convert_fixture_to_all_eligible_profile(
            fixture
        )
        cache_root = root / "raw-token-cache"
        config = CacheConfig(
            expected_vocab_size=fixture.VOCAB_SIZE,
            max_documents_per_archive=100,
            max_document_bytes=1024,
            max_document_tokens=1024,
            tokenizer_batch_documents=8,
            tokenizer_batch_bytes=8 * 1024,
            tokenizer_batch_tokens=8 * 1024,
            max_manifest_member_bytes=1024 * 1024,
            max_json_line_bytes=64 * 1024,
            minimum_free_bytes=1,
        )
        jobs = discover_cache_jobs(
            fixture.root, fixture.preprocess, config=config
        )
        run_cache_jobs(
            jobs,
            cache_root,
            fixture.tokenizer_root,
            config=config,
            workers=1,
        )

        # Build the publication from authorities already authenticated by the
        # materializer's v7 selection/report inventory, not cache self-report.
        probe = fixture.materializer(root / "unused-probe")
        tokenizer_identity = verify_tokenizer_identity(
            fixture.tokenizer_root,
            expected_vocab_size=fixture.VOCAB_SIZE,
        )
        validation = probe.tokenizer_manifest["validation"]
        tokenizer = TokenizerAuthority(
            repo_id=probe.tokenizer_manifest["repo_id"],
            resolved_revision=probe.tokenizer_manifest["resolved_revision"],
            manifest_sha256=tokenizer_identity.manifest_sha256,
            vocabulary_sha256=tokenizer_identity.vocabulary_sha256,
            vocab_size=tokenizer_identity.vocab_size,
            eos_token=validation["eos_token"],
            eos_token_id=validation["eos_token_id"],
        )
        sources = [
            InventorySource(
                ordinal=archive.ordinal,
                archive=ArchiveAuthority(
                    path=archive.archive,
                    bucket=archive.bucket,
                    index=archive.archive_index,
                    bytes=archive.raw_path.stat().st_size,
                    sha256=archive.raw_sha256,
                ),
                preprocess_report=FileAuthority(
                    path=archive.report_relative,
                    bytes=archive.report_path.stat().st_size,
                    sha256=archive.report_sha256,
                ),
                fingerprint=FileAuthority(
                    path=archive.fingerprint_relative,
                    bytes=archive.fingerprint_path.stat().st_size,
                    sha256=archive.fingerprint_sha256,
                ),
                records=archive.documents,
                clean_bytes=archive.clean_bytes,
                content_tokens=archive.content_tokens,
            )
            for archive in probe.archives
        ]
        inventory_root = root / "raw-token-cache-inventory"
        publish_raw_token_cache_inventory(
            cache_root=cache_root,
            inventory_root=inventory_root,
            dataset_root=fixture.root,
            preprocess_root=fixture.preprocess,
            tokenizer_root=fixture.tokenizer_root,
            selection_manifest_sha256=file_sha256(
                fixture.selection / "manifest.json"
            ),
            selection_format_version=7,
            tokenizer=tokenizer,
            sources=sources,
        )
        return fixture, cache_root, inventory_root

    @staticmethod
    def _cached_materializer(
        fixture: materialize_fixture.MaterializationFixture,
        output: Path,
        cache_root: Path,
        inventory_root: Path,
        *,
        fault_injector: Any | None = None,
    ):
        return fixture.materializer(
            output,
            raw_token_cache_root=cache_root,
            raw_token_cache_inventory_root=inventory_root,
            fault_injector=fault_injector,
        )

    def test_multi_archive_cache_and_raw_paths_are_payload_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            raw_output = root / "raw-output"
            cache_output = root / "cache-output"
            raw = fixture.materializer(raw_output).run()
            cached = self._cached_materializer(
                fixture, cache_output, cache_root, inventory_root
            ).run()
            self.assertTrue(raw["complete"] and cached["complete"])

            for relative in ("packed", "orders", "provenance/documents"):
                self.assertEqual(
                    materialize_fixture.directory_bytes(raw_output / relative),
                    materialize_fixture.directory_bytes(cache_output / relative),
                    relative,
                )
            for name in ("source", "policy", "tokenizer", "fingerprints"):
                self.assertEqual(
                    (raw_output / f"provenance/{name}.json").read_bytes(),
                    (cache_output / f"provenance/{name}.json").read_bytes(),
                    name,
                )
            self.assertTrue(
                (cache_output / "provenance/raw_token_cache.json").is_file()
            )
            raw_manifest = raw["manifest"]
            cache_manifest = cached["manifest"]
            cache_identity = cache_manifest["identity"].pop("raw_token_cache")
            cache_provenance = cache_manifest["provenance"].pop(
                "raw_token_cache"
            )
            self.assertEqual(cache_manifest, raw_manifest)
            self.assertEqual(cache_identity["format"], "raw-token-cache-inventory")
            self.assertEqual(
                cache_provenance["path"], "provenance/raw_token_cache.json"
            )

    def test_inventory_cli_publishes_same_canonical_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            cli_inventory = root / "cli-inventory"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/publish_raw_token_cache_inventory.py",
                    "--root",
                    str(fixture.root),
                    "--preprocess-root",
                    str(fixture.preprocess),
                    "--selection-root",
                    str(fixture.selection),
                    "--tokenizer-root",
                    str(fixture.tokenizer_root),
                    "--cache-root",
                    str(cache_root),
                    "--output",
                    str(cli_inventory),
                    "--expected-vocab-size",
                    str(fixture.VOCAB_SIZE),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["archives"], 4)
            self.assertEqual(
                materialize_fixture.directory_bytes(cli_inventory),
                materialize_fixture.directory_bytes(inventory_root),
            )

    def test_cache_payload_corruption_fails_before_archive_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            inventory = json.loads(
                (inventory_root / "manifest.json").read_text(encoding="utf-8")
            )
            token_path = (
                cache_root / inventory["archives"][0]["cache_directory"] / "tokens.u16"
            )
            raw = bytearray(token_path.read_bytes())
            raw[0] ^= 1
            token_path.write_bytes(raw)
            output = root / "corrupt-output"
            with self.assertRaisesRegex(MaterializationError, "authentication failed"):
                self._cached_materializer(
                    fixture, output, cache_root, inventory_root
                ).run()
            journal = json.loads(
                (output / ".materialization-journal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(journal["state"]["completed_archives"], 0)
            inventory_manifest_sha = file_sha256(inventory_root / "manifest.json")
            self.assertEqual(
                journal["identity"]["raw_token_cache"]["manifest"]["sha256"],
                inventory_manifest_sha,
            )
            self.assertEqual(
                journal["identity"]["raw_token_cache"]["sidecar"]["sha256"],
                file_sha256(inventory_root / "manifest.sha256"),
            )
            self.assertTrue(
                all(
                    cursor["next_archive"] == 0
                    for cursor in journal["state"]["writer_cursors"].values()
                )
            )

    def test_source_mutation_during_cache_read_fails_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            inventory = json.loads(
                (inventory_root / "manifest.json").read_text(encoding="utf-8")
            )
            raw_path = fixture.root / inventory["archives"][0]["archive"]["path"]
            fired = False

            def mutate_source(event: str, _payload: Any) -> None:
                nonlocal fired
                if event == "document_added" and not fired:
                    fired = True
                    metadata = raw_path.stat()
                    os.utime(
                        raw_path,
                        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
                    )

            output = root / "mutated-source-output"
            with self.assertRaisesRegex(MaterializationError, "changed before checkpoint"):
                self._cached_materializer(
                    fixture,
                    output,
                    cache_root,
                    inventory_root,
                    fault_injector=mutate_source,
                ).run()
            journal = json.loads(
                (output / ".materialization-journal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(journal["state"]["completed_archives"], 0)

    def test_cache_crash_and_resume_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            fresh = root / "fresh"
            resumed = root / "resumed"
            self._cached_materializer(
                fixture, fresh, cache_root, inventory_root
            ).run()
            fired = False

            def interrupt(event: str, _payload: Any) -> None:
                nonlocal fired
                if event == "document_added" and not fired:
                    fired = True
                    raise RuntimeError("cache crash")

            with self.assertRaisesRegex(RuntimeError, "cache crash"):
                self._cached_materializer(
                    fixture,
                    resumed,
                    cache_root,
                    inventory_root,
                    fault_injector=interrupt,
                ).run()
            completed = self._cached_materializer(
                fixture, resumed, cache_root, inventory_root
            ).run()
            self.assertTrue(completed["complete"])
            self.assertEqual(
                materialize_fixture.directory_bytes(resumed),
                materialize_fixture.directory_bytes(fresh),
            )

    def test_cache_writer_checkpoint_crash_reconciles_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            fresh = root / "fresh"
            resumed = root / "resumed"
            self._cached_materializer(
                fixture, fresh, cache_root, inventory_root
            ).run()
            fired = False

            def interrupt(event: str, _payload: Any) -> None:
                nonlocal fired
                if event == "writer_checkpoint" and not fired:
                    fired = True
                    raise RuntimeError("cache checkpoint crash")

            with self.assertRaisesRegex(RuntimeError, "cache checkpoint crash"):
                self._cached_materializer(
                    fixture,
                    resumed,
                    cache_root,
                    inventory_root,
                    fault_injector=interrupt,
                ).run()
            journal = json.loads(
                (resumed / ".materialization-journal.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(journal["state"]["completed_archives"], 0)
            completed = self._cached_materializer(
                fixture, resumed, cache_root, inventory_root
            ).run()
            self.assertTrue(completed["complete"])
            self.assertEqual(
                materialize_fixture.directory_bytes(resumed),
                materialize_fixture.directory_bytes(fresh),
            )

    def test_inventory_selection_and_order_corruption_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            manifest_path = inventory_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["archives"][0], manifest["archives"][1] = (
                manifest["archives"][1], manifest["archives"][0]
            )
            manifest["archive_inventory_sha256"] = hashlib.sha256(
                canonical_json(manifest["archives"])
            ).hexdigest()
            manifest_raw = canonical_json(manifest)
            manifest_path.write_bytes(manifest_raw)
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            (inventory_root / "manifest.sha256").write_text(
                f"{manifest_sha}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(MaterializationError, "order/ordinal"):
                self._cached_materializer(
                    fixture, root / "bad-inventory", cache_root, inventory_root
                )

    def test_inventory_generation_change_cannot_resume_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, cache_root, inventory_root = self._prepare(root)
            output = root / "partial"
            partial = self._cached_materializer(
                fixture, output, cache_root, inventory_root
            ).run(max_archives=1)
            self.assertFalse(partial["complete"])
            manifest_path = inventory_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builder"]["implementation_sha256"] = "f" * 64
            manifest_raw = canonical_json(manifest)
            manifest_path.write_bytes(manifest_raw)
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            (inventory_root / "manifest.sha256").write_text(
                f"{manifest_sha}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(MaterializationError, "journal identity mismatch"):
                self._cached_materializer(
                    fixture, output, cache_root, inventory_root
                ).run()

    def test_cache_adapter_rejects_non_v7_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = materialize_fixture.MaterializationFixture(root / "source")
            with self.assertRaisesRegex(MaterializationError, "supplied together"):
                fixture.materializer(
                    root / "one-sided-output",
                    raw_token_cache_root=root / "cache",
                )
            with self.assertRaisesRegex(
                MaterializationError, "supported only for selection v7"
            ):
                # The paths need not exist: the v7 gate is evaluated first.
                fixture.materializer(
                    root / "output",
                    raw_token_cache_root=root / "cache",
                    raw_token_cache_inventory_root=root / "inventory",
                )


if __name__ == "__main__":
    unittest.main()
