from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tokenizers import Tokenizer, models

from posttrain.sft_data import (
    REQUIRED_COLUMNS,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    file_sha256,
    prompt_group_id,
    split_for_group,
    SplitPolicy,
)
from scripts import prepare_prime_sft as prepare


class InlineExecutor:
    """ProcessPoolExecutor-compatible deterministic test double."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> "InlineExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def submit(self, function, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:  # make as_completed surface worker errors
            future.set_exception(exc)
        return future


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _row(index: int, *, prompt: str, completion: str) -> dict[str, object]:
    return {
        "id": f"{index:032x}",
        "input": prompt,
        "output": completion,
        "domain": "generic",
        "generation_algorithm": "self-instruct",
        "llm_judgement": "metadata is deliberately not copied",
        "unit_tests": json.dumps(["assert solution() is not None"]),
        "tests_execution_status": json.dumps(["pass"]),
        "average_test_score": 1.0,
    }


class PreparePrimeSFTIntegrationTests(unittest.TestCase):
    def _write_tokenizer(self, root: Path) -> None:
        root.mkdir(parents=True)
        tokenizer = Tokenizer(
            models.WordLevel(
                vocab={"<|endoftext|>": 0, "<unk>": 1},
                unk_token="<unk>",
            )
        )
        tokenizer.save(str(root / "tokenizer.json"))
        manifest = {
            "repo_id": TOKENIZER_REPO_ID,
            "resolved_revision": TOKENIZER_REVISION,
            "validation": {"vocab_size": 2, "eos_token_id": 0},
            "files": {
                "tokenizer.json": {
                    "bytes": (root / "tokenizer.json").stat().st_size,
                    "sha256": file_sha256(root / "tokenizer.json"),
                }
            },
        }
        (root / "TOKENIZER_MANIFEST.json").write_bytes(_json_bytes(manifest))

    def _write_source(self, root: Path, rows_by_shard: list[list[dict[str, object]]]) -> None:
        data = root / "raw" / "data"
        data.mkdir(parents=True)
        schema = pa.schema(
            [
                *[
                    pa.field(column, pa.string(), nullable=False)
                    for column in REQUIRED_COLUMNS[:-1]
                ],
                pa.field("average_test_score", pa.float64(), nullable=False),
            ]
        )
        records = []
        for index, rows in enumerate(rows_by_shard):
            relative = f"data/train-{index:05d}-of-{len(rows_by_shard):05d}.parquet"
            path = root / "raw" / relative
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
            records.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "rows": len(rows),
                    "sha256": file_sha256(path),
                }
            )
        inventory_core = {
            "inventory_version": 1,
            "files": records,
            "repository_metadata": [],
            "train_parquet_files": len(records),
            "compressed_download_bytes": sum(record["bytes"] for record in records),
            "rows": sum(record["rows"] for record in records),
        }
        inventory = {
            **inventory_core,
            "inventory_sha256": hashlib.sha256(
                json.dumps(
                    inventory_core,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        source = {
            "manifest_version": 1,
            "kind": "raw_sft_dataset_snapshot",
            "repo_id": SOURCE_REPO_ID,
            "repo_type": "dataset",
            "requested_revision": SOURCE_REVISION,
            "resolved_revision": SOURCE_REVISION,
            "raw_subdirectory": "raw",
            "allow_patterns": [
                "data/train-*.parquet",
                "README.md",
                ".gitattributes",
            ],
            "raw_files_preserved": True,
            "expected": {
                "train_parquet_files": len(records),
                "compressed_download_bytes": inventory["compressed_download_bytes"],
                "rows": inventory["rows"],
            },
            "inventory": inventory,
        }
        source_bytes = _json_bytes(source)
        (root / "SOURCE.json").write_bytes(source_bytes)
        completion = {
            "completion_version": 1,
            "kind": "raw_sft_dataset_snapshot",
            "status": "complete",
            "source_manifest": "SOURCE.json",
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "repo_id": SOURCE_REPO_ID,
            "repo_type": "dataset",
            "resolved_revision": SOURCE_REVISION,
            "raw_subdirectory": "raw",
            "raw_files_preserved": True,
            "train_parquet_files": len(records),
            "compressed_download_bytes": inventory["compressed_download_bytes"],
            "rows": inventory["rows"],
            "inventory_sha256": inventory["inventory_sha256"],
        }
        (root / "COMPLETION.json").write_bytes(_json_bytes(completion))

    @staticmethod
    def _prompt_for_split(split: str, policy: SplitPolicy) -> str:
        for index in range(10_000):
            prompt = f"Independent coding prompt {index}"
            if split_for_group(prompt_group_id(prompt), policy) == split:
                return prompt
        raise AssertionError(f"could not synthesize a deterministic {split} prompt")

    def _build_fixture(self, temporary_root: Path) -> dict[str, object]:
        raw_root = temporary_root / "opencodeinstruct"
        tokenizer_root = temporary_root / "tokenizer"
        output = raw_root / "derived" / "prime-sft-v1"
        policy_path = temporary_root / "policy.json"
        denylist_path = temporary_root / "mbpp_denylist.json"
        self._write_tokenizer(tokenizer_root)

        split_policy = SplitPolicy(
            seed="prime-sft-integration-test",
            validation_per_million=500_000,
        )
        rows = [
            [
                _row(
                    1,
                    prompt="Write an otherwise harmless shared function.",
                    completion="MBPP benchmark payload",
                ),
                _row(
                    2,
                    prompt=self._prompt_for_split("train", split_policy),
                    completion="def solution(): return 1",
                ),
            ],
            [
                _row(
                    3,
                    prompt="Write an otherwise harmless shared function.",
                    completion="def solution(): return 2",
                ),
                _row(
                    4,
                    prompt=self._prompt_for_split("validation", split_policy),
                    completion="def solution(): return 3",
                ),
            ],
        ]
        self._write_source(raw_root, rows)
        policy = json.loads(prepare.DEFAULT_POLICY.read_text(encoding="utf-8"))
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "configs" / "mbpp_denylist.json",
            denylist_path,
        )
        policy["benchmark_denylists"] = [str(denylist_path)]
        policy["split"] = {
            "seed": split_policy.seed,
            "validation_per_million": split_policy.validation_per_million,
        }
        policy_path.write_bytes(_json_bytes(policy))
        total_bytes = sum(
            path.stat().st_size
            for path in (raw_root / "raw" / "data").glob("*.parquet")
        )
        return {
            "raw_root": raw_root,
            "tokenizer_root": tokenizer_root,
            "output": output,
            "policy_path": policy_path,
            "denylist_path": denylist_path,
            "patches": {
                "EXPECTED_SOURCE_SHARDS": 2,
                "EXPECTED_SOURCE_ROWS": 4,
                "EXPECTED_SOURCE_BYTES": total_bytes,
                "TOKENIZER_VOCAB_SIZE": 2,
            },
        }

    def test_end_to_end_publication_is_hf_loadable_restartable_and_group_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prime-sft-curation-") as temporary:
            temporary_root = Path(temporary)
            fixture = self._build_fixture(temporary_root)
            raw_root = fixture["raw_root"]
            tokenizer_root = fixture["tokenizer_root"]
            output = fixture["output"]
            policy_path = fixture["policy_path"]
            denylist_path = fixture["denylist_path"]
            patches = fixture["patches"]
            assert isinstance(raw_root, Path)
            assert isinstance(tokenizer_root, Path)
            assert isinstance(output, Path)
            assert isinstance(policy_path, Path)
            assert isinstance(denylist_path, Path)
            assert isinstance(patches, dict)
            raw_checksums = {
                path.name: file_sha256(path)
                for path in sorted((raw_root / "raw" / "data").glob("*.parquet"))
            }

            with (
                mock.patch.multiple(prepare, **patches),
                mock.patch.object(
                    prepare.concurrent.futures,
                    "ProcessPoolExecutor",
                    InlineExecutor,
                ),
            ):
                published = prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )
                repeated = prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )
                independently_validated = prepare.validate_published_dataset(output)

            self.assertFalse(published["already_complete"])
            self.assertTrue(repeated["already_complete"])
            self.assertEqual(
                independently_validated["curation_identity_sha256"],
                published["manifest"]["identity"]["curation_identity_sha256"],
            )
            self.assertTrue((output / "COMPLETION.json").is_file())
            self.assertEqual(
                raw_checksums,
                {
                    path.name: file_sha256(path)
                    for path in sorted((raw_root / "raw" / "data").glob("*.parquet"))
                },
            )

            manifest = json.loads((output / "DATASET_MANIFEST.json").read_text())
            self.assertEqual(manifest["statistics"]["counts"]["train"], 1)
            self.assertEqual(manifest["statistics"]["counts"]["validation"], 1)
            rejection_counts = {
                key: value
                for key, value in manifest["statistics"]["counts"].items()
                if key.startswith("rejected:")
            }
            self.assertEqual(sum(rejection_counts.values()), 2)
            self.assertIn("rejected:benchmark:group-propagated", rejection_counts)
            self.assertEqual(manifest["identity"]["contaminated_groups"], 1)
            self.assertEqual(
                set(manifest["identity"]["implementation"]["files"]),
                set(prepare.IMPLEMENTATION_FILES),
            )
            self.assertEqual(
                set(manifest["identity"]["implementation"]["packages"]),
                set(prepare.IMPLEMENTATION_PACKAGES),
            )

            cache = temporary_root / "hf-cache"
            train = load_dataset(str(output), split="train", cache_dir=str(cache))
            validation = load_dataset(
                str(output), split="validation", cache_dir=str(cache)
            )
            self.assertEqual(len(train), 1)
            self.assertEqual(len(validation), 1)
            for example in (train[0], validation[0]):
                self.assertEqual([message["role"] for message in example["messages"]], ["user", "assistant"])
                self.assertNotIn("unit_tests", example)
                self.assertNotIn("llm_judgement", example)

            rejected = pq.read_table(
                sorted((output / "audit").glob("rejected-*.parquet"))
            ).to_pylist()
            self.assertEqual(len(rejected), 2)
            self.assertTrue(all("reason" in row and "messages" not in row for row in rejected))

            # A completed publication is re-authenticated as an exact tree; a
            # file matching HF's split glob cannot be smuggled into a cache hit.
            rogue = output / "data" / "train-rogue.parquet"
            shutil.copy2(sorted((output / "data").glob("train-*.parquet"))[0], rogue)
            with (
                mock.patch.multiple(prepare, **patches),
                self.assertRaisesRegex(prepare.SFTPreparationError, "rogue"),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

            rogue.unlink()

            # The denylist digest is part of the curation identity even when
            # its configured pathname remains unchanged.
            denylist_path.write_bytes(denylist_path.read_bytes() + b"\n")
            with (
                mock.patch.multiple(prepare, **patches),
                self.assertRaisesRegex(
                    prepare.SFTPreparationError,
                    "different curation identity",
                ),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

    def test_interrupted_resume_binds_implementation_and_rejects_corruption_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prime-sft-resume-") as temporary:
            temporary_root = Path(temporary)
            fixture = self._build_fixture(temporary_root)
            raw_root = fixture["raw_root"]
            tokenizer_root = fixture["tokenizer_root"]
            output = fixture["output"]
            policy_path = fixture["policy_path"]
            patches = fixture["patches"]
            assert isinstance(raw_root, Path)
            assert isinstance(tokenizer_root, Path)
            assert isinstance(output, Path)
            assert isinstance(policy_path, Path)
            assert isinstance(patches, dict)

            real_process = prepare._process_shard

            def interrupt_after_first(task: dict[str, object]) -> dict[str, object]:
                if task["index"] == 1:
                    raise RuntimeError("synthetic interruption")
                return real_process(task)

            with (
                mock.patch.multiple(prepare, **patches),
                mock.patch.object(
                    prepare.concurrent.futures,
                    "ProcessPoolExecutor",
                    InlineExecutor,
                ),
                mock.patch.object(
                    prepare,
                    "_process_shard",
                    side_effect=interrupt_after_first,
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic interruption"),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

            work = output.with_name(f".{output.name}.work")
            self.assertTrue((work / "audit" / "shard-00000.json").is_file())
            self.assertFalse((work / "audit" / "shard-00001.json").exists())

            # A hard process/pod loss can leave an unpublished Parquet file.
            # Resume must remove the exact deterministic partial for the
            # missing source shard before it is resubmitted.
            with mock.patch.multiple(prepare, **patches):
                stale_partial = prepare._partial_output_path(
                    prepare._output_record(work, "train", 1)
                )
            stale_partial.write_bytes(b"synthetic interrupted parquet")

            changed_implementation = json.loads(
                json.dumps(prepare._implementation_identity())
            )
            changed_implementation["packages"]["pyarrow"] += "+changed"
            with (
                mock.patch.multiple(prepare, **patches),
                mock.patch.object(
                    prepare,
                    "_implementation_identity",
                    return_value=changed_implementation,
                ),
                self.assertRaisesRegex(
                    prepare.SFTPreparationError,
                    "different curation identity",
                ),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

            resumed_indices: list[int] = []

            def observe_missing_shard(task: dict[str, object]) -> dict[str, object]:
                self.assertFalse(stale_partial.exists())
                resumed_indices.append(int(task["index"]))
                raise RuntimeError("resume probe")

            with (
                mock.patch.multiple(prepare, **patches),
                mock.patch.object(
                    prepare.concurrent.futures,
                    "ProcessPoolExecutor",
                    InlineExecutor,
                ),
                mock.patch.object(
                    prepare,
                    "_process_shard",
                    side_effect=observe_missing_shard,
                ),
                self.assertRaisesRegex(RuntimeError, "resume probe"),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )
            self.assertEqual(resumed_indices, [1])
            self.assertFalse(stale_partial.exists())

            corrupt_sidecar = work / "audit" / "shard-00000.json"
            corrupt_payload = json.loads(corrupt_sidecar.read_text(encoding="utf-8"))
            corrupt_payload["counts"]["train"] = 99
            corrupt_payload.pop("result_sha256")
            corrupt_payload["result_sha256"] = prepare.sha256_bytes(
                prepare.canonical_json_bytes(corrupt_payload)
            )
            corrupt_sidecar.write_bytes(_json_bytes(corrupt_payload))

            regenerated_indices: list[int] = []

            def regenerate(task: dict[str, object]) -> dict[str, object]:
                regenerated_indices.append(int(task["index"]))
                return real_process(task)

            with (
                mock.patch.multiple(prepare, **patches),
                mock.patch.object(
                    prepare.concurrent.futures,
                    "ProcessPoolExecutor",
                    InlineExecutor,
                ),
                mock.patch.object(
                    prepare,
                    "_process_shard",
                    side_effect=regenerate,
                ),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )
            self.assertEqual(regenerated_indices, [0, 1])

            data_file = next((output / "data").glob("*.parquet"))
            outside_copy = temporary_root / "outside.parquet"
            shutil.copy2(data_file, outside_copy)
            data_file.unlink()
            data_file.symlink_to(outside_copy)
            with (
                mock.patch.multiple(prepare, **patches),
                self.assertRaisesRegex(
                    prepare.SFTPreparationError,
                    "curation sidecar",
                ),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

            data_file.unlink()
            shutil.copy2(outside_copy, data_file)
            identity_file = output / "IDENTITY.json"
            outside_identity = temporary_root / "outside-identity.json"
            shutil.copy2(identity_file, outside_identity)
            identity_file.unlink()
            identity_file.symlink_to(outside_identity)
            with (
                mock.patch.multiple(prepare, **patches),
                self.assertRaisesRegex(
                    prepare.SFTPreparationError,
                    "JSON authority",
                ),
            ):
                prepare.prepare_dataset(
                    root=raw_root,
                    output=output,
                    tokenizer_root=tokenizer_root,
                    policy_path=policy_path,
                    workers=2,
                    read_batch_size=2,
                    output_row_group_size=1,
                )

    def test_prime_config_freezes_six_gpu_boundary_aware_dry_run(self) -> None:
        import tomllib

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "posttrain"
            / "prime_sft_6gpu.toml"
        )
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["deployment"]["num_train_gpus"], 6)
        self.assertEqual(config["deployment"]["num_infer_gpus"], 0)
        self.assertEqual(config["model"]["impl"], "custom")
        self.assertEqual(config["model"]["seq_len"], 4096)
        self.assertEqual(config["model"]["ac_offloading"], "None")
        self.assertEqual(config["model"]["cp"], 1)
        self.assertEqual(config["model"]["ep"], 1)
        self.assertEqual(config["renderer"]["name"], "starcoder2-coding-chat-v1")
        self.assertEqual(config["data"]["seq_len"], 4096)
        self.assertEqual(config["val"]["data"]["seq_len"], 4096)
        self.assertEqual(config["data"]["num_workers"], 1)
        self.assertEqual(config["val"]["data"]["num_workers"], 1)
        self.assertFalse(config["ckpt"]["output_dir"].endswith("/checkpoints"))
        self.assertEqual(
            config["ckpt"]["output_dir"] + "/checkpoints",
            "/workspace/posttraining-runs/prime-sft/"
            "coding-llm-1p3b-opencodeinstruct-sft-v1/checkpoints",
        )
        self.assertTrue(config["data"]["loss_mask"]["assistant"])
        self.assertFalse(config["data"]["loss_mask"]["user"])


if __name__ == "__main__":
    unittest.main()
