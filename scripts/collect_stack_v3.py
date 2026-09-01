#!/usr/bin/env python3
"""Stream Stack v3 into lossless raw-code archives until code quotas are full.

Only languages in configs/code_languages.json are retained. Source rows are
read through Hugging Face streaming and are never saved. Retained UTF-8 code is
stored byte-for-byte in .tar.zst archives; tokenization is only used to enforce
the exact Python and other-code collection thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import signal
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from benchmark_guard import BenchmarkGuard
from quota_tracker import DEFAULT_CONFIG, load_config, quota_status, write_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGUAGES = PROJECT_ROOT / "configs" / "code_languages.json"
DEFAULT_BENCHMARK_DENYLIST = PROJECT_ROOT / "configs" / "mbpp_denylist.json"
DEFAULT_DATASET = "HuggingFaceCode/stack-v3-train"
CHECKPOINT_VERSION = 3
COLLECTOR_VERSION = 4
QUARANTINE_MARKERS = (
    "mbpp",
    "evalplus",
    "eval_plus",
    "multi-pl-e",
    "multipl-e",
    "multiple-e",
    "mbxp",
    "mxeval",
    "human_eval",
    "humaneval",
)


class StopRequest:
    requested = False

    def __call__(self, signum: int, _frame: Any) -> None:
        self.requested = True
        print(f"\nReceived signal {signum}; checkpointing after the current repository...", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_rejection_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rejection_id = json.loads(line).get("rejection_id")
                if rejection_id:
                    result.add(str(rejection_id))
    return result


def record_benchmark_rejection(
    path: Path,
    seen: set[str],
    reason: str,
    source: str,
    source_shard_index: int,
    row_offset: int,
    repo: dict[str, Any],
    file: dict[str, Any],
) -> None:
    identity = {
        "source": source,
        "repo_id": repo.get("repo_id"),
        "commit_id": repo.get("commit_id"),
        "content_id": file.get("content_id"),
        "file_path": file.get("file_path"),
    }
    rejection_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if rejection_id in seen:
        return
    payload = {
        "rejection_id": rejection_id,
        "reason": reason,
        "source": source,
        "source_shard_index": source_shard_index,
        "row_offset": row_offset,
        "repo_path": repo.get("repo_path"),
        "repo_id": repo.get("repo_id"),
        "commit_id": repo.get("commit_id"),
        "file_path": file.get("file_path"),
        "content_id": file.get("content_id"),
        "language": file.get("language"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    seen.add(rejection_id)


def load_language_policy(path: Path) -> tuple[set[str], set[str]]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("version") != 1:
        raise ValueError(f"Unsupported language policy version in {path}")
    python = set(policy.get("python", []))
    other = set(policy.get("other_code", []))
    overlap = python & other
    if overlap:
        raise ValueError(f"Languages appear in both categories: {sorted(overlap)}")
    if python != {"Python"}:
        raise ValueError("The Python category must contain exactly the go-enry language 'Python'")
    return python, other


def classify_language(language: Any, python: set[str], other: set[str]) -> str | None:
    if language in python:
        return "python"
    if language in other:
        return "other_code"
    return None


def is_quarantined(repo_path: str, file_path: str) -> bool:
    searchable = f"{repo_path}/{file_path}".lower().replace("\\", "/")
    return any(marker in searchable for marker in QUARANTINE_MARKERS)


def safe_suffix(file_path: str) -> str:
    suffix = PurePosixPath(file_path.replace("\\", "/")).suffix
    if len(suffix) <= 16 and re.fullmatch(r"\.[A-Za-z0-9_+.-]+", suffix):
        return suffix
    return ""


def collection_targets(config: dict[str, Any]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for quota in config["quotas"]:
        if quota.get("phase") != "collection" or "split" in quota:
            continue
        category = quota.get("category")
        if category in ("python", "other_code"):
            if quota.get("token_field") != "exact_tokens":
                raise ValueError(f"Collection quota {quota['name']} is not exact-token based")
            targets[category] = int(quota["target"])
    if set(targets) != {"python", "other_code"}:
        raise ValueError("Expected exact collection quotas for python and other_code")
    return targets


def committed_totals(root: Path, config: dict[str, Any]) -> dict[str, int]:
    totals = {"python": 0, "other_code": 0}
    for row in quota_status(root, config, phase="collection"):
        category = row.get("category")
        if category in totals and "split" not in row:
            totals[category] = int(row["current"])
    return totals


class RawArchiveWriter:
    def __init__(
        self,
        root: Path,
        category: str,
        index: int,
        compression_level: int,
        compression_threads: int,
    ) -> None:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("Raw archive writing requires zstandard") from exc

        self.category = category
        self.index = index
        self.documents = 0
        self.clean_bytes = 0
        self.exact_tokens = 0
        self.languages: dict[str, int] = {}
        self._manifest = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b")
        output_dir = root / "raw" / category
        output_dir.mkdir(parents=True, exist_ok=True)
        unique = uuid.uuid4().hex
        self.pending_path = output_dir / f".part-{index:06d}-{unique}.tar.zst"
        self.final_path = output_dir / f"part-{index:06d}.tar.zst"
        self._raw = self.pending_path.open("wb")
        compressor = zstandard.ZstdCompressor(
            level=compression_level,
            threads=compression_threads,
            write_checksum=True,
        )
        self._compressed = compressor.stream_writer(self._raw, closefd=False)
        self._tar = tarfile.open(fileobj=self._compressed, mode="w|", format=tarfile.PAX_FORMAT)

    def add(
        self,
        content: bytes,
        token_count: int,
        repo: dict[str, Any],
        file: dict[str, Any],
    ) -> None:
        ordinal = self.documents
        repo_id = str(repo.get("repo_id", "unknown"))
        content_id = str(file.get("content_id", "unknown"))
        member_path = f"files/{repo_id}/{ordinal:09d}-{content_id}{safe_suffix(str(file.get('file_path', '')))}"
        info = tarfile.TarInfo(member_path)
        info.size = len(content)
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        self._tar.addfile(info, io.BytesIO(content))

        manifest_row = {
            "member_path": member_path,
            "repo_path": repo.get("repo_path"),
            "repo_id": repo.get("repo_id"),
            "commit_id": repo.get("commit_id"),
            "file_path": file.get("file_path"),
            "content_id": file.get("content_id"),
            "language": file.get("language"),
            "license_type": file.get("license_type"),
            "detected_licenses": file.get("detected_licenses"),
            "size_bytes": len(content),
            "starcoder2_tokens": token_count,
        }
        self._manifest.write(
            (json.dumps(manifest_row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        self.documents += 1
        self.clean_bytes += len(content)
        self.exact_tokens += token_count
        language = str(file.get("language"))
        self.languages[language] = self.languages.get(language, 0) + token_count

    def close(self) -> dict[str, Any] | None:
        if self._tar is None:
            raise RuntimeError("Archive writer has already been closed")
        if self.documents == 0:
            self._tar.close()
            self._compressed.close()
            self._raw.close()
            self._manifest.close()
            try:
                self.pending_path.unlink()
            except FileNotFoundError:
                pass
            self._tar = None
            return None

        self._manifest.flush()
        manifest_size = self._manifest.tell()
        self._manifest.seek(0)
        info = tarfile.TarInfo("_manifest.jsonl")
        info.size = manifest_size
        info.mode = 0o644
        info.mtime = 0
        self._tar.addfile(info, self._manifest)
        self._tar.close()
        self._compressed.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._manifest.close()
        self._tar = None
        return {
            "category": self.category,
            "index": self.index,
            "pending_path": str(self.pending_path),
            "final_path": str(self.final_path),
            "documents": self.documents,
            "clean_bytes": self.clean_bytes,
            "exact_tokens": self.exact_tokens,
            "tokens_by_language": self.languages,
            "compressed_bytes": self.pending_path.stat().st_size,
        }


def next_archive_indices(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(r"part-(\d{6})\.tar\.zst$")
    for category in ("python", "other_code"):
        indexes = []
        directory = root / "raw" / category
        if directory.exists():
            for path in directory.iterdir():
                match = pattern.match(path.name)
                if match:
                    indexes.append(int(match.group(1)))
        result[category] = max(indexes, default=-1) + 1
    return result


def recover_checkpoint_archive(
    root: Path,
    archive: dict[str, Any],
    *,
    checkpoint_path: Path,
) -> Path:
    """Recover one checkpoint archive into ``root`` without trusting old roots.

    Checkpoints historically stored absolute pending/final paths.  A frozen
    dataset generation may now be hard-link cloned before collection resumes,
    so a finalized archive is authoritative at the canonical path derived from
    the *current* root, category, and index.  Descriptor paths are consulted
    only when recovering an actually pending same-root archive.
    """

    if not isinstance(archive, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} has an invalid archive descriptor")
    category = archive.get("category")
    index = archive.get("index")
    if category not in ("python", "other_code"):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has an invalid archive category: {category!r}"
        )
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index > 999_999
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has an invalid archive index: {index!r}"
        )

    root = root.resolve(strict=True)
    expected_directory = root / "raw" / category
    expected_final = expected_directory / f"part-{index:06d}.tar.zst"
    final_name = expected_final.name
    pending_pattern = re.compile(rf"\.part-{index:06d}-[0-9a-f]{{32}}\.tar\.zst")

    def descriptor_path(field: str) -> Path:
        value = archive.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} archive has invalid {field}"
            )
        candidate = Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} archive has unsafe {field}: {value!r}"
            )
        return candidate

    descriptor_final = descriptor_path("final_path")
    descriptor_pending = descriptor_path("pending_path")
    if tuple(descriptor_final.parts[-3:]) != ("raw", category, final_name):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} final_path identity mismatch: "
            f"{descriptor_final}"
        )
    if (
        tuple(descriptor_pending.parts[-3:-1]) != ("raw", category)
        or pending_pattern.fullmatch(descriptor_pending.name) is None
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} pending_path identity mismatch: "
            f"{descriptor_pending}"
        )

    def require_finalized(path: Path) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"Checkpoint {checkpoint_path} is missing archive {path}") from error
        if path.is_symlink() or not path.is_file() or metadata.st_size <= 0:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} archive is unsafe or empty: {path}"
            )

    # This is the normal cloned-generation path.  Never dereference an old
    # absolute descriptor when the current root already owns the final name.
    if expected_final.exists() or expected_final.is_symlink():
        require_finalized(expected_final)
        return expected_final

    # A pending rename is valid only for a checkpoint created in this same
    # root.  This prevents a cloned checkpoint from reaching back into its
    # frozen source generation if the clone is incomplete.
    if (
        descriptor_final.resolve(strict=False) != expected_final
        or descriptor_pending.parent.resolve(strict=False) != expected_directory
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no finalized archive in the current root "
            "and its pending paths belong to a different root"
        )
    if expected_directory.is_symlink() or not expected_directory.is_dir():
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} raw archive directory is unsafe: "
            f"{expected_directory}"
        )
    try:
        pending_metadata = descriptor_pending.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing pending archive "
            f"{descriptor_pending}"
        ) from error
    if (
        descriptor_pending.is_symlink()
        or not descriptor_pending.is_file()
        or pending_metadata.st_size <= 0
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} pending archive is unsafe or empty: "
            f"{descriptor_pending}"
        )
    os.replace(descriptor_pending, expected_final)
    require_finalized(expected_final)
    return expected_final


def recover_checkpoints(root: Path, source: str, benchmark_guard_sha256: str) -> dict[str, int]:
    checkpoint_dir = root / "state" / "collector_checkpoints"
    if not checkpoint_dir.exists():
        return {"sequence": 0, "repos_consumed": 0, "source_shard_index": 0, "row_offset": 0}
    cursor = {"sequence": 0, "repos_consumed": 0, "source_shard_index": 0, "row_offset": 0}
    for path in sorted(checkpoint_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise RuntimeError(f"Unsupported checkpoint version in {path}")
        if checkpoint.get("source") != source:
            raise RuntimeError(f"Checkpoint source mismatch in {path}")
        if checkpoint.get("benchmark_guard_sha256") != benchmark_guard_sha256:
            raise RuntimeError(f"Benchmark denylist mismatch in {path}")
        for archive in checkpoint.get("archives", []):
            recover_checkpoint_archive(root, archive, checkpoint_path=path)
            shard_id = f"stack-v3-{checkpoint['dataset_revision'][:12]}-{archive['category']}-{archive['index']:06d}"
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": archive["category"],
                    "shard_id": shard_id,
                    "source": source,
                    "documents": archive["documents"],
                    "clean_bytes": archive["clean_bytes"],
                    "exact_tokens": archive["exact_tokens"],
                },
            )
        sequence = int(checkpoint["sequence"])
        if sequence >= cursor["sequence"]:
            cursor = {
                "sequence": sequence,
                "repos_consumed": int(checkpoint["repos_consumed"]),
                "source_shard_index": int(checkpoint["source_shard_index"]),
                "row_offset": int(checkpoint["row_offset"]),
            }
    return cursor


def commit_checkpoint(
    root: Path,
    writers: dict[str, RawArchiveWriter],
    repos_consumed: int,
    source_shard_index: int,
    row_offset: int,
    sequence: int,
    dataset_revision: str,
    source: str,
    benchmark_guard_sha256: str,
) -> list[dict[str, Any]]:
    archives = []
    for writer in writers.values():
        archive = writer.close()
        if archive is not None:
            archives.append(archive)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "source": source,
        "benchmark_guard_sha256": benchmark_guard_sha256,
        "dataset_revision": dataset_revision,
        "sequence": sequence,
        "repos_consumed": repos_consumed,
        "source_shard_index": source_shard_index,
        "row_offset": row_offset,
        "archives": archives,
    }
    checkpoint_path = (
        root / "state" / "collector_checkpoints" / f"checkpoint-{sequence:08d}.json"
    )
    atomic_json(checkpoint_path, checkpoint)
    recover_checkpoints(root, source, benchmark_guard_sha256)
    return archives


def ordered_source_shards(filenames: list[str], seed: int) -> list[str]:
    return sorted(
        filenames,
        key=lambda name: hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest(),
    )


def resolve_dataset_source(
    root: Path,
    repo_id: str,
    requested: str | None,
    shard_seed: int,
    benchmark_guard_sha256: str,
) -> tuple[str, list[str]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before collecting Stack v3") from exc
    manifest_path = root / "manifests" / "STACK_V3_SOURCE.json"
    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("repo_id") != repo_id:
            raise ValueError(f"Existing source manifest uses {existing.get('repo_id')!r}")
    revision = requested or (existing and existing.get("resolved_revision")) or "main"
    info = HfApi().dataset_info(repo_id, revision=revision)
    resolved = info.sha
    if not isinstance(resolved, str) or len(resolved) != 40:
        raise RuntimeError(f"Invalid dataset revision returned by Hugging Face: {resolved!r}")
    if existing and existing.get("resolved_revision") != resolved:
        raise ValueError(
            "The existing raw corpus is pinned to a different dataset revision; use a new root"
        )
    filenames = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet")
    ]
    if not filenames:
        raise RuntimeError("No Stack v3 data shards were found")
    shards = ordered_source_shards(filenames, shard_seed)
    shard_digest = hashlib.sha256("\n".join(shards).encode("utf-8")).hexdigest()
    if existing:
        if existing.get("shard_seed") != shard_seed:
            raise ValueError("The existing corpus uses a different source-shard seed")
        if existing.get("source_shards_sha256") != shard_digest:
            raise ValueError("The pinned source-shard list differs from the existing manifest")
        if existing.get("benchmark_guard_sha256") != benchmark_guard_sha256:
            raise ValueError("The existing corpus uses a different benchmark denylist")
    atomic_json(
        manifest_path,
        {
            "manifest_version": 3,
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "shard_seed": shard_seed,
            "source_shard_count": len(shards),
            "source_shards_sha256": shard_digest,
            "source_shards": shards,
            "benchmark_guard_sha256": benchmark_guard_sha256,
        },
    )
    return resolved, shards


def tokenizer_revision(tokenizer_dir: Path) -> str:
    manifest_path = tokenizer_dir / "TOKENIZER_MANIFEST.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing pinned tokenizer manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest.get("resolved_revision")
    if manifest.get("validation", {}).get("vocab_size") != 49_152:
        raise ValueError("Tokenizer manifest does not validate a 49,152-token vocabulary")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"Invalid tokenizer revision in {manifest_path}")
    return revision


def make_writer(
    root: Path, category: str, indexes: dict[str, int], args: argparse.Namespace
) -> RawArchiveWriter:
    writer = RawArchiveWriter(
        root,
        category,
        indexes[category],
        args.compression_level,
        args.compression_threads,
    )
    indexes[category] += 1
    return writer


def checkpointed_exit(code: int) -> None:
    """Exit after fsync without waiting for an in-flight remote read to drain."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/workspace/dataset/tokenizer/starcoder2")
    )
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--languages", type=Path, default=DEFAULT_LANGUAGES)
    parser.add_argument(
        "--benchmark-denylist", type=Path, default=DEFAULT_BENCHMARK_DENYLIST
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/huggingface-cache"))
    parser.add_argument(
        "--shard-seed",
        type=int,
        default=1307,
        help="seed for deterministic hash-ordering of the 8,192 source shards",
    )
    parser.add_argument("--checkpoint-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--checkpoint-repos", type=int, default=100_000)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--compression-threads", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--log-every-repos", type=int, default=1_000)
    parser.add_argument("--max-new-repos", type=int, help="cleanly stop after N new repos (pilot mode)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.checkpoint_bytes <= 0 or args.checkpoint_repos <= 0:
            raise ValueError("Checkpoint thresholds must be positive")
        args.root.mkdir(parents=True, exist_ok=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        python_languages, other_languages = load_language_policy(args.languages)
        benchmark_guard = BenchmarkGuard(args.benchmark_denylist)
        quota_config = load_config(args.quota_config)
        targets = collection_targets(quota_config)
        tokenizer_sha = tokenizer_revision(args.tokenizer)
        tokenizer_json = args.tokenizer / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise ValueError(f"Missing tokenizer file: {tokenizer_json}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers before collecting Stack v3") from exc
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer.no_padding()
        tokenizer.no_truncation()

        dataset_sha, source_shards = resolve_dataset_source(
            args.root,
            args.dataset_repo,
            args.dataset_revision,
            args.shard_seed,
            benchmark_guard.manifest_sha256,
        )
        source = f"{args.dataset_repo}@{dataset_sha}"
        benchmark_rejections_path = args.root / "logs" / "benchmark_rejections.jsonl"
        benchmark_rejections_seen = load_rejection_ids(benchmark_rejections_path)
        cursor = recover_checkpoints(args.root, source, benchmark_guard.manifest_sha256)
        repos_consumed = cursor["repos_consumed"]
        source_shard_index = cursor["source_shard_index"]
        row_offset = cursor["row_offset"]
        checkpoint_sequence = cursor["sequence"]
        totals = committed_totals(args.root, quota_config)
        if all(totals[category] >= targets[category] for category in targets):
            print("Both collection buckets are already full.")
            return 0

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets before collecting Stack v3") from exc
        if repos_consumed:
            print(
                f"Resuming at source shard {source_shard_index:,}, row {row_offset:,} "
                f"after {repos_consumed:,} repositories",
                flush=True,
            )

        stop_request = StopRequest()
        signal.signal(signal.SIGINT, stop_request)
        signal.signal(signal.SIGTERM, stop_request)
        indexes = next_archive_indices(args.root)
        writers: dict[str, RawArchiveWriter] = {}
        interval_bytes = 0
        interval_repos = 0
        new_repos = 0
        skipped = {
            "not_code": 0,
            "vendor": 0,
            "quarantine": 0,
            "benchmark_content": 0,
            "bucket_full": 0,
        }
        started = time.monotonic()
        accepted_bytes = 0

        def live_total(category: str) -> int:
            writer = writers.get(category)
            return totals[category] + (writer.exact_tokens if writer else 0)

        def checkpoint(next_source_shard_index: int, next_row_offset: int) -> None:
            nonlocal writers, totals, interval_bytes, interval_repos, checkpoint_sequence
            checkpoint_sequence += 1
            commit_checkpoint(
                args.root,
                writers,
                repos_consumed,
                next_source_shard_index,
                next_row_offset,
                checkpoint_sequence,
                dataset_sha,
                source,
                benchmark_guard.manifest_sha256,
            )
            writers = {}
            totals = committed_totals(args.root, quota_config)
            interval_bytes = 0
            interval_repos = 0

        for current_shard_index in range(source_shard_index, len(source_shards)):
            source_filename = source_shards[current_shard_index]
            source_uri = (
                f"hf://datasets/{args.dataset_repo}@{dataset_sha}/{source_filename}"
            )
            stream = load_dataset(
                "parquet",
                data_files={"train": source_uri},
                split="train",
                streaming=True,
                cache_dir=str(args.cache_dir),
            )
            current_row_offset = row_offset if current_shard_index == source_shard_index else 0
            if current_row_offset:
                stream = stream.skip(current_row_offset)

            for repo in stream:
                repo_path = str(repo.get("repo_path") or "")
                repo_quarantined = is_quarantined(repo_path, "")
                for file in repo.get("files") or []:
                    category = classify_language(
                        file.get("language"), python_languages, other_languages
                    )
                    if category is None:
                        skipped["not_code"] += 1
                        continue
                    if live_total(category) >= targets[category]:
                        skipped["bucket_full"] += 1
                        continue
                    if file.get("is_vendor") is True:
                        skipped["vendor"] += 1
                        continue
                    if file.get("license_type") not in ("permissive", "no_license"):
                        skipped["not_code"] += 1
                        continue
                    file_path = str(file.get("file_path") or "")
                    if repo_quarantined or is_quarantined(repo_path, file_path):
                        skipped["quarantine"] += 1
                        continue
                    content_text = file.get("content")
                    if not isinstance(content_text, str) or not content_text:
                        skipped["not_code"] += 1
                        continue
                    benchmark_reason = benchmark_guard.contamination_reason(
                        str(file.get("language")), content_text
                    )
                    if benchmark_reason:
                        record_benchmark_rejection(
                            benchmark_rejections_path,
                            benchmark_rejections_seen,
                            benchmark_reason,
                            source,
                            current_shard_index,
                            current_row_offset,
                            repo,
                            file,
                        )
                        skipped["benchmark_content"] += 1
                        continue
                    content = content_text.encode("utf-8")
                    encoding = tokenizer.encode(content_text, add_special_tokens=False)
                    token_count = len(encoding.ids)
                    if token_count == 0:
                        skipped["not_code"] += 1
                        continue
                    if category not in writers:
                        writers[category] = make_writer(args.root, category, indexes, args)
                    writers[category].add(content, token_count, repo, file)
                    interval_bytes += len(content)
                    accepted_bytes += len(content)

                current_row_offset += 1
                repos_consumed += 1
                new_repos += 1
                interval_repos += 1
                bucket_reached = any(
                    category in writers and live_total(category) >= targets[category]
                    for category in targets
                )
                disk_free = shutil.disk_usage(args.root).free
                low_disk = disk_free < int(args.min_free_gb * 1_000_000_000)
                pilot_done = args.max_new_repos is not None and new_repos >= args.max_new_repos
                checkpoint_due = (
                    interval_bytes >= args.checkpoint_bytes
                    or interval_repos >= args.checkpoint_repos
                    or bucket_reached
                    or stop_request.requested
                    or low_disk
                    or pilot_done
                )
                if checkpoint_due:
                    checkpoint(current_shard_index, current_row_offset)

                if new_repos % args.log_every_repos == 0 or checkpoint_due:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "source_shard_index": current_shard_index,
                                "source_shards": len(source_shards),
                                "row_offset": current_row_offset,
                                "repos_consumed": repos_consumed,
                                "new_repos": new_repos,
                                "python_tokens": live_total("python"),
                                "python_target": targets["python"],
                                "other_code_tokens": live_total("other_code"),
                                "other_code_target": targets["other_code"],
                                "retained_MB_per_second": round(
                                    accepted_bytes / elapsed / 1_000_000, 3
                                ),
                                "free_GB": round(disk_free / 1_000_000_000, 2),
                                "skipped_files": skipped,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

                complete = all(totals[category] >= targets[category] for category in targets)
                if complete or stop_request.requested or low_disk or pilot_done:
                    if complete:
                        atomic_json(
                            args.root / "state" / "COLLECTION_COMPLETE.json",
                            {
                                "source": source,
                                "tokenizer_revision": tokenizer_sha,
                                "python_tokens": totals["python"],
                                "other_code_tokens": totals["other_code"],
                            },
                        )
                        print("Both exact-token collection buckets reached their targets.")
                    elif low_disk:
                        print("Stopped cleanly because the free-space threshold was reached.")
                    elif pilot_done:
                        print("Pilot repository limit reached; resume with the same command.")
                    else:
                        print("Stopped cleanly after checkpointing.")
                    checkpointed_exit(0)

            row_offset = 0
            checkpoint(current_shard_index + 1, 0)

        print("The source stream ended before both quotas were reached.")
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
