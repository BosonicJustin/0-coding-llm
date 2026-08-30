#!/usr/bin/env python3
"""Stream cleaned English Wikipedia articles until the exact token quota is full.

The pinned ``20231101.en`` source is read one Parquet shard at a time through
Hugging Face streaming. Source Parquet is never copied to the dataset volume.
Accepted article text is stored byte-for-byte in lossless tar.zst archives with
an internal provenance manifest. Checkpoints contain an exact source-file and
row cursor and are safe to resume after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from benchmark_guard import BenchmarkGuard
from quota_tracker import DEFAULT_CONFIG, load_config, quota_status, write_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DENYLIST = PROJECT_ROOT / "configs" / "mbpp_denylist.json"
DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_DATASET_CONFIG = "20231101.en"
LANGUAGE_GROUP = "wikipedia"
CONFIG_PREFIXES = {
    "20231101.en": "20231101.en/",
}
CHECKPOINT_VERSION = 1
COLLECTOR_VERSION = 1
URL_QUARANTINE_MARKERS = (
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
        print(f"\nReceived signal {signum}; checkpointing after the current document...", flush=True)


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


def english_collection_target(config: dict[str, Any]) -> int:
    matches = [
        quota
        for quota in config["quotas"]
        if quota.get("phase") == "collection"
        and quota.get("category") == "english"
        and quota.get("language_group") == LANGUAGE_GROUP
        and "split" not in quota
    ]
    if len(matches) != 1 or matches[0].get("token_field") != "exact_tokens":
        raise ValueError("Expected one exact-token collection quota for English")
    return int(matches[0]["target"])


def committed_total(root: Path, config: dict[str, Any]) -> int:
    for row in quota_status(root, config, phase="collection"):
        if row.get("language_group") == LANGUAGE_GROUP and "split" not in row:
            return int(row["current"])
    return 0


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


def is_url_quarantined(url: str) -> bool:
    searchable = url.lower()
    return any(marker in searchable for marker in URL_QUARANTINE_MARKERS)


def rejection_id(source: str, row: dict[str, Any]) -> str:
    identity = {
        "source": source,
        "id": row.get("id"),
        "url": row.get("url"),
        "title": row.get("title"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_rejection_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(str(json.loads(line)["rejection_id"]))
    return result


def record_rejection(
    path: Path,
    seen: set[str],
    reason: str,
    source: str,
    source_shard_index: int,
    row_offset: int,
    row: dict[str, Any],
) -> None:
    identifier = rejection_id(source, row)
    if identifier in seen:
        return
    payload = {
        "rejection_id": identifier,
        "reason": reason,
        "source": source,
        "source_shard_index": source_shard_index,
        "row_offset": row_offset,
        "id": row.get("id"),
        "title": row.get("title"),
        "url": row.get("url"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    seen.add(identifier)


class RawTextArchiveWriter:
    def __init__(self, root: Path, index: int, compression_level: int, compression_threads: int):
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("Raw archive writing requires zstandard") from exc
        self.index = index
        self.documents = 0
        self.clean_bytes = 0
        self.exact_tokens = 0
        self._manifest = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b")
        output_dir = root / "raw" / "english" / "wikipedia"
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

    def add(self, content: bytes, token_count: int, row: dict[str, Any]) -> None:
        source_id = str(row.get("id") or row.get("url") or self.documents)
        safe_id = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
        member_path = f"documents/{self.documents:09d}-{safe_id}.txt"
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
            "id": row.get("id"),
            "title": row.get("title"),
            "url": row.get("url"),
            "language": "en",
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
            "index": self.index,
            "pending_path": str(self.pending_path),
            "final_path": str(self.final_path),
            "documents": self.documents,
            "clean_bytes": self.clean_bytes,
            "exact_tokens": self.exact_tokens,
            "compressed_bytes": self.pending_path.stat().st_size,
        }


def next_archive_index(root: Path) -> int:
    directory = root / "raw" / "english" / "wikipedia"
    indexes = []
    if directory.exists():
        for path in directory.glob("part-*.tar.zst"):
            try:
                indexes.append(int(path.name.removeprefix("part-").removesuffix(".tar.zst")))
            except ValueError:
                continue
    return max(indexes, default=-1) + 1


def ordered_source_shards(filenames: list[str], seed: int) -> list[str]:
    return sorted(
        filenames,
        key=lambda name: hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest(),
    )


def resolve_dataset_source(
    root: Path,
    repo_id: str,
    dataset_config: str,
    requested_revision: str | None,
    shard_seed: int,
    tokenizer_sha256: str,
    benchmark_guard_sha256: str,
) -> tuple[str, list[str]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before collecting Wikipedia") from exc
    if dataset_config not in CONFIG_PREFIXES:
        raise ValueError(f"Unsupported Wikipedia config: {dataset_config}")
    manifest_path = root / "manifests" / "WIKIPEDIA_SOURCE.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if existing and (existing.get("repo_id") != repo_id or existing.get("dataset_config") != dataset_config):
        raise ValueError("Existing Wikipedia source manifest uses a different repository/config")
    revision = requested_revision or (existing and existing.get("resolved_revision")) or "main"
    info = HfApi().dataset_info(repo_id, revision=revision)
    resolved = info.sha
    if not isinstance(resolved, str) or len(resolved) != 40:
        raise RuntimeError(f"Invalid dataset revision returned by Hugging Face: {resolved!r}")
    if existing and existing.get("resolved_revision") != resolved:
        raise ValueError("The existing English corpus is pinned to a different dataset revision")
    card = vars(info.card_data) if info.card_data else {}
    if not dataset_config.endswith(".en"):
        raise ValueError(f"Wikipedia config is not English: {dataset_config!r}")
    prefix = CONFIG_PREFIXES[dataset_config]
    filenames = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet")
    )
    if not filenames:
        raise RuntimeError(f"No source files found for {dataset_config}")
    shards = ordered_source_shards(filenames, shard_seed)
    shard_digest = hashlib.sha256("\n".join(shards).encode("utf-8")).hexdigest()
    fixed = {
        "shard_seed": shard_seed,
        "source_shards_sha256": shard_digest,
        "tokenizer_revision": tokenizer_sha256,
        "benchmark_guard_sha256": benchmark_guard_sha256,
    }
    if existing:
        for key, value in fixed.items():
            if existing.get(key) != value:
                raise ValueError(f"Existing Wikipedia manifest has a different {key}")
    atomic_json(
        manifest_path,
        {
            "manifest_version": 1,
            "repo_id": repo_id,
            "dataset_config": dataset_config,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "license": card.get("license"),
            "language": card.get("language"),
            "source_shard_count": len(shards),
            "source_shards": shards,
            **fixed,
        },
    )
    return resolved, shards


def checkpoint_directory(root: Path) -> Path:
    return root / "state" / "wikipedia_checkpoints"


def recover_checkpoints(
    root: Path, source: str, tokenizer_sha256: str, benchmark_guard_sha256: str
) -> dict[str, int]:
    directory = checkpoint_directory(root)
    if not directory.exists():
        return {"sequence": 0, "documents_consumed": 0, "source_shard_index": 0, "row_offset": 0}
    cursor = {"sequence": 0, "documents_consumed": 0, "source_shard_index": 0, "row_offset": 0}
    for path in sorted(directory.glob("checkpoint-*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise RuntimeError(f"Unsupported Wikipedia checkpoint version in {path}")
        for key, expected in (
            ("source", source),
            ("tokenizer_revision", tokenizer_sha256),
            ("benchmark_guard_sha256", benchmark_guard_sha256),
        ):
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Wikipedia checkpoint {key} mismatch in {path}")
        archive = checkpoint.get("archive")
        if archive:
            pending = Path(archive["pending_path"])
            final = Path(archive["final_path"])
            if not final.exists():
                if not pending.exists():
                    raise RuntimeError(f"Checkpoint {path} is missing archive {final}")
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pending, final)
            shard_id = f"wikipedia-{checkpoint['dataset_revision'][:12]}-{archive['index']:06d}"
            write_record(
                root,
                {
                    "phase": "collection",
                    "category": "english",
                    "language_group": LANGUAGE_GROUP,
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
                "documents_consumed": int(checkpoint["documents_consumed"]),
                "source_shard_index": int(checkpoint["source_shard_index"]),
                "row_offset": int(checkpoint["row_offset"]),
            }
    return cursor


def commit_checkpoint(
    root: Path,
    writer: RawTextArchiveWriter | None,
    documents_consumed: int,
    source_shard_index: int,
    row_offset: int,
    sequence: int,
    dataset_revision: str,
    source: str,
    tokenizer_sha256: str,
    benchmark_guard_sha256: str,
) -> None:
    archive = writer.close() if writer else None
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "source": source,
        "dataset_revision": dataset_revision,
        "tokenizer_revision": tokenizer_sha256,
        "benchmark_guard_sha256": benchmark_guard_sha256,
        "sequence": sequence,
        "documents_consumed": documents_consumed,
        "source_shard_index": source_shard_index,
        "row_offset": row_offset,
        "archive": archive,
    }
    path = checkpoint_directory(root) / f"checkpoint-{sequence:08d}.json"
    atomic_json(path, payload)
    recover_checkpoints(root, source, tokenizer_sha256, benchmark_guard_sha256)


def checkpointed_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/workspace/dataset/tokenizer/starcoder2"))
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_BENCHMARK_DENYLIST)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wikipedia-cache"))
    parser.add_argument("--shard-seed", type=int, default=2027)
    parser.add_argument("--checkpoint-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--checkpoint-documents", type=int, default=100_000)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--compression-threads", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--log-every-documents", type=int, default=1_000)
    parser.add_argument("--max-new-documents", type=int, help="cleanly stop after N source rows")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.checkpoint_bytes <= 0 or args.checkpoint_documents <= 0:
            raise ValueError("Checkpoint thresholds must be positive")
        args.root.mkdir(parents=True, exist_ok=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        quota_config = load_config(args.quota_config)
        target = english_collection_target(quota_config)
        tokenizer_sha = tokenizer_revision(args.tokenizer)
        tokenizer_json = args.tokenizer / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise ValueError(f"Missing tokenizer file: {tokenizer_json}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers before collecting Wikipedia") from exc
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        benchmark_guard = BenchmarkGuard(args.benchmark_denylist)
        dataset_sha, source_shards = resolve_dataset_source(
            args.root,
            args.dataset_repo,
            args.dataset_config,
            args.dataset_revision,
            args.shard_seed,
            tokenizer_sha,
            benchmark_guard.manifest_sha256,
        )
        source = f"{args.dataset_repo}@{dataset_sha}#{args.dataset_config}"
        cursor = recover_checkpoints(
            args.root, source, tokenizer_sha, benchmark_guard.manifest_sha256
        )
        documents_consumed = cursor["documents_consumed"]
        source_shard_index = cursor["source_shard_index"]
        row_offset = cursor["row_offset"]
        checkpoint_sequence = cursor["sequence"]
        committed = committed_total(args.root, quota_config)
        if committed >= target:
            print("The English collection bucket is already full.")
            return 0
        if documents_consumed:
            print(
                f"Resuming at source shard {source_shard_index:,}, row {row_offset:,} "
                f"after {documents_consumed:,} documents",
                flush=True,
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets before collecting Wikipedia") from exc
        stop_request = StopRequest()
        signal.signal(signal.SIGINT, stop_request)
        signal.signal(signal.SIGTERM, stop_request)
        archive_index = next_archive_index(args.root)
        writer: RawTextArchiveWriter | None = None
        interval_bytes = 0
        interval_documents = 0
        new_documents = 0
        accepted_documents = 0
        accepted_bytes = 0
        skipped = {"empty": 0, "url_quarantine": 0, "benchmark_content": 0}
        rejections_path = args.root / "logs" / "wikipedia_benchmark_rejections.jsonl"
        rejection_ids = load_rejection_ids(rejections_path)
        started = time.monotonic()

        def live_total() -> int:
            return committed + (writer.exact_tokens if writer else 0)

        def checkpoint(next_source_shard_index: int, next_row_offset: int) -> None:
            nonlocal writer, committed, interval_bytes, interval_documents, checkpoint_sequence
            checkpoint_sequence += 1
            commit_checkpoint(
                args.root,
                writer,
                documents_consumed,
                next_source_shard_index,
                next_row_offset,
                checkpoint_sequence,
                dataset_sha,
                source,
                tokenizer_sha,
                benchmark_guard.manifest_sha256,
            )
            writer = None
            committed = committed_total(args.root, quota_config)
            interval_bytes = 0
            interval_documents = 0

        for current_shard_index in range(source_shard_index, len(source_shards)):
            source_filename = source_shards[current_shard_index]
            source_uri = f"hf://datasets/{args.dataset_repo}@{dataset_sha}/{source_filename}"
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
            for row in stream:
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    skipped["empty"] += 1
                elif is_url_quarantined(str(row.get("url") or "")):
                    skipped["url_quarantine"] += 1
                else:
                    benchmark_reason = benchmark_guard.contamination_reason("English", text)
                    if benchmark_reason:
                        skipped["benchmark_content"] += 1
                        record_rejection(
                            rejections_path,
                            rejection_ids,
                            benchmark_reason,
                            source,
                            current_shard_index,
                            current_row_offset,
                            row,
                        )
                    else:
                        encoding = tokenizer.encode(text, add_special_tokens=False)
                        token_count = len(encoding.ids)
                        if token_count:
                            if writer is None:
                                writer = RawTextArchiveWriter(
                                    args.root,
                                    archive_index,
                                    args.compression_level,
                                    args.compression_threads,
                                )
                                archive_index += 1
                            content = text.encode("utf-8")
                            writer.add(content, token_count, row)
                            interval_bytes += len(content)
                            accepted_bytes += len(content)
                            accepted_documents += 1
                        else:
                            skipped["empty"] += 1
                current_row_offset += 1
                documents_consumed += 1
                new_documents += 1
                interval_documents += 1
                reached = live_total() >= target
                disk_free = shutil.disk_usage(args.root).free
                low_disk = disk_free < int(args.min_free_gb * 1_000_000_000)
                pilot_done = (
                    args.max_new_documents is not None
                    and new_documents >= args.max_new_documents
                )
                checkpoint_due = (
                    interval_bytes >= args.checkpoint_bytes
                    or interval_documents >= args.checkpoint_documents
                    or reached
                    or low_disk
                    or pilot_done
                    or stop_request.requested
                )
                if checkpoint_due:
                    checkpoint(current_shard_index, current_row_offset)
                if new_documents % args.log_every_documents == 0 or checkpoint_due:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "source_shard_index": current_shard_index,
                                "source_shards": len(source_shards),
                                "row_offset": current_row_offset,
                                "documents_consumed": documents_consumed,
                                "new_documents": new_documents,
                                "accepted_documents": accepted_documents,
                                "english_tokens": live_total(),
                                "english_target": target,
                                "retained_MB_per_second": round(
                                    accepted_bytes / elapsed / 1_000_000, 3
                                ),
                                "free_GB": round(disk_free / 1_000_000_000, 2),
                                "skipped_documents": skipped,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if reached or low_disk or pilot_done or stop_request.requested:
                    if reached:
                        atomic_json(
                            args.root / "state" / "ENGLISH_WIKIPEDIA_COMPLETE.json",
                            {
                                "source": source,
                                "tokenizer_revision": tokenizer_sha,
                                "benchmark_guard_sha256": benchmark_guard.manifest_sha256,
                                "english_tokens": committed,
                                "target": target,
                            },
                        )
                        print("The exact-token English collection target was reached.")
                    elif low_disk:
                        print("Stopped cleanly because the free-space threshold was reached.")
                    elif pilot_done:
                        print("Pilot document limit reached; resume with the same command.")
                    else:
                        print("Stopped cleanly after checkpointing.")
                    checkpointed_exit(0)
            row_offset = 0
            checkpoint(current_shard_index + 1, 0)
        print("The configured Wikipedia source ended before the English quota was reached.")
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
