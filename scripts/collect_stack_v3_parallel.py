#!/usr/bin/env python3
"""Parallel, checkpoint-compatible Stack v3 raw-code collector.

The first run imports the legacy sequential cursor into an immutable worker
plan. Worker zero resumes the partially consumed shard at its exact row offset;
the remaining workers start on later disjoint shards. Each worker owns separate
checkpoints, rejection logs, cache files, and archive-index sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import collect_stack_v3 as stack
from quota_tracker import DEFAULT_CONFIG, load_config


PARALLEL_PLAN_VERSION = 1
WORKER_CHECKPOINT_VERSION = 1


class StopRequest:
    def __init__(self, message: str) -> None:
        self.requested = False
        self.message = message

    def __call__(self, signum: int, _frame: Any) -> None:
        if not self.requested:
            print(f"\nReceived signal {signum}; {self.message}", flush=True)
        self.requested = True


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_path(root: Path) -> Path:
    return root / "state" / "collector_parallel" / "PLAN.json"


def worker_checkpoint_directory(root: Path, worker_id: int) -> Path:
    return root / "state" / "collector_parallel" / f"worker-{worker_id:02d}" / "checkpoints"


def assigned_source_shard(plan: dict[str, Any], worker_id: int, position: int) -> int:
    return int(plan["frontier_source_shard_index"]) + worker_id + (
        position * int(plan["worker_count"])
    )


def worker_archive_index(
    plan: dict[str, Any], category: str, worker_id: int, sequence: int
) -> int:
    return (
        int(plan["base_archive_indices"][category])
        + worker_id
        + int(plan["worker_count"]) * sequence
    )


def create_or_load_plan(
    args: argparse.Namespace,
    source: str,
    dataset_revision: str,
    source_shards: list[str],
    benchmark_guard_sha256: str,
) -> dict[str, Any]:
    path = plan_path(args.root)
    legacy_cursor = stack.recover_checkpoints(
        args.root, source, benchmark_guard_sha256
    )
    source_shards_sha256 = hashlib.sha256(
        "\n".join(source_shards).encode("utf-8")
    ).hexdigest()
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "plan_version": PARALLEL_PLAN_VERSION,
            "source": source,
            "dataset_revision": dataset_revision,
            "source_shards_sha256": source_shards_sha256,
            "benchmark_guard_sha256": benchmark_guard_sha256,
            "worker_count": args.workers,
        }
        for key, value in expected.items():
            if plan.get(key) != value:
                raise RuntimeError(
                    f"Parallel plan {key} mismatch: {plan.get(key)!r} != {value!r}"
                )
        saved_cursor = plan.get("legacy_cursor") or {}
        for key in ("sequence", "repos_consumed", "source_shard_index", "row_offset"):
            if int(saved_cursor.get(key, -1)) != int(legacy_cursor[key]):
                raise RuntimeError(
                    "Legacy cursor changed after the parallel plan was created; "
                    "refusing a potentially overlapping resume"
                )
        return plan

    plan = {
        "plan_version": PARALLEL_PLAN_VERSION,
        "created_unix": int(time.time()),
        "source": source,
        "dataset_revision": dataset_revision,
        "source_shards_sha256": source_shards_sha256,
        "source_shard_count": len(source_shards),
        "benchmark_guard_sha256": benchmark_guard_sha256,
        "worker_count": args.workers,
        "frontier_source_shard_index": int(legacy_cursor["source_shard_index"]),
        "frontier_row_offset": int(legacy_cursor["row_offset"]),
        "base_archive_indices": stack.next_archive_indices(args.root),
        "legacy_cursor": legacy_cursor,
    }
    stack.atomic_json(path, plan)
    return plan


def recover_worker_checkpoints(
    root: Path,
    plan: dict[str, Any],
    worker_id: int,
) -> dict[str, Any]:
    plan_sha256 = canonical_sha256(plan)
    cursor: dict[str, Any] = {
        "sequence": 0,
        "assignment_position": 0,
        "row_offset": int(plan["frontier_row_offset"]) if worker_id == 0 else 0,
        "repos_consumed": 0,
        "archive_sequences": {"python": 0, "other_code": 0},
    }
    directory = worker_checkpoint_directory(root, worker_id)
    if not directory.exists():
        return cursor
    for path in sorted(directory.glob("checkpoint-*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_version": WORKER_CHECKPOINT_VERSION,
            "plan_sha256": plan_sha256,
            "source": plan["source"],
            "worker_id": worker_id,
            "worker_count": int(plan["worker_count"]),
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(
                    f"Worker checkpoint {path} {key} mismatch: "
                    f"{checkpoint.get(key)!r} != {value!r}"
                )
        for archive in checkpoint.get("archives", []):
            pending = Path(archive["pending_path"])
            final = Path(archive["final_path"])
            if not final.exists():
                if not pending.exists():
                    raise RuntimeError(f"Checkpoint {path} is missing archive {final}")
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pending, final)
            shard_id = (
                f"stack-v3-{plan['dataset_revision'][:12]}-"
                f"{archive['category']}-{archive['index']:06d}"
            )
            stack.write_record(
                root,
                {
                    "phase": "collection",
                    "category": archive["category"],
                    "shard_id": shard_id,
                    "source": plan["source"],
                    "documents": archive["documents"],
                    "clean_bytes": archive["clean_bytes"],
                    "exact_tokens": archive["exact_tokens"],
                },
            )
        if int(checkpoint["sequence"]) >= int(cursor["sequence"]):
            cursor = {
                "sequence": int(checkpoint["sequence"]),
                "assignment_position": int(checkpoint["assignment_position"]),
                "row_offset": int(checkpoint["row_offset"]),
                "repos_consumed": int(checkpoint["repos_consumed"]),
                "archive_sequences": {
                    category: int(checkpoint["archive_sequences"][category])
                    for category in ("python", "other_code")
                },
            }
    return cursor


def commit_worker_checkpoint(
    root: Path,
    plan: dict[str, Any],
    worker_id: int,
    writers: dict[str, stack.RawArchiveWriter],
    sequence: int,
    assignment_position: int,
    row_offset: int,
    repos_consumed: int,
    archive_sequences: dict[str, int],
) -> None:
    archives = []
    for writer in writers.values():
        archive = writer.close()
        if archive is not None:
            archives.append(archive)
    checkpoint = {
        "checkpoint_version": WORKER_CHECKPOINT_VERSION,
        "collector_version": 1,
        "plan_sha256": canonical_sha256(plan),
        "source": plan["source"],
        "worker_id": worker_id,
        "worker_count": int(plan["worker_count"]),
        "sequence": sequence,
        "assignment_position": assignment_position,
        "source_shard_index": assigned_source_shard(
            plan, worker_id, assignment_position
        ),
        "row_offset": row_offset,
        "repos_consumed": repos_consumed,
        "archive_sequences": archive_sequences,
        "archives": archives,
    }
    path = worker_checkpoint_directory(root, worker_id) / f"checkpoint-{sequence:08d}.json"
    stack.atomic_json(path, checkpoint)
    recover_worker_checkpoints(root, plan, worker_id)


def make_worker_writer(
    args: argparse.Namespace,
    plan: dict[str, Any],
    worker_id: int,
    category: str,
    archive_sequences: dict[str, int],
) -> stack.RawArchiveWriter:
    index = worker_archive_index(
        plan, category, worker_id, archive_sequences[category]
    )
    archive_sequences[category] += 1
    return stack.RawArchiveWriter(
        args.root,
        category,
        index,
        args.compression_level,
        args.compression_threads,
    )


def run_worker(
    args: argparse.Namespace,
    plan: dict[str, Any],
    worker_id: int,
) -> int:
    python_languages, other_languages = stack.load_language_policy(args.languages)
    benchmark_guard = stack.BenchmarkGuard(args.benchmark_denylist)
    quota_config = load_config(args.quota_config)
    targets = stack.collection_targets(quota_config)
    tokenizer_sha = stack.tokenizer_revision(args.tokenizer)
    tokenizer_json = args.tokenizer / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError(f"Missing tokenizer file: {tokenizer_json}")
    from tokenizers import Tokenizer
    from datasets import load_dataset

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    source_manifest = json.loads(
        (args.root / "manifests" / "STACK_V3_SOURCE.json").read_text(encoding="utf-8")
    )
    source_shards = list(source_manifest["source_shards"])
    if hashlib.sha256("\n".join(source_shards).encode("utf-8")).hexdigest() != plan[
        "source_shards_sha256"
    ]:
        raise RuntimeError("Worker source-shard manifest differs from the parallel plan")
    if tokenizer_sha != stack.tokenizer_revision(args.tokenizer):
        raise RuntimeError("Tokenizer changed during worker startup")

    worker_cache = args.cache_dir / f"worker-{worker_id:02d}"
    worker_cache.mkdir(parents=True, exist_ok=True)
    rejection_path = (
        args.root / "logs" / f"benchmark_rejections.worker-{worker_id:02d}.jsonl"
    )
    rejection_ids = stack.load_rejection_ids(rejection_path)
    cursor = recover_worker_checkpoints(args.root, plan, worker_id)
    assignment_position = int(cursor["assignment_position"])
    row_offset = int(cursor["row_offset"])
    checkpoint_sequence = int(cursor["sequence"])
    repos_consumed = int(cursor["repos_consumed"])
    archive_sequences = dict(cursor["archive_sequences"])
    totals = stack.committed_totals(args.root, quota_config)
    writers: dict[str, stack.RawArchiveWriter] = {}
    interval_bytes = 0
    interval_repos = 0
    new_repos = 0
    accepted_bytes = 0
    started = time.monotonic()
    skipped = {
        "not_code": 0,
        "vendor": 0,
        "quarantine": 0,
        "benchmark_content": 0,
        "bucket_full": 0,
    }
    stop_request = StopRequest("worker will checkpoint after the current repository")
    for handled_signal in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(handled_signal, stop_request)

    def live_total(category: str) -> int:
        writer = writers.get(category)
        return totals[category] + (writer.exact_tokens if writer else 0)

    def checkpoint(next_assignment_position: int, next_row_offset: int) -> None:
        nonlocal writers, totals, interval_bytes, interval_repos, checkpoint_sequence
        checkpoint_sequence += 1
        commit_worker_checkpoint(
            args.root,
            plan,
            worker_id,
            writers,
            checkpoint_sequence,
            next_assignment_position,
            next_row_offset,
            repos_consumed,
            archive_sequences,
        )
        writers = {}
        totals = stack.committed_totals(args.root, quota_config)
        interval_bytes = 0
        interval_repos = 0

    while True:
        current_shard_index = assigned_source_shard(
            plan, worker_id, assignment_position
        )
        if current_shard_index >= len(source_shards):
            return 0
        source_filename = source_shards[current_shard_index]
        source_uri = (
            f"hf://datasets/{args.dataset_repo}@{plan['dataset_revision']}/{source_filename}"
        )
        stream = load_dataset(
            "parquet",
            data_files={"train": source_uri},
            split="train",
            streaming=True,
            cache_dir=str(worker_cache),
        )
        if row_offset:
            stream = stream.skip(row_offset)

        for repo in stream:
            repo_path = str(repo.get("repo_path") or "")
            repo_quarantined = stack.is_quarantined(repo_path, "")
            candidates: list[tuple[str, dict[str, Any], str]] = []
            for file in repo.get("files") or []:
                category = stack.classify_language(
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
                if repo_quarantined or stack.is_quarantined(repo_path, file_path):
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
                    stack.record_benchmark_rejection(
                        rejection_path,
                        rejection_ids,
                        benchmark_reason,
                        plan["source"],
                        current_shard_index,
                        row_offset,
                        repo,
                        file,
                    )
                    skipped["benchmark_content"] += 1
                    continue
                candidates.append((category, file, content_text))

            for start in range(0, len(candidates), args.token_batch_size):
                batch = candidates[start : start + args.token_batch_size]
                encodings = tokenizer.encode_batch(
                    [content for _, _, content in batch], add_special_tokens=False
                )
                for (category, file, content_text), encoding in zip(
                    batch, encodings, strict=True
                ):
                    if live_total(category) >= targets[category]:
                        skipped["bucket_full"] += 1
                        continue
                    token_count = len(encoding.ids)
                    if token_count == 0:
                        skipped["not_code"] += 1
                        continue
                    if category not in writers:
                        writers[category] = make_worker_writer(
                            args, plan, worker_id, category, archive_sequences
                        )
                    content = content_text.encode("utf-8")
                    writers[category].add(content, token_count, repo, file)
                    interval_bytes += len(content)
                    accepted_bytes += len(content)

            row_offset += 1
            repos_consumed += 1
            new_repos += 1
            interval_repos += 1
            bucket_reached = any(
                category in writers and live_total(category) >= targets[category]
                for category in targets
            )
            disk_free = shutil.disk_usage(args.root).free
            low_disk = disk_free < int(args.min_free_gb * 1_000_000_000)
            pilot_done = (
                args.max_new_repos is not None and new_repos >= args.max_new_repos
            )
            checkpoint_due = (
                interval_bytes >= args.checkpoint_bytes
                or interval_repos >= args.checkpoint_repos
                or bucket_reached
                or stop_request.requested
                or low_disk
                or pilot_done
            )
            if checkpoint_due:
                checkpoint(assignment_position, row_offset)
            if new_repos % args.log_every_repos == 0 or checkpoint_due:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "event": "worker_progress",
                            "worker_id": worker_id,
                            "worker_count": int(plan["worker_count"]),
                            "source_shard_index": current_shard_index,
                            "source_shards": len(source_shards),
                            "row_offset": row_offset,
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
            if stop_request.requested or low_disk or pilot_done:
                return 0
            if all(totals[category] >= targets[category] for category in targets):
                return 0

        assignment_position += 1
        row_offset = 0
        checkpoint(assignment_position, 0)


def worker_entry(args: argparse.Namespace, plan: dict[str, Any], worker_id: int) -> None:
    try:
        code = run_worker(args, plan, worker_id)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "event": "worker_error",
                    "worker_id": worker_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    raise SystemExit(code)


def parallel_main(args: argparse.Namespace) -> int:
    args.root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    benchmark_guard = stack.BenchmarkGuard(args.benchmark_denylist)
    quota_config = load_config(args.quota_config)
    targets = stack.collection_targets(quota_config)
    stack.tokenizer_revision(args.tokenizer)
    dataset_revision, source_shards = stack.resolve_dataset_source(
        args.root,
        args.dataset_repo,
        args.dataset_revision,
        args.shard_seed,
        benchmark_guard.manifest_sha256,
    )
    source = f"{args.dataset_repo}@{dataset_revision}"
    plan = create_or_load_plan(
        args,
        source,
        dataset_revision,
        source_shards,
        benchmark_guard.manifest_sha256,
    )
    for worker_id in range(args.workers):
        recover_worker_checkpoints(args.root, plan, worker_id)
    totals = stack.committed_totals(args.root, quota_config)
    if all(totals[category] >= targets[category] for category in targets):
        print("Both collection buckets are already full.")
        return 0

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=worker_entry,
            args=(args, plan, worker_id),
            name=f"stack-worker-{worker_id:02d}",
        )
        for worker_id in range(args.workers)
    ]
    stop_request = StopRequest("supervisor will stop workers after their current repositories")
    for handled_signal in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(handled_signal, stop_request)
    for process in processes:
        process.start()

    signaled = False
    failure = False
    last_status = 0.0
    last_quota_poll = 0.0
    while any(process.is_alive() for process in processes):
        now = time.monotonic()
        if now - last_quota_poll >= args.supervisor_poll_seconds:
            totals = stack.committed_totals(args.root, quota_config)
            last_quota_poll = now
        complete = all(totals[category] >= targets[category] for category in targets)
        if stop_request.requested or complete or failure:
            if not signaled:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                signaled = True
        if now - last_status >= args.supervisor_log_seconds:
            print(
                json.dumps(
                    {
                        "event": "parallel_status",
                        "parallel_workers": args.workers,
                        "active_workers": sum(process.is_alive() for process in processes),
                        "python_tokens": totals["python"],
                        "python_target": targets["python"],
                        "other_code_tokens": totals["other_code"],
                        "other_code_target": targets["other_code"],
                        "free_GB": round(
                            shutil.disk_usage(args.root).free / 1_000_000_000, 2
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_status = now
        for process in processes:
            process.join(timeout=0)
            if process.exitcode not in (None, 0) and not signaled:
                failure = True
        time.sleep(1)

    for process in processes:
        process.join()
    totals = stack.committed_totals(args.root, quota_config)
    complete = all(totals[category] >= targets[category] for category in targets)
    if complete:
        stack.atomic_json(
            args.root / "state" / "COLLECTION_COMPLETE.json",
            {
                "source": source,
                "benchmark_guard_sha256": benchmark_guard.manifest_sha256,
                "python_tokens": totals["python"],
                "other_code_tokens": totals["other_code"],
                "targets": targets,
                "parallel_workers": args.workers,
            },
        )
        print("Both exact-token collection targets were reached.", flush=True)
        return 0
    if stop_request.requested:
        print("Parallel collection stopped cleanly after worker checkpoints.", flush=True)
        return 0
    return 1 if failure or not complete else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/workspace/dataset/tokenizer/starcoder2")
    )
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--languages", type=Path, default=stack.DEFAULT_LANGUAGES)
    parser.add_argument(
        "--benchmark-denylist", type=Path, default=stack.DEFAULT_BENCHMARK_DENYLIST
    )
    parser.add_argument("--dataset-repo", default=stack.DEFAULT_DATASET)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/huggingface-cache"))
    parser.add_argument("--shard-seed", type=int, default=1307)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-bytes", type=int, default=100_000_000)
    parser.add_argument("--checkpoint-repos", type=int, default=10_000)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--compression-threads", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--log-every-repos", type=int, default=1_000)
    parser.add_argument("--supervisor-log-seconds", type=float, default=15.0)
    parser.add_argument("--supervisor-poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-new-repos", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.token_batch_size < 1
        or args.checkpoint_bytes < 1
        or args.checkpoint_repos < 1
        or args.compression_threads < 1
        or args.supervisor_log_seconds <= 0
        or args.supervisor_poll_seconds <= 0
    ):
        parser.error("worker, batch, checkpoint, compression, and log values must be positive")
    try:
        return parallel_main(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
