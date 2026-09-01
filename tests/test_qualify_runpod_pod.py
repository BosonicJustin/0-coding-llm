from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import qualify_runpod_pod as qualify
from pretrain import run_authority


def _gpu_observation() -> dict[str, object]:
    topology_raw = "fixture"
    devices = [
        {
            "visible_index": index,
            "physical_index": index,
            "uuid": f"GPU-{index:032x}",
            "name": "Fixture H100",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "compute_capability": [9, 0],
            "total_memory_bytes": 80 * 1024**3,
            "available_memory_bytes": 79 * 1024**3,
            "multiprocessor_count": 120,
            "bf16_supported": True,
            "nvidia_smi_memory_bytes": 80 * 1024**3,
            "mig_mode": "disabled",
        }
        for index in range(qualify.WORLD_SIZE)
    ]
    topology = [
        ["X" if left == right else "NV18" for right in range(qualify.WORLD_SIZE)]
        for left in range(qualify.WORLD_SIZE)
    ]
    return {
        "devices": devices,
        "peer_access_matrix": [
            [True for _ in range(qualify.WORLD_SIZE)]
            for _ in range(qualify.WORLD_SIZE)
        ],
        "nvidia_topology": {
            "labels_in_visible_order": [
                f"GPU{index}" for index in range(qualify.WORLD_SIZE)
            ],
            "matrix": topology,
            "nvlink_pairs": 15,
            "possible_pairs": 15,
            "raw_sha256": hashlib.sha256(topology_raw.encode("utf-8")).hexdigest(),
            "raw": topology_raw,
        },
        "driver_version": "570.86.15",
        "driver_supported_cuda_version": "12.8",
        "cuda_runtime_version": "12.8",
        "torch_version": "2.7.1+cu128",
        "cudnn_version": "9.8.0",
        "nccl_version": "2.25.1",
        "distributed_nccl_available": True,
        "torch_cuda_arch_list": ["sm_90"],
        "compiled_arch_supported": True,
        "nccl_smoke": {
            "status": "pass",
            "backend": "nccl",
            "world_size": 6,
            "completed_ranks": 6,
            "all_reduce_sum": 21,
            "all_reduce_dtype": "bfloat16",
            "local_ranks": list(range(6)),
            "device_uuids_in_rank_order": [
                f"GPU-{index:032x}" for index in range(6)
            ],
            "rank_results_sha256": "b" * 64,
            "elapsed_seconds": 1.0,
        },
    }


def _mount(
    *,
    path: str,
    device: int,
    classification: str,
    read_only: bool,
    mount_read_only: bool,
    filesystem_type: str,
    free_bytes: int,
    minimum_free_bytes: int,
) -> dict[str, object]:
    return {
        "path": path,
        "device": device,
        "mount_point": path,
        "filesystem_type": filesystem_type,
        "mount_source": "fixture",
        "read_only": read_only,
        "mount_read_only": mount_read_only,
        "classification": classification,
        "total_bytes": 2 * free_bytes,
        "used_bytes": free_bytes,
        "free_bytes": free_bytes,
        "minimum_free_bytes": minimum_free_bytes,
    }


def _host_observation() -> dict[str, object]:
    gib = 1024**3
    return {
        "storage": {
            "network": _mount(
                path="/network-volume",
                device=11,
                classification="network",
                read_only=False,
                mount_read_only=False,
                filesystem_type="nfs4",
                free_bytes=500 * gib,
                minimum_free_bytes=200 * gib,
            ),
            "local_work": _mount(
                path="/workspace",
                device=22,
                classification="local-or-block",
                read_only=False,
                mount_read_only=False,
                filesystem_type="ext4",
                free_bytes=200 * gib,
                minimum_free_bytes=100 * gib,
            ),
            "local_data": _mount(
                path="/workspace/data-ro",
                device=22,
                classification="local-or-block",
                read_only=True,
                mount_read_only=True,
                filesystem_type="ext4",
                free_bytes=200 * gib,
                minimum_free_bytes=1,
            ),
            "shared_memory": _mount(
                path="/dev/shm",
                device=33,
                classification="ephemeral",
                read_only=False,
                mount_read_only=False,
                filesystem_type="tmpfs",
                free_bytes=32 * gib,
                minimum_free_bytes=16 * gib,
            ),
            "receipt_parent": "/network-volume/audits",
        },
        "data": {"status": "pass"},
        "environment": {
            "required": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "secrets_recorded": False,
        },
        "resource_limits": {
            "nofile_sufficient": True,
            "stack_sufficient": True,
            "memlock_unlimited": True,
        },
        "wandb": {
            "available": True,
            "mode": "offline",
            "credential_value_recorded": False,
        },
        "package_lock": {
            "lock": {
                "path": "/network-volume/lock.json",
                "bytes": 100,
                "sha256": "c" * 64,
            },
            "python": {
                "implementation": "cpython",
                "version": "3.12.11",
                "executable": "/workspace/pretrain-venv/bin/python",
                "executable_bytes": 100,
                "executable_sha256": "a" * 64,
            },
            "packages_sha256": "e" * 64,
            "package_count": 12,
            "torch_version": "2.7.1+cu128",
            "numpy_version": "2.2.6",
        },
        "platform": {"system": "Linux"},
    }


def _observation() -> dict[str, object]:
    source_artifact = lambda path, digest: {
        "path": path,
        "bytes": 100,
        "sha256": digest,
    }
    return {
        "gpu": _gpu_observation(),
        "host": _host_observation(),
        "source": {
            "qualification_script": source_artifact(
                "/workspace/repo/scripts/qualify_runpod_pod.py", "d" * 64
            ),
            "requirements_train": source_artifact(
                "/workspace/repo/requirements-train.txt", "1" * 64
            ),
            "requirements_wandb": source_artifact(
                "/workspace/repo/requirements-wandb.txt", "2" * 64
            ),
            "git": {
                "project_root": "/workspace/repo",
                "commit": "3" * 40,
                "tree": "4" * 40,
                "head_archive_sha256": "5" * 64,
                "head_archive_bytes": 10240,
                "branch": "main",
                "origin_sha256": None,
                "clean": True,
            },
            "argv": [
                "/workspace/repo/scripts/qualify_runpod_pod.py",
                "verify",
                "--package-lock",
                "/network-volume/lock.json",
            ],
        },
    }


class PodQualificationValidationTests(unittest.TestCase):
    def test_cpu_mock_cli_publishes_once_and_preserves_failure_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "hardware.json"
            argv = [
                "verify",
                "--network-root",
                str(root),
                "--local-work-root",
                str(root),
                "--local-data-root",
                str(root),
                "--tokenizer",
                str(root),
                "--train-order-manifest",
                str(root / "train.json"),
                "--validation-order-manifest",
                str(root / "validation.json"),
                "--package-lock",
                str(root / "lock.json"),
                "--wandb-dir",
                str(root),
                "--receipt",
                str(receipt),
                "--wandb-mode",
                "offline",
                "--nvlink-policy",
                "require-all",
                "--omp-threads",
                "8",
                "--eval-batches",
                "2",
                "--minimum-network-free-bytes",
                "1GiB",
                "--minimum-local-free-bytes",
                "1GiB",
            ]
            stdout = io.StringIO()
            with mock.patch.object(
                qualify, "collect_observation", return_value=_observation()
            ), redirect_stdout(stdout):
                self.assertEqual(qualify.main(argv), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["path"], str(receipt.resolve()))
            stderr = io.StringIO()
            with mock.patch.object(
                qualify, "collect_observation", return_value=_observation()
            ), redirect_stderr(stderr):
                self.assertEqual(qualify.main(argv), 2)
            self.assertIn("Refusing to overwrite", stderr.getvalue())

    def test_passing_receipt_is_run_authority_hardware_contract(self) -> None:
        receipt = qualify.build_hardware_receipt(
            _observation(),
            nvlink_policy="require-all",
            created_utc="2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(receipt["format"], run_authority.HARDWARE_FORMAT)
        self.assertEqual(receipt["gpu_count"], 6)
        self.assertEqual(receipt["qualification"]["status"], "pass")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            published = qualify.publish_receipt(path, receipt)
            inspected = run_authority.inspect_hardware_contract(path)
            self.assertEqual(inspected["expected"], receipt)
            sidecar = path.with_name("hardware.json.sha256")
            self.assertEqual(
                sidecar.read_text(encoding="ascii"),
                f"{published['sha256']}  hardware.json\n",
            )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), published["sha256"]
            )

    def test_receipt_pair_is_write_once(self) -> None:
        receipt = qualify.build_hardware_receipt(
            _observation(),
            nvlink_policy="require-all",
            created_utc="2026-09-01T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            qualify.publish_receipt(path, receipt)
            with self.assertRaisesRegex(qualify.PodQualificationError, "overwrite"):
                qualify.publish_receipt(path, receipt)

    def test_run_authority_rejects_hardware_receipt_mutation_after_sidecar(self) -> None:
        receipt = qualify.build_hardware_receipt(
            _observation(),
            nvlink_policy="require-all",
            created_utc="2026-09-01T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            qualify.publish_receipt(path, receipt)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(run_authority.RunAuthorityError, "sidecar"):
                run_authority.inspect_hardware_contract(path)

    def test_exactly_six_homogeneous_native_bf16_gpus_are_required(self) -> None:
        for mutation, message in (
            (lambda gpu: gpu["devices"].pop(), "Exactly six"),
            (
                lambda gpu: gpu["devices"][5].__setitem__("name", "Other GPU"),
                "heterogeneous",
            ),
            (
                lambda gpu: gpu["devices"][2].__setitem__("bf16_supported", False),
                "BF16",
            ),
        ):
            with self.subTest(message=message):
                gpu = _gpu_observation()
                mutation(gpu)
                with self.assertRaisesRegex(qualify.PodQualificationError, message):
                    qualify.validate_gpu_observation(gpu, nvlink_policy="require-all")

    def test_peer_matrix_and_nccl_collective_are_fail_closed(self) -> None:
        peer = _gpu_observation()
        peer["peer_access_matrix"][1][4] = False
        peer["peer_access_matrix"][4][1] = False
        with self.assertRaisesRegex(qualify.PodQualificationError, "peer access"):
            qualify.validate_gpu_observation(peer, nvlink_policy="require-all")

        smoke = _gpu_observation()
        smoke["nccl_smoke"]["all_reduce_sum"] = 20
        with self.assertRaisesRegex(qualify.PodQualificationError, "all-reduce"):
            qualify.validate_gpu_observation(smoke, nvlink_policy="require-all")

        mapping = _gpu_observation()
        mapping["nccl_smoke"]["device_uuids_in_rank_order"][5] = mapping[
            "nccl_smoke"
        ]["device_uuids_in_rank_order"][0]
        with self.assertRaisesRegex(qualify.PodQualificationError, "rank-to-GPU"):
            qualify.validate_gpu_observation(mapping, nvlink_policy="require-all")

    def test_nvlink_policy_is_explicit(self) -> None:
        gpu = _gpu_observation()
        gpu["nvidia_topology"]["matrix"] = [
            ["X" if left == right else "PIX" for right in range(6)]
            for left in range(6)
        ]
        gpu["nvidia_topology"]["nvlink_pairs"] = 0
        qualify.validate_gpu_observation(gpu, nvlink_policy="observe")
        with self.assertRaisesRegex(qualify.PodQualificationError, "at least one"):
            qualify.validate_gpu_observation(gpu, nvlink_policy="require-any")
        with self.assertRaisesRegex(qualify.PodQualificationError, "all 15"):
            qualify.validate_gpu_observation(gpu, nvlink_policy="require-all")

    def test_driver_cuda_and_compiled_architecture_must_cover_runtime(self) -> None:
        gpu = _gpu_observation()
        gpu["driver_supported_cuda_version"] = "12.4"
        with self.assertRaisesRegex(qualify.PodQualificationError, "older"):
            qualify.validate_gpu_observation(gpu, nvlink_policy="require-all")
        gpu = _gpu_observation()
        gpu["compiled_arch_supported"] = False
        with self.assertRaisesRegex(qualify.PodQualificationError, "architecture"):
            qualify.validate_gpu_observation(gpu, nvlink_policy="require-all")

    def test_storage_resource_and_wandb_gates_fail_closed(self) -> None:
        cases = (
            (
                lambda host: host["storage"]["network"].__setitem__(
                    "classification", "local-or-block"
                ),
                "Network root",
            ),
            (
                lambda host: host["storage"]["local_data"].__setitem__(
                    "mount_read_only", False
                ),
                "read-only",
            ),
            (
                lambda host: host["storage"]["shared_memory"].__setitem__(
                    "free_bytes", 1
                ),
                "shared memory capacity",
            ),
            (
                lambda host: host["resource_limits"].__setitem__(
                    "memlock_unlimited", False
                ),
                "MEMLOCK",
            ),
            (
                lambda host: host["wandb"].__setitem__("available", False),
                "W&B",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                host = _host_observation()
                mutate(host)
                with self.assertRaisesRegex(qualify.PodQualificationError, message):
                    qualify.validate_host_observation(host)

    def test_deterministic_environment_requires_exact_values_without_secrets(self) -> None:
        environment = {
            **qualify._EXPECTED_ENVIRONMENT,
            "OMP_NUM_THREADS": "8",
            "WANDB_MODE": "offline",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",
            "WANDB_API_KEY": "must-not-be-recorded",
        }
        evidence = qualify.inspect_deterministic_environment(
            environment, wandb_mode="offline", omp_threads=8
        )
        self.assertNotIn("WANDB_API_KEY", json.dumps(evidence))
        changed = dict(environment)
        changed["PYTHONHASHSEED"] = "1"
        with self.assertRaisesRegex(qualify.PodQualificationError, "not frozen"):
            qualify.inspect_deterministic_environment(
                changed, wandb_mode="offline", omp_threads=8
            )
        disabled = dict(environment)
        disabled["NCCL_P2P_DISABLE"] = "1"
        with self.assertRaisesRegex(qualify.PodQualificationError, "disabled"):
            qualify.inspect_deterministic_environment(
                disabled, wandb_mode="offline", omp_threads=8
            )


class NvidiaEvidenceParsingTests(unittest.TestCase):
    def test_inventory_and_visible_uuid_selection(self) -> None:
        rows = "\n".join(
            f"{index}, GPU-{index:032x}, Fixture H100, 81920, 9.0, "
            f"570.86.15, 00000000:{index + 1:02X}:00.0, Disabled"
            for index in range(8)
        )
        inventory = qualify._parse_nvidia_smi_inventory(rows)
        selected = qualify._resolve_visible_inventory(
            ["1", "2", "3", "4", "5", "6"], inventory
        )
        self.assertEqual([item["index"] for item in selected], [1, 2, 3, 4, 5, 6])
        selected_by_uuid = qualify._resolve_visible_inventory(
            [item["uuid"] for item in selected], inventory
        )
        self.assertEqual(selected, selected_by_uuid)
        self.assertEqual(
            qualify._normalize_pci_bus_id("0000:0A:00.0"), "00000000:0a:00.0"
        )

    def test_topology_parser_preserves_visible_physical_order(self) -> None:
        header = "        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 CPU Affinity"
        rows = []
        for left in range(6):
            links = ["X" if left == right else "NV18" for right in range(6)]
            rows.append(f"GPU{left}   " + "  ".join(links) + "  0-31")
        parsed = qualify._parse_topology("\n".join([header, *rows]), physical_indices=[5, 4, 3, 2, 1, 0])
        self.assertEqual(parsed["labels_in_visible_order"], ["GPU5", "GPU4", "GPU3", "GPU2", "GPU1", "GPU0"])
        self.assertEqual(parsed["nvlink_pairs"], 15)

    def test_byte_parser_is_binary_and_rejects_zero(self) -> None:
        self.assertEqual(qualify.parse_bytes("16GiB"), 16 * 1024**3)
        with self.assertRaises(Exception):
            qualify.parse_bytes("0")


class NcclWorkerContractTests(unittest.TestCase):
    def test_worker_gathers_rank_tensor_with_matching_int64_outputs(self) -> None:
        class FakeTensor:
            def __init__(self, value: int, dtype: object) -> None:
                self.value = value
                self.dtype = dtype

            def item(self) -> int:
                return self.value

        bfloat16 = object()
        int64 = object()
        fake_torch = types.ModuleType("torch")
        fake_distributed = types.ModuleType("torch.distributed")
        calls: dict[str, object] = {}

        def tensor(values, *, device, dtype):
            del device
            return FakeTensor(int(values[0]), dtype)

        def zeros_like(value):
            return FakeTensor(0, value.dtype)

        def all_reduce(value):
            value.value = sum(range(1, qualify.WORLD_SIZE + 1))

        def all_gather(outputs, value):
            calls["gather_dtypes"] = ([item.dtype for item in outputs], value.dtype)
            if any(item.dtype is not value.dtype for item in outputs):
                raise ValueError("all_gather dtype mismatch")
            for rank, output in enumerate(outputs):
                output.value = rank

        def broadcast(value, *, src):
            self.assertEqual(src, 0)
            value.value = 8675309

        fake_torch.bfloat16 = bfloat16
        fake_torch.int64 = int64
        fake_torch.tensor = tensor
        fake_torch.zeros_like = zeros_like
        fake_torch.cuda = types.SimpleNamespace(
            set_device=lambda rank: calls.setdefault("set_device", rank),
            synchronize=lambda rank: calls.setdefault("synchronize", rank),
            get_device_properties=lambda rank: types.SimpleNamespace(
                uuid=f"GPU-{rank:032x}"
            ),
        )
        fake_torch.distributed = fake_distributed
        fake_distributed.init_process_group = lambda *, backend: calls.setdefault(
            "backend", backend
        )
        fake_distributed.all_reduce = all_reduce
        fake_distributed.all_gather = all_gather
        fake_distributed.broadcast = broadcast
        fake_distributed.barrier = lambda: calls.setdefault("barrier", True)
        fake_distributed.destroy_process_group = lambda: calls.setdefault(
            "destroyed", True
        )

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "torch.distributed": fake_distributed},
        ), mock.patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "6"},
        ):
            self.assertEqual(qualify._nccl_worker(Path(temporary)), 0)
            payload = json.loads(
                (Path(temporary) / "rank-0.json").read_text(encoding="utf-8")
            )

        output_dtypes, input_dtype = calls["gather_dtypes"]
        self.assertTrue(all(dtype is input_dtype for dtype in output_dtypes))
        self.assertIs(input_dtype, int64)
        self.assertEqual(payload["all_gather_ranks"], list(range(6)))
        self.assertEqual(payload["device_uuid"], "GPU-00000000000000000000000000000000")
        self.assertEqual(calls["backend"], "nccl")
        self.assertTrue(calls["destroyed"])

    def test_run_timeout_invokes_process_tree_cleanup(self) -> None:
        process = mock.Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired(["fixture"], 1)
        with mock.patch.object(
            qualify.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(qualify, "_terminate_process_tree") as terminate:
            with self.assertRaisesRegex(qualify.PodQualificationError, "timed out"):
                qualify._run(["fixture"], timeout_seconds=1)
        terminate.assert_called_once_with(process)
        self.assertEqual(
            popen.call_args.kwargs["start_new_session"], os.name == "posix"
        )


if __name__ == "__main__":
    unittest.main()
