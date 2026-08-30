# /// script
# dependencies = []
# ///
"""Generate cases and score one submitted function inside a Verifiers runtime.

This script is uploaded and invoked exclusively with ``Runtime.run_uv_script``.
It starts the candidate in a second process inside that already-isolated runtime
so time and memory limits apply to model-written code rather than to the scorer.
Generated cases and expected values never enter ``TaskData`` or trace JSON.
"""

from __future__ import annotations

import base64
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 32 * 1024
CHILD_TIMEOUT_SECONDS = 3.0
RESULT_MARKER = "__VF_CODING_SMOKE_RESULT__="


class StableRng:
    """Tiny fixed LCG so case generation does not depend on stdlib RNG changes."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randint(self, low: int, high: int) -> int:
        return low + self._next() % (high - low + 1)

    def choice(self, values: str) -> str:
        return values[self._next() % len(values)]


def _stride_negate_cases(rng: StableRng) -> list[list[Any]]:
    cases: list[list[Any]] = [[[], 1], [[0], 1], [[1, -2, 3, 4], 2]]
    for _ in range(13):
        values = [rng.randint(-50, 50) for _ in range(rng.randint(0, 30))]
        cases.append([values, rng.randint(1, 9)])
    return cases


def _chunk_mirror_cases(rng: StableRng) -> list[list[Any]]:
    alphabet = "abCD09_- "
    cases: list[list[Any]] = [["", 1], ["abcdefg", 3], ["x", 8]]
    for _ in range(13):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 48)))
        cases.append([text, rng.randint(1, 11)])
    return cases


def _neighbor_delta_cases(rng: StableRng) -> list[list[Any]]:
    cases: list[list[Any]] = [[[]], [[8]], [[3, 3, -2, 10]]]
    for _ in range(13):
        cases.append([[rng.randint(-100, 100) for _ in range(rng.randint(0, 35))]])
    return cases


def _zip_bias_cases(rng: StableRng) -> list[list[Any]]:
    cases: list[list[Any]] = [[[], [], 3], [[1, 2], [9], -1], [[0], [0], 0]]
    for _ in range(13):
        left = [rng.randint(-30, 30) for _ in range(rng.randint(0, 24))]
        right = [rng.randint(-30, 30) for _ in range(rng.randint(0, 24))]
        cases.append([left, right, rng.randint(-20, 20)])
    return cases


def _border_count_cases(rng: StableRng) -> list[list[Any]]:
    cases: list[list[Any]] = [[["x"], "x"], [["xxx"], "x"], [["x", "x", "x"], "x"]]
    alphabet = "xyz."
    for _ in range(13):
        height, width = rng.randint(1, 9), rng.randint(1, 10)
        grid = [
            "".join(rng.choice(alphabet) for _ in range(width)) for _ in range(height)
        ]
        cases.append([grid, rng.choice(alphabet)])
    return cases


def _run_totals_cases(rng: StableRng) -> list[list[Any]]:
    cases: list[list[Any]] = [[[]], [[5]], [[2, 2, 3, 3, 3, -1]]]
    for _ in range(13):
        values: list[int] = []
        for _ in range(rng.randint(0, 12)):
            value = rng.randint(-8, 8)
            values.extend([value] * rng.randint(1, 7))
        cases.append([values])
    return cases


def _stride_negate(args: list[Any]) -> list[int]:
    values, stride = args
    return [
        -value if index % stride == 0 else value for index, value in enumerate(values)
    ]


def _chunk_mirror(args: list[Any]) -> str:
    text, width = args
    return "".join(
        text[start : start + width][::-1] for start in range(0, len(text), width)
    )


def _neighbor_delta(args: list[Any]) -> list[int]:
    (values,) = args
    return values[:1] + [current - previous for previous, current in pairwise(values)]


def _zip_bias(args: list[Any]) -> list[int]:
    left, right, bias = args
    return [a + b + bias for a, b in zip(left, right)]


def _border_count(args: list[Any]) -> int:
    grid, marker = args
    height, width = len(grid), len(grid[0])
    return sum(
        grid[row][column] == marker
        for row in range(height)
        for column in range(width)
        if row in (0, height - 1) or column in (0, width - 1)
    )


def _run_totals(args: list[Any]) -> list[int]:
    (values,) = args
    runs: list[list[int]] = []
    for value in values:
        if not runs or runs[-1][0] != value:
            runs.append([value, 1])
        else:
            runs[-1][1] += 1
    return [value * count for value, count in runs]


TaskDefinition = tuple[
    str,
    Callable[[StableRng], list[list[Any]]],
    Callable[[list[Any]], Any],
]
TASKS: dict[str, TaskDefinition] = {
    "stride-negate": ("pi_smoke_stride_negate", _stride_negate_cases, _stride_negate),
    "chunk-mirror": ("pi_smoke_chunk_mirror", _chunk_mirror_cases, _chunk_mirror),
    "neighbor-delta": (
        "pi_smoke_neighbor_delta",
        _neighbor_delta_cases,
        _neighbor_delta,
    ),
    "zip-bias": ("pi_smoke_zip_bias", _zip_bias_cases, _zip_bias),
    "border-count": ("pi_smoke_border_count", _border_count_cases, _border_count),
    "run-totals": ("pi_smoke_run_totals", _run_totals_cases, _run_totals),
}


def _seed(task_id: str) -> int:
    return 0xC0D10000 + list(TASKS).index(task_id)


def _child_limits() -> None:
    memory = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _metrics(**updates: float) -> dict[str, float]:
    result = {
        "syntax_valid": 0.0,
        "entrypoint_present": 0.0,
        "tests_passed": 0.0,
        "tests_total": 0.0,
        "pass_rate": 0.0,
        "timed_out": 0.0,
        "runtime_error": 0.0,
        "all_tests_passed": 0.0,
    }
    result.update(updates)
    return result


def _runner_source(candidate: str, entrypoint: str) -> str:
    # The candidate is intentionally executed only by the child interpreter.
    return f"""{candidate}

import contextlib as _vf_contextlib
import io as _vf_io
import json as _vf_json
import sys as _vf_sys

_vf_cases = _vf_json.loads(_vf_sys.stdin.read())
_vf_fn = globals().get({entrypoint!r})
_vf_records = []
if not callable(_vf_fn):
    _vf_payload = {{"entrypoint_present": False, "records": []}}
else:
    for _vf_args in _vf_cases:
        try:
            with (
                _vf_contextlib.redirect_stdout(_vf_io.StringIO()),
                _vf_contextlib.redirect_stderr(_vf_io.StringIO()),
            ):
                _vf_value = _vf_fn(*_vf_args)
            _vf_json.dumps(_vf_value, allow_nan=False)
            _vf_records.append({{"ok": True, "value": _vf_value}})
        except BaseException as _vf_error:
            _vf_records.append({{"ok": False, "error": type(_vf_error).__name__}})
    _vf_payload = {{"entrypoint_present": True, "records": _vf_records}}
print(
    {RESULT_MARKER!r}
    + _vf_json.dumps(_vf_payload, allow_nan=False, separators=(",", ":"))
)
"""


def evaluate(task_id: str, encoded_source: str) -> dict[str, float]:
    if task_id not in TASKS:
        raise ValueError(f"unknown task id: {task_id}")
    try:
        source_bytes = base64.urlsafe_b64decode(encoded_source.encode("ascii"))
        source = source_bytes.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid source encoding") from exc
    if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES:
        return _metrics()

    try:
        compile(source, "candidate.py", "exec")
    except SyntaxError:
        return _metrics()

    entrypoint, case_factory, oracle = TASKS[task_id]
    cases = case_factory(StableRng(_seed(task_id)))
    expected = [oracle(args) for args in cases]
    total = float(len(cases))

    with tempfile.TemporaryDirectory(prefix="vf-coding-smoke-") as tmp:
        candidate_path = Path(tmp) / "candidate.py"
        candidate_path.write_text(_runner_source(source, entrypoint), encoding="utf-8")
        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
        stdout_path = Path(tmp) / "stdout.txt"
        stderr_path = Path(tmp) / "stderr.txt"
        with (
            stdout_path.open("wb") as child_stdout,
            stderr_path.open("wb") as child_stderr,
        ):
            child = subprocess.Popen(
                [sys.executable, "-I", "-S", str(candidate_path)],
                stdin=subprocess.PIPE,
                stdout=child_stdout,
                stderr=child_stderr,
                cwd=tmp,
                env=child_env,
                start_new_session=True,
                # The verifier process has no threads; limits must be installed
                # after fork and before any model-written code can execute.
                preexec_fn=_child_limits,  # noqa: PLW1509
            )
            timed_out = False
            try:
                child.communicate(
                    input=json.dumps(cases, separators=(",", ":")).encode(),
                    timeout=CHILD_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # A candidate may fork. Its private process group is torn down on
                # every path, including after the top-level process exits normally.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            if timed_out:
                return _metrics(
                    syntax_valid=1.0,
                    tests_total=total,
                    timed_out=1.0,
                )
        child_output = stdout_path.read_text(encoding="utf-8", errors="replace")

    marker_lines = [
        line[len(RESULT_MARKER) :]
        for line in child_output.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if child.returncode != 0 or not marker_lines:
        return _metrics(
            syntax_valid=1.0,
            tests_total=total,
            runtime_error=1.0,
        )
    try:
        payload = json.loads(marker_lines[-1])
    except json.JSONDecodeError:
        return _metrics(
            syntax_valid=1.0,
            tests_total=total,
            runtime_error=1.0,
        )

    present = float(payload.get("entrypoint_present") is True)
    records = payload.get("records")
    if present == 0.0 and records == []:
        return _metrics(
            syntax_valid=1.0,
            tests_total=total,
        )
    if not isinstance(records, list) or len(records) != len(expected):
        return _metrics(
            syntax_valid=1.0,
            entrypoint_present=present,
            tests_total=total,
            runtime_error=1.0,
        )

    def canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    passed = float(
        sum(
            record.get("ok") is True
            and canonical_json(record.get("value")) == canonical_json(answer)
            for record, answer in zip(records, expected)
            if isinstance(record, dict)
        )
    )
    pass_rate = passed / total if total else 0.0
    return _metrics(
        syntax_valid=1.0,
        entrypoint_present=present,
        tests_passed=passed,
        tests_total=total,
        pass_rate=pass_rate,
        all_tests_passed=float(present == 1.0 and passed == total),
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify.py TASK_ID URLSAFE_BASE64_SOURCE")
    print(json.dumps(evaluate(sys.argv[1], sys.argv[2]), sort_keys=True))


if __name__ == "__main__":
    main()
