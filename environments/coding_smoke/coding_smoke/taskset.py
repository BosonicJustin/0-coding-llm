"""Small, original Python tasks for post-training infrastructure smoke tests.

This package is an evaluation/RL environment. Offline SFT consumes a static
``messages`` dataset directly and must not instantiate this taskset.

Candidate programs are never executed by the environment host process. The
scorer sends them to ``Runtime.run_uv_script`` and the task declares
``NEEDS_CONTAINER`` so Verifiers rejects its unsafe subprocess runtime.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import ClassVar

import verifiers.v1 as vf

VERIFY_SCRIPT = (Path(__file__).parent / "verify.py").read_bytes()
MAX_SOURCE_BYTES = 32 * 1024

SYSTEM_PROMPT = (
    "Implement the requested Python function. Return exactly one ```python code "
    "block containing a self-contained function definition. Do not read input, "
    "print output, use the network, or access files."
)


class CodingSmokeData(vf.TaskData):
    """Wire-safe task data; generated verification cases are deliberately absent."""

    task_id: str


def extract_python(reply: str | None) -> str:
    """Extract the first Python/unlabelled fenced block, or accept raw source."""

    text = reply or ""
    fenced = re.search(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (fenced.group(1) if fenced else text).strip()


def _empty_metrics(source_bytes: int) -> dict[str, float]:
    return {
        "source_bytes": float(source_bytes),
        "syntax_valid": 0.0,
        "entrypoint_present": 0.0,
        "tests_passed": 0.0,
        "tests_total": 0.0,
        "pass_rate": 0.0,
        "timed_out": 0.0,
        "runtime_error": 0.0,
        "all_tests_passed": 0.0,
    }


class CodingSmokeTask(vf.Task[CodingSmokeData]):
    """Run one submitted function against generated cases inside an isolated runtime."""

    NEEDS_CONTAINER: ClassVar[bool] = True

    @property
    def key(self) -> str:
        return f"coding-smoke-v1:{self.data.task_id}"

    @staticmethod
    def _require_container(runtime: vf.Runtime) -> None:
        if runtime.type == "subprocess":
            raise RuntimeError(
                "coding-smoke executes untrusted code and refuses the subprocess "
                "runtime; "
                "select Docker, Prime, or another container runtime"
            )

    async def setup(self, runtime: vf.Runtime) -> None:
        self._require_container(runtime)
        # Prepare uv while trusted setup still has egress. verify.py has no
        # third-party dependencies, so execution remains offline after the task
        # network policy applies.
        await runtime.prepare_uv_script(VERIFY_SCRIPT)

    async def _score_source(self, source: str, runtime: vf.Runtime) -> dict[str, float]:
        self._require_container(runtime)
        encoded = source.encode("utf-8")
        metrics = _empty_metrics(len(encoded))
        if not encoded or len(encoded) > MAX_SOURCE_BYTES:
            return metrics

        payload = base64.urlsafe_b64encode(encoded).decode("ascii")
        result = await runtime.run_uv_script(
            VERIFY_SCRIPT,
            args=[self.data.task_id, payload],
        )
        if result.exit_code != 0:
            detail = result.stderr.strip()[-500:]
            raise RuntimeError(f"coding-smoke verifier failed: {detail}")

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("coding-smoke verifier returned no result")
        try:
            scored = json.loads(lines[-1])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("coding-smoke verifier returned invalid JSON") from exc

        expected = set(metrics) - {"source_bytes"}
        if not isinstance(scored, dict) or set(scored) != expected:
            raise RuntimeError(
                "coding-smoke verifier returned an invalid metric schema"
            )
        metrics.update({name: float(scored[name]) for name in expected})
        return metrics

    @vf.metric
    async def evaluate(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        return await self._score_source(extract_python(trace.last_reply), runtime)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        """The sole optimization signal: exact binary pass/fail."""

        return float(trace.metrics.get("all_tests_passed", 0.0))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Exercise the full sandboxed verifier with the package's reference source."""

        source = REFERENCE_SOURCES[self.data.task_id]
        metrics = await self._score_source(source, runtime)
        return metrics["all_tests_passed"] == 1.0


# These prompts and reference implementations were written specifically for this
# smoke environment. They are not copied or transformed from any evaluation corpus.
TASK_SPECS: tuple[tuple[str, str], ...] = (
    (
        "stride-negate",
        (
            "Define `pi_smoke_stride_negate(values: list[int], stride: int) -> "
            "list[int]`. Return a new list where values at zero-based indices "
            "divisible by the positive `stride` are negated and all other values "
            "are unchanged. Do not mutate `values`."
        ),
    ),
    (
        "chunk-mirror",
        (
            "Define `pi_smoke_chunk_mirror(text: str, width: int) -> str`. Split "
            "`text` from left to right into consecutive chunks of positive size "
            "`width`, reverse each chunk independently, and concatenate the "
            "reversed chunks. The final chunk may be shorter than `width`."
        ),
    ),
    (
        "neighbor-delta",
        (
            "Define `pi_smoke_neighbor_delta(values: list[int]) -> list[int]`. "
            "Return an empty list for empty input. Otherwise, the first output is "
            "the first input and each later output is the current input minus the "
            "immediately preceding input."
        ),
    ),
    (
        "zip-bias",
        (
            "Define `pi_smoke_zip_bias(left: list[int], right: list[int], bias: "
            "int) -> list[int]`. Pair elements only while both lists have an "
            "element and return each pair's sum plus `bias`. Ignore any unpaired "
            "suffix and do not mutate either list."
        ),
    ),
    (
        "border-count",
        (
            "Define `pi_smoke_border_count(grid: list[str], marker: str) -> int`. "
            "`grid` is a non-empty rectangular list of non-empty equal-length "
            "strings and `marker` is one character. Count cells equal to `marker` "
            "on the outer border, counting every cell at most once even for a "
            "one-row or one-column grid."
        ),
    ),
    (
        "run-totals",
        (
            "Define `pi_smoke_run_totals(values: list[int]) -> list[int]`. Within "
            "each maximal consecutive run of equal values, replace the run with "
            "one integer equal to the run value multiplied by its length. Preserve "
            "run order; empty input returns an empty list."
        ),
    ),
)

REFERENCE_SOURCES: dict[str, str] = {
    "stride-negate": """\
def pi_smoke_stride_negate(values: list[int], stride: int) -> list[int]:
    return [
        -value if index % stride == 0 else value
        for index, value in enumerate(values)
    ]
""",
    "chunk-mirror": """\
def pi_smoke_chunk_mirror(text: str, width: int) -> str:
    return \"\".join(
        text[start:start + width][::-1]
        for start in range(0, len(text), width)
    )
""",
    "neighbor-delta": """\
def pi_smoke_neighbor_delta(values: list[int]) -> list[int]:
    return values[:1] + [
        current - previous
        for previous, current in zip(values, values[1:])
    ]
""",
    "zip-bias": """\
def pi_smoke_zip_bias(left: list[int], right: list[int], bias: int) -> list[int]:
    return [a + b + bias for a, b in zip(left, right)]
""",
    "border-count": """\
def pi_smoke_border_count(grid: list[str], marker: str) -> int:
    height, width = len(grid), len(grid[0])
    return sum(
        grid[row][column] == marker
        for row in range(height)
        for column in range(width)
        if row in (0, height - 1) or column in (0, width - 1)
    )
""",
    "run-totals": """\
def pi_smoke_run_totals(values: list[int]) -> list[int]:
    totals = []
    for value in values:
        if not totals or value != totals[-1][0]:
            totals.append([value, 1])
        else:
            totals[-1][1] += 1
    return [value * count for value, count in totals]
""",
}


class CodingSmokeTaskset(vf.Taskset[CodingSmokeTask, vf.TasksetConfig]):
    """Six finite, deterministic, externally unaffiliated smoke tasks."""

    def load(self) -> list[CodingSmokeTask]:
        return [
            CodingSmokeTask(
                CodingSmokeData(
                    idx=index,
                    name=task_id,
                    task_id=task_id,
                    system_prompt=SYSTEM_PROMPT,
                    prompt=prompt,
                    network_allow=[],
                    network_block=["*"],
                    resources=vf.TaskResources(cpu=1.0, memory=1.0, disk=1.0),
                    timeout=vf.TaskTimeout(
                        setup=120.0,
                        agent=120.0,
                        scoring=20.0,
                    ),
                ),
                self.config.task,
            )
            for index, (task_id, prompt) in enumerate(TASK_SPECS)
        ]
