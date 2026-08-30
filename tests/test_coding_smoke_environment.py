from __future__ import annotations

import ast
import asyncio
import base64
import importlib
import importlib.util
import json
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import tomllib

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "coding_smoke"
PACKAGE_ROOT = ENV_ROOT / "coding_smoke"


def _verifiers_stubs() -> tuple[types.ModuleType, types.ModuleType]:
    class GenericStub:
        @classmethod
        def __class_getitem__(cls, _params):
            return cls

    class TaskData:
        def __init__(self, **values):
            self.__dict__.update(values)

        def model_dump(self, **_kwargs):
            return dict(self.__dict__)

    class Task(GenericStub):
        def __init__(self, data, config=None):
            self.data = data
            self.config = config

    class Taskset(GenericStub):
        def __init__(self, config):
            self.config = config

    class TasksetConfig:
        def __init__(self):
            self.task = object()

    class TaskResources:
        def __init__(self, **values):
            self.__dict__.update(values)

    class TaskTimeout(TaskResources):
        pass

    def metric(fn):
        return fn

    def reward(*, weight=1.0):
        def decorate(fn):
            fn._vf_weight = weight
            return fn

        return decorate

    vf = types.ModuleType("verifiers.v1")
    vf.TaskData = TaskData
    vf.Task = Task
    vf.Taskset = Taskset
    vf.TasksetConfig = TasksetConfig
    vf.TaskResources = TaskResources
    vf.TaskTimeout = TaskTimeout
    vf.Runtime = object
    vf.Trace = object
    vf.metric = metric
    vf.reward = reward
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = vf
    return verifiers, vf


@contextmanager
def _imported_taskset():
    verifiers, vf = _verifiers_stubs()
    modules = {"verifiers": verifiers, "verifiers.v1": vf}
    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(sys, "path", [str(ENV_ROOT), *sys.path]),
    ):
        for name in ("coding_smoke", "coding_smoke.taskset"):
            sys.modules.pop(name, None)
        try:
            yield importlib.import_module("coding_smoke.taskset")
        finally:
            for name in ("coding_smoke", "coding_smoke.taskset"):
                sys.modules.pop(name, None)


class CodingSmokeEnvironmentTests(unittest.TestCase):
    def test_package_exports_exactly_one_taskset(self) -> None:
        with _imported_taskset():
            package = importlib.import_module("coding_smoke")
            self.assertEqual(package.__all__, ["CodingSmokeTaskset"])

        project = tomllib.loads((ENV_ROOT / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["name"], "coding-smoke")
        self.assertEqual(
            project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"],
            ["coding_smoke"],
        )

    def test_task_data_is_deterministic_and_contains_no_tests(self) -> None:
        with _imported_taskset() as module:
            taskset = module.CodingSmokeTaskset(module.vf.TasksetConfig())
            first = list(taskset.load())
            second = list(taskset.load())

            self.assertEqual(len(first), 6)
            self.assertEqual(
                [task.key for task in first],
                [task.key for task in second],
            )
            self.assertEqual(len({task.key for task in first}), len(first))
            for task in first:
                wire = task.data.model_dump()
                self.assertTrue(
                    set(wire).isdisjoint(
                        {"tests", "hidden_tests", "expected", "solution"}
                    )
                )
                self.assertEqual(wire["network_allow"], [])
                self.assertEqual(wire["network_block"], ["*"])
                self.assertIs(task.NEEDS_CONTAINER, True)

    def test_host_only_delegates_candidate_execution_to_runtime(self) -> None:
        tree = ast.parse((PACKAGE_ROOT / "taskset.py").read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("subprocess", imported)

        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        attrs = {
            node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        }
        self.assertIn("run_uv_script", attrs)
        self.assertIn("prepare_uv_script", attrs)
        self.assertNotIn("exec", attrs)
        self.assertNotIn("eval", attrs)

    def test_reference_sources_score_through_runtime_contract(self) -> None:
        with _imported_taskset() as module:

            class Result:
                exit_code = 0
                stderr = ""
                stdout = json.dumps(
                    {
                        "syntax_valid": 1.0,
                        "entrypoint_present": 1.0,
                        "tests_passed": 16.0,
                        "tests_total": 16.0,
                        "pass_rate": 1.0,
                        "timed_out": 0.0,
                        "runtime_error": 0.0,
                        "all_tests_passed": 1.0,
                    }
                )

            class Runtime:
                type = "docker"

                def __init__(self):
                    self.calls = []

                async def run_uv_script(self, script, args):
                    self.calls.append((script, args))
                    return Result()

            task = module.CodingSmokeTaskset(module.vf.TasksetConfig()).load()[0]
            runtime = Runtime()
            source = module.REFERENCE_SOURCES[task.data.task_id]
            metrics = asyncio.run(task._score_source(source, runtime))

            self.assertEqual(metrics["all_tests_passed"], 1.0)
            self.assertEqual(len(runtime.calls), 1)
            _script, args = runtime.calls[0]
            self.assertEqual(args[0], task.data.task_id)
            self.assertEqual(base64.urlsafe_b64decode(args[1]).decode(), source)

    def test_verifier_has_no_external_dependencies_and_compiles(self) -> None:
        source = (PACKAGE_ROOT / "verify.py").read_text()
        self.assertIn("# dependencies = []", source)
        compile(source, str(PACKAGE_ROOT / "verify.py"), "exec")

    def test_generated_cases_are_fixed_and_cover_every_task(self) -> None:
        with _imported_taskset() as taskset:
            path = PACKAGE_ROOT / "verify.py"
            spec = importlib.util.spec_from_file_location(
                "coding_smoke_verify_test", path
            )
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertIsNotNone(spec.loader)
            assert spec.loader is not None
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)

            task_ids = [task_id for task_id, _prompt in taskset.TASK_SPECS]
            self.assertEqual(list(verifier.TASKS), task_ids)
            for task_id, (_entrypoint, case_factory, oracle) in verifier.TASKS.items():
                first = case_factory(verifier.StableRng(verifier._seed(task_id)))
                second = case_factory(verifier.StableRng(verifier._seed(task_id)))
                self.assertEqual(first, second)
                self.assertEqual(len(first), 16)
                json.dumps([oracle(args) for args in first], allow_nan=False)

    def test_authored_content_does_not_match_mbpp_denylist(self) -> None:
        path = ROOT / "scripts" / "benchmark_guard.py"
        spec = importlib.util.spec_from_file_location("coding_smoke_guard_test", path)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertIsNotNone(spec.loader)
        assert spec.loader is not None
        guard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard_module)
        guard = guard_module.BenchmarkGuard(ROOT / "configs" / "mbpp_denylist.json")

        tree = ast.parse((PACKAGE_ROOT / "taskset.py").read_text())
        values = {
            node.target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"TASK_SPECS", "REFERENCE_SOURCES"}
        }
        hits = [
            (task_id, guard.contamination_reason("Python", source))
            for task_id, source in values["REFERENCE_SOURCES"].items()
            if guard.contamination_reason("Python", source)
        ]
        hits.extend(
            (task_id, guard.contamination_reason("English", prompt))
            for task_id, prompt in values["TASK_SPECS"]
            if guard.contamination_reason("English", prompt)
        )
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
