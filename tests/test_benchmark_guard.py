import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_guard import BenchmarkGuard, build_mbpp_manifest


CODE = """def mbpp_unique_example(values):
    selected = [value * 17 for value in values if value % 3 == 1]
    return tuple(reversed(selected))
"""


class BenchmarkGuardTest(unittest.TestCase):
    def make_guard(self, directory: Path) -> BenchmarkGuard:
        row = {
            "text": "Write a function whose deliberately distinctive synthetic behavior is tested.",
            "code": CODE,
            "test_list": ["assert mbpp_unique_example([1, 2, 4]) == (68, 17)"],
            "challenge_test_list": [],
        }
        manifest = build_mbpp_manifest([row] * 974, "0" * 64, "https://example.invalid/mbpp.jsonl")
        path = directory / "denylist.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return BenchmarkGuard(path)

    def test_exact_and_embedded_python_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = self.make_guard(Path(temporary))
            self.assertEqual(guard.contamination_reason("Python", CODE), "mbpp-exact-code")
            embedded = "# wrapper\n" + CODE + "\nprint(mbpp_unique_example([1]))\n"
            self.assertIsNotNone(guard.contamination_reason("Python", embedded))

    def test_test_assertion_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = self.make_guard(Path(temporary))
            content = "def test_behavior():\n    assert mbpp_unique_example([1, 2, 4]) == (68, 17)\n"
            self.assertEqual(
                guard.contamination_reason("Python", content), "mbpp-problem-or-test"
            )
            self.assertEqual(
                guard.contamination_reason("English", content), "mbpp-problem-or-test"
            )

    def test_other_languages_and_unrelated_python_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = self.make_guard(Path(temporary))
            self.assertIsNone(guard.contamination_reason("Java", CODE))
            unrelated = "def mbpp_unique_example(values):\n    return list(values)\n"
            self.assertIsNone(guard.contamination_reason("Python", unrelated))


if __name__ == "__main__":
    unittest.main()
