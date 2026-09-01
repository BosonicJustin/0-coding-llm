from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PretrainPackageImportTest(unittest.TestCase):
    def _run(self, source: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_existing_public_data_exports_resolve_lazily(self) -> None:
        result = self._run(
            """
            import sys
            import pretrain

            assert "pretrain.data" not in sys.modules
            from pretrain import IGNORE_INDEX, PackedShardWriter
            import pretrain.data as data

            assert IGNORE_INDEX == data.IGNORE_INDEX
            assert PackedShardWriter is data.PackedShardWriter
            assert pretrain.PackedShardWriter is data.PackedShardWriter
            assert "pretrain.data" in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_module_and_cli_help_do_not_import_torch(self) -> None:
        result = self._run(
            """
            import importlib.abc
            import runpy
            import sys

            class RejectTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise RuntimeError(f"forbidden torch import: {fullname}")
                    return None

            sys.meta_path.insert(0, RejectTorch())
            import pretrain
            import pretrain.raw_token_cache

            assert "pretrain.data" not in sys.modules
            assert "torch" not in sys.modules
            sys.argv = ["cache_raw_tokens.py", "--help"]
            try:
                runpy.run_path(
                    "scripts/cache_raw_tokens.py",
                    run_name="__main__",
                )
            except SystemExit as error:
                assert error.code == 0
            else:
                raise AssertionError("CLI --help did not exit")
            assert "pretrain.data" not in sys.modules
            assert "torch" not in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("raw-document token caches", result.stdout)


if __name__ == "__main__":
    unittest.main()
