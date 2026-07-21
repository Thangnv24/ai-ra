"""Regression tests for complete batch-output handling."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("batch_runner", ROOT / "tests" / "test.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class RunnerGuardTests(unittest.TestCase):
    def test_part2_default_output_is_separate(self) -> None:
        output = RUNNER.default_output_dir(ROOT / "input_part2" / "input" / "input")

        self.assertTrue(output.name.endswith("_part2"))

    def test_failed_batch_keeps_incomplete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text("sample", encoding="utf-8")

            with patch.object(RUNNER, "run_file", side_effect=RuntimeError("request failed")):
                with self.assertRaises(RuntimeError):
                    RUNNER.run_target(input_dir, output_dir, "http://local", 1, 1, None, False)

            self.assertTrue((output_dir / "_INCOMPLETE_RUN.txt").exists())

    def test_complete_batch_removes_incomplete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            input_path = input_dir / "1.txt"
            input_path.write_text("sample", encoding="utf-8")

            def fake_run_file(path, destination, *args):
                output_path = destination / f"{path.stem}.json"
                output_path.write_text("[]", encoding="utf-8")
                return RUNNER.FileResult(path, output_path, 0, True)

            with patch.object(RUNNER, "run_file", side_effect=fake_run_file):
                results = RUNNER.run_target(
                    input_dir, output_dir, "http://local", 1, 1, None, False
                )

            self.assertEqual(len(results), 1)
            self.assertFalse((output_dir / "_INCOMPLETE_RUN.txt").exists())


if __name__ == "__main__":
    unittest.main()
