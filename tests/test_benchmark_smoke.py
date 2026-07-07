from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BenchmarkSmokeTest(unittest.TestCase):
    def test_benchmark_script_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "inputs"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text("aspirin 81 mg po daily", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark.py"),
                    "--input-dir",
                    str(input_dir),
                    "--limit",
                    "1",
                    "--no-write",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertIn('"files": 1', proc.stdout)


if __name__ == "__main__":
    unittest.main()
