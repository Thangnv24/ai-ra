from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import TYPE_SYMPTOM
from medkg.io import load_json, write_output
from medkg.schema import Concept


class IOTest(unittest.TestCase):
    def test_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_output(path, [Concept("ho", TYPE_SYMPTOM, (0, 2))])
            payload = load_json(path)
            self.assertEqual(payload[0]["text"], "ho")
            self.assertNotIn("candidates", payload[0])


if __name__ == "__main__":
    unittest.main()
