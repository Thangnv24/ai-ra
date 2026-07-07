from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import ASSERTION_HISTORICAL, TYPE_DRUG
from medkg.schema import Concept, validate_output


class SchemaTest(unittest.TestCase):
    def test_valid_drug_concept(self) -> None:
        source = "aspirin 81 mg"
        concept = Concept(
            text=source,
            type=TYPE_DRUG,
            position=(0, len(source)),
            assertions=(ASSERTION_HISTORICAL,),
            candidates=("243670",),
        )
        self.assertEqual(validate_output([concept.to_dict()], source_text=source), [])

    def test_rejects_bad_position(self) -> None:
        payload = [{"text": "aspirin", "type": TYPE_DRUG, "assertions": [], "position": [1, 4], "candidates": []}]
        errors = validate_output(payload, source_text="aspirin")
        self.assertTrue(any("does not match" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
