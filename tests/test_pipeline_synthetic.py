from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import ASSERTION_NEGATED, TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM
from medkg.pipeline import MedicalKGPipeline


class PipelineSyntheticTest(unittest.TestCase):
    def test_extracts_context_and_candidates(self) -> None:
        text = (
            "Benh nhan khong ho. Co tien su hen suyen. "
            "Dang dung aspirin 81 mg po daily."
        )
        concepts = MedicalKGPipeline().process_text(text)
        by_text = {concept.text.casefold(): concept for concept in concepts}
        self.assertEqual(by_text["ho"].type, TYPE_SYMPTOM)
        self.assertIn(ASSERTION_NEGATED, by_text["ho"].assertions)
        self.assertEqual(by_text["hen suyen"].type, TYPE_DIAGNOSIS)
        self.assertTrue(by_text["hen suyen"].candidates)
        drug = next(concept for concept in concepts if concept.type == TYPE_DRUG)
        self.assertIn("243670", drug.candidates)


if __name__ == "__main__":
    unittest.main()
