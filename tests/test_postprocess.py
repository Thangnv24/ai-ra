from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ASSERTION_HISTORICAL, TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.schema import Concept
from extraction.context import ContextDetector
from services.postprocess import refine_concepts


class PostprocessTest(unittest.TestCase):
    def test_trims_progressive_edema_symptom(self) -> None:
        text = "Phù ngoại vi tăng dần trong vài tuần gần đây."
        concepts = [Concept(text[:-1], TYPE_SYMPTOM, (0, len(text) - 1))]

        refined = refine_concepts(text, concepts)

        self.assertEqual(refined[0].text, "Phù ngoại vi")
        self.assertEqual(text[refined[0].position[0] : refined[0].position[1]], refined[0].text)

    def test_trims_repeated_weight_gain_symptom(self) -> None:
        text = "Tăng tăng cân 3 pound trong 7 ngày qua."
        concepts = [Concept(text[:-1], TYPE_SYMPTOM, (0, len(text) - 1))]

        refined = refine_concepts(text, concepts)

        self.assertEqual(refined[0].text, "tăng cân")
        self.assertEqual(refined[0].position, (5, 13))

    def test_lab_result_keeps_value_or_qualifier_only(self) -> None:
        text = "UA: 12 bạch cầu, âm tính nitrite."
        first_start = text.index("12")
        second_start = text.index("âm")
        concepts = [
            Concept("12 bạch cầu", TYPE_TEST_RESULT, (first_start, first_start + len("12 bạch cầu"))),
            Concept("âm tính nitrite", TYPE_TEST_RESULT, (second_start, second_start + len("âm tính nitrite"))),
        ]

        refined = refine_concepts(text, concepts)

        self.assertEqual([concept.text for concept in refined], ["12", "âm tính"])

    def test_drops_diagnosis_inside_test_name(self) -> None:
        text = "xét nghiệm phân tìm cryptosporidium âm tính."
        test_name = "xét nghiệm phân tìm cryptosporidium"
        diag = "cryptosporidium"
        concepts = [
            Concept(test_name, TYPE_TEST_NAME, (0, len(test_name))),
            Concept(diag, TYPE_DIAGNOSIS, (text.index(diag), text.index(diag) + len(diag)), candidates=("A07.2",)),
        ]

        refined = refine_concepts(text, concepts)

        self.assertEqual([concept.type for concept in refined], [TYPE_TEST_NAME])

    def test_reapplies_historical_context_for_drug(self) -> None:
        text = "Bệnh nhân đã dùng levafloxacin trước nhập viện."
        drug = "levafloxacin"
        start = text.index(drug)
        concepts = [Concept(drug, TYPE_DRUG, (start, start + len(drug)), candidates=("123",))]

        refined = refine_concepts(text, concepts, context_detector=ContextDetector())

        self.assertIn(ASSERTION_HISTORICAL, refined[0].assertions)


if __name__ == "__main__":
    unittest.main()
