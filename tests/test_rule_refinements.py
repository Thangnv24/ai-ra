from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ASSERTION_FAMILY, ASSERTION_HISTORICAL, TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM
from extraction.context import ContextDetector
from extraction.ner import MedicalNER
from knowledge.ontology import load_ontology_index
from knowledge.retrieval import CandidateRetriever


class RuleRefinementTest(unittest.TestCase):
    def test_diagnosis_imaging_heading_is_not_context_diagnosis(self) -> None:
        text = "Kết quả chẩn đoán hình ảnh: chụp x-quang ngực không có gì đáng chú ý."
        spans = MedicalNER().extract(text)
        self.assertFalse(any(span.type.endswith("ĐOÁN") for span in spans))

    def test_drug_span_keeps_attached_mg_and_x_dose(self) -> None:
        text = "Được chỉ định điều trị aspirin 325mg x 1."
        spans = [span for span in MedicalNER().extract(text) if span.type == TYPE_DRUG]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "aspirin 325mg x 1")
        self.assertEqual(text[spans[0].start : spans[0].end], spans[0].text)

    def test_present_illness_heading_does_not_make_symptom_historical(self) -> None:
        text = "Tiền sử bệnh hiện tại\nTriệu chứng hiện tại: khó thở."
        start = text.index("khó thở")
        assertions = ContextDetector().assertions_for(text, start, start + len("khó thở"), TYPE_SYMPTOM)
        self.assertNotIn(ASSERTION_HISTORICAL, assertions)

    def test_family_assertion_avoids_short_word_false_positives(self) -> None:
        detector = ContextDetector()
        examples = [
            ("duoc chi dinh dieu tri aspirin 325mg x 1", "aspirin", TYPE_DRUG),
            ("dau chan sau khi di bo vai chang - dau nguc", "dau nguc", TYPE_SYMPTOM),
            ("chan doan hinh anh: sieu am goi y tac nghen duong mat", "tac nghen duong mat", TYPE_DIAGNOSIS),
        ]
        for text, phrase, concept_type in examples:
            with self.subTest(text=text):
                start = text.index(phrase)
                assertions = detector.assertions_for(text, start, start + len(phrase), concept_type)
                self.assertNotIn(ASSERTION_FAMILY, assertions)

    def test_family_assertion_keeps_explicit_kinship_context(self) -> None:
        text = "me bi tang huyet ap"
        start = text.index("tang huyet ap")
        assertions = ContextDetector().assertions_for(text, start, start + len("tang huyet ap"), TYPE_DIAGNOSIS)
        self.assertIn(ASSERTION_FAMILY, assertions)

    def test_aspirin_325_dose_prefers_matching_rxnorm_candidate(self) -> None:
        retriever = CandidateRetriever(load_ontology_index())
        candidates = retriever.candidates_for("aspirin 325mg x 1", TYPE_DRUG, limit=3)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "1191")


if __name__ == "__main__":
    unittest.main()
