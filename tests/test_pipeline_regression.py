"""Fast regression tests for deterministic pipeline decisions."""

from __future__ import annotations

import unittest
from pathlib import Path

from core.config import (
    ASSERTION_HISTORICAL,
    ASSERTION_NEGATED,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)
from core.medication import medication_match_score
from core.schema import Concept, validate_output
from extraction.context import ContextDetector
from extraction.llm_entities import _chunk_units, _span_from_mention_with_reason
from extraction.ner import MedicalNER, SpanCandidate, resolve_span_types
from extraction.sectioning import TextChunk, split_chunks
from knowledge.candidates import CandidateHit, CandidateRecord, SlimCandidateIndex
from knowledge.retrieval import _select_diagnosis_codes
from services.pipeline import _merge_span_candidates_with_summary
from services.postprocess import refine_concepts


ROOT = Path(__file__).resolve().parents[1]


class EntityOccurrenceTests(unittest.TestCase):
    def test_aligns_repeated_quotes_to_distinct_occurrences(self) -> None:
        quote = "\u0111\u00e1nh tr\u1ed1ng ng\u1ef1c"
        text = f"{quote}, {quote}"
        chunk = TextChunk("c1", "present_illness", 0, len(text), text)
        units = _chunk_units(chunk)
        used: dict[tuple[str, str], set[int]] = {}

        spans = []
        for occurrence_index in (0, 1):
            span, reason = _span_from_mention_with_reason(
                text,
                chunk,
                {
                    "unit_id": "c1u1",
                    "quote": quote,
                    "occurrence_index": occurrence_index,
                    "type": TYPE_SYMPTOM,
                    "confidence": 0.9,
                },
                units=units,
                used_occurrences=used,
            )
            self.assertIsNone(reason)
            self.assertIsNotNone(span)
            spans.append(span)

        self.assertNotEqual(
            (spans[0].start, spans[0].end),
            (spans[1].start, spans[1].end),
        )
        self.assertEqual([text[span.start : span.end] for span in spans], [quote, quote])

    def test_first_ten_inputs_keep_chunk_and_unit_offsets(self) -> None:
        for file_number in range(1, 11):
            text = (ROOT / "input" / f"{file_number}.txt").read_text(encoding="utf-8")
            chunks = split_chunks(text, max_chars=1000, overlap=0)
            self.assertTrue(chunks, file_number)
            self.assertLessEqual(max(len(chunk.text) for chunk in chunks), 1000, file_number)
            for chunk in chunks:
                for unit in _chunk_units(chunk):
                    self.assertEqual(text[unit.start : unit.end], unit.text, (file_number, unit.unit_id))

    def test_rule_extractor_keeps_all_input_one_palpitations(self) -> None:
        phrase = "\u0111\u00e1nh tr\u1ed1ng ng\u1ef1c"
        text = (ROOT / "input" / "1.txt").read_text(encoding="utf-8")
        spans = MedicalNER().extract(text)

        self.assertEqual(sum(span.text.casefold() == phrase for span in spans), 10)


class SpanAndTypeTests(unittest.TestCase):
    def test_rule_span_wins_over_broad_llm_overlap(self) -> None:
        rule = SpanCandidate(5, 20, "rule", TYPE_SYMPTOM, 0.8, source="rule")
        broad_llm = SpanCandidate(0, 30, "llm", TYPE_SYMPTOM, 0.99, source="llm")
        selected, summary = _merge_span_candidates_with_summary([broad_llm, rule])

        self.assertEqual(selected, [rule])
        self.assertEqual(summary.overlap_conflicts, 1)

    def test_resolves_sinus_rhythm_as_test_result(self) -> None:
        phrase = "Nh\u1ecbp xoang"
        text = f"\u0110i\u1ec7n t\u00e2m \u0111\u1ed3 ghi nh\u1eadn {phrase}."
        start = text.index(phrase)
        span = SpanCandidate(start, start + len(phrase), phrase, TYPE_DIAGNOSIS, 0.8, source="llm")

        resolved = resolve_span_types(text, [span])

        self.assertEqual(resolved[0].type, TYPE_TEST_RESULT)

    def test_resolves_disease_terms_from_each_occurrence_context(self) -> None:
        pneumonia = "vi\u00eam ph\u1ed5i"
        syncope = "ng\u1ea5t x\u1ec9u"
        atrial_ectopy = "ngo\u1ea1i t\u00e2m thu nh\u0129"
        cases = (
            (f"Ch\u1ea9n \u0111o\u00e1n: {pneumonia}.", pneumonia, TYPE_TEST_RESULT, TYPE_DIAGNOSIS),
            (f"X-quang ng\u1ef1c cho th\u1ea5y {pneumonia}.", pneumonia, TYPE_DIAGNOSIS, TYPE_TEST_RESULT),
            (
                f"monitor Holter cho th\u1ea5y nh\u1ecbp xoang. Ghi nh\u1eadn {atrial_ectopy}.",
                atrial_ectopy,
                TYPE_DIAGNOSIS,
                TYPE_TEST_RESULT,
            ),
            (f"G\u1ea7n \u0111\u00e2y \u0111\u01b0\u1ee3c ch\u1ea9n \u0111o\u00e1n {syncope}.", syncope, TYPE_SYMPTOM, TYPE_DIAGNOSIS),
            (f"Tri\u1ec7u ch\u1ee9ng hi\u1ec7n t\u1ea1i: {syncope}.", syncope, TYPE_SYMPTOM, TYPE_SYMPTOM),
        )
        for text, phrase, proposed_type, expected_type in cases:
            with self.subTest(text=text):
                start = text.index(phrase)
                span = SpanCandidate(start, start + len(phrase), phrase, proposed_type, 0.8, source="llm")
                resolved = resolve_span_types(text, [span])
                self.assertEqual(resolved[0].type, expected_type)

    def test_result_and_diagnosis_headings_change_the_same_term_type(self) -> None:
        cardiomegaly = "tim to"
        text = (
            f"K\u1ebft qu\u1ea3 ch\u1ea9n \u0111o\u00e1n h\u00ecnh \u1ea3nh\n- {cardiomegaly}\n"
            f"C\u00e1c ph\u00e1t hi\u1ec7n ch\u1ea9n \u0111o\u00e1n kh\u00e1c\n- {cardiomegaly}"
        )
        first_start = text.index(cardiomegaly)
        second_start = text.rindex(cardiomegaly)
        spans = [
            SpanCandidate(first_start, first_start + len(cardiomegaly), cardiomegaly, TYPE_DIAGNOSIS),
            SpanCandidate(second_start, second_start + len(cardiomegaly), cardiomegaly, TYPE_DIAGNOSIS),
        ]

        resolved = resolve_span_types(text, spans)

        self.assertEqual([span.type for span in resolved], [TYPE_TEST_RESULT, TYPE_DIAGNOSIS])

    def test_procedure_target_is_not_confused_with_its_finding(self) -> None:
        nodule = "n\u1ed1t tuy\u1ebfn gi\u00e1p"
        text = (
            f"Th\u1ee7 thu\u1eadt \u0111\u00e3 th\u1ef1c hi\u1ec7n: Ch\u1ecdc h\u00fat {nodule}. "
            f"T\u1ed5n th\u01b0\u01a1ng ghi nh\u1eadn: {nodule}."
        )
        first_start = text.index(nodule)
        second_start = text.rindex(nodule)
        spans = [
            SpanCandidate(first_start, first_start + len(nodule), nodule, TYPE_DIAGNOSIS),
            SpanCandidate(second_start, second_start + len(nodule), nodule, TYPE_DIAGNOSIS),
        ]

        resolved = resolve_span_types(text, spans)

        self.assertEqual([span.type for span in resolved], [TYPE_DIAGNOSIS, TYPE_TEST_RESULT])

        narrative = f"Kh\u00f4ng ghi nh\u1eadn kh\u00f3 th\u1edf; \u0111\u00e1nh gi\u00e1 k\u1ebft qu\u1ea3 ch\u1ecdc h\u00fat c\u1ee7a {nodule}."
        target_start = narrative.index(nodule)
        target = SpanCandidate(target_start, target_start + len(nodule), nodule, TYPE_DIAGNOSIS)
        self.assertEqual(resolve_span_types(narrative, [target])[0].type, TYPE_DIAGNOSIS)

    def test_splits_atomic_mentions_and_keeps_repeated_positions(self) -> None:
        palpitation = "\u0111\u00e1nh tr\u1ed1ng ng\u1ef1c"
        dyspnea = "kh\u00f3 th\u1edf"
        text = f"{palpitation} v\u00e0 {dyspnea}. {palpitation}."
        first_sentence_end = text.index(".")
        second_start = text.rindex(palpitation)
        concepts = [
            Concept(text[:first_sentence_end], TYPE_SYMPTOM, (0, first_sentence_end)),
            Concept(palpitation, TYPE_SYMPTOM, (second_start, second_start + len(palpitation))),
        ]

        refined = refine_concepts(text, concepts)
        palpitation_spans = [item.position for item in refined if item.text == palpitation]

        self.assertEqual(len(palpitation_spans), 2)
        self.assertIn(dyspnea, [item.text for item in refined])


class AssertionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = ContextDetector()

    def test_negation_stops_at_contrast(self) -> None:
        dysphagia = "kh\u00f3 nu\u1ed1t"
        hoarseness = "kh\u00e0n ti\u1ebfng"
        text = f"Ph\u1ee7 nh\u1eadn {dysphagia} nh\u01b0ng c\u00f3 {hoarseness}."

        first = self.detector.assertions_for(
            text, text.index(dysphagia), text.index(dysphagia) + len(dysphagia), TYPE_SYMPTOM
        )
        second = self.detector.assertions_for(
            text, text.index(hoarseness), text.index(hoarseness) + len(hoarseness), TYPE_SYMPTOM
        )

        self.assertIn(ASSERTION_NEGATED, first)
        self.assertNotIn(ASSERTION_NEGATED, second)

    def test_history_is_assigned_per_occurrence(self) -> None:
        syncope = "ng\u1ea5t x\u1ec9u"
        text = (
            f"Ti\u1ec1n s\u1eed b\u1ec7nh:\n- \u0110\u00e3 t\u1eebng {syncope}.\n"
            f"Tri\u1ec7u ch\u1ee9ng hi\u1ec7n t\u1ea1i:\n- {syncope}."
        )
        first_start = text.index(syncope)
        second_start = text.rindex(syncope)

        first = self.detector.assertions_for(text, first_start, first_start + len(syncope), TYPE_SYMPTOM)
        second = self.detector.assertions_for(text, second_start, second_start + len(syncope), TYPE_SYMPTOM)

        self.assertIn(ASSERTION_HISTORICAL, first)
        self.assertNotIn(ASSERTION_HISTORICAL, second)


class CandidateMappingTests(unittest.TestCase):
    def test_rxnorm_prefers_matching_strength_and_oral_form(self) -> None:
        query = "clonazepam 0.5 mg po qam prn"
        matching = medication_match_score(query, "clonazepam 0.5 MG Oral Tablet")
        wrong_strength = medication_match_score(query, "clonazepam 1 MG Oral Tablet")
        ingredient_only = medication_match_score(query, "clonazepam")

        self.assertGreater(matching, wrong_strength)
        self.assertGreater(matching, ingredient_only)

    def test_rxnorm_index_ranks_matching_clinical_drug_first(self) -> None:
        records = {
            (TYPE_DRUG, "197527"): CandidateRecord(
                "197527", "clonazepam 0.5 MG Oral Tablet", "RxNorm", TYPE_DRUG, 10, ttys=("SCD",)
            ),
            (TYPE_DRUG, "200379"): CandidateRecord(
                "200379", "clonazepam 1 MG Oral Tablet", "RxNorm", TYPE_DRUG, 10, ttys=("SCD",)
            ),
            (TYPE_DRUG, "2598"): CandidateRecord(
                "2598", "clonazepam", "RxNorm", TYPE_DRUG, 10, ttys=("IN",)
            ),
        }
        index = SlimCandidateIndex(records, {})

        hits = index.lookup("clonazepam 0.5 mg po qam prn", TYPE_DRUG, 10)

        self.assertTrue(hits)
        self.assertEqual(hits[0].record.code, "197527")

    def test_icd_canonical_alias_wins_and_generic_diagnosis_is_rejected(self) -> None:
        i10 = CandidateRecord("I10", "Essential hypertension", "ICD10", TYPE_DIAGNOSIS, 10)
        i110 = CandidateRecord("I11.0", "Hypertensive heart disease", "ICD10", TYPE_DIAGNOSIS, 10)
        records = {(TYPE_DIAGNOSIS, i10.code): i10, (TYPE_DIAGNOSIS, i110.code): i110}
        aliases = {(TYPE_DIAGNOSIS, "benh tang huyet ap vo can nguyen phat"): ("I10",)}
        index = SlimCandidateIndex(records, aliases)

        hits = index.lookup("t\u0103ng huy\u1ebft \u00e1p", TYPE_DIAGNOSIS, 10)
        selected = _select_diagnosis_codes("t\u0103ng huy\u1ebft \u00e1p", hits, 5)
        generic_hits = [CandidateHit(i110, "diagnosis_lexical", 0.8)]

        self.assertEqual(selected, ("I10",))
        self.assertEqual(_select_diagnosis_codes("u tuy\u1ebfn", generic_hits, 5), ())


class SchemaTests(unittest.TestCase):
    def test_schema_requires_exact_source_offset(self) -> None:
        text = "abc test xyz"
        concept = Concept("test", TYPE_TEST_NAME, (4, 8))

        self.assertEqual(validate_output([concept.to_dict()], source_text=text), [])


if __name__ == "__main__":
    unittest.main()
