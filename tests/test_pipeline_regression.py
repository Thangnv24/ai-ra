"""Fast regression tests for deterministic pipeline decisions."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
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
from core.medication import (
    medication_lookup_keys,
    medication_match_score,
    normalize_prescription_text,
    strip_drug_context,
)
from core.schema import Concept, validate_output
from extraction.context import ContextDetector
from extraction.llm_entities import (
    _chunk_payload,
    _chunk_units,
    _mentions_from_payload,
    _neighbor_context,
    _skip_non_patient_article,
    _span_from_mention_with_reason,
)
from extraction.ner import MedicalNER, SpanCandidate, resolve_span_types
from extraction.sectioning import TextChunk, split_chunks
from knowledge.candidates import (
    CandidateHit,
    CandidateRecord,
    SlimCandidateIndex,
    diagnosis_qualifier_adjustment,
)
from knowledge.candidate_policy import CandidateEmissionPolicy, CandidatePolicyRule
from knowledge.ontology import OntologyIndex
from knowledge.retrieval import CandidateRetriever, _select_diagnosis_codes, _select_drug_code
from services.pipeline import _gate_rule_spans_by_structure, _merge_span_candidates_with_summary
from services.postprocess import refine_concepts
from integrations.openai_client import _is_max_tokens_error, _rate_limit_retry_after
from integrations.prompts import build_entity_extraction_prompt


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

    def test_current_inputs_keep_short_chunk_and_unit_offsets(self) -> None:
        for file_number in range(1, 101):
            text = (ROOT / "input" / f"{file_number}.txt").read_text(encoding="utf-8")
            chunks = split_chunks(text, max_chars=650, overlap=0)
            self.assertTrue(chunks, file_number)
            self.assertLessEqual(max(len(chunk.text) for chunk in chunks), 650, file_number)
            for chunk in chunks:
                self.assertEqual(text[chunk.start : chunk.end], chunk.text, file_number)
                for unit in _chunk_units(chunk):
                    self.assertEqual(text[unit.start : unit.end], unit.text, (file_number, unit.unit_id))

    def test_rule_extractor_keeps_repeated_palpitations(self) -> None:
        phrase = "\u0111\u00e1nh tr\u1ed1ng ng\u1ef1c"
        text = ", ".join([phrase] * 10)
        spans = MedicalNER().extract(text)

        self.assertEqual(sum(span.text.casefold() == phrase for span in spans), 10)

    def test_part_two_chunks_start_new_patient_cases(self) -> None:
        text = (ROOT / "input_part2" / "input" / "input" / "2.txt").read_text(encoding="utf-8")
        chunks = split_chunks(text, max_chars=1000)
        second_case = "L\u00fac 08h30: C\u1eadp nh\u1eadt th\u00f4ng tin"
        case_start = text.index(second_case)

        self.assertFalse(any(chunk.start < case_start < chunk.end for chunk in chunks))
        first_after_boundary = next(chunk for chunk in chunks if chunk.start >= case_start)
        self.assertIn("case_", first_after_boundary.section)

    def test_new_input_structure_markers_are_hard_boundaries(self) -> None:
        examples = (
            (1, "3.  \u0110\u00e1nh gi\u00e1 t\u1ea1i b\u1ec7nh vi\u1ec7n", "hospital_evaluation"),
            (7, "Tr\u1ea3 l\u1eddi :", "answer"),
            (37, "C\u00e1c s\u1ef1 ki\u1ec7n tr\u01b0\u1edbc khi nh\u1eadp vi\u1ec7n", "pre_admission_events"),
            (37, "D\u00f9 hi\u1ec7n t\u1ea1i", "answer"),
            (53, "C\u00e1c s\u1ef1 ki\u1ec7n tr\u01b0\u1edbc khi nh\u1eadp vi\u1ec7n", "pre_admission_events"),
            (82, "2.  Ti\u1ec1n s\u1eed b\u1ec7nh hi\u1ec7n t\u1ea1i", "present_illness"),
        )
        for file_number, marker, expected_role in examples:
            text = (ROOT / "input" / f"{file_number}.txt").read_text(encoding="utf-8")
            boundary = text.index(marker)
            chunks = split_chunks(text, max_chars=650)

            self.assertFalse(
                any(chunk.start < boundary < chunk.end for chunk in chunks),
                (file_number, marker),
            )
            boundary_chunk = next(chunk for chunk in chunks if chunk.start <= boundary < chunk.end)
            self.assertEqual(boundary_chunk.structure_role, expected_role, (file_number, marker))

    def test_neighbor_context_stays_inside_structural_block(self) -> None:
        text = (ROOT / "input" / "7.txt").read_text(encoding="utf-8")
        chunks = split_chunks(text, max_chars=650)
        answer_index = next(index for index, chunk in enumerate(chunks) if chunk.structure_role == "answer")

        before, _ = _neighbor_context(chunks, answer_index)
        self.assertEqual(before, "")
        self.assertTrue(
            any(
                _neighbor_context(chunks, index)[1]
                for index in range(len(chunks) - 1)
                if chunks[index].context_scope == chunks[index + 1].context_scope
            )
        )

    def test_pathological_token_run_is_not_sent_to_llm(self) -> None:
        prefix = "B\u1ec7nh nh\u00e2n t\u0103ng huy\u1ebft \u00e1p "
        noise = "n\u00f4ng " * 30
        suffix = "\nCh\u1ea9n \u0111o\u00e1n: vi\u00eam ph\u1ed5i"
        text = prefix + noise + suffix

        chunks = split_chunks(text, max_chars=1000)

        payload = " ".join(chunk.text for chunk in chunks)
        self.assertIn("t\u0103ng huy\u1ebft \u00e1p", payload)
        self.assertIn("vi\u00eam ph\u1ed5i", payload)
        self.assertNotIn("n\u00f4ng n\u00f4ng n\u00f4ng", payload)

    def test_accepts_compact_mentions_for_a_smaller_llm_response(self) -> None:
        quote = "B\u1ec7nh nh\u00e2n t\u1ec9nh"
        chunk = TextChunk("c1", "case_1:document", 0, len(quote), quote)

        span, reason = _span_from_mention_with_reason(
            quote,
            chunk,
            [quote, TYPE_SYMPTOM, 0],
            units=_chunk_units(chunk),
        )

        self.assertIsNone(reason)
        self.assertEqual((span.text, span.type), (quote, TYPE_SYMPTOM))

    def test_accepts_common_small_model_mentions_wrappers(self) -> None:
        mentions = [["sốt", TYPE_SYMPTOM]]

        self.assertEqual(_mentions_from_payload({"entities": mentions}), mentions)
        self.assertEqual(_mentions_from_payload({"result": {"medical_mentions": mentions}}), mentions)
        self.assertIsNone(_mentions_from_payload({"status": "ok"}))

    def test_prompt_separates_context_from_extraction_target(self) -> None:
        chunk = TextChunk(
            "c2",
            "case_1:document",
            0,
            len("Tim \u0111\u1ec1u, T1 T2 r\u00f5"),
            "Tim \u0111\u1ec1u, T1 T2 r\u00f5",
            structure_role="answer",
            context_scope="case_1:block_2:segment_1",
        )
        payload = _chunk_payload(
            chunk,
            context_before="B\u1ec7nh nh\u00e2n v\u00e0o vi\u1ec7n",
            context_after="Ch\u1ea9n \u0111o\u00e1n",
        )
        prompt = build_entity_extraction_prompt(payload)
        prompt_data = json.loads(prompt)

        self.assertIn("target_text", prompt)
        self.assertEqual(payload["structure_role"], "answer")
        self.assertEqual(prompt_data["target"]["structure_role"], "answer")
        self.assertIn("clinical meaning", prompt)
        self.assertIn("must never be quoted", prompt)
        self.assertIn("Exclude general explanations", prompt)
        self.assertIn("Never label a drug strength", prompt)
        self.assertIn('Return exactly', prompt)
        self.assertEqual(prompt_data["mention_format"], ["exact_quote", "type"])

    def test_rate_limit_retry_parser_uses_provider_delay(self) -> None:
        class RateLimitError(Exception):
            status_code = 429

        error = RateLimitError("Please try again in 3.25s")

        self.assertEqual(_rate_limit_retry_after(error), 3.25)

    def test_provider_json_truncation_is_classified_as_max_tokens(self) -> None:
        error = RuntimeError("max completion tokens reached before generating a valid document")

        self.assertTrue(_is_max_tokens_error(error))


class SpanAndTypeTests(unittest.TestCase):
    def test_low_trust_article_rules_require_llm_confirmation(self) -> None:
        text = (ROOT / "input" / "72.txt").read_text(encoding="utf-8")
        chunks = split_chunks(text, max_chars=650)
        phrase = "lo âu"
        article = next(
            chunk
            for chunk in chunks
            if chunk.structure_role == "medical_article" and phrase in chunk.text.casefold()
        )
        start = text.casefold().index(phrase, article.start, article.end)
        rule = SpanCandidate(
            start,
            start + len(phrase),
            text[start : start + len(phrase)],
            TYPE_SYMPTOM,
            0.8,
        )

        self.assertEqual(_gate_rule_spans_by_structure(text, [rule], []), [])
        self.assertEqual(
            _gate_rule_spans_by_structure(text, [rule], [rule]),
            [replace(rule, structure_role="medical_article")],
        )

    def test_high_confidence_article_rule_does_not_depend_on_small_llm_type(self) -> None:
        text = (ROOT / "input" / "72.txt").read_text(encoding="utf-8")
        chunks = split_chunks(text, max_chars=650)
        phrase = "clonidine"
        article = next(
            chunk
            for chunk in chunks
            if chunk.structure_role == "medical_article" and phrase in chunk.text.casefold()
        )
        start = text.casefold().index(phrase, article.start, article.end)
        rule = SpanCandidate(
            start,
            start + len(phrase),
            text[start : start + len(phrase)],
            TYPE_DRUG,
            0.98,
        )

        self.assertEqual(_gate_rule_spans_by_structure(text, [rule], []), [replace(rule, structure_role="medical_article")])

    def test_skips_only_non_patient_medical_article_chunks(self) -> None:
        general = TextChunk(
            "c1",
            "case_1:document",
            0,
            40,
            "Điều trị phụ thuộc vào mức độ nặng.",
            structure_role="medical_article",
        )
        patient = TextChunk(
            "c2",
            "case_1:document",
            0,
            50,
            "Thuốc đã điều trị trước khi nhập viện: clonidine",
            structure_role="medical_article",
        )

        self.assertTrue(_skip_non_patient_article(general))
        self.assertFalse(_skip_non_patient_article(patient))

    def test_rule_extractor_handles_inline_labs_and_prescription_lines(self) -> None:
        text = (
            "XN: BC 5,38 G/l; N 51,4%\n"
            "HBsAg (+), Anti HBe (-)\n"
            "Creatinin: 89 micromol/l; GOT/GPT < 1\n"
            "Điều trị:\n"
            "  Glucose 5% x 1000ml truyền tĩnh mạch\n"
            "  Philpovin 5g x 2 ống, pha vào dịch\n"
            "  Vitamin 3B x 4 viên, sáng 2 viên"
        )

        spans = MedicalNER().extract(text)
        tagged = {(span.text, span.type) for span in spans}

        for expected in (
            ("BC", TYPE_TEST_NAME),
            ("5,38 G/l", TYPE_TEST_RESULT),
            ("N", TYPE_TEST_NAME),
            ("51,4%", TYPE_TEST_RESULT),
            ("HBsAg", TYPE_TEST_NAME),
            ("(+)", TYPE_TEST_RESULT),
            ("Anti HBe", TYPE_TEST_NAME),
            ("(-)", TYPE_TEST_RESULT),
            ("Creatinin", TYPE_TEST_NAME),
            ("89 micromol/l", TYPE_TEST_RESULT),
            ("GOT/GPT < 1", TYPE_TEST_RESULT),
            ("Glucose 5%", TYPE_DRUG),
            ("Philpovin 5g x 2 ống", TYPE_DRUG),
            ("Vitamin 3B x 4 viên", TYPE_DRUG),
        ):
            self.assertIn(expected, tagged)

    def test_context_diagnosis_rejects_a_narrative_transfer_sentence(self) -> None:
        text = (
            "Chẩn đoán:\n"
            "- Khi được chuyển vào khoa điều trị, bệnh nhân không còn cảm giác khó chịu vùng ngực"
        )

        diagnoses = [span.text for span in MedicalNER().extract(text) if span.type == TYPE_DIAGNOSIS]

        self.assertNotIn(
            "- Khi được chuyển vào khoa điều trị, bệnh nhân không còn cảm giác khó chịu vùng ngực",
            diagnoses,
        )

    def test_rule_span_wins_over_broad_llm_overlap(self) -> None:
        rule = SpanCandidate(5, 20, "rule", TYPE_SYMPTOM, 0.8, source="rule")
        broad_llm = SpanCandidate(0, 30, "llm", TYPE_SYMPTOM, 0.99, source="llm")
        selected, summary = _merge_span_candidates_with_summary([broad_llm, rule])

        self.assertEqual(selected, [rule])
        self.assertEqual(summary.overlap_conflicts, 1)

    def test_preserves_contextual_type_for_sinus_rhythm(self) -> None:
        phrase = "Nh\u1ecbp xoang"
        text = f"\u0110i\u1ec7n t\u00e2m \u0111\u1ed3 ghi nh\u1eadn {phrase}."
        start = text.index(phrase)
        span = SpanCandidate(start, start + len(phrase), phrase, TYPE_DIAGNOSIS, 0.8, source="llm")

        resolved = resolve_span_types(text, [span])

        self.assertEqual(resolved[0].type, TYPE_DIAGNOSIS)

    def test_preserves_proposed_types_without_lexical_override(self) -> None:
        pneumonia = "vi\u00eam ph\u1ed5i"
        syncope = "ng\u1ea5t x\u1ec9u"
        atrial_ectopy = "ngo\u1ea1i t\u00e2m thu nh\u0129"
        cases = (
            (f"Ch\u1ea9n \u0111o\u00e1n: {pneumonia}.", pneumonia, TYPE_TEST_RESULT, TYPE_TEST_RESULT),
            (f"X-quang ng\u1ef1c cho th\u1ea5y {pneumonia}.", pneumonia, TYPE_DIAGNOSIS, TYPE_DIAGNOSIS),
            (
                f"monitor Holter cho th\u1ea5y nh\u1ecbp xoang. Ghi nh\u1eadn {atrial_ectopy}.",
                atrial_ectopy,
                TYPE_DIAGNOSIS,
                TYPE_DIAGNOSIS,
            ),
            (f"G\u1ea7n \u0111\u00e2y \u0111\u01b0\u1ee3c ch\u1ea9n \u0111o\u00e1n {syncope}.", syncope, TYPE_SYMPTOM, TYPE_SYMPTOM),
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

        self.assertEqual([span.type for span in resolved], [TYPE_DIAGNOSIS, TYPE_DIAGNOSIS])

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

        self.assertEqual([span.type for span in resolved], [TYPE_DIAGNOSIS, TYPE_DIAGNOSIS])

        narrative = f"Kh\u00f4ng ghi nh\u1eadn kh\u00f3 th\u1edf; \u0111\u00e1nh gi\u00e1 k\u1ebft qu\u1ea3 ch\u1ecdc h\u00fat c\u1ee7a {nodule}."
        target_start = narrative.index(nodule)
        target = SpanCandidate(target_start, target_start + len(nodule), nodule, TYPE_DIAGNOSIS)
        self.assertEqual(resolve_span_types(narrative, [target])[0].type, TYPE_DIAGNOSIS)

    def test_keeps_compound_mentions_and_repeated_positions(self) -> None:
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
        self.assertEqual([item.text for item in refined], [text[:first_sentence_end], palpitation])
        self.assertEqual(refined[1].position, (second_start, second_start + len(palpitation)))

    def test_keeps_full_numeric_and_qualitative_results(self) -> None:
        numeric = "1703 UI/L"
        imaging = "T\u00fay m\u1eadt kh\u00f4ng d\u00e0y, l\u00f2ng kh\u00f4ng th\u1ea5y s\u1ecfi. T\u1ee5y ph\u00f9 n\u1ec1 nh\u1eb9."
        text = f"Amylase: {numeric}\nCT: {imaging}"
        concepts = [
            Concept(numeric, TYPE_TEST_RESULT, (text.index(numeric), text.index(numeric) + len(numeric))),
            Concept(imaging, TYPE_TEST_RESULT, (text.index(imaging), text.index(imaging) + len(imaging))),
        ]

        refined = refine_concepts(text, concepts)

        self.assertEqual([(item.text, item.position) for item in refined], [(item.text, item.position) for item in concepts])


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

    def test_long_narrative_defaults_to_empty_assertions(self) -> None:
        symptom = "kh\u00f3 th\u1edf"
        text = "B\u1ec7nh nh\u00e2n " + ("c\u00f3 di\u1ec5n bi\u1ebfn k\u00e9o d\u00e0i, " * 18) + f"ph\u1ee7 nh\u1eadn {symptom}."
        start = text.index(symptom)

        assertions = self.detector.assertions_for(text, start, start + len(symptom), TYPE_SYMPTOM)

        self.assertEqual(assertions, ())

    def test_negation_inside_structured_mention_is_detected(self) -> None:
        mention = "Kh\u00f4ng \u0111au \u0111\u1ea7u"
        assertions = self.detector.assertions_for(mention, 0, len(mention), TYPE_SYMPTOM)

        self.assertIn(ASSERTION_NEGATED, assertions)

    def test_family_assertion_requires_family_member_as_subject(self) -> None:
        symptom = "\u0111au ng\u1ef1c"
        reported = f"Theo l\u1eddi ng\u01b0\u1eddi nh\u00e0, b\u1ec7nh nh\u00e2n {symptom}."
        inherited = f"M\u1eb9 b\u1ec7nh nh\u00e2n m\u1eafc {symptom}."

        reported_start = reported.index(symptom)
        inherited_start = inherited.index(symptom)

        self.assertNotIn(
            "isFamily",
            self.detector.assertions_for(reported, reported_start, reported_start + len(symptom), TYPE_SYMPTOM),
        )
        self.assertIn(
            "isFamily",
            self.detector.assertions_for(inherited, inherited_start, inherited_start + len(symptom), TYPE_SYMPTOM),
        )


class CandidateMappingTests(unittest.TestCase):
    def test_curated_exact_alias_is_a_high_recall_retrieval_channel(self) -> None:
        expected = CandidateRecord("R74.0", "Abnormal enzyme levels", "ICD10", TYPE_DIAGNOSIS, 20)
        lexical = CandidateRecord("R74.8", "Other abnormal enzyme levels", "ICD10", TYPE_DIAGNOSIS, 10)
        index = SlimCandidateIndex(
            {
                (TYPE_DIAGNOSIS, expected.code): expected,
                (TYPE_DIAGNOSIS, lexical.code): lexical,
            },
            {(TYPE_DIAGNOSIS, "tang men gan"): (lexical.code,)},
            {(TYPE_DIAGNOSIS, "tang men gan"): (expected.code,)},
        )

        hits = index.lookup("Tăng men gan", TYPE_DIAGNOSIS, 10)

        self.assertEqual(hits[0].source, "curated_exact")
        self.assertEqual(_select_diagnosis_codes("Tăng men gan", hits, 5), ("R74.0",))

    def test_curated_rxnorm_alias_can_retain_gold_ingredient_identifier(self) -> None:
        ingredient = CandidateRecord("5224", "heparin", "RxNorm", TYPE_DRUG, 20, True, ("IN",))
        product = CandidateRecord("1857598", "heparin Injection", "RxNorm", TYPE_DRUG, 10, False, ("SCDF",))
        index = SlimCandidateIndex(
            {(TYPE_DRUG, ingredient.code): ingredient, (TYPE_DRUG, product.code): product},
            {},
            {(TYPE_DRUG, "heparin drip"): (ingredient.code,)},
        )

        hits = index.lookup("heparin drip", TYPE_DRUG, 10)

        self.assertEqual(_select_drug_code("heparin drip", hits), ("5224",))

    def test_drug_context_stripping_preserves_name(self) -> None:
        self.assertEqual(strip_drug_context("Vancomycin trong 20 ngày"), "vancomycin")
        self.assertEqual(strip_drug_context("Romidepsin trong tổng cộng 7 chu kỳ"), "romidepsin")

    def test_candidate_policy_is_profile_specific_and_can_abstain(self) -> None:
        policy = CandidateEmissionPolicy(
            (
                CandidatePolicyRule(
                    TYPE_DRUG,
                    "crestor 10 mg",
                    "short_line",
                    (),
                    support=3,
                    file_support=3,
                ),
            )
        )
        short_text = "Thu\u1ed1c: Crestor 10 mg"
        long_text = (
            "B\u1ec7nh nh\u00e2n \u0111\u01b0\u1ee3c theo d\u00f5i v\u00e0 \u0111i\u1ec1u tr\u1ecb. " * 12
        ) + "Crestor 10 mg"

        self.assertEqual(
            policy.apply(
                "Crestor 10 mg",
                TYPE_DRUG,
                ("576402",),
                source_text=short_text,
                start=short_text.index("Crestor"),
                end=len(short_text),
            ),
            (),
        )
        self.assertEqual(
            policy.apply(
                "Crestor 10 mg",
                TYPE_DRUG,
                ("576402",),
                source_text=long_text,
                start=long_text.index("Crestor"),
                end=len(long_text),
            ),
            ("576402",),
        )

    def test_candidate_policy_can_resolve_same_alias_by_assertion_profile(self) -> None:
        policy = CandidateEmissionPolicy(
            (
                CandidatePolicyRule(
                    TYPE_DIAGNOSIS,
                    "tram cam",
                    "short_line",
                    (),
                    assertions=(ASSERTION_HISTORICAL,),
                    support=3,
                    file_support=3,
                ),
            )
        )
        text = "Tiền sử: trầm cảm"
        start = text.index("trầm cảm")

        self.assertEqual(
            policy.apply(
                "trầm cảm",
                TYPE_DIAGNOSIS,
                ("F32.8",),
                source_text=text,
                start=start,
                end=len(text),
                assertions=(ASSERTION_HISTORICAL,),
            ),
            (),
        )

    def test_candidate_policy_uses_renal_context_for_secondary_hypertension(self) -> None:
        text = "Ch\u1ea9n \u0111o\u00e1n: h\u1eb9p \u0111\u1ed9ng m\u1ea1ch th\u1eadn tr\u00e1i, t\u0103ng huy\u1ebft \u00e1p"
        mention = "t\u0103ng huy\u1ebft \u00e1p"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("I10",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("I15.0",),
        )

    def test_candidate_policy_keeps_primary_code_in_renal_history(self) -> None:
        text = (
            "B\u1ec7nh s\u1eed: h\u1eb9p kh\u00edt \u0111\u1ed9ng m\u1ea1ch th\u1eadn tr\u00e1i, c\u00f3 "
            "t\u0103ng huy\u1ebft \u00e1p c\u00e1ch m\u1ed9t n\u0103m."
        )
        mention = "t\u0103ng huy\u1ebft \u00e1p"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("I10",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("I10",),
        )

    def test_candidate_policy_resolves_diabetes_neurologic_complication(self) -> None:
        text = "Chẩn đoán: ĐTĐ type 2 biến chứng thần kinh ngoại biên"
        mention = "ĐTĐ type 2"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("E11.9",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("E11.4†",),
        )

    def test_candidate_policy_does_not_force_conflicting_atrial_fibrillation_code(self) -> None:
        text = "Chẩn đoán: Rung nhĩ cơn, xem xét can thiệp"
        mention = "Rung nhĩ"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("I48.2",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("I48.2",),
        )

    def test_candidate_policy_resolves_alcohol_related_cirrhosis(self) -> None:
        text = "Tiền sử uống rượu 20 năm. Chẩn đoán: Xơ gan child C"
        mention = "Xơ gan"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("K74.6",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("K70.3",),
        )

    def test_candidate_policy_does_not_retype_historical_cirrhosis(self) -> None:
        text = "Tiền sử uống rượu, xơ gan"
        mention = "xơ gan"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("K74.6",),
                source_text=text,
                start=start,
                end=start + len(mention),
                assertions=(ASSERTION_HISTORICAL,),
            ),
            ("K74.6",),
        )

    def test_candidate_policy_context_does_not_invent_a_suppressed_code(self) -> None:
        text = "Bệnh nhân uống rượu nhiều, có xơ gan trong phần tư vấn."
        mention = "xơ gan"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                (),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            (),
        )

    def test_candidate_policy_resolves_current_gallbladder_stone(self) -> None:
        text = "Chẩn đoán: Sỏi túi mật. Phẫu thuật nội soi cắt túi mật."
        mention = "Sỏi túi mật"
        start = text.index(mention)

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                mention,
                TYPE_DIAGNOSIS,
                ("K80.2",),
                source_text=text,
                start=start,
                end=start + len(mention),
            ),
            ("K80.0",),
        )

    def test_candidate_policy_keeps_unknown_short_drug_mapping_as_fallback(self) -> None:
        text = "Thuốc: Solumedrol"
        start = text.index("Solumedrol")

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                "Solumedrol",
                TYPE_DRUG,
                ("203856",),
                source_text=text,
                start=start,
                end=start + len("Solumedrol"),
            ),
            ("203856",),
        )

    def test_candidate_policy_uses_positive_short_drug_profile(self) -> None:
        text = "Thuốc: Aspirin 81 mg"
        start = text.index("Aspirin")

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                "Aspirin 81 mg",
                TYPE_DRUG,
                ("wrong",),
                source_text=text,
                start=start,
                end=len(text),
                profile_candidates=("315431",),
            ),
            ("315431",),
        )

    def test_candidate_policy_honors_known_empty_profile(self) -> None:
        text = "Thuá»‘c: Solumedrol"
        start = text.index("Solumedrol")

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                "Solumedrol",
                TYPE_DRUG,
                ("203856",),
                source_text=text,
                start=start,
                end=start + len("Solumedrol"),
                profile_candidates=(),
            ),
            (),
        )

    def test_candidate_policy_keeps_unknown_long_drug_mapping(self) -> None:
        text = ("Bệnh nhân được điều trị và theo dõi. " * 15) + "Vancomycin"
        start = text.index("Vancomycin")

        self.assertEqual(
            CandidateEmissionPolicy.empty().apply(
                "Vancomycin",
                TYPE_DRUG,
                ("11124",),
                source_text=text,
                start=start,
                end=start + len("Vancomycin"),
            ),
            ("11124",),
        )

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

    def test_rxnorm_normalizes_vietnamese_spellings_and_parenthetical_names(self) -> None:
        self.assertEqual(normalize_prescription_text("ASA 325mg"), "aspirin 325 mg")
        self.assertEqual(normalize_prescription_text("Nifedipin 20 mg"), "nifedipine 20 mg")
        self.assertEqual(normalize_prescription_text("Forxiga 10 mg"), "farxiga 10 mg")
        self.assertIn("clobazam", medication_lookup_keys("10mg clobazam"))
        self.assertIn("nifedipine", medication_lookup_keys("Nifedipine (Adalat 60mg)"))
        self.assertIn("adalat", medication_lookup_keys("Nifedipine (Adalat 60mg)"))

    def test_rxnorm_indexes_brand_and_prefers_better_supported_duplicate(self) -> None:
        old_lipitor = CandidateRecord(
            "565191", "atorvastatin 20 MG [Lipitor]", "RxNorm", TYPE_DRUG, 2, ttys=("SBDC",)
        )
        supported_lipitor = CandidateRecord(
            "617317", "atorvastatin 20 MG [Lipitor]", "RxNorm", TYPE_DRUG, 2, ttys=("SBDC",)
        )
        records = {
            (TYPE_DRUG, old_lipitor.code): old_lipitor,
            (TYPE_DRUG, supported_lipitor.code): supported_lipitor,
        }
        aliases = {
            (TYPE_DRUG, "atorvastatin 20 mg lipitor"): ("565191", "617317"),
            (TYPE_DRUG, "atorvastatin calcium 20 mg lipitor"): ("617317",),
        }
        index = SlimCandidateIndex(records, aliases)

        hits = index.lookup("Lipitor 20 mg", TYPE_DRUG, 10)

        self.assertEqual(hits[0].record.code, "617317")
        self.assertEqual(_select_drug_code("Lipitor 20 mg", hits), ("617317",))

    def test_rxnorm_prefers_brand_component_when_no_form_requested(self) -> None:
        oral_tablet = CandidateRecord(
            "617318", "atorvastatin 20 MG Oral Tablet [Lipitor]", "RxNorm", TYPE_DRUG, 1, ttys=("SBD",)
        )
        component = CandidateRecord(
            "617317", "atorvastatin 20 MG [Lipitor]", "RxNorm", TYPE_DRUG, 2, ttys=("SBDC",)
        )
        index = SlimCandidateIndex(
            {(TYPE_DRUG, oral_tablet.code): oral_tablet, (TYPE_DRUG, component.code): component},
            {},
        )

        hits = index.lookup("Lipitor 20 mg", TYPE_DRUG, 10)

        self.assertEqual(_select_drug_code("Lipitor 20 mg", hits), ("617317",))

    def test_rxnorm_maps_forxiga_brand_with_matching_strength(self) -> None:
        record = CandidateRecord(
            "1534397", "dapagliflozin 10 MG [Farxiga]", "RxNorm", TYPE_DRUG, 2, ttys=("SBDC",)
        )
        index = SlimCandidateIndex({(TYPE_DRUG, record.code): record}, {})

        hits = index.lookup("Forxiga 10 mg", TYPE_DRUG, 10)

        self.assertEqual(_select_drug_code("Forxiga 10 mg", hits), ("1534397",))

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

    def test_icd_gold_style_heart_failure_override_wins(self) -> None:
        first = CandidateRecord("I50.0", "Suy tim sung huyet", "ICD10", TYPE_DIAGNOSIS, 10)
        second = CandidateRecord("I50.9", "Suy tim khong xac dinh", "ICD10", TYPE_DIAGNOSIS, 10)
        hits = [CandidateHit(first, "exact", 0.99), CandidateHit(second, "exact", 0.98)]

        self.assertEqual(_select_diagnosis_codes("suy tim", hits, 5), ("I50.9",))

    def test_icd_stage_qualifier_selects_specific_code(self) -> None:
        stage_five = CandidateRecord(
            "N18.5", "Bệnh thận mạn tính, giai đoạn 5", "ICD10", TYPE_DIAGNOSIS, 10
        )
        unspecified = CandidateRecord(
            "N18.9", "Bệnh thận mạn tính, không xác định", "ICD10", TYPE_DIAGNOSIS, 10
        )
        query = "suy thận mạn giai đoạn V"
        aliases = {(TYPE_DIAGNOSIS, "suy than man giai doan v"): ("N18.9", "N18.5")}
        index = SlimCandidateIndex(
            {(TYPE_DIAGNOSIS, stage_five.code): stage_five, (TYPE_DIAGNOSIS, unspecified.code): unspecified},
            aliases,
        )

        hits = index.lookup(query, TYPE_DIAGNOSIS, 10)

        self.assertEqual(hits[0].record.code, "N18.5")
        self.assertEqual(_select_diagnosis_codes(query, hits, 5), ("N18.5",))

    def test_icd_qualifier_penalizes_acute_chronic_and_laterality_conflicts(self) -> None:
        self.assertGreater(
            diagnosis_qualifier_adjustment("viêm tụy cấp", "Viêm tụy cấp tính, không xác định"),
            diagnosis_qualifier_adjustment("viêm tụy cấp", "Viêm tụy mạn tính thể khác"),
        )
        self.assertGreater(
            diagnosis_qualifier_adjustment("tổn thương thận trái", "Tổn thương thận trái"),
            diagnosis_qualifier_adjustment("tổn thương thận trái", "Tổn thương thận phải"),
        )

    def test_candidate_gate_abstains_for_generic_diagnosis_in_long_narrative_line(self) -> None:
        record = CandidateRecord("I10", "Essential hypertension", "ICD10", TYPE_DIAGNOSIS, 10)
        index = SlimCandidateIndex(
            {(TYPE_DIAGNOSIS, "I10"): record},
            {(TYPE_DIAGNOSIS, "benh tang huyet ap vo can nguyen phat"): ("I10",)},
        )
        retriever = CandidateRetriever(OntologyIndex(()), index)
        diagnosis = "t\u0103ng huy\u1ebft \u00e1p"
        narrative = "B\u1ec7nh nh\u00e2n " + ("c\u00f3 di\u1ec5n bi\u1ebfn k\u00e9o d\u00e0i, " * 18) + diagnosis
        start = narrative.index(diagnosis)

        self.assertEqual(
            retriever.candidates_for(
                diagnosis,
                TYPE_DIAGNOSIS,
                source_text=narrative,
                start=start,
                end=start + len(diagnosis),
            ),
            (),
        )

    def test_candidate_gate_allows_high_value_long_line_override(self) -> None:
        record = CandidateRecord("I25.1", "Bệnh tim mạch do xơ vữa động mạch", "ICD10", TYPE_DIAGNOSIS, 10)
        index = SlimCandidateIndex(
            {(TYPE_DIAGNOSIS, "I25.1"): record},
            {(TYPE_DIAGNOSIS, "benh tim mach do xo vua dong mach"): ("I25.1",)},
        )
        retriever = CandidateRetriever(OntologyIndex(()), index)
        diagnosis = "bệnh tim mạch do xơ vữa động mạch"
        narrative = "Bệnh nhân " + ("có diễn biến kéo dài, " * 18) + diagnosis
        start = narrative.index(diagnosis)

        self.assertEqual(
            retriever.candidates_for(
                diagnosis,
                TYPE_DIAGNOSIS,
                source_text=narrative,
                start=start,
                end=start + len(diagnosis),
            ),
            ("I25.1",),
        )

    def test_candidate_gate_uses_training_profile_evidence_on_long_lines(self) -> None:
        record = CandidateRecord("N62", "Breast hypertrophy", "ICD10", TYPE_DIAGNOSIS, 20)
        index = SlimCandidateIndex(
            {(TYPE_DIAGNOSIS, record.code): record},
            {},
            {(TYPE_DIAGNOSIS, "phi dai vu"): (record.code,)},
            {(TYPE_DIAGNOSIS, "phi dai vu", "long_line"): (record.code,)},
        )
        retriever = CandidateRetriever(OntologyIndex(()), index)
        mention = "phì đại vú"
        narrative = ("Bệnh nhân có nhiều thông tin lâm sàng. " * 15) + mention
        start = narrative.index(mention)

        self.assertEqual(
            retriever.candidates_for(
                mention,
                TYPE_DIAGNOSIS,
                source_text=narrative,
                start=start,
                end=start + len(mention),
            ),
            ("N62",),
        )

    def test_drug_ingredient_only_alias_is_not_emitted(self) -> None:
        record = CandidateRecord("69749", "valsartan", "RxNorm", TYPE_DRUG, 10, ttys=("IN",))
        hit = CandidateHit(record, "drug_ingredient", 0.9)

        self.assertEqual(_select_drug_code("Valsartan 50 mg", [hit]), ())


class SchemaTests(unittest.TestCase):
    def test_schema_requires_exact_source_offset(self) -> None:
        text = "abc test xyz"
        concept = Concept("test", TYPE_TEST_NAME, (4, 8))

        self.assertEqual(validate_output([concept.to_dict()], source_text=text), [])


if __name__ == "__main__":
    unittest.main()
