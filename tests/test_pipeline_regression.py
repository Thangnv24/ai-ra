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
from core.medication import medication_lookup_keys, medication_match_score, normalize_prescription_text
from core.schema import Concept, validate_output
from extraction.context import ContextDetector
from extraction.llm_entities import _chunk_units, _span_from_mention_with_reason
from extraction.ner import MedicalNER, SpanCandidate, resolve_span_types
from extraction.sectioning import TextChunk, split_chunks
from knowledge.candidates import (
    CandidateHit,
    CandidateRecord,
    SlimCandidateIndex,
    diagnosis_qualifier_adjustment,
)
from knowledge.ontology import OntologyIndex
from knowledge.retrieval import CandidateRetriever, _select_diagnosis_codes, _select_drug_code
from services.pipeline import _merge_span_candidates_with_summary
from services.postprocess import refine_concepts
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

    def test_part_two_chunks_start_new_patient_cases(self) -> None:
        text = (ROOT / "input_part2" / "input" / "input" / "2.txt").read_text(encoding="utf-8")
        chunks = split_chunks(text, max_chars=1000)
        second_case = "L\u00fac 08h30: C\u1eadp nh\u1eadt th\u00f4ng tin"
        case_start = text.index(second_case)

        self.assertFalse(any(chunk.start < case_start < chunk.end for chunk in chunks))
        first_after_boundary = next(chunk for chunk in chunks if chunk.start > case_start)
        self.assertIn("case_", first_after_boundary.section)

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

    def test_prompt_separates_context_from_extraction_target(self) -> None:
        prompt = build_entity_extraction_prompt(
            {
                "chunk_id": "c2",
                "section": "case_1:document",
                "text": "Tim \u0111\u1ec1u, T1 T2 r\u00f5",
                "context_before": "B\u1ec7nh nh\u00e2n v\u00e0o vi\u1ec7n",
                "context_after": "Ch\u1ea9n \u0111o\u00e1n",
            }
        )

        self.assertIn("target_text", prompt)
        self.assertIn("clinical finding", prompt)
        self.assertIn("must never be quoted", prompt)
        self.assertIn("Be exhaustive", prompt)
        self.assertIn("trầm cảm", prompt)
        self.assertIn("CTM", prompt)


class SpanAndTypeTests(unittest.TestCase):
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
