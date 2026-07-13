"""Deterministic cleanup after rule/LLM mention decisions."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from core.config import (
    ALLOWED_ASSERTIONS,
    ASSERTION_TYPES,
    CODED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)
from core.schema import Concept
from core.text import normalize_key, trim_span_text

if TYPE_CHECKING:
    from extraction.context import ContextDetector
    from knowledge.retrieval import CandidateRetriever


_NUMERIC_RESULT_RE = re.compile(
    r"[<>]?\s*\d+(?:[,.]\d+)?(?:\s*(?:mg/dL|g/L|g/dL|mmol/L|mmol|IU/L|U/L|"
    r"mEq/L|ng/mL|pg/mL|x10\^?\d*/L|%))?",
    re.IGNORECASE,
)

_DROP_DIAGNOSIS_KEYS = {
    "am tinh",
    "duong tinh",
    "binh thuong",
    "nhip xoang",
    "khong co gi dang chu y",
}


def refine_concepts(
    text: str,
    concepts: list[Concept],
    retriever: "CandidateRetriever | None" = None,
    context_detector: "ContextDetector | None" = None,
) -> list[Concept]:
    """Apply conservative deterministic fixes before schema validation."""

    refined: list[Concept] = []
    for concept in concepts:
        updated = _refine_single_span(text, concept)
        if updated is None:
            continue
        updated = _repair_assertions(text, updated, context_detector)
        updated = _repair_candidates(updated, retriever, text_changed=updated.text != concept.text)
        if _should_drop_concept(updated):
            continue
        refined.append(updated)

    refined = _drop_diagnoses_inside_test_names(refined)
    refined = _dedupe(refined)
    refined = _dedupe_repeated_concepts(refined, text)
    return sorted(refined, key=lambda c: (c.position[0], c.position[1], c.type))


def _refine_single_span(text: str, concept: Concept) -> Concept | None:
    if concept.type == TYPE_SYMPTOM:
        return _refine_symptom_span(text, concept)
    if concept.type == TYPE_TEST_RESULT:
        return _refine_lab_result_span(text, concept)
    if concept.type == TYPE_DIAGNOSIS:
        return _refine_diagnosis_span(text, concept)
    return concept


def _refine_symptom_span(text: str, concept: Concept) -> Concept | None:
    start, end = concept.position
    key = normalize_key(concept.text)
    if key == "ho" and normalize_key(text[start : min(len(text), start + 32)]).startswith("ho dai thao duong"):
        return None

    expansions = (
        ("non", ("nôn ra máu",)),
        ("kho tho", ("khó thở khi gắng sức", "khó thở khi nằm đột ngột", "khó thở khi nằm")),
        ("dau bung", ("đau bụng gián đoạn",)),
    )
    for source_key, phrases in expansions:
        if key == source_key:
            expanded = _expand_to_phrase_at(text, start, phrases)
            if expanded is not None:
                return _replace_span(concept, text, *expanded)

    targeted: list[tuple[str, bool]] = [
        ("phu ngoai vi", key.startswith("phu ngoai vi ") and _has_any(key, ("tang dan", "gan day", "trong vai"))),
        ("tang can", key.startswith("tang tang can") or " pound" in key),
        ("cam giac that chat nguc", key.startswith("cac dot ") or key.startswith("cac dot")),
        ("di lai kho khan", "can dung gay" in key),
        ("ton thuong chi duoi nghiem trong", key.startswith("ton thuong ton thuong")),
    ]
    for target_key, condition in targeted:
        if condition:
            narrowed = _narrow_to_normalized_subspan(text, start, end, target_key)
            if narrowed is not None:
                return _replace_span(concept, text, *narrowed)

    comma_rules = (", can dung gay", ", can ho tro", ", tang dan")
    for marker in comma_rules:
        idx = key.find(marker)
        if idx > 0:
            narrowed = _narrow_to_prefix_before_normalized(text, start, end, marker)
            if narrowed is not None:
                return _replace_span(concept, text, *narrowed)

    return concept


def _refine_lab_result_span(text: str, concept: Concept) -> Concept | None:
    key = normalize_key(concept.text)
    if "dang cho ket qua" in key or "cho ket qua" in key:
        return None

    for target_key in ("am tinh", "duong tinh", "binh thuong", "tang", "giam"):
        if key.startswith(target_key + " ") or key == target_key:
            narrowed = _narrow_to_normalized_subspan(text, concept.position[0], concept.position[1], target_key)
            if narrowed is not None:
                return _replace_span(concept, text, *narrowed)

    match = _NUMERIC_RESULT_RE.search(concept.text)
    if not match:
        return concept
    before = concept.text[: match.start()]
    after = concept.text[match.end() :]
    has_extra_words = bool(re.search(r"[A-Za-z]", before + after, re.IGNORECASE))
    if not has_extra_words and match.start() == 0:
        return concept
    start = concept.position[0] + match.start()
    end = concept.position[0] + match.end()
    start, end, span_text = trim_span_text(text, start, end)
    if not span_text:
        return None
    return replace(concept, text=span_text, position=(start, end), assertions=(), candidates=())


def _refine_diagnosis_span(text: str, concept: Concept) -> Concept | None:
    key = normalize_key(concept.text)
    if key in _DROP_DIAGNOSIS_KEYS:
        return None
    if key.startswith("viem da day ") and "ruot do virus" in key:
        expanded = _expand_to_phrase_at(text, concept.position[0], ("viêm dạ dày ruột do virus", "viêm dạ dày - ruột do virus"))
        if expanded is not None:
            return _replace_span(concept, text, *expanded)
    for marker in (" phan ung", " dap ung", " dieu tri", " duoc dieu tri"):
        idx = key.find(marker)
        if idx > 0:
            narrowed = _narrow_to_prefix_before_normalized(text, concept.position[0], concept.position[1], marker)
            if narrowed is not None:
                return _replace_span(concept, text, *narrowed)
    bad_prefixes = (
        "ket qua",
        "hinh anh",
        "chup x quang",
        "sieu am",
        "ct ",
        "mri ",
        "dien tam do",
        "ecg",
    )
    if any(key.startswith(prefix) for prefix in bad_prefixes):
        return None
    return concept


def _repair_assertions(
    text: str,
    concept: Concept,
    context_detector: "ContextDetector | None",
) -> Concept:
    if concept.type not in ASSERTION_TYPES:
        return replace(concept, assertions=())
    assertions = [item for item in concept.assertions if item in ALLOWED_ASSERTIONS]
    if context_detector is not None and concept.type == TYPE_DRUG:
        start, end = concept.position
        assertions.extend(context_detector.assertions_for(text, start, end, concept.type))
    return replace(concept, assertions=_ordered_assertions(assertions))


def _repair_candidates(
    concept: Concept,
    retriever: "CandidateRetriever | None",
    text_changed: bool,
) -> Concept:
    if concept.type not in CODED_TYPES:
        return replace(concept, candidates=())
    if retriever is None:
        return concept
    if text_changed or not concept.candidates:
        candidates = retriever.candidates_for(concept.text, concept.type)
        return replace(concept, candidates=candidates)
    return concept


def _should_drop_concept(concept: Concept) -> bool:
    key = normalize_key(concept.text)
    if not key:
        return True
    if concept.type == TYPE_TEST_RESULT and key in {"dang cho ket qua", "cho ket qua"}:
        return True
    if concept.type == TYPE_DRUG and len(key) <= 1:
        return True
    return False


def _drop_diagnoses_inside_test_names(concepts: list[Concept]) -> list[Concept]:
    test_ranges = [concept.position for concept in concepts if concept.type == TYPE_TEST_NAME]
    output: list[Concept] = []
    for concept in concepts:
        if concept.type == TYPE_DIAGNOSIS:
            start, end = concept.position
            if any(test_start <= start and end <= test_end and (end - start) < (test_end - test_start) for test_start, test_end in test_ranges):
                continue
        output.append(concept)
    return output


def _dedupe(concepts: list[Concept]) -> list[Concept]:
    best: dict[tuple[int, int, str], Concept] = {}
    for concept in concepts:
        key = (concept.position[0], concept.position[1], concept.type)
        current = best.get(key)
        if current is None:
            best[key] = concept
            continue
        if len(concept.candidates) > len(current.candidates):
            best[key] = concept
    return list(best.values())


def _dedupe_repeated_concepts(concepts: list[Concept], text: str) -> list[Concept]:
    best: dict[tuple[str, str], Concept] = {}
    for concept in concepts:
        key = (concept.type, normalize_key(concept.text))
        current = best.get(key)
        if current is None or _concept_rank(text, concept) > _concept_rank(text, current):
            best[key] = concept
    return list(best.values())


def _concept_rank(text: str, concept: Concept) -> tuple[int, int, int, int, int]:
    assertions = set(concept.assertions)
    start, end = concept.position
    context = normalize_key(text[max(0, start - 120) : min(len(text), end + 120)])
    current_context = int(any(marker in context for marker in ("trieu chung hien tai", "ly do nhap vien", "chan doan hinh anh", "ket qua")))
    return (
        int("isNegated" not in assertions),
        int("isFamily" not in assertions),
        int("isHistorical" not in assertions),
        current_context,
        len(concept.text),
    )


def _replace_span(concept: Concept, text: str, start: int, end: int) -> Concept:
    start, end, span_text = trim_span_text(text, start, end)
    return replace(concept, text=span_text, position=(start, end))


def _expand_to_phrase_at(text: str, start: int, phrases: tuple[str, ...]) -> tuple[int, int] | None:
    segment = text[start : min(len(text), start + 80)]
    for phrase in sorted(phrases, key=len, reverse=True):
        if normalize_key(segment[: len(phrase) + 8]).startswith(normalize_key(phrase)):
            return start, start + len(phrase)
    return None


def _narrow_to_normalized_subspan(text: str, start: int, end: int, target_key: str) -> tuple[int, int] | None:
    segment = text[start:end]
    target_key = normalize_key(target_key)
    if not target_key:
        return None
    for local_start in range(0, len(segment)):
        if segment[local_start].isspace():
            continue
        for local_end in range(min(len(segment), local_start + len(target_key) + 24), local_start, -1):
            candidate = segment[local_start:local_end]
            if normalize_key(candidate) == target_key:
                return start + local_start, start + local_end
    return None


def _narrow_to_prefix_before_normalized(text: str, start: int, end: int, marker_key: str) -> tuple[int, int] | None:
    segment = text[start:end]
    for local_end in range(1, len(segment)):
        suffix_key = normalize_key(segment[local_end:])
        if suffix_key.startswith(normalize_key(marker_key).lstrip()):
            return start, start + local_end
    return None


def _ordered_assertions(assertions: list[str]) -> tuple[str, ...]:
    seen = set()
    output: list[str] = []
    for allowed in ALLOWED_ASSERTIONS:
        if allowed in assertions and allowed not in seen:
            output.append(allowed)
            seen.add(allowed)
    return tuple(output)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
