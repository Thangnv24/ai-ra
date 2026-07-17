"""Conservative deterministic cleanup after mention extraction."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from core.config import (
    ALLOWED_ASSERTIONS,
    ASSERTION_TYPES,
    CODED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)
from core.schema import Concept
from core.text import normalize_key

if TYPE_CHECKING:
    from extraction.context import ContextDetector
    from knowledge.retrieval import CandidateRetriever


def refine_concepts(
    text: str,
    concepts: list[Concept],
    retriever: "CandidateRetriever | None" = None,
    context_detector: "ContextDetector | None" = None,
) -> list[Concept]:
    """Repair metadata without changing extraction span boundaries or types."""

    refined: list[Concept] = []
    for concept in concepts:
        updated = _repair_assertions(text, concept, context_detector)
        updated = _repair_candidates(updated, retriever, source_text=text)
        if not _should_drop_concept(updated):
            refined.append(updated)

    refined = _drop_diagnoses_inside_test_names(refined)
    refined = _dedupe(refined)
    return sorted(refined, key=lambda concept: (concept.position[0], concept.position[1], concept.type))


def _repair_assertions(
    text: str,
    concept: Concept,
    context_detector: "ContextDetector | None",
) -> Concept:
    if concept.type not in ASSERTION_TYPES:
        return replace(concept, assertions=())
    if context_detector is not None:
        start, end = concept.position
        assertions = list(context_detector.assertions_for(text, start, end, concept.type))
    else:
        assertions = [item for item in concept.assertions if item in ALLOWED_ASSERTIONS]
    return replace(concept, assertions=_ordered_assertions(assertions))


def _repair_candidates(
    concept: Concept,
    retriever: "CandidateRetriever | None",
    source_text: str,
) -> Concept:
    if concept.type not in CODED_TYPES:
        return replace(concept, candidates=())
    if retriever is None or concept.candidates:
        return concept
    candidates = retriever.candidates_for(
        concept.text,
        concept.type,
        source_text=source_text,
        start=concept.position[0],
        end=concept.position[1],
    )
    return replace(concept, candidates=candidates)


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
            inside_larger_test = any(
                test_start <= start
                and end <= test_end
                and (end - start) < (test_end - test_start)
                for test_start, test_end in test_ranges
            )
            if inside_larger_test:
                continue
        output.append(concept)
    return output


def _dedupe(concepts: list[Concept]) -> list[Concept]:
    best: dict[tuple[int, int, str], Concept] = {}
    for concept in concepts:
        key = (concept.position[0], concept.position[1], concept.type)
        current = best.get(key)
        if current is None or len(concept.candidates) > len(current.candidates):
            best[key] = concept
    return list(best.values())


def _ordered_assertions(assertions: list[str]) -> tuple[str, ...]:
    return tuple(allowed for allowed in ALLOWED_ASSERTIONS if allowed in assertions)
