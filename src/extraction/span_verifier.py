"""Calibrate, adjudicate, and decode competing medical span proposals."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Iterable

from core.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.text import normalize_key, token_count
from extraction.annotation_memory import AnnotationMemory, section_at
from extraction.learned_models import SpanAcceptanceModel
from extraction.ner import SpanCandidate
from extraction.sectioning import detect_sections, detect_subsections


_SOURCE_PRIORS = {
    "lab_rule": 0.96,
    "drug_rule": 0.92,
    "context_rule": 0.84,
    "result_rule": 0.82,
    "memory": 0.86,
    "grammar": 0.88,
    "lexical_rule": 0.76,
    "rule": 0.78,
    "llm": 0.72,
    "sequence_model": 0.74,
}
_SOURCE_PRIORITY = {
    "lab_rule": 8,
    "drug_rule": 7,
    "memory": 6,
    "grammar": 6,
    "context_rule": 5,
    "result_rule": 4,
    "lexical_rule": 3,
    "rule": 2,
    "llm": 1,
    "sequence_model": 2,
}
_TYPE_PRIORITY = {
    TYPE_DRUG: 5,
    TYPE_DIAGNOSIS: 4,
    TYPE_TEST_RESULT: 3,
    TYPE_TEST_NAME: 2,
    TYPE_SYMPTOM: 1,
}
_SOURCE_THRESHOLDS = {
    "lab_rule": 0.76,
    "drug_rule": 0.76,
    "memory": 0.7,
    "grammar": 0.72,
    "context_rule": 0.73,
    "result_rule": 0.74,
    "lexical_rule": 0.72,
    "rule": 0.72,
    "llm": 0.68,
    "sequence_model": 0.7,
}
_GENERIC_RESULT_FRAGMENTS = {
    "anh",
    "chi so",
    "hinh anh",
    "ket qua",
    "ket qua xet nghiem",
}
_GENERIC_DIAGNOSIS_FRAGMENTS = {"benh", "chan doan"}


@dataclass(frozen=True, slots=True)
class SpanVerificationSummary:
    inputs: int
    valid: int
    invalid: int
    exact_duplicates: int
    type_conflicts: int
    below_threshold: int
    corroborated: int
    llm_only: int
    sequence_only_rejected: int
    generic_llm_rejected: int
    overlap_conflicts: int
    selected: int

    def to_dict(self) -> dict[str, int]:
        return {
            "inputs": self.inputs,
            "valid": self.valid,
            "invalid": self.invalid,
            "exact_duplicates": self.exact_duplicates,
            "type_conflicts": self.type_conflicts,
            "below_threshold": self.below_threshold,
            "corroborated": self.corroborated,
            "llm_only": self.llm_only,
            "sequence_only_rejected": self.sequence_only_rejected,
            "generic_llm_rejected": self.generic_llm_rejected,
            "overlap_conflicts": self.overlap_conflicts,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedSpan:
    span: SpanCandidate
    sources: frozenset[str]
    utility: float


class SpanTypeVerifier:
    def __init__(
        self,
        memory: AnnotationMemory | None = None,
        acceptance_model: SpanAcceptanceModel | None = None,
    ) -> None:
        self.memory = memory or AnnotationMemory.empty()
        self.acceptance_model = acceptance_model or SpanAcceptanceModel.empty()

    def select(self, text: str, proposals: Iterable[SpanCandidate]) -> tuple[list[SpanCandidate], SpanVerificationSummary]:
        raw = list(proposals)
        sections = detect_sections(text)
        subsections = detect_subsections(text)
        valid: list[SpanCandidate] = []
        for span in raw:
            if 0 <= span.start < span.end <= len(text) and text[span.start : span.end] == span.text:
                valid.append(span)

        grouped: dict[tuple[int, int, str], list[SpanCandidate]] = {}
        for span in valid:
            grouped.setdefault((span.start, span.end, span.type), []).append(span)
        exact_duplicates = sum(max(0, len(group) - 1) for group in grouped.values())

        verified: list[_VerifiedSpan] = []
        below_threshold = 0
        corroborated = 0
        llm_only = 0
        sequence_only_rejected = 0
        generic_llm_rejected = 0
        for group in grouped.values():
            scored = [
                self._calibrate(
                    text,
                    span,
                    section_at(sections, span.start),
                    section_at(subsections, span.start),
                )
                for span in group
            ]
            representative, score = max(
                scored,
                key=lambda item: (item[1], _SOURCE_PRIORITY.get(item[0].source, 0), item[0].score),
            )
            sources = frozenset(span.source for span in group)
            evidence_sources = _evidence_sources(group)
            if not evidence_sources:
                sequence_only_rejected += 1
                continue
            if evidence_sources == {"llm"}:
                llm_only += 1
                if _is_generic_llm_fragment(text, representative):
                    generic_llm_rejected += 1
                    continue
            elif len(evidence_sources) >= 2:
                corroborated += 1
            score = min(
                0.995,
                score
                + 0.045 * max(0, len(evidence_sources) - 1)
                + (0.01 if "sequence_model" in sources else 0.0),
            )
            threshold = min(_SOURCE_THRESHOLDS.get(source, 0.72) for source in sources)
            if score < threshold:
                below_threshold += 1
                continue
            representative = replace(representative, score=score, source=_best_source(sources))
            verified.append(
                _VerifiedSpan(
                    representative,
                    sources,
                    _span_utility(representative, len(evidence_sources)),
                )
            )

        by_boundary: dict[tuple[int, int], list[_VerifiedSpan]] = {}
        for item in verified:
            by_boundary.setdefault((item.span.start, item.span.end), []).append(item)
        type_conflicts = sum(max(0, len(group) - 1) for group in by_boundary.values())
        typed = [
            max(
                group,
                key=lambda item: (
                    item.utility,
                    _SOURCE_PRIORITY.get(item.span.source, 0),
                    _TYPE_PRIORITY.get(item.span.type, 0),
                ),
            )
            for group in by_boundary.values()
        ]

        selected = _decode_non_overlapping(typed)
        overlap_conflicts = len(typed) - len(selected)
        spans = sorted((item.span for item in selected), key=lambda span: (span.start, span.end, span.type))
        return spans, SpanVerificationSummary(
            inputs=len(raw),
            valid=len(valid),
            invalid=len(raw) - len(valid),
            exact_duplicates=exact_duplicates,
            type_conflicts=type_conflicts,
            below_threshold=below_threshold,
            corroborated=corroborated,
            llm_only=llm_only,
            sequence_only_rejected=sequence_only_rejected,
            generic_llm_rejected=generic_llm_rejected,
            overlap_conflicts=overlap_conflicts,
            selected=len(spans),
        )

    def _calibrate(
        self,
        text: str,
        span: SpanCandidate,
        section: str,
        subsection: str,
    ) -> tuple[SpanCandidate, float]:
        prior = _SOURCE_PRIORS.get(span.source, 0.7)
        score = 0.6 * max(0.0, min(1.0, span.score)) + 0.4 * prior
        evidence = self.memory.evidence_for_context(span, section, subsection, text)
        memory_score: float | None = None
        if evidence is not None:
            entry, memory_score = evidence
            evidence_weight = 0.42 if entry.observed_count >= 3 else 0.2
            score = (1.0 - evidence_weight) * score + evidence_weight * memory_score
        learned_score = self.acceptance_model.score(
            text,
            span,
            section,
            subsection,
            memory_score,
        )
        if learned_score is not None:
            source = span.parent_source or span.source
            if span.type == TYPE_TEST_RESULT:
                learned_weight = 0.08
            elif span.type == TYPE_DRUG:
                learned_weight = 0.16
            elif span.variant != "original":
                learned_weight = 0.42
            elif source == "llm":
                learned_weight = 0.42
            elif source == "sequence_model":
                learned_weight = 0.38
            else:
                learned_weight = 0.2
            score = (1.0 - learned_weight) * score + learned_weight * learned_score
            learned_threshold = self.acceptance_model.threshold_for(span.type, source)
            if learned_score >= learned_threshold:
                score += 0.025
            elif (
                source in {"llm", "sequence_model"}
                and span.type not in {TYPE_TEST_RESULT, TYPE_DRUG}
                and learned_score < max(0.12, learned_threshold - 0.12)
            ):
                score -= 0.1
        return span, score


def _best_source(sources: frozenset[str]) -> str:
    return max(sources, key=lambda source: (_SOURCE_PRIORITY.get(source, 0), source))


def _evidence_sources(group: list[SpanCandidate]) -> set[str]:
    evidence: set[str] = set()
    for span in group:
        if span.source == "sequence_model":
            continue
        if span.source == "grammar":
            evidence.add(span.parent_source or "grammar")
            continue
        evidence.add(span.source)
    return evidence


def _is_generic_llm_fragment(text: str, span: SpanCandidate) -> bool:
    key = normalize_key(span.text)
    if len(key) < 2 or not re.search(r"[a-z0-9]", key):
        return True
    if span.type == TYPE_TEST_RESULT and key in _GENERIC_RESULT_FRAGMENTS:
        return True
    if span.type == TYPE_DIAGNOSIS and key in _GENERIC_DIAGNOSIS_FRAGMENTS:
        return True
    line_start = text.rfind("\n", 0, span.start) + 1
    line_end = text.find("\n", span.end)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    return line.endswith(":") and normalize_key(line) == key


def _span_utility(span: SpanCandidate, source_count: int) -> float:
    tokens = token_count(span.text)
    length_bonus = 0.0
    if span.type == TYPE_TEST_NAME:
        length_bonus = 0.008 * min(8, tokens)
    elif span.type in {TYPE_DIAGNOSIS, TYPE_SYMPTOM}:
        length_bonus = 0.003 * min(6, tokens)
    variant_adjustment = {
        "drug_without_sig": 0.025,
        "drug_core": -0.005,
        "test_full_label": 0.035,
        "symptom_with_modifier": 0.02,
        "symptom_with_negation": -0.005,
        "diagnosis_core": 0.015,
        "result_with_unit": -0.01,
        "vital_label_with_value": 0.005,
        "grammar_drug_sig": 0.025,
        "grammar_result_unit": 0.025,
        "grammar_symptom_modifier": 0.02,
        "grammar_symptom_negation": 0.01,
        "grammar_diagnosis_core": 0.015,
    }.get(span.variant, 0.0)
    corroboration_bonus = 0.025 * max(0, source_count - 1)
    short_penalty = 0.035 if len(span.text.strip()) <= 2 else 0.0
    return max(0.0, span.score - 0.55) + length_bonus + variant_adjustment + corroboration_bonus - short_penalty


def _decode_non_overlapping(candidates: list[_VerifiedSpan]) -> list[_VerifiedSpan]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.span.end, item.span.start, -item.utility))
    ends = [item.span.end for item in ordered]
    predecessors = [bisect_right(ends, item.span.start, 0, index) - 1 for index, item in enumerate(ordered)]
    values = [0.0]
    choices: list[tuple[int, ...]] = [()]
    for index, item in enumerate(ordered, start=1):
        predecessor = predecessors[index - 1] + 1
        include_value = values[predecessor] + item.utility
        include_choice = choices[predecessor] + (index - 1,)
        exclude_value = values[index - 1]
        exclude_choice = choices[index - 1]
        if include_value > exclude_value + 1e-9:
            values.append(include_value)
            choices.append(include_choice)
        elif exclude_value > include_value + 1e-9:
            values.append(exclude_value)
            choices.append(exclude_choice)
        else:
            preferred = max(
                (include_choice, exclude_choice),
                key=lambda choice: (len(choice), sum(ordered[position].span.score for position in choice)),
            )
            values.append(include_value)
            choices.append(preferred)
    return [ordered[index] for index in choices[-1]]
