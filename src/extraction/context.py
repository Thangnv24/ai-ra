"""Section- and clause-aware assertion detection."""

from __future__ import annotations

import re
from functools import lru_cache

from core.config import (
    ALLOWED_ASSERTIONS,
    ASSERTION_FAMILY,
    ASSERTION_HISTORICAL,
    ASSERTION_NEGATED,
    ASSERTION_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
)
from core.text import normalize_key
from extraction.sectioning import Section, case_bounds_at, detect_sections
from extraction.assertion_model import AssertionClassifier


NEGATION_RE = re.compile(
    r"(?:\bkhong\b|\bchua\b|\bphu nhan\b|\bkhong thay\b|\bno\b|\bdenies\b|"
    r"\bwithout\b|\bnegative for\b)"
)
NEGATION_EXCLUSIONS = (
    "khong dien hinh",
    "khong dac hieu",
    "khong the",
    "khong ro",
    "khong co cai thien",
    "khong lien quan",
    "khong kin",
    "khong mo",
    "khong on dinh",
    "khong dac hieu",
    "khong xac dinh",
    "khong bien chung",
    "khong ro nguyen nhan",
)
CONTRAST_RE = re.compile(r"\b(?:nhung|tuy nhien|song|however|but)\b")
FAMILY_SUBJECT_RE = re.compile(
    r"(?:\btien su gia dinh\s*(?::|la|co|ghi nhan)?\s*$|\bfamily history\s*(?::|of)?\s*$|"
    r"\b(?:bo|cha|me|ong|ba|anh trai|chi gai|em trai|em gai|father|mother)"
    r"(?:\s+benh nhan)?\s+(?:bi|mac|co|tien su|duoc chan doan|da tung)\s*$)"
)
EXPLICIT_HISTORY_RE = re.compile(
    r"(?:\btien su\b|\bda tung\b|\btung bi\b|\btruoc day\b|\bpreviously\b|\bhistory of\b|"
    r"\bpast medical history\b|\bpmh\b|\bkeo dai tu lau\b|\btrong vai nam\b|"
    r"\bman tinh\b|\bman tinh\b|\bcu\b)"
)

HISTORICAL_DRUG_RE = re.compile(
    r"(?:\bthuoc truoc khi nhap vien\b|\bthuoc truoc nhap vien\b|\bhome meds?\b|"
    r"\bprior to admission\b|\bda dung\b|\btung dung\b|\bsu dung\b|\bda ngung\b|"
    r"\bngung su dung\b|\bo nha\b)"
)
CURRENT_TREATMENT_RE = re.compile(
    r"(?:\bduoc chi dinh dieu tri\b|\btai khoa cap cuu\b|\bden khoa cap cuu\b|"
    r"\bdanh gia tai benh vien\b|\bdieu tri tai benh vien\b|\bdang dieu tri\b)"
)
_NARRATIVE_LINE_MIN_CHARS = 300
HISTORICAL_SUBSECTION_RE = re.compile(
    r"(?:\bcac benh ly (?:noi khoa )?man tinh\b|\bbenh ly man tinh\b|\bbenh ly man tinh\b|"
    r"\bthuoc truoc khi nhap vien\b|\bthuoc truoc nhap vien\b|\bcac tap kinh lam sang truoc day\b|"
    r"\btien su phau thuat\b|\bcac su kien truoc khi nhap vien\b|\bcac dien bien truoc khi nhap vien\b)"
)
PRESENT_ILLNESS_RE = re.compile(
    r"(?:\btien su benh hien tai\b|\bbenh su hien tai\b|\blich su benh hien tai\b|"
    r"\btrieu chung hien tai\b|\bly do nhap vien\b)"
)


class ContextDetector:
    def __init__(self, classifier: AssertionClassifier | None = None) -> None:
        self.classifier = classifier or AssertionClassifier.empty()

    def assertions_for(self, text: str, start: int, end: int, concept_type: str) -> tuple[str, ...]:
        if concept_type not in ASSERTION_TYPES:
            return ()

        case_start, _ = case_bounds_at(text, start)
        line_start = max(case_start, _line_start(text, start))
        line_end = text.find("\n", end)
        if line_end < 0:
            line_end = len(text)
        if line_end - line_start >= _NARRATIVE_LINE_MIN_CHARS:
            return self._merge_classifier(text, start, end, concept_type, ())
        clause_start = max(case_start, _clause_start(text, start))
        clause_before = normalize_key(text[clause_start:start])
        line_before = normalize_key(text[line_start:start])
        mention_key = normalize_key(text[start:end])
        recent_before = normalize_key(text[max(case_start, start - 700) : start])
        section = _section_at(text, start)

        assertions: list[str] = []
        if self._has_negation(clause_before, mention_key):
            assertions.append(ASSERTION_NEGATED)
        if FAMILY_SUBJECT_RE.search(line_before[-180:]) or FAMILY_SUBJECT_RE.search(recent_before[-180:]):
            assertions.append(ASSERTION_FAMILY)
        if self._has_historical_context(
            concept_type=concept_type,
            section=section,
            clause_before=clause_before,
            line_before=line_before,
            recent_before=recent_before,
        ):
            assertions.append(ASSERTION_HISTORICAL)
        return self._merge_classifier(text, start, end, concept_type, tuple(assertions))

    def _merge_classifier(
        self,
        text: str,
        start: int,
        end: int,
        concept_type: str,
        rule_assertions: tuple[str, ...],
    ) -> tuple[str, ...]:
        probabilities = self.classifier.probabilities(
            text,
            start,
            end,
            concept_type,
            rule_assertions,
        )
        if not probabilities:
            return rule_assertions
        selected: set[str] = {
            label for label in rule_assertions if label not in probabilities
        }
        for label in ALLOWED_ASSERTIONS:
            if label not in probabilities:
                continue
            probability = probabilities.get(label, 0.0)
            threshold = self.classifier.threshold(label)
            if probability >= threshold:
                selected.add(label)
            elif label in rule_assertions and probability >= max(0.05, threshold * 0.55):
                selected.add(label)
        return tuple(label for label in ALLOWED_ASSERTIONS if label in selected)

    @staticmethod
    def _has_negation(clause_before: str, mention_key: str = "") -> bool:
        context = " ".join(part for part in (clause_before, mention_key) if part)
        if not context:
            return False
        matches = list(NEGATION_RE.finditer(context))
        if not matches:
            return False
        last = matches[-1]
        negated_scope = context[last.start() :]
        if any(negated_scope.startswith(prefix) for prefix in NEGATION_EXCLUSIONS):
            return False
        contrast = list(CONTRAST_RE.finditer(negated_scope))
        return not contrast

    @staticmethod
    def _has_historical_context(
        concept_type: str,
        section: str,
        clause_before: str,
        line_before: str,
        recent_before: str,
    ) -> bool:
        local_context = " ".join((line_before, clause_before))
        if concept_type == TYPE_DRUG:
            if CURRENT_TREATMENT_RE.search(local_context) or CURRENT_TREATMENT_RE.search(recent_before[-180:]):
                return False
            if HISTORICAL_DRUG_RE.search(local_context):
                return True
            if section == "pre_admission":
                return bool(HISTORICAL_DRUG_RE.search(recent_before) or "thuoc" in recent_before[-240:])
            if HISTORICAL_SUBSECTION_RE.search(recent_before[-450:]):
                return True
            return False

        if PRESENT_ILLNESS_RE.search(recent_before[-500:]) and section == "present_illness":
            if concept_type == TYPE_SYMPTOM:
                return bool(EXPLICIT_HISTORY_RE.search(local_context))
            if not EXPLICIT_HISTORY_RE.search(local_context):
                return False

        if EXPLICIT_HISTORY_RE.search(local_context):
            return True
        if section == "pre_admission":
            if concept_type == TYPE_DIAGNOSIS:
                return True
            if concept_type == TYPE_SYMPTOM:
                return bool(
                    EXPLICIT_HISTORY_RE.search(recent_before[-350:])
                    or re.search(r"\b(?:tap kinh|cac con).*(?:truoc day|vai nam)\b", recent_before[-350:])
                )
        if concept_type == TYPE_DIAGNOSIS and HISTORICAL_SUBSECTION_RE.search(recent_before[-450:]):
            return True
        return False


def _line_start(text: str, offset: int) -> int:
    index = text.rfind("\n", 0, offset)
    return index + 1 if index >= 0 else 0


def _clause_start(text: str, offset: int) -> int:
    last = _line_start(text, offset)
    for separator in (".", ";", ":", "!", "?"):
        index = text.rfind(separator, last, offset)
        if index >= 0:
            last = max(last, index + 1)
    return last


@lru_cache(maxsize=128)
def _sections(text: str) -> tuple[Section, ...]:
    return tuple(detect_sections(text))


def _section_at(text: str, offset: int) -> str:
    for section in _sections(text):
        if section.start <= offset < section.end:
            return section.name
    return "document"
