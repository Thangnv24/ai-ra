"""Section- and clause-aware assertion detection."""

from __future__ import annotations

import re
from functools import lru_cache

from core.config import (
    ASSERTION_FAMILY,
    ASSERTION_HISTORICAL,
    ASSERTION_NEGATED,
    ASSERTION_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
)
from core.text import normalize_key
from extraction.sectioning import Section, detect_sections


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
)
CONTRAST_RE = re.compile(r"\b(?:nhung|tuy nhien|song|however|but)\b")
FAMILY_RE = re.compile(
    r"(?:\btien su gia dinh\b|\bgia dinh\b|\bnguoi than\b|\bfamily\b|\bfather\b|\bmother\b"
    r"|\banh trai\b|\bchi gai\b|\bem trai\b|\bem gai\b"
    r"|\b(?:bo|cha|me|ong|ba)\s+(?:bi|mac|co|tien su|duoc chan doan|da tung)\b)"
)
EXPLICIT_HISTORY_RE = re.compile(
    r"(?:\bda tung\b|\btung bi\b|\btruoc day\b|\bpreviously\b|\bhistory of\b|"
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
    def assertions_for(self, text: str, start: int, end: int, concept_type: str) -> tuple[str, ...]:
        if concept_type not in ASSERTION_TYPES:
            return ()

        clause_before = normalize_key(text[_clause_start(text, start) : start])
        line_before = normalize_key(text[_line_start(text, start) : start])
        recent_before = normalize_key(text[max(0, start - 700) : start])
        section = _section_at(text, start)

        assertions: list[str] = []
        if self._has_negation(clause_before):
            assertions.append(ASSERTION_NEGATED)
        if FAMILY_RE.search(line_before) or FAMILY_RE.search(recent_before[-220:]):
            assertions.append(ASSERTION_FAMILY)
        if self._has_historical_context(
            concept_type=concept_type,
            section=section,
            clause_before=clause_before,
            line_before=line_before,
            recent_before=recent_before,
        ):
            assertions.append(ASSERTION_HISTORICAL)
        return tuple(assertions)

    @staticmethod
    def _has_negation(clause_before: str) -> bool:
        if not clause_before:
            return False
        matches = list(NEGATION_RE.finditer(clause_before))
        if not matches:
            return False
        last = matches[-1]
        negated_scope = clause_before[last.start() :]
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
    for separator in (".", ";", ":"):
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
