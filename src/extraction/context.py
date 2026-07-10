"""Rule-based assertion/context detection."""

from __future__ import annotations

import re

from core.config import (
    ASSERTION_FAMILY,
    ASSERTION_HISTORICAL,
    ASSERTION_NEGATED,
    ASSERTION_TYPES,
    TYPE_DRUG,
)
from core.text import normalize_key


NEGATION_RE = re.compile(
    r"(?:\bkhong\b|\bkhong ghi nhan\b|\bchua\b|\bphu nhan\b|\bkhong thay\b|\bno\b|\bdenies\b|\bwithout\b|\bnegative for\b)"
)
FAMILY_RE = re.compile(
    r"(?:\btien su gia dinh\b|\bgia dinh\b|\bnguoi than\b|\bfamily\b|\bfather\b|\bmother\b"
    r"|\banh trai\b|\bchi gai\b|\bem trai\b|\bem gai\b"
    r"|\b(?:bo|cha|me|ong|ba)\s+(?:bi|mac|co|tien su|duoc chan doan|da tung)\b)"
)
HISTORICAL_RE = re.compile(
    r"(?:\btien su\b|\btruoc nhap vien\b|\bda tung\b|\bbenh su\b|\bpast medical history\b|\bpmh\b|\bhistory of\b|\bhome meds?\b|\bprior to admission\b|\bpreviously\b)"
)


class ContextDetector:
    def assertions_for(self, text: str, start: int, end: int, concept_type: str) -> tuple[str, ...]:
        if concept_type not in ASSERTION_TYPES:
            return ()

        before = normalize_key(text[max(0, start - 180) : start])
        sentence_before = normalize_key(text[max(0, _sentence_start(text, start)) : start])
        broad_before = normalize_key(text[max(0, start - 600) : start])

        assertions: list[str] = []
        if self._has_close_negation(before, sentence_before):
            assertions.append(ASSERTION_NEGATED)
        if FAMILY_RE.search(sentence_before) or FAMILY_RE.search(before[-80:]):
            assertions.append(ASSERTION_FAMILY)
        if self._has_historical_context(sentence_before, broad_before, concept_type):
            assertions.append(ASSERTION_HISTORICAL)

        return tuple(assertions)

    @staticmethod
    def _has_close_negation(before: str, sentence_before: str) -> bool:
        if not before:
            return False
        close = " ".join(before.split()[-8:])
        if NEGATION_RE.search(close):
            return True
        return bool(NEGATION_RE.search(sentence_before) and len(sentence_before.split()) <= 16)

    @staticmethod
    def _has_historical_context(sentence_before: str, broad_before: str, concept_type: str) -> bool:
        if _is_present_illness_context(sentence_before) or _is_present_illness_context(broad_before):
            return False
        if HISTORICAL_RE.search(sentence_before):
            return True
        if concept_type == TYPE_DRUG:
            return bool(re.search(r"\b(?:thuoc truoc nhap vien|truoc nhap vien|home meds?|prior to admission)\b", broad_before))
        return bool(re.search(r"\b(?:tien su benh noi khoa|past medical history|pmh|da tung|previously)\b", broad_before))


def _sentence_start(text: str, offset: int) -> int:
    last = 0
    for sep in ".;\n":
        idx = text.rfind(sep, 0, offset)
        if idx >= 0:
            last = max(last, idx + 1)
    return last


def _is_present_illness_context(text: str) -> bool:
    return bool(re.search(r"\b(?:tien su benh hien tai|benh su hien tai|history of present illness)\b", text))
