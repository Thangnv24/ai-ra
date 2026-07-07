"""Rule-based assertion/context detection."""

from __future__ import annotations

import re

from medkg.config import (
    ASSERTION_FAMILY,
    ASSERTION_HISTORICAL,
    ASSERTION_NEGATED,
    ASSERTION_TYPES,
)
from medkg.normalization import normalize_key


NEGATION_RE = re.compile(
    r"(?:\bkhong\b|\bkhong ghi nhan\b|\bchua\b|\bphu nhan\b|\bkhong thay\b|\bno\b|\bdenies\b|\bwithout\b|\bnegative for\b)"
)
FAMILY_RE = re.compile(
    r"(?:\bgia dinh\b|\bbo\b|\bcha\b|\bme\b|\banh\b|\bchi\b|\bem\b|\bong\b|\bba\b|\bfamily\b|\bfather\b|\bmother\b)"
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
        if HISTORICAL_RE.search(sentence_before) or HISTORICAL_RE.search(broad_before):
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


def _sentence_start(text: str, offset: int) -> int:
    last = 0
    for sep in ".;\n":
        idx = text.rfind(sep, 0, offset)
        if idx >= 0:
            last = max(last, idx + 1)
    return last

