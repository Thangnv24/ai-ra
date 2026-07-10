"""Deterministic lab name/result extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.config import TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.text import normalize_key, trim_span_text


@dataclass(frozen=True, slots=True)
class LabSpan:
    start: int
    end: int
    text: str
    type: str


LAB_ABBREVIATIONS = {
    "wbc",
    "twbc",
    "rbc",
    "hgb",
    "hct",
    "plt",
    "neut",
    "neut%",
    "lyph",
    "lyph%",
    "lymph",
    "lymph%",
    "mono",
    "mono%",
    "eos",
    "eos%",
    "baso",
    "baso%",
    "ast",
    "alt",
    "ap",
    "alp",
    "tbili",
    "dbili",
    "crp",
    "creatinine",
    "glucose",
    "ure",
    "bun",
    "na",
    "k",
    "cl",
    "ca",
}

LAB_KEYWORDS = (
    "bach cau",
    "hong cau",
    "tieu cau",
    "hemoglobin",
    "hematocrit",
    "bilirubin",
    "phosphatase",
    "aspartate aminotransferase",
    "alanine aminotransferase",
    "creatinine",
    "glucose",
    "protein",
    "albumin",
    "men gan",
)

LAB_PAIR_RE = re.compile(
    r"(?P<name>[^\n;:=]{1,120}?)\s*(?P<sep>:|=|\bl\u00e0\b)\s*"
    r"(?P<result>[<>]?\s*\d+(?:[,.]\d+)?(?:\s*(?:mg/dL|g/L|g/dL|mmol/L|"
    r"mmol|IU/L|U/L|mEq/L|ng/mL|pg/mL|x10\^?\d*/L|%))?)",
    re.IGNORECASE,
)


def extract_lab_spans(text: str) -> list[LabSpan]:
    spans: list[LabSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for match in LAB_PAIR_RE.finditer(text):
        name_start, name_end = _tighten_name_span(text, match.start("name"), match.end("name"))
        result_start, result_end = match.span("result")
        name_start, name_end, name_text = trim_span_text(text, name_start, name_end)
        name_start, name_end, name_text = _strip_lab_prefix(text, name_start, name_end)
        result_start, result_end, result_text = trim_span_text(text, result_start, result_end)
        if not name_text or not result_text or not _looks_like_lab_name(name_text):
            continue
        for start, end, span_text, span_type in (
            (name_start, name_end, name_text, TYPE_TEST_NAME),
            (result_start, result_end, result_text, TYPE_TEST_RESULT),
        ):
            key = (start, end, span_type)
            if key not in seen:
                seen.add(key)
                spans.append(LabSpan(start, end, span_text, span_type))
    return sorted(spans, key=lambda span: (span.start, span.end, span.type))


def _tighten_name_span(text: str, start: int, end: int) -> tuple[int, int]:
    segment = text[start:end]
    cut = 0
    for separator in ("\n", ";", "."):
        idx = segment.rfind(separator)
        if idx >= 0:
            cut = max(cut, idx + 1)
    return start + cut, end


def _strip_lab_prefix(text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[start] in "-*•":
        start += 1
    while start < end and text[start].isspace():
        start += 1
    return start, end, text[start:end]


def _looks_like_lab_name(name: str) -> bool:
    key = normalize_key(name)
    if not key:
        return False

    first = key.split()[0].strip(":")
    first_base = first.rstrip("%")
    if first in LAB_ABBREVIATIONS or first_base in LAB_ABBREVIATIONS:
        return True
    if any(keyword in key for keyword in LAB_KEYWORDS):
        return True
    if re.fullmatch(r"[a-z]{2,8}%?(?:\s*\([^)]+\))?", key):
        return True
    return False
