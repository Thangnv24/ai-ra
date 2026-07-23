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
    "bc",
    "ha",
    "n",
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
    "hbsag",
    "spo2",
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
    "creatinin",
    "glucose",
    "protein",
    "albumin",
    "men gan",
    "anti hbe",
    "anti hbc",
    "prothrombin",
    "deritis",
    "huyet ap",
    "nhiet do",
    "mach",
    "nhip tim",
    "nhip tho",
    "do bao hoa oxy",
)

_LAB_UNIT = (
    r"mg/dL|g/L|g/dL|mmol/L|mmol|micromol/L|umol/L|IU/L|U/L|mEq/L|"
    r"ng/mL|ng/L|pg/mL|x10\^?\d*/L|G/L|T/L|ck/p|lần/phút|lan/phut|mmHg|°C|%"
)
_LAB_RESULT = (
    rf"(?:\d{{2,3}}\s*/\s*\d{{2,3}}(?:\s*mmHg)?|"
    rf"[<>]?\s*\d+(?:[,.]\d+)?(?:\s*(?:{_LAB_UNIT}))?|\(\s*[+-]\s*\))"
)
_LAB_QUALITATIVE_RESULT = (
    r"(?:chưa phát hiện bất thường(?: trên phim chụp)?|"
    r"không phát hiện bất thường(?: trên phim chụp)?|"
    r"đều tăng|bình thường|bất thường|âm tính|dương tính)"
)

LAB_PAIR_RE = re.compile(
    r"(?P<name>[^\n;:=]{1,120}?)\s*(?P<sep>:|=|\bl\u00e0\b)\s*"
    rf"(?P<result>{_LAB_RESULT})",
    re.IGNORECASE,
)
LAB_RATIO_RESULT_RE = re.compile(
    rf"(?P<result>[A-Za-z]{{2,10}}\s*/\s*[A-Za-z]{{2,10}}\s*[<>=]\s*"
    rf"\d+(?:[,.]\d+)?(?:\s*(?:{_LAB_UNIT}))?)",
    re.IGNORECASE,
)
_LAB_INLINE_NAMES = LAB_ABBREVIATIONS | {
    "anti hbe",
    "anti hbc igg",
    "anti hbc igm",
}
LAB_INLINE_PAIR_RE = re.compile(
    rf"(?<!\w)(?P<name>{'|'.join(re.escape(name) for name in sorted(_LAB_INLINE_NAMES, key=len, reverse=True))})"
    rf"(?!\w)\s+(?P<result>{_LAB_RESULT})",
    re.IGNORECASE,
)
LAB_QUALITATIVE_PAIR_RE = re.compile(
    rf"(?<!\w)(?P<name>{'|'.join(re.escape(name) for name in sorted(_LAB_INLINE_NAMES, key=len, reverse=True))})"
    rf"(?!\w)(?P<middle>[^\n.;:]{{0,45}}?)\s+(?P<result>{_LAB_QUALITATIVE_RESULT})",
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
        if not name_text or not result_text:
            continue
        looks_like_name = _looks_like_lab_name(name_text)
        candidates = []
        if looks_like_name:
            candidates.append((name_start, name_end, name_text, TYPE_TEST_NAME))
        if looks_like_name or _line_has_lab_context(text, match.start()):
            candidates.append((result_start, result_end, result_text, TYPE_TEST_RESULT))
        for start, end, span_text, span_type in candidates:
            key = (start, end, span_type)
            if key not in seen:
                seen.add(key)
                spans.append(LabSpan(start, end, span_text, span_type))

    for match in LAB_RATIO_RESULT_RE.finditer(text):
        start, end, result_text = trim_span_text(text, *match.span("result"))
        key = (start, end, TYPE_TEST_RESULT)
        if result_text and key not in seen:
            seen.add(key)
            spans.append(LabSpan(start, end, result_text, TYPE_TEST_RESULT))

    for match in LAB_INLINE_PAIR_RE.finditer(text):
        name_start, name_end, name_text = trim_span_text(text, *match.span("name"))
        result_start, result_end, result_text = trim_span_text(text, *match.span("result"))
        for start, end, span_text, span_type in (
            (name_start, name_end, name_text, TYPE_TEST_NAME),
            (result_start, result_end, result_text, TYPE_TEST_RESULT),
        ):
            key = (start, end, span_type)
            if span_text and key not in seen:
                seen.add(key)
                spans.append(LabSpan(start, end, span_text, span_type))

    for match in LAB_QUALITATIVE_PAIR_RE.finditer(text):
        name_start, name_end, name_text = trim_span_text(text, *match.span("name"))
        result_start, result_end, result_text = trim_span_text(text, *match.span("result"))
        for start, end, span_text, span_type in (
            (name_start, name_end, name_text, TYPE_TEST_NAME),
            (result_start, result_end, result_text, TYPE_TEST_RESULT),
        ):
            key = (start, end, span_type)
            if span_text and key not in seen:
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


def _line_has_lab_context(text: str, offset: int) -> bool:
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end < 0:
        line_end = len(text)
    line_key = normalize_key(text[line_start:line_end])
    if any(keyword in line_key for keyword in LAB_KEYWORDS):
        return True
    tokens = {token.strip(":()") for token in line_key.split()}
    return bool(tokens & LAB_ABBREVIATIONS)
