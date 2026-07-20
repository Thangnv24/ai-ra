"""Generate exact-substring boundary alternatives for span adjudication."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from core.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.text import normalize_key, trim_span_text
from extraction.ner import SpanCandidate
from extraction.sectioning import detect_subsections
from extraction.annotation_memory import section_at


_DRUG_SIG_CUT_RE = re.compile(
    r"\s+(?:"
    r"x\s*\d+(?:[.,]\d+)?(?:\s*(?:v|viên|ống|gói|lọ|chai))?"
    r"|uống|tiêm|truyền|bôi|nhỏ|đặt"
    r"|sáng|trưa|chiều|tối"
    r"|\d+(?:[.,]\d+)?\s*lần\s*/?\s*ngày"
    r"|po|iv|im|sc|sq|sl|pr|bid|tid|qid|qd|qam|qpm|qhs|prn|daily"
    r")\b",
    re.IGNORECASE,
)
_STRENGTH_RE = re.compile(
    r"\s+\d+(?:[.,/]\d+)*(?:\s*(?:mg/ml|mcg/ml|mg|mcg|g|ml|iu|%))\b",
    re.IGNORECASE,
)
_RESULT_WITH_UNIT_RE = re.compile(
    r"^[ \t]*(?:mmhg|mmol/l|µmol/l|umol/l|mg/dl|g/l|g/dl|10\^?\d+/l|%|bpm|lần/phút|nhịp/phút|°c|c)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:[.,/]\d+)*$")
_NEGATION_PREFIX_RE = re.compile(r"(?:không|chưa|phủ nhận)\s+$", re.IGNORECASE)
_SYMPTOM_MODIFIER_RE = re.compile(
    r"^(?:\s+(?:trái|phải|hai bên|toàn thân|khi gắng sức|khi nằm|về đêm|"
    r"dữ dội|âm ỉ|nhẹ|nặng|từng cơn|liên tục|vùng [^,.;:\n]{1,30}))",
    re.IGNORECASE,
)
_DIAGNOSIS_CORE_CUT_RE = re.compile(
    r"\s*(?:,|;|-)?\s+(?:do|nghi do|kèm|trên nền|thứ phát sau)\b",
    re.IGNORECASE,
)
_VITAL_PREFIXES = (
    "ha ",
    "huyet ap ",
    "mach ",
    "nhip tim ",
    "nhip tho ",
    "spo2 ",
    "nhiet do ",
)


@dataclass(frozen=True, slots=True)
class BoundaryVariantSummary:
    inputs: int
    outputs: int
    generated: int
    by_variant: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "generated": self.generated,
            "by_variant": dict(self.by_variant),
        }


class BoundaryVariantGenerator:
    def expand(
        self,
        text: str,
        spans: Iterable[SpanCandidate],
    ) -> tuple[list[SpanCandidate], BoundaryVariantSummary]:
        original = list(spans)
        subsections = detect_subsections(text)
        output: list[SpanCandidate] = []
        generated_counts: Counter[str] = Counter()
        for span in original:
            output.append(span)
            subsection = section_at(subsections, span.start)
            for variant in self._variants_for(text, span, subsection):
                output.append(variant)
                generated_counts[variant.variant] += 1
        deduplicated = _deduplicate(output)
        return deduplicated, BoundaryVariantSummary(
            inputs=len(original),
            outputs=len(deduplicated),
            generated=max(0, len(deduplicated) - len(original)),
            by_variant=tuple(sorted(generated_counts.items())),
        )

    def _variants_for(
        self,
        text: str,
        span: SpanCandidate,
        subsection: str,
    ) -> list[SpanCandidate]:
        if span.type == TYPE_DRUG:
            return _drug_variants(text, span)
        if span.type == TYPE_TEST_NAME:
            return _test_name_variants(text, span, subsection)
        if span.type == TYPE_TEST_RESULT:
            return _test_result_variants(text, span, subsection)
        if span.type == TYPE_SYMPTOM:
            return _symptom_variants(text, span)
        if span.type == TYPE_DIAGNOSIS:
            return _diagnosis_variants(text, span)
        return []


def _drug_variants(text: str, span: SpanCandidate) -> list[SpanCandidate]:
    output: list[SpanCandidate] = []
    local = text[span.start:span.end]
    sig = _DRUG_SIG_CUT_RE.search(local)
    if sig and sig.start() > 1:
        candidate = _make_variant(text, span, span.start, span.start + sig.start(), "drug_without_sig", -0.01)
        if candidate:
            output.append(candidate)
    strength = _STRENGTH_RE.search(local)
    if strength and strength.start() > 1:
        candidate = _make_variant(text, span, span.start, span.start + strength.start(), "drug_core", -0.04)
        if candidate:
            output.append(candidate)
    return output


def _test_name_variants(text: str, span: SpanCandidate, subsection: str) -> list[SpanCandidate]:
    if subsection not in {"laboratory", "imaging_procedure", "vital_signs"}:
        return []
    line_start, line_end = _line_bounds(text, span.start, span.end)
    raw_line = text[line_start:line_end]
    leading = len(raw_line) - len(raw_line.lstrip(" \t-*•"))
    candidate_start = line_start + leading
    candidate_end = line_end
    result_separator = re.search(
        r"\s*[:=]\s*(?=[+-]?\d|dương\s+tính|âm\s+tính|positive|negative)",
        text[candidate_start:candidate_end],
        re.IGNORECASE,
    )
    if result_separator:
        candidate_end = candidate_start + result_separator.start()
    if not (candidate_start <= span.start and span.end <= candidate_end):
        return []
    if candidate_end - candidate_start > 320 or candidate_end - candidate_start <= span.end - span.start:
        return []
    candidate = _make_variant(
        text,
        span,
        candidate_start,
        candidate_end,
        "test_full_label",
        0.02,
    )
    return [candidate] if candidate else []


def _test_result_variants(text: str, span: SpanCandidate, subsection: str) -> list[SpanCandidate]:
    output: list[SpanCandidate] = []
    if _NUMERIC_RE.match(normalize_key(span.text)):
        tail = text[span.end:min(len(text), span.end + 24)]
        unit = _RESULT_WITH_UNIT_RE.match(tail)
        if unit:
            candidate = _make_variant(
                text,
                span,
                span.start,
                span.end + unit.end(),
                "result_with_unit",
                0.03,
            )
            if candidate:
                output.append(candidate)
    if subsection == "vital_signs" or _line_has_vital_prefix(text, span.start, span.end):
        line_start, line_end = _line_bounds(text, span.start, span.end)
        raw = text[line_start:line_end]
        stripped_start = line_start + len(raw) - len(raw.lstrip(" \t-*•"))
        if line_end - stripped_start <= 100 and stripped_start < span.start:
            candidate = _make_variant(
                text,
                span,
                stripped_start,
                line_end,
                "vital_label_with_value",
                0.0,
            )
            if candidate:
                output.append(candidate)
    return output


def _symptom_variants(text: str, span: SpanCandidate) -> list[SpanCandidate]:
    output: list[SpanCandidate] = []
    prefix = text[max(0, span.start - 24):span.start]
    negation = _NEGATION_PREFIX_RE.search(prefix)
    if negation:
        candidate = _make_variant(
            text,
            span,
            max(0, span.start - 24) + negation.start(),
            span.end,
            "symptom_with_negation",
            0.0,
        )
        if candidate:
            output.append(candidate)
    tail = text[span.end:min(len(text), span.end + 50)]
    modifier = _SYMPTOM_MODIFIER_RE.match(tail)
    if modifier:
        candidate = _make_variant(
            text,
            span,
            span.start,
            span.end + modifier.end(),
            "symptom_with_modifier",
            0.02,
        )
        if candidate:
            output.append(candidate)
    return output


def _diagnosis_variants(text: str, span: SpanCandidate) -> list[SpanCandidate]:
    local = text[span.start:span.end]
    cut = _DIAGNOSIS_CORE_CUT_RE.search(local)
    if not cut or cut.start() <= 2:
        return []
    candidate = _make_variant(
        text,
        span,
        span.start,
        span.start + cut.start(),
        "diagnosis_core",
        0.01,
    )
    return [candidate] if candidate else []


def _make_variant(
    text: str,
    parent: SpanCandidate,
    start: int,
    end: int,
    variant: str,
    score_delta: float,
) -> SpanCandidate | None:
    start, end, value = trim_span_text(text, start, end)
    if not value or (start, end) == (parent.start, parent.end):
        return None
    return SpanCandidate(
        start=start,
        end=end,
        text=value,
        type=parent.type,
        score=max(0.0, min(0.99, parent.score + score_delta)),
        source=parent.source,
        variant=variant,
        parent_source=parent.parent_source or parent.source,
    )


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return line_start, len(text) if line_end < 0 else line_end


def _line_has_vital_prefix(text: str, start: int, end: int) -> bool:
    line_start, _ = _line_bounds(text, start, end)
    prefix = normalize_key(text[line_start:start]).lstrip("-* ")
    return any(prefix.startswith(value) for value in _VITAL_PREFIXES)


def _deduplicate(spans: Iterable[SpanCandidate]) -> list[SpanCandidate]:
    best: dict[tuple[int, int, str, str], SpanCandidate] = {}
    for span in spans:
        key = (span.start, span.end, span.type, span.source)
        current = best.get(key)
        if current is None or span.score > current.score:
            best[key] = span
    return sorted(best.values(), key=lambda item: (item.start, item.end, item.type, item.source))
