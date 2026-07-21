"""Deterministic, template-aware boundary grammar."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.config import TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_RESULT
from core.text import normalize_key, trim_span_text
from extraction.ner import SpanCandidate


_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:[.,/]\d+)*$")


@dataclass(frozen=True, slots=True)
class SpanGrammarSummary:
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


class SpanGrammar:
    def __init__(self, policy: dict[str, object] | None = None) -> None:
        self.policy = policy or {}
        configured_variants = self.policy.get("enabled_variants")
        self.enabled_variants = (
            {str(value) for value in configured_variants}
            if isinstance(configured_variants, list)
            else {"drug_sig", "result_unit", "symptom_modifier"}
        )
        drug = self.policy.get("drug") or {}
        result = self.policy.get("test_result") or {}
        symptom = self.policy.get("symptom") or {}
        diagnosis = self.policy.get("diagnosis") or {}

        self.drug_units = tuple(str(value) for value in drug.get("units", ()))
        self.drug_tokens = tuple(
            str(value)
            for value in (
                *(drug.get("route_frequency_tokens") or ()),
                *(drug.get("form_tokens") or ()),
            )
        )
        self.vietnamese_instructions = tuple(
            str(value).casefold() for value in drug.get("vietnamese_instruction_tokens", ())
        )
        self.result_units = tuple(str(value) for value in result.get("units", ()))
        self.negation_prefixes = tuple(str(value) for value in symptom.get("negation_prefixes", ()))
        self.symptom_modifiers = tuple(str(value) for value in symptom.get("trailing_modifiers", ()))
        self.diagnosis_connectors = tuple(
            str(value) for value in diagnosis.get("explanation_connectors", ())
        )

        drug_unit_pattern = _alternatives(self.drug_units)
        drug_token_pattern = _alternatives(self.drug_tokens)
        self._drug_tail_re = re.compile(
            rf"^(?:(?:\s+)(?:\d+(?:[.,/]\d+)*\s*(?:{drug_unit_pattern})|"
            rf"x\s*\d+|q\s*\d+\s*h(?:\s*:?\s*prn)?|{drug_token_pattern})){{1,8}}",
            re.IGNORECASE,
        )
        self._result_unit_re = re.compile(
            rf"^[ \t]*(?:{_alternatives(self.result_units)})(?=$|\s|[,;.)])",
            re.IGNORECASE,
        )
        self._symptom_modifier_re = re.compile(
            rf"^(?:(?:\s+)(?:{_alternatives(self.symptom_modifiers)})){{1,3}}",
            re.IGNORECASE,
        )
        self._negation_re = re.compile(
            rf"(?:{_alternatives(self.negation_prefixes)})\s+$",
            re.IGNORECASE,
        )
        self._diagnosis_cut_re = re.compile(
            rf"\s*(?:,|;|-)??\s+(?:{_alternatives(self.diagnosis_connectors)})\b",
            re.IGNORECASE,
        )

    @classmethod
    def load(cls, path: Path) -> SpanGrammar:
        if not path.exists():
            return cls({})
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload if isinstance(payload, dict) else {})

    def expand(
        self,
        text: str,
        spans: Iterable[SpanCandidate],
    ) -> tuple[list[SpanCandidate], SpanGrammarSummary]:
        original = list(spans)
        output = list(original)
        generated = Counter()
        for span in original:
            variant = self._variant_for(text, span)
            if variant is None:
                continue
            output.append(variant)
            generated[variant.variant] += 1
        deduplicated = _deduplicate(output)
        return deduplicated, SpanGrammarSummary(
            inputs=len(original),
            outputs=len(deduplicated),
            generated=max(0, len(deduplicated) - len(original)),
            by_variant=tuple(sorted(generated.items())),
        )

    def _variant_for(self, text: str, span: SpanCandidate) -> SpanCandidate | None:
        if span.type == TYPE_DRUG and "drug_sig" in self.enabled_variants:
            return self._drug_variant(text, span)
        if span.type == TYPE_TEST_RESULT and "result_unit" in self.enabled_variants:
            return self._result_variant(text, span)
        if span.type == TYPE_SYMPTOM and "symptom_modifier" in self.enabled_variants:
            return self._symptom_variant(text, span)
        return None

    def _drug_variant(self, text: str, span: SpanCandidate) -> SpanCandidate | None:
        line_end = text.find("\n", span.end)
        if line_end < 0:
            line_end = len(text)
        tail = text[span.end:min(line_end, span.end + 100)]
        match = self._drug_tail_re.match(tail)
        if not match:
            return None
        extension = match.group(0)
        count = re.search(r"\s+x\s*\d+", extension, re.IGNORECASE)
        remaining = tail[match.end():].lstrip().casefold()
        if count and any(remaining.startswith(token) for token in self.vietnamese_instructions):
            extension = extension[:count.start()]
        end = span.end + len(extension.rstrip())
        return _make_variant(text, span, span.start, end, "grammar_drug_sig", 0.035)

    def _result_variant(self, text: str, span: SpanCandidate) -> SpanCandidate | None:
        if not _NUMERIC_RE.fullmatch(normalize_key(span.text)):
            return None
        tail = text[span.end:min(len(text), span.end + 32)]
        match = self._result_unit_re.match(tail)
        if not match:
            return None
        return _make_variant(
            text,
            span,
            span.start,
            span.end + match.end(),
            "grammar_result_unit",
            0.04,
        )

    def _symptom_variant(self, text: str, span: SpanCandidate) -> SpanCandidate | None:
        tail = text[span.end:min(len(text), span.end + 60)]
        modifier = self._symptom_modifier_re.match(tail)
        if modifier:
            return _make_variant(
                text,
                span,
                span.start,
                span.end + modifier.end(),
                "grammar_symptom_modifier",
                0.03,
            )
        return None


def _alternatives(values: Iterable[str]) -> str:
    ordered = sorted({value for value in values if value}, key=len, reverse=True)
    return "|".join(re.escape(value) for value in ordered) or r"(?!)"


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
        source="grammar",
        variant=variant,
        parent_source=parent.parent_source or parent.source,
    )


def _deduplicate(spans: Iterable[SpanCandidate]) -> list[SpanCandidate]:
    best: dict[tuple[int, int, str, str], SpanCandidate] = {}
    for span in spans:
        key = (span.start, span.end, span.type, span.source)
        current = best.get(key)
        if current is None or span.score > current.score:
            best[key] = span
    return sorted(best.values(), key=lambda item: (item.start, item.end, item.type, item.source))
