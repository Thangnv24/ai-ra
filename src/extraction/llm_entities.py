"""LLM-assisted entity proposal with deterministic quote alignment."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.config import (
    ALLOWED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)
from core.text import normalize_key, trim_span_text
from extraction.annotation_memory import AnnotationMemory
from extraction.ner import SpanCandidate
from extraction.sectioning import TextChunk, split_chunks
from integrations.prompts import ENTITY_SYSTEM_PROMPT, build_entity_extraction_prompt


@dataclass(frozen=True, slots=True)
class EntityProposalSummary:
    chunks: int
    mentions: int
    aligned: int
    calls: int = 0
    base_calls: int = 0
    rescue_calls: int = 0
    aligned_before_dedup: int = 0
    deduplicated: int = 0
    rejection_reasons: tuple[tuple[str, int], ...] = ()
    mentions_by_type: tuple[tuple[str, int], ...] = ()
    aligned_by_type: tuple[tuple[str, int], ...] = ()
    rescue_by_type: tuple[tuple[str, int], ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "mentions": self.mentions,
            "aligned": self.aligned,
            "calls": self.calls,
            "base_calls": self.base_calls,
            "rescue_calls": self.rescue_calls,
            "aligned_before_dedup": self.aligned_before_dedup,
            "deduplicated": self.deduplicated,
            "rejected": sum(count for _, count in self.rejection_reasons),
            "rejection_reasons": dict(self.rejection_reasons),
            "mentions_by_type": dict(self.mentions_by_type),
            "aligned_by_type": dict(self.aligned_by_type),
            "rescue_by_type": dict(self.rescue_by_type),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class TextUnit:
    unit_id: str
    start: int
    end: int
    text: str


class LLMEntityExtractor:
    def __init__(
        self,
        llm_client: Any,
        memory: AnnotationMemory | None = None,
        max_chars: int = 1000,
        overlap: int = 0,
    ) -> None:
        self.llm_client = llm_client
        self.memory = memory or AnnotationMemory.empty()
        self.max_chars = max_chars
        self.overlap = overlap

    def extract(self, text: str) -> tuple[list[SpanCandidate], EntityProposalSummary]:
        if not getattr(self.llm_client, "enabled", False):
            raise RuntimeError("LLM entity extraction is required but the LLM client is disabled")

        chunks = _merge_small_chunks(
            text,
            split_chunks(text, max_chars=self.max_chars, overlap=self.overlap),
            max_chars=self.max_chars,
        )
        spans: list[SpanCandidate] = []
        mention_count = 0
        call_count = 0
        base_call_count = 0
        rescue_call_count = 0
        mentions_by_type: Counter[str] = Counter()
        aligned_by_type: Counter[str] = Counter()
        rescue_by_type: Counter[str] = Counter()
        rejection_reasons: Counter[str] = Counter()
        for chunk_index, chunk in enumerate(chunks):
            units = _chunk_units(chunk)
            context_before, context_after = _neighbor_context(chunks, chunk_index)
            payload = _chunk_payload(
                chunk,
                units,
                context_before=context_before,
                context_after=context_after,
            )
            result = self.llm_client.chat_json(
                ENTITY_SYSTEM_PROMPT,
                build_entity_extraction_prompt(payload),
            )
            call_count += 1
            base_call_count += 1
            if not result.ok or not isinstance(result.data, dict):
                raise RuntimeError(
                    f"LLM entity extraction failed for {chunk.chunk_id}: "
                    f"{result.error or 'LLM returned no data'}"
                )
            mentions = result.data.get("mentions")
            if not isinstance(mentions, list):
                raise RuntimeError(
                    f"LLM entity extraction failed for {chunk.chunk_id}: JSON has no mentions list"
                )
            mention_count += len(mentions)
            used_occurrences: dict[tuple[str, str], set[int]] = {}
            chunk_spans: list[SpanCandidate] = []

            def align_mentions(raw_mentions: list[Any], forced_type: str | None = None) -> None:
                for raw_mention in raw_mentions:
                    mention = (
                        _mention_for_requested_type(raw_mention, forced_type)
                        if forced_type is not None
                        else _coerce_compact_mention(raw_mention)
                    )
                    if isinstance(mention, dict):
                        raw_type = str(mention.get("type") or "")
                        if raw_type in ALLOWED_TYPES:
                            mentions_by_type[raw_type] += 1
                    span, rejection_reason = _span_from_mention_with_reason(
                        text,
                        chunk,
                        mention,
                        units=units,
                        used_occurrences=used_occurrences,
                    )
                    if span is None:
                        rejection_reasons[rejection_reason or "unknown"] += 1
                        continue
                    spans.append(span)
                    chunk_spans.append(span)
                    aligned_by_type[span.type] += 1

            # The base response was already counted above.
            align_mentions(mentions)
            for rescue_type in _coverage_rescue_types(chunk, chunk_spans):
                rescue_result = self.llm_client.chat_json(
                    ENTITY_SYSTEM_PROMPT,
                    build_entity_extraction_prompt(payload, rescue_type),
                )
                call_count += 1
                rescue_call_count += 1
                rescue_by_type[rescue_type] += 1
                if not rescue_result.ok or not isinstance(rescue_result.data, dict):
                    raise RuntimeError(
                        f"LLM entity rescue failed for {chunk.chunk_id}/{rescue_type}: "
                        f"{rescue_result.error or 'LLM returned no data'}"
                    )
                rescue_mentions = rescue_result.data.get("mentions")
                if not isinstance(rescue_mentions, list):
                    raise RuntimeError(
                        f"LLM entity rescue failed for {chunk.chunk_id}/{rescue_type}: "
                        "JSON has no mentions list"
                    )
                mention_count += len(rescue_mentions)
                align_mentions(rescue_mentions, rescue_type)
        aligned_before_dedup = len(spans)
        spans = _dedup_spans(spans)
        return spans, EntityProposalSummary(
            chunks=len(chunks),
            mentions=mention_count,
            aligned=len(spans),
            calls=call_count,
            base_calls=base_call_count,
            rescue_calls=rescue_call_count,
            aligned_before_dedup=aligned_before_dedup,
            deduplicated=aligned_before_dedup - len(spans),
            rejection_reasons=tuple(sorted(rejection_reasons.items())),
            mentions_by_type=tuple(sorted(mentions_by_type.items())),
            aligned_by_type=tuple(sorted(aligned_by_type.items())),
            rescue_by_type=tuple(sorted(rescue_by_type.items())),
        )


def _merge_small_chunks(
    text: str,
    chunks: list[TextChunk],
    *,
    max_chars: int,
    min_chars: int = 120,
) -> list[TextChunk]:
    """Attach structural headers and other tiny chunks to adjacent case text."""

    merged: list[TextChunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        previous = merged[-1]
        same_section = previous.section == chunk.section
        gap_is_whitespace = not text[previous.end : chunk.start].strip()
        combined_length = chunk.end - previous.start
        should_merge = len(previous.text) < min_chars or len(chunk.text) < min_chars
        if same_section and gap_is_whitespace and should_merge and combined_length <= max_chars:
            merged[-1] = TextChunk(
                chunk_id=previous.chunk_id,
                section=previous.section,
                start=previous.start,
                end=chunk.end,
                text=text[previous.start : chunk.end],
                subsection=(
                    chunk.subsection
                    if previous.subsection == "document" and chunk.subsection != "document"
                    else previous.subsection
                ),
            )
            continue
        merged.append(chunk)
    return merged


def _coverage_rescue_types(
    chunk: TextChunk,
    aligned_spans: list[SpanCandidate],
    *,
    limit: int = 2,
) -> tuple[str, ...]:
    """Choose specialist passes from deterministic section coverage gaps."""

    counts = Counter(span.type for span in aligned_spans)
    lines = [line.strip() for line in chunk.text.splitlines() if len(line.strip()) >= 3]
    substantive_lines = max(1, len(lines))
    subsection = chunk.subsection
    normalized = normalize_key(chunk.text)
    expectations: list[tuple[str, int, int]] = []

    if subsection == "medications":
        expectations.append((TYPE_DRUG, min(3, max(1, substantive_lines // 3)), 100))
    elif subsection == "laboratory":
        result_lines = sum(
            bool(re.search(r"(?:[:=]\s*)?[+-]?\d|\b(?:positive|negative|duong tinh|am tinh)\b", normalize_key(line)))
            for line in lines
        )
        minimum = min(3, max(1, result_lines // 2))
        expectations.extend(((TYPE_TEST_NAME, minimum, 100), (TYPE_TEST_RESULT, minimum, 95)))
    elif subsection == "imaging_procedure":
        minimum = min(2, max(1, substantive_lines // 3))
        expectations.extend(((TYPE_TEST_NAME, 1, 100), (TYPE_TEST_RESULT, minimum, 95)))
    elif subsection == "diagnoses":
        expectations.append((TYPE_DIAGNOSIS, min(3, max(1, substantive_lines // 2)), 100))
    elif subsection == "symptoms_exam":
        expectations.append((TYPE_SYMPTOM, min(3, max(1, substantive_lines // 3)), 100))
    elif subsection == "vital_signs":
        expectations.extend(((TYPE_TEST_NAME, 1, 100), (TYPE_TEST_RESULT, 1, 95)))
    elif subsection == "history":
        if any(cue in normalized for cue in ("chan doan", "benh ly", "history", "tien su")):
            expectations.append((TYPE_DIAGNOSIS, 1, 80))
        if any(cue in normalized for cue in ("thuoc", "medication", "dang dung")):
            expectations.append((TYPE_DRUG, 1, 75))

    gaps = [
        (priority, concept_type)
        for concept_type, minimum, priority in expectations
        if counts[concept_type] < minimum
    ]
    gaps.sort(key=lambda item: (-item[0], ALLOWED_TYPES.index(item[1])))
    return tuple(concept_type for _, concept_type in gaps[:limit])


def _mention_for_requested_type(mention: Any, concept_type: str) -> Any:
    if isinstance(mention, dict):
        output = dict(mention)
        output["type"] = concept_type
        return output
    if isinstance(mention, list) and mention:
        if len(mention) >= 2 and str(mention[1]) in ALLOWED_TYPES:
            output = list(mention)
            output[1] = concept_type
            return output
        return {
            "unit_id": mention[0] if len(mention) > 1 else "",
            "quote": mention[1] if len(mention) > 1 else mention[0],
            "occurrence_index": mention[2] if len(mention) > 2 else 0,
            "type": concept_type,
        }
    return mention


def align_quote_in_chunk(chunk: TextChunk, quote: str) -> tuple[int, int] | None:
    quote = " ".join(str(quote or "").split())
    if not quote:
        return None

    direct = chunk.text.find(quote)
    if direct >= 0:
        return (chunk.start + direct, chunk.start + direct + len(quote))

    lowered_text = chunk.text.casefold()
    lowered_quote = quote.casefold()
    insensitive = lowered_text.find(lowered_quote)
    if insensitive >= 0:
        return (chunk.start + insensitive, chunk.start + insensitive + len(quote))

    return _align_normalized_quote(chunk, quote)


def _chunk_payload(
    chunk: TextChunk,
    units: list[TextUnit] | None = None,
    context_before: str = "",
    context_after: str = "",
) -> dict[str, Any]:
    units = units or _chunk_units(chunk)
    return {
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
        "context_before": context_before,
        "context_after": context_after,
        "units": [
            {
                "unit_id": unit.unit_id,
                "text": unit.text,
            }
            for unit in units
        ],
    }


def _neighbor_context(chunks: list[TextChunk], index: int, max_chars: int = 240) -> tuple[str, str]:
    chunk = chunks[index]
    case_id = chunk.section.split(":", 1)[0]
    before = ""
    after = ""
    if index > 0 and chunks[index - 1].section.split(":", 1)[0] == case_id:
        before = chunks[index - 1].text[-max_chars:]
    if index + 1 < len(chunks) and chunks[index + 1].section.split(":", 1)[0] == case_id:
        after = chunks[index + 1].text[:max_chars]
    return before, after


def _span_from_mention(text: str, chunk: TextChunk, mention: Any) -> SpanCandidate | None:
    span, _ = _span_from_mention_with_reason(text, chunk, mention)
    return span


def _span_from_mention_with_reason(
    text: str,
    chunk: TextChunk,
    mention: Any,
    units: list[TextUnit] | None = None,
    used_occurrences: dict[tuple[str, str], set[int]] | None = None,
) -> tuple[SpanCandidate | None, str | None]:
    mention = _coerce_compact_mention(mention)
    if not isinstance(mention, dict):
        return None, "invalid_mention"
    quote = str(mention.get("quote") or "").strip()
    if not quote:
        return None, "empty_quote"
    concept_type = str(mention.get("type") or "").strip()
    if concept_type not in ALLOWED_TYPES:
        return None, "invalid_type"

    available_units = units or _chunk_units(chunk)
    unit_by_id = {unit.unit_id: unit for unit in available_units}
    raw_unit_id = str(mention.get("unit_id") or "").strip()
    if raw_unit_id:
        unit = unit_by_id.get(raw_unit_id)
        if unit is None:
            return None, "invalid_unit_id"
    else:
        unit = TextUnit(chunk.chunk_id, chunk.start, chunk.end, chunk.text)

    occurrences = _quote_occurrences(unit.text, quote)
    if not occurrences:
        return None, "quote_not_found"

    raw_occurrence_index = mention.get("occurrence_index")
    if raw_occurrence_index is not None:
        try:
            occurrence_index = int(raw_occurrence_index)
        except (TypeError, ValueError):
            return None, "invalid_occurrence_index"
        if occurrence_index < 0 or occurrence_index >= len(occurrences):
            return None, "occurrence_out_of_range"
    else:
        occurrence_key = (unit.unit_id, normalize_key(quote))
        consumed = (used_occurrences or {}).setdefault(occurrence_key, set())
        occurrence_index = next((index for index in range(len(occurrences)) if index not in consumed), 0)

    if used_occurrences is not None:
        occurrence_key = (unit.unit_id, normalize_key(quote))
        used_occurrences.setdefault(occurrence_key, set()).add(occurrence_index)

    local_start, local_end = occurrences[occurrence_index]
    aligned = (unit.start + local_start, unit.start + local_end)
    start, end = aligned
    start, end, span_text = trim_span_text(text, start, end)
    if not span_text or text[start:end] != span_text:
        return None, "invalid_aligned_span"
    return SpanCandidate(
        start,
        end,
        span_text,
        concept_type,
        _confidence(mention.get("confidence")),
        source="llm",
    ), None


def _coerce_compact_mention(mention: Any) -> Any:
    if not isinstance(mention, list) or len(mention) < 2:
        return mention
    return {
        "quote": mention[0],
        "type": mention[1],
        "occurrence_index": mention[2] if len(mention) > 2 else 0,
    }


def _chunk_units(chunk: TextChunk) -> list[TextUnit]:
    units: list[TextUnit] = []
    local_offset = 0
    for raw_line in chunk.text.splitlines(keepends=True):
        line_start = local_offset
        local_offset += len(raw_line)
        leading = len(raw_line) - len(raw_line.lstrip())
        trailing = len(raw_line.rstrip())
        start = line_start + leading
        end = line_start + trailing
        if start >= end:
            continue
        units.append(
            TextUnit(
                unit_id=f"{chunk.chunk_id}u{len(units) + 1}",
                start=chunk.start + start,
                end=chunk.start + end,
                text=chunk.text[start:end],
            )
        )
    if units:
        return units
    stripped = chunk.text.strip()
    if not stripped:
        return []
    leading = len(chunk.text) - len(chunk.text.lstrip())
    start = chunk.start + leading
    return [TextUnit(f"{chunk.chunk_id}u1", start, start + len(stripped), stripped)]


def _quote_occurrences(source: str, quote: str) -> list[tuple[int, int]]:
    direct = _literal_occurrences(source, quote)
    if direct:
        return direct

    insensitive = _literal_occurrences(source.casefold(), quote.casefold())
    if insensitive:
        return insensitive

    target = normalize_key(quote)
    if not target:
        return []
    quote_len = len(quote)
    min_len = max(1, quote_len - 8)
    max_len = min(len(source), quote_len + 16)
    matches: list[tuple[int, int]] = []
    for start in range(0, len(source)):
        for end in range(min(len(source), start + max_len), start + min_len - 1, -1):
            if normalize_key(source[start:end]) == target:
                matches.append((start, end))
                break
    return matches


def _literal_occurrences(source: str, quote: str) -> list[tuple[int, int]]:
    if not quote:
        return []
    matches: list[tuple[int, int]] = []
    cursor = 0
    while cursor <= len(source) - len(quote):
        start = source.find(quote, cursor)
        if start < 0:
            break
        matches.append((start, start + len(quote)))
        cursor = start + max(1, len(quote))
    return matches


def _align_normalized_quote(chunk: TextChunk, quote: str) -> tuple[int, int] | None:
    target = normalize_key(quote)
    if not target:
        return None
    quote_len = len(quote)
    min_len = max(1, quote_len - 8)
    max_len = min(len(chunk.text), quote_len + 16)
    for start in range(0, len(chunk.text)):
        for end in range(min(len(chunk.text), start + max_len), start + min_len - 1, -1):
            candidate = chunk.text[start:end]
            if normalize_key(candidate) == target:
                return (chunk.start + start, chunk.start + end)
    return None


def _confidence(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.75
    return max(0.0, min(1.0, score))


def _dedup_spans(spans: list[SpanCandidate]) -> list[SpanCandidate]:
    best: dict[tuple[int, int, str], SpanCandidate] = {}
    for span in spans:
        key = (span.start, span.end, span.type)
        current = best.get(key)
        if current is None or span.score > current.score:
            best[key] = span
    return sorted(best.values(), key=lambda item: (item.start, item.end, item.type))
