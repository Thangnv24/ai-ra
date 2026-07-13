"""LLM-assisted entity proposal with deterministic quote alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import ALLOWED_TYPES
from core.text import normalize_key, trim_span_text
from extraction.ner import SpanCandidate
from extraction.sectioning import TextChunk, split_chunks
from integrations.prompts import ENTITY_SYSTEM_PROMPT, build_entity_extraction_prompt


@dataclass(frozen=True, slots=True)
class EntityProposalSummary:
    chunks: int
    mentions: int
    aligned: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "mentions": self.mentions,
            "aligned": self.aligned,
            "errors": list(self.errors),
        }


class LLMEntityExtractor:
    def __init__(self, llm_client: Any, max_chars: int = 1800, overlap: int = 160) -> None:
        self.llm_client = llm_client
        self.max_chars = max_chars
        self.overlap = overlap

    def extract(self, text: str) -> tuple[list[SpanCandidate], EntityProposalSummary]:
        if not getattr(self.llm_client, "enabled", False):
            raise RuntimeError("LLM entity extraction is required but the LLM client is disabled")

        chunks = split_chunks(text, max_chars=self.max_chars, overlap=self.overlap)
        spans: list[SpanCandidate] = []
        mention_count = 0
        for chunk in chunks:
            result = self.llm_client.chat_json(
                ENTITY_SYSTEM_PROMPT,
                build_entity_extraction_prompt(_chunk_payload(chunk)),
            )
            if not result.ok or not isinstance(result.data, dict):
                raise RuntimeError(
                    f"LLM entity extraction failed for {chunk.chunk_id}: "
                    f"{result.error or 'LLM returned no data'}"
                )
            mentions = result.data.get("mentions")
            if not isinstance(mentions, list):
                raise RuntimeError(f"LLM entity extraction failed for {chunk.chunk_id}: JSON has no mentions list")
            mention_count += len(mentions)
            for mention in mentions:
                span = _span_from_mention(text, chunk, mention)
                if span is not None:
                    spans.append(span)
        spans = _dedup_spans(spans)
        return spans, EntityProposalSummary(
            chunks=len(chunks),
            mentions=mention_count,
            aligned=len(spans),
        )


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


def _chunk_payload(chunk: TextChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
    }


def _span_from_mention(text: str, chunk: TextChunk, mention: Any) -> SpanCandidate | None:
    if not isinstance(mention, dict):
        return None
    quote = str(mention.get("quote") or "").strip()
    concept_type = str(mention.get("type") or "").strip()
    if concept_type not in ALLOWED_TYPES:
        return None
    aligned = align_quote_in_chunk(chunk, quote)
    if aligned is None:
        return None
    start, end = aligned
    start, end, span_text = trim_span_text(text, start, end)
    if not span_text or text[start:end] != span_text:
        return None
    return SpanCandidate(start, end, span_text, concept_type, _confidence(mention.get("confidence")))


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
