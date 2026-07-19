"""Context-aware evidence learned from reviewed annotation corpora."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.config import ALLOWED_TYPES
from core.text import normalize_key, trim_span_text
from extraction.ner import SpanCandidate
from extraction.sectioning import Section, detect_sections


_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    key: str
    surfaces: tuple[str, ...]
    concept_type: str
    positive_count: int
    observed_count: int
    support_documents: int
    observed_documents: int
    type_purity: float
    annotation_rate: float
    sections: dict[str, tuple[int, int]]
    left_cues: tuple[str, ...]
    right_cues: tuple[str, ...]
    assertions: dict[str, int]
    candidates: dict[str, int]
    candidate_sets: dict[str, int]

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> MemoryEntry | None:
        key = normalize_key(str(item.get("key") or ""))
        concept_type = str(item.get("type") or "")
        if not key or concept_type not in ALLOWED_TYPES:
            return None
        sections: dict[str, tuple[int, int]] = {}
        raw_sections = item.get("sections") or {}
        if isinstance(raw_sections, dict):
            for name, counts in raw_sections.items():
                if not isinstance(counts, dict):
                    continue
                sections[str(name)] = (
                    _nonnegative_int(counts.get("positive")),
                    _nonnegative_int(counts.get("observed")),
                )
        return cls(
            key=key,
            surfaces=tuple(str(value) for value in item.get("surfaces") or () if str(value)),
            concept_type=concept_type,
            positive_count=_nonnegative_int(item.get("positive_count")),
            observed_count=_nonnegative_int(item.get("observed_count")),
            support_documents=_nonnegative_int(item.get("support_documents")),
            observed_documents=_nonnegative_int(item.get("observed_documents")),
            type_purity=_probability(item.get("type_purity")),
            annotation_rate=_probability(item.get("annotation_rate")),
            sections=sections,
            left_cues=tuple(str(value) for value in item.get("left_cues") or () if str(value)),
            right_cues=tuple(str(value) for value in item.get("right_cues") or () if str(value)),
            assertions={str(key): _nonnegative_int(value) for key, value in (item.get("assertions") or {}).items()},
            candidates={str(key): _nonnegative_int(value) for key, value in (item.get("candidates") or {}).items()},
            candidate_sets={
                str(key): _nonnegative_int(value) for key, value in (item.get("candidate_sets") or {}).items()
            },
        )

    def section_rate(self, section: str) -> float | None:
        positive, observed = self.sections.get(section, (0, 0))
        if observed < 2:
            return None
        return (positive + 1.0) / (observed + 2.0)

    def proposal_score(self, text: str, start: int, end: int, section: str) -> float:
        global_rate = (self.positive_count + 2.0) / (self.observed_count + 4.0)
        section_rate = self.section_rate(section)
        contextual_rate = section_rate if section_rate is not None else global_rate
        cue_score = _cue_agreement(text, start, end, self.left_cues, self.right_cues)
        score = 0.45 * global_rate + 0.25 * contextual_rate + 0.2 * self.type_purity + 0.1 * cue_score
        return max(0.0, min(0.99, score))

    def can_propose(self, section: str) -> bool:
        if self.positive_count < 3 or self.support_documents < 2 or self.type_purity < 0.85:
            return False
        section_rate = self.section_rate(section)
        evidence_rate = section_rate if section_rate is not None else self.annotation_rate
        if len(self.key.split()) == 1:
            return self.positive_count >= 8 and evidence_rate >= 0.8
        return evidence_rate >= 0.62


class AnnotationMemory:
    def __init__(self, entries: Iterable[MemoryEntry] = ()) -> None:
        self.entries = tuple(entries)
        self._entries_by_key: dict[str, list[MemoryEntry]] = {}
        for entry in self.entries:
            self._entries_by_key.setdefault(entry.key, []).append(entry)
        self._matcher = NormalizedKeyMatcher(self._entries_by_key)

    @classmethod
    def empty(cls) -> AnnotationMemory:
        return cls(())

    @classmethod
    def load(cls, path: Path) -> AnnotationMemory:
        if not path.exists():
            return cls.empty()
        entries: list[MemoryEntry] = []
        with path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    entry = MemoryEntry.from_dict(item)
                    if entry is not None:
                        entries.append(entry)
        return cls(entries)

    def propose(self, text: str) -> list[SpanCandidate]:
        if not text or not self.entries:
            return []
        projection = normalized_projection(text)
        sections = detect_sections(text)
        proposals: list[SpanCandidate] = []
        for key, start, end in self._matcher.find(projection):
            for entry in self._entries_by_key[key]:
                section = section_at(sections, start)
                if not entry.can_propose(section):
                    continue
                score = entry.proposal_score(text, start, end, section)
                if score < 0.7:
                    continue
                start, end, span_text = trim_span_text(text, start, end)
                if span_text:
                    proposals.append(
                        SpanCandidate(start, end, span_text, entry.concept_type, score, source="memory")
                    )
        return proposals

    def evidence_for(self, text: str, span: SpanCandidate) -> tuple[MemoryEntry, float] | None:
        key = normalize_key(span.text)
        section = section_at(detect_sections(text), span.start)
        matching = [entry for entry in self._entries_by_key.get(key, ()) if entry.concept_type == span.type]
        if not matching:
            return None
        entry = max(matching, key=lambda value: (value.positive_count, value.type_purity))
        return entry, entry.proposal_score(text, span.start, span.end, section)

    def evidence_for_section(self, span: SpanCandidate, section: str, text: str) -> tuple[MemoryEntry, float] | None:
        key = normalize_key(span.text)
        matching = [entry for entry in self._entries_by_key.get(key, ()) if entry.concept_type == span.type]
        if not matching:
            return None
        entry = max(matching, key=lambda value: (value.positive_count, value.type_purity))
        return entry, entry.proposal_score(text, span.start, span.end, section)

    def candidate_decision(self, text: str, concept_type: str) -> tuple[str, ...] | None:
        key = normalize_key(text)
        matching = [entry for entry in self._entries_by_key.get(key, ()) if entry.concept_type == concept_type]
        if not matching:
            return None
        entry = max(matching, key=lambda value: (value.positive_count, value.type_purity))
        total = sum(entry.candidate_sets.values())
        if total < 3 or entry.support_documents < 2:
            return None
        null_count = entry.candidate_sets.get("", 0)
        nonempty_count = total - null_count
        nonempty_rate = nonempty_count / total
        if null_count >= 3 and nonempty_rate <= 0.2:
            return ()
        nonempty_sets = [(value, count) for value, count in entry.candidate_sets.items() if value]
        if not nonempty_sets:
            return None
        value, support = max(nonempty_sets, key=lambda item: (item[1], item[0]))
        if support < 3 or support / total < 0.6 or nonempty_rate < 0.5:
            return None
        return tuple(code for code in value.split("|") if code)


@dataclass(frozen=True, slots=True)
class NormalizedProjection:
    text: str
    offsets: tuple[int, ...]
    source: str

    def find(self, key: str) -> list[tuple[int, int]]:
        key = normalize_key(key)
        if not key or not self.text:
            return []
        output: list[tuple[int, int]] = []
        cursor = 0
        while True:
            match_start = self.text.find(key, cursor)
            if match_start < 0:
                break
            match_end = match_start + len(key)
            left_ok = match_start == 0 or not _is_word_char(self.text[match_start - 1])
            right_ok = match_end == len(self.text) or not _is_word_char(self.text[match_end])
            if left_ok and right_ok:
                start = self.offsets[match_start]
                end = self.offsets[match_end - 1] + 1
                output.append((start, end))
            cursor = match_start + 1
        return output


class NormalizedKeyMatcher:
    _END = "\0"

    def __init__(self, keys: Iterable[str]) -> None:
        self.root: dict[str, Any] = {}
        for raw_key in keys:
            key = normalize_key(raw_key)
            if not key:
                continue
            node = self.root
            for char in key:
                node = node.setdefault(char, {})
            node.setdefault(self._END, []).append(key)

    def find(self, projection: NormalizedProjection) -> list[tuple[str, int, int]]:
        normalized = projection.text
        output: list[tuple[str, int, int]] = []
        for start_index, char in enumerate(normalized):
            if start_index > 0 and _is_word_char(normalized[start_index - 1]):
                continue
            node = self.root.get(char)
            if node is None:
                continue
            cursor = start_index + 1
            while True:
                keys = node.get(self._END, ())
                if keys and (cursor == len(normalized) or not _is_word_char(normalized[cursor])):
                    source_start = projection.offsets[start_index]
                    source_end = projection.offsets[cursor - 1] + 1
                    output.extend((key, source_start, source_end) for key in keys)
                if cursor >= len(normalized):
                    break
                node = node.get(normalized[cursor])
                if node is None:
                    break
                cursor += 1
        return output


def normalized_projection(text: str) -> NormalizedProjection:
    normalized: list[str] = []
    offsets: list[int] = []
    previous_space = True
    for index, char in enumerate(text):
        folded = normalize_key(char)
        if not folded:
            folded = " "
        for output_char in folded:
            is_space = output_char.isspace()
            if is_space and previous_space:
                continue
            normalized.append(" " if is_space else output_char)
            offsets.append(index)
            previous_space = is_space
    if normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return NormalizedProjection("".join(normalized), tuple(offsets), text)


def section_at(sections: Iterable[Section], offset: int) -> str:
    for section in sections:
        if section.start <= offset < section.end:
            return section.name
    return "document"


def context_cues(text: str, start: int, end: int, width: int = 3) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left = normalize_key(text[max(0, start - 120) : start])
    right = normalize_key(text[end : min(len(text), end + 120)])
    left_words = _WORD_RE.findall(left)
    right_words = _WORD_RE.findall(right)
    return tuple(left_words[-width:]), tuple(right_words[:width])


def _cue_agreement(
    text: str,
    start: int,
    end: int,
    left_cues: tuple[str, ...],
    right_cues: tuple[str, ...],
) -> float:
    left, right = context_cues(text, start, end)
    expected = set(left_cues) | set(right_cues)
    if not expected:
        return 0.5
    observed = set(left) | set(right)
    return len(expected & observed) / len(expected)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _probability(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"
