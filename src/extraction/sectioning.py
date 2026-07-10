"""Section and chunk helpers with stable source offsets."""

from __future__ import annotations

from dataclasses import dataclass

from core.text import normalize_key


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    start: int
    end: int

    @property
    def text_span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    section: str
    start: int
    end: int
    text: str


SECTION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("history", ("tien su benh noi khoa", "tien su", "past medical history", "pmh")),
    ("present_illness", ("tien su benh hien tai", "benh su hien tai", "history of present illness")),
    ("symptoms", ("trieu chung chinh", "cac trieu chung lien quan", "ly do nhap vien")),
    ("assessment", ("danh gia tai benh vien", "chan doan", "assessment", "diagnosis")),
    ("labs", ("ket qua xet nghiem", "xet nghiem", "can lam sang", "laboratory", "labs")),
    ("imaging", ("ket qua chan doan hinh anh", "chan doan hinh anh", "sieu am", "ct", "x quang")),
    ("procedures", ("thu thuat da thuc hien", "cac thu thuat da thuc hien", "procedure")),
    ("medications", ("danh sach thuoc", "thuoc truoc nhap vien", "su dung thuoc", "home medication")),
)


def detect_sections(text: str) -> list[Section]:
    hits = _section_hits(text)
    if not hits:
        return [Section("document", 0, len(text))]

    sections: list[Section] = []
    for index, (start, name) in enumerate(hits):
        end = hits[index + 1][0] if index + 1 < len(hits) else len(text)
        if start < end:
            sections.append(Section(name, start, end))
    return sections or [Section("document", 0, len(text))]


def split_chunks(text: str, max_chars: int = 1800, overlap: int = 160) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section in detect_sections(text):
        cursor = section.start
        section_end = section.end
        while cursor < section_end:
            raw_end = min(section_end, cursor + max_chars)
            end = _choose_chunk_end(text, cursor, raw_end, section_end)
            start, end = _trim_offsets(text, cursor, end)
            if start < end:
                chunks.append(
                    TextChunk(
                        chunk_id=f"c{len(chunks) + 1}",
                        section=section.name,
                        start=start,
                        end=end,
                        text=text[start:end],
                    )
                )
            if end >= section_end:
                break
            next_cursor = max(section.start, end - max(0, overlap))
            if next_cursor <= cursor:
                next_cursor = end
            cursor = next_cursor
    return chunks or [TextChunk("c1", "document", 0, len(text), text)]


def _section_hits(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip(" \t\r\n:-")
        key = normalize_key(stripped)
        if key:
            for name, markers in SECTION_MARKERS:
                if any(_is_section_heading(key, marker) for marker in markers):
                    hits.append((offset + line.find(stripped), name))
                    break
        offset += len(line)

    dedup: list[tuple[int, str]] = []
    seen_starts: set[int] = set()
    for start, name in sorted(hits):
        if start in seen_starts:
            continue
        seen_starts.add(start)
        dedup.append((start, name))
    return dedup


def _is_section_heading(line_key: str, marker: str) -> bool:
    if line_key == marker:
        return True
    if line_key.startswith(marker + " "):
        return True
    if line_key.startswith(marker + ":"):
        return True
    numbered = line_key.split(maxsplit=1)
    return len(numbered) == 2 and numbered[0].rstrip(".").isdigit() and numbered[1].startswith(marker)


def _choose_chunk_end(text: str, start: int, raw_end: int, section_end: int) -> int:
    if raw_end >= section_end:
        return section_end
    window = text[start:raw_end]
    min_cut = max(240, len(window) // 2)
    best = -1
    for separator in ("\n\n", "\n", ". ", "; "):
        idx = window.rfind(separator)
        if idx >= min_cut:
            best = max(best, idx + len(separator))
    return start + best if best > 0 else raw_end


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
