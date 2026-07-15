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


MAJOR_SECTION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pre_admission", ("tien su benh", "lich su benh", "past medical history", "pmh")),
    (
        "present_illness",
        (
            "tien su benh hien tai",
            "benh su hien tai",
            "lich su benh hien tai",
            "benh su",
            "history of present illness",
        ),
    ),
    (
        "hospital_evaluation",
        (
            "danh gia tai benh vien",
            "kham tai benh vien",
            "kham tai vien",
            "luc vao vien",
            "assessment",
        ),
    ),
)

FALLBACK_SECTION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "pre_admission",
        (
            "benh ly man tinh",
            "cac benh ly man tinh",
            "tien su phau thuat",
            "thuoc truoc khi nhap vien",
            "thuoc dang dieu tri",
            "home medication",
        ),
    ),
    (
        "present_illness",
        (
            "ly do nhap vien",
            "ly do vao vien",
            "trieu chung hien tai",
            "cac trieu chung hien tai",
            "dien bien benh",
            "tinh trang luc vao vien",
            "tinh trang ngay truoc khi nhap vien",
        ),
    ),
    (
        "hospital_evaluation",
        (
            "ket qua xet nghiem",
            "ket qua phong thi nghiem",
            "can lam sang",
            "ket qua chan doan hinh anh",
            "chan doan hinh anh",
            "thu thuat da thuc hien",
            "thu thuat thuc hien",
            "cac phat hien chan doan khac",
            "cac ket qua chan doan khac",
            "chan doan",
            "dieu tri",
            "xu tri thuoc",
        ),
    ),
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


def split_chunks(text: str, max_chars: int = 1000, overlap: int = 0) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section in detect_sections(text):
        for chunk_start, chunk_end in _semantic_section_spans(text, section.start, section.end, max_chars):
            start, end = _trim_offsets(text, chunk_start, chunk_end)
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
    return chunks or [TextChunk("c1", "document", 0, len(text), text)]


def _semantic_section_spans(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]

    blocks = _line_blocks(text, start, end, max_chars)
    spans: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    for block_start, block_end in blocks:
        if block_end - block_start > max_chars:
            if chunk_start is not None and chunk_end is not None:
                spans.append((chunk_start, chunk_end))
                chunk_start = None
                chunk_end = None
            spans.extend(_split_long_span(text, block_start, block_end, max_chars))
            continue
        if chunk_start is None:
            chunk_start, chunk_end = block_start, block_end
            continue
        assert chunk_end is not None
        if block_end - chunk_start <= max_chars:
            chunk_end = block_end
        else:
            spans.append((chunk_start, chunk_end))
            chunk_start, chunk_end = block_start, block_end
    if chunk_start is not None and chunk_end is not None:
        spans.append((chunk_start, chunk_end))
    return spans


def _line_blocks(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end: int | None = None
    block_has_bullet = False
    offset = start
    for line in text[start:end].splitlines(keepends=True):
        line_start = offset
        line_end = offset + len(line)
        offset = line_end
        stripped = line.strip()
        if not stripped:
            if block_start is not None:
                block_end = line_end
            continue

        is_bullet = _is_bullet_line(stripped)
        starts_new_group = (
            block_start is not None
            and (
                (not is_bullet and block_has_bullet)
                or _is_bullet_group_heading(stripped)
                or (block_end is not None and line_end - block_start > max_chars)
            )
        )
        if starts_new_group:
            assert block_end is not None
            blocks.append((block_start, block_end))
            block_start = line_start
            block_has_bullet = False
        elif block_start is None:
            block_start = line_start

        block_end = line_end
        block_has_bullet = block_has_bullet or is_bullet

    if block_start is not None and block_end is not None:
        blocks.append((block_start, block_end))
    return blocks or [(start, end)]


def _split_long_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        raw_end = min(cursor + max_chars, end)
        cut = _choose_chunk_end(text, cursor, raw_end, end)
        if cut <= cursor:
            cut = raw_end
        spans.append((cursor, cut))
        cursor = cut
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return spans


def _is_bullet_line(stripped: str) -> bool:
    return stripped.startswith(("-", "*", "•"))


def _is_bullet_group_heading(stripped: str) -> bool:
    if not _is_bullet_line(stripped):
        return False
    marker_stripped = stripped.lstrip("-*• ").strip()
    key = normalize_key(marker_stripped.strip("*: "))
    if not key or len(key) > 90:
        return False
    detail_prefixes = (
        "vi tri",
        "muc do",
        "thoi gian",
        "tan suat",
        "chieu xa",
        "lan toa",
        "yeu to",
        "cac yeu to",
        "trieu chung lien quan",
        "cac trieu chung",
        "khong",
        "cam thay",
        "duoc",
        "da",
        "bat dau",
        "keo dai",
        "xuat hien",
    )
    if any(key.startswith(prefix) for prefix in detail_prefixes):
        return False
    return marker_stripped.startswith("**") or marker_stripped.endswith(":")


def _section_hits(text: str) -> list[tuple[int, str]]:
    major_hits: list[tuple[int, str]] = []
    fallback_hits: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip(" \t\r\n:-*")
        key = normalize_key(stripped)
        if key:
            major = _major_section_name(key)
            if major is not None:
                major_hits.append((offset + line.find(stripped), major))
            else:
                fallback = _fallback_section_name(key)
                if fallback is not None:
                    fallback_hits.append((offset + line.find(stripped), fallback))
        offset += len(line)

    hits = major_hits if len(major_hits) >= 2 else fallback_hits
    dedup: list[tuple[int, str]] = []
    seen_starts: set[int] = set()
    for start, name in sorted(hits):
        if start in seen_starts:
            continue
        seen_starts.add(start)
        dedup.append((start, name))
    if dedup and dedup[0][0] > 0:
        dedup.insert(0, (0, "document"))
    return dedup


def _major_section_name(line_key: str) -> str | None:
    numbered = line_key.split(maxsplit=1)
    if len(numbered) != 2 or not numbered[0].rstrip(".").isdigit():
        return None
    number = numbered[0].rstrip(".")
    title = numbered[1]
    if number == "1" and _starts_with_any(title, MAJOR_SECTION_MARKERS[0][1]):
        return "pre_admission"
    if number == "2" and _starts_with_any(title, MAJOR_SECTION_MARKERS[1][1]):
        return "present_illness"
    if number == "3" and _starts_with_any(title, MAJOR_SECTION_MARKERS[2][1]):
        return "hospital_evaluation"
    return None


def _fallback_section_name(line_key: str) -> str | None:
    if len(line_key) > 140:
        return None
    for name, markers in FALLBACK_SECTION_MARKERS:
        if _starts_with_any(line_key, markers):
            return name
    return None


def _starts_with_any(line_key: str, markers: tuple[str, ...]) -> bool:
    return any(
        line_key == marker
        or line_key.startswith(marker + " ")
        or line_key.startswith(marker + ":")
        for marker in markers
    )


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
