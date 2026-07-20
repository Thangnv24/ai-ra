"""Section and chunk helpers with stable source offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

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
    subsection: str = "document"


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

SUBSECTION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "medications",
        (
            "thuoc truoc khi nhap vien",
            "thuoc truoc nhap vien",
            "thuoc dang dieu tri",
            "thuoc hien tai",
            "don thuoc",
            "dieu tri thuoc",
            "xu tri thuoc",
            "home medication",
            "medications",
        ),
    ),
    (
        "laboratory",
        (
            "ket qua xet nghiem",
            "ket qua phong thi nghiem",
            "xet nghiem",
            "cong thuc mau",
            "sinh hoa mau",
            "laboratory",
            "lab results",
        ),
    ),
    (
        "imaging_procedure",
        (
            "chan doan hinh anh",
            "ket qua chan doan hinh anh",
            "thu thuat da thuc hien",
            "thu thuat thuc hien",
            "noi soi",
            "sieu am",
            "imaging",
        ),
    ),
    (
        "diagnoses",
        (
            "chan doan",
            "chan doan ra vien",
            "benh kem theo",
            "problem list",
        ),
    ),
    (
        "symptoms_exam",
        (
            "trieu chung",
            "trieu chung hien tai",
            "ly do nhap vien",
            "ly do vao vien",
            "kham lam sang",
            "kham benh",
            "physical examination",
        ),
    ),
    (
        "history",
        (
            "tien su",
            "tien su benh",
            "tien su gia dinh",
            "tien su phau thuat",
            "past medical history",
            "family history",
        ),
    ),
    (
        "vital_signs",
        (
            "dau hieu sinh ton",
            "sinh hieu",
            "vital signs",
        ),
    ),
    (
        "exposure_poisoning",
        (
            "ngo doc",
            "phoi nhiem",
            "tiep xuc hoa chat",
            "chat doc",
            "poisoning",
            "exposure",
        ),
    ),
)

_CASE_ORDINALS = (
    "nhat",
    "hai",
    "ba",
    "tu",
    "nam",
    "sau",
    "bay",
    "tam",
    "chin",
    "muoi",
)
_CASE_NOUNS = ("ho so", "truong hop", "ca benh", "benh nhan")
_ADMIN_LINE_PREFIXES = (
    "phieu ban giao",
    "cong hoa xa hoi chu nghia viet nam",
    "bao cao tong hop",
    "tom tat danh sach",
    "ngay xuat",
    "nguoi xuat",
    "ca truc tu",
    "nguoi ban giao",
    "nguoi nhan",
    "ma so ho so",
    "thoi diem tiep nhan",
    "noi dung chi tiet",
    "phan i",
    "phan ii",
    "phan iii",
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CLAUSE_RE = re.compile(r"[,;:\n]|\b(?:va|hoac|khong|phu nhan)\b")
_TARGET_EXTRACTION_LOAD = 22


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


def detect_subsections(text: str) -> list[Section]:
    hits = _subsection_hits(text)
    if not hits:
        return [Section("document", 0, len(text))]
    if hits[0][0] > 0:
        hits.insert(0, (0, "document"))
    return [
        Section(name, start, hits[index + 1][0] if index + 1 < len(hits) else len(text))
        for index, (start, name) in enumerate(hits)
        if start < (hits[index + 1][0] if index + 1 < len(hits) else len(text))
    ]


def split_chunks(text: str, max_chars: int = 480, overlap: int = 100) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    semantic_sections = detect_sections(text)
    subsections = detect_subsections(text)
    for case_index, (case_start, case_end) in enumerate(_case_spans(text), start=1):
        for segment_start, segment_end in _usable_segments(text, case_start, case_end):
            for structure_start, structure_end in _structure_spans(
                segment_start,
                segment_end,
                semantic_sections,
                subsections,
            ):
                for chunk_start, chunk_end in _semantic_section_spans(
                    text, structure_start, structure_end, max_chars
                ):
                    start, end = _expand_with_overlap(
                        text,
                        chunk_start,
                        chunk_end,
                        structure_start,
                        structure_end,
                        overlap,
                    )
                    start, end = _trim_offsets(text, start, end)
                    if start >= end or _is_administrative_only(text[start:end]):
                        continue
                    section_name = _section_name_at(semantic_sections, start)
                    subsection_name = _section_name_at(subsections, start)
                    chunks.append(
                        TextChunk(
                            chunk_id=f"c{len(chunks) + 1}",
                            section=f"case_{case_index}:{section_name}",
                            start=start,
                            end=end,
                            text=text[start:end],
                            subsection=subsection_name,
                        )
                    )
    return chunks or [TextChunk("c1", "document", 0, len(text), text)]


def _structure_spans(
    start: int,
    end: int,
    sections: list[Section],
    subsections: list[Section],
) -> list[tuple[int, int]]:
    boundaries = {start, end}
    for item in (*sections, *subsections):
        if start < item.start < end:
            boundaries.add(item.start)
        if start < item.end < end:
            boundaries.add(item.end)
    ordered = sorted(boundaries)
    return [
        (left, right)
        for left, right in zip(ordered, ordered[1:])
        if left < right
    ]


def _expand_with_overlap(
    text: str,
    start: int,
    end: int,
    scope_start: int,
    scope_end: int,
    overlap: int,
) -> tuple[int, int]:
    if overlap <= 0:
        return start, end
    expanded_start = max(scope_start, start - overlap)
    expanded_end = min(scope_end, end + overlap)
    if expanded_start < start:
        window = text[expanded_start:start]
        cut = max(window.find("\n"), window.find(". "))
        if cut >= 0:
            expanded_start += cut + 1
    if end < expanded_end:
        window = text[end:expanded_end]
        cuts = [index + len(separator) for separator in ("\n", ". ") if (index := window.find(separator)) >= 0]
        if cuts:
            expanded_end = end + min(cuts)
    return expanded_start, expanded_end


def _case_spans(text: str) -> list[tuple[int, int]]:
    return list(_cached_case_spans(text))


@lru_cache(maxsize=128)
def _cached_case_spans(text: str) -> tuple[tuple[int, int], ...]:
    hits = _case_boundary_hits(text)
    if not hits:
        return ((0, len(text)),)
    starts = [0] if hits[0] > 0 else []
    starts.extend(hits)
    starts = sorted(set(starts))
    return tuple(
        (start, starts[index + 1] if index + 1 < len(starts) else len(text))
        for index, start in enumerate(starts)
    )


def case_bounds_at(text: str, offset: int) -> tuple[int, int]:
    for start, end in _cached_case_spans(text):
        if start <= offset < end:
            return start, end
    return 0, len(text)


def _case_boundary_hits(text: str) -> list[int]:
    hits: list[int] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.strip()
        key = normalize_key(stripped)
        if key and len(key) <= 190 and _is_case_boundary_line(stripped, key):
            start = offset + max(0, raw_line.find(stripped))
            if not hits or not _is_administrative_only(text[hits[-1] : start]):
                hits.append(start)
        offset += len(raw_line)
    return hits


def _is_case_boundary_line(raw_line: str, key: str) -> bool:
    if key.startswith("ma so ho so"):
        return True
    if re.match(r"^\s*\d+\.\d+\.?\s+", raw_line) and any(noun in key for noun in _CASE_NOUNS):
        return True
    return any(f"{noun} thu {ordinal}" in key for noun in _CASE_NOUNS for ordinal in _CASE_ORDINALS)


def _usable_segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    segment_start = start
    offset = start
    for raw_line in text[start:end].splitlines(keepends=True):
        line_end = offset + len(raw_line)
        repeat_start = _pathological_repeat_start(raw_line)
        if repeat_start is not None:
            absolute_repeat = offset + repeat_start
            if segment_start < absolute_repeat:
                segments.append((segment_start, absolute_repeat))
            segment_start = line_end
        offset = line_end
    if segment_start < end:
        segments.append((segment_start, end))
    return segments or [(start, end)]


def _pathological_repeat_start(line: str, threshold: int = 20) -> int | None:
    previous = ""
    run_start = 0
    run_length = 0
    for match in _WORD_RE.finditer(line):
        token = normalize_key(match.group(0))
        if token and token == previous:
            run_length += 1
        else:
            previous = token
            run_start = match.start()
            run_length = 1
        if run_length >= threshold:
            return run_start
    return None


def _is_administrative_only(value: str) -> bool:
    lines = [normalize_key(line) for line in value.splitlines() if normalize_key(line)]
    if not lines:
        return True
    for line in lines:
        if not line.strip("._- "):
            continue
        if any(line.startswith(prefix) for prefix in _ADMIN_LINE_PREFIXES):
            continue
        if _is_case_boundary_line(line, line):
            continue
        return False
    return True


def _section_name_at(sections: list[Section], offset: int) -> str:
    for section in sections:
        if section.start <= offset < section.end:
            return section.name
    return "document"


def _semantic_section_spans(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]

    blocks = _line_blocks(text, start, end, max_chars)
    spans: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    for block_start, block_end in blocks:
        block_text = text[block_start:block_end]
        block_load = _estimated_extraction_load(block_text)
        if block_end - block_start > max_chars or (block_load > _TARGET_EXTRACTION_LOAD and len(block_text) > 360):
            if chunk_start is not None and chunk_end is not None:
                spans.append((chunk_start, chunk_end))
                chunk_start = None
                chunk_end = None
            dense_max_chars = min(
                max_chars,
                max(360, int(max_chars * _TARGET_EXTRACTION_LOAD / max(1, block_load))),
            )
            spans.extend(_split_long_span(text, block_start, block_end, dense_max_chars))
            continue
        if chunk_start is None:
            chunk_start, chunk_end = block_start, block_end
            continue
        assert chunk_end is not None
        combined_text = text[chunk_start:block_end]
        if (
            block_end - chunk_start <= max_chars
            and _estimated_extraction_load(combined_text) <= _TARGET_EXTRACTION_LOAD
        ):
            chunk_end = block_end
        else:
            spans.append((chunk_start, chunk_end))
            chunk_start, chunk_end = block_start, block_end
    if chunk_start is not None and chunk_end is not None:
        spans.append((chunk_start, chunk_end))
    return spans


def _estimated_extraction_load(value: str) -> int:
    key = normalize_key(value)
    structural_cues = len(_CLAUSE_RE.findall(key))
    nonempty_lines = sum(bool(line.strip()) for line in value.splitlines())
    return max(1, structural_cues + max(0, nonempty_lines - 1), len(key) // 220)


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


def _subsection_hits(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.strip(" \t\r\n:-*#")
        key = normalize_key(stripped)
        if key and len(key) <= 140:
            for name, markers in SUBSECTION_MARKERS:
                if _starts_with_any(key, markers):
                    local_start = raw_line.find(stripped)
                    hits.append((offset + max(0, local_start), name))
                    break
        offset += len(raw_line)
    dedup: list[tuple[int, str]] = []
    for start, name in hits:
        if dedup and dedup[-1] == (start, name):
            continue
        dedup.append((start, name))
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
