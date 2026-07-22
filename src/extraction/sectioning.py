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
    structure_role: str = "document"
    context_scope: str = ""


@dataclass(frozen=True, slots=True)
class StructuralBlock:
    role: str
    start: int
    end: int


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

_PRESERVE_REPEATED_ROLE_BOUNDARIES = {
    "question",
    "answer",
    "medical_article",
    "patient_case",
    "personal_history",
    "family_history",
    "epidemiology_history",
}

_QUESTION_MARKERS = (
    "hoi",
    "cau hoi",
    "cau hoi tu nguoi dung",
    "cau hoi cua nguoi dung",
    "question",
)
_ANSWER_MARKERS = (
    "tra loi",
    "cau tra loi",
    "cau tra loi cua bac si",
    "bac si tra loi",
    "dap",
    "answer",
)
_ARTICLE_HEADING_CUES = (
    "la gi",
    "dau hieu",
    "trieu chung",
    "nguyen nhan",
    "bien chung",
    "co nguy hiem",
    "can lam gi",
    "phong ngua",
    "dieu tri",
    "chan doan",
)
_HISTORY_BLOCK_MARKERS = (
    "tien su ban than",
    "tien su gia dinh",
    "tien su dich te",
)
_LAB_BLOCK_MARKERS = (
    "can lam sang",
    "xet nghiem",
    "xn luc vao vien",
    "ket qua xet nghiem",
    "ket qua laboratory",
    "ket qua phong thi nghiem",
    "ket qua chan doan hinh anh",
    "chan doan hinh anh va tham do",
    "dien tam do",
    "men tim",
    "sieu am tim",
)
_DIAGNOSIS_BLOCK_MARKERS = (
    "chan doan",
    "cac chan doan",
)
_TREATMENT_BLOCK_MARKERS = (
    "dieu tri",
    "don thuoc",
    "thuoc dieu tri",
    "xu tri thuoc",
)
_SYMPTOM_PROFILE_MARKERS = (
    "dac diem cua trieu chung",
    "dac diem trieu chung",
    "trieu chung khi nhap vien",
    "trieu chung hien tai",
    "cac trieu chung lien quan",
    "thoi diem khoi phat trieu chung",
    "vi tri",
)
_PRE_ADMISSION_EVENT_MARKERS = (
    "cac su kien truoc khi nhap vien",
    "su kien truoc khi nhap vien",
    "cac dien bien truoc khi nhap vien",
    "dien bien truoc khi nhap vien",
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
    del overlap  # Context overlap is carried separately; extraction targets remain disjoint.
    chunks: list[TextChunk] = []
    semantic_sections = detect_sections(text)
    for case_index, (case_start, case_end) in enumerate(_case_spans(text), start=1):
        blocks = _structural_blocks(text, case_start, case_end, semantic_sections)
        for block_index, block in enumerate(blocks, start=1):
            for segment_index, (segment_start, segment_end) in enumerate(
                _usable_segments(text, block.start, block.end),
                start=1,
            ):
                context_scope = f"case_{case_index}:block_{block_index}:segment_{segment_index}"
                for chunk_start, chunk_end in _semantic_section_spans(
                    text,
                    segment_start,
                    segment_end,
                    max_chars,
                ):
                    start, end = _trim_offsets(text, chunk_start, chunk_end)
                    if start >= end or _is_administrative_only(text[start:end]):
                        continue
                    section_name = _section_name_at(semantic_sections, start)
                    chunks.append(
                        TextChunk(
                            chunk_id=f"c{len(chunks) + 1}",
                            section=f"case_{case_index}:{section_name}",
                            start=start,
                            end=end,
                            text=text[start:end],
                            structure_role=block.role,
                            context_scope=context_scope,
                        )
                    )
    return chunks or [
        TextChunk(
            "c1",
            "document",
            0,
            len(text),
            text,
            structure_role="document",
            context_scope="case_1:block_1:segment_1",
        )
    ]


def _structural_blocks(
    text: str,
    start: int,
    end: int,
    semantic_sections: list[Section],
) -> list[StructuralBlock]:
    hits: dict[int, str] = {}
    for section in semantic_sections:
        if start <= section.start < end:
            hits[section.start] = _role_for_section(section.name)
    for boundary_start, role in _structure_hits(text, start, end):
        hits[boundary_start] = role

    if start not in hits:
        hits[start] = _role_for_section(_section_name_at(semantic_sections, start))

    ordered_hits = _collapse_redundant_structure_hits(sorted(hits.items()))
    starts = [position for position, _ in ordered_hits]
    roles = dict(ordered_hits)
    blocks = [
        StructuralBlock(
            role=roles[block_start],
            start=block_start,
            end=starts[index + 1] if index + 1 < len(starts) else end,
        )
        for index, block_start in enumerate(starts)
        if block_start < end
    ]
    return _merge_heading_only_blocks(text, blocks)


def _collapse_redundant_structure_hits(
    hits: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    collapsed: list[tuple[int, str]] = []
    for position, role in hits:
        if (
            collapsed
            and collapsed[-1][1] == role
            and role not in _PRESERVE_REPEATED_ROLE_BOUNDARIES
        ):
            continue
        collapsed.append((position, role))
    return collapsed


def _structure_hits(text: str, start: int, end: int) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    offset = start
    seen_question = False
    active_role: str | None = None
    for raw_line in text[start:end].splitlines(keepends=True):
        stripped = raw_line.strip()
        key = normalize_key(stripped.strip("*- "))
        role = _structure_role_for_line(stripped, key)
        if (
            seen_question
            and active_role not in {None, "question", "answer"}
            and _looks_like_advice_reentry(key)
        ):
            role = "answer"
        if role is not None:
            line_start = offset + max(0, raw_line.find(stripped))
            hits.append((line_start, role))
            active_role = role
            seen_question = seen_question or role == "question"
        offset += len(raw_line)
    return hits


def _structure_role_for_line(raw_line: str, key: str) -> str | None:
    if not key:
        return None
    if _is_question_line(raw_line, key):
        return "question"
    if _is_answer_line(raw_line, key):
        return "answer"

    major = _major_section_name(key)
    if major is not None:
        return _role_for_section(major)

    if _is_patient_restart(key):
        return "patient_case"
    heading_key = _strip_heading_number(key)
    if _starts_with_any(heading_key, _HISTORY_BLOCK_MARKERS):
        return _history_role(heading_key)
    if _starts_with_any(heading_key, _PRE_ADMISSION_EVENT_MARKERS):
        return "pre_admission_events"
    if _starts_with_any(heading_key, _SYMPTOM_PROFILE_MARKERS):
        return "symptom_profile"
    if _looks_like_heading(raw_line, key) and _starts_with_any(heading_key, _LAB_BLOCK_MARKERS):
        return "lab_or_imaging"
    if _looks_like_heading(raw_line, key) and _starts_with_any(
        heading_key,
        _DIAGNOSIS_BLOCK_MARKERS,
    ):
        return "diagnosis"
    if _looks_like_heading(raw_line, key) and _starts_with_any(heading_key, _TREATMENT_BLOCK_MARKERS):
        return "treatment"
    if _looks_like_article_heading(raw_line, heading_key):
        return "medical_article"
    return None


def _history_role(heading_key: str) -> str:
    if heading_key.startswith("tien su ban than"):
        return "personal_history"
    if heading_key.startswith("tien su gia dinh"):
        return "family_history"
    if heading_key.startswith("tien su dich te"):
        return "epidemiology_history"
    return "medical_history"


def _is_question_line(raw_line: str, key: str) -> bool:
    raw = raw_line.lstrip("*- ").casefold()
    return bool(
        re.match(r"^(?:hỏi|hoi|question)\s*[:：-]", raw)
        or _starts_with_any(key, _QUESTION_MARKERS[1:])
    )


def _is_answer_line(raw_line: str, key: str) -> bool:
    raw = raw_line.lstrip("*- ").casefold()
    return bool(
        re.match(r"^(?:trả lời|tra loi|đáp|dap|answer)\s*[:：-]", raw)
        or _starts_with_any(key, _ANSWER_MARKERS[1:-1])
    )


def _strip_heading_number(key: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*[.)/]?\s*", "", key)


def _looks_like_advice_reentry(key: str) -> bool:
    return bool(
        _starts_with_any(
            key,
            (
                "du hien tai",
                "de theo doi tai nha",
                "ban nen",
                "em nen",
                "gia dinh hay",
            ),
        )
        or " em nen " in f" {key} "
        or " ban nen " in f" {key} "
    )


def _role_for_section(section_name: str) -> str:
    return {
        "pre_admission": "medical_history",
        "present_illness": "present_illness",
        "hospital_evaluation": "hospital_evaluation",
    }.get(section_name, "document")


def _is_patient_restart(key: str) -> bool:
    patient_prefix = r"^(?:\d+(?:\.\d+)*[.)/]?\s*)?(?:benh nhan|bn)\s+(?:nam|nu)"
    return bool(
        re.match(patient_prefix + r"\s+\d+\s+tuoi\b", key)
        or (re.match(patient_prefix + r"\b", key) and "vao vien" in key[:180])
    )


def _looks_like_heading(raw_line: str, key: str) -> bool:
    if len(key) > 140:
        return False
    if re.match(r"^\s*\d+(?:\.\d+)*[.)/]?\s+", raw_line):
        return True
    if raw_line.rstrip().endswith((':', '：')):
        return True
    return len(key.split()) <= 12 and not raw_line.rstrip().endswith(('.', '?', '!'))


def _looks_like_article_heading(raw_line: str, key: str) -> bool:
    if len(key) > 150 or not any(cue in key for cue in _ARTICLE_HEADING_CUES):
        return False
    if raw_line.lstrip().startswith(("-", "*", "•")):
        return False
    if re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s+", raw_line):
        return True
    return len(key.split()) <= 14 and not raw_line.rstrip().endswith('.')


def _merge_heading_only_blocks(text: str, blocks: list[StructuralBlock]) -> list[StructuralBlock]:
    merged: list[StructuralBlock] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        value = text[block.start:block.end].strip()
        if (
            index + 1 < len(blocks)
            and block.role not in {"question", "answer", "patient_case"}
            and len(value) <= 160
            and len(value.splitlines()) == 1
            and _is_structure_heading_only(value)
        ):
            following = blocks[index + 1]
            merged.append(StructuralBlock(following.role, block.start, following.end))
            index += 2
            continue
        merged.append(block)
        index += 1
    return merged


def _is_structure_heading_only(value: str) -> bool:
    key = _strip_heading_number(normalize_key(value.strip("*- "))).rstrip(":").strip()
    exact_markers = {
        *[marker.rstrip(":") for _, markers in MAJOR_SECTION_MARKERS for marker in markers],
        *_HISTORY_BLOCK_MARKERS,
        *_LAB_BLOCK_MARKERS,
        *_DIAGNOSIS_BLOCK_MARKERS,
        *_TREATMENT_BLOCK_MARKERS,
        *_SYMPTOM_PROFILE_MARKERS,
        *_PRE_ADMISSION_EVENT_MARKERS,
    }
    return key in exact_markers


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
