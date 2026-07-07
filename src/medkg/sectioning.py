"""Lightweight section detection for Vietnamese clinical notes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    end: int


SECTION_MARKERS = (
    ("history", "Tiền sử"),
    ("present_illness", "Tiền sử bệnh hiện tại"),
    ("assessment", "Đánh giá tại bệnh viện"),
    ("labs", "Kết quả xét nghiệm"),
    ("imaging", "Kết quả chẩn đoán hình ảnh"),
)


def detect_sections(text: str) -> list[Section]:
    hits: list[tuple[int, str]] = []
    lower = text.casefold()
    for name, marker in SECTION_MARKERS:
        idx = lower.find(marker.casefold())
        if idx >= 0:
            hits.append((idx, name))
    hits.sort()
    if not hits:
        return [Section("document", 0, len(text))]
    sections: list[Section] = []
    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        sections.append(Section(name, start, end))
    return sections
