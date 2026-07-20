"""Dependency-free assertion classifier used after final span selection."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from core.config import ALLOWED_ASSERTIONS
from core.text import normalize_key
from extraction.sectioning import Section, detect_sections, detect_subsections
from extraction.annotation_memory import section_at


_NEGATION_RE = re.compile(r"\b(?:khong|chua|phu nhan|denies|without|negative for)\b")
_HISTORY_RE = re.compile(
    r"\b(?:tien su|da tung|truoc day|cach day|history of|past medical history|pmh|thuoc truoc)\b"
)
_FAMILY_RE = re.compile(
    r"\b(?:tien su gia dinh|family history|bo|cha|me|ong|ba|anh trai|chi gai|em trai|em gai)\b"
)


class AssertionClassifier:
    def __init__(self, models: dict[str, dict[str, object]] | None = None) -> None:
        self.models = models or {}

    @classmethod
    def empty(cls) -> AssertionClassifier:
        return cls({})

    @classmethod
    def load(cls, path: Path) -> AssertionClassifier:
        if not path.exists():
            return cls.empty()
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload.get("models") or {}
        return cls({str(label): dict(model) for label, model in models.items() if isinstance(model, dict)})

    @property
    def enabled(self) -> bool:
        return bool(self.models)

    def probabilities(
        self,
        text: str,
        start: int,
        end: int,
        concept_type: str,
        rule_assertions: Iterable[str] = (),
    ) -> dict[str, float]:
        if not self.models:
            return {}
        features = assertion_features(text, start, end, concept_type, rule_assertions)
        output: dict[str, float] = {}
        for label, model in self.models.items():
            weights = model.get("weights") or {}
            if not isinstance(weights, dict):
                continue
            logit = float(model.get("bias") or 0.0) + sum(
                float(weights.get(feature) or 0.0) for feature in features
            )
            output[label] = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        return output

    def threshold(self, label: str) -> float:
        model = self.models.get(label) or {}
        return float(model.get("threshold") or 0.5)


def assertion_features(
    text: str,
    start: int,
    end: int,
    concept_type: str,
    rule_assertions: Iterable[str] = (),
) -> tuple[str, ...]:
    mention = normalize_key(text[start:end])
    left = normalize_key(text[max(0, start - 320):start])
    right = normalize_key(text[end:min(len(text), end + 100)])
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    local_left = normalize_key(text[max(line_start, start - 180):start])
    words = left.split()
    sections = _sections(text)
    subsections = _subsections(text)
    features = [
        "bias",
        f"type={concept_type}",
        f"section={section_at(sections, start)}",
        f"subsection={section_at(subsections, start)}",
        f"mention={mention}",
        f"mention_first={mention.split()[0] if mention.split() else ''}",
        f"line_length={_bucket(line_end - line_start, (80, 160, 300, 600))}",
    ]
    for width in range(1, min(6, len(words)) + 1):
        features.append(f"left{width}={' '.join(words[-width:])}")
    context = " ".join((local_left, mention))
    if _NEGATION_RE.search(context):
        features.append("has_negation_cue")
    if _HISTORY_RE.search(" ".join((left, mention))):
        features.append("has_history_cue")
    if _FAMILY_RE.search(" ".join((left[-180:], mention))):
        features.append("has_family_cue")
    if _NEGATION_RE.search(mention):
        features.append("cue_inside_mention")
    if re.search(r"\b(?:nhung|tuy nhien|however|but)\b", local_left):
        features.append("contrast_before")
    if re.search(r"\b(?:benh su|hien tai|kham vao vien|current)\b", left[-220:]):
        features.append("current_scope")
    if re.search(r"\b(?:tien su|past medical history|pmh)\b", left[-220:]):
        features.append("history_scope")
    if re.search(r"\b(?:theo loi|gia dinh cho biet|reported by)\b", left[-120:]):
        features.append("reported_by_other")
    for label in rule_assertions:
        if label in ALLOWED_ASSERTIONS:
            features.append(f"rule={label}")
    return tuple(features)


@lru_cache(maxsize=128)
def _sections(text: str) -> tuple[Section, ...]:
    return tuple(detect_sections(text))


@lru_cache(maxsize=128)
def _subsections(text: str) -> tuple[Section, ...]:
    return tuple(detect_subsections(text))


def _bucket(value: int, limits: tuple[int, ...]) -> str:
    for limit in limits:
        if value <= limit:
            return f"<={limit}"
    return f">{limits[-1]}"
