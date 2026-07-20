"""Small dependency-free models for token proposals and span acceptance."""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.config import ALLOWED_TYPES
from core.text import normalize_key
from extraction.ner import SpanCandidate
from extraction.sectioning import detect_subsections
from extraction.annotation_memory import section_at


_TOKEN_RE = re.compile(r"\d+(?:[.,/]\d+)*|[^\W\d_]+(?:[-'][^\W\d_]+)*|[^\w\s]", re.UNICODE)
_UNIT_RE = re.compile(r"\b(?:mmhg|mmol/l|umol/l|mg/dl|g/l|g/dl|mg|mcg|ml|iu|bpm|%)\b")
_POISON_RE = re.compile(r"\b(?:ngo doc|phoi nhiem|hoa chat|thuoc tru sau|cocaine|heroin)\b")
_GENERIC_KEYS = {
    "bat thuong",
    "dau",
    "giam",
    "ket qua",
    "tang",
    "ton thuong",
    "xet nghiem",
}


@dataclass(frozen=True, slots=True)
class Token:
    start: int
    end: int
    text: str
    key: str


def tokenize(text: str) -> list[Token]:
    return [
        Token(match.start(), match.end(), match.group(0), normalize_key(match.group(0)))
        for match in _TOKEN_RE.finditer(text)
    ]


class TokenSpanModel:
    def __init__(
        self,
        labels: Iterable[str] = (),
        weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.labels = tuple(labels)
        self.weights = weights or {}

    @classmethod
    def empty(cls) -> TokenSpanModel:
        return cls((), {})

    @classmethod
    def load(cls, path: Path) -> TokenSpanModel:
        if not path.exists():
            return cls.empty()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(
            payload.get("labels") or (),
            {
                str(feature): {str(label): float(value) for label, value in values.items()}
                for feature, values in (payload.get("weights") or {}).items()
                if isinstance(values, dict)
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8", newline="\n") as fh:
            json.dump(
                {"format_version": 1, "labels": list(self.labels), "weights": self.weights},
                fh,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

    def propose(self, text: str) -> list[SpanCandidate]:
        if not self.labels or not self.weights or not text:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        subsections = detect_subsections(text)
        tags: list[str] = []
        margins: list[float] = []
        previous = "<START>"
        for index, token in enumerate(tokens):
            subsection = section_at(subsections, token.start)
            features = token_features(tokens, index, previous, subsection)
            scores = {
                label: sum(self.weights.get(feature, {}).get(label, 0.0) for feature in features)
                for label in self.labels
            }
            ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
            tag = ordered[0][0]
            if tag.startswith("I:") and previous not in {tag, "B:" + tag[2:]}:
                tag = "B:" + tag[2:]
            tags.append(tag)
            margins.append(ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1])
            previous = tag
        return _tags_to_spans(text, tokens, tags, margins)


class AveragedPerceptronTrainer:
    def __init__(self, labels: Iterable[str]) -> None:
        self.labels = tuple(labels)
        self.weights: dict[str, dict[str, float]] = defaultdict(dict)
        self._totals: dict[tuple[str, str], float] = defaultdict(float)
        self._timestamps: dict[tuple[str, str], int] = defaultdict(int)
        self._step = 0

    def train(self, sequences: Iterable[tuple[list[Token], list[str], list[str]]], epochs: int = 5) -> TokenSpanModel:
        materialized = list(sequences)
        for epoch in range(epochs):
            ordered = materialized if epoch % 2 == 0 else list(reversed(materialized))
            for tokens, truth, subsections in ordered:
                previous = "<START>"
                for index, gold in enumerate(truth):
                    features = token_features(tokens, index, previous, subsections[index])
                    guess = self._predict(features)
                    self._step += 1
                    if guess != gold:
                        weight = 1.6 if gold != "O" else 1.0
                        for feature in features:
                            self._update(feature, gold, weight)
                            self._update(feature, guess, -weight)
                    previous = guess
        self._average()
        pruned = {
            feature: {
                label: round(value, 6)
                for label, value in values.items()
                if abs(value) >= 0.02
            }
            for feature, values in self.weights.items()
        }
        return TokenSpanModel(self.labels, {key: value for key, value in pruned.items() if value})

    def _predict(self, features: Iterable[str]) -> str:
        scores = {label: 0.0 for label in self.labels}
        for feature in features:
            for label, value in self.weights.get(feature, {}).items():
                scores[label] += value
        return max(self.labels, key=lambda label: (scores[label], label))

    def _update(self, feature: str, label: str, value: float) -> None:
        key = (feature, label)
        current = self.weights[feature].get(label, 0.0)
        self._totals[key] += (self._step - self._timestamps[key]) * current
        self._timestamps[key] = self._step
        self.weights[feature][label] = current + value

    def _average(self) -> None:
        if not self._step:
            return
        for feature, values in list(self.weights.items()):
            for label, current in list(values.items()):
                key = (feature, label)
                total = self._totals[key] + (self._step - self._timestamps[key]) * current
                values[label] = total / self._step


class SpanAcceptanceModel:
    def __init__(
        self,
        bias: float = 0.0,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.bias = float(bias)
        self.weights = weights or {}
        self.thresholds = thresholds or {}

    @classmethod
    def empty(cls) -> SpanAcceptanceModel:
        return cls()

    @classmethod
    def load(cls, path: Path) -> SpanAcceptanceModel:
        if not path.exists():
            return cls.empty()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            payload.get("bias", 0.0),
            {str(key): float(value) for key, value in (payload.get("weights") or {}).items()},
            {str(key): float(value) for key, value in (payload.get("thresholds") or {}).items()},
        )

    def score(
        self,
        text: str,
        span: SpanCandidate,
        section: str,
        subsection: str,
        memory_score: float | None,
    ) -> float | None:
        if not self.weights:
            return None
        features = span_features(text, span, section, subsection, memory_score)
        logit = self.bias + sum(self.weights.get(feature, 0.0) for feature in features)
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

    def threshold_for(self, concept_type: str, source: str) -> float:
        return self.thresholds.get(f"{concept_type}|{source}", self.thresholds.get(concept_type, 0.5))


def token_features(tokens: list[Token], index: int, previous_tag: str, subsection: str) -> tuple[str, ...]:
    token = tokens[index]
    previous = tokens[index - 1].key if index else "<START>"
    following = tokens[index + 1].key if index + 1 < len(tokens) else "<END>"
    key = token.key or token.text
    return (
        "bias",
        f"token={key}",
        f"prefix2={key[:2]}",
        f"prefix3={key[:3]}",
        f"suffix2={key[-2:]}",
        f"suffix3={key[-3:]}",
        f"shape={_shape(token.text)}",
        f"previous={previous}",
        f"following={following}",
        f"previous_tag={previous_tag}",
        f"previous_tag+token={previous_tag}|{key}",
        f"subsection={subsection}",
    )


def span_features(
    text: str,
    span: SpanCandidate,
    section: str,
    subsection: str,
    memory_score: float | None,
) -> tuple[str, ...]:
    key = normalize_key(span.text)
    words = key.split()
    left = normalize_key(text[max(0, span.start - 80):span.start])
    right = normalize_key(text[span.end:min(len(text), span.end + 80)])
    source = span.parent_source or span.source
    features = [
        "bias",
        f"type={span.type}",
        f"source={source}",
        f"variant={span.variant}",
        f"type+source={span.type}|{source}",
        f"type+variant={span.type}|{span.variant}",
        f"section={section}",
        f"subsection={subsection}",
        f"type+subsection={span.type}|{subsection}",
        f"tokens={_bucket(len(words), (1, 2, 4, 8, 16))}",
        f"chars={_bucket(len(span.text), (4, 8, 16, 32, 64, 128))}",
        f"first={words[0] if words else ''}",
        f"last={words[-1] if words else ''}",
        f"mention={key}",
        f"shape={_shape(span.text)}",
    ]
    if any(char.isdigit() for char in span.text):
        features.append("has_digit")
    if _UNIT_RE.search(key):
        features.append("has_unit")
    if "(" in span.text and ")" in span.text:
        features.append("has_parenthetical")
    if key in _GENERIC_KEYS:
        features.append("generic_key")
    if _POISON_RE.search(" ".join((left[-80:], key, right[:80]))):
        features.append("poison_context")
    if re.search(r"\b(?:khong|chua|phu nhan)\b", " ".join((left[-35:], key))):
        features.append("negation_context")
    if memory_score is not None:
        features.append(f"memory={int(max(0.0, min(0.99, memory_score)) * 5)}")
    return tuple(features)


def _tags_to_spans(
    text: str,
    tokens: list[Token],
    tags: list[str],
    margins: list[float],
) -> list[SpanCandidate]:
    output: list[SpanCandidate] = []
    index = 0
    while index < len(tokens):
        tag = tags[index]
        if not tag.startswith("B:"):
            index += 1
            continue
        concept_type = tag[2:]
        end_index = index + 1
        while end_index < len(tokens) and tags[end_index] == f"I:{concept_type}":
            end_index += 1
        start = tokens[index].start
        end = tokens[end_index - 1].end
        margin = sum(margins[index:end_index]) / max(1, end_index - index)
        score = 0.68 + 0.2 * (1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, margin)))) - 0.5)
        output.append(
            SpanCandidate(start, end, text[start:end], concept_type, score, "sequence_model")
        )
        index = end_index
    return output


def _shape(value: str) -> str:
    output: list[str] = []
    for char in value[:24]:
        marker = "D" if char.isdigit() else "U" if char.isupper() else "L" if char.isalpha() else char
        if not output or output[-1] != marker:
            output.append(marker)
    return "".join(output)


def _bucket(value: int, limits: tuple[int, ...]) -> str:
    for limit in limits:
        if value <= limit:
            return f"<={limit}"
    return f">{limits[-1]}"
